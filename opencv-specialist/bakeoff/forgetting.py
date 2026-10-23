"""Forgetting check (success criterion 3): paired pass@1 on an MBPP+ slice, base model vs fine-tuned model.

    python -m bakeoff.forgetting info                                    # the slice: size, sha256, revision
    python -m bakeoff.forgetting validate --workers 2                    # reference solutions through the sandbox (all must pass)
    python -m bakeoff.forgetting predict --tok-s 53 --tokens base=450 --tokens ft=200
    python -m bakeoff.forgetting generate --run NAME --arm base=TAG --arm ft=TAG    # Ollama, GPU, resumable
    python -m bakeoff.forgetting grade    --run NAME --workers 2                    # CPU, sandbox, resumable
    python -m bakeoff.forgetting report   --run NAME --a base --b ft                # report.md, summary.json

The slice (data/forgetting/mbppplus_slice.jsonl, built from EvalPlus MBPP+ by data/forgetting/build_slice.py, see the
README beside it) holds, per problem: the task text, the entry point, the original MBPP asserts and the EvalPlus
test script (``inputs``/``results`` literals plus an ``assertion`` helper and a loop that calls the entry point).

Protocol (the same for both arms, so the comparison is paired)
    * one system message, ``SYSTEM_GENERAL``, sent to every arm in place of the model's own SYSTEM; the user message
      is the problem text plus the entry-point name and the first original MBPP assert (the EvalPlus prompt style);
    * one greedy reply per problem: T 0, repeat_penalty 1, num_predict 1024, ``think: false``; the reply is cut
      by the bake-off's extractor, except that when the first python block does not define the entry point and a
      later one does, that one is taken (a formatting choice must not decide a correctness comparison);
    * the answer passes iff the EvalPlus script runs to the end in the sandbox without raising.

How the tests run (CLAUDE.md Rule 4: model code runs only through ``sandbox.run_solution``)
    The sandbox contract is ``solve(**kwargs) -> dict`` in a limited child process. The graded program is the
    model's code, unchanged, followed by a fixed adapter that defines ``solve(test_src)``: it looks up the entry point
    that the model's code defined, ``exec``s the EvalPlus script in a fresh namespace in which that name is bound, and
    returns ``{"passed": True}``. So the model's code never runs in this process, and neither does the dataset's test
    script. A wrong answer is the script's AssertionError, any other exception is a runtime failure, a missing
    entry point is its own class, and the sandbox timeout / memory limits apply. ``sandbox/`` is unchanged.

    Two deliberate departures from ``run_solution``'s defaults, both visible in the grade rows:
      * ``lint=False``. The project lint is an OpenCV-contract gate (it demands a top-level ``solve``, bans
        ``except Exception``, bans ``import sys``); none of that is correctness of a general Python answer. The lint
        findings (other than the contract rule) are still recorded per answer and the report shows how many answers
        of each arm would have been flagged.
      * a safety screen instead (``screen_code``): code that imports a module the project lint bans (``os``,
        ``subprocess``, ``socket``, ... except ``sys``) or calls ``open/exec/eval/compile/__import__/input/...`` is
        not run and counts as a failure of class ``blocked``. The child still has no network (own namespace),
        write guards, rlimits and the wall-clock kill. The report counts blocked answers per arm.
      * an output cap (``OUTPUT_CAP``): the sandbox parent buffers the child's stdout in memory, so printing in a
        loop would fill it; prints are discarded and more than 1 MiB of them is a RuntimeError in the child.

Time: generation is the long part (GPU, ~0.5-2 h for 2 x 300 replies, see ``predict``); grading is CPU, a few minutes
(0.2 s per answer in the sandbox, up to ``TIMEOUT_S`` for an infinite loop), two worker threads at most.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import re
import sys
import traceback
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from bakeoff.extract import _PY_LABELS, _THINK, extract_code
from bakeoff.generate import GenSettings, _info, run_generation
from bakeoff.predict import parse_per_rung, predict, render
from bakeoff.rungs import Rung
from bakeoff.stats import binomial_se, mcnemar_exact, paired_compare, percentile
from bakeoff.store import JsonlWriter, RunDir, key_of, keys, read_json, read_jsonl, write_json
from bakeoff.tasks import Task
from sandbox.lint import BANNED_BUILTINS, BANNED_IMPORTS, lint as lint_code
from sandbox.run import run_solution
from train.config import OLLAMA, WORK

DATASET = "evalplus/mbppplus"
REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
SLICE_PATH = Path(__file__).resolve().parents[1] / "data" / "forgetting" / "mbppplus_slice.jsonl"
DEFAULT_ROOT = WORK / "forgetting-runs"   # ~/opencv-expert/forgetting/ holds the untouched download; runs go beside it
SLICE_SEED = 20261022
SLICE_SIZE = 300
NGRAM = 13
TIMEOUT_S = 90.0  # per problem; the slowest reference solution takes ~16 s (Mbpp/599), the rest under 4 s
MEM_MB = 2048     # the sandbox default; Mbpp/255 needs more than this even for its reference, so it is not in the slice

SYSTEM_GENERAL = "You are a helpful Python programmer. Write the requested function."

# Words that mark a problem as OpenCV / image work. Deliberately narrow: MBPP's "colors" (pattern matching) and
# "resolution" (quotation marks) are not image processing.
IMAGE_WORDS = re.compile(r"\bcv2?\b|opencv|\bimages?\b|\bpixels?\b|\bphotos?\b|\bpictures?\b|\bbitmaps?\b|"
                         r"\bgrayscale\b|\bgreyscale\b|\brgb\b|computer vision|\bcamera\b", re.I)

SCREEN_ALLOW = frozenset({"sys"})  # banned by the project lint, but plain MBPP code uses sys.maxsize

CLASS_ORDER = ("pass", "wrong", "runtime", "syntax", "no_entry", "timeout", "killed", "blocked", "empty",
               "violation", "bad_output", "grader_error")
PAYLOAD_KEYS = ("passed", "cls", "status", "error_type", "error", "blocked_by", "lint_rules", "run_s", "netns",
                "n_inputs", "grader_error")
Z95 = 1.959963984540054


def _parse(src: str) -> ast.Module:
    """ast.parse without the SyntaxWarnings (an unescaped backslash in a regex string) that dataset and model code trigger."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src)


# ================================================================ the problems

def entry_point_of(test_src: str) -> str:
    """The function an EvalPlus script tests: the callee in ``assertion(F(*inp), ...)`` of its final loop (parsed, not run)."""
    loop = _parse(test_src).body[-1]
    for node in ast.walk(loop):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "assertion"
                and node.args and isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Name)):
            return node.args[0].func.id
    raise ValueError("no assertion(F(*inp), ...) call in the final statement of the test script")


def user_prompt(text: str, entry_point: str, first_test: str) -> str:
    """The EvalPlus prompt style: the task, then one assert that fixes the name and signature."""
    return (f"{text.strip()}\n\nThe function must be named `{entry_point}` and should pass this test:\n"
            f"{first_test.strip()}")


def exclusion_reason(rec: dict) -> str | None:
    """The OpenCV / image-processing word found in a raw MBPP+ record (text, reference solution, asserts), or None."""
    blob = "\n".join([rec["prompt"], rec["code"], *rec["test_list"]])
    m = IMAGE_WORDS.search(blob)
    return m.group(0).lower() if m else None


def select_slice(ids: Iterable[int], n: int, seed: int = SLICE_SEED) -> list[int]:
    """n ids chosen by sorting on sha256(seed|id): deterministic on any Python version, independent of input order.
    Returned sorted ascending. All ids when n >= the pool."""
    ranked = sorted(set(ids), key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).hexdigest())
    return sorted(ranked[:n])


# ---------------------------------------------------------------- 13-gram overlap

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def overlap_scan(queries: dict, corpora: dict, n: int = NGRAM) -> dict:
    """Word n-grams (lowercased, ``\\w+`` tokens) of every query text found in any row of any corpus.

    ``queries``: {key: text}. ``corpora``: {name: iterable of (row_id, text)}, streamed row by row, so memory is the
    query n-grams only. Returns {key: {corpus: {"n_grams": grams in the query, "shared": most query grams found in
    one row, "row": that row's id, "rows": rows sharing at least one}}}."""
    index: dict[tuple, list] = {}
    totals = {}
    for key, text in queries.items():
        grams = ngrams(words(text), n)
        totals[key] = len(grams)
        for g in grams:
            index.setdefault(g, []).append(key)
    out = {key: {c: {"n_grams": totals[key], "shared": 0, "row": None, "rows": 0} for c in corpora} for key in queries}
    for cname, rows in corpora.items():
        for row_id, text in rows:
            toks = words(text)
            if len(toks) < n:
                continue
            seen: set[tuple] = set()
            per_key: Counter = Counter()
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if g in index and g not in seen:
                    seen.add(g)
                    for key in index[g]:
                        per_key[key] += 1
            for key, cnt in per_key.items():
                o = out[key][cname]
                o["rows"] += 1
                if cnt > o["shared"]:
                    o["shared"], o["row"] = cnt, row_id
    return out


def corpus_text(row: dict) -> str:
    """A training row as text for the overlap scan: the user and assistant turns (the system prompt is shared by all rows)."""
    return "\n".join(m["content"] for m in row["messages"] if m["role"] != "system")


# ---------------------------------------------------------------- slice file

def load_slice(path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path else SLICE_PATH
    rows = read_jsonl(p)
    if not rows:
        raise FileNotFoundError(f"{p} is missing or empty; see data/forgetting/README.md")
    need = {"task_id", "prompt", "entry_point", "test_list", "test", "reference"}
    for r in rows:
        if need - set(r):
            raise ValueError(f"{p}: row {r.get('task_id')} lacks {sorted(need - set(r))}")
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError(f"{p}: duplicate task ids")
    return rows


def file_sha256(path: Path | str | None = None) -> str:
    return hashlib.sha256(Path(path or SLICE_PATH).read_bytes()).hexdigest()


def slice_tasks(rows: list[dict], limit: int | None = None) -> list[Task]:
    """Bake-off ``Task`` objects (so the generation loop is reused unchanged); first ``limit`` by MBPP id."""
    rows = sorted(rows, key=lambda r: r.get("mbpp_id", 0))
    if limit:
        rows = rows[:limit]
    return [Task(id=r["task_id"], kind="forgetting", family="mbpp+", subset="general", held_out=False, role="task",
                 seed=None, prompt=user_prompt(r["prompt"], r["entry_point"], r["test_list"][0]), cluster=r["task_id"])
            for r in rows]


# ================================================================ reply -> code

def python_fences(text: str) -> list[str]:
    """Every fenced block labelled python/py/python3 or unlabelled, in order (same rules as bakeoff.extract)."""
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s.startswith("```"):
            i += 1
            continue
        ws = s[3:].strip().lower().split()
        label = ws[0] if ws else ""
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("```"):
            j += 1
        if label in _PY_LABELS:
            out.append("".join(lines[i + 1:j]))
        i = j + 1
    return out


def defines(code: str, name: str) -> bool:
    """Does the code bind ``name`` at top level (def, class or assignment)? Falls back to a regex when it does not parse."""
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*=", code, re.M) is not None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
    return False


def pick_code(reply: str, entry_point: str) -> tuple[str, str]:
    """(code, path). The bake-off extraction (first python block, else the whole reply) unless that code does not
    define the entry point and a later python block does: then the first such block, path ``fence-later``."""
    ex = extract_code(reply)
    if defines(ex.code, entry_point):
        return ex.code, ex.path
    for code in python_fences(_THINK.sub("", reply, count=1)):
        if defines(code, entry_point):
            return code, "fence-later"
    return ex.code, ex.path


# ================================================================ running the tests in the sandbox

# Output cap. The sandbox parent reads the child's stdout/stderr into memory until the child ends, so a model's
# `while True: print(...)` would fill this process (and the machine) for the whole timeout. The cap discards output
# and raises once 1 MiB has been written; nothing in the graded path needs the model's prints.
OUTPUT_CAP = '''import sys as _sys_cap


class _OutputCap:
    encoding = "utf-8"
    errors = "strict"

    def __init__(self, limit=1 << 20):
        self._n, self._limit = 0, limit

    def write(self, s):
        self._n += len(s)
        if self._n > self._limit:
            raise RuntimeError("output limit exceeded (1 MiB)")
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def _install_output_cap():
    _sys_cap.stdout = _OutputCap()
    _sys_cap.stderr = _OutputCap()
'''

ADAPTER = '''

# ---- appended by bakeoff.forgetting: runs the EvalPlus script on the entry point the code above defined ----
_CANDIDATE = globals().get({entry!r})


def solve(test_src):
    _install_output_cap()
    if _CANDIDATE is None:
        raise NameError("entry point " + {entry!r} + " is not defined")
    ns = {{"__name__": "mbppplus_test", {entry!r}: _CANDIDATE}}
    exec(compile(test_src, "<mbppplus test>", "exec"), ns)
    return {{"passed": True, "n_inputs": len(ns["inputs"])}}
'''
_FUTURE_IMPORT = re.compile(r"^\s*from\s+__future__\s+import", re.M)


def build_program(code: str, entry_point: str) -> str:
    """The model's code unchanged, between the output cap (in front, unless the code has a ``from __future__`` import,
    which must come first; the adapter installs the cap again before the tests run) and the adapter."""
    future = bool(_FUTURE_IMPORT.search(code))
    head = "" if future else OUTPUT_CAP + "_install_output_cap()\n\n\n"
    tail = (OUTPUT_CAP if future else "") + ADAPTER.format(entry=entry_point)
    return head + code.rstrip() + "\n\n" + tail


def screen_code(code: str) -> str | None:
    """Why this code must not be run at all (a banned module or builtin), or None. Pure ``ast``; nothing is executed."""
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None  # fails in the sandbox as a SyntaxError, which is the honest class
    for node in ast.walk(tree):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots = [(node.module or "").split(".")[0]]
        for r in roots:
            if r in BANNED_IMPORTS and r not in SCREEN_ALLOW:
                return f"import {r}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
            return f"{node.func.id}()"
    return None


def classify_run(res) -> str:
    """Failure class of a sandbox RunResult."""
    if res.ok:
        return "pass"
    if res.status == "exception":
        et = res.error_type or ""
        if et in ("SyntaxError", "IndentationError", "TabError"):
            return "syntax"
        if et == "NameError" and (res.error or "").startswith("entry point "):
            return "no_entry"
        return "wrong" if et == "AssertionError" else "runtime"
    return res.status if res.status in ("timeout", "killed", "violation", "bad_output") else "runtime"


def grade_code(code: str, rec: dict, *, timeout_s: float = TIMEOUT_S, mem_mb: int = MEM_MB) -> dict:
    """Grade one answer for one problem. Never raises: a bug here is recorded as class ``grader_error``."""
    try:
        lint_rules = sorted({f.rule for f in lint_code(code)} - {"contract"})
        if not code.strip():
            return {"passed": False, "cls": "empty", "status": "empty", "lint_rules": lint_rules}
        why = screen_code(code)
        if why:
            return {"passed": False, "cls": "blocked", "status": "blocked", "blocked_by": why, "lint_rules": lint_rules}
        res = run_solution(build_program(code, rec["entry_point"]), [{"test_src": rec["test"]}], timeout_s=timeout_s,
                           mem_mb=mem_mb, lint=False)
        ok = res.ok and bool(res.outputs and res.outputs[0].get("passed") is True)
        out = {"passed": ok, "cls": "pass" if ok else (classify_run(res) if not res.ok else "bad_output"),
               "status": res.status, "error_type": res.error_type, "error": (res.error or "")[:300],
               "lint_rules": lint_rules, "run_s": round(res.seconds, 3), "netns": res.netns}
        if ok:
            out["n_inputs"] = res.outputs[0]["n_inputs"]
        return out
    except Exception:  # noqa: BLE001 - a verifier bug must be visible, not scored as a model failure
        return {"passed": False, "cls": "grader_error", "grader_error": traceback.format_exc()[-1200:]}


def _run_jobs(jobs: list[tuple], workers: int, timeout_s: float, mem_mb: int):
    """jobs: [(key, code, rec)] -> yields (key, payload) as they finish. Threads are enough: the work is in the
    sandbox child processes, the threads only wait on them."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(grade_code, code, rec, timeout_s=timeout_s, mem_mb=mem_mb): key for key, code, rec in jobs}
        try:
            for f in as_completed(futs):
                yield futs[f], f.result()
        finally:
            for f in futs:
                f.cancel()


def validate_reference(rows: list[dict], *, workers: int = 2, timeout_s: float = TIMEOUT_S, mem_mb: int = MEM_MB) -> list[dict]:
    """Run each problem's reference solution through the same path as a model answer. Returns one dict per problem."""
    by_id = {r["task_id"]: r for r in rows}
    out = []
    for tid, payload in _run_jobs([(r["task_id"], r["reference"], r) for r in rows], workers, timeout_s, mem_mb):
        out.append({"task_id": tid, "mbpp_id": by_id[tid].get("mbpp_id"), **payload})
    return sorted(out, key=lambda x: (x["mbpp_id"] or 0, x["task_id"]))


# ================================================================ generation

def parse_arms(values: list[str]) -> list[Rung]:
    """['base=base-qwen3.5-9b', 'ft=my-tag'] -> Rungs (think false: both are Qwen3.5 and must not think)."""
    arms = []
    for v in values:
        label, _, tag = v.partition("=")
        if not tag:
            label, tag = v, v
        if not label or not tag:
            raise SystemExit(f"--arm wants LABEL=OLLAMA_TAG, got {v!r}")
        arms.append(Rung(label, tag, think=False))
    if len({a.name for a in arms}) != len(arms):
        raise SystemExit("arm labels must be unique")
    return arms


def resolve_think(arms: list[Rung], base_url: str, mode: str = "auto") -> list[Rung]:
    """Set each arm's ``think`` flag. ``auto`` (as train/hydro_eval.py does): ``think: false`` when ``/api/show`` lists the
    thinking capability (Ollama then thinks by default, which would replace the trained non-thinking format), no flag when it
    does not (a model without the capability can reject the field). ``false`` / ``omit`` force it for every arm."""
    out = []
    for r in arms:
        if mode == "auto":
            caps = _info(base_url, "/api/show", {"model": r.model}).get("capabilities") or []
            think = False if ("thinking" in caps or not caps) else None
        else:
            think = False if mode == "false" else None
        out.append(dataclasses.replace(r, think=think))
    return out


def check_slice(run: RunDir, sha: str) -> None:
    """A run folder holds one slice: refuse to resume with a different file."""
    cfg = read_json(run.config)
    old = {i.get("slice_sha256") for i in (cfg or {}).get("invocations", []) if i.get("slice_sha256")}
    if old and old != {sha}:
        raise SystemExit(f"{run.config} was started on a different slice file (sha256 {sorted(old)}, now {sha}); use a new --run")


# ================================================================ grading a run

def grading_settings(timeout_s: float, mem_mb: int, slice_sha: str) -> dict:
    return {"timeout_s": timeout_s, "mem_mb": mem_mb, "lint": False, "screen": "banned imports (except sys) and builtins",
            "slice_sha256": slice_sha}


def check_grading_settings(run: RunDir, gs: dict) -> None:
    old = read_json(run.grading)
    if old and old != gs and read_jsonl(run.grades):
        raise SystemExit(f"{run.grading} says the existing grades used {old}, not {gs}; use a new --run or the old settings")
    write_json(run.grading, gs)


def grade_run(run: RunDir, rows: list[dict], gs: dict, *, workers: int = 2, log=lambda m: print(m, file=sys.stderr, flush=True)) -> dict:
    """Grade every ungraded generation. Identical (problem, code) pairs are graded once (the sandbox is deterministic)."""
    check_grading_settings(run, gs)
    by_id = {r["task_id"]: r for r in rows}
    prior = read_jsonl(run.grades)
    done = keys(prior)
    cache = {(r["task_id"], r["code_sha1"]): {k: r[k] for k in PAYLOAD_KEYS if k in r} for r in prior if not r.get("cached")}
    counts = {"graded": 0, "executed": 0, "cached": 0, "grader_errors": 0}
    waiting: dict[tuple, list[dict]] = {}
    jobs = []
    with JsonlWriter(run.grades) as out:
        def emit(gen, code_path, sha, payload, cached):
            row = {k: gen[k] for k in ("rung", "task_id", "mode", "sample")}
            row.update(payload)
            row.update(code_sha1=sha, code_path=code_path, cached=cached)
            if cached:
                row.pop("run_s", None)
            out.write(row)
            counts["graded"] += 1
            counts["cached" if cached else "executed"] += 1
            counts["grader_errors"] += bool(payload.get("grader_error"))

        for g in read_jsonl(run.generations):
            if key_of(g) in done or g["task_id"] not in by_id:
                continue
            code, path = pick_code(g["reply"], by_id[g["task_id"]]["entry_point"])
            sha = hashlib.sha1(code.encode("utf-8")).hexdigest()
            k = (g["task_id"], sha)
            if k in cache:
                emit(g, path, sha, cache[k], True)
            elif k in waiting:
                waiting[k].append((g, path))
            else:
                waiting[k] = [(g, path)]
                jobs.append((k, code, by_id[g["task_id"]]))
        log(f"  {counts['graded']} reused from the cache, {len(jobs)} distinct answers to run in the sandbox")
        for k, payload in _run_jobs(jobs, workers, gs["timeout_s"], gs["mem_mb"]):
            cache[k] = payload
            (first, fpath), *rest = waiting.pop(k)
            emit(first, fpath, k[1], payload, False)
            for g, path in rest:
                emit(g, path, k[1], payload, True)
            if counts["executed"] % 25 == 0:
                log(f"  graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused)")
    return counts


# ================================================================ statistics

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired(both: int, a_only: int, b_only: int, neither: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe (1998, Stat Med 17:2635) method 10: a CI for p_A - p_B from paired pass/fail counts, the hybrid
    of the two Wilson intervals with a continuity-corrected correlation estimate."""
    a, b, c, d = both, a_only, b_only, neither
    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan")
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    big_a = (a + b) * (c + d) * (a + c) * (b + d)
    c0 = a * d - b * c
    c_adj = c0 - n / 2 if c0 > n / 2 else (0.0 if c0 >= 0 else c0)
    phi = c_adj / math.sqrt(big_a) if big_a > 0 else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return max(-1.0, lo), min(1.0, hi)


def paired_report(a: list[int], b: list[int]) -> dict:
    """A (reference arm) vs B on the same problems. ``drop`` = pass(A) - pass(B), positive when B is worse."""
    pc = paired_compare(a, b)
    lo, hi = newcombe_paired(pc["both"], pc["a_only"], pc["b_only"], pc["neither"])
    return {**pc, "drop": pc["diff"], "ci_lo": lo, "ci_hi": hi, "p_mcnemar": pc["p"]}


# ================================================================ report

def _pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def _stat(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values) if values else float("nan"), "median": percentile(values, 50),
            "p90": percentile(values, 90), "max": max(values) if values else float("nan")}


def summarize(run: RunDir, arm_a: str, arm_b: str) -> dict:
    gens = [g for g in read_jsonl(run.generations) if g["mode"] == "greedy"]
    grades = {(g["rung"], g["task_id"]): g for g in read_jsonl(run.grades) if g["mode"] == "greedy"}
    cfg = read_json(run.config, {})
    ga = {g["task_id"]: g for g in gens if g["rung"] == arm_a}
    gb = {g["task_id"]: g for g in gens if g["rung"] == arm_b}
    common = sorted(t for t in ga if t in gb and (arm_a, t) in grades and (arm_b, t) in grades)
    if not common:
        raise SystemExit(f"no problem is generated and graded for both {arm_a!r} and {arm_b!r} in {run.path}")
    arms = {}
    for name, gen in ((arm_a, ga), (arm_b, gb)):
        rows = [(gen[t], grades[(name, t)]) for t in common]
        k = sum(1 for _, gr in rows if gr["passed"])
        lo, hi = wilson(k, len(rows))
        eval_s = [g["eval_count"] / (g["eval_duration"] / 1e9) for g, _ in rows if g["eval_duration"]]
        arms[name] = {
            "model": cfg.get("rungs", {}).get(name, {}).get("model"), "think": cfg.get("rungs", {}).get(name, {}).get("think"),
            "capabilities": (cfg.get("rungs", {}).get(name, {}).get("show") or {}).get("capabilities"), "n": len(rows), "passed": k, "pass1": k / len(rows),
            "se": binomial_se(k / len(rows), len(rows)), "ci_lo": lo, "ci_hi": hi,
            "classes": dict(Counter(gr["cls"] for _, gr in rows)),
            "truncated": sum(1 for g, _ in rows if g["truncated"]),
            "extract": dict(Counter(g["extract"] for g, _ in rows)),
            "code_path": dict(Counter(gr["code_path"] for _, gr in rows)),
            "blocked_by": dict(Counter(gr["blocked_by"] for _, gr in rows if gr.get("blocked_by"))),
            "lint_flagged": sum(1 for _, gr in rows if gr.get("lint_rules")),
            "lint_rules": dict(Counter(r for _, gr in rows for r in gr.get("lint_rules", []))),
            "pass_if_lint_enforced": sum(1 for _, gr in rows if gr["passed"] and not gr.get("lint_rules")),
            "tokens": _stat([g["eval_count"] for g, _ in rows]), "decode_tps": _stat(eval_s),
            "gen_wall_s": _stat([g["wall_s"] for g, _ in rows]), "gen_wall_total_min": sum(g["wall_s"] for g, _ in rows) / 60,
            "prompt_tokens": _stat([g["prompt_eval_count"] for g, _ in rows]),
            "grader_errors": sum(1 for _, gr in rows if gr["cls"] == "grader_error"),
        }
    run_s = [gr["run_s"] for gr in grades.values() if gr.get("run_s") is not None and gr["rung"] in (arm_a, arm_b)]
    va = [int(grades[(arm_a, t)]["passed"]) for t in common]
    vb = [int(grades[(arm_b, t)]["passed"]) for t in common]
    warm = {}
    for w in read_jsonl(run.warmups):
        warm.setdefault(w["rung"], w.get("prompt_eval_count"))
    discordant = [{"task_id": t, "passed": arm_a if va[i] else arm_b,
                   "other_cls": grades[(arm_b if va[i] else arm_a, t)]["cls"],
                   "other_error": grades[(arm_b if va[i] else arm_a, t)].get("error_type") or grades[(arm_b if va[i] else arm_a, t)].get("blocked_by")}
                  for i, t in enumerate(common) if va[i] != vb[i]]
    return {"run": str(run.path), "arm_a": arm_a, "arm_b": arm_b, "n": len(common), "arms": arms,
            "paired": paired_report(va, vb), "sandbox_s": _stat(run_s), "warmup_prompt_tokens": warm,
            "system": cfg.get("rungs", {}).get(arm_a, {}).get("system"), "decoding": cfg.get("decoding", {}).get("greedy"),
            "slice_sha256": (read_json(run.grading) or {}).get("slice_sha256"), "discordant": discordant,
            "ollama_version": cfg.get("ollama_version")}


def render_report(s: dict) -> str:
    a, b = s["arm_a"], s["arm_b"]
    A, B, P = s["arms"][a], s["arms"][b], s["paired"]
    drop, lo, hi = P["drop"], P["ci_lo"], P["ci_hi"]
    met = drop <= 0.03
    shown = hi <= 0.03
    L = [f"# Forgetting check: {b} vs {a} on MBPP+ ({s['n']} problems)", "",
         f"Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.", "",
         f"- reference arm A = `{a}` ({A['model']}), candidate arm B = `{b}` ({B['model']})",
         f"- dataset {DATASET} @ {REVISION}; slice file sha256 {s['slice_sha256']}",
         f"- system prompt (both arms): {s['system']!r}", f"- decoding: {s['decoding']}, one greedy reply per problem",
         f"- `think` flag sent: {a} {A['think']}, {b} {B['think']} (None: omitted); capabilities reported: {a} {A['capabilities']}, {b} {B['capabilities']}",
         f"- Ollama {s['ollama_version']}", "",
         "## Result", "",
         f"| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |", "|---|---|---|---|---|"]
    for name, x in ((a, A), (b, B)):
        L.append(f"| {name} | {x['passed']}/{x['n']} | {_pct(x['pass1'])} | [{_pct(x['ci_lo'])}, {_pct(x['ci_hi'])}] | {_pct(x['se'])} |")
    L += ["", f"Paired on the same {s['n']} problems: both pass {P['both']}, only {a} {P['a_only']}, only {b} {P['b_only']}, neither {P['neither']}.", "",
          f"- **drop = pass@1({a}) - pass@1({b}) = {100 * drop:+.1f} points** (positive: {b} is worse)",
          f"- 95 % CI of the drop (Newcombe method 10): [{100 * lo:+.1f}, {100 * hi:+.1f}] points; Wald paired SE {100 * P['se']:.1f}",
          f"- exact McNemar p = {P['p_mcnemar']:.3g} ({P['a_only'] + P['b_only']} discordant problems)",
          f"- criterion 3 on the point estimate (drop <= 3 points): **{'MET' if met else 'NOT MET'}**",
          f"- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **{'yes' if shown else 'no'}**"
          + ("" if shown or not met else " (the point estimate passes but the interval is too wide to exclude a larger drop)"), "",
          "## What the failures were", "", "| class | " + " | ".join([a, b]) + " |", "|---|---|---|"]
    for c in CLASS_ORDER:
        if A["classes"].get(c) or B["classes"].get(c):
            L.append(f"| {c} | {A['classes'].get(c, 0)} | {B['classes'].get(c, 0)} |")
    L += ["", "| | " + " | ".join([a, b]) + " |", "|---|---|---|",
          f"| replies cut at num_predict (graded as they are) | {A['truncated']} | {B['truncated']} |",
          f"| reply format (first-block extraction path) | {A['extract']} | {B['extract']} |",
          f"| code taken from a later block (first block lacked the entry point) | {A['code_path'].get('fence-later', 0)} | {B['code_path'].get('fence-later', 0)} |",
          f"| blocked by the safety screen (counted as failures) | {sum(A['blocked_by'].values())} {A['blocked_by'] or ''} | {sum(B['blocked_by'].values())} {B['blocked_by'] or ''} |",
          f"| answers the project lint would flag (info only: lint is not enforced here) | {A['lint_flagged']} {A['lint_rules'] or ''} | {B['lint_flagged']} {B['lint_rules'] or ''} |",
          f"| passes if the project lint were enforced | {A['pass_if_lint_enforced']} | {B['pass_if_lint_enforced']} |",
          f"| grader errors | {A['grader_errors']} | {B['grader_errors']} |", ""]
    wa, wb = s["warmup_prompt_tokens"].get(a), s["warmup_prompt_tokens"].get(b)
    same = wa is not None and wb is not None and abs(wa - wb) <= 2
    L += ["## Time per problem", "", "| | " + " | ".join([a, b]) + " |", "|---|---|---|"]
    for label, key, unit, d in (("generation wall, mean", ("gen_wall_s", "mean"), "s", 1), ("generation wall, median", ("gen_wall_s", "median"), "s", 1),
                                ("generation wall, p90", ("gen_wall_s", "p90"), "s", 1), ("tokens per answer, mean", ("tokens", "mean"), "", 0),
                                ("decode tok/s, median", ("decode_tps", "median"), "", 1)):
        L.append(f"| {label} {unit} | {A[key[0]][key[1]]:.{d}f} | {B[key[0]][key[1]]:.{d}f} |")
    L += ["", f"Generation total: {a} {A['gen_wall_total_min']:.1f} min, {b} {B['gen_wall_total_min']:.1f} min. "
              f"Sandbox time per executed answer: mean {s['sandbox_s']['mean']:.2f} s, p90 {s['sandbox_s']['p90']:.2f} s, max {s['sandbox_s']['max']:.2f} s.", "",
          f"Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens {a} {wa}, {b} {wb} "
          + ("(equal: yes)" if same else "(**differ: the Modelfile SYSTEM may have been prepended; investigate before trusting the comparison**)"), "",
          f"## Problems where the arms differ ({len(s['discordant'])})", "", "| problem | passed | the other arm failed as |", "|---|---|---|"]
    for d in s["discordant"][:80]:
        L.append(f"| {d['task_id']} | {d['passed']} | {d['other_cls']} {d['other_error'] or ''} |")
    if len(s["discordant"]) > 80:
        L.append(f"| ... | | {len(s['discordant']) - 80} more in summary.json |")
    return "\n".join(L) + "\n"


# ================================================================ CLI

def _root(a) -> Path:
    return Path(a.root) if getattr(a, "root", None) else DEFAULT_ROOT


def cmd_info(a) -> int:
    rows = load_slice(a.slice)
    ids = sorted(r.get("mbpp_id", 0) for r in rows)
    print(f"{a.slice or SLICE_PATH}\n  {len(rows)} problems, MBPP ids {ids[0]}..{ids[-1]}, sha256 {file_sha256(a.slice)}\n"
          f"  dataset {DATASET} @ {REVISION}\n  system prompt: {SYSTEM_GENERAL!r}")
    return 0


def cmd_validate(a) -> int:
    rows = load_slice(a.slice)
    res = validate_reference(rows, workers=a.workers, timeout_s=a.timeout)
    bad = [r for r in res if not r["passed"]]
    times = [r["run_s"] for r in res if r.get("run_s") is not None]
    print(f"{len(res) - len(bad)}/{len(res)} reference solutions pass; sandbox seconds per problem: "
          f"mean {sum(times) / len(times):.2f}, p90 {percentile(times, 90):.2f}, max {max(times):.2f}; netns {Counter(r.get('netns') for r in res)}")
    for r in bad:
        print(f"  FAIL {r['task_id']}: {r['cls']} {r.get('error_type')} {r.get('error')}")
    return 1 if bad else 0


def cmd_predict(a) -> int:
    arms = [x.partition("=")[0] for x in (a.arm or [])] or ["base", "ft"]
    n = a.tasks or len(load_slice(a.slice))
    print(render(predict(arms, parse_per_rung(a.tok_s, arms, "--tok-s"), parse_per_rung(a.tokens, arms, "--tokens"), n, 0,
                         overhead_s=a.overhead_s, load_s=a.load_s), n, 0, a.overhead_s, a.load_s))
    return 0


def cmd_generate(a) -> int:
    rows = load_slice(a.slice)
    tasks = slice_tasks(rows, a.limit)
    arms = resolve_think(parse_arms(a.arm), a.ollama, a.think)
    st = GenSettings(base_url=a.ollama, samples=0, num_predict=a.num_predict, num_ctx=a.num_ctx, keep_alive=a.keep_alive,
                     http_timeout=a.http_timeout)
    run = RunDir.of(a.run, _root(a))
    sha = file_sha256(a.slice)
    check_slice(run, sha)
    inv = {"arms": {r.name: r.model for r in arms}, "think": {r.name: r.think for r in arms}, "n_tasks": len(tasks), "limit": a.limit, "slice_sha256": sha,
           "dataset": f"{DATASET}@{REVISION}"}
    written = run_generation(run, arms, tasks, st, invocation=inv, system=SYSTEM_GENERAL)
    print(f"generated {written} into {run.generations}")
    return 0


def cmd_grade(a) -> int:
    rows = load_slice(a.slice)
    run = RunDir.of(a.run, _root(a))
    gs = grading_settings(a.timeout, MEM_MB, file_sha256(a.slice))
    counts = grade_run(run, rows, gs, workers=a.workers)
    print(f"graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused, "
          f"{counts['grader_errors']} grader errors) into {run.grades}")
    return 1 if counts["grader_errors"] else 0


def cmd_report(a) -> int:
    run = RunDir.of(a.run, _root(a))
    cfg = read_json(run.config)
    if not cfg:
        raise SystemExit(f"{run.config} not found: run generate first")
    labels = list(cfg["rungs"])
    arm_a, arm_b = a.a or labels[0], a.b or (labels[1] if len(labels) > 1 else None)
    if arm_b is None:
        raise SystemExit("the run has one arm; a paired report needs two (--a, --b)")
    s = summarize(run, arm_a, arm_b)
    write_json(run.summary, s)
    run.report_md.write_text(render_report(s), encoding="utf-8")
    P = s["paired"]
    for name in (arm_a, arm_b):
        x = s["arms"][name]
        print(f"{name:<12} pass@1 {_pct(x['pass1'])}  [{_pct(x['ci_lo'])}, {_pct(x['ci_hi'])}]  ({x['passed']}/{x['n']})")
    print(f"drop {100 * P['drop']:+.1f} pts, 95 % CI [{100 * P['ci_lo']:+.1f}, {100 * P['ci_hi']:+.1f}], McNemar p {P['p_mcnemar']:.3g}")
    print(f"wrote {run.report_md}, {run.summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bakeoff.forgetting", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=False):
        p.add_argument("--slice", help=f"slice file (default {SLICE_PATH})")
        if run:
            p.add_argument("--run", required=True, help="run name (folder under --root)")
            p.add_argument("--root", help=f"parent folder of runs (default {DEFAULT_ROOT})")

    p = sub.add_parser("info", help="slice size, sha256, revision")
    common(p)
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("validate", help="run every reference solution through the sandbox path (all must pass)")
    common(p)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=TIMEOUT_S)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("predict", help="predicted generation minutes per arm (Rule 1)")
    common(p)
    p.add_argument("--arm", action="append", help="arm labels (default base, ft)")
    p.add_argument("--tok-s", action="append", required=True, help="decode tokens/s: a number, or arm=number")
    p.add_argument("--tokens", action="append", required=True, help="expected tokens per answer: a number, or arm=number")
    p.add_argument("--tasks", type=int)
    p.add_argument("--overhead-s", type=float, default=0.5)
    p.add_argument("--load-s", type=float, default=90.0)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("generate", help="ask the Ollama models every problem (GPU, resumable)")
    common(p, True)
    p.add_argument("--arm", action="append", required=True, help="LABEL=OLLAMA_TAG, repeatable (the first is the reference arm)")
    p.add_argument("--limit", type=int, help="first N problems by MBPP id (smoke test)")
    p.add_argument("--num-predict", type=int, default=1024)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--keep-alive", default="30m")
    p.add_argument("--ollama", default=OLLAMA)
    p.add_argument("--http-timeout", type=float, default=900.0)
    p.add_argument("--think", choices=("auto", "false", "omit"), default="auto",
                   help="auto: send think:false to models that list the thinking capability (like train/hydro_eval.py); "
                        "false / omit: the same for every arm")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("grade", help="run the EvalPlus tests on the replies in the sandbox (CPU, resumable)")
    common(p, True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per problem in the sandbox")
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("report", help="pass@1 per arm, paired difference, McNemar, CI, timing")
    common(p, True)
    p.add_argument("--a", help="reference arm label (default: first arm of the run)")
    p.add_argument("--b", help="candidate arm label (default: second arm)")
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    # model code with an unescaped backslash in a string would make every ast.parse here print a SyntaxWarning
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
