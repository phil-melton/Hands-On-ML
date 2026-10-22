"""Verification: completions.jsonl -> grades.jsonl on the CPU, with a tier for every sample.

Each completion is reduced to code (``bakeoff.extract.extract_code``: first python fence, else the whole reply) and graded
by ``bakeoff.grade.grade_job``, which calls ``cvbench.evaluate.evaluate_solution`` for the spec's family at the spec's
seed (20 scenes, 5 metamorphic scenes by default) and so runs the code only inside ``sandbox.run_solution`` (CLAUDE.md
Rule 4). The four gates are independent; ``api`` only reports.

Tiers (PLAN.md section 4.1; Gate 2 decision: Tier A needs every gate):

    A     exec-only, lint, manufactured and metamorphic all pass. The metamorphic gate also runs the typed contract
          (``check_outputs``) on every output, and the manufactured gate does so first ("contract: ..." reasons).
    B     exec-only and lint pass, the metamorphic gate (which includes the contract) passes, and the manufactured gate
          fails with a *value* reason: right shape, dtype and order, wrong number. A "contract" reason from the manufactured
          gate (``contract: ...``, "not sorted", wrong order, wrong shape) or an exception disqualifies. Lint is required
          because a bare ``except`` can make a constant answer look invariant (PLAN.md 8: a stuck gauge is repeatable).
    fail  anything else, including a reply with no code and a grader that raised (``grader_error`` is kept apart so a
          verifier bug is visible, not scored as a model failure).

Whether a manufactured-gate reason is a contract problem is decided by ``bakeoff.taxonomy.classify_reason`` (the
project's own rule: ``convention`` = contract). Tier C (runs + contract only) is never a training target and is "fail" here.

Drill specs (``kind == "drill"``, foundry.drills) are graded by ``cvbench.gotchas.core.grade`` instead (lint, sandbox run,
contract, verifier on fixed cases). A drill has no metamorphic gate, so its Tier A means "the grade passed"; it has no Tier B.

Only the answer is graded. A completion row carries ``answer`` (the text after a thinking teacher's ``</think>``, see
colab_job.py and foundry.answer.answer_of); older rows are split again from ``text``. A reply whose reasoning never closed
(``think_status`` "truncated") has no answer and fails as "no code". Completions of any kind but "generate" and "repair" in the
same file (the phrasing requests of the combined session, ``kind`` "phrasing") are not graded here: ``screen-phrasings`` reads them.

Selection (``select_examples``): per spec, the Tier-A samples in sample order, the first of each distinct AST-normalized
code (foundry.astnorm), at most ``cap`` = 2.

Thinking vs non-thinking. When a run holds samples of both modes (the prompt's ``mode`` "think" / "nothink", else thinking
tokens present or not), ``mode_table`` compares their Tier-A rate per family, on all samples and on the specs that have both,
with the tokens each costs per Tier-A sample; ``verify`` prints it.

Resumable like the bake-off: rerunning skips graded (id, sample) pairs; identical (spec, code) pairs are graded once.
"""
from __future__ import annotations

import importlib
import json
import math
import multiprocessing
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

from bakeoff.extract import extract_code
from bakeoff.grade import PASS_GATES, _static, code_sha1, grade_job, payload_of
from bakeoff.taxonomy import classify, classify_reason
from foundry.answer import answer_of
from foundry.astnorm import ast_hash
from foundry.store import FoundryRun, JsonlWriter, read_json, read_jsonl, write_json

FOUR = tuple(PASS_GATES)  # exec-only, lint, manufactured, metamorphic
MAX_WORKERS = 4           # a GPU bake-off may be running: the foundry never uses more
SCENE_PREFIX = re.compile(r"\Ascene \d+:\s*")
ITEM_PREFIX = re.compile(r"\Aitem \d+:\s*")
SOLUTION_KINDS = ("generate", "repair")   # completion kinds that hold code to grade; "phrasing" holds task wordings


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class VerifySettings:
    workers: int = 4
    n_scenes: int = 20
    n_meta: int = 5
    timeout_s: float | None = None

    def as_dict(self) -> dict:
        return {"n_scenes": self.n_scenes, "n_meta": self.n_meta, "timeout_s": self.timeout_s}


# ---------------------------------------------------------------- tiers and failure messages

def _passed(gates: dict, name: str) -> bool:
    return bool(gates.get(name, {}).get("passed", False))


def manufactured_is_value_error(detail: str) -> bool:
    """True when the manufactured gate failed on the numbers (a verifier reason that is not a contract reason)."""
    if re.match(r"\A(exception|violation|bad_output|timeout|killed)\b", detail or ""):
        return False  # the code did not run to the end: not a wrong number
    cls, _ = classify_reason(SCENE_PREFIX.sub("", detail or ""))
    return cls != "convention"


def tier_of(rec: dict) -> str:
    """"A", "B" or "fail" for a grade record (the bake-off payload: gates, code_empty, grader_error). A drill record (no gates,
    ``kind`` "drill") is Tier A when its grade passed with the lint enforced, and "fail" otherwise: a drill graded with
    ``lint=False`` (it needs I/O the lint bans) can never yield a training target."""
    if rec.get("code_empty") or rec.get("grader_error"):
        return "fail"
    if rec.get("kind") == "drill":
        return "A" if rec.get("passed") and rec.get("lint_enforced", True) else "fail"
    gates = rec.get("gates") or {}
    if all(_passed(gates, g) for g in FOUR):
        return "A"
    if (_passed(gates, "exec-only") and _passed(gates, "lint") and _passed(gates, "metamorphic")
            and not _passed(gates, "manufactured") and manufactured_is_value_error(gates.get("manufactured", {}).get("detail", ""))):
        return "B"
    return "fail"


def failure_of(rec: dict) -> dict | None:
    """What went wrong, for repair prompts and the report; None for Tier A.

    ``message`` is the grader's text for the first failing gate (exec-only, lint, manufactured, metamorphic) with the
    scene seed removed, plus a line for each removed-in-5.0 API found. ``repairable`` is False for an empty reply, for a
    reply that is prose (no fenced code, and it does not parse), and for a grader crash (nothing to repair, or not the
    model's fault)."""
    if rec.get("tier", tier_of(rec)) == "A":
        return None
    c = classify(rec)
    cls, rule = (c.cls, c.rule) if c else ("other", "unexplained")
    if rec.get("code_empty"):
        if rec.get("think_status") == "truncated":   # the reasoning used the whole budget: there is no answer at all
            return {"gate": None, "message": "the reasoning never finished (token budget)", "class": "format",
                    "rule": "thinking-truncated", "repairable": False}
        return {"gate": None, "message": "the reply contained no code", "class": cls, "rule": rule, "repairable": False}
    if rec.get("grader_error"):
        return {"gate": None, "message": "grader error", "class": cls, "rule": rule, "repairable": False}
    if rec.get("extract") == "bare" and any(f.get("rule") == "syntax" for f in rec.get("lint_findings") or []):
        return {"gate": None, "message": "the reply contained no code (prose only)", "class": "format", "rule": "no-code",
                "repairable": False}
    if rec.get("kind") == "drill":
        gate, message = rec.get("stage"), ITEM_PREFIX.sub("", rec.get("detail") or "failed")
    else:
        gates = rec["gates"]
        gate = next((g for g in FOUR if not _passed(gates, g)), None)
        message = SCENE_PREFIX.sub("", gates[gate]["detail"]) if gate else "failed"
    hints = [f"`{f['name']}` is not in OpenCV 5.0" + (f"; use {f['replacement']}" if f.get("replacement") else "")
             for f in (rec.get("api_findings") or [])[:3]]
    if hints:
        message += " (" + "; ".join(hints) + ")"
    return {"gate": gate, "message": message, "class": cls, "rule": rule, "repairable": True}


def select_examples(rows: list[dict], cap: int = 2, tier: str = "A") -> tuple[list[dict], dict]:
    """Per spec: the rows of ``tier`` in sample order, the first of each distinct ``ast_sha1``, at most ``cap``.

    Returns (kept rows with ``rank`` 0.., {"candidates", "duplicates", "over_cap"})."""
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("tier") == tier:
            by_spec[r["spec_id"]].append(r)
    kept, dup, over = [], 0, 0
    for spec_id in sorted(by_spec):
        seen: set[str] = set()
        n = 0
        for r in sorted(by_spec[spec_id], key=lambda r: (r["sample"], r["id"])):
            if r["ast_sha1"] in seen:
                dup += 1
            elif n >= cap:
                seen.add(r["ast_sha1"])
                over += 1
            else:
                seen.add(r["ast_sha1"])
                kept.append({**r, "rank": n})
                n += 1
    return kept, {"candidates": sum(len(v) for v in by_spec.values()), "duplicates": dup, "over_cap": over}


# ---------------------------------------------------------------- grading

def make_grade_row(comp: dict, spec_row: dict, code: str, ex_path: str, ex_think: str, payload: dict, *, cached: bool) -> dict:
    row = {"id": comp["id"], "spec_id": comp["spec_id"], "sample": comp["sample"], "family": spec_row["family"],
           "kind": spec_row.get("kind", "family"), "mode": comp.get("mode"), "think_status": comp.get("think_status"),
           "thinking_tokens": comp.get("thinking_tokens"), "answer_tokens": comp.get("answer_tokens"),
           "finish_reason": comp.get("finish_reason"), "completion_tokens": comp.get("completion_tokens"),
           "extract": ex_path, "think": ex_think, "code": code, "code_sha1": code_sha1(code),
           "ast_sha1": ast_hash(code) if code.strip() else "", **payload, "cached": cached}
    if cached:
        row["grade_s"] = 0.0
    row["tier"] = tier_of(row)
    row["failure"] = failure_of(row)
    return row


def check_settings(path, settings: VerifySettings, grades_exist: bool) -> None:
    """One grades file must come from one grader configuration."""
    old = read_json(path)
    if old and old != settings.as_dict() and grades_exist:
        raise SystemExit(f"{path} says the existing grades used {old}, not {settings.as_dict()}; "
                         "use a new run folder or the old settings")
    write_json(path, settings.as_dict())


def grade_any(job: dict) -> dict:
    """Grade one (task, code) in a pool worker: a drill through ``cvbench.gotchas.core.grade``, everything else through the
    bake-off's ``grade_job``. Both run the code only inside ``sandbox.run_solution`` (CLAUDE.md Rule 4)."""
    task = job["task"]
    if task["kind"] != "drill":
        return grade_job(job)
    t0 = time.perf_counter()
    try:
        from cvbench.gotchas.core import grade
        check = importlib.import_module("foundry.drills").DRILLS[task["drill"]]
        g = grade(check, job["code"], timeout_s=job["settings"]["timeout_s"] or 30.0)
        out = {"passed": g.passed, "ran": g.ran, "stage": g.stage, "detail": g.detail, "error_type": g.error_type,
               "error_msg": g.error, "lint_enforced": check.lint}
        out.update(_static(job["code"]))
    except Exception:  # noqa: BLE001 - a verifier bug must be visible, not scored as a model failure
        out = {"passed": False, "ran": False, "grader_error": traceback.format_exc()[-1500:]}
    out["grade_s"] = round(time.perf_counter() - t0, 3)
    return out


def job_task(spec: dict) -> dict:
    if spec.get("kind") == "drill":
        return {"task_id": spec["id"], "kind": "drill", "drill": spec["drill"]}
    return {"task_id": spec["id"], "kind": "family", "family": spec["family"], "task_seed": spec["seed"]}


def grade_stage(run: FoundryRun, stage: str, settings: VerifySettings = VerifySettings(), *, log=_log) -> dict:
    """Grade every completion of ``stage`` not yet graded. Returns counts."""
    st = run.stage(stage)
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    if not specs:
        raise SystemExit(f"{run.specs} is missing or empty; run `prepare` first")
    comps = read_jsonl(st.completions)
    if not comps:
        raise SystemExit(f"{st.completions} is missing or empty: nothing to verify (download the Colab job's output first)")
    other = [c for c in comps if c.get("kind", "generate") not in SOLUTION_KINDS]
    if other:
        log(f"skipping {len(other)} completions of other kinds ({sorted({c['kind'] for c in other})}): not solutions")
        comps = [c for c in comps if c.get("kind", "generate") in SOLUTION_KINDS]
    workers = min(settings.workers, MAX_WORKERS)
    if workers != settings.workers:
        log(f"warning: using {workers} workers, not {settings.workers} (the foundry never uses more than {MAX_WORKERS})")
    settings = VerifySettings(workers, settings.n_scenes, settings.n_meta, settings.timeout_s)
    prior = read_jsonl(st.grades)
    check_settings(st.grading, settings, bool(prior))
    done = {(r["id"], r["sample"]) for r in prior}
    cache = {(r["spec_id"], r["code_sha1"]): payload_of(r) for r in prior if not r.get("cached") and r.get("code_sha1")}
    counts = {"graded": 0, "executed": 0, "cached": 0, "grader_errors": 0, "A": 0, "B": 0, "fail": 0}
    todo = [c for c in comps if (c["id"], c["sample"]) not in done]
    for c in todo:
        if c["spec_id"] not in specs:
            raise SystemExit(f"completion {c['id']} names spec {c['spec_id']!r}, which is not in {run.specs}")

    in_flight: dict = {}                    # future -> cache key
    waiting: dict[tuple, list[tuple]] = {}  # cache key -> [(completion, code, extract path, think)]

    with JsonlWriter(st.grades) as out:
        def emit(comp, code, path, think, payload, cached):
            row = make_grade_row(comp, specs[comp["spec_id"]], code, path, think, payload, cached=cached)
            out.write(row)
            counts["graded"] += 1
            counts["cached" if cached else "executed"] += 1
            counts["grader_errors"] += bool(row.get("grader_error"))
            counts[row["tier"]] += 1
            if counts["graded"] % 50 == 0:
                log(f"  graded {counts['graded']} / {len(todo)} (A {counts['A']}, B {counts['B']})")

        pool = None
        try:
            for comp in todo:
                ex = extract_code(answer_of(comp))   # the answer only, never the reasoning
                if not ex.code.strip():
                    emit(comp, "", ex.path, ex.think, {"passed": False, "ran": False, "code_empty": True, "grade_s": 0.0}, False)
                    continue
                key = (comp["spec_id"], code_sha1(ex.code))
                if key in cache:
                    emit(comp, ex.code, ex.path, ex.think, cache[key], True)
                elif key in waiting:
                    waiting[key].append((comp, ex.code, ex.path, ex.think))
                else:
                    if pool is None:
                        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                    waiting[key] = [(comp, ex.code, ex.path, ex.think)]
                    job = {"task": job_task(specs[comp["spec_id"]]), "code": ex.code, "settings": settings.as_dict()}
                    in_flight[pool.submit(grade_any, job)] = key
            while in_flight:
                finished, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    key = in_flight.pop(fut)
                    payload = fut.result()
                    cache[key] = payload
                    first, *rest = waiting.pop(key)
                    emit(*first, payload, False)
                    for item in rest:
                        emit(*item, payload, True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    return counts


def write_kept(run: FoundryRun, cap: int = 2) -> dict:
    """kept.jsonl: the Tier-A samples kept (at most ``cap`` per spec, distinct AST). Returns the selection statistics."""
    rows = read_jsonl(run.stage("generate").grades)
    kept, stats = select_examples(rows, cap)
    with run.kept.open("w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps({"spec_id": r["spec_id"], "sample": r["sample"], "family": r["family"], "rank": r["rank"],
                                 "ast_sha1": r["ast_sha1"], "tier": "A"}, sort_keys=True) + "\n")
    return {**stats, "kept": len(kept)}


def summarize(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        d = out.setdefault(r["family"], {"A": 0, "B": 0, "fail": 0})
        d[r["tier"]] += 1
    return out


# ---------------------------------------------------------------- thinking vs non-thinking

def mode_of(row: dict) -> str:
    """"think" or "nothink" for a grade row: the prompt's own ``mode`` label, else whether the completion thought."""
    if row.get("mode"):
        return row["mode"]
    thought = (row.get("thinking_tokens") or 0) > 0 or row.get("think_status") in ("closed", "truncated")
    return "think" if thought else "nothink"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson interval of a proportion, in percent; None without samples."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def _mode_stats(rows: list[dict]) -> dict:
    n = len(rows)
    a = sum(1 for r in rows if r["tier"] == "A")
    tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    specs = {r["spec_id"] for r in rows}
    solved = {r["spec_id"] for r in rows if r["tier"] == "A"}
    return {"samples": n, "specs": len(specs), "tier_A": a, "tier_A_pct": 100 * a / n if n else None, "ci": wilson(a, n),
            "specs_solved": len(solved), "mean_tokens": tokens / n if n else None,
            "mean_thinking_tokens": sum(r.get("thinking_tokens") or 0 for r in rows) / n if n else None,
            "truncated_thinking": sum(1 for r in rows if r.get("think_status") == "truncated"),
            "cut_off": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "tokens_per_A": tokens / a if a else None}


def mode_table(rows: list[dict]) -> dict | None:
    """Tier-A yield by thinking mode per family, on all samples and on the specs that both modes were asked; None when the
    grades hold a single mode."""
    rows = [r for r in rows if "tier" in r]
    modes = sorted({mode_of(r) for r in rows})
    if len(modes) < 2:
        return None
    fams = list(dict.fromkeys(r["family"] for r in rows))
    asked = {(f, m): {r["spec_id"] for r in rows if r["family"] == f and mode_of(r) == m} for f in fams for m in modes}
    common = {f: set.intersection(*(asked[f, m] for m in modes)) for f in fams}   # specs both modes were asked
    out = {"modes": modes, "families": {}, "all": {}, "paired": {}, "paired_all": {}}
    for f in fams:
        mine = [r for r in rows if r["family"] == f]
        out["families"][f] = {m: _mode_stats([r for r in mine if mode_of(r) == m]) for m in modes}
        out["paired"][f] = {"specs": len(common[f]),
                            **{m: _mode_stats([r for r in mine if mode_of(r) == m and r["spec_id"] in common[f]]) for m in modes}}
    out["all"] = {m: _mode_stats([r for r in rows if mode_of(r) == m]) for m in modes}
    paired = [r for r in rows if r["spec_id"] in common[r["family"]]]
    out["paired_all"] = {"specs": sum(len(v) for v in common.values()), **{m: _mode_stats([r for r in paired if mode_of(r) == m]) for m in modes}}
    return out


def _fmt(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def render_mode_table(t: dict) -> str:
    """Plain-text version of ``mode_table`` for the console and report.md."""
    modes = t["modes"]
    head = "| family | " + " | ".join(f"{m} A % [95 % CI] (n) | {m} tok/sample | {m} tok/A" for m in modes) + " | paired specs | " + \
        " | ".join(f"{m} A % (paired)" for m in modes) + " |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]

    def cell(s):
        ci = s["ci"]
        return (f"{_fmt(s['tier_A_pct'])} [{_fmt(ci[0], 0) if ci else '-'}-{_fmt(ci[1], 0) if ci else '-'}] ({s['samples']}) | "
                f"{_fmt(s['mean_tokens'], 0)} | {_fmt(s['tokens_per_A'], 0)}")

    for f, per in {**t["families"], "**all**": t["all"]}.items():
        p = t["paired_all"] if f == "**all**" else t["paired"][f]
        lines.append(f"| {f} | " + " | ".join(cell(per[m]) for m in modes) + f" | {p['specs']} | " +
                     " | ".join(_fmt(p[m]["tier_A_pct"]) for m in modes) + " |")
    trunc = {m: t["all"][m]["truncated_thinking"] for m in modes}
    lines.append("")
    lines.append("tok/A = completion tokens spent per Tier-A sample (what a kept example costs); "
                 f"reasoning cut off before the answer: {trunc}; mean reasoning tokens per sample: "
                 f"{ {m: round(t['all'][m]['mean_thinking_tokens'] or 0) for m in modes} }.")
    return "\n".join(lines)

