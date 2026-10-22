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

Selection (``select_examples``): per spec, the Tier-A samples in sample order, the first of each distinct AST-normalized
code (foundry.astnorm), at most ``cap`` = 2.

Resumable like the bake-off: rerunning skips graded (id, sample) pairs; identical (spec, code) pairs are graded once.
"""
from __future__ import annotations

import json
import multiprocessing
import re
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

from bakeoff.extract import extract_code
from bakeoff.grade import PASS_GATES, code_sha1, grade_job, payload_of
from bakeoff.taxonomy import classify, classify_reason
from foundry.astnorm import ast_hash
from foundry.store import FoundryRun, JsonlWriter, read_json, read_jsonl, write_json

FOUR = tuple(PASS_GATES)  # exec-only, lint, manufactured, metamorphic
MAX_WORKERS = 4           # a GPU bake-off may be running: the foundry never uses more
SCENE_PREFIX = re.compile(r"\Ascene \d+:\s*")


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
    """"A", "B" or "fail" for a grade record (the bake-off payload: gates, code_empty, grader_error)."""
    if rec.get("code_empty") or rec.get("grader_error"):
        return "fail"
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
        return {"gate": None, "message": "the reply contained no code", "class": cls, "rule": rule, "repairable": False}
    if rec.get("grader_error"):
        return {"gate": None, "message": "grader error", "class": cls, "rule": rule, "repairable": False}
    if rec.get("extract") == "bare" and any(f.get("rule") == "syntax" for f in rec.get("lint_findings") or []):
        return {"gate": None, "message": "the reply contained no code (prose only)", "class": "format", "rule": "no-code",
                "repairable": False}
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


def grade_stage(run: FoundryRun, stage: str, settings: VerifySettings = VerifySettings(), *, log=_log) -> dict:
    """Grade every completion of ``stage`` not yet graded. Returns counts."""
    st = run.stage(stage)
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    if not specs:
        raise SystemExit(f"{run.specs} is missing or empty; run `prepare` first")
    comps = read_jsonl(st.completions)
    if not comps:
        raise SystemExit(f"{st.completions} is missing or empty: nothing to verify (download the Colab job's output first)")
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
                ex = extract_code(comp["text"])
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
                    spec = specs[comp["spec_id"]]
                    job = {"task": {"task_id": spec["id"], "kind": "family", "family": spec["family"], "task_seed": spec["seed"]},
                           "code": ex.code, "settings": settings.as_dict()}
                    in_flight[pool.submit(grade_job, job)] = key
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

