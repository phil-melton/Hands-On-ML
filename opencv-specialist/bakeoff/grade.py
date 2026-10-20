"""Grading: generations.jsonl -> grades.jsonl through the existing graders, in a process pool.

Family tasks go through ``cvbench.evaluate.evaluate_solution`` (seed = the task's seed, so scenes are the same for
every rung), gotchas through ``cvbench.gotchas.grade``. Both run the model's code only inside
``sandbox.run_solution`` (CLAUDE.md Rule 4); this module adds no execution path of its own. A task passes iff its
four gates pass (exec-only, lint, manufactured, metamorphic); the report-only ``api`` gate is recorded and feeds the
taxonomy, but never decides a pass (Gate 2 decision 4).

Resumable and decoupled from generation: rerunning skips graded rows; ``--follow`` keeps polling a growing
generations file until generate writes its done marker, so CPU grading overlaps GPU generation. Identical
(task, code) pairs are graded once (greedy replies of different rungs often coincide, and so do tasks that share a
prompt); a repeated pair reuses the earlier result and is marked ``cached``. Grading is deterministic, so this
changes nothing but the CPU time.

A grader that raises is recorded as ``grader_error`` and counted in the report: a verifier bug must be visible,
not scored as a model failure.
"""
from __future__ import annotations

import hashlib
import importlib
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

from bakeoff.store import JsonlWriter, RunDir, key_of, keys, read_json, read_jsonl, write_json
from bakeoff.taxonomy import RUN_GATES, classify, parse_run_detail

PASS_GATES = ("exec-only", "lint", "manufactured", "metamorphic")  # `api` only reports
PAYLOAD_KEYS = ("passed", "ran", "gates", "stage", "detail", "error_gate", "error_type", "error_msg", "lint_enforced",
                "lint_findings", "api_findings", "code_empty", "grader_error", "grade_s")


@dataclass(frozen=True)
class GradeSettings:
    workers: int = 4
    n_scenes: int = 20
    n_meta: int = 5
    timeout_s: float | None = None  # None: each grader's own default (60 s families, 30 s gotchas)

    def as_dict(self) -> dict:
        return {"n_scenes": self.n_scenes, "n_meta": self.n_meta, "timeout_s": self.timeout_s}


def _static(code: str) -> dict:
    """Lint rules and removed-API findings by AST inspection (nothing is executed)."""
    from cvbench.api_check import check_removed_api
    from sandbox.lint import lint
    return {"lint_findings": [{"rule": f.rule, "line": f.line, "message": f.message} for f in lint(code)],
            "api_findings": [{"line": f.line, "name": f.name, "api": f.api, "label": f.label, "replacement": f.replacement}
                             for f in check_removed_api(code)]}


def grade_job(job: dict) -> dict:
    """Grade one (task, code). Runs in a pool worker; returns the payload fields of a grade row."""
    task, code, s = job["task"], job["code"], job["settings"]
    t0 = time.perf_counter()
    try:
        if task["kind"] == "family":
            from cvbench.evaluate import evaluate_solution
            fam = importlib.import_module(f"cvbench.families.{task['family']}")
            gates = evaluate_solution(fam, code, n_scenes=s["n_scenes"], n_meta=s["n_meta"], seed=task["task_seed"],
                                      timeout_s=s["timeout_s"] or 60.0)
            out = {"passed": all(gates[g].passed for g in PASS_GATES), "ran": gates["exec-only"].passed,
                   "gates": {k: {"passed": g.passed, "detail": g.detail} for k, g in gates.items()}}
            first = next((g for g in RUN_GATES if not gates[g].passed), None)  # the sandbox run that failed first, if any
            run = parse_run_detail(gates[first].detail) if first else None
            if run:
                out.update(error_gate=first, error_type=run["etype"] or run["status"], error_msg=run["msg"])
        else:
            from cvbench.gotchas import CHECKS, grade
            check = CHECKS[task["task_id"]]
            g = grade(check, code, timeout_s=s["timeout_s"] or 30.0)
            out = {"passed": g.passed, "ran": g.ran, "stage": g.stage, "detail": g.detail,
                   "error_type": g.error_type, "error_msg": g.error, "lint_enforced": check.lint}
        out.update(_static(code))
    except Exception:  # noqa: BLE001 - see the module docstring
        out = {"passed": False, "ran": False, "grader_error": traceback.format_exc()[-1500:]}
    out["grade_s"] = round(time.perf_counter() - t0, 3)
    return out


def code_sha1(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


def _task_of(gen: dict) -> dict:
    return {k: gen[k] for k in ("task_id", "kind", "family", "task_seed")}


def payload_of(row: dict) -> dict:
    return {k: row[k] for k in PAYLOAD_KEYS if k in row}


def make_grade_row(gen: dict, payload: dict, *, cached: bool) -> dict:
    row = {k: gen[k] for k in ("rung", "task_id", "mode", "sample", "kind", "family", "subset", "held_out", "role", "cluster")}
    row.update(payload)
    row["code_sha1"] = code_sha1(gen["code"])
    row["cached"] = cached
    if cached:
        row["grade_s"] = 0.0  # the CPU time was spent on the first copy
    c = classify(row)
    row["taxonomy"] = c.as_dict() if c else None
    return row


def check_settings(run: RunDir, gs: GradeSettings) -> None:
    """One grades.jsonl must come from one grader configuration."""
    old = read_json(run.grading)
    if old and old != gs.as_dict() and read_jsonl(run.grades):
        raise SystemExit(f"{run.grading} says the existing grades used {old}, not {gs.as_dict()}; "
                         "use a new --run name or the old settings")
    write_json(run.grading, gs.as_dict())


def grade_run(run: RunDir, gs: GradeSettings, *, follow: bool = False, poll_s: float = 10.0,
              log=lambda m: print(m, file=sys.stderr, flush=True)) -> dict:
    """Grade every ungraded generation. With ``follow``, keep polling until generate's done marker appears and
    everything is graded. Returns counts."""
    check_settings(run, gs)
    prior = read_jsonl(run.grades)
    done = keys(prior)
    cache = {(r["task_id"], r["code_sha1"]): payload_of(r) for r in prior if not r.get("cached")}
    counts = {"graded": 0, "executed": 0, "cached": 0, "grader_errors": 0}
    in_flight: dict = {}                    # future -> (task_id, code sha1)
    waiting: dict[tuple, list[dict]] = {}   # (task_id, code sha1) -> generations waiting for that result
    claimed: set[tuple] = set()             # generation keys already emitted or waiting

    with ProcessPoolExecutor(max_workers=gs.workers, mp_context=multiprocessing.get_context("spawn")) as pool, \
            JsonlWriter(run.grades) as out:
        def emit(gen: dict, payload: dict, cached: bool) -> None:
            row = make_grade_row(gen, payload, cached=cached)
            out.write(row)
            counts["graded"] += 1
            counts["cached" if cached else "executed"] += 1
            counts["grader_errors"] += bool(row.get("grader_error"))
            if counts["graded"] % 50 == 0:
                log(f"  graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused)")

        def refill() -> int:
            """Read the generations file and dispatch what is new. Returns how many rows were new."""
            n = 0
            for g in read_jsonl(run.generations):
                if key_of(g) in done or key_of(g) in claimed:
                    continue
                n += 1
                claimed.add(key_of(g))
                if not g["code"].strip():
                    emit(g, {"passed": False, "ran": False, "code_empty": True, "grade_s": 0.0}, cached=False)
                    continue
                k = (g["task_id"], code_sha1(g["code"]))
                if k in cache:
                    emit(g, cache[k], cached=True)
                elif k in waiting:
                    waiting[k].append(g)
                else:
                    waiting[k] = [g]
                    in_flight[pool.submit(grade_job, {"task": _task_of(g), "code": g["code"],
                                                      "settings": gs.as_dict()})] = k
            return n

        refill()
        last_read = time.monotonic()
        while True:
            if in_flight:
                finished, _ = wait(set(in_flight), timeout=poll_s if follow else None, return_when=FIRST_COMPLETED)
                for fut in finished:
                    k = in_flight.pop(fut)
                    payload = fut.result()
                    cache[k] = payload
                    first, *rest = waiting.pop(k)
                    emit(first, payload, cached=False)
                    for g in rest:
                        emit(g, payload, cached=True)
            elif not follow:
                break
            if follow and (not in_flight or time.monotonic() - last_read >= poll_s):
                generation_finished = run.done.exists()  # looked at before reading: rows written before it are seen
                new = refill()
                last_read = time.monotonic()
                if not new and not in_flight:
                    if generation_finished:
                        break
                    time.sleep(poll_s)
    return counts
