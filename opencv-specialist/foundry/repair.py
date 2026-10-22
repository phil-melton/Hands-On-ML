"""Repair: failed samples + the grader's message -> a second teacher job -> repair pairs (PLAN.md 4.1 "keep the scrap").

``build_repair_prompts`` writes ``repair_prompts.jsonl`` in the format colab_job.py reads. For one failed sample the
conversation is the teacher's original messages, then the failed code as the assistant turn, then a user turn:

    That failed: <the grader's message>
    Return the corrected full function.

The wording is PLAN.md 8.4's serve-time repair turn, so the model is trained on the feedback it will get in production
(CLAUDE.md Rule 7). The message is the first failing gate's text (exec-only, lint, manufactured, metamorphic) with the
scene seed removed, and a removed-in-5.0 hint when the api report found one (foundry.verify.failure_of).

Which failures are sent: per spec, the distinct (by AST) non-Tier-A samples with code and without a grader crash, in
sample order, at most ``max_per_spec`` (default 2); specs that already kept ``cap`` Tier-A solutions are skipped unless
``include_solved``. The teacher is asked once (k = 1 by default, ``colab --stage repair``).

``write_repair_pairs`` runs after ``verify --stage repair``: a pair is a repair completion that reached Tier A (or B, flagged)
with code different from the failed code. Stored apart from the examples: ``messages`` is the candidate chat (training
SYSTEM, the task prompt, the failed code, the feedback, the fixed code) and the pair keeps the pieces to rebuild any format.
"""
from __future__ import annotations

import json
from collections import defaultdict

from foundry.store import FoundryRun, read_jsonl, teacher_info
from foundry.verify import select_examples
from train.config import SYSTEM

FEEDBACK = "That failed: {message}\nReturn the corrected full function."


def fenced(code: str) -> str:
    return "```python\n" + code.rstrip("\n") + "\n```"


def feedback_text(message: str) -> str:
    return FEEDBACK.format(message=message)


def build_repair_prompts(run: FoundryRun, *, max_per_spec: int = 2, include_solved: bool = False, cap: int = 2) -> list[dict]:
    original = {r["id"]: r for r in read_jsonl(run.stage("generate").prompts)}
    grades = read_jsonl(run.stage("generate").grades)
    if not grades:
        raise SystemExit(f"{run.stage('generate').grades} is missing or empty: run `verify` first")
    kept, _ = select_examples(grades, cap)
    solved = {r["spec_id"] for r in kept if r["rank"] == cap - 1}
    tier_a_ast = defaultdict(set)
    for r in grades:
        if r["tier"] == "A":
            tier_a_ast[r["spec_id"]].add(r["ast_sha1"])
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for r in grades:
        by_spec[r["spec_id"]].append(r)
    rows = []
    for spec_id in sorted(by_spec):
        if spec_id in solved and not include_solved:
            continue
        if spec_id not in original:
            raise SystemExit(f"grades name spec {spec_id!r}, which is not in {run.stage('generate').prompts}")
        seen: set[str] = set()
        n = 0
        for r in sorted(by_spec[spec_id], key=lambda r: (r["sample"], r["id"])):
            fail = r.get("failure")
            if r["tier"] == "A" or not fail or not fail["repairable"] or not r["code"].strip():
                continue
            if r["ast_sha1"] in seen or r["ast_sha1"] in tier_a_ast[spec_id]:
                continue
            seen.add(r["ast_sha1"])
            if n >= max_per_spec:
                break
            n += 1
            fb = feedback_text(fail["message"])
            rows.append({"id": f"{spec_id}/s{r['sample']}", "spec_id": spec_id, "family": r["family"], "kind": "repair",
                         "sample_of": r["sample"], "tier_before": r["tier"], "failure": fail, "failed_code": r["code"],
                         "failed_ast_sha1": r["ast_sha1"], "feedback": fb,
                         "messages": [*original[spec_id]["messages"], {"role": "assistant", "content": fenced(r["code"])},
                                      {"role": "user", "content": fb}]})
    return rows


def write_repair_prompts(run: FoundryRun, rows: list[dict]) -> None:
    path = run.stage("repair").prompts
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def build_repair_pairs(run: FoundryRun) -> list[dict]:
    st = run.stage("repair")
    prompts = {r["id"]: r for r in read_jsonl(st.prompts)}
    texts = {(r["id"], r["sample"]): r for r in read_jsonl(st.completions)}
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    teacher = teacher_info(run, "repair")
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for g in read_jsonl(st.grades):
        if g["tier"] in ("A", "B") and g["id"] in prompts:
            by_prompt[g["id"]].append(g)
    pairs = []
    for pid in sorted(by_prompt):
        p = prompts[pid]
        # the best tier wins; within it the first sample whose code differs from the failed code
        for g in sorted(by_prompt[pid], key=lambda g: (g["tier"], g["sample"])):
            if g["ast_sha1"] == p["failed_ast_sha1"]:
                continue
            spec = specs[p["spec_id"]]
            pairs.append({
                "id": f"{pid}#{g['sample']}", "spec_id": p["spec_id"], "family": p["family"], "tier": g["tier"],
                "tier_before": p["tier_before"], "failure": p["failure"], "failed_code": p["failed_code"],
                "feedback": p["feedback"], "fixed_code": g["code"], "fixed_reply": texts[(pid, g["sample"])]["text"],
                "fixed_ast_sha1": g["ast_sha1"], "sample": g["sample"],
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec["prompt"]},
                             {"role": "assistant", "content": p["failed_code"]},
                             {"role": "user", "content": p["feedback"]},
                             {"role": "assistant", "content": g["code"]}],
                "provenance": {"teacher": teacher, "gates": {k: v["passed"] for k, v in (g.get("gates") or {}).items()},
                               "finish_reason": g.get("finish_reason"), "completion_tokens": g.get("completion_tokens")},
            })
            break
    return pairs


def write_repair_pairs(run: FoundryRun) -> int:
    pairs = build_repair_pairs(run)
    with run.repair_pairs.open("w", encoding="utf-8", newline="\n") as fh:
        for r in pairs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(pairs)
