"""Repair examples. Gate 4a decision 3: they come from PAIRING (``build_pairing_pairs``, below), not from teacher repair rounds.

Pairing. A spec that has at least one Tier-A sample (a verified answer) and at least one failed sample with a grader message
gives one repair example: the task prompt, the failed code as the assistant's first turn, the grader's message asking for a fix
as the user's second turn, and the verified answer as the final turn. At most one per spec, so a spec that failed in five ways
teaches one lesson once. Failures of class convention or api are preferred (the measured 5.0 lessons: shapes, removed or
changed calls), then the earliest sample. Not used as the failed turn: an empty reply, one cut off by the token budget, a grader
crash, a reply that is prose. No teacher call is made; the examples cost nothing beyond the generation that produced the
samples. Both assistant turns are in the training answer format (foundry.answer.format_answer). Whoever trains on these must
compute the loss on the last assistant turn only: the first one is a wrong answer.

The older teacher repair round (a second GPU job that asks the teacher to fix its own failures) is kept below; the
Gate 4a decision is not to use it.

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

from foundry.answer import format_answer, row_problems
from foundry.store import FoundryRun, read_jsonl, teacher_info
from foundry.verify import select_examples
from train.config import SYSTEM

FEEDBACK = "That failed: {message}\nReturn the corrected full function."
PREFERRED_CLASSES = ("convention", "api")   # the 5.0 lessons: pair these failures first
MAX_MESSAGE_CHARS = 800


def fenced(code: str) -> str:
    return "```python\n" + code.rstrip("\n") + "\n```"


def feedback_text(message: str) -> str:
    return FEEDBACK.format(message=message)


def _usable_failure(g: dict) -> bool:
    """A failed grade row that can be the first assistant turn of a repair example."""
    f = g.get("failure")
    return bool(g["tier"] != "A" and f and f.get("repairable") and f.get("message") and g.get("code", "").strip()
                and g.get("finish_reason") in (None, "stop") and g.get("extract") != "fence-unclosed" and not g.get("grader_error"))


def build_pairing_pairs(run: FoundryRun, *, preferred: tuple = PREFERRED_CLASSES) -> list[dict]:
    """One repair example per spec that has a verified answer and a usable failed sample (module docstring), ordered by spec id.

    Each pair: ``messages`` (training SYSTEM, the task prompt, the failed code, the grader's message, the verified code; both
    code turns fenced), the pieces to rebuild it, and the provenance of both samples."""
    grades = read_jsonl(run.stage("generate").grades)
    if not grades:
        raise SystemExit(f"{run.stage('generate').grades} is missing or empty: run `verify` first")
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    teacher = teacher_info(run)
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for g in grades:
        by_spec[g["spec_id"]].append(g)
    pairs = []
    for spec_id in sorted(by_spec):
        rows = sorted(by_spec[spec_id], key=lambda g: (g["sample"], g["id"]))
        good = next((g for g in rows if g["tier"] == "A" and not row_problems(g)), None)
        if good is None:
            continue
        fails = [g for g in rows if _usable_failure(g) and g["ast_sha1"] != good["ast_sha1"]]
        if not fails:
            continue
        bad = min(fails, key=lambda g: (0 if g["failure"]["class"] in preferred else 1, g["sample"], g["id"]))
        message = " ".join(bad["failure"]["message"].split())
        if len(message) > MAX_MESSAGE_CHARS:
            message = message[:MAX_MESSAGE_CHARS - 3] + "..."
        fb = feedback_text(message)
        spec = specs[spec_id]
        pairs.append({
            "id": f"{spec_id}/pair", "spec_id": spec_id, "family": bad["family"], "kind": "repair", "tier": "A",
            "failure": {k: bad["failure"][k] for k in ("gate", "class", "rule", "message")}, "failed_code": bad["code"],
            "failed_ast_sha1": bad["ast_sha1"], "failed_sample": bad["sample"], "feedback": fb, "fixed_code": good["code"],
            "fixed_ast_sha1": good["ast_sha1"], "fixed_sample": good["sample"],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec["prompt"]},
                         {"role": "assistant", "content": format_answer(bad["code"])}, {"role": "user", "content": fb},
                         {"role": "assistant", "content": format_answer(good["code"])}],
            "provenance": {"teacher": teacher, "gates": {k: v["passed"] for k, v in (good.get("gates") or {}).items()},
                           "stage": good.get("stage"), "kind": good.get("kind"), "scene_seed": spec["seed"],
                           "phrasing_index": spec["phrasing_index"], "prompt_sha1": spec.get("prompt_sha1"),
                           "thinking": (good.get("thinking_tokens") or 0) > 0,
                           "completion_tokens": good.get("completion_tokens")}})
    return pairs


def write_pairing_pairs(run: FoundryRun) -> int:
    pairs = build_pairing_pairs(run)
    with (run.path / "pairing_pairs.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in pairs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(pairs)


def build_repair_prompts(run: FoundryRun, *, max_per_spec: int = 2, include_solved: bool = False, cap: int = 2) -> list[dict]:
    original: dict[str, dict] = {}
    for r in read_jsonl(run.stage("generate").prompts):
        if r.get("kind", "generate") == "generate":
            original.setdefault(r.get("spec_id", r["id"]), r)   # a spec asked in two modes keeps its first prompt row
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
