"""Prompt rows of a Colab session: several kinds of request in one prompts.jsonl, so one model load serves all of them.

Loading the 27B teacher costs about ten minutes of A100 time (about 1 CU) whatever it is then asked, so the phrasing requests
and the thinking-mode test run in ONE job. Every prompt row carries an ``origin``:

    phrasings    kind "phrasing": the teacher-written task phrasings (foundry.phrasings), thinking off, short answers
    think-test   kind "generate": a small spec set asked twice, mode "nothink" and mode "think", each with its own settings
    full         kind "generate": the full generation (foundry.specs.full_specs); mode "think" for the families chosen to think
    (none)       the pilot's rows, written by ``prepare``

A prepare command replaces the rows of its own origin and keeps the others; the rows are grouped by origin in a fixed order.
Row ids are unique across origins (``phr-<family>-NN``, ``<spec id>@<mode>``, the spec id). A row may override the job's
sampling settings (colab_job.py); ``mode`` and ``origin`` are copied to its completions, which is how verify and report tell
the modes apart. Completions of kind "phrasing" are read by ``screen-phrasings``, never graded.
"""
from __future__ import annotations

import json

from foundry.store import FoundryRun, read_jsonl

ORIGINS = ("phrasings", "think-test", "full")

# Sampling settings per mode. "nothink" repeats the pilot's (T 0.7, top_p 0.8, 2,048 tokens, no top-k). "think" follows
# the usual recommendation for Qwen3-family thinking (T 0.6, top_p 0.95, top-k 20) and leaves room for the reasoning;
# NOT checked against the model card of Qwen3.8-27B, so the commands pass them explicitly and they can be changed there.
MODE_SETTINGS = {
    "nothink": {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 2048, "temperature": 0.7, "top_p": 0.8},
    "think": {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 12288, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
}


def mode_settings(mode: str, **over) -> dict:
    s = json.loads(json.dumps(MODE_SETTINGS[mode]))
    for key, value in over.items():
        if value is not None:
            s[key] = value
    return s


def spec_rows(specs, *, origin: str, mode: str | None = None, settings: dict | None = None, suffix: bool = False) -> list[dict]:
    """Prompt rows for solution specs. ``suffix``: ids are ``<spec id>@<mode>`` (a spec asked in several modes)."""
    rows = []
    for s in specs:
        row = {"id": f"{s.id}@{mode}" if suffix else s.id, "spec_id": s.id, "family": s.family, "kind": "generate",
               "origin": origin, "messages": s.messages()}
        if mode:
            row["mode"] = mode
        if s.k is not None:
            row["k"] = s.k
        row.update(json.loads(json.dumps(settings or {})))
        rows.append(row)
    return rows


def _norm(rows: list[dict]) -> list[dict]:
    return json.loads(json.dumps(rows, sort_keys=True))


def merge_prompts(run: FoundryRun, origin: str, rows: list[dict], *, force: bool = False) -> int:
    """Put ``rows`` (all of origin ``origin``) into the generate stage's prompts.jsonl, replacing the file's rows of that origin.
    Refuses to change an origin's rows once the run has completions or grades, unless ``force``. Returns the file's row count."""
    if origin not in ORIGINS:
        raise ValueError(f"unknown origin {origin!r}; choose from {ORIGINS}")
    st = run.stage("generate")
    old = read_jsonl(st.prompts)
    mine = [r for r in old if r.get("origin") == origin]
    others = [r for r in old if r.get("origin") != origin]
    if mine and _norm(mine) != _norm(rows) and (st.completions.is_file() or st.grades.is_file()) and not force:
        raise SystemExit(f"{st.prompts} has different {origin!r} prompts and the run already has completions or grades; "
                         "use a new --run name or --force")
    ids = {r["id"] for r in others}
    clash = [r["id"] for r in rows if r["id"] in ids]
    if clash:
        raise SystemExit(f"prompt ids already used by another origin: {clash[:3]}")
    merged = sorted([*others, *rows], key=lambda r: ORIGINS.index(r["origin"]) if r.get("origin") in ORIGINS else -1)
    st.prompts.parent.mkdir(parents=True, exist_ok=True)
    with st.prompts.open("w", encoding="utf-8", newline="\n") as fh:
        for r in merged:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(merged)


def merge_specs(run: FoundryRun, specs, *, force: bool = False) -> int:
    """Add spec rows to specs.jsonl (by id); a spec already there must be identical. Returns the file's row count."""
    old = {r["id"]: r for r in read_jsonl(run.specs)}
    for s in specs:
        row = json.loads(json.dumps(s.to_row()))
        if s.id in old and old[s.id] != row and not force:
            raise SystemExit(f"spec {s.id} is already in {run.specs} with different content")
        old[s.id] = row
    run.specs.parent.mkdir(parents=True, exist_ok=True)
    with run.specs.open("w", encoding="utf-8", newline="\n") as fh:
        for r in old.values():
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(old)


def label_of(row: dict) -> str:
    """The group a prompt belongs to for cost accounting: its mode, else its kind (colab_job.py labels the same way)."""
    return row.get("mode") or row.get("kind") or "generate"
