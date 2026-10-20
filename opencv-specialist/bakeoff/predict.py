"""Predicted generation time per rung: the written prediction CLAUDE.md Rule 1 asks for before every GPU job.

    seconds per reply = tokens per answer / decode tok/s + overhead
    hours per rung    = (tasks x (1 greedy + n samples) x seconds per reply + load and warm-up) / 3600

The decode rate and the answer length are inputs, not guesses made here: take them from a smoke run
(``generate --split dev --per-family 1 ...`` then ``report``) or from the Stage 1 hydrotest; ``--from-run`` reads
them from a finished run's summary.json. ``overhead`` is prompt processing plus HTTP, about half a second for a
400-token prompt; the notes rung's longer system prompt adds to it, so raise it for that rung if needed.
A measurement more than 3x off the prediction is a bug to diagnose before continuing (Rule 1).
"""
from __future__ import annotations

from bakeoff.store import RunDir, read_json


def parse_per_rung(values: list[str], rungs: list[str], what: str, fallback: dict[str, float] | None = None) -> dict[str, float]:
    """['40', 'r2-4b=70'] -> {rung: value}: a bare number is the default, name=value overrides it."""
    default, per = None, {}
    for v in values or []:
        if "=" in v:
            name, x = v.split("=", 1)
            per[name] = float(x)
        else:
            default = float(v)
    out = {}
    for r in rungs:
        if r in per:
            out[r] = per[r]
        elif default is not None:
            out[r] = default
        elif fallback and r in fallback:
            out[r] = fallback[r]
        else:
            raise SystemExit(f"no {what} for rung {r}: give {what} VALUE for all rungs, or {what} {r}=VALUE")
    return out


def measured(run: RunDir) -> tuple[dict[str, float], dict[str, float]]:
    """(tok/s, tokens per answer) per rung from a finished run's summary.json."""
    s = read_json(run.summary)
    if not s:
        raise SystemExit(f"{run.summary} not found: run the report command on that run first")
    tps, toks = {}, {}
    for rung, t in s["throughput"].items():
        if t["decode_tps"]["median"] and t["tokens_per_answer"]["mean"]:
            tps[rung], toks[rung] = t["decode_tps"]["median"], t["tokens_per_answer"]["mean"]
    return tps, toks


def predict(rungs: list[str], tok_s: dict[str, float], tokens: dict[str, float], n_tasks: int, samples: int, *,
            overhead_s: float = 0.5, load_s: float = 90.0) -> list[dict]:
    rows = []
    for r in rungs:
        replies = n_tasks * (1 + samples)
        per = tokens[r] / tok_s[r] + overhead_s
        rows.append({"rung": r, "replies": replies, "tok_s": tok_s[r], "tokens": tokens[r], "s_per_reply": per,
                     "minutes": (replies * per + load_s) / 60, "hours": (replies * per + load_s) / 3600})
    return rows


def render(rows: list[dict], n_tasks: int, samples: int, overhead_s: float, load_s: float) -> str:
    lines = [f"Predicted generation time ({n_tasks} tasks x (1 greedy + {samples} sampled) replies per rung; "
             f"overhead {overhead_s:g} s per reply, {load_s:g} s load + warm-up per rung)", "",
             f"{'rung':<14}{'replies':>8}{'tok/s':>8}{'tok/answer':>12}{'s/reply':>9}{'minutes':>9}{'hours':>7}"]
    for x in rows:
        lines.append(f"{x['rung']:<14}{x['replies']:>8}{x['tok_s']:>8.1f}{x['tokens']:>12.0f}{x['s_per_reply']:>9.1f}"
                     f"{x['minutes']:>9.0f}{x['hours']:>7.2f}")
    lines.append(f"{'total':<14}{sum(x['replies'] for x in rows):>8}{'':>8}{'':>12}{'':>9}"
                 f"{sum(x['minutes'] for x in rows):>9.0f}{sum(x['hours'] for x in rows):>7.2f}")
    lines += ["", "Record this in LOGBOOK.md before the job (Rule 1), then the measured values after it. "
                  "A miss of more than 3x is a bug: diagnose it before continuing."]
    return "\n".join(lines)
