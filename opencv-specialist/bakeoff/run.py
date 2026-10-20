"""Command line for the Stage 3 bake-off. Run from opencv-specialist/ with the cv5 venv:

    python -m bakeoff.run predict  --tok-s 40 --tokens 300                       # write this into LOGBOOK.md first
    python -m bakeoff.run generate --run main --rung r1-7b --rung r2-4b --rung r3-9b \\
                                   --rung r4-7b-notes --notes data/notes.md        # GPU; resumable
    python -m bakeoff.run grade    --run main --workers 6 [--follow]             # CPU; resumable; can overlap generate
    python -m bakeoff.run report   --run main                                    # report.md, summary.json, audit.md

A run lives in ~/opencv-expert/bakeoff/<name>/ (``--root`` changes the parent). Tasks default to the 198 test tasks
plus the 40 gotchas; ``--split dev`` gives the 99 dev tasks, and ``--split`` can be repeated. For a smoke test add
``--per-family 1`` or ``--only REGEX``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bakeoff.grade import GradeSettings, grade_run
from bakeoff.predict import measured, parse_per_rung, predict, render
from bakeoff.rungs import RUNGS
from bakeoff.store import DEFAULT_ROOT, RunDir
from bakeoff.tasks import SPLIT_CHOICES, load_tasks, split_sizes
from train.config import OLLAMA


def _rung_names(values: list[str]) -> list[str]:
    names = [n for v in values for n in v.split(",") if n]
    unknown = [n for n in names if n not in RUNGS]
    if unknown:
        raise SystemExit(f"unknown rung(s) {unknown}; known: {list(RUNGS)}")
    return list(dict.fromkeys(names))


def _run_dir(a) -> RunDir:
    gen = Path(a.generations) if getattr(a, "generations", None) else None
    if a.run:
        return RunDir(Path(a.root or DEFAULT_ROOT) / a.run, generations_file=gen)
    if gen:
        return RunDir(gen.resolve().parent, generations_file=gen)
    raise SystemExit("give --run NAME (or --generations FILE)")


def cmd_generate(a) -> int:
    from bakeoff.generate import GenSettings, run_generation
    rungs = [RUNGS[n] for n in _rung_names(a.rung)]
    notes_text = Path(a.notes).read_text(encoding="utf-8") if a.notes else None
    if any(r.notes for r in rungs) and not notes_text:
        raise SystemExit("a notes rung is selected: pass --notes PATH")
    if notes_text:
        from bakeoff.rungs import NOTES_TOKEN_BUDGET, notes_token_estimate
        est = notes_token_estimate(notes_text)
        if est > NOTES_TOKEN_BUDGET:
            print(f"warning: the notes block is about {est} tokens (3 chars/token estimate), over the {NOTES_TOKEN_BUDGET} budget; "
                  "the report shows the measured size", file=sys.stderr)
    tasks = load_tasks(a.split or ["test", "gotchas"], only=a.only, per_family=a.per_family)
    st = GenSettings(base_url=a.ollama, samples=a.samples, num_predict=a.num_predict, num_ctx=a.num_ctx,
                     keep_alive=a.keep_alive, http_timeout=a.http_timeout, dedupe_greedy=a.dedupe_greedy)
    run = RunDir.of(a.run, a.root)
    inv = {"rungs": [r.name for r in rungs], "splits": a.split or ["test", "gotchas"], "only": a.only, "per_family": a.per_family,
           "samples": a.samples, "n_tasks": len(tasks), "dedupe_greedy": a.dedupe_greedy}
    written = run_generation(run, rungs, tasks, st, notes_text=notes_text, invocation=inv)
    print(f"generated {written} into {run.generations}")
    return 0


def cmd_grade(a) -> int:
    run = _run_dir(a)
    counts = grade_run(run, GradeSettings(workers=a.workers, n_scenes=a.n_scenes, n_meta=a.n_meta, timeout_s=a.timeout),
                       follow=a.follow, poll_s=a.poll)
    print(f"graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused, "
          f"{counts['grader_errors']} grader errors) into {run.grades}")
    return 1 if counts["grader_errors"] else 0


def cmd_report(a) -> int:
    from bakeoff.report import write_report
    run = _run_dir(a)
    s = write_report(run, k=a.k, boot=a.bootstrap, seed=a.seed, audit_k=a.audit_k)
    for rung in s["rungs"]:
        b = s["groups"]["all"]["rungs"][rung]
        print(f"{rung:<14} pass@1 {100 * b['pass1']:.1f} ± {100 * b['se1']:.1f}"
              + (f"   pass@{s['k']} {100 * b['passk']:.1f} ± {100 * b['sek']:.1f}" if b["passk"] is not None else ""))
    print(f"wrote {run.report_md}, {run.summary}, {run.audit_md}")
    return 0


def cmd_predict(a) -> int:
    rungs = _rung_names(a.rung) if a.rung else list(RUNGS)
    fb_tps, fb_tok = measured(RunDir.of(a.from_run, a.root)) if a.from_run else ({}, {})
    tok_s = parse_per_rung(a.tok_s, rungs, "--tok-s", fb_tps)
    tokens = parse_per_rung(a.tokens, rungs, "--tokens", fb_tok)
    n_tasks = a.tasks or split_sizes(a.split or ["test", "gotchas"])
    print(render(predict(rungs, tok_s, tokens, n_tasks, a.samples, overhead_s=a.overhead_s, load_s=a.load_s),
                 n_tasks, a.samples, a.overhead_s, a.load_s))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bakeoff.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run(p, required_name: bool):
        p.add_argument("--run", required=required_name, help="run name (folder under --root)")
        p.add_argument("--root", help=f"parent folder of runs (default {DEFAULT_ROOT})")

    g = sub.add_parser("generate", help="ask the Ollama models every task (GPU, resumable)")
    add_run(g, True)
    g.add_argument("--rung", action="append", required=True, help=f"rung name(s), repeatable or comma separated; known: {list(RUNGS)}")
    g.add_argument("--split", action="append", choices=SPLIT_CHOICES, help="default: test and gotchas")
    g.add_argument("--only", help="regex searched in task ids")
    g.add_argument("--per-family", type=int, help="first K tasks of each family (smoke test)")
    g.add_argument("--notes", help="file with the OpenCV 5.0 notes block (r4-7b-notes)")
    g.add_argument("--samples", type=int, default=5, help="sampled replies per task at T=0.8 (0: greedy only)")
    g.add_argument("--num-predict", type=int, default=1024)
    g.add_argument("--num-ctx", type=int, default=4096)
    g.add_argument("--keep-alive", default="30m")
    g.add_argument("--ollama", default=OLLAMA)
    g.add_argument("--http-timeout", type=float, default=900.0)
    g.add_argument("--dedupe-greedy", action="store_true",
                   help="reuse one greedy reply for tasks with an identical prompt (they would get the same reply at T=0)")
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("grade", help="grade the replies in the sandbox (CPU, resumable)")
    add_run(r, False)
    r.add_argument("--generations", help="a generations.jsonl elsewhere (its folder becomes the run folder if --run is absent)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--n-scenes", type=int, default=20)
    r.add_argument("--n-meta", type=int, default=5)
    r.add_argument("--timeout", type=float, help="seconds per sandbox run (default: 60 families, 30 gotchas)")
    r.add_argument("--follow", action="store_true", help="keep grading while generate is still running")
    r.add_argument("--poll", type=float, default=10.0)
    r.set_defaults(fn=cmd_grade)

    p = sub.add_parser("report", help="write report.md, summary.json and audit.md")
    add_run(p, False)
    p.add_argument("--generations")
    p.add_argument("--k", type=int, default=5, help="k of pass@k")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--audit-k", type=int, default=3, help="failures per class in audit.md")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_report)

    q = sub.add_parser("predict", help="predicted generation hours per rung from a decode rate and answer length")
    q.add_argument("--rung", action="append")
    q.add_argument("--tok-s", action="append", help="decode tokens/s: a number for all rungs, or rung=number; repeatable")
    q.add_argument("--tokens", action="append", help="expected tokens per answer: a number, or rung=number")
    q.add_argument("--from-run", help="take tok/s and tokens per answer from this finished run's summary.json")
    q.add_argument("--root")
    q.add_argument("--split", action="append", choices=SPLIT_CHOICES)
    q.add_argument("--tasks", type=int, help="override the task count")
    q.add_argument("--samples", type=int, default=5)
    q.add_argument("--overhead-s", type=float, default=0.5)
    q.add_argument("--load-s", type=float, default=90.0)
    q.set_defaults(fn=cmd_predict)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
