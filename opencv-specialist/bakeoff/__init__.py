"""Stage 3 baseline bake-off: the same tasks, graders and metrics for every rung of the ladder.

    python -m bakeoff.run generate --run NAME --rung r1-7b ...   Ollama replies -> generations.jsonl
    python -m bakeoff.run grade    --run NAME                    sandbox grading -> grades.jsonl
    python -m bakeoff.run report   --run NAME                    tables, statistics, error taxonomy
    python -m bakeoff.run predict  --tok-s 40 --tokens 300       predicted GPU hours (CLAUDE.md Rule 1)

Generation and grading are separate commands on purpose: generation holds the GPU, grading is CPU-bound
sandbox work, and the two can overlap (``grade --follow``) or run one after the other. Model code is only ever
run by the existing graders, which go through ``sandbox.run_solution`` (CLAUDE.md Rule 4).

Modules: rungs (the ladder and the exact request), tasks (split rows -> prompts), extract (reply -> code),
generate (Ollama loop), grade (process pool), taxonomy (why a grade failed), stats (pass@k, SEs, McNemar),
report (summary + markdown), predict, store (run folder, JSONL), run (CLI).
"""
