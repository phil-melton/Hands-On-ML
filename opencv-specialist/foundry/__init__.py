"""Stage 4 data foundry: teacher-written training data, everything around the GPU job.

    python -m foundry.run specs [--full]                      the pilot's specs, or the sized full-generation specs, and their invariants
    python -m foundry.run prepare --run NAME                  pilot: specs.jsonl + prompts.jsonl (teacher messages) in the run folder
    python -m foundry.run prepare-phrasings  --run NAME       prompt rows asking the teacher for new task phrasings
    python -m foundry.run prepare-think-test --run NAME       the same specs asked with and without thinking
    python -m foundry.run prepare-full       --run NAME       the full generation (drills included, k per spec, thinking per family)
    python -m foundry.run predict --run NAME ...              predicted minutes and CU for the LOGBOOK (CLAUDE.md Rule 1)
    python -m foundry.run colab   --run NAME --stage generate --dry-run              print the Colab CLI commands
    python -m foundry.run colab   --run NAME --stage generate --i-have-approval ...  the billed job (needs Phil's OK)
    python -m foundry.run verify  --run NAME                  CPU: extract the answers, grade, assign tiers; thinking vs non-thinking
    python -m foundry.run screen-phrasings --run NAME         parse and screen the teacher's phrasings -> foundry/phrasings/<family>.json
    python -m foundry.run pair    --run NAME                  repair examples by pairing a failed sample with its spec's verified answer
    python -m foundry.run report  --run NAME                  yield table, thinking vs non-thinking, cost per Tier-A example, examples.jsonl
    python -m foundry.run build-dataset --run NAME ...        data/train.jsonl, data/dev.jsonl, manifest.json (Gate 4a mix)
    python -m foundry.run tokens                              token lengths of those files under the Qwen3.5-9B template

The only step that needs a GPU is the teacher job. ``colab_job.py`` is that job (standalone, vLLM, no imports from this
repo) and ``colab.py`` drives it through the Colab CLI; one job (one model load, about 1 CU on an A100) can serve prompts of
several kinds with their own sampling settings (``session.py``), and nothing in this package launches it unless
``--i-have-approval`` is given. A run lives in ~/opencv-expert/foundry/<name>/ (outside git, CLAUDE.md Rule 8):

    config.json            what the prepare commands were asked for (n specs, notes file hash, system prompts hashes, plans)
    specs.jsonl            one row per spec: id, family (or "drills"), phrasing index, scene seed, prompt, prompt hash
    prompts.jsonl          one row per request: the teacher's messages and its sampling overrides (what colab_job.py reads)
    completions.jsonl      colab_job.py output: raw text, thinking, answer, token counts of each, finish reason
    timing.json            load time, generation time, output tokens per second, tokens per group (the cost basis)
    grades.jsonl           verify: one row per completion (extracted code, gates, tier, failure message)
    kept.jsonl             verify: the Tier-A solutions kept (at most 2 per spec, AST-deduplicated)
    pairing_pairs.jsonl    pair: one repair example per spec
    repair_prompts.jsonl, repair_completions.jsonl, ...   the older teacher repair round (not used since Gate 4a)
    session_<stage>.json   the Colab driver's record: commands, usage before and after, prediction
    report.md, summary.json, examples.jsonl

Model code is only ever run by the existing graders, which go through ``sandbox.run_solution`` (CLAUDE.md Rule 4).
Modules: specs (pilot, full and think-test specs, invariants), phrasings (teacher-written task wordings and their screen),
session (prompt rows of a Colab session), store, astnorm (dedup key), answer (the training answer format), colab_job (GPU job),
colab (driver), verify, repair (pairing), report, dataset (the training set), run (CLI). ``foundry.drills`` (training-only
tasks on the measured 5.0 changes) is written elsewhere and consumed by specs and verify.
"""
