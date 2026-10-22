"""Stage 4 data foundry: the teacher pilot, everything around the GPU job.

    python -m foundry.run specs                          the pilot specs, their invariants, a table per family
    python -m foundry.run prepare --run NAME             specs.jsonl + prompts.jsonl (teacher messages) in the run folder
    python -m foundry.run predict --run NAME ...         predicted minutes and CU for the LOGBOOK (CLAUDE.md Rule 1)
    python -m foundry.run colab   --run NAME --stage generate --dry-run        print the Colab CLI commands
    python -m foundry.run colab   --run NAME --stage generate --i-have-approval ...   the billed job (needs Phil's OK)
    python -m foundry.run verify  --run NAME [--stage generate|repair]         CPU: extract, grade, assign tiers
    python -m foundry.run repair  --run NAME             repair_prompts.jsonl from the failed samples
    python -m foundry.run report  --run NAME             yield table, cost per Tier-A example, examples.jsonl

The only step that needs a GPU is the teacher job. ``colab_job.py`` is that job (standalone, vLLM, no imports from this
repo) and ``colab.py`` drives it through the Colab CLI; both stages (generate, repair) use the same job script, and
nothing in this package launches it unless ``--i-have-approval`` is given. A run lives in ~/opencv-expert/foundry/<name>/
(outside git, CLAUDE.md Rule 8):

    config.json            what prepare was asked for (n specs, notes file hash, system prompts hashes)
    specs.jsonl            one row per spec: id, family, phrasing index, scene seed, prompt, prompt hash
    prompts.jsonl          one row per spec: the teacher's messages (what colab_job.py reads)
    completions.jsonl      colab_job.py output: spec id, sample index, text, token counts, finish reason
    timing.json            load time, generation time, output tokens per second (the cost basis)
    grades.jsonl           verify: one row per completion (extracted code, gates, tier, failure message)
    kept.jsonl             verify: the Tier-A solutions kept (at most 2 per spec, AST-deduplicated)
    repair_prompts.jsonl   repair: the original messages + failed code + the grader's message
    repair_completions.jsonl, repair_timing.json, repair_grades.jsonl, repair_pairs.jsonl
    session_<stage>.json   the Colab driver's record: commands, usage before and after, prediction
    report.md, summary.json, examples.jsonl

Model code is only ever run by the existing graders, which go through ``sandbox.run_solution`` (CLAUDE.md Rule 4).
Modules: specs, store, astnorm (dedup key), colab_job (GPU job), colab (driver), verify, repair, report, run (CLI).
"""
