"""Command line for the Stage 4 data foundry. Run from opencv-specialist/ with the cv5 venv:

  The pilot (done 2026-10-22)
    python -m foundry.run specs                                   # the 200 pilot specs, checked; a table per family
    python -m foundry.run prepare --run pilot1                    # specs.jsonl, prompts.jsonl in ~/opencv-expert/foundry/pilot1/
    python -m foundry.run repair  --run pilot1                    # repair_prompts.jsonl from the failures (teacher repair round)

  Session 1: teacher-written phrasings + the thinking-mode test, ONE Colab job (one model load)
    python -m foundry.run prepare-phrasings  --run s1             # prompt rows, kind "phrasing" (origin phrasings)
    python -m foundry.run prepare-think-test --run s1             # the same specs asked with and without thinking (origin think-test)
    python -m foundry.run predict --run s1 --tok-s 600 --tokens-by-label '{"phrasing": 450, "nothink": 800, "think": 4500}'
    python -m foundry.run colab   --run s1 --stage generate --dry-run                       # print the Colab CLI commands
    python -m foundry.run colab   --run s1 --stage generate --i-have-approval ...           # BILLED: needs Phil's OK
    python -m foundry.run verify  --run s1                        # grades the think-test samples; prints thinking vs non-thinking
    python -m foundry.run screen-phrasings --run s1               # parse + screen -> foundry/phrasings/<family>.json (committed)
    python -m foundry.run report  --run s1

  Session 2: the full generation
    python -m foundry.run specs --full --target-kept 2100         # the sized spec set and its invariants, a table per group
    python -m foundry.run prepare-full --run full1 [--think-families hough_segments,text_layout]
    python -m foundry.run colab   --run full1 --stage generate --i-have-approval ...        # BILLED
    python -m foundry.run verify  --run full1 --workers 4         # CPU, resumable
    python -m foundry.run pair    --run full1                     # pairing_pairs.jsonl: repair examples by pairing (no teacher call)
    python -m foundry.run build-dataset --run full1                       # data/train.jsonl, data/dev.jsonl, manifest.json (replay: ~/opencv-expert/replay/replay.jsonl)
    ~/venvs/train/bin/python -m foundry.run tokens                # token lengths under the Qwen3.5-9B template -> manifest.json

  Other
    python -m foundry.run calibrate-phrasings                     # the numbers behind the phrasing screen's thresholds
    python -m foundry.run predict --run NAME --tok-s 600 --tokens 700       # minutes and CU, for LOGBOOK.md first (Rule 1)
    python -m foundry.run report  --run NAME --gpu-rate 5.4       # report.md, summary.json, examples.jsonl

The GPU job is the only billed step and it never starts without ``--i-have-approval`` (CLAUDE.md Rule 2 and the project
budget: Colab needs Phil's explicit OK per stage). ``--root`` changes the parent of run folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from foundry import colab as colab_mod
from foundry.store import DEFAULT_ROOT, FoundryRun, read_json, read_jsonl, teacher_info, write_json
from train.config import SYSTEM


def _run(a) -> FoundryRun:
    return FoundryRun.of(a.run, a.root)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- specs and prepare

def _full_kwargs(a) -> dict:
    try:
        rates = json.loads(a.rates) if a.rates else None
    except json.JSONDecodeError as e:
        raise SystemExit(f"--rates is not JSON: {e}")
    try:
        weights = json.loads(a.weights) if a.weights else None
    except json.JSONDecodeError as e:
        raise SystemExit(f"--weights is not JSON: {e}")
    return {"target_kept": a.target_kept, "rates": rates, "k": a.k, "drill_weight": a.drill_weight, "weights": weights,
            "include_drills": not a.no_drills, "use_generated": not a.no_generated}


def cmd_specs(a) -> int:
    from cvbench.splits import FAMILIES
    from foundry.specs import FOUNDRY_BLOCK, SpecError, full_specs, pilot_specs, sizing, summarize_specs
    try:
        specs = full_specs(**_full_kwargs(a)) if a.full else pilot_specs(a.n)  # raises SpecError on any invariant
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    print(f"{len(specs)} specs, invariants hold (trainable families and drills only, no dev/test/gotcha prompt, phrasings that "
          f"no scored task uses, seed windows inside FOUNDRY_RANGE and disjoint)")
    if a.full:
        kw = _full_kwargs(a)
        plan = sizing(a.target_kept, rates=kw["rates"], drill_weight=a.drill_weight, include_drills=not a.no_drills, weights=kw["weights"])
        print(f"target {a.target_kept} kept examples (cap 2 per spec); the group's kept target / its rate = its specs")
        print(f"{'group':<22}{'specs':>6}{'prompts':>9}{'k':>6}{'samples':>9}{'weight':>8}{'kept target':>12}{'rate':>6}")
        for r in summarize_specs(specs):
            p = plan[r["group"]]
            print(f"{r['group']:<22}{r['specs']:>6}{r['prompts']:>9}{','.join(map(str, r['k'])):>6}{r['samples']:>9}"
                  f"{p['weight']:>8.1f}{p['kept_target']:>12.0f}{p['rate']:>6.2f}")
        print(f"total: {len(specs)} specs, {sum(r['samples'] for r in summarize_specs(specs))} samples, "
              f"{len({s.prompt_sha1 for s in specs})} distinct prompts")
        return 0
    print(f"{'family':<22}{'subset':<10}{'specs':>6}  seeds")
    for f in dict.fromkeys(s.family for s in specs):
        mine = [s for s in specs if s.family == f]
        print(f"{f:<22}{FAMILIES[f].subset:<10}{len(mine):>6}  {mine[0].seed} .. {mine[-1].seed + 19}  "
              f"(block {FOUNDRY_BLOCK}, phrasings {sorted({s.phrasing_index for s in mine})})")
    return 0


def prompt_rows(specs) -> list[dict]:
    rows = [{"id": s.id, "spec_id": s.id, "family": s.family, "kind": "generate", "messages": s.messages()} for s in specs]
    for row, s in zip(rows, specs):
        if s.k is not None:
            row["k"] = s.k
    return rows


def cmd_prepare(a) -> int:
    from foundry.specs import FOUNDRY_BLOCK, NOTES_PATH, pilot_specs
    run = _run(a)
    notes_path = Path(a.notes) if a.notes else NOTES_PATH
    notes = notes_path.read_text(encoding="utf-8")
    specs = pilot_specs(a.n, notes_text=notes)
    prompts = prompt_rows(specs)
    st = run.stage("generate")
    if st.prompts.is_file() and not a.force:
        old = read_jsonl(st.prompts)
        if old != json.loads(json.dumps(prompts, sort_keys=True)) and (st.completions.is_file() or st.grades.is_file()):
            raise SystemExit(f"{st.prompts} exists with different prompts and the run already has completions or grades; "
                             "use a new --run name or --force")
    _write_jsonl(run.specs, [s.to_row() for s in specs])
    _write_jsonl(st.prompts, prompts)
    write_json(run.config, {"n_specs": len(specs), "notes_path": str(notes_path), "notes_sha1": _sha1(notes),
                            "system_sha1": _sha1(SYSTEM), "teacher_system_sha1": _sha1(specs[0].teacher_system),
                            "foundry_block": FOUNDRY_BLOCK, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    print(f"wrote {len(specs)} specs and prompts to {run.path}")
    return 0


def _update_config(run: FoundryRun, **kv) -> None:
    cfg = read_json(run.config, {}) or {}
    cfg.update(kv)
    cfg["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run.config, cfg)


def _notes(a):
    from foundry.specs import NOTES_PATH
    path = Path(a.notes) if a.notes else NOTES_PATH
    return path, path.read_text(encoding="utf-8")


def _settings(a, prefix: str, mode: str) -> dict:
    from foundry.session import mode_settings
    return mode_settings(mode, max_tokens=getattr(a, f"{prefix}_max_tokens"), temperature=getattr(a, f"{prefix}_temperature"),
                         top_p=getattr(a, f"{prefix}_top_p"), top_k=getattr(a, f"{prefix}_top_k"))


def cmd_prepare_full(a) -> int:
    from foundry.session import merge_prompts, merge_specs, spec_rows
    from foundry.specs import SpecError, full_specs, sizing, summarize_specs
    run = _run(a)
    notes_path, notes = _notes(a)
    try:
        specs = full_specs(**_full_kwargs(a), notes_text=notes)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    thinkers = {f for f in (a.think_families or "").split(",") if f}
    unknown = thinkers - {s.family for s in specs}
    if unknown:
        raise SystemExit(f"--think-families names groups with no specs: {sorted(unknown)}")
    think = _settings(a, "think", "think")
    rows = spec_rows([s for s in specs if s.family not in thinkers], origin="full", mode="nothink" if thinkers else None)
    rows += spec_rows([s for s in specs if s.family in thinkers], origin="full", mode="think", settings=think)
    order = {s.id: i for i, s in enumerate(specs)}
    rows.sort(key=lambda r: order[r["id"]])
    n = merge_prompts(run, "full", rows, force=a.force)   # first: it refuses to change a run that has output, before specs.jsonl moves
    merge_specs(run, specs, force=a.force)
    kw = _full_kwargs(a)
    plan = sizing(a.target_kept, rates=kw["rates"], drill_weight=a.drill_weight, include_drills=not a.no_drills, weights=kw["weights"])
    _update_config(run, full={"target_kept": a.target_kept, "n_specs": len(specs), "plan": plan, "think_families": sorted(thinkers),
                              "think_settings": think if thinkers else None, "notes_path": str(notes_path), "notes_sha1": _sha1(notes),
                              "system_sha1": _sha1(SYSTEM), "teacher_system_sha1": _sha1(specs[0].teacher_system),
                              "groups": summarize_specs(specs)})
    samples = sum(r["samples"] for r in summarize_specs(specs))
    print(f"wrote {len(specs)} specs ({samples} samples) to {run.path}; prompts.jsonl now has {n} rows")
    return 0


def cmd_prepare_phrasings(a) -> int:
    from foundry import phrasings as phr
    from foundry.session import merge_prompts
    run = _run(a)
    families = [f for f in (a.families or "").split(",") if f] or None
    rows = phr.build_requests(families, variants=a.variants, per_request=a.per_request, k=a.k, max_tokens=a.max_tokens,
                              temperature=a.temperature, top_p=a.top_p)
    n = merge_prompts(run, "phrasings", rows, force=a.force)
    _update_config(run, phrasings={"variants": a.variants, "per_request": a.per_request, "k": a.k, "max_tokens": a.max_tokens,
                                   "temperature": a.temperature, "top_p": a.top_p, "requests": len(rows),
                                   "families": sorted({r["family"] for r in rows})})
    print(f"wrote {len(rows)} phrasing requests ({len(rows) * a.k} replies of {a.per_request} wordings) to {run.stage('generate').prompts}; "
          f"the file has {n} rows")
    return 0


def cmd_prepare_think_test(a) -> int:
    from foundry.session import merge_prompts, merge_specs, spec_rows
    from foundry.specs import SpecError, think_test_specs
    run = _run(a)
    notes_path, notes = _notes(a)
    try:
        try:
            more = json.loads(a.more) if a.more else None
        except json.JSONDecodeError as e:
            raise SystemExit(f"--more is not JSON: {e}")
        specs = think_test_specs(a.per_family, a.drills, a.k, notes_text=notes, more=more)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    nothink, think = _settings(a, "nothink", "nothink"), _settings(a, "think", "think")
    rows = spec_rows(specs, origin="think-test", mode="nothink", settings=nothink, suffix=True) + \
        spec_rows(specs, origin="think-test", mode="think", settings=think, suffix=True)
    n = merge_prompts(run, "think-test", rows, force=a.force)
    merge_specs(run, specs, force=a.force)
    _update_config(run, think_test={"per_family": a.per_family, "drills": a.drills, "k": a.k, "n_specs": len(specs),
                                    "nothink": nothink, "think": think, "notes_path": str(notes_path), "notes_sha1": _sha1(notes)})
    print(f"wrote {len(specs)} specs x 2 modes x k {a.k} = {len(specs) * 2 * a.k} samples to {run.path}; prompts.jsonl has {n} rows")
    return 0


# ---------------------------------------------------------------- phrasings

def cmd_screen_phrasings(a) -> int:
    from foundry import phrasings as phr
    run = _run(a)
    comps = [c for c in read_jsonl(run.stage("generate").completions) if c.get("kind") == "phrasing"]
    if not comps:
        raise SystemExit(f"{run.stage('generate').completions} has no completions of kind 'phrasing': run the job first")
    teacher = teacher_info(run)
    for fam in phr.trainable():
        cands, cstats = phr.candidates_from(comps, fam)
        if not cstats["replies"]:
            continue
        record = phr.load_record(fam)
        existing = [p["text"] for p in (record or {}).get("phrasings", [])]
        new, rejected, examples = phr.screen_family(fam, cands, existing=existing, max_keep=a.max_keep,
                                                    dup_threshold=a.threshold, run_limit=a.run_limit)
        kept = len(existing) + len(new)
        print(f"{fam:<22} replies {cstats['replies']:>3}  candidates {cstats['candidates']:>4}  kept {kept:>3} "
              f"({len(existing)} from before)  rejected {dict(rejected)}")
        for reason, shown in examples.items():
            for line in shown[:a.show]:
                print(f"    {reason}: {line}")
        if kept < a.max_keep:
            print(f"    WARNING: only {kept} of {a.max_keep}; ask for more requests (prepare-phrasings --variants/--k) and screen again")
        if not a.dry_run:
            phr.write_family(fam, record, new, teacher=teacher, run=run.name, stats=cstats, rejected=rejected,
                             dup_threshold=a.threshold, run_limit=a.run_limit)
    if a.dry_run:
        print("dry run: nothing written")
    return 0


def cmd_calibrate_phrasings(a) -> int:
    from foundry import phrasings as phr
    c = phr.calibrate()
    print(f"{'family':<22}{'pairs':>6}{'max 5-gram Jaccard':>20}{'mean':>8}{'max shared run (words)':>25}")
    for f, v in c["per_family"].items():
        print(f"{f:<22}{v['pairs']:>6}{v['max_jaccard']:>14.3f} ({v['max_jaccard_pair'][0]:>2},{v['max_jaccard_pair'][1]:>2})"
              f"{v['mean_jaccard']:>8.3f}{v['max_run']:>14} ({v['max_run_pair'][0]:>2},{v['max_run_pair'][1]:>2})")
    print(f"largest: Jaccard {c['max_jaccard']:.3f}, run {c['max_run']} words -> threshold {c['recommended_threshold']}, "
          f"run limit {c['recommended_run_limit']}; in force: threshold {phr.DUP_THRESHOLD}, run limit {phr.RUN_LIMIT}")
    return 0 if (c["recommended_threshold"], c["recommended_run_limit"]) == (phr.DUP_THRESHOLD, phr.RUN_LIMIT) else 1


# ---------------------------------------------------------------- predict

def predict_text(n_prompts: int, k: int, tokens: float, tok_s: float, load_s: float, overhead_s: float, rate: float,
                 usd_per_cu: float) -> str:
    out_tokens = n_prompts * k * tokens
    gen_s = out_tokens / tok_s
    total_s = load_s + gen_s + overhead_s
    cu = total_s / 3600.0 * rate
    return (f"prompts {n_prompts} x k {k} = {n_prompts * k} samples, {tokens:.0f} tokens each = {out_tokens:.0f} output tokens\n"
            f"decode {tok_s:.0f} tok/s -> generation {gen_s / 60:.1f} min; model load {load_s / 60:.1f} min; install and download "
            f"{overhead_s / 60:.1f} min\n"
            f"predicted {total_s / 60:.1f} min at {rate} CU/h = {cu:.2f} CU (${cu * usd_per_cu:.2f}); "
            f"kill at 2x = {2 * total_s / 60:.0f} min / {2 * cu:.2f} CU\n"
            f"LOGBOOK line: predicted {tok_s:.0f} tok/s, {total_s / 60:.0f} min, {cu:.2f} CU")


def predict_mix(rows: list[dict], default_k: int, tokens: dict, tok_s: dict, default_tokens: float | None, default_tok_s: float,
                load_s: float, overhead_s: float, rate: float, usd_per_cu: float, group_by: str = "label") -> str:
    """Predicted tokens, minutes and CU of a prompts file that mixes labels (mode, else kind): each label's samples x its mean
    output tokens, decoded at its own tokens/s (reasoning decodes slower: it holds more KV cache per sequence). ``group_by``
    "family": one line per family (``<family>/<mode>`` where the row has a mode), for pricing a full generation family by family."""
    from foundry.session import label_of
    groups: dict[str, dict] = {}
    for r in rows:
        key = label_of(r) if group_by == "label" else (f"{r.get('family', '?')}/{r['mode']}" if r.get("mode") else r.get("family", "?"))
        g = groups.setdefault(key, {"prompts": 0, "samples": 0})
        g["prompts"] += 1
        g["samples"] += int(r.get("k", default_k))
    lines, gen_s, total_tokens = [], 0.0, 0.0
    for label, g in groups.items():
        t = tokens.get(label, default_tokens)
        if t is None:
            raise SystemExit(f"no mean output tokens for label {label!r}: pass --tokens or --tokens-by-label")
        rate_s = tok_s.get(label, default_tok_s)
        out = g["samples"] * t
        secs = out / rate_s
        gen_s += secs
        total_tokens += out
        lines.append(f"  {label:<10} {g['prompts']:>5} prompts, {g['samples']:>6} samples x {t:>6.0f} tokens = {out:>10.0f} output tokens; "
                     f"at {rate_s:.0f} tok/s {secs / 60:>6.1f} min")
    total_s = load_s + gen_s + overhead_s
    cu = total_s / 3600.0 * rate
    return ("\n".join(lines) + f"\ntotal {total_tokens:.0f} output tokens; generation {gen_s / 60:.1f} min; model load {load_s / 60:.1f} min; "
            f"install and download {overhead_s / 60:.1f} min\n"
            f"predicted {total_s / 60:.1f} min at {rate} CU/h = {cu:.2f} CU (${cu * usd_per_cu:.2f}); "
            f"kill at 2x = {2 * total_s / 60:.0f} min / {2 * cu:.2f} CU\n"
            f"LOGBOOK line: predicted {total_tokens / gen_s if gen_s else 0:.0f} tok/s overall, {total_s / 60:.0f} min, {cu:.2f} CU")


def cmd_predict(a) -> int:
    run = _run(a)
    rows = read_jsonl(run.stage(a.stage).prompts)
    if not rows:
        raise SystemExit(f"{run.stage(a.stage).prompts} is missing or empty")
    k = a.k or (4 if a.stage == "generate" else 1)
    if a.tokens_by_label or a.tok_s_by_label or a.group_by == "family" or any("k" in r or "mode" in r or r.get("kind", "generate") != "generate" for r in rows):
        try:
            tokens = json.loads(a.tokens_by_label or "{}")
            speeds = json.loads(a.tok_s_by_label or "{}")
        except json.JSONDecodeError as e:
            raise SystemExit(f"--tokens-by-label / --tok-s-by-label must be JSON objects: {e}")
        print(predict_mix(rows, k, tokens, speeds, a.tokens, a.tok_s, a.load_s, a.overhead_s, a.rate, a.usd_per_cu, a.group_by))
        return 0
    if a.tokens is None:
        raise SystemExit("--tokens (mean output tokens per sample) is required")
    print(predict_text(len(rows), k, a.tokens, a.tok_s, a.load_s, a.overhead_s, a.rate, a.usd_per_cu))
    return 0


# ---------------------------------------------------------------- colab

def _default_colab_bin() -> str:
    return shutil.which("colab") or str(Path.home() / "venvs" / "colab" / "bin" / "colab")


def job_config(a) -> colab_mod.JobConfig:
    try:
        template_kwargs = json.loads(a.chat_template_kwargs)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--chat-template-kwargs is not JSON: {e}")
    return colab_mod.JobConfig(
        run=_run(a), stage=a.stage, gpu=a.gpu, model=a.model, revision=a.revision, quantization=a.quantization, max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization, k=a.k, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
        max_tokens=a.max_tokens, seed=a.seed, chunk=a.chunk, limit=a.limit, enforce_eager=a.enforce_eager,
        max_num_seqs=a.max_num_seqs, trust_remote_code=a.trust_remote_code, chat_template_kwargs=template_kwargs,
        packages=tuple(p for p in a.packages.split(",") if p), install_via=a.install_via, session=a.session,
        predicted_minutes=a.predicted_minutes, predicted_cu=a.predicted_cu, teacher_license=a.teacher_license,
        colab_bin=a.colab_bin or _default_colab_bin(), colab_auth=a.colab_auth, exec_timeout_s=a.exec_timeout)


def cmd_colab(a) -> int:
    cfg = job_config(a)
    try:
        rec = colab_mod.run_job(cfg, approved=a.i_have_approval, dry_run=a.dry_run)
    except colab_mod.ApprovalRequired as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except colab_mod.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except colab_mod.ColabError as e:
        print(f"colab job failed: {e}", file=sys.stderr)
        return 1
    if rec.get("stop_ok") is False:
        return 3
    return 0


# ---------------------------------------------------------------- verify, repair, report

def cmd_verify(a) -> int:
    from foundry import repair as repair_mod
    from foundry.verify import VerifySettings, grade_stage, mode_table, render_mode_table, summarize, write_kept
    run = _run(a)
    counts = grade_stage(run, a.stage, VerifySettings(a.workers, a.n_scenes, a.n_meta, a.timeout))
    print(f"graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused, {counts['grader_errors']} grader "
          f"errors): Tier A {counts['A']}, Tier B {counts['B']}, fail {counts['fail']}")
    rows = read_jsonl(run.stage(a.stage).grades)
    for fam, d in summarize(rows).items():
        print(f"  {fam:<22} A {d['A']:>4}  B {d['B']:>4}  fail {d['fail']:>4}")
    table = mode_table(rows)
    if table:
        print("\nThinking vs non-thinking (Tier-A rate per sample; paired = the specs asked in both modes):")
        print(render_mode_table(table))
        print()
    if a.stage == "generate":
        s = write_kept(run)
        print(f"kept {s['kept']} Tier-A solutions of {s['candidates']} candidates ({s['duplicates']} AST duplicates, "
              f"{s['over_cap']} over the cap) -> {run.kept}")
    else:
        print(f"{repair_mod.write_repair_pairs(run)} repair pairs -> {run.repair_pairs}")
    return 1 if counts["grader_errors"] else 0


def cmd_pair(a) -> int:
    from foundry import repair as repair_mod
    run = _run(a)
    pairs = repair_mod.build_pairing_pairs(run)
    n = repair_mod.write_pairing_pairs(run)
    classes = Counter(p["failure"]["class"] for p in pairs)
    print(f"{n} repair pairs by pairing -> {run.path / 'pairing_pairs.jsonl'}; failure classes {dict(classes)}; "
          f"by family {dict(Counter(p['family'] for p in pairs))}")
    return 0


def cmd_build_dataset(a) -> int:
    from foundry import dataset as ds
    out = Path(a.out) if a.out else ds.REPO_DATA
    replay = Path(a.replay) if a.replay else (None if a.no_replay else ds.default_replay())
    if replay is None and not a.no_replay:
        print(f"warning: no replay file ({ds.DEFAULT_REPLAY} is missing and --replay was not given): the set has no replay rows", file=sys.stderr)
    m = ds.build_dataset(a.run, root=a.root, replay=replay, out_dir=out, total=None if a.all else a.total,
                         shares=ds.parse_shares(a.shares), dev_frac=a.dev_frac, seed=a.seed, cap=a.cap, base=a.base, tokens=a.tokens,
                         dry_run=a.dry_run, keep_shares=a.keep_shares)
    print(ds.render_summary(m))
    print(("dry run: nothing written" if a.dry_run else f"wrote {out / 'train.jsonl'}, {out / 'dev.jsonl'}, {out / 'manifest.json'}"))
    short = [k for k, v in m["stages"]["mix"].items() if v.get("shortfall")]
    return 3 if short else 0


def cmd_tokens(a) -> int:
    from foundry import dataset as ds
    data = Path(a.data) if a.data else ds.REPO_DATA
    rep = ds.tokens_for_files(data, a.base)
    for name, r in rep.items():
        if r:
            print(f"{name}: n {r['n']} mean {r['mean']} p50 {r['p50']} p99 {r['p99']} max {r['max']}; template mismatches "
                  f"{r['template_mismatches']}; rows without exactly one response marker {r['rows_without_exactly_one_response_marker']}; "
                  f"by kind (n, p99, max) { {k: (v['n'], v['p99'], v['max']) for k, v in r['by_kind'].items()} }")
    return 1 if any(r and (r["template_mismatches"] or r["rows_without_exactly_one_response_marker"]) for r in rep.values()) else 0


def cmd_repair(a) -> int:
    from foundry import repair as repair_mod
    run = _run(a)
    rows = repair_mod.build_repair_prompts(run, max_per_spec=a.max_per_spec, include_solved=a.include_solved)
    repair_mod.write_repair_prompts(run, rows)
    print(f"wrote {len(rows)} repair prompts to {run.stage('repair').prompts}")
    return 0


def cmd_report(a) -> int:
    from foundry import report as report_mod
    run = _run(a)
    s = report_mod.summarize_run(run, gpu_rate=a.gpu_rate, usd_per_cu=a.usd_per_cu, overhead_s=a.overhead_s)
    md = report_mod.render_markdown(s)
    run.report_md.write_text(md, encoding="utf-8")
    write_json(run.summary, s)
    n = report_mod.write_examples(run, answer_format=a.answer_format, include_tier_b=not a.no_tier_b,
                                  cross_spec_dedup=not a.keep_cross_spec_duplicates)
    print(md)
    print(f"wrote {run.report_md}, {run.summary} and {n} examples to {run.examples}")
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m foundry.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_run(p):
        p.add_argument("--run", required=True, help="run name (a folder under --root)")
        p.add_argument("--root", help=f"parent folder of runs (default {DEFAULT_ROOT})")

    def add_full(p):
        p.add_argument("--target-kept", type=int, default=2100, help="verified (Tier-A) solutions to size the run for")
        p.add_argument("--rates", help='JSON {group: kept examples per spec} overriding the defaults (the pilot\'s, foundry.specs)')
        p.add_argument("--drill-weight", type=float, default=3.0, help="family-equivalents of kept examples the drills get as a group")
        p.add_argument("--weights", help='JSON {group: weight} for the share of the target a group gets (default 1.0 per family)')
        p.add_argument("-k", "--k", type=int, help="samples per spec for every group (default 4; 8 for hough_segments, text_layout)")
        p.add_argument("--no-drills", action="store_true", help="families only (foundry.drills not needed)")
        p.add_argument("--no-generated", action="store_true", help="foundry phrasings 12-15 only, no teacher-written ones")

    def add_mode(p, prefix: str, mode: str):
        d = {"think": (12288, 0.6, 0.95, 20), "nothink": (2048, 0.7, 0.8, 0)}[mode]
        p.add_argument(f"--{prefix}-max-tokens", type=int, default=d[0])
        p.add_argument(f"--{prefix}-temperature", type=float, default=d[1])
        p.add_argument(f"--{prefix}-top-p", type=float, default=d[2])
        p.add_argument(f"--{prefix}-top-k", type=int, default=d[3])

    p = sub.add_parser("specs", help="build and check the pilot specs (or --full: the sized full-generation specs); print a table")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--full", action="store_true", help="the full-generation specs of foundry.specs.full_specs")
    add_full(p)
    p.set_defaults(fn=cmd_specs)

    p = sub.add_parser("prepare", help="write specs.jsonl and prompts.jsonl (the pilot's 200 specs)")
    add_run(p)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--notes", help="the OpenCV 5.0 notes file (default bakeoff/notes_5_0.md)")
    p.add_argument("--force", action="store_true", help="overwrite prompts even if the run has completions")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("prepare-full", help="specs.jsonl and prompts.jsonl of the full generation (origin full)")
    add_run(p)
    add_full(p)
    p.add_argument("--think-families", help="comma-separated groups (families, or drills) whose rows think; the rest do not")
    add_mode(p, "think", "think")
    p.add_argument("--notes", help="the OpenCV 5.0 notes file (default bakeoff/notes_5_0.md)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare_full)

    p = sub.add_parser("prepare-phrasings", help="prompt rows asking the teacher for new task phrasings (origin phrasings)")
    add_run(p)
    p.add_argument("--families", help="comma-separated trainable families (default all 7)")
    p.add_argument("--variants", type=int, default=8, help="requests per family, each in another register (at most 8 differ)")
    p.add_argument("--per-request", type=int, default=6, help="wordings asked for per reply")
    p.add_argument("-k", "--k", type=int, default=3, help="replies sampled per request")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare_phrasings)

    p = sub.add_parser("prepare-think-test", help="the same specs asked with and without thinking (origin think-test)")
    add_run(p)
    p.add_argument("--per-family", type=int, default=8, help="specs per trainable family")
    p.add_argument("--more", help="JSON {family: specs} for families that get more than --per-family")
    p.add_argument("--drills", type=int, default=16, help="drills to include (needs foundry.drills; 0: none)")
    p.add_argument("-k", "--k", type=int, default=4, help="samples per spec and mode")
    add_mode(p, "think", "think")
    add_mode(p, "nothink", "nothink")
    p.add_argument("--notes", help="the OpenCV 5.0 notes file (default bakeoff/notes_5_0.md)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare_think_test)

    p = sub.add_parser("screen-phrasings", help="parse and screen the teacher's phrasings -> foundry/phrasings/<family>.json")
    add_run(p)
    from foundry import phrasings as _phr
    p.add_argument("--max-keep", type=int, default=_phr.MAX_PER_FAMILY)
    p.add_argument("--threshold", type=float, default=_phr.DUP_THRESHOLD, help="word 5-gram Jaccard at or above which two texts are near-duplicates")
    p.add_argument("--run-limit", type=int, default=_phr.RUN_LIMIT, help="shared run of this many words or more is an overlap")
    p.add_argument("--show", type=int, default=1, help="rejected examples to show per reason")
    p.add_argument("--dry-run", action="store_true", help="screen and report, write no file")
    p.set_defaults(fn=cmd_screen_phrasings)

    p = sub.add_parser("calibrate-phrasings", help="similarity of the 16 hand-written phrasings; the thresholds in force")
    p.set_defaults(fn=cmd_calibrate_phrasings)

    p = sub.add_parser("pair", help="repair examples by pairing a failed sample with its spec's verified answer")
    add_run(p)
    p.set_defaults(fn=cmd_pair)

    p = sub.add_parser("build-dataset", help="verified runs + replay -> data/train.jsonl, data/dev.jsonl, manifest.json")
    p.add_argument("--run", action="append", required=True, help="a verified run folder name; repeat for several")
    p.add_argument("--root", help=f"parent folder of runs (default {DEFAULT_ROOT})")
    p.add_argument("--replay", help="JSONL of general-code chats: messages, source, license (default ~/opencv-expert/replay/replay.jsonl, "
                                    "600 rows of nvidia/OpenCodeInstruct, CC-BY-4.0)")
    p.add_argument("--no-replay", action="store_true", help="leave replay out (the default file is used otherwise)")
    p.add_argument("--out", help="output folder (default <repo>/data)")
    p.add_argument("--total", type=int, default=3000)
    p.add_argument("--all", action="store_true", help="no mix target: keep every example that survives decontamination and dedup")
    p.add_argument("--shares", help="solutions,repairs,replay shares summing to 1 (default 0.70,0.20,0.10)")
    p.add_argument("--keep-shares", action="store_true", help="when a kind is short, shrink the total so the mix keeps its shares")
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap", type=int, default=2, help="solutions kept per spec")
    p.add_argument("--base", default="qwen3.5-9b", help="train.config.BASES key for the token lengths")
    p.add_argument("--tokens", choices=("auto", "yes", "no"), default="auto", help="token lengths (needs transformers and the cached tokenizer)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_build_dataset)

    p = sub.add_parser("tokens", help="token lengths of data/train.jsonl and dev.jsonl -> manifest.json (run with the train venv)")
    p.add_argument("--data", help="folder with train.jsonl and dev.jsonl (default <repo>/data)")
    p.add_argument("--base", default="qwen3.5-9b")
    p.set_defaults(fn=cmd_tokens)

    p = sub.add_parser("predict", help="predicted minutes and CU of a teacher job (write it to LOGBOOK.md first)")
    add_run(p)
    p.add_argument("--stage", choices=("generate", "repair"), default="generate")
    p.add_argument("--k", type=int, help="samples per prompt (default 4 generate, 1 repair)")
    p.add_argument("--tok-s", type=float, required=True, help="aggregate decode tokens/s of the whole batch")
    p.add_argument("--tokens", type=float, help="mean output tokens per sample")
    p.add_argument("--tokens-by-label", help="JSON {label: mean output tokens}; a label is a prompt's mode, else its kind "
                                             "(with --group-by family: its family, or family/mode)")
    p.add_argument("--group-by", choices=("label", "family"), default="label", help="price per label (mode, else kind) or per family")
    p.add_argument("--tok-s-by-label", help="JSON {label: decode tokens/s} where it differs from --tok-s (reasoning decodes slower)")
    p.add_argument("--load-s", type=float, default=300.0, help="vLLM start-up and weight loading")
    p.add_argument("--overhead-s", type=float, default=600.0, help="VM start, package install, upload, download")
    p.add_argument("--rate", type=float, default=1.7, help="CU per hour (L4 1.7)")
    p.add_argument("--usd-per-cu", type=float, default=0.10)
    p.set_defaults(fn=cmd_predict)

    c = sub.add_parser("colab", help="run the teacher job on a Colab GPU (billed; needs --i-have-approval) or --dry-run it")
    add_run(c)
    c.add_argument("--stage", choices=("generate", "repair"), default="generate")
    mode = c.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the commands and the launcher; run nothing")
    mode.add_argument("--i-have-approval", action="store_true", help="Phil has approved this spend (CLAUDE.md Rule 2)")
    c.add_argument("--gpu", default="L4", help=f"one of {colab_mod.GPU_CHOICES}")
    c.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    c.add_argument("--revision", help="model repo commit to load (pin it: CLAUDE.md Rule 5)")
    c.add_argument("--quantization", help="vLLM quantization of the checkpoint (awq, gptq, fp8 ...)")
    c.add_argument("--max-model-len", type=int, default=8192)
    c.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    c.add_argument("-k", type=int, help="samples per prompt (default 4 generate, 1 repair)")
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--top-p", type=float, default=0.95)
    c.add_argument("--top-k", type=int, default=0, help="0: off")
    c.add_argument("--max-tokens", type=int, default=2048)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--chunk", type=int, default=50, help="prompts per generate() call; saved after each")
    c.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    c.add_argument("--enforce-eager", action="store_true")
    c.add_argument("--max-num-seqs", type=int, default=0)
    c.add_argument("--trust-remote-code", action="store_true")
    c.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. \'{"enable_thinking": false}\'')
    c.add_argument("--packages", default="vllm", help="comma-separated pip requirements for the VM (pin the version)")
    c.add_argument("--install-via", choices=("pip", "colab"), default="pip")
    c.add_argument("--session", help="Colab session name (default foundry-<run>-<stage>)")
    c.add_argument("--predicted-minutes", type=float, help="whole job incl. install and load (LOGBOOK.md); exec timeout = 2x")
    c.add_argument("--predicted-cu", type=float)
    c.add_argument("--exec-timeout", type=float, help="seconds; overrides 2 x predicted minutes")
    c.add_argument("--teacher-license", help="license of the teacher model (CLAUDE.md Rule 5)")
    c.add_argument("--colab-bin", help="path to the colab executable (default: PATH, else ~/venvs/colab/bin/colab)")
    c.add_argument("--colab-auth", choices=("oauth2", "adc"), help="pin the CLI's global --auth (its documented default disagrees with its help)")
    c.set_defaults(fn=cmd_colab)

    v = sub.add_parser("verify", help="grade the completions on the CPU and assign tiers (resumable)")
    add_run(v)
    v.add_argument("--stage", choices=("generate", "repair"), default="generate")
    v.add_argument("--workers", type=int, default=4, help="at most 4")
    v.add_argument("--n-scenes", type=int, default=20)
    v.add_argument("--n-meta", type=int, default=5)
    v.add_argument("--timeout", type=float, help="seconds per sandbox run (default 60)")
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser("repair", help="build repair_prompts.jsonl from the failed samples")
    add_run(r)
    r.add_argument("--max-per-spec", type=int, default=2, help="failed samples (distinct code) sent per spec")
    r.add_argument("--include-solved", action="store_true", help="also repair specs that already have 2 Tier-A examples")
    r.set_defaults(fn=cmd_repair)

    q = sub.add_parser("report", help="yield table, cost per Tier-A example, examples.jsonl")
    add_run(q)
    q.add_argument("--gpu-rate", type=float, default=1.7, help="CU per hour (L4 1.7)")
    q.add_argument("--usd-per-cu", type=float, default=0.10)
    q.add_argument("--overhead-s", type=float, default=0.0, help="install and download seconds the job cannot see; added to GPU time")
    q.add_argument("--answer-format", choices=("code", "fenced", "raw"), default="code",
                   help="the assistant turn of examples.jsonl: the extracted code, the code in one fenced block (the Gate 4a "
                        "training format), or the answer as written (all are stored)")
    q.add_argument("--no-tier-b", action="store_true", help="leave the Tier-B fallback examples out of examples.jsonl")
    q.add_argument("--keep-cross-spec-duplicates", action="store_true",
                   help="keep examples whose prompt and AST-normalized code equal an earlier spec's (specs of one family and phrasing share a prompt)")
    q.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
