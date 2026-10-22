"""Command line for the Stage 4 teacher pilot. Run from opencv-specialist/ with the cv5 venv:

    python -m foundry.run specs                                   # the 200 pilot specs, checked; a table per family
    python -m foundry.run prepare --run pilot1                    # specs.jsonl, prompts.jsonl in ~/opencv-expert/foundry/pilot1/
    python -m foundry.run predict --run pilot1 --tok-s 600 --tokens 700     # minutes and CU, for LOGBOOK.md first (Rule 1)
    python -m foundry.run colab   --run pilot1 --stage generate --dry-run   # print the Colab CLI commands, run nothing
    python -m foundry.run colab   --run pilot1 --stage generate --i-have-approval --predicted-minutes 45 --predicted-cu 1.3 \\
                                  --teacher-license apache-2.0 --model ... --quantization awq      # BILLED: needs Phil's OK
    python -m foundry.run verify  --run pilot1 --workers 4        # CPU: grade every completion, assign tiers (resumable)
    python -m foundry.run repair  --run pilot1                    # repair_prompts.jsonl from the failures
    python -m foundry.run colab   --run pilot1 --stage repair ...                 # the same job script, k = 1
    python -m foundry.run verify  --run pilot1 --stage repair     # grades the repairs, writes repair_pairs.jsonl
    python -m foundry.run report  --run pilot1 --gpu-rate 1.7     # report.md, summary.json, examples.jsonl

The GPU job is the only billed step and it never starts without ``--i-have-approval`` (CLAUDE.md Rule 2 and the project
budget: Colab needs Phil's explicit OK per stage). ``--root`` changes the parent of run folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from foundry import colab as colab_mod
from foundry.store import DEFAULT_ROOT, FoundryRun, read_jsonl, write_json
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

def cmd_specs(a) -> int:
    from foundry.specs import FOUNDRY_BLOCK, pilot_specs
    from cvbench.splits import FAMILIES
    specs = pilot_specs(a.n)  # raises SpecError on any invariant
    print(f"{len(specs)} specs, invariants hold (trainable families only, foundry phrasings, no dev/test/gotcha prompt, "
          f"seed windows inside FOUNDRY_RANGE and disjoint)")
    print(f"{'family':<22}{'subset':<10}{'specs':>6}  seeds")
    for f in dict.fromkeys(s.family for s in specs):
        mine = [s for s in specs if s.family == f]
        print(f"{f:<22}{FAMILIES[f].subset:<10}{len(mine):>6}  {mine[0].seed} .. {mine[-1].seed + 19}  "
              f"(block {FOUNDRY_BLOCK}, phrasings {sorted({s.phrasing_index for s in mine})})")
    return 0


def prompt_rows(specs) -> list[dict]:
    return [{"id": s.id, "spec_id": s.id, "family": s.family, "kind": "generate", "messages": s.messages()} for s in specs]


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


def cmd_predict(a) -> int:
    run = _run(a)
    n = len(read_jsonl(run.stage(a.stage).prompts))
    if not n:
        raise SystemExit(f"{run.stage(a.stage).prompts} is missing or empty")
    k = a.k or (4 if a.stage == "generate" else 1)
    print(predict_text(n, k, a.tokens, a.tok_s, a.load_s, a.overhead_s, a.rate, a.usd_per_cu))
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
        gpu_memory_utilization=a.gpu_memory_utilization, k=a.k, temperature=a.temperature, top_p=a.top_p,
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
    from foundry.verify import VerifySettings, grade_stage, summarize, write_kept
    run = _run(a)
    counts = grade_stage(run, a.stage, VerifySettings(a.workers, a.n_scenes, a.n_meta, a.timeout))
    print(f"graded {counts['graded']} ({counts['executed']} executed, {counts['cached']} reused, {counts['grader_errors']} grader "
          f"errors): Tier A {counts['A']}, Tier B {counts['B']}, fail {counts['fail']}")
    rows = read_jsonl(run.stage(a.stage).grades)
    for fam, d in summarize(rows).items():
        print(f"  {fam:<22} A {d['A']:>4}  B {d['B']:>4}  fail {d['fail']:>4}")
    if a.stage == "generate":
        s = write_kept(run)
        print(f"kept {s['kept']} Tier-A solutions of {s['candidates']} candidates ({s['duplicates']} AST duplicates, "
              f"{s['over_cap']} over the cap) -> {run.kept}")
    else:
        print(f"{repair_mod.write_repair_pairs(run)} repair pairs -> {run.repair_pairs}")
    return 1 if counts["grader_errors"] else 0


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

    p = sub.add_parser("specs", help="build and check the pilot specs; print a table")
    p.add_argument("--n", type=int, default=200)
    p.set_defaults(fn=cmd_specs)

    p = sub.add_parser("prepare", help="write specs.jsonl and prompts.jsonl")
    add_run(p)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--notes", help="the OpenCV 5.0 notes file (default bakeoff/notes_5_0.md)")
    p.add_argument("--force", action="store_true", help="overwrite prompts even if the run has completions")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("predict", help="predicted minutes and CU of a teacher job (write it to LOGBOOK.md first)")
    add_run(p)
    p.add_argument("--stage", choices=("generate", "repair"), default="generate")
    p.add_argument("--k", type=int, help="samples per prompt (default 4 generate, 1 repair)")
    p.add_argument("--tok-s", type=float, required=True, help="aggregate decode tokens/s of the whole batch")
    p.add_argument("--tokens", type=float, required=True, help="mean output tokens per sample")
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
    q.add_argument("--answer-format", choices=("code", "raw"), default="code", help="the assistant turn of examples.jsonl (both are stored)")
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
