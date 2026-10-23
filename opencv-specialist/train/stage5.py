"""Stage 5 driver (local): predict, pre-flight, the Colab convergence study, grading of the dev answers, the report.

    python -m train.stage5 predict                                   # CPU: minutes and CU of the whole study (CLAUDE.md Rule 1)
    python -m train.stage5 tokens                                    # train venv: real token counts of every row, cached
    python -m train.stage5 check                                     # train venv: template, one marker per row, lengths, dev prompts
    python -m train.stage5 prepare --study S1                        # study folder: dev_prompts.jsonl and the upload set
    python -m train.stage5 colab   --study S1 --dry-run              # print every Colab CLI command and the launchers
    python -m train.stage5 colab   --study S1 --i-have-approval --predicted-minutes M --predicted-cu C   # BILLED
    python -m train.stage5 grade   --study S1 [--follow]             # CPU: grade the downloaded dev answers (resumable)
    python -m train.stage5 report   --study S1                       # curves, selection by dev pass@1, convergence rule

The convergence study (PLAN.md 4.3) trains on nested 25 / 50 / 100 % subsets for two epochs each, in ONE Colab session, one
``colab exec`` per run, so a failed run costs one run and the model download is paid once. ``train/colab_train_job.py`` is the
job (standalone); ``foundry/colab.py`` supplies the command runner and the guards' vocabulary. A study lives in
``~/opencv-expert/stage5/<study>/`` (outside git, CLAUDE.md Rule 8):

    config.json               what was asked: percents, epochs, revision, hashes of the data and of the job
    dev_prompts.jsonl         one row per distinct cvbench dev prompt: id, messages, the dev task ids that share it
    launch_setup.py, launch_<run>.py   the launchers `colab exec` runs (the kernel has no argv)
    runs/<run>/               everything downloaded from the VM: <run>.log.jsonl, .summary.json, .manifest.json,
                              <run>.<tag>.adapter.tar and <run>.<tag>.dev_answers.jsonl for every checkpoint
    session_stage5.json       the driver's record: commands, usage before and after, predicted vs measured per run
    grading/                  a bakeoff run folder: generations.jsonl (the dev answers as bake-off rows), grades.jsonl
    report.md, summary.json, curves.csv

Nothing here executes model code except through ``bakeoff.grade`` (which goes through ``sandbox.run_solution``, Rule 4).
Nothing launches a billed job without ``--i-have-approval`` and a prediction.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from foundry.colab import (ApprovalRequired, CmdResult, ColabError, ConfigError, GPU_CHOICES, Runner, Step,
                           subprocess_runner)
from train import colab_train_job as job
from train.config import BASES, SYSTEM, WORK

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH, DEV_PATH = ROOT / "data" / "train.jsonl", ROOT / "data" / "dev.jsonl"
JOB_SCRIPT = Path(job.__file__).resolve()
REQUIREMENTS = Path(__file__).resolve().parent / "requirements-stage5.txt"
STAGE5_ROOT = WORK / "stage5"
BASE = "qwen3.5-9b"

# Colab rates (COMPUTE.md); H100 and G4 are unmeasured, so they need --cu-per-hour.
CU_PER_HOUR = {"T4": 1.2, "L4": 1.7, "A100": 5.4}
PEAK_TFLOPS = {"T4": 65.0, "L4": 121.0, "A100": 312.0, "H100": 989.0}   # dense bf16 / fp16
N_PARAMS_B = BASES[BASE]["params_b"]                                       # 9.65, vision tower included (conservative)
CHARS_PER_TOKEN = 3.5                                                      # fallback only; `tokens` caches the real counts
CAP_CU = 60.0                                                              # Phil, Stage 5 cap (2026-10-22)

REMOTE = {"job": "/content/stage5_job.py", "train": "/content/stage5_train.jsonl", "dev": "/content/stage5_dev.jsonl",
          "prompts": "/content/stage5_dev_prompts.jsonl", "data": "/content/stage5_data", "out": "/content/stage5_out",
          "work": "/content/stage5_work", "venv": "/content/stage5_venv"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_name(percent: int) -> str:
    return f"r{percent:03d}"


def sha256_file(path: Path) -> str:
    return job.sha256_file(path)


# ================================================================ token counts

@dataclass(frozen=True)
class RowTokens:
    train: list[int]
    dev: list[int]
    source: str          # "tokenizer" (the cache from `tokens`) or "estimate" (characters / CHARS_PER_TOKEN)


def token_cache_path(root: Path | None = None) -> Path:
    return (root or STAGE5_ROOT) / "token_counts.json"


def estimate_tokens(rows: list[dict]) -> list[int]:
    return [max(1, round(len(job.render_row(r["messages"])) / CHARS_PER_TOKEN)) for r in rows]


def load_row_tokens(train_rows: list[dict], dev_rows: list[dict], *, train_path: Path = TRAIN_PATH, dev_path: Path = DEV_PATH,
                    cache: Path | None = None) -> RowTokens:
    """Real token counts from the cache written by ``tokens`` (keyed by the SHA-256 of both files), else an estimate."""
    cache = cache or token_cache_path()
    if cache.is_file() and train_path.is_file() and dev_path.is_file():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("train_sha256") == sha256_file(train_path) and c.get("dev_sha256") == sha256_file(dev_path):
            return RowTokens(c["train"], c["dev"], "tokenizer")
    return RowTokens(estimate_tokens(train_rows), estimate_tokens(dev_rows), "estimate")


# ================================================================ predict (CLAUDE.md Rule 1, COMPUTE.md method)

@dataclass(frozen=True)
class Assumptions:
    """Every number a prediction rests on. The throughput is derived, not measured: the first run measures it."""
    gpu: str = "A100"
    cu_per_hour: float = 5.4
    peak_tflops: float = 312.0
    mfu: float = 0.20               # conservative: COMPUTE.md uses 0.35; the 3060 hydrotests reached 0.53-0.69 of measured peak
    n_params_b: float = N_PARAMS_B
    tok_s: float | None = None      # override the derived tokens/s
    fwd_speedup: float = 3.0        # forward-only (eval) tokens/s = 3 x training tokens/s: 2N against 6N FLOP per token
    gen_step_s: float = 0.09        # HF generate, LoRA unmerged, all dev prompts in one batch: seconds per decoded token (unmeasured)
    gen_steps: int = 2048           # the batch runs to the cap, as if one answer loops (conservative)
    gen_prefill_s: float = 15.0
    install_min: float = 8.0        # uv venv + the pinned wheels (torch and its CUDA libraries are ~4.5 GB)
    download_min: float = 8.0       # 19.3 GB at ~40 MB/s (Stage 4 loaded a 19.5 GB model in ~10 min including load)
    setup_misc_min: float = 3.0     # colab new, uploads, tokenizer pre-flight
    load_min: float = 3.0           # per run: weights from the VM disk to the GPU, unsloth patching, LoRA
    prep_min: float = 1.5           # per run: tokenize ~3k rows, mask check
    save_s: float = 15.0            # per checkpoint: adapter tar
    adapter_mb: float = 175.0       # 43.3 M LoRA parameters in fp32 (the 4B adapter was 130 MB for 32.5 M)
    download_mb_s: float = 10.0     # colab download through the Jupyter API: unmeasured, deliberately low
    download_file_s: float = 10.0   # fixed cost of each `colab download` call
    teardown_min: float = 0.5

    @property
    def train_tok_s(self) -> float:
        return self.tok_s or self.peak_tflops * 1e12 * self.mfu / (6.0 * self.n_params_b * 1e9)


def assumptions_for(gpu: str, cu_per_hour: float | None = None, **kw) -> Assumptions:
    rate = cu_per_hour or CU_PER_HOUR.get(gpu)
    if rate is None:
        raise ConfigError(f"no known CU/h for {gpu}: pass --cu-per-hour (COMPUTE.md lists T4, L4 and A100 only)")
    return Assumptions(gpu=gpu, cu_per_hour=rate, peak_tflops=PEAK_TFLOPS.get(gpu, 312.0), **kw)


def n_checkpoints(epochs: float, every: float) -> int:
    return len(job.ckpt_plan(10 ** 6, epochs, every))


def predict_run(tokens: RowTokens, train_rows: list[dict], percent: int, as_: Assumptions, *, epochs: float, ckpt_every: float,
                step0_gen: bool, probe_rows: int = 64, seed: int = 0, eff_batch: int = 16) -> dict:
    """Minutes of one training run, by component. Training time is COMPUTE.md's: tokens / (tokens/s)."""
    frac = percent / 100.0
    idx = job.select_subset(train_rows, frac, seed)
    epoch_tokens = sum(tokens.train[i] for i in idx)
    probe_tokens = sum(tokens.train[i] for i in job.probe_indices(train_rows, probe_rows, seed))
    n_ck = n_checkpoints(epochs, ckpt_every)
    n_eval = n_ck + 1
    n_gen = n_ck + (1 if step0_gen else 0)
    n_adapters = n_gen
    tok_s = as_.train_tok_s
    gen_min = (as_.gen_steps * as_.gen_step_s + as_.gen_prefill_s) / 60.0
    eval_min = (sum(tokens.dev) + probe_tokens) / (as_.fwd_speedup * tok_s) / 60.0
    n_files = 2 * n_gen + 3                                              # adapters + answers, manifest, summary, log
    parts = {
        "load": as_.load_min, "prep": as_.prep_min,
        "train": epochs * epoch_tokens / tok_s / 60.0,
        "eval": n_eval * eval_min,
        "generate": n_gen * gen_min,
        "save": n_adapters * as_.save_s / 60.0,
        "download": (n_adapters * as_.adapter_mb / as_.download_mb_s + n_files * as_.download_file_s) / 60.0,
        "teardown": as_.teardown_min,
    }
    total = sum(parts.values())
    return {"run": run_name(percent), "percent": percent, "rows": len(idx), "epoch_tokens": epoch_tokens, "train_tokens": epochs * epoch_tokens,
            "steps": math.ceil(epochs * len(idx) / eff_batch), "checkpoints": n_ck, "generations": n_gen, "parts": parts,
            "minutes": total, "cu": total / 60.0 * as_.cu_per_hour}


def predict_setup(as_: Assumptions) -> dict:
    parts = {"install": as_.install_min, "model download": as_.download_min, "new, uploads, pre-flight": as_.setup_misc_min}
    total = sum(parts.values())
    return {"parts": parts, "minutes": total, "cu": total / 60.0 * as_.cu_per_hour}


def predict_study(tokens: RowTokens, train_rows: list[dict], percents: tuple[int, ...], as_: Assumptions, *, epochs: float = 2.0,
                  ckpt_every: float = 0.5, step0_gen: bool = True, probe_rows: int = 64, seed: int = 0) -> dict:
    """One Colab session: setup, then the runs in order (the first also generates the step-0 dev answers)."""
    runs = [predict_run(tokens, train_rows, p, as_, epochs=epochs, ckpt_every=ckpt_every, step0_gen=step0_gen and i == 0,
                        probe_rows=probe_rows, seed=seed) for i, p in enumerate(percents)]
    setup = predict_setup(as_)
    minutes = setup["minutes"] + sum(r["minutes"] for r in runs)
    return {"setup": setup, "runs": runs, "minutes": minutes, "cu": minutes / 60.0 * as_.cu_per_hour,
            "train_tok_s": as_.train_tok_s, "assumptions": dataclasses.asdict(as_), "token_source": tokens.source}


def extra_round(tokens: RowTokens, train_rows: list[dict], percent: int, as_: Assumptions, *, epochs: float, ckpt_every: float,
                probe_rows: int = 64) -> dict:
    """One more training round in its own session: setup plus one run (no step-0 generation)."""
    r = predict_run(tokens, train_rows, percent, as_, epochs=epochs, ckpt_every=ckpt_every, step0_gen=False, probe_rows=probe_rows)
    minutes = predict_setup(as_)["minutes"] + r["minutes"]
    return {"label": f"{percent} % of the data, {epochs:g} epochs", "run": r, "minutes": minutes, "cu": minutes / 60.0 * as_.cu_per_hour}


def render_prediction(study: dict, extra: dict | None, as_: Assumptions, *, cap_cu: float, spent_cu: float, token_source: str,
                      scenarios: list[tuple[str, dict, dict | None]] | None = None, split: list[tuple[str, dict]] | None = None) -> str:
    """The text for LOGBOOK.md: per run, total, headroom under the cap, the extra round, the kill points."""
    L = []
    tok_s = study["train_tok_s"]
    L.append(f"Stage 5 convergence study on {as_.gpu} ({as_.cu_per_hour} CU/h), 16-bit LoRA r=16 on Qwen3.5-9B; row tokens: {token_source}")
    L.append(f"throughput: {tok_s:,.0f} training tokens/s = {as_.peak_tflops:.0f} TFLOP/s x {as_.mfu:.0%} / (6 x {as_.n_params_b} B)"
             f"{' [override]' if as_.tok_s else ''}; eval = {as_.fwd_speedup:.0f}x that; dev generation {as_.gen_steps} steps x {as_.gen_step_s * 1000:.0f} ms"
             f" + {as_.gen_prefill_s:.0f} s per pass (the 36 distinct dev prompts in one batch, run to the cap)")
    L.append(f"setup (once per session): {study['setup']['minutes']:.0f} min = " +
             ", ".join(f"{k} {v:.0f}" for k, v in study["setup"]["parts"].items()) + f"  ->  {study['setup']['cu']:.2f} CU")
    for r in study["runs"]:
        p = r["parts"]
        L.append(f"{r['run']}  {r['rows']:>5} rows, {r['epoch_tokens'] / 1e6:5.2f} M tokens/epoch, {r['steps']:>3} steps, {r['checkpoints']} checkpoints, "
                 f"{r['generations']} dev generations: train {p['train']:5.1f} + eval {p['eval']:4.1f} + generate {p['generate']:4.1f} + load/prep/save/"
                 f"download {p['load'] + p['prep'] + p['save'] + p['download'] + p['teardown']:4.1f} = {r['minutes']:6.1f} min = {r['cu']:5.2f} CU "
                 f"(exec timeout {2 * r['minutes']:.0f} min)")
    L.append(f"STUDY: {study['minutes']:.0f} min = {study['cu']:.1f} CU; killed at 2x = {2 * study['minutes']:.0f} min / {2 * study['cu']:.1f} CU")
    if split:
        L.append("or in two sessions, so run 1 measures the throughput before the other two are paid for: " +
                 "; ".join(f"{name}: {s['minutes']:.0f} min, {s['cu']:.1f} CU" for name, s in split) +
                 f"; together {sum(s['cu'] for _, s in split):.1f} CU (one more setup, {study['setup']['cu']:.1f} CU)")
    if scenarios:
        for name, s, ex in scenarios:
            left = cap_cu - spent_cu - s["cu"]
            L.append(f"range, {name}: study {s['minutes']:.0f} min, {s['cu']:.1f} CU, {left:.1f} CU left under the cap" +
                     (f"; the extra round would then cost {ex['cu']:.1f} CU -> {'fits' if left >= ex['cu'] else 'DOES NOT FIT'}" if ex else ""))
    room = cap_cu - spent_cu - study["cu"]
    L.append(f"cap {cap_cu:.0f} CU (Phil, 2026-10-22), spent in Stage 5 so far {spent_cu:.1f}: study {study['cu']:.1f} -> {room:.1f} CU left")
    if extra:
        fits = room >= extra["cu"]
        L.append(f"one extra round ({extra['label']}, its own session: setup + {extra['run']['minutes']:.0f} min run, no step-0 generation): "
                 f"{extra['minutes']:.0f} min = {extra['cu']:.1f} CU -> {'fits' if fits else 'DOES NOT FIT'}: {room - extra['cu']:+.1f} CU left"
                 f" (if appended to the study's session it saves {study['setup']['cu']:.1f} CU of setup)")
        L.append(f"each session gets its own prediction and is killed at 2x it; the driver refuses a session if spent + 2 x its predicted CU passes the cap: "
                 f"the study alone is {2 * study['cu']:.1f} CU at its kill point, and the extra round may start only if the study cost at most "
                 f"{cap_cu - 2 * extra['cu']:.1f} CU (it is predicted at {study['cu']:.1f})")
    L.append(f"LOGBOOK line: predicted {tok_s:,.0f} tok/s, {study['minutes']:.0f} min, {study['cu']:.1f} CU for the study; "
             f"first run {study['runs'][0]['minutes']:.0f} min / {study['runs'][0]['cu']:.2f} CU")
    return "\n".join(L)


# ================================================================ dev prompts, study files

def build_dev_prompts(tasks=None) -> list[dict]:
    """One row per distinct dev prompt. At T = 0 tasks with the same prompt (same family and phrasing, different scenes) get
    the same answer, so the VM generates 36 answers, not 99; each answer is graded on every task of its prompt (as the
    bake-off's --dedupe-greedy did for the baselines)."""
    if tasks is None:
        from bakeoff.tasks import load_tasks
        tasks = load_tasks(("dev",))
    by_prompt: dict[str, dict] = {}
    for t in tasks:
        row = by_prompt.setdefault(t.prompt_sha1, {"id": "d" + t.prompt_sha1[:10], "prompt_sha1": t.prompt_sha1, "cluster": t.cluster,
                                                   "messages": [{"role": "system", "content": SYSTEM},
                                                                {"role": "user", "content": t.prompt}], "task_ids": []})
        row["task_ids"].append(t.id)
    return list(by_prompt.values())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    from bakeoff.store import read_jsonl as _r
    return _r(path)


# ================================================================ the Colab study

@dataclass
class StudyConfig:
    name: str
    root: Path | None = None
    percents: tuple[int, ...] = (25, 50, 100)
    epochs: float = 2.0
    ckpt_every: float = 0.5
    gpu: str = "A100"
    repo: str = job.DEFAULT_REPO
    revision: str = job.DEFAULT_REVISION
    base_license: str | None = None          # CLAUDE.md Rule 5: the base's license, recorded with the session
    max_len: int = 4608
    seed: int = 0
    batch: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    probe_rows: int = 64
    dev_max_new_tokens: int = 2048
    dev_batch: int = 0
    step0_gen: bool = True
    qlora: bool = False
    cu_per_hour: float | None = None
    cap_cu: float = 60.0                     # Phil's Stage 5 cap
    spent_cu: float = 0.0                    # already spent on Stage 5 (LOGBOOK.md)
    predicted_minutes: float | None = None   # the whole session (LOGBOOK.md); every exec timeout is 2x its share of it
    predicted_cu: float | None = None
    session: str | None = None
    colab_bin: str = "colab"
    colab_auth: str | None = None
    abort_when_slow: bool = True             # a run slower than predicted/3 stops the study (Rule 1: a >3x miss is a bug)
    extra_job_args: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return (Path(self.root) if self.root else STAGE5_ROOT) / self.name

    @property
    def session_name(self) -> str:
        return self.session or f"stage5-{self.name}"

    @property
    def runs(self) -> list[str]:
        return [run_name(p) for p in self.percents]

    def run_dir(self, run: str) -> Path:
        return self.path / "runs" / run

    def packages(self) -> list[str]:
        pins = [ln.strip() for ln in REQUIREMENTS.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        return pins                                      # unsloth_zoo 2026.9.9 vendors fla 0.5.1, so no flash-linear-attention is installed


def validate(cfg: StudyConfig, *, real_run: bool) -> None:
    if cfg.gpu not in GPU_CHOICES:
        raise ConfigError(f"unknown GPU {cfg.gpu!r}; choose from {GPU_CHOICES} (the CLI would silently use an A100)")
    if cfg.gpu in ("T4", "L4") and not cfg.qlora:
        raise ConfigError(f"16-bit LoRA on the 9B needs more than {cfg.gpu}'s memory (weights alone are 19 GB): use A100, or --qlora (4-bit) "
                          "on a T4 as AGENT_PROMPT allows")
    if not cfg.percents or list(cfg.percents) != sorted(set(cfg.percents)) or not all(0 < p <= 100 for p in cfg.percents):
        raise ConfigError("--runs must be increasing percents in (0, 100], e.g. 25,50,100")
    if real_run:
        missing = [n for n, v in (("predicted_minutes", cfg.predicted_minutes), ("predicted_cu", cfg.predicted_cu)) if not v or v <= 0]
        if missing:
            raise ConfigError(f"{', '.join(missing)} required: write the prediction to LOGBOOK.md first (CLAUDE.md Rule 1); "
                              "every exec timeout is 2x its share of the predicted minutes (Rule 2)")
        if not cfg.base_license:
            raise ConfigError("base_license required: record the license of every model you use (CLAUDE.md Rule 5)")
        rate = cfg.cu_per_hour or CU_PER_HOUR.get(cfg.gpu)
        if rate is None:
            raise ConfigError(f"no known CU/h for {cfg.gpu}: pass --cu-per-hour")
        if cfg.spent_cu + 2.0 * cfg.predicted_cu > cfg.cap_cu + 1e-9:
            raise ConfigError(f"spent {cfg.spent_cu} + 2 x predicted {cfg.predicted_cu} CU (the kill point, Rule 2) = {cfg.spent_cu + 2 * cfg.predicted_cu:.1f} "
                              f"is over the {cfg.cap_cu:.0f} CU cap: shrink the session (--runs) or ask Phil")
        implied = cfg.predicted_cu / (cfg.predicted_minutes / 60.0)
        if not 0.7 * rate <= implied <= 1.5 * rate:
            raise ConfigError(f"predicted {cfg.predicted_cu} CU over {cfg.predicted_minutes} min implies {implied:.2f} CU/h, not the {rate} CU/h of "
                              f"{cfg.gpu}: a wrong CU number would weaken the 2x kill (Rule 2)")


def job_args(cfg: StudyConfig, run: str, *, first: bool) -> list[str]:
    """argv of ``colab_train_job.py train`` for one run (parsed by the same parser that runs on the VM)."""
    pct = int(run[1:])
    a = ["train", "--data-dir", REMOTE["data"], "--out-dir", REMOTE["out"], "--run", run, "--frac", str(pct / 100.0),
         "--repo", cfg.repo, "--revision", cfg.revision, "--max-len", str(cfg.max_len), "--seed", str(cfg.seed),
         "--epochs", str(cfg.epochs), "--ckpt-every", str(cfg.ckpt_every), "--batch", str(cfg.batch),
         "--grad-accum", str(cfg.grad_accum), "--lr", str(cfg.lr), "--warmup-ratio", str(cfg.warmup_ratio),
         "--probe-rows", str(cfg.probe_rows), "--dev-max-new-tokens", str(cfg.dev_max_new_tokens), "--dev-batch", str(cfg.dev_batch)]
    if first and cfg.step0_gen:
        a.append("--step0-gen")
    if cfg.qlora:
        a.append("--qlora")
    return a + list(cfg.extra_job_args)


SETUP_LAUNCHER = '''\
# Written by train/stage5.py, run with `colab exec -f` (the kernel has no argv). Idempotent: a rerun reuses the venv and the cache.
import os, shutil, subprocess, sys, time

VENV, PY = @@VENV@@, @@VENV@@ + "/bin/python"
PINS = @@PINS@@
JOB, DATA, OUT, WORK = @@JOB@@, @@DATA@@, @@OUT@@, @@WORK@@
UPLOADS = @@UPLOADS@@
REPO, REVISION, MAX_LEN = @@REPO@@, @@REVISION@@, @@MAX_LEN@@


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        raise RuntimeError("failed (%d): %s" % (r.returncode, " ".join(cmd)))


t0 = time.time()
for d in (DATA, OUT, WORK):
    os.makedirs(d, exist_ok=True)
for src, dst in UPLOADS:
    shutil.copy(src, dst)
if not os.path.exists(PY):
    if shutil.which("uv") is None:
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "venv", "--python", "3.12", VENV])
    sh(["uv", "pip", "install", "--python", PY, "-q"] + PINS)
print("setup: packages ready after %.0f s" % (time.time() - t0), flush=True)
sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
sh([PY, JOB, "prefetch", "--repo", REPO, "--revision", REVISION])
sh([PY, JOB, "check", "--data-dir", DATA, "--repo", REPO, "--revision", REVISION, "--max-len", str(MAX_LEN)])
print("launcher: setup finished OK after %.0f s" % (time.time() - t0), flush=True)
'''

RUN_LAUNCHER = '''\
# Written by train/stage5.py, run with `colab exec -f`. The job runs in a subprocess of the venv's Python, in its own folder.
import os, subprocess

PY, JOB, WORK = @@PY@@, @@JOB@@, @@WORK@@
ARGS = @@ARGS@@
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
cmd = [PY, JOB] + ARGS
print("+", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc:
    raise RuntimeError("colab_train_job.py exited with status %d" % rc)
print("launcher: job finished OK", flush=True)
'''


def setup_launcher_source(cfg: StudyConfig) -> str:
    uploads = [(REMOTE["train"], REMOTE["data"] + "/train.jsonl"), (REMOTE["dev"], REMOTE["data"] + "/dev.jsonl"),
               (REMOTE["prompts"], REMOTE["data"] + "/dev_prompts.jsonl")]
    rep = {"VENV": repr(REMOTE["venv"]), "PINS": repr(cfg.packages()), "JOB": repr(REMOTE["job"]), "DATA": repr(REMOTE["data"]),
           "OUT": repr(REMOTE["out"]), "WORK": repr(REMOTE["work"]), "UPLOADS": repr(uploads), "REPO": repr(cfg.repo),
           "REVISION": repr(cfg.revision), "MAX_LEN": repr(cfg.max_len)}
    src = SETUP_LAUNCHER
    for k, v in rep.items():
        src = src.replace(f"@@{k}@@", v)
    return src


def run_launcher_source(cfg: StudyConfig, run: str, first: bool) -> str:
    rep = {"PY": repr(REMOTE["venv"] + "/bin/python"), "JOB": repr(REMOTE["job"]), "WORK": repr(REMOTE["work"]),
           "ARGS": repr(job_args(cfg, run, first=first))}
    src = RUN_LAUNCHER
    for k, v in rep.items():
        src = src.replace(f"@@{k}@@", v)
    return src


@dataclass(frozen=True)
class Budget:
    """Minutes the exec timeouts derive from: the model's per-phase prediction, scaled to the caller's predicted total."""
    setup_min: float
    run_min: dict[str, float]
    total_min: float

    def timeout_s(self, phase: str) -> float:
        return 2.0 * 60.0 * (self.setup_min if phase == "setup" else self.run_min[phase])


def make_budget(cfg: StudyConfig, study: dict) -> Budget:
    scale = (cfg.predicted_minutes / study["minutes"]) if cfg.predicted_minutes else 1.0
    return Budget(setup_min=study["setup"]["minutes"] * scale,
                  run_min={r["run"]: r["minutes"] * scale for r in study["runs"]},
                  total_min=study["minutes"] * scale)


def _pre(cfg: StudyConfig) -> list[str]:
    return [cfg.colab_bin] + (["--auth", cfg.colab_auth] if cfg.colab_auth else [])


def plan(cfg: StudyConfig, budget: Budget, todo: list[str] | None = None) -> list[Step]:
    """Every command of the session in order. ``download-<run>`` expands to one command per file of the run's manifest."""
    pre, s = _pre(cfg), cfg.session_name
    todo = cfg.runs if todo is None else todo
    steps = [Step("usage-before", pre + ["usage"], 120, "pre", required=False),
             Step("new", pre + ["new", "-s", s, "--gpu", cfg.gpu], 900, note=f"billed from here ({cfg.gpu})"),
             Step("upload-job", pre + ["upload", "-s", s, str(JOB_SCRIPT), REMOTE["job"]], 600),
             Step("upload-train", pre + ["upload", "-s", s, str(TRAIN_PATH), REMOTE["train"]], 900),
             Step("upload-dev", pre + ["upload", "-s", s, str(DEV_PATH), REMOTE["dev"]], 600),
             Step("upload-dev-prompts", pre + ["upload", "-s", s, str(cfg.path / "dev_prompts.jsonl"), REMOTE["prompts"]], 600),
             Step("exec-setup", pre + ["exec", "-s", s, "-f", str(cfg.path / "launch_setup.py"), "--timeout", str(int(budget.timeout_s("setup")))],
                  budget.timeout_s("setup") + 300, log=True, note="uv venv, pinned wheels, weights, tokenizer pre-flight")]
    for run in todo:
        steps.append(Step(f"exec-{run}", pre + ["exec", "-s", s, "-f", str(cfg.path / f"launch_{run}.py"), "--timeout", str(int(budget.timeout_s(run)))],
                          budget.timeout_s(run) + 300, log=True, note=f"training run {run}: 2 epochs, checkpoints, dev answers"))
        steps.append(Step(f"download-{run}", pre + ["download", "-s", s, f"{REMOTE['out']}/{run}.manifest.json", str(cfg.run_dir(run) / f"{run}.manifest.json")], 900,
                          note="the manifest, then one download per file it lists"))
    steps.append(Step("stop", pre + ["stop", "-s", s], 300, "finally", required=False, note="always runs"))
    steps.append(Step("usage-after", pre + ["usage"], 120, "finally", required=False))
    return steps


def print_plan(cfg: StudyConfig, budget: Budget, steps: list[Step], log: Callable[[str], None]) -> None:
    log(f"# dry run: study {cfg.name}, session {cfg.session_name}, {cfg.gpu}, runs {cfg.runs}, base {cfg.repo}@{cfg.revision[:8]}")
    log(f"# predicted session {budget.total_min:.0f} min; exec timeouts (2x): setup {budget.timeout_s('setup') / 60:.0f} min, " +
        ", ".join(f"{r} {budget.timeout_s(r) / 60:.0f} min" for r in cfg.runs))
    for st in steps:
        log(f"[{st.phase:7}] {st.name:<20} {st.shell()}" + (f"    # {st.note}" if st.note else ""))
    log("# setup launcher (launch_setup.py):")
    for line in setup_launcher_source(cfg).splitlines():
        log("#   " + line)
    log(f"# run launcher (launch_{cfg.runs[0]}.py):")
    for line in run_launcher_source(cfg, cfg.runs[0], True).splitlines():
        log("#   " + line)


def run_complete(cfg: StudyConfig, run: str) -> bool:
    """A run is done locally when its summary says complete and every file of its manifest is present with the listed hash."""
    d = cfg.run_dir(run)
    man, summ = d / f"{run}.manifest.json", d / f"{run}.summary.json"
    if not (man.is_file() and summ.is_file()):
        return False
    try:
        m, s = json.loads(man.read_text(encoding="utf-8")), json.loads(summ.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if m.get("state") != "complete" or s.get("state") != "complete":
        return False
    return all(_have(d / f["name"], f) for f in m["files"])


def _have(path: Path, f: dict) -> bool:
    """A listed file is here: present, and equal to the manifest's hash when it has one (the live log has none)."""
    return path.is_file() and (f.get("sha256") is None or sha256_file(path) == f["sha256"])


def remote_path(name: str) -> str:
    return f"{REMOTE['out']}/{name}"


def download_run(cfg: StudyConfig, run: str, do: Callable[[Step], CmdResult], log) -> dict | None:
    """Download the run's manifest and then every file it lists that is not already here with the same hash."""
    pre, s = _pre(cfg), cfg.session_name
    d = cfg.run_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    man_path = d / f"{run}.manifest.json"
    do(Step(f"download-{run}-manifest", pre + ["download", "-s", s, remote_path(man_path.name), str(man_path)], 900, required=False))
    if not man_path.is_file():
        log(f"no manifest for {run} on the VM (the job may not have started)")
        return None
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for f in man["files"]:
        local = d / f["name"]
        if f.get("sha256") is not None and _have(local, f):
            continue  # unhashed files (the live log) are always fetched again: they grow until the run ends
        do(Step(f"download-{f['name']}", pre + ["download", "-s", s, remote_path(f["name"]), str(local)], 900, required=False))
        if not _have(local, f):
            log(f"WARNING: {f['name']} is missing or differs from the manifest after download")
    return man


def run_study(cfg: StudyConfig, *, approved: bool = False, dry_run: bool = False, runner: Runner | None = None, log: Callable[[str], None] = print,
              tokens: RowTokens | None = None, train_rows: list[dict] | None = None) -> dict:
    """Run (or, with ``dry_run``, show) the convergence study. Returns the session record. See the module docstring."""
    validate(cfg, real_run=not dry_run)
    train_rows = train_rows if train_rows is not None else read_jsonl(TRAIN_PATH)
    tokens = tokens or load_row_tokens(train_rows, read_jsonl(DEV_PATH))
    as_ = assumptions_for(cfg.gpu, cfg.cu_per_hour)
    study = predict_study(tokens, train_rows, cfg.percents, as_, epochs=cfg.epochs, ckpt_every=cfg.ckpt_every, step0_gen=cfg.step0_gen,
                          probe_rows=cfg.probe_rows, seed=cfg.seed)
    budget = make_budget(cfg, study)
    todo = [r for r in cfg.runs if not run_complete(cfg, r)]
    steps = plan(cfg, budget, todo)
    if dry_run:
        print_plan(cfg, budget, steps, log)
        if not todo:
            log("# note: every run is already complete locally; a real run would start no VM")
        return {"dry_run": True, "commands": [s.argv for s in steps], "steps": [s.name for s in steps], "todo": todo, "budget": dataclasses.asdict(budget)}
    if not approved:
        raise ApprovalRequired("a billed Colab session needs Phil's explicit OK: pass --i-have-approval (or use --dry-run)")
    if not todo:
        log("every run is already complete locally; no VM started")
        return {"skipped": True, "todo": []}

    runner = runner or subprocess_runner
    prepare_study(cfg, train_rows)
    by_phase = {p: [s for s in steps if s.phase == p] for p in ("pre", "main", "finally")}
    log_path = cfg.path / "colab_stage5.log"
    record: dict = {"study": cfg.name, "session": cfg.session_name, "gpu": cfg.gpu, "runs": cfg.runs, "todo": todo, "repo": cfg.repo,
                    "revision": cfg.revision, "base_license": cfg.base_license, "predicted_minutes": cfg.predicted_minutes,
                    "predicted_cu": cfg.predicted_cu, "predicted_train_tok_s": as_.train_tok_s, "budget": dataclasses.asdict(budget),
                    "commands": [], "per_run": {}, "usage_before": None, "usage_after": None, "started_at": _now(),
                    "state": "starting", "error": None, "stop_ok": None}
    t_start = time.monotonic()
    deadline = 2.0 * 60.0 * (cfg.predicted_minutes or budget.total_min)    # Rule 2: no more than 2x the predicted session, from `new`
    t_new = [None]

    def do(step: Step) -> CmdResult:
        timeout = step.timeout_s
        if t_new[0] is not None and step.phase == "main":
            remaining = deadline - (time.monotonic() - t_new[0])
            if remaining <= 0:
                raise ColabError(f"the session passed 2x its predicted {cfg.predicted_minutes} min (Rule 2): stopping before {step.name}")
            if step.name.startswith("exec-"):                                 # keep the remote timeout and the local kill in line
                argv = list(step.argv)
                i = argv.index("--timeout")
                argv[i + 1] = str(int(min(float(argv[i + 1]), remaining)))
                step = dataclasses.replace(step, argv=argv, timeout_s=min(timeout, remaining + 300))
        log(f"$ {step.shell()}")
        res = runner(step.argv, timeout=step.timeout_s, log_path=log_path if step.log else None)
        record["commands"].append({"name": step.name, "argv": step.argv, "returncode": res.returncode, "seconds": round(res.seconds, 2)})
        if res.returncode and step.required:
            raise ColabError(f"{step.name} exited with status {res.returncode}: {step.shell()}")
        return res

    def save(state: str) -> None:
        record["state"] = state
        record["wall_s"] = round(time.monotonic() - t_start, 1)
        job.write_json_atomic(cfg.path / "session_stage5.json", record)

    attempted_new = False
    current_run: str | None = None
    try:
        for step in by_phase["pre"]:
            try:
                record["usage_before"] = do(step).stdout.strip()
            except Exception as e:  # noqa: BLE001 - usage is informational; a broken login fails `new` just as well
                log(f"warning: {step.name} failed: {e!r}")
        for name, text in (("launch_setup.py", setup_launcher_source(cfg)),):
            (cfg.path / name).write_text(text, encoding="utf-8")
        for i, run in enumerate(cfg.runs):
            (cfg.path / f"launch_{run}.py").write_text(run_launcher_source(cfg, run, first=(i == 0)), encoding="utf-8")
        for step in by_phase["main"]:
            if step.name.startswith("download-"):
                continue                                                     # expanded after each exec below
            if step.name == "new":
                attempted_new = True
                save("new-attempted")                                        # evidence of a possibly billed VM if this process dies hard
            is_run_exec = step.name.startswith("exec-") and step.name != "exec-setup"
            if is_run_exec:
                current_run = step.name[len("exec-"):]
                aside_partial(cfg, current_run)
            do(step)
            if step.name == "new":
                t_new[0] = time.monotonic()
            if is_run_exec:
                man = download_run(cfg, current_run, do, log)
                check_run(cfg, current_run, man, as_, study, record, log)
                save("running")
                current_run = None
        left = [r for r in cfg.runs if not run_complete(cfg, r)]
        if left:
            raise ColabError(f"the session finished but runs {left} are not complete locally")
        record["state"] = "ok"
    except BaseException as e:
        record["error"] = repr(e)
        record["state"] = "failed"
        if attempted_new:
            log(f"session failed ({e!r}); stopping the VM")
            if current_run and not run_complete(cfg, current_run):
                try:                                                         # best effort: whatever the failed run wrote
                    download_run(cfg, current_run, lambda st: runner(st.argv, timeout=st.timeout_s, log_path=None), log)
                except Exception as e2:  # noqa: BLE001
                    log(f"salvage of {current_run} failed: {e2!r}")
        raise
    finally:
        if attempted_new:
            stop = next(s for s in by_phase["finally"] if s.name == "stop")
            record["stop_ok"] = False
            for attempt in (1, 2):
                try:
                    res = do(dataclasses.replace(stop, phase="finally"))
                    if res.returncode == 0:
                        record["stop_ok"] = True
                        break
                except BaseException as e3:  # noqa: BLE001 - stopping is the one thing that must not be skipped
                    log(f"stop attempt {attempt} raised {e3!r}")
            if not record["stop_ok"]:
                log("!" * 70)
                log(f"!!! `colab stop -s {cfg.session_name}` DID NOT SUCCEED. A billed VM may still be running.")
                log("!!! Run `colab sessions`, then `colab stop -s NAME` yourself, now.")
                log("!" * 70)
            usage_after = next((s for s in by_phase["finally"] if s.name == "usage-after"), None)
            if usage_after:
                try:
                    record["usage_after"] = do(usage_after).stdout.strip()
                except Exception as e4:  # noqa: BLE001
                    log(f"warning: usage-after failed: {e4!r}")
        save(record["state"] if record["state"] in ("ok", "failed") else "failed")
        (cfg.path / "study.done").write_text(_now() + "\n")                  # grade --follow stops on it
    return record


def aside_partial(cfg: StudyConfig, run: str) -> None:
    """A rerun starts from nothing: an incomplete earlier attempt's files are kept under runs/<run>.partial-N."""
    d = cfg.run_dir(run)
    if d.is_dir() and any(d.iterdir()):
        n = 1
        while (d.parent / f"{run}.partial-{n}").exists():
            n += 1
        d.rename(d.parent / f"{run}.partial-{n}")


def check_run(cfg: StudyConfig, run: str, man: dict | None, as_: Assumptions, study: dict, record: dict, log) -> None:
    """After a run: it must be complete, and its measured tokens/s is compared with the prediction (Rule 1)."""
    if man is None or man.get("state") != "complete" or not run_complete(cfg, run):
        raise ColabError(f"run {run} is not complete after its exec (manifest state {None if man is None else man.get('state')}); "
                         "the partial files are in the study folder")
    s = json.loads((cfg.run_dir(run) / f"{run}.summary.json").read_text(encoding="utf-8"))
    measured = s["train"]["tokens_per_s"]
    pred = as_.train_tok_s
    pr = next(r for r in study["runs"] if r["run"] == run)
    ratio = measured / pred
    record["per_run"][run] = {"predicted_tok_s": round(pred, 1), "measured_tok_s": measured, "ratio": round(ratio, 3),
                              "predicted_minutes": round(pr["minutes"], 1), "measured_train_s": s["train"]["step_s"],
                              "predicted_train_s": round(pr["parts"]["train"] * 60, 1), "peak_train_gib": max(c["peak_train_gib"] for c in s["checkpoints"]),
                              "load_s": s.get("load_s"), "checkpoints": [(c["tag"], c.get("gen_s"), c.get("eval_s")) for c in s["checkpoints"]]}
    log(f"{run}: {measured:.0f} training tokens/s measured, {pred:.0f} predicted (x{ratio:.2f}); peak {record['per_run'][run]['peak_train_gib']} GiB")
    if ratio < 1.0 / 3.0:
        msg = f"{run}: tokens/s is {1 / ratio:.1f}x BELOW the prediction (Rule 1: a miss of more than 3x is a bug)"
        if cfg.abort_when_slow:
            raise ColabError(msg + "; the remaining runs are not started (stopping the VM). Diagnose, then rerun the same command.")
        log("WARNING: " + msg)
    elif ratio > 3.0:
        log(f"WARNING: {run}: tokens/s is {ratio:.1f}x ABOVE the prediction (Rule 1: a miss of more than 3x is a bug); the budget is "
            "not at risk, so the study continues, but diagnose this before using the number")


def prepare_study(cfg: StudyConfig, train_rows: list[dict] | None = None) -> dict:
    """Write config.json and dev_prompts.jsonl (idempotent; refuses to change a prompts file a finished run used)."""
    cfg.path.mkdir(parents=True, exist_ok=True)
    prompts = build_dev_prompts()
    pp = cfg.path / "dev_prompts.jsonl"
    if pp.is_file() and read_jsonl(pp) != prompts and any((cfg.path / "runs").glob("*/*.dev_answers.jsonl")):
        raise ConfigError(f"{pp} differs from the dev prompts of this repository but answers for the old ones exist: use a new --study name")
    write_jsonl(pp, prompts)
    cfg_json = {"study": cfg.name, "percents": list(cfg.percents), "epochs": cfg.epochs, "ckpt_every": cfg.ckpt_every, "gpu": cfg.gpu,
                "repo": cfg.repo, "revision": cfg.revision, "base_license": cfg.base_license, "max_len": cfg.max_len, "seed": cfg.seed,
                "batch": cfg.batch, "grad_accum": cfg.grad_accum, "lr": cfg.lr, "warmup_ratio": cfg.warmup_ratio, "qlora": cfg.qlora,
                "step0_gen": cfg.step0_gen, "dev_max_new_tokens": cfg.dev_max_new_tokens,
                "train_sha256": sha256_file(TRAIN_PATH), "dev_sha256": sha256_file(DEV_PATH), "job_sha256": sha256_file(JOB_SCRIPT),
                "requirements_sha256": sha256_file(REQUIREMENTS), "n_dev_prompts": len(prompts), "n_dev_tasks": sum(len(p["task_ids"]) for p in prompts),
                "created": _now(), "created_by": "train/stage5.py"}
    old = job_read_json(cfg.path / "config.json")
    if old:
        cfg_json["created"] = old.get("created", cfg_json["created"])
    job.write_json_atomic(cfg.path / "config.json", cfg_json)
    return cfg_json


def job_read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ================================================================ grading the dev answers (CPU)

def answer_files(study_dir: Path) -> list[tuple[str, str, Path]]:
    """(run, tag, path) of every dev-answers file downloaded for the study."""
    out = []
    suffix = ".dev_answers.jsonl"
    for p in sorted((study_dir / "runs").glob(f"*/*{suffix}")):
        run = p.name.split(".", 1)[0]
        out.append((run, p.name[len(run) + 1:-len(suffix)], p))      # the tag may hold a dot: e0.5
    return out


def rung_of(run: str, tag: str) -> str:
    return "step0" if tag == "step0" else f"{run}@{tag}"


def generation_rows(prompts: list[dict], answers: list[dict], tasks: dict, rung: str, model: str) -> list[dict]:
    """The dev answers of one checkpoint as bake-off generation rows, one per dev task: the answer to a prompt is the answer of
    every task sharing it (T = 0)."""
    from bakeoff.extract import extract_code
    by_prompt = {a["prompt_id"]: a for a in answers}
    rows = []
    for p in prompts:
        a = by_prompt.get(p["id"])
        if a is None:
            continue
        ex = extract_code(a["text"])
        for k, tid in enumerate(p["task_ids"]):
            t = tasks.get(tid)
            if t is None:
                continue
            rows.append({"rung": rung, "model": model, **t.meta(), "mode": "greedy", "sample": 0, "reply": a["text"], "code": ex.code,
                         "extract": ex.path, "think_block": ex.think, "eval_count": a["n_tokens"], "done_reason": a["finish"],
                         "truncated": a["finish"] == "length", "reused_from": None if k == 0 else p["task_ids"][0]})
    return rows


def ingest(study_dir: Path, tasks: dict | None = None, prompts: list[dict] | None = None) -> int:
    """Append the dev answers not yet in grading/generations.jsonl. Returns the number of rows added."""
    from bakeoff.store import JsonlWriter, RunDir, keys
    if tasks is None:
        from bakeoff.tasks import load_tasks
        tasks = {t.id: t for t in load_tasks(("dev",))}
    prompts = prompts or read_jsonl(study_dir / "dev_prompts.jsonl")
    gdir = RunDir(study_dir / "grading")
    have = keys(read_jsonl(gdir.generations))
    added = 0
    with JsonlWriter(gdir.generations) as out:
        for run, tag, path in answer_files(study_dir):
            rung = rung_of(run, tag)
            for r in generation_rows(prompts, read_jsonl(path), tasks, rung, f"{run}/{tag}"):
                if (r["rung"], r["task_id"], r["mode"], r["sample"]) not in have:
                    out.write(r)
                    added += 1
    return added


def grade_study(study_dir: Path, *, workers: int = 4, follow: bool = False, poll_s: float = 15.0, tasks: dict | None = None,
                log=lambda m: print(m, file=sys.stderr, flush=True)) -> dict:
    """Ingest and grade (resumable). With ``follow``, keep polling the downloads until the driver writes study.done."""
    from bakeoff.grade import GradeSettings, grade_run
    from bakeoff.store import RunDir
    gdir = RunDir(study_dir / "grading")
    totals = {"graded": 0, "executed": 0, "cached": 0, "grader_errors": 0, "ingested": 0}
    while True:
        finished = (study_dir / "study.done").exists()                         # read before the answers, so nothing is missed
        n = ingest(study_dir, tasks)
        totals["ingested"] += n
        if len(read_jsonl(gdir.generations)) > len(read_jsonl(gdir.grades)):
            c = grade_run(gdir, GradeSettings(workers=workers))
            for k in ("graded", "executed", "cached", "grader_errors"):
                totals[k] += c[k]
            log(f"ingested {n} dev answers, graded {c['graded']} ({c['grader_errors']} grader errors)")
        if not follow or (finished and not n):
            return totals
        time.sleep(poll_s)


# ================================================================ the report

def run_curves(study_dir: Path) -> dict[str, dict]:
    """{run: summary.json} for every downloaded run, plus the training curve (step, loss) from its log."""
    out = {}
    for p in sorted((study_dir / "runs").glob("*/*.summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        run = p.name.split(".")[0]
        log = read_jsonl(p.with_name(f"{run}.log.jsonl"))
        s["curve"] = [(e["step"], e["loss"]) for e in log if e.get("event") == "step"]
        out[run] = s
    return out


def tag_epoch(tag: str) -> float:
    return 0.0 if tag == "step0" else float(tag[1:])


def pick_best(rows: list[dict]) -> dict | None:
    """The checkpoint with the highest dev pass@1; ties go to the lower dev loss, then to the earlier checkpoint."""
    rows = [r for r in rows if r.get("pass1") is not None and r["tag"] != "step0"]
    if not rows:
        return None
    return sorted(rows, key=lambda r: (-r["pass1"], r.get("dev_loss") if r.get("dev_loss") is not None else math.inf, r["epoch"]))[0]


def _f(x):
    """NaN -> None, so rows are valid JSON and comparable."""
    return None if isinstance(x, float) and math.isnan(x) else x


def summarize_study(study_dir: Path, *, boot: int = 200) -> dict:
    from bakeoff.report import GROUPS, Outcomes, stat_block
    from bakeoff.stats import paired_compare
    from bakeoff.store import RunDir
    gdir = RunDir(study_dir / "grading")
    grades, gens = read_jsonl(gdir.grades), read_jsonl(gdir.generations)
    runs = run_curves(study_dir)
    out = Outcomes(grades)
    preds = dict((g[0], g[2]) for g in GROUPS)
    sel = {k: out.ids(preds[k]) for k in ("all", "trained families", "held-out families", "5.0", "timeless")}
    by_rung: dict[str, dict[str, dict]] = {}
    for g in gens:
        by_rung.setdefault(g["rung"], {})[g["cluster"]] = g           # one answer per prompt (the tasks of a prompt share it)
    step0 = next((s["step0"] for s in runs.values() if "step0" in s), {})
    entries = [("step0", "step0", None, step0)] + [(rung_of(r, c["tag"]), c["tag"], r, c) for r, s in runs.items() for c in s["checkpoints"]]
    rows = []
    for rung, tag, run, ck in entries:
        if rung not in out.greedy:
            continue
        blocks = {k: stat_block(out, rung, ids, 5, boot, 0) for k, ids in sel.items()}
        distinct = by_rung.get(rung, {})
        rows.append({"rung": rung, "run": run, "tag": tag, "epoch": tag_epoch(tag), "step": ck.get("step", 0),
                     "pass1": _f(blocks["all"]["pass1"]), "se1": _f(blocks["all"]["se1_cluster"]), "n": blocks["all"]["n1"],
                     "pass1_trained": _f(blocks["trained families"]["pass1"]), "pass1_held_out": _f(blocks["held-out families"]["pass1"]),
                     "pass1_5_0": _f(blocks["5.0"]["pass1"]), "pass1_timeless": _f(blocks["timeless"]["pass1"]),
                     "train_loss": ck.get("train_loss_window"), "probe_loss": ck.get("probe_loss"), "dev_loss": ck.get("dev_loss"),
                     "tok_s_window": ck.get("tok_s_window"), "gen_s": ck.get("gen_s"), "adapter": f"{run}.{tag}.adapter.tar" if run else None,
                     "cut_at_cap": sum(1 for g in distinct.values() if g["truncated"]),
                     "fenced": sum(1 for g in distinct.values() if g["extract"] == "fence"), "answers": len(distinct)})
    paired = []
    if any(r["rung"] == "step0" for r in rows):
        for r in rows:
            if r["rung"] == "step0":
                continue
            ids = [t for t in sel["all"] if t in out.greedy.get(r["rung"], {}) and t in out.greedy["step0"]]
            if ids:
                pc = paired_compare([out.greedy[r["rung"]][t] for t in ids], [out.greedy["step0"][t] for t in ids], [out.meta[t]["cluster"] for t in ids])
                paired.append({"a": r["rung"], "b": "step0", **{k: _f(v) for k, v in pc.items()}})
    best = {}
    for run in runs:
        b = pick_best([r for r in rows if r["run"] == run])
        if b:
            best[run] = b
    final = {run: max((x for x in rows if x["run"] == run), key=lambda x: x["epoch"])["rung"] for run in runs if any(x["run"] == run for x in rows)}
    conv = []
    order = sorted(best, key=lambda r: int(r[1:]))
    for lo, hi in zip(order, order[1:]):
        for which, pick in (("best", {k: v["rung"] for k, v in best.items()}), ("final", final)):
            a, b = pick[hi], pick[lo]
            ids = [t for t in sel["all"] if t in out.greedy.get(a, {}) and t in out.greedy.get(b, {})]
            if ids:
                pc = paired_compare([out.greedy[a][t] for t in ids], [out.greedy[b][t] for t in ids], [out.meta[t]["cluster"] for t in ids])
                noise = _f(pc.get("se_cluster"))
                noise = noise if noise is not None else _f(pc["se"])
                conv.append({"from": lo, "to": hi, "which": which, "a": a, "b": b, "diff": pc["diff"], "noise_se": noise,
                             "p_sign_cluster": _f(pc.get("p_sign_cluster")), "p_mcnemar": pc["p"],
                             "moves_more_than_noise": None if noise is None else abs(pc["diff"]) > noise})
    return {"rows": rows, "best": {k: v["rung"] for k, v in best.items()}, "final": final, "paired_vs_step0": paired, "convergence": conv,
            "tasks": {k: len(v) for k, v in sel.items()},
            "runs": {r: {k: s.get(k) for k in ("frac", "n_train", "train", "load_s", "trainable_params", "dtype", "loss_mask", "dev_loss_step0",
                                               "max_steps", "state", "subset")} for r, s in runs.items()}}


def pct(x, digits=1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def fnum(x, fmt=".4f") -> str:
    return "-" if x is None else format(x, fmt)


def render_report(s: dict, study_name: str) -> str:
    L = [f"# Stage 5 convergence study: {study_name}", ""]
    L.append(f"Dev pass@1 at T = 0 on the {s['tasks']['all']} cvbench dev tasks ({s['tasks']['trained families']} in trained families, "
             f"{s['tasks']['held-out families']} in held-out families, {s['tasks']['5.0']} subset 5.0). SE is the prompt-cluster-robust one "
             "(tasks of one prompt share their answer). Selecting the best of four checkpoints on this set inflates its score: "
             "the test set (Stage 6) is the unbiased number.")
    L += ["", "## Checkpoints", "", "| checkpoint | step | train loss (window) | probe loss | dev loss | pass@1 all (SE) | trained | held-out | 5.0 | timeless | cut at cap | fenced | train tok/s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        star = " **best**" if s["best"].get(r["run"]) == r["rung"] else ""
        L.append(f"| {r['rung']}{star} | {r['step']} | {fnum(r['train_loss'])} | {fnum(r['probe_loss'])} | {fnum(r['dev_loss'])} | {pct(r['pass1'])} ({pct(r['se1'])}) | "
                 f"{pct(r['pass1_trained'])} | {pct(r['pass1_held_out'])} | {pct(r['pass1_5_0'])} | {pct(r['pass1_timeless'])} | {r['cut_at_cap']}/{r['answers']} | "
                 f"{r['fenced']}/{r['answers']} | {fnum(r['tok_s_window'], '.0f')} |")
    L += ["", "The train-validation gap is probe loss (a fixed sample of the run's own training rows) against dev loss, at the same checkpoint.", ""]
    L += ["## Paired against the untrained model (same dev tasks)", "", "| checkpoint | diff (pts) | SE cluster | p (McNemar) | p (sign over prompts) | gained | lost |", "|---|---|---|---|---|---|---|"]
    for p in s["paired_vs_step0"]:
        L.append(f"| {p['a']} | {pct(p['diff'])} | {pct(p.get('se_cluster'))} | {fnum(p['p'], '.3g')} | {fnum(p.get('p_sign_cluster'), '.3g')} | {p['a_only']} | {p['b_only']} |")
    L += ["", "## Convergence rule (PLAN.md 4.3): stop growing when a doubling moves dev pass@1 by less than the noise", "",
          "| from -> to | compared | diff (pts) | noise (SE, pts) | p (sign over prompts) | moves more than the noise |", "|---|---|---|---|---|---|"]
    for c in s["convergence"]:
        L.append(f"| {c['from']} -> {c['to']} | {c['which']}: {c['a']} vs {c['b']} | {pct(c['diff'])} | {pct(c['noise_se'])} | {fnum(c['p_sign_cluster'], '.3g')} | {'yes' if c['moves_more_than_noise'] else 'no'} |")
    L += ["", "The decision is Phil's (PLAN.md section 7): this table is the evidence, not the answer."]
    return "\n".join(L) + "\n"


def write_report(study_dir: Path, study_name: str) -> dict:
    s = summarize_study(study_dir)
    (study_dir / "report.md").write_text(render_report(s, study_name), encoding="utf-8")
    job.write_json_atomic(study_dir / "summary.json", s)
    with (study_dir / "curves.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "run", "tag", "epoch", "step", "train_loss", "probe_loss", "dev_loss", "pass1", "se1", "pass1_trained", "pass1_held_out", "pass1_5_0"])
        for r in s["rows"]:
            w.writerow([r["rung"], r["run"], r["tag"], r["epoch"], r["step"], r["train_loss"], r["probe_loss"], r["dev_loss"], r["pass1"], r["se1"],
                        r["pass1_trained"], r["pass1_held_out"], r["pass1_5_0"]])
    return s


# ================================================================ commands

def _default_colab_bin() -> str:
    return shutil.which("colab") or str(Path.home() / "venvs" / "colab" / "bin" / "colab")


def _percents(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _assumptions(a) -> Assumptions:
    kw = {k: getattr(a, k) for k in ("mfu", "tok_s", "gen_step_s", "download_mb_s") if getattr(a, k, None) is not None}
    return assumptions_for(a.gpu, a.cu_per_hour, **kw)


def cmd_predict(a) -> int:
    train_rows, dev_rows = read_jsonl(TRAIN_PATH), read_jsonl(DEV_PATH)
    tokens = load_row_tokens(train_rows, dev_rows)
    as_ = _assumptions(a)
    pcts = _percents(a.runs)
    study = predict_study(tokens, train_rows, pcts, as_, epochs=a.epochs, ckpt_every=a.ckpt_every, step0_gen=not a.no_step0_gen, probe_rows=a.probe_rows)
    kw = dict(epochs=a.epochs, ckpt_every=a.ckpt_every, probe_rows=a.probe_rows)
    extra = extra_round(tokens, train_rows, a.extra_round, as_, **kw) if a.extra_round else None
    scen = []
    for name, mfu in (("35 % utilization (COMPUTE.md)", 0.35), ("10 % utilization (2x slower than predicted, still not a bug)", 0.10)):
        sa = dataclasses.replace(as_, mfu=mfu, tok_s=None)
        scen.append((name, predict_study(tokens, train_rows, pcts, sa, step0_gen=not a.no_step0_gen, **kw),
                     extra_round(tokens, train_rows, a.extra_round, sa, **kw) if a.extra_round else None))
    split = None
    if len(pcts) > 1:
        split = [(f"A (runs {pcts[0]})", predict_study(tokens, train_rows, pcts[:1], as_, step0_gen=not a.no_step0_gen, **kw)),
                 (f"B (runs {','.join(map(str, pcts[1:]))}, no step-0 generation)", predict_study(tokens, train_rows, pcts[1:], as_, step0_gen=False, **kw))]
    if a.json:
        print(json.dumps({"study": study, "extra": extra, "cap_cu": a.cap_cu, "spent_cu": a.spent_cu}, indent=1, default=str))
    else:
        print(render_prediction(study, extra, as_, cap_cu=a.cap_cu, spent_cu=a.spent_cu, token_source=tokens.source, scenarios=scen, split=split))
    if tokens.source != "tokenizer":
        print("note: row token counts are an estimate (characters / %.1f); run `python -m train.stage5 tokens` in the train venv for the real ones" % CHARS_PER_TOKEN,
              file=sys.stderr)
    return 0


def cmd_tokens(a) -> int:
    """Real token counts of every row under the Qwen3.5 tokenizer (train venv), cached for predict."""
    from transformers import AutoTokenizer
    train_rows, dev_rows = read_jsonl(TRAIN_PATH), read_jsonl(DEV_PATH)
    tok = AutoTokenizer.from_pretrained(BASES[BASE]["upstream"], local_files_only=True)
    pf_t, pf_d = job.preflight(tok, train_rows, a.max_len, "train"), job.preflight(tok, dev_rows, a.max_len, "dev")
    chars = sum(len(t) for t in pf_t["texts"])
    cache = token_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    job.write_json_atomic(cache, {"train": pf_t["lengths"], "dev": pf_d["lengths"], "train_sha256": sha256_file(TRAIN_PATH), "dev_sha256": sha256_file(DEV_PATH),
                                  "tokenizer": BASES[BASE]["upstream"], "chars_per_token": round(chars / sum(pf_t["lengths"]), 3), "created": _now()})
    print(f"train {pf_t['rows']} rows: mean {pf_t['mean']} p99 {pf_t['p99']} max {pf_t['max']}; dev {pf_d['rows']} rows: mean {pf_d['mean']} max {pf_d['max']}; "
          f"{chars / sum(pf_t['lengths']):.2f} characters per token; cached in {cache}")
    return 0


def cmd_check(a) -> int:
    """The pre-flight of the job on the real tokenizer, plus the repository-side checks (train venv, no GPU)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASES[BASE]["upstream"], local_files_only=True)
    train_rows, dev_rows = read_jsonl(TRAIN_PATH), read_jsonl(DEV_PATH)
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        if job.validate_rows(rows, name) != SYSTEM:
            raise SystemExit(f"{name}: the rows' system prompt is not train.config.SYSTEM (Rule 7)")
        s = job.preflight(tok, rows, a.max_len, name)
        print(json.dumps({k: v for k, v in s.items() if k not in ("texts", "lengths")}))
    prompts = build_dev_prompts()
    for p in prompts:
        job.check_prompt_template(tok, p["messages"][0]["content"], p["messages"][1]["content"])
    print(f"ok: {len(prompts)} distinct dev prompts ({sum(len(p['task_ids']) for p in prompts)} dev tasks) render as the serve-time prompt under the "
          "tokenizer's own template, non-thinking")
    return 0


def cmd_prepare(a) -> int:
    cfg = _study_config(a)
    c = prepare_study(cfg)
    print(f"prepared {cfg.path}: {c['n_dev_prompts']} distinct dev prompts for {c['n_dev_tasks']} dev tasks")
    return 0


def _study_config(a) -> StudyConfig:
    return StudyConfig(
        name=a.study, root=Path(a.root) if a.root else None, percents=_percents(a.runs), epochs=a.epochs, ckpt_every=a.ckpt_every, gpu=a.gpu,
        revision=a.revision, base_license=a.base_license, max_len=a.max_len, qlora=a.qlora, step0_gen=not a.no_step0_gen,
        cu_per_hour=a.cu_per_hour, cap_cu=getattr(a, "cap_cu", CAP_CU), spent_cu=getattr(a, "spent_cu", 0.0),
        predicted_minutes=getattr(a, "predicted_minutes", None), predicted_cu=getattr(a, "predicted_cu", None),
        session=getattr(a, "session", None), colab_bin=getattr(a, "colab_bin", None) or _default_colab_bin(), colab_auth=getattr(a, "colab_auth", None),
        abort_when_slow=not getattr(a, "no_abort_when_slow", False), dev_max_new_tokens=a.dev_max_new_tokens)


def cmd_colab(a) -> int:
    cfg = _study_config(a)
    try:
        rec = run_study(cfg, approved=a.i_have_approval, dry_run=a.dry_run)
    except ApprovalRequired as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except ColabError as e:
        print(f"colab study failed: {e}", file=sys.stderr)
        return 1
    if rec.get("stop_ok") is False:
        return 3
    return 0


def _study_dir(a) -> Path:
    return (Path(a.root) if a.root else STAGE5_ROOT) / a.study


def cmd_grade(a) -> int:
    sd = _study_dir(a)
    if not (sd / "dev_prompts.jsonl").is_file():
        raise SystemExit(f"{sd / 'dev_prompts.jsonl'} is missing: run `prepare` or `colab` first")
    t = grade_study(sd, workers=a.workers, follow=a.follow, poll_s=a.poll)
    print(f"ingested {t['ingested']}, graded {t['graded']} ({t['executed']} executed, {t['cached']} reused, {t['grader_errors']} grader errors) in {sd / 'grading'}")
    return 1 if t["grader_errors"] else 0


def cmd_report(a) -> int:
    sd = _study_dir(a)
    s = write_report(sd, a.study)
    for r in s["rows"]:
        print(f"{r['rung']:<12} pass@1 {pct(r['pass1'])} ± {pct(r['se1'])}  dev loss {fnum(r['dev_loss'])}  probe loss {fnum(r['probe_loss'])}")
    print(f"best checkpoints: {s['best']}; wrote {sd / 'report.md'}, summary.json, curves.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m train.stage5", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, study=True):
        p.add_argument("--runs", default="25,50,100", help="percents of train.jsonl, one run each (nested subsets)")
        p.add_argument("--epochs", type=float, default=2.0)
        p.add_argument("--ckpt-every", type=float, default=0.5, help="epochs between checkpoints")
        p.add_argument("--gpu", default="A100", help=f"one of {GPU_CHOICES}")
        p.add_argument("--cu-per-hour", type=float, help="needed for H100 and G4 (COMPUTE.md lists T4 1.2, L4 1.7, A100 5.4)")
        p.add_argument("--no-step0-gen", action="store_true", help="skip the untrained model's dev answers")
        if study:
            p.add_argument("--study", required=True, help="folder name under --root")
            p.add_argument("--root", help=f"parent folder of studies (default {STAGE5_ROOT})")
            p.add_argument("--revision", default=job.DEFAULT_REVISION)
            p.add_argument("--base-license", default=BASES[BASE]["license"], help="license of the base model (CLAUDE.md Rule 5)")
            p.add_argument("--max-len", type=int, default=4608)
            p.add_argument("--qlora", action="store_true", help="4-bit base: only for a T4/L4 fallback or a smoke test")
            p.add_argument("--dev-max-new-tokens", type=int, default=2048)

    p = sub.add_parser("predict", help="predicted minutes and CU of the convergence study and one extra round")
    common(p, study=False)
    p.add_argument("--mfu", type=float, help="utilization of the GPU's peak (default 0.20)")
    p.add_argument("--tok-s", type=float, help="override the derived training tokens/s")
    p.add_argument("--gen-step-s", type=float, help="seconds per decoded token of the dev generation (default 0.09)")
    p.add_argument("--download-mb-s", type=float)
    p.add_argument("--probe-rows", type=int, default=64)
    p.add_argument("--extra-round", type=int, default=100, help="percent of the data of the one extra round to leave room for (0: none)")
    p.add_argument("--cap-cu", type=float, default=CAP_CU)
    p.add_argument("--spent-cu", type=float, default=0.0, help="CU already spent on Stage 5")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("tokens", help="real token counts per row (train venv), cached")
    p.add_argument("--max-len", type=int, default=4608)
    p.set_defaults(fn=cmd_tokens)

    p = sub.add_parser("check", help="template, markers, lengths and dev prompts on the real tokenizer (train venv)")
    p.add_argument("--max-len", type=int, default=4608)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("prepare", help="write the study folder's config and dev prompts")
    common(p)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("colab", help="run the convergence study on a Colab GPU (billed; needs --i-have-approval) or --dry-run it")
    common(p)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the commands and the launchers; run nothing")
    mode.add_argument("--i-have-approval", action="store_true", help="Phil has approved this spend (CLAUDE.md Rule 2)")
    p.add_argument("--predicted-minutes", type=float, help="the whole session incl. setup (LOGBOOK.md); every exec timeout is 2x its share")
    p.add_argument("--predicted-cu", type=float)
    p.add_argument("--cap-cu", type=float, default=CAP_CU, help="Phil's Stage 5 cap; the driver refuses a session whose 2x kill point would pass it")
    p.add_argument("--spent-cu", type=float, default=0.0, help="CU already spent on Stage 5 (from LOGBOOK.md)")
    p.add_argument("--session", help="Colab session name (default stage5-<study>)")
    p.add_argument("--colab-bin")
    p.add_argument("--colab-auth", choices=("oauth2", "adc"))
    p.add_argument("--no-abort-when-slow", action="store_true", help="only warn when a run is >3x slower than predicted (default: stop the study)")
    p.set_defaults(fn=cmd_colab)

    p = sub.add_parser("grade", help="grade the downloaded dev answers on the CPU (resumable; --follow overlaps the next run's GPU time)")
    p.add_argument("--study", required=True)
    p.add_argument("--root")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--poll", type=float, default=15.0)
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("report", help="curves, the best checkpoint by dev pass@1, the convergence rule")
    p.add_argument("--study", required=True)
    p.add_argument("--root")
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
