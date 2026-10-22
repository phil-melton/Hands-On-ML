"""Run the teacher job through the Colab CLI (COMPUTE.md Path B). Nothing here runs unless ``approved=True``.

The sequence of one job (``plan`` returns it; ``--dry-run`` prints it, executes nothing):

    usage-before   colab usage                                                 (recorded in session_<stage>.json)
    new            colab new -s NAME --gpu L4
    [install]      colab install -s NAME PKG...                                (--install-via colab; default: pip, below)
    upload-job     colab upload -s NAME colab_job.py /content/colab_job.py
    upload-prompts colab upload -s NAME <stage prompts> /content/<stage>_prompts.jsonl
    [upload-partial]  the completions and timing already downloaded, so a job killed half-way resumes
    exec           colab exec -s NAME -f launch_<stage>.py --timeout T         (the launcher: pip install, then colab_job.py)
    download-completions, download-timing    colab download -s NAME /content/<stage>_completions.jsonl <run folder>
    stop           colab stop -s NAME                                          ALWAYS, in ``finally``
    usage-after    colab usage

Guards (CLAUDE.md Rules 1, 2 and 5). A real run needs ``approved=True`` (the CLI's ``--i-have-approval``), a predicted
duration and CU (written to LOGBOOK.md first; the ``exec`` timeout is 2x the predicted minutes, which is Rule 2's
"kill any job that passes 2x its predicted CU"), and the teacher's license. The GPU name is checked against the CLI's list
because the CLI silently turns an unknown ``--gpu`` into an A100 (5.4 CU/h). ``stop`` runs even when ``new`` itself failed
or timed out (the VM may exist), is retried once, and a failure to stop is reported loudly. A stage whose completions are
already complete locally starts no VM. If the job fails, whatever it wrote is downloaded (best effort) before the stop, and
rerunning the same command resumes from it.

Verified against ``colab help <cmd>`` (colab 0.7.5): ``new`` takes ``-s/--session`` and ``--gpu`` (T4, L4, G4,
H100, A100); ``install [-s] [-r FILE] PKG...``; ``upload [-s] LOCAL REMOTE``; ``download [-s] REMOTE LOCAL``; ``exec [-s] [-f
FILE] [--timeout SECONDS, default 30] [--env K=V]``; ``stop [-s]``; ``usage`` (no options); the global ``--auth``
precedes the subcommand. tests/test_foundry.py re-checks every flag used against the live help when the CLI is installed.

Not confirmed (nothing was launched): (1) ``colab exec`` has no argv and its default timeout is 30 s, hence the launcher
and ``--timeout``; whether ``exec`` exits non-zero when the kernel raises is not documented, so success is judged by the
downloaded files, not the exit code. (2) ``colab install`` has no timeout option, so by default the launcher runs the
install under exec's timeout (``uv pip install --system`` if uv exists, else pip); ``--install-via colab`` uses the CLI's own
command. (3) Whether ``upload`` creates missing remote directories: every remote path is flat under /content. (4) That
``colab usage`` reflects the job's CU immediately after ``stop``. (5) The README says the global ``--auth`` defaults to adc,
``colab --help`` says oauth2; pass ``--colab-auth`` to pin it. (6) That the model's vLLM install resolves against Colab's
preinstalled torch: ``pip install vllm`` pulls its own torch and the job runs in a fresh subprocess so that it is used.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from foundry.store import FoundryRun, read_jsonl, write_json

GPU_CHOICES = ("T4", "L4", "G4", "A100", "H100")
JOB_SCRIPT = Path(__file__).resolve().parent / "colab_job.py"
REMOTE_JOB = "/content/colab_job.py"
DEFAULT_EXEC_TIMEOUT_S = 4 * 3600  # dry-run display only; a real run derives the timeout from the prediction

LAUNCHER_TEMPLATE = '''\
# Written by foundry/colab.py, run with `colab exec -f`. The kernel has no argv, so the job runs in a subprocess.
import shutil, subprocess, sys

JOB = @@JOB@@
ARGS = @@ARGS@@
PIP = @@PIP@@

if PIP:
    cmd = (["uv", "pip", "install", "--system", "-q"] if shutil.which("uv")
           else [sys.executable, "-m", "pip", "install", "-q"]) + PIP
    print("+", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode:
        raise RuntimeError("package install failed: " + " ".join(PIP))

cmd = [sys.executable, JOB] + ARGS
print("+", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc:
    raise RuntimeError(f"colab_job.py exited with status {rc}")
print("launcher: job finished OK", flush=True)
'''


class ConfigError(ValueError):
    pass


class ApprovalRequired(RuntimeError):
    pass


class ColabError(RuntimeError):
    pass


@dataclass(frozen=True)
class CmdResult:
    returncode: int
    stdout: str = ""
    seconds: float = 0.0


Runner = Callable[..., CmdResult]  # runner(argv, timeout=..., log_path=...) -> CmdResult


@dataclass
class JobConfig:
    run: FoundryRun
    stage: str = "generate"
    gpu: str = "L4"
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    revision: str | None = None   # model repo commit (Rule 5)
    quantization: str | None = None
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    k: int | None = None            # samples per prompt; None: 4 for generate, 1 for repair (repair "once")
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0                  # 0: off
    max_tokens: int = 2048
    seed: int = 0
    chunk: int = 50
    limit: int = 0
    enforce_eager: bool = False
    max_num_seqs: int = 0
    trust_remote_code: bool = False
    chat_template_kwargs: dict = field(default_factory=dict)
    packages: tuple[str, ...] = ("vllm",)
    install_via: str = "pip"        # pip: the launcher installs under exec's timeout; colab: `colab install`
    session: str | None = None
    predicted_minutes: float | None = None   # the job as one block: install, load, generate (LOGBOOK.md, Rule 1)
    predicted_cu: float | None = None
    teacher_license: str | None = None
    colab_bin: str = "colab"
    colab_auth: str | None = None
    exec_timeout_s: float | None = None      # default 2 x predicted_minutes (Rule 2)

    @property
    def samples(self) -> int:
        return self.k if self.k is not None else (4 if self.stage == "generate" else 1)

    @property
    def session_name(self) -> str:
        return self.session or f"foundry-{self.run.name}-{self.stage}"

    @property
    def timeout_s(self) -> float:
        if self.exec_timeout_s:
            return float(self.exec_timeout_s)
        return 2.0 * 60.0 * self.predicted_minutes if self.predicted_minutes else float(DEFAULT_EXEC_TIMEOUT_S)

    # remote files are flat under /content: whether `colab upload` makes missing directories is not documented
    def remote(self, what: str) -> str:
        return f"/content/{self.stage}_{what}"


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]
    timeout_s: float
    phase: str = "main"          # pre | main | finally
    required: bool = True
    log: bool = False            # tee the output to colab_<stage>.log
    note: str = ""

    def shell(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


# ---------------------------------------------------------------- configuration -> commands

def job_args(cfg: JobConfig) -> list[str]:
    """argv of colab_job.py for this job (the same parser that runs on the VM reads it)."""
    a = ["--prompts", cfg.remote("prompts.jsonl"), "--out-dir", "/content",
         "--completions-name", f"{cfg.stage}_completions.jsonl", "--timing-name", f"{cfg.stage}_timing.json",
         "--model", cfg.model, "--dtype", "auto", "--max-model-len", str(cfg.max_model_len),
         "--gpu-memory-utilization", str(cfg.gpu_memory_utilization), "-k", str(cfg.samples),
         "--temperature", str(cfg.temperature), "--top-p", str(cfg.top_p), "--max-tokens", str(cfg.max_tokens),
         "--seed", str(cfg.seed), "--chunk", str(cfg.chunk),
         "--chat-template-kwargs", json.dumps(cfg.chat_template_kwargs, sort_keys=True)]
    if cfg.top_k:
        a += ["--top-k", str(cfg.top_k)]
    if cfg.quantization:
        a += ["--quantization", cfg.quantization]
    if cfg.revision:
        a += ["--revision", cfg.revision]
    if cfg.limit:
        a += ["--limit", str(cfg.limit)]
    if cfg.max_num_seqs:
        a += ["--max-num-seqs", str(cfg.max_num_seqs)]
    if cfg.enforce_eager:
        a.append("--enforce-eager")
    if cfg.trust_remote_code:
        a.append("--trust-remote-code")
    return a


def launcher_source(cfg: JobConfig) -> str:
    pip = list(cfg.packages) if cfg.install_via == "pip" else []
    return (LAUNCHER_TEMPLATE.replace("@@JOB@@", repr(REMOTE_JOB)).replace("@@ARGS@@", repr(job_args(cfg)))
            .replace("@@PIP@@", repr(pip)))


def validate(cfg: JobConfig, *, real_run: bool) -> None:
    if cfg.stage not in ("generate", "repair"):
        raise ConfigError(f"unknown stage {cfg.stage!r}")
    if cfg.gpu not in GPU_CHOICES:
        raise ConfigError(f"unknown GPU {cfg.gpu!r}; choose from {GPU_CHOICES} (the CLI would silently use an A100)")
    if cfg.install_via not in ("pip", "colab"):
        raise ConfigError("install_via must be 'pip' or 'colab'")
    if cfg.samples < 1:
        raise ConfigError("k must be at least 1")
    if real_run:
        missing = [n for n, v in (("predicted_minutes", cfg.predicted_minutes), ("predicted_cu", cfg.predicted_cu)) if not v or v <= 0]
        if missing:
            raise ConfigError(f"{', '.join(missing)} required: write the prediction to LOGBOOK.md first (CLAUDE.md Rule 1); "
                              "the exec timeout is 2x the predicted minutes (Rule 2)")
        if not cfg.teacher_license:
            raise ConfigError("teacher_license required: record the license of every model you use (CLAUDE.md Rule 5)")


def downloads(cfg: JobConfig) -> list[tuple[str, str, Path]]:
    st = cfg.run.stage(cfg.stage)
    return [("download-completions", cfg.remote("completions.jsonl"), st.completions),
            ("download-timing", cfg.remote("timing.json"), st.timing)]


def _partial_uploads(cfg: JobConfig) -> list[Step]:
    st, pre = cfg.run.stage(cfg.stage), [cfg.colab_bin] + (["--auth", cfg.colab_auth] if cfg.colab_auth else [])
    steps = []
    if st.completions.is_file() and st.completions.stat().st_size:
        steps.append(Step("upload-partial-completions", pre + ["upload", "-s", cfg.session_name, str(st.completions),
                                                                  cfg.remote("completions.jsonl")], 600,
                          note="resume: the completions downloaded by an earlier attempt"))
        if st.timing.is_file():
            steps.append(Step("upload-partial-timing", pre + ["upload", "-s", cfg.session_name, str(st.timing),
                                                                 cfg.remote("timing.json")], 600,
                              note="resume: so the timing accumulates"))
    return steps


def plan(cfg: JobConfig) -> list[Step]:
    """Every command of the job in order, with its phase. The launcher path is where a real run writes it."""
    st = cfg.run.stage(cfg.stage)
    pre = [cfg.colab_bin] + (["--auth", cfg.colab_auth] if cfg.colab_auth else [])
    s = cfg.session_name
    steps = [Step("usage-before", pre + ["usage"], 120, "pre", required=False)]
    steps.append(Step("new", pre + ["new", "-s", s, "--gpu", cfg.gpu], 900, note=f"billed from here ({cfg.gpu})"))
    if cfg.install_via == "colab":
        steps.append(Step("install", pre + ["install", "-s", s, *cfg.packages], 2400, note="the CLI has no timeout option"))
    steps += [Step("upload-job", pre + ["upload", "-s", s, str(JOB_SCRIPT), REMOTE_JOB], 600),
              Step("upload-prompts", pre + ["upload", "-s", s, str(st.prompts), cfg.remote("prompts.jsonl")], 600)]
    steps += _partial_uploads(cfg)
    launcher = cfg.run.path / f"launch_{cfg.stage}.py"
    steps.append(Step("exec", pre + ["exec", "-s", s, "-f", str(launcher), "--timeout", str(int(cfg.timeout_s))],
                      cfg.timeout_s + 300, log=True, note="pip install (if --install-via pip), model load, generation"))
    steps += [Step(name, pre + ["download", "-s", s, remote, str(local)], 900) for name, remote, local in downloads(cfg)]
    steps.append(Step("stop", pre + ["stop", "-s", s], 300, "finally", required=False, note="always runs"))
    steps.append(Step("usage-after", pre + ["usage"], 120, "finally", required=False))
    return steps


def missing_prompts(cfg: JobConfig) -> int:
    """Prompts that do not yet have their k samples in the local completions file (a prompt row may carry its own ``k``)."""
    st = cfg.run.stage(cfg.stage)
    prompts = read_jsonl(st.prompts)
    if cfg.limit:
        prompts = prompts[:cfg.limit]
    have: dict[str, set[int]] = {}
    for r in read_jsonl(st.completions):
        have.setdefault(r["id"], set()).add(r["sample"])
    return sum(1 for p in prompts if not set(range(int(p.get("k", cfg.samples)))) <= have.get(p["id"], set()))


# ---------------------------------------------------------------- execution

def subprocess_runner(argv: list[str], *, timeout: float, log_path: Path | None = None) -> CmdResult:
    """Run a command, echo its output to stderr as it arrives, tee it to ``log_path``, kill its process group on timeout
    or Ctrl-C."""
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1, start_new_session=True)
    lines: list[str] = []
    fh = log_path.open("a", encoding="utf-8") if log_path else None

    def pump() -> None:
        for line in proc.stdout:
            lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()
            if fh:
                fh.write(line)
                fh.flush()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    try:
        rc = proc.wait(timeout=timeout)
    except BaseException:  # TimeoutExpired, KeyboardInterrupt
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    finally:
        t.join(timeout=10)
        if fh:
            fh.close()
    return CmdResult(rc, "".join(lines), time.monotonic() - t0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def print_plan(cfg: JobConfig, steps: list[Step], log: Callable[[str], None]) -> None:
    log(f"# dry run: stage {cfg.stage}, session {cfg.session_name}, {cfg.gpu}, model {cfg.model}, k = {cfg.samples}")
    log(f"# exec timeout {int(cfg.timeout_s)} s" + ("" if cfg.predicted_minutes else " (no prediction given; a real run requires one)"))
    for st in steps:
        log(f"[{st.phase:7}] {st.name:<26} {st.shell()}" + (f"    # {st.note}" if st.note else ""))
    log("# launcher (launch_%s.py):" % cfg.stage)
    for line in launcher_source(cfg).splitlines():
        log("#   " + line)


def run_job(cfg: JobConfig, *, approved: bool = False, dry_run: bool = False, runner: Runner | None = None,
            log: Callable[[str], None] = print) -> dict:
    """Run (or, with ``dry_run``, show) one teacher job. Returns the session record. See the module docstring."""
    validate(cfg, real_run=not dry_run)
    st = cfg.run.stage(cfg.stage)
    if not read_jsonl(st.prompts):
        raise ConfigError(f"{st.prompts} is missing or empty; run `prepare` (or `repair`) first")
    steps = plan(cfg)
    todo = missing_prompts(cfg)
    if dry_run:
        print_plan(cfg, steps, log)
        if todo == 0:
            log("# note: every prompt already has its samples locally; a real run would start no VM")
        return {"dry_run": True, "commands": [s.argv for s in steps], "steps": [s.name for s in steps], "missing_prompts": todo}
    if not approved:
        raise ApprovalRequired("a billed Colab job needs Phil's explicit OK: pass --i-have-approval (or use --dry-run)")
    if todo == 0:
        log(f"stage {cfg.stage}: all prompts already have {cfg.samples} samples in {st.completions}; no VM started")
        return {"skipped": True, "missing_prompts": 0}

    runner = runner or subprocess_runner
    by_phase = {p: [s for s in steps if s.phase == p] for p in ("pre", "main", "finally")}
    record: dict = {"stage": cfg.stage, "session": cfg.session_name, "gpu": cfg.gpu, "model": cfg.model, "revision": cfg.revision,
                    "quantization": cfg.quantization, "k": cfg.samples, "n_prompts_missing": todo,
                    "chat_template_kwargs": cfg.chat_template_kwargs, "max_tokens": cfg.max_tokens,
                    "max_model_len": cfg.max_model_len, "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
                    "predicted_minutes": cfg.predicted_minutes, "predicted_cu": cfg.predicted_cu,
                    "teacher_license": cfg.teacher_license, "exec_timeout_s": cfg.timeout_s, "install_via": cfg.install_via,
                    "packages": list(cfg.packages), "commands": [], "usage_before": None, "usage_after": None,
                    "started_at": _now(), "state": "starting", "error": None, "stop_ok": None}
    t_start = time.monotonic()

    def do(step: Step) -> CmdResult:
        log(f"$ {step.shell()}")
        res = runner(step.argv, timeout=step.timeout_s, log_path=st.log if step.log else None)
        record["commands"].append({"name": step.name, "argv": step.argv, "returncode": res.returncode,
                                   "seconds": round(res.seconds, 2)})
        if res.returncode and step.required:
            raise ColabError(f"{step.name} exited with status {res.returncode}: {step.shell()}")
        return res

    def save(state: str) -> None:
        record["state"] = state
        record["wall_s"] = round(time.monotonic() - t_start, 1)
        write_json(st.session, record)

    attempted_new = created = False
    try:
        for step in by_phase["pre"]:
            try:
                res = do(step)
                record["usage_before"] = res.stdout.strip()
            except Exception as e:  # noqa: BLE001 - usage is informational; a broken login will fail `new` just as well
                log(f"warning: {step.name} failed: {e!r}")
        cfg.run.path.mkdir(parents=True, exist_ok=True)
        (cfg.run.path / f"launch_{cfg.stage}.py").write_text(launcher_source(cfg), encoding="utf-8")
        for step in by_phase["main"]:
            if step.name == "new":
                attempted_new = True
                save("new-attempted")  # evidence of a possibly billed VM if this process dies hard
            do(step)
            if step.name == "new":
                created = True
            if step.name == "exec":
                log("exec finished; downloading results")
        left = missing_prompts(cfg)
        if left:
            raise ColabError(f"the job finished but {left} prompts still lack {cfg.samples} samples in {st.completions}")
        record["state"] = "ok"
    except BaseException as e:
        record["error"] = repr(e)
        record["state"] = "failed"
        if attempted_new:
            log(f"job failed ({e!r}); " + ("downloading whatever it wrote, then " if created else "") + "stopping the VM")
        if created:
            for name, remote, local in downloads(cfg):  # best effort: a partial result still resumes
                try:
                    runner(plan_download(cfg, remote, local), timeout=900, log_path=None)
                except Exception as e2:  # noqa: BLE001
                    log(f"salvage {name} failed: {e2!r}")
        raise
    finally:
        if attempted_new:
            stop = next(s for s in by_phase["finally"] if s.name == "stop")
            record["stop_ok"] = False
            for attempt in (1, 2):
                try:
                    res = do(stop)
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
        if attempted_new and usage_after:
            try:
                record["usage_after"] = do(usage_after).stdout.strip()
            except Exception as e4:  # noqa: BLE001
                log(f"warning: usage-after failed: {e4!r}")
        save(record["state"] if record["state"] in ("ok", "failed") else "failed")
    return record


def plan_download(cfg: JobConfig, remote: str, local: Path) -> list[str]:
    pre = [cfg.colab_bin] + (["--auth", cfg.colab_auth] if cfg.colab_auth else [])
    return pre + ["download", "-s", cfg.session_name, remote, str(local)]
