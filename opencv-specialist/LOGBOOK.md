# opencv-expert logbook

**Next step:** Gate 0 report sent 2026-10-05; waiting for Phil on (a) pushing `opencv-expert-training` (needs a
GitHub sign-in once) and (b) WSL mirrored networking so WSL Python can reach the Windows Ollama API. After his OK:
Stage 1 hydrotest, local (Path L): create `train/train_lora.py` from PLAN.md §8.1 and a 64-example `contour_area`
format set, verify base repo IDs/licenses/chat templates, then run `QLORA=1 MAX_STEPS=30` for the 7B inside tmux
session `train` (predicted ~107 tok/s, ~52 min; see 2026-10-05 entry).

Working copy of record: the WSL clone `~/Hands-On-ML` (Ubuntu-24.04). The Windows clone at
`C:\Users\ptmel\Documents\GitHub\Hands-On-ML` only syncs through GitHub.

## 2026-10-05 Stage 0 complete inside WSL2 (Gate 0)

What ran
- Phil installed WSL 3.0.1 + Ubuntu 24.04.5 (kernel 6.18.40.1), distro vhdx at `D:\WSL\Ubuntu-24.04\ext4.vhdx`, `~` has
  946 GB free. Phil installed `python3-venv python3-dev build-essential cmake libcurl4-openssl-dev` (gcc 13.3, cmake 3.28).
- `C:\Users\ptmel\.wslconfig`: `memory=24GB`, `swap=16GB` (default was 15 GB). Reason: merging a 7.6B LoRA into 16-bit
  weights for GGUF export handles ~15 GB of tensors; 15 GB of RAM left no margin.
- Cloned the branch into WSL (`~/Hands-On-ML`) from a git bundle of the Windows clone (Linux git refuses the `/mnt/c`
  repo as "dubious ownership"); `origin` points to GitHub.
- Reference env `~/venvs/cv5` (requirements-cv5.txt): `pytest` 33 passed (6.6 s); `demo_exec_vs_mms`: exec-only accepts
  7 of 8 wrong solutions, lint + manufactured + metamorphic reject 8 of 8. Both match the reference exactly.
- Training env `~/venvs/train` via `setup/train_env.sh` (log and `pip freeze` in `~/opencv-expert/logs/`):
  torch 2.12.1+cu130, unsloth 2026.9.14, unsloth_zoo 2026.9.9, transformers 5.5.0, trl 0.24.0, peft 0.21.2,
  bitsandbytes 0.50.2, xformers 0.0.35, triton 3.7.1, accelerate 1.15.0, datasets 4.3.0, opencv 5.0.0, numpy 2.4.6.
  `pytest` in this venv: 33 passed, so one venv can both train and grade.
- GPU inside WSL: RTX 3060, bf16 supported, 10.93 / 12.00 GiB free (desktop holds the rest).

Predicted vs measured
- fp16 4096^2 matmul: predicted 20-27 TFLOP/s (spec 25.6 dense, fp32 accumulate); measured 24.5 TFLOP/s.
  Consequence: Path L's peak assumption holds; at 20 % utilization the 7.6B QLoRA prediction is 24.5e12 x 0.2 / (6 x 7.6e9)
  = ~107 tok/s, so the hydrotest (30 steps x 16 x ~700 tok = 336k tok) predicts ~52 min.

Findings and open items
- Ollama 0.32.9 runs on Windows (model present: gemma4:latest, 9.6 GB). WSL uses NAT networking, so the Windows API at
  127.0.0.1:11434 is unreachable from WSL (both localhost and the host IP time out). `ollama.exe` via interop works
  (the client runs on the Windows side). Recommendation: `networkingMode=mirrored` in `.wslconfig` rather than a second
  Ollama inside WSL, because two servers would compete for the same 12 GB of VRAM.
- Before every GPU job: `ollama ps` must show no loaded model; a loaded model would take VRAM from training.
- Version watch (Rule 6): trl 0.24.0 is old relative to transformers 5.5.0; it is what unsloth 2026.9.14 resolved.
  Check `train_on_responses_only` and SFTConfig argument names against this trl at the hydrotest.
- Docker is not installed; Rule 4 bulk runs fall back to the sandbox alone until Stage 4.
- Monitoring: long jobs run detached in tmux 3.4 (`tmux attach -t train`), log to `~/opencv-expert/logs/`, with
  TensorBoard for curves.

Decisions
- The distro's vhdx lives on D: and all work happens in `~` (ext4). Mechanism: `~` is physically on D: but avoids
  9P overhead on `/mnt/d` for venvs, pytest and dataloaders. Only GGUF hand-off to the Windows Ollama uses `/mnt/d`.

## 2026-10-04 Decision: train locally on the RTX 3060, not on Colab

Phil chose the local path. Mechanistic basis: the card has 12 GB, not the <= 8 GB AGENT_PROMPT assumed, so
QLoRA on a 7B (about 6.5 GB weights + activations at batch 1, grad-accum 16, MAX_LEN 2048) fits, and a 4B fits
with headroom. The cost is speed: predicted with COMPUTE.md's formula at 25.6 TFLOP/s dense fp16 and 20 %
utilization (QLoRA dequant overhead on a 360 GB/s card): about 110 tok/s for a 7.6B, 210 tok/s for a 4B.

| stage | local prediction | notes |
|---|---|---|
| hydrotest, 30 steps x eff. batch 16 x ~700 tok (7B QLoRA) | ~50 min | use MAX_STEPS=30; proves LoRA -> merge -> GGUF -> Ollama |
| bake-off, ~800 generations per rung on Ollama Q4 | ~2 h per rung, 3-5 rungs overnight | decode ~50 tok/s for a 4.7 GB Q4 on 360 GB/s |
| teacher pilot, 200 specs x k=4 | ~4 h with a 14B coder at 4-bit (~9 GB) | 30B-A3B does not fit VRAM; try CPU offload and measure |
| full teacher generation, ~3k specs x k=4 | ~60 h at ~30 tok/s | the one stage where Colab or an API clearly pays; decide at Gate 4a |
| fine-tune 7B, 2.1 M tok/epoch | ~5 h per epoch, 2 epochs overnight | 4B: ~2.7 h per epoch |
| quant ladder evals | Q4/Q5/Q8 all fit locally | bf16 rung (15 GB) needs CPU offload or Colab |

Rules unchanged except Rule 2: the 60 CU cap is now 0 CU unless Phil re-authorizes Colab for a named stage.
Rule 1 still applies: predict tok/s and hours before every local GPU job and record the measurement.
Open question for PLAN.md §7.6: if the measured tok/s is more than 3x below ~110, suspect fp32 fallback,
no packing, or the dataloader before blaming the card.

Housekeeping: the first push of `opencv-expert-training` waited on a Git Credential Manager sign-in that was
never completed; the branch exists only locally until Phil runs `git push -u origin opencv-expert-training` once.

## 2026-10-04 Stage 0, environment survey (Windows side)

What ran
- Cloned `phil-melton/Hands-On-ML`, checked out `claude/opencv-specialist-plan` (commits 728c5a1 2026-10-03, f562f7b 2026-10-04),
  created working branch `opencv-expert-training`, copied the Rules into `CLAUDE.md`.
- Informational only, on Windows (not the reference environment): venv at `D:\opencv-expert\.venv-cv5` with
  `opencv-python-headless 5.0.0.93`, `numpy 2.4.6`, `pytest 9.1.1`. `pytest`: 12 passed, 20 failed, 1 error.
  `demo_exec_vs_mms`: every gate reports "killed: child exited with code 1 and no result".
  Mechanism: the sandbox runner relies on POSIX rlimits and process groups, so the child never starts on Windows;
  the 12 passes are the lint and contract tests that never spawn a child. This confirms the README: the kit needs WSL2.

Measured hardware (differs from AGENT_PROMPT's assumptions)
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM (the prompt assumed <= 8 GB). Driver 595.95, CUDA 13.2 runtime visible.
  Consequence: a 7B Q4 (~4.7 GB) serves with room to spare; a 7B Q8 (~8 GB) likely fits too, so the Q8 rung of the
  quantization ladder may be evaluable locally instead of on Colab (verify: Ollama with num_ctx 8192 needs KV cache on top).
- CPU: i7-12700 (20 threads). RAM: 31.7 GB.
- Disk: C: 16.8 GB free (below the ~40 GB the prompt asks for), D: 931 GB free, G: 16 GB free.
  Decision: put the WSL distro, venvs, datasets and GGUFs on D:.
- Ollama 0.32.9 is installed on Windows. Docker: not installed (Rule 4 bulk-run isolation falls back to the sandbox alone
  unless Docker Desktop or Docker inside WSL is added).
- WSL: not installed (`wsl --status`: "The Windows Subsystem for Linux is not installed").
- git: credential helper = manager (Git for Windows), user phil-melton; remote reachable.

Predicted vs measured: no GPU spend yet (0 CU).

Decisions
- Work in WSL2, as the prompt requires; do not port the sandbox to Windows (it would change the thing under test).
- Keep the D: drive for everything large; WSL2 distro install target must be D: as well (`wsl --import` or move the vhdx).
