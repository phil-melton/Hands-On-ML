# opencv-expert logbook

**Next step:** Stage 0 is blocked on WSL2. Phil installs WSL2 + Ubuntu (see 2026-10-04 entry), then a new
session reads `CLAUDE.md` and this file, repeats the Stage 0 checks inside WSL (venv, `pytest` = 33 passed,
`demo_exec_vs_mms` = exec-only accepts 7 of 8), verifies the GPU is visible inside WSL (`nvidia-smi`), installs
Unsloth + bitsandbytes there, and sends the Gate 0 report. The Colab CLI is NOT installed by default any more
(decision below); it stays a fallback.

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
