# opencv-expert logbook

**Next step:** Stage 0 is blocked on WSL2. Phil installs WSL2 + Ubuntu (see 2026-10-04 entry), then a new
session reads `CLAUDE.md` and this file, repeats the Stage 0 checks inside WSL (venv, `pytest` = 33 passed,
`demo_exec_vs_mms` = exec-only accepts 7 of 8), installs and authenticates the Colab CLI, and sends the Gate 0 report.

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
