# Agent prompt: run the opencv-expert training with Claude Code on your computer

## How to use it

1. **On Windows:** install WSL2 + Ubuntu, install Claude Code *inside* WSL2, and work in the WSL
   filesystem (`~/`, not `/mnt/c`). The sandbox and the Colab CLI need Linux. Ollama installed
   inside WSL2 uses your NVIDIA GPU through WSL's CUDA support.
2. **Start Claude Code** in an empty folder (or an existing clone of the repo) and paste this
   one-liner:

   > Clone https://github.com/phil-melton/Hands-On-ML (skip if already cloned), check out branch
   > `claude/opencv-specialist-plan`, then read `opencv-specialist/AGENT_PROMPT.md` and follow the
   > prompt in its "The prompt" section, starting with Stage 0.

   Or paste the whole prompt below. Optionally press Shift+Tab into plan mode first, so it proposes
   Stage 0 before touching anything.
3. **Expect to be pulled in at:**
   - the Colab browser login
   - Gate 2: the task-family design review
   - Gate 3: go/no-go, and freezing the success criteria
   - Gate 4a: the teacher plan and spend
   - the final report
4. **It's multi-session.** In a new session, say: *"Read opencv-specialist/CLAUDE.md and
   LOGBOOK.md, then continue from 'Next step'."*

## The prompt

```text
You're the build engineer for `opencv-expert`: a small open model fine-tuned to write OpenCV 5.0 (Python cv2) code, served locally by Ollama and exposed to Claude Code as an MCP tool. The design is done and partly built; your job is to execute the rest carefully, measure everything, and keep me in the loop at the gates below. This is a multi-session project.

## Where things are
- Repo: https://github.com/phil-melton/Hands-On-ML, branch `claude/opencv-specialist-plan`, folder `opencv-specialist/`. Create a working branch `opencv-expert-training` from it. Commit at the end of every stage and push.
- Read these first, in order. They are the spec:
  1. `opencv-specialist/PLAN.md`: §0 measured findings, §4 the three key design choices, §5 phases, §6 success criteria, §7 open questions, §8 untested templates
  2. `opencv-specialist/COMPUTE.md`: Colab budget, how to predict CU, runbook, troubleshooting
  3. `opencv-specialist/cvbench/families/README.md`: how a task family is built, and the next families to build
  4. `opencv-specialist/probes/results/diff_4.14_vs_5.0.md`: measured OpenCV 4.14 → 5.0 differences (the curriculum)
- Exists and is tested: version probes, `sandbox/` (lint gate + subprocess runner), `cvbench/` (task contract, the `contour_area` family, the four-gate grader, `demo_exec_vs_mms`), 33 tests.
- Doesn't exist yet: the other task families, any training data, and every GPU step. The templates in PLAN.md §8 have never been run.

## Hardware and accounts
- My machine has an NVIDIA GPU with ≤ 8 GB VRAM. It serves the finished model (Ollama, Q4) and does CPU work (verification, tests). It does not train.
- GPU work runs on Google Colab through my Google AI Pro plan (200 compute units per month). Drive it with the Colab CLI (`pip install google-colab-cli`; Linux/macOS only). The first use needs my Google login in a browser: ask me to complete it.
- A frontier-API teacher is off by default. Use one only if I give you a key, confirm its terms allow training on its outputs, and set a dollar cap.

## Rules
First, copy this whole Rules section into `opencv-specialist/CLAUDE.md` and commit it, so every future session follows it.
1. Predict before you spend. Before every GPU job, write the predicted tokens/s, minutes and CU into `opencv-specialist/LOGBOOK.md`, using COMPUTE.md's method. Afterwards, record the measured values and `colab usage`. A miss of more than 3× is a bug: diagnose it before continuing.
2. Budget: at most 60 CU in total without asking me. Kill any job that passes 2× its predicted CU. Run `colab stop` the moment a job ends; an idle GPU still burns units. API spend is $0 unless I approve a cap.
3. Verification integrity: never loosen a tolerance, delete or skip a test, or change a verifier to make results look better. If you believe a tolerance is wrong, show me calibration evidence and ask.
4. Generated code runs only through `sandbox.run_solution`. For bulk runs, run it inside Docker with `--network none` if Docker is available.
5. Data provenance: training targets come only from a teacher whose license permits training. The default is an Apache-2.0 open-weight coder you host on Colab. Don't write training targets yourself. Record the license of every model and dataset you use.
6. Measure, don't assume. Unsloth, TRL, vLLM, Ollama, the Colab CLI and the Qwen chat templates all move fast. When docs, templates and reality disagree, trust the measurement and log the discrepancy.
7. Train/serve consistency: in every training example, the user message is `TaskSpec.to_prompt()`, the system prompt is identical to the Modelfile's SYSTEM, and the chat template is the base model's own.
8. No weights or secrets in git. Model artifacts go to Drive, Hugging Face or local disk. Commit data files under 50 MB; for anything larger, commit a manifest with SHA-256 hashes.
9. Logbook: keep `LOGBOOK.md` current enough that a fresh session can resume from it. Put a "Next step" line at the top, then dated entries: what ran, predicted vs measured, and each decision with a one-line physical or mechanistic justification.
10. Learning: when a result bears on one of PLAN.md §7's questions, point that out and show me the evidence. Leave the answer to me.
11. Gates: at each ⛔ gate, stop and send me a short report: what ran, predicted vs measured, the decision needed, and your recommendation with its reasoning. Then wait.

## Stages

Stage 0: Workbench (local)
- Detect the OS. On Windows, work inside WSL2 Ubuntu, because the sandbox and the Colab CLI need POSIX. If WSL2 isn't set up, stop and tell me how to set it up.
- Create a venv and `pip install -r requirements-cv5.txt`. Then, from `opencv-specialist/`, run `pytest` (expect 33 passed) and `python -m cvbench.demo_exec_vs_mms` (expect exec-only to accept 7 of 8 wrong solutions). Any difference means your environment differs from the reference: stop and report.
- Record the GPU (`nvidia-smi`), free disk space (about 40 GB needed) and RAM.
- Install the Colab CLI and authenticate. Confirm that `colab usage` shows the allowance. Read the CLI repo's `skills/colab-operator/`; if it's sound, install it under `~/.claude/skills/`.
- Check `ollama --version`. If Ollama is missing, ask me to install it.
- ⛔ Gate 0: environment report.

Stage 1: Hydrotest (Colab, at most 6 CU). This tests the plumbing, not the model's quality.
- Create `train/train_lora.py` from PLAN.md §8.1, and a 64-example format-test dataset built from `contour_area` (user message `SPEC.to_prompt()`, assistant message `REFERENCE`).
- Base candidates: Qwen2.5-Coder-7B-Instruct (the default), Qwen3.5-4B and Qwen3.5-9B. Verify the exact repo IDs, the licenses, and Unsloth support.
- For each candidate, on an L4: 30 training steps → merge → GGUF Q4_K_M → download → `ollama create` → generate one `contour_area` solution → grade it with `cvbench.evaluate`.
- Check each base's chat template, and set `train_on_responses_only`'s instruction and response markers to match. The `<|im_start|>` markers in §8.1 are Qwen2.5's.
- Persistence: try mounting Drive. If the CLI session can't mount it, `colab download` the checkpoints periodically.
- Continue to Stage 2 if the default base passes. Report which candidates failed and why.

Stage 2: Benchmark (local, CPU)
- If I've already added task families beyond `contour_area`, use them. Otherwise, draft 7–9 from the ranked list in `cvbench/families/README.md`, each following its checklist:
  - write the tolerance's scaling law in the docstring before writing any code
  - calibrate the constant on gauge blocks, with the measured numbers in a comment
  - define the task's symmetries
  - write 5–8 mutants, preferring the 4.x idioms from the diff report
  - pin the gate-matrix test
- Add a lint rule, or a separate gate, that flags APIs the probe measured as absent in 5.0.
- Build the gotcha suite, about 40 executable checks, from the diff report's behaviour changes and timeless gotchas.
- Splits: about 100 dev and 200 test tasks. Hold out 2–3 whole families from training.
- ⛔ Gate 2 (design review): for each family, show me the tolerance law, the calibration numbers and the gate matrix, and wait for my OK. This is the part I want to understand.

Stage 3: Baseline bake-off (Colab vLLM at about 3–5 CU, or local Ollama overnight)
- Rungs:
  - each hydrotested base at bf16
  - the default 7B with a system prompt of at most 1.5k tokens of "OpenCV 5.0 notes", distilled from the diff report
  - if I give you API access, the orchestrator model as the competitor
- Metrics:
  - pass@1 at T=0 and unbiased pass@5 at T=0.8 (Chen et al. 2021)
  - an error taxonomy: hallucinated or removed API, convention, algorithm
  - the 5.0 subset reported separately
  - paired comparisons (McNemar) with standard errors
- ⛔ Gate 3 (go/no-go): recommend whether to fine-tune and on which base, citing the lever table in PLAN.md §5. Then ask me to confirm the success criteria in PLAN.md §6 and record them in LOGBOOK.md. They are frozen after this.

Stage 4: Data foundry
- Pilot: 200 specs per teacher, each teacher given the 5.0 notes in context.
  - Default teacher: an Apache-2.0 Qwen coder served by vLLM on Colab, for example Qwen3-Coder-30B-A3B-Instruct at 4-bit on an L4. Verify that it fits and check its license.
  - Metric: cost per Tier-A example, per family. A cheap teacher with near-zero yield on the hardest families isn't cheap.
- ⛔ Gate 4a: the yield table, your recommended teacher plan, and the predicted CU and $ for full generation.
- Full loop (PLAN.md §5 Phase 1): k = 4 samples → lint → sandbox → Tier A/B gates → keep at most 2 passes per spec → failures plus their messages → teacher repair → repair pairs.
- Spend GPU time only on generation. Download batches and verify locally on CPU; never let a billed GPU idle while a CPU verifies.
- Data mix:
  - about 70 % verified solutions, 15–20 % repairs, and about 10 % general-code replay with a permissive license
  - AST-normalized dedup; decontaminate against cvbench; exclude the held-out families
- Output: write `data/train.jsonl` and `data/dev.jsonl`. Record the token-length p99, which sets `MAX_LEN`. Report the dataset stats.

Stage 5: Train (Colab L4, about 7–10 CU)
- 16-bit LoRA on an L4 or A100; QLoRA only on a T4. Hyperparameters per the table in PLAN.md §5 Phase 2: r = 16, α = 16, all 7 projections, lr 2e-4 with cosine decay, at most 2 epochs, loss on responses only.
- Log the step-0 held-out loss before training. Then run a convergence study on 25 %, 50 % and 100 % subsets.
- Select checkpoints by dev pass@1 through the sandbox, not by loss. Report the train–val gap.
- Report: the curves, the chosen checkpoint, and predicted vs measured CU per epoch.

Stage 6: Package
- Export the adapter, a merged 16-bit model, and GGUF Q8_0, Q5_K_M and Q4_K_M.
- Evaluate each, paired, on the test set and the 5.0 subset. Run the bf16 and Q8 rungs on Colab; a 7B Q8 doesn't fit in 8 GB. Ship the smallest that stays within noise, and test PLAN.md's prediction that the 5.0-specific behaviours degrade first.
- Write the Modelfile from PLAN.md §8.3, using the base's real chat template (Unsloth's generated Modelfile, or `ollama show <base> --modelfile`). Then `ollama create opencv-expert -f Modelfile`.
- Final evaluation: the shipped GGUF against the same base with the same prompt (paired), plus a forgetting check on a slice of a general Python benchmark.

Stage 7: Integrate
- Implement `serve/server.py` from PLAN.md §8.4. Add runtime metamorphic checks: rerun the code on rot90, mirrored and dimmed copies of the image, and report whether the outputs stay invariant.
- Register it with `claude mcp add`, then smoke-test 5 multi-step tasks through the tool.
- ⛔ Final report: PLAN.md §6's success criteria, target vs measured; total CU and $ spent; what surprised you; and what you would change next.

Begin with Stage 0.
```
