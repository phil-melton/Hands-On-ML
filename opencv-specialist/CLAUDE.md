# opencv-expert: rules for every session

Copied verbatim from `AGENT_PROMPT.md` ("Rules"). Resume a session with: read this file and
`LOGBOOK.md`, then continue from the "Next step" line at the top of the logbook.

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

## Where things are

- Spec, in reading order: `PLAN.md`, `COMPUTE.md`, `cvbench/families/README.md`, `probes/results/diff_4.14_vs_5.0.md`.
- Working branch: `opencv-expert-training` (from `claude/opencv-specialist-plan`). Commit at the end of every stage and push.
- Stages and gates: `AGENT_PROMPT.md` ("Stages").
