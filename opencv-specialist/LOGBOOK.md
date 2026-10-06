# opencv-expert logbook

**Next step:** Stage 1 is complete (2026-10-06 summary below). Push `opencv-expert-training` (needs Phil's GitHub
sign-in), then Stage 2 (benchmark, CPU): draft 7-9 task families from `cvbench/families/README.md` with their tolerance
laws, calibration and gate matrices, the absent-in-5.0 lint rule, the ~40-check gotcha suite and the dev/test splits,
then stop at Gate 2 for Phil's design review. For any GPU job: run `setup/commit_watchdog.ps1`, keep Ollama idle while
training, and discard the first Ollama request after a model is created before timing decode.

Working copy of record: the WSL clone `~/Hands-On-ML` (Ubuntu-24.04). The Windows clone at
`C:\Users\ptmel\Documents\GitHub\Hands-On-ML` only syncs through GitHub.

## 2026-10-06 Stage 1 complete: summary

Plumbing (QLoRA -> adapter -> merge -> GGUF Q4_K_M -> Ollama -> generate -> four-gate grade) works locally on the
RTX 3060. The default base passes, so per AGENT_PROMPT Stage 1 continues to Stage 2 (no gate).

| base | train tok/s (pred orig / revised) | peak VRAM | export | Q4_K_M | decode, warm | hydrotest |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B (default) | 357 (107 / 369) | 7.7 GiB | 17.1 min (from adapter, incl. 14.2 GB download) | 4.36 GiB | 64.6-65.3 tok/s (pred 50-60) | PASS: 128-token REFERENCE, 4/4 gates |
| Qwen3.5-4B | 472 (174 / 600) | 6.46 GiB | 2.3 + 8.0 min (+ llama.cpp build) | 2.59 GiB + 0.63 GiB mmproj | 78.6-84.6 tok/s | PASS non-thinking: 133 tokens, 4/4 gates |
| Qwen3.5-9B | - | - | - | - | - | dropped from local training (Phil, 2026-10-06): 7.38 GiB resident in 4-bit, ~10-10.5 GiB peak predicted vs 10.9 GiB free, 19.3 GB through WSL. Bake-off rung only |

- 7B grade at 07:31: first request decoded at 17.1 tok/s, a > 3x miss. Diagnosis: cold start of a just-created model.
  All 29/29 layers were on the GPU (11.25 GiB free at load; Ollama server.log), and two back-to-back runs right after
  decoded at 64.6 and 65.3 tok/s at full clocks (P0, 1950 MHz). Rule for the bake-off: discard the first request after
  a model is loaded before timing.
- Watchdog minimum during the export: 10.9 GB free commit (RAM free fell to 3.2 GB; the D: page file absorbed it).
- Unsloth's Modelfile for the 7B (kept as `Modelfile.unsloth`) carries the base's full Qwen2.5 Ollama template (tools,
  FIM, no `<|im_end|>` after the last assistant turn) and Qwen's default SYSTEM. For a single system + user turn it
  renders the same text as `train.config`'s TEMPLATE (the grade matched byte for byte). For Stage 6, prefer the base's
  own full template with our SYSTEM (Rule 7), and re-check it against `render_turn` for multi-turn repair dialogs.

Findings to carry forward
1. Ollama 0.32.9 thinks by default for qwen35 GGUFs; send `think: false` (`hydro_eval` does; PLAN.md §8.4 must).
2. Output format: §8.1 trains bare code, §8.4 serves JSON `{"code": ...}`. Pick one before Stage 4 (Rule 7).
3. Speed: QLoRA runs at ~69 % of the measured fp16 peak, not 20 %: a Stage 5 epoch (2.1 M tokens) on the 7B is ~1.6 h.
   Evidence for PLAN.md §7 question 6, left to Phil.
4. `save_pretrained_gguf` merges on its own; skip the separate 16-bit merge unless the merged weights are needed.
5. transformers 5.5 ignores `logging_dir`; `train_lora` sets `TENSORBOARD_LOGGING_DIR`.
6. Windows commit (~26-27 GB at idle) is the binding local constraint, not VRAM. WSL is capped at 14 GB, the D: page
   file raises the limit, and every GPU job runs under `setup/commit_watchdog.ps1`.
7. Never move or rename files a live job writes; never run Ollama while training; chains stop when a step is killed.

## 2026-10-05 Stage 1 hydrotest: setup and predictions (Phil OK'd Gate 0: mirrored networking, push, Stage 1)

Setup
- `.wslconfig` gained `networkingMode=mirrored`; WSL reaches the Windows Ollama at 127.0.0.1:11434 (version 0.32.9).
- New `train/`: `config.py` (one source for SYSTEM, chat-template rendering, Ollama TEMPLATE/PARAMETERs),
  `make_format_set.py` (64 train + 4 dev examples: 4 phrasings of `contour_area`, assistant = REFERENCE; plumbing
  only, never for real training, Rule 5), `train_lora.py` (PLAN.md §8.1 adapted: QLoRA, batch 1 x GA 16, asserts the
  base template equals `render_turn`, checks the loss mask, logs step-0 dev loss, tokens/s, peak VRAM, TensorBoard),
  `ollama_create.py` (uploads the GGUF over the Ollama API, writes the Modelfile), `hydro_eval.py` (generates one
  solution through Ollama, grades it with the four gates in the sandbox), `hydrotest.sh` (all three bases in sequence).
- tensorboard 2.21.0 added to `~/venvs/train`.

Base candidates (verified on the HF Hub, 2026-10-05)
| base | repo used | license | arch | weights |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (default) | unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit | apache-2.0 | Qwen2ForCausalLM | 5.5 GB NF4 (15.2 GB bf16 upstream) |
| Qwen3.5-4B | unsloth/Qwen3.5-4B (no pre-quantized copy exists) | apache-2.0 | Qwen3_5ForConditionalGeneration (VLM) | 9.3 GB bf16 |
| Qwen3.5-9B | unsloth/Qwen3.5-9B (no pre-quantized copy exists) | apache-2.0 | Qwen3_5ForConditionalGeneration (VLM) | 19.3 GB bf16 |

Chat templates (measured with each upstream tokenizer)
- Qwen2.5: ChatML; matches PLAN.md §8.3 exactly. Answer prefix `<|im_start|>assistant\n`.
- Qwen3.5: ChatML plus an empty `<think>\n\n</think>\n\n` before every non-thinking answer, and it trims the answer
  (the REFERENCE's trailing newline is dropped before `<|im_end|>`). The first check failed on exactly that newline;
  `render_turn` now trims for Qwen3.5 and all three bases render identically to their own templates (64/64).
  Consequence: for Qwen3.5 the `train_on_responses_only` response marker and the Ollama TEMPLATE both end with the
  empty think block, so serving matches training in non-thinking mode.
- Format finding to raise before Stage 4 (Rule 7): PLAN.md §8.4's server asks Ollama for JSON (`format: {"code": ...}`)
  but §8.1's targets are bare code. Training and serving must use one output format; decide before data generation.

Tokens: 64 examples, mean 357 tokens (Qwen2.5) / 370 (Qwen3.5), max 372; answer 127 / 133 tokens. MAX_LEN 2048 is
ample. 30 steps x 16 = 480 sequences = 7.5 passes over the 64 examples (fine for plumbing; expect train loss -> ~0).

Predictions (Rule 1), COMPUTE.md method with the measured 24.5 TFLOP/s and 20 % utilization: tok/s = 4.9e12 / (6N)
| base | N (B) | tok/s | train tokens | train min | peak VRAM | download | merge + GGUF |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder-7b | 7.6 | ~107 | 171k | ~27 | ~7-8 GiB | 5.5 + 15.2 GB (merge source) | ~15 min |
| qwen3.5-4b | 4.7 | ~174 | 178k | ~17 | ~4-5 GiB | 9.3 GB | ~10 min |
| qwen3.5-9b | 9.65 | ~85 | 178k | ~35 | ~8-9 GiB | 19.3 GB | ~20 min |
Plus a one-time llama.cpp build (~5-10 min). Downloads at an assumed 20-50 MB/s. Whole chain: ~3-3.5 h wall.
Risk to the tok/s prediction: 360-token sequences at batch 1 underfill the GPU's tiles, so a miss on the slow side
is likely; it only counts as a bug beyond 3x (< 36 tok/s for the 7B). N for Qwen3.5 includes the vision tower,
which a text-only forward skips, so those two predictions lean slow.
VRAM: 10.9 GiB free with the desktop running; `hydrotest.sh` refuses to start a base while Ollama holds a model.

7B, early measurement (steps 1-3, launched 21:03 in tmux `train`)
- Load 1.6 min (5.5 GB NF4 download at ~25 MB/s). Loss mask: 128 tokens = answer + end token. Step-0 dev loss 0.5947.
- Steady 369 tok/s (step 2 -> 3: 5,715 tokens in 15.49 s), 16.5 s/step: **3.4x faster than the ~107 predicted**.
- Rule 1 diagnosis (a >3x miss in either direction): tokens are real (5,715 per step = 16 x 357, batch 1, no padding);
  loss sane (0.595 -> 0.594 -> 0.579 during warmup); implied work 6 x 7.6e9 x 369 = 16.8 TFLOP/s = 69 % of the measured
  24.5 TFLOP/s peak, so physically possible (a counting bug would show > 100 %). The miss is the assumed 20 % utilization:
  NF4 dequantization is cheap next to the matmuls at 357-token micro-batches (MLP weight 3584 x 18944: ~2 ms of matmul
  vs ~0.5 ms to dequantize 136 MB at 360 GB/s), and Unsloth overlaps activation offload with compute.
  Bears on PLAN.md §7 question 6 (expected vs measured cost): evidence for Phil, answer left to him.
- Revised predictions for the jobs not yet started, scaled by 369 x 7.6 / N (same utilization assumed for the Qwen3.5
  path, though FastModel's VLM route may be less tuned): qwen3.5-4b ~600 tok/s, 178k tokens ~5 min;
  qwen3.5-9b ~290 tok/s, ~10 min. Full Stage 5 epoch (2.1 M tokens) on the 7B: ~1.6 h, not ~5.3 h.

7B attempt 1 killed at step 9/30 (operator error, 21:08)
- transformers 5.5 ignores `SFTConfig(logging_dir=...)` (deprecated; it reads env `TENSORBOARD_LOGGING_DIR`), so the
  events went to `./runs` inside the repo. I moved that folder to `~/opencv-expert/runs/...` mid-run, assuming the writer
  kept its file descriptor; it reopens the file by path on every flush, so the next flush raised FileNotFoundError and
  the run died. No checkpoint existed yet (SAVE_STEPS=15). Lesson: never move or rename files a live job writes.
- Fix: `train_lora.py` sets `TENSORBOARD_LOGGING_DIR=$OUT/tb` before building the trainer. The partial run's events and
  summary are kept in `~/opencv-expert/runs/hydro-qwen2.5-coder-7b-attempt1/`. Its 9 steps confirm the speed:
  2:25 for 9 steps, ~362 tok/s.
- The chain continued to qwen3.5-4b as designed. A 7B rerun is queued in tmux window `rerun-7b` (`train/after.sh`),
  starting when the main log prints "hydrotest finished"; the 5.5 GB NF4 weights are cached.

qwen3.5-4b: HYDROTEST PASS (21:08-21:30)
| item | predicted | measured |
|---|---|---|
| load (incl. 9.3 GB download) | 3-8 min | 2.3 min |
| train tok/s | ~174 original / ~600 revised | 472 (177,721 tokens in 376.5 s); within 3x of both |
| peak VRAM | 4-5 GiB | 6.46 GiB reserved |
| merge 16-bit + GGUF Q4_K_M | ~10 min + llama.cpp build | 2.3 + 8.0 min (llama.cpp build included) |
| Q4_K_M size / decode | | 2.59 GiB (+ 0.63 GiB BF16 mmproj vision projector) / 76.5 tok/s |
- Loss mask 133 tokens; step-0 dev loss 0.3974 -> final 0.0001; train loss 0.0395. LoRA 32.5 M of 4.57 B params
  (language layers only). Memorized the 64 near-identical examples, as a plumbing run should.
- Generated code is byte-identical to REFERENCE; all four gates pass.
- Finding (Rule 6/7): Ollama 0.32.9 tags the qwen35 GGUF with capabilities [tools, thinking, completion] and THINKS BY
  DEFAULT in /api/chat, whatever our TEMPLATE says: the first eval generated 526 tokens (~390 of hidden thinking, then the
  code). The same model given our exact training prompt via raw /api/generate emits the 133-token answer and stops.
  With `think: false` /api/chat also returns 133 tokens, identical to raw: so non-thinking serving matches training.
  `hydro_eval.py` now sends `think: false` whenever the model reports the thinking capability; the thinking-mode
  result is kept as `hydro_eval_thinking_default.json`. PLAN.md §8.4's MCP server must also send `think: false`.
- Finding: `save_pretrained_gguf` writes its own merged 16-bit copy (8.7 GB in `gguf/`) and the GGUFs into `gguf_gguf/`,
  so the separate `save_pretrained_merged` call duplicates ~2 min and 8.7 GB per run. Keep one in Stage 6.
- Fixed `ollama_create.py`: it wrote our Modelfile before looking for Unsloth's, so an Unsloth Modelfile in the same
  directory would have been overwritten unseen. It now keeps Unsloth's as `Modelfile.unsloth` and compares TEMPLATEs.

qwen3.5-9b attempt 1: CUDA OOM while loading (21:30-21:34)
- Download (19.3 GB) completed into the shared `~/.cache/huggingface/hub/blobs` store (hub cache now 32 GB).
- `transformers.modeling_utils.caching_allocator_warmup` asked for one 7.38 GiB fp16 block and CUDA refused it with
  10.92 GiB reported free. Mechanism (hypothesis): the WSL2/WDDM paravirtualized driver cannot hand out a single block of
  that size even when the total is free; the 4B's smaller warm-up block succeeded.
- 7.38 GiB is also the measured resident size of the 4-bit 9B (NF4 language layers plus bf16 embeddings and vision
  tower), so the peak-VRAM prediction rises from ~8-9 to ~10-10.5 GiB (4B: 6.46 GiB peak = weights + ~2.6 GiB).
  That is within ~0.5 GiB of the 10.9 GiB free: training may OOM even if loading succeeds.
- Retry queued in tmux window `retry-9b`, after the 7B rerun: same run with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (lets the allocator map the block from smaller pieces).
  Prediction: loads; trains at ~290 tok/s (~10 min) if the peak stays under 10.9 GiB, else OOM at step 1.
  If it fails again, report the 9B as not trainable on 12 GB with this stack; it stays a bake-off (inference) candidate.

21:35-21:37: 7B rerun and 9B retry both failed with CUDA OOM; idle until 22:53
- 7B rerun: OOM in the step-0 dev evaluation asking for 220 MiB, with 5.17 GiB reported free (5.73 GiB held by PyTorch).
- 9B retry (expandable_segments): the same 7.38 GiB warm-up refusal with 10.92 GiB reported free.
- My log watcher had a bug (it captured "0" twice and its failure test errored every loop), so neither failure was seen
  until it timed out at 22:52: ~75 min of idle GPU. Fixed.
- Allocation probe at 22:53 on the idle GPU (`torch.empty` + fill): largest single block 10 GiB OK; 26.25 GiB total in
  256 MiB chunks (WDDM pages CUDA allocations into shared system memory beyond the 12 GiB of VRAM, so a "fit" can
  silently become slow rather than fail). So neither the 7.38 GiB block nor the 7B's working set exceeds what the
  card can give when nothing else holds it.
- Hypothesis: interference from my own Ollama calls on the same GPU. Between 21:33 and 21:35 I loaded the 4B in the
  Windows Ollama three times to investigate the thinking finding (one with the default 5-minute keep-alive). The
  failures cluster in exactly that window, and `hydrotest.sh` only checks `/api/ps` at the start of each base; Ollama
  also returns from a `keep_alive: 0` request before its runner has released VRAM. Test: rerun with no Ollama activity.
- Rule for operators (me): no Ollama or other GPU work while a training job runs.
- Relaunched 22:55: 7B then 9B in the tmux `job` window (`bash train/hydrotest.sh qwen2.5-coder-7b qwen3.5-9b`), with
  the predictions above (7B ~369 tok/s, ~8 min of training; 9B ~290 tok/s, ~10 min, peak ~10-10.5 GiB, which may spill
  into shared memory and run slow instead of failing).

qwen2.5-coder-7b run 2: training complete, export killed by a Windows out-of-memory event (22:55-23:06)
- With no Ollama activity the 7B passed the step that OOM'd at 21:36 (supports the interference hypothesis). Its first
  three losses (0.5953, 0.5942, 0.5786) equal attempt 1's exactly: the seeded run reproduces.
| item | predicted | measured |
|---|---|---|
| train tok/s | ~107 original / ~369 revised | 356.9 (171,481 tokens in 480.4 s) |
| peak VRAM | ~7-8 GiB | 7.7 GiB reserved |
| loss | | step-0 dev 0.5947 -> final dev 0.0005; train 0.094; LoRA 40.4 M params |
- The adapter (161 MB) and summary.json were saved at 23:04. The export then began downloading the 14.2 GB bf16 base
  (the merge source) and at 23:06:15 Windows ran out of commit: 8x iaStorVD "driver failed to allocate memory",
  dwm.exe terminated "could not allocate additional memory" (23:06:30), and the WSL VM died (Hyper-V switch ports
  deleted 23:06:16). No sleep or hibernate event.
- Mechanism: commit limit 45.0 GB = 31.7 GB RAM + 13.3 GB system-managed page file on C: (C: has 18 GB free, so the
  page file cannot grow much). Windows apps already commit ~25-27 GB (measured 26.8 GB with WSL at 1.85 GB after
  restart). `.wslconfig memory=24GB` let the VM's Linux page cache (filled by the 14 GB download) grow until the sum
  passed 45 GB. My 24 GB setting was sized for the merge without checking Windows' own commit.
- Not relaunched: this affects the stability of Phil's desktop, so the fix needs his OK. Proposal: `memory=14GB` (26.8 +
  14 = 40.8 < 45) plus page-cache reclaim; close heavy apps during runs; optionally move/enlarge the page file on D:
  (a Windows setting, his to change). Prepared `train/export_gguf.py` to export the saved 7B adapter without retraining.

2026-10-06 06:24 (launched; log `hydro_run3_20261006_062423.log`; Windows commit free 17.6 GB at start, so the
14 GB VM leaves ~3.6 GB in the worst case) Phil chose: `memory=14GB` + `[experimental] autoMemoryReclaim=dropcache` (accepted without warnings by
WSL 3.0.1; the VM now shows 13 GiB), then export + grade the 7B, then one 9B attempt.
- Safeguards: a Windows-side commit watchdog (every 5 s; below 3 GB free commit it pkills `train.train_lora` /
  `train.export_gguf` in WSL and logs the minimum seen), and `hydrotest.sh` now waits up to 2 min for an idle GPU
  (empty `/api/ps` and < 2.5 GB used) before each base instead of only checking `/api/ps`.
- Predictions (Rule 1):
  - 7B export from adapter: 14.2 GB bf16 download at 25-50 MB/s (5-10 min) + merge and Q4_K_M (~10 min; llama.cpp
    already built) = ~15-20 min; WSL RAM pinned at its 14 GB cap by page cache; Windows commit free stays >= ~4 GB.
  - 7B grade: 128-token reply identical to REFERENCE, all four gates pass; decode ~50-60 tok/s (4.4 GB Q4 on 360 GB/s).
  - 9B: load from cache 3-5 min; train ~290 tok/s (~10 min) if the ~10-10.5 GiB peak fits in VRAM. If it spills into
    shared memory, tok/s drops several-fold and Windows commit falls; the watchdog stops it at 3 GB free.

06:24-06:27 result: the watchdog stopped the 7B export; I stopped the 9B
- Windows free commit: 17.6 GB at 06:24:13 -> 7.3 GB at 06:25:15 -> 2.31 GB at 06:25:46, when the watchdog killed
  `train.export_gguf` (it had loaded the adapter in 0.7 min and just begun the 14.2 GB bf16 download). Prediction
  (">= ~4 GB free") missed: 15.3 GB consumed = the 14 GB VM cap (page cache from the download fills it) + ~1.3 GB more,
  plausibly the WDDM-backed CUDA allocations of the 4-bit model.
- Two tooling faults of mine: the resume script went on to the 9B after the kill, and the watchdog had exited after
  firing, so the 9B was loading unguarded; I killed the chain by hand at 06:27 (no Windows errors this time).
  Fixed: the watchdog now kills the chain scripts before the Python jobs; `hydrotest.sh` stops the chain when a step
  exits 137/143 instead of starting the next base.
- Conclusion: with ~26-27 GB of Windows commit at idle and a 45 GB limit, this PC has ~18 GB for WSL, minus margin.
  Any step that streams a multi-GB download or checkpoint through WSL fills the VM to its cap. The 7B export (14.2 GB
  download + merge) and anything 9B do not fit safely as configured. Decision needed from Phil (page file on D:, a
  smaller VM cap with fewer apps open, or moving the export off this PC); 9B local training not recommended.

Decisions (Phil, 2026-10-06)
- Headroom: Phil adds a page file on D: (custom size, e.g. 16-48 GB). Mechanism: the commit limit is RAM + page files,
  so it rises from 45 GB to ~60-90 GB; when RAM fills, Windows pages to D: (slower) instead of failing allocations
  (the 23:06 crash). The watchdog stays on for every GPU job.
- Qwen3.5-9B: dropped from local training. It needs ~10-10.5 GiB of the 10.9 GiB free VRAM (7.38 GiB resident in 4-bit)
  and streams 19.3 GB through WSL. It stays an inference-only rung (its Q4 GGUF fits for serving) in the Stage 3
  bake-off; revisit training only if it wins there (then Colab, with Phil's OK).

07:10 7B export relaunched after the page file (log `hydro_run4_20261006_071025.log`)
- After Phil's restart: page files C: 2 GB (system managed) + D: 16 GB initial / 48 GB max; commit limit 49.7 GB and
  able to grow with the D: file; 29.7 GB free commit at launch (fewer apps open after the restart). WSL 13 GiB, GPU
  idle at 908 MiB, Ollama empty. Watchdog running (threshold 3 GB). Predictions unchanged from the 06:24 entry
  (export ~15-20 min; grade: the 128-token REFERENCE, four gates pass, ~50-60 tok/s decode).

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
