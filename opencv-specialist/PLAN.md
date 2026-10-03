# Plan: `opencv-expert`, specializing an open model for OpenCV 5.0

**Goal:** a small open model that writes OpenCV 5.0 code as a *verified worker* for an orchestrator
(Claude Code via MCP). It runs locally, and every answer comes back with an inspection report.

**Status:**
- **Phase 0, the CPU half, is done and tested:** the version probe, the sandbox, the task contract,
  one worked task family, and the demo.
- **Remaining Phase 0 work is yours:** the bake-off, the hydrotest, and the other task families.

**Where to look:**
- GPU logistics: [COMPUTE.md](COMPUTE.md)
- How to run the kit: [README.md](README.md)

---

## 0. What Phase 0 has already measured

Everything below was measured in this repo on `opencv-python-headless` **4.14.0.94** vs
**5.0.0.93** (numpy 2.4.6). It is reproducible with `probes/api_diff.py`. The full table is
[probes/results/diff_4.14_vs_5.0.md](probes/results/diff_4.14_vs_5.0.md).

1. **The project has a reason to exist.** OpenCV 5.0 shipped in June 2026, and a bare
   `pip install` now pulls it. Of 213 probes, 40 changed behaviour:
   - *Removed or moved to contrib:* `CascadeClassifier` (and the Haar XMLs), `HOGDescriptor`,
     `cv2.ml`, G-API, `readNetFromCaffe`/`Darknet`.
   - *Missing from the 5.0 main wheel but not on the wiki's breaking-changes list:* **AKAZE, KAZE,
     BRISK, AGAST, `BOWKMeansTrainer`, the GOTURN tracker.**
   - *New:* `FontFace`, and the dtypes `CV_Bool`/`CV_32U`/`CV_64U`/`CV_64S`. bool masks, uint32
     and float16 arrays are now accepted where 4.14 raised.
   - *`VideoCapture.get()`* on an unsupported property returns −1 instead of 0.
2. **The "1D output" change is a set of point defects, not a rule.** That matters for training:
   a model that over-generalizes will break code that still works.
   - *Lost the middle axis:* `HoughLinesP` `(N,1,4)→(N,4)`, `findNonZero` `(N,1,2)→(N,2)`,
     `findChessboardCorners`/`SB` `(N,1,2)→(N,2)`, `convexityDefects`, `minEnclosingTriangle`,
     `HoughLinesWithAccumulator`, `calcHist` `(256,1)→(256,)`, ArUco `ids` `(N,1)→(N,)`.
   - *Did not change:* contours `(N,1,2)`, `goodFeaturesToTrack`, `approxPolyDP`, `convexHull`,
     `HoughLines`/`HoughCircles`, `fitLine`, `kmeans` labels, the LK flow status, `findHomography`
     masks.
   - *Version-proof idiom* (prefer it in training data): `.reshape(-1, k)`.
3. **Behaviour drift that never raises:**
   - **MSER returns 0 regions** with default parameters on a text image where 4.14 returns 7. The
     defaults are identical; `min_diversity=0` gives 170. This looks like a regression in 5.0.0's
     diversity pruning; I found no upstream issue yet.
   - `warpAffine`/`warpPerspective` changed by up to 4 gray levels on ~3 % of pixels (uint8).
   - Text rendering changed: `getTextSize` (155,22)→(136,27), and 13 % of glyph pixels differ.
   - The wiki says `INTER_NEAREST` "now follows Pillow". **Not observed**: the mapping is
     identical on all 450 size pairs swept. Measure; don't trust anyone's claim, mine included.
4. **Timeless gotchas** (same in both versions; they seed the gotcha suite):
   - int64 arrays are silently **narrowed to int32 with wraparound** (`cv2.add` on `2**40` gives `0`).
   - Drawing on `img[:, :, ::-1]` or `img.T` raises.
   - `getPerspectiveTransform` needs float32.
   - `perspectiveTransform` needs `(N,1,2)`.
   - `fillPoly` rejects float32 points but accepts int64.
   - `imread`/`imdecode` return `None` rather than raising.
5. **`exec()` is not a verifier: measured.** `python -m cvbench.demo_exec_vs_mms` grades the
   reference and 9 plausible mistakes. The `exec()` filter would have **accepted 7 of 8 wrong
   solutions**. The other three gates together rejected 8 of 8, and each gate caught something the
   others missed (§4.1).
6. **The tolerance law holds.** Over 3,500 single-shape gauge blocks, error/perimeter ranged over:
   - `contourArea`: [−0.589, +0.120]; the half-pixel inward shave on curved shapes, exactly 0
     where polygon vertices sit on pixel centres.
   - pixel counting: [−0.151, +0.598]; the half-pixel outward rim.
   - Worst case 0.598, so `TOL_C = 0.75`.
   - Under the task symmetries, the reference's areas moved by exactly **0.0** px².

---

## 1. Context and decisions

You want an OpenCV "master machinist" under an orchestrator (the "general contractor"). The
blueprint is synthetic data → QLoRA (Unsloth) → GGUF/Ollama → orchestrator tool. Every model in
the loop (base, teacher, orchestrator) learned from a mix of OpenCV 2/3/4 code, and almost none
of 5.0. That is Gorilla's regime: an API that moved after training. On 4.x a frontier
orchestrator is already a master machinist, and a specialist could win only on cost.

| Decision | Choice | Consequence |
|---|---|---|
| Target | OpenCV 5.0, pinned (`5.0.0.93`) | 4.14 is only the "before" side of the diff |
| Compute | Colab through Google AI Pro (200 CU/month) for all GPU work; your NVIDIA ≤ 8 GB card for serving | L4 with 16-bit LoRA (T4 fallback: QLoRA). This repo's CPU-only parts run anywhere |
| Teacher | Pilot two, pick by verified yield per family | Frontier API vs a self-hosted Apache-licensed Qwen coder on Colab |
| Integration | MCP server for Claude Code/Desktop | One `opencv_solve` tool: generate → lint → sandbox → check → repair |

## 2. Compute (summary; details in [COMPUTE.md](COMPUTE.md))

- **Budget.** Since 2026-09-22, Google AI Pro includes 200 Colab compute units per month (1 CU ≈ $0.10).
  The whole project needs about **50 CU** with a 2× margin. Rates (March 2026, check live):
  T4 ≈ 1.2 CU/h, **L4 ≈ 1.7**, A100-40 ≈ 5.4, A100-80 ≈ 7.5.
- **Most of the project is CPU or API work.** Only the hydrotest, training, the bake-off and the
  quantization ladder need a GPU.
- **Choose the GPU by CU per epoch, not by name.** Predicted for a 7B on 3k examples:
  T4 ≈ 2.0, L4 ≈ 1.1, A100 ≈ 1.35 CU/epoch.
- **Serving is a conveyor belt.** Decode is bandwidth-bound: tok/s ≈ memory bandwidth ÷ model
  bytes, roughly 40–75 tok/s for a 4.7 GB Q4 7B on an 8 GB card. Batching rides the same belt
  pass, so batch your evals.

## 3. Architecture: steelman, then fix

**Steelman.** A narrow model has a sharp prior over one API. It hallucinates fewer parameters, and
it runs free and private on local images.

**Where it breaks:**
1. **A machinist without a micrometer is guessing.** The worker is *model + sandbox + verifier +
   repair loop*, and it returns a measured part with an inspection report (think AS9102
   first-article inspection).
   - In control terms, fine-tuning is the **feedforward** term, re-identified for the new plant
     (OpenCV 5.0). The sandbox loop is **feedback**.
   - *Limit of the analogy:* feedback only corrects what the sensor measures. Verifier coverage
     is sensor coverage.
2. **The contractor has to hand over a toleranced drawing.** That's the typed contract in
   [`cvbench/contract.py`](cvbench/contract.py). Train on exactly the format you serve with, so the
   part is fixtured the same way in qualification as in production.
3. **"Deterministic" is the wrong word.** Temperature 0 makes the model repeatable, not correct;
   correctness comes from the verifier.

## 4. Three blueprint choices to change

### 4.1 `exec()` pass/fail → manufactured-solution verifiers *(now measured)*

- **Steelman:** it catches Gorilla's failure class (`AttributeError`, `TypeError`, `cv2.error`)
  cheaply and at scale.
- **Break:** it is a one-sided test. The demo's matrix shows each gate catching a different kind
  of defect:

  | candidate | exec-only | lint | manufactured | metamorphic |
  |---|---|---|---|---|
  | reference | PASS | PASS | PASS | PASS |
  | `rgb2gray` (harmless here) | PASS | PASS | PASS | PASS |
  | `retr_tree` (holes counted as shapes) | PASS | PASS | **FAIL** | PASS |
  | `points_as_area` | PASS | PASS | **FAIL** | PASS |
  | `inverted_threshold` | PASS | PASS | **FAIL** | PASS |
  | `fixed_threshold_127` | PASS | PASS | PASS | **FAIL** |
  | `component_pixel_areas` (holes subtracted) | PASS | PASS | **FAIL** | PASS |
  | `unsorted` | PASS | PASS | **FAIL** | PASS |
  | `opencv3_unpack` | **FAIL** | PASS | **FAIL** | **FAIL** |
  | `swallowed_opencv3_unpack` | PASS | **FAIL** | **FAIL** | PASS |

  Read the rows the way you'd read a sensor survey:
  - **The hard-coded threshold** passes every base scene. Only the *dimmed* symmetry finds it.
  - **The metamorphic gate passes the swallowed bug**, because a constant output is invariant under
    everything: a stuck gauge is perfectly repeatable.
  - **`rgb2gray` passes, correctly.** Channel order can't matter when every shape is bright in
    every channel. The verifier measures what the task depends on, not style.
- **A filter is a selection pressure.** Exec-only filtering keeps the `try/except` variant, so it
  would teach the student to swallow errors. A leak test that only checks "no drip in 5 s"
  selects for slow leakers.
- **Trust tiers** for the training data:
  - A: manufactured solution
  - B: metamorphic + contract
  - C: runs + contract only. Never used as a fine-tuning target.
- **Keep the scrap.** Failed attempt + error message + fix = a repair example. Under 5.0 these
  are mostly 4.x-isms, the most valuable rows you have.

### 4.2 "Training loss < 0.8" → step-0 baseline + held-out pass@1

- **Steelman:** falling loss means the format is being absorbed, and a threshold gives a stopping rule.
- **Break:**
  - A strong code model may *start* below 0.8.
  - The loss floor is the data's irreducible entropy; going below it means fitting the teacher's
    variable names.
  - The decisive tokens (`FaceDetectorYN` vs `CascadeClassifier`, `line` vs `line[0]`) are about
    1 % of tokens, so boilerplate dominates the average. A heat exchanger's mean temperature won't
    find the hot spot that cracks it.
  - Gudibande et al. 2024: imitation copies style while capability barely moves.
- **Fix:** log these, with the 5.0 subset reported separately, and **pick checkpoints by dev pass@1**:
  - step-0 held-out loss and Δloss
  - the train–val gap
  - dev pass@1 every half-epoch

### 4.3 "2–5k pairs" → coverage-first data, sized by an information budget

- **Capacity:** LoRA r = 16 on all 7 projections of Qwen2.5-7B (hidden 3584, MLP 18944, KV 512,
  28 layers) is 1.44 M params per layer, so **≈ 40 M trainable (0.53 %)**. At about 2 bits/param
  (Allen-Zhu & Li) that's ~80 Mbit.
- **Signal:** 3k examples × ~400 supervised tokens × ≲ 1 bit/token ≈ **≤ 1.2 Mbit**.
- **Capacity is about 60× the signal**, so the risk is memorization:
  - 1–2 epochs
  - r = 16 (8 is likely enough)
  - **coverage over count.** 5,000 thresholding rows teach nothing about `FaceDetectorYN`.
- *Limit:* 2 bits/param is a storage ceiling. It bounds the capacity; it doesn't predict it.
- **Size the data with a convergence study** (like mesh independence): train on 25/50/100 %
  subsets, and stop growing when a doubling moves dev pass@1 by less than the noise.

## 5. Phases

### Phase 0: instrument before you machine *(CPU half done)*

- [x] Two environments: 5.0.0.93 and 4.14.0.94 (headless), see `requirements-cv*.txt`.
- [x] Differential API probe, 213 probes: `probes/`. **The diff is the curriculum**, a
      correction table from running the same gauge blocks through both versions.
- [x] Sandbox: lint gate, subprocess, timeout + process-group kill, rlimits, GUI/network/write
      guards, sticky violations (`sandbox/`, 14 tests).
- [x] Task contract plus one worked family with calibrated tolerance and symmetries:
      `cvbench/` (19 tests). The demo pins the gate matrix above.
- [ ] **Gotcha suite** (~40 checks). Seed it from the diff's 40 behaviour changes and 31 timeless
      gotchas.
- [ ] **cvbench v0**: 8–10 families (candidates in
      [`cvbench/families/README.md`](cvbench/families/README.md)). Split ~100 dev / ~200 test
      tasks, plus 2–3 **held-out families** that are never trained on.
- [ ] **Baseline bake-off** on a Colab L4 (batched vLLM, ~1.5 h) or overnight on your GPU.
  - Candidates: Qwen2.5-Coder-7B-Instruct, Qwen3.5-4B, Qwen3.5-9B; plus base 7B *with the 5.0
    migration notes in its prompt*, and the orchestrator alone.
  - Measure pass@1 (T=0), unbiased pass@5 (T=0.8), and an error taxonomy (hallucinated API /
    convention / algorithm).
- [ ] **Toolchain hydrotest** per candidate: 30-step LoRA → merge → GGUF → `ollama create` →
      generate. Pressure-test the plumbing before pouring product through it, especially for
      Qwen3.5's hybrid Gated-DeltaNet architecture.

**Go/no-go.** If the prompting rung lands within noise of the orchestrator on the 5.0 subset,
fine-tuning only buys cost and latency; decide now whether that's worth it. The error class picks
the lever:

| error class | lever |
|---|---|
| removed API, wrong convention | prior → fine-tune |
| wrong algorithm | capacity → bigger model |
| anything measurable | feedback → repair loop |

**Eval noise.** At n = 200 and p ≈ 0.6 the SE is ≈ 3.5 pts, so an unpaired difference needs
~10 pts to clear 2σ. Compare **paired** (McNemar): a differential measurement cancels the
common-mode task difficulty.

### Phase 1: data foundry (weekends 2–3)

1. **Seeds:** `opencv/samples/python` and the 5.0 tutorials, used for *ideas* (OSS-Instruct).
   Run the seeds under 5.0 too; their failures are curriculum.
2. **About 25 families**, weighted by Phase-0 error rate × 5.0 exposure.
3. **Teacher pilot:** 200 specs × 2 teachers, with the 5.0 notes and the probe table in context.
   - (a) a frontier API, paid per token; check its terms on training with outputs.
   - (b) an Apache-licensed Qwen coder on Colab via vLLM (e.g. Qwen3-Coder-30B-A3B at 4-bit on
     an L4), paid in CUs.
   - Metric: **cost per Tier-A example, per family.** A sieve changes the gradation, not just the
     quantity. Expect to use (b) for bulk and (a) for the families where (b)'s yield is low.
4. **Generation loop:** spec → teacher (k = 4) → `sandbox.run_solution` (Docker
   `--network none` for the foundry) → Tier A/B gates → keep ≤ 2 passes per spec → failures +
   messages → teacher repair → repair pairs.
5. **Mix:** ~70 % verified solutions, 15–20 % repairs, ~10 % general-code replay (LoRA forgets
   less, not zero: Biderman et al.).
6. **Hygiene:** AST-normalized dedup; decontaminate against cvbench by n-gram and scene seed;
   split by family.

### Phase 2: train (weekend 4)

| Knob | Value | Physical justification |
|---|---|---|
| Base | Bake-off winner (default Qwen2.5-Coder-7B-Instruct) | Fine-tuning moves priors, not reasoning capacity: you can retune the ECU, not the displacement. If the 4B is within ~5 pts, ship the 4B; it serves ~2× faster |
| Precision | L4/A100: 16-bit LoRA. T4: QLoRA | QLoRA trades base precision for memory. A 7B in bf16 (15 GB) fits the L4's 22.5 GB, so there's nothing to trade for, and it removes the NF4 → k-quant grid mismatch |
| Seq len | p99 dataset tokens × 1.2 (likely 2048) | Measured, not habit |
| LoRA | r=16, α=16, dropout 0; q,k,v,o,gate,up,down | API recall is thought to live in the MLP key-value memories (Geva et al.), so attention-only LoRA under-reaches |
| Loss | `train_on_responses_only` | The gradient comes only from what you want to change |
| Opt | lr 2e-4, cosine, 5 % warmup, eff. batch 16, ≤ 2 epochs | Capacity ≫ signal |
| Compute | ≈ 6N FLOP/token with checkpointing → ~1e17 FLOP/epoch | L4 ≈ 40 min (≈ 1.1 CU)/epoch. If Unsloth's tokens/s is > 3× off, something is misconfigured |
| VRAM (7B) | L4 16-bit: 15.2 + 0.5 + 0.5 + 2–4 GB act ≈ 19–20 GB. T4 QLoRA ≈ 6.5 GB + act | Both fit. Your 8 GB card serves, it doesn't train |

If the base has a thinking mode, train and serve non-thinking; the repair loop is the reasoning budget.

### Phase 3: package (weekends 4–5)

- **Artifacts:** the adapter, a merged 16-bit model (the re-quantization source), and the GGUFs.
- **Quantization ladder:** bf16 → Q8_0 → Q5_K_M → Q4_K_M, evaluated paired; ship the smallest
  within noise. A 7B Q8 doesn't fit in 8 GB, so the upper rungs run on Colab.
- **Prediction to test:** the 5.0-specific behaviours break first. They're newly learned,
  low-margin decisions, and quantization noise flips the smallest logit gaps first, like a
  comparator chattering near its threshold. The cure is margin (targeted examples), not a fatter
  quant.
- **Evaluate the GGUF you ship, never the checkpoint.** Quantization is heat treatment, so inspect
  after treatment.
- **Use the right chat template:** Unsloth's generated Modelfile, or `TEMPLATE` from
  `ollama show qwen2.5-coder:7b --modelfile`. A mismatch is the #1 cause of "great in training,
  garbage in Ollama".

### Phase 4: MCP integration (weekend 5)

- **Tool:** `opencv_solve(task, image_path, outputs)` (template in §8.4). It runs generate → lint
  → sandbox → contract check, with ≤ 3 repair rounds that feed the error back. It saves output
  images, so Claude can *look* at the part.
- **Runtime metamorphic checks:** at serve time there is no manufactured answer, but the
  symmetries still exist. Rerun the code on rot90/mirror/dimmed copies of the user's image and
  report the invariance as a confidence signal.
- **Register:** `claude mcp add opencv-expert -- python /abs/path/opencv-specialist/serve/server.py`.
- **E2E eval:** 20 multi-step tasks (HuggingGPT-style DAGs), comparing the orchestrator alone vs
  orchestrator + specialist on success rate, orchestrator tokens, and wall time.

### Phase 5: optional

- **Expert iteration:** the student samples k solutions, the verifier keeps the passes, and you
  retrain. The sieve lets the student approach the teacher's pass@k, not just its pass@1.
- **Vision in the loop:** the Qwen3.5 small models are natively multimodal; the machinist could
  look at the workpiece.
- **A version-conditioned model** (4.14 + 5.0).
- **Report the MSER finding upstream**, with `probes/api_diff.py` as the minimal repro.

## 6. Success criteria (write your numbers down before Phase 2)

Defaults, measured paired on the **shipped Q4 GGUF**:

| criterion | default |
|---|---|
| specialization | ≥ +10 pts pass@1 over the same base + prompt on the cvbench test set; ≥ +25 pts on the 5.0 subset |
| generalization | held-out-family gain ≥ half the in-family gain |
| forgetting | ≤ 3-pt drop on a general Python slice (MBPP+); the before/after delta cancels shared contamination |
| integration | orchestrator + specialist ≥ orchestrator alone on the E2E set, with fewer orchestrator tokens |

## 7. Questions to push on

1. Which error class dominates your baseline, and so which lever does it call for?
2. The contour verifier passes at r = 50 px and fails at r = 5 px. Bug in the code, or in the
   tolerance? (Check `tolerance()`: why `c·P + c0` and not `k·A`?)
3. Train loss falls to 0.2 while dev pass@1 stays flat. What did the model learn?
4. Without `train_on_responses_only`, what would the 40 M parameters spend themselves on?
5. Which behaviours do you predict break first at Q4_K_M, and what measurement would falsify that?
6. Before the first Colab run, write down the CU per epoch you expect. If the measurement is
   > 2× off, is it padding, no packing, an fp32 fallback, or a starved dataloader?
7. The 1D change hit `findChessboardCorners` but not `goodFeaturesToTrack`, though both produce
   2-D points. What in the C++ implementations could make the difference? (Hint: look at how
   each one writes its `OutputArray`.) And what does that imply about whether a model can
   *infer* the rule rather than memorize it per function?
8. The metamorphic gate passed a function that always returns `{"areas": [], "count": 0}`. Which
   other gate would you trust to catch a stuck gauge, and why?

## 8. Templates (untested: no GPU or Ollama in the planning container)

APIs move fast; check argument names against the versions you install. Each template is
validated by the Phase-0 hydrotest, not by this document.

### 8.1 Training script (`train/train_lora.py`)

```python
# UNTESTED TEMPLATE. Run on a Colab L4 (QLORA=0) or T4 (QLORA=1).
import os
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

BASE = os.environ.get("BASE", "unsloth/Qwen2.5-Coder-7B-Instruct")
OUT = os.environ.get("OUT", "/content/drive/MyDrive/opencv-expert")
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))     # measured p99 tokens x 1.2
QLORA = os.environ.get("QLORA", "0") == "1"
CKPT = f"{OUT}/checkpoints"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=MAX_LEN, load_in_4bit=QLORA, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "dev": "data/dev.jsonl"})
ds = ds.map(lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
            batched=True)

trainer = SFTTrainer(
    model=model, processing_class=tokenizer,          # older TRL: tokenizer=tokenizer
    train_dataset=ds["train"], eval_dataset=ds["dev"],
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_LEN,  # older TRL: max_seq_length
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=2,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),  # hydrotest: MAX_STEPS=30
        logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100, save_total_limit=3,  # a disconnect costs minutes
        output_dir=CKPT, optim="adamw_8bit", seed=3407, report_to="none"))
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")
trainer.train(resume_from_checkpoint=get_last_checkpoint(CKPT) if os.path.isdir(CKPT) else None)

model.save_pretrained(f"{OUT}/lora_adapter")
tokenizer.save_pretrained(f"{OUT}/lora_adapter")
model.save_pretrained_merged(f"{OUT}/merged_16bit", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf(f"{OUT}/gguf", tokenizer, quantization_method="q4_k_m")
```

Each line of `data/train.jsonl` is `{"messages": [{"role": "system", ...}, {"role": "user",
"content": spec.to_prompt()}, {"role": "assistant", "content": "<the verified solve() code>"}]}`.

### 8.2 Dev pass@1 during training (sketch)

```python
# UNTESTED SKETCH: checkpoint selection by what you care about, not by loss.
from transformers import TrainerCallback

class DevPassAt1(TrainerCallback):
    def __init__(self, dev_specs, generate_fn, grade_fn):
        self.dev_specs, self.generate, self.grade = dev_specs, generate_fn, grade_fn

    def on_evaluate(self, args, state, control, model=None, **kw):
        FastLanguageModel.for_inference(model)
        codes = [self.generate(model, s.to_prompt()) for s in self.dev_specs[:60]]  # greedy
        score = sum(self.grade(s, c) for s, c in zip(self.dev_specs, codes)) / len(codes)
        state.log_history.append({"step": state.global_step, "dev_pass@1": score})
        FastLanguageModel.for_training(model)
```

`grade_fn` should call `cvbench.evaluate.evaluate_solution` (manufactured + metamorphic gates).

### 8.3 Ollama `Modelfile`

```text
# UNTESTED TEMPLATE. Prefer the Modelfile Unsloth writes next to the GGUF, or copy TEMPLATE from
# `ollama show qwen2.5-coder:7b --modelfile`. The template must match the one used in training.
FROM ./gguf/<the .gguf file Unsloth wrote>
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER num_ctx 8192
SYSTEM """You are opencv-expert, an OpenCV 5.0 (Python cv2) worker. Given a task contract, return one Python function `solve(...) -> dict` that satisfies it. No file, GUI or network access; never catch exceptions to hide failures."""
```

Then: `ollama create opencv-expert -f Modelfile`.

### 8.4 MCP server (`serve/server.py`)

```python
# UNTESTED TEMPLATE. Register with:
#   claude mcp add opencv-expert -- python /abs/path/opencv-specialist/serve/server.py
import json
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox.run import run_solution  # noqa: E402

OLLAMA, MODEL = "http://localhost:11434/api/chat", "opencv-expert"
SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
RUNS = Path.home() / ".opencv-expert" / "runs"
mcp = FastMCP("opencv-expert")


def describe(out: dict, run_dir: Path) -> dict:
    """Small values inline; images saved to disk so the orchestrator can look at them."""
    report = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.dtype == np.uint8 and a.ndim in (2, 3) and min(a.shape[:2]) > 8:
            path = run_dir / f"{k}.png"
            cv2.imwrite(str(path), a)
            report[k] = {"image": str(path), "shape": list(a.shape)}
        else:
            report[k] = a.tolist() if a.size <= 64 else {"shape": list(a.shape), "dtype": str(a.dtype)}
    return report


@mcp.tool()
def opencv_solve(task: str, image_path: str, outputs: dict[str, str], max_repairs: int = 3) -> dict:
    """Write, run and check OpenCV 5.0 code for an image-processing task on a local image.

    `outputs` maps each expected output name to its description,
    e.g. {"areas": "float (N,), px^2, largest first", "count": "int"}.
    Returns the code, measured outputs (images saved as PNG paths) and the attempt log.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"verified": False, "log": [f"cannot read {image_path}"]}
    contract = (f"Task: {task}\nOpenCV: 5.0 (Python cv2, headless)\nInputs:\n"
                f"  - img: uint8, shape {img.shape}, BGR\nReturn a dict with:\n"
                + "".join(f"  - {k!r}: {v}\n" for k, v in outputs.items())
                + "Write one function, `def solve(img) -> dict`. No file, GUI or network access.")
    messages, log, code = [{"role": "user", "content": contract}], [], ""
    for attempt in range(1, max_repairs + 2):
        reply = httpx.post(OLLAMA, timeout=180, json={
            "model": MODEL, "messages": messages, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0}}).json()
        code = json.loads(reply["message"]["content"])["code"]
        res = run_solution(code, [{"img": img}])
        missing = [k for k in outputs if res.ok and k not in res.outputs[0]]
        if res.ok and not missing:
            run_dir = RUNS / f"{abs(hash(code)) % 10**8}"
            run_dir.mkdir(parents=True, exist_ok=True)
            return {"verified": True, "attempts": attempt, "code": code, "log": log,
                    "measurements": describe(res.outputs[0], run_dir)}
        problem = res.summary() if not res.ok else f"missing outputs {missing}"
        log.append(problem)
        messages += [{"role": "assistant", "content": json.dumps({"code": code})},
                     {"role": "user", "content": f"That failed: {problem}\nReturn the corrected full function."}]
    return {"verified": False, "attempts": max_repairs + 1, "code": code, "log": log}


if __name__ == "__main__":
    mcp.run()
```

## 9. References

**Blueprint's four:**
- Gorilla (Patil et al., NeurIPS 2024, arXiv:2305.15334)
- HuggingGPT (Shen et al., NeurIPS 2023, arXiv:2303.17580)
- Magicoder / OSS-Instruct (Wei et al., ICML 2024, arXiv:2312.02120)
- QLoRA (Dettmers et al., NeurIPS 2023, arXiv:2305.14314)

**Added:**
- Allen-Zhu & Li, *Knowledge Capacity Scaling Laws*, ICLR 2025, arXiv:2404.05405. The bits/param budget.
- Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024, arXiv:2405.09673. Replay and forgetting.
- Gudibande et al., *The False Promise of Imitating Proprietary LLMs*, ICLR 2024, arXiv:2305.15717. Loss ≠ capability.
- Chen et al., *Teaching Large Language Models to Self-Debug*, 2023, arXiv:2304.05128. The repair loop and repair data.
- Chen et al., *Evaluating Large Language Models Trained on Code*, 2021, arXiv:2107.03374. Unbiased pass@k.
- Lai et al., *DS-1000*, ICML 2023, arXiv:2211.11501. Executable library-specific eval with anti-memorization perturbations.
- Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP 2021, arXiv:2012.14913. Why LoRA should target the MLP.
- Roache, *Code Verification by the Method of Manufactured Solutions*, J. Fluids Eng. 124(1), 2002. The verifier pattern.
- OpenCV wiki, *OpenCV 4 to 5 migration*. Compare it against `probes/results/diff_4.14_vs_5.0.md`.
