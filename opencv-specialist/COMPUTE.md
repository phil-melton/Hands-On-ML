# Compute runbook

You don't need to buy a computer for this project. You rent minutes on Google's GPUs with the
Colab allowance in your Google AI Pro plan, and your own machine only serves the finished model.

## Budget at a glance

- **Allowance:** 200 Colab compute units (CU) per month with Google AI Pro. Colab benefits were
  added to Google AI plans on 2026-09-22. 1 CU ≈ $0.10.
- **Rates:** third-party measurements from March 2026. Colab shows the live rate for your runtime.

| GPU | VRAM | ≈ CU/h | Use it for |
|---|---|---|---|
| T4 | 15 GB | 1.2 | QLoRA fallback; cheap inference |
| **L4** | 22.5 GB | **1.7** | **Default:** 16-bit LoRA on a 7B; batched vLLM evals |
| A100 40 GB | 40 GB | 5.4 | Fastest training; a self-hosted teacher at 4-bit |
| A100 80 GB | 80 GB | 7.5 | A Qwen3-Coder-Next-class teacher at 4-bit |

**Plan for the whole project:**

| Item | GPU time | CU |
|---|---|---|
| Hydrotest (3 base-model candidates) | ~2 h L4 | ~4 |
| Baseline bake-off (5 rungs × ~800 generations) | ~1.5 h L4 (vLLM) | ~3 |
| Open-weight teacher: pilot, plus full generation if it wins | ~4–6 h L4 | ~10 |
| Convergence study + final run (3.5 epoch-equivalents) + dev evals | ~4 h L4 | ~7 |
| GGUF export + quantization ladder (bf16 and Q8 rungs) | ~2 h | ~4 |
| Re-runs and mistakes (2× margin) | | ~25 |
| **Total** | | **~50 of 200 per month** |

**The only line that can cost real money is a frontier-API teacher.** A full run is about
20 M input and 6 M output tokens. Multiply by your teacher's current per-million-token prices
(tens of dollars on a fast/cheap tier, low hundreds on a top tier). The Phase 1 pilot tells you
whether you need it at all.

## Predict before you spend

```
CU per epoch = (CU/h) × tokens_per_epoch / (tokens_per_s × 3600)
tokens_per_epoch ≈ examples × mean tokens            ≈ 3,000 × 700 = 2.1 M
tokens_per_s     ≈ peak FLOP/s × utilization / (6 N)  (6N: forward + recompute + backward, LoRA)
```

| GPU | Peak (bf16/fp16) | Assumed utilization | tok/s (7.6 B) | min/epoch | CU/epoch |
|---|---|---|---|---|---|
| T4 | 65 TFLOP/s | 25 % | ~360 | ~98 | ~2.0 |
| L4 | 121 TFLOP/s | 35 % | ~930 | ~38 | ~1.1 |
| A100 40 GB | 312 TFLOP/s | 35 % | ~2,400 | ~15 | ~1.35 |

Unsloth prints tokens/s after a few steps. Log the prediction next to the measurement in the
usage log below: **a miss of more than 3× is a bug, not bad luck.**

## Path A: Colab notebook (start here)

Your first session is the **hydrotest**: about 30 min on an L4, ~1 CU. It proves the plumbing
(LoRA → merge → GGUF), not the model.

`train/train_lora.py` doesn't exist yet: create it from the template in PLAN.md §8.1. It is left
as an untested template on purpose, because its first run *is* the hydrotest.

```python
# 0. Runtime → Change runtime type → L4 GPU.  (A100 is fine. On a T4, set QLORA=1.)
!nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. Persist everything on Drive: the VM is wiped when it stops.
from google.colab import drive
drive.mount("/content/drive")

# 2. Code. If the repo is private, use a GitHub token or upload the folder to Drive.
!git clone -b claude/opencv-specialist-plan https://github.com/phil-melton/Hands-On-ML.git
%cd Hands-On-ML/opencv-specialist
!pip uninstall -y -q opencv-python opencv-contrib-python   # Colab's preinstalled 4.x would shadow 5.0
!pip install -q unsloth -r requirements-cv5.txt

# 3. Hydrotest: 30 steps on a tiny data file, then merge and export.
%env MAX_STEPS=30
%env OUT=/content/drive/MyDrive/opencv-expert/hydrotest
!python train/train_lora.py        # from the template in PLAN.md §8.1
```

When it finishes, use **Runtime → Disconnect and delete runtime.** An idle GPU keeps burning CUs.

## Path B: Colab CLI (drive it from your terminal, or let Claude Code drive it)

The Colab CLI was released in June 2026. It runs on Linux and macOS; on Windows, use WSL2.

```bash
pip install google-colab-cli                 # or: uv tool install google-colab-cli
colab run --gpu L4 train/train_lora.py       # one shot: provision, run, release
# or keep a session for several steps (the VM starts empty: upload or clone the repo first)
colab new --gpu L4
colab upload data/train.jsonl /content/data/train.jsonl
colab exec -f train/train_lora.py
colab download /content/drive/MyDrive/opencv-expert/gguf ./gguf
colab stop
colab usage                                  # compute units spent
```

- **CLI details** (per its README):
  - GPU names are T4, L4, G4, A100 and H100.
  - The first use links your Google account in a browser.
  - Run `colab --help` before relying on any flag shown here.
- **Letting Claude Code drive it:** copy the CLI repository's `skills/colab-operator/` folder into
  `~/.claude/skills/`. Claude Code on your machine can then provision, run and stop jobs for you.

## Path C: Kaggle (backup)

- **Quota:** about 30 free GPU-hours per week, on 2× T4 or 1× P100. GPU and internet access need
  a phone-verified account.
- **Recipe:** use the T4 recipe (`QLORA=1`). Two T4s don't double a single LoRA run without extra
  setup, so treat them as one trainer plus one spare GPU for evals.
- **Saving outputs:** persist them as a Kaggle dataset, or push them to a private Hugging Face repo.

## Shop rules

1. **Checkpoint every ~100 steps** to Drive and resume with `get_last_checkpoint`; the template
   does both. A disconnect then costs minutes, not the run.
2. **Stop the runtime the moment a job ends.**
3. **The GPU type isn't guaranteed.** Code for the L4, accept an A100, fall back to a T4 (`QLORA=1`).
4. **Record the versions** of torch, transformers, trl, unsloth and opencv in the usage log for
   every run.
5. **Model artifacts never go in git.** `.gitignore` covers `*.gguf` and `*.safetensors`; use
   Drive or HF instead.

## Usage log (prediction vs measurement)

| date | GPU | job | predicted CU | tokens/s measured | CU measured (`colab usage`) | versions / notes |
|---|---|---|---|---|---|---|
| | | hydrotest | 1 | | | |
| | | bake-off | 3 | | | |
| | | teacher pilot | 2 | | | |
| | | train 25 % | 0.6 | | | |
| | | train 50 % | 1.1 | | | |
| | | train 100 % | 2.2 | | | |
| | | quant ladder | 4 | | | |

## Troubleshooting by symptom

| Symptom | Likely cause | Fix |
|---|---|---|
| tokens/s more than 3× below the prediction | padding waste, no packing, fp32 fallback, CPU-bound data loading | check bf16 on L4/A100; bucket or pack by length; `num_proc` in `datasets.map` |
| OOM on the L4 in 16-bit | batch × sequence too large | batch 1 with grad-accum 16, `MAX_LEN` from the measured p99; last resort `QLORA=1` |
| run dies mid-way | session limit or disconnect | rerun the same command; it resumes from the last checkpoint |
| GGUF output is garbage in Ollama | chat-template mismatch | use Unsloth's Modelfile, or the `TEMPLATE` from `ollama show qwen2.5-coder:7b --modelfile` |
| `cv2.__version__` is 4.x on Colab | preinstalled `opencv-python` shadows the headless 5.0 wheel | `pip uninstall -y opencv-python opencv-contrib-python`, reinstall `requirements-cv5.txt` |

## Path L: local RTX 3060 (chosen 2026-10-04)

The card has 12 GB, so a 7B trains with QLoRA (`QLORA=1`) at batch 1, grad-accum 16, MAX_LEN from the
measured p99. Predictions use the formula above with 25.6 TFLOP/s dense fp16 and 20 % utilization.

| GPU | Peak (fp16) | Assumed utilization | tok/s (7.6 B) | tok/s (4 B) | h/epoch (7.6 B, 2.1 M tok) |
|---|---|---|---|---|---|
| RTX 3060 12 GB | 25.6 TFLOP/s | 20 % (QLoRA) | ~110 | ~210 | ~5.3 |

Runbook (inside WSL2 Ubuntu, GPU visible via `nvidia-smi`):

```bash
cd /mnt/d/opencv-expert            # large files live on D:
python3 -m venv .venv-train && . .venv-train/bin/activate
pip install unsloth bitsandbytes -r /path/to/opencv-specialist/requirements-cv5.txt
QLORA=1 MAX_STEPS=30 OUT=/mnt/d/opencv-expert/hydrotest nohup python train/train_lora.py > hydrotest.log 2>&1 &
```

Shop rules that change: no CU to stop, but close other GPU apps first (the desktop already holds ~1.2 GB),
launch every job detached with a log, and record tokens/s from the Unsloth banner next to the prediction.
What does not fit: a 30B teacher at 4-bit (~18 GB) and a 7B bf16 eval (15 GB). Measure CPU offload for
the MoE teacher (only ~3 B active) before paying for Colab.
