"""LoRA / QLoRA fine-tune with Unsloth, then merge and export a GGUF.

Adapted from the untested template in PLAN.md §8.1 for Path L (RTX 3060 12 GB, QLoRA, batch 1 x
grad-accum 16). Its first run is the Stage 1 hydrotest.

Run from opencv-specialist/:  python -m train.train_lora
Environment (default):
  BASE         key in train.config.BASES (qwen2.5-coder-7b)
  DATA         directory holding train.jsonl and dev.jsonl, relative to opencv-specialist/ (data/hydrotest)
  OUT          output directory (~/opencv-expert/runs/<BASE>)
  MAX_LEN      tokens per example; set to the measured p99 x 1.2 (2048)
  QLORA        1 = 4-bit NF4 base (1)
  MAX_STEPS    -1 = run NUM_EPOCHS (-1)
  NUM_EPOCHS   (2)
  BATCH        per-device batch (1)
  GRAD_ACCUM   gradient accumulation steps (16)
  EVAL_STEPS   (50)
  SAVE_STEPS   checkpoint interval (100)
  EXPORT       GGUF quantization method, or "none" to stop after the adapter (q4_k_m)
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import unsloth  # noqa: F401  (must be imported before transformers, trl and peft)
import torch
import transformers
import trl
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, FastModel
from unsloth.chat_templates import train_on_responses_only

from train.config import BASES, USER_MARK, WORK, render_turn

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("BASE", "qwen2.5-coder-7b")
CFG = BASES[BASE]
DATA = ROOT / os.environ.get("DATA", "data/hydrotest")
OUT = Path(os.environ.get("OUT", WORK / "runs" / BASE)).expanduser()
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))
QLORA = os.environ.get("QLORA", "1") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", 2))
BATCH = int(os.environ.get("BATCH", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 16))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", 50))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 100))
EXPORT = os.environ.get("EXPORT", "q4_k_m")
CKPT = OUT / "checkpoints"

summary: dict = {"base": BASE, "repo": CFG["repo"], "license": CFG["license"], "qlora": QLORA, "max_len": MAX_LEN,
                 "batch": BATCH, "grad_accum": GRAD_ACCUM, "max_steps": MAX_STEPS, "data": str(DATA),
                 "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "trl": trl.__version__, "unsloth": unsloth.__version__,
                              "python": platform.python_version()}}


def phase(name: str, t0: float) -> float:
    dt = time.time() - t0
    summary.setdefault("seconds", {})[name] = round(dt, 1)
    print(f"== {name}: {dt / 60:.1f} min", flush=True)
    return time.time()


def write_summary() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


t = time.time()
loader = FastModel if CFG["loader"] == "FastModel" else FastLanguageModel
model, processor = loader.from_pretrained(model_name=CFG["repo"], max_seq_length=MAX_LEN, load_in_4bit=QLORA,
                                          dtype=None)
tokenizer = getattr(processor, "tokenizer", processor)  # vision-language bases return a processor
lora = dict(r=16, lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
            random_state=3407)
if loader is FastModel:
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True, **lora)
else:
    model = FastLanguageModel.get_peft_model(
        model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], **lora)
summary["dtype"] = str(next(p for p in model.parameters() if p.is_floating_point()).dtype)
summary["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
t = phase("load", t)

ds = load_dataset("json", data_files={"train": str(DATA / "train.jsonl"), "dev": str(DATA / "dev.jsonl")})
ds = ds.map(lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
            batched=True)

# Rule 7: the base's own chat template must render exactly what Ollama will be sent at serve time.
for row in ds["train"].select(range(min(4, len(ds["train"])))):
    m = {x["role"]: x["content"] for x in row["messages"]}
    expected = render_turn(BASE, m["system"], m["user"], m["assistant"])
    if row["text"] != expected:
        raise SystemExit(f"chat template drift for {BASE}:\n--- base template\n{row['text']!r}\n"
                         f"--- train.config.render_turn\n{expected!r}")

lengths = sorted(len(tokenizer(x, add_special_tokens=False).input_ids) for x in ds["train"]["text"])
p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
summary["tokens"] = {"n": len(lengths), "mean": round(sum(lengths) / len(lengths), 1), "p99": p99,
                     "max": lengths[-1]}
print("token lengths:", summary["tokens"], flush=True)
if lengths[-1] > MAX_LEN:
    raise SystemExit(f"longest example has {lengths[-1]} tokens > MAX_LEN {MAX_LEN}: raise MAX_LEN, never truncate")

# transformers 5.5 ignores SFTConfig(logging_dir=...) (deprecated); without this the events land in ./runs (measured).
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(OUT / "tb"))
trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["dev"],
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_LEN, dataset_num_proc=1,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS, logging_steps=1, eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        output_dir=str(CKPT), optim="adamw_8bit", seed=3407,
        report_to="tensorboard", include_num_input_tokens_seen="all"))
trainer = train_on_responses_only(trainer, instruction_part=USER_MARK, response_part=CFG["answer_prefix"])

# The loss must see only the answer and its end token (PLAN.md §7 question 4).
labels = [x for x in trainer.train_dataset[0]["labels"] if x != -100]
answer_seen = tokenizer.decode(labels)
first = {x["role"]: x["content"] for x in ds["train"][0]["messages"]}
answer_rendered = render_turn(BASE, first["system"], first["user"], first["assistant"])[
    len(render_turn(BASE, first["system"], first["user"])):]  # answer + end token, as the template renders it
summary["loss_mask_ok"] = answer_seen.strip() == answer_rendered.strip()
print(f"loss sees {len(labels)} tokens; matches answer + end token: {summary['loss_mask_ok']}", flush=True)
if not summary["loss_mask_ok"]:
    raise SystemExit(f"train_on_responses_only masked the wrong span; loss sees:\n{answer_seen!r}")

summary["dev_loss_step0"] = round(trainer.evaluate()["eval_loss"], 4)
print("step-0 dev loss:", summary["dev_loss_step0"], flush=True)
write_summary()
t = phase("prepare", t)

torch.cuda.reset_peak_memory_stats()
resume = get_last_checkpoint(str(CKPT)) if CKPT.is_dir() else None
result = trainer.train(resume_from_checkpoint=resume)
tokens_seen = int(trainer.state.num_input_tokens_seen)
summary["train"] = {"runtime_s": round(result.metrics["train_runtime"], 1), "steps": trainer.state.global_step,
                    "tokens_seen": tokens_seen,
                    "tokens_per_s": round(tokens_seen / result.metrics["train_runtime"], 1),
                    "final_train_loss": round(result.metrics["train_loss"], 4),
                    "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    "resumed_from": resume}
summary["dev_loss_final"] = round(trainer.evaluate()["eval_loss"], 4)
print("train:", summary["train"], "| final dev loss:", summary["dev_loss_final"], flush=True)
write_summary()
t = phase("train", t)

model.save_pretrained(str(OUT / "lora_adapter"))
processor.save_pretrained(str(OUT / "lora_adapter"))
t = phase("save_adapter", t)

if EXPORT != "none":
    os.chdir(WORK)  # Unsloth clones and builds llama.cpp in the working directory; keep it out of the repo
    model.save_pretrained_merged(str(OUT / "merged_16bit"), processor, save_method="merged_16bit")
    t = phase("merge_16bit", t)
    model.save_pretrained_gguf(str(OUT / "gguf"), processor, quantization_method=EXPORT)
    t = phase("gguf_" + EXPORT, t)
    summary["gguf"] = sorted(str(p) for p in OUT.rglob("*.gguf"))
    print("gguf files:", summary["gguf"], flush=True)

write_summary()
print(json.dumps(summary, indent=1))
