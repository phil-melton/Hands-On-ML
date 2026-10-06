"""Export a saved LoRA adapter to GGUF without retraining.

Loads ``$OUT/lora_adapter`` (Unsloth resolves its base), then lets ``save_pretrained_gguf`` merge and
quantize. It skips ``train_lora``'s separate ``save_pretrained_merged`` step, which the 4B hydrotest
showed is a duplicate: the GGUF export writes its own merged 16-bit copy (LOGBOOK 2026-10-05).

Run from opencv-specialist/:  BASE=<key> OUT=<run dir> python -m train.export_gguf
Environment: BASE, OUT, QLORA (1), MAX_LEN (2048), EXPORT (q4_k_m), as in train_lora.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import unsloth  # noqa: F401  (must be imported before transformers, trl and peft)
from unsloth import FastLanguageModel, FastModel

from train.config import BASES, WORK

BASE = os.environ.get("BASE", "qwen2.5-coder-7b")
CFG = BASES[BASE]
OUT = Path(os.environ.get("OUT", WORK / "runs" / BASE)).expanduser()
EXPORT = os.environ.get("EXPORT", "q4_k_m")

t = time.time()
loader = FastModel if CFG["loader"] == "FastModel" else FastLanguageModel
model, processor = loader.from_pretrained(model_name=str(OUT / "lora_adapter"),
                                          max_seq_length=int(os.environ.get("MAX_LEN", 2048)),
                                          load_in_4bit=os.environ.get("QLORA", "1") == "1", dtype=None)
print(f"== load_adapter: {(time.time() - t) / 60:.1f} min", flush=True)

t = time.time()
os.chdir(WORK)  # Unsloth builds llama.cpp in the working directory; reuse ~/opencv-expert/llama.cpp
model.save_pretrained_gguf(str(OUT / "gguf"), processor, quantization_method=EXPORT)
minutes = (time.time() - t) / 60
print(f"== gguf_{EXPORT}: {minutes:.1f} min", flush=True)

summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary.setdefault("seconds", {})[f"export_only_gguf_{EXPORT}"] = round(minutes * 60, 1)
summary["gguf"] = sorted(str(p) for p in OUT.rglob("*.gguf"))
summary_path.write_text(json.dumps(summary, indent=1))
print("gguf files:", summary["gguf"], flush=True)
