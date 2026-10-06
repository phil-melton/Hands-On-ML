"""One source of truth for what training and serving must agree on (Rule 7).

The system prompt, the chat template the model is trained under, and the Ollama TEMPLATE it is
served under all live here. ``train_lora`` asserts that the base tokenizer's own chat template
renders every training example exactly as ``render_turn`` does, so a drift fails before any
GPU time is spent rather than as "great in training, garbage in Ollama".
"""
from __future__ import annotations

from pathlib import Path

WORK = Path.home() / "opencv-expert"  # large artifacts: runs, logs, llama.cpp (ext4 inside the D:-hosted vhdx)
OLLAMA = "http://127.0.0.1:11434"  # the Windows Ollama server, reached through WSL mirrored networking

# Identical to the Modelfile SYSTEM in PLAN.md §8.3.
SYSTEM = ("You are opencv-expert, an OpenCV 5.0 (Python cv2) worker. Given a task contract, return one Python "
          "function `solve(...) -> dict` that satisfies it. No file, GUI or network access; never catch "
          "exceptions to hide failures.")

# ChatML turn markers shared by the Qwen2.5 and Qwen3.5 families.
USER_MARK = "<|im_start|>user\n"
END = "<|im_end|>"

# Qwen3.5 renders an empty think block in front of every non-thinking answer (measured 2026-10-05
# with the Qwen/Qwen3.5-4B tokenizer); it belongs to the answer prefix, so it is masked from the loss
# and must also end the serve-time prompt. PLAN.md Phase 2: train and serve non-thinking.
# Its template also trims the answer (the reference's trailing newline disappears before <|im_end|>).
_NO_THINK = "<think>\n\n</think>\n\n"

_OLLAMA_TEMPLATE = (
    '{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}'
    '<|im_start|>assistant\n')

BASES: dict[str, dict] = {
    "qwen2.5-coder-7b": {
        "repo": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",  # pre-quantized NF4, 5.5 GB
        "upstream": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "license": "apache-2.0",
        "loader": "FastLanguageModel",
        "params_b": 7.6,
        "answer_prefix": "<|im_start|>assistant\n",
        "trim_answer": False,
        "ollama_template": _OLLAMA_TEMPLATE,
    },
    "qwen3.5-4b": {
        "repo": "unsloth/Qwen3.5-4B",  # bf16, 9.3 GB; no pre-quantized copy exists, quantized at load
        "upstream": "Qwen/Qwen3.5-4B",
        "license": "apache-2.0",
        "loader": "FastModel",  # Qwen3_5ForConditionalGeneration (vision-language); text-only LoRA
        "params_b": 4.7,  # 9.3 GB / 2 bytes, vision tower included
        "answer_prefix": "<|im_start|>assistant\n" + _NO_THINK,
        "trim_answer": True,
        "ollama_template": _OLLAMA_TEMPLATE + _NO_THINK,
    },
    "qwen3.5-9b": {
        "repo": "unsloth/Qwen3.5-9B",  # bf16, 19.3 GB
        "upstream": "Qwen/Qwen3.5-9B",
        "license": "apache-2.0",
        "loader": "FastModel",
        "params_b": 9.65,
        "answer_prefix": "<|im_start|>assistant\n" + _NO_THINK,
        "trim_answer": True,
        "ollama_template": _OLLAMA_TEMPLATE + _NO_THINK,
    },
}

OLLAMA_PARAMETERS = {"stop": [END], "temperature": 0, "num_ctx": 8192}


def render_turn(base: str, system: str, user: str, answer: str | None = None) -> str:
    """The exact text the model sees: the serve-time prompt, plus the answer and its end token when training."""
    prompt = (f"<|im_start|>system\n{system}{END}\n{USER_MARK}{user}{END}\n" + BASES[base]["answer_prefix"])
    if answer is None:
        return prompt
    return prompt + (answer.strip() if BASES[base]["trim_answer"] else answer) + END + "\n"


def modelfile(base: str, gguf_name: str) -> str:
    """Modelfile text recorded next to each GGUF (PLAN.md §8.3)."""
    params = "".join(f'PARAMETER stop "{s}"\n' for s in OLLAMA_PARAMETERS["stop"])
    params += "".join(f"PARAMETER {k} {v}\n" for k, v in OLLAMA_PARAMETERS.items() if k != "stop")
    return (f"FROM ./{gguf_name}\nTEMPLATE \"\"\"{BASES[base]['ollama_template']}\"\"\"\n{params}"
            f"SYSTEM \"\"\"{SYSTEM}\"\"\"\n")
