"""The rungs of the ladder and the exact request each one sends.

A rung is an Ollama model tag plus a system prompt. The tags are created by hand from the base GGUFs; adding a
rung means adding one line to RUNGS. Everything that could change a reply is explicit here and recorded in the
run's config.json: Ollama's defaults are not trusted (``repeat_penalty`` defaults to 1.1, which hurts code, and the
Qwen3.5 GGUFs think by default in /api/chat, which would replace the non-thinking format we train and serve).

Two decoding modes, matching PLAN.md "Baseline bake-off":

    greedy   temperature 0, one reply per task               -> pass@1
    sample   temperature 0.8, top_p 0.95, n replies per task  -> unbiased pass@k (Chen et al. 2021)

Each request carries a seed derived from (rung, task id, mode, sample index) so a rerun asks the same question.
At temperature 0 the seed cannot matter; it is sent and recorded anyway.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from train.config import SYSTEM

NOTES_TOKEN_BUDGET = 1500  # PLAN.md: the notes block is at most 1.5k tokens


@dataclass(frozen=True)
class Rung:
    name: str
    model: str                 # Ollama tag, created from the base GGUF with no system prompt of its own
    think: bool | None = None  # False for the Qwen3.5 models; None omits the field (Qwen2.5-Coder cannot think)
    notes: bool = False        # True: SYSTEM followed by the OpenCV 5.0 notes block


RUNGS: dict[str, Rung] = {r.name: r for r in (
    Rung("r1-7b", "base-qwen2.5-coder-7b"),
    Rung("r2-4b", "base-qwen3.5-4b", think=False),
    Rung("r3-9b", "base-qwen3.5-9b", think=False),
    Rung("r4-7b-notes", "base-qwen2.5-coder-7b", notes=True),
    Rung("r5-9b-notes", "base-qwen3.5-9b", think=False, notes=True),  # Gate 3: the Stage 6 comparator for the 9B
    # Stage 6: the chosen Stage 5 checkpoint (r100 @ e1), merged and quantized (train/merge_lora.py). Same system prompt as
    # the baselines (train.config.SYSTEM) and no notes block: the fine-tune replaces the notes.
    Rung("s6-q4", "opencv-expert-q4", think=False),
    Rung("s6-q5", "opencv-expert-q5", think=False),
    Rung("s6-q8", "opencv-expert-q8", think=False),
    # Stage 6: Q5_K_M ships (the smallest quant within noise of the best), so its like-for-like comparator is the same
    # base + notes at Q5_K_M, quantized from the base BF16 GGUF with the same llama.cpp build. r5 is the base at Q4_K_M.
    Rung("r6-9b-notes-q5", "base-qwen3.5-9b-q5", think=False, notes=True),
)}

MODES: dict[str, dict] = {
    "greedy": {"temperature": 0.0},
    "sample": {"temperature": 0.8, "top_p": 0.95},
}


def system_prompt(rung: Rung, notes_text: str | None = None) -> str:
    """The system message: train.config.SYSTEM, plus the notes block for a notes rung."""
    if not rung.notes:
        return SYSTEM
    if not notes_text or not notes_text.strip():
        raise ValueError(f"rung {rung.name} needs the notes block (--notes PATH)")
    return SYSTEM + "\n\n" + notes_text.strip()


def notes_token_estimate(text: str) -> int:
    """Rough token count (3 characters per token, deliberately high). The measured count is the difference of the
    warm-up prompt_eval_count between the notes rung and its plain twin, shown in the report."""
    return -(-len(text.strip()) // 3)


def request_options(mode: str, *, num_predict: int, num_ctx: int, seed: int) -> dict:
    """The Ollama ``options`` of one request."""
    return {**MODES[mode], "repeat_penalty": 1.0, "num_predict": num_predict, "num_ctx": num_ctx, "seed": seed}


def sample_seed(rung: str, task_id: str, mode: str, sample: int) -> int:
    """Deterministic 31-bit seed from the request's identity (not Python's per-process salted hash())."""
    digest = hashlib.sha256(f"{rung}|{task_id}|{mode}|{sample}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
