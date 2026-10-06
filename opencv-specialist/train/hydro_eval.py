"""Stage 1 end check: generate one ``contour_area`` solution through Ollama and grade it on the four gates.

Generation goes through the served GGUF, not the training checkpoint (PLAN.md Phase 3: inspect after
heat treatment). The code runs only inside the sandbox, via cvbench.evaluate (Rule 4).

Run from opencv-specialist/:  python -m train.hydro_eval <ollama-model> [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx

from cvbench.evaluate import evaluate_solution
from cvbench.families import contour_area as fam
from train.config import OLLAMA


def extract_code(reply: str) -> str:
    """The trained format is bare code; tolerate a fenced block in case the model adds one."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return m.group(1) if m else reply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    # Ollama 0.32.9 marks qwen35 GGUFs as thinking-capable and then thinks by default in /api/chat, overriding
    # the trained non-thinking format (measured 2026-10-05: 526 tokens generated for a 133-token answer).
    # Train and serve non-thinking (PLAN.md Phase 2), so ask for it explicitly.
    caps = httpx.post(f"{OLLAMA}/api/show", json={"model": a.model}).json().get("capabilities", [])
    req = {"model": a.model, "messages": [{"role": "user", "content": fam.SPEC.to_prompt()}],
           "stream": False, "keep_alive": 0}  # unload afterwards: the next stage needs the VRAM
    if "thinking" in caps:
        req["think"] = False
    r = httpx.post(f"{OLLAMA}/api/chat", timeout=900, json=req)
    r.raise_for_status()
    body = r.json()
    reply = body["message"]["content"]
    if body["message"].get("thinking"):
        print(f"WARNING: the model returned {len(body['message']['thinking'])} chars of thinking")
    code = extract_code(reply)
    decode_tps = body.get("eval_count", 0) / max(body.get("eval_duration", 1) / 1e9, 1e-9)
    print(f"--- reply from {a.model} ({body.get('eval_count')} tokens, {decode_tps:.1f} tok/s decode):\n{reply}\n---")

    gates = evaluate_solution(fam, code)
    for name, g in gates.items():
        print(f"{name:13s} {g}  {g.detail}")
    passed = all(g.passed for g in gates.values())
    print("HYDROTEST", "PASS" if passed else "FAIL", flush=True)

    if a.out:
        a.out.write_text(json.dumps({
            "model": a.model, "passed": passed, "think": req.get("think"), "capabilities": caps,
            "thinking_chars": len(body["message"].get("thinking") or ""),
            "fenced": code != reply, "matches_reference": code.strip() == fam.REFERENCE.strip(),
            "decode_tokens_per_s": round(decode_tps, 1), "eval_count": body.get("eval_count"),
            "load_s": round(body.get("load_duration", 0) / 1e9, 1), "reply": reply,
            "gates": {k: {"passed": g.passed, "detail": g.detail} for k, g in gates.items()}}, indent=1))


if __name__ == "__main__":
    main()
