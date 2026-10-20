"""Reply -> code, one function for every rung.

The trained format is bare code, but the untrained base models answer with a fenced block and a paragraph of
explanation. Extraction order:

    1. strip a leading <think>...</think> block (empty for a non-thinking Qwen3.5; if it holds text the model
       thought despite think:false, and its draft code must not be mistaken for the answer)
    2. take the first fenced block labelled python/py/python3 or unlabelled; blocks in other languages are skipped
    3. else the whole reply

A fence that never closes (the reply hit num_predict) is still taken, so a truncated answer is graded on what
the model wrote rather than on a stray ```python line. Which path was taken is recorded with the reply.
"""
from __future__ import annotations

import re
from typing import NamedTuple

_THINK = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.S)
_PY_LABELS = {"", "python", "py", "python3"}


class Extracted(NamedTuple):
    code: str
    path: str   # fence | fence-unclosed | bare | empty
    think: str  # "" | empty | nonempty: the leading think block that was stripped


def _first_python_fence(text: str) -> tuple[str, str] | None:
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s.startswith("```"):
            i += 1
            continue
        words = s[3:].strip().lower().split()
        label = words[0] if words else ""
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("```"):
            j += 1
        if label in _PY_LABELS:
            return "".join(lines[i + 1:j]), ("fence" if j < len(lines) else "fence-unclosed")
        i = j + 1  # skip the whole non-python block
    return None


def extract_code(reply: str) -> Extracted:
    think = ""
    m = _THINK.match(reply)
    if m:
        think = "nonempty" if m.group(1).strip() else "empty"
        reply = reply[m.end():]
    found = _first_python_fence(reply)
    code, path = found if found else (reply, "bare")
    if not code.strip():
        return Extracted("", "empty", think)
    return Extracted(code, path, think)
