"""The training answer format and what a completion must look like to become one (Gate 4a decisions 5 and 6).

A training target is a fenced python block holding the imports and ``solve`` (with any helpers), and nothing else: no prose
before or after. ``format_answer`` builds it from the extracted code. The code is never edited: a completion whose code is
not already a clean module is not used (``answer_problems``), because training targets are the teacher's own output
(CLAUDE.md Rule 5), not something rewritten by hand.

``answer_of`` returns the part of a completion that is the answer. With a thinking teacher that is the text after
``</think>`` (colab_job.split_thinking); completions written before thinking mode existed have no ``answer`` field and are
split again from their ``text``, which leaves them unchanged. Code extraction, grading and examples all start from it.
"""
from __future__ import annotations

import ast

from foundry.colab_job import split_thinking

# Statements allowed at module level: imports, definitions, constants and a docstring. Anything else runs at import time
# (a demo call, a print, an ``if __name__`` block) and is not part of "the imports and solve".
_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def answer_of(comp: dict) -> str:
    """The answer part of one completions.jsonl row (never the reasoning)."""
    if "answer" in comp:
        return comp["answer"]
    return split_thinking(comp.get("text", ""), False)["answer"]


def format_answer(code: str) -> str:
    """The assistant turn of a training example: the code in one fenced python block."""
    return "```python\n" + code.strip("\n") + "\n```"


def row_problems(g: dict) -> list[str]:
    """Why the code of a grade row (foundry.verify) cannot be a training target as it is; [] when it can. The reply must have
    finished (not cut off by the token budget), its fence must be closed, and the code must be a clean module."""
    problems = []
    if g.get("finish_reason") not in (None, "stop"):
        problems.append(f"finish_reason {g['finish_reason']}")
    if g.get("extract") == "fence-unclosed":
        problems.append("unclosed code fence")
    return problems + answer_problems(g.get("code", ""))


def answer_problems(code: str) -> list[str]:
    """Why ``code`` cannot be a training answer as it is; [] when it can."""
    if not code.strip():
        return ["empty"]
    if "```" in code:
        return ["contains a code fence"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg}"]
    problems = []
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body):
        problems.append("no top-level solve()")
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            continue  # a module docstring
        if not isinstance(n, _TOP_LEVEL):
            problems.append(f"top-level {type(n).__name__} statement (line {n.lineno})")
    return problems
