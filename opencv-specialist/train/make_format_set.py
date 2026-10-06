"""Stage 1 hydrotest data: 64 format-test examples built from the ``contour_area`` family.

This tests plumbing, not learning. The user message is ``SPEC.to_prompt()`` under each of the
family's four phrasings and the assistant message is the family's REFERENCE solution, as
AGENT_PROMPT.md Stage 1 specifies. The targets are the hand-written reference, not teacher output,
so this set must never feed a real training run (Rule 5). The dev split repeats the four phrasings:
it exercises the eval path, it does not measure generalization.

Run from opencv-specialist/:  python -m train.make_format_set
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from cvbench.families import contour_area as fam
from train.config import SYSTEM

OUT = Path(__file__).resolve().parents[1] / "data" / "hydrotest"
N_TRAIN = 64


def example(phrasing: str) -> dict:
    spec = dataclasses.replace(fam.SPEC, task=phrasing)
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": spec.to_prompt()},
                         {"role": "assistant", "content": fam.REFERENCE}]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {"train": [example(fam.PHRASINGS[i % len(fam.PHRASINGS)]) for i in range(N_TRAIN)],
              "dev": [example(p) for p in fam.PHRASINGS]}
    for name, rows in splits.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"wrote {len(rows)} examples to {OUT / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
