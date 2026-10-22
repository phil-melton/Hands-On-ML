"""Shared plumbing for the drills: the gotcha suite's own types, plus a registry that keeps each drill's three phrasings.

A drill is a ``cvbench.gotchas.core.Check`` (spec, make_cases, verify, reference, wrong variants with ``fails=``), graded
with ``cvbench.gotchas.core.grade``. What differs from the gotcha suite is the purpose. Gotchas are TEST data; drills are
TRAINING data, so no drill reuses a gotcha's task, wording or scene (tests/test_drills.py pins that with a 5-gram check).

Conventions every drill module follows:

* ``spec.task`` is a normal request: no version, no mention of what goes wrong, no tolerance, no answer. It names an
  API only when a person asking for that job would ("with a probabilistic Hough transform", "with cv2.kmeans");
  it never names an API the correct answer must avoid.
* the three phrasings are ``[spec.task, alternative 1, alternative 2]``; they restate the same job with a different wording
  and keep the same inputs and outputs, so the contract below the task line is identical.
* ``make_cases`` is deterministic and returns several cases that differ in everything that matters (sizes, counts,
  angles), so that a hard-coded answer fails. Seeds live in 61_000..69_999, a block no family, split or gotcha uses.
* every wrong variant pins HOW it fails (``fails="IndexError"``, ``"error"`` for ``cv2.error``, or ``"verifier"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# re-exported for the drill modules
from cvbench.gotchas.core import (PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail,  # noqa: F401
                                  mask_iou, match_points, point_line_distance, sorted_rows)

SEED_BASE = 61_000  # drills draw their scenes from SEED_BASE .. SEED_BASE + 9_999
PHRASINGS_PER_DRILL = 3


def spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="drill", task=task, inputs=inputs, outputs=outputs)


def code(sig: str, body: str, extra_imports: str = "") -> str:
    """A solve() source text with the two standard imports (a drill's reference and wrong variants share them)."""
    return f"import cv2\nimport numpy as np\n{extra_imports}\n\ndef solve({sig}):\n{body}"


@dataclass
class Registry:
    """The drills of one module and their phrasings, keyed by drill id."""
    checks: list[Check] = field(default_factory=list)
    phrasings: dict[str, list[str]] = field(default_factory=dict)

    def add(self, check: Check, alt1: str, alt2: str) -> Check:
        assert check.id not in self.phrasings, f"duplicate drill id {check.id}"
        self.checks.append(check)
        self.phrasings[check.id] = [check.spec.task, alt1, alt2]
        return check
