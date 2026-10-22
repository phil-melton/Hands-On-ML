"""Drills: small verified training tasks, each on one measured OpenCV 5.0 behaviour (or one timeless gotcha).

    from foundry.drills import DRILLS, DRILL_PHRASINGS
    from cvbench.gotchas.core import grade

    check = DRILLS["drill_hough_longest"]
    check.prompt()                        # TaskSpec.to_prompt(): the user message (phrasing 0)
    DRILL_PHRASINGS[check.id]             # 3 task lines, index 0 == check.spec.task
    grade(check, model_code).passed       # lint -> sandbox -> contract -> verifier
"""
from __future__ import annotations

from . import calling, drift, dtypes, numpy2, removed, shapes

_MODULES = (shapes, dtypes, calling, removed, numpy2, drift)

DRILLS: dict[str, "shapes.Check"] = {}
DRILL_PHRASINGS: dict[str, list[str]] = {}
for _m in _MODULES:
    for _c in _m.CHECKS:
        assert _c.id not in DRILLS, f"duplicate drill id {_c.id}"
        DRILLS[_c.id] = _c
    DRILL_PHRASINGS.update(_m.PHRASINGS)

__all__ = ["DRILLS", "DRILL_PHRASINGS"]
