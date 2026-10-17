"""The gotcha suite: small checks whose common wrong answer fails and whose correct 5.0 answer passes.

    from cvbench.gotchas import CHECKS, grade, run_suite

    check = CHECKS["shape_houghlinesp"]
    check.prompt()                      # what a model is given (TaskSpec.to_prompt())
    grade(check, model_code)            # lint -> sandbox -> contract -> verifier
    run_suite(lambda c: ask_model(c.prompt()), workers=8).summary()

See ``core.py`` for the types and ``tests/test_gotchas.py`` for the pinned behaviour.
"""
from __future__ import annotations

from . import calling, drawing, drift, dtypes, shapes
from .core import (GROUP_TITLES, GROUPS, ROLES, Case, Check, Grade, SuiteReport, Verdict, Wrong, grade,
                   run_suite)

CHECKS: dict[str, Check] = {}
for _module in (shapes, dtypes, drawing, calling, drift):
    for _check in _module.CHECKS:
        assert _check.id not in CHECKS, f"duplicate check id {_check.id}"
        CHECKS[_check.id] = _check


def catalogue_markdown() -> str:
    """The catalogue as a markdown table (id, group, source probe, subset, role, what goes wrong)."""
    rows = ["| id | group | source probe | subset | role | the gotcha |", "|---|---|---|---|---|---|"]
    for c in CHECKS.values():
        rows.append(f"| `{c.id}` | {GROUP_TITLES[c.group]} | {', '.join(f'`{p}`' for p in c.probes)} | "
                    f"{c.subset} | {c.role} | {c.gotcha} |")
    return "\n".join(rows)


__all__ = ["catalogue_markdown", "CHECKS", "GROUPS", "GROUP_TITLES", "ROLES", "Case", "Check", "Grade", "SuiteReport", "Verdict",
           "Wrong", "grade", "run_suite"]
