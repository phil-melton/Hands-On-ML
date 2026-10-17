"""Types and runner for the gotcha suite.

A *check* is one small task whose common wrong answer (a 4.x idiom or a timeless
mistake) fails and whose correct OpenCV 5.0 answer passes. It carries

    spec         a ``TaskSpec``: ``spec.to_prompt()`` is what a model sees
    make_cases   deterministic cases (a batch, run in ONE sandbox process); a case is the
                 inputs handed to ``solve`` plus the manufactured truth only the verifier sees
    verify       judges one (case, outputs) pair: shape, dtype, values
    reference    a correct ``solve()`` (source text)
    wrong        named wrong ``solve()`` variants that reproduce the gotcha

All solution code runs through ``sandbox.run_solution``; nothing in this package
``exec()``s model code. Verifiers run in the parent process on the decoded outputs.

Every wrong variant pins HOW it fails: ``fails="IndexError"`` (the run must raise
that exception type) or ``fails="verifier"`` (the run must finish and the verifier
must reject it). The second kind is the dangerous one: an ``exec()``-only harness
accepts it. ``run_suite`` reports both numbers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs  # noqa: F401  (re-exported for the modules)
from sandbox.run import RunResult, run_solution

GROUPS = ("shapes", "dtypes", "drawing", "calling", "drift")
GROUP_TITLES = {
    "shapes": "Output shapes",
    "dtypes": "dtypes",
    "drawing": "Views and drawing",
    "calling": "Calling conventions",
    "drift": "Behaviour drift",
}
ROLES = ("gotcha", "control")  # a control is behaviour that did NOT change: it catches over-application


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


PASS = Verdict(True)


@dataclass(frozen=True)
class Case:
    inputs: dict  # keyword arguments for solve(): numpy arrays and JSON scalars
    truth: object = None  # manufactured answer, visible to the verifier only


def fail(reason: str) -> Verdict:
    return Verdict(False, reason)


@dataclass(frozen=True)
class Wrong:
    why: str  # what the variant does, and which idiom it reproduces
    code: str
    fails: str  # exception class name the run must raise, or "verifier"
    error_contains: str = ""  # optional substring the error text must contain


@dataclass(frozen=True)
class Check:
    id: str
    group: str
    probes: tuple[str, ...]  # source probe names in probes/results/probe_*.json (or "new:<what>" if measured here)
    depends_on_5_0: bool  # True = the "5.0 subset" (behaviour differs from 4.14); False = timeless
    role: str  # "gotcha" | "control"
    gotcha: str  # one line: what the common wrong answer gets wrong
    spec: TaskSpec
    make_cases: Callable[[], list[Case]]
    verify: Callable[[Case, dict], Verdict]
    reference: str
    wrong: Mapping[str, Wrong]
    lint: bool = True  # False only where the task needs an API the production lint bans (documented per check)
    note: str = ""

    def prompt(self) -> str:
        return self.spec.to_prompt()

    def inputs(self) -> list[dict]:
        return [c.inputs for c in self.make_cases()]

    @property
    def subset(self) -> str:
        return "5.0" if self.depends_on_5_0 else "timeless"


@dataclass
class Grade:
    check_id: str
    passed: bool
    stage: str  # ok | lint | run | contract | verify | malformed
    detail: str = ""
    ran: bool = False  # the code ran without raising: all an exec()-only harness can see
    error_type: str | None = None
    error: str | None = None
    seconds: float = 0.0

    def __str__(self) -> str:
        return f"{self.check_id}: {'PASS' if self.passed else 'FAIL'} [{self.stage}] {self.detail}".rstrip()


def grade(check: Check, code: str, *, timeout_s: float = 30.0) -> Grade:
    """Grade one code string against one check: lint, sandbox run, contract, verifier."""
    cases = check.make_cases()
    items = [c.inputs for c in cases]
    r: RunResult = run_solution(code, items, lint=check.lint, timeout_s=timeout_s)
    base = dict(check_id=check.id, error_type=r.error_type, error=r.error, seconds=r.seconds)
    if r.status == "lint":
        return Grade(passed=False, stage="lint", detail=r.summary(), **base)
    if not r.ok:
        return Grade(passed=False, stage="run", detail=r.summary(), **base)
    for i, (case, out) in enumerate(zip(cases, r.outputs)):
        problems = check_outputs(check.spec, out)
        if problems:
            return Grade(passed=False, stage="contract", ran=True, detail=f"item {i}: " + "; ".join(problems), **base)
        try:
            v = check.verify(case, out)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
            # only reachable for outputs the verifier cannot interpret; the test-suite pins that no wrong
            # variant and no reference ends here, so a verifier bug cannot hide behind this branch
            return Grade(passed=False, stage="malformed", ran=True,
                         detail=f"item {i}: verifier could not read the output ({type(e).__name__}: {e})", **base)
        if not v:
            return Grade(passed=False, stage="verify", ran=True, detail=f"item {i}: {v.reason}", **base)
    return Grade(passed=True, stage="ok", ran=True, detail=f"{len(items)} case(s) verified", **base)


# ---------------------------------------------------------------- suite

Solutions = Mapping[str, str] | Callable[[Check], str]


@dataclass
class SuiteReport:
    grades: dict[str, Grade] = field(default_factory=dict)
    checks: dict[str, Check] = field(default_factory=dict)

    def _rate(self, pred: Callable[[Check], bool], key: Callable[[Grade], bool]) -> tuple[int, int]:
        sel = [g for cid, g in self.grades.items() if pred(self.checks[cid])]
        return sum(key(g) for g in sel), len(sel)

    def passed(self, pred: Callable[[Check], bool] = lambda c: True) -> tuple[int, int]:
        return self._rate(pred, lambda g: g.passed)

    def ran(self, pred: Callable[[Check], bool] = lambda c: True) -> tuple[int, int]:
        """How many answers an exec()-only harness would accept."""
        return self._rate(pred, lambda g: g.ran)

    def by_group(self) -> dict[str, tuple[int, int]]:
        return {g: self.passed(lambda c, g=g: c.group == g) for g in GROUPS}

    def by_subset(self) -> dict[str, tuple[int, int]]:
        return {"5.0 subset": self.passed(lambda c: c.depends_on_5_0 and c.role == "gotcha"),
                "timeless": self.passed(lambda c: not c.depends_on_5_0 and c.role == "gotcha"),
                "controls": self.passed(lambda c: c.role == "control")}

    def summary(self) -> str:
        p, n = self.passed()
        e, _ = self.ran()
        lines = [f"gotcha suite: {p}/{n} pass the verifier ({e}/{n} would pass an exec()-only harness)"]
        for name, (a, b) in self.by_group().items():
            lines.append(f"  {GROUP_TITLES[name]:<22s} {a}/{b}")
        for name, (a, b) in self.by_subset().items():
            lines.append(f"  {name:<22s} {a}/{b}")
        return "\n".join(lines)


def run_suite(solutions: Solutions, checks: Mapping[str, Check] | None = None, *, workers: int = 1,
              timeout_s: float = 30.0) -> SuiteReport:
    """Run every check against one candidate solution each.

    ``solutions`` is either ``{check_id: code}`` or a callable ``Check -> code`` (e.g. a model call
    that renders ``check.prompt()``). A missing or empty answer fails the check at stage ``run``.
    """
    if checks is None:
        from cvbench.gotchas import CHECKS
        checks = CHECKS
    report = SuiteReport(checks=dict(checks))

    def one(check: Check) -> Grade:
        code = solutions(check) if callable(solutions) else solutions.get(check.id)
        if not code:
            return Grade(check_id=check.id, passed=False, stage="run", detail="no solution supplied")
        return grade(check, code, timeout_s=timeout_s)

    if workers <= 1:
        results = [one(c) for c in checks.values()]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, checks.values()))
    report.grades = {g.check_id: g for g in results}
    return report


# ---------------------------------------------------------------- helpers shared by the verifiers

def arr(out: dict, key: str) -> np.ndarray:
    return np.asarray(out[key])


def sorted_rows(a: np.ndarray) -> np.ndarray:
    """Rows in lexicographic order (for set-style comparisons of point lists)."""
    a = np.asarray(a)
    return a[np.lexsort(a.T[::-1])] if a.size else a


def match_points(truth: np.ndarray, got: np.ndarray, tol: float) -> str | None:
    """One-to-one nearest matching of two point sets within ``tol`` px. Returns a reason or None."""
    truth = np.asarray(truth, np.float64).reshape(-1, 2)
    got = np.asarray(got, np.float64).reshape(-1, 2)
    if len(truth) != len(got):
        return f"{len(got)} points for {len(truth)} expected"
    if not len(truth):
        return None
    d = np.linalg.norm(truth[:, None, :] - got[None, :, :], axis=2)
    used: set[int] = set()
    for i in np.argsort(d.min(axis=1)):
        order = [j for j in np.argsort(d[i]) if j not in used]
        if not order or d[i, order[0]] > tol:
            return f"expected point {truth[i].round(1).tolist()} has no match within {tol} px (closest {d[i].min():.2f} px)"
        used.add(int(order[0]))
    return None


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def point_line_distance(p, a, b) -> float:
    """Distance from p to the segment a-b (all in px)."""
    p, a, b = (np.asarray(v, np.float64) for v in (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))
