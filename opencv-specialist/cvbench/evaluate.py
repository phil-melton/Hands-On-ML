"""Grade a candidate solution against a task family through four independent gates.

    exec-only     does it run without raising?  (all a bare exec() harness can see)
    lint          does it announce forbidden behaviour? (sandbox/lint.py)
    manufactured  does it recover the manufactured answer within tolerance?
    metamorphic   is its answer invariant under the task's symmetries?

The gates are run independently so you can see which defect each one catches.
In production they run in order and stop at the first failure.

A family module provides: SPEC, make_scene(seed), inputs(scene),
verify_manufactured(scene, out), transforms(scene), verify_metamorphic(base, variants).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from sandbox.lint import lint
from sandbox.run import run_solution


@dataclass
class Gate:
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        return "PASS" if self.passed else "FAIL"


def evaluate_solution(family: ModuleType, code: str, *, n_scenes: int = 20, n_meta: int = 5,
                      seed: int = 0, timeout_s: float = 60.0) -> dict[str, Gate]:
    scenes = [family.make_scene(seed + i) for i in range(n_scenes)]
    gates: dict[str, Gate] = {}

    r = run_solution(code, [family.inputs(scenes[0])], lint=False, timeout_s=timeout_s)
    gates["exec-only"] = Gate(r.ok, r.summary())

    findings = lint(code)
    gates["lint"] = Gate(not findings, "; ".join(map(str, findings)) or "clean")

    r = run_solution(code, [family.inputs(s) for s in scenes], lint=False, timeout_s=timeout_s)
    if not r.ok:
        gates["manufactured"] = Gate(False, r.summary())
    else:
        gates["manufactured"] = Gate(True, f"{n_scenes} scenes within tolerance")
        for s, out in zip(scenes, r.outputs):
            v = family.verify_manufactured(s, out)
            if not v:
                gates["manufactured"] = Gate(False, f"scene {s.seed}: {v.reason}")
                break

    batch, layout = [], []
    for s in scenes[:n_meta]:
        variants = family.transforms(s)
        layout.append(list(variants))
        batch.append(family.inputs(s))
        batch.extend(variants.values())
    r = run_solution(code, batch, lint=False, timeout_s=timeout_s)
    if not r.ok:
        gates["metamorphic"] = Gate(False, r.summary())
    else:
        gates["metamorphic"] = Gate(True, f"{n_meta} scenes x {len(layout[0])} symmetries invariant")
        i = 0
        for s, names in zip(scenes, layout):
            base, variants = r.outputs[i], dict(zip(names, r.outputs[i + 1:i + 1 + len(names)]))
            i += 1 + len(names)
            v = family.verify_metamorphic(base, variants)
            if not v:
                gates["metamorphic"] = Gate(False, f"scene {s.seed}: {v.reason}")
                break
    return gates
