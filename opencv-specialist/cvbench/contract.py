"""The task contract: the toleranced drawing the orchestrator hands the worker.

Training prompts, eval prompts and the MCP tool all render specs through
``TaskSpec.to_prompt()``, so the model is fixtured the same way in qualification
as in production.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class IOSpec:
    dtype: str  # numpy dtype name, or the kinds "float" / "int" / "any"
    shape: str  # human-readable, e.g. "HxWx3", "N", "scalar"
    ndim: int | None = None  # enforced when given (None for scalars)
    note: str = ""  # units, colour order, coordinate order, sorting...


@dataclass(frozen=True)
class TaskSpec:
    family: str
    task: str
    inputs: dict[str, IOSpec]
    outputs: dict[str, IOSpec]
    opencv: str = "5.0"

    @property
    def signature(self) -> str:
        return f"def solve({', '.join(self.inputs)}) -> dict"

    def to_prompt(self) -> str:
        lines = [f"Task: {self.task}", f"OpenCV: {self.opencv} (Python cv2, headless)", "Inputs:"]
        for name, s in self.inputs.items():
            lines.append(f"  - {name}: {s.dtype}, shape {s.shape}" + (f". {s.note}" if s.note else ""))
        lines.append("Return a dict with:")
        for name, s in self.outputs.items():
            lines.append(f"  - {name!r}: {s.dtype}, shape {s.shape}" + (f". {s.note}" if s.note else ""))
        lines.append(f"Write one function, `{self.signature}`. No file, GUI or network access; "
                     "do not catch exceptions to hide failures.")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def check_outputs(spec: TaskSpec, out: dict) -> list[str]:
    """Contract violations: missing keys, wrong kind of scalar, wrong dtype kind or ndim."""
    problems = []
    for name, s in spec.outputs.items():
        if name not in out:
            problems.append(f"missing output {name!r}")
            continue
        v = out[name]
        if s.shape == "scalar":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append(f"{name!r} should be a scalar {s.dtype}, got {type(v).__name__}")
            elif s.dtype == "int" and not isinstance(v, int):
                problems.append(f"{name!r} should be an int, got {v!r}")
            continue
        a = np.asarray(v)
        if s.ndim is not None and a.ndim != s.ndim:
            problems.append(f"{name!r} should have ndim {s.ndim}, got shape {a.shape}")
        kind_ok = {"float": "f", "int": "iu"}.get(s.dtype)
        if kind_ok is not None:
            if a.size and a.dtype.kind not in kind_ok:
                problems.append(f"{name!r} should be {s.dtype}, got {a.dtype}")
        elif s.dtype != "any" and str(a.dtype) != s.dtype:
            problems.append(f"{name!r} should be {s.dtype}, got {a.dtype}")
    return problems
