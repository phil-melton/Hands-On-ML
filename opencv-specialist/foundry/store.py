"""Run folder layout for the foundry; the JSONL helpers are bakeoff.store's (append-only, torn last line tolerated)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bakeoff.store import JsonlWriter, read_json, read_jsonl, write_json  # noqa: F401  (re-exported)
from train.config import WORK

DEFAULT_ROOT = WORK / "foundry"
STAGES = ("generate", "repair")


@dataclass(frozen=True)
class Stage:
    """The files of one teacher job: its input prompts, what the GPU job returned, and what grading made of it."""
    name: str
    prompts: Path
    completions: Path
    timing: Path
    grades: Path
    grading: Path   # the grader settings the grades were made with (a resume must match)
    session: Path   # the Colab driver's record of the job
    log: Path       # the job's console output


@dataclass(frozen=True)
class FoundryRun:
    path: Path

    @classmethod
    def of(cls, name: str, root: Path | str | None = None) -> "FoundryRun":
        return cls((Path(root) if root else DEFAULT_ROOT) / name)

    @property
    def name(self) -> str:
        return self.path.name

    config = property(lambda self: self.path / "config.json")
    specs = property(lambda self: self.path / "specs.jsonl")
    kept = property(lambda self: self.path / "kept.jsonl")
    repair_pairs = property(lambda self: self.path / "repair_pairs.jsonl")
    examples = property(lambda self: self.path / "examples.jsonl")
    report_md = property(lambda self: self.path / "report.md")
    summary = property(lambda self: self.path / "summary.json")

    def stage(self, name: str) -> Stage:
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; choose from {STAGES}")
        p = self.path
        pre = "" if name == "generate" else "repair_"
        return Stage(name=name, prompts=p / f"{pre}prompts.jsonl", completions=p / f"{pre}completions.jsonl",
                     timing=p / f"{pre}timing.json", grades=p / f"{pre}grades.jsonl", grading=p / f"{pre}grading.json",
                     session=p / f"session_{name}.json", log=p / f"colab_{name}.log")


def teacher_info(run: FoundryRun, stage: str = "generate") -> dict:
    """Which model produced a stage's completions, and under what license (CLAUDE.md Rule 5): from the Colab driver's
    session record, else the job's timing.json, else the completions themselves."""
    st = run.stage(stage)
    session = read_json(st.session, {}) or {}
    timing = read_json(st.timing, {}) or {}
    model = session.get("model") or timing.get("model")
    if not model:
        rows = read_jsonl(st.completions)
        model = rows[0].get("model") if rows else None
    return {"model": model or "unknown", "license": session.get("teacher_license") or "unrecorded",
            "revision": session.get("revision") or (timing.get("args") or {}).get("revision"),
            "quantization": session.get("quantization") or (timing.get("args") or {}).get("quantization"),
            "gpu": session.get("gpu") or (timing.get("env") or {}).get("gpu")}
