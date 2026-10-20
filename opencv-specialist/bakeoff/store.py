"""Run folder layout and append-only JSONL helpers.

One bake-off lives in one folder outside git (default ``~/opencv-expert/bakeoff/<name>/``):

    config.json        options, rungs and system prompts the generations were made with; a resume must match
    generations.jsonl  one row per model reply (append-only, so an interrupted run resumes where it stopped)
    warmups.jsonl      each rung's discarded warm-up request, kept apart so it cannot enter a statistic
    grades.jsonl       one row per graded reply
    grading.json       the grader settings (scenes, timeout) the grades were made with
    report.md, summary.json, audit.md

Rows are one JSON object per line. A process killed mid-write leaves at most one torn last line; readers skip
it and the writer cuts it off, so the file stays usable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from train.config import WORK

DEFAULT_ROOT = WORK / "bakeoff"


def read_jsonl(path: Path) -> list[dict]:
    """All rows of a JSONL file ([] if missing). A torn final line is skipped; a bad line elsewhere raises."""
    if not path.is_file():
        return []
    lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise ValueError(f"{path}:{i + 1}: not valid JSON")
    return rows


def _repair_tail(path: Path) -> None:
    """After a crash the last line may lack its newline. If it is a whole row, add the newline; if it is torn
    (readers already skip it), cut it off, so the next row starts on a clean line."""
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    start = data.rfind(b"\n") + 1
    try:
        json.loads(data[start:])
        path.write_bytes(data + b"\n")
    except ValueError:  # JSONDecodeError, or a multi-byte character cut in half
        path.write_bytes(data[:start])


class JsonlWriter:
    """Append rows to a JSONL file, flushing after each so a crash loses at most the row being written."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size:
            _repair_tail(path)
        self._f = path.open("a", encoding="utf-8", newline="\n")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, sort_keys=True) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@dataclass(frozen=True)
class RunDir:
    path: Path
    generations_file: Path | None = None  # grade/report over a generations file kept elsewhere
    grades_file: Path | None = None

    @classmethod
    def of(cls, name: str, root: Path | None = None) -> "RunDir":
        return cls((Path(root) if root else DEFAULT_ROOT) / name)

    config = property(lambda self: self.path / "config.json")
    generations = property(lambda self: self.generations_file or self.path / "generations.jsonl")
    warmups = property(lambda self: self.path / "warmups.jsonl")
    grades = property(lambda self: self.grades_file or self.path / "grades.jsonl")
    grading = property(lambda self: self.path / "grading.json")
    done = property(lambda self: self.path / "generate.done")  # written when generate finishes; grade --follow stops on it
    report_md = property(lambda self: self.path / "report.md")
    summary = property(lambda self: self.path / "summary.json")
    audit_md = property(lambda self: self.path / "audit.md")


def key_of(row: dict) -> tuple:
    """Identity of one generation or its grade: resume skips keys already present."""
    return (row["rung"], row["task_id"], row["mode"], row["sample"])


def keys(rows: Iterable[dict]) -> set[tuple]:
    return {key_of(r) for r in rows}
