"""Split rows -> tasks: the prompt a model is given and the facts the report groups by.

A family task is one row of data/splits/{dev,test}.jsonl: its prompt is the family SPEC with the row's phrasing
(``TaskSpec.to_prompt()``, the same call training and serving use, CLAUDE.md Rule 7) and it is graded on scenes
``seed .. seed + n_scenes - 1`` by cvbench.evaluate. The row's ``prompt_sha1`` is checked, so a prompt that drifted
from the manifest fails loudly here instead of silently changing what is measured.

A gotcha task is one row of data/splits/gotchas.jsonl: its prompt is ``Check.prompt()``.

Note for reading the statistics: tasks of one family cycle through the family's phrasings, so many tasks share an
identical prompt (37 distinct prompts across the 198 test tasks). ``cluster`` names the prompt; it is the unit
that is really independent at T=0.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

from bakeoff.store import read_jsonl
from cvbench.splits import SPLITS_DIR

SPLIT_CHOICES = ("test", "dev", "gotchas")  # test = the 198 family tasks; gotchas = the 40 checks (also test split)


@dataclass(frozen=True)
class Task:
    id: str
    kind: str          # family | gotcha
    family: str        # family name, or "gotcha:<group>"
    subset: str        # "5.0" | "timeless"
    held_out: bool
    role: str          # task (family) | gotcha | control
    seed: int | None   # first scene seed (family tasks)
    prompt: str
    cluster: str       # identical prompt <=> identical cluster

    @property
    def prompt_sha1(self) -> str:
        return hashlib.sha1(self.prompt.encode("utf-8")).hexdigest()

    def meta(self) -> dict:
        """The fields copied into every generation row, so later stages need not reload the splits."""
        return {"task_id": self.id, "kind": self.kind, "family": self.family, "subset": self.subset,
                "held_out": self.held_out, "role": self.role, "task_seed": self.seed, "cluster": self.cluster,
                "prompt_sha1": self.prompt_sha1}


def _family_task(row: dict) -> Task:
    fam = importlib.import_module(f"cvbench.families.{row['family']}")
    prompt = dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[row["phrasing_index"]]).to_prompt()
    if hashlib.sha1(prompt.encode("utf-8")).hexdigest() != row["prompt_sha1"]:
        raise ValueError(f"{row['id']}: prompt no longer matches the manifest's prompt_sha1; rerun python -m cvbench.splits")
    return Task(id=row["id"], kind="family", family=row["family"], subset=row["subset"], held_out=row["held_out"],
                role="task", seed=row["seed"], prompt=prompt, cluster=f"{row['family']}#{row['phrasing_index']}")


def _gotcha_task(row: dict, checks) -> Task:
    check = checks[row["id"]]
    return Task(id=row["id"], kind="gotcha", family=f"gotcha:{row['group']}", subset=row["subset"], held_out=False,
                role=row["role"], seed=None, prompt=check.prompt(), cluster=row["id"])


def load_tasks(splits=("test", "gotchas"), *, only: str | None = None, per_family: int | None = None,
               splits_dir: Path = SPLITS_DIR) -> list[Task]:
    """The tasks of the requested splits, in manifest order. ``only`` is a regex searched in the task id;
    ``per_family`` keeps the first K tasks of each family (a smoke test)."""
    unknown = set(splits) - set(SPLIT_CHOICES)
    if unknown:
        raise ValueError(f"unknown split(s) {sorted(unknown)}; choose from {SPLIT_CHOICES}")
    tasks: list[Task] = []
    for split in splits:
        rows = read_jsonl(splits_dir / f"{split}.jsonl")
        if not rows:
            raise FileNotFoundError(f"{splits_dir / (split + '.jsonl')} is missing or empty; run python -m cvbench.splits")
        if split == "gotchas":
            from cvbench.gotchas import CHECKS
            tasks += [_gotcha_task(r, CHECKS) for r in rows]
        else:
            tasks += [_family_task(r) for r in rows]
    if only:
        pat = re.compile(only)
        tasks = [t for t in tasks if pat.search(t.id)]
    if per_family:
        seen: dict[str, int] = {}
        kept = []
        for t in tasks:
            seen[t.family] = seen.get(t.family, 0) + 1
            if seen[t.family] <= per_family:
                kept.append(t)
        tasks = kept
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids across the requested splits")
    return tasks


def split_sizes(splits=("test", "gotchas"), splits_dir: Path = SPLITS_DIR) -> int:
    """Number of tasks in the splits, counted from the manifests without importing any family."""
    return sum(len(read_jsonl(splits_dir / f"{s}.jsonl")) for s in splits)
