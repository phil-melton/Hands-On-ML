"""The task specs the teacher is asked about (Stage 4 pilot: 200 specs over the 7 trainable families).

A spec is one task the way cvbench poses it: a family, one of the family's *foundry* phrasings (indices 12-15, never
scored), and a scene seed. Everything the teacher sees is derived from it:

    user message    ``TaskSpec.to_prompt()`` with the foundry phrasing  (CLAUDE.md Rule 7: the training user message)
    training system ``train.config.SYSTEM``                            (what the student is trained and served with)
    teacher system  SYSTEM + "\\n\\n" + bakeoff/notes_5_0.md             (PLAN.md Phase 1: each teacher gets the 5.0 notes)

Seeds. A task is graded on scenes ``seed .. seed + N_SCENES - 1`` (cvbench.evaluate), so the seed windows must stay
disjoint from each other and from dev and test. cvbench.splits gives a family a block of SEED_BLOCK = 1_000 seeds, which
fits 25 tasks; a teacher run needs thousands, so the foundry gives each family a block of FOUNDRY_BLOCK = 100_000 seeds
inside FOUNDRY_RANGE = [5_000_000, 6_000_000), at the family's index in ``cvbench.splits.FAMILIES`` (the held-out
families' blocks stay reserved and unused), and steps by SCENE_STRIDE like the other splits:

    seed = FOUNDRY_RANGE[0] + FOUNDRY_BLOCK * family_index + SCENE_STRIDE * j      (j = the spec's index in its family)

Spec j of a family never changes with n, so ``pilot_specs(200)`` is a prefix of ``pilot_specs(2000)`` family by family.
Balance: ``n // 7`` specs per family, the remainder going one each to the first families in FAMILIES order (so 200 =
4 x 29 + 3 x 28). Phrasing for spec j is 12 + j % 4, so the four foundry phrasings are used equally often.

``check_specs`` enforces: trainable families only, foundry phrasings only, the prompt equals a fresh render, no prompt
hash equals any dev, test or gotcha prompt hash, every seed window lies inside FOUNDRY_RANGE, windows are pairwise
disjoint, ids are unique. ``pilot_specs`` runs it before returning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import bakeoff
from cvbench.splits import (FAMILIES, FOUNDRY_PHRASINGS, FOUNDRY_RANGE, N_SCENES, SCENE_STRIDE,
                            build_splits, foundry_phrasings)
from train.config import SYSTEM

NOTES_PATH = Path(bakeoff.__file__).resolve().parent / "notes_5_0.md"
FOUNDRY_BLOCK = 100_000  # seeds per family inside FOUNDRY_RANGE
MAX_SPECS_PER_FAMILY = FOUNDRY_BLOCK // SCENE_STRIDE  # window of the last spec ends at SCENE_STRIDE * (m - 1) + N_SCENES

assert FOUNDRY_BLOCK * len(FAMILIES) <= FOUNDRY_RANGE[1] - FOUNDRY_RANGE[0], "family blocks must fit in FOUNDRY_RANGE"
assert SCENE_STRIDE * (MAX_SPECS_PER_FAMILY - 1) + N_SCENES <= FOUNDRY_BLOCK


class SpecError(ValueError):
    """A spec violates an invariant (held-out family, reused prompt, overlapping or out-of-range seed window ...)."""


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def trainable_families() -> list[str]:
    """The families that may be trained on, in FAMILIES order (the held-out ones are never in a foundry spec)."""
    return [name for name, row in FAMILIES.items() if not row.held_out]


def family_module(name: str):
    return importlib.import_module(f"cvbench.families.{name}")


def teacher_system(notes_text: str | None = None) -> str:
    """The teacher's system message: the training SYSTEM, then the OpenCV 5.0 notes (same layout as the bake-off's
    notes rungs, bakeoff.rungs.system_prompt)."""
    notes = NOTES_PATH.read_text(encoding="utf-8") if notes_text is None else notes_text
    if not notes.strip():
        raise ValueError("the OpenCV 5.0 notes are empty")
    return SYSTEM + "\n\n" + notes.strip()


@dataclass(frozen=True)
class Spec:
    id: str
    family: str
    index: int            # j: the spec's index inside its family
    phrasing_index: int   # 12..15
    seed: int             # first scene seed; graded on seed .. seed + N_SCENES - 1
    prompt: str           # the user message, TaskSpec.to_prompt()
    system: str           # the training system prompt (train.config.SYSTEM)
    teacher_system: str   # SYSTEM + the 5.0 notes

    @property
    def prompt_sha1(self) -> str:
        return sha1(self.prompt)

    @property
    def window(self) -> tuple[int, int]:
        return (self.seed, self.seed + N_SCENES)

    @property
    def subset(self) -> str:
        return FAMILIES[self.family].subset

    def messages(self) -> list[dict]:
        """What the teacher is asked."""
        return [{"role": "system", "content": self.teacher_system}, {"role": "user", "content": self.prompt}]

    def to_row(self) -> dict:
        return {"id": self.id, "family": self.family, "index": self.index, "phrasing_index": self.phrasing_index,
                "seed": self.seed, "subset": self.subset, "prompt": self.prompt, "prompt_sha1": self.prompt_sha1,
                "system_sha1": sha1(self.system), "teacher_system_sha1": sha1(self.teacher_system)}

    @classmethod
    def from_row(cls, row: dict, *, system: str = SYSTEM, teacher: str = "") -> "Spec":
        """Rebuild from a specs.jsonl row. The system prompts are not stored in full (the hashes are); verify and report
        need only the id, family, seed and prompt."""
        return cls(id=row["id"], family=row["family"], index=row["index"], phrasing_index=row["phrasing_index"],
                   seed=row["seed"], prompt=row["prompt"], system=system, teacher_system=teacher)


def family_seed_base(family: str) -> int:
    return FOUNDRY_RANGE[0] + FOUNDRY_BLOCK * list(FAMILIES).index(family)


def spec_seed(family: str, j: int) -> int:
    if not 0 <= j < MAX_SPECS_PER_FAMILY:
        raise SpecError(f"spec index {j} outside 0..{MAX_SPECS_PER_FAMILY - 1}")
    return family_seed_base(family) + SCENE_STRIDE * j


def family_counts(n: int) -> dict[str, int]:
    fams = trainable_families()
    base, extra = divmod(n, len(fams))
    return {f: base + (1 if i < extra else 0) for i, f in enumerate(fams)}


def _render(family: str, phrasing_index: int) -> str:
    fam = family_module(family)
    task = foundry_phrasings(family)[phrasing_index - FOUNDRY_PHRASINGS.start]
    return dataclasses.replace(fam.SPEC, task=task).to_prompt()


def pilot_specs(n: int = 200, *, notes_text: str | None = None) -> list[Spec]:
    """``n`` specs, balanced over the trainable families, ordered family by family (FAMILIES order), then by index."""
    if n < 1:
        raise ValueError("n must be at least 1")
    system = teacher_system(notes_text)
    specs = []
    for family, count in family_counts(n).items():
        if count > MAX_SPECS_PER_FAMILY:
            raise SpecError(f"{family}: {count} specs, at most {MAX_SPECS_PER_FAMILY} fit in a foundry seed block")
        for j in range(count):
            i = FOUNDRY_PHRASINGS.start + j % len(FOUNDRY_PHRASINGS)
            specs.append(Spec(id=f"foundry-{family}-{j:04d}", family=family, index=j, phrasing_index=i,
                              seed=spec_seed(family, j), prompt=_render(family, i), system=SYSTEM,
                              teacher_system=system))
    check_specs(specs)
    return specs


def reserved_prompt_hashes() -> dict[str, str]:
    """sha1 of every prompt a scored task uses -> the task id: the 99 dev and 198 test family tasks and the gotcha
    checks. No foundry prompt may be one of them."""
    from cvbench.gotchas import CHECKS

    dev, test = build_splits()
    out = {r["prompt_sha1"]: r["id"] for r in dev + test}
    out.update({sha1(c.prompt()): cid for cid, c in CHECKS.items()})
    return out


def check_specs(specs: list[Spec], *, reserved: dict[str, str] | None = None) -> None:
    """Raise SpecError when an invariant fails (see the module docstring)."""
    reserved = reserved_prompt_hashes() if reserved is None else reserved
    ids, windows = set(), []
    for s in specs:
        if s.id in ids:
            raise SpecError(f"duplicate spec id {s.id}")
        ids.add(s.id)
        if s.family not in FAMILIES:
            raise SpecError(f"{s.id}: unknown family {s.family!r}")
        if FAMILIES[s.family].held_out:
            raise SpecError(f"{s.id}: {s.family} is a held-out family and may not be trained on")
        if s.phrasing_index not in FOUNDRY_PHRASINGS:
            raise SpecError(f"{s.id}: phrasing {s.phrasing_index} is not a foundry phrasing {list(FOUNDRY_PHRASINGS)}")
        if s.prompt != _render(s.family, s.phrasing_index):
            raise SpecError(f"{s.id}: the prompt is not TaskSpec.to_prompt() of the family spec with that phrasing")
        if s.prompt_sha1 in reserved:
            raise SpecError(f"{s.id}: its prompt is the prompt of scored task {reserved[s.prompt_sha1]}")
        if s.system != SYSTEM:
            raise SpecError(f"{s.id}: the training system prompt differs from train.config.SYSTEM")
        if not s.teacher_system.startswith(SYSTEM + "\n\n"):
            raise SpecError(f"{s.id}: the teacher system prompt must start with SYSTEM")
        lo, hi = s.window
        if not (FOUNDRY_RANGE[0] <= lo and hi <= FOUNDRY_RANGE[1]):
            raise SpecError(f"{s.id}: seed window [{lo}, {hi}) is outside FOUNDRY_RANGE {FOUNDRY_RANGE}")
        windows.append((lo, hi, s.id))
    windows.sort()
    for (lo1, hi1, id1), (lo2, hi2, id2) in zip(windows, windows[1:]):
        if lo2 < hi1:
            raise SpecError(f"seed windows of {id1} [{lo1}, {hi1}) and {id2} [{lo2}, {hi2}) overlap")


def load_specs(path: Path) -> list[Spec]:
    from foundry.store import read_jsonl
    return [Spec.from_row(r) for r in read_jsonl(path)]
