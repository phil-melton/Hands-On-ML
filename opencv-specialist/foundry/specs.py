"""The task specs the teacher is asked about: the 200-spec pilot, the full generation, and the thinking-mode test.

A spec is one task the way cvbench poses it: a family, one of its phrasings that no scored task uses, and a scene seed.
Everything the teacher sees is derived from it:

    user message    ``TaskSpec.to_prompt()`` with the phrasing  (CLAUDE.md Rule 7: the training user message)
    training system ``train.config.SYSTEM``                     (what the student is trained and served with)
    teacher system  SYSTEM + "\\n\\n" + bakeoff/notes_5_0.md      (PLAN.md Phase 1: each teacher gets the 5.0 notes)

Phrasing indices of a family spec: 12-15 are the hand-written foundry phrasings (``cvbench.splits.foundry_phrasings``, never
scored); 16, 17, ... are the teacher-written ones in ``foundry/phrasings/<family>.json`` (``foundry.phrasings``, screened
against every dev/test and gotcha prompt). Dev and test phrasings (0-11) are never used.

Drill specs (``kind == "drill"``) come from ``foundry.drills`` (training-only tasks on measured 5.0 changes, written to be
disjoint from the gotcha suite): ``DRILLS`` maps a name to a ``cvbench.gotchas.core.Check`` and ``DRILL_PHRASINGS`` to its
three phrasings, index 0 being ``check.spec.task``; the phrasing index of a drill spec indexes that list. A drill has no
scene seed (its cases are fixed) and is graded with ``cvbench.gotchas.core.grade``: its "Tier A" is "the grade passed",
because a drill has no metamorphic gate; the lint and the contract check inside ``grade`` still apply.

Seeds. A family task is graded on scenes ``seed .. seed + N_SCENES - 1`` (cvbench.evaluate), so the seed windows must stay
disjoint from each other and from dev and test. cvbench.splits gives a family a block of SEED_BLOCK = 1_000 seeds, which
fits 25 tasks; a teacher run needs thousands, so the foundry gives each family a block of FOUNDRY_BLOCK = 100_000 seeds
inside FOUNDRY_RANGE = [5_000_000, 6_000_000), at the family's index in ``cvbench.splits.FAMILIES`` (the held-out
families' blocks stay reserved and unused), and steps by SCENE_STRIDE like the other splits:

    seed = FOUNDRY_RANGE[0] + FOUNDRY_BLOCK * family_index + SCENE_STRIDE * j      (j = the spec's index in its family)

Spec j of a family never changes with n, so ``pilot_specs(200)`` is a prefix of ``pilot_specs(2000)`` family by family (and the
same for ``full_specs``, whose targets only decide how many specs each family gets). Pilot balance: ``n // 7`` specs per
family, the remainder going one each to the first families in FAMILIES order; the phrasing of spec j is 12 + j % 4.

``full_specs`` sizes the run by a target number of kept (Tier-A) examples: a group (a family, or all drills together, with
``drill_weight`` family-equivalents) gets ``target * weight / total weight`` kept examples, and
``ceil(that / kept-per-spec rate)`` specs, the rate being what the group measured in the pilot (``PILOT_KEPT_PER_SPEC``, k = 4,
at most 2 kept per spec) or an estimate where k differs. Spec j takes the phrasing ``pool[j % len(pool)]`` of the family's
pool (the 4 foundry phrasings, then the teacher-written ones).

``check_specs`` enforces, for family specs: trainable family only, a foundry or screened teacher-written phrasing, the prompt
equals a fresh render, no prompt hash equals any dev, test or gotcha prompt hash, every seed window lies inside FOUNDRY_RANGE and
windows are pairwise disjoint; for drills: a known drill, a valid phrasing, the prompt equals a fresh render, no scored prompt
hash and no overlap with a scored task line (the phrasing screen's measure); for all: unique ids, the training system prompt
is ``SYSTEM``, the teacher's starts with it. The ``pilot_specs``, ``full_specs`` and ``think_test_specs`` builders run it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
from dataclasses import dataclass
from pathlib import Path

import bakeoff
from cvbench.splits import (FAMILIES, FOUNDRY_PHRASINGS, FOUNDRY_RANGE, N_SCENES, SCENE_STRIDE,
                            build_splits, foundry_phrasings)
from foundry import phrasings as phr
from train.config import SYSTEM

NOTES_PATH = Path(bakeoff.__file__).resolve().parent / "notes_5_0.md"
FOUNDRY_BLOCK = 100_000  # seeds per family inside FOUNDRY_RANGE
MAX_SPECS_PER_FAMILY = FOUNDRY_BLOCK // SCENE_STRIDE  # window of the last spec ends at SCENE_STRIDE * (m - 1) + N_SCENES
GENERATED_BASE = phr.GENERATED_BASE                   # phrasing index of a family's first teacher-written phrasing
DRILLS_GROUP = "drills"                               # the family name drill specs carry

assert FOUNDRY_BLOCK * len(FAMILIES) <= FOUNDRY_RANGE[1] - FOUNDRY_RANGE[0], "family blocks must fit in FOUNDRY_RANGE"
assert SCENE_STRIDE * (MAX_SPECS_PER_FAMILY - 1) + N_SCENES <= FOUNDRY_BLOCK

# Pilot (Qwen3.8-27B, non-thinking, k = 4, 200 specs, cap 2 per spec): kept examples per spec, by family
# (foundry/results/pilot-a-20261022/summary.json). Hough and text_layout are the families the teacher mostly fails: 25 % and 14 % of
# their samples reached Tier A, and 58.6 % and 39.3 % of their specs.
PILOT_KEPT_PER_SPEC = {"contour_area": 2.0, "hough_segments": 0.93, "perspective_rectify": 2.0, "feature_homography": 1.97,
                       "histogram_stats": 2.0, "components_stats": 2.0, "text_layout": 0.54}
# Rates the sizing uses by default. With k = 8 the two weak families keep more per spec; the estimate is the expected
# min(2, Binomial(8, p)) at the pilot's per-sample Tier-A rate (1.53 and 1.03), cut by about a fifth because specs differ in
# difficulty (the pilot solved 58.6 % and 39.3 % of its specs at k = 4, a homogeneous model says 68 % and 46 %). Drills have no
# pilot: 1.5 of the 2 the cap allows.
DEFAULT_RATES = {**PILOT_KEPT_PER_SPEC, "hough_segments": 1.3, "text_layout": 0.8, DRILLS_GROUP: 1.5}
DEFAULT_K = 4
K_BY_GROUP = {"hough_segments": 8, "text_layout": 8}   # the two families the teacher mostly fails get twice the samples
DRILL_PHRASINGS_EACH = 3


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


# ---------------------------------------------------------------- drills (foundry.drills, written by another agent)

def drills_module():
    try:
        return importlib.import_module("foundry.drills")
    except ModuleNotFoundError as e:
        if e.name != "foundry.drills":
            raise
        raise SpecError("foundry.drills is not available: drill specs need DRILLS and DRILL_PHRASINGS from it") from e


def drill_names() -> list[str]:
    """Drills that can yield a training target: those graded with the lint enforced (``verify.tier_of``)."""
    return sorted(name for name, check in drills_module().DRILLS.items() if check.lint)


def drill_task(name: str, phrasing_index: int) -> str:
    mod = drills_module()
    if name not in mod.DRILLS:
        raise SpecError(f"unknown drill {name!r}")
    lines = mod.DRILL_PHRASINGS[name]
    if not 0 <= phrasing_index < len(lines):
        raise SpecError(f"drill {name}: phrasing {phrasing_index} outside 0..{len(lines) - 1}")
    return lines[phrasing_index]


# ---------------------------------------------------------------- the spec

@dataclass(frozen=True)
class Spec:
    id: str
    family: str
    index: int            # j: the spec's index inside its family (or drill)
    phrasing_index: int   # family: 12-15 foundry, 16+ teacher-written; drill: index into its phrasings
    seed: int             # first scene seed; graded on seed .. seed + N_SCENES - 1 (drills: 0, unused)
    prompt: str           # the user message, TaskSpec.to_prompt()
    system: str           # the training system prompt (train.config.SYSTEM)
    teacher_system: str   # SYSTEM + the 5.0 notes
    kind: str = "family"  # "family" | "drill"
    drill: str = ""       # the drill's name when kind == "drill"
    k: int | None = None  # samples to ask for; None: the job's -k

    @property
    def prompt_sha1(self) -> str:
        return sha1(self.prompt)

    @property
    def window(self) -> tuple[int, int]:
        return (self.seed, self.seed + (N_SCENES if self.kind == "family" else 0))

    @property
    def subset(self) -> str:
        return FAMILIES[self.family].subset if self.kind == "family" else "drill"

    def messages(self) -> list[dict]:
        """What the teacher is asked."""
        return [{"role": "system", "content": self.teacher_system}, {"role": "user", "content": self.prompt}]

    def to_row(self) -> dict:
        row = {"id": self.id, "family": self.family, "index": self.index, "phrasing_index": self.phrasing_index,
               "seed": self.seed, "subset": self.subset, "prompt": self.prompt, "prompt_sha1": self.prompt_sha1,
               "system_sha1": sha1(self.system), "teacher_system_sha1": sha1(self.teacher_system)}
        if self.kind != "family":
            row.update(kind=self.kind, drill=self.drill)
        if self.k is not None:
            row["k"] = self.k
        return row

    @classmethod
    def from_row(cls, row: dict, *, system: str = SYSTEM, teacher: str = "") -> "Spec":
        """Rebuild from a specs.jsonl row. The system prompts are not stored in full (the hashes are); verify and report
        need only the id, family, seed and prompt."""
        return cls(id=row["id"], family=row["family"], index=row["index"], phrasing_index=row["phrasing_index"],
                   seed=row["seed"], prompt=row["prompt"], system=system, teacher_system=teacher,
                   kind=row.get("kind", "family"), drill=row.get("drill", ""), k=row.get("k"))


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


def family_task(family: str, phrasing_index: int) -> str:
    """The task line of a family phrasing index (12-15 foundry, 16+ teacher-written)."""
    if phrasing_index in FOUNDRY_PHRASINGS:
        return foundry_phrasings(family)[phrasing_index - FOUNDRY_PHRASINGS.start]
    gen = phr.generated_phrasings(family)
    if phrasing_index >= GENERATED_BASE and phrasing_index - GENERATED_BASE < len(gen):
        return gen[phrasing_index - GENERATED_BASE]
    raise SpecError(f"{family}: phrasing {phrasing_index} is neither a foundry phrasing {list(FOUNDRY_PHRASINGS)} nor one of the "
                    f"{len(gen)} teacher-written ones (indices {GENERATED_BASE}..{GENERATED_BASE + len(gen) - 1})")


def _render(family: str, phrasing_index: int) -> str:
    return dataclasses.replace(family_module(family).SPEC, task=family_task(family, phrasing_index)).to_prompt()


def _render_drill(name: str, phrasing_index: int) -> str:
    return dataclasses.replace(drills_module().DRILLS[name].spec, task=drill_task(name, phrasing_index)).to_prompt()


def phrasing_pool(family: str, use_generated: bool = True) -> list[int]:
    """Phrasing indices a family's specs cycle through: the foundry ones, then the teacher-written ones."""
    pool = list(FOUNDRY_PHRASINGS)
    if use_generated:
        pool += [GENERATED_BASE + i for i in range(len(phr.generated_phrasings(family)))]
    return pool


def _family_spec(family: str, j: int, phrasing_index: int, *, prefix: str, system: str, k: int | None = None) -> Spec:
    return Spec(id=f"{prefix}-{family}-{j:04d}", family=family, index=j, phrasing_index=phrasing_index, seed=spec_seed(family, j),
                prompt=_render(family, phrasing_index), system=SYSTEM, teacher_system=system, k=k)


def _drill_spec(name: str, j: int, phrasing_index: int, *, prefix: str, system: str, k: int | None = None) -> Spec:
    return Spec(id=f"{prefix}-{name}-{j:03d}", family=DRILLS_GROUP, index=j, phrasing_index=phrasing_index, seed=0,
                prompt=_render_drill(name, phrasing_index), system=SYSTEM, teacher_system=system, kind="drill", drill=name, k=k)


# ---------------------------------------------------------------- builders

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
            specs.append(_family_spec(family, j, i, prefix="foundry", system=system))
    check_specs(specs)
    return specs


def _group_k(group: str, k) -> int:
    if isinstance(k, dict):
        return int(k.get(group, K_BY_GROUP.get(group, DEFAULT_K)))
    return int(k) if k else K_BY_GROUP.get(group, DEFAULT_K)


def sizing(target_kept: int, *, rates: dict | None = None, drill_weight: float = 3.0, include_drills: bool = True,
           weights: dict | None = None) -> dict[str, dict]:
    """How many specs each group gets: {group: {"weight", "kept_target", "rate", "specs"}} (see the module docstring).
    ``weights``: {group: weight} overriding the default 1.0 per family (and ``drill_weight`` for the drills)."""
    if target_kept < 1:
        raise ValueError("target_kept must be at least 1")
    rate = {**DEFAULT_RATES, **(rates or {})}
    groups = {f: 1.0 for f in trainable_families()}
    if include_drills:
        groups[DRILLS_GROUP] = float(drill_weight)
    for g, w in (weights or {}).items():
        if g not in groups:
            raise ValueError(f"weight for unknown group {g!r}; groups: {sorted(groups)}")
        groups[g] = float(w)
    total = sum(groups.values())
    out = {}
    for g, w in groups.items():
        if rate[g] <= 0:
            raise ValueError(f"kept-per-spec rate of {g} must be positive, got {rate[g]}")
        share = target_kept * w / total
        out[g] = {"weight": w, "kept_target": share, "rate": rate[g], "specs": math.ceil(share / rate[g] - 1e-9)}
    return out


def full_specs(target_kept: int = 2100, *, rates: dict | None = None, k=None, drill_weight: float = 3.0,
               include_drills: bool = True, use_generated: bool = True, notes_text: str | None = None,
               prefix: str = "full", weights: dict | None = None) -> list[Spec]:
    """The specs of the full generation, sized to ``target_kept`` verified solutions (module docstring).

    ``rates``: kept examples per spec by group (families and "drills"), overriding DEFAULT_RATES. ``weights``: share weights by
    group, default 1.0 per family and ``drill_weight`` for the drills. ``k``: samples per spec, an int
    for all or {group: k}; default 4, and 8 for hough_segments and text_layout. Family specs cycle through the foundry and
    teacher-written phrasings (``use_generated=False``: the foundry ones only); drill specs through each drill's phrasings, the
    drills sharing their group's specs equally. For drills, "Tier A" means the drill's grade passed (no metamorphic gate).
    Needs foundry.drills when ``include_drills``."""
    plan = sizing(target_kept, rates=rates, drill_weight=drill_weight, include_drills=include_drills, weights=weights)
    system = teacher_system(notes_text)
    specs = []
    for family in trainable_families():
        n = plan[family]["specs"]
        if n > MAX_SPECS_PER_FAMILY:
            raise SpecError(f"{family}: {n} specs, at most {MAX_SPECS_PER_FAMILY} fit in a foundry seed block")
        pool = phrasing_pool(family, use_generated)
        for j in range(n):
            specs.append(_family_spec(family, j, pool[j % len(pool)], prefix=prefix, system=system, k=_group_k(family, k)))
    if include_drills:
        names = drill_names()
        if not names:
            raise SpecError("foundry.drills has no drills")
        base, extra = divmod(plan[DRILLS_GROUP]["specs"], len(names))
        for d, name in enumerate(names):
            lines = len(drills_module().DRILL_PHRASINGS[name])
            for j in range(base + (1 if d < extra else 0)):
                specs.append(_drill_spec(name, j, j % lines, prefix=prefix, system=system, k=_group_k(DRILLS_GROUP, k)))
    check_specs(specs)
    return specs


def think_test_specs(per_family: int = 8, drills: int = 16, k: int = 4, *, notes_text: str | None = None,
                     prefix: str = "tt", more: dict | None = None) -> list[Spec]:
    """A small spec set for the thinking-mode test: the first ``per_family`` specs of every trainable family (``more``: {family:
    count} for families that need more, such as the two the teacher mostly fails; foundry phrasings only, because the
    teacher-written ones do not exist before the phrasing job has run) and one spec of each of ``drills`` drills, spread evenly
    over the sorted drill names. Every spec gets ``k`` samples (so both modes are asked the same thing)."""
    system = teacher_system(notes_text)
    specs = []
    for family in trainable_families():
        pool = phrasing_pool(family, use_generated=False)
        specs += [_family_spec(family, j, pool[j % len(pool)], prefix=prefix, system=system, k=k)
                  for j in range((more or {}).get(family, per_family))]
    if drills:
        names = drill_names()   # SpecError when foundry.drills is missing: ask for drills=0 to leave them out
        picked = names if len(names) <= drills else [names[int(i * len(names) / drills)] for i in range(drills)]
        specs += [_drill_spec(n, 0, 0, prefix=prefix, system=system, k=k) for n in picked]
    check_specs(specs)
    return specs


def summarize_specs(specs: list[Spec]) -> list[dict]:
    """One row per group (family or drills): specs, distinct prompts, samples asked for."""
    out = []
    for g in dict.fromkeys(s.family for s in specs):
        mine = [s for s in specs if s.family == g]
        out.append({"group": g, "specs": len(mine), "prompts": len({s.prompt_sha1 for s in mine}),
                    "samples": sum(s.k or DEFAULT_K for s in mine), "k": sorted({s.k or DEFAULT_K for s in mine}),
                    "phrasings": sorted({s.phrasing_index for s in mine})})
    return out


# ---------------------------------------------------------------- invariants

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
        if s.system != SYSTEM:
            raise SpecError(f"{s.id}: the training system prompt differs from train.config.SYSTEM")
        if not s.teacher_system.startswith(SYSTEM + "\n\n"):
            raise SpecError(f"{s.id}: the teacher system prompt must start with SYSTEM")
        if s.kind == "drill":
            _check_drill(s, reserved)
            continue
        if s.kind != "family":
            raise SpecError(f"{s.id}: unknown spec kind {s.kind!r}")
        if s.family not in FAMILIES:
            raise SpecError(f"{s.id}: unknown family {s.family!r}")
        if FAMILIES[s.family].held_out:
            raise SpecError(f"{s.id}: {s.family} is a held-out family and may not be trained on")
        try:
            task = family_task(s.family, s.phrasing_index)
        except SpecError as e:
            raise SpecError(f"{s.id}: not a foundry phrasing: {e}") from None
        if s.prompt != _render(s.family, s.phrasing_index):
            raise SpecError(f"{s.id}: the prompt is not TaskSpec.to_prompt() of the family spec with that phrasing")
        if s.phrasing_index >= GENERATED_BASE:
            problems = phr.hard_problems(s.family, task)
            if problems:
                raise SpecError(f"{s.id}: teacher-written phrasing {s.phrasing_index} fails the screen: {problems[0]}")
        if s.prompt_sha1 in reserved:
            raise SpecError(f"{s.id}: its prompt is the prompt of scored task {reserved[s.prompt_sha1]}")
        lo, hi = s.window
        if not (FOUNDRY_RANGE[0] <= lo and hi <= FOUNDRY_RANGE[1]):
            raise SpecError(f"{s.id}: seed window [{lo}, {hi}) is outside FOUNDRY_RANGE {FOUNDRY_RANGE}")
        windows.append((lo, hi, s.id))
    windows.sort()
    for (lo1, hi1, id1), (lo2, hi2, id2) in zip(windows, windows[1:]):
        if lo2 < hi1:
            raise SpecError(f"seed windows of {id1} [{lo1}, {hi1}) and {id2} [{lo2}, {hi2}) overlap")


def _check_drill(s: Spec, reserved: dict[str, str]) -> None:
    mod = drills_module()
    if s.drill not in mod.DRILLS:
        raise SpecError(f"{s.id}: unknown drill {s.drill!r}")
    task = drill_task(s.drill, s.phrasing_index)
    if s.prompt != _render_drill(s.drill, s.phrasing_index):
        raise SpecError(f"{s.id}: the prompt is not TaskSpec.to_prompt() of the drill with that phrasing")
    if s.prompt_sha1 in reserved:
        raise SpecError(f"{s.id}: its prompt is the prompt of scored task {reserved[s.prompt_sha1]}")
    cand = phr.make_line("drill", task)
    for other in phr.scored_lines():
        why = phr.closeness(cand, other)
        if why:
            raise SpecError(f"{s.id}: drill {s.drill} overlaps a scored task ({why})")


def load_specs(path: Path) -> list[Spec]:
    from foundry.store import read_jsonl
    return [Spec.from_row(r) for r in read_jsonl(path)]
