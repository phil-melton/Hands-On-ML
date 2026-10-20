"""Dev and test splits for cvbench: which tasks are scored, with which seeds and prompt phrasings.

Seed ranges (one million seeds each, disjoint):

    DEV      [3_100_000, 4_000_000)   development: iterate prompts and models here
    TEST     [4_000_000, 5_000_000)   held out: score a model here, do not tune on it
    FOUNDRY  [5_000_000, 6_000_000)   reserved for Stage 4 training-data generation; never used here

Each family owns a block of SEED_BLOCK seeds inside its split's range, so seed = start + SEED_BLOCK * f + SCENE_STRIDE * j
(f = family index, j = task index). Task j is graded on scenes seed .. seed + N_SCENES - 1 (cvbench.evaluate), so the
scene windows are pairwise disjoint within and across splits. 2026-10-20: the stride was 1 until then, so adjacent tasks
shared 19 of their 20 scenes and a family's 22 test tasks covered only 41 distinct scenes; tasks are meant to be independent draws.

Seed audit, grepped over cvbench/ and tests/ when the ranges were fixed: exactly one literal falls in
[3_000_000, 6_000_000), the default seed=3_000_000 (n=9_000, i.e. seeds 3_000_000..3_008_999) of
feature_homography.measure_metamorphic_drift. It is a calibration seed and was left alone. It overlapped the
first DEV seeds when DEV started at 3_000_000, so DEV now starts at 3_100_000 (2026-10-17): no scene a tolerance
was calibrated on is a dev or test task.

``build_splits()`` is deterministic. ``write_manifests()`` writes data/splits/{dev,test,gotchas}.jsonl, and
``python -m cvbench.splits`` calls it. tests/test_splits.py fails if the files drift from a fresh build.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FamilyRow:
    held_out: bool  # True: dev and test only, never training data
    subset: str     # "5.0": the answer depends on a measured 4.14 -> 5.0 change; "timeless": it does not
    reason: str     # one line


FAMILIES: dict[str, FamilyRow] = {
    "contour_area": FamilyRow(
        held_out=False, subset="timeless",
        reason="findContours is measured unchanged 4.14 -> 5.0 (control shape_contours_ctrl); no README exposure."),
    "hough_segments": FamilyRow(
        held_out=False, subset="5.0",
        reason="HoughLinesP returns (N, 4) in 5.0 instead of (N, 1, 4)."),
    "chessboard_corners": FamilyRow(
        held_out=True, subset="5.0",
        reason="findChessboardCorners returns (N, 2) in 5.0 instead of (N, 1, 2)."),
    "perspective_rectify": FamilyRow(
        held_out=False, subset="timeless",
        reason="warpPerspective moves at most 4 grey levels on ~3 % of pixels, inside the image tolerance TOL_IMG."),
    "feature_homography": FamilyRow(
        held_out=False, subset="5.0",
        reason="AKAZE, KAZE, BRISK and AGAST are gone from the 5.0 wheel; the answer depends on the detectors left."),
    "histogram_stats": FamilyRow(
        held_out=False, subset="5.0",
        reason="calcHist returns (256,) in 5.0 instead of (256, 1)."),
    "color_segmentation": FamilyRow(
        held_out=True, subset="timeless",
        reason="The family README records no 4.14 -> 5.0 exposure for HSV inRange; its traps are bounds and hue wrap."),
    "components_stats": FamilyRow(
        held_out=False, subset="5.0",
        reason="5.0 accepts bool masks in some calls, but connectedComponents* still require uint8."),
    "text_layout": FamilyRow(
        held_out=False, subset="5.0",
        reason="getTextSize and glyph pixels changed in 5.0 (line height, baseline, about 13 % of pixels)."),
}

DEV_RANGE = (3_100_000, 4_000_000)  # starts past feature_homography's drift-calibration seeds (3_000_000..3_008_999)
TEST_RANGE = (4_000_000, 5_000_000)
FOUNDRY_RANGE = (5_000_000, 6_000_000)
SEED_BLOCK = 1_000  # seeds per family inside a split's range
SCENE_STRIDE = 40   # seed gap between consecutive tasks of a family: at least the grader's n_scenes = 20, with headroom
N_SCENES = 20       # cvbench.evaluate.evaluate_solution's default n_scenes; tests pin it to the grader's default

DEV_PER_FAMILY = 11   # 9 families -> 99 dev tasks
TEST_PER_FAMILY = 22  # 9 families -> 198 test tasks

assert SCENE_STRIDE >= N_SCENES, "consecutive tasks would share scenes"
assert SCENE_STRIDE * (TEST_PER_FAMILY - 1) + N_SCENES <= SEED_BLOCK, "a family's scene windows must fit inside its seed block"

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"


def _family(name: str):
    return importlib.import_module(f"cvbench.families.{name}")


def _prompt(fam, phrasing_index: int) -> str:
    return dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[phrasing_index]).to_prompt()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _split_rows(split: str, start: int, per_family: int) -> list[dict]:
    rows = []
    for f, name in enumerate(FAMILIES):
        fam = _family(name)
        row = FAMILIES[name]
        for j in range(per_family):
            i = j % len(fam.PHRASINGS)
            rows.append({
                "id": f"{split}-{name}-{j:03d}",
                "split": split,
                "family": name,
                "phrasing_index": i,
                "seed": start + SEED_BLOCK * f + SCENE_STRIDE * j,
                "held_out": row.held_out,
                "subset": row.subset,
                "prompt_sha1": _sha1(_prompt(fam, i)),
            })
    return rows


def build_splits() -> tuple[list[dict], list[dict]]:
    """(dev, test): one row per task, families in FAMILIES order."""
    dev = _split_rows("dev", DEV_RANGE[0], DEV_PER_FAMILY)
    test = _split_rows("test", TEST_RANGE[0], TEST_PER_FAMILY)
    return dev, test


def build_gotcha_manifest() -> list[dict]:
    """One row per gotcha-suite check, in catalogue order. Gotcha cases are never training data: all TEST."""
    from cvbench.gotchas import CHECKS

    return [{"id": c.id, "group": c.group, "subset": c.subset, "role": c.role, "split": "test"}
            for c in CHECKS.values()]


def write_manifests(out_dir: Path = SPLITS_DIR) -> dict[str, Path]:
    """Write dev.jsonl, test.jsonl and gotchas.jsonl (one JSON object per line) under out_dir."""
    dev, test = build_splits()
    tables = {"dev": dev, "test": test, "gotchas": build_gotcha_manifest()}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in tables.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in write_manifests().items():
        print(f"{name}: {path}")
