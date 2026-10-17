"""The dev/test splits and the gotcha manifest: the files on disk, their seeds, counts, tags and prompt hashes."""
import dataclasses
import hashlib
import itertools
import json
from collections import Counter
from importlib import import_module

from cvbench import splits as sp
from cvbench.gotchas import CHECKS

# The family tags as specified, pinned here so that editing the table cannot silently change them.
EXPECTED_SUBSET = {
    "contour_area": "timeless", "hough_segments": "5.0", "chessboard_corners": "5.0",
    "perspective_rectify": "timeless", "feature_homography": "5.0", "histogram_stats": "5.0",
    "color_segmentation": "timeless", "components_stats": "5.0", "text_layout": "5.0",
}
EXPECTED_HELD_OUT = {"chessboard_corners", "color_segmentation"}

DEV, TEST = sp.build_splits()
GOTCHAS = sp.build_gotcha_manifest()


def _family(name):
    return import_module(f"cvbench.families.{name}")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_files_on_disk_equal_a_fresh_build():
    for name, rows in (("dev", DEV), ("test", TEST), ("gotchas", GOTCHAS)):
        path = sp.SPLITS_DIR / f"{name}.jsonl"
        assert path.is_file(), f"missing {path}; run: python -m cvbench.splits"
        assert _read_jsonl(path) == rows, f"{path.name} has drifted from build_splits(); rerun python -m cvbench.splits"


def test_seed_ranges_are_pinned_and_disjoint():
    assert sp.DEV_RANGE[0] == 3_100_000 and sp.TEST_RANGE[0] == 4_000_000 and sp.FOUNDRY_RANGE[0] == 5_000_000
    for (a0, a1), (b0, b1) in itertools.combinations((sp.DEV_RANGE, sp.TEST_RANGE, sp.FOUNDRY_RANGE), 2):
        assert a1 <= b0 or b1 <= a0


def test_no_scored_seed_was_used_to_calibrate_a_tolerance():
    # feature_homography calibrated its metamorphic tolerance on seeds [seed, seed + n); none may be a dev/test task.
    import inspect
    from cvbench.families import feature_homography
    defaults = inspect.signature(feature_homography.measure_metamorphic_drift).parameters
    lo, n = defaults["seed"].default, defaults["n"].default
    dev, test = sp.build_splits()
    assert not any(lo <= row["seed"] < lo + n for row in dev + test)


def test_dev_and_test_seeds_are_disjoint_unique_and_inside_their_ranges():
    dev_seeds = [t["seed"] for t in DEV]
    test_seeds = [t["seed"] for t in TEST]
    assert all(sp.DEV_RANGE[0] <= s < sp.DEV_RANGE[1] for s in dev_seeds)
    assert all(sp.TEST_RANGE[0] <= s < sp.TEST_RANGE[1] for s in test_seeds)
    assert len(set(dev_seeds)) == len(dev_seeds)
    assert len(set(test_seeds)) == len(test_seeds)
    assert not set(dev_seeds) & set(test_seeds)


def test_no_seed_is_in_the_foundry_range():
    lo, hi = sp.FOUNDRY_RANGE
    assert not [t["seed"] for t in DEV + TEST if lo <= t["seed"] < hi]


def test_counts_and_per_family_balance():
    assert set(sp.FAMILIES) == set(EXPECTED_SUBSET)
    assert len(DEV) == 9 * sp.DEV_PER_FAMILY == 99
    assert len(TEST) == 9 * sp.TEST_PER_FAMILY == 198
    for split, rows, per in (("dev", DEV, sp.DEV_PER_FAMILY), ("test", TEST, sp.TEST_PER_FAMILY)):
        assert all(r["split"] == split for r in rows)
        assert len({r["id"] for r in rows}) == len(rows)
        for name in EXPECTED_SUBSET:
            mine = [r for r in rows if r["family"] == name]
            assert len(mine) == per
            n_phrasings = len(_family(name).PHRASINGS)
            counts = Counter(r["phrasing_index"] for r in mine)
            assert set(counts) == set(range(n_phrasings)), f"{name}: every phrasing should be used"
            assert max(counts.values()) - min(counts.values()) <= 1


def test_held_out_families_are_present_in_dev_and_test_and_flagged():
    assert {n for n, row in sp.FAMILIES.items() if row.held_out} == EXPECTED_HELD_OUT
    for rows in (DEV, TEST):
        for name in EXPECTED_HELD_OUT:
            assert any(r["family"] == name for r in rows), f"{name} missing from a split"
        assert all(r["held_out"] == (r["family"] in EXPECTED_HELD_OUT) for r in rows)


def test_subset_tags_match_the_table():
    assert {n: row.subset for n, row in sp.FAMILIES.items()} == EXPECTED_SUBSET
    assert all(row.reason and "\n" not in row.reason for row in sp.FAMILIES.values())
    assert all(r["subset"] == EXPECTED_SUBSET[r["family"]] for r in DEV + TEST)


def test_every_task_scene_can_be_built():
    for r in DEV + TEST:
        fam = _family(r["family"])
        fam.inputs(fam.make_scene(r["seed"]))


def test_prompt_sha1_matches_recomputation():
    for r in DEV + TEST:
        fam = _family(r["family"])
        prompt = dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[r["phrasing_index"]]).to_prompt()
        assert hashlib.sha1(prompt.encode("utf-8")).hexdigest() == r["prompt_sha1"]


def test_gotcha_manifest_covers_every_check_once_in_test_only():
    ids = [r["id"] for r in GOTCHAS]
    assert len(ids) == len(set(ids)) == len(CHECKS)
    assert set(ids) == set(CHECKS)
    for r in GOTCHAS:
        c = CHECKS[r["id"]]
        assert (r["group"], r["subset"], r["role"]) == (c.group, c.subset, c.role)
        assert r["split"] == "test"
