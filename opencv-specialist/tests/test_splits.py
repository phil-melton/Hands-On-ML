"""The dev/test splits and the gotcha manifest: the files on disk, their seeds, counts, tags and prompt hashes.

Also pins how prompts are shared out: 16 phrasings per family, disjoint between dev (0-3), test (4-11) and foundry
(12-15), and the wording every prompt ends with.
"""
import dataclasses
import hashlib
import itertools
import json
import re
from collections import Counter
from importlib import import_module

from cvbench import splits as sp
from cvbench.gotchas import CHECKS
from train.config import SYSTEM, modelfile

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


def _prompt(name, index):
    fam = _family(name)
    return dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[index]).to_prompt()


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
    # every graded scene of a task, not just its first seed
    assert not any(lo < row["seed"] + sp.N_SCENES and row["seed"] < lo + n for row in dev + test)


def test_scene_windows_are_disjoint_and_match_the_grader():
    # A task is graded on scenes seed .. seed + n_scenes - 1 (cvbench.evaluate); no two tasks may share a scene.
    import inspect
    from cvbench.evaluate import evaluate_solution
    assert sp.N_SCENES == inspect.signature(evaluate_solution).parameters["n_scenes"].default
    windows = sorted((r["seed"], r["seed"] + sp.N_SCENES) for r in DEV + TEST)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in itertools.pairwise(windows))


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
    for split, rows, per, phrasings in (("dev", DEV, sp.DEV_PER_FAMILY, range(0, 4)),
                                        ("test", TEST, sp.TEST_PER_FAMILY, range(4, 12))):
        assert all(r["split"] == split for r in rows)
        assert len({r["id"] for r in rows}) == len(rows)
        for name in EXPECTED_SUBSET:
            mine = [r for r in rows if r["family"] == name]
            assert len(mine) == per
            counts = Counter(r["phrasing_index"] for r in mine)
            assert set(counts) == set(phrasings), f"{name}: every phrasing of the {split} split should be used, and only those"
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
        assert _sha1(_prompt(r["family"], r["phrasing_index"])) == r["prompt_sha1"]


# ---------------------------------------------------------------- phrasings: 16 per family, disjoint between splits

def test_phrasing_ranges_are_pinned_and_tile_the_sixteen_phrasings():
    assert (sp.PHRASINGS_PER_FAMILY, sp.DEV_PHRASINGS, sp.TEST_PHRASINGS, sp.FOUNDRY_PHRASINGS) == \
        (16, range(0, 4), range(4, 12), range(12, 16))
    assert sorted(itertools.chain(sp.DEV_PHRASINGS, sp.TEST_PHRASINGS, sp.FOUNDRY_PHRASINGS)) == list(range(16))


def test_every_family_has_sixteen_distinct_phrasings_and_prompts():
    for name in EXPECTED_SUBSET:
        fam = _family(name)
        assert len(fam.PHRASINGS) == sp.PHRASINGS_PER_FAMILY == 16, name
        assert fam.PHRASINGS[0] == fam.SPEC.task, name
        assert len(set(fam.PHRASINGS)) == 16, name
        assert len({_sha1(_prompt(name, i)) for i in range(16)}) == 16, name


def test_dev_and_test_cycle_their_own_phrasings():
    # dev task j uses phrasing j % 4, test task j uses 4 + j % 8
    for rows, offset, width in ((DEV, 0, 4), (TEST, 4, 8)):
        for name in EXPECTED_SUBSET:
            mine = [r for r in rows if r["family"] == name]
            assert [r["phrasing_index"] for r in mine] == [offset + j % width for j in range(len(mine))], name


def test_dev_and_test_share_no_prompt():
    dev_hashes = {r["prompt_sha1"] for r in DEV}
    test_hashes = {r["prompt_sha1"] for r in TEST}
    assert not dev_hashes & test_hashes
    for name in EXPECTED_SUBSET:  # also per family, so a shared prompt cannot hide behind a different family's
        assert not {r["prompt_sha1"] for r in DEV if r["family"] == name} & \
            {r["prompt_sha1"] for r in TEST if r["family"] == name}, name


def test_distinct_prompt_counts_per_split():
    for rows, per_family in ((DEV, 4), (TEST, 8)):
        for name in EXPECTED_SUBSET:
            assert len({r["prompt_sha1"] for r in rows if r["family"] == name}) == per_family, name
    assert len({r["prompt_sha1"] for r in DEV}) == 36
    assert len({r["prompt_sha1"] for r in TEST}) == 72


def test_foundry_phrasings_are_reserved_and_never_scored():
    scored = {r["prompt_sha1"] for r in DEV + TEST}
    for name in EXPECTED_SUBSET:
        fam = _family(name)
        foundry = sp.foundry_phrasings(name)
        assert foundry == fam.PHRASINGS[12:16] and len(foundry) == 4, name
        for index in range(12, 16):
            assert _sha1(_prompt(name, index)) not in scored, (name, index)
        assert not {r["phrasing_index"] for r in DEV + TEST if r["family"] == name} & set(range(12, 16)), name


def _words(text):
    return set(re.findall(r"[a-z0-9_.\[\]]+", text.lower()))


def test_phrasings_are_not_near_duplicates():
    # Word-set Jaccard similarity of two phrasings, after removing the contract text that is shared on purpose.
    # Measured over the 16 phrasings of each family when they were written: at most 0.62. Reworded copies score above 0.8.
    for name in EXPECTED_SUBSET:
        fam = _family(name)
        shared = [getattr(fam, attr, "") for attr in ("ACCURACY", "_RANGES", "_BOUNDS")]
        clean = []
        for ph in fam.PHRASINGS:
            for s in shared:
                if s:
                    ph = ph.replace(s, "")
            clean.append(_words(ph))
        for (i, a), (j, b) in itertools.combinations(enumerate(clean), 2):
            assert len(a & b) / len(a | b) < 0.8, (name, i, j)


# Names that would turn a task statement into a recipe, or leak how the task is graded. SPEC.task may mention the
# cv2 constants and functions the contract is built on; no other phrasing may add one.
HINT_WORDS = ("ransac", "sift", "orb", "akaze", "brisk", "hough", "houghlinesp", "findcontours", "contourarea", "inrange",
              "calchist", "connectedcomponents", "warpperspective", "getperspectivetransform", "findhomography",
              "approxpolydp", "cornersubpix", "findchessboardcorners", "tolerance", "gate", "seed", "benchmark",
              "metamorphic", "version", "deprecated", "4.14", "countnonzero", "minarearect", "boundingrect")


def test_phrasings_add_no_hints():
    for name in EXPECTED_SUBSET:
        fam = _family(name)
        spec_api = set(re.findall(r"cv2\.\w+", fam.SPEC.task))
        for i, ph in enumerate(fam.PHRASINGS):
            assert set(re.findall(r"cv2\.\w+", ph)) <= spec_api, (name, i)
            found = [w for w in HINT_WORDS if re.search(rf"\b{re.escape(w)}\b", ph.lower())]
            assert not found, (name, i, found)


# ---------------------------------------------------------------- the wording every prompt ends with

CLOSING = ("Write one function, `{signature}`, with the imports it needs. No file, GUI or network access; "
           "do not catch exceptions to hide failures.")


def test_every_prompt_ends_by_asking_for_the_imports():
    # Stage 3 finding: "Write one function" was taken literally and many replies left out `import cv2` / `import numpy as np`.
    for name in EXPECTED_SUBSET:
        fam = _family(name)
        for i in range(sp.PHRASINGS_PER_FAMILY):
            assert _prompt(name, i).splitlines()[-1] == CLOSING.format(signature=fam.SPEC.signature), (name, i)
    for cid, c in CHECKS.items():  # the gotcha prompts render through the same method
        assert c.prompt().splitlines()[-1] == CLOSING.format(signature=c.spec.signature), cid


def test_system_prompt_asks_for_the_imports_and_is_the_modelfile_system():
    assert "return one Python function `solve(...) -> dict`, with the imports it needs, that satisfies it." in SYSTEM
    assert f'SYSTEM """{SYSTEM}"""' in modelfile("qwen2.5-coder-7b", "x.gguf")  # Rule 7: one source for both


def test_gotcha_manifest_covers_every_check_once_in_test_only():
    ids = [r["id"] for r in GOTCHAS]
    assert len(ids) == len(set(ids)) == len(CHECKS)
    assert set(ids) == set(CHECKS)
    for r in GOTCHAS:
        c = CHECKS[r["id"]]
        assert (r["group"], r["subset"], r["role"]) == (c.group, c.subset, c.role)
        assert r["split"] == "test"
