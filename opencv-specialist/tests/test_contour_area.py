import math

import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import contour_area as fam

# Expected verdicts per gate: this is the demo's claim, pinned as a regression test.
EXPECTED = {
    #                           exec-only lint   manufactured metamorphic
    "reference":                (True,     True,  True,        True),
    "rgb2gray (harmless here)": (True,     True,  True,        True),
    "retr_tree":                (True,     True,  False,       True),
    "points_as_area":           (True,     True,  False,       True),
    "inverted_threshold":       (True,     True,  False,       True),
    "fixed_threshold_127":      (True,     True,  True,        False),
    "component_pixel_areas":    (True,     True,  False,       True),
    "unsorted":                 (True,     True,  False,       True),
    "opencv3_unpack":           (False,    True,  False,       False),
    "swallowed_opencv3_unpack": (True,     False, False,       True),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    return {name: evaluate_solution(fam, code, n_scenes=20, n_meta=5) for name, (_, code) in candidates.items()}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n not in ("reference", "rgb2gray (harmless here)")]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == 7
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


def test_reference_passes_many_manufactured_scenes_in_process():
    for seed in range(200, 400):
        scene = fam.make_scene(seed)
        v = fam.verify_manufactured(scene, fam.reference_solve(scene.img))
        assert v, (seed, v.reason)


def test_pixel_count_estimator_also_passes():
    """Both standard estimators sit inside the band: the tolerance encodes physics, not one API."""
    import cv2
    for seed in range(400, 450):
        scene = fam.make_scene(seed)
        gray = cv2.cvtColor(scene.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        areas = []
        for c in cs:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [c], -1, 255, -1)
            areas.append(float(cv2.countNonZero(filled)))
        out = {"areas": np.array(sorted(areas, reverse=True)), "count": len(cs)}
        v = fam.verify_manufactured(scene, out)
        assert v, (seed, v.reason)


def test_calibration_still_covers_fresh_gauge_blocks():
    summary = fam.calibrate(n=700, seed=999)
    assert summary["worst_abs"] < fam.TOL_C
    # and the physics: contourArea shaves inward on curved shapes, pixel counts bulge outward on rects
    assert summary["by_kind"]["disk/contourArea"][1] < 0
    assert summary["by_kind"]["rect/pixel_count"][0] > 0.45


def test_metamorphic_symmetries_are_exact_for_reference():
    assert fam.measure_metamorphic_drift(n=40, seed=7_000) == 0.0


def test_relative_tolerance_scales_as_one_over_r():
    def rel(r):
        s = fam.Shape("disk", math.pi * r * r, 2 * math.pi * r, (0, 0, 0, 0))
        return fam.tolerance(s) / s.area
    assert 8 < rel(5) / rel(50) < 11  # ~10x looser for a 10x smaller disk (TOL_C0 bends it slightly)


def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.img, b.img) and a.shapes == b.shapes
    kinds = set()
    for seed in range(60):
        s = fam.make_scene(seed)
        assert 3 <= len(s.shapes) <= 7
        kinds |= {sh.kind for sh in s.shapes}
        boxes = [sh.box for sh in s.shapes]
        for i, p in enumerate(boxes):
            for q in boxes[i + 1:]:
                assert p[2] < q[0] or q[2] < p[0] or p[3] < q[1] or q[3] < p[1], "shapes overlap"
    assert kinds == set(fam.KINDS)


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img) -> dict" in p and "OpenCV: 5.0" in p and "largest first" in p
