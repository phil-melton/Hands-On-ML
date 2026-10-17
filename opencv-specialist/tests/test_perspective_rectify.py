import math
from functools import lru_cache

import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import perspective_rectify as fam


class _Family:
    """The family with make_scene memoised: evaluate_solution re-renders the same 20 scenes for every candidate
    and rendering dominates the run time. Scenes are never mutated (inputs go through np.savez)."""
    make_scene = staticmethod(lru_cache(maxsize=None)(fam.make_scene))
    inputs = staticmethod(fam.inputs)
    verify_manufactured = staticmethod(fam.verify_manufactured)
    transforms = staticmethod(fam.transforms)
    verify_metamorphic = staticmethod(fam.verify_metamorphic)


# Expected verdicts per gate: this is the demo's claim, pinned as a regression test.
EXPECTED = {
    #                                exec-only lint   manufactured metamorphic
    "reference":                     (True,     True,  True,        True),
    "float64_points":                (False,    True,  False,       False),
    "int32_points":                  (False,    True,  False,       False),
    "dsize_hw":                      (True,     True,  False,       False),
    "contour_order":                 (True,     True,  False,       False),
    "first_four_of_polygon":         (True,     True,  False,       False),
    # Invariant under every symmetry (a channel swap commutes with shift, rotate, mirror and dim): only the
    # manufactured texture can see it.
    "bgr_to_rgb_output":             (True,     True,  False,       True),
    # Right on every base photo; wrong on the dimmed one: only the metamorphic gate sees it.
    "fixed_threshold_127":           (True,     True,  True,        False),
    # A consistent offset is covariant: blind spot of the metamorphic gate, closed by the manufactured one.
    "min_area_rect":                 (True,     True,  False,       True),
    # exec-only keeps it; lint rejects it; the zero warp it returns is covariant, so metamorphic misses it.
    "swallowed_float64":             (True,     False, False,       True),
    # True negatives: legitimate variations the tolerance must not punish.
    "inter_nearest (harmless)":      (True,     True,  True,        True),
    "dst_wh_convention (harmless)":  (True,     True,  True,        True),
    "extremes_of_contour (harmless)": (True,    True,  True,        True),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")
HARMLESS = [n for n in EXPECTED if "harmless" in n]
WRONG = [n for n in EXPECTED if n != "reference" and n not in HARMLESS]


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    return {name: evaluate_solution(_Family, code, n_scenes=20, n_meta=5) for name, (_, code) in candidates.items()}


def test_expected_matrix_covers_exactly_the_mutants():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_exec_only_misses_most_wrong_solutions(matrix):
    assert len(WRONG) == 9
    assert sum(matrix[n]["exec-only"].passed for n in WRONG) == 7
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in WRONG)


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


def test_failure_reasons_name_the_defect(matrix):
    assert "shape" in matrix["dsize_hw"]["manufactured"].detail
    assert "wrong order" in matrix["contour_order"]["manufactured"].detail
    assert "NRMSE" in matrix["bgr_to_rgb_output"]["manufactured"].detail
    assert "checkVector" in matrix["float64_points"]["exec-only"].detail
    assert "checkVector" in matrix["int32_points"]["exec-only"].detail
    assert "empty" in matrix["fixed_threshold_127"]["metamorphic"].detail
    assert "px from the manufactured" in matrix["min_area_rect"]["manufactured"].detail
    assert "broad-except" in matrix["swallowed_float64"]["lint"].detail


# ---------------------------------------------------------------- scenes

def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.img, b.img) and np.array_equal(a.corners, b.corners)
    assert (a.out_w, a.out_h, a.blur, a.noise, a.card) == (b.out_w, b.out_h, b.blur, b.noise, b.card)
    assert np.array_equal(a.H_ci, b.H_ci) and np.array_equal(a.tex["freq"], b.tex["freq"])
    assert not np.array_equal(a.img, fam.make_scene(4).img)
    turns, tilts = set(), []
    for seed in range(40):
        s = fam.make_scene(seed)
        assert s.img.shape == (fam.IMG_H, fam.IMG_W, 3) and s.img.dtype == np.uint8
        assert s.corners.shape == (4, 2)
        assert s.corners.min() >= fam.BORDER - 1e-9
        assert s.corners[:, 0].max() <= fam.IMG_W - 1 - fam.BORDER + 1e-9
        assert s.corners[:, 1].max() <= fam.IMG_H - 1 - fam.BORDER + 1e-9
        idx, gap = fam.position_order(s.corners)
        assert idx == [0, 1, 2, 3] and gap >= fam.GAP, "the position rule must be unambiguous"
        ang = fam.interior_angles(s.corners)
        assert fam.ANGLE_RANGE[0] <= min(ang) and max(ang) <= fam.ANGLE_RANGE[1]
        assert abs(s.out_w - s.out_h) >= 24
        assert fam.BLUR_RANGE[0] <= s.blur <= fam.BLUR_RANGE[1] and fam.NOISE_RANGE[0] <= s.noise <= fam.NOISE_RANGE[1]
        turns.add(round(s.roll_deg / 90) % 4)
        tilts.append(s.tilt_deg)
        inp = fam.inputs(s)
        assert set(inp) == {"img", "out_w", "out_h"} and isinstance(inp["out_w"], int)
        assert seed >= 8 or fam.truth_warp(s).shape == (s.out_h, s.out_w, 3)
    assert turns == {0, 1, 2, 3}, "the card must appear in every quarter turn"
    assert max(tilts) > 30 and min(tilts) < 10


def test_label_margin_exceeds_what_the_corner_tolerance_can_flip():
    worst = fam.TOL_A + fam.TOL_B * fam.BLUR_RANGE[1]
    assert fam.GAP >= 2 * math.sqrt(2) * worst


def test_manufactured_texture_is_what_the_photo_shows():
    """Rectifying the (nearly sharp, noiseless) photo with the true corners reproduces the manufactured image."""
    for seed in range(10):
        s = fam.make_scene(seed, blur=0.6, noise=0.0)
        warped = fam.warp_with(s.img, s.corners, s.out_w, s.out_h)
        assert fam.nrmse(warped, fam.truth_warp(s)) < 0.08, seed


def test_scenes_tell_orientation_apart():
    """The texture is asymmetric: mirrored, rotated and channel-swapped warps are far from the truth."""
    for seed in range(10):
        s = fam.make_scene(seed)
        t = fam.truth_warp(s)
        assert fam.nrmse(t[:, ::-1], t) > 0.5 and fam.nrmse(t[::-1], t) > 0.5 and fam.nrmse(t[..., ::-1], t) > 0.5


def test_cv2_5_0_warp_equals_an_independent_bilinear_within_rounding():
    """On the smooth manufactured texture the 5.0 warpPerspective is a plain bilinear sampler to half a gray level,
    so a verifier that scores the texture does not depend on which warp implementation produced the pixels."""
    s = fam.make_scene(5, noise=0.0)
    ow, oh = s.out_w, s.out_h
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    M = cv2.getPerspectiveTransform(s.corners.astype(np.float32), dst)
    got = cv2.warpPerspective(s.img, M, (ow, oh), flags=cv2.INTER_LINEAR)
    xs, ys = np.meshgrid(np.arange(ow), np.arange(oh))
    p = np.linalg.inv(M.astype(np.float64)) @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    sx, sy = p[0] / p[2], p[1] / p[2]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = (sx - x0)[:, None], (sy - y0)[:, None]
    img = s.img.astype(np.float64)
    want = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy).reshape(oh, ow, 3)
    assert np.abs(got - want).max() <= 0.6  # 0.50 measured: the uint8 rounding
    assert fam.nrmse(got, want) < 0.01 < fam.TOL_IMG


# ---------------------------------------------------------------- verifier behaviour

@pytest.fixture(scope="module")
def scene():
    return fam.make_scene(11)


def _good(scene):
    return {"corners": scene.corners.astype(np.float32),
            "warped": fam.warp_with(scene.img, scene.corners, scene.out_w, scene.out_h)}


def test_manufactured_accepts_the_truth_and_the_reference(scene):
    assert fam.verify_manufactured(scene, _good(scene))
    assert fam.verify_manufactured(scene, fam.reference_solve(**fam.inputs(scene)))


def test_manufactured_rejects_each_defect(scene):
    good = _good(scene)
    cases = {
        "order": ({**good, "corners": np.roll(good["corners"], 1, axis=0)}, "wrong order"),
        "swapped xy": ({**good, "corners": good["corners"][:, ::-1].copy()}, "px from the manufactured"),
        "offset": ({**good, "corners": good["corners"] + np.float32([15, 0])}, "px from the manufactured"),
        "four-point polygon with a middle axis": ({**good, "corners": good["corners"].reshape(4, 1, 2)}, "ndim"),
        "integer corners": ({**good, "corners": good["corners"].astype(np.int32)}, "float"),
        "five points": ({**good, "corners": np.vstack([good["corners"], good["corners"][:1]])}, "shape"),
        "transposed canvas": ({**good, "warped": np.ascontiguousarray(good["warped"].transpose(1, 0, 2))}, "shape"),
        "grayscale": ({**good, "warped": good["warped"][..., 0]}, "ndim"),
        "float image": ({**good, "warped": good["warped"].astype(np.float32)}, "uint8"),
        "channels swapped": ({**good, "warped": good["warped"][..., ::-1].copy()}, "NRMSE"),
        "mirrored": ({**good, "warped": good["warped"][:, ::-1].copy()}, "NRMSE"),
        "zero image": ({**good, "warped": np.zeros_like(good["warped"])}, "NRMSE"),
        "missing key": ({"corners": good["corners"]}, "missing"),
    }
    for name, (out, word) in cases.items():
        v = fam.verify_manufactured(scene, out)
        assert not v and word in v.reason, (name, v.reason)


def test_tolerance_is_judged_on_corners_once_and_the_warp_against_its_own_corners(scene):
    ow, oh = scene.out_w, scene.out_h
    tol = fam.corner_tolerance(scene)
    inside = scene.corners + np.array([0.9 * tol, 0.0])  # all four corners off by 0.9 tolerance
    out = {"corners": inside.astype(np.float32), "warped": fam.warp_with(scene.img, inside, ow, oh)}
    assert fam.verify_manufactured(scene, out), "a warp made from the reported corners must not pay the corner error twice"
    outside = scene.corners + np.array([1.1 * tol, 0.0])
    out = {"corners": outside.astype(np.float32), "warped": fam.warp_with(scene.img, outside, ow, oh)}
    assert not fam.verify_manufactured(scene, out)
    # right corners reported, but the image was warped from corners 7 px away: inconsistent
    skew = scene.corners + np.array([7.0, 7.0])
    out = {"corners": scene.corners.astype(np.float32), "warped": fam.warp_with(scene.img, skew, ow, oh)}
    v = fam.verify_manufactured(scene, out)
    assert not v and "NRMSE" in v.reason


def test_corner_tolerance_follows_the_blur_law():
    a, b = fam.make_scene(1, blur=0.6), fam.make_scene(1, blur=2.2)
    assert fam.corner_tolerance(b) - fam.corner_tolerance(a) == pytest.approx(fam.TOL_B * 1.6)
    assert fam.corner_tolerance(a) == pytest.approx(fam.TOL_A + fam.TOL_B * 0.6)


def test_metamorphic_rejects_a_stuck_gauge_and_accepts_a_covariant_estimator(scene):
    good = _good(scene)
    stuck = {n: good for n in fam.transforms(scene)}
    v = fam.verify_metamorphic(good, stuck)
    assert not v and "corner set" in v.reason
    variants = fam.transforms(scene)
    outs = {n: fam.linefit_solve(**kw) for n, kw in variants.items()}
    assert fam.verify_metamorphic(fam.linefit_solve(**fam.inputs(scene)), outs)
    # same corners, image not following the symmetry
    outs = {n: dict(o) for n, o in outs.items()}
    outs[fam.T_MIRROR]["warped"] = np.ascontiguousarray(outs[fam.T_MIRROR]["warped"][:, ::-1])
    v = fam.verify_metamorphic(fam.linefit_solve(**fam.inputs(scene)), outs)
    assert not v and fam.T_MIRROR in v.reason and "NRMSE" in v.reason


# ---------------------------------------------------------------- calibration

def test_reference_and_standard_estimators_pass_fresh_manufactured_scenes_in_process():
    for seed in range(200, 224):
        s = fam.make_scene(seed)
        for name, fn in fam.STANDARD.items():
            v = fam.verify_manufactured(s, fn(**fam.inputs(s)))
            assert v, (seed, name, v.reason)
        ref = fam.reference_solve(**fam.inputs(s))
        for kw in ({"pixel_edges": True}, {"flags": cv2.INTER_NEAREST}, {"flags": cv2.INTER_CUBIC}):
            out = {"corners": ref["corners"], "warped": fam.warp_with(s.img, ref["corners"], s.out_w, s.out_h, **kw)}
            assert fam.verify_manufactured(s, out), (seed, kw)


def test_calibration_still_covers_fresh_gauge_blocks_and_matches_the_recorded_measurement():
    summary = fam.calibrate(n=48, seed=999_000, k_scenes=12)
    fit, corner = summary["corner_fit"], summary["corner"]
    assert fit["worst_ratio_to_tolerance"] < 1.0  # inside the band (recorded worst 0.79 on 2000 blocks)
    assert summary["image_worst_ratio_to_tolerance"] < 1.0  # recorded 0.78
    # recorded on 2000 blocks: contour estimators 1.29 / 1.44 px mean, line fit 0.24 mean and 1.15 max
    assert 1.0 < corner["reference"]["mean"] < 1.7 and 1.1 < corner["extremes"]["mean"] < 1.9
    assert corner["line fit"]["mean"] < 0.4 and corner["line fit"]["max"] < 1.6
    # the physics: contour estimators are shaved inwards more as the edge blurs (recorded slope 0.59 px/sigma),
    # and the unbiased line fit is better than either everywhere
    assert 0.3 < fit["slope_px_per_sigma"] < 0.9
    for name in ("reference", "extremes"):
        by_bin = corner[name]["mean_by_blur_bin"]
        assert np.mean(by_bin[3:]) > np.mean(by_bin[:2]) + 0.2, by_bin
    assert corner["line fit"]["max"] < 0.5 * corner["reference"]["max"]
    assert 0.10 < summary["texture_k_rms"] < 0.15  # recorded 0.124 rad per output px
    assert 0.10 < summary["image_worst_max"] < fam.TOL_IMG  # recorded 0.188 (nearest + pixel-edge convention)


def test_constants_carry_the_25_percent_margin_over_the_recorded_worst_cases():
    # recorded by calibrate(n=2000, seed=500_000) and measure_metamorphic_drift(n=300, seed=50_000)
    intercept, slope, image_worst, meta_corner, meta_image = 5.06, 0.59, 0.188, 5.00, 0.231
    for sigma in (fam.BLUR_RANGE[0], 1.4, fam.BLUR_RANGE[1]):
        tol = fam.TOL_A + fam.TOL_B * sigma
        assert 1.25 * (intercept + slope * sigma) - 0.05 <= tol <= 1.5 * (intercept + slope * sigma)
    assert 1.25 * image_worst - 0.01 <= fam.TOL_IMG <= 1.5 * image_worst
    assert 1.25 * meta_corner - 0.1 <= fam.META_TOL_CORNER <= 1.5 * meta_corner
    assert 1.25 * meta_image - 0.01 <= fam.META_TOL_IMG <= 1.5 * meta_image


def test_symmetries_are_exact_for_the_unbiased_estimator_and_inside_tolerance_for_the_others():
    drift = fam.measure_metamorphic_drift(n=12, seed=7_000)
    for key, v in drift["by"].items():
        assert v["order_violations"] == 0, key
        assert v["corner_max"] <= fam.META_TOL_CORNER and v["image_max"] <= fam.META_TOL_IMG, (key, v)
        if key.startswith("line fit"):  # recorded <= 0.25 px and 0.027 over 300 scenes: the relations are exact
            assert v["corner_max"] < 0.6 and v["image_max"] < 0.06, (key, v)


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img, out_w, out_h) -> dict" in p and "OpenCV: 5.0" in p
    assert "TL, TR, BR, BL" in p and "smallest x+y" in p and "warped[0, 0]" in p
    assert 3 <= len(fam.PHRASINGS) <= 5 and len(set(fam.PHRASINGS)) == len(fam.PHRASINGS)
