import dataclasses
import functools
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from cvbench.contract import check_outputs
from cvbench.evaluate import evaluate_solution
from cvbench.families import chessboard_corners as fam

# Expected verdicts per gate: the verifier's claim, pinned as a regression test. The rows with a "statistical" note
# fail on a handful of scenes of the 20 (seeds 0..19); a different seed range can change them.
EXPECTED = {
    #                               exec-only lint   manufactured metamorphic
    "reference":                    (True,     True,  True,        True),
    "rgb2gray (harmless here)":     (True,     True,  True,        True),
    "SB + cornerSubPix (correct)":  (True,     True,  True,        True),
    "index_corners_0":              (False,    True,  False,       False),  # 5.0 (N, 2): IndexError
    "no_cornersubpix":              (True,     True,  False,       False),
    "sb_unrefined":                 (True,     True,  False,       False),  # strict-law consequence, see docstring
    "pattern_in_squares":           (False,    True,  False,       False),
    "swapped_rows_cols":            (True,     True,  False,       True),   # same points, wrong order: sets agree
    "no_order_fix":                 (True,     True,  False,       True),   # ditto
    "sorted_by_y_then_x":           (True,     True,  False,       True),   # ditto
    "xy_swapped":                   (True,     True,  False,       False),
    "half_pixel_offset":            (True,     True,  False,       False),
    "integer_pixels":               (True,     True,  False,       False),
    "subpix_on_color":              (False,    True,  False,       False),
    "tiny_subpix_window":           (True,     True,  False,       False),
    "huge_subpix_window":           (True,     True,  False,       True),   # statistical: fails scene 6 by 0.07 px
    "fixed_threshold_127":          (True,     True,  False,       False),  # fails scene 11 / the dimmed image
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")
HARMLESS = ("reference", "rgb2gray (harmless here)", "SB + cornerSubPix (correct)")


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    with ThreadPoolExecutor(4) as ex:  # run_solution spawns a child process per call: threads only wait
        futs = {n: ex.submit(evaluate_solution, fam, code, n_scenes=20, n_meta=5) for n, (_, code) in candidates.items()}
        return {n: f.result() for n, f in futs.items()}


def test_expected_table_covers_every_candidate():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n not in HARMLESS]
    assert len(wrong) == 14
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == 11
    assert all(not matrix[n]["manufactured"].passed for n in wrong)


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


def test_reference_passes_many_manufactured_scenes_in_process():
    worst = 0.0
    for seed in range(200, 228):
        scene = fam.make_scene(seed)
        out = fam.reference_solve(**fam.inputs(scene))
        v = fam.verify_manufactured(scene, out)
        assert v, (seed, v.reason)
        worst = max(worst, float(np.linalg.norm(out["corners"] - scene.corners, axis=1).max()) / fam.tolerance(scene))
    assert worst < 0.8  # the tolerance has real headroom over the reference, not a knife edge


@functools.lru_cache(maxsize=None)
def _estimates(seed: int) -> dict:
    return fam._estimates(fam.make_scene(seed))


def _estimator_pass_rate(name: str, seeds) -> float:
    hits = 0
    for seed in seeds:
        est = _estimates(seed)[name]
        hits += int(est is not None and fam.verify_manufactured(fam.make_scene(seed), {"corners": est}).ok)
    return hits / len(seeds)


def test_the_tolerance_encodes_physics_not_one_recipe():
    """Every cornerSubPix window from 8 to 13 half-pixels, loose or tight termination, passes."""
    seeds = range(300, 330)
    assert _estimator_pass_rate("R8", seeds) == 1.0
    assert _estimator_pass_rate("R11l", seeds) == 1.0
    assert _estimator_pass_rate("sbR11", seeds) >= 0.95  # SB misdetects ~1 board in 1000


def test_unrefined_detectors_are_rejected_on_a_fraction_of_scenes():
    """Recorded decision (see the docstring): refinement is the point of the task. Unrefined
    findChessboardCorners passes ~40 % of scenes; unrefined SB passes ~85-87 %, so it passes a whole
    batch of 20 scenes only ~5 % of the time."""
    seeds = range(300, 330)
    assert _estimator_pass_rate("raw", seeds) < 0.6
    assert _estimator_pass_rate("sb", seeds) < 0.97


def test_opencv5_exposure_and_the_ordering_trap():
    """The measured facts behind the `index_corners_0` and `no_order_fix` mutants."""
    n_reversed = n_forward = 0
    for seed in range(24):
        scene = fam.make_scene(seed, blur=1.0, noise=1.0, contrast=140)
        sx, sy = scene.squares
        gray = cv2.cvtColor(scene.img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (sx - 1, sy - 1))
        assert found and corners.shape == ((sx - 1) * (sy - 1), 2)  # (N, 2) in 5.0; (N, 1, 2) in 4.x
        with pytest.raises(IndexError):
            corners[:, 0, :]
        c = corners.reshape(-1, 2)
        fwd = np.linalg.norm(c - scene.corners, axis=1).max() < 5
        rev = np.linalg.norm(c[::-1] - scene.corners, axis=1).max() < 5
        assert fwd != rev  # always exactly one of the two
        assert fwd == scene.tl_dark  # row-major iff the top-left square is black
        n_forward += fwd
        n_reversed += rev
    assert n_forward and n_reversed


def test_tolerance_law_scales_as_the_physics_says():
    base = dataclasses.replace(fam.make_scene(0), blur=1.0, noise=2.0, contrast=100.0)
    tol = lambda **kw: fam.tolerance(dataclasses.replace(base, **kw))  # noqa: E731
    # sampling bias falls with blur, then the noise term (which grows with the edge width) takes over
    quiet = {b: tol(blur=b, noise=0.0, contrast=150.0) for b in (0.0, 0.5, 1.0, 2.0)}
    assert quiet[0.0] > quiet[0.5] > quiet[1.0]
    assert quiet[0.0] / quiet[1.0] > 3
    # noise enters through noise / contrast: doubling both leaves the law alone (up to the 8-bit rounding term)
    assert tol(noise=4.0, contrast=200.0) == pytest.approx(tol(noise=2.0, contrast=100.0), rel=0.01)
    assert tol(noise=4.0) > tol(noise=2.0) > tol(noise=0.0)
    # the noise slope is steeper on blurrier edges
    slope = lambda b: tol(blur=b, noise=4.0) - tol(blur=b, noise=0.0)  # noqa: E731
    assert slope(2.0) > 1.8 * slope(0.6)
    # the verifier's tolerance is the bare law times the recorded margin
    assert fam.tolerance(base) == pytest.approx(fam.TOL_M * fam._law(1.0, 2.0, 100.0))


def test_constants_match_the_recorded_measurement():
    rec = fam.RECORDED
    assert (fam.A_ALIAS, fam.W_ALIAS, fam.K_NOISE) == pytest.approx(rec["refit_A_W_K"], rel=0.01)
    worst = max(rec["gauge_worst_over_law"], rec["prior_worst_over_law"])
    assert fam.TOL_M / worst >= 1.25  # at least 25 % headroom over the worst ratio ever measured
    assert rec["gauge_blocks"] == len(fam.GRID_BLUR) * len(fam.GRID_RATIO) * 40


def test_calibration_still_covers_fresh_gauge_blocks():
    summary = fam.calibrate(reps=1, seed=9_000_000, workers=4)
    est = summary["estimators"]
    assert summary["n_blocks"] == len(fam.GRID_BLUR) * len(fam.GRID_RATIO)
    assert summary["worst_over_tolerance"] < 1.0
    assert summary["law_class_worst_over_law"] < fam.TOL_M
    for name in fam.LAW_CLASS:
        assert est[name]["not_found"] == 0
        assert est[name]["median_over_law"] < 0.6  # the law is an envelope, not a mean: typical error is well inside
    # the physics, not just the pass rate: refinement is what makes the error small
    assert est["raw"]["median_over_law"] > 1.0 > est["R11"]["median_over_law"]
    assert est["raw"]["pass_fraction_at_tolerance"] < 0.6
    assert est["sbR11"]["pass_fraction_at_tolerance"] > 0.9


def _lanczos(gray: np.ndarray, pts: np.ndarray, a: int = 6) -> np.ndarray:
    """Lanczos-a interpolation at float (x, y) positions: accurate for these band-limited images
    (OpenCV's a = -0.75 cubic kernel is not: it leaves ~3 gray levels on the steepest edges)."""
    offs = np.arange(-a + 1, a + 1)
    out = np.empty(len(pts))
    for i, (x, y) in enumerate(pts):
        xs, ys = int(np.floor(x)) + offs, int(np.floor(y)) + offs
        wx, wy = np.sinc(x - xs) * np.sinc((x - xs) / a), np.sinc(y - ys) * np.sinc((y - ys) / a)
        out[i] = wy @ gray[ys[:, None], xs[None, :]] @ wx
    return out


def _point_symmetry_residual(scene, shift=(0.0, 0.0), radius=3.0) -> float:
    gray = cv2.cvtColor(scene.img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ang = np.linspace(0, np.pi, 7, endpoint=False)
    u = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    c = scene.corners + np.asarray(shift)
    p, q = (c[:, None, :] + u[None]).reshape(-1, 2), (c[:, None, :] - u[None]).reshape(-1, 2)
    return float(np.abs(_lanczos(gray, p) - _lanczos(gray, q)).max())


def test_truth_is_the_point_symmetry_centre_of_the_rendered_board():
    """Independent of every OpenCV estimator. A saddle between straight edges is point-symmetric
    about its corner (reflection maps quadrant ++ onto --, the same colour), so the image at c + u
    and c - u must agree. Measured: exact truth <= 1.4 gray levels (8-bit rounding; median 0.3);
    truth displaced by (0.03, 0.03) px: 2.4 to 5.2 depending on blur and scene. The test checks
    that it can fail (a 0.05 px diagonal displacement) before checking that the truth does not."""
    for blur in (1.0, 2.0):
        for seed in (1, 2):
            scene = fam.make_scene(seed, blur=blur, noise=0.0, contrast=150.0)
            assert _point_symmetry_residual(scene) < 2.0
            assert _point_symmetry_residual(scene, shift=(0.05, 0.05)) > 2.5


def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam._build(3), fam._build(3)
    assert np.array_equal(a.img, b.img) and np.array_equal(a.corners, b.corners) and a.squares == b.squares
    c = fam.make_scene(3)
    assert np.array_equal(c.img, a.img)
    c.img[:] = 0  # callers get their own copy of the cached scene
    assert np.array_equal(fam.make_scene(3).img, a.img)
    tl, shapes = set(), set()
    for seed in range(30):
        s = fam.make_scene(seed)
        sx, sy = s.squares
        assert (sx - sy) % 2 == 1 and s.img.shape == (fam.IMG_H, fam.IMG_W, 3) and s.img.dtype == np.uint8
        assert s.corners.shape == ((sx - 1) * (sy - 1), 2)
        assert s.min_square_px >= fam.SQ_MIN
        assert s.corners.min() >= fam.MARGIN and s.corners[:, 0].max() <= fam.IMG_W - 1 - fam.MARGIN
        assert s.corners[0].sum() < s.corners[-1].sum()  # upright row-major: the first corner is top-left of the last
        assert 0.0 <= s.blur <= 2.0 and 0.0 <= s.noise <= 4.0 and 70.0 * 0.92 <= s.contrast <= 190.0 * 1.08
        tl.add(s.tl_dark)
        shapes.add(sx > sy)
    assert tl == {True, False} and shapes == {True, False}  # both colours of top-left square, landscape and portrait
    # pinning a factor leaves the geometry and the noise pattern of that seed untouched
    p, q = fam.make_scene(7, blur=0.5), fam.make_scene(7, blur=1.5)
    assert np.array_equal(p.corners, q.corners) and p.noise == q.noise and not np.array_equal(p.img, q.img)


@pytest.mark.parametrize("name", [n for n in fam._COVARIANT if n != "dim x0.5"])
def test_covariant_maps_follow_a_marked_pixel(name):
    """verify_metamorphic maps corner sets by transform name alone: check the map against the real image op."""
    scene = fam.make_scene(0)
    for (x, y) in ((100, 60), (500, 400), (33, 301)):
        img = np.zeros_like(scene.img)
        img[y, x] = 255
        t = fam.transforms(dataclasses.replace(scene, img=img, surround=(0.0, 0.0, 0.0)))[name]["img"]
        r, c = np.unravel_index(np.argmax(t[..., 0]), t.shape[:2])
        assert fam._COVARIANT[name](np.array([[x, y]], dtype=np.float64))[0] == pytest.approx([c, r])
    sx, sy = scene.squares
    tr = fam.transforms(scene)[name]
    assert (tr["squares_x"], tr["squares_y"]) == ((sy, sx) if name == "rot90" else (sx, sy))


def test_verify_manufactured_diagnoses_what_is_wrong():
    scene = fam.make_scene(4)
    truth = scene.corners
    tol = fam.tolerance(scene)
    assert fam.verify_manufactured(scene, {"corners": truth + 0.4 * tol})
    assert fam.verify_manufactured(scene, {"corners": truth.astype(np.float32)})
    off = truth.copy()
    off[5, 0] += 1.5 * tol
    v = fam.verify_manufactured(scene, {"corners": off})
    assert not v and "corner 5" in v.reason
    v = fam.verify_manufactured(scene, {"corners": truth[::-1].copy()})
    assert not v and "reversed" in v.reason
    sx, sy = scene.squares
    cm = truth.reshape(sy - 1, sx - 1, 2).transpose(1, 0, 2).reshape(-1, 2)
    v = fam.verify_manufactured(scene, {"corners": cm})
    assert not v and "column" in v.reason
    v = fam.verify_manufactured(scene, {"corners": truth[:-1]})
    assert not v and "expected" in v.reason
    v = fam.verify_manufactured(scene, {"corners": truth.reshape(-1, 1, 2)})  # the 4.x shape
    assert not v and "ndim" in v.reason
    v = fam.verify_manufactured(scene, {"corners": np.full_like(truth, np.nan)})
    assert not v


def test_verify_metamorphic_checks_covariance_and_ignores_order():
    scene = fam.make_scene(4)
    base = {"corners": scene.corners}
    good = {n: {"corners": f(scene.corners)[::-1].copy()} for n, f in fam._COVARIANT.items()}  # order is free
    assert fam.verify_metamorphic(base, good)
    bad = {n: {"corners": f(scene.corners + 0.5)} for n, f in fam._COVARIANT.items()}  # pixel-centre convention slip
    v = fam.verify_metamorphic(base, {"rot90": bad["rot90"]})
    assert not v and "rot90" in v.reason
    assert not fam.verify_metamorphic(base, {"mirror": bad["mirror"]})
    assert not fam.verify_metamorphic(base, {"mirror": {"corners": good["mirror"]["corners"][:-1]}})
    assert not fam.verify_metamorphic(base, {"nonsense": good["mirror"]})
    assert not fam.verify_metamorphic(base, {"mirror": {"corners": good["mirror"]["corners"].reshape(-1, 1, 2)}})


def test_metamorphic_symmetries_hold_for_the_reference():
    """Exact symmetries move the reference's corners by ~1e-3 px; dimming costs the rounding noise (~0.1 px)."""
    d = fam.measure_metamorphic_drift(n=12, seed=7_000)
    assert d["not_found"] == {}
    for name in ("translate(+7,-5)", "rot90", "mirror"):
        assert d["worst_drift_px"][name] < 0.01, d
    assert d["worst_drift_px"]["dim x0.5"] < 0.75 * fam.META_TOL, d


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img, squares_x, squares_y) -> dict" in p and "OpenCV: 5.0" in p
    assert "Row-major" in p and "(N, 2)" not in p and "Nx2" in p
    assert 3 <= len(fam.PHRASINGS) <= 5 and fam.PHRASINGS[0] == fam.SPEC.task
    scene = fam.make_scene(0)
    assert check_outputs(fam.SPEC, fam.reference_solve(**fam.inputs(scene))) == []
