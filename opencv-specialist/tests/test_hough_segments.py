import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import hough_segments as fam
from sandbox.run import run_solution

# Expected verdicts per gate: the family's claim, pinned as a regression test.
EXPECTED = {
    #                      exec-only lint   manufactured metamorphic
    "reference":           (True,    True,  True,        True),
    "unpack_line0":        (False,   True,  False,       False),
    "n1x4_layout":         (True,    True,  False,       False),
    "houghlines_infinite": (True,    True,  False,       False),
    "no_merge":            (True,    True,  False,       False),
    "plain_union":         (True,    True,  False,       False),
    "fixed_canny":         (True,    True,  True,        False),
    "fixed_low_canny":     (True,    True,  True,        False),
    "fixed_threshold_127": (True,    True,  True,        False),
    "unordered":           (True,    True,  False,       True),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")
# Which check fires, so a mutant cannot start failing for a different reason unnoticed.
REASON = {
    ("unpack_line0", "exec-only"): "cannot unpack non-iterable",
    ("n1x4_layout", "manufactured"): "ndim 2",
    ("no_merge", "manufactured"): "duplicate or fragmentary",
    ("plain_union", "manufactured"): "has no row within",
    ("fixed_canny", "metamorphic"): "dim x0.45",
    ("fixed_low_canny", "metamorphic"): "dim x0.15",  # and not x0.45: that dimming was the gap this one closes
    ("fixed_threshold_127", "metamorphic"): "dim x0.45: 0 segments",
    ("unordered", "manufactured"): "smaller x",
}
# Worst end point error (px) by stroke width, pooled over four seeds x 6000 gauge blocks (see TOL_A in the module),
# and the reference's own worst case over 8000 multi-stroke scenes.
RECORDED_PANEL_WORST = {1: 1.42, 2: 2.24, 3: 3.16, 4: 3.16}
RECORDED_SCENE_WORST = {1: 1.41, 2: 2.00, 3: 2.24, 4: 2.24}


def canon(rows) -> np.ndarray:
    """Integer rows in the contract's canonical order."""
    out = []
    for x1, y1, x2, y2 in np.round(np.asarray(rows, dtype=float)).astype(int).reshape(-1, 4):
        p, q = sorted([(x1, y1), (x2, y2)])
        out.append(p + q)
    return np.array(sorted(out), dtype=np.int32).reshape(-1, 4)


def truth_rows(scene) -> np.ndarray:
    return canon([[*s.p1, *s.p2] for s in scene.segments])


def otsu_canny_code() -> str:
    """The fixed_canny mutant with its absolute thresholds replaced by Otsu-driven ones: a correct alternative."""
    old = "mask = cv2.morphologyEx(cv2.Canny(gray, 100, 200), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))"
    new = ("hi, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "    mask = cv2.morphologyEx(cv2.Canny(gray, 0.5 * hi, hi), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))")
    code = fam.MUTANTS["fixed_canny"][1]
    assert old in code
    return code.replace(old, new)


FIXED_CANNY_MASK = "mask = cv2.morphologyEx(cv2.Canny(gray, 100, 200), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))"


def mask_code(mask_step: str) -> str:
    """The fixed_canny mutant with its Canny(100, 200) mask line replaced (extra lines indented 4 spaces)."""
    code = fam.MUTANTS["fixed_canny"][1]
    assert FIXED_CANNY_MASK in code
    return code.replace(FIXED_CANNY_MASK, mask_step)


def adaptive_canny_code(prelude: str, thresholds: str) -> str:
    """fixed_canny with its absolute Canny thresholds computed from the image: a correct alternative."""
    return mask_code(f"{prelude}\n    mask = cv2.morphologyEx(cv2.Canny(gray, {thresholds}), cv2.MORPH_CLOSE, "
                     "np.ones((7, 7), np.uint8))")


def fixed_canny_code(lo: int, hi: int) -> str:
    return mask_code(f"mask = cv2.morphologyEx(cv2.Canny(gray, {lo}, {hi}), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))")


MEDIAN_CANNY = ("med = float(np.median(gray))", "int(0.66 * med), int(1.33 * med)")
MAX_CANNY = ("mx = float(gray.max())", "0.4 * mx, 0.9 * mx")
OTSU_CLOSING = ("_, m0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
                "    mask = cv2.morphologyEx(m0, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))")


def sandbox_drift(code: str, n: int, seed: int) -> dict[str, float]:
    """Largest move of ``code``'s end points against the covariant prediction, per symmetry (via the sandbox)."""
    batch, layout = [], []
    for i in range(n):
        scene = fam.make_scene(seed + i)
        variants = fam.transforms(scene)
        layout.append(list(variants))
        batch.append(fam.inputs(scene))
        batch.extend(variants.values())
    r = run_solution(code, batch, lint=False, timeout_s=120)
    assert r.ok, r.summary()
    worst = {name: 0.0 for name in fam.META_TOL}
    i = 0
    for names in layout:
        base = np.asarray(r.outputs[i]["segments"], dtype=float)
        i += 1
        for name in names:
            got = np.asarray(r.outputs[i]["segments"], dtype=float)
            i += 1
            want = fam.expected_under(name, base)
            if len(got) != len(want):
                worst[name] = float("inf")
                continue
            _, dist = fam._unmatched(want, np.full(len(want), np.inf), got)
            worst[name] = max(worst[name], float(dist.min(axis=1).max()))
    return worst


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    return {name: evaluate_solution(fam, code, n_scenes=20, n_meta=5) for name, (_, code) in candidates.items()}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


@pytest.mark.parametrize("key", list(REASON))
def test_mutants_fail_for_the_documented_reason(matrix, key):
    name, gate = key
    assert REASON[key] in matrix[name][gate].detail, matrix[name][gate].detail


def test_expected_matrix_covers_every_mutant():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}
    assert 5 <= len(fam.MUTANTS) <= 9  # was 8 (families/README.md: 5-8); fixed_low_canny is the ninth


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n != "reference"]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == len(wrong) - 1  # only the 4.x unpack crashes
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


def test_a_correct_alternative_passes_every_gate():
    """Otsu-driven Canny + the same merge is a different pipeline with the same answer: all four gates pass."""
    gates = evaluate_solution(fam, otsu_canny_code(), n_scenes=20, n_meta=5)
    assert all(g.passed for g in gates.values()), {k: g.detail for k, g in gates.items()}


def test_reference_passes_many_manufactured_scenes_in_process():
    for seed in range(200, 400):
        scene = fam.make_scene(seed)
        v = fam.verify_manufactured(scene, fam.reference_solve(scene.img))
        assert v, (seed, v.reason)


def test_hough_union_is_the_reference():
    """calibrate() measures the reference through _hough_union: its defaults must be the reference."""
    for seed in range(15):
        scene = fam.make_scene(seed)
        assert np.array_equal(canon(fam._hough_union(scene.img)), fam.reference_solve(scene.img)["segments"]), seed


@pytest.mark.parametrize("est", [fam._est_fitline, lambda img: fam._est_fitline(img, widen=True),
                                 fam._est_minarearect], ids=["fitLine", "fitLine(dilated)", "minAreaRect"])
def test_estimators_without_hough_also_pass(est):
    """The tolerance encodes the physics of the measurement, not one API."""
    for seed in range(400, 450):
        scene = fam.make_scene(seed)
        v = fam.verify_manufactured(scene, {"segments": canon(est(scene.img))})
        assert v, (seed, v.reason)


def test_calibration_still_covers_fresh_gauge_blocks():
    c = fam.calibrate(n=1200, seed=4242)
    for w, worst in c["panel_worst"].items():
        assert worst * 1.2 <= fam.TOL_A + fam.TOL_B * w, (w, worst)
    assert c["worst_ratio"] < 1 / 1.2
    assert c["hausdorff_minus_endpoint"] < 0.01  # the metric question: end point distance = Hausdorff distance
    assert c["angle"]["reject_atan(2*tol/L)"] == 0.0  # the end point test already bounds the angle
    assert c["misses"]["reference"] == 0
    # the physics: the stroke's own pixels end ceil(t/2) beyond the end point (round caps) and no further,
    # except that a 1-px line has no cap (measured: 0, 1, 2, 2 px for t = 1..4)
    fit = {w: v["max"] for w, v in c["panel"]["fitLine"].items()}
    assert fit[1] < 0.35 and 0.9 < fit[2] < 1.2 and 1.9 < fit[3] < 2.2 and 1.9 < fit[4] < 2.2, fit
    # the angle step does not enter the law once the stroke is widened and refined ...
    for dth, by_w in c["dtheta_reference"].items():
        for w, worst in by_w.items():
            assert worst * 1.2 <= fam.TOL_A + fam.TOL_B * w, (dth, w, worst)
    # ... but a bare 1-px stroke has a heavy tail that grows with it, and textbook parameters are worse still
    ph = c["plain_hough"]
    tol1 = fam.TOL_A + fam.TOL_B
    assert ph["0.5deg"]["p50"] == 0.0
    assert ph["2deg"]["p95"] > ph["0.5deg"]["p95"]
    assert ph["2deg"]["frac_over_tol"] > ph["0.5deg"]["frac_over_tol"]
    assert ph["1deg"]["frac_over_tol"] > 0.02 and ph["1deg"]["max"] > 3 * tol1
    assert ph["1deg textbook thr20/len30/gap5"]["max"] > 5 * tol1


def test_constants_follow_from_the_recorded_measurement():
    a_needed = max(1.25 * v - fam.TOL_B * w for w, v in RECORDED_PANEL_WORST.items())
    assert fam.TOL_B == 0.7
    assert a_needed <= fam.TOL_A < a_needed + 0.1, (a_needed, fam.TOL_A)
    tols = {w: fam.TOL_A + fam.TOL_B * w for w in RECORDED_PANEL_WORST}
    assert [round(t, 1) for t in tols.values()] == [2.6, 3.3, 4.0, 4.7]
    assert all(tols[w] >= 1.25 * v for w, v in RECORDED_PANEL_WORST.items())
    assert all(tols[w] >= 1.25 * v for w, v in RECORDED_SCENE_WORST.items())
    assert fam.META_TOL[fam.T_DIM] == 1.0 and fam.META_TOL[fam.T_SHIFT] == fam.META_TOL[fam.T_ROT] == 5.0
    assert fam.GAIN == 0.45 and fam.GAIN_STRONG == 0.15  # GAIN is shared with contour_area: not this family's to change
    assert fam.META_TOL[fam.T_DIM_STRONG] == 1.0 and fam.T_DIM_STRONG == "dim x0.15"
    assert fam.T_DIM in fam.transforms(fam.make_scene(0)) and fam.T_DIM_STRONG in fam.transforms(fam.make_scene(0))


def test_tolerance_scales_with_stroke_width():
    def tol(w):
        return fam.tolerance(fam.Segment((0, 0), (50, 0), w))
    assert tol(4) - tol(1) == pytest.approx(3 * fam.TOL_B)
    assert 2.5 < tol(1) < tol(2) < tol(3) < tol(4) < 5.0


def test_metamorphic_drift_of_the_reference_is_inside_the_tolerances():
    d = fam.measure_metamorphic_drift(n=40, seed=7_000)
    for name in (fam.T_SHIFT, fam.T_ROT, fam.T_MIRROR):  # recorded: max 2.83 px over 600 scenes
        assert d[name]["max"] <= fam.META_TOL[name] / 1.25, (name, d[name])
    for name in (fam.T_DIM, fam.T_DIM_STRONG):  # dimming does not touch the geometry: exactly invariant
        assert d[name] == {"max": 0.0, "p99": 0.0, "mean": 0.0}, (name, d[name])


def test_adaptive_pipelines_are_invariant_to_dimming_and_fixed_thresholds_are_not():
    adaptive = sandbox_drift(otsu_canny_code(), n=25, seed=8_000)
    assert adaptive[fam.T_DIM] == adaptive[fam.T_DIM_STRONG] == 0.0
    for name in (fam.T_SHIFT, fam.T_ROT, fam.T_MIRROR):
        assert adaptive[name] <= 4.0 <= fam.META_TOL[name] / 1.25, (name, adaptive[name])
    fixed = sandbox_drift(fam.MUTANTS["fixed_canny"][1], n=25, seed=8_000)
    assert fixed[fam.T_DIM] > fam.META_TOL[fam.T_DIM]


def test_a_low_fixed_canny_survives_the_gentle_dimming_and_not_the_strong_one():
    """Gate 2 review, decision 4. Canny(30, 100) is exactly invariant under x0.45 (the gap), while under x0.15
    some strokes move more than the tolerance or vanish (1006 of 2000 scenes at seeds 7_610_000-7_611_999)."""
    low = sandbox_drift(fam.MUTANTS["fixed_low_canny"][1], n=40, seed=7_100_000)
    assert low[fam.T_DIM] == 0.0
    assert low[fam.T_DIM_STRONG] > fam.META_TOL[fam.T_DIM_STRONG], low


def test_the_strong_dimming_reaches_canny_20_60_and_the_gentle_one_does_not():
    """The step from x0.25 to x0.15 (2026-10-22): Canny(20, 60) fails 394 of 2000 scenes (19.7 %, seeds 7_610_000-
    7_611_999) at x0.15 and 5 at x0.25, and none at x0.45. (Canny(10, 30) still passes: 0 of 2000.)"""
    drift = sandbox_drift(fixed_canny_code(20, 60), n=40, seed=7_630_000)
    assert drift[fam.T_DIM] == 0.0
    assert drift[fam.T_DIM_STRONG] > fam.META_TOL[fam.T_DIM_STRONG], drift


def test_strong_dimming_keeps_the_weakest_thin_stroke_the_generator_can_draw():
    """1-px strokes of gray 150 on gray 50 are the least contrast make_scene can produce (100 gray levels); at x0.15
    they are 22 on 8, and every correct pipeline still sees every one of them: the gate must not reject correct code."""
    img = np.full((fam.H, fam.W, 3), 50, np.uint8)
    segs = [fam.Segment((40, 40), (200, 40), 1), fam.Segment((60, 90), (60, 250), 1),
            fam.Segment((120, 100), (260, 220), 1), fam.Segment((300, 60), (420, 300), 1)]
    for s in segs:
        cv2.line(img, s.p1, s.p2, (150, 150, 150), 1)
    assert all(fam.segment_distance(a, b) >= fam.CLEAR for i, a in enumerate(segs) for b in segs[i + 1:])
    scene = fam.Scene(img, segs, 50, -1)
    dimmed = fam.transforms(scene)[fam.T_DIM_STRONG]["img"]
    assert int(dimmed[40, 100, 0]) - int(dimmed[0, 0, 0]) == 22 - 8
    correct = {"reference": fam.REFERENCE, "Otsu-driven Canny": otsu_canny_code(),
               "median-driven Canny": adaptive_canny_code(*MEDIAN_CANNY),
               "max-relative Canny": adaptive_canny_code(*MAX_CANNY), "Otsu mask + 7x7 closing": mask_code(OTSU_CLOSING)}
    for name, code in correct.items():
        r = run_solution(code, [{"img": img}, {"img": dimmed}], lint=False, timeout_s=60)
        assert r.ok, r.summary()
        assert fam.verify_manufactured(scene, r.outputs[0]), name
        assert np.array_equal(np.asarray(r.outputs[0]["segments"]), np.asarray(r.outputs[1]["segments"])), name


@pytest.mark.parametrize("prelude, thresholds", [MEDIAN_CANNY, MAX_CANNY], ids=["median-driven", "max-relative"])
def test_other_adaptive_canny_pipelines_are_invariant_to_dimming(prelude, thresholds):
    code = adaptive_canny_code(prelude, thresholds)
    assert code != fam.MUTANTS["fixed_canny"][1]
    drift = sandbox_drift(code, n=15, seed=9_000)
    assert drift[fam.T_DIM] == drift[fam.T_DIM_STRONG] == 0.0
    for name in (fam.T_SHIFT, fam.T_ROT, fam.T_MIRROR):
        assert drift[name] <= fam.META_TOL[name], (name, drift[name])


def test_covariant_maps_match_numpy():
    """expected_under() lands where np.rot90 / mirror / the shift actually put a pixel."""
    img = np.zeros((fam.H, fam.W, 3), np.uint8)
    x, y = 123, 45
    img[y, x] = 255
    scene = fam.Scene(img, [], 0, -1)
    seg = np.array([[x, y, x, y]], dtype=float)
    for name, variant in fam.transforms(scene).items():
        ys, xs = np.nonzero(variant["img"][..., 0] > 0)
        want = fam.expected_under(name, seg)[0]
        assert len(xs) == 1 and (xs[0], ys[0]) == (want[0], want[1]) == (want[2], want[3]), (name, xs, ys, want)


def test_verifier_accepts_the_truth_and_pins_the_law():
    scene = fam.make_scene(5)
    rows = truth_rows(scene)
    assert fam.verify_manufactured(scene, {"segments": rows})
    assert fam.verify_manufactured(scene, {"segments": rows.astype(np.float64)})
    # move the second end point of the thinnest stroke along y by just under / just over its tolerance
    s = min(scene.segments, key=lambda g: g.width)
    k = int(np.nonzero((rows == canon([[*s.p1, *s.p2]])[0]).all(axis=1))[0][0])
    inside, outside = rows.astype(float), rows.astype(float)
    inside[k, 3] += fam.tolerance(s) - 0.1
    outside[k, 3] += fam.tolerance(s) + 0.3
    assert fam.verify_manufactured(scene, {"segments": inside})
    bad = fam.verify_manufactured(scene, {"segments": outside})
    assert not bad and "has no row within" in bad.reason


@pytest.mark.parametrize("what", ["duplicate", "missing", "unsorted rows", "ends reversed", "wrong shape", "nan",
                                  "object", "x and y swapped"])
def test_verifier_rejects_contract_violations(what):
    scene = fam.make_scene(6)
    rows = truth_rows(scene).astype(float)
    if what == "duplicate":
        out = np.vstack([rows, rows[:1]])
    elif what == "missing":
        out = rows[:-1]
    elif what == "unsorted rows":
        out = rows[::-1]
    elif what == "ends reversed":
        out = rows.copy()
        out[0] = out[0][[2, 3, 0, 1]]
    elif what == "wrong shape":
        out = rows[:, :3]
    elif what == "nan":
        out = rows.copy()
        out[0, 0] = np.nan
    elif what == "object":
        out = rows.astype(object)
    else:
        out = rows[:, [1, 0, 3, 2]]
    assert not fam.verify_manufactured(scene, {"segments": out})


def test_metamorphic_verifier_compares_sets_and_catches_moves():
    scene = fam.make_scene(7)
    base = truth_rows(scene).astype(float)
    # the same set of segments with the rows reversed and every row's ends swapped
    variants = {name: {"segments": fam.expected_under(name, base)[::-1][:, [2, 3, 0, 1]]} for name in fam.META_TOL}
    assert fam.verify_metamorphic({"segments": base}, variants)
    moved = {fam.T_SHIFT: {"segments": fam.expected_under(fam.T_SHIFT, base) + [0, fam.META_TOL[fam.T_SHIFT] + 1, 0, 0]}}
    assert not fam.verify_metamorphic({"segments": base}, moved)
    assert not fam.verify_metamorphic({"segments": base}, {fam.T_SHIFT: {"segments": base}})  # stuck: not covariant
    dim_moved = {fam.T_DIM: {"segments": base + [0, 0, 0, fam.META_TOL[fam.T_DIM] + 0.5]}}
    assert not fam.verify_metamorphic({"segments": base}, dim_moved)


def test_hough_lines_p_shape_is_the_5_0_exposure():
    """The measured 4.14 -> 5.0 change the family is about: (N, 1, 4) became (N, 4); no lines is still None."""
    assert cv2.__version__.startswith("5."), cv2.__version__
    img = np.zeros((100, 200), np.uint8)
    cv2.line(img, (10, 10), (180, 80), 255, 3)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=5)
    assert lines.ndim == 2 and lines.shape[1] == 4 and lines.dtype == np.int32
    assert lines.reshape(-1, 4).shape == lines.shape  # the version-proof idiom is a no-op here
    assert cv2.HoughLinesP(np.zeros((50, 50), np.uint8), 1, np.pi / 180, 20) is None


def test_lsd_detect_returns_n_by_4_float32_in_5_0():
    """Measured on OpenCV 5.0.0: lines are (N, 4) float32 (no singleton axis); width and prec are (N,) float64;
    nfa is None; an image without a segment gives None in all four."""
    assert cv2.__version__.startswith("5."), cv2.__version__
    gray = cv2.cvtColor(fam.make_scene(7_160_001).img, cv2.COLOR_BGR2GRAY)
    lines, width, prec, nfa = cv2.createLineSegmentDetector().detect(gray)
    assert lines.ndim == 2 and lines.shape[1] == 4 and lines.shape[0] > 0 and lines.dtype == np.float32, lines.shape
    assert width.shape == prec.shape == (len(lines),) and width.dtype == prec.dtype == np.float64 and nfa is None
    assert lines.reshape(-1, 4).shape == lines.shape  # the version-proof idiom of _est_lsd is a no-op here
    assert cv2.createLineSegmentDetector().detect(np.zeros((100, 100), np.uint8)) == (None, None, None, None)


def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.img, b.img) and a.segments == b.segments and a.background == b.background
    assert not np.array_equal(fam.make_scene(4).img, a.img)
    widths, axis_aligned = set(), 0
    for seed in range(60):
        s = fam.make_scene(seed)
        assert s.img.shape == (fam.H, fam.W, 3) and s.img.dtype == np.uint8
        assert fam.N_RANGE[0] <= len(s.segments) <= fam.N_RANGE[1]
        assert 10 <= s.background <= 50
        for i, g in enumerate(s.segments):
            widths.add(g.width)
            axis_aligned += g.p1[0] == g.p2[0] or g.p1[1] == g.p2[1]
            assert g.length >= fam.LMIN
            m = fam.BORDER + g.width
            assert all(m <= p[0] < fam.W - m and m <= p[1] < fam.H - m for p in (g.p1, g.p2))
            mid = ((g.p1[0] + g.p2[0]) // 2, (g.p1[1] + g.p2[1]) // 2)
            around = cv2.cvtColor(s.img[mid[1] - 1:mid[1] + 2, mid[0] - 1:mid[0] + 2], cv2.COLOR_BGR2GRAY)
            assert int(around.max()) - s.background >= 100, "stroke is not bright on the background"
            for h in s.segments[i + 1:]:
                assert fam.segment_distance(g, h) >= fam.CLEAR, "strokes too close"
    assert widths == set(range(fam.WIDTHS[0], fam.WIDTHS[1] + 1)) and axis_aligned > 0


def test_segment_distance():
    S = fam.Segment
    assert fam.segment_distance(S((0, 0), (10, 0), 1), S((5, -5), (5, 5), 1)) == 0.0  # crossing
    assert fam.segment_distance(S((0, 0), (10, 0), 1), S((0, 7), (10, 7), 1)) == pytest.approx(7.0)
    assert fam.segment_distance(S((0, 0), (10, 0), 1), S((13, 4), (20, 4), 1)) == pytest.approx(5.0)


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img) -> dict" in p and "OpenCV: 5.0" in p and "'segments'" in p
    assert "smaller x" in p and "one row per stroke" in p and "sorted by x1" in p
    assert len(fam.PHRASINGS) == 16 and len(set(fam.PHRASINGS)) == len(fam.PHRASINGS) and fam.PHRASINGS[0] == fam.SPEC.task


def test_every_phrasing_states_the_task():
    for ph in fam.PHRASINGS:  # a rewording must still ask for the end points of each stroke, in rows
        low = ph.lower()
        assert "stroke" in low or "line" in low or "segment" in low, ph
        assert "end" in low, ph
        assert "row" in low or "N x 4" in ph, ph
