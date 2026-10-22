import functools
import math

import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import color_segmentation as fam

# Expected verdicts per gate: this is the family's claim, pinned as a regression test.
EXPECTED = {
    #                                    exec-only lint   manufactured metamorphic
    "reference":                         (True,     True,  True,        True),
    "rgb2hsv_on_bgr":                    (True,     True,  False,       True),
    "red_low_lobe_only":                 (True,     True,  False,       True),
    "red_wrapped_bounds":                (True,     True,  False,       True),
    "hue_0_360":                         (True,     True,  False,       True),
    "hsv_full_with_180_bounds":          (True,     True,  False,       True),
    "no_saturation_value_gate":          (True,     True,  False,       True),
    "value_floor_100":                   (True,     True,  True,        False),  # only dimming sees it
    "exclusive_hue_bounds":              (True,     True,  False,       True),
    "mask_sum_not_count":                (True,     True,  False,       True),
    "contour_area_sum":                  (True,     True,  False,       True),
    "blur_before_hsv":                   (True,     True,  False,       False),
    "countnonzero_3channel":             (False,    True,  False,       False),
    "swallowed_countnonzero_3channel":   (True,     False, False,       True),   # constant 0 is invariant
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")

scene_of = functools.lru_cache(maxsize=None)(fam.make_scene)  # scenes are never mutated by the tests

# Correct solutions written differently from the reference: the verifier must not be tied to one API.
ALTERNATIVES = {
    "numpy_masks": '''import cv2
import numpy as np


def solve(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    ok = (s >= 100) & (v >= 80)
    red = ok & (((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 179)))
    yellow = ok & (h >= 20) & (h <= 35)
    green = ok & (h >= 40) & (h <= 80)
    blue = ok & (h >= 100) & (h <= 130)
    return {"red": int(red.sum()), "yellow": int(yellow.sum()), "green": int(green.sum()), "blue": int(blue.sum())}
''',
    "bitwise_or_count_nonzero": '''import cv2
import numpy as np


def count(hsv, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array([lo, 100, 80]), np.array([hi, 255, 255]))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return int(np.count_nonzero(mask))


def solve(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {"red": count(hsv, [(0, 10), (170, 179)]), "yellow": count(hsv, [(20, 35)]),
            "green": count(hsv, [(40, 80)]), "blue": count(hsv, [(100, 130)])}
''',
}


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    return {name: evaluate_solution(fam, code, n_scenes=20, n_meta=5) for name, (_, code) in candidates.items()}


def test_expected_table_covers_every_mutant():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


@pytest.mark.parametrize("name", list(ALTERNATIVES))
def test_correct_alternative_implementations_pass_every_gate(name):
    gates = evaluate_solution(fam, ALTERNATIVES[name], n_scenes=20, n_meta=5)
    assert all(gates[g].passed for g in GATES), {g: gates[g].detail for g in GATES}


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n != "reference"]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == len(wrong) - 1  # only the raising one is caught
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)


def test_the_gates_are_independent(matrix):
    """Each gate has a mutant that only it catches (or, for lint, that only it separates from exec-only)."""
    only_metamorphic = matrix["value_floor_100"]
    assert only_metamorphic["manufactured"].passed and not only_metamorphic["metamorphic"].passed
    assert "dim" in only_metamorphic["metamorphic"].detail
    only_manufactured = matrix["rgb2hsv_on_bgr"]
    assert not only_manufactured["manufactured"].passed and only_manufactured["metamorphic"].passed
    swallowed = matrix["swallowed_countnonzero_3channel"]
    assert swallowed["exec-only"].passed and not swallowed["lint"].passed


def test_every_mutant_differs_from_reference():
    assert 5 <= len(fam.MUTANTS)
    for name, (why, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE and why, name


# ---------------------------------------------------------------- the reference and the contract

def test_reference_passes_many_manufactured_scenes_in_process():
    for seed in range(200, 320):
        scene = scene_of(seed)
        v = fam.verify_manufactured(scene, fam.reference_solve(scene.img))
        assert v, (seed, v.reason)


def test_reference_matches_the_contract_constants_on_the_whole_colour_cube():
    """Every 8-bit colour once: the reference's literals must agree with HUE, S_MIN and V_MIN everywhere."""
    cube = fam.hsv_cube()
    hsv = cv2.cvtColor(cube, cv2.COLOR_BGR2HSV)
    cls = fam.classify_array(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    expected = {c: int((cls == i).sum()) for i, c in enumerate(fam.COLOURS, 1)}
    assert fam.reference_solve(cube) == expected
    assert all(expected[c] > 0 for c in fam.COLOURS)


def test_cv2_8bit_hsv_scale_facts_the_contract_relies_on():
    fp = fam.hsv_cube_fingerprint()
    assert (fp["h_max"], fp["s_max"], fp["v_max"]) == (179, 255, 255)  # hue never reaches 180: red is 170-179
    assert len(fp["sha1"]) == 40
    px = np.array([[(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]], np.uint8)  # red yellow green blue
    assert cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, :, 0].tolist() == [0, 30, 60, 120]
    assert cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, :, 0].tolist() == [120, 90, 60, 0]  # the RGB-on-BGR swap
    assert cv2.cvtColor(px, cv2.COLOR_BGR2HSV_FULL)[0, :, 0].tolist() == [0, 43, 85, 171]  # FULL hue is 0-255


def test_inrange_with_lower_hue_above_upper_matches_nothing():
    """Why 'one inRange from 170 to 10' is a bug, not a wrap-around."""
    px = np.array([[(0, 0, 255), (40, 40, 250)]], np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
    assert not cv2.inRange(hsv, (170, 100, 80), (10, 255, 255)).any()


# ---------------------------------------------------------------- tolerance and calibration

def test_tolerance_is_zero_and_consistent_with_the_recorded_calibration():
    assert fam.TOL_PX == 0 and fam.META_TOL_PX == 0
    cal = fam.calibrate(n=480, seed=999, evidence=False)
    assert cal["worst_abs_error_px"] <= fam.TOL_PX and cal["blocks_with_any_error"] == 0
    assert set(cal["by_kind"]) == set(fam.KINDS) and all(lo == hi == 0 for lo, hi in cal["by_kind"].values())


def test_antialiasing_would_break_exactness():
    """The recorded reason shapes are drawn with LINE_8: a blended edge adds ~0.5-0.9 x perimeter extra pixels."""
    ev = fam.antialias_evidence(n=60, seed=1)
    for lo, hi in ev.values():
        assert 0.4 < lo <= hi < 1.0


def test_conversion_rounding_is_real_but_far_inside_the_generator_margin():
    ev = fam.conversion_evidence(n=300_000, seed=5)
    assert 0.01 < ev["hue_not_rounded_float"] < 0.03  # cv2 hue is not round(float hue) for ~1.7 % of colours
    assert ev["class_flipped"] > 0  # so exact bound colours would be unsafe in general...
    assert ev["mismatch_max_dist_from_rounding_edge"] < 0.2  # ...but the disagreement stays within 0.2 of an edge
    assert ev["max_abs_hue_err"] < 0.7
    assert fam.MARGIN["hue"] >= 5 * ev["mismatch_max_dist_from_rounding_edge"]


def test_metamorphic_symmetries_are_exact_for_reference():
    assert fam.measure_metamorphic_drift(n=60, seed=7_000) == 0


def test_dim_is_not_a_symmetry_of_the_contract():
    """A pixel with V in [V_MIN, 2*V_MIN) leaves its colour when halved: dim only holds on this scene class."""
    red_v120 = cv2.cvtColor(np.array([[[5, 200, 120]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]  # V in [80, 160)
    inside = np.full((20, 20, 3), red_v120, np.uint8)
    assert fam.reference_solve(inside)["red"] == 400
    assert fam.reference_solve(inside >> 1)["red"] == 0
    # and the scenes honour the condition: no in-class pixel in the flip band, every in-class colour even
    for seed in range(40):
        scene = scene_of(seed)
        for sh in scene.shapes:
            h, s, v = fam._hsv_cv(sh.bgr)
            if sh.cls is not None:
                assert v >= 2 * fam.V_MIN and all(c % 2 == 0 for c in sh.bgr), (seed, sh.tag)
            if sh.tag.startswith(f"edge V={fam.V_MIN - 1}"):
                assert v == fam.V_MIN - 1  # the one odd colour: excluded by V before and after halving
            else:
                assert all(c % 2 == 0 for c in sh.bgr), (seed, sh.tag)
        assert all(c % 2 == 0 for c in scene.background)
        assert fam.reference_solve(scene.img >> 1) == fam.reference_solve(scene.img) == scene.truth


# ---------------------------------------------------------------- the verifiers themselves

def _truth_out(scene):
    return dict(scene.truth)


def test_verify_manufactured_is_exact_and_checks_the_contract():
    scene = scene_of(3)
    good = _truth_out(scene)
    assert fam.verify_manufactured(scene, good)
    off = dict(good, green=good["green"] + 1)
    v = fam.verify_manufactured(scene, off)
    assert not v and "green" in v.reason
    assert not fam.verify_manufactured(scene, {c: float(n) for c, n in good.items()})  # counts are ints
    assert not fam.verify_manufactured(scene, {c: n for c, n in good.items() if c != "red"})  # missing key
    assert not fam.verify_manufactured(scene, dict(good, red=True))  # a bool is not a count
    assert not fam.verify_manufactured(scene, {c: 0 for c in fam.COLOURS})  # constant zero


def test_verify_metamorphic_is_exact_and_blind_to_constant_output():
    base = {"red": 10, "yellow": 20, "green": 30, "blue": 40}
    assert fam.verify_metamorphic(base, {"a": dict(base), "b": dict(base)})
    v = fam.verify_metamorphic(base, {"a": dict(base), "dim x0.5": dict(base, blue=39)})
    assert not v and "dim x0.5" in v.reason and "blue 40 -> 39" in v.reason
    assert not fam.verify_metamorphic(base, {"a": dict(base, red=10.0)})  # contract violation, not just a value
    zeros = {c: 0 for c in fam.COLOURS}
    assert fam.verify_metamorphic(zeros, {"a": dict(zeros)})  # the documented stuck-gauge blind spot


def test_transforms_move_pixels_without_changing_them():
    scene = scene_of(11)
    tf = fam.transforms(scene)
    assert list(tf) == ["translate(+7,-5)", "rot90", "mirror", "dim x0.5"]
    H, W = scene.img.shape[:2]
    assert tf["rot90"]["img"].shape == (W, H, 3) and tf["mirror"]["img"].shape == (H, W, 3)
    colours = lambda a: np.unique(a.reshape(-1, 3), axis=0)  # noqa: E731
    base = colours(scene.img)
    for name in ("translate(+7,-5)", "rot90", "mirror"):
        assert np.array_equal(colours(tf[name]["img"]), base), name
    assert tf["dim x0.5"]["img"].dtype == np.uint8 and tf["dim x0.5"]["img"].shape == (H, W, 3)
    bg = np.array(scene.background)
    assert (tf["translate(+7,-5)"]["img"][0, 0] == bg).all() and (tf["translate(+7,-5)"]["img"][-1, -1] == bg).all()


# ---------------------------------------------------------------- the scenes

def test_scene_generation_is_deterministic_and_well_posed():
    a, b = scene_of(3), fam.make_scene(3)
    assert np.array_equal(a.img, b.img) and np.array_equal(a.labels, b.labels)
    assert a.shapes == b.shapes and a.truth == b.truth and a.background == b.background
    assert not np.array_equal(scene_of(4).img, a.img)
    kinds, lobes, bg_modes = set(), set(), set()
    for seed in range(60):
        s = scene_of(seed)
        bodies = [x for x in s.shapes if x.role == "body"]
        dist = [x for x in s.shapes if x.role == "distractor"]
        edges = [x for x in s.shapes if x.role == "edge"]
        dropped = sum(1 for _, c in fam.IN_EDGES if c == s.omitted)  # in-range swatches of an omitted colour
        assert 4 <= len(bodies) <= 7 and 4 <= len(dist) <= 6 and len(edges) == 22 - dropped, seed
        assert {x.tag.split()[0] for x in dist} >= {"gray", "pastel", "dark", "gap"}, seed
        kinds |= {x.kind for x in bodies + dist}
        lobes |= {fam._hsv_cv(x.bgr)[0] >= 170 for x in bodies if x.cls == "red"}
        bg_modes.add("tint" if len(set(s.background)) > 1 else "grey")
        boxes = [x.box for x in s.shapes]
        for i, p in enumerate(boxes):
            for q in boxes[i + 1:]:
                assert p[2] < q[0] or q[2] < p[0] or p[3] < q[1] or q[3] < p[1], "boxes overlap"
        assert all(x.area > 0 for x in s.shapes)
    assert kinds == set(fam.KINDS)
    assert lobes == {True, False}  # both lobes of red occur
    assert bg_modes == {"tint", "grey"}


def test_every_colour_is_sometimes_present_and_sometimes_absent():
    present, absent = {c: 0 for c in fam.COLOURS}, {c: 0 for c in fam.COLOURS}
    for seed in range(200):
        s = scene_of(seed)
        for c in fam.COLOURS:
            present[c] += s.truth[c] > 0
            absent[c] += s.truth[c] == 0
        assert (s.omitted is None) == all(s.truth[c] > 0 for c in fam.COLOURS), seed
        if s.omitted:
            assert s.truth[s.omitted] == 0 and not any(sh.cls == s.omitted for sh in s.shapes)
    assert all(n > 0 for n in present.values()) and all(n > 0 for n in absent.values())
    assert 0.15 <= sum(absent.values()) / 200 <= 0.35  # OMIT_P = 0.25 of scenes lack exactly one colour


def test_image_is_rendered_from_the_labels_with_flat_colours():
    for seed in range(25):
        s = scene_of(seed)
        assert s.img.dtype == np.uint8 and s.img.shape == (*fam.SIZE, 3) and s.img.flags["C_CONTIGUOUS"]
        assert (s.img[s.labels == 0] == np.array(s.background)).all()
        for k, sh in enumerate(s.shapes, 1):
            m = s.labels == k
            assert (s.img[m] == np.array(sh.bgr)).all(), (seed, sh.tag)  # flat colour: no blending, no gradient
            assert int(m.sum()) == sh.area
            ys, xs = np.nonzero(m)
            assert sh.box[0] <= xs.min() and xs.max() <= sh.box[2] and sh.box[1] <= ys.min() and ys.max() <= sh.box[3]
        packed = (s.img[..., 0].astype(np.int32) << 16) | (s.img[..., 1].astype(np.int32) << 8) | s.img[..., 2]
        allowed = {(b << 16) | (g << 8) | r for b, g, r in (s.background, *(sh.bgr for sh in s.shapes))}
        assert set(np.unique(packed).tolist()) <= allowed


def test_truth_is_the_label_pixel_count_per_colour():
    for seed in range(40):
        s = scene_of(seed)
        assert set(s.truth) == set(fam.COLOURS)
        counts = np.bincount(s.labels.ravel(), minlength=len(s.shapes) + 1)
        for c in fam.COLOURS:
            assert s.truth[c] == sum(int(counts[k]) for k, sh in enumerate(s.shapes, 1) if sh.cls == c)
        assert all(isinstance(n, int) for n in s.truth.values())


def test_analytic_areas_agree_with_the_raster_labels():
    """The label counts are not circular: closed-form areas reproduce them (exactly where the geometry is integer)."""
    for seed in range(40):
        for sh in scene_of(seed).shapes:
            if sh.kind in ("rect", "ring", "speckle"):
                assert sh.area == sh.analytic, (seed, sh.kind)
            elif sh.kind == "disk":
                r = math.sqrt(sh.analytic / math.pi)
                # measured: (raster area - pi r^2) / perimeter in [-0.151, +0.078] for r = 4..32
                assert abs(sh.area - sh.analytic) <= 0.25 * 2 * math.pi * r, (seed, sh.area, sh.analytic)


def test_scene_colours_classify_identically_everywhere():
    """The intended class of every shape is what cv2's fixed point, the float formula and a dimmed copy all say."""
    for seed in range(60):
        for sh in scene_of(seed).shapes:
            for bgr in (sh.bgr, tuple(c >> 1 for c in sh.bgr)):
                assert fam.classify(*fam._hsv_cv(bgr)) == sh.cls, (seed, sh.tag, bgr)
            hf, sf, vf = fam._hsv_f(sh.bgr)
            assert fam.classify(math.floor(hf + 0.5) % 180, math.floor(sf + 0.5), vf) == sh.cls, (seed, sh.tag)
            if sh.role == "body":  # comfortably inside, not on a rounding edge
                assert sf >= fam.S_MIN + fam.MARGIN["sat"] and vf >= 2 * fam.V_MIN + fam.MARGIN["val"]


def test_edge_chart_covers_every_bound_of_the_contract():
    in_hues = sorted({b for spans in fam.HUE.values() for lo, hi in spans for b in (lo, hi)})
    out_hues = sorted({b for spans in fam.HUE.values() for lo, hi in spans for b in (lo - 1, hi + 1)
                       if 0 <= b <= 179 and not any(l <= b <= h for sp in fam.HUE.values() for l, h in sp)})
    assert sorted(h for h, _ in fam.IN_EDGES) == in_hues
    assert sorted(fam.OUT_EDGES) == out_hues
    for h, cls in fam.IN_EDGES:
        assert fam.classify(h, 120, 254) == cls
    for h in fam.OUT_EDGES:
        assert fam.classify(h, 120, 254) is None
    for seed in range(40):
        s = scene_of(seed)
        edge = {sh.tag: sh for sh in s.shapes if sh.role == "edge"}
        hues_in = sorted(fam._hsv_cv(sh.bgr)[0] for sh in edge.values()
                         if sh.tag.startswith("edge H=") and sh.tag.split()[2] == "in")
        hues_out = sorted(fam._hsv_cv(sh.bgr)[0] for sh in edge.values()
                          if sh.tag.startswith("edge H=") and sh.tag.split()[2] == "out")
        assert hues_in == sorted(h for h, c in fam.IN_EDGES if c != s.omitted), seed  # all of them, or minus a colour
        assert hues_out == out_hues, seed  # the out-of-range hues are in every scene, omitted colour or not
        sat_in = next(sh for t, sh in edge.items() if t.startswith(f"edge S={fam.S_MIN} in"))
        sat_out = next(sh for t, sh in edge.items() if t.startswith(f"edge S={fam.S_MIN - 1} out"))
        val_in = next(sh for t, sh in edge.items() if t.startswith(f"edge V={2 * fam.V_MIN} in"))
        val_out = next(sh for t, sh in edge.items() if t.startswith(f"edge V={fam.V_MIN - 1} out"))
        assert fam._hsv_cv(sat_in.bgr)[1] == fam.S_MIN and sat_in.cls not in (None, s.omitted)
        assert fam._hsv_cv(sat_out.bgr)[1] == fam.S_MIN - 1 and sat_out.cls is None
        assert fam._hsv_cv(val_in.bgr)[2] == 2 * fam.V_MIN and val_in.cls not in (None, s.omitted)
        assert fam._hsv_cv(val_out.bgr)[2] == fam.V_MIN - 1 and val_out.cls is None


def _threshold_solver(spans, s_min=fam.S_MIN, v_min=fam.V_MIN):
    """A solution with the contract's structure and chosen bounds: in-process, for probing the verifier only."""
    def solve(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out = {}
        for name, ranges in spans.items():
            mask = np.zeros(img.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
            out[name] = int(cv2.countNonZero(mask))
        return out
    return solve


def _caught(solver, n_scenes=20, n_meta=5):
    scenes = [scene_of(i) for i in range(n_scenes)]
    if any(not fam.verify_manufactured(s, solver(s.img)) for s in scenes):
        return "manufactured"
    for s in scenes[:n_meta]:
        tf = fam.transforms(s)
        if not fam.verify_metamorphic(solver(s.img), {k: solver(t["img"]) for k, t in tf.items()}):
            return "metamorphic"
    return None


def test_every_unit_perturbation_of_a_bound_is_caught():
    """Resolution of the verifier: moving any single bound of the contract by one unit fails a gate."""
    assert _caught(_threshold_solver({c: list(s) for c, s in fam.HUE.items()})) is None  # control: the contract
    for name, spans in fam.HUE.items():
        for i, (lo, hi) in enumerate(spans):
            for j, bound in enumerate((lo, hi)):
                for new in (bound - 1, bound + 1):
                    if not 0 <= new <= 179:
                        continue
                    moved = {c: list(s) for c, s in fam.HUE.items()}
                    moved[name][i] = (new, hi) if j == 0 else (lo, new)
                    assert _caught(_threshold_solver(moved)) == "manufactured", (name, i, j, new)
    hue = {c: list(s) for c, s in fam.HUE.items()}
    for new in (fam.S_MIN - 1, fam.S_MIN + 1):
        assert _caught(_threshold_solver(hue, s_min=new)) == "manufactured", f"S_MIN {new}"
    assert _caught(_threshold_solver(hue, v_min=fam.V_MIN - 1)) == "manufactured"  # the V = 79 swatch
    assert _caught(_threshold_solver(hue, v_min=fam.V_MIN + 1)) == "metamorphic"  # V = 160 -> 80 under dim only


# ---------------------------------------------------------------- the prompt

def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img) -> dict" in p and "OpenCV: 5.0" in p and "170-179" in p
    for c in fam.COLOURS:
        assert f"'{c}'" in p
    assert [fam.SPEC.outputs[c].dtype for c in fam.COLOURS] == ["int"] * 4


def test_every_phrasing_states_the_whole_contract():
    assert len(fam.PHRASINGS) == 16 and fam.PHRASINGS[0] == fam.SPEC.task
    assert len(set(fam.PHRASINGS)) == len(fam.PHRASINGS)
    for text in fam.PHRASINGS:
        for needle in ("0-10", "170-179", "20-35", "40-80", "100-130", f"S >= {fam.S_MIN}", f"V >= {fam.V_MIN}",
                       "0-179", "inclusive"):
            assert needle in text, (needle, text[:60])
        assert "BGR" in text
        assert all(colour in text for colour in fam.COLOURS), text[:60]
