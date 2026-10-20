import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import text_layout as fam
from sandbox.run import run_solution

# Expected verdicts per gate: this is the family's claim, pinned as a regression test.
EXPECTED = {
    #                           exec-only lint   manufactured metamorphic
    "reference":                (True,     True,  True,        True),
    "int_rounding (harmless)":  (True,     True,  True,        True),
    "ink_centred (harmless)":   (True,     True,  True,        True),
    "hardcoded_4x_metrics":     (True,     True,  False,       False),
    "baseline_ignored":         (True,     True,  False,       True),
    "wh_swapped":               (True,     True,  False,       True),
    "float_origin":             (False,    True,  False,       False),
    "swallowed_float_origin":   (True,     False, False,       True),   # lint: `except cv2.error: pass` is swallowed-except (not broad)
    "no_shrink":                (True,     True,  False,       False),
    "top_left_origin":          (True,     True,  False,       False),
    "proportional_shrink":      (True,     True,  False,       True),
    "fontface_api":             (True,     True,  False,       True),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")
HARMLESS = ("reference", "int_rounding (harmless)", "ink_centred (harmless)")

# The measurements recorded next to the constants in text_layout.py (OpenCV 5.0.0.93, 4000 gauge blocks).
RECORDED = {"tau_x": 0.052, "tau_y": 0.143, "size_w": (0.059, 0.196), "size_h": (0.118, 0.300)}

PROBES = Path(__file__).resolve().parents[1] / "probes" / "results"


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    with ThreadPoolExecutor(max_workers=6) as pool:  # each evaluation is mostly waiting on sandbox subprocesses
        futures = {name: pool.submit(evaluate_solution, fam, code, n_scenes=20, n_meta=5)
                   for name, (_, code) in candidates.items()}
        return {name: f.result() for name, f in futures.items()}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_every_candidate_is_in_the_matrix_and_differs_from_reference():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n not in HARMLESS]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == len(wrong) - 1  # only float_origin raises
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)


def test_swallowed_float_origin_is_caught_by_swallowed_except_not_by_broad_except(matrix):
    detail = matrix["swallowed_float_origin"]["lint"].detail
    assert "swallowed-except" in detail and "broad-except" not in detail  # `except cv2.error` is specific


def test_the_two_verifier_gates_catch_different_mutants(matrix):
    man = {n for n in EXPECTED if n not in HARMLESS and not matrix[n]["manufactured"].passed}
    meta = {n for n in EXPECTED if n not in HARMLESS and not matrix[n]["metamorphic"].passed}
    assert man > meta  # the manufactured gate catches strictly more, the metamorphic one only mis-placements
    assert meta == {"hardcoded_4x_metrics", "float_origin", "no_shrink", "top_left_origin"}


# ---------------------------------------------------------------- reference and the tolerance law

def test_reference_passes_many_manufactured_scenes_in_process():
    for seed in range(200, 300):
        scene = fam.make_scene(seed)
        v = fam.verify_manufactured(scene, fam.reference_solve(**fam.inputs(scene)))
        assert v, (seed, v.reason)


def test_calibration_still_covers_fresh_gauge_blocks():
    s = fam.calibrate(n=500, seed=999, n_scenes=40, scene_seed=90_000, n_side=0)
    tall = s["tall"]
    # the constants are the recorded measurements plus a margin of at least 20 %
    assert fam.TOL_X >= 1.2 * RECORDED["tau_x"] and fam.TOL_Y >= 1.2 * RECORDED["tau_y"]
    assert fam.SIZE_W[0] >= 1.2 * RECORDED["size_w"][0] and fam.SIZE_W[1] >= 1.2 * RECORDED["size_w"][1]
    assert fam.SIZE_H[0] >= 1.2 * RECORDED["size_h"][0] and fam.SIZE_H[1] >= 1.2 * RECORDED["size_h"][1]
    # and cover a fresh sample of the same population
    assert tall["centre"]["metric_box"]["tau_x"] < fam.TOL_X
    assert tall["centre"]["metric_box"]["tau_y"] < fam.TOL_Y
    assert tall["size_w"]["lo"] < fam.SIZE_W[0] and tall["size_w"]["hi"] < fam.SIZE_W[1]
    assert tall["size_h"]["lo"] < fam.SIZE_H[0] and tall["size_h"]["hi"] < fam.SIZE_H[1]
    # the ink-centred idiom is exact to the pixel quantum: TOL_C0 (1 px) is two quanta
    assert tall["centre"]["ink_centred"]["max_abs_px"] <= 0.5 < fam.TOL_C0
    # layer 2 on scenes: the reference draws what putText draws, the colour model holds
    sc = s["scenes"]
    assert sc["reference_fails_verifier"] == 0
    assert sc["dim_diff_vs_render_max"] <= fam.DIM_TOL
    assert sc["peak_alpha_min"] > fam.ALPHA_MIN and sc["peak_alpha_spread_max"] < fam.ALPHA_SPREAD
    assert 0.9 < sc["shrink_scale_over_s_fit"][0] and sc["shrink_scale_over_s_fit"][1] < 1.05
    assert tall["peak_coverage_min"] > fam.ALPHA_MIN


def test_the_physics_behind_the_centring_band():
    """The textbook idiom is correct but leaves the 5.0 ink LOW (line height 27 px > cap height 21 px at scale 1)."""
    rng = np.random.default_rng(31)
    errs, ihs, base_gap = [], [], []
    for _ in range(150):
        face = fam.FACES[int(rng.integers(len(fam.FACES)))]
        thick, scale, text = int(rng.integers(1, 4)), round(float(rng.uniform(0.5, 2.5)), 2), fam._rand_text(rng)
        if len(text) < 1 or not fam._reaches_cap(text, face, scale, thick):
            continue
        g = fam._gauge(text, face, scale, thick)
        errs.append(g["err"]["metric_box"][1])
        ihs.append(g["ih"])
        base_gap.append(g["err"]["baseline_ignored"][1] - g["err"]["metric_box"][1])
    assert min(errs) > 0, "the metric box centres the ink too high everywhere: the ink is always low"
    assert 0.08 < np.mean(np.array(errs) / np.array(ihs)) < 0.16  # about 0.12 of the ink height on average
    # ignoring the baseline moves the ink down by a further base/2, never up
    assert min(base_gap) >= 0 and max(base_gap) > 2


def test_5_0_text_metrics_that_the_docstring_relies_on():
    (w, h), base = cv2.getTextSize("OpenCV 5", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    assert ((w, h), base) == ((136, 27), 6)  # the 5.0.0 probe value; 4.14 gave ((155, 22), 10)
    # the height is the font's line height, not the string's: the same for a string with and without descenders
    h_flat = cv2.getTextSize("HHH", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][1]
    h_desc = cv2.getTextSize("gypsy", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][1]
    assert h_flat == h_desc == 27
    # the baseline is the string's own descent
    assert cv2.getTextSize("HHH", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[1] == 0
    assert cv2.getTextSize("gypsy", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[1] >= 6
    # cap height 21 px: the 6 px of headroom are what leaves centred ink low
    cap = -fam._ink_box("H", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[1]
    assert cap == 21 and h_flat - cap == 6


def test_ink_leaves_the_text_size_box_only_on_the_left_by_a_leading_j():
    """The overhang figures recorded in text_layout.py: left side only, a leading 'j' only, at most 5 px."""
    rng = np.random.default_rng(7_000_000)
    n, worst, hooked = 0, 0, []
    while n < 400:
        scale, thick = round(float(rng.uniform(0.5, 2.5)), 2), int(rng.integers(1, 4))
        text, face = fam._rand_text(rng), fam.FACES[int(rng.integers(len(fam.FACES)))]
        if len(text) < 1 or not fam._reaches_cap(text, face, scale, thick):
            continue
        n += 1
        x0, y0, x1, y1 = fam._ink_box(text, face, scale, thick)
        (tw, th), base = cv2.getTextSize(text, face, scale, thick)  # the box: x in [0, tw-1], y in [-th, base-1]
        assert x1 <= tw - 1 and y0 >= -th and y1 <= base - 1, (text, face, scale, thick)
        if x0 < 0:
            hooked.append((text[0], -x0))
        worst = max(worst, -x0)
    assert hooked and {c for c, _ in hooked} == {"j"}, hooked
    assert worst <= 5


def test_mutant_failure_rates_are_measured_by_a_function_and_the_fitting_scenes_never_fail():
    r = fam.measure_mutant_failure_rates(n=80, seed=7_000_000)
    assert r["seeds"] == (7_000_000, 7_000_079)
    for name in ("baseline_ignored", "proportional_shrink"):
        assert r[name]["all"][1] == 80 and r[name]["fits"][1] == r[name]["must_shrink"][1] == 40
        assert r[name]["all"][0] == r[name]["fits"][0] + r[name]["must_shrink"][0]
    # proportional_shrink only differs from the reference when the width overflows, so a scene that fits cannot fail it
    assert r["proportional_shrink"]["fits"][0] == 0 and r["proportional_shrink"]["must_shrink"][0] > 0
    assert r["baseline_ignored"]["all"][0] > 0


def test_putText_ignores_lineType_for_hershey_fonts_on_5_0():
    """Sentinel for the finding behind the coverage-based colour check: if OpenCV starts honouring lineType this fails."""
    for face in fam.FACES:
        for scale, thick in ((0.6, 1), (1.0, 2), (2.0, 3)):
            ims = []
            for lt in (cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA):
                im = np.zeros((120, 400), np.uint8)
                cv2.putText(im, "Hello World 123", (10, 80), face, scale, 255, thick, lt)
                ims.append(im)
            assert np.array_equal(ims[0], ims[1]) and np.array_equal(ims[1], ims[2])
            assert len(np.unique(ims[0])) > 50  # anti-aliased: many grey levels, not two


def test_recorded_probes_agree_with_the_4x_constants_of_the_hardcoded_mutant():
    old = json.loads((PROBES / "probe_4.14.0.json").read_text())["probes"]
    new = json.loads((PROBES / "probe_5.0.0.json").read_text())["probes"]
    (w4, h4), b4 = (old["getTextSize"]["obs"]["tuple"][0]["tuple"], old["getTextSize"]["obs"]["tuple"][1])
    assert (w4, h4, b4) == (155, 22, 10)
    code = fam.MUTANTS["hardcoded_4x_metrics"][1]
    assert "19.4 * len(text)" in code and "int(22 * s)" in code and "int(10 * s)" in code
    assert int(19.4 * len("OpenCV 5")) == w4
    assert new["getTextSize"]["obs"]["tuple"][0]["tuple"] == [136, 27]
    # 4.14 drew LINE_8 text with two grey levels, 5.0 does not
    o4, o5 = old["putText_hershey_pixels"]["obs"], new["putText_hershey_pixels"]["obs"]
    assert o4["sum"] == 255 * o4["nonzero"] and o5["sum"] != 255 * o5["nonzero"]


# ---------------------------------------------------------------- the verifier, check by check

@pytest.fixture(scope="module")
def good():
    scenes = {False: fam.make_scene(100), True: fam.make_scene(101)}  # even seed: fits, odd seed: must shrink
    return {kind: (sc, fam.reference_solve(**fam.inputs(sc))) for kind, sc in scenes.items()}


@pytest.mark.parametrize("kind", [False, True], ids=["fits", "must-shrink"])
def test_each_layer_rejects_its_own_defect(good, kind):
    sc, ok = good[kind]
    assert fam.verify_manufactured(sc, ok)
    x, y, w, h = sc.box

    def bad(**changes):
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ok.items()}
        out.update(changes)
        return fam.verify_manufactured(sc, out)

    # contract
    assert "contract" in bad(text_size=np.array([1.5, 2.5])).reason
    assert "nothing was drawn" in bad(img=sc.img.copy()).reason
    assert "shape" in bad(img=ok["img"][:-1]).reason
    # 1. containment: one changed pixel outside the box
    outside = ok["img"].copy()
    outside[0, 0] ^= 1
    assert "outside the box" in bad(img=outside).reason
    # 4. scale rule
    assert "scale" in bad(scale=ok["scale"] * 0.5).reason
    assert "scale" in bad(scale=0.0).reason
    # 5. glyph size: right place and scale claim, but the text is drawn smaller than claimed
    small = sc.img.copy()
    (tw, th), base = cv2.getTextSize(sc.text, sc.face, ok["scale"] * 0.8, sc.thickness)
    cv2.putText(small, sc.text, (x + (w - tw) // 2, y + (h - th - base) // 2 + th), sc.face, ok["scale"] * 0.8,
                tuple(sc.color), sc.thickness)
    assert "putText draws" in bad(img=small).reason
    # 2. centring: the reference drawing moved by 40 % of the line height inside a roomy box is still contained
    #    but not centred (use a copy of the scene with a much larger box so containment cannot fire first)
    big = fam.Scene(np.full((sc.img.shape[0] + 400, sc.img.shape[1] + 400, 3), sc.img[0, 0], np.uint8),
                    (x + 200, y + 200, w + 200, h + 200), sc.text, sc.face, ok["scale"], sc.thickness, sc.color,
                    False, ok["scale"], sc.seed)
    mv = fam.reference_solve(**fam.inputs(big))
    assert fam.verify_manufactured(big, mv)
    shifted = np.roll(mv["img"], (0, int(0.4 * mv["text_size"][1])), axis=(1, 0))
    r = fam.verify_manufactured(big, {**mv, "img": shifted})
    assert not r and "ink centre" in r.reason
    # 3. reported size: swapped, and the 4.14 numbers
    assert "reported width" in bad(text_size=ok["text_size"][::-1].copy()).reason
    assert "reported" in bad(text_size=np.array([ok["text_size"][0] + 20, ok["text_size"][1]])).reason
    # 6. colour: the same drawing in the channel-reversed colour
    wrong = sc.img.copy()
    (tw, th), base = cv2.getTextSize(sc.text, sc.face, ok["scale"], sc.thickness)
    cv2.putText(wrong, sc.text, (x + (w - tw) // 2, y + (h - th - base) // 2 + th), sc.face, ok["scale"],
                tuple(sc.color[::-1]), sc.thickness)
    r = bad(img=wrong)
    assert not r and "colour" in r.reason


def test_a_text_that_does_not_shrink_enough_or_shrinks_too_much_is_rejected(good):
    sc, ok = good[True]
    assert "scale" in fam.verify_manufactured(sc, {**ok, "scale": sc.scale}).reason  # claims no shrink
    r = fam.verify_manufactured(sc, {**ok, "scale": 0.89 * sc.s_fit})
    assert not r and "scale" in r.reason  # below the 10 % floor
    sc, ok = good[False]
    r = fam.verify_manufactured(sc, {**ok, "scale": sc.scale * 0.95})
    assert not r and "fits at scale" in r.reason  # the text fits: shrinking is wrong too


def test_baseline_ignored_fails_only_where_there_is_a_descender():
    scenes = [fam.make_scene(2000 + i) for i in range(70)]
    inp = [fam.inputs(s) for s in scenes]
    ref = run_solution(fam.REFERENCE, inp, lint=False)
    mut = run_solution(fam.MUTANTS["baseline_ignored"][1], inp, lint=False)
    assert ref.ok and mut.ok
    failing = []
    for s, a, b in zip(scenes, ref.outputs, mut.outputs):
        assert fam.verify_manufactured(s, a)
        base = cv2.getTextSize(s.text, s.face, a["scale"], s.thickness)[1]
        if not fam.verify_manufactured(s, b):
            failing.append(base)
    same = sum(np.array_equal(a["img"], b["img"]) for a, b in zip(ref.outputs, mut.outputs))
    # a font-agnostic band cannot see a shift smaller than its width: it fails the mutant only when the descender is
    # deep enough, which is exactly when the output differs from the right one by more than the band
    assert len(failing) >= 10 and min(failing) >= 2
    assert same >= 20  # and for descender-free strings the two solutions are the same picture


def test_int_rounding_moves_the_shift_by_one_pixel_so_META_POS_is_tight():
    scenes = [fam.make_scene(3000 + i) for i in range(60)]
    batch = []
    for s in scenes:
        batch += [fam.inputs(s), fam.transforms(s)[fam.NAME_SHIFT]]
    res = run_solution(fam.MUTANTS["int_rounding (harmless)"][1], batch, lint=False)
    assert res.ok
    errs = Counter()
    for i in range(len(scenes)):
        a, b = res.outputs[2 * i], res.outputs[2 * i + 1]
        ba, bb = fam._bbox(fam._ink_mask(a["img"])), fam._bbox(fam._ink_mask(b["img"]))
        errs[max(abs(bb[0] - ba[0] - fam.SHIFT[0]), abs(bb[1] - ba[1] - fam.SHIFT[1]))] += 1
    assert set(errs) == {0, 1}, errs  # banker's rounding breaks exact covariance by one pixel, never more
    assert fam.META_POS == 1.0


def test_metamorphic_symmetries_hold_for_the_reference():
    d = fam.measure_metamorphic_drift(n=60, seed=7_000)
    assert d["shift_pos"] == 0 and d["shift_dim"] == 0
    assert d["scale_increase"] == 0 and d["narrowed"] == 0
    assert d["longer_centre_ratio"] < 0.6  # the reference uses well under its allowance


def test_metamorphic_is_blind_to_a_stuck_output(matrix):
    """The usual blind spot, pinned: nothing drawn is invariant under every symmetry (the manufactured gate has it)."""
    g = matrix["swallowed_float_origin"]
    assert g["metamorphic"].passed and not g["manufactured"].passed


def test_metamorphic_rejects_a_wrong_symmetry_by_itself():
    sc = fam.make_scene(4)
    base = fam.reference_solve(**fam.inputs(sc))
    t = fam.transforms(sc)
    shifted = fam.reference_solve(**t[fam.NAME_SHIFT])
    longer = fam.reference_solve(**t[fam.NAME_LONGER])
    assert fam.verify_metamorphic(base, {fam.NAME_SHIFT: shifted, fam.NAME_LONGER: longer})
    # an ink that did not follow the box
    assert "ink moved" in fam.verify_metamorphic(base, {fam.NAME_SHIFT: base}).reason
    # a longer string that was given a larger scale than the original
    assert "larger scale" in fam.verify_metamorphic(
        {**base, "scale": longer["scale"] * 0.5}, {fam.NAME_LONGER: longer}).reason
    assert "unknown variant" in fam.verify_metamorphic(base, {"rot90": base}).reason


# ---------------------------------------------------------------- scenes

def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.img, b.img)
    assert all(getattr(a, f) == getattr(b, f) for f in ("box", "text", "face", "scale", "thickness", "color", "shrink",
                                                       "s_fit", "seed"))
    assert fam.make_scene(3).text != fam.make_scene(5).text or fam.make_scene(3).box != fam.make_scene(5).box
    assert {fam.make_scene(s).shrink for s in range(5)} == {False, True}  # the metamorphic gate sees both kinds
    faces, shapes = set(), Counter()
    for seed in range(80):
        s = fam.make_scene(seed)
        x, y, w, h = s.box
        H, W = s.img.shape[:2]
        assert s.shrink == bool(seed % 2)
        assert x >= fam.MARGIN and y >= fam.MARGIN and x + w <= W - fam.MARGIN and y + h <= H - fam.MARGIN
        assert (s.img == s.img[0, 0]).all(), "uniform background"
        assert all(abs(int(c) - int(b)) >= 100 for c, b in zip(s.color, s.img[0, 0])), "contrast in every channel"
        assert s.face in fam.FACES and 3 <= len(s.text) <= 22 and s.text == s.text.strip()
        assert all(32 <= ord(c) < 127 for c in s.text)
        assert fam._reaches_cap(s.text, s.face, s.s_fit, s.thickness), "population: ink reaches the cap height"
        faces.add(s.face)
        (tw, th), base = cv2.getTextSize(s.text, s.face, s.scale, s.thickness)
        if not s.shrink:
            assert s.s_fit == s.scale
            assert w - tw >= 6 and h - th - base >= 6  # >= 3 px each side: every notion of "fits" agrees
        else:
            iw, ih = fam._ink_dims(s.text, s.face, s.scale, s.thickness)
            assert iw >= 1.1 * w or ih >= 1.1 * h  # the ink overflows by >= 10 % at the requested scale
            assert fam._fits(s.text, s.face, s.s_fit, s.thickness, w, h)
            assert not fam._fits(s.text, s.face, round(s.s_fit + 0.01, 2), s.thickness, w, h)  # s_fit is the largest
            assert 0.5 <= s.s_fit < s.scale
            (fw, fh), fbase = cv2.getTextSize(s.text, s.face, s.s_fit, s.thickness)
            roomy = ("width " if w - fw > 3 else "") + ("height" if h - fh - fbase > 3 else "")
            shapes[roomy.strip() or "tight"] += 1
    assert faces == set(fam.FACES)
    assert set(shapes) == {"tight", "width", "height"}  # tight in both, roomy in the width, roomy in the height


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img, box, text, face, scale, thickness, color) -> dict" in p and "OpenCV: 5.0" in p
    assert "centred" in p and "getTextSize" in p and "Never change a pixel outside the box" in p
    assert 3 <= len(fam.PHRASINGS) <= 5 and fam.PHRASINGS[0] == fam.SPEC.task
    for ph in fam.PHRASINGS:
        assert all(word in ph for word in ("scale", "thickness", "color", "getTextSize")), ph
        assert any(word in ph for word in ("box", "rectangle", "region")), ph
