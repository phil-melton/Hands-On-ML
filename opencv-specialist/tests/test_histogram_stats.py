from decimal import Decimal, getcontext

import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import histogram_stats as fam

# Expected verdicts per gate: this is the family's claim, pinned as a regression test.
# Nothing passes every gate, so unlike contour_area there is no harmless mutant here: rgb2gray is wrong
# because the scenes are coloured. The metamorphic gate fires on everything the manufactured gate does
# (no mutant isolates it, see test_every_mutant_is_caught_twice); bins_255 and float_counts fail it through
# the shared contract check, hist_col0 because the run crashes.
EXPECTED = {
    #                      exec-only lint   manufactured metamorphic
    "reference":           (True,     True,  True,        True),
    "hist_col0":           (False,    True,  False,       False),
    "range_255":           (True,     True,  False,       False),
    "bins_255":            (True,     True,  False,       False),
    "no_gray_conversion":  (True,     True,  False,       False),
    "float_counts":        (True,     True,  False,       False),
    "rgb2gray":            (True,     True,  False,       False),
    "gray_truncate":       (True,     True,  False,       False),
    "std_ddof1":           (True,     True,  False,       False),
    "median_np_int":       (True,     True,  False,       False),
    "median_strict":       (True,     True,  False,       False),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")

# Correct solutions that share no code path with REFERENCE (np.bincount / np.histogram, sorted pixels /
# np.percentile, cv2.meanStdDev / hand-rolled moments): the tolerance encodes the physics, not one API.
ALT_BINCOUNT = '''
import cv2
import numpy as np


def solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256)
    mean, std = cv2.meanStdDev(gray)
    pixels = np.sort(gray.ravel())
    return {"hist": hist, "mean": float(mean[0, 0]), "std": float(std[0, 0]),
            "median": int(pixels[(pixels.size + 1) // 2 - 1])}
'''
ALT_NUMPY = '''
import cv2
import numpy as np


def solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    levels = np.arange(256)
    mean = float((levels * hist).sum() / hist.sum())
    std = float(np.sqrt((((levels - mean) ** 2) * hist).sum() / hist.sum()))
    return {"hist": hist, "mean": mean, "std": std,
            "median": int(np.percentile(gray, 50, method="inverted_cdf"))}
'''


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    return {name: evaluate_solution(fam, code, n_scenes=20, n_meta=5) for name, (_, code) in candidates.items()}


@pytest.fixture(scope="module")
def cal():
    return fam.calibrate(n=150, seed=777)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_matrix_covers_every_mutant():
    assert set(fam.MUTANTS) == set(EXPECTED) - {"reference"}
    assert 5 <= len(fam.MUTANTS) <= 10


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n != "reference"]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == 9  # only the 4.x hist[:, 0] idiom crashes
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)


def test_every_mutant_is_caught_twice(matrix):
    """Independent detectors agree on this family: record it, so a verifier change that breaks one shows up."""
    for n in set(EXPECTED) - {"reference"}:
        assert not matrix[n]["manufactured"].passed and not matrix[n]["metamorphic"].passed, n


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


@pytest.mark.parametrize("code", [ALT_BINCOUNT, ALT_NUMPY], ids=["bincount+meanStdDev", "histogram+percentile"])
def test_equivalent_implementations_pass_all_gates(code):
    gates = evaluate_solution(fam, code, n_scenes=20, n_meta=5)
    assert all(g.passed for g in gates.values()), {k: g.detail for k, g in gates.items()}


def test_reference_passes_many_manufactured_scenes_in_process():
    for seed in range(200, 400):
        scene = fam.make_scene(seed)
        v = fam.verify_manufactured(scene, fam.reference_solve(scene.img))
        assert v, (seed, v.reason)


# ---------------------------------------------------------------- the 5.0 premise

def test_premise_calcHist_is_1d_in_5_0():
    g = np.random.default_rng(0).integers(0, 256, (40, 50), dtype=np.uint8)
    g[0, :3] = 255
    h = cv2.calcHist([g], [0], None, [256], [0, 256])
    assert h.shape == (256,) and h.dtype == np.float32  # (256, 1) on 4.14; floats, so the contract's ints need a cast
    with pytest.raises(IndexError):
        h[:, 0]
    bc = np.bincount(g.ravel(), minlength=256)
    assert np.array_equal(h.astype(np.int64), bc)
    h255 = cv2.calcHist([g], [0], None, [256], [0, 255]).ravel()  # exclusive upper bound: level 255 is dropped
    assert np.flatnonzero(h255 != bc).tolist() == [255] and h255[255] == 0
    assert cv2.calcHist([g], [0], None, [255], [0, 256]).shape == (255,)


# ---------------------------------------------------------------- tolerance law and its evidence

def test_tolerance_constants_are_pinned():
    assert fam.COUNT_TOL == 0
    assert fam.FLOAT_RTOL == 1e-9
    assert fam.META_RTOL == fam.FLOAT_RTOL
    assert fam.TIE_MARGIN == 0.05


def test_cvtColor_matches_exact_luma_outside_the_tie_margin(cal):
    c = cal["cvtColor_vs_exact_luma"]
    assert c["colours"] == 2 ** 24
    assert c["max_level_error"] <= 1
    assert c["mismatches_at_or_beyond_TIE_MARGIN"] == 0
    assert 10 * c["max_tie_distance_of_a_mismatch"] <= fam.TIE_MARGIN  # measured 0.003, margin 0.05


def test_counts_are_exact_so_their_tolerance_is_zero(cal):
    assert cal["gauge_constant_images"]["max_abs_count_error"] == 0
    assert cal["scenes"]["max_abs_count_error"] == 0
    assert cal["scenes"]["gray_plane_mismatches_vs_cvtColor"] == 0
    hidden = cal["range_255_hidden_by_slack"]
    assert hidden["abs_1_count"] > 0.02 and hidden["rel_1_percent_of_pixels"] > 0.4  # what slack would cost


def test_float_tolerance_sits_between_roundoff_and_convention_effects(cal):
    assert cal["worst_float64_rel_error"] * 1e3 < fam.FLOAT_RTOL  # measured 1.2e-13 (cv2.meanStdDev)
    assert cal["ddof1_min_rel_effect"] > 1e3 * fam.FLOAT_RTOL  # measured 2.7e-5
    # a documented consequence of the law: float32 arithmetic (~1e-7) does not pass
    assert cal["float_rel_error"]["numpy float32"] > fam.FLOAT_RTOL


def test_metamorphic_symmetries_are_exact_for_reference():
    drift = fam.measure_metamorphic_drift(n=40, seed=7_000)
    assert drift["count"] == 0 and drift["median"] == 0
    assert drift["mean"] < fam.META_RTOL * 1e-3 and drift["std"] < fam.META_RTOL * 1e-3


# ---------------------------------------------------------------- scene generation

def test_scene_generation_is_deterministic():
    a, b = fam.make_scene(3), fam.make_scene(3)
    for f in ("img", "gray", "hist"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert (a.mean, a.std, a.median, a.median_upper, a.kind) == (b.mean, b.std, b.median, b.median_upper, b.kind)
    ta, tb = fam.transforms(a), fam.transforms(b)
    assert list(ta) == list(tb)
    assert all(np.array_equal(ta[k]["img"], tb[k]["img"]) for k in ta)
    c = fam.make_scene(4)
    assert c.img.shape != a.img.shape or not np.array_equal(c.img, a.img)


def test_scene_is_well_posed_and_the_answer_is_exact():
    getcontext().prec = 60
    kinds, ties, wide_ties, odd_boundary, chroma = set(), 0, 0, 0, []
    for seed in range(200):
        s = fam.make_scene(seed)
        H, W = s.img.shape[:2]
        assert s.img.dtype == np.uint8 and s.img.shape == (H, W, 3) and 40 <= H <= 120 and 40 <= W <= 160
        # the manufactured answer is what OpenCV actually sees ...
        gray = cv2.cvtColor(s.img, cv2.COLOR_BGR2GRAY)
        assert np.array_equal(gray, s.gray), seed
        assert np.array_equal(np.bincount(gray.ravel(), minlength=256), s.hist), seed
        assert s.hist.sum() == H * W and s.hist[0] >= 1 and s.hist[255] >= 1  # saturated levels are occupied
        # ... and its statistics are exact: re-derived here from the pixels and from 60-digit arithmetic
        pixels = np.sort(s.gray.ravel())
        assert s.median == pixels[(s.n + 1) // 2 - 1] and s.median_upper == pixels[s.n // 2]
        n, s1 = s.n, int(pixels.astype(np.int64).sum())
        s2 = int((pixels.astype(np.int64) ** 2).sum())
        mean, std = Decimal(s1) / Decimal(n), (Decimal(n * s2 - s1 * s1) / Decimal(n * n)).sqrt()
        assert abs(Decimal(s.mean) - mean) <= Decimal("3e-16") * mean
        assert abs(Decimal(s.std) - std) <= Decimal("3e-16") * std
        kinds.add(s.kind)
        ties += s.tie
        wide_ties += s.tie and s.median_upper - s.median >= 2
        cum = np.cumsum(s.hist)
        odd_boundary += s.n % 2 == 1 and int(np.searchsorted(cum, s.n // 2)) != s.median  # n // 2 convention differs
        chroma.append(float(np.mean(s.img.max(2) != s.img.min(2))))
    assert kinds == set(fam.KINDS)
    assert np.mean(chroma) > 0.9  # coloured scenes: channel order and luma weights matter
    # the median convention is exercised: even-N ties, ties with a gap of 2+ levels, odd-N boundaries
    assert ties >= 80 and wide_ties >= 50 and odd_boundary >= 15, (ties, wide_ties, odd_boundary)


def test_default_evaluation_sets_contain_median_ties():
    """evaluate_solution uses seeds 0..19 (manufactured) and 0..4 (metamorphic): the median checks must bite there."""
    scenes = [fam.make_scene(i) for i in range(20)]
    assert sum(s.tie for s in scenes) >= 10
    assert sum(s.tie and s.median_upper - s.median >= 2 for s in scenes) >= 5
    assert sum(s.tie for s in scenes[:5]) >= 2


def test_lower_median_matches_the_cumulative_count_definition():
    rng = np.random.default_rng(1)
    for _ in range(300):
        h = rng.integers(0, 4, 256) * (rng.random(256) < 0.3)
        if h.sum() == 0:
            continue
        n, cum = int(h.sum()), np.cumsum(h)
        assert fam.lower_median(h) == int(np.searchsorted(cum, n / 2))
        assert fam.upper_median(h) == int(np.searchsorted(cum, n / 2, side="right"))


# ---------------------------------------------------------------- the verifiers, on hand-made outputs

@pytest.fixture(scope="module")
def tie_scene():
    s = fam.make_scene(0)
    assert s.tie and s.median_upper - s.median >= 2 and s.mean > 1
    return s


def _good(scene):
    return fam.reference_solve(scene.img)


def test_manufactured_accepts_the_reference(tie_scene):
    assert fam.verify_manufactured(tie_scene, _good(tie_scene))


@pytest.mark.parametrize("edit", [
    lambda o: o.update(hist=np.roll(o["hist"], 1)),
    lambda o: o["hist"].__setitem__(10, o["hist"][10] + 1),  # one count too many
    lambda o: (o["hist"].__setitem__(10, o["hist"][10] + 1), o["hist"].__setitem__(11, o["hist"][11] - 1)),  # sum kept
    lambda o: o["hist"].__setitem__(255, o["hist"][255] - 1),  # a single dropped saturated pixel
    lambda o: o.update(hist=o["hist"].astype(np.float32)),
    lambda o: o.update(hist=o["hist"][:255]),
    lambda o: o.update(hist=o["hist"].reshape(256, 1)),  # the 4.x shape
    lambda o: o.update(mean=float("nan")),
    lambda o: o.update(median=float(o["median"])),  # float where the contract says int
    lambda o: o.update(median=o["median"] + 1),
    lambda o: o.update(median=fam.upper_median(o["hist"])),
    lambda o: o.pop("std"),
], ids=["shifted bin", "one count over", "one count moved", "one pixel dropped", "float32 counts", "255 bins",
        "(256, 1) shape", "nan mean", "float median", "median + 1", "upper median", "missing key"])
def test_manufactured_rejects_each_small_error(tie_scene, edit):
    out = _good(tie_scene)
    out["hist"] = out["hist"].copy()
    edit(out)
    assert not fam.verify_manufactured(tie_scene, out)


def test_float_tolerance_is_relative_with_a_unit_floor(tie_scene):
    for key in ("mean", "std"):
        want = getattr(tie_scene, key)
        scale = max(1.0, abs(want))
        inside, outside = _good(tie_scene), _good(tie_scene)
        inside[key] = want + 0.5 * fam.FLOAT_RTOL * scale
        outside[key] = want + 2.0 * fam.FLOAT_RTOL * scale
        assert fam.verify_manufactured(tie_scene, inside), key
        assert not fam.verify_manufactured(tie_scene, outside), key


def _meta_outputs(scene):
    base = fam.reference_solve(scene.img)
    return base, {k: fam.reference_solve(t["img"]) for k, t in fam.transforms(scene).items()}


def test_metamorphic_accepts_the_reference_and_names_the_symmetries(tie_scene):
    base, variants = _meta_outputs(tie_scene)
    assert set(variants) == {fam.T_ROT90, fam.T_MIRROR, fam.T_TRANSPOSE, fam.T_ROLL, fam.T_RECOLOR, fam.T_TILE,
                             fam.T_INVERT}
    assert fam.verify_metamorphic(base, variants)
    # the relations really are what the docstring says
    h = base["hist"]
    assert np.array_equal(variants[fam.T_INVERT]["hist"], h[::-1])
    assert np.array_equal(variants[fam.T_TILE]["hist"], 4 * h)
    assert variants[fam.T_INVERT]["median"] == 255 - tie_scene.median_upper != 255 - tie_scene.median
    assert abs(variants[fam.T_INVERT]["mean"] - (255 - base["mean"])) < 1e-12


@pytest.mark.parametrize("name, edit", [
    (fam.T_INVERT, lambda o, b: o.update(hist=b["hist"].copy())),  # not reversed
    (fam.T_INVERT, lambda o, b: o.update(mean=b["mean"])),  # not 255 - mean
    (fam.T_INVERT, lambda o, b: o.update(median=255 - b["median"])),  # 255 - lower instead of 255 - upper
    (fam.T_TILE, lambda o, b: o.update(hist=b["hist"].copy())),  # not x4
    (fam.T_TILE, lambda o, b: o.update(std=b["std"] * (1 + 1e-6))),  # sample-std-sized drift
    (fam.T_RECOLOR, lambda o, b: o["hist"].__setitem__(40, o["hist"][40] + 1)),  # one count moved
    (fam.T_ROT90, lambda o, b: o.update(median=b["median"] + 1)),
    (fam.T_MIRROR, lambda o, b: o.update(hist=o["hist"].astype(np.float64))),  # contract
], ids=["invert: hist", "invert: mean", "invert: median", "tile: hist", "tile: std", "recolor: one count",
        "rot90: median", "mirror: float counts"])
def test_metamorphic_rejects_broken_relations(tie_scene, name, edit):
    base, variants = _meta_outputs(tie_scene)
    variants[name] = dict(variants[name], hist=variants[name]["hist"].copy())
    edit(variants[name], base)
    assert not fam.verify_metamorphic(base, variants)


def test_metamorphic_tolerates_float_roundoff_but_not_more(tie_scene):
    base, variants = _meta_outputs(tie_scene)
    variants[fam.T_TILE] = dict(variants[fam.T_TILE], std=base["std"] * (1 + 1e-11))
    assert fam.verify_metamorphic(base, variants)
    variants[fam.T_TILE] = dict(variants[fam.T_TILE], std=base["std"] * (1 + 1e-8))
    assert not fam.verify_metamorphic(base, variants)


def test_metamorphic_refuses_an_unknown_symmetry(tie_scene):
    base, variants = _meta_outputs(tie_scene)
    assert not fam.verify_metamorphic(base, {"shear": variants[fam.T_ROT90]})


# ---------------------------------------------------------------- contract and prompts

def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img) -> dict" in p and "OpenCV: 5.0" in p
    for needle in ("BT.601", "population standard deviation", "lower median", "'hist': int, shape 256"):
        assert needle in p, needle


def test_phrasings_keep_the_contract():
    assert 3 <= len(fam.PHRASINGS) <= 5 and len(set(fam.PHRASINGS)) == len(fam.PHRASINGS)
    for p in fam.PHRASINGS:
        assert "BT.601" in p and "median" in p and "mean" in p, p
