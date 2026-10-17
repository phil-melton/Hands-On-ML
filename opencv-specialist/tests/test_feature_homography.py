import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import feature_homography as fam

# Expected verdicts per gate for the active profile (sift+orb), measured on scenes 0-19 (manufactured) and
# scenes 0-4 x 4 symmetries (metamorphic).
EXPECTED = {
    #                                              exec-only lint   manufactured metamorphic
    "reference":                                   (True,     True,  True,        True),
    "akaze_removed":                               (False,    True,  False,       False),
    "swallowed_akaze_identity":                    (True,     False, False,       False),
    "orb_norm_l2 (harmless here)":                 (True,     True,  True,        True),
    "no_ratio_test (harmless here)":               (True,     True,  True,        True),
    "no_ransac":                                   (True,     True,  False,       False),
    "perspective_transform_Nx2":                   (False,    True,  False,       False),
    "swapped_direction":                           (True,     True,  False,       False),
    "affine_instead_of_homography":                (True,     True,  False,       True),
    "best_10_matches (within accepted accuracy)":  (True,     True,  True,        True),
    "orb_hamming (valid alternative)":             (True,     True,  True,        True),
    "rgb2gray (harmless here)":                    (True,     True,  True,        True),
}
# The same candidates that change under the strict profile (sift-only): accuracy becomes a gate.
EXPECTED_STRICT = {
    "orb_norm_l2 (harmless here)":                 (True,     True,  False,       True),
    "best_10_matches (within accepted accuracy)":  (True,     True,  False,       True),
    "orb_hamming (valid alternative)":             (True,     True,  False,       False),
    "affine_instead_of_homography":                (True,     True,  False,       False),
    "reference":                                   (True,     True,  True,        True),
}
GATES = ("exec-only", "lint", "manufactured", "metamorphic")
# Why each all-gates-pass row is not a verifier blind spot (the README asks for this to be written down).
PASSES_BECAUSE = {
    "reference": "correct",
    "orb_hamming (valid alternative)": "a correct pipeline; its tail is what sets the active tolerance",
    "rgb2gray (harmless here)": "both images get the same grey conversion",
    "no_ratio_test (harmless here)": "RANSAC and the 1.5 px re-fit remove the extra bad matches again",
    "orb_norm_l2 (harmless here)": "L2 on descriptor bytes still ranks neighbours; ratio test + RANSAC do the rest",
    "best_10_matches (within accepted accuracy)": "10 SIFT matches are as accurate as ORB, the accepted class; "
                                                  "the sift-only profile rejects it",
}

# The measurement behind the constants: calibrate(n_bench=24000, n_wide=2000, n_floor=60, seed=2_000_000) and
# measure_metamorphic_drift(n=9000, seed=3_000_000) on OpenCV 5.0.0 (see the module docstring).
RECORDED = {
    "sift_e_sqrtN_median": 2.38,  # probe error x sqrt(N), SIFT pipeline, 24000 benchmark pairs, N = 60-494
    "sift_e_sqrtN_max": 15.75,  # worst pair: the strict profile's TOL_B = 20 carries 27 % over it
    "orb_over_sift_median_error": 3.95,  # 0.656 px vs 0.166 px
    "orb_e_sqrtN_q998": 59.0,  # 99.8th percentile for plain ORB: the active TOL_B = 60 leaves it 0.18 % of pairs
    "orb_share_above_strict": 0.129,  # plain ORB pairs above 20 / sqrt(N)
    "orb_share_above_active": 0.0018,  # ... above 60 / sqrt(N) (0.30 % in 3000 further make_scene scenes)
    "sift_drift_max": 1.50,  # symmetry discrepancy, SIFT pipeline, 9000 scenes x 4 symmetries (0.97 in a second run)
    "orb_drift_q998": 5.64,  # ... plain ORB, 99.8th percentile of 36000 comparisons (median 0.8-0.9, max 11-28)
}


@pytest.fixture(scope="module")
def matrix():
    candidates = {"reference": ("", fam.REFERENCE), **fam.MUTANTS}
    with ThreadPoolExecutor(12) as ex:  # each evaluation is one child process; run them side by side
        futs = {n: ex.submit(evaluate_solution, fam, code, n_scenes=20, n_meta=5) for n, (_, code) in candidates.items()}
        return {n: f.result() for n, f in futs.items()}


def test_mutants_and_expectations_cover_the_same_names():
    assert set(EXPECTED) == {"reference", *fam.MUTANTS}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_gate_matrix(matrix, name):
    got = tuple(matrix[name][g].passed for g in GATES)
    assert got == EXPECTED[name], {g: matrix[name][g].detail for g in GATES}


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n not in PASSES_BECAUSE]
    assert len(wrong) == 6
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)  # every one is caught by a later gate
    silent = [n for n in wrong if matrix[n]["exec-only"].passed]
    assert len(silent) == len(wrong) - 2  # only the two that raise are visible to exec-only


def test_every_all_pass_row_is_explained(matrix):
    assert {n for n in EXPECTED if all(matrix[n][g].passed for g in GATES)} == set(PASSES_BECAUSE)


def test_strict_profile_turns_accuracy_into_a_gate(monkeypatch):
    """The reviewer's alternative: with the sift-only constants, ORB and the 10-match mutant stop passing."""
    a, b, meta = fam.TOLERANCE_PROFILES["sift-only"]
    for attr, value in (("PROFILE", "sift-only"), ("TOL_A", a), ("TOL_B", b), ("META_TOL", meta)):
        monkeypatch.setattr(fam, attr, value)
    candidates = {"reference": ("", fam.REFERENCE), **{n: fam.MUTANTS[n] for n in EXPECTED_STRICT if n != "reference"}}
    with ThreadPoolExecutor(len(candidates)) as ex:
        futs = {n: ex.submit(evaluate_solution, fam, code, n_scenes=20, n_meta=5) for n, (_, code) in candidates.items()}
        got = {n: tuple(f.result()[g].passed for g in GATES) for n, f in futs.items()}
    assert got == EXPECTED_STRICT


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


# ---------------------------------------------------------------- scenes: determinism and truth consistency

def _noiseless(seed=11, **kw):
    knobs = dict(density=3.0, noise=0.0, blur=0.7, rot_max=20.0, scale_max=1.2, shift_max=0.05, persp=0.08)
    knobs.update(kw)
    return fam.render(seed, **knobs)


def test_scene_generation_is_deterministic():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.img_a, b.img_a) and np.array_equal(a.img_b, b.img_b)
    assert np.array_equal(a.H, b.H) and a.n_corr == b.n_corr and a.knobs == b.knobs
    c = fam.make_scene(4)
    assert not np.array_equal(a.img_a, c.img_a)


def test_scenes_are_well_posed():
    for seed in range(40):
        s = fam.make_scene(seed)
        assert s.img_a.shape == (fam.A_H, fam.A_W, 3) and s.img_b.shape == (fam.B_H, fam.B_W, 3)
        assert s.img_a.dtype == s.img_b.dtype == np.uint8
        assert s.H.shape == (3, 3) and s.H[2, 2] == 1.0
        assert s.n_corr >= fam.N_MIN
        probes = fam.project(s.H, fam.frame_corners(fam.GUARD))  # the verifier's probe points land inside img_b
        assert np.all(probes > 6) and np.all(probes < [fam.B_W - 7, fam.B_H - 7])
        assert abs(s.knobs["rot"]) <= 20.0 and 1 / 1.2 <= s.knobs["scale"] <= 1.2


def test_manufactured_H_really_maps_img_a_to_img_b():
    """Warping img_a by the manufactured H must reproduce img_b on the overlap: the answer is the answer."""
    for seed in (11, 12, 13):
        s = _noiseless(seed)
        warped = cv2.warpPerspective(s.img_a, s.H, (fam.B_W, fam.B_H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(s.img_a.shape[:2], 255, np.uint8), s.H, (fam.B_W, fam.B_H),
                                   flags=cv2.INTER_NEAREST)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        assert inner.sum() > 0.4 * fam.A_H * fam.A_W
        diff = np.abs(warped.astype(int) - s.img_b.astype(int))[inner]
        assert diff.max() <= 2 and diff.mean() < 0.1, (seed, diff.max(), diff.mean())


def test_symmetry_predictions_match_the_transformed_images():
    """The algebra behind verify_metamorphic: warping each variant's first image with the predicted H
    reproduces the variant's second image."""
    s = _noiseless(21)
    var = fam.transforms(s)
    for name in ("mirror", "rot90", "crop"):
        a, b = var[name]["img_a"], var[name]["img_b"]
        Ma, Mb = {"mirror": (fam._flip_x(fam.A_W), fam._flip_x(fam.B_W)),
                  "rot90": (fam._rot90(fam.A_W), fam._rot90(fam.B_W)),
                  "crop": (fam._shift(*fam.CROP_A), fam._shift(*fam.CROP_B))}[name]
        Hp = Mb @ s.H @ np.linalg.inv(Ma)
        Hp /= Hp[2, 2]
        warped = cv2.warpPerspective(a, Hp, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(a.shape[:2], 255, np.uint8), Hp, (b.shape[1], b.shape[0]),
                                   flags=cv2.INTER_NEAREST)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        assert np.abs(warped.astype(int) - b.astype(int))[inner].max() <= 2, name
        assert fam.discrepancy(name, s.H, Hp) < 1e-9, name  # the prediction is a fixed point of the check
    assert fam.discrepancy("swap", s.H, np.linalg.inv(s.H)) < 1e-9
    assert np.array_equal(var["swap"]["img_a"], s.img_b) and np.array_equal(var["swap"]["img_b"], s.img_a)


# ---------------------------------------------------------------- the verifier itself

def _ok_out(scene, H=None):
    return {"H": scene.H.copy() if H is None else H}


def test_truth_passes_and_perturbations_fail_in_proportion():
    s = fam.make_scene(5)
    assert fam.verify_manufactured(s, _ok_out(s))
    tol = fam.tolerance(s)
    for dx, ok in ((0.5 * tol, True), (1.5 * tol, False)):
        T = np.array([[1.0, 0, dx], [0, 1, 0], [0, 0, 1]])  # a pure shift of the answer in img_b
        assert bool(fam.verify_manufactured(s, _ok_out(s, T @ s.H))) is ok, dx


def test_contract_violations_are_rejected():
    s = fam.make_scene(5)
    good = s.H
    bad = {
        "unnormalized": good * 2.0,
        "flat 9-vector": good.ravel(),
        "wrong shape": good[:2],
        "nan": np.where(np.eye(3) > 0, np.nan, good),
        "int dtype": np.round(good).astype(np.int32),
        "singular": np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        "zeros": np.zeros((3, 3)),
    }
    for name, H in bad.items():
        v = fam.verify_manufactured(s, {"H": H})
        assert not v, name
        assert v.reason, name
    assert not fam.verify_manufactured(s, {})
    assert "normalized" in fam.verify_manufactured(s, {"H": good * 2.0}).reason
    assert fam.verify_manufactured(s, {"H": good.astype(np.float32)})  # the contract asks for a float, not float64


def test_tolerance_follows_the_inverse_square_root_law():
    s = fam.make_scene(5)
    lo = fam.tolerance(s)
    s.n_corr *= 4
    hi = fam.tolerance(s)
    assert hi < lo and abs((lo - fam.TOL_A) / (hi - fam.TOL_A) - 2.0) < 1e-9  # b / sqrt(N) halves when N quadruples
    s.n_corr = 61
    assert 7.3 < fam.tolerance(s) < 8.1 and 1.9 < fam.tolerance(s, "sift-only") < 2.7  # sparsest scene served
    s.n_corr = 490
    assert 2.5 < fam.tolerance(s) < 2.9 and 0.8 < fam.tolerance(s, "sift-only") < 1.0  # densest
    assert fam.TOLERANCE_PROFILES[fam.PROFILE] == (fam.TOL_A, fam.TOL_B, fam.META_TOL)


def test_metamorphic_rejects_inconsistent_variants():
    s = fam.make_scene(5)
    base = _ok_out(s)
    consistent = {"swap": _ok_out(s, np.linalg.inv(s.H))}
    assert fam.verify_metamorphic(base, consistent)
    shifted = np.array([[1.0, 0, 3 * fam.META_TOL], [0, 1, 0], [0, 0, 1]]) @ np.linalg.inv(s.H)
    v = fam.verify_metamorphic(base, {"swap": _ok_out(s, shifted)})
    assert not v and "swap" in v.reason
    assert not fam.verify_metamorphic(base, {"teleport": base})
    assert not fam.verify_metamorphic({"H": np.zeros(9)}, consistent)
    singular = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1]])  # must be a failing verdict, not a LinAlgError
    assert not fam.verify_metamorphic({"H": singular}, consistent)


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(img_a, img_b) -> dict" in p and "OpenCV: 5.0" in p and "H[2, 2] = 1" in p
    assert "Accuracy requirement" in p and "32 px inside" in p
    assert 3 <= len(fam.PHRASINGS) <= 5 and len(set(fam.PHRASINGS)) == len(fam.PHRASINGS)
    for ph in fam.PHRASINGS:  # same contract in every wording
        assert ("H[2, 2]" in ph or "bottom-right entry is 1" in ph) and fam.ACCURACY in ph
        assert "img_a" in ph and "img_b" in ph


# ---------------------------------------------------------------- calibration: the constants against fresh gauge blocks

@pytest.fixture(scope="module")
def fresh():
    return fam.calibrate(n_bench=150, n_wide=0, n_floor=3, seed=900_000, processes=8, keep_records=True)


def _tol(r, profile):
    a, b, _ = fam.TOLERANCE_PROFILES[profile]
    return a + b / math.sqrt(r["N"])


def test_reference_pipeline_obeys_the_inverse_square_root_law(fresh):
    assert fresh["bench"]["sift"]["failures"] == 0
    assert abs(fresh["bench"]["sift"]["inset32"]["e_sqrtN_med"] / RECORDED["sift_e_sqrtN_median"] - 1) < 0.15
    # every fresh gauge block, both domains, sits inside even the strict tolerance
    for dom in ("bench", "floor"):
        for r in fresh["records"][dom]:
            assert r["sift"] <= _tol(r, "sift-only"), (dom, r)
    # log error against log N: slope about -1/2 (an N-independent floor would flatten it)
    recs = fresh["records"]["bench"] + fresh["records"]["floor"]
    slope = np.polyfit(np.log([r["N"] for r in recs]), np.log([r["sift"] for r in recs]), 1)[0]
    assert -0.8 < slope < -0.25, slope


def test_constants_are_consistent_with_the_recorded_measurement():
    a_strict, b_strict, meta_strict = fam.TOLERANCE_PROFILES["sift-only"]
    a_act, b_act, meta_act = fam.TOLERANCE_PROFILES["sift+orb"]
    assert a_strict == a_act == 0.0  # no N-independent floor was measurable
    assert 1.25 * RECORDED["sift_e_sqrtN_max"] <= b_strict < 1.35 * RECORDED["sift_e_sqrtN_max"]  # 25 % margin
    assert RECORDED["orb_e_sqrtN_q998"] <= b_act < 1.1 * RECORDED["orb_e_sqrtN_q998"]  # 0.2 % false-reject budget
    assert 1.25 * RECORDED["sift_drift_max"] <= meta_strict < 1.35 * RECORDED["sift_drift_max"]
    assert RECORDED["orb_drift_q998"] <= meta_act < 1.1 * RECORDED["orb_drift_q998"]
    assert fam.PROFILE == "sift+orb"


def test_orb_does_not_average_down_and_the_profiles_differ_by_its_tail(fresh):
    sift, orb = fresh["bench"]["sift"]["inset32"], fresh["bench"]["orb"]["inset32"]
    ratio = orb["err_med"] / sift["err_med"]
    assert 0.6 * RECORDED["orb_over_sift_median_error"] < ratio < 1.6 * RECORDED["orb_over_sift_median_error"], ratio
    # at N ~ 600 (noiseless dense textures) ORB still sits at its floor: the error is not 1/sqrt(N)
    f_sift, f_orb = fresh["floor"]["sift"]["inset32"], fresh["floor"]["orb"]["inset32"]
    assert f_orb["err_med"] > 3 * f_sift["err_med"] and f_orb["err_med"] > 0.35
    bench = fresh["records"]["bench"]
    strict = np.mean([r["orb"] > _tol(r, "sift-only") for r in bench])
    active = np.mean([r["orb"] > _tol(r, "sift+orb") for r in bench])
    assert 0.05 < strict < 0.25, strict  # recorded 12.9 %: plain ORB mostly fails a SIFT-class bound over 20 scenes
    assert active <= 0.02, active  # recorded 0.18 %
    assert all(r["sift"] <= _tol(r, "sift+orb") for r in bench)


def test_probing_the_frame_corners_instead_of_the_inset_ones_is_worse(fresh):
    # why GUARD = 32: ORB keypoints stay 31+ px from the border, so the corners themselves are extrapolation
    for name in ("sift", "orb"):
        b = fresh["bench"][name]
        assert b["frame_corners"]["err_med"] > 1.15 * b["inset32"]["err_med"], name


def test_symmetry_drift_stays_inside_META_TOL():
    drift = fam.measure_metamorphic_drift(n=60, seed=500_000, processes=8, keep_rows=True)
    sift, orb = np.array(drift["sift"]["pooled"]["values"]), np.array(drift["orb"]["pooled"]["values"])
    strict, active = fam.TOLERANCE_PROFILES["sift-only"][2], fam.TOLERANCE_PROFILES["sift+orb"][2]
    assert sift.max() < strict  # the SIFT pipeline is equivariant to well under a pixel
    assert np.mean(orb > active) <= 0.02  # ... and plain ORB fits inside the active bound
    assert 0.03 < np.mean(orb > strict) < 0.35  # but not inside the strict one: its randomized fit drifts
    ratio = np.mean([drift["orb"][t]["med"] / drift["sift"][t]["med"] for t in fam.SYMMETRIES])
    assert ratio > 3, ratio


# ---------------------------------------------------------------- facts about OpenCV 5.0 that the mutants rely on

def test_opencv5_facts_behind_the_mutants():
    for gone in ("AKAZE_create", "KAZE_create", "BRISK_create", "AgastFeatureDetector_create", "BOWKMeansTrainer"):
        assert not hasattr(cv2, gone), gone
    assert hasattr(cv2, "SIFT_create") and hasattr(cv2, "ORB_create") and not hasattr(cv2, "xfeatures2d")
    pts = np.float32([[1, 2], [3, 4], [5, 6]])
    assert cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.eye(3)).shape == (3, 1, 2)
    with pytest.raises(cv2.error):
        cv2.perspectiveTransform(pts, np.eye(3))  # (N, 2) is rejected
    assert cv2.findHomography(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * 10, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * 20)[0] is not None
    sift_desc = np.zeros((3, 128), np.float32)
    with pytest.raises(cv2.error):
        cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(sift_desc, sift_desc, k=2)  # Hamming needs uint8 descriptors
