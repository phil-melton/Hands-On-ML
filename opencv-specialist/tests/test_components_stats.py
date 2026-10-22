import cv2
import numpy as np
import pytest

from cvbench.evaluate import evaluate_solution
from cvbench.families import components_stats as fam

# Expected verdicts per gate, as measured: this is the family's claim, pinned as a regression test.
EXPECTED = {
    #                            exec-only lint   manufactured metamorphic
    "reference":                (True,     True,  True,        True),
    "bool_direct":              (False,    True,  False,       False),
    "include_background":       (True,     True,  False,       False),
    "connectivity_hardcoded_8": (True,     True,  False,       True),   # metamorphic blind spot: constant in the symmetry
    "connectivity_swapped":     (True,     True,  False,       False),
    "centroid_row_col":         (True,     True,  False,       False),
    "box_wh_swapped":           (True,     True,  False,       False),
    "drop_small_noise":         (True,     True,  False,       False),
    "label_order_ties":         (True,     True,  False,       True),   # metamorphic compares sets: blind to order
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


def test_matrix_covers_exactly_the_registered_mutants():
    assert set(EXPECTED) == {"reference"} | set(fam.MUTANTS)
    assert 5 <= len(fam.MUTANTS) <= 8


def test_exec_only_misses_most_wrong_solutions(matrix):
    wrong = [n for n in EXPECTED if n != "reference"]
    assert sum(matrix[n]["exec-only"].passed for n in wrong) == len(wrong) - 1  # only bool_direct raises
    assert all(not all(matrix[n][g].passed for g in GATES[1:]) for n in wrong)  # no mutant survives all gates


def test_failure_reasons_name_the_defect(matrix):
    assert "iDepth == CV_8U" in matrix["bool_direct"]["exec-only"].detail
    assert "connectivity 4" in matrix["connectivity_hardcoded_8"]["manufactured"].detail  # right on 8, wrong on 4
    assert "stated order" in matrix["label_order_ties"]["manufactured"].detail
    assert "area changed" in matrix["drop_small_noise"]["metamorphic"].detail  # the connectivity relation


def test_every_mutant_differs_from_reference():
    for name, (_, code) in fam.MUTANTS.items():
        assert code != fam.REFERENCE, name


# ---------------------------------------------------------------- the reference, in process

def test_reference_passes_many_scenes_in_process():
    for seed in range(200, 400):
        scene = fam.make_scene(seed)
        out = fam.reference_solve(**fam.inputs(scene))
        v = fam.verify_manufactured(scene, out)
        assert v, (seed, v.reason)
        variants = {name: fam.reference_solve(**t) for name, t in fam.transforms(scene).items()}
        v = fam.verify_metamorphic(out, variants)
        assert v, (seed, v.reason)


def flood_components(mask, connectivity):
    """An independent labeller (pure python flood fill, no OpenCV) for the manufactured truth."""
    H, W = mask.shape
    nb = [(0, 1), (1, 0), (0, -1), (-1, 0)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if connectivity == 8 else [])
    seen = np.zeros_like(mask)
    comps = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, pix = [(r, c)], []
        seen[r, c] = True
        while stack:
            a, b = stack.pop()
            pix.append((a, b))
            for da, db in nb:
                p, q = a + da, b + db
                if 0 <= p < H and 0 <= q < W and mask[p, q] and not seen[p, q]:
                    seen[p, q] = True
                    stack.append((p, q))
        ys, xs = np.array([p[0] for p in pix]), np.array([p[1] for p in pix])
        n = len(pix)
        comps.append(fam.Component(n, int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                   int(ys.max() - ys.min() + 1), int(xs.sum()) / n, int(ys.sum()) / n))
    return sorted(comps, key=lambda c: c.order_key)


def test_manufactured_truth_matches_an_independent_flood_fill():
    for seed in range(60):
        s = fam.make_scene(seed)
        for conn in (4, 8):
            assert s.truth_for(conn) == flood_components(s.mask, conn), (seed, conn)


# ---------------------------------------------------------------- scene generation

def test_scene_generation_is_deterministic_and_well_posed():
    a, b = fam.make_scene(3), fam.make_scene(3)
    assert np.array_equal(a.mask, b.mask) and a.objects == b.objects and a.connectivity == b.connectivity
    assert not np.array_equal(fam.make_scene(4).mask, a.mask)
    kinds, edge = set(), 0
    for seed in range(120):
        s = fam.make_scene(seed)
        assert s.mask.dtype == bool and s.mask.shape == fam.SIZE
        assert s.connectivity == (8 if seed % 2 == 0 else 4)
        names = [o.kind for o in s.objects]
        kinds |= set(names)
        assert names.count("dot") >= 2 and "stagger_pair" in names and any(k in names for k in fam.DIAGONAL), seed
        assert 7 <= len(s.objects) <= 11
        boxes = [o.box for o in s.objects]
        for i, p in enumerate(boxes):  # reserved boxes keep >= GAP empty pixels between them
            assert 0 <= p[0] <= p[2] < fam.W0 and 0 <= p[1] <= p[3] < fam.H0
            for q in boxes[i + 1:]:
                assert fam._free(p, [q]), (seed, p, q)
        t4, t8 = s.truth_for(4), s.truth_for(8)
        assert sum(c.area for c in t4) == sum(c.area for c in t8) == int(s.mask.sum())
        assert len(t4) > len(t8), seed  # the scene is sensitive to connectivity
        keys = [c.order_key for c in s.truth]
        assert len(set(keys)) == len(keys) and keys == sorted(keys), seed  # the contract order is a strict order
        assert sum(c.area == 1 for c in s.truth) >= 2  # equal areas exist, so the tie-break is exercised
        edge += any(o.box[0] == 0 or o.box[1] == 0 or o.box[2] == fam.W0 - 1 or o.box[3] == fam.H0 - 1
                    for o in s.objects)
    assert kinds == set(fam.KINDS)
    assert 30 < edge < 110  # border contact is common but not universal


def test_stagger_pair_separates_label_order_from_the_contract_order_at_connectivity_8_only():
    def label_order_areas_and_tops(scene):
        _, _, st, _ = cv2.connectedComponentsWithStats(scene.mask.astype(np.uint8), connectivity=scene.connectivity)
        st = st[1:]
        stable = np.argsort(-st[:, cv2.CC_STAT_AREA], kind="stable")
        contract = np.lexsort((st[:, cv2.CC_STAT_LEFT], st[:, cv2.CC_STAT_TOP], -st[:, cv2.CC_STAT_AREA]))
        return stable, contract

    for seed in range(0, 40, 2):  # connectivity 8: the shortcut disagrees with the contract in every scene
        stable, contract = label_order_areas_and_tops(fam.make_scene(seed))
        assert not np.array_equal(stable, contract), seed
    for seed in range(1, 40, 2):  # connectivity 4: raster-order labels agree with (y, x) in these scenes
        stable, contract = label_order_areas_and_tops(fam.make_scene(seed))
        assert np.array_equal(stable, contract), seed


# ---------------------------------------------------------------- tolerance constants vs the recorded measurement

def test_tolerance_constants_are_consistent_with_the_recorded_measurement():
    assert fam.CENTROID_TOL == 1e-6 and fam.META_TOL == 1e-6
    s = fam.calibrate(n=240, seed=999)  # fresh gauge blocks, not the 1200 the constants were recorded on
    assert s["area"] == 0 and s["box"] == 0  # integers: zero tolerance is attainable
    assert s["algorithm_disagreements"] == 0
    assert s["centroid_ccl"] < 1e-12 and s["centroid_moments"] < 1e-12  # float64 rounding floor, predicted 2.8e-14
    assert fam.CENTROID_TOL > 1e6 * max(s["centroid_ccl"], 5.7e-14)  # far above the noise ...
    assert s["centroid_float32"] > 10 * fam.CENTROID_TOL  # ... and float32 would not fit: hence the float64 contract
    biggest = max(c.area for seed in range(200) for c in fam.make_scene(seed).truth_for(8 - 4 * (seed % 2)))
    assert biggest <= 60 * 40
    assert 1 / (60 * 40) > 100 * fam.CENTROID_TOL  # ... and far below a one-pixel membership error


def test_metamorphic_maps_are_exact_for_the_reference():
    d = fam.measure_metamorphic_drift(n=20, seed=7_000)
    assert d["int_mismatches"] == 0 and d["refinement_failures"] == 0
    assert d["centroid_drift"] < 1e-12  # one ulp of the covariant map itself (5.7e-14 measured on 100 scenes)
    assert fam.META_TOL > 1e6 * 5.7e-14


def test_bool_mask_support_on_this_build():
    """The measured 5.0 claim behind the input type: the binding takes bools, the CCL algorithm still does not."""
    s = fam.measure_bool_mask_support()
    assert s["countNonZero"] == s["moments"] == s["bitwise_not"] == "accepted"
    for name in ("connectedComponentsWithStats", "connectedComponents", "findContours", "threshold", "dilate"):
        assert s[name] == "raises cv2.error", name
    m = np.zeros((6, 6), bool)
    m[1:3, 1:4] = True
    assert cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)[2][1].tolist() == [1, 1, 3, 2, 6]


# ---------------------------------------------------------------- the verifiers, attacked directly

def _case(seed=3):
    scene = fam.make_scene(seed)
    out = fam.reference_solve(**fam.inputs(scene))
    return scene, out


def _copy(out):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in out.items()}


def test_centroid_tolerance_boundary():
    scene, out = _case()
    ok = _copy(out)
    ok["centroids"][0, 0] += 5e-7
    assert fam.verify_manufactured(scene, ok)
    bad = _copy(out)
    bad["centroids"][0, 0] += 2e-6
    assert not fam.verify_manufactured(scene, bad)
    nan = _copy(out)
    nan["centroids"][2, 1] = np.nan
    assert not fam.verify_manufactured(scene, nan)


def test_integers_get_zero_tolerance():
    scene, out = _case()
    for key, idx in (("areas", (0,)), ("boxes", (0, 0)), ("boxes", (1, 1)), ("boxes", (2, 2)), ("boxes", (3, 3))):
        bad = _copy(out)
        bad[key][idx] += 1
        assert not fam.verify_manufactured(scene, bad), (key, idx)


def test_float32_centroids_are_a_contract_violation():
    scene, out = _case()
    bad = _copy(out)
    bad["centroids"] = bad["centroids"].astype(np.float32)
    v = fam.verify_manufactured(scene, bad)
    assert not v and "float64" in v.reason


def test_order_is_enforced_including_the_tie_break():
    scene, out = _case()
    k = int(np.nonzero(out["areas"][:-1] == out["areas"][1:])[0][0])  # two equal areas, adjacent in the contract order
    swapped = _copy(out)
    for key in ("areas", "boxes", "centroids"):
        swapped[key][[k, k + 1]] = swapped[key][[k + 1, k]]
    v = fam.verify_manufactured(scene, swapped)
    assert not v and "stated order" in v.reason
    ascending = {k_: (v_[::-1].copy() if isinstance(v_, np.ndarray) else v_) for k_, v_ in out.items()}
    assert not fam.verify_manufactured(scene, ascending)


def test_centroids_must_stay_attached_to_their_component():
    scene, out = _case()
    bad = _copy(out)
    bad["centroids"] = np.roll(bad["centroids"], 1, axis=0)  # right values, sorted/shuffled apart from the stats
    assert not fam.verify_manufactured(scene, bad)


def test_wrong_connectivity_answer_is_rejected_in_both_directions():
    for seed in (4, 5):
        scene = fam.make_scene(seed)
        other = fam.reference_solve(scene.mask, 12 - scene.connectivity)
        assert not fam.verify_manufactured(scene, other)


def test_dropping_border_components_is_rejected():
    for seed in range(40):
        scene, out = _case(seed)
        b = out["boxes"]
        touching = (b[:, 0] == 0) | (b[:, 1] == 0) | (b[:, 0] + b[:, 2] == fam.W0) | (b[:, 1] + b[:, 3] == fam.H0)
        if touching.any():
            kept = {k: v[~touching] for k, v in out.items() if k != "count"}
            kept["count"] = int((~touching).sum())
            assert not fam.verify_manufactured(scene, kept), seed
            return
    pytest.fail("no scene with a border-touching component among 40")


def test_shape_and_count_inconsistencies_are_rejected():
    scene, out = _case()
    assert not fam.verify_manufactured(scene, {**out, "count": out["count"] + 1})
    assert not fam.verify_manufactured(scene, {**out, "boxes": out["boxes"][:, :3]})
    assert not fam.verify_manufactured(scene, {**out, "areas": out["areas"][:-1]})
    assert not fam.verify_manufactured(scene, {k: v for k, v in out.items() if k != "centroids"})


def test_metamorphic_verifier_catches_what_it_is_for():
    scene, out = _case(2)
    variants = {name: fam.reference_solve(**t) for name, t in fam.transforms(scene).items()}
    assert fam.verify_metamorphic(out, variants)
    for name in (fam.TRANSLATE, "rot90", "mirror"):  # the output did not move with the image
        assert not fam.verify_metamorphic(out, {**variants, name: out}), name
    swapped = {**variants, "rot90": variants["mirror"], "mirror": variants["rot90"]}  # right answers, wrong transforms
    assert not fam.verify_metamorphic(out, swapped)
    assert not fam.verify_metamorphic(out, {"rotate 45": variants["rot90"]})  # no covariance rule: refuse, do not pass
    name = next(n for n in variants if n.startswith("connectivity"))
    assert not fam.verify_metamorphic(out, {name: {**out, "count": 0, "areas": out["areas"][:0],
                                                   "boxes": out["boxes"][:0], "centroids": out["centroids"][:0]}})
    nan = _copy(variants["mirror"])
    nan["centroids"][0, 0] = np.nan
    assert not fam.verify_metamorphic(out, {"mirror": nan})


def test_metamorphic_blind_spot_is_a_constant_in_the_symmetry():
    """connectivity_hardcoded_8 returns the same answer for both connectivities: refinement allows equality."""
    scene = fam.make_scene(2)
    same = fam.reference_solve(scene.mask, 8)
    variants = {name: fam.reference_solve(t["mask"], 8) for name, t in fam.transforms(scene).items()}
    assert fam.verify_metamorphic(same, variants)  # passes ...
    assert not fam.verify_manufactured(fam.make_scene(3), fam.reference_solve(fam.make_scene(3).mask, 8))  # ... yet wrong


def test_metamorphic_gate_alone_accepts_an_empty_answer():
    """The stuck gauge: a constant output is invariant under every symmetry; only the manufactured gate rejects it."""
    scene = fam.make_scene(2)
    empty = {"count": 0, "areas": np.zeros(0, np.int64), "boxes": np.zeros((0, 4), np.int64),
             "centroids": np.zeros((0, 2))}
    assert fam.verify_metamorphic(empty, {name: empty for name in fam.transforms(scene)})
    assert not fam.verify_manufactured(scene, empty)


def test_prompt_renders_the_contract():
    p = fam.SPEC.to_prompt()
    assert "def solve(mask, connectivity) -> dict" in p and "OpenCV: 5.0" in p
    assert "float64" in p and "largest area first" in p and "no +0.5" in p and "w = x_max - x_min + 1" in p
    assert len(fam.PHRASINGS) == 16 and fam.PHRASINGS[0] == fam.SPEC.task
    assert len(set(fam.PHRASINGS)) == len(fam.PHRASINGS)
    for ph in fam.PHRASINGS:  # a rewording must still carry the whole contract
        low = ph.lower()
        assert "connect" in low and "box" in low and "centroid" in low and "background" in low, ph
        assert "4" in ph and "8" in ph and "area" in low, ph  # both connectivities, and the area to report
        # the output notes say "in the order stated in the task": every wording must state it
        assert any(word in low for word in ("largest", "biggest", "descending")), ph
        assert any(word in low for word in ("top-left", "box top", "box y", "starts higher", "top to bottom")), ph
