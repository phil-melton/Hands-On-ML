"""The gotcha suite's own regression tests.

Four kinds of test:

* catalogue integrity: 40 checks, unique ids, tags that agree with the recorded 4.14-vs-5.0 diff, prompts that
  render, solve() signatures that match the specs;
* the behaviour matrix: for every check the reference passes and every wrong variant fails, in the way it was
  declared to fail (an exception of a given type, or a silent wrong answer only the verifier catches);
* verifier sensitivity: every tolerance is exercised from both sides (legitimate alternatives accepted, genuinely
  different answers rejected), with the measurements the tolerance was derived from re-run here;
* measurements of OpenCV 5.0.0 itself that the checks stand on, so a library change shows up as a test failure.

All model-style code runs through ``sandbox.run_solution`` (via ``cvbench.gotchas.grade``). The matrix runs its ~110
sandbox processes on a thread pool; the whole file takes well under the 90 s budget.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvbench.contract import TaskSpec
from cvbench.gotchas import CHECKS, GROUPS, ROLES, calling, catalogue_markdown, drawing, drift, dtypes, grade, run_suite, shapes
from cvbench.gotchas.core import Case
from sandbox.lint import lint

ROOT = Path(__file__).resolve().parents[1]
REC4 = json.loads((ROOT / "probes/results/probe_4.14.0.json").read_text())["probes"]
REC5 = json.loads((ROOT / "probes/results/probe_5.0.0.json").read_text())["probes"]

PER_GROUP = {"shapes": 14, "dtypes": 11, "drawing": 6, "calling": 5, "drift": 4}
PREFIX = {"shapes": "shape_", "dtypes": "dtype_", "drawing": "draw_", "calling": "call_", "drift": "drift_"}
VARIANTS = [(cid, name) for cid, c in CHECKS.items() for name in c.wrong]
RUNTIME_BUDGET_S = 90.0


# ---------------------------------------------------------------- catalogue integrity

def test_the_build_under_test_is_opencv_5_0():
    assert cv2.__version__.startswith("5.0"), cv2.__version__


def test_forty_checks_in_the_planned_groups():
    assert len(CHECKS) == 40
    got = {g: sum(c.group == g for c in CHECKS.values()) for g in GROUPS}
    assert got == PER_GROUP
    assert list(GROUPS) == list(PER_GROUP)


def test_ids_are_unique_and_prefixed_by_group():
    all_ids = [c.id for module in (shapes, dtypes, drawing, calling, drift) for c in module.CHECKS]
    assert len(all_ids) == len(set(all_ids)) == len(CHECKS)
    for cid, c in CHECKS.items():
        assert cid == c.id and cid.startswith(PREFIX[c.group]), cid


def test_every_check_carries_its_tags():
    for cid, c in CHECKS.items():
        assert c.group in GROUPS and c.role in ROLES, cid
        assert isinstance(c.spec, TaskSpec) and c.spec.outputs, cid
        assert c.probes and all(isinstance(p, str) and p for p in c.probes), cid
        assert isinstance(c.depends_on_5_0, bool) and c.subset in ("5.0", "timeless"), cid
        assert c.gotcha.strip() and c.reference.strip(), cid
        assert len(c.wrong) >= 1, cid
        for name, w in c.wrong.items():
            assert w.why.strip() and w.code.strip() and w.fails.strip(), (cid, name)
            assert w.code != c.reference, (cid, name)
        codes = [w.code for w in c.wrong.values()] + [c.reference]
        assert len(set(codes)) == len(codes), f"{cid}: duplicate solution text"


def test_controls_are_timeless_gotchas_are_tagged_from_the_recorded_diff():
    n_5_0 = n_timeless = n_controls = 0
    for cid, c in CHECKS.items():
        recorded = [p for p in c.probes if not p.startswith("new:")]
        assert recorded, f"{cid} cites no recorded probe"
        for p in recorded:
            assert p in REC5 and p in REC4, f"{cid}: {p} is not a recorded probe"
        if c.role == "control":
            n_controls += 1
            assert not c.depends_on_5_0, cid
            continue
        changed = any(REC4[p] != REC5[p] for p in recorded)
        assert c.depends_on_5_0 == changed, f"{cid}: tagged {c.subset} but the recorded probes say changed={changed}"
        n_5_0 += c.depends_on_5_0
        n_timeless += not c.depends_on_5_0
    assert (n_5_0, n_timeless, n_controls) == (17, 17, 6)


def test_measurements_not_in_the_recorded_probes_are_tagged_as_new():
    assert [cid for cid, c in CHECKS.items() if any(p.startswith("new:") for p in c.probes)] == ["dtype_bool_mask_ops"]


def _solve_params(code: str) -> list[str]:
    fn = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    assert len(fn) == 1
    return [a.arg for a in fn[0].args.args]


def test_every_solution_has_the_signature_the_prompt_announces():
    for cid, c in CHECKS.items():
        want = list(c.spec.inputs)
        assert f"def solve({', '.join(want)}) -> dict" in c.prompt(), cid
        for label, code in [("reference", c.reference)] + [(n, w.code) for n, w in c.wrong.items()]:
            assert _solve_params(code) == want, (cid, label)


def test_prompts_read_as_requests_not_quizzes():
    banned = ("gotcha", "pitfall", "trap", "4.14", "regress", "middle axis", "lost axis", "narrow", "wraps around")
    for cid, c in CHECKS.items():
        text = c.prompt().lower()
        assert text.startswith("task: ") and "opencv: 5.0" in text, cid
        for word in banned:
            assert word not in text, f"{cid}: prompt mentions {word!r}"


def test_cases_are_deterministic_and_match_the_declared_inputs():
    exact = {"uint8", "int32", "int64", "uint32", "float16", "float32", "float64", "bool"}
    for cid, c in CHECKS.items():
        a, b = c.make_cases(), c.make_cases()
        assert len(a) == len(b) >= 1, cid
        for x, y in zip(a, b):
            assert list(x.inputs) == list(c.spec.inputs), cid
            for k in x.inputs:
                u, v = x.inputs[k], y.inputs[k]
                if isinstance(u, np.ndarray):
                    assert u.dtype == v.dtype and np.array_equal(u, v), (cid, k)
                    s = c.spec.inputs[k]
                    if s.dtype in exact:
                        assert str(u.dtype) == s.dtype, (cid, k, str(u.dtype), s.dtype)
                    elif s.dtype in ("int", "float"):
                        assert u.dtype.kind in {"int": "iu", "float": "f"}[s.dtype], (cid, k)
                    if s.ndim is not None:
                        assert u.ndim == s.ndim, (cid, k, u.shape)
                else:
                    assert u == v, (cid, k)


def test_catalogue_table_lists_every_check():
    table = catalogue_markdown()
    assert all(f"`{cid}`" in table for cid in CHECKS) and table.count("\n") == 41


def test_the_recorded_probes_the_checks_cite_reproduce_on_this_build():
    spec = importlib.util.spec_from_file_location("api_diff", ROOT / "probes/api_diff.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    now = api.run_all()
    cited = {p for c in CHECKS.values() for p in c.probes if not p.startswith("new:")}
    drifted = {p for p in cited if now[p] != REC5[p]}
    assert not drifted, f"recorded 5.0.0 probe results that no longer reproduce: {sorted(drifted)}"


# ---------------------------------------------------------------- the behaviour matrix

@pytest.fixture(scope="module")
def matrix():
    jobs = [(cid, "reference", c.reference) for cid, c in CHECKS.items()]
    jobs += [(cid, name, w.code) for cid, c in CHECKS.items() for name, w in c.wrong.items()]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        grades = list(pool.map(lambda j: grade(CHECKS[j[0]], j[2]), jobs))
    out = {(cid, name): g for (cid, name, _), g in zip(jobs, grades)}
    out["seconds"] = time.monotonic() - t0
    return out


@pytest.mark.parametrize("cid", list(CHECKS))
def test_reference_passes(matrix, cid):
    g = matrix[(cid, "reference")]
    assert g.passed and g.stage == "ok", str(g)


@pytest.mark.parametrize("cid,name", VARIANTS)
def test_wrong_variant_fails_the_way_it_is_declared_to_(matrix, cid, name):
    w, g = CHECKS[cid].wrong[name], matrix[(cid, name)]
    assert not g.passed, f"{cid}/{name} was accepted: {g}"
    if w.fails == "verifier":
        # runs fine (an exec()-only harness accepts it) and the contract or the verifier rejects it
        assert g.ran and g.stage in ("contract", "verify"), str(g)
    else:
        assert g.stage == "run" and not g.ran and g.error_type == w.fails, str(g)
        assert w.error_contains in (g.error or ""), (w.error_contains, g.error)


def test_exec_only_filtering_would_keep_the_silent_wrong_answers(matrix):
    wrong = [matrix[v] for v in VARIANTS]
    silent = [g for g in wrong if g.ran]
    assert len(wrong) == len(VARIANTS) == 88
    assert len(silent) == sum(w.fails == "verifier" for c in CHECKS.values() for w in c.wrong.values()) == 42
    assert not any(g.passed for g in wrong)


def test_the_matrix_ran_inside_the_time_budget(matrix):
    assert matrix["seconds"] < RUNTIME_BUDGET_S, matrix["seconds"]


def test_run_suite_scores_a_candidate_per_check_and_reports_the_exec_only_gap(matrix):
    refs = run_suite({cid: c.reference for cid, c in CHECKS.items()}, workers=8)
    assert refs.passed() == (40, 40) and refs.ran() == (40, 40)
    assert refs.by_group() == {g: (n, n) for g, n in PER_GROUP.items()}
    assert refs.by_subset() == {"5.0 subset": (17, 17), "timeless": (17, 17), "controls": (6, 6)}
    first_wrong = run_suite(lambda c: next(iter(c.wrong.values())).code, workers=8)
    assert first_wrong.passed()[0] == 0
    assert first_wrong.ran()[0] == sum(next(iter(c.wrong.values())).fails == "verifier" for c in CHECKS.values())
    partial = run_suite({"shape_houghlinesp": CHECKS["shape_houghlinesp"].reference})
    assert partial.passed() == (1, 40) and partial.grades["shape_calchist"].detail == "no solution supplied"
    assert "1/40" in partial.summary() and "Output shapes" in partial.summary()


def test_the_videocapture_reference_needs_the_lint_exemption():
    c = CHECKS["drift_videocapture_get"]
    assert not c.lint and [f.rule for f in lint(c.reference)] == ["io"]
    assert all(CHECKS[cid].lint for cid in CHECKS if cid != "drift_videocapture_get")


def test_an_empty_answer_never_passes():
    for cid in ("shape_calchist", "dtype_uint32_add", "draw_float_center", "call_imdecode_none", "drift_mser_default"):
        g = grade(CHECKS[cid], "def solve(**kwargs):\n    return {}\n")
        assert not g.passed and g.stage == "contract" and "missing output" in g.detail, str(g)


# ---------------------------------------------------------------- verifier sensitivity: shapes

def _verify(cid: str, case: Case, out: dict):
    return CHECKS[cid].verify(case, out)


def test_hough_verifier_accepts_reasonable_parameters_and_rejects_wrong_geometry():
    bad = 0
    n = 0
    for seed in range(40):
        img, gt = shapes.line_scene(seed)
        case = Case({"edges": img}, gt)
        for thr, mll, gap in itertools.product([20, 30, 50, 80, 100], [20, 30, 50, 60], [5, 10, 20]):
            n += 1
            lines = cv2.HoughLinesP(img, 1, np.pi / 180, thr, minLineLength=mll, maxLineGap=gap)
            bad += not _verify("shape_houghlinesp", case, {"segments": lines.reshape(-1, 4)})
    assert bad == 0, f"{bad}/{n} reasonable parameter choices rejected"
    img, gt = shapes.line_scene(0)
    case = Case({"edges": img}, gt)
    shifted = np.array([[x1 + 8, y1, x2 + 8, y2] for (x1, y1), (x2, y2) in gt])  # lines moved 8 px sideways
    assert not _verify("shape_houghlinesp", case, {"segments": shifted})
    assert not _verify("shape_houghlinesp", case, {"segments": np.array([[gt[0][0][0], gt[0][0][1], gt[0][1][0], gt[0][1][1]]])})


def test_hough_normal_form_verifiers_accept_any_threshold_at_or_below_the_votes_and_reject_wrong_lines():
    for seed in range(5000, 5040):
        img, gt = shapes.line_scene(seed)
        case = Case({"edges": img, "min_votes": shapes.LINE_MIN_VOTES}, gt)
        for thr in (60, 80):
            hl = cv2.HoughLines(img, 1, np.pi / 180, thr).reshape(-1, 2)
            assert _verify("shape_houghlines_ctrl", case, {"lines": hl}), (seed, thr)
        acc = cv2.HoughLinesWithAccumulator(img, 1, np.pi / 180, shapes.LINE_MIN_VOTES).reshape(-1, 3)
        assert _verify("shape_houghlines_acc", case, {"lines": acc}), seed
    img, gt = shapes.line_scene(5000)
    case = Case({"edges": img, "min_votes": shapes.LINE_MIN_VOTES}, gt)
    acc = cv2.HoughLinesWithAccumulator(img, 1, np.pi / 180, shapes.LINE_MIN_VOTES).reshape(-1, 3)
    assert not _verify("shape_houghlines_acc", case, {"lines": acc[::-1]}), "votes must be strongest first"
    assert not _verify("shape_houghlines_acc", case, {"lines": acc[:1]}), "every manufactured line must be present"


def test_convexity_defect_scenes_stay_inside_the_calibrated_tolerance():
    worst_pt = worst_d = 0.0
    for seed in range(1000, 1150):
        mask, pts_gt, depth_gt = shapes.star_scene(seed)
        for mode in (cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE):
            cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, mode)[0], key=cv2.contourArea)
            d = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False)).reshape(-1, 4)
            depth = d[:, 3] / 256.0
            keep = depth > 3
            order = np.argsort(-depth[keep])
            pts, dep = cnt.reshape(-1, 2)[d[keep, 2]][order], depth[keep][order]
            assert len(dep) == len(depth_gt)
            worst_pt = max(worst_pt, np.abs(pts - pts_gt).max())
            worst_d = max(worst_d, np.abs(dep - depth_gt).max())
    assert worst_pt <= 2.0 < shapes.CD_POINT_TOL and worst_d <= 1.35 < shapes.CD_DEPTH_TOL, (worst_pt, worst_d)


def test_chessboard_tolerance_holds_for_both_finders_on_each_others_boards():
    for check_id, finder_ids in (("shape_chessboard", [cv2.findChessboardCorners, cv2.findChessboardCornersSB]),
                                 ("shape_chessboard_sb", [cv2.findChessboardCorners, cv2.findChessboardCornersSB])):
        for case in CHECKS[check_id].make_cases():
            for finder in finder_ids:
                ok, c = finder(case.inputs["img"], (case.inputs["cols"], case.inputs["rows"]))
                assert ok and _verify(check_id, case, {"corners": c.reshape(-1, 2)}), (check_id, finder.__name__)
            half_square = case.truth + [10, 0]
            assert not _verify(check_id, case, {"corners": half_square}), "a half-square shift must be rejected"
    worst = 0.0
    for cols, rows, sq, angle, blur in itertools.product((6, 7), (4, 5), (20, 28), (0.0, 6.0, -5.0), (0.0, 1.2)):
        img, gt = shapes.board(cols, rows, sq=sq, angle=angle, blur=blur)
        for finder in (cv2.findChessboardCorners, cv2.findChessboardCornersSB):
            ok, c = finder(img, (cols, rows))
            assert ok
            worst = max(worst, np.linalg.norm(gt[:, None] - c.reshape(-1, 2)[None], axis=2).min(axis=1).max())
    assert worst < shapes.CB_TOL, worst


def test_min_triangle_tolerance_is_tight_and_rejects_a_bigger_triangle():
    case = CHECKS["shape_mintriangle"].make_cases()[0]
    tri = case.truth
    area, got = cv2.minEnclosingTriangle(case.inputs["pts"])
    assert _verify("shape_mintriangle", case, {"triangle": got.reshape(3, 2), "area": float(area)})
    assert np.abs(got.reshape(3, 2) - tri).max() < 1e-3
    c = tri.mean(axis=0)
    bigger = (c + (tri - c) * 1.05).astype(np.float32)  # still encloses every point, but not the minimum
    assert not _verify("shape_mintriangle", case, {"triangle": bigger, "area": shapes._tri_area(bigger)})


def test_gftt_is_accepted_for_twelve_reasonable_settings():
    for q, md in itertools.product((0.01, 0.05, 0.1, 0.2), (5, 10, 20)):
        for seed in range(300, 340):
            img, truth = shapes.rect_scene(seed)
            corners = cv2.goodFeaturesToTrack(img, len(truth), q, md).reshape(-1, 2)
            assert _verify("shape_gftt_ctrl", Case({"img": img, "max_corners": len(truth)}, truth), {"corners": corners}), (q, md, seed)
    img, truth = shapes.rect_scene(300)
    case = Case({"img": img, "max_corners": len(truth)}, truth)
    assert not _verify("shape_gftt_ctrl", case, {"corners": truth[::2] + 30}), "corners at the wrong places"


def test_approxpoly_verifier_accepts_both_chain_modes_and_rejects_the_wrong_vertex_count():
    ns = set()
    for seed in range(3000, 3300):
        mask, v = shapes.polygon_scene(seed)
        ns.add(len(v))
        for mode in (cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE):
            c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, mode)[0], key=cv2.contourArea)
            a = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
            assert _verify("shape_approxpoly_ctrl", Case({"mask": mask}, v), {"vertices": a}), (seed, mode)
    assert ns == {3, 4, 5, 6, 7, 8}
    mask, v = shapes.polygon_scene(3000)
    assert not _verify("shape_approxpoly_ctrl", Case({"mask": mask}, v), {"vertices": v[:-1]})
    assert not _verify("shape_approxpoly_ctrl", Case({"mask": mask}, v), {"vertices": v + 12})


def test_aruco_centres_survive_corner_refinement():
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for ref in (cv2.aruco.CORNER_REFINE_NONE, cv2.aruco.CORNER_REFINE_SUBPIX, cv2.aruco.CORNER_REFINE_CONTOUR):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = ref
        for case in CHECKS["shape_aruco_corners_ctrl"].make_cases():
            corners, ids, _ = cv2.aruco.ArucoDetector(dic, params).detectMarkers(case.inputs["img"])
            centers = np.array([c.reshape(4, 2).mean(axis=0) for c in corners])
            assert _verify("shape_aruco_corners_ctrl", case, {"centers": centers[np.argsort(centers[:, 0])]})


# ---------------------------------------------------------------- verifier sensitivity: dtypes, drawing, calling

def test_fillpoly_accepts_any_sane_float_to_int_conversion_and_rejects_swapped_axes():
    for case in CHECKS["dtype_fillpoly"].make_cases():
        pf, pi = case.inputs["poly_f"], case.inputs["poly_i"]

        def fill(p):
            m = np.zeros((128, 128), np.uint8)
            cv2.fillPoly(m, [np.asarray(p).astype(np.int32)], 255)
            return m
        for conv in (np.round, np.floor, np.ceil, np.trunc):
            assert _verify("dtype_fillpoly", case, {"mask_f": fill(conv(pf)), "mask_i": fill(pi)}), conv.__name__
        assert not _verify("dtype_fillpoly", case, {"mask_f": fill(np.round(pf)[:, ::-1]), "mask_i": fill(pi)})
        assert not _verify("dtype_fillpoly", case, {"mask_f": fill(np.round(pf)), "mask_i": fill(pi[:, ::-1])})
        assert not _verify("dtype_fillpoly", case, {"mask_f": np.zeros((128, 128), np.uint8), "mask_i": fill(pi)})


def test_fillpoly_conversions_measured_over_200_polygons():
    lo = 1.0
    for seed in range(6000, 6200):
        rng = np.random.default_rng(seed)
        pf = dtypes._poly(rng, int(rng.integers(5, 8))).astype(np.float32)
        base = np.zeros((128, 128), np.uint8)
        cv2.fillPoly(base, [np.round(pf).astype(np.int32)], 255)
        for conv in (np.floor, np.ceil, np.trunc):
            m = np.zeros((128, 128), np.uint8)
            cv2.fillPoly(m, [conv(pf).astype(np.int32)], 255)
            lo = min(lo, float(np.logical_and(m, base).sum() / np.logical_or(m, base).sum()))
    assert 0.95 <= lo < 0.96 and lo > dtypes.FP_IOU


def test_float16_tolerance_is_two_ulp():
    case = CHECKS["dtype_float16_pipeline"].make_cases()[0]
    a, b = case.inputs["a"], case.inputs["b"]
    hybrid = cv2.GaussianBlur(cv2.addWeighted(a, 0.5, b, 0.5, 0).astype(np.float32), (5, 5), 0).astype(np.float16)
    assert _verify("dtype_float16_pipeline", case, {"smoothed": hybrid})
    up = case.truth.copy()
    for _ in range(3):
        up = np.nextafter(up, np.float16(np.inf))
    assert not _verify("dtype_float16_pipeline", case, {"smoothed": up}), "3 ulp off must be rejected"
    up1 = np.nextafter(case.truth, np.float16(np.inf))
    assert _verify("dtype_float16_pipeline", case, {"smoothed": up1}), "1 ulp off is within the rounding of the pipeline"


def test_remap_verifier_accepts_float32_arithmetic_and_numpy_bilinear_and_rejects_a_half_pixel_shift():
    for seed in range(7000, 7050):
        img, amp, period = dtypes.smooth_image(seed), 3.7, 40.0
        case = Case({"img": img, "amplitude": amp, "period": period}, dtypes._rm_truth(img, amp, period))
        h, w = img.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx = xs + np.float32(amp) * np.sin(np.float32(2 * np.pi) * ys / np.float32(period))
        assert _verify("dtype_remap_maps", case, {"warped": cv2.remap(img, mx.astype(np.float32), ys, cv2.INTER_LINEAR)})
        X = np.arange(w)[None, :] + amp * np.sin(2 * np.pi * np.arange(h)[:, None] / period)
        x0 = np.floor(X).astype(int)
        fx = X - x0

        def px(ix):
            out = np.zeros((h, w))
            ok = (ix >= 0) & (ix < w)
            out[ok] = img[np.nonzero(ok)[0], ix[ok]]
            return out
        val = np.round(px(x0) * (1 - fx) + px(x0 + 1) * fx).astype(np.uint8)
        assert _verify("dtype_remap_maps", case, {"warped": val}), seed
    shifted = cv2.remap(img, (mx + 0.5).astype(np.float32), ys, cv2.INTER_LINEAR)
    assert not _verify("dtype_remap_maps", case, {"warped": shifted})


def test_float_center_iou_threshold_separates_one_pixel_from_wrong_place():
    lo, hi = 1.0, 0.0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        p = np.array([rng.integers(15, 45), rng.integers(15, 65)])
        q = np.array([rng.integers(70, 105), rng.integers(15, 65)])
        mid = np.floor((p + q) / 2).astype(int)

        def disk(c):
            m = np.zeros((80, 120), np.uint8)
            cv2.circle(m, (int(c[0]), int(c[1])), 14, 255, -1)
            return m
        for dx, dy in itertools.product((0, 1), (0, 1)):
            lo = min(lo, drawing.mask_iou(disk(mid + [dx, dy]), disk(mid)))
        hi = max(hi, drawing.mask_iou(disk(p), disk(mid)))
    assert lo >= 0.88 and hi <= 0.28 and hi < drawing.FC_IOU < lo


def test_resize_content_check_accepts_every_interpolation_and_rejects_right_shaped_junk():
    for seed in range(8000, 8015):
        for case in [Case({}, cv2.resize(calling._color_smooth(seed, h1, w1), (w2, h2), interpolation=cv2.INTER_LINEAR))
                     for (h1, w1), (h2, w2) in (((48, 64), (60, 90)), ((90, 70), (40, 100)))]:
            h2, w2 = case.truth.shape[:2]
            src = calling._color_smooth(seed, 48, 64) if (h2, w2) == (60, 90) else calling._color_smooth(seed, 90, 70)
            for flag in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4):
                assert _verify("call_resize_order", case, {"resized": cv2.resize(src, (w2, h2), interpolation=flag)}), flag
            assert not _verify("call_resize_order", case, {"resized": np.ones_like(case.truth)})
            assert not _verify("call_resize_order", case, {"resized": case.truth[::-1]})
            assert not _verify("call_resize_order", case, {"resized": np.roll(case.truth, 10, axis=1)})


def test_gaussian_check_accepts_standard_kernel_choices_and_rejects_a_box_blur():
    for case in CHECKS["call_gaussian_even"].make_cases():
        img, sigma = case.inputs["img"], case.inputs["sigma"]
        for k in (2 * math.ceil(3 * sigma) + 1, 2 * math.ceil(4 * sigma) + 1):
            assert _verify("call_gaussian_even", case, {"blurred": cv2.GaussianBlur(img, (k, k), sigma)}), (sigma, k)
        f32 = np.round(cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)).astype(np.uint8)
        assert _verify("call_gaussian_even", case, {"blurred": f32})
        assert not _verify("call_gaussian_even", case, {"blurred": cv2.blur(img, (5, 5))})
        assert not _verify("call_gaussian_even", case, {"blurred": img})


# ---------------------------------------------------------------- verifier sensitivity: drift

def test_warp_tolerance_is_pinned_to_the_recorded_414_arrays():
    z4 = np.load(ROOT / "probes/results/probe_4.14.0.npz")
    z5 = np.load(ROOT / "probes/results/probe_5.0.0.npz")
    case = CHECKS["drift_warp_tolerance"].make_cases()[0]
    a4, p4 = z4["warpAffine_rotate_fingerprint"], z4["warpPerspective_fingerprint"]
    # the scene regenerated here is the probes' scene: the 5.0.0 arrays are reproduced bit for bit
    assert np.array_equal(case.truth["affine"], z5["warpAffine_rotate_fingerprint"])
    assert np.array_equal(case.truth["perspective"], z5["warpPerspective_fingerprint"])
    # 4.14 and 5.0 differ: pixel-exact equality would reject the other version's output...
    for old, new in ((a4, case.truth["affine"]), (p4, case.truth["perspective"])):
        diff = np.abs(old.astype(int) - new.astype(int))
        assert diff.max() == 4 and 0.030 < (diff > 0).mean() < 0.034
        assert drift.pearson(old, new) > 0.99998
    # ...and the verifier does not
    assert _verify("drift_warp_tolerance", case, {"affine": a4, "perspective": p4})
    assert not np.array_equal(a4, case.truth["affine"])
    # while genuinely different warps are rejected, by the max-difference bound or the correlation
    img, M, H = case.inputs["img"], case.inputs["M"], case.inputs["H"]
    size = (200, 160)
    variants = {
        "nearest": cv2.warpAffine(img, M, size, flags=cv2.INTER_NEAREST),
        "bicubic": cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC),
        "half a degree": cv2.warpAffine(img, cv2.getRotationMatrix2D((100.0, 80.0), 16.5, 1.0), size),
        "half a pixel": cv2.warpAffine(img, M + np.array([[0, 0, 0.5], [0, 0, 0]]), size),
        "replicated border": cv2.warpAffine(img, M, size, borderMode=cv2.BORDER_REPLICATE),
    }
    for name, got in variants.items():
        assert not _verify("drift_warp_tolerance", case, {"affine": got, "perspective": p4}), name
    assert not _verify("drift_warp_tolerance", case, {"affine": a4, "perspective": cv2.warpPerspective(img, H, size, flags=cv2.INTER_CUBIC)})


def test_text_box_slack_calibration_sweep():
    font = cv2.FONT_HERSHEY_SIMPLEX
    worst = 0.0
    n = 0
    for text, scale, thick in itertools.product(["OpenCV 5", "gjpqy yaq", "Quality 42", "HELLO", "Wide text here", "i l 1 ."],
                                                [0.5, 0.8, 1.0, 1.5, 2.0], [1, 2, 3, 4, 5, 6]):
        canvas = np.zeros((140, 700), np.uint8)
        cv2.putText(canvas, text, (20, 80), font, scale, 255, thick)
        ys, xs = np.nonzero(canvas)
        ix0, iy0, ix1, iy1 = xs.min(), ys.min(), xs.max(), ys.max()
        (w, h), base = cv2.getTextSize(text, font, scale, thick)
        x0, y0, x1, y1 = 20, 80 - h, 20 + w, 80 + base
        n += 1
        assert x0 <= ix0 and y0 <= iy0 and x1 >= ix1 and y1 >= iy1, (text, scale, thick)
        inkh = iy1 - iy0 + 1
        worst = max(worst, max((v - drift.TB_SLACK_PX) / inkh for v in (ix0 - x0, iy0 - y0, x1 - ix1, y1 - iy1)))
    assert n == 180 and 0.25 < worst < 0.27 < drift.TB_SLACK_FRAC / 1.45


def test_text_box_check_requires_blended_glyphs_filled_box_and_nothing_outside():
    case = CHECKS["drift_gettextsize_box"].make_cases()[0]
    t = case.inputs
    (w, h), base = cv2.getTextSize(t["text"], cv2.FONT_HERSHEY_SIMPLEX, t["scale"], t["thickness"])
    box = np.array([t["x"], t["y"] - h, t["x"] + w, t["y"] + base])
    img = np.zeros_like(t["img"])
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), 128, -1)
    cv2.putText(img, t["text"], (t["x"], t["y"]), cv2.FONT_HERSHEY_SIMPLEX, t["scale"], 255, t["thickness"])
    assert drift.text_box_problem(case.truth, img, box) is None
    # 5.0 draws Hershey glyphs with intermediate gray values even with LINE_8 (4.x drew only the colour itself)
    assert len(np.unique(img[case.truth[0]])) > 20
    assert drift.text_box_problem(case.truth, np.where(img == 128, 0, img).astype(np.uint8), box) is not None  # box not filled
    assert drift.text_box_problem(case.truth, img, box + [0, 0, 40, 0]) is not None  # loose
    assert drift.text_box_problem(case.truth, img, box + [5, 0, 0, 0]) is not None  # clips the ink
    stray = img.copy()
    stray[0, 0] = 77
    assert drift.text_box_problem(case.truth, stray, box) is not None  # something drawn outside the box


def test_mser_verifier_wants_a_box_per_glyph():
    for case in CHECKS["drift_mser_default"].make_cases():
        glyphs = np.array(case.truth)
        assert _verify("drift_mser_default", case, {"boxes": glyphs})
        assert not _verify("drift_mser_default", case, {"boxes": glyphs[:-1]})
        assert not _verify("drift_mser_default", case, {"boxes": np.zeros((0, 4), np.int32)})
        assert not _verify("drift_mser_default", case, {"boxes": glyphs + [40, 0, 0, 0]})


# ---------------------------------------------------------------- the 5.0.0 measurements the checks stand on

def test_opencv_5_0_int64_is_narrowed_with_wraparound_even_with_a_64_bit_dtype_argument():
    a = np.array([[2 ** 40, 5], [2 ** 41 + 3, 7]], np.int64)
    assert cv2.add(a, a).dtype == np.int32 and cv2.add(a, a).tolist() == [[0, 10], [6, 14]]
    assert cv2.add(a, a, dtype=cv2.CV_64S).tolist() == [[0, 10], [6, 14]]
    assert cv2.subtract(a, a[::-1]).dtype == np.int32 and cv2.multiply(a, a).dtype == np.int32


def test_opencv_5_0_bool_support_is_partial_and_bitwise_not_does_not_invert():
    m = np.zeros((4, 6), bool)
    m[1:3, 2:5] = True
    assert cv2.countNonZero(m) == 6 and cv2.bitwise_and(m, m).dtype == bool
    inv = cv2.bitwise_not(m)
    assert inv.dtype == bool and set(np.unique(inv.view(np.uint8)).tolist()) == {254, 255}  # every element non-zero
    assert inv.view(np.uint8).astype(bool).all() and not np.array_equal(inv.view(np.uint8) != 0, ~m)
    assert set(np.unique(cv2.bitwise_not(m.astype(np.uint8))).tolist()) == {254, 255}
    for fn in (lambda: cv2.connectedComponentsWithStats(m), lambda: cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
               lambda: cv2.threshold(m, 0, 255, cv2.THRESH_BINARY), lambda: cv2.dilate(m, np.ones((3, 3), np.uint8)),
               lambda: cv2.GaussianBlur(m, (3, 3), 0)):
        with pytest.raises(cv2.error):
            fn()


def test_opencv_5_0_uint32_and_float16_are_accepted_by_arithmetic_but_not_by_filters():
    u = np.array([[4000000000, 1]], np.uint32)
    assert cv2.add(u, u).tolist() == [[4294967295, 2]] and cv2.add(u, u).dtype == np.uint32
    h = np.ones((8, 8), np.float16)
    assert cv2.add(h, h).dtype == np.float16 and cv2.addWeighted(h, 0.5, h, 0.5, 0).dtype == np.float16
    for fn in (lambda: cv2.GaussianBlur(u.astype(np.uint32), (3, 3), 0), lambda: cv2.resize(u, (4, 4)),
               lambda: cv2.GaussianBlur(h, (3, 3), 0), lambda: cv2.resize(h, (4, 4)), lambda: cv2.Canny(h, 1, 2)):
        with pytest.raises(cv2.error):
            fn()


def test_opencv_5_0_text_is_antialiased_and_mser_defaults_find_nothing():
    img = np.zeros((60, 240), np.uint8)
    cv2.putText(img, "OpenCV 5", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_8)
    assert len(np.unique(img)) > 20 and cv2.countNonZero(img) == 1621  # the recorded probe: 1045 on 4.14
    assert cv2.getTextSize("OpenCV 5", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2) == ((136, 27), 6)
    for case in CHECKS["drift_mser_default"].make_cases():
        assert len(cv2.MSER_create().detectRegions(case.inputs["img"])[0]) == 0
        assert len(cv2.MSER_create(min_diversity=0.1).detectRegions(case.inputs["img"])[0]) == 0
        assert len(cv2.MSER_create(min_diversity=0.0).detectRegions(case.inputs["img"])[0]) > 50
    assert cv2.VideoCapture().get(cv2.CAP_PROP_FPS) == -1.0 and cv2.VideoCapture().get(999) == -1.0


def test_opencv_5_0_remap_does_not_check_the_second_map_when_the_first_is_float32():
    img = (np.arange(64 * 96).reshape(64, 96) % 251).astype(np.uint8)
    xs, ys = np.meshgrid(np.arange(96), np.arange(64))
    mx = (xs + 0.5).astype(np.float32)
    good = cv2.remap(img, mx, ys.astype(np.float32), cv2.INTER_LINEAR)
    assert np.abs(cv2.remap(img, mx, ys, cv2.INTER_LINEAR).astype(int) - good).max() > 100  # int64 map2: no exception, garbage
    with pytest.raises(cv2.error):
        cv2.remap(img, mx.astype(np.float64), ys.astype(np.float32), cv2.INTER_LINEAR)
    with pytest.raises(cv2.error):
        cv2.remap(img, xs.astype(np.int32), ys.astype(np.int32), cv2.INTER_LINEAR)


def test_opencv_5_0_drawing_accepts_numpy_ints_and_row_strided_views_only():
    img = np.zeros((30, 30), np.uint8)
    cv2.circle(img, tuple(np.array([10, 12], np.int64)), 3, 255, -1)
    cv2.circle(img, np.array([10, 12]), 3, 255, -1)
    cv2.circle(img[::2], (5, 5), 2, 255, -1)
    for bad in (lambda: cv2.circle(img, (10.0, 12.0), 3, 255, -1), lambda: cv2.circle(img, tuple(np.array([10, 12], np.float32)), 3, 255, -1),
                lambda: cv2.circle(img.T, (5, 5), 2, 255, -1), lambda: cv2.circle(np.asfortranarray(img), (5, 5), 2, 255, -1),
                lambda: cv2.rectangle(np.zeros((9, 9, 3), np.uint8)[:, :, ::-1], (1, 1), (4, 4), (255, 0, 0), 1)):
        with pytest.raises(cv2.error):
            bad()
    assert cv2.imdecode(np.frombuffer(b"not an image", np.uint8), cv2.IMREAD_COLOR) is None
