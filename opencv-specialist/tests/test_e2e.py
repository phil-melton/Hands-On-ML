"""Stage 7 E2E set: 20 multi-step tasks, their grader and the runner.

What is pinned, in order: the registry (20 tasks, 2-4 dependent steps, every family, at least 4 tasks on the held-out
families), the seeds (inside [6_000_000, 7_000_000), disjoint from dev/test/foundry), determinism (same bytes twice),
solvability and truth (each task's reference pipeline, built from the families' REFERENCE solutions run through
sandbox.run_solution, recovers the manufactured answer and passes the grader), the grader (rejects a missing or
malformed answer and an answer perturbed beyond each tolerance), the traps the tasks are meant to carry, the tolerance
provenance (imported from the families, not restated), the two arms' command lines, the work dirs (inputs only, no
truth), and the report arithmetic on a small synthetic set. No model and no claude.exe session is started.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvbench import splits
from cvbench.families import (color_segmentation as cs, components_stats as cc, feature_homography as fh,
                              histogram_stats as hi, hough_segments as hs)
from serve.e2e import compose as cp
from serve.e2e import grade as G
from serve.e2e import reference as REF
from serve.e2e import run as R
from serve.e2e import tasks as T

IDS = list(T.TASK_IDS)
E2E = Path(T.__file__).resolve().parent


@pytest.fixture(scope="module")
def all_tasks():
    return T.build_all()


@lru_cache(maxsize=None)
def ref_answer_and_dir(n: int, tmp_root: str):
    task = T.build(n)
    wd = Path(tmp_root) / f"ref-{n}"
    wd.mkdir(parents=True, exist_ok=True)
    return REF.reference_answer(task, wd), wd


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("e2e-ref"))


# ---------------------------------------------------------------- the registry

def test_twenty_tasks_with_two_to_four_steps(all_tasks):
    assert len(all_tasks) == 20
    assert [t.id for t in all_tasks] == IDS and len(set(IDS)) == 20
    assert [t.number for t in all_tasks] == list(range(1, 21))
    for t in all_tasks:
        assert 2 <= len(t.steps) <= 4, (t.id, t.steps)
        assert t.title and t.prompt_body.strip() and t.images


def test_every_family_is_used_and_four_tasks_use_the_held_out_ones(all_tasks):
    used = {f for t in all_tasks for f in t.families}
    assert used == set(splits.FAMILIES)
    held = {name for name, row in splits.FAMILIES.items() if row.held_out}
    assert held == {"chessboard_corners", "color_segmentation"}
    on_held = [t.id for t in all_tasks if t.uses_held_out]
    assert len(on_held) >= 4, on_held
    for t in all_tasks:
        assert set(t.held_out) == set(t.families)
        assert all(t.held_out[f] == (f in held) for f in t.families)
    # the board and the colour tasks are the held-out ones
    assert {t.number for t in all_tasks if t.uses_held_out} == {6, 7, 8, 9, 10, 20}


def test_truth_keys_match_the_schema_and_every_spec_names_its_tolerance_source(all_tasks):
    for t in all_tasks:
        assert set(t.schema) <= set(t.truth), t.id
        extra = set(t.truth) - set(t.schema)
        assert extra <= {"layout"}, (t.id, extra)
        for k, spec in t.truth.items():
            assert spec["kind"] in {"int", "ints", "str", "bool", "num", "rel", "point", "points", "angle", "sorted_nums",
                                    "dict", "homography", "present", "layout"}, (t.id, k)
            assert spec.get("src"), (t.id, k)
            for x in (spec.get("tol"), spec.get("rtol")):
                assert x is None or (math.isfinite(x) and x >= 0), (t.id, k, x)
            for x in spec.get("tols", []):
                assert math.isfinite(x) and x >= 0
            for kid in spec.get("children", {}).values():
                assert kid.get("src")


def test_prompts_name_the_keys_the_images_and_no_tool(all_tasks):
    for t in all_tasks:
        p = T.render_prompt(t, "/home/x/run/alone/" + t.id, "\\\\wsl.localhost\\Ubuntu-24.04\\home\\x\\run\\alone\\" + t.id)
        for k in t.schema:
            assert f'"{k}"' in p, (t.id, k)
        for name in t.images:
            assert f"/home/x/run/alone/{t.id}/{name}.png" in p
        assert "/home/x/run/alone/" + t.id + "/answer.json" in p
        assert "MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- /home/ptmel/venvs/cv5/bin/python" in p
        low = p.lower()
        for banned in ("opencv_solve", "specialist", "mcp", "tolerance", "ollama"):
            assert banned not in low, (t.id, banned)
        assert "graded automatically against an exact manufactured truth" in p


# ---------------------------------------------------------------- seeds

def test_seed_range_is_disjoint_from_dev_test_and_foundry(all_tasks):
    lo, hi_ = cp.SEED_RANGE
    assert (lo, hi_) == (6_000_000, 7_000_000)
    for rng in (splits.DEV_RANGE, splits.TEST_RANGE, splits.FOUNDRY_RANGE):
        assert hi_ <= rng[0] or lo >= rng[1], rng
    assert splits.FOUNDRY_RANGE[1] == lo  # the E2E range starts where the foundry's ends
    for t in all_tasks:
        for name, seed in t.seeds.items():
            assert lo <= seed < hi_, (t.id, name, seed)
            blo = lo + T.BLOCK * t.number
            assert blo <= seed < blo + T.BLOCK, (t.id, name, seed)


def test_scene_seeds_are_not_used_elsewhere_in_the_task_set(all_tasks):
    seeds = [s for t in all_tasks for s in t.seeds.values()]
    assert len(seeds) == len(set(seeds))


# ---------------------------------------------------------------- determinism and inputs

def _digest(task: T.Task) -> dict:
    return {"images": {k: cv2.imencode(".png", v)[1].tobytes() for k, v in sorted(task.images.items())},
            "prompt": task.prompt_body, "schema": task.schema, "truth": task.truth_json(), "seeds": task.seeds}


@pytest.mark.parametrize("n", range(1, 21))
def test_task_builds_deterministically(n):
    assert _digest(T.fresh(n)) == _digest(T.fresh(n))


def test_images_are_8_bit_png_round_trips(all_tasks):
    for t in all_tasks:
        for name, im in t.images.items():
            assert im.dtype == np.uint8 and im.ndim in (2, 3) and im.flags["C_CONTIGUOUS"], (t.id, name)
            ok, buf = cv2.imencode(".png", im)
            assert ok and np.array_equal(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), im)


# ---------------------------------------------------------------- solvability and truth

@pytest.mark.parametrize("n", range(1, 21))
def test_reference_pipeline_recovers_the_truth_and_passes_the_grader(n, tmp_root):
    task = T.build(n)
    ans, wd = ref_answer_and_dir(n, tmp_root)
    res = G.grade_answer(task, json.loads(json.dumps(ans)), wd)  # through JSON, as the orchestrator's answer.json is
    assert res["passed"], {k: v for k, v in res["keys"].items() if not v["ok"]}
    assert res["n_pass"] == res["n_keys"] == len(task.truth)
    assert res["extra_keys"] == []


def test_the_reference_answers_use_only_the_schema_keys(tmp_root):
    for n in range(1, 21):
        ans, _ = ref_answer_and_dir(n, tmp_root)
        assert set(ans) == set(T.build(n).schema), n


# ---------------------------------------------------------------- the grader

def _bump(spec: dict, v):
    """A value beyond the tolerance of ``spec`` (or malformed) derived from the correct value v."""
    k = spec["kind"]
    if k == "int":
        return v + 1
    if k == "ints":
        return [v[0] + 1] + list(v[1:])
    if k == "str":
        return "zzz"
    if k == "bool":
        return not v
    if k == "num":
        return v + 2 * spec["tol"] + 1e-3
    if k == "rel":
        return v * (1 + 1e-6) + 1e-6
    if k == "point":
        return [v[0] + 2 * spec["tol"] + 0.01, v[1]]
    if k == "points":
        return [[v[0][0] + 2 * spec["tol"] + 0.01, v[0][1]]] + [list(p) for p in v[1:]]
    if k == "angle":
        return v + 2 * spec["tol"] + 0.1
    if k == "sorted_nums":
        return [v[0] + 2 * spec["tols"][0] + 1e-3] + list(v[1:])
    if k == "dict":
        first = next(iter(spec["children"]))
        out = dict(v)
        out[first] = _bump(spec["children"][first], v[first])
        return out
    if k == "homography":
        H = copy.deepcopy(v)
        H[0][2] += 3 * spec["tol"] + 3
        return H
    if k == "present":
        return "not a number"
    raise AssertionError(k)


@pytest.mark.parametrize("n", range(1, 21))
def test_grader_fails_each_key_perturbed_beyond_its_tolerance(n, tmp_root):
    task = T.build(n)
    ans, wd = ref_answer_and_dir(n, tmp_root)
    ans = json.loads(json.dumps(ans))
    for key, spec in task.truth.items():
        if spec["kind"] == "layout":
            continue
        bad = dict(ans)
        bad[key] = _bump(spec, ans[key])
        res = G.grade_answer(task, bad, wd)
        assert not res["passed"], (task.id, key)
        assert not res["keys"][key]["ok"], (task.id, key)
        # the text tasks' layout check reads scale and text_size, so it fails with them
        others = {k: r for k, r in res["keys"].items() if k != key and not (k == "layout" and key in ("scale", "text_size"))}
        assert all(r["ok"] for r in others.values()), (task.id, key)


def test_grader_accepts_values_just_inside_each_numeric_tolerance(tmp_root):
    for n in range(1, 21):
        task = T.build(n)
        ans = json.loads(json.dumps(ref_answer_and_dir(n, tmp_root)[0]))
        wd = ref_answer_and_dir(n, tmp_root)[1]
        for key, spec in task.truth.items():
            if spec["kind"] == "num" and spec["tol"] > 0:
                ok = dict(ans)
                ok[key] = spec["value"] + 0.9 * spec["tol"]
                assert G.grade_answer(task, ok, wd)["keys"][key]["ok"], (task.id, key)
            if spec["kind"] == "angle":
                ok = dict(ans)
                ok[key] = spec["value"] - 0.9 * spec["tol"] + spec["period"]  # a full period away is the same angle
                assert G.grade_answer(task, ok, wd)["keys"][key]["ok"], (task.id, key)


def test_missing_malformed_and_wrong_shaped_answers_fail_and_are_recorded(tmp_path):
    task = T.build(3)
    res = G.grade_workdir(task, tmp_path)
    assert res["status"] == "missing" and not res["passed"] and res["n_pass"] == 0 and res["n_keys"] == len(task.truth)
    (tmp_path / "answer.json").write_text("{not json")
    assert G.grade_workdir(task, tmp_path)["status"] == "malformed"
    (tmp_path / "answer.json").write_text("[1, 2]")
    assert G.grade_workdir(task, tmp_path)["status"] == "not_object"
    (tmp_path / "answer.json").write_bytes(b"\xff\xfe\x00")
    assert G.grade_workdir(task, tmp_path)["status"] == "malformed"
    (tmp_path / "answer.json").write_text("{}")
    res = G.grade_workdir(task, tmp_path)
    assert res["status"] == "ok" and not res["passed"] and all(not r["ok"] for r in res["keys"].values())


def test_numbers_must_be_finite_json_numbers_and_extra_keys_are_recorded_not_failed(tmp_root):
    task = T.build(3)
    ans, wd = ref_answer_and_dir(3, tmp_root)
    ans = json.loads(json.dumps(ans))
    for bad in (float("nan"), float("inf"), True, "172.0", None, [172.0]):
        a = dict(ans, longest_length_px=bad)
        assert not G.grade_answer(task, a, wd)["keys"]["longest_length_px"]["ok"], bad
    a = dict(ans, comment="hello")
    res = G.grade_answer(task, a, wd)
    assert res["passed"] and res["extra_keys"] == ["comment"]
    a = dict(ans, n_strokes=ans["n_strokes"] + 0.5)
    assert not G.grade_answer(task, a, wd)["keys"]["n_strokes"]["ok"]
    a = dict(ans, n_strokes=float(ans["n_strokes"]))  # 4.0 is an integer value
    assert G.grade_answer(task, a, wd)["keys"]["n_strokes"]["ok"]


def test_angle_comparison_wraps_around_the_period():
    spec = {"kind": "angle", "value": 179.0, "tol": 3.0, "period": 180.0}
    assert G.grade_value(spec, 1.0).ok and G.grade_value(spec, -1.0).ok and not G.grade_value(spec, 10.0).ok
    spec = {"kind": "angle", "value": 179.0, "tol": 3.0, "period": 360.0}
    assert G.grade_value(spec, -179.0).ok and not G.grade_value(spec, 0.0).ok


def test_sorted_nums_must_be_sorted_and_homography_must_be_normalised(tmp_root):
    spec = {"kind": "sorted_nums", "value": [10.0, 5.0], "tols": [1.0, 1.0]}
    assert G.grade_value(spec, [10.2, 4.9]).ok
    assert not G.grade_value(spec, [5.0, 10.0]).ok
    task = T.build(11)
    ans = json.loads(json.dumps(ref_answer_and_dir(11, tmp_root)[0]))
    H = np.array(ans["H"])
    scaled = (H * 2).tolist()  # the same mapping, but H[2][2] = 2
    assert not G.grade_value(task.truth["H"], scaled).ok


@pytest.mark.parametrize("n", [13, 14])
def test_text_layout_is_graded_on_out_png_with_the_manufactured_canvas(n, tmp_path, tmp_root):
    task = T.build(n)
    ans, wd = ref_answer_and_dir(n, tmp_root)
    ans = json.loads(json.dumps(ans))
    assert G.grade_answer(task, ans, wd)["passed"]
    out = cv2.imread(str(wd / "out.png"))
    assert out is not None and out.shape == task.images["field"].shape
    # no out.png
    other = tmp_path / "empty"
    other.mkdir()
    res = G.grade_answer(task, ans, other)
    assert not res["keys"]["layout"]["ok"] and "out.png is missing" in res["keys"]["layout"]["detail"]
    # the unmodified canvas (nothing drawn), the text drawn outside the field, and a wrong scale all fail
    for name, img, scale in (("blank", task.images["field"].copy(), ans["scale"]),
                             ("outside", _draw_outside(task), ans["scale"]),
                             ("scale", out, ans["scale"] * 0.5)):
        d = tmp_path / name
        d.mkdir()
        cv2.imwrite(str(d / "out.png"), img)
        res = G.grade_answer(task, dict(ans, scale=scale), d)
        assert not res["keys"]["layout"]["ok"], name
    # an edit of the input canvas on disk does not change the verdict: the grader uses the manufactured canvas
    d = tmp_path / "tampered"
    d.mkdir()
    shutil.copy(wd / "out.png", d / "out.png")
    cv2.imwrite(str(d / "field.png"), np.zeros_like(task.images["field"]))
    assert G.grade_answer(task, ans, d)["passed"]


def _draw_outside(task):
    sc = task.extras["tl_scene"]
    img = task.images["field"].copy()
    cv2.putText(img, "X", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [int(c) for c in sc.color], 1)
    return img


# ---------------------------------------------------------------- the traps the tasks carry

def test_connectivity_matters_in_the_component_tasks(all_tasks):
    t5, t18 = T.build(5), T.build(18)
    for t, name in ((t5, "micrograph"), (t18, "mask")):
        mask = (cv2.cvtColor(t.images[name], cv2.COLOR_BGR2GRAY) > 40) if t.images[name].ndim == 3 else t.images[name] > 0
        n8, n4 = (len(cp.label_components(mask, c)[1]) for c in (8, 4))
        assert n4 > n8 == t.truth["n_components"]["value"], t.id
    t7 = T.build(7)
    sc = cs.make_scene(t7.seeds["scene"])
    dom = t7.truth["dominant"]["value"]
    ids = [i for i, s in enumerate(sc.shapes, 1) if s.cls == dom]
    m = np.isin(sc.labels, ids)
    assert len(cp.label_components(m, 4)[1]) > len(cp.label_components(m, 8)[1]) == t7.truth["n_components"]["value"]
    assert np.array_equal(cs.make_scene(t7.seeds["scene"]).img, t7.images["scene"])


def test_the_median_tasks_have_a_real_tie_so_np_median_is_wrong(all_tasks):
    t15 = T.build(15)
    g = cv2.cvtColor(t15.images["image"], cv2.COLOR_BGR2GRAY)
    assert g.size % 2 == 0 and float(np.median(g)) != t15.truth["median_gray"]["value"]
    t5 = T.build(5)
    gray = cv2.cvtColor(t5.images["micrograph"], cv2.COLOR_BGR2GRAY)
    labels, comps = cp.label_components(gray > 40, 8)
    k = 1 + [c.area for c in comps].index(t5.truth["largest_area"]["value"])
    vals = gray[labels == k]
    assert vals.size % 2 == 0 and float(np.median(vals)) != t5.truth["largest_median_gray"]["value"]
    assert int(np.sort(vals)[(vals.size + 1) // 2 - 1]) == t5.truth["largest_median_gray"]["value"]  # the lower median
    # the sample standard deviation (ddof=1), the usual slip, is far outside the population-std tolerance
    pop = float(vals.std())
    assert abs(float(vals.std(ddof=1)) - pop) / pop > 1000 * hi.FLOAT_RTOL
    assert math.isclose(pop, t5.truth["largest_std_gray"]["value"], rel_tol=1e-9)


def test_the_boards_pin_both_corner_order_cases():
    t8, t9 = T.build(8), T.build(9)
    from cvbench.families import chessboard_corners as cb
    assert cb.make_scene(t8.seeds["scene"]).tl_dark is False  # OpenCV lists the corners reversed
    assert cb.make_scene(t9.seeds["scene"]).tl_dark is True
    assert t9.truth["top_left_square_is_dark"]["value"] is True
    # a reversed list answers the "first corner" question wrongly: the last corner is far from the first
    sc = cb.make_scene(t8.seeds["scene"])
    assert np.linalg.norm(sc.corners[-1] - sc.corners[0]) > 10 * t8.truth["first_corner_xy"]["tol"]


def test_one_label_task_fits_and_one_must_shrink():
    assert T.build(13).extras["tl_scene"].shrink is False
    assert T.build(14).extras["tl_scene"].shrink is True


def test_the_longest_stroke_and_the_dominant_colour_are_unambiguous(all_tasks):
    sc = hs.make_scene(T.build(3).seeds["scene"])
    L = sorted(sc.segments, key=lambda s: -s.length)
    assert L[0].length - L[1].length > 2 * hs.tolerance(L[0]) + 2 * hs.tolerance(L[1])
    t6 = T.build(6)
    kids = sorted(t6.truth["area_px2_by_colour"]["children"].items(), key=lambda kv: -kv[1]["value"])
    (best, top), (_, second) = kids[0], kids[1]
    assert top["value"] - second["value"] > top["tol"] + second["tol"]
    assert t6.truth["largest_total_colour"]["value"] == best


# ---------------------------------------------------------------- tolerance provenance

def test_tolerances_come_from_the_family_laws_not_from_restated_numbers():
    # 1. the sources of tasks.py and compose.py hold no literal equal to a family constant
    restated = {6.4, 0.75, 1.9, 1e-9, 1e-6, 7.3, 0.26, 0.15, 0.24, 0.29, 6.3}
    for path in (E2E / "tasks.py", E2E / "compose.py", E2E / "grade.py"):
        tree = ast.parse(path.read_text())
        lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)}
        assert not (lits & restated), (path.name, lits & restated)
    # 2. spot checks: the number in the spec is the family function of the scene
    from cvbench.families import (chessboard_corners as cb, contour_area as ca, perspective_rectify as pr)
    t1 = T.build(1)
    assert t1.truth["corners"]["tol"] == pr.corner_tolerance(pr.make_scene(t1.seeds["scene"]))
    t3 = T.build(3)
    seg = max(hs.make_scene(t3.seeds["scene"]).segments, key=lambda s: s.length)
    assert t3.truth["longest_length_px"]["tol"] == 2 * hs.tolerance(seg)
    assert t3.truth["longest_midpoint_xy"]["tol"] == hs.tolerance(seg)
    t5 = T.build(5)
    assert t5.truth["largest_mean_gray"]["rtol"] == hi.FLOAT_RTOL and t5.truth["largest_std_gray"]["rtol"] == hi.FLOAT_RTOL
    assert T.build(7).truth["largest_centroid"]["tol"] == cc.CENTROID_TOL == T.build(18).truth["largest_centroid"]["tol"]
    assert T.build(7).truth["counts"]["children"]["red"]["kind"] == "int" and cs.TOL_PX == 0
    t8 = T.build(8)
    d8 = cb.tolerance(cb.make_scene(t8.seeds["scene"]))
    assert t8.truth["first_corner_xy"]["tol"] == d8 and t8.truth["pitch_x_px"]["tol"] == 2 * d8
    t11 = T.build(11)
    assert t11.truth["H"]["tol"] == fh.tolerance(fh.make_scene(t11.seeds["scene"]))
    t17 = T.build(17)
    sc17 = ca.make_scene(t17.seeds["scene"])
    assert t17.truth["areas_desc_px2"]["tols"] == [ca.tolerance(s) for s in sorted(sc17.shapes, key=lambda s: -s.area)]
    t6 = T.build(6)
    sc6 = ca.make_scene(t6.seeds["scene"])
    assert sum(k["tol"] for k in t6.truth["area_px2_by_colour"]["children"].values()) <= sum(ca.tolerance(s) for s in sc6.shapes) + 1e-9


def test_propagation_helpers():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert cp.poly_area(sq) == 100 and cp.poly_perimeter(sq) == 40
    # moving every vertex outward by exactly d (along the diagonal) changes the area by less than the bound, and the
    # bound is tight there: it is the exact expansion, the cross terms are what the 0.5 n d^2 covers
    d = 0.5
    e = d / math.sqrt(2)
    grown = np.array([[-e, -e], [10 + e, -e], [10 + e, 10 + e], [-e, 10 + e]])
    dA = cp.poly_area(grown) - 100
    assert dA <= cp.area_tol(sq, d) < 1.01 * dA
    assert math.isclose(cp.angle_tol_deg(100.0, 1.0), math.degrees(math.asin(0.02)))
    assert cp.direction_deg((0, 0), (1, -1)) == 135.0 and cp.direction_deg((5, 5), (5, 9)) == 90.0
    assert cp.ratio_tol(10.0, 0.0, 5.0, 0.0) == 0.0
    assert math.isclose(cp.ratio_tol(10.0, 1.0, 5.0, 1.0), max(abs(11 / 4 - 2), abs(9 / 4 - 2), abs(11 / 6 - 2), abs(9 / 6 - 2)))
    oct_ = cp.octagon(2.0)
    assert oct_.shape == (8, 2) and np.allclose(np.hypot(*oct_.T), 2.0 / math.cos(math.pi / 8))
    # worst_case of a linear map of the corners is attained at an octagon vertex combination and bounds every disc displacement
    c = sq.copy()
    w = cp.worst_case(c, 1.0, lambda cn: [cn[0][0] - cn[2][0]])
    assert 1.9 < w < 2.2  # two corners, each off by up to 1 along x: the difference moves by up to 2 (octagon: 2.16)


def test_labelling_matches_the_family_truth_on_random_scenes():
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        sc = cc.make_scene(seed)
        _, comps = cp.label_components(sc.mask, sc.connectivity)
        truth = sc.truth
        assert sorted((c.area, c.x, c.y, c.w, c.h) for c in comps) == sorted((c.area, c.x, c.y, c.w, c.h) for c in truth)
        assert sorted(comps, key=lambda c: c.order_key)[0].cx == truth[0].cx


# ---------------------------------------------------------------- the runner: command lines

def test_the_two_arms_differ_only_in_the_mcp_flag_and_the_allowed_tool():
    kw = dict(model="sonnet", claude_exe="/c/claude.exe", mcp_config_win="C:\\x\\mcp.json", workdir_win="C:\\x\\w", max_turns=40)
    alone = R.build_command(arm="alone", **kw)
    spec = R.build_command(arm="specialist", **kw)
    assert "--mcp-config" not in alone and "--mcp-config" in spec
    assert spec[spec.index("--mcp-config") + 1] == "C:\\x\\mcp.json"
    assert R.MCP_TOOL == "mcp__opencv-expert__opencv_solve"
    assert R.MCP_TOOL not in alone[alone.index("--allowedTools") + 1].split(",")
    assert spec[spec.index("--allowedTools") + 1].split(",") == ["Bash", "Read", "Write", R.MCP_TOOL]
    stripped = list(spec)
    i = stripped.index("--mcp-config")
    del stripped[i:i + 2]
    stripped[stripped.index("--allowedTools") + 1] = "Bash,Read,Write"
    assert stripped == alone
    for cmd in (alone, spec):
        assert cmd[0] == "/c/claude.exe" and "-p" in cmd and "--strict-mcp-config" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "stream-json" and "--verbose" in cmd
        assert cmd[cmd.index("--model") + 1] == "sonnet" and cmd[cmd.index("--tools") + 1] == "Bash,Read,Write"
        assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk" and "--no-session-persistence" in cmd
        assert cmd[cmd.index("--max-turns") + 1] == "40"
        assert cmd[cmd.index("--setting-sources") + 1] == "" and "--disable-slash-commands" in cmd and "--no-chrome" in cmd
        assert cmd[cmd.index("--add-dir") + 1] == "C:\\x\\w"
    with pytest.raises(ValueError):
        R.build_command(arm="specialist", model="sonnet")
    with pytest.raises(ValueError):
        R.build_command(arm="both", model="sonnet")
    assert "--setting-sources" not in R.build_command(arm="alone", model="m", isolation="none")  # presets change only the isolation flags
    assert "--safe-mode" in R.build_command(arm="alone", model="m", isolation="safe-mode")
    assert "--strict-mcp-config" in R.build_command(arm="alone", model="m", isolation="bare")


def test_environment_is_forwarded_through_wslenv():
    env = R.build_env({"WSLENV": "FOO/p", "PATH": "/bin"}, arm="specialist")
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["MCP_TIMEOUT"] == "60000"
    names = env["WSLENV"].split(":")
    assert "FOO/p" in names and "CLAUDE_CODE_DISABLE_AUTO_MEMORY/u" in names and "MCP_TIMEOUT/u" in names
    alone = R.build_env({}, arm="alone", extra={"X": "1"})
    assert "MCP_TIMEOUT" not in alone and "X/u" in alone["WSLENV"].split(":")
    assert R.build_env({"WSLENV": "A/u"}, arm="alone", isolation="none")["WSLENV"] == "A/u"


def test_windows_paths_and_the_mcp_config():
    assert R.to_windows_path("/mnt/c/Users/me/x", "D") == "C:\\Users\\me\\x"
    assert R.to_windows_path("/mnt/d", "D") == "D:\\"
    assert R.to_windows_path("/home/ptmel/a b/c", "Ubuntu-24.04") == "\\\\wsl.localhost\\Ubuntu-24.04\\home\\ptmel\\a b\\c"
    cfg = R.mcp_config()
    srv = cfg["mcpServers"]["opencv-expert"]
    assert srv["command"] == "wsl.exe" and srv["args"] == ["-d", "Ubuntu-24.04", "--", "/home/ptmel/venvs/serve/bin/python",
                                                            "/home/ptmel/Hands-On-ML/opencv-specialist/serve/server.py"]
    assert list(cfg["mcpServers"]) == ["opencv-expert"]


def test_build_and_dry_run_write_inputs_only_and_print_the_commands(tmp_path, capsys):
    root = str(tmp_path)
    R.main(["build", "--run", "t", "--root", root, "--only", "e2e-0[1-2]|e2e-13"])
    run = tmp_path / "t"
    cfg = json.loads((run / "config.json").read_text())
    assert [t["id"] for t in cfg["tasks"]] == ["e2e-01-card-geometry", "e2e-02-card-blobs", "e2e-13-field-label-fit"]
    # inputs only, separately per arm, no truth anywhere
    for arm in R.ARMS:
        for t in cfg["tasks"]:
            files = sorted(p.name for p in (run / arm / t["id"]).iterdir())
            assert files == sorted(f"{n}.png" for n in t["images"]), (arm, t["id"], files)
    blob = (run / "config.json").read_text() + (run / "mcp.json").read_text()
    for forbidden in ('"tol"', '"truth"', '"value"', "answer.json"):
        assert forbidden not in blob
    assert (run / "alone" / "e2e-01-card-geometry" / "photo.png").read_bytes() == (run / "specialist" / "e2e-01-card-geometry" / "photo.png").read_bytes()
    # the two arms get the same prompt apart from the work dir
    for t in cfg["tasks"]:
        a = (run / "prompts" / "alone" / f"{t['id']}.txt").read_text()
        s = (run / "prompts" / "specialist" / f"{t['id']}.txt").read_text()
        assert a != s
        assert a.replace("/alone/", "/ARM/").replace("\\alone\\", "\\ARM\\") == s.replace("/specialist/", "/ARM/").replace("\\specialist\\", "\\ARM\\")
    capsys.readouterr()
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    R.main(["run", "--run", "t", "--root", root, "--arm", "specialist", "--model", "sonnet", "--limit", "2", "--dry-run", "--max-turns", "30"])
    out_s = capsys.readouterr().out
    R.main(["run", "--run", "t", "--root", root, "--arm", "alone", "--model", "sonnet", "--only", "card-geometry", "--dry-run", "--show-prompt"])
    out_a = capsys.readouterr().out
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before  # a dry run writes nothing
    assert out_s.count("[dry-run] specialist") == 2 and "--mcp-config" in out_s and R.MCP_TOOL in out_s
    assert "--max-turns 30" in out_s and "stdin: the prompt" in out_s
    assert "[dry-run] alone e2e-01-card-geometry" in out_a and "--mcp-config" not in out_a and R.MCP_TOOL not in out_a
    assert "answer.json" in out_a and "INPUT IMAGES" in out_a
    with pytest.raises(SystemExit):
        R.main(["build", "--run", "t", "--root", root])  # an existing run is not overwritten without --force
    R.main(["predict", "--run", "t", "--root", root, "--tokens-per-task", "400000", "--s-per-task", "240"])
    out = capsys.readouterr().out
    assert "3 tasks per arm" in out and "1,200,000 tokens" in out and "both arms" in out and "2,400,000" in out


# ---------------------------------------------------------------- the runner: transcripts and report math

def _msg(mid, content, usage):
    return {"type": "assistant", "message": {"id": mid, "content": content, "usage": usage}}


def _events(arm="specialist", verified=True):
    tools = ["Bash", "Read", "Write"] + ([R.MCP_TOOL] if arm == "specialist" else [])
    servers = [{"name": "opencv-expert", "status": "connected"}] if arm == "specialist" else []
    ev = [{"type": "system", "subtype": "init", "model": "claude-sonnet-x", "tools": tools, "mcp_servers": servers, "cwd": "C:\\w"}]
    ev.append(_msg("m1", [{"type": "text", "text": "thinking"}], {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0, "output_tokens": 5}))
    ev.append(_msg("m1", [{"type": "tool_use", "id": "t1", "name": R.MCP_TOOL if arm == "specialist" else "Bash", "input": {"task": "x"}}],
                   {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0, "output_tokens": 40}))
    ev.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                                        "content": [{"type": "text", "text": json.dumps({"verified": verified, "code": "..."})}]}]}})
    ev.append(_msg("m2", [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls /home/ptmel/Hands-On-ML/opencv-specialist/serve/e2e"}}],
                   {"input_tokens": 3, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 110, "output_tokens": 20}))
    ev.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": "x", "is_error": True}]}})
    ev.append({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "total_cost_usd": 0.05, "result": "DONE",
               "usage": {"input_tokens": 13, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 110, "output_tokens": 60}})
    return ev


def test_summarize_events_counts_tokens_tools_and_solve_calls():
    r = R.summarize_events(_events(), arm="specialist", task_id="e2e-01", run_name="t")
    assert r["usage"] == {"input": 13, "cache_creation": 100, "cache_read": 110, "output": 60, "fresh": 173, "total": 283}
    assert r["usage_source"] == "result" and r["usage_messages"]["output"] == 60 and r["usage_messages"]["input"] == 13
    assert r["tool_calls"] == {R.MCP_TOOL: 1, "Bash": 1} and r["n_tool_calls"] == 2 and r["n_tool_errors"] == 1
    assert r["n_solve"] == 1 and r["n_solve_verified"] == 1 and r["solve"]["unverified"] == 0
    assert r["cost_usd"] == 0.05 and r["num_turns"] == 3 and r["n_assistant_messages"] == 2
    assert r["isolation_ok"] and r["init"]["model"] == "claude-sonnet-x" and r["permission_denials"] == []
    denied = _events()
    denied[-1] = dict(denied[-1], permission_denials=[{"tool_name": "Write", "tool_use_id": "t9", "tool_input": {}}])
    assert R.summarize_events(denied, arm="specialist", task_id="e2e-01", run_name="t")["permission_denials"] == ["Write"]
    assert any("opencv-specialist" in p for p in r["peeked"])  # the Bash call looked into the repo
    r2 = R.summarize_events(_events(verified=False), arm="specialist", task_id="e2e-01", run_name="t")
    assert r2["n_solve_verified"] == 0 and r2["solve"]["unverified"] == 1
    # without a result event the usage comes from the messages (max per message id, summed)
    r3 = R.summarize_events([e for e in _events() if e["type"] != "result"], arm="specialist", task_id="e2e-01", run_name="t")
    assert r3["usage_source"] == "messages" and r3["usage"]["output"] == 60 and r3["usage"]["input"] == 13 and not r3["has_result_event"]


def test_isolation_problems_are_flagged():
    ev = _events(arm="alone")
    assert R.summarize_events(ev, arm="alone", task_id="x", run_name="t")["isolation_ok"]
    ev[0]["tools"] = ev[0]["tools"] + [R.MCP_TOOL, "WebFetch"]
    ev[0]["mcp_servers"] = [{"name": "other", "status": "connected"}]
    r = R.summarize_events(ev, arm="alone", task_id="x", run_name="t")
    assert not r["isolation_ok"] and len(r["isolation_notes"]) >= 2
    ev = _events(arm="specialist")
    ev[0]["mcp_servers"] = [{"name": "opencv-expert", "status": "failed"}]
    assert not R.summarize_events(ev, arm="specialist", task_id="x", run_name="t")["isolation_ok"]


def test_peeking_at_other_runs_is_flagged_but_the_own_work_dir_is_not():
    own = "/home/ptmel/opencv-expert/e2e-runs/t/alone/e2e-01/answer.json"
    other = "/home/ptmel/opencv-expert/e2e-runs/t/logs/alone/e2e-01/result.json"
    other_arm = "C:\\\\wsl.localhost\\\\x\\\\e2e-runs\\\\t\\\\specialist\\\\e2e-01\\\\photo.png"
    ev = [{"type": "assistant", "message": {"id": "a", "content": [
        {"type": "tool_use", "id": "1", "name": "Read", "input": {"file_path": own}},
        {"type": "tool_use", "id": "2", "name": "Read", "input": {"file_path": other}},
        {"type": "tool_use", "id": "3", "name": "Read", "input": {"file_path": other_arm}}], "usage": {}}}]
    r = R.summarize_events(ev, arm="alone", task_id="e2e-01", run_name="t")
    assert len(r["peeked"]) == 2 and not any("answer.json" in p for p in r["peeked"])


def test_wilson_and_sign_test_values():
    lo, hi_ = R.wilson(7, 10)
    assert math.isclose(lo, 0.3968, abs_tol=2e-3) and math.isclose(hi_, 0.8922, abs_tol=2e-3)
    assert R.wilson(0, 5)[0] == 0.0 and R.wilson(5, 5)[1] == 1.0 and math.isnan(R.wilson(0, 0)[0])
    assert R.sign_test_p(3, 0) == 0.25 and R.sign_test_p(0, 3) == 0.25
    assert math.isclose(R.sign_test_p(5, 1), 0.21875) and R.sign_test_p(2, 2) == 1.0 and R.sign_test_p(0, 0) == 1.0
    assert math.isclose(R.sign_test_p(10, 0), 2 / 1024)


def _synthetic():
    ids = [f"e2e-{i:02d}-x" for i in range(1, 7)]
    tasks = [{"id": i, "steps": ["a", "b"], "families": ["perspective_rectify"] if k % 2 else ["chessboard_corners"],
              "held_out": {"perspective_rectify": False} if k % 2 else {"chessboard_corners": True}} for k, i in enumerate(ids)]
    a_tok = [100, 200, 300, 400, 500, 600]
    s_tok = [50, 100, 300, 200, 250, 900]
    a_pass = [1, 1, 1, 0, 0, 1]
    s_pass = [1, 1, 0, 1, 0, 1]

    def result(tok, wall, solve=0, ver=0):
        return {"usage": {"input": tok // 10, "cache_creation": tok // 10, "cache_read": tok // 2, "output": tok - tok // 10 - tok // 10 - tok // 2,
                          "fresh": tok // 2, "total": tok}, "wall_s": wall, "n_tool_calls": 4, "tool_calls": {"Bash": 3}, "num_turns": 5,
                "n_tool_errors": 0, "n_solve": solve, "n_solve_verified": ver, "stopped_by": "completed", "isolation_ok": True, "peeked": [],
                "cost_usd": tok / 1000}

    results = {"alone": {i: result(t, w) for i, t, w in zip(ids, a_tok, [10, 20, 30, 40, 50, 60])},
               "specialist": {i: result(t, w, solve=2, ver=1) for i, t, w in zip(ids, s_tok, [30, 20, 60, 20, 100, 30])}}

    def grade(p):
        return {"passed": bool(p), "status": "ok", "n_pass": 2 if p else 1, "n_keys": 2}

    grades = {"alone": {i: grade(p) for i, p in zip(ids, a_pass)}, "specialist": {i: grade(p) for i, p in zip(ids, s_pass)}}
    return ids, tasks, results, grades


def test_summary_arithmetic_on_a_small_synthetic_set():
    ids, tasks, results, grades = _synthetic()
    s = R.summarize(results, grades, tasks)
    assert s["arms"]["alone"]["n_pass"] == 4 and s["arms"]["specialist"]["n_pass"] == 4
    assert math.isclose(s["arms"]["alone"]["rate"], 4 / 6)
    assert s["arms"]["alone"]["wilson95"] == list(R.wilson(4, 6))
    p = s["paired"]
    assert (p["n"], p["both"], p["alone_only"], p["specialist_only"], p["neither"]) == (6, 3, 1, 1, 1)
    assert p["tasks"]["alone_only"] == [ids[2]] and p["tasks"]["specialist_only"] == [ids[3]] and p["sign_test_p"] == 1.0
    tot_a, tot_s = s["tokens"]["alone"]["total"], s["tokens"]["specialist"]["total"]
    assert tot_a["mean"] == 350 and tot_a["median"] == 350 and tot_a["total"] == 2100
    assert tot_s["mean"] == 300 and tot_s["median"] == 225 and tot_s["total"] == 1800
    ratios = [0.5, 0.5, 1.0, 0.5, 0.5, 1.5]
    r = s["token_ratio_specialist_over_alone"]["total"]
    assert r["n"] == 6 and r["median"] == 0.5 and math.isclose(r["mean_of_ratios"], sum(ratios) / 6)
    assert np.allclose(r["iqr"], np.percentile(ratios, [25, 75])) and r["iqr"][1] == 0.875
    assert r["specialist_fewer"] == 4 and r["specialist_more"] == 1 and math.isclose(r["sign_test_p"], R.sign_test_p(4, 1))
    assert math.isclose(r["ratio_of_means"], 300 / 350)
    assert s["wall"]["alone"]["mean"] == 35 and s["wall"]["specialist"]["median"] == 30
    w = s["wall_ratio_specialist_over_alone"]
    assert w["n"] == 6 and w["median"] == 1.5
    assert s["arms"]["specialist"]["tools"]["solve_calls_total"] == 12 and s["arms"]["specialist"]["tools"]["solve_verified_total"] == 6
    assert s["arms"]["specialist"]["tools"]["tasks_using_solve"] == 6 and s["arms"]["alone"]["tools"]["tasks_using_solve"] == 0
    assert s["arms"]["alone"]["held_out_family_tasks"] == {"n": 3, "n_pass": 2}
    assert s["arms"]["specialist"]["held_out_family_tasks"] == {"n": 3, "n_pass": 1}
    c = s["criterion4_inputs"]
    assert c["specialist_success_ge_alone"] is True and c["mean_tokens_specialist_lt_alone"] is True and c["median_token_ratio_total"] == 0.5
    rows = {row["id"]: row for row in s["per_task"]}
    assert rows[ids[2]]["alone"]["passed"] and not rows[ids[2]]["specialist"]["passed"] and rows[ids[0]]["specialist"]["solve"] == 2
    text = R.render_report("synthetic", s)
    for needle in ("Wilson 95 % CI", "sign test", "IQR", "Per task", ids[0], "Criterion 4 inputs", "chessboard_corners*"):
        assert needle in text, needle
    json.dumps(s)  # summary.json must be serialisable


def test_partial_runs_are_reported_without_crashing():
    ids, tasks, results, grades = _synthetic()
    results["alone"] = {i: results["alone"][i] for i in ids[:2]}
    grades["alone"] = {i: grades["alone"][i] for i in ids[:2]}
    results["specialist"] = {}
    grades["specialist"] = {}
    s = R.summarize(results, grades, tasks)
    assert s["arms"]["specialist"]["n_run"] == 0 and s["arms"]["specialist"]["rate"] is None
    assert s["paired"]["n"] == 0 and s["token_ratio_specialist_over_alone"]["total"]["median"] is None
    assert "-" in R.render_report("partial", s)


def test_grade_and_report_commands_end_to_end_on_reference_answers(tmp_path, capsys):
    root = str(tmp_path)
    R.main(["build", "--run", "g", "--root", root, "--only", "e2e-03|e2e-13"])
    run = tmp_path / "g"
    cfg = json.loads((run / "config.json").read_text())
    for arm in R.ARMS:
        for t in cfg["tasks"]:
            n = int(t["id"].split("-")[1])
            ans = REF.reference_answer(T.build(n), run / arm / t["id"])
            if arm == "alone" or n == 3:  # the specialist arm's text task gets a wrong answer
                (run / arm / t["id"] / "answer.json").write_text(json.dumps(ans))
            else:
                (run / arm / t["id"] / "answer.json").write_text(json.dumps(dict(ans, scale=ans["scale"] * 0.5)))
            res = R.summarize_events(_events(arm=arm), arm=arm, task_id=t["id"], run_name="g")
            res.update({"task": t["id"], "arm": arm, "wall_s": 12.0 if arm == "alone" else 7.0, "stopped_by": "completed", "exit_code": 0})
            (run / "logs" / arm / t["id"]).mkdir(parents=True)
            (run / "logs" / arm / t["id"] / "result.json").write_text(json.dumps(res))
    capsys.readouterr()
    R.main(["grade", "--run", "g", "--root", root])
    out = capsys.readouterr().out
    assert out.count("PASS") == 3 and out.count("FAIL") == 1 and "layout" in out
    R.main(["report", "--run", "g", "--root", root])
    s = json.loads((run / "summary.json").read_text())
    assert s["arms"]["alone"]["n_pass"] == 2 and s["arms"]["specialist"]["n_pass"] == 1 and s["paired"]["alone_only"] == 1
    assert (run / "report.md").read_text().startswith("# E2E report: run `g`")
    assert len(list((run / "grades" / "alone").glob("*.json"))) == 2


# ---------------------------------------------------------------- the runner against a stand-in for claude.exe

STUB = '''#!{python}
import json, os, shutil, sys, time
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_MODE", "ok")
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({{"cwd": os.getcwd(), "prompt_chars": len(prompt), "argv": sys.argv[1:], "env_mcp_timeout": os.environ.get("MCP_TIMEOUT")}}) + chr(10))


def emit(o):
    print(json.dumps(o), flush=True)


def asst(i, content):
    return {{"type": "assistant", "message": {{"id": "m%d" % i, "content": content, "usage": {{"input_tokens": 5, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 20, "output_tokens": 7}}}}}}


emit({{"type": "system", "subtype": "init", "model": "fake", "tools": ["Bash", "Read", "Write"], "mcp_servers": [], "cwd": os.getcwd()}})
if mode == "ok":
    emit(asst(1, [{{"type": "tool_use", "id": "t1", "name": "Bash", "input": {{"command": "ls"}}}}]))
    emit({{"type": "user", "message": {{"content": [{{"type": "tool_result", "tool_use_id": "t1", "content": "x"}}]}}}})
    emit(asst(2, [{{"type": "text", "text": "DONE"}}]))
    if os.environ.get("FAKE_ANSWER"):
        shutil.copy(os.environ["FAKE_ANSWER"], "answer.json")
    emit({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "total_cost_usd": 0.01, "result": "DONE",
          "usage": {{"input_tokens": 10, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 40, "output_tokens": 14}}}})
elif mode == "sleep":
    time.sleep(120)
elif mode == "many":
    for i in range(1, 13):
        emit(asst(i, [{{"type": "text", "text": "hm"}}]))
    time.sleep(120)
elif mode == "crash":
    sys.exit(3)
elif mode == "claude_max_turns":
    emit(asst(1, [{{"type": "text", "text": "hm"}}]))
    emit({{"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 3,
          "usage": {{"input_tokens": 1, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3, "output_tokens": 4}}}})
    sys.exit(1)
'''


@pytest.fixture()
def stub_run(tmp_path, monkeypatch):
    import stat
    import sys
    stub = tmp_path / "fake_claude.py"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "stub.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    root = str(tmp_path / "runs")
    R.main(["build", "--run", "f", "--root", root, "--only", "e2e-03"])
    return stub, log, root, tmp_path / "runs" / "f"


def _calls(log):
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_a_normal_session_is_recorded_graded_and_not_repeated(stub_run, monkeypatch, tmp_path, capsys):
    stub, log, root, run = stub_run
    tid = "e2e-03-strokes-longest"
    ans = tmp_path / "ans.json"
    ans.write_text(json.dumps(REF.reference_answer(T.build(3))))
    monkeypatch.setenv("FAKE_ANSWER", str(ans))
    (run / "alone" / tid / "answer.json").write_text("stale")  # a leftover must not survive into the next attempt
    args = ["run", "--run", "f", "--root", root, "--arm", "alone", "--model", "m", "--claude", str(stub), "--max-turns", "20"]
    R.main(args)
    res = json.loads((run / "logs" / "alone" / tid / "result.json").read_text())
    assert res["stopped_by"] == "completed" and res["exit_code"] == 0 and res["wall_s"] > 0
    assert res["usage"] == {"input": 10, "cache_creation": 20, "cache_read": 40, "output": 14, "fresh": 44, "total": 84}
    assert res["n_tool_calls"] == 1 and res["num_turns"] == 2 and res["cost_usd"] == 0.01 and res["isolation_ok"]
    assert res["cmd"][0] == str(stub) and "--mcp-config" not in res["cmd"] and "--max-turns" in res["cmd"]
    lines = (run / "logs" / "alone" / tid / "transcript.jsonl").read_text().splitlines()
    assert [json.loads(l)["type"] for l in lines] == ["system", "assistant", "user", "assistant", "result"]
    (call,) = _calls(log)
    assert call["cwd"].endswith(f"/alone/{tid}") and call["prompt_chars"] == len((run / "prompts" / "alone" / f"{tid}.txt").read_text())
    assert call["env_mcp_timeout"] is None
    R.main(["grade", "--run", "f", "--root", root])
    assert json.loads((run / "grades" / "alone" / f"{tid}.json").read_text())["passed"] is True
    R.main(args)  # resume: the result exists, the stand-in is not started again
    assert len(_calls(log)) == 1 and "skip alone" in capsys.readouterr().out
    R.main(args + ["--rerun"])
    assert len(_calls(log)) == 2


def test_the_specialist_arm_passes_the_mcp_config_and_timeout(stub_run, monkeypatch):
    stub, log, root, run = stub_run
    monkeypatch.setenv("FAKE_MODE", "ok")
    R.main(["run", "--run", "f", "--root", root, "--arm", "specialist", "--model", "m", "--claude", str(stub)])
    (call,) = _calls(log)
    assert "--mcp-config" in call["argv"] and call["env_mcp_timeout"] == "60000"
    res = json.loads((run / "logs" / "specialist" / "e2e-03-strokes-longest" / "result.json").read_text())
    assert not res["isolation_ok"] and "not among the session's tools" in res["isolation_notes"][0]  # the stand-in has no MCP tool: flagged


def test_a_session_without_an_answer_is_graded_missing(stub_run):
    stub, log, root, run = stub_run
    R.main(["run", "--run", "f", "--root", root, "--arm", "alone", "--model", "m", "--claude", str(stub)])
    R.main(["grade", "--run", "f", "--root", root])
    g = json.loads((run / "grades" / "alone" / "e2e-03-strokes-longest.json").read_text())
    assert g["status"] == "missing" and not g["passed"]


def test_timeout_max_turns_and_crash_are_recorded(stub_run, monkeypatch):
    stub, log, root, run = stub_run
    tid = "e2e-03-strokes-longest"
    base = ["run", "--run", "f", "--root", root, "--arm", "alone", "--model", "m", "--claude", str(stub)]
    monkeypatch.setenv("FAKE_MODE", "sleep")
    R.main(base + ["--timeout", "2", "--rerun"])
    res = json.loads((run / "logs" / "alone" / tid / "result.json").read_text())
    assert res["stopped_by"] == "timeout" and 2 <= res["wall_s"] < 30 and not res["has_result_event"]
    monkeypatch.setenv("FAKE_MODE", "many")
    R.main(base + ["--timeout", "60", "--max-turns", "2", "--rerun"])
    res = json.loads((run / "logs" / "alone" / tid / "result.json").read_text())
    assert res["stopped_by"] == "max_turns" and res["n_assistant_messages"] > 2 and res["wall_s"] < 30
    assert res["usage_source"] == "messages" and res["usage"]["output"] == 7 * res["n_assistant_messages"]
    monkeypatch.setenv("FAKE_MODE", "crash")
    R.main(base + ["--rerun"])
    res = json.loads((run / "logs" / "alone" / tid / "result.json").read_text())
    assert res["stopped_by"] == "error" and res["exit_code"] == 3
    monkeypatch.setenv("FAKE_MODE", "claude_max_turns")  # claude's own limit: exit code 1, subtype error_max_turns
    R.main(base + ["--rerun", "--max-turns", "3"])
    res = json.loads((run / "logs" / "alone" / tid / "result.json").read_text())
    assert res["stopped_by"] == "max_turns" and res["exit_code"] == 1 and res["usage"]["total"] == 10
