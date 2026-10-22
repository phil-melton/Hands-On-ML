"""The drills' own regression tests (foundry/drills/).

Drills are small verified TRAINING tasks, one measured OpenCV 5.0 behaviour (or one timeless gotcha) each. The tests pin:

* catalogue integrity: ids, groups, tags that agree with the recorded 4.14-vs-5.0 diff, prompts that render, solve() signatures
  that match the specs, three distinct phrasings per drill, the held-out families untouched;
* the behaviour matrix: every reference passes ``cvbench.gotchas.core.grade``, every wrong variant fails the way it is declared
  to (an exception of a given type, or a silent wrong answer only the verifier catches), a constant answer fails;
* no leakage: no prompt mentions the benchmark, a version, a tolerance, the gotcha or the answer, and none names an API the
  correct answer must avoid;
* disjointness: no drill prompt or phrasing overlaps a gotcha prompt, a dev or test prompt, or a foundry phrasing beyond a
  calibrated word 5-gram Jaccard threshold, and no drill id equals a gotcha id;
* calibration: the cheap tolerance measurements written into the modules' comments are re-run on smaller samples;
* coverage: every behaviour that changed between 4.14 and 5.0 in the recorded probes either has a drill or a stated reason.

All model-style code runs through ``sandbox.run_solution`` (via ``grade``); the matrix runs on a pool of 4 workers.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import itertools
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvbench import api_check
from cvbench import splits as sp
from cvbench.contract import check_outputs
from cvbench.gotchas import CHECKS as GOTCHAS
from cvbench.gotchas import grade
from cvbench.gotchas.core import Case
from foundry.drills import DRILL_PHRASINGS, DRILLS
from foundry.drills import calling, drift, dtypes, numpy2, removed, shapes
from sandbox.lint import lint
from sandbox.run import run_solution

ROOT = Path(__file__).resolve().parents[1]
REC4 = json.loads((ROOT / "probes/results/probe_4.14.0.json").read_text())["probes"]
REC5 = json.loads((ROOT / "probes/results/probe_5.0.0.json").read_text())["probes"]

PER_GROUP = {"shapes": 8, "dtypes": 10, "calling": 5, "removed": 8, "numpy2": 4, "drift": 6}
MODULE_OF = {"shapes": shapes, "dtypes": dtypes, "calling": calling, "removed": removed, "numpy2": numpy2, "drift": drift}
ROLES = ("gotcha", "control")
VARIANTS = [(did, name) for did, d in DRILLS.items() for name in d.wrong]
RUNTIME_BUDGET_S = 90.0

# Word 5-gram Jaccard thresholds, calibrated on the 184 reference prompts below (see test_the_overlap_thresholds_are_calibrated):
# the largest overlap a drill has with any gotcha, dev, test or foundry text is 0.13 on the task line and 0.28 on the whole
# prompt (the rest of the prompt is the shared template and similar input notes); two phrasings of one family share 0.31 (90th
# percentile) to 0.51 of their task line and 0.63 to 0.87 of the whole prompt. A threshold halfway up the gap catches a copy
# or a light paraphrase and passes honest reuse of common words.
TASK_JACCARD_MAX = 0.25
PROMPT_JACCARD_MAX = 0.40
PHRASING_DUPLICATE_MAX = 0.60  # two phrasings of ONE drill: their task lines must be different enough (5-gram Jaccard)


# ---------------------------------------------------------------- helpers

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _grams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _prompt(drill_id: str, index: int) -> str:
    return dataclasses.replace(DRILLS[drill_id].spec, task=DRILL_PHRASINGS[drill_id][index]).to_prompt()


def _authored_text(drill_id: str, index: int) -> str:
    """Everything a drill author wrote into a prompt: the task line and the input/output notes (not the template)."""
    s = DRILLS[drill_id].spec
    parts = [DRILL_PHRASINGS[drill_id][index]]
    for io in (*s.inputs.values(), *s.outputs.values()):
        parts += [io.dtype, io.shape, io.note]
    return "\n".join(parts)


def _solve_params(code: str) -> list[str]:
    fn = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    assert len(fn) == 1
    return [a.arg for a in fn[0].args.args]


def _other_prompts() -> dict[str, tuple[str, str]]:
    """name -> (task line, whole prompt) for every text the drills must stay away from: the 40 gotcha prompts, the dev and test
    prompts (re-rendered from data/splits/*.jsonl with cvbench.splits' recipe, hash-checked against the manifests) and the foundry
    phrasings reserved for Stage 4."""
    out: dict[str, tuple[str, str]] = {f"gotcha:{c.id}": (c.spec.task, c.prompt()) for c in GOTCHAS.values()}
    for split in ("dev", "test"):
        rows = [json.loads(line) for line in (sp.SPLITS_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            fam = importlib.import_module(f"cvbench.families.{row['family']}")
            s = dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[row["phrasing_index"]])
            assert hashlib.sha1(s.to_prompt().encode("utf-8")).hexdigest() == row["prompt_sha1"], row["id"]
            out[f"{split}:{row['family']}:{row['phrasing_index']}"] = (s.task, s.to_prompt())
    for name in sp.FAMILIES:
        fam = importlib.import_module(f"cvbench.families.{name}")
        for i in sp.FOUNDRY_PHRASINGS:
            s = dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[i])
            out[f"foundry:{name}:{i}"] = (s.task, s.to_prompt())
    return out


def overlaps() -> dict:
    """Largest task-line and whole-prompt overlap of every drill phrasing with the other prompts (and who it is with)."""
    others = _other_prompts()
    task_g = {k: _grams(v[0]) for k, v in others.items()}
    full_g = {k: _grams(v[1]) for k, v in others.items()}
    rows = []
    for did in DRILLS:
        for i, task in enumerate(DRILL_PHRASINGS[did]):
            gt, gf = _grams(task), _grams(_prompt(did, i))
            bt = max(((_jaccard(gt, g), k) for k, g in task_g.items()), key=lambda t: t[0])
            bf = max(((_jaccard(gf, g), k) for k, g in full_g.items()), key=lambda t: t[0])
            rows.append({"drill": did, "phrasing": i, "task": bt, "prompt": bf})
    return {"rows": rows, "n_others": len(others), "others": others}


# ---------------------------------------------------------------- catalogue integrity

def test_the_build_under_test_is_opencv_5_0_with_numpy_2():
    assert cv2.__version__.startswith("5.0"), cv2.__version__
    assert np.__version__.startswith("2."), np.__version__


def test_forty_one_drills_in_the_planned_groups():
    assert len(DRILLS) == 41
    got = {g: sum(d.group == g for d in DRILLS.values()) for g in PER_GROUP}
    assert got == PER_GROUP
    assert {d.group for d in DRILLS.values()} == set(PER_GROUP)
    for g, module in MODULE_OF.items():
        assert [c.id for c in module.CHECKS] == [d.id for d in DRILLS.values() if d.group == g]


def test_ids_are_unique_prefixed_and_never_a_gotcha_id():
    ids = [c.id for module in MODULE_OF.values() for c in module.CHECKS]
    assert len(ids) == len(set(ids)) == len(DRILLS)
    assert not set(ids) & set(GOTCHAS), "a drill id equals a gotcha id"
    assert all(i.startswith("drill_") for i in ids)
    assert list(DRILLS) == list(DRILL_PHRASINGS)


def test_every_drill_carries_its_tags():
    n_controls = 0
    for did, d in DRILLS.items():
        assert d.id == did and d.group in PER_GROUP and d.role in ROLES, did
        assert d.spec.family == "drill" and d.spec.outputs and d.spec.inputs, did
        assert d.probes and all(isinstance(p, str) and p for p in d.probes), did
        assert isinstance(d.depends_on_5_0, bool) and d.subset in ("5.0", "timeless"), did
        assert d.gotcha.strip() and d.reference.strip() and len(d.wrong) >= 1, did
        if d.role == "control":
            n_controls += 1
            assert not d.depends_on_5_0, did
        for name, w in d.wrong.items():
            assert w.why.strip() and w.code.strip() and w.fails.strip(), (did, name)
            assert w.code != d.reference, (did, name)
            assert w.fails == "verifier" or w.error_contains, f"{did}/{name}: pin the error text of an exception variant"
        codes = [w.code for w in d.wrong.values()] + [d.reference]
        assert len(set(codes)) == len(codes), f"{did}: duplicate solution text"
    assert n_controls == 2  # drill_coin_circles (HoughCircles) and drill_palette_kmeans (kmeans labels)


def test_5_0_tags_agree_with_the_recorded_4_14_vs_5_0_diff():
    """depends_on_5_0 must equal "a cited recorded probe differs between the two versions"; a drill citing only `new:` probes
    (a behaviour measured while building, not in the recorded diff) is timeless unless it says why not."""
    for did, d in DRILLS.items():
        recorded = [p for p in d.probes if not p.startswith("new:")]
        for p in recorded:
            assert p in REC5 and p in REC4, f"{did}: {p} is not a recorded probe"
        if not recorded:
            assert not d.depends_on_5_0, f"{did}: no recorded probe, so it cannot be tagged 5.0"
            continue
        changed = any(REC4[p] != REC5[p] for p in recorded)
        assert d.depends_on_5_0 == changed, f"{did}: tagged {d.subset} but the probes say changed={changed}"
    assert sum(d.depends_on_5_0 for d in DRILLS.values()) == 24


def test_every_solution_has_the_signature_the_prompt_announces():
    for did, d in DRILLS.items():
        want = list(d.spec.inputs)
        assert f"def solve({', '.join(want)}) -> dict" in d.prompt(), did
        for label, code in [("reference", d.reference)] + [(n, w.code) for n, w in d.wrong.items()]:
            assert _solve_params(code) == want, (did, label)


def test_cases_are_deterministic_vary_and_match_the_declared_inputs():
    exact = {"uint8", "int32", "int64", "uint32", "float16", "float32", "float64", "bool"}
    for did, d in DRILLS.items():
        a, b = d.make_cases(), d.make_cases()
        assert len(a) == len(b) >= 2, did
        for x, y in zip(a, b):
            assert list(x.inputs) == list(d.spec.inputs), did
            for k in x.inputs:
                u, v = x.inputs[k], y.inputs[k]
                if isinstance(u, np.ndarray):
                    assert u.dtype == v.dtype and np.array_equal(u, v), (did, k)
                    s = d.spec.inputs[k]
                    if s.dtype in exact:
                        assert str(u.dtype) == s.dtype, (did, k, str(u.dtype), s.dtype)
                    elif s.dtype in ("int", "float"):
                        assert u.dtype.kind in {"int": "iu", "float": "f"}[s.dtype], (did, k)
                    if s.ndim is not None:
                        assert u.ndim == s.ndim, (did, k, u.shape)
                else:
                    assert u == v, (did, k)
        # no two cases are the same input: a hard-coded answer for one case cannot serve another
        for (i, x), (j, y) in itertools.combinations(enumerate(a), 2):
            same = all(np.array_equal(x.inputs[k], y.inputs[k]) if isinstance(x.inputs[k], np.ndarray) else x.inputs[k] == y.inputs[k]
                       for k in x.inputs)
            assert not same, f"{did}: cases {i} and {j} are identical"


def test_drills_do_not_borrow_the_gotcha_suites_scenes_and_keep_to_their_own_seed_block():
    """Gotchas are test data: the only thing a drill may import from the gotcha suite is its types (cvbench.gotchas.core), never a
    scene generator or a verifier helper; and the drills' seeds sit in a block no split uses."""
    from foundry.drills import common
    pkg = Path(common.__file__).parent
    for path in sorted(pkg.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cvbench"):
                assert node.module in ("cvbench.gotchas.core",) and path.name == "common.py", (path.name, node.module)
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("cvbench.gotchas") for a in node.names), path.name
    lo, hi = common.SEED_BASE, common.SEED_BASE + 9_999
    for a, b in (sp.DEV_RANGE, sp.TEST_RANGE, sp.FOUNDRY_RANGE):
        assert hi < a or lo >= b
    used = [m.S for m in MODULE_OF.values()]
    assert len(set(used)) == len(used) and all(lo <= u <= hi - 999 for u in used)


def test_the_held_out_families_have_no_drill():
    """chessboard_corners and color_segmentation are dev/test only (cvbench/splits.py): no prompt, reference or variant of a
    drill may be a chessboard or an HSV-range task, or call the functions those families are about."""
    banned = ("findChessboardCorners", "chessboard", "inRange", "COLOR_BGR2HSV", "COLOR_RGB2HSV")
    for did, d in DRILLS.items():
        texts = [d.reference, *[w.code for w in d.wrong.values()], *[_authored_text(did, i) for i in range(3)]]
        for t in texts:
            for b in banned:
                assert b.lower() not in t.lower(), f"{did} mentions {b}"
    assert {n for n, r in sp.FAMILIES.items() if r.held_out} == {"chessboard_corners", "color_segmentation"}


def test_references_pass_the_production_lint_except_the_documented_videocapture_one():
    for did, d in DRILLS.items():
        findings = lint(d.reference)
        if d.lint:
            assert not findings, (did, [str(f) for f in findings])
        else:
            assert did == "drill_capture_properties" and [f.rule for f in findings] == ["io"], did
    assert [did for did, d in DRILLS.items() if not d.lint] == ["drill_capture_properties"]


# ---------------------------------------------------------------- phrasings

def test_three_distinct_phrasings_per_drill_and_the_first_is_the_spec_task():
    for did, d in DRILLS.items():
        ph = DRILL_PHRASINGS[did]
        assert len(ph) == 3 and ph[0] == d.spec.task, did
        assert all(isinstance(p, str) and p.strip() == p and len(p) > 40 for p in ph), did
        norm = [" ".join(_words(p)) for p in ph]
        assert len(set(norm)) == 3, f"{did}: duplicate phrasing"
        for (i, a), (j, b) in itertools.combinations(enumerate(ph), 2):
            wa, wb = set(_words(a)), set(_words(b))
            assert len(wa & wb) / len(wa | wb) < 0.8, f"{did}: phrasings {i} and {j} share most of their words"
            assert _jaccard(_grams(a), _grams(b)) < PHRASING_DUPLICATE_MAX, f"{did}: phrasings {i} and {j} repeat each other"


def test_every_phrasing_renders_the_same_contract():
    for did, d in DRILLS.items():
        base = d.prompt().split("\n")[1:]
        for i in range(3):
            lines = _prompt(did, i).split("\n")
            assert lines[0] == f"Task: {DRILL_PHRASINGS[did][i]}" and lines[1:] == base, (did, i)
        assert d.prompt().splitlines()[1] == "OpenCV: 5.0 (Python cv2, headless)"


# ---------------------------------------------------------------- no leakage

BANNED_WORDS = (r"cvbench", r"tolerances?", r"gates?", r"seeds?", r"verifiers?", r"graders?", r"benchmarks?", r"gotchas?",
                r"pitfalls?", r"traps?", r"regress\w*", r"deprecated", r"removed", r"no longer", r"used to", r"silently",
                r"middle axis", r"lost axis", r"extra axis", r"squeeze", r"reshape", r"flatten", r"ravel", r"wraps? around",
                r"overflow\w*", r"hidden test", r"unit tests?", r"numpy ?2", r"opencv ?[45]", r"[45]\.(?:x|\d+)(?:\.\d+)?x?")


def test_prompts_read_as_requests_not_quizzes():
    pattern = re.compile(r"(?<![\w.])(?:" + "|".join(BANNED_WORDS) + r")(?![\w])", re.I)
    for did in DRILLS:
        for i in range(3):
            text = _authored_text(did, i)
            m = pattern.search(text)
            assert not m, f"{did} phrasing {i} says {m.group(0)!r}"
            assert _prompt(did, i).lower().startswith("task: ")


def _forbidden_names(matrix) -> dict[str, set[str]]:
    """Names a prompt must not contain: every removed or absent cv2 name in the api_check catalogue, the numpy names that no longer
    exist, and, per drill, the attribute named in the AttributeError of each of its wrong variants."""
    global_names: set[str] = {"np.int0", "np.Inf", "np.float_", "np.product", "np.round_", ".ptp(", "BOWKMeansTrainer",
                              "HOGDescriptor", "estimateRigidTransform", "CV_AA", "bgsegm"}
    for key in api_check.CATALOGUE:
        tail = key.split(".")[-1]
        for name in {tail, tail.removesuffix("_create")}:
            if len(name) >= 4:
                global_names.add(name)
    global_names |= {"cv2.ml", "cv2.cv"}
    per_drill: dict[str, set[str]] = {did: set() for did in DRILLS}
    for (did, name), g in matrix.items():
        if isinstance(did, str) and name != "reference" and g.error_type == "AttributeError":
            m = re.search(r"attribute '([^']+)'", g.error or "") or re.search(r"`(np\.[A-Za-z_0-9]+)`", g.error or "")
            assert m, (did, name, g.error)
            per_drill[did].add(m.group(1))
    return {did: global_names | per_drill[did] for did in DRILLS}


def test_no_prompt_names_an_api_the_correct_answer_must_avoid(matrix):
    forbidden = _forbidden_names({k: v for k, v in matrix.items() if k != "seconds"})
    for did in DRILLS:
        for i in range(3):
            text = _authored_text(did, i)
            for name in forbidden[did]:
                hit = re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w])", text) if name[0].isalnum() else (name in text)
                assert not hit, f"{did} phrasing {i} names {name!r}, which the correct answer must avoid"
    # the check sees what it should: the tracker drill's wrong variants name these four, none of them in its prompts
    assert {"TrackerKCF_create", "TrackerCSRT_create", "TrackerGOTURN_create", "legacy"} <= forbidden["drill_track_box"]
    assert {"int0"} <= forbidden["drill_rotated_box_draw"] and {"ptp"} <= forbidden["drill_channel_stretch"]
    assert {"np.Inf"} <= forbidden["drill_nearest_foreground"]


def test_the_api_a_task_is_about_may_be_named():
    """Naming is allowed where a person asking for the job would: these phrasings say so on purpose."""
    named = {"drill_nonzero_extremes": "cv2.findNonZero", "drill_hough_acc_skew": "cv2.HoughLinesWithAccumulator",
             "drill_bool_centroid": "cv2.moments", "drill_label_map_resize": "INTER_NEAREST_EXACT",
             "drill_fontface_label": "cv2.FontFace", "drill_bubble_count_mser": "cv2.MSER_create",
             "drill_palette_kmeans": "cv2.kmeans", "drill_aruco_sizes": "DICT_5X5_100"}
    for did, name in named.items():
        assert any(name in p for p in DRILL_PHRASINGS[did]), (did, name)


# ---------------------------------------------------------------- disjointness from the gotcha suite, dev, test and foundry

def test_the_overlap_metric_catches_a_copy_and_a_light_paraphrase():
    g = GOTCHAS["draw_float_center"]
    a, b = _grams(g.prompt()), _grams(g.prompt())
    assert _jaccard(a, b) == 1.0
    light = g.prompt().replace("Draw a filled red circle", "Paint a filled red circle").replace("midpoint", "middle")
    assert _jaccard(a, _grams(light)) > PROMPT_JACCARD_MAX
    assert _jaccard(_grams(g.spec.task), _grams(g.spec.task.replace("Draw", "Paint"))) > TASK_JACCARD_MAX


def test_overlap_with_gotcha_dev_test_and_foundry_prompts_is_below_the_calibrated_threshold():
    r = overlaps()
    assert r["n_others"] == 40 + 36 + 72 + 36, r["n_others"]  # gotchas, dev (9 x 4), test (9 x 8), foundry (9 x 4)
    worst_task = max(r["rows"], key=lambda x: x["task"][0])
    worst_prompt = max(r["rows"], key=lambda x: x["prompt"][0])
    print(f"\nmax task-line 5-gram Jaccard {worst_task['task'][0]:.3f} ({worst_task['drill']}[{worst_task['phrasing']}] vs "
          f"{worst_task['task'][1]}); max whole-prompt {worst_prompt['prompt'][0]:.3f} ({worst_prompt['drill']}"
          f"[{worst_prompt['phrasing']}] vs {worst_prompt['prompt'][1]})")
    for x in r["rows"]:
        assert x["task"][0] < TASK_JACCARD_MAX, x
        assert x["prompt"][0] < PROMPT_JACCARD_MAX, x
    assert len(r["rows"]) == 3 * len(DRILLS)


def test_overlap_thresholds_sit_in_the_gap_between_honest_reuse_and_near_duplicates():
    """The calibration the thresholds stand on, re-measured: the drills' worst overlap is well under the thresholds, the
    prompts of one family (near-duplicates by construction) are well over."""
    r = overlaps()
    others = r["others"]
    worst_task = max(x["task"][0] for x in r["rows"])
    worst_prompt = max(x["prompt"][0] for x in r["rows"])
    fam = {}
    for k, (t, p) in others.items():
        if k.split(":")[0] in ("dev", "test", "foundry"):
            fam.setdefault(k.split(":")[1], []).append((_grams(t), _grams(p)))
    pairs_task, pairs_prompt = [], []
    for items in fam.values():
        for (t1, p1), (t2, p2) in itertools.combinations(items, 2):
            pairs_task.append(_jaccard(t1, t2))
            pairs_prompt.append(_jaccard(p1, p2))
    assert worst_task < 0.20 and worst_prompt < 0.32, (worst_task, worst_prompt)
    assert np.percentile(pairs_prompt, 10) > 0.45 and np.median(pairs_prompt) > PROMPT_JACCARD_MAX
    assert np.percentile(pairs_task, 90) > 0.20 and max(pairs_task) > TASK_JACCARD_MAX


def test_no_two_drills_share_a_task_line():
    grams = {(did, i): _grams(t) for did, ph in DRILL_PHRASINGS.items() for i, t in enumerate(ph)}
    worst = max(_jaccard(grams[a], grams[b]) for a, b in itertools.combinations(grams, 2) if a[0] != b[0])
    assert worst < TASK_JACCARD_MAX, worst


# ---------------------------------------------------------------- the behaviour matrix

@pytest.fixture(scope="module")
def matrix():
    jobs = [(did, "reference", d.reference) for did, d in DRILLS.items()]
    jobs += [(did, name, w.code) for did, d in DRILLS.items() for name, w in d.wrong.items()]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        grades = list(pool.map(lambda j: grade(DRILLS[j[0]], j[2]), jobs))
    out = {(did, name): g for (did, name, _), g in zip(jobs, grades)}
    out["seconds"] = time.monotonic() - t0
    return out


@pytest.mark.parametrize("did", list(DRILLS))
def test_reference_passes(matrix, did):
    g = matrix[(did, "reference")]
    assert g.passed and g.stage == "ok", str(g)


@pytest.mark.parametrize("did,name", VARIANTS)
def test_wrong_variant_fails_the_way_it_is_declared_to_(matrix, did, name):
    w, g = DRILLS[did].wrong[name], matrix[(did, name)]
    assert not g.passed, f"{did}/{name} was accepted: {g}"
    if w.fails == "verifier":
        # runs fine (an exec()-only harness accepts it) and the contract or the verifier rejects it
        assert g.ran and g.stage in ("contract", "verify"), str(g)
    else:
        assert g.stage == "run" and not g.ran and g.error_type == w.fails, str(g)
        assert w.error_contains in (g.error or ""), (w.error_contains, g.error)


def test_the_matrix_ran_inside_the_time_budget(matrix):
    assert matrix["seconds"] < RUNTIME_BUDGET_S, matrix["seconds"]


def test_exec_only_filtering_would_keep_the_silent_wrong_answers(matrix):
    wrong = [matrix[v] for v in VARIANTS]
    silent = [g for g in wrong if g.ran]
    assert len(wrong) == len(VARIANTS) == 122
    assert len(silent) == sum(w.fails == "verifier" for d in DRILLS.values() for w in d.wrong.values()) == 68
    assert not any(g.passed for g in wrong)
    # every exception variant is a cv2.error, an AttributeError, or another named Python error: none ends in a sandbox violation
    assert not any(g.stage == "run" and g.error_type in (None, "SandboxViolation", "ContractError") for g in wrong)


def test_a_constant_answer_fails(matrix):
    """A hard-coded answer (the reference's outputs for the first case, returned for every case) is rejected."""
    for did, d in DRILLS.items():
        cases = d.make_cases()
        r = run_solution(d.reference, [c.inputs for c in cases], lint=d.lint)
        assert r.ok, (did, r.summary())
        const = r.outputs[0]
        rejected = False
        for case in cases[1:]:
            if check_outputs(d.spec, const):
                rejected = True
                break
            try:
                if not d.verify(case, const):
                    rejected = True
                    break
            except (ValueError, TypeError, IndexError, KeyError, AttributeError):
                rejected = True
                break
        assert rejected, f"{did}: the first case's answer passes every other case"


def test_an_empty_answer_never_passes():
    for did in ("drill_hist_unused_levels", "drill_u32_extrema", "drill_hierarchy_holes", "drill_caption_wrap", "drill_aa_overlay"):
        g = grade(DRILLS[did], "def solve(**kwargs):\n    return {}\n")
        assert not g.passed and g.stage == "contract" and "missing output" in g.detail, str(g)


# ---------------------------------------------------------------- calibration: the measurements the tolerances stand on (smaller samples)

def test_hough_longest_holds_for_reasonable_parameters():
    bad = n = 0
    for seed in range(20):
        img, longest = shapes.stroke_scene(seed)
        case = Case({"edges": img}, longest)
        for thr, mll, gap in itertools.product((30, 50), (20, 40), (10, 15)):
            lines = cv2.HoughLinesP(img, 1, np.pi / 180, thr, minLineLength=mll, maxLineGap=gap).reshape(-1, 4)
            seg = lines[int(np.argmax(np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])))]
            n += 1
            bad += not DRILLS["drill_hough_longest"].verify(case, {"segment": seg})
    assert bad == 0, f"{bad}/{n} reasonable parameter choices rejected"
    img, longest = shapes.stroke_scene(0)
    shifted = np.array([longest[0][0] + 9, longest[0][1], longest[1][0] + 9, longest[1][1]])
    assert not DRILLS["drill_hough_longest"].verify(Case({"edges": img}, longest), {"segment": shifted})


def test_skew_estimate_is_within_tolerance_for_both_vote_thresholds():
    worst = 0.0
    for a in np.arange(-12, 12.01, 0.75):
        img = shapes.ruled_sheet(float(a))
        for votes in (100, 120):
            lines = cv2.HoughLinesWithAccumulator(img, 1, np.pi / 180, votes).reshape(-1, 3)
            worst = max(worst, abs(90 - np.average(np.degrees(lines[:, 1]), weights=lines[:, 2]) - a))
    assert worst < shapes.SKEW_TOL_DEG * 0.8, worst


def test_notched_plate_defects_match_the_manufactured_chips():
    worst_pt = worst_d = 0.0
    for seed in range(100, 140):
        mask, chips = shapes.notched_plate(seed)
        for mode in (cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE):
            cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, mode)[0], key=cv2.contourArea)
            d = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False)).reshape(-1, 4)
            depth = d[:, 3] / 256.0
            keep = depth > 3
            assert keep.sum() == len(chips)
            i = int(np.argmax(np.where(keep, depth, -1)))
            worst_pt = max(worst_pt, np.abs(cnt.reshape(-1, 2)[d[i, 2]] - chips[0][0]).max())
            worst_d = max(worst_d, abs(depth[i] - chips[0][1]))
    assert worst_pt <= 2.0 < shapes.NOTCH_POINT_TOL and worst_d <= 2.2 < shapes.NOTCH_DEPTH_TOL, (worst_pt, worst_d)


def test_enclosing_triangle_stays_inside_the_tolerance():
    worst_v = 0.0
    lo, hi = 9.0, -9.0
    for seed in range(60):
        mask, v = shapes.sign_scene(seed)
        cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        area, tri = cv2.minEnclosingTriangle(cnt)
        tri = tri.reshape(3, 2)
        d = np.linalg.norm(v[:, None] - tri[None], axis=2)
        worst_v = max(worst_v, min(max(d[i, p[i]] for i in range(3)) for p in itertools.permutations(range(3))))
        ratio = area / shapes._tri_area(v)
        lo, hi = min(lo, ratio), max(hi, ratio)
    assert worst_v < shapes.TRI_VERTEX_TOL / 1.5 and 1.0 <= lo and hi < 1 + shapes.TRI_AREA_REL / 1.5, (worst_v, lo, hi)


def test_aruco_side_length_is_the_drawn_size_minus_one_pixel():
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(shapes.AR_DICT))
    for seed in range(30):
        img, ids, sides = shapes.marker_sheet(seed)
        corners, got, _ = det.detectMarkers(img)
        assert sorted(got.reshape(-1).tolist()) == ids.tolist()
        s = np.array([np.linalg.norm(np.roll(c.reshape(4, 2), -1, axis=0) - c.reshape(4, 2), axis=1).mean() for c in corners])
        assert np.abs(s[np.argsort(got.reshape(-1))] - sides).max() < 0.5
    blank, ids, sides = shapes.marker_sheet(7, 0)
    assert det.detectMarkers(blank)[1] is None and len(ids) == len(sides) == 0


def test_coin_circles_are_found_for_the_usual_parameter_choices():
    check = DRILLS["drill_coin_circles"]
    bad = n = 0
    for seed in range(30):
        img, discs = shapes.coin_sheet(seed)
        for dp, md, p2 in itertools.product((1.0, 1.2, 1.5), (25, 45), (20, 30)):
            c = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, md, param1=100, param2=p2, minRadius=10, maxRadius=40)
            n += 1
            bad += c is None or not check.verify(Case({"gray": img}, discs), {"circles": c[0][np.argsort(c[0][:, 0])]})
    assert bad == 0, f"{bad}/{n}"
    img, discs = shapes.coin_sheet(3)
    assert not check.verify(Case({"gray": img}, discs), {"circles": discs[::-1]}), "right circles in the wrong order"
    assert not check.verify(Case({"gray": img}, discs), {"circles": discs + [0, 0, 6]}), "radius 6 px off"


def test_rod_axis_is_recovered_from_the_box_points_for_every_orientation():
    worst_a = worst_s = 0.0
    for deg in range(0, 180, 5):
        m, (theta, length, width) = calling.rod_scene(deg + 31, angle=float(deg))
        cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        e0, e1 = box[1] - box[0], box[2] - box[1]
        long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
        worst_a = max(worst_a, abs((math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180 - theta + 90) % 180 - 90))
        worst_s = max(worst_s, abs(max(np.linalg.norm(e0), np.linalg.norm(e1)) - length), abs(min(np.linalg.norm(e0), np.linalg.norm(e1)) - width))
    assert worst_a < calling.ROD_ANGLE_TOL and worst_s < calling.ROD_SIZE_TOL, (worst_a, worst_s)


def test_rotation_canvas_and_marker_follow_the_clockwise_convention():
    check = DRILLS["drill_rotate_clockwise"]
    for seed in range(5):
        img = calling.flag_image(seed)
        for angle in (-150.0, -60.0, 15.0, 90.0, 135.0):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
            cs, sn = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(round(h * sn + w * cs)), int(round(h * cs + w * sn))
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            case = Case({"img": img, "angle": angle}, None)
            assert check.verify(case, {"rotated": cv2.warpAffine(img, M, (nw, nh))}), (seed, angle)
            wrong = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            wrong[0, 2] += nw / 2 - w / 2
            wrong[1, 2] += nh / 2 - h / 2
            if abs(math.sin(math.radians(angle))) > 0.2 and abs(math.cos(math.radians(angle))) > 0.2:
                assert not check.verify(case, {"rotated": cv2.warpAffine(img, wrong, (nw, nh))}), (seed, angle)


def test_similarity_fit_recovers_the_manufactured_transform_with_outliers():
    check = DRILLS["drill_similarity_fit"]
    for seed in range(40):
        src, dst, truth = removed.similarity_pairs(seed + 4000)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        out = {"angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), "scale": float(np.hypot(M[0, 0], M[1, 0])),
               "shift": M[:, 2]}
        assert check.verify(Case({"src": src, "dst": dst}, truth), out), seed


def test_orb_and_sift_both_recover_rotation_and_scale():
    rot, sc = DRILLS["drill_rotation_from_features"], DRILLS["drill_scale_from_features"]
    for case in rot.make_cases()[:2]:
        for det, norm in ((cv2.ORB_create(1500), cv2.NORM_HAMMING), (cv2.SIFT_create(), cv2.NORM_L2)):
            ka, da = det.detectAndCompute(case.inputs["img_a"], None)
            kb, db = det.detectAndCompute(case.inputs["img_b"], None)
            m = cv2.BFMatcher(norm, crossCheck=True).match(da, db)
            M, _ = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]),
                                               method=cv2.RANSAC, ransacReprojThreshold=3.0)
            assert rot.verify(case, {"angle_deg": float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))})
    for case in sc.make_cases()[:2]:
        for det, norm in ((cv2.ORB_create(1500), cv2.NORM_HAMMING), (cv2.SIFT_create(), cv2.NORM_L2)):
            ka, da = det.detectAndCompute(case.inputs["img_a"], None)
            kb, db = det.detectAndCompute(case.inputs["img_b"], None)
            m = cv2.BFMatcher(norm, crossCheck=True).match(da, db)
            M, _ = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]),
                                               method=cv2.RANSAC, ransacReprojThreshold=3.0)
            assert sc.verify(case, {"scale": float(math.hypot(M[0, 0], M[1, 0]))})


def test_tracker_cases_are_ones_where_mil_holds_and_a_frozen_box_does_not():
    for case in DRILLS["drill_track_box"].make_cases():
        frames, box0 = case.inputs["frames"], tuple(int(v) for v in case.inputs["box0"])
        tr = cv2.TrackerMIL_create()
        tr.init(frames[0], box0)
        for f in frames[1:]:
            ok, box = tr.update(f)
        assert removed._iou(box, case.truth) >= 0.65
        assert removed._iou(box0, case.truth) < removed.TRACK_MIN_IOU - 0.1


def test_background_subtractor_reference_is_exact_on_fast_clips_across_settings():
    for seed in range(12):
        frames, truth = removed.conveyor_clip(seed + 300)
        for hist, thr, shadows in ((100, 16, False), (500, 25, True)):
            mog = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thr, detectShadows=shadows)
            for f in frames:
                m = mog.apply(f)
            pred = m == 255
            assert (pred & truth).sum() / (pred | truth).sum() > removed.BG_MIN_IOU + 0.2


def test_exact_nearest_cases_separate_the_flag_from_the_numpy_formula():
    check = DRILLS["drill_label_map_resize"]
    nearest_rejected = 0
    for case in check.make_cases():
        lm, h, w = case.inputs["labels"], case.inputs["out_h"], case.inputs["out_w"]
        assert check.verify(case, {"resized": cv2.resize(lm, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)})
        nearest_rejected += not check.verify(case, {"resized": cv2.resize(lm, (w, h), interpolation=cv2.INTER_NEAREST)})
        rows = np.minimum(((np.arange(h) + 0.5) * lm.shape[0] / h).astype(int), lm.shape[0] - 1)
        cols = np.minimum(((np.arange(w) + 0.5) * lm.shape[1] / w).astype(int), lm.shape[1] - 1)
        assert not check.verify(case, {"resized": lm[rows][:, cols]}), "a hand-rolled centre formula must differ on these sizes"
    assert nearest_rejected >= 2, "plain INTER_NEAREST must differ from INTER_NEAREST_EXACT on most of the cases"


def test_text_wrap_verifier_measures_whole_lines_and_accepts_both_ways_of_filling_a_line():
    check = DRILLS["drill_caption_wrap"]
    for case in check.make_cases():
        i = case.inputs
        words = i["text"].split()
        lines, cur = [], ""
        for w in words:
            trial = w if not cur else cur + " " + w
            if cur and drift.text_width(trial, i["scale"], i["thickness"]) > i["max_width"]:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        assert check.verify(case, {"lines": lines})
        assert not check.verify(case, {"lines": words}), "one word per line is not greedy"
        assert not check.verify(case, {"lines": [" ".join(words)]}), "one long line does not fit"


def test_bubble_counting_with_mser_needs_min_diversity_zero():
    check = DRILLS["drill_bubble_count_mser"]
    for case in check.make_cases():
        n_default = len(cv2.MSER_create().detectRegions(case.inputs["gray"])[0])
        n_zero = len(cv2.MSER_create(min_diversity=0.0).detectRegions(case.inputs["gray"])[0])
        assert n_default == 0 and n_zero >= 4 * len(case.truth), (n_default, n_zero)


def test_the_text_ink_mask_keeps_the_soft_edge_pixels_the_old_idiom_loses():
    for case in DRILLS["drill_text_ink_mask"].make_cases():
        out = case.inputs["canvas"].copy()
        cv2.putText(out, case.inputs["text"], (case.inputs["x"], case.inputs["y"]), cv2.FONT_HERSHEY_SIMPLEX,
                    case.inputs["scale"], 255, case.inputs["thickness"])
        soft = (out > 0) & (out < 255)
        assert soft.sum() > 0.2 * (out > 0).sum(), "5.0 renders Hershey text anti-aliased even with LINE_8"


_ORDER_SRC = (
    "import cv2\nimport numpy as np\n\n\ndef solve(hershey):\n"
    "    def render(size):\n"
    "        img = np.full((60, 220, 3), 255, np.uint8)\n"
    "        cv2.putText(img, 'Lot 42-B', (10, 40), (0, 0, 0), cv2.FontFace('sans'), size)\n"
    "        return img\n"
    "    render(24)\n"
    "    if hershey:\n"
    "        cv2.getTextSize('Inspection OK', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)\n"
    "    return {'img': render(18)}\n")


def test_a_hershey_call_changes_later_fontface_rendering_on_5_0_0():
    """The measurement behind drift._ff_truth: each run below is a fresh sandbox process, so the only difference is one Hershey
    getTextSize between a size-24 and a size-18 FontFace putText. If this starts failing the library fixed it and the truth
    no longer needs a clean process."""
    clean = run_solution(_ORDER_SRC, [{"hershey": False}])
    polluted = run_solution(_ORDER_SRC, [{"hershey": True}])
    assert clean.ok and polluted.ok
    assert (clean.outputs[0]["img"] != polluted.outputs[0]["img"]).sum() > 500


def test_fontface_truth_is_rendered_in_a_clean_process_and_is_stable():
    first = drift._ff_truth()
    assert first is drift._ff_truth()
    # a Hershey call in THIS process must not change what the check expects: the truth was rendered elsewhere
    cv2.getTextSize("polluting the process", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    ref = run_solution(DRILLS["drill_fontface_label"].reference, drift._ff_inputs())
    assert ref.ok and all(np.array_equal(o["img"], t[0]) and np.array_equal(o["rect"], t[1]) for o, t in zip(ref.outputs, first))
    # and the cases the check hands out agree with it
    assert all(np.array_equal(c.truth[0], t[0]) for c, t in zip(DRILLS["drill_fontface_label"].make_cases(), first))


def test_a_constant_unopened_capture_report_is_all_nan_and_only_its_length_varies():
    lengths = {c.truth for c in DRILLS["drill_capture_properties"].make_cases()}
    assert len(lengths) >= 3
    cap = cv2.VideoCapture()
    assert not cap.isOpened() and all(cap.get(p) == -1.0 for p in drift.PROPS.values())


def test_numpy_two_names_and_rules_the_numpy_drills_stand_on():
    a = np.zeros((2, 2, 3), np.uint8)
    for name in ("int0", "Inf", "float_", "product", "round_"):
        assert not hasattr(np, name), name
    assert not hasattr(a, "ptp") and np.ptp(a) == 0
    with pytest.raises(OverflowError):
        a * 587
    assert (a * 255).dtype == np.uint8  # fits the dtype: wraps silently, which is what the uint8_wrap variant relies on
    assert cv2.boxPoints(((5, 5), (4, 2), 30)).dtype == np.float32


# ---------------------------------------------------------------- coverage of PLAN.md section 0

# Every probe whose recorded 4.14.0 and 5.0.0 results differ (the "behaviour changes" of probes/results/diff_4.14_vs_5.0.md plus its one
# error-text change) is either cited by a drill's `probes` or listed here with the reason it has no drill.
NOT_DRILLED = {
    "has cv2.CascadeClassifier": "needs a Haar XML file; the sandbox has no files and the 5.0 wheel ships none",
    "haarcascade_xml_shipped": "same: the bundled Haar XML files are not in the wheel",
    "has cv2.HOGDescriptor": "no confirmed replacement (api_check has none); a verified numpy HOG is a task of its own",
    "has cv2.gapi": "no use in a one-function task",
    "has cv2.dnn.readNetFromCaffe": "needs a model file; the replacement (ONNX) needs one too",
    "has cv2.dnn.readNetFromDarknet": "needs a model file",
    "has cv2.AgastFeatureDetector_create": "a task cannot mention AGAST without naming the API the answer must avoid",
    "has cv2.CV_Bool": "constant only; the dtype behaviours behind it are drilled per function (bool, uint32, float16)",
    "has cv2.CV_32U": "constant only; see drill_u32_extrema",
    "has cv2.CV_64U": "constant only; no function that takes it is drillable without int64 narrowing getting in the way",
    "has cv2.CV_64S": "constant only; the gotcha suite's dtype_int64_add covers the int64 narrowing it cannot fix",
    "findChessboardCorners": "held out of training: the chessboard_corners family is dev/test only (cvbench/splits.py)",
    "findChessboardCornersSB": "held out of training, same family",
    "warpAffine_rotate_fingerprint": "changes up to 4 gray levels on 3 % of pixels; nothing a model should do differently "
                                      "(the gotcha suite's drift_warp_tolerance pins the verifier side)",
    "warpPerspective_fingerprint": "same",
    "findContours_3channel": "same failure with a different error message; matters only for repair prompts that quote errors",
}


def _changed_probes() -> list[str]:
    return [p for p in REC5 if REC4[p] != REC5[p]]


def coverage_table() -> str:
    cited: dict[str, list[str]] = {}
    for did, d in DRILLS.items():
        for p in d.probes:
            cited.setdefault(p, []).append(did)
    rows = ["| changed probe (4.14.0 -> 5.0.0) | drills | not drilled because |", "|---|---|---|"]
    for p in _changed_probes():
        rows.append(f"| `{p}` | {', '.join(f'`{x}`' for x in cited.get(p, [])) or '-'} | {NOT_DRILLED.get(p, '')} |")
    return "\n".join(rows)


def test_every_changed_probe_has_a_drill_or_a_stated_reason():
    changed = _changed_probes()
    assert len(changed) == 41, len(changed)  # 40 behaviour changes + 1 error-text change in diff_4.14_vs_5.0.md
    cited = {p for d in DRILLS.values() for p in d.probes}
    for p in changed:
        assert (p in cited) != (p in NOT_DRILLED), f"{p}: cited by a drill XOR listed as not drilled"
    assert set(NOT_DRILLED) <= set(changed) and not set(NOT_DRILLED) & cited
    table = coverage_table()
    print("\n" + table)
    assert table.count("\n") == 2 + len(changed) - 1
    drilled = [p for p in changed if p in cited]
    assert len(drilled) == 25 and len(NOT_DRILLED) == 16


# PLAN.md section 0, item by item: the statement and the drills that exercise it. An empty tuple says "no drill, on purpose".
PLAN_SECTION_0 = {
    "1. removed or moved to contrib: CascadeClassifier / HOG / cv2.ml / G-API / Caffe / Darknet (only cv2.ml is drillable)": (
        "drill_knn_labels",),
    "1. missing from the main wheel: AKAZE, KAZE, BRISK, AGAST, BOWKMeansTrainer, GOTURN (AGAST is not drillable)": (
        "drill_rotation_from_features", "drill_scale_from_features", "drill_visual_words", "drill_track_box"),
    "1. new: FontFace": ("drill_fontface_label",),
    "1. bool masks accepted per function (countNonZero, mask=, moments) and rejected elsewhere (findContours, dilate)": (
        "drill_bool_background_mean", "drill_bool_centroid", "drill_bool_blob_box", "drill_bool_grow"),
    "1. bitwise_not on a bool mask returns all-true bytes": ("drill_bool_background_mean",),
    "1. uint32 and float16 accepted per function": ("drill_u32_extrema", "drill_f16_downscale"),
    "1. VideoCapture.get on an unsupported property returns -1": ("drill_capture_properties",),
    "2. HoughLinesP (N, 4)": ("drill_hough_longest",),
    "2. HoughLinesWithAccumulator (N, 3)": ("drill_hough_acc_skew",),
    "2. findNonZero (N, 2)": ("drill_nonzero_extremes",),
    "2. convexityDefects (K, 4)": ("drill_defect_notches",),
    "2. calcHist (256,)": ("drill_hist_unused_levels",),
    "2. minEnclosingTriangle (3, 2)": ("drill_enclosing_triangle",),
    "2. ArUco ids (N,)": ("drill_aruco_sizes",),
    "2. findChessboardCorners / SB (N, 2)": (),  # held out of training
    "2. unchanged (controls): HoughCircles, kmeans labels, the contour hierarchy": (
        "drill_coin_circles", "drill_palette_kmeans", "drill_hierarchy_holes"),
    "3. MSER default parameters": ("drill_bubble_count_mser",),
    "3. getTextSize and glyph metrics, putText always anti-aliased": ("drill_caption_wrap", "drill_text_ink_mask"),
    "3. INTER_NEAREST_EXACT": ("drill_label_map_resize",),
    "3. warpAffine / warpPerspective moved by up to 4 gray levels": (),  # nothing for a model to do differently
    "4. timeless, PLAN's own list: int64 narrowing, drawing views, perspective dtypes, fillPoly, imread None": (),  # gotcha suite
    "4. timeless, probed but not in the gotcha suite: Canny / findContours on float, cvtColor on float64": (
        "drill_float_edges", "drill_soft_mask_boxes", "drill_rgb_float_gray"),
    "4. timeless, found while building: Otsu, matchTemplate, minAreaRect, rotation, hierarchy, kmeans": (
        "drill_otsu_binarize", "drill_template_center", "drill_rod_axis", "drill_rotate_clockwise", "drill_hierarchy_holes",
        "drill_palette_kmeans"),
    "removed APIs with replacements in api_check.SUGGESTIONS": (
        "drill_track_box", "drill_similarity_fit", "drill_knn_labels", "drill_visual_words", "drill_moving_mask",
        "drill_aa_overlay", "drill_rotation_from_features", "drill_scale_from_features"),
    "NumPy 2 removals met by OpenCV code (np.int0, np.Inf, ndarray.ptp, NEP 50)": (
        "drill_rotated_box_draw", "drill_nearest_foreground", "drill_channel_stretch", "drill_int_luma"),
}


def test_plan_section_0_behaviours_map_to_existing_drills():
    for statement, ids in PLAN_SECTION_0.items():
        assert all(i in DRILLS for i in ids), statement
    covered = {i for ids in PLAN_SECTION_0.values() for i in ids}
    assert covered == set(DRILLS), f"drills in no statement: {sorted(set(DRILLS) - covered)}"
    assert sorted(s for s, ids in PLAN_SECTION_0.items() if not ids) == sorted([
        "2. findChessboardCorners / SB (N, 2)",
        "3. warpAffine / warpPerspective moved by up to 4 gray levels",
        "4. timeless, PLAN's own list: int64 narrowing, drawing views, perspective dtypes, fillPoly, imread None"])
    text = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    for needle in ("HoughLinesP", "findNonZero", "convexityDefects", "minEnclosingTriangle", "HoughLinesWithAccumulator", "calcHist",
                   "INTER_NEAREST_EXACT", "getTextSize", "MSER", "VideoCapture", "FontFace", "BOWKMeansTrainer"):
        assert needle in text, f"PLAN.md section 0 no longer mentions {needle}: update PLAN_SECTION_0"
