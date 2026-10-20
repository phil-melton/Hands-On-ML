"""The Stage 3 bake-off harness: statistics against hand-computed values, code extraction, the error taxonomy on
synthetic grades, generation against a stubbed Ollama, and a dry run in which the "model" answers with a family's
REFERENCE (must pass) or a known mutant (must fail, with the right class). No GPU, no real network: the dry run talks
to a fake Ollama on a loopback port.
"""
from __future__ import annotations

import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from bakeoff import generate
from bakeoff.extract import extract_code
from bakeoff.generate import GenSettings, OllamaError, run_generation
from bakeoff.grade import grade_job
from bakeoff.predict import parse_per_rung, predict
from bakeoff.rungs import RUNGS, request_options, sample_seed, system_prompt
from bakeoff.run import main
from bakeoff.stats import (binomial_se, bootstrap_se, cluster_se, mcnemar_exact, pass_at_k, paired_compare)
from bakeoff.store import JsonlWriter, RunDir, read_json, read_jsonl
from bakeoff.taxonomy import CLASSES, classify, parse_run_detail, sample_failures
from bakeoff.tasks import load_tasks
from cvbench.families import contour_area, hough_segments
from cvbench.gotchas import CHECKS
from train.config import SYSTEM


# ================================================================ statistics, against hand computation

def test_pass_at_k_hand_values():
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)                    # c/n
    assert pass_at_k(5, 1, 2) == pytest.approx(1 - 6 / 10)             # 1 - C(4,2)/C(5,2)
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)          # 1 - C(7,5)/C(10,5) = 11/12
    assert pass_at_k(20, 4, 3) == pytest.approx(1 - 560 / 1140)        # general n and k
    assert pass_at_k(5, 3, 3) == 1.0                                   # only 2 wrong replies, fewer than k = 3


def test_pass_at_k_edges_and_validation():
    for k in (1, 3, 5):
        assert pass_at_k(5, 0, k) == 0.0
        assert pass_at_k(5, 5, k) == 1.0
    assert pass_at_k(1, 1, 1) == 1.0 and pass_at_k(1, 0, 1) == 0.0
    for n, c, k in ((5, 2, 6), (5, 6, 1), (5, -1, 1), (5, 2, 0)):
        with pytest.raises(ValueError):
            pass_at_k(n, c, k)


def test_pass_at_k_is_unbiased_by_enumeration():
    # estimator 1 equals the exact probability that a random k-subset of the n replies contains a correct one
    from itertools import combinations
    n, c, k = 7, 3, 4
    replies = [1] * c + [0] * (n - c)
    subsets = list(combinations(range(n), k))
    exact = sum(any(replies[i] for i in s) for s in subsets) / len(subsets)
    assert pass_at_k(n, c, k) == pytest.approx(exact)


def test_mcnemar_exact_against_binomial_tails():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0                                   # tail reaches the middle: capped at 1
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)                 # 2 * P(X<=0 | n=5)
    assert mcnemar_exact(5, 0) == mcnemar_exact(0, 5)                   # symmetric
    assert mcnemar_exact(1, 5) == pytest.approx(2 * 7 / 64)             # 2 * (1+6)/2^6
    assert mcnemar_exact(3, 10) == pytest.approx(2 * 378 / 8192)        # 2 * (1+13+78+286)/2^13
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(2, 2) == 1.0


def test_paired_compare_counts_difference_and_se():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 0]
    r = paired_compare(a, b)
    assert (r["n"], r["a_only"], r["b_only"], r["both"], r["neither"]) == (5, 2, 1, 1, 1)
    assert r["diff"] == pytest.approx(0.2)
    assert r["se"] == pytest.approx(math.sqrt(0.7 / 5))                 # d = [0, 1, -1, 0, 1]: var(ddof=1) = 2.8/4
    assert r["p"] == pytest.approx(mcnemar_exact(2, 1))
    with pytest.raises(ValueError):
        paired_compare([1, 0], [1])


def test_binomial_se():
    assert binomial_se(0.5, 100) == pytest.approx(0.05)
    assert binomial_se(0.0, 50) == 0.0
    assert math.isnan(binomial_se(0.5, 0))


def test_cluster_se_by_hand():
    # two prompts, one all-pass and one all-fail: the mean 0.5 has the SD of two cluster means / sqrt(2) = 0.5
    assert cluster_se([1, 1, 0, 0], ["a", "a", "b", "b"]) == pytest.approx(0.5)
    assert math.isnan(cluster_se([1, 0], ["a", "a"]))                   # one cluster: undefined


def test_bootstrap_se_sanity():
    rng = np.random.default_rng(1)
    v = (rng.random(400) < 0.5).astype(float)
    p = v.mean()
    assert bootstrap_se(v, B=2000, seed=3) == pytest.approx(math.sqrt(p * (1 - p) / 400), rel=0.1)
    assert bootstrap_se(v, B=2000, seed=3) == bootstrap_se(v, B=2000, seed=3)           # seeded
    assert bootstrap_se([0.7] * 30, B=200) == pytest.approx(0.0, abs=1e-12)             # constant: no spread
    assert math.isnan(bootstrap_se([1.0]))


def test_cluster_bootstrap_is_wider_when_tasks_share_outcomes():
    rng = np.random.default_rng(2)
    per_cluster = (rng.random(20) < 0.5).astype(float)
    v = np.repeat(per_cluster, 10)                                      # 20 prompts x 10 identical tasks
    clusters = np.repeat(np.arange(20), 10)
    plain, clustered = bootstrap_se(v, B=2000), bootstrap_se(v, clusters, B=2000)
    assert clustered > 2.5 * plain
    assert cluster_se(v, clusters) == pytest.approx(clustered, rel=0.25)


# ================================================================ rungs, seeds, prediction

def test_seed_is_deterministic_and_depends_on_every_part():
    s = sample_seed("r1-7b", "dev-contour_area-000", "sample", 3)
    assert s == sample_seed("r1-7b", "dev-contour_area-000", "sample", 3)
    assert 0 <= s < 2 ** 31
    others = {sample_seed("r2-4b", "dev-contour_area-000", "sample", 3), sample_seed("r1-7b", "dev-contour_area-001", "sample", 3),
              sample_seed("r1-7b", "dev-contour_area-000", "greedy", 3), sample_seed("r1-7b", "dev-contour_area-000", "sample", 4)}
    assert s not in others and len(others) == 4


def test_request_options_for_both_modes():
    g = request_options("greedy", num_predict=1024, num_ctx=4096, seed=9)
    assert g == {"temperature": 0.0, "repeat_penalty": 1.0, "num_predict": 1024, "num_ctx": 4096, "seed": 9}
    s = request_options("sample", num_predict=512, num_ctx=2048, seed=1)
    assert s == {"temperature": 0.8, "top_p": 0.95, "repeat_penalty": 1.0, "num_predict": 512, "num_ctx": 2048, "seed": 1}


def test_system_prompts_and_rung_table():
    assert system_prompt(RUNGS["r1-7b"]) == SYSTEM
    notes = system_prompt(RUNGS["r4-7b-notes"], "  NOTES: HoughLinesP returns (N, 4).  \n")
    assert notes.startswith(SYSTEM + "\n\n") and notes.endswith("(N, 4).")
    with pytest.raises(ValueError):
        system_prompt(RUNGS["r4-7b-notes"], None)
    assert RUNGS["r2-4b"].think is False and RUNGS["r3-9b"].think is False and RUNGS["r1-7b"].think is None
    assert RUNGS["r1-7b"].model == RUNGS["r4-7b-notes"].model == "base-qwen2.5-coder-7b"


def test_predict_hours_and_per_rung_parsing():
    rows = predict(["r1-7b", "r2-4b"], {"r1-7b": 50.0, "r2-4b": 100.0}, {"r1-7b": 300.0, "r2-4b": 300.0}, 238, 5,
                   overhead_s=0.5, load_s=90.0)
    assert rows[0]["replies"] == 1428
    assert rows[0]["s_per_reply"] == pytest.approx(300 / 50 + 0.5)
    assert rows[0]["hours"] == pytest.approx((1428 * 6.5 + 90) / 3600)
    assert rows[1]["hours"] == pytest.approx((1428 * 3.5 + 90) / 3600)
    assert parse_per_rung(["40", "r2-4b=70"], ["r1-7b", "r2-4b"], "--tok-s") == {"r1-7b": 40.0, "r2-4b": 70.0}
    with pytest.raises(SystemExit):
        parse_per_rung(["r1-7b=40"], ["r1-7b", "r2-4b"], "--tok-s")


def test_predict_cli_prints_a_table(capsys):
    assert main(["predict", "--tok-s", "40", "--tokens", "300", "--rung", "r1-7b"]) == 0
    out = capsys.readouterr().out
    assert "r1-7b" in out and "238 tasks" in out and "1428" in out


def test_jsonl_writer_repairs_a_torn_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')                              # killed in the middle of a row
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    with JsonlWriter(p) as w:
        w.write({"a": 3})
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}, {"a": 3}]
    p.write_text('{"a": 1}\n{"a": 2}')                                      # a whole row that only lost its newline
    with JsonlWriter(p) as w:
        w.write({"a": 3})
    assert [r["a"] for r in read_jsonl(p)] == [1, 2, 3]
    p.write_text('{"a": 1}\nnot json\n{"a": 2}\n')                          # damage in the middle is an error, not skipped
    with pytest.raises(ValueError):
        read_jsonl(p)
    assert read_jsonl(tmp_path / "missing.jsonl") == []


# ================================================================ tasks

def test_split_loading():
    dev = load_tasks(["dev"])
    assert len(dev) == 99 and all(t.kind == "family" and t.id.startswith("dev-") for t in dev)
    full = load_tasks()                                               # the default: test families + gotchas
    assert len(full) == 238 and sum(t.kind == "gotcha" for t in full) == 40
    assert {t.subset for t in full} == {"5.0", "timeless"}
    assert any(t.held_out for t in full) and not any(t.held_out for t in full if t.kind == "gotcha")
    one = load_tasks(["dev"], per_family=1)
    assert len(one) == 9 and len({t.family for t in one}) == 9
    assert [t.id for t in load_tasks(["dev", "gotchas"], only=r"^(dev-hough_segments-000|shape_calchist)$")] == \
        ["dev-hough_segments-000", "shape_calchist"]
    with pytest.raises(ValueError):
        load_tasks(["train"])
    # tasks that share a family and phrasing share the prompt, and therefore the cluster
    by_prompt = {}
    for t in load_tasks(["test"]):
        by_prompt.setdefault(t.prompt, set()).add(t.cluster)
    assert all(len(c) == 1 for c in by_prompt.values()) and len(by_prompt) < 198


# ================================================================ extraction

def test_extract_fenced_block():
    ex = extract_code("Here you go:\n```python\nimport cv2\n\ndef solve(img):\n    return {}\n```\nHope it helps.")
    assert ex.path == "fence" and ex.code.startswith("import cv2") and "Hope" not in ex.code and ex.think == ""


def test_extract_unfenced_reply_is_taken_whole():
    reply = "import cv2\n\ndef solve(img):\n    return {}\n"
    ex = extract_code(reply)
    assert ex.path == "bare" and ex.code == reply


def test_extract_strips_an_empty_think_block():
    ex = extract_code("<think>\n\n</think>\n\n```python\nx = 1\n```")
    assert ex.think == "empty" and ex.path == "fence" and ex.code == "x = 1\n"
    ex = extract_code("<think></think>def solve(img):\n    return {}")
    assert ex.think == "empty" and ex.path == "bare" and ex.code.startswith("def solve")


def test_extract_nonempty_think_block_is_stripped_too():
    ex = extract_code("<think>maybe ```python\nWRONG\n``` first</think>\n```python\nRIGHT\n```")
    assert ex.think == "nonempty" and ex.code == "RIGHT\n"


def test_extract_first_of_two_fences_and_language_handling():
    two = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
    assert extract_code(two).code == "first = 1\n"
    unlabelled = "```\nplain = 1\n```"
    assert extract_code(unlabelled).code == "plain = 1\n"
    assert extract_code("```py\nshort = 1\n```").code == "short = 1\n"
    # a bash block before the python block must not be mistaken for the answer, nor pair up with its neighbour
    mixed = "```bash\npip install opencv-python\n```\n```python\nreal = 1\n```"
    assert extract_code(mixed).code == "real = 1\n"
    assert extract_code("```json\n{}\n```").path == "bare"            # no python block: the whole reply


def test_extract_unclosed_fence_and_empty_reply():
    ex = extract_code("```python\ndef solve(img):\n    return {")       # truncated at num_predict
    assert ex.path == "fence-unclosed" and ex.code.startswith("def solve")
    for reply in ("", "   \n", "<think>\n\n</think>\n\n", "```python\n```"):
        assert extract_code(reply).path == "empty" and extract_code(reply).code == ""


# ================================================================ taxonomy on synthetic grade records

OK = {"passed": True, "detail": "ok (0.2s)"}


def _gates(**fail):
    g = {name: dict(OK) for name in ("exec-only", "lint", "api", "manufactured", "metamorphic")}
    for name, detail in fail.items():
        g[name.replace("_", "-")] = {"passed": False, "detail": detail}
    return g


def fam_rec(lint=(), api=(), **fail):
    gates = _gates(**fail)
    if lint:                                                           # findings mean the lint gate failed
        gates["lint"] = {"passed": False, "detail": "; ".join(f"line {f['line']}: [{f['rule']}] {f['message']}" for f in lint)}
    return {"passed": False, "gates": gates, "lint_findings": list(lint), "api_findings": list(api)}


def lint_f(rule, msg="..."):
    return {"rule": rule, "line": 3, "message": msg}


REMOVED = {"line": 4, "name": "cv2.xfeatures2d.SURF_create", "api": "cv2.xfeatures2d", "label": "removed_in_5",
           "replacement": "cv2.SIFT_create"}
IDX = "exception on input 0: IndexError too many indices for array: array is 2-dimensional, but 3 were indexed"

CASES = {
    "no-code": ({"passed": False, "code_empty": True}, "format", "no-code"),
    "syntax": (fam_rec(lint=[lint_f("syntax", "invalid syntax")], exec_only="exception: SyntaxError invalid syntax"), "format", "lint:syntax"),
    "no-solve": (fam_rec(lint=[lint_f("contract", "no top-level `def solve(...)`")],
                         exec_only="bad_output: ContractError no callable solve()"), "format", "lint:contract"),
    "removed-api-finding": (fam_rec(api=[REMOVED], exec_only="exception on input 0: AttributeError module 'cv2' has no attribute 'xfeatures2d'"),
                            "api", "removed-api:removed_in_5"),
    "cv2-attribute": (fam_rec(exec_only="exception on input 0: AttributeError module 'cv2' has no attribute 'createFoo'"), "api", "cv2-attribute"),
    "cv2-object-attribute": (fam_rec(exec_only="exception on input 0: AttributeError 'cv2.ORB' object has no attribute 'compute2'"),
                             "api", "cv2-attribute"),
    "unexpected-keyword": (fam_rec(exec_only="exception on input 0: TypeError 'foo' is an invalid keyword argument for findContours()"),
                           "api", "call-signature"),
    "required-argument": (fam_rec(exec_only="exception on input 0: error OpenCV(5.0.0) :-1: error: (-5:Bad argument) in function 'HoughLinesP'\n"
                                            "> Overload resolution failed:\n>  - Required argument 'threshold' (pos 3) not found"),
                          "api", "cv2-call-signature"),
    "import": (fam_rec(exec_only="exception on input 0: ModuleNotFoundError No module named 'skimage'"), "api", "import"),
    "solve-signature": (fam_rec(exec_only="exception on input 0: TypeError solve() got an unexpected keyword argument 'image'"),
                        "convention", "solve-signature"),
    "lint-import": (fam_rec(lint=[lint_f("import", "`import os` is not allowed")], manufactured="scene 4000000: count 3, but the scene has 4 shapes"),
                    "lint", "lint:import"),
    "lint-hides-algorithm": (fam_rec(lint=[lint_f("broad-except")], manufactured="scene 4000000: 0 areas for 4 shapes"), "lint", "lint:broad-except"),
    "sandbox-violation": (fam_rec(exec_only="violation on input 0: SandboxViolation cv2.imshow: GUI calls are not available in the worker"),
                          "lint", "sandbox-violation"),
    "timeout": (fam_rec(exec_only="timeout:  no result after 60s; process group killed"), "timeout", "timeout"),
    "index-error": (fam_rec(exec_only=IDX), "convention", "IndexError"),
    "unpack-type-error": (fam_rec(exec_only="exception on input 0: TypeError cannot unpack non-iterable numpy.int32 object"), "convention", "TypeError"),
    "unpack-value-error": (fam_rec(exec_only="exception on input 0: ValueError not enough values to unpack (expected 3, got 2)"), "convention", "ValueError"),
    "none-attribute": (fam_rec(exec_only="exception on input 1: AttributeError 'NoneType' object has no attribute 'shape'"), "convention", "AttributeError"),
    "cv2-assertion": (fam_rec(exec_only="exception on input 0: error OpenCV(5.0.0) /io/opencv/modules/imgproc/src/thresh.cpp:1743: error: "
                                        "(-210:Unsupported format or combination of formats)"), "convention", "cv2-assertion"),
    "cv2-argument-type": (fam_rec(exec_only="exception on input 0: error OpenCV(5.0.0) :-1: error: (-5:Bad argument) in function 'putText'\n"
                                            "> Overload resolution failed:\n>  - Can't parse 'org'. Sequence item with index 0 has a wrong type"),
                          "convention", "cv2-assertion"),
    "bad-output": (fam_rec(exec_only="bad_output on input 0: ContractError solve() must return a dict, got ndarray"), "convention", "bad-output"),
    "no-callable": (fam_rec(exec_only="bad_output: ContractError no callable solve()"), "format", "no-solve"),
    "verifier-contract": (fam_rec(manufactured="scene 4000001: contract: 'segments' should have ndim 2, got shape (7, 1, 4)"),
                          "convention", "verifier-contract"),
    "verifier-order": (fam_rec(manufactured="scene 4000001: areas are not sorted largest-first"), "convention", "verifier-contract"),
    "verifier-corner-order": (fam_rec(manufactured="scene 4000001: the corner positions are right but the order is wrong: reversed"),
                              "convention", "verifier-contract"),
    "verifier-value": (fam_rec(manufactured="scene 4000001: count 4, but the scene has 3 shapes"), "algorithm", "verifier-value"),
    "metamorphic-value": (fam_rec(metamorphic="scene 4000002: dim x0.45: count 0 vs 3 on the original image"), "algorithm", "verifier-value"),
    "gave-up": (fam_rec(manufactured="exception on input 13: ValueError chessboard not found"), "other", "exception:ValueError"),
    "name-error": (fam_rec(exec_only="exception on input 0: NameError name 'np' is not defined"), "other", "exception:NameError"),
    "killed": (fam_rec(exec_only="killed: child exited with code -9 and no result"), "other", "killed"),
    "grader-error": ({"passed": False, "grader_error": "Traceback...\nValueError: verifier blew up"}, "other", "grader-error"),
    "gotcha-lint-syntax": ({"passed": False, "stage": "lint", "detail": "lint: line 1: [syntax] invalid syntax", "lint_findings": [lint_f("syntax")]},
                           "format", "lint:syntax"),
    "gotcha-run-index": ({"passed": False, "stage": "run", "detail": IDX}, "convention", "IndexError"),
    "gotcha-contract": ({"passed": False, "stage": "contract", "detail": "item 0: 'segments' should have ndim 2, got shape (7,)"},
                        "convention", "stage:contract"),
    "gotcha-verify-shape": ({"passed": False, "stage": "verify", "detail": "item 0: resized has shape (90, 60, 3), ref has shape (60, 90, 3)"},
                            "convention", "verifier-contract"),
    "gotcha-verify-value": ({"passed": False, "stage": "verify", "detail": "item 0: 48 of 48 sums are wrong (int64 values above 2**31 were wrapped)"},
                            "algorithm", "gotcha-value"),
    "gotcha-malformed": ({"passed": False, "stage": "malformed", "detail": "item 0: verifier could not read the output (TypeError: x)"},
                         "convention", "stage:malformed"),
    "gotcha-api": ({"passed": False, "stage": "run", "detail": "exception on input 0: AttributeError module 'cv2' has no attribute 'foo'",
                    "api_findings": [REMOVED]}, "api", "removed-api:removed_in_5"),
    "unexplained": ({"passed": False, "gates": {}}, "other", "unexplained"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_taxonomy_rule_table(name):
    rec, cls, rule = CASES[name]
    c = classify(rec)
    assert (c.cls, c.rule) == (cls, rule), c
    assert c.cls in CLASSES and c.evidence and len(c.evidence) <= 300


def test_every_class_is_covered_by_the_table():
    assert {cls for _, cls, _ in CASES.values()} == set(CLASSES)


def test_taxonomy_keeps_the_other_signals_in_also():
    c = classify(CASES["lint-hides-algorithm"][0])
    assert c.cls == "lint" and "algorithm:verifier-value" in c.also
    c = classify(CASES["removed-api-finding"][0])
    assert c.cls == "api" and "api:cv2-attribute" in c.also
    assert classify({"passed": True}) is None
    assert classify(CASES["index-error"][0]).as_dict()["class"] == "convention"


def test_run_detail_parsing():
    assert parse_run_detail("timeout:  no result after 60s; process group killed") == \
        {"status": "timeout", "item": None, "etype": None, "msg": "no result after 60s; process group killed"}
    d = parse_run_detail("exception on input 3: IndexError index 1 is out of bounds")
    assert (d["status"], d["item"], d["etype"], d["msg"]) == ("exception", 3, "IndexError", "index 1 is out of bounds")
    assert parse_run_detail("scene 4000001: count 4, but the scene has 3 shapes") is None
    assert parse_run_detail("20 scenes within tolerance") is None


def test_sample_failures_is_bounded_deterministic_and_covers_classes():
    recs = []
    for i in range(12):
        rec, _, _ = CASES[["index-error", "verifier-value", "timeout"][i % 3]]
        recs.append({**rec, "rung": "r1-7b", "task_id": f"t{i:02d}", "mode": "greedy", "sample": 0})
    recs.append({"passed": True, "rung": "r1-7b", "task_id": "ok", "mode": "greedy", "sample": 0})
    a, b = sample_failures(recs, k=2, seed=7), sample_failures(recs, k=2, seed=7)
    assert set(a) == set(CLASSES)
    assert [(r["task_id"], c.rule) for r, c in a["convention"]] == [(r["task_id"], c.rule) for r, c in b["convention"]]
    assert all(len(v) <= 2 for v in a.values()) and len(a["convention"]) == 2 and len(a["format"]) == 0
    assert all(not r["passed"] for rows in a.values() for r, _ in rows)


# ================================================================ generation against a stubbed Ollama

PROMPT_TASKS = load_tasks(["dev"], only=r"^dev-contour_area-00[01]$")


class Stub:
    """Replaces generate.chat: records every request body and answers like Ollama."""

    def __init__(self, monkeypatch, reply=None, fail_after=None, fail_first=0):
        self.calls: list[dict] = []
        self.reply = reply or (lambda body: "```python\n# stub reply\nx = 1\n```")
        self.fail_after, self.fail_first, self.n_gen = fail_after, fail_first, 0
        monkeypatch.setattr(generate, "chat", self)
        monkeypatch.setattr(generate, "_info", lambda base, path, payload, timeout=30.0: {"version": "stub"} if path == "/api/version" else {})
        monkeypatch.setattr(generate.time, "sleep", lambda s: None)

    def __call__(self, base_url, body, timeout):
        if not body["messages"]:
            self.calls.append(body)
            return {"model": body["model"], "done": True, "done_reason": "unload" if body["keep_alive"] == 0 else "load", "load_duration": 7}
        if self.fail_after is not None and self.n_gen >= self.fail_after:
            raise OllamaError("stub: server went away")
        if self.fail_first and self.n_gen < self.fail_first:
            self.n_gen += 1
            raise OllamaError("stub: transient")
        self.n_gen += 1
        self.calls.append(body)
        warm = body["options"]["num_predict"] == generate.WARMUP_NUM_PREDICT
        text = self.reply(body)
        return {"message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop",
                "eval_count": 7777 if warm else 100, "eval_duration": 2_000_000_000, "prompt_eval_count": 300,
                "prompt_eval_duration": 500_000_000, "load_duration": 0, "total_duration": 3_000_000_000}

    def generations(self):
        return [c for c in self.calls if c["messages"] and c["options"]["num_predict"] != generate.WARMUP_NUM_PREDICT]


def settings(**kw):
    return GenSettings(base_url="http://stub", retries=kw.pop("retries", 0), backoff_s=0.0, **kw)


def test_requests_carry_the_protocol(tmp_path, monkeypatch):
    stub = Stub(monkeypatch)
    run = RunDir.of("proto", tmp_path)
    rungs = [RUNGS["r1-7b"], RUNGS["r2-4b"]]
    run_generation(run, rungs, PROMPT_TASKS, settings(samples=2))
    rows = read_jsonl(run.generations)
    assert len(rows) == 2 * (2 + 2 * 2)                                         # rungs x tasks x (1 greedy + 2 samples)

    for rung in rungs:
        mine = [c for c in stub.calls if c["model"] == rung.model]                # load, warm-up, 6 replies, unload
        assert len(mine) == 1 + 1 + 6 + 1
        assert mine[0]["messages"] == [] and mine[0]["keep_alive"] == "30m"      # load
        assert mine[1]["options"]["num_predict"] == generate.WARMUP_NUM_PREDICT  # warm-up first, before anything is timed
        assert mine[-1]["messages"] == [] and mine[-1]["keep_alive"] == 0         # unloaded at the end
        for body in mine[2:-1]:
            assert body["stream"] is False and body["keep_alive"] == "30m" and body["model"] == rung.model
            assert body["messages"][0] == {"role": "system", "content": SYSTEM}
            assert body["messages"][1]["role"] == "user"
            assert ("think" in body) == (rung.think is not None)
            if rung.think is not None:
                assert body["think"] is False
        gens = mine[2:-1]
        greedy, sampled = gens[:2], gens[2:]
        for body, task in zip(greedy, PROMPT_TASKS):
            assert body["messages"][1]["content"] == task.prompt
            assert body["options"] == {"temperature": 0.0, "repeat_penalty": 1.0, "num_predict": 1024, "num_ctx": 4096,
                                       "seed": sample_seed(rung.name, task.id, "greedy", 0)}
        for i, body in enumerate(sampled):
            task, s = PROMPT_TASKS[i % 2], i // 2
            assert body["options"] == {"temperature": 0.8, "top_p": 0.95, "repeat_penalty": 1.0, "num_predict": 1024,
                                       "num_ctx": 4096, "seed": sample_seed(rung.name, task.id, "sample", s)}
        assert len({b["options"]["seed"] for b in sampled}) == 4                   # every sample has its own seed

    r = rows[0]
    for field in ("rung", "task_id", "mode", "sample", "temperature", "seed", "reply", "code", "extract", "eval_count",
                  "eval_duration", "prompt_eval_count", "prompt_eval_duration", "load_duration", "done_reason", "truncated", "wall_s"):
        assert field in r
    assert r["extract"] == "fence" and r["code"].strip().endswith("x = 1") and r["done_reason"] == "stop" and r["truncated"] is False
    cfg = read_json(run.config)
    assert cfg["decoding"]["greedy"]["temperature"] == 0.0 and cfg["decoding"]["sample"]["top_p"] == 0.95
    assert cfg["decoding"]["greedy"]["repeat_penalty"] == 1.0 and "seed" not in cfg["decoding"]["greedy"]
    assert cfg["rungs"]["r2-4b"]["think"] is False and cfg["rungs"]["r1-7b"]["system"] == SYSTEM


def test_warm_up_is_discarded_and_kept_apart(tmp_path, monkeypatch):
    Stub(monkeypatch)
    run = RunDir.of("warm", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=1))
    assert all(r["eval_count"] == 100 for r in read_jsonl(run.generations))       # the stub's warm-up answer says 7777
    warm = read_jsonl(run.warmups)
    assert len(warm) == 1 and warm[0]["eval_count"] == 7777 and warm[0]["rung"] == "r1-7b"


def test_resume_skips_done_keys_and_survives_a_torn_line(tmp_path, monkeypatch):
    stub = Stub(monkeypatch)
    run = RunDir.of("resume", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=2))
    n_before = len(read_jsonl(run.generations))
    stub.calls.clear()
    assert run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=2)) == {"r1-7b": 0}
    assert stub.calls == []                                                       # nothing to do: the model is not even loaded

    lines = run.generations.read_text().splitlines()
    run.generations.write_text("\n".join(lines[:-3]) + "\n" + lines[-3][:40])      # lose 3 rows, the last one torn mid-line
    assert len(read_jsonl(run.generations)) == n_before - 3
    assert run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=2)) == {"r1-7b": 3}
    rows = read_jsonl(run.generations)
    assert len(rows) == n_before and len({(r["task_id"], r["mode"], r["sample"]) for r in rows}) == n_before
    assert len(stub.generations()) == 3

    stub.calls.clear()                                                            # asking for more samples adds only those
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=3))
    assert len(stub.generations()) == 2 and len(read_jsonl(run.generations)) == n_before + 2


def test_truncation_is_flagged(tmp_path, monkeypatch):
    class Truncating(Stub):
        def __call__(self, base_url, body, timeout):
            out = super().__call__(base_url, body, timeout)
            if body["messages"] and body["messages"][1]["content"] == PROMPT_TASKS[0].prompt and body["options"]["num_predict"] == 1024:
                out.update(done_reason="length", eval_count=1024)
            return out
    Truncating(monkeypatch)
    run = RunDir.of("trunc", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=1))
    rows = read_jsonl(run.generations)
    assert {r["task_id"]: r["truncated"] for r in rows if r["mode"] == "greedy"} == {PROMPT_TASKS[0].id: True, PROMPT_TASKS[1].id: False}
    assert all(r["done_reason"] == "length" for r in rows if r["truncated"])


def test_a_resume_with_other_options_is_refused(tmp_path, monkeypatch):
    Stub(monkeypatch)
    run = RunDir.of("cfg", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=1))
    with pytest.raises(SystemExit):
        run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=1, num_predict=512))
    run_generation(run, [RUNGS["r4-7b-notes"]], PROMPT_TASKS, settings(samples=0), notes_text="some notes")
    with pytest.raises(SystemExit):                                               # same rung, different system prompt
        run_generation(run, [RUNGS["r4-7b-notes"]], PROMPT_TASKS, settings(samples=0), notes_text="other notes")


def test_notes_rung_sends_the_notes_in_the_system_message(tmp_path, monkeypatch):
    stub = Stub(monkeypatch)
    run = RunDir.of("notes", tmp_path)
    run_generation(run, [RUNGS["r4-7b-notes"]], PROMPT_TASKS[:1], settings(samples=0), notes_text="NOTE: HoughLinesP returns (N, 4).")
    body = stub.generations()[0]
    assert body["model"] == "base-qwen2.5-coder-7b"
    assert body["messages"][0]["content"] == SYSTEM + "\n\nNOTE: HoughLinesP returns (N, 4)."
    assert read_json(run.config)["rungs"]["r4-7b-notes"]["notes_chars"] == len("NOTE: HoughLinesP returns (N, 4).")
    with pytest.raises(ValueError):
        run_generation(RunDir.of("notes2", tmp_path), [RUNGS["r4-7b-notes"]], PROMPT_TASKS[:1], settings(samples=0))


def test_dedupe_greedy_reuses_replies_for_identical_prompts(tmp_path, monkeypatch):
    dev = load_tasks(["dev"], only=r"^dev-contour_area-")
    same = [t for t in dev if t.prompt == dev[0].prompt]
    assert len(same) >= 2
    stub = Stub(monkeypatch)
    run = RunDir.of("dedupe", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], same, settings(samples=0, dedupe_greedy=True))
    assert len(stub.generations()) == 1
    rows = read_jsonl(run.generations)
    assert [r["reused_from"] for r in rows] == [None] + [same[0].id] * (len(same) - 1)
    assert all(r["wall_s"] == 0.0 for r in rows[1:]) and {r["reply"] for r in rows} == {rows[0]["reply"]}
    stub.calls.clear()
    run_generation(RunDir.of("nodedupe", tmp_path), [RUNGS["r1-7b"]], same, settings(samples=0))
    assert len(stub.generations()) == len(same)


def test_the_model_is_unloaded_after_an_error(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, fail_after=3)                                         # warm-up + 2 replies, then the server dies
    run = RunDir.of("fail", tmp_path)
    with pytest.raises(OllamaError):
        run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS, settings(samples=2))
    assert stub.calls[-1]["messages"] == [] and stub.calls[-1]["keep_alive"] == 0
    assert len(read_jsonl(run.generations)) == 2 and not run.done.exists()          # what was done is kept, no done marker


def test_transient_failures_are_retried(tmp_path, monkeypatch):
    Stub(monkeypatch, fail_first=1)
    run = RunDir.of("retry", tmp_path)
    run_generation(run, [RUNGS["r1-7b"]], PROMPT_TASKS[:1], GenSettings(base_url="http://stub", samples=0, retries=2, backoff_s=0.0))
    assert len(read_jsonl(run.generations)) == 1 and run.done.exists()


# ================================================================ the transport, against a fake Ollama on loopback

class FakeOllama:
    """Answers /api/chat, /api/show, /api/version and /api/ps like Ollama; the reply text comes from reply(model, prompt)."""

    def __init__(self, reply):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/chat":
                    outer.requests.append(body)
                    if not body["messages"]:
                        return self._send({"model": body["model"], "done": True, "done_reason": "load", "load_duration": 5})
                    text = outer.reply(body["model"], body["messages"][-1]["content"])
                    return self._send({"message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop",
                                       "eval_count": 120, "eval_duration": 3_000_000_000, "prompt_eval_count": 400,
                                       "prompt_eval_duration": 400_000_000, "load_duration": 0, "total_duration": 4_000_000_000})
                if self.path == "/api/show":
                    return self._send({"capabilities": ["completion"], "parameters": "stop \"<|im_end|>\"", "details": {"family": "stub"}})
                self._send({"error": "not found"}, 404)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._send({"version": "0.0.0-fake"})
                if self.path == "/api/ps":
                    last = outer.requests[-1]["model"] if outer.requests else "none"
                    return self._send({"models": [{"name": last + ":latest", "size": 1000, "size_vram": 1000}]})
                self._send({"error": "not found"}, 404)

        self.reply = reply
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_transport_round_trip_and_errors():
    fake = FakeOllama(lambda model, prompt: f"echo {model}")
    try:
        r = generate.chat(fake.url, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, 10)
        assert r["message"]["content"] == "echo m" and r["eval_count"] == 120
        with pytest.raises(OllamaError):
            generate._http("POST", fake.url + "/api/nope", {}, 5)
    finally:
        fake.close()
    with socket.socket() as s:                                                    # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    with pytest.raises(OllamaError):
        generate.chat(f"http://127.0.0.1:{dead}", {"model": "m", "messages": []}, 2)


# ================================================================ grading edge cases, direct

def _job(task, code, **settings):
    return {"task": {"task_id": task.id, "kind": task.kind, "family": task.family, "task_seed": task.seed}, "code": code,
            "settings": {"n_scenes": 20, "n_meta": 5, "timeout_s": None, **settings}}


def _grade_rec(task, payload, **extra):
    from bakeoff.grade import make_grade_row
    gen = {"rung": "r1-7b", "mode": "greedy", "sample": 0, "code": "x", **task.meta(), **extra}
    return make_grade_row(gen, payload, cached=False)


def test_timeout_and_syntax_error_are_graded_and_classified():
    task = load_tasks(["dev"], only=r"^dev-histogram_stats-000$")[0]
    slow = grade_job(_job(task, "import cv2\n\ndef solve(img):\n    while True:\n        pass\n", timeout_s=1.0))
    assert slow["passed"] is False and slow["gates"]["exec-only"]["detail"].startswith("timeout")
    assert _grade_rec(task, slow)["taxonomy"]["class"] == "timeout"
    broken = grade_job(_job(task, "def solve(img:\n    return {}\n"))
    assert broken["passed"] is False and broken["lint_findings"][0]["rule"] == "syntax"
    assert _grade_rec(task, broken)["taxonomy"]["class"] == "format"


def test_grades_follow_the_four_gate_rule_and_api_only_reports():
    task = load_tasks(["dev"], only=r"^dev-histogram_stats-000$")[0]
    from cvbench.families import histogram_stats
    ok = grade_job(_job(task, histogram_stats.REFERENCE))
    assert ok["passed"] is True and all(g["passed"] for g in ok["gates"].values()) and ok["ran"] is True
    # a removed API that is never called still fires the report-only api gate, but does not decide the pass
    with_api = histogram_stats.REFERENCE.replace("def solve(", "UNUSED = lambda: cv2.AKAZE_create\n\n\ndef solve(", 1)
    r = grade_job(_job(task, with_api))
    assert r["gates"]["api"]["passed"] is False and r["passed"] is True and r["api_findings"][0]["label"] == "removed_in_5"


# ================================================================ end to end: a fake Ollama, the CLI, the real graders

def _code(fam, which="REFERENCE"):
    return fam.REFERENCE if which == "REFERENCE" else fam.MUTANTS[which][1]


def _fenced(code):
    return f"Here is the function.\n\n```python\n{code}```\n\nIt follows the contract."


def test_dry_run_reference_passes_and_mutants_fail_with_the_right_class(tmp_path, capsys):
    ids = ["dev-contour_area-000", "dev-contour_area-001", "dev-hough_segments-000", "shape_houghlinesp", "shape_findnonzero"]
    tasks = {t.id: t for t in load_tasks(["dev", "gotchas"], only="^(" + "|".join(ids) + ")$")}
    assert set(tasks) == set(ids)
    ca = contour_area
    inject_api = ca.REFERENCE.replace("def solve(img):\n", "def solve(img):\n    cv2.xfeatures2d.SURF_create()\n", 1)
    assert inject_api != ca.REFERENCE

    def reference_for(tid):
        if tid.startswith("dev-contour"):
            return ca.REFERENCE
        if tid.startswith("dev-hough"):
            return hough_segments.REFERENCE
        return CHECKS[tid].reference

    answers = {  # (model tag, task id) -> reply text
        **{("base-qwen2.5-coder-7b", t): _fenced(reference_for(t)) for t in ids},
        ("base-qwen3.5-4b", "dev-contour_area-000"): _fenced(_code(ca, "opencv3_unpack")),               # convention: ValueError
        ("base-qwen3.5-4b", "dev-contour_area-001"): _fenced(_code(ca, "retr_tree")),                    # algorithm
        ("base-qwen3.5-4b", "dev-hough_segments-000"): _fenced(_code(hough_segments, "n1x4_layout")),    # convention: contract
        ("base-qwen3.5-4b", "shape_houghlinesp"): _fenced(CHECKS["shape_houghlinesp"].wrong["legacy_index"].code),  # convention
        ("base-qwen3.5-4b", "shape_findnonzero"): "I am not able to help with that request.",           # format: no code
        ("base-qwen3.5-9b", "dev-contour_area-000"): _fenced(inject_api),                                # api
        ("base-qwen3.5-9b", "dev-contour_area-001"): _fenced("import os\n" + ca.REFERENCE),               # lint
        **{("base-qwen3.5-9b", t): _fenced(reference_for(t)) for t in ids[2:]},
    }
    by_prompt = {(m, tasks[t].prompt): text for (m, t), text in answers.items()}
    fake = FakeOllama(lambda model, prompt: by_prompt[(model, prompt)])
    root = str(tmp_path)
    try:
        argv = ["--run", "dry", "--root", root]
        assert main(["generate", *argv, "--ollama", fake.url, "--rung", "r1-7b,r2-4b", "--rung", "r3-9b", "--split", "dev",
                     "--split", "gotchas", "--only", "^(" + "|".join(ids) + ")$", "--samples", "5"]) == 0
        run = RunDir.of("dry", tmp_path)
        gens = read_jsonl(run.generations)
        assert len(gens) == 3 * 5 * 6 and run.done.exists()
        assert {g["extract"] for g in gens if g["rung"] == "r1-7b"} == {"fence"}
        assert {g["extract"] for g in gens if g["task_id"] == "shape_findnonzero" and g["rung"] == "r2-4b"} == {"bare"}
        n_requests = len(fake.requests)
        assert main(["generate", *argv, "--ollama", fake.url, "--rung", "r1-7b,r2-4b,r3-9b", "--split", "dev", "--split", "gotchas",
                     "--only", "^(" + "|".join(ids) + ")$"]) == 0
        assert len(fake.requests) == n_requests and len(read_jsonl(run.generations)) == len(gens)   # resumed: nothing asked again
    finally:
        fake.close()
    assert fake.requests[-1]["keep_alive"] == 0 and fake.requests[0]["messages"] == []

    assert main(["grade", *argv, "--workers", "3"]) == 0
    grades = read_jsonl(run.grades)
    assert len(grades) == len(gens)
    assert sum(g["cached"] for g in grades) > 0.6 * len(grades)                         # identical (task, code) graded once
    assert main(["grade", *argv, "--workers", "2", "--follow", "--poll", "0.1"]) == 0   # all graded, done marker present
    assert len(read_jsonl(run.grades)) == len(gens)
    assert not [g for g in grades if g.get("grader_error")]

    def tax(rung, tid):
        g = next(g for g in grades if g["rung"] == rung and g["task_id"] == tid and g["mode"] == "greedy")
        return g["passed"], (g["taxonomy"] or {}).get("class"), (g["taxonomy"] or {}).get("rule")

    for tid in ids:
        assert tax("r1-7b", tid) == (True, None, None), tid                             # the REFERENCE passes everywhere
    assert tax("r2-4b", "dev-contour_area-000") == (False, "convention", "ValueError")
    assert tax("r2-4b", "dev-contour_area-001") == (False, "algorithm", "verifier-value")
    assert tax("r2-4b", "dev-hough_segments-000") == (False, "convention", "verifier-contract")
    assert tax("r2-4b", "shape_houghlinesp") == (False, "convention", "IndexError")
    assert tax("r2-4b", "shape_findnonzero")[:2] == (False, "format")
    crash = next(g for g in grades if g["rung"] == "r2-4b" and g["task_id"] == "dev-contour_area-000" and g["mode"] == "greedy")
    assert (crash["error_gate"], crash["error_type"]) == ("exec-only", "ValueError") and "unpack" in crash["error_msg"]
    api = tax("r3-9b", "dev-contour_area-000")
    assert api[:2] == (False, "api") and api[2].startswith("removed-api")
    assert tax("r3-9b", "dev-contour_area-001") == (False, "lint", "lint:import")
    assert all(tax("r3-9b", t)[0] for t in ids[2:])
    g = next(g for g in grades if g["rung"] == "r3-9b" and g["task_id"] == "dev-contour_area-000")
    assert g["gates"]["exec-only"]["passed"] is False and g["api_findings"][0]["name"] == "cv2.xfeatures2d.SURF_create"
    assert read_json(run.grading) == {"n_scenes": 20, "n_meta": 5, "timeout_s": None}

    assert main(["report", *argv, "--bootstrap", "300", "--audit-k", "2"]) == 0
    s = read_json(run.summary)
    allr = s["groups"]["all"]["rungs"]
    assert allr["r1-7b"]["pass1"] == 1.0 and allr["r1-7b"]["passk"] == 1.0
    assert allr["r2-4b"]["pass1"] == 0.0 and allr["r2-4b"]["passk"] == 0.0
    assert allr["r3-9b"]["pass1"] == pytest.approx(0.6) and allr["r3-9b"]["passk"] == pytest.approx(0.6)
    assert allr["r1-7b"]["se1"] == 0.0 and allr["r3-9b"]["se1"] == pytest.approx(math.sqrt(0.6 * 0.4 / 5))
    assert s["groups"]["families"]["n_tasks"] == 3 and s["groups"]["gotchas"]["n_tasks"] == 2
    assert s["groups"]["5.0"]["n_tasks"] == 3 and s["groups"]["timeless"]["n_tasks"] == 2     # hough + 2 gotchas vs 2 contour
    pairs = {(x["a"], x["b"]): x for x in s["paired"]["all"]}
    assert pairs[("r1-7b", "r2-4b")]["a_only"] == 5 and pairs[("r1-7b", "r2-4b")]["b_only"] == 0
    assert pairs[("r1-7b", "r2-4b")]["p"] == pytest.approx(2 / 32) and pairs[("r1-7b", "r2-4b")]["diff"] == pytest.approx(1.0)
    assert pairs[("r1-7b", "r3-9b")]["a_only"] == 2 and pairs[("r1-7b", "r3-9b")]["p"] == pytest.approx(0.5)
    assert pairs[("r2-4b", "r3-9b")]["b_only"] == 3 and pairs[("r2-4b", "r3-9b")]["p"] == pytest.approx(0.25)
    tx = s["taxonomy"]["r2-4b"]["greedy"]["classes"]
    assert tx == {"format": 1, "api": 0, "lint": 0, "timeout": 0, "convention": 3, "algorithm": 1, "other": 0}
    assert s["taxonomy"]["r3-9b"]["greedy"]["classes"]["api"] == 1 and s["taxonomy"]["r3-9b"]["greedy"]["classes"]["lint"] == 1
    t = s["throughput"]["r1-7b"]
    assert t["decode_tps"]["median"] == pytest.approx(40.0) and t["truncated"] == 0 and t["replies"] == 30
    assert s["counts"]["ungraded"] == 0 and s["counts"]["grader_errors"] == 0
    md = run.report_md.read_text()
    for needle in ("pass@1", "pass@5", "McNemar", "tok/s", "r3-9b", "convention", "held-out"):
        assert needle in md
    assert "## 3. Error taxonomy" in md and "## 5. Throughput" in md
    audit = run.audit_md.read_text()
    assert "## api" in audit and "SURF_create" in audit and "## format" in audit

    # the prediction can be seeded from this run's measured decode rate and answer length
    assert main(["predict", "--from-run", "dry", "--root", root, "--rung", "r1-7b"]) == 0
    out = capsys.readouterr().out
    assert "40.0" in out and "120" in out and "238 tasks" in out


def test_report_for_a_greedy_only_run(tmp_path, monkeypatch):
    Stub(monkeypatch)                                                            # every reply is `x = 1`: no solve()
    run = RunDir.of("greedy", tmp_path)
    run_generation(run, [RUNGS["r1-7b"], RUNGS["r2-4b"]], PROMPT_TASKS, settings(samples=0))
    assert main(["grade", "--run", "greedy", "--root", str(tmp_path), "--workers", "2"]) == 0
    assert main(["report", "--run", "greedy", "--root", str(tmp_path), "--bootstrap", "100"]) == 0
    s = read_json(run.summary)
    assert s["groups"]["all"]["rungs"]["r1-7b"]["pass1"] == 0.0 and s["groups"]["all"]["rungs"]["r1-7b"]["passk"] is None
    assert s["taxonomy"]["r1-7b"]["greedy"]["classes"]["format"] == 2
    assert s["paired"]["all"][0]["p"] == 1.0
    assert "pass@5" in run.report_md.read_text()
