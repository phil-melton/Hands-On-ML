"""The forgetting check (bakeoff/forgetting.py): slice determinism and integrity, the 13-gram overlap filter, the
EvalPlus tests run through the sandbox (reference solutions pass, broken answers fail with the right class), the
paired statistics against independently computed values, and the whole pipeline (generation against a stubbed Ollama
-> grading -> report) on fake model answers. No GPU, no network, no real model.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bakeoff import forgetting as F
from bakeoff import generate
from bakeoff.generate import GenSettings, run_generation
from bakeoff.stats import mcnemar_exact
from bakeoff.store import JsonlWriter, RunDir, read_json, read_jsonl, write_json

REPO = Path(__file__).resolve().parents[1]
ROWS = F.load_slice()
BY_ID = {r["mbpp_id"]: r for r in ROWS}

MINI_TEST = '''
import numpy as np

def is_floats(x) -> bool:
    return isinstance(x, float)

def assertion(out, exp, atol):
    assert out == exp, f"out: {out}, exp: {exp}"

inputs = [[1, 2], [3, 4], [10, 20]]
results = [3, 7, 30]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(add(*inp), exp, 0)
'''
MINI = {"entry_point": "add", "test": MINI_TEST}


# ================================================================ slice determinism and integrity

def test_select_slice_is_deterministic_independent_of_order_and_matches_a_plain_reimplementation():
    ids = list(range(1, 60))
    a = F.select_slice(ids, 20, seed=7)
    assert a == F.select_slice(reversed(ids), 20, seed=7) == F.select_slice(ids + ids[:5], 20, seed=7)
    assert a == sorted(a) and len(a) == 20 and set(a) <= set(ids)
    expect = sorted(sorted(ids, key=lambda i: hashlib.sha256(("7|%d" % i).encode()).hexdigest())[:20])
    assert a == expect
    assert F.select_slice(ids, 20, seed=8) != a                       # the seed matters
    assert F.select_slice(ids, 500, seed=7) == ids                    # n above the pool: everything
    assert F.select_slice([], 5) == []


def test_the_stored_slice_is_what_the_seeded_selection_gives():
    rep = json.loads((REPO / "data/forgetting/build_report.json").read_text())
    pool = rep["slice_ids"] + rep["pool_not_sampled"]
    assert F.select_slice(pool, rep["n_requested"], rep["seed"]) == rep["slice_ids"]
    assert [r["mbpp_id"] for r in sorted(ROWS, key=lambda r: r["mbpp_id"])] == rep["slice_ids"]
    assert rep["seed"] == F.SLICE_SEED and rep["ngram"] == F.NGRAM
    assert rep["dataset"] == f"{F.DATASET}@{F.REVISION}"
    # every problem that left the pool is accounted for: 378 = slice + not sampled + overlap + image words + failed reference
    assert rep["raw"] == (len(rep["slice_ids"]) + len(rep["pool_not_sampled"]) + rep["n_overlap_dropped"]
                          + len(rep["image_word_dropped"]) + len(rep["reference_failed"]))


def test_slice_file_invariants():
    assert 280 <= len(ROWS) <= 320 and len({r["task_id"] for r in ROWS}) == len(ROWS)
    for r in ROWS:
        assert r["task_id"] == f"Mbpp/{r['mbpp_id']}"
        assert F.entry_point_of(r["test"]) == r["entry_point"]
        assert F.defines(r["reference"], r["entry_point"])
        assert r["entry_point"] in r["test_list"][0]                    # the example assert shown in the prompt names it
        assert F.exclusion_reason({"prompt": r["prompt"], "code": r["reference"], "test_list": r["test_list"]}) is None
        assert r["source"] == f"{F.DATASET}@{F.REVISION}" and r["ngram13_shared"] == 0
    assert F.file_sha256() == hashlib.sha256(F.SLICE_PATH.read_bytes()).hexdigest()
    assert F.SLICE_PATH.stat().st_size < 6_000_000                       # Rule 8 allows 50 MB; the brief said a few MB
    tasks = F.slice_tasks(ROWS)
    assert [t.id for t in tasks] == [f"Mbpp/{i}" for i in sorted(BY_ID)]
    assert all(t.kind == "forgetting" and t.prompt.startswith(BY_ID[int(t.id.split("/")[1])]["prompt"]) for t in tasks)
    assert len(F.slice_tasks(ROWS, limit=7)) == 7


def test_image_word_filter_is_narrow():
    rec = lambda text, code="def f(): pass", tests=("assert f() == 1",): {"prompt": text, "code": code, "test_list": list(tests)}
    for text in ("Write a function to blur an image.", "Count the pixels above 128.", "Open the file with cv2.imread.",
                 "Use OpenCV to resize.", "Convert an RGB value to hex.", "A computer vision toy."):
        assert F.exclusion_reason(rec(text)), text
    assert F.exclusion_reason(rec("Write a function to check the colors in the patterns array.")) is None
    assert F.exclusion_reason(rec("Extract values between quotation marks (resolution of a string).")) is None
    assert F.exclusion_reason(rec("Sum a list.", code="import cv2\ndef f(): pass")) == "cv2"   # the reference solution counts too


# ================================================================ 13-gram overlap

def test_words_and_ngrams():
    assert F.words("Hello, World! x_1 = f(2)") == ["hello", "world", "x_1", "f", "2"]
    toks = [str(i) for i in range(20)]
    assert len(F.ngrams(toks, 13)) == 8 and F.ngrams(toks[:12], 13) == set()
    assert F.ngrams(["a"] * 14, 13) == {("a",) * 13}                       # repeated windows collapse


def test_overlap_scan_finds_a_shared_13gram_in_any_row_and_nothing_shorter():
    span = " ".join(f"w{i}" for i in range(13))
    queries = {"hit": f"intro text {span} outro", "twelve": " ".join(f"w{i}" for i in range(12)),
               "case": span.upper(), "miss": "completely different words " * 5}
    replay = [("r0", "nothing to see here " * 10),
              ("r1", "before, " + span.replace(" w5 ", " w5, ") + "! after"),     # punctuation and case do not matter
              ("r2", "x " * 4 + span + " y " * 4 + span)]
    train = iter([("t0", "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11"), ("t1", "short")])      # a stream, not a list
    out = F.overlap_scan(queries, {"replay": replay, "train": train})
    assert out["hit"]["replay"]["rows"] == 2 and out["hit"]["replay"]["shared"] == 1 and out["hit"]["replay"]["row"] == "r1"
    assert out["case"]["replay"]["rows"] == 2                              # lowercased
    assert out["twelve"]["replay"]["shared"] == 0 and out["twelve"]["replay"]["n_grams"] == 0   # fewer than 13 words
    assert out["miss"]["replay"]["shared"] == 0 and out["hit"]["train"]["shared"] == 0
    # 14 shared words = two shared 13-grams, counted once per row
    out = F.overlap_scan({"q": span + " w13"}, {"c": [("a", span + " w13 more")]})
    assert out["q"]["c"] == {"n_grams": 2, "shared": 2, "row": "a", "rows": 1}


def test_corpus_text_skips_the_system_message():
    row = {"messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]}
    assert F.corpus_text(row) == "U\nA"


def test_the_slice_shares_no_13gram_with_the_replay_set_or_the_training_set():
    """The replay README's caveat, kept as a regression test: rebuild data/train.jsonl and this fails if a slice
    problem now overlaps it (then rebuild the slice with data/forgetting/build_slice.py)."""
    files = {"replay": REPO / "data/replay/replay.jsonl", "train": REPO / "data/train.jsonl"}
    if not all(p.is_file() for p in files.values()):
        pytest.skip("replay.jsonl / train.jsonl not present")
    queries = {}
    for r in ROWS:
        queries[(r["mbpp_id"], "text")] = r["prompt"]
        queries[(r["mbpp_id"], "shown")] = F.user_prompt(r["prompt"], r["entry_point"], r["test_list"][0])
        queries[(r["mbpp_id"], "reference")] = r["reference"]
        queries[(r["mbpp_id"], "tests")] = "\n".join(r["test_list"])
    corpora = {n: ((row["id"], F.corpus_text(row)) for row in read_jsonl(p)) for n, p in files.items()}
    hits = {k: v for k, v in F.overlap_scan(queries, corpora).items() if any(c["shared"] for c in v.values())}
    assert not hits, f"{len(hits)} slice problems overlap the training data: {sorted(hits)[:10]}"


# ================================================================ the safety rule: no in-process execution

def test_the_module_never_executes_code_in_this_process():
    tree = ast.parse((REPO / "bakeoff/forgetting.py").read_text())
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"exec", "eval", "compile", "__import__"}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"subprocess", "os", "importlib", "runpy"}
    assert "run_solution(" in (REPO / "bakeoff/forgetting.py").read_text()   # the one execution path


# ================================================================ reply -> code

def fence(code, lang="python"):
    return f"```{lang}\n{code}\n```"


def test_pick_code_takes_the_first_block_when_it_defines_the_entry_point():
    reply = f"Here you go.\n{fence('def add(a, b):\n    return a + b')}\nUsage:\n{fence('print(add(1, 2))')}"
    code, path = F.pick_code(reply, "add")
    assert path == "fence" and code.startswith("def add")


def test_pick_code_looks_further_when_the_first_block_lacks_the_entry_point():
    reply = f"Setup:\n{fence('import math')}\nThe function:\n{fence('def add(a, b):\n    return a + b')}"
    code, path = F.pick_code(reply, "add")
    assert path == "fence-later" and code.startswith("def add")
    code, path = F.pick_code(f"{fence('int main() {}', 'c')}\n{fence('x = 1')}", "add")     # nothing defines it: bake-off rule
    assert path == "fence" and code.strip() == "x = 1"
    code, path = F.pick_code("def add(a, b):\n    return a + b\n", "add")
    assert path == "bare" and code.startswith("def add")
    code, path = F.pick_code("<think>\n\n</think>\n\n" + fence("def add(a, b):\n    return a + b"), "add")
    assert path == "fence"
    code, path = F.pick_code("```python\ndef add(a, b):\n    return a + b", "add")          # reply cut before the closing fence
    assert path == "fence-unclosed" and "return a + b" in code
    assert F.pick_code("", "add") == ("", "empty")


def test_defines():
    assert F.defines("def f(x):\n  return x", "f") and F.defines("class f:\n  pass", "f") and F.defines("f = lambda x: x", "f")
    assert not F.defines("def g(x):\n  return x", "f") and not F.defines("def outer():\n  def f(): pass", "f")
    assert F.defines("def f(:\n", "f")                                            # does not parse: regex fallback


# ================================================================ the entry point and the prompt

def test_entry_point_of_reads_the_final_loop_only():
    assert F.entry_point_of(MINI_TEST) == "add"
    ref = MINI_TEST.replace("assertion(add(*inp), exp, 0)", "assertion(add(*inp), ref_func(*inp), 0)")
    assert F.entry_point_of(ref) == "add"                                        # the candidate, not the reference function
    trick = MINI_TEST.replace("results = [3, 7, 30]", "results = ['assertion(evil(*inp), 1, 0)']")
    assert F.entry_point_of(trick) == "add"
    with pytest.raises(ValueError):
        F.entry_point_of("x = 1\nfor i in range(3):\n    print(i)\n")


def test_user_prompt():
    p = F.user_prompt("  Write a function to add two numbers.\n", "add", "assert add(1, 2) == 3\n")
    assert p == ("Write a function to add two numbers.\n\nThe function must be named `add` and should pass this test:\n"
                 "assert add(1, 2) == 3")


# ================================================================ the EvalPlus tests through the sandbox

def grade(code, rec=MINI, **kw):
    return F.grade_code(code, rec, **kw)


def test_reference_solutions_of_the_slice_pass_through_the_sandbox():
    ids = sorted(BY_ID)
    picks = ids[:6] + [i for i in (88, 462, 630) if i in BY_ID]                 # the plain form, a huge script, the ref_func forms
    for i in picks:
        r = BY_ID[i]
        g = grade(r["reference"], r)
        assert g["passed"] and g["cls"] == "pass" and g["n_inputs"] > 3, (i, g)
        assert g["netns"] in (True, False) and g["lint_rules"] is not None


def test_a_correct_answer_passes_and_the_harness_reports_the_number_of_inputs():
    g = grade("def add(a, b):\n    return a + b\n")
    assert g["passed"] and g["cls"] == "pass" and g["n_inputs"] == 3 and g["status"] == "ok"


@pytest.mark.parametrize("code, cls, error_type", [
    ("def add(a, b):\n    return a + b + 1\n", "wrong", "AssertionError"),                 # wrong value
    ("def add(a, b):\n    return None\n", "wrong", "AssertionError"),
    ("def add(a, b):\n    return (a + b, 0)\n", "wrong", "AssertionError"),                # right number, wrong type
    ("def add(a, b):\n    return a + b if a < 10 else 'x'\n", "wrong", "AssertionError"),  # wrong only on a later input
    ("def add(a, b):\n    raise ValueError('boom')\n", "runtime", "ValueError"),
    ("def add(a):\n    return a\n", "runtime", "TypeError"),                               # wrong signature
    ("def add(a, b):\n    return a + undefined_name\n", "runtime", "NameError"),
    ("def sub(a, b):\n    return a - b\n", "no_entry", "NameError"),                       # entry point missing
    ("def add(a, b:\n    return a + b\n", "syntax", "SyntaxError"),
    ("def add(a, b):\nreturn a + b\n", "syntax", "IndentationError"),
    ("x = 1 / 0\ndef add(a, b):\n    return a + b\n", "runtime", "ZeroDivisionError"),    # a failing top-level statement
    ("def add(a, b):\n    return a + b\nassert add(1, 1) == 3\n", "wrong", "AssertionError"),   # the answer's own example assert fails
])
def test_broken_answers_fail_with_the_right_class(code, cls, error_type):
    g = grade(code)
    assert not g["passed"] and g["cls"] == cls and g["error_type"] == error_type, g


def test_empty_and_whitespace_answers_are_their_own_class():
    for code in ("", "  \n\t\n"):
        g = grade(code)
        assert not g["passed"] and g["cls"] == "empty"


def test_an_infinite_loop_is_a_timeout_not_a_hang():
    g = grade("def add(a, b):\n    while True:\n        pass\n", timeout_s=2.0)
    assert not g["passed"] and g["cls"] == "timeout" and g["status"] == "timeout"


def test_runaway_printing_is_cut_off_instead_of_filling_the_parent():
    g = grade("def add(a, b):\n    while True:\n        print('x' * 1000)\n", timeout_s=20.0)
    assert not g["passed"] and g["cls"] == "runtime" and "output limit" in g["error"]
    g = grade("import sys\nwhile True:\n    sys.stderr.write('x' * 1000)\ndef add(a, b):\n    return a + b\n", timeout_s=20.0)   # at import time
    assert not g["passed"] and "output limit" in g["error"]
    g = grade("def add(a, b):\n    print('hello')\n    return a + b\n")                  # ordinary prints are harmless and discarded
    assert g["passed"]


def test_a_future_import_still_works_with_the_output_cap_placed_after_it():
    code = "from __future__ import annotations\n\ndef add(a: int, b: int) -> int:\n    return a + b\n"
    assert grade(code)["passed"]
    prog = F.build_program(code, "add")
    assert prog.startswith("from __future__ import annotations") and prog.count("def _install_output_cap") == 1
    runaway = "from __future__ import annotations\n\ndef add(a, b):\n    while True:\n        print('x' * 1000)\n"
    g = grade(runaway, timeout_s=20.0)
    assert not g["passed"] and "output limit" in g["error"]                               # the adapter installs the cap before the tests
    assert F.build_program("def add(a, b):\n    return a + b\n", "add").count("def _install_output_cap") == 1


def test_a_memory_hog_is_killed_by_the_sandbox_limits():
    g = grade("def add(a, b):\n    x = bytearray(4 * 1024 ** 3)\n    return a + b\n", mem_mb=512, timeout_s=20.0)
    assert not g["passed"] and g["cls"] in ("runtime", "killed")               # MemoryError inside, or the process was killed


def test_model_code_cannot_rewrite_the_test_it_is_judged_by():
    cheat = ("def assertion(*a, **k):\n    pass\n\ninputs = []\nresults = []\n\ndef is_floats(x):\n    return True\n\n"
             "def add(a, b):\n    return -1\n")
    assert grade(cheat)["cls"] == "wrong"                                       # the script runs in a fresh namespace


def test_an_entry_point_named_solve_is_not_shadowed_by_the_adapter():
    rec = {"entry_point": "solve", "test": MINI_TEST.replace("add(*inp)", "solve(*inp)")}
    assert grade("def solve(a, b):\n    return a + b\n", rec)["passed"]
    assert grade("def solve(a, b):\n    return a * b\n", rec)["cls"] == "wrong"


def test_the_safety_screen_blocks_without_running():
    for code, why in (("import os\ndef add(a, b):\n    return a + b\n", "import os"),
                      ("from subprocess import run\ndef add(a, b):\n    return a + b\n", "import subprocess"),
                      ("def add(a, b):\n    import socket\n    return a + b\n", "import socket"),
                      ("def add(a, b):\n    open('/tmp/x', 'w').write('hi')\n    return a + b\n", "open()"),
                      ("def add(a, b):\n    return eval('a + b')\n", "eval()"),
                      ("def add(a, b):\n    exec('import os')\n    return a + b\n", "exec()"),
                      ("def add(a, b):\n    return __import__('os').getpid() * 0 + a + b\n", "__import__()")):
        g = grade(code)
        assert g["cls"] == "blocked" and g["blocked_by"] == why and "run_s" not in g, (code, g)
    assert grade("import sys\ndef add(a, b):\n    return a + b + 0 * sys.maxsize\n")["passed"]      # sys is allowed
    assert grade("import math, re, collections, heapq, itertools\ndef add(a, b):\n    return a + b\n")["passed"]
    assert F.screen_code("def add(:\n") is None                                   # syntax errors fail in the sandbox, honestly


def test_the_project_lint_is_recorded_but_not_enforced():
    code = "def add(a, b):\n    try:\n        return a + b\n    except Exception:\n        return 0\n"
    g = grade(code)
    assert g["passed"] and "broad-except" in g["lint_rules"] and "contract" not in g["lint_rules"]
    assert grade("def add(a, b):\n    return a + b\n")["lint_rules"] == []


def test_classify_run_maps_every_sandbox_status():
    from sandbox.run import RunResult
    r = lambda status, **kw: RunResult(status=status, **kw)
    assert F.classify_run(r("timeout")) == "timeout" and F.classify_run(r("killed")) == "killed"
    assert F.classify_run(r("violation")) == "violation" and F.classify_run(r("bad_output")) == "bad_output"
    assert F.classify_run(r("exception", error_type="KeyError")) == "runtime"
    assert F.classify_run(r("exception", error_type="NameError", error="entry point f is not defined")) == "no_entry"
    assert F.classify_run(r("exception", error_type="NameError", error="name 'x' is not defined")) == "runtime"
    assert F.classify_run(r("lint")) == "runtime"


def test_a_grader_bug_is_a_grader_error_not_a_model_failure(monkeypatch):
    monkeypatch.setattr(F, "run_solution", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("harness broke")))
    g = grade("def add(a, b):\n    return a + b\n")
    assert g["cls"] == "grader_error" and not g["passed"] and "harness broke" in g["grader_error"]


# ================================================================ statistics against independently computed values

def test_wilson_interval():
    lo, hi = F.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.236593090512564, 0.7634069094874361))
    lo, hi = F.wilson(80, 100)
    assert (lo, hi) == pytest.approx((0.7111708344068411, 0.8666330666689676))
    assert F.wilson(0, 20)[0] == 0.0 and F.wilson(0, 20)[1] == pytest.approx(0.16112515805281938)
    assert F.wilson(20, 20)[1] == 1.0
    assert math.isnan(F.wilson(0, 0)[0])


@pytest.mark.parametrize("table, lo, hi", [        # (both, A only, B only, neither): Newcombe method 10, derived step by step on paper
    ((70, 10, 5, 15), -0.02863760734505308, 0.12934890546374833),
    ((200, 8, 7, 85), -0.02306349982330052, 0.029762213740110833),
    ((40, 3, 3, 4), -0.11237245507648207, 0.11237245507648207),
])
def test_newcombe_paired_interval(table, lo, hi):
    assert F.newcombe_paired(*table) == pytest.approx((lo, hi), abs=1e-12)


def test_newcombe_properties():
    lo, hi = F.newcombe_paired(70, 10, 5, 15)
    assert lo < (10 - 5) / 100 < hi
    nlo, nhi = F.newcombe_paired(70, 5, 10, 15)                                   # swapping the arms negates the interval
    assert (nlo, nhi) == pytest.approx((-hi, -lo))
    lo0, hi0 = F.newcombe_paired(30, 0, 0, 70)                                      # identical outputs: still uncertain about 0
    assert lo0 < 0 < hi0 and lo0 == pytest.approx(-hi0)
    lo1, hi1 = F.newcombe_paired(100, 0, 0, 0)                                      # everything passes in both arms
    assert lo1 < 0 < hi1 and lo1 == pytest.approx(-hi1) and hi1 < 0.05
    lo, hi = F.newcombe_paired(0, 30, 0, 0)
    assert -1.0 <= lo <= hi <= 1.0
    assert math.isnan(F.newcombe_paired(0, 0, 0, 0)[0])
    # a larger sample with the same rates gives a narrower interval
    w1 = F.newcombe_paired(70, 10, 5, 15); w2 = F.newcombe_paired(700, 100, 50, 150)
    assert (w2[1] - w2[0]) < (w1[1] - w1[0])


def test_paired_report_counts_difference_and_mcnemar():
    a = [1] * 70 + [1] * 10 + [0] * 5 + [0] * 15
    b = [1] * 70 + [0] * 10 + [1] * 5 + [0] * 15
    p = F.paired_report(a, b)
    assert (p["both"], p["a_only"], p["b_only"], p["neither"], p["n"]) == (70, 10, 5, 15, 100)
    assert p["drop"] == pytest.approx(0.05) and p["p_mcnemar"] == pytest.approx(0.3017578125) == pytest.approx(mcnemar_exact(10, 5))
    assert (p["ci_lo"], p["ci_hi"]) == pytest.approx((-0.02863760734505308, 0.12934890546374833))
    assert p["se"] == pytest.approx(math.sqrt(0.1475 / 99))                  # sample SD of the paired differences / sqrt(n)


# ================================================================ the report on a fake pair of result files

def fake_run(tmp_path, table, *, name="fake", warm=(300, 300)):
    """A run folder with arms 'base' and 'ft' on sum(table) problems: table = (both, base only, ft only, neither)."""
    run = RunDir.of(name, tmp_path)
    both, a_only, b_only, neither = table
    outcome = [(1, 1)] * both + [(1, 0)] * a_only + [(0, 1)] * b_only + [(0, 0)] * neither
    with JsonlWriter(run.generations) as gw, JsonlWriter(run.grades) as rw:
        for i, (pa, pb) in enumerate(outcome):
            tid = f"Mbpp/{i + 1}"
            for arm, passed, tok in (("base", pa, 400), ("ft", pb, 150)):
                gw.write({"rung": arm, "task_id": tid, "mode": "greedy", "sample": 0, "eval_count": tok, "eval_duration": int(tok / 50 * 1e9),
                          "wall_s": tok / 50 + 0.5, "truncated": False, "extract": "fence", "prompt_eval_count": 60})
                rw.write({"rung": arm, "task_id": tid, "mode": "greedy", "sample": 0, "passed": bool(passed), "cls": "pass" if passed else "wrong",
                          "code_path": "fence", "lint_rules": [], "run_s": 0.2, "error_type": None if passed else "AssertionError"})
    with JsonlWriter(run.warmups) as w:
        for arm, n in zip(("base", "ft"), warm):
            w.write({"rung": arm, "prompt_eval_count": n})
    write_json(run.config, {"rungs": {"base": {"model": "base-qwen3.5-9b", "system": F.SYSTEM_GENERAL}, "ft": {"model": "ft-tag", "system": F.SYSTEM_GENERAL}},
                            "decoding": {"greedy": {"temperature": 0.0}}, "ollama_version": "0.0-test", "invocations": []})
    write_json(run.grading, {"slice_sha256": "abc"})
    return run


def test_report_numbers_on_a_fake_pair_of_result_files(tmp_path):
    run = fake_run(tmp_path, (70, 10, 5, 15))
    s = F.summarize(run, "base", "ft")
    assert s["n"] == 100 and s["arms"]["base"]["passed"] == 80 and s["arms"]["ft"]["passed"] == 75
    assert s["arms"]["base"]["pass1"] == pytest.approx(0.80)
    assert (s["arms"]["base"]["ci_lo"], s["arms"]["base"]["ci_hi"]) == pytest.approx((0.7111708344068411, 0.8666330666689676))
    assert s["arms"]["base"]["se"] == pytest.approx(math.sqrt(0.8 * 0.2 / 100))
    P = s["paired"]
    assert P["drop"] == pytest.approx(0.05) and P["p_mcnemar"] == pytest.approx(0.3017578125)
    assert (P["ci_lo"], P["ci_hi"]) == pytest.approx((-0.02863760734505308, 0.12934890546374833))
    assert s["arms"]["base"]["tokens"]["mean"] == 400 and s["arms"]["ft"]["gen_wall_s"]["mean"] == pytest.approx(3.5)
    assert s["arms"]["base"]["decode_tps"]["median"] == pytest.approx(50.0)
    assert len(s["discordant"]) == 15 and {d["passed"] for d in s["discordant"]} == {"base", "ft"}
    text = F.render_report(s)
    assert "**drop = pass@1(base) - pass@1(ft) = +5.0 points**" in text
    assert "[-2.9, +12.9]" in text and "p = 0.302" in text
    assert "criterion 3 on the point estimate (drop <= 3 points): **NOT MET**" in text
    assert "non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **no**" in text
    assert "(equal: yes)" in text


def test_report_verdict_when_the_drop_is_small_and_the_interval_is_tight(tmp_path):
    s = F.summarize(fake_run(tmp_path, (200, 8, 7, 85), name="tight"), "base", "ft")     # 300 problems
    assert s["n"] == 300 and s["paired"]["drop"] == pytest.approx(1 / 300) and s["paired"]["ci_hi"] < 0.03
    text = F.render_report(s)
    assert "criterion 3 on the point estimate (drop <= 3 points): **MET**" in text
    assert "non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **yes**" in text


def test_report_verdict_point_estimate_met_but_interval_too_wide(tmp_path):
    s = F.summarize(fake_run(tmp_path, (40, 3, 3, 4), name="wide"), "base", "ft")          # drop 0, CI +-11 points
    text = F.render_report(s)
    assert "(drop <= 3 points): **MET**" in text and "**no** (the point estimate passes but the interval is too wide" in text


def test_report_only_pairs_problems_present_in_both_arms(tmp_path):
    run = fake_run(tmp_path, (6, 2, 1, 1), name="partial")
    # drop the last two problems of the ft arm from both files, as an interrupted generation would leave them
    gens = [g for g in read_jsonl(run.generations) if not (g["rung"] == "ft" and g["task_id"] in ("Mbpp/9", "Mbpp/10"))]
    run.generations.write_text("".join(json.dumps(g) + "\n" for g in gens))
    s = F.summarize(run, "base", "ft")
    assert s["n"] == 8 and s["arms"]["base"]["n"] == 8 == s["arms"]["ft"]["n"]


def test_report_flags_a_system_prompt_that_was_not_replaced(tmp_path):
    run = fake_run(tmp_path, (7, 2, 1, 0), name="sys", warm=(60, 101))
    text = F.render_report(F.summarize(run, "base", "ft"))
    assert "differ: the Modelfile SYSTEM may have been prepended" in text


def test_report_cli_writes_report_and_summary(tmp_path, capsys):
    fake_run(tmp_path, (70, 10, 5, 15), name="cli")
    assert F.main(["report", "--run", "cli", "--root", str(tmp_path)]) == 0
    run = RunDir.of("cli", tmp_path)
    assert run.report_md.is_file() and read_json(run.summary)["paired"]["a_only"] == 10
    out = capsys.readouterr().out
    assert "drop +5.0 pts" in out and "McNemar p 0.302" in out
    with pytest.raises(SystemExit):
        F.main(["report", "--run", "cli", "--root", str(tmp_path), "--a", "base", "--b", "nope"])


# ================================================================ generation against a stubbed Ollama

class Stub:
    """Replaces generate.chat: records every request body and answers like Ollama."""

    def __init__(self, monkeypatch, reply):
        self.calls: list[dict] = []
        self.reply = reply
        monkeypatch.setattr(generate, "chat", self)
        monkeypatch.setattr(generate, "_info", lambda base, path, payload, timeout=30.0: {"version": "stub"} if path == "/api/version" else {})
        monkeypatch.setattr(generate.time, "sleep", lambda s: None)

    def __call__(self, base_url, body, timeout):
        self.calls.append(body)
        if not body["messages"]:
            return {"model": body["model"], "done": True, "done_reason": "unload" if body["keep_alive"] == 0 else "load", "load_duration": 7}
        return {"message": {"role": "assistant", "content": self.reply(body)}, "done": True, "done_reason": "stop", "eval_count": 100,
                "eval_duration": 2_000_000_000, "prompt_eval_count": 90, "prompt_eval_duration": 500_000_000, "load_duration": 0,
                "total_duration": 3_000_000_000}

    def generations(self):
        return [c for c in self.calls if c["messages"] and c["options"]["num_predict"] != generate.WARMUP_NUM_PREDICT]


def test_generation_sends_one_neutral_prompt_to_both_arms_with_the_frozen_decoding(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, lambda body: "```python\nx = 1\n```")
    arms = F.parse_arms(["base=base-qwen3.5-9b", "ft=my-finetune"])
    tasks = F.slice_tasks(ROWS, limit=3)
    run = RunDir.of("gen", tmp_path)
    written = run_generation(run, arms, tasks, GenSettings(base_url="http://stub", samples=0, retries=0, backoff_s=0.0),
                             system=F.SYSTEM_GENERAL, invocation={"slice_sha256": "s"})
    assert written == {"base": 3, "ft": 3}
    by_model = {a.model: [c for c in stub.generations() if c["model"] == a.model] for a in arms}
    assert {len(v) for v in by_model.values()} == {3}
    for model, bodies in by_model.items():
        for body, task in zip(bodies, tasks):
            assert body["messages"] == [{"role": "system", "content": F.SYSTEM_GENERAL}, {"role": "user", "content": task.prompt}]
            assert body["think"] is False and body["stream"] is False
            o = body["options"]
            assert (o["temperature"], o["repeat_penalty"], o["num_predict"], o["num_ctx"]) == (0.0, 1.0, 1024, 4096)
    # identical requests for the two arms apart from the model name and the (irrelevant at T=0) seed
    strip = lambda b: {k: (v if k != "options" else {x: y for x, y in v.items() if x != "seed"}) for k, v in b.items() if k != "model"}
    assert [strip(b) for b in by_model["base-qwen3.5-9b"]] == [strip(b) for b in by_model["my-finetune"]]
    assert "opencv" not in json.dumps(stub.generations()).lower()                 # the OpenCV system prompt is gone
    cfg = read_json(run.config)
    assert cfg["rungs"]["ft"]["system"] == F.SYSTEM_GENERAL == cfg["rungs"]["base"]["system"] and cfg["rungs"]["ft"]["model"] == "my-finetune"
    rows = read_jsonl(run.generations)
    assert len(rows) == 6 and {r["rung"] for r in rows} == {"base", "ft"} and all(r["mode"] == "greedy" for r in rows)
    # a resume asks for nothing, and refuses another slice file
    n_calls = len(stub.calls)
    assert run_generation(run, arms, tasks, GenSettings(base_url="http://stub", samples=0), system=F.SYSTEM_GENERAL) == {"base": 0, "ft": 0}
    assert len(stub.generations()) == 6 and len(stub.calls) >= n_calls
    F.check_slice(run, "s")
    with pytest.raises(SystemExit):
        F.check_slice(run, "other")


def test_generation_default_system_prompt_is_unchanged_without_the_new_argument(tmp_path, monkeypatch):
    from bakeoff.rungs import RUNGS
    from train.config import SYSTEM
    stub = Stub(monkeypatch, lambda body: "```python\nx = 1\n```")
    run_generation(RunDir.of("dflt", tmp_path), [RUNGS["r1-7b"]], F.slice_tasks(ROWS, limit=1),
                   GenSettings(base_url="http://stub", samples=0, retries=0, backoff_s=0.0))
    assert stub.generations()[0]["messages"][0] == {"role": "system", "content": SYSTEM}


def test_resolve_think(monkeypatch):
    caps = {"has-think": ["completion", "thinking"], "no-think": ["completion"], "unknown": []}
    monkeypatch.setattr(F, "_info", lambda base, path, payload, timeout=30.0: {"capabilities": caps[payload["model"]]} if payload["model"] in caps else {})
    arms = F.parse_arms(["a=has-think", "b=no-think", "c=unknown"])
    assert [r.think for r in F.resolve_think(arms, "http://x")] == [False, None, False]      # unknown capabilities: the protocol default
    assert [r.think for r in F.resolve_think(arms, "http://x", "false")] == [False, False, False]
    assert [r.think for r in F.resolve_think(arms, "http://x", "omit")] == [None, None, None]
    assert [r.model for r in F.resolve_think(arms, "http://x")] == ["has-think", "no-think", "unknown"]


def test_parse_arms():
    arms = F.parse_arms(["base=a:b", "ft=c"])
    assert [(a.name, a.model, a.think) for a in arms] == [("base", "a:b", False), ("ft", "c", False)]
    assert F.parse_arms(["tag-only"])[0].name == "tag-only"
    with pytest.raises(SystemExit):
        F.parse_arms(["x=a", "x=b"])
    with pytest.raises(SystemExit):
        F.parse_arms(["=a"])


# ================================================================ generation -> grading -> report, end to end

def test_end_to_end_with_fake_model_answers_resume_and_cache(tmp_path):
    picks = sorted(BY_ID)[:5]
    rows = [BY_ID[i] for i in picks]
    run = RunDir.of("e2e", tmp_path)
    with JsonlWriter(run.generations) as w:
        for r in rows:
            e = r["entry_point"]
            good = f"Sure! Here is the function.\n\n```python\n{r['reference']}```\n\nIt works because it is correct."
            bad = f"```python\ndef {e}(*a, **k):\n    return None\n```"
            helper_first = f"```python\nimport math\n```\n\n```python\n{r['reference']}```"
            # base: always right; ft: wrong on the first two problems, right via the second block on the third, right otherwise
            ft = bad if r is rows[0] or r is rows[1] else (helper_first if r is rows[2] else good)
            for arm, reply in (("base", good), ("ft", ft)):
                w.write({"rung": arm, "task_id": r["task_id"], "mode": "greedy", "sample": 0, "reply": reply, "eval_count": 120,
                         "eval_duration": 2_000_000_000, "wall_s": 2.6, "truncated": False, "extract": "fence", "prompt_eval_count": 80})
    gs = F.grading_settings(30.0, F.MEM_MB, "sha")
    counts = F.grade_run(run, ROWS, gs, workers=2, log=lambda m: None)
    # base and ft end up with the same code on the last three problems (the second block is the same text): graded once each
    assert counts == {"graded": 10, "executed": 7, "cached": 3, "grader_errors": 0}
    grades = {(g["rung"], g["task_id"]): g for g in read_jsonl(run.grades)}
    assert all(grades[("base", r["task_id"])]["passed"] for r in rows)
    assert [grades[("ft", r["task_id"])]["passed"] for r in rows] == [False, False, True, True, True]
    assert grades[("ft", rows[2]["task_id"])]["code_path"] == "fence-later"
    assert grades[("ft", rows[0]["task_id"])]["cls"] in ("wrong", "runtime")
    assert sum(g["cached"] for g in grades.values()) == 3
    # resume: nothing left to do, nothing rewritten
    before = run.grades.read_bytes()
    assert F.grade_run(run, ROWS, gs, workers=2, log=lambda m: None)["graded"] == 0 and run.grades.read_bytes() == before
    # another grader configuration on the same grades is refused
    with pytest.raises(SystemExit):
        F.grade_run(run, ROWS, F.grading_settings(5.0, F.MEM_MB, "sha"), workers=2, log=lambda m: None)
    s = F.summarize(run, "base", "ft")
    assert s["n"] == 5 and s["arms"]["base"]["passed"] == 5 and s["arms"]["ft"]["passed"] == 3
    assert s["paired"]["a_only"] == 2 and s["paired"]["b_only"] == 0 and s["paired"]["drop"] == pytest.approx(0.4)
    assert s["arms"]["ft"]["code_path"].get("fence-later") == 1
    assert "criterion 3 on the point estimate (drop <= 3 points): **NOT MET**" in F.render_report(s)


def test_a_torn_last_generation_line_is_survived_by_grading(tmp_path):
    r = BY_ID[sorted(BY_ID)[0]]
    run = RunDir.of("torn", tmp_path)
    good = {"rung": "base", "task_id": r["task_id"], "mode": "greedy", "sample": 0, "reply": f"```python\n{r['reference']}```"}
    run.generations.parent.mkdir(parents=True, exist_ok=True)
    run.generations.write_text(json.dumps(good) + '\n{"rung": "base", "task_id": "Mbpp/9')
    c = F.grade_run(run, ROWS, F.grading_settings(30.0, F.MEM_MB, "sha"), workers=1, log=lambda m: None)
    assert c["graded"] == 1 and read_jsonl(run.grades)[0]["passed"] is True


# ================================================================ the whole command line against a fake Ollama on loopback

class LoopbackOllama:
    """Answers /api/chat, /api/show, /api/version and /api/ps over HTTP; the reply comes from reply(model, user_prompt)."""

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
                    text = reply(body["model"], body["messages"][-1]["content"])
                    return self._send({"message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop", "eval_count": 120,
                                       "eval_duration": 3_000_000_000, "prompt_eval_count": 400, "prompt_eval_duration": 400_000_000,
                                       "load_duration": 0, "total_duration": 4_000_000_000})
                if self.path == "/api/show":
                    caps = ["completion", "thinking"] if body["model"] == "base-m" else ["completion"]
                    return self._send({"capabilities": caps, "parameters": "stop \"<|im_end|>\"", "details": {"family": "stub"}})
                self._send({"error": "not found"}, 404)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._send({"version": "0.0.0-fake"})
                if self.path == "/api/ps":
                    return self._send({"models": []})
                self._send({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_cli_generate_grade_report_over_http_and_resume(tmp_path, capsys):
    tasks = F.slice_tasks(ROWS, limit=6)
    row_of = {t.prompt: BY_ID[int(t.id.split("/")[1])] for t in tasks}
    first = row_of[tasks[0].prompt]

    def reply(model, prompt):
        r = row_of[prompt]
        if model == "base-m":                                                  # chatty, like the untrained base
            return f"Sure, here is the function.\n\n```python\n{r['reference']}```\n\nIt works because it is correct."
        if r is first:                                                          # the fine-tuned model "forgot" one problem
            return f"```python\ndef {r['entry_point']}(*a, **k):\n    return None\n```"
        return f"```python\n{r['reference']}```"                                # terse, like the fine-tuned model

    fake = LoopbackOllama(reply)
    try:
        argv = ["--run", "dry", "--root", str(tmp_path)]
        gen = ["generate", *argv, "--arm", "base=base-m", "--arm", "ft=ft-m", "--limit", "6", "--ollama", fake.url]
        assert F.main(gen) == 0
        sent = len(fake.requests)
        assert F.main(gen) == 0 and len(fake.requests) == sent                  # resumed: nothing asked again
    finally:
        fake.close()
    chats = [r for r in fake.requests if r["messages"] and r["options"]["num_predict"] != generate.WARMUP_NUM_PREDICT]
    assert len(chats) == 12 and all(r["messages"][0] == {"role": "system", "content": F.SYSTEM_GENERAL} for r in chats)
    assert fake.requests[0]["messages"] == [] and fake.requests[-1]["keep_alive"] == 0
    # think:false only for the model that lists the thinking capability (as train/hydro_eval.py does)
    assert all(r.get("think") is False for r in chats if r["model"] == "base-m") and all("think" not in r for r in chats if r["model"] == "ft-m")

    assert F.main(["grade", *argv, "--workers", "2"]) == 0
    run = RunDir.of("dry", tmp_path)
    assert len(read_jsonl(run.grades)) == 12 and read_json(run.grading)["slice_sha256"] == F.file_sha256()
    assert F.main(["report", *argv]) == 0
    s = read_json(run.summary)
    assert s["n"] == 6 and s["arms"]["base"]["passed"] == 6 and s["arms"]["ft"]["passed"] == 5
    assert s["paired"]["a_only"] == 1 and s["paired"]["drop"] == pytest.approx(1 / 6)
    assert s["arms"]["base"]["tokens"]["mean"] == 120 and s["decoding"]["temperature"] == 0.0
    assert s["arms"]["ft"]["model"] == "ft-m" and s["warmup_prompt_tokens"] == {"base": 400, "ft": 400}
    assert (s["arms"]["base"]["think"], s["arms"]["ft"]["think"]) == (False, None)
    assert s["arms"]["base"]["capabilities"] == ["completion", "thinking"]
    out = capsys.readouterr().out
    assert "base" in out and "drop +16.7 pts" in out
    md = run.report_md.read_text()
    assert "(equal: yes)" in md and "`think` flag sent: base False, ft None" in md


# ================================================================ command line

def test_cli_info_validate_and_predict(capsys):
    assert F.main(["info"]) == 0
    assert f"{len(ROWS)} problems" in capsys.readouterr().out
    assert F.main(["predict", "--tok-s", "50", "--tokens", "base=400", "--tokens", "ft=150"]) == 0
    out = capsys.readouterr().out
    assert "base" in out and "ft" in out and "total" in out
    n = len(ROWS)
    assert f"{n} tasks" in out
    with pytest.raises(SystemExit):
        F.main(["predict", "--tok-s", "50", "--tokens", "base=400"])           # no length for the ft arm


def test_validate_flags_a_reference_that_does_not_pass(capsys, tmp_path):
    r = dict(BY_ID[sorted(BY_ID)[0]])
    ok = F.validate_reference([r], workers=1)
    assert ok[0]["passed"]
    r["reference"] = r["reference"].replace("return", "return None and", 1) if "return" in r["reference"] else "def f(): pass"
    bad = F.validate_reference([r], workers=1)
    assert not bad[0]["passed"]
    p = tmp_path / "slice.jsonl"
    p.write_text(json.dumps(r) + "\n")
    assert F.main(["validate", "--slice", str(p), "--workers", "1"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_load_slice_validates_the_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        F.load_slice(tmp_path / "missing.jsonl")
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"task_id": "Mbpp/1"}) + "\n")
    with pytest.raises(ValueError):
        F.load_slice(p)
    r = dict(ROWS[0])
    p.write_text(json.dumps(r) + "\n" + json.dumps(r) + "\n")
    with pytest.raises(ValueError):
        F.load_slice(p)
