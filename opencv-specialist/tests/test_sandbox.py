import textwrap
import time

import numpy as np
import pytest

from sandbox.lint import lint
from sandbox.run import run_solution

GOOD = """\
import cv2
import numpy as np

def solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"gray": gray, "n": 3, "pair": (1.5, "x"), "nested": {"pts": [np.zeros((2, 2), np.float32)]}}
"""


def test_outputs_round_trip_with_types_preserved():
    img = np.zeros((4, 5, 3), np.uint8)
    img[..., 2] = 255
    r = run_solution(GOOD, [{"img": img}])
    assert r.ok, r.summary()
    out = r.outputs[0]
    assert out["gray"].shape == (4, 5) and out["gray"].dtype == np.uint8
    assert out["n"] == 3 and out["pair"] == (1.5, "x")
    assert out["nested"]["pts"][0].dtype == np.float32


def test_batch_runs_every_input_in_one_process():
    code = "def solve(x, k):\n    return {'y': x * k}\n"
    r = run_solution(code, [{"x": np.arange(3), "k": 2}, {"x": np.arange(2), "k": 10}])
    assert r.ok, r.summary()
    assert [o["y"].tolist() for o in r.outputs] == [[0, 2, 4], [0, 10]]


def test_exception_reports_type_and_failing_item():
    r = run_solution("def solve(x):\n    return {'y': 1 / x}\n", [{"x": 1}, {"x": 0}])
    assert r.status == "exception"
    assert r.error_type == "ZeroDivisionError" and r.item == 1
    assert "solution.py" in r.traceback


def test_timeout_kills_an_infinite_loop():
    t0 = time.monotonic()
    r = run_solution("def solve():\n    while True:\n        pass\n", [{}], timeout_s=2)
    assert r.status == "timeout"
    assert time.monotonic() - t0 < 8


def test_lint_flags_gui_network_io_and_swallowed_errors():
    code = """\
import socket
import cv2

def solve(img):
    try:
        cv2.imshow("x", img)
        cv2.imwrite("/tmp/out.png", img)
    except Exception:
        pass
    return {}
"""
    rules = {f.rule for f in lint(code)}
    assert {"import", "gui", "io", "broad-except"} <= rules


# ---------------------------------------------------------------- swallowed-except

def _in_solve(body: str, header: str = "import cv2\n") -> str:
    return header + "\n\ndef solve(img):\n" + textwrap.indent(textwrap.dedent(body), "    ") + "    return {}\n"


def _handler(exc: str, body: str, loop: bool = False) -> str:
    """A solve() whose try/except <exc> runs <body> in the handler (inside a for loop if `loop`: continue/break)."""
    code = f"try:\n    cv2.blur(img, (3, 3))\nexcept {exc}:\n" + textwrap.indent(textwrap.dedent(body), "    ")
    return _in_solve("for _ in range(2):\n" + textwrap.indent(code, "    ") if loop else code)


def _rules(code: str) -> list[str]:
    return [f.rule for f in lint(code)]


@pytest.mark.parametrize("exc, body, loop", [
    pytest.param("cv2.error", "pass\n", False, id="specific type"),
    pytest.param("(ValueError, cv2.error)", "pass\n", False, id="tuple of types"),
    pytest.param("cv2.error as e", "pass\n", False, id="bound name never used"),
    pytest.param("cv2.error", "...\n", False, id="ellipsis"),
    pytest.param("cv2.error", "'ignored'\n", False, id="bare string"),
    pytest.param("cv2.error", "pass\n...\n'x'\n", False, id="several discards"),
    pytest.param("cv2.error", "continue\n", True, id="continue"),
    pytest.param("cv2.error", "break\n", True, id="break"),
])
def test_lint_flags_a_handler_that_only_discards_the_error(exc, body, loop):
    assert _rules(_handler(exc, body, loop)) == ["swallowed-except"]


@pytest.mark.parametrize("exc, body, loop", [
    pytest.param("cv2.error", "out = img\n", False, id="fallback assignment"),
    pytest.param("cv2.error", "return {'img': img}\n", False, id="returned fallback"),
    pytest.param("cv2.error", "raise\n", False, id="re-raise"),
    pytest.param("cv2.error as e", "raise ValueError('bad kernel') from e\n", False, id="raise another"),
    pytest.param("cv2.error as e", "print(e)\n", False, id="logging call"),
    pytest.param("cv2.error", "pass\nout = img\n", False, id="pass then a real statement"),
    pytest.param("cv2.error", "out = None\ncontinue\n", True, id="real statement then continue"),
])
def test_lint_does_not_flag_a_handler_that_does_something_visible(exc, body, loop):
    assert _rules(_handler(exc, body, loop)) == []


@pytest.mark.parametrize("header, expr", [
    pytest.param("import contextlib\n", "contextlib.suppress(cv2.error)", id="contextlib.suppress"),
    pytest.param("import contextlib as cl\n", "cl.suppress(cv2.error)", id="module alias"),
    pytest.param("from contextlib import suppress\n", "suppress(cv2.error)", id="from import"),
    pytest.param("from contextlib import suppress as s\n", "s(ValueError, cv2.error)", id="from import alias"),
    pytest.param("from contextlib import *\n", "suppress(cv2.error)", id="star import"),
])
def test_lint_flags_contextlib_suppress_however_it_is_spelled(header, expr):
    with_form = _in_solve(f"with {expr}:\n    cv2.blur(img, (3, 3))\n", "import cv2\n" + header)
    assert _rules(with_form) == ["swallowed-except"]
    # held in a variable first: still a call of suppress, so still flagged
    held = _in_solve(f"ctx = {expr}\nwith ctx:\n    cv2.blur(img, (3, 3))\n", "import cv2\n" + header)
    assert _rules(held) == ["swallowed-except"]


def test_lint_does_not_flag_other_things_named_suppress():
    local = _in_solve("def suppress(*errors):\n    return errors\n\nsuppress(ValueError)\n")  # not from contextlib
    assert _rules(local) == []
    kwarg = _in_solve("np.set_printoptions(suppress=True)\n", "import numpy as np\n")
    assert _rules(kwarg) == []
    other = _in_solve("with cl.nullcontext():\n    pass\n", "import contextlib as cl\n")  # same module, other manager
    assert _rules(other) == []


def test_lint_reports_both_rules_for_a_broad_handler_that_discards_broad_first():
    assert _rules(_handler("Exception", "pass\n")) == ["broad-except", "swallowed-except"]
    assert _rules(_in_solve("try:\n    cv2.blur(img, (3, 3))\nexcept:\n    pass\n")) == ["broad-except", "swallowed-except"]
    assert lint(_handler("Exception", "pass\n"))[0].rule == "broad-except"  # RunResult.error reads as before
    # the rules are independent: a broad handler that returns a fallback is broad only
    assert _rules(_handler("Exception", "return {'img': img}\n")) == ["broad-except"]


def test_swallowing_a_specific_error_is_stopped_before_it_runs_a_visible_fallback_is_not():
    template = "import cv2\n\ndef solve(img):\n    try:\n        out = cv2.blur(img, (3, 3))\n    except cv2.error:\n" \
               "        HANDLER\n    return {'img': out}\n"
    img = np.zeros((4, 4), np.uint8)
    r = run_solution(template.replace("HANDLER", "pass"), [{"img": img}])
    assert r.status == "lint" and "swallowed-except" in r.error
    r = run_solution(template.replace("HANDLER", "out = img"), [{"img": img}])
    assert r.ok, r.summary()


def test_lint_requires_solve_and_parses():
    assert [f.rule for f in lint("def main():\n    pass\n")] == ["contract"]
    assert [f.rule for f in lint("def solve(:\n")] == ["syntax"]


def test_lint_failure_short_circuits_execution():
    r = run_solution("import os\ndef solve():\n    return {}\n", [{}])
    assert r.status == "lint" and "import os" in r.error


def test_gui_call_blocked_at_runtime_even_without_lint():
    code = "import cv2\nimport numpy as np\n\ndef solve():\n    cv2.imshow('x', np.zeros((4, 4), np.uint8))\n    return {}\n"
    r = run_solution(code, [{}], lint=False)
    assert r.status == "violation" and "imshow" in r.error


def test_violation_is_sticky_even_when_swallowed():
    code = """\
import cv2
import numpy as np

def solve():
    try:
        cv2.imshow("x", np.zeros((2, 2), np.uint8))
    except Exception:
        pass
    return {}
"""
    r = run_solution(code, [{}], lint=False)
    assert r.status == "violation"


def test_network_blocked_at_runtime():
    code = "import socket\n\ndef solve():\n    socket.create_connection(('example.com', 80), timeout=1)\n    return {}\n"
    r = run_solution(code, [{}], lint=False)
    assert r.status == "violation" and "network" in r.error


def test_writes_outside_sandbox_blocked(tmp_path):
    target = tmp_path / "escaped.txt"
    code = f"def solve():\n    open({str(target)!r}, 'w').write('x')\n    return {{}}\n"
    r = run_solution(code, [{}], lint=False)
    assert r.status == "violation"
    assert not target.exists()


def test_writes_inside_sandbox_allowed():
    code = "def solve():\n    open('scratch.txt', 'w').write('x')\n    return {'n': len(open('scratch.txt').read())}\n"
    r = run_solution(code, [{}], lint=False)
    assert r.ok and r.outputs[0]["n"] == 1


def test_memory_limit_raises_memory_error():
    code = "import numpy as np\n\ndef solve():\n    a = np.ones(10**9)\n    return {'s': float(a.sum())}\n"
    r = run_solution(code, [{}], mem_mb=1024)
    assert r.status == "exception" and "MemoryError" in r.error_mro


def test_contract_violations_are_bad_output():
    assert run_solution("def solve():\n    return [1, 2]\n", [{}]).status == "bad_output"
    code = "import cv2\n\ndef solve():\n    return {'kp': cv2.KeyPoint(1.0, 2.0, 3.0)}\n"
    r = run_solution(code, [{}])
    assert r.status == "bad_output" and "KeyPoint" in r.error
