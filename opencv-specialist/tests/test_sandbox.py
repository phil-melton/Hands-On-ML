import time

import numpy as np

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
