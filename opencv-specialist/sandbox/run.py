"""Run a generated ``solve()`` in a separate, limited Python process.

    from sandbox.run import run_solution
    res = run_solution(code, [{"img": img}], timeout_s=10)
    if res.ok:
        areas = res.outputs[0]["areas"]

What it is: a seatbelt.
  * a separate process, so crashes and hangs cannot take the caller down
  * wall-clock timeout with a process-group kill; address-space and CPU rlimits
  * headless OpenCV with GUI calls stubbed to raise
  * best-effort Python-level guards on sockets and on writes outside the sandbox dir
  * a static lint gate (sandbox/lint.py) before anything runs
  * violations are sticky: a guard that fires is reported even if the code swallows it

What it is not: a vault. Python-level guards can be bypassed by determined code
(the lint bans the obvious escape hatches such as ctypes, importlib and resource).
For the data foundry, where thousands of untrusted snippets run, put this same
runner inside a container with ``--network none`` and a read-only root filesystem.

Inputs are a list of keyword-argument dicts (a batch, run in one process); values are
numpy arrays or JSON scalars. ``solve`` must return a dict of arrays, numbers,
strings, or lists/tuples/dicts of those.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sandbox.lint import Finding, lint as lint_code

_HARNESS = r'''
import builtins, io, json, os, resource, socket, sys, traceback
TMP = os.path.realpath(os.getcwd())
MEM_MB, CPU_S = int(sys.argv[1]), int(sys.argv[2])
VIOLATIONS = []


def _limit(kind, value):
    soft, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value))


_limit(resource.RLIMIT_AS, MEM_MB * 1024 * 1024)
_limit(resource.RLIMIT_CPU, CPU_S)
_limit(resource.RLIMIT_FSIZE, 256 * 1024 * 1024)


class SandboxViolation(Exception):
    pass


def _violate(msg):
    VIOLATIONS.append(msg)
    raise SandboxViolation(msg)


def _inside(path):
    p = os.path.realpath(os.fspath(path))
    return p == TMP or p.startswith(TMP + os.sep)


_real_open = builtins.open


def _guarded_open(file, mode="r", *a, **k):
    if not isinstance(file, int) and any(c in mode for c in "wax+") and not _inside(file):
        _violate(f"write outside the sandbox dir: {file!r}")
    return _real_open(file, mode, *a, **k)


builtins.open = io.open = _guarded_open

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_real_os_open = os.open


def _guarded_os_open(path, flags, *a, **k):
    if flags & _WRITE_FLAGS and not _inside(path):
        _violate(f"write outside the sandbox dir: {path!r}")
    return _real_os_open(path, flags, *a, **k)


os.open = _guarded_os_open


def _guard_paths(fn, name):
    def wrapper(*args, **kwargs):
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)) and not _inside(a):
                _violate(f"os.{name} outside the sandbox dir: {a!r}")
        return fn(*args, **kwargs)
    return wrapper


for _n in ("remove", "unlink", "rmdir", "mkdir", "rename", "replace", "symlink", "link",
           "chmod", "truncate", "utime"):
    if hasattr(os, _n):
        setattr(os, _n, _guard_paths(getattr(os, _n), _n))


def _no_network(*a, **k):
    _violate("network access is disabled in the sandbox")


socket.getaddrinfo = socket.gethostbyname = socket.create_connection = _no_network
for _n in ("connect", "connect_ex", "sendto", "bind"):
    setattr(socket.socket, _n, _no_network)

import numpy as np
import cv2

cv2.setNumThreads(1)


def _gui(name):
    def blocked(*a, **k):
        _violate(f"cv2.{name}: GUI calls are not available in the worker")
    return blocked


for _n in ("imshow", "waitKey", "waitKeyEx", "pollKey", "namedWindow", "destroyWindow",
           "destroyAllWindows", "setMouseCallback", "createTrackbar", "selectROI", "selectROIs"):
    if hasattr(cv2, _n):
        setattr(cv2, _n, _gui(_n))

_real_imwrite = cv2.imwrite


def _guarded_imwrite(filename, *a, **k):
    if not _inside(filename):
        _violate(f"cv2.imwrite outside the sandbox dir: {filename!r}")
    return _real_imwrite(filename, *a, **k)


cv2.imwrite = _guarded_imwrite


def _write_result(**payload):
    if VIOLATIONS:
        payload = {"status": "violation", "error_type": "SandboxViolation", "error": VIOLATIONS[0],
                   "item": payload.get("item")}
    with _real_open(os.path.join(TMP, "result.json"), "w") as f:
        json.dump(payload, f)


def _pack(x, store, where):
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            raise TypeError(f"{where}: object arrays are not allowed; return a list of arrays")
        key = f"a{len(store)}"
        store[key] = x
        return {"__nd__": key}
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return {"__seq__": type(x).__name__, "items": [_pack(v, store, f"{where}[{i}]") for i, v in enumerate(x)]}
    if isinstance(x, dict):
        return {"__map__": {str(k): _pack(v, store, f"{where}.{k}") for k, v in x.items()}}
    raise TypeError(f"{where}: unsupported output type {type(x).__name__}; convert it to arrays/numbers")


def _error(e, item):
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if f.filename.endswith("solution.py")] or tb[-1:]
    _write_result(status="exception", error_type=type(e).__name__, error=str(e).strip()[:2000],
                  error_mro=[c.__name__ for c in type(e).__mro__],
                  traceback="".join(traceback.format_list(frames)), item=item)


def main():
    spec = json.load(_real_open(os.path.join(TMP, "inputs.json")))
    arrays = np.load(os.path.join(TMP, "inputs.npz"), allow_pickle=False)
    items = [{name: (arrays[v["npz"]] if "npz" in v else v["json"]) for name, v in item.items()}
             for item in spec]
    import importlib.util
    s = importlib.util.spec_from_file_location("solution", os.path.join(TMP, "solution.py"))
    mod = importlib.util.module_from_spec(s)
    try:
        s.loader.exec_module(mod)
    except BaseException as e:
        return _error(e, None)
    solve = getattr(mod, "solve", None)
    if not callable(solve):
        return _write_result(status="bad_output", error_type="ContractError", error="no callable solve()", item=None)
    packed, store = [], {}
    for k, kwargs in enumerate(items):
        try:
            out = solve(**kwargs)
        except BaseException as e:
            return _error(e, k)
        if VIOLATIONS:
            return _write_result(status="violation", item=k)
        if not isinstance(out, dict):
            return _write_result(status="bad_output", error_type="ContractError",
                                 error=f"solve() must return a dict, got {type(out).__name__}", item=k)
        try:
            packed.append(_pack(out, store, "output"))
        except TypeError as e:
            return _write_result(status="bad_output", error_type="ContractError", error=str(e), item=k)
    np.savez(os.path.join(TMP, "outputs.npz"), **store)
    with _real_open(os.path.join(TMP, "outputs.json"), "w") as f:
        json.dump(packed, f)
    _write_result(status="ok", n=len(packed))


main()
'''


@dataclass
class RunResult:
    status: str  # ok | lint | exception | violation | bad_output | timeout | killed
    outputs: list[dict] | None = None
    error_type: str | None = None
    error: str | None = None
    error_mro: list[str] = field(default_factory=list)
    traceback: str | None = None
    item: int | None = None  # index of the input set that failed
    lint: list[Finding] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        if self.ok:
            return f"ok ({self.seconds:.2f}s)"
        if self.status == "lint":
            return "lint: " + "; ".join(map(str, self.lint))
        where = f" on input {self.item}" if self.item is not None else ""
        return f"{self.status}{where}: {self.error_type or ''} {self.error or ''}".strip()


def _write_inputs(tmp: Path, inputs: list[dict[str, Any]]) -> None:
    spec, arrays = [], {}
    for k, item in enumerate(inputs):
        entry = {}
        for name, value in item.items():
            if isinstance(value, np.ndarray):
                key = f"in{k}__{name}"
                arrays[key] = value
                entry[name] = {"npz": key}
            else:
                entry[name] = {"json": value.item() if isinstance(value, np.generic) else value}
        spec.append(entry)
    np.savez(tmp / "inputs.npz", **arrays)
    (tmp / "inputs.json").write_text(json.dumps(spec))


def _unpack(x, arrays):
    if isinstance(x, dict):
        if "__nd__" in x:
            return arrays[x["__nd__"]]
        if "__seq__" in x:
            items = [_unpack(v, arrays) for v in x["items"]]
            return tuple(items) if x["__seq__"] == "tuple" else items
        if "__map__" in x:
            return {k: _unpack(v, arrays) for k, v in x["__map__"].items()}
    return x


def run_solution(code: str, inputs: list[dict[str, Any]], *, timeout_s: float = 10.0,
                 mem_mb: int = 2048, lint: bool = True, python: str | None = None,
                 keep_dir: bool = False) -> RunResult:
    """Lint, then run ``solve(**item)`` for every item in ``inputs`` inside one child process."""
    if lint:
        findings = lint_code(code)
        if findings:
            return RunResult(status="lint", lint=findings, error=str(findings[0]))
    tmp = Path(tempfile.mkdtemp(prefix="cvsandbox-")).resolve()
    t0 = time.monotonic()
    try:
        (tmp / "solution.py").write_text(code)
        (tmp / "_harness.py").write_text(_HARNESS)
        _write_inputs(tmp, inputs)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "TMPDIR": str(tmp), "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENCV_LOG_LEVEL": "ERROR", "QT_QPA_PLATFORM": "offscreen"}
        proc = subprocess.Popen(
            [python or sys.executable, "-I", "-B", "_harness.py", str(mem_mb), str(int(timeout_s) + 1)],
            cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            return RunResult(status="timeout", error=f"no result after {timeout_s:g}s; process group killed",
                             stdout=out.decode(errors="replace")[-4000:], stderr=err.decode(errors="replace")[-4000:],
                             seconds=time.monotonic() - t0)
        stdout, stderr = out.decode(errors="replace")[-4000:], err.decode(errors="replace")[-4000:]
        result_file = tmp / "result.json"
        if not result_file.exists():
            return RunResult(status="killed", error=f"child exited with code {proc.returncode} and no result",
                             stdout=stdout, stderr=stderr, seconds=time.monotonic() - t0)
        res = json.loads(result_file.read_text())
        rr = RunResult(status=res["status"], error_type=res.get("error_type"), error=res.get("error"),
                       error_mro=res.get("error_mro", []), traceback=res.get("traceback"), item=res.get("item"),
                       stdout=stdout, stderr=stderr, seconds=time.monotonic() - t0)
        if rr.ok:
            arrays = dict(np.load(tmp / "outputs.npz", allow_pickle=False))
            rr.outputs = [_unpack(o, arrays) for o in json.loads((tmp / "outputs.json").read_text())]
        return rr
    finally:
        if not keep_dir:
            shutil.rmtree(tmp, ignore_errors=True)
