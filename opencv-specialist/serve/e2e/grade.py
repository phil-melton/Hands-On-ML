"""Grade a work dir's ``answer.json`` against a task's truth.

Per key: pass/fail with the measured error and the tolerance. A task passes when every key passes. A missing or
malformed answer fails the task and is recorded as such (``status`` = "missing" / "malformed" / "not_object").

The truth is never read from the work dir: the grader takes the ``Task`` object (rebuilt from its number, deterministic),
so the orchestrator cannot influence it by editing files, and a copy of the input image it may have overwritten is not
used either (the text-layout check uses the manufactured canvas, not the file on disk).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_ANSWER_BYTES = 2_000_000


@dataclass
class KeyResult:
    ok: bool
    error: float | None = None  # measured error in the unit of ``tol``; None where the check is not numeric
    tol: float | None = None
    detail: str = ""

    def to_json(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in asdict(self).items()}


def _num(x) -> float | None:
    """A finite JSON number (not a bool), else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _intlike(x) -> int | None:
    v = _num(x)
    if v is None or v != int(v):
        return None
    return int(v)


def _bad(why: str) -> KeyResult:
    return KeyResult(False, None, None, "malformed: " + why)


def _vec(x, n: int | None = None) -> list[float] | None:
    if not isinstance(x, list) or (n is not None and len(x) != n):
        return None
    out = [_num(v) for v in x]
    return None if any(v is None for v in out) else out  # type: ignore[return-value]


def grade_value(spec: dict, got: Any, ctx: dict | None = None) -> KeyResult:
    kind = spec["kind"]
    ctx = ctx or {}
    if kind == "int":
        v = _intlike(got)
        if v is None:
            return _bad(f"expected an integer, got {got!r}")
        return KeyResult(v == spec["value"], abs(v - spec["value"]), 0.0, "" if v == spec["value"] else f"got {v}, truth {spec['value']}")
    if kind == "ints":
        if not isinstance(got, list) or any(_intlike(g) is None for g in got):
            return _bad(f"expected a list of integers, got {got!r}")
        g = [int(v) for v in got]
        return KeyResult(g == spec["value"], None, 0.0, "" if g == spec["value"] else f"got {g}, truth {spec['value']}")
    if kind == "str":
        if not isinstance(got, str):
            return _bad(f"expected a string, got {got!r}")
        ok = got.strip().lower() == spec["value"].lower()
        return KeyResult(ok, None, None, "" if ok else f"got {got!r}, truth {spec['value']!r}")
    if kind == "bool":
        if not isinstance(got, bool):
            return _bad(f"expected true or false, got {got!r}")
        return KeyResult(got == spec["value"], None, None, "" if got == spec["value"] else f"got {got}, truth {spec['value']}")
    if kind == "num":
        v = _num(got)
        if v is None:
            return _bad(f"expected a number, got {got!r}")
        e = abs(v - spec["value"])
        return KeyResult(e <= spec["tol"], e, spec["tol"], f"got {v:.6g}, truth {spec['value']:.6g}")
    if kind == "rel":
        v = _num(got)
        if v is None:
            return _bad(f"expected a number, got {got!r}")
        e = abs(v - spec["value"]) / max(1.0, abs(spec["value"]))
        return KeyResult(e <= spec["rtol"], e, spec["rtol"], f"got {v!r}, truth {spec['value']!r} (relative error)")
    if kind == "angle":
        v = _num(got)
        if v is None:
            return _bad(f"expected a number, got {got!r}")
        p = spec["period"]
        e = abs((v - spec["value"] + p / 2) % p - p / 2)
        return KeyResult(e <= spec["tol"], e, spec["tol"], f"got {v:.6g}, truth {spec['value']:.6g} (period {p:g})")
    if kind == "point":
        v = _vec(got, 2)
        if v is None:
            return _bad(f"expected [x, y], got {got!r}")
        e = math.hypot(v[0] - spec["value"][0], v[1] - spec["value"][1])
        return KeyResult(e <= spec["tol"], e, spec["tol"], f"got {v}, truth {spec['value']}")
    if kind == "points":
        want = spec["value"]
        if not isinstance(got, list) or len(got) != len(want):
            return _bad(f"expected {len(want)} points, got {got!r}")
        errs = []
        for g, w in zip(got, want):
            v = _vec(g, 2)
            if v is None:
                return _bad(f"expected [x, y] rows, got {g!r}")
            errs.append(math.hypot(v[0] - w[0], v[1] - w[1]))
        e = max(errs)
        return KeyResult(e <= spec["tol"], e, spec["tol"], "per-point errors " + ", ".join(f"{x:.3g}" for x in errs))
    if kind == "sorted_nums":
        v = _vec(got)
        want, tols = spec["value"], spec["tols"]
        if v is None or len(v) != len(want):
            return _bad(f"expected {len(want)} numbers, got {got!r}")
        if any(a < b for a, b in zip(v, v[1:])):
            return KeyResult(False, None, None, f"not sorted largest first: {v}")
        errs = [abs(a - b) for a, b in zip(v, want)]
        ok = all(e <= t for e, t in zip(errs, tols))
        worst = max(range(len(errs)), key=lambda i: errs[i] / max(tols[i], 1e-12))
        return KeyResult(ok, errs[worst], tols[worst], "per-element errors " + ", ".join(f"{e:.3g}/{t:.3g}" for e, t in zip(errs, tols)))
    if kind == "dict":
        if not isinstance(got, dict):
            return _bad(f"expected an object, got {got!r}")
        kids = {k: (grade_value(s, got[k], ctx) if k in got else KeyResult(False, None, None, "missing key"))
                for k, s in spec["children"].items()}
        bad = [k for k, r in kids.items() if not r.ok]
        return KeyResult(not bad, None, None, "" if not bad else "; ".join(f"{k}: {kids[k].detail}" for k in bad))
    if kind == "homography":
        if not isinstance(got, list) or len(got) != 3 or any(_vec(r, 3) is None for r in got):
            return _bad(f"expected a 3x3 array, got {got!r}")
        Hm = np.array(got, float)
        if abs(Hm[2, 2] - 1.0) > spec["norm_tol"]:
            return KeyResult(False, None, None, f"H[2][2] = {Hm[2, 2]:.6g}, the homography must be normalised to 1")
        pts = np.array(spec["pts"], float)
        p = np.c_[pts, np.ones(len(pts))] @ Hm.T
        with np.errstate(divide="ignore", invalid="ignore"):
            est = p[:, :2] / p[:, 2:3]
        Ht = np.array(spec["value"], float)
        q = np.c_[pts, np.ones(len(pts))] @ Ht.T
        want = q[:, :2] / q[:, 2:3]
        d = np.linalg.norm(est - want, axis=1)
        e = float(np.max(d)) if np.all(np.isfinite(d)) else float("inf")
        return KeyResult(e <= spec["tol"], e, spec["tol"], "probe error at the ROI corners")
    if kind == "present":
        ok = _num(got) is not None or _vec(got) is not None
        return KeyResult(ok, None, None, "" if ok else f"expected a number or a list of numbers, got {got!r}")
    if kind == "layout":
        return _grade_layout(ctx)
    raise ValueError(f"unknown spec kind {kind!r}")


def _grade_layout(ctx: dict) -> KeyResult:
    """text_layout's own verifier on out.png, with the manufactured canvas, box and text as the scene."""
    import cv2

    from cvbench.families import text_layout as tl

    scene, answer, workdir = ctx.get("tl_scene"), ctx.get("answer", {}), ctx.get("workdir")
    path = Path(workdir) / "out.png" if workdir else None
    if scene is None or path is None:
        return KeyResult(False, None, None, "no scene or work dir to check the layout against")
    if not path.is_file():
        return KeyResult(False, None, None, "out.png is missing")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return KeyResult(False, None, None, "out.png is not a readable image")
    scale, ts = _num(answer.get("scale")), answer.get("text_size")
    if scale is None or _vec(ts, 2) is None or any(_intlike(v) is None for v in ts):
        return KeyResult(False, None, None, "scale must be a number and text_size a pair of integers")
    out = {"img": img, "scale": scale, "text_size": np.array([int(v) for v in ts], dtype=np.int64)}
    v = tl.verify_manufactured(scene, out)
    return KeyResult(bool(v), None, None, v.reason if not v else "text_layout.verify_manufactured passed")


def load_answer(workdir) -> tuple[Any, str, str]:
    """(answer, status, detail); status is "ok", "missing", "malformed" or "not_object"."""
    p = Path(workdir) / "answer.json"
    if not p.is_file():
        return None, "missing", "no answer.json in the work dir"
    if p.stat().st_size > MAX_ANSWER_BYTES:
        return None, "malformed", f"answer.json is larger than {MAX_ANSWER_BYTES} bytes"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, "malformed", f"{type(e).__name__}: {e}"
    if not isinstance(obj, dict):
        return None, "not_object", f"answer.json holds a {type(obj).__name__}, not an object"
    return obj, "ok", ""


def grade_answer(task, answer: dict | None, workdir=None, status: str = "ok", detail: str = "") -> dict:
    """Grade a parsed answer (or record why there is none). Returns a JSON-able dict."""
    res: dict = {"task": task.id, "status": status, "detail": detail, "passed": False, "keys": {}, "extra_keys": []}
    if status != "ok" or answer is None:
        res["n_keys"], res["n_pass"] = len(task.truth), 0
        return res
    ctx = {"tl_scene": task.extras.get("tl_scene"), "answer": answer, "workdir": str(workdir) if workdir else None}
    for key, spec in task.truth.items():
        if spec["kind"] == "layout":
            r = grade_value(spec, None, ctx)
        elif key not in answer:
            r = KeyResult(False, None, None, "missing key")
        else:
            r = grade_value(spec, answer[key], ctx)
        res["keys"][key] = r.to_json()
    res["extra_keys"] = sorted(set(answer) - set(task.truth))
    res["n_keys"] = len(task.truth)
    res["n_pass"] = sum(1 for r in res["keys"].values() if r["ok"])
    res["passed"] = res["n_pass"] == res["n_keys"]
    return res


def grade_workdir(task, workdir) -> dict:
    answer, status, detail = load_answer(workdir)
    return grade_answer(task, answer, workdir, status, detail)
