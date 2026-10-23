"""Runtime metamorphic checks: a confidence signal for code that has no reference answer.

At serve time nobody knows the right answer, but the task's symmetries still exist (PLAN.md Phase 4). After a
run verified, the same code is rerun, through ``sandbox.run_solution`` (CLAUDE.md Rule 4), on three transformed
copies of every input image:

    rot90    cv2.rotate(img, ROTATE_90_CLOCKWISE)   pixel-exact permutation, moves every coordinate
    mirror   cv2.flip(img, 1)  (horizontal flip)    pixel-exact permutation, moves x only
    dim      round(0.6 * pixel), uint8              same geometry, every intensity scaled down

and every output key is compared with the original run's value. Per (transform, key) the verdict is one of

    invariant   the value did not change: scalars, counts, areas, lengths, sorted values (relative tolerance
                below); point/box/segment arrays whose rows are the same set; images equal as they are
    covariant   the value moved with the input: a point/box/segment array that equals the original after the
                original coordinates are pushed through the same transform (unordered set, pixel tolerance);
                an image/mask that equals the original after the inverse transform (IoU / mean abs diff)
    changed     the rerun succeeded and the value differs
    error       the rerun failed (exception, timeout, lint, sandbox violation) or lacks the key

``stable_fraction`` = (invariant + covariant) / all checks; an error or a changed value counts against it.

THIS IS A CONFIDENCE SIGNAL, NOT A GATE. Nothing here fails the tool, and a low fraction does not mean the code is
wrong: some tasks are legitimately not invariant.

    * dim changes intensity statistics: a mean/median gray level, a fixed-threshold count (``> 127``), a gray
      image output, or a histogram *should* change when every pixel is multiplied by 0.6.
    * rot90 swaps width and height and moves "top-left"-based conventions: outputs sorted "by x" or "by y", a
      "first corner is the top-left one" ordering, (w, h) pairs, angles, and a homography between two images all
      legitimately change. Coordinates that are not in the input frame (a rectified image) are *invariant*,
      not covariant; both readings are tried.
    * mirror flips handedness: signed angles, orientation, and left/right-sorted lists change.
    * (N, 2) and (N, 4) arrays are read as points, segments (x1, y1, x2, y2) or boxes (x, y, w, h) / (x1, y1, x2,
      y2) in pixel coordinates of the FIRST input image. For N > MAX_SET_ROWS they are not compared.
    * Lists of unequal arrays and dicts are compared shallowly (lengths and sizes), not value by value.

Conversely a high fraction is evidence of a geometry- and brightness-robust implementation, not of correctness:
code that returns a constant is invariant under everything.

Tolerances are named constants, each with the reason for its value. They were chosen from the measurement
physics of the pixel grid (not tuned on outputs), and are not calibrated against labelled good/bad runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from sandbox.run import run_solution

# --- tolerances -------------------------------------------------------------------------------------------------
# rot90 and mirror only permute pixels: a deterministic pixel-level algorithm gives identical areas and counts, and
# float summation order gives ~1e-7. 1 % leaves room for scan-order effects (a contour start point, a Hough bin tie)
# without letting a one-object-off count (>= 1/N for N < 100) pass.
REL_TOL_PERMUTE = 0.01
# dim rounds every pixel to the nearest 1/255 after x0.6, so an edge pixel can flip across a threshold. A boundary
# shift of <= 1 px changes the area of a disc of radius r by ~1/r in relative terms: 5 % is "objects of radius
# >= ~20 px (about 1250 px^2) are measured to within the rounding".
REL_TOL_DIM = 0.05
# float32 accumulation on values up to ~1e3 is good to ~1e-4 absolute; keeps values near 0 from failing on noise.
ABS_TOL = 1e-4
# Coordinate tolerance per component, in px. Conventions differ by 0.5-1 px (integer pixel centres, x' = h-1-y, vs
# a corner origin, x' = h-y), and a detector may pick the neighbouring pixel on the transformed image: 2 px covers
# both, and is below the spacing of anything one would call two separate objects.
COORD_TOL_PX = 2.0
# Mask IoU after mapping back. Permuting transforms: a mask that differs on 2 % of its union is a different mask.
# dim: a 1 px boundary shift on an object of radius 20 px gives IoU ~ 0.90.
MASK_IOU_PERMUTE = 0.98
MASK_IOU_DIM = 0.90
# Mean absolute difference of a non-mask uint8 image, in gray levels. The pixel permutations introduce no
# interpolation, so a faithful implementation differs by ~0; 3 levels (1.2 %) absorbs border-handling asymmetry.
IMAGE_MAD_LEVELS = 3.0
# Pairwise L-inf distances are computed in chunks of at most CHUNK_ELEMS floats; sets above MAX_SET_ROWS rows are
# skipped (20000 x 20000 pairs is already seconds of numpy).
MAX_SET_ROWS = 20000
CHUNK_ELEMS = 2_000_000
DIM_FACTOR = 0.6
MIN_IMAGE_SIDE = 9  # a uint8 array is "an image" only if both sides exceed 8 px (so (N, 4) boxes never are)

STABLE = ("invariant", "covariant")
RANK = {"invariant": 0, "covariant": 1, "changed": 2, "error": 3}
SHALLOW_NOTE = "shallow: lengths and element sizes only"


# --- the transforms ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    name: str
    geometric: bool                                   # moves coordinates (False: same pixel grid)
    apply: Callable[[np.ndarray], np.ndarray]         # input image -> transformed input image
    inverse: Callable[[np.ndarray], np.ndarray]       # image in the transformed frame -> original frame
    points: Callable[[np.ndarray, int, int], np.ndarray]       # (N, 2) x, y; original h, w -> transformed frame
    boxes: Callable[[np.ndarray, int, int], np.ndarray]        # (N, 4) x, y, w, h (inclusive pixels) -> same
    rtol: float
    iou_min: float


def _rot90_points(p: np.ndarray, h: int, w: int) -> np.ndarray:
    # a clockwise quarter turn sends pixel (col x, row y) to (col h-1-y, row x)
    return np.stack([h - 1 - p[:, 1], p[:, 0]], axis=1)


def _rot90_boxes(b: np.ndarray, h: int, w: int) -> np.ndarray:
    # columns x..x+bw-1 and rows y..y+bh-1 become columns h-y-bh..h-1-y and rows x..x+bw-1
    return np.stack([h - b[:, 1] - b[:, 3], b[:, 0], b[:, 3], b[:, 2]], axis=1)


def _mirror_points(p: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack([w - 1 - p[:, 0], p[:, 1]], axis=1)


def _mirror_boxes(b: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack([w - b[:, 0] - b[:, 2], b[:, 1], b[:, 2], b[:, 3]], axis=1)


def _dim(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img.astype(np.float32) * DIM_FACTOR), 0, 255).astype(np.uint8)


TRANSFORMS: dict[str, Transform] = {t.name: t for t in (
    Transform("rot90", True, lambda im: np.ascontiguousarray(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)),
              lambda im: np.ascontiguousarray(np.rot90(im, 1)), _rot90_points, _rot90_boxes,
              REL_TOL_PERMUTE, MASK_IOU_PERMUTE),
    Transform("mirror", True, lambda im: np.ascontiguousarray(cv2.flip(im, 1)),
              lambda im: np.ascontiguousarray(im[:, ::-1]), _mirror_points, _mirror_boxes,
              REL_TOL_PERMUTE, MASK_IOU_PERMUTE),
    Transform("dim", False, _dim, lambda im: im, lambda p, h, w: p, lambda b, h, w: b,
              REL_TOL_DIM, MASK_IOU_DIM),
)}


# --- comparing one value ----------------------------------------------------------------------------------------

def _normalise(v: Any) -> Any:
    """str stays str; scalars and regular nested numeric lists become ndarrays; ragged lists and dicts stay."""
    if isinstance(v, (str, dict)):
        return v
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
        except ValueError:
            return list(v)
        return a if a.dtype != object else list(v)
    return np.asarray(v)


def _as_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    return list(x) if isinstance(x, np.ndarray) and x.ndim >= 1 else [x]


def _worst(results: list[tuple[str, str]]) -> tuple[str, str]:
    return max(results, key=lambda r: RANK[r[0]]) if results else ("invariant", "empty")


def _is_image(a: np.ndarray) -> bool:
    return (a.dtype == np.uint8 and a.ndim in (2, 3) and (a.ndim == 2 or a.shape[2] in (1, 3, 4))
            and min(a.shape[:2]) >= MIN_IMAGE_SIDE) or (a.dtype == bool and a.ndim == 2
                                                         and min(a.shape) >= MIN_IMAGE_SIDE)


def _is_mask(a: np.ndarray) -> bool:
    if a.dtype == bool:
        return True
    vals = set(np.unique(a).tolist())
    return vals <= {0, 255} or vals <= {0, 1}


def _image_metric(a: np.ndarray, b: np.ndarray, mask: bool) -> tuple[float, str]:
    """(score, text); the score is IoU for masks (higher is better) and mean abs difference otherwise (lower)."""
    if mask:
        pa, pb = a > 0, b > 0
        union = int(np.logical_or(pa, pb).sum())
        iou = 1.0 if union == 0 else float(np.logical_and(pa, pb).sum()) / union
        return iou, f"IoU {iou:.3f}"
    mad = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    return mad, f"mean abs diff {mad:.2f} levels"


def _compare_images(a: np.ndarray, b: np.ndarray, tf: Transform) -> tuple[str, str]:
    mask = _is_mask(a) and _is_mask(b)

    def ok(score: float) -> bool:
        return score >= tf.iou_min if mask else score <= IMAGE_MAD_LEVELS

    seen = []
    if a.shape == b.shape:  # as it is: for dim, or an output that does not live in the input frame
        score, text = _image_metric(a, b, mask)
        if ok(score):
            return "invariant", text
        seen.append(text)
    if tf.geometric:
        back = tf.inverse(b)
        if back.shape == a.shape:
            score, text = _image_metric(a, back, mask)
            if ok(score):
                return "covariant", text + " after the inverse transform"
            seen.append(text + " after the inverse transform")
    if not seen:
        return "changed", f"image shape {a.shape} -> {b.shape}"
    return "changed", "; ".join(seen)


def _hausdorff(A: np.ndarray, B: np.ndarray, swap: bool = False) -> float:
    """Symmetric Hausdorff distance between two row sets in L-inf (so a px tolerance applies per component).
    swap: the second half of a 4-column row is a second endpoint that may trade places with the first."""
    def directed(X: np.ndarray, Y: np.ndarray) -> float:
        worst, step = 0.0, max(1, CHUNK_ELEMS // max(len(Y), 1))
        for i in range(0, len(X), step):
            x = X[i:i + step, None, :]
            d = np.abs(x - Y[None, :, :]).max(axis=2)
            if swap:
                d = np.minimum(d, np.abs(x - Y[None, :, [2, 3, 0, 1]]).max(axis=2))
            worst = max(worst, float(d.min(axis=1).max()))
        return worst
    return max(directed(A, B), directed(B, A))


def _compare_rows(a: np.ndarray, b: np.ndarray, tf: Transform, dims: list[tuple[int, int]]) -> tuple[str, str]:
    """(N, 2) points or (N, 4) segments/boxes as unordered sets, in pixel units."""
    if a.shape[0] != b.shape[0]:
        return "changed", f"{a.shape[0]} rows -> {b.shape[0]} rows"
    if a.shape[0] == 0:
        return "invariant", "no rows"
    if a.shape[0] > MAX_SET_ROWS:
        return "changed", f"{a.shape[0]} rows: too many to match, not compared"
    A, B = a.astype(np.float64), b.astype(np.float64)
    k = A.shape[1]
    kinds = ("rows",) if k == 2 else ("rows", "segment")
    best, best_text = np.inf, ""
    for kind in kinds:  # as they are (a segment may list its endpoints in the other order)
        d = _hausdorff(A, B, swap=kind == "segment")
        if d <= COORD_TOL_PX:
            return "invariant", f"same {'segments' if kind == 'segment' else 'rows'} within {d:.2f} px"
        if d < best:
            best, best_text = d, f"{d:.1f} px from the original"
    if tf.geometric:
        Bn = None if k == 2 else np.concatenate([np.minimum(B[:, :2], B[:, 2:]), np.maximum(B[:, :2], B[:, 2:])], axis=1)
        for h, w in dims:
            if k == 2:
                cands = [("points", tf.points(A, h, w), B, False)]
            else:
                p1, p2 = tf.points(A[:, :2], h, w), tf.points(A[:, 2:], h, w)
                cands = [("segments", np.concatenate([p1, p2], axis=1), B, True),
                         ("boxes (x, y, w, h)", tf.boxes(A, h, w), B, False),
                         ("boxes (x1, y1, x2, y2)",
                          np.concatenate([np.minimum(p1, p2), np.maximum(p1, p2)], axis=1), Bn, False)]
            for label, mapped, target, swap in cands:
                d = _hausdorff(mapped, target, swap=swap)
                if d <= COORD_TOL_PX:
                    return "covariant", f"matches the original mapped as {label} within {d:.2f} px"
                if d < best:
                    best, best_text = d, f"{d:.1f} px from the mapped original"
    return "changed", best_text


def compare_value(a: Any, b: Any, tf: Transform, dims: list[tuple[int, int]]) -> tuple[str, str]:
    """Classify one output value of the original run (``a``) against the transformed run's (``b``)."""
    a, b = _normalise(a), _normalise(b)
    if isinstance(a, str) or isinstance(b, str):
        return ("invariant", "equal") if a == b else ("changed", f"{str(a)[:30]!r} vs {str(b)[:30]!r}")
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return "changed", "keys differ"
        return _worst([compare_value(a[k], b[k], tf, dims) for k in a])
    if isinstance(a, list) or isinstance(b, list):
        la, lb = _as_list(a), _as_list(b)
        if len(la) != len(lb):
            return "changed", f"{len(la)} items -> {len(lb)} items"
        sa = sorted(np.asarray(x).size for x in la)
        sb = sorted(np.asarray(x).size for x in lb)
        return ("invariant", SHALLOW_NOTE) if sa == sb else ("changed", "element sizes differ")
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return ("invariant", "equal") if np.array_equal(a, b) else ("changed", "differs")
    if a.size == 0 and b.size == 0:
        return "invariant", "empty"
    if _is_image(a) and _is_image(b) and a.ndim == b.ndim:
        return _compare_images(a, b, tf)
    if a.ndim == 2 and a.shape[1] in (2, 4) and b.ndim == 2 and b.shape[1] == a.shape[1]:
        return _compare_rows(a, b, tf, dims)
    if a.shape != b.shape:
        if a.ndim == 1 and b.ndim == 1:
            return "changed", f"length {a.shape[0]} -> {b.shape[0]}"
        return "changed", f"shape {a.shape} -> {b.shape}"
    A, B = a.astype(np.float64), b.astype(np.float64)
    if np.allclose(A, B, rtol=tf.rtol, atol=ABS_TOL, equal_nan=True):
        if A.ndim == 0:
            return "invariant", f"{A.item():.6g}"
        return "invariant", f"max abs diff {float(np.nanmax(np.abs(A - B))):.3g}"
    if A.ndim == 1 and np.allclose(np.sort(A), np.sort(B), rtol=tf.rtol, atol=ABS_TOL, equal_nan=True):
        return "invariant", "same values, different order"
    if A.ndim == 0:
        return "changed", f"{A.item():.6g} -> {B.item():.6g}"
    diff = np.abs(A - B)
    close = np.isclose(A, B, rtol=tf.rtol, atol=ABS_TOL, equal_nan=True)
    return "changed", f"{int((~close).sum())} of {A.size} values differ, max abs diff {float(np.nanmax(diff)):.3g}"


# --- the runner -------------------------------------------------------------------------------------------------

def _is_image_input(v: Any) -> bool:
    return (isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim in (2, 3)
            and (v.ndim == 2 or v.shape[2] in (1, 3, 4)))


def _verdict(checks: dict[str, dict[str, dict]], reruns: dict[str, dict]) -> tuple[float | None, dict, str]:
    cells = [(t, k, c["class"]) for t, ks in checks.items() for k, c in ks.items()]
    counts = {name: sum(1 for _, _, c in cells if c == name) for name in RANK}
    n = len(cells)
    if n == 0:
        return None, counts, "no outputs to compare"
    stable = counts["invariant"] + counts["covariant"]
    frac = round(stable / n, 3)
    if counts["error"] == n:
        why = "; ".join(f"{t}: {r.get('error', 'failed')}"[:120] for t, r in reruns.items() if not r["ok"])
        return frac, counts, f"no signal: the code failed on every transformed copy ({why})"
    if stable == n:
        return frac, counts, (f"stable: all {n} checks invariant or covariant under {', '.join(checks)} "
                              "(robust to these changes, which is not the same as correct)")
    bad = [f"{t}: {', '.join(k for k, c in ks.items() if c['class'] not in STABLE)}"
           for t, ks in checks.items() if any(c["class"] not in STABLE for c in ks.values())]
    return frac, counts, (f"{stable}/{n} stable; not stable under " + "; ".join(bad)
                          + ". Confidence signal only: check whether those outputs should be invariant for this task")


def run_metamorphic(code: str, inputs: dict[str, Any], outputs: dict[str, Any], *, keys: list[str] | None = None,
                    timeout_s: float = 30.0, mem_mb: int = 4096,
                    run: Callable[..., Any] = run_solution) -> dict:
    """Rerun ``code`` on rot90 / mirror / dim copies of every image in ``inputs`` and classify each key of ``outputs``.

    ``inputs`` are the keyword arguments of the verified run (images as uint8 arrays, extra scalars untouched),
    ``outputs`` its output dict; ``keys`` restricts the keys compared (default: all of ``outputs``). Never raises.
    Returns {"checks": {transform: {key: {"class", "detail"}}}, "reruns": {transform: {"ok", "seconds", "error"?}},
    "counts", "stable_fraction", "verdict", "note"} or {"skipped": reason}.
    """
    images = {k: v for k, v in inputs.items() if _is_image_input(v)}
    if not images:
        return {"skipped": "no uint8 image among the inputs"}
    keys = list(outputs) if keys is None else [k for k in keys if k in outputs]
    dims: list[tuple[int, int]] = []
    for v in images.values():
        if (v.shape[0], v.shape[1]) not in dims:
            dims.append((v.shape[0], v.shape[1]))
    checks: dict[str, dict[str, dict]] = {}
    reruns: dict[str, dict] = {}
    for name, tf in TRANSFORMS.items():
        moved = {**inputs, **{k: tf.apply(v) for k, v in images.items()}}
        try:
            res = run(code, [moved], timeout_s=timeout_s, mem_mb=mem_mb)
        except Exception as e:  # a sandbox that cannot start is an error cell, not a tool failure
            checks[name] = {k: {"class": "error", "detail": f"rerun not started: {type(e).__name__}: {e}"[:200]} for k in keys}
            reruns[name] = {"ok": False, "seconds": 0.0, "error": f"{type(e).__name__}: {e}"[:200]}
            continue
        reruns[name] = {"ok": bool(res.ok), "seconds": round(float(res.seconds), 2)}
        if not res.ok:
            msg = res.summary()[:200]
            reruns[name]["error"] = msg
            checks[name] = {k: {"class": "error", "detail": msg} for k in keys}
            continue
        out2 = res.outputs[0]
        checks[name] = {}
        for k in keys:
            if k not in out2:
                checks[name][k] = {"class": "error", "detail": "missing from the rerun's output"}
                continue
            try:
                cls, detail = compare_value(outputs[k], out2[k], tf, dims)
            except Exception as e:
                cls, detail = "error", f"comparison failed: {type(e).__name__}: {e}"[:200]
            checks[name][k] = {"class": cls, "detail": detail}
    frac, counts, verdict = _verdict(checks, reruns)
    return {"checks": checks, "reruns": reruns, "counts": counts, "stable_fraction": frac, "verdict": verdict,
            "note": "confidence signal, not a gate: dim legitimately changes intensity statistics, rot90/mirror "
                    "legitimately change ordering conventions and two-image geometry; see serve/metamorphic.py"}
