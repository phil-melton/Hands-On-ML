"""Group 2, dtypes: what OpenCV accepts, silently converts, or rejects, per element type.

Timeless (same on 4.14 and 5.0):
  * int64 (numpy's default int) is narrowed to int32 with wraparound at the binding, in EVERY cv2
    function (``cv2.add`` on 2**40 gives 0), even with ``dtype=cv2.CV_64S``;
  * ``getPerspectiveTransform`` wants float32 points, ``perspectiveTransform`` wants (N, 1, 2),
    ``fillPoly``/``polylines`` want int32 points (int64 is accepted, float32 is not), ``remap`` wants
    float32 maps, ``addWeighted`` of different dtypes wants ``dtype=``, ``threshold`` rejects int64.

5.0 subset: bool, uint32 and float16 arrays are accepted where 4.14 raised, but only by SOME functions,
and ``cv2.bitwise_not`` on a bool mask does not invert it. So each 5.0 check is scoped to the function
the probe measured, and a counter-case shows the functions that still reject the new dtype.

Measured on 5.0.0 while building (not all are in the recorded probes; the new ones are tagged ``new:``):
  bool     accepted: countNonZero, bitwise_and/or/xor (bool in, bool out), ``mask=`` arguments, findNonZero,
           flip, merge/split, copyMakeBorder, INTER_NEAREST resize, moments(binaryImage=True), sumElems
           rejected: findContours, connectedComponents*, threshold, dilate/erode, resize (linear), GaussianBlur,
           medianBlur, distanceTransform, cvtColor, warpAffine, add/subtract/absdiff, boundingRect
           trap:     ``bitwise_not`` runs but returns bytes 0xFE/0xFF, i.e. every element True (``new:bitwise_not_bool``)
  uint32   accepted: add/subtract/multiply (saturating), absdiff, max/min, bitwise_*, compare, minMaxLoc, sumElems,
           countNonZero, normalize, flip, transpose, INTER_NEAREST resize; rejected: GaussianBlur, resize (linear),
           reduce, threshold
  float16  accepted: add/subtract/multiply/addWeighted, normalize, minMaxLoc, mean, flip, transpose, countNonZero;
           rejected: GaussianBlur, resize, blur, medianBlur, filter2D, Sobel, Canny, cvtColor, threshold, warp*,
           remap, dilate, matchTemplate, integral, calcHist
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail, mask_iou

GROUP = "dtypes"


def _spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="gotcha", task=task, inputs=inputs, outputs=outputs)


def _code(sig: str, body: str) -> str:
    return f"import cv2\nimport numpy as np\n\n\ndef solve({sig}):\n{body}"


# ---------------------------------------------------------------- 1. int64 arithmetic wraps inside cv2

def _i64_cases():
    out = []
    for seed in (201, 202):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 2 ** 41, (6, 8), dtype=np.int64)
        b = rng.integers(0, 2 ** 41, (6, 8), dtype=np.int64)
        a[0, :3] = rng.integers(0, 100, 3)  # a few small counts so a lazy "all huge" assumption also fails
        out.append(Case({"a": a, "b": b}, a + b))
    return out


def _i64_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "total")
    if not np.array_equal(got, case.truth):
        bad = int((got != case.truth).sum())
        return fail(f"{bad} of {got.size} sums are wrong (int64 values above 2**31 were wrapped)")
    return PASS


INT64_ADD = Check(
    id="dtype_int64_add", group=GROUP, probes=("int64_overflow_narrowing", "int64_array_input"),
    depends_on_5_0=False, role="gotcha",
    gotcha="cv2 narrows int64 to int32 with wraparound: cv2.add on 2**40 gives 0, even with dtype=cv2.CV_64S",
    spec=_spec("Add two per-pixel count maps elementwise. The counts are int64 and can reach about 2**41.",
               {"a": IOSpec("int64", "HxW", 2, "counts"), "b": IOSpec("int64", "HxW", 2, "counts")},
               {"total": IOSpec("int64", "HxW", 2, "a + b, exact")}),
    make_cases=_i64_cases, verify=_i64_verify,
    reference=_code("a, b", '    return {"total": a + b}\n'),
    wrong={
        "cv2_add": Wrong("cv2.add(a, b): the output is int32", _code("a, b", '    return {"total": cv2.add(a, b)}\n'),
                         "verifier"),
        "cv2_add_recast": Wrong("cv2.add(a, b).astype(np.int64): right dtype, wrapped values",
                                _code("a, b", '    return {"total": cv2.add(a, b).astype(np.int64)}\n'), "verifier"),
        "cv2_add_dtype_arg": Wrong("cv2.add(a, b, dtype=cv2.CV_64S): the inputs were already narrowed",
                                   _code("a, b", '    return {"total": cv2.add(a, b, dtype=cv2.CV_64S)}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 2. threshold on int64

def _thr_cases():
    out = []
    for seed, t in ((211, 500), (212, 300)):
        v = np.random.default_rng(seed).integers(0, 1000, (12, 16), dtype=np.int64)
        out.append(Case({"values": v, "thresh": t}, (v > t).astype(np.uint8) * 255))
    return out


def _thr_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "mask")
    if got.shape != case.truth.shape or not np.array_equal(got, case.truth):
        return fail(f"{int((got != case.truth).sum())} pixels differ from (values > thresh) * 255")
    return PASS


INT64_THRESHOLD = Check(
    id="dtype_int64_threshold", group=GROUP, probes=("threshold_int64_input",), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2.threshold rejects int64 input (-210); casting to uint8 instead wraps values above 255",
    spec=_spec("Binarize the sensor readings: 255 where the reading is greater than the threshold, else 0.",
               {"values": IOSpec("int64", "HxW", 2, "readings in 0..999"),
                "thresh": IOSpec("int", "scalar", None, "threshold")},
               {"mask": IOSpec("uint8", "HxW", 2, "0 or 255")}),
    make_cases=_thr_cases, verify=_thr_verify,
    reference=_code("values, thresh",
                    '    _, mask = cv2.threshold(values.astype(np.float32), thresh, 255, cv2.THRESH_BINARY)\n'
                    '    return {"mask": mask.astype(np.uint8)}\n'),
    wrong={
        "direct": Wrong("cv2.threshold on the int64 array",
                        _code("values, thresh", '    _, mask = cv2.threshold(values, thresh, 255, cv2.THRESH_BINARY)\n'
                                                '    return {"mask": mask.astype(np.uint8)}\n'),
                        "error", "Unsupported format"),
        "uint8_wrap": Wrong("values.astype(np.uint8): readings above 255 wrap around",
                            _code("values, thresh",
                                  '    _, mask = cv2.threshold(values.astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)\n'
                                  '    return {"mask": mask}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 3. getPerspectiveTransform needs float32

# calibrated on 200 random quads (float and rounded): the float32 reference maps src onto dst to 1.3e-5 px at worst, and
# cv2.perspectiveTransform agrees with a numpy projection to 3.8e-6 px; 1e-3 px leaves ~75x margin
PT_TOL = 1e-3


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts.astype(np.float64), np.ones(len(pts))] @ np.asarray(H, np.float64).T
    return p[:, :2] / p[:, 2:3]


def _pers_cases():
    out = []
    for seed, dtype in ((221, np.float64), (222, np.int32)):
        rng = np.random.default_rng(seed)
        w, h = int(rng.integers(120, 220)), int(rng.integers(80, 160))
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        dst = src * rng.uniform(0.8, 1.1) + rng.uniform(-12, 12, (4, 2)) + [10, 10]
        if dtype == np.int32:
            src, dst = np.round(src), np.round(dst)
        out.append(Case({"src": src.astype(dtype), "dst": dst.astype(dtype)}, (src, dst)))
    return out


def _pers_verify(case: Case, out: dict) -> Verdict:
    src, dst = case.truth
    H = arr(out, "H")
    if H.shape != (3, 3):
        return fail(f"H has shape {H.shape}")
    err = np.abs(_project(H, src) - dst).max()
    return fail(f"H maps src onto dst with an error of {err:.3f} px") if not err <= PT_TOL else PASS


PERSPECTIVE_POINTS = Check(
    id="dtype_persp_points", group=GROUP, probes=("getPerspectiveTransform_float64", "getPerspectiveTransform_int32"),
    depends_on_5_0=False, role="gotcha",
    gotcha="getPerspectiveTransform accepts only float32 points: float64 and int32 (pixel coordinates) both raise",
    spec=_spec("Compute the 3x3 perspective transform that maps the four source corners onto the four "
               "destination corners.",
               {"src": IOSpec("float64 or int32", "4x2", 2, "(x, y) corners"),
                "dst": IOSpec("float64 or int32", "4x2", 2, "(x, y) corners, same order as src")},
               {"H": IOSpec("float", "3x3", 2, "maps src to dst: p_dst ~ H @ [x, y, 1]")}),
    make_cases=_pers_cases, verify=_pers_verify,
    reference=_code("src, dst", '    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))\n'
                                '    return {"H": H}\n'),
    wrong={
        "direct": Wrong("passes the float64 points straight in",
                        _code("src, dst", '    return {"H": cv2.getPerspectiveTransform(src, dst)}\n'),
                        "error", "src.checkVector(2, CV_32F) == 4"),
        "casts_floats_only": Wrong("converts float64 but not the int32 pixel coordinates",
                                   _code("src, dst",
                                         '    if src.dtype == np.float64:\n        src = src.astype(np.float32)\n'
                                         '        dst = dst.astype(np.float32)\n'
                                         '    return {"H": cv2.getPerspectiveTransform(src, dst)}\n'),
                                   "error", "src.checkVector(2, CV_32F) == 4"),
    })


# ---------------------------------------------------------------- 4. perspectiveTransform needs (N,1,2)

def _pnx2_cases():
    out = []
    for seed in (231, 232):
        rng = np.random.default_rng(seed)
        H = np.eye(3) + rng.uniform(-0.05, 0.05, (3, 3)) * [[1, 1, 20], [1, 1, 20], [0.002, 0.002, 1]]
        pts = rng.uniform(0, 100, (6, 2)).astype(np.float32)
        out.append(Case({"pts": pts, "H": H}, _project(H, pts)))
    return out


def _pnx2_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "mapped").astype(np.float64)
    if got.shape != case.truth.shape:
        return fail(f"mapped has shape {got.shape}")
    err = np.abs(got - case.truth).max()
    return fail(f"mapped points are off by {err:.3f} px") if not err <= PT_TOL else PASS


PERSPECTIVE_NX2 = Check(
    id="dtype_persp_nx2", group=GROUP, probes=("perspectiveTransform_Nx2_input",), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2.perspectiveTransform needs (N, 1, 2) input and returns (N, 1, 2); (N, 2) raises",
    spec=_spec("Map these points through the homography.",
               {"pts": IOSpec("float32", "Nx2", 2, "(x, y) points"),
                "H": IOSpec("float64", "3x3", 2, "homography: p' ~ H @ [x, y, 1]")},
               {"mapped": IOSpec("float", "Nx2", 2, "(x, y) of each mapped point, same order")}),
    make_cases=_pnx2_cases, verify=_pnx2_verify,
    reference=_code("pts, H", '    mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)\n'
                              '    return {"mapped": mapped.reshape(-1, 2)}\n'),
    wrong={
        "direct": Wrong("cv2.perspectiveTransform(pts, H) with (N, 2) points",
                        _code("pts, H", '    return {"mapped": cv2.perspectiveTransform(pts, H)}\n'),
                        "error", "scn + 1 == m.cols"),
        "no_reshape_back": Wrong("reshapes in but returns the (N, 1, 2) result as is",
                                 _code("pts, H", '    return {"mapped": cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)}\n'),
                                 "verifier"),
    })


# ---------------------------------------------------------------- 5. fillPoly: float32 rejected, int64 accepted

# calibrated on 200 polygons (5-7 vertices, R = 40..55 px on a 128 px canvas): converting the float vertices by
# floor, ceil or truncation instead of rounding (a 1 px shift) gives IoU >= 0.952 against the rounded fill, so a
# sane conversion must pass. 0.90 accepts all of them; polygons with x/y swapped or a vertex dropped fall below
# it on the check's own scenes (pinned in tests/test_gotchas.py).
FP_IOU = 0.90


def _poly(rng: np.random.Generator, n: int, size: int = 128):
    R, a0 = rng.uniform(40, 55), rng.uniform(0, 2 * math.pi)
    c = size / 2 + rng.uniform(-6, 6, 2)
    ang = a0 + 2 * math.pi * np.arange(n) / n
    return np.c_[c[0] + R * np.cos(ang), c[1] + R * np.sin(ang)]


def _fp_cases():
    out = []
    for seed in (241, 242):
        rng = np.random.default_rng(seed)
        pf = _poly(rng, int(rng.integers(5, 8))).astype(np.float32)
        pi = np.round(_poly(rng, int(rng.integers(5, 8)))).astype(np.int64)
        truth = []
        for poly in (np.round(pf), pi):
            m = np.zeros((128, 128), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            truth.append(m)
        out.append(Case({"poly_f": pf, "poly_i": pi, "height": 128, "width": 128}, truth))
    return out


def _fp_verify(case: Case, out: dict) -> Verdict:
    for key, truth in zip(("mask_f", "mask_i"), case.truth):
        got = arr(out, key)
        if got.shape != truth.shape:
            return fail(f"{key} has shape {got.shape}")
        if not set(np.unique(got).tolist()) <= {0, 255}:
            return fail(f"{key} is not a 0/255 mask")
        iou = mask_iou(got, truth)
        if iou < FP_IOU:
            return fail(f"{key} overlaps the expected polygon with IoU {iou:.2f}")
    return PASS


_FP_SIG = "poly_f, poly_i, height, width"
FILLPOLY = Check(
    id="dtype_fillpoly", group=GROUP, probes=("fillPoly_float32_points", "fillPoly_int64_points"),
    depends_on_5_0=False, role="gotcha",
    gotcha="cv2.fillPoly rejects float32 vertices (raises) but accepts numpy's default int64",
    spec=_spec("Rasterize each polygon into its own filled mask.",
               {"poly_f": IOSpec("float32", "Nx2", 2, "(x, y) vertices in pixels, e.g. from a projection"),
                "poly_i": IOSpec("int64", "Mx2", 2, "(x, y) vertices in pixels, e.g. np.array([[x, y], ...])"),
                "height": IOSpec("int", "scalar", None, "mask height"),
                "width": IOSpec("int", "scalar", None, "mask width")},
               {"mask_f": IOSpec("uint8", "HxW", 2, "255 inside poly_f, 0 outside"),
                "mask_i": IOSpec("uint8", "HxW", 2, "255 inside poly_i, 0 outside")}),
    make_cases=_fp_cases, verify=_fp_verify,
    reference=_code(_FP_SIG, '    mask_f = np.zeros((height, width), np.uint8)\n'
                             '    cv2.fillPoly(mask_f, [np.round(poly_f).astype(np.int32)], 255)\n'
                             '    mask_i = np.zeros((height, width), np.uint8)\n'
                             '    cv2.fillPoly(mask_i, [poly_i], 255)\n'
                             '    return {"mask_f": mask_f, "mask_i": mask_i}\n'),
    wrong={
        "direct": Wrong("fillPoly with the float32 vertices as given",
                        _code(_FP_SIG, '    mask_f = np.zeros((height, width), np.uint8)\n'
                                       '    cv2.fillPoly(mask_f, [poly_f], 255)\n'
                                       '    mask_i = np.zeros((height, width), np.uint8)\n'
                                       '    cv2.fillPoly(mask_i, [poly_i], 255)\n'
                                       '    return {"mask_f": mask_f, "mask_i": mask_i}\n'),
                        "error", "p.checkVector(2, CV_32S) > 0"),
        "overcast_int64": Wrong("over-converts: casts the int64 vertices to float32 before fillPoly",
                                _code(_FP_SIG, '    mask_f = np.zeros((height, width), np.uint8)\n'
                                               '    cv2.fillPoly(mask_f, [np.round(poly_f).astype(np.int32)], 255)\n'
                                               '    mask_i = np.zeros((height, width), np.uint8)\n'
                                               '    cv2.fillPoly(mask_i, [poly_i.astype(np.float32)], 255)\n'
                                               '    return {"mask_f": mask_f, "mask_i": mask_i}\n'),
                                "error", "p.checkVector(2, CV_32S) > 0"),
    })


# ---------------------------------------------------------------- 6. addWeighted with mixed dtypes

def _aw_cases():
    out = []
    for seed in (251, 252):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        ov = (rng.random((12, 16, 3)) * 255).astype(np.float32)
        out.append(Case({"img": img, "overlay": ov}, np.clip(np.round(0.5 * img + 0.5 * ov), 0, 255)))
    return out


def _aw_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "blend").astype(np.int16)
    if got.shape != case.truth.shape:
        return fail(f"blend has shape {got.shape}")
    err = np.abs(got - case.truth).max()
    return fail(f"blend is off by {err} gray levels from 0.5 * img + 0.5 * overlay") if err > 1 else PASS


ADDWEIGHTED_MIXED = Check(
    id="dtype_addweighted_mixed", group=GROUP, probes=("addWeighted_mixed_dtypes",), depends_on_5_0=False,
    role="gotcha",
    gotcha="addWeighted of a uint8 and a float32 array raises unless dtype= is given",
    spec=_spec("Blend the image and the overlay 50/50.",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
                "overlay": IOSpec("float32", "HxWx3", 3, "overlay in the same 0..255 range")},
               {"blend": IOSpec("uint8", "HxWx3", 3, "0.5 * img + 0.5 * overlay, rounded")}),
    make_cases=_aw_cases, verify=_aw_verify,
    reference=_code("img, overlay", '    return {"blend": cv2.addWeighted(img, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)}\n'),
    wrong={
        "direct": Wrong("cv2.addWeighted(img, 0.5, overlay, 0.5, 0)",
                        _code("img, overlay", '    return {"blend": cv2.addWeighted(img, 0.5, overlay, 0.5, 0)}\n'),
                        "error", "output array type must be explicitly specified"),
        "forgot_cast_back": Wrong("converts img to float32 and returns the float32 blend",
                                  _code("img, overlay",
                                        '    return {"blend": cv2.addWeighted(img.astype(np.float32), 0.5, overlay, 0.5, 0)}\n'),
                                  "verifier"),
    })


# ---------------------------------------------------------------- 7. remap maps must be float32

# calibrated on 50 smooth scenes: maps computed in float32 arithmetic, and a pure-numpy float64 bilinear sampler,
# both differ from the reference by <= 1 gray level (remap's 1/32-px fixed-point interpolation); 2 keeps a 2x margin.
# The image is smooth on purpose: on hard edges that quantization alone would move a pixel by tens of levels.
RM_TOL = 2


def smooth_image(seed: int, h: int = 64, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w))
    for _ in range(4):
        fx, fy, ph = rng.uniform(0.02, 0.09), rng.uniform(0.02, 0.09), rng.uniform(0, 6.28)
        img += np.sin(2 * math.pi * (fx * x + fy * y) + ph)
    return np.round(127.5 + 127.5 * img / 4).astype(np.uint8)


def _rm_truth(img: np.ndarray, amp: float, period: float) -> np.ndarray:
    h, w = img.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mx = (xs + amp * np.sin(2 * np.pi * ys / period)).astype(np.float32)
    return cv2.remap(img, mx, ys.astype(np.float32), cv2.INTER_LINEAR)


def _rm_cases():
    out = []
    for seed, amp, period in ((261, 3.7, 40.0), (262, 2.2, 25.0)):
        img = smooth_image(seed)
        out.append(Case({"img": img, "amplitude": amp, "period": period}, _rm_truth(img, amp, period)))
    return out


def _rm_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "warped").astype(np.int16)
    if got.shape != case.truth.shape:
        return fail(f"warped has shape {got.shape}")
    err = np.abs(got - case.truth.astype(np.int16)).max()
    return fail(f"warped image differs from the expected sampling by up to {err} gray levels") if err > RM_TOL else PASS


REMAP_MAPS = Check(
    id="dtype_remap_maps", group=GROUP, probes=("remap_float64_maps",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.remap needs float32 maps; the float64 (or int64) arrays numpy builds raise",
    spec=_spec("Apply a horizontal wobble: output pixel (x, y) is the input sampled bilinearly at "
               "(x + amplitude * sin(2 * pi * y / period), y). Samples that fall outside the image are 0.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image"),
                "amplitude": IOSpec("float", "scalar", None, "px"),
                "period": IOSpec("float", "scalar", None, "px")},
               {"warped": IOSpec("uint8", "HxW", 2, "the distorted image")}),
    make_cases=_rm_cases, verify=_rm_verify,
    reference=_code("img, amplitude, period",
                    '    h, w = img.shape\n    xs, ys = np.meshgrid(np.arange(w), np.arange(h))\n'
                    '    map_x = (xs + amplitude * np.sin(2 * np.pi * ys / period)).astype(np.float32)\n'
                    '    return {"warped": cv2.remap(img, map_x, ys.astype(np.float32), cv2.INTER_LINEAR)}\n'),
    wrong={
        "float64_maps": Wrong("maps left as the float64 numpy produced",
                              _code("img, amplitude, period",
                                    '    h, w = img.shape\n    xs, ys = np.meshgrid(np.arange(w), np.arange(h))\n'
                                    '    map_x = xs + amplitude * np.sin(2 * np.pi * ys / period)\n'
                                    '    return {"warped": cv2.remap(img, map_x, ys.astype(np.float64), cv2.INTER_LINEAR)}\n'),
                              "error", "map1.type() == CV_32FC2"),
        "int_y_map": Wrong("float32 x map but the int64 meshgrid y map: 5.0.0 does NOT raise here, it reinterprets "
                           "the y map's bytes as float32 (measured: the output depends on map2's bytes, not its values)",
                           _code("img, amplitude, period",
                                 '    h, w = img.shape\n    xs, ys = np.meshgrid(np.arange(w), np.arange(h))\n'
                                 '    map_x = (xs + amplitude * np.sin(2 * np.pi * ys / period)).astype(np.float32)\n'
                                 '    return {"warped": cv2.remap(img, map_x, ys, cv2.INTER_LINEAR)}\n'),
                           "verifier"),
    })


# ---------------------------------------------------------------- 8. bool masks (5.0): accepted by some functions, bitwise_not is a trap

def _bm_cases():
    out = []
    for seed in (271, 272):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        mask = cv2.GaussianBlur(rng.random((24, 32)).astype(np.float32), (0, 0), 2.5) > 0.5
        out.append(Case({"img": img, "mask": mask}, (img * mask[..., None], int(mask.sum()), ~mask)))
    return out


def _bm_verify(case: Case, out: dict) -> Verdict:
    kept, count, inverse = case.truth
    if not np.array_equal(arr(out, "kept"), kept):
        return fail("kept image is not the input with the background zeroed")
    if out["count"] != count:
        return fail(f"count {out['count']}, expected {count}")
    inv = arr(out, "inverse").view(np.uint8) != 0  # read the raw bytes: a bool array may hold 0xFE/0xFF
    if inv.shape != inverse.shape or not np.array_equal(inv, inverse):
        return fail(f"inverse mask is wrong in {int((inv != inverse).sum())} pixels "
                    f"(it marks {int(inv.sum())} of {inv.size} pixels True, the truth is {int(inverse.sum())})")
    return PASS


_BM_SIG = "img, mask"
BOOL_MASK_OPS = Check(
    id="dtype_bool_mask_ops", group=GROUP, probes=("bool_array_input", "bitwise_and_bool_mask", "new:bitwise_not_bool"),
    depends_on_5_0=True, role="gotcha",
    gotcha="5.0 accepts a bool mask in bitwise_and(mask=) and countNonZero, but cv2.bitwise_not(bool_mask) returns "
           "all True (bytes 0xFE/0xFF); on a 0/1 uint8 mask bitwise_not gives 254/255",
    spec=_spec("Black out everything outside the foreground mask, count the foreground pixels, and return the "
               "background mask (the logical inverse of the foreground mask).",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
                "mask": IOSpec("bool", "HxW", 2, "True where foreground")},
               {"kept": IOSpec("uint8", "HxWx3", 3, "img where mask is True, 0 elsewhere"),
                "count": IOSpec("int", "scalar", None, "number of True pixels in mask"),
                "inverse": IOSpec("bool", "HxW", 2, "True where mask is False")}),
    make_cases=_bm_cases, verify=_bm_verify,
    reference=_code(_BM_SIG, '    kept = cv2.bitwise_and(img, img, mask=mask)\n    count = cv2.countNonZero(mask)\n'
                             '    return {"kept": kept, "count": count, "inverse": np.logical_not(mask)}\n'),
    note="the reference hands the bool mask to cv2 directly (4.14 raises there); only bitwise_not is avoided",
    wrong={
        "bitwise_not_bool": Wrong("inverse = cv2.bitwise_not(mask) on the bool mask",
                                  _code(_BM_SIG, '    kept = cv2.bitwise_and(img, img, mask=mask)\n'
                                                 '    return {"kept": kept, "count": cv2.countNonZero(mask),\n'
                                                 '            "inverse": cv2.bitwise_not(mask)}\n'), "verifier"),
        "bitwise_not_01": Wrong("4.x habit: mask.astype(np.uint8) (0/1), then cv2.bitwise_not -> 254/255",
                                _code(_BM_SIG, '    m = mask.astype(np.uint8)\n'
                                               '    kept = cv2.bitwise_and(img, img, mask=m)\n'
                                               '    inverse = cv2.bitwise_not(m).astype(bool)\n'
                                               '    return {"kept": kept, "count": cv2.countNonZero(m), "inverse": inverse}\n'),
                                "verifier"),
    })


# ---------------------------------------------------------------- 9. bool masks (5.0 counter-case): blobs still need uint8

def _bb_cases():
    out = []
    for seed in (281, 282):
        rng = np.random.default_rng(seed)
        m = np.zeros((60, 90), bool)
        k = int(rng.integers(3, 6))
        areas = []
        for i in range(k):
            w, h = int(rng.integers(6, 14)), int(rng.integers(6, 40))
            x0, y0 = 4 + i * 17, int(rng.integers(2, 60 - h - 2))
            m[y0:y0 + h, x0:x0 + w] = True
            areas.append(w * h)
        out.append(Case({"mask": m}, (k, max(areas))))
    return out


def _bb_verify(case: Case, out: dict) -> Verdict:
    n, largest = case.truth
    if out["n_blobs"] != n or out["largest"] != largest:
        return fail(f"got {out['n_blobs']} blobs / largest {out['largest']}, expected {n} / {largest}")
    return PASS


BOOL_BLOBS = Check(
    id="dtype_bool_blobs", group=GROUP, probes=("bool_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="COUNTER-CASE: 5.0 still rejects a bool mask in connectedComponents* (-215 iDepth == CV_8U) and "
           "findContours (-210); it needs .astype(np.uint8)",
    spec=_spec("Count the separate blobs in the boolean mask (8-connectivity) and report the pixel area of "
               "the largest one.",
               {"mask": IOSpec("bool", "HxW", 2, "True where foreground; the blobs are solid, axis-aligned rectangles "
                                                  "at least 4 px apart")},
               {"n_blobs": IOSpec("int", "scalar", None, "number of blobs"),
                "largest": IOSpec("int", "scalar", None, "pixels in the largest blob")}),
    make_cases=_bb_cases, verify=_bb_verify,
    reference=_code("mask", '    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)\n'
                            '    return {"n_blobs": n - 1, "largest": int(stats[1:, cv2.CC_STAT_AREA].max())}\n'),
    wrong={
        "bool_to_components": Wrong("over-generalizes 'bool is accepted': connectedComponentsWithStats(mask)",
                                    _code("mask", '    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)\n'
                                                  '    return {"n_blobs": n - 1, "largest": int(stats[1:, cv2.CC_STAT_AREA].max())}\n'),
                                    "error", "iDepth == CV_8U || iDepth == CV_8S"),
        "bool_to_contours": Wrong("over-generalizes: findContours(mask)",
                                  _code("mask", '    cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                                '    return {"n_blobs": len(cs), "largest": int(max(cv2.contourArea(c) for c in cs))}\n'),
                                  "error", "Unsupported format"),
    })


# ---------------------------------------------------------------- 10. uint32 (5.0): cv2.add saturates natively

def _u32_cases():
    out = []
    for seed in (291, 292):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 2 ** 32, (6, 8), dtype=np.uint64).astype(np.uint32)
        b = rng.integers(0, 2 ** 32, (6, 8), dtype=np.uint64).astype(np.uint32)
        a[0, :3], b[0, :3] = 10, 20  # some sums that do not overflow
        out.append(Case({"a": a, "b": b}, np.minimum(a.astype(np.uint64) + b, 2 ** 32 - 1).astype(np.uint32)))
    return out


def _u32_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "total")
    if not np.array_equal(got, case.truth):
        return fail(f"{int((got != case.truth).sum())} of {got.size} saturating sums are wrong")
    return PASS


UINT32_ADD = Check(
    id="dtype_uint32_add", group=GROUP, probes=("uint32_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="5.0 accepts uint32 in cv2.add (saturating at 2**32 - 1); the 4.x workarounds (int32 or float32 cast, "
           "numpy +) wrap or lose precision",
    spec=_spec("Add two uint32 counter images elementwise, saturating at the largest uint32 (2**32 - 1) "
               "instead of wrapping around.",
               {"a": IOSpec("uint32", "HxW", 2, "values up to 2**32 - 1"), "b": IOSpec("uint32", "HxW", 2, "values up to 2**32 - 1")},
               {"total": IOSpec("uint32", "HxW", 2, "saturating sum")}),
    make_cases=_u32_cases, verify=_u32_verify,
    reference=_code("a, b", '    return {"total": cv2.add(a, b)}\n'),
    note="scoped to cv2.add: GaussianBlur, linear resize, reduce and threshold still reject uint32 on 5.0.0",
    wrong={
        "int32_cast": Wrong("4.x workaround: cast to int32 (values above 2**31 turn negative)",
                            _code("a, b", '    s = cv2.add(a.astype(np.int32), b.astype(np.int32))\n'
                                          '    return {"total": s.astype(np.uint32)}\n'), "verifier"),
        "float32_cast": Wrong("4.x workaround: cast to float32 (24-bit mantissa)",
                              _code("a, b", '    s = cv2.add(a.astype(np.float32), b.astype(np.float32))\n'
                                            '    return {"total": np.clip(s, 0, 2 ** 32 - 1).astype(np.uint64).astype(np.uint32)}\n'),
                              "verifier"),
        "numpy_wraps": Wrong("a + b in numpy: wraps modulo 2**32 instead of saturating",
                             _code("a, b", '    return {"total": a + b}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 11. float16 (5.0): add accepted, GaussianBlur is not

# calibrated on 50 scenes: the reference (float16 addWeighted, float32 blur, cast back) is within 1.0 ulp of the
# all-float32 pipeline; a numpy float16 average gives the same worst case. 2 ulp keeps a 2x margin.
F16_ULP = 2.0


def _f16_cases():
    out = []
    for seed in (301, 302):
        rng = np.random.default_rng(seed)
        a = (rng.random((48, 64)) * 100).astype(np.float16)
        b = (rng.random((48, 64)) * 100).astype(np.float16)
        avg = (a.astype(np.float32) + b.astype(np.float32)) / 2
        out.append(Case({"a": a, "b": b}, cv2.GaussianBlur(avg, (5, 5), 0).astype(np.float16)))
    return out


def _f16_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "smoothed")
    if got.shape != case.truth.shape:
        return fail(f"smoothed has shape {got.shape}")
    ulp = np.spacing(np.abs(case.truth)).astype(np.float64)
    err = (np.abs(got.astype(np.float64) - case.truth.astype(np.float64)) / np.maximum(ulp, 1e-9)).max()
    return fail(f"smoothed differs from the float32 pipeline by {err:.1f} float16 ulp (allowed {F16_ULP})") if err > F16_ULP else PASS


FLOAT16_PIPELINE = Check(
    id="dtype_float16_pipeline", group=GROUP, probes=("float16_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="5.0 accepts float16 in cv2.add/addWeighted, but GaussianBlur (and most filters) still reject it "
           "(ktype == CV_32F || CV_64F); it must go through float32",
    spec=_spec("Average the two float16 frames, then smooth the average with a 5x5 Gaussian blur. Keep the result "
               "in float16.",
               {"a": IOSpec("float16", "HxW", 2, "frame"), "b": IOSpec("float16", "HxW", 2, "frame")},
               {"smoothed": IOSpec("float16", "HxW", 2, "blurred average")}),
    make_cases=_f16_cases, verify=_f16_verify,
    reference=_code("a, b", '    avg = cv2.addWeighted(a, 0.5, b, 0.5, 0)\n'
                            '    blurred = cv2.GaussianBlur(avg.astype(np.float32), (5, 5), 0)\n'
                            '    return {"smoothed": blurred.astype(np.float16)}\n'),
    note="scoped to addWeighted + GaussianBlur: float16 is rejected by resize, cvtColor, Canny, warp*, remap, ... too",
    wrong={
        "blur_float16_direct": Wrong("over-generalizes 'float16 is accepted': GaussianBlur on the float16 array",
                                     _code("a, b", '    avg = cv2.addWeighted(a, 0.5, b, 0.5, 0)\n'
                                                   '    return {"smoothed": cv2.GaussianBlur(avg, (5, 5), 0)}\n'),
                                     "error", "CV_16F"),
        "returns_float32": Wrong("goes through float32 and forgets to cast back",
                                 _code("a, b", '    avg = cv2.addWeighted(a, 0.5, b, 0.5, 0)\n'
                                               '    return {"smoothed": cv2.GaussianBlur(avg.astype(np.float32), (5, 5), 0)}\n'),
                                 "verifier"),
    })


CHECKS = [INT64_ADD, INT64_THRESHOLD, PERSPECTIVE_POINTS, PERSPECTIVE_NX2, FILLPOLY, ADDWEIGHTED_MIXED,
          REMAP_MAPS, BOOL_MASK_OPS, BOOL_BLOBS, UINT32_ADD, FLOAT16_PIPELINE]
