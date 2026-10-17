"""Group 1, output shapes: 9 functions whose 5.0 return value lost an axis, plus 5 controls.

In 5.0 a handful of OpenCV functions return arrays with the singleton middle axis removed
(``HoughLinesP`` (N,1,4) -> (N,4), ``findNonZero`` (N,1,2) -> (N,2), ...). The 4.x idioms
``lines[:, 0, :]``, ``for s, e, f, d in defects[:, 0]``, ``ids[i][0]`` then raise or silently
produce the wrong shape. The version-proof idiom is ``.reshape(-1, k)``.

The change is a set of point defects, not a rule (PLAN.md section 0). The five *controls* are
functions that did NOT change: a model that over-applies "OpenCV 5 dropped the middle axis"
writes ``for x, y in corners`` for ``goodFeaturesToTrack`` and crashes. A control fails only
the over-applied variants; the 4.x idiom is correct there.

Verifiers judge values against a manufactured truth with tolerances measured on the reference
(see each ``# calibrated`` comment). Detection quality is not the point of these checks, so
every verifier is lenient about WHICH parameters were used and strict about shape, dtype and
the recovered geometry.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import (PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail, match_points,
                   point_line_distance)

GROUP = "shapes"


def _spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="gotcha", task=task, inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------- shared scenes

def line_scene(seed: int, k: int = 3, H: int = 140, W: int = 180):
    """k bright 2 px lines, each at least 100 px long, one per horizontal band (so they never merge)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    gt = []
    band = H // k
    for i in range(k):
        lo, hi = i * band + 12, (i + 1) * band - 12
        while True:
            x1, x2 = int(rng.integers(8, 40)), int(rng.integers(W - 40, W - 8))
            y1, y2 = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
            if math.hypot(x2 - x1, y2 - y1) >= 100:
                break
        cv2.line(img, (x1, y1), (x2, y2), 255, 2)
        gt.append(((x1, y1), (x2, y2)))
    return img, gt


def _near_normal_form(rho: float, theta: float, a, b, tol: float) -> bool:
    """Both endpoints of the manufactured segment lie within tol px of the line x cos t + y sin t = rho."""
    c, s = math.cos(theta), math.sin(theta)
    return all(abs(p[0] * c + p[1] * s - rho) <= tol for p in (a, b))


def _covered_fraction(segs: np.ndarray, a, b, tol: float) -> float:
    """Fraction of the manufactured segment a-b covered by the union of the returned segments lying on it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / length
    spans = []
    for s in segs:
        p, q = s[:2].astype(float), s[2:].astype(float)
        if point_line_distance(p, a, b) <= tol and point_line_distance(q, a, b) <= tol:
            t0, t1 = sorted((float(np.dot(p - a, d)), float(np.dot(q - a, d))))
            spans.append((max(t0, 0.0), min(t1, length)))
    covered, cur = 0.0, 0.0
    for t0, t1 in sorted(spans):
        t0 = max(t0, cur)
        if t1 > t0:
            covered += t1 - t0
            cur = t1
    return covered / length


# ---------------------------------------------------------------- 1. HoughLinesP  (N,1,4) -> (N,4)

_HLP_REF = '''\
import cv2
import numpy as np


def solve(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)
    return {"segments": lines.reshape(-1, 4)}
'''
_HLP_HEAD = '''\
import cv2
import numpy as np


def solve(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)
'''

# calibrated: 3200 (threshold x minLineLength x maxLineGap) configurations over 40 scenes: endpoints
# within 3 px of a manufactured line always; every line >= 50 % covered in 3198/3200 (the two misses
# are maxLineGap=2). >= 80 % holds for the reference parameters on 200 scenes.
HLP_TOL, HLP_COVER = 3.0, 0.5


def _hlp_cases():
    out = []
    for seed in (11, 12, 13):
        img, gt = line_scene(seed)
        out.append(Case({"edges": img}, gt))
    return out


def _hlp_verify(case: Case, out: dict) -> Verdict:
    segs = arr(out, "segments")
    if len(segs) == 0:
        return fail("no segments returned")
    for s in segs:
        for p in (s[:2], s[2:]):
            if min(point_line_distance(p, a, b) for a, b in case.truth) > HLP_TOL:
                return fail(f"endpoint {p.tolist()} is not on any manufactured line (> {HLP_TOL} px)")
    for a, b in case.truth:
        frac = _covered_fraction(segs, a, b, HLP_TOL)
        if frac < HLP_COVER:
            return fail(f"line {a}-{b} only {frac:.0%} covered by the returned segments")
    return PASS


HOUGHLINESP = Check(
    id="shape_houghlinesp", group=GROUP, probes=("HoughLinesP",), depends_on_5_0=True, role="gotcha",
    gotcha="HoughLinesP returns (N, 4) in 5.0, was (N, 1, 4): line[0] and lines[:, 0, :] break",
    spec=_spec("Detect the straight line segments in this binary edge image and return their endpoints, "
               "one row per segment.",
               {"edges": IOSpec("uint8", "HxW", 2, "0/255 edge map; every line is at least 100 px long and the lines are far apart")},
               {"segments": IOSpec("int", "Nx4", 2, "one row (x1, y1, x2, y2) per detected segment, in pixels")}),
    make_cases=_hlp_cases, verify=_hlp_verify, reference=_HLP_REF,
    wrong={
        "legacy_index": Wrong("4.x: lines[:, 0, :] to drop the middle axis", _HLP_HEAD +
                              '    return {"segments": lines[:, 0, :]}\n', "IndexError", "too many indices"),
        "legacy_unpack": Wrong("4.x loop: x1, y1, x2, y2 = line[0]", _HLP_HEAD +
                               '    segs = []\n    for line in lines:\n        x1, y1, x2, y2 = line[0]\n'
                               '        segs.append((x1, y1, x2, y2))\n    return {"segments": np.array(segs)}\n',
                               "TypeError", "cannot unpack"),
        "legacy_first_item": Wrong("4.x: np.array([l[0] for l in lines]) collapses to 1-D on 5.0", _HLP_HEAD +
                                   '    return {"segments": np.array([l[0] for l in lines])}\n', "verifier"),
    })


# ---------------------------------------------------------------- 2. findNonZero  (N,1,2) -> (N,2)

_FNZ_REF = '''\
import cv2
import numpy as np


def solve(mask):
    pts = cv2.findNonZero(mask)
    return {"points": pts.reshape(-1, 2)}
'''
_FNZ_HEAD = '''\
import cv2
import numpy as np


def solve(mask):
    pts = cv2.findNonZero(mask)
'''


def _fnz_cases():
    out = []
    for seed in (21, 22, 23):
        rng = np.random.default_rng(seed)
        m = np.zeros((60, 80), np.uint8)
        for _ in range(3):
            x, y = int(rng.integers(5, 55)), int(rng.integers(5, 40))
            cv2.circle(m, (x, y), int(rng.integers(3, 9)), 255, -1)
        m[rng.random(m.shape) < 0.01] = 255
        out.append(Case({"mask": m}, np.argwhere(m > 0)[:, ::-1]))
    return out


def _fnz_verify(case: Case, out: dict) -> Verdict:
    pts = arr(out, "points")
    if pts.shape != case.truth.shape or not np.array_equal(pts, case.truth):
        return fail(f"points differ from the {len(case.truth)} non-zero pixels in raster order "
                    f"(got shape {pts.shape})")
    return PASS


FINDNONZERO = Check(
    id="shape_findnonzero", group=GROUP, probes=("findNonZero",), depends_on_5_0=True, role="gotcha",
    gotcha="findNonZero returns (N, 2) in 5.0, was (N, 1, 2): pts[:, 0, 0] and pts[:, 0] break",
    spec=_spec("List the coordinates of every non-zero pixel of the mask.",
               {"mask": IOSpec("uint8", "HxW", 2, "0/255 mask with at least one non-zero pixel")},
               {"points": IOSpec("int", "Nx2", 2, "(x, y) of each non-zero pixel, in raster order "
                                                   "(row by row, left to right)")}),
    make_cases=_fnz_cases, verify=_fnz_verify, reference=_FNZ_REF,
    wrong={
        "legacy_index": Wrong("4.x: xs = pts[:, 0, 0]; ys = pts[:, 0, 1]", _FNZ_HEAD +
                              '    xs, ys = pts[:, 0, 0], pts[:, 0, 1]\n'
                              '    return {"points": np.stack([xs, ys], axis=1)}\n', "IndexError", "too many indices"),
        "legacy_squeeze": Wrong("4.x: pts[:, 0] to get (N, 2); on 5.0 it keeps only x", _FNZ_HEAD +
                                '    return {"points": pts[:, 0]}\n', "verifier"),
    })


# ---------------------------------------------------------------- 3. convexityDefects  (K,1,4) -> (K,4)

_CD_REF = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull).reshape(-1, 4)
    depth = defects[:, 3] / 256.0
    keep = depth > 3.0
    order = np.argsort(-depth[keep])
    points = cnt.reshape(-1, 2)[defects[keep, 2]][order]
    return {"points": points, "depths": depth[keep][order]}
'''
_CD_HEAD = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
'''

# calibrated on 300 star scenes x {CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE}: the reference's deepest point is within
# 2.00 px of the manufactured inner vertex (the contour tip is a pixel or two off the polygon vertex) and its depth
# within 1.34 px of the distance from the drawn vertex to the hull edge; no shallow (<= 3 px) defects occur at all
# and the depth gaps are >= 4.1 px. Tolerances carry a 1.5x margin over those worst cases.
CD_POINT_TOL, CD_DEPTH_TOL = 3.0, 2.0


def star_scene(seed: int):
    """A star-shaped polygon with n notches of distinct depths. Truth: each notch's inner vertex and its
    depth, the distance from that vertex to the hull edge joining the two neighbouring tips.

    The depths are chosen first (12 px + 6.5 px x a random rank, +-0.8 px) and each inner vertex is placed that far
    from the chord between its neighbouring tips, towards the centre; the truth is then recomputed from the rounded
    integer vertices that are actually drawn. Gaps between depths stay >= 3.5 px, more than twice the largest
    depth error measured on the reference (see CD_DEPTH_TOL), so the deepest-first order is unambiguous.
    """
    rng = np.random.default_rng(seed)
    centre = np.array([100.0, 100.0])
    while True:
        n = int(rng.integers(4, 7))
        R = rng.uniform(65, 80)
        a0 = rng.uniform(0, 2 * math.pi / n)
        tips = centre + R * np.c_[np.cos(a0 + 2 * math.pi * np.arange(n) / n), np.sin(a0 + 2 * math.pi * np.arange(n) / n)]
        want = 12 + 6.5 * rng.permutation(n) + rng.uniform(-0.8, 0.8, n)
        inner = []
        for k in range(n):
            mid = (tips[k] + tips[(k + 1) % n]) / 2
            u = (centre - mid) / np.linalg.norm(centre - mid)
            inner.append(mid + u * want[k])
        tips_i = [tuple(int(round(v)) for v in p) for p in tips]
        inner_i = [tuple(int(round(v)) for v in p) for p in inner]
        depths = [point_line_distance(inner_i[k], tips_i[k], tips_i[(k + 1) % n]) for k in range(n)]
        if min(np.diff(sorted(depths))) >= 3.5:
            break
    poly = []
    for k in range(n):
        poly += [tips_i[k], inner_i[k]]
    mask = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    order = np.argsort(depths)[::-1]
    return mask, np.array([inner_i[k] for k in order]), np.array([depths[k] for k in order])


def _cd_cases():
    out = []
    for seed in (31, 32, 33):
        mask, pts, depths = star_scene(seed)
        out.append(Case({"mask": mask}, (pts, depths)))
    return out


def _cd_verify(case: Case, out: dict) -> Verdict:
    pts_gt, depth_gt = case.truth
    pts, depths = arr(out, "points"), arr(out, "depths")
    if len(depths) != len(depth_gt) or len(pts) != len(pts_gt):
        return fail(f"{len(depths)} defects reported, the shape has {len(depth_gt)} notches deeper than 3 px")
    if len(depths) > 1 and np.any(np.diff(depths) > 1e-6):
        return fail("defects are not sorted deepest first")
    if np.abs(depths - depth_gt).max() > CD_DEPTH_TOL:
        return fail(f"depths {np.round(depths, 1).tolist()} vs manufactured {np.round(depth_gt, 1).tolist()} "
                    f"(tolerance {CD_DEPTH_TOL} px)")
    if np.abs(pts - pts_gt).max() > CD_POINT_TOL:
        return fail(f"deepest points {pts.tolist()} vs manufactured {pts_gt.tolist()}")
    return PASS


CONVEXITYDEFECTS = Check(
    id="shape_convexitydefects", group=GROUP, probes=("convexityDefects",), depends_on_5_0=True, role="gotcha",
    gotcha="convexityDefects returns (K, 4) in 5.0, was (K, 1, 4): defects[:, 0] and defects[i, 0] break",
    spec=_spec("Find the convexity defects of the single non-convex blob in the mask. Report the deepest point "
               "and the depth of every defect deeper than 3 px, deepest first.",
               {"mask": IOSpec("uint8", "HxW", 2, "0/255; one star-shaped blob")},
               {"points": IOSpec("int", "Kx2", 2, "(x, y) of the deepest point of each defect, deepest defect first"),
                "depths": IOSpec("float", "K", 1, "defect depth in px (not OpenCV's fixed-point value), deepest first")}),
    make_cases=_cd_cases, verify=_cd_verify, reference=_CD_REF,
    wrong={
        "legacy_loop": Wrong("4.x: for s, e, f, d in defects[:, 0]", _CD_HEAD +
                             '    pts, depths = [], []\n    for s, e, f, d in defects[:, 0]:\n'
                             '        if d / 256.0 > 3.0:\n            pts.append(cnt[f][0])\n            depths.append(d / 256.0)\n'
                             '    return {"points": np.array(pts), "depths": np.array(depths)}\n',
                             "TypeError", "cannot unpack"),
        "fixed_point": Wrong("forgets that OpenCV's defect depth is fixed-point (x 256)", _CD_HEAD +
                             '    defects = defects.reshape(-1, 4)\n    keep = defects[:, 3] > 3.0 * 256\n'
                             '    order = np.argsort(-defects[keep, 3])\n'
                             '    points = cnt.reshape(-1, 2)[defects[keep, 2]][order]\n'
                             '    return {"points": points, "depths": defects[keep, 3][order].astype(np.float64)}\n',
                             "verifier"),
    })


# ---------------------------------------------------------------- 4. calcHist  (256,1) -> (256,)

_CH_REF = '''\
import cv2
import numpy as np


def solve(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    return {"hist": hist, "peak": int(np.argmax(hist))}
'''
_CH_HEAD = '''\
import cv2
import numpy as np


def solve(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
'''


def _ch_cases():
    out = []
    for seed in (41, 42, 43):
        rng = np.random.default_rng(seed)
        g = np.clip(rng.normal(rng.integers(60, 200), 25, (64, 96)), 0, 255).astype(np.uint8)
        g[:8] = int(rng.integers(0, 256))  # a flat band makes the peak unique
        out.append(Case({"gray": g}, np.bincount(g.ravel(), minlength=256)))
    return out


def _ch_verify(case: Case, out: dict) -> Verdict:
    hist = arr(out, "hist")
    if hist.shape != (256,) or not np.array_equal(hist, case.truth):
        return fail("histogram differs from the exact per-level pixel counts")
    if out["peak"] != int(np.argmax(case.truth)):
        return fail(f"peak {out['peak']}, expected {int(np.argmax(case.truth))}")
    return PASS


CALCHIST = Check(
    id="shape_calchist", group=GROUP, probes=("calcHist",), depends_on_5_0=True, role="gotcha",
    gotcha="calcHist returns (256,) in 5.0, was (256, 1): hist[i][0] and hist[:, 0] break",
    spec=_spec("Compute the 256-bin histogram of a grayscale image and the gray level that occurs most often.",
               {"gray": IOSpec("uint8", "HxW", 2, "grayscale image")},
               {"hist": IOSpec("float", "256", 1, "number of pixels at each gray level 0..255"),
                "peak": IOSpec("int", "scalar", None, "the most frequent gray level")}),
    make_cases=_ch_cases, verify=_ch_verify, reference=_CH_REF,
    wrong={
        "legacy_bin_index": Wrong("4.x: counts = [h[0] for h in hist] (each bin was a length-1 row)", _CH_HEAD +
                                  '    counts = np.array([h[0] for h in hist])\n'
                                  '    return {"hist": counts, "peak": int(np.argmax(counts))}\n',
                                  "IndexError", "invalid index to scalar"),
        "legacy_column": Wrong("4.x: hist[:, 0] to get a flat vector", _CH_HEAD +
                               '    flat = hist[:, 0]\n    return {"hist": flat, "peak": int(np.argmax(flat))}\n',
                               "IndexError", "too many indices"),
    })


# ---------------------------------------------------------------- 5/6. chessboard finders  (N,1,2) -> (N,2)

# calibrated on 144 boards (4 sizes x 3 square sizes x rotation 0/6/-5 deg x blur 0..1.5 px), both finders found every
# board: worst corner error 0.75 px (classic) and 0.39 px (SB); on the six boards the checks use, worst 0.35 px.
# A wrong detection is off by half a square (>= 10 px), so 1.0 px separates the two cleanly.
CB_TOL = 1.0


def board(cols: int, rows: int, sq: int = 20, margin: int = 24, angle: float = 0.0, blur: float = 0.0):
    """Gray image of a (cols+1) x (rows+1)-square board; truth = inner corners in pixel-centre coordinates."""
    H, W = (rows + 1) * sq + 2 * margin, (cols + 1) * sq + 2 * margin
    img = np.full((H, W), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    gt = np.array([[margin + (c + 1) * sq - 0.5, margin + (r + 1) * sq - 0.5]
                   for r in range(rows) for c in range(cols)], np.float64)
    if angle:
        M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
        gt = gt @ M[:, :2].T + M[:, 2]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img, gt


def _cb_cases(boards):
    out = []
    for cols, rows, sq, angle, blur in boards:
        img, gt = board(cols, rows, sq=sq, angle=angle, blur=blur)
        out.append(Case({"img": img, "cols": cols, "rows": rows}, gt))
    return out


def _cb_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "corners")
    why = match_points(case.truth, got, CB_TOL)
    return fail(why) if why else PASS


_CB_SPEC_IN = {"img": IOSpec("uint8", "HxW", 2, "grayscale image of a flat chessboard filling most of the frame"),
               "cols": IOSpec("int", "scalar", None, "inner corners per row"),
               "rows": IOSpec("int", "scalar", None, "inner corners per column")}
_CB_SPEC_OUT = {"corners": IOSpec("float", "Nx2", 2, "(x, y) of every inner corner, N = cols * rows, in any order")}


def _cb_code(fn: str, tail: str) -> str:
    return (f"import cv2\nimport numpy as np\n\n\ndef solve(img, cols, rows):\n"
            f"    found, corners = cv2.{fn}(img, (cols, rows))\n{tail}")


CHESSBOARD = Check(
    id="shape_chessboard", group=GROUP, probes=("findChessboardCorners",), depends_on_5_0=True, role="gotcha",
    gotcha="findChessboardCorners returns (N, 2) in 5.0, was (N, 1, 2)",
    spec=_spec("Find the inner corners of the chessboard in this image.", _CB_SPEC_IN, _CB_SPEC_OUT),
    make_cases=lambda: _cb_cases([(6, 4, 20, 0.0, 0.0), (7, 5, 24, 6.0, 0.8), (5, 4, 28, -5.0, 1.2)]),
    verify=_cb_verify,
    reference=_cb_code("findChessboardCorners", '    if not found:\n        return {"corners": np.zeros((0, 2), np.float32)}\n'
                                                '    return {"corners": corners.reshape(-1, 2)}\n'),
    wrong={
        "legacy_index": Wrong("4.x: corners[:, 0, :]", _cb_code("findChessboardCorners",
                              '    return {"corners": corners[:, 0, :]}\n'), "IndexError", "too many indices"),
        "legacy_unpack": Wrong("4.x loop: x, y = c[0]", _cb_code("findChessboardCorners",
                               '    pts = [tuple(c[0]) for c in corners]\n    return {"corners": np.array(pts, np.float32)}\n'),
                               "TypeError", "not iterable"),
    })

CHESSBOARD_SB = Check(
    id="shape_chessboard_sb", group=GROUP, probes=("findChessboardCornersSB",), depends_on_5_0=True, role="gotcha",
    gotcha="findChessboardCornersSB returns (N, 2) in 5.0, was (N, 1, 2)",
    spec=_spec("Find the inner corners of the chessboard in this blurry image with OpenCV's sector-based "
               "detector, which is more robust to blur than the classic one.", _CB_SPEC_IN, _CB_SPEC_OUT),
    make_cases=lambda: _cb_cases([(6, 4, 22, 0.0, 1.5), (7, 5, 20, 4.0, 1.0), (4, 3, 30, -6.0, 1.5)]),
    verify=_cb_verify,
    reference=_cb_code("findChessboardCornersSB", '    if not found:\n        return {"corners": np.zeros((0, 2), np.float32)}\n'
                                                   '    return {"corners": corners.reshape(-1, 2)}\n'),
    wrong={
        "legacy_index": Wrong("4.x: corners[:, 0, :]", _cb_code("findChessboardCornersSB",
                              '    return {"corners": corners[:, 0, :]}\n'), "IndexError", "too many indices"),
        "legacy_unpack": Wrong("4.x loop: x, y = c[0]", _cb_code("findChessboardCornersSB",
                               '    pts = [tuple(c[0]) for c in corners]\n    return {"corners": np.array(pts, np.float32)}\n'),
                               "TypeError", "not iterable"),
    })


# ---------------------------------------------------------------- 7. minEnclosingTriangle  (3,1,2) -> (3,2)

_MT_REF = '''\
import cv2
import numpy as np


def solve(pts):
    area, tri = cv2.minEnclosingTriangle(pts)
    return {"triangle": tri.reshape(3, 2), "area": float(area)}
'''
_MT_HEAD = '''\
import cv2
import numpy as np


def solve(pts):
    area, tri = cv2.minEnclosingTriangle(pts)
'''

# calibrated on 300 scenes: the reference returns the manufactured vertices to float32 precision (max error 0.0000 px)
# and the area to 7.5e-7 relative; the tolerances below are ~100x that.
MT_VERTEX_TOL, MT_AREA_REL = 0.01, 1e-4


def _tri_area(t: np.ndarray) -> float:
    t = np.asarray(t, np.float64)
    return 0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1]))


def _mt_cases():
    out = []
    for seed in (51, 52, 53):
        rng = np.random.default_rng(seed)
        while True:
            v = rng.uniform(10, 190, (3, 2))
            if _tri_area(v) > 3000:
                break
        w = rng.dirichlet(np.ones(3), size=25)  # interior points: convex combinations of the vertices
        pts = np.vstack([v, w @ v]).astype(np.float32)
        out.append(Case({"pts": pts}, v.astype(np.float32)))
    return out


def _mt_verify(case: Case, out: dict) -> Verdict:
    tri, area = arr(out, "triangle"), out["area"]
    if tri.shape != (3, 2):
        return fail(f"triangle has shape {tri.shape}")
    why = match_points(case.truth, tri, MT_VERTEX_TOL)
    if why:
        return fail("the minimum enclosing triangle of points that include a triangle's three vertices is that "
                    "triangle: " + why)
    true_area = _tri_area(case.truth)
    if abs(area - true_area) > MT_AREA_REL * true_area:
        return fail(f"area {area:.1f}, expected {true_area:.1f}")
    return PASS


MINENCLOSINGTRIANGLE = Check(
    id="shape_mintriangle", group=GROUP, probes=("minEnclosingTriangle",), depends_on_5_0=True, role="gotcha",
    gotcha="minEnclosingTriangle returns a (3, 2) triangle in 5.0, was (3, 1, 2)",
    spec=_spec("Find the smallest-area triangle that encloses all of these points.",
               {"pts": IOSpec("float32", "Nx2", 2, "(x, y) points")},
               {"triangle": IOSpec("float", "3x2", 2, "the three vertices (x, y)"),
                "area": IOSpec("float", "scalar", None, "area of the triangle in px^2")}),
    make_cases=_mt_cases, verify=_mt_verify, reference=_MT_REF,
    wrong={
        "legacy_index": Wrong("4.x: tri[:, 0, :]", _MT_HEAD +
                              '    return {"triangle": tri[:, 0, :], "area": float(area)}\n', "IndexError",
                              "too many indices"),
        "legacy_squeeze": Wrong("4.x: tri[:, 0] to get (3, 2); on 5.0 it keeps only x", _MT_HEAD +
                                '    return {"triangle": tri[:, 0], "area": float(area)}\n', "verifier"),
    })


# ---------------------------------------------------------------- 8. ArUco ids  (N,1) -> (N,)

_AR_HEAD = '''\
import cv2
import numpy as np


def solve(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(img)
'''


def aruco_scene(seed: int, k: int = 3):
    """k DICT_4X4_50 markers with distinct random ids on a white sheet. Truth: id and centre of each."""
    rng = np.random.default_rng(seed)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    size = int(rng.choice([80, 90, 100]))
    H, W = size + 80, k * (size + 30) + 30
    img = np.full((H, W), 255, np.uint8)
    ids = rng.choice(50, size=k, replace=False)
    markers = []
    for i, mid in enumerate(ids):
        x, y = 30 + i * (size + 30), int(rng.integers(20, 60))
        img[y:y + size, x:x + size] = cv2.aruco.generateImageMarker(dic, int(mid), size)
        markers.append((int(mid), (x + (size - 1) / 2, y + (size - 1) / 2)))
    return img, markers


def _ar_ids_cases():
    out = []
    for seed in (61, 62, 63):
        img, markers = aruco_scene(seed)
        out.append(Case({"img": img}, np.array(sorted(m[0] for m in markers))))
    return out


def _ar_ids_verify(case: Case, out: dict) -> Verdict:
    ids = arr(out, "ids")
    if ids.shape != case.truth.shape or not np.array_equal(ids, case.truth):
        return fail(f"ids {ids.tolist()} vs the markers on the sheet {case.truth.tolist()}")
    return PASS


ARUCO_IDS = Check(
    id="shape_aruco_ids", group=GROUP, probes=("aruco_detectMarkers",), depends_on_5_0=True, role="gotcha",
    gotcha="ArucoDetector.detectMarkers returns ids as (N,) in 5.0, was (N, 1): ids[i][0] and ids[:, 0] break",
    spec=_spec("Detect the ArUco markers (DICT_4X4_50 dictionary) in this image and return their ids in "
               "ascending order.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image: dark-bordered markers on a white sheet")},
               {"ids": IOSpec("int", "N", 1, "marker ids, ascending")}),
    make_cases=_ar_ids_cases, verify=_ar_ids_verify,
    reference=_AR_HEAD + '    ids = np.sort(ids.reshape(-1))\n    return {"ids": ids}\n',
    wrong={
        "legacy_index": Wrong("4.x: [int(i[0]) for i in ids]", _AR_HEAD +
                              '    return {"ids": np.array(sorted(int(i[0]) for i in ids))}\n', "IndexError",
                              "invalid index to scalar"),
        "legacy_column": Wrong("4.x: ids[:, 0]", _AR_HEAD +
                               '    return {"ids": np.sort(ids[:, 0])}\n', "IndexError", "too many indices"),
        "keeps_column": Wrong("returns the 4.x column vector (N, 1)", _AR_HEAD +
                              '    return {"ids": np.sort(ids.reshape(-1, 1), axis=0)}\n', "verifier"),
    })


# ---------------------------------------------------------------- 9. HoughLinesWithAccumulator  (N,1,3) -> (N,3)

_HA_REF = '''\
import cv2
import numpy as np


def solve(edges, min_votes):
    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi / 180, min_votes)
    return {"lines": lines.reshape(-1, 3)}
'''
_HA_HEAD = '''\
import cv2
import numpy as np


def solve(edges, min_votes):
    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi / 180, min_votes)
'''
LINE_MIN_VOTES = 80  # a 100+ px line of thickness 2 collects well over 100 votes at 1 px / 1 degree
NF_TOL = 3.0  # px: both endpoints of a manufactured line within 3 px of a returned (rho, theta) line


def _normal_cases(seeds):
    out = []
    for seed in seeds:
        img, gt = line_scene(seed)
        out.append(Case({"edges": img, "min_votes": LINE_MIN_VOTES}, gt))
    return out


def _normal_form_verify(case: Case, out: dict, key: str, votes_col: int | None) -> Verdict:
    lines = arr(out, key).astype(np.float64)
    if len(lines) == 0:
        return fail("no lines returned")
    if votes_col is not None:
        if lines[:, votes_col].min() < LINE_MIN_VOTES:
            return fail(f"a row has fewer than {LINE_MIN_VOTES} votes")
        if np.any(np.diff(lines[:, votes_col]) > 1e-6):
            return fail("rows are not ordered by votes, strongest first")
    r0, t0 = lines[0, 0], lines[0, 1]
    if not any(_near_normal_form(r0, t0, a, b, NF_TOL) for a, b in case.truth):
        return fail(f"the first row (rho={r0:.1f}, theta={t0:.3f}) is not one of the manufactured lines")
    for a, b in case.truth:
        if not any(_near_normal_form(r, t, a, b, NF_TOL) for r, t in lines[:, :2]):
            return fail(f"manufactured line {a}-{b} has no (rho, theta) row within {NF_TOL} px")
    return PASS


HOUGHLINES_ACC = Check(
    id="shape_houghlines_acc", group=GROUP, probes=("HoughLinesWithAccumulator",), depends_on_5_0=True,
    role="gotcha",
    gotcha="HoughLinesWithAccumulator returns (N, 3) in 5.0, was (N, 1, 3): line[0] and lines[:, 0, :] break",
    spec=_spec("Run the standard Hough line transform at 1 px / 1 degree resolution and return every line "
               "with its vote count, strongest first.",
               {"edges": IOSpec("uint8", "HxW", 2, "0/255 edge map; every line is at least 100 px long"),
                "min_votes": IOSpec("int", "scalar", None, "discard lines with fewer votes than this")},
               {"lines": IOSpec("float", "Nx3", 2, "one row (rho, theta, votes) per line, strongest first; "
                                                    "rho in px, theta in radians")}),
    make_cases=lambda: _normal_cases((71, 72, 73)),
    verify=lambda case, out: _normal_form_verify(case, out, "lines", 2), reference=_HA_REF,
    wrong={
        "legacy_unpack": Wrong("4.x: rho, theta, votes = line[0]", _HA_HEAD +
                               '    rows = []\n    for line in lines:\n        rho, theta, votes = line[0]\n'
                               '        rows.append((rho, theta, votes))\n    return {"lines": np.array(rows)}\n',
                               "TypeError", "cannot unpack"),
        "legacy_index": Wrong("4.x: lines[:, 0, :]", _HA_HEAD + '    return {"lines": lines[:, 0, :]}\n',
                              "IndexError", "too many indices"),
    })


# ---------------------------------------------------------------- controls (did not change in 5.0)

# 10. findContours keeps (N,1,2)

_CT_REF = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    pts = []
    for c in contours:
        p = c.reshape(-1, 2)
        pts.append(p[np.lexsort((p[:, 1], p[:, 0]))[0]])
    pts = sorted(pts, key=lambda q: (int(q[0]), int(q[1])))
    return {"points": np.array(pts)}
'''
_CT_HEAD = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    pts = []
'''


def blob_row_scene(seed: int):
    """3-5 separate blobs (rectangle, disc or triangle) side by side. Truth: the leftmost, then topmost,
    pixel of each blob, in increasing x."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 6))
    H, W = 120, 44 * k + 20
    mask = np.zeros((H, W), np.uint8)
    truth = []
    for i in range(k):
        one = np.zeros_like(mask)
        x0 = 10 + 44 * i + int(rng.integers(0, 6))
        y0 = int(rng.integers(14, 50))
        kind = int(rng.integers(3))
        if kind == 0:
            cv2.rectangle(one, (x0, y0), (x0 + int(rng.integers(14, 32)), y0 + int(rng.integers(14, 50))), 255, -1)
        elif kind == 1:
            r = int(rng.integers(8, 17))
            cv2.circle(one, (x0 + r, y0 + r), r, 255, -1)
        else:
            cv2.fillPoly(one, [np.array([[x0, y0 + 40], [x0 + 30, y0 + 20], [x0 + 12, y0]], np.int32)], 255)
        ys, xs = np.nonzero(one)
        lx = xs.min()
        truth.append((int(lx), int(ys[xs == lx].min())))
        mask |= one
    return mask, np.array(sorted(truth))


def _ct_cases():
    return [Case({"mask": m}, t) for m, t in (blob_row_scene(s) for s in (81, 82, 83))]


def _ct_verify(case: Case, out: dict) -> Verdict:
    pts = arr(out, "points")
    if pts.shape != case.truth.shape or not np.array_equal(pts, case.truth):
        return fail(f"leftmost points {pts.tolist()} vs manufactured {case.truth.tolist()}")
    return PASS


CONTOURS_CTRL = Check(
    id="shape_contours_ctrl", group=GROUP, probes=("findContours_tree", "findContours_approx_none"),
    depends_on_5_0=False, role="control",
    gotcha="CONTROL: findContours contours are still (N, 1, 2); treating them as (N, 2) breaks",
    spec=_spec("For each separate blob in the mask, find the leftmost point of its outline (the topmost one "
               "if several share the smallest x).",
               {"mask": IOSpec("uint8", "HxW", 2, "0/255; separate blobs that touch neither each other nor the border")},
               {"points": IOSpec("int", "Kx2", 2, "(x, y) per blob, ordered by increasing x")}),
    make_cases=_ct_cases, verify=_ct_verify, reference=_CT_REF,
    wrong={
        "overapplied_columns": Wrong("over-applies the lost-axis rule: x, y = c[:, 0], c[:, 1]", _CT_HEAD +
                                     '    for c in contours:\n        x, y = c[:, 0], c[:, 1]\n'
                                     '        i = np.lexsort((y, x))[0]\n        pts.append((x[i], y[i]))\n'
                                     '    return {"points": np.array(sorted(pts))}\n', "IndexError", "out of bounds"),
        "overapplied_unpack": Wrong("over-applies the rule: for x, y in c", _CT_HEAD +
                                    '    for c in contours:\n        best = min((x, y) for x, y in c)\n'
                                    '        pts.append(best)\n    return {"points": np.array(sorted(pts))}\n',
                                    "ValueError", "not enough values to unpack"),
    })

# 11. goodFeaturesToTrack keeps (N,1,2)

_GF_REF = '''\
import cv2
import numpy as np


def solve(img, max_corners):
    corners = cv2.goodFeaturesToTrack(img, max_corners, 0.05, 10)
    return {"corners": corners.reshape(-1, 2)}
'''
_GF_HEAD = '''\
import cv2
import numpy as np


def solve(img, max_corners):
    corners = cv2.goodFeaturesToTrack(img, max_corners, 0.05, 10)
'''
GF_TOL, GF_RECALL = 2.5, 0.9  # calibrated: see the sweep in rect_scene's docstring


def rect_scene(seed: int):
    """4-5 filled bright rectangles (each side >= 30 px, >= 24 px apart, contrast 150-235 gray levels). Truth: their
    4 corners each.

    Calibrated on 40 scenes x 12 settings (qualityLevel 0.01/0.05/0.1/0.2 x minDistance 5/10/20): the reference
    call finds every corner at the exact pixel (error 0.0 px) and returns nothing else; the settings sweep is
    re-run in tests/test_gotchas.py.
    """
    rng = np.random.default_rng(seed)
    img = np.full((160, 320), 20, np.uint8)
    k = int(rng.integers(4, 6))
    truth, w = [], 320 // k
    for i in range(k):
        x0 = i * w + int(rng.integers(6, 12))
        x1 = x0 + int(rng.integers(30, w - 30))
        y0 = int(rng.integers(8, 40))
        y1 = y0 + int(rng.integers(30, 100))
        cv2.rectangle(img, (x0, y0), (x1, y1), int(rng.integers(170, 256)), -1)
        truth += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    return img, np.array(truth, np.float64)


def _gf_cases():
    return [Case({"img": img, "max_corners": len(t)}, t) for img, t in (rect_scene(s) for s in (91, 92, 93))]


def _gf_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "corners").astype(np.float64)
    if len(got) == 0:
        return fail("no corners returned")
    d = np.linalg.norm(case.truth[:, None, :] - got[None, :, :], axis=2)
    if (d.min(axis=1) <= GF_TOL).mean() < GF_RECALL:
        return fail(f"only {(d.min(axis=1) <= GF_TOL).mean():.0%} of the rectangle corners were found")
    if (d.min(axis=0) <= GF_TOL).mean() < GF_RECALL:
        return fail("most returned points are not rectangle corners")
    return PASS


GFTT_CTRL = Check(
    id="shape_gftt_ctrl", group=GROUP, probes=("goodFeaturesToTrack",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: goodFeaturesToTrack still returns (N, 1, 2); `for x, y in corners` breaks",
    spec=_spec("Find the strongest corners in the image.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image of bright filled rectangles on a dark background"),
                "max_corners": IOSpec("int", "scalar", None, "return at most this many corners")},
               {"corners": IOSpec("float", "Nx2", 2, "(x, y) of each corner")}),
    make_cases=_gf_cases, verify=_gf_verify, reference=_GF_REF,
    wrong={
        "overapplied_unpack": Wrong("over-applies the rule: for x, y in corners", _GF_HEAD +
                                    '    pts = [(x, y) for x, y in corners]\n    return {"corners": np.array(pts, np.float32)}\n',
                                    "ValueError", "not enough values to unpack"),
        "overapplied_asis": Wrong("over-applies the rule: returns the (N, 1, 2) array as if it were (N, 2)", _GF_HEAD +
                                  '    return {"corners": corners}\n', "verifier"),
    })

# 12. approxPolyDP keeps (K,1,2)

_AP_REF = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    return {"vertices": approx.reshape(-1, 2)}
'''
_AP_HEAD = '''\
import cv2
import numpy as np


def solve(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
'''
# approxPolyDP keeps contour points, but WHICH point near a corner it keeps drifts by up to ~eps along the edges
# (worst 9.2 px against the manufactured vertices over 300 polygons, n = 3..8), so the verifier does not match
# vertices one-to-one. It uses the algorithm's own guarantee instead: the returned polygon has exactly n vertices,
# every returned vertex lies on the true outline (within AP_ON_EDGE px) and every true vertex is within
# eps + AP_ON_EDGE of the returned polygon, eps being 2 % of the true perimeter.
# Measured on the reference over 300 polygons, both chain modes: n correct every time, returned vertices
# 0.49 px from the outline at worst, true corners 1.04 eps from the returned polygon at worst. AP_ON_EDGE = 1.5
# is 3x the first number and covers the second (eps + 1.5 px > 1.04 eps).
AP_ON_EDGE = 1.5


def polygon_scene(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    R, a0 = rng.uniform(40, 60), rng.uniform(0, 2 * math.pi)
    verts = np.array([[80 + R * math.cos(a0 + 2 * math.pi * k / n), 80 + R * math.sin(a0 + 2 * math.pi * k / n)]
                      for k in range(n)])
    v = np.round(verts).astype(np.int32)
    mask = np.zeros((160, 160), np.uint8)
    cv2.fillPoly(mask, [v], 255)
    return mask, v


def _dist_to_polygon(p, verts) -> float:
    return min(point_line_distance(p, verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts)))


def _ap_cases():
    return [Case({"mask": m}, v) for m, v in (polygon_scene(s) for s in (101, 102, 103, 104))]


def _ap_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "vertices").astype(np.float64)
    truth = case.truth.astype(np.float64)
    if len(got) != len(truth):
        return fail(f"{len(got)} vertices for a {len(truth)}-gon")
    for p in got:
        if _dist_to_polygon(p, truth) > AP_ON_EDGE:
            return fail(f"vertex {p.tolist()} is not on the polygon's outline")
    perimeter = sum(np.linalg.norm(truth[i] - truth[(i + 1) % len(truth)]) for i in range(len(truth)))
    for p in truth:
        if _dist_to_polygon(p, got) > 0.02 * perimeter + AP_ON_EDGE:
            return fail(f"polygon corner {p.tolist()} is not approximated by the returned vertices")
    return PASS


APPROXPOLY_CTRL = Check(
    id="shape_approxpoly_ctrl", group=GROUP, probes=("approxPolyDP",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: approxPolyDP still returns (K, 1, 2); returning or unpacking it as (K, 2) breaks",
    spec=_spec("Outline the polygon with as few vertices as possible: simplify its contour with a tolerance "
               "of 2% of the contour's perimeter.",
               {"mask": IOSpec("uint8", "HxW", 2, "0/255; one filled convex polygon")},
               {"vertices": IOSpec("int", "Kx2", 2, "(x, y) of each polygon vertex, in any order")}),
    make_cases=_ap_cases, verify=_ap_verify, reference=_AP_REF,
    wrong={
        "overapplied_asis": Wrong("over-applies the rule: returns approx as if it were (K, 2)", _AP_HEAD +
                                  '    return {"vertices": approx}\n', "verifier"),
        "overapplied_unpack": Wrong("over-applies the rule: for x, y in approx", _AP_HEAD +
                                    '    return {"vertices": np.array([(int(x), int(y)) for x, y in approx])}\n',
                                    "ValueError", "not enough values to unpack"),
    })

# 13. HoughLines keeps (N,1,2)

_HL_REF = '''\
import cv2
import numpy as np


def solve(edges, min_votes):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
    return {"lines": lines.reshape(-1, 2)}
'''
_HL_HEAD = '''\
import cv2
import numpy as np


def solve(edges, min_votes):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
'''

HOUGHLINES_CTRL = Check(
    id="shape_houghlines_ctrl", group=GROUP, probes=("HoughLines",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: HoughLines still returns (N, 1, 2); `rho, theta = line` breaks",
    spec=_spec("Run the standard Hough line transform at 1 px / 1 degree resolution and return the lines it "
               "finds, strongest first.",
               {"edges": IOSpec("uint8", "HxW", 2, "0/255 edge map; every line is at least 100 px long"),
                "min_votes": IOSpec("int", "scalar", None, "discard lines with fewer votes than this")},
               {"lines": IOSpec("float", "Nx2", 2, "one row (rho, theta) per line, strongest first; "
                                                    "rho in px, theta in radians")}),
    make_cases=lambda: _normal_cases((111, 112, 113)),
    verify=lambda case, out: _normal_form_verify(case, out, "lines", None), reference=_HL_REF,
    wrong={
        "overapplied_unpack": Wrong("over-applies the rule: rho, theta = line", _HL_HEAD +
                                    '    rows = []\n    for line in lines:\n        rho, theta = line\n'
                                    '        rows.append((rho, theta))\n    return {"lines": np.array(rows)}\n',
                                    "ValueError", "not enough values to unpack"),
        "overapplied_asis": Wrong("over-applies the rule: returns the (N, 1, 2) array as (N, 2)", _HL_HEAD +
                                  '    return {"lines": lines}\n', "verifier"),
    })

# 14. ArUco corners keep (1,4,2) per marker

_AC_REF = '''\
import cv2
import numpy as np


def solve(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(img)
    centers = np.array([c.reshape(4, 2).mean(axis=0) for c in corners])
    return {"centers": centers[np.argsort(centers[:, 0])]}
'''
_AC_HEAD = '''\
import cv2
import numpy as np


def solve(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(img)
'''
AC_TOL = 1.0  # calibrated on 300 scenes: the reference's centres are exact (0.000 px); 1.0 px leaves room for corner refinement


def _ac_cases():
    out = []
    for seed in (121, 122, 123):
        img, markers = aruco_scene(seed)
        out.append(Case({"img": img}, np.array(sorted(m[1] for m in markers))))
    return out


def _ac_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "centers")
    if got.shape != case.truth.shape:
        return fail(f"centers have shape {got.shape}, expected {case.truth.shape}")
    err = np.abs(got - case.truth).max()
    if err > AC_TOL:
        return fail(f"centers {np.round(got, 1).tolist()} vs {np.round(case.truth, 1).tolist()} (left to right)")
    return PASS


ARUCO_CORNERS_CTRL = Check(
    id="shape_aruco_corners_ctrl", group=GROUP, probes=("aruco_detectMarkers",), depends_on_5_0=False,
    role="control",
    gotcha="CONTROL: only ids lost an axis; each marker's corners are still (1, 4, 2): c.mean(axis=0) is wrong",
    spec=_spec("Detect the ArUco markers (DICT_4X4_50 dictionary) in this image and return the centre of each "
               "marker, ordered left to right.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image: dark-bordered markers on a white sheet")},
               {"centers": IOSpec("float", "Nx2", 2, "(x, y) centre of each marker, increasing x")}),
    make_cases=_ac_cases, verify=_ac_verify, reference=_AC_REF,
    wrong={
        "overapplied_mean": Wrong("over-applies the rule: corners[i] is (4, 2), so c.mean(axis=0)", _AC_HEAD +
                                  '    centers = np.array([c.mean(axis=0) for c in corners])\n'
                                  '    return {"centers": centers[np.argsort(centers[:, 0, 0])]}\n', "verifier"),
        "overapplied_unpack": Wrong("over-applies the rule: for p0, p1, p2, p3 in c", _AC_HEAD +
                                    '    cs = []\n    for c in corners:\n        p0, p1, p2, p3 = c\n'
                                    '        cs.append((p0 + p1 + p2 + p3) / 4)\n    return {"centers": np.array(cs)}\n',
                                    "ValueError", "not enough values to unpack"),
    })


CHECKS = [HOUGHLINESP, FINDNONZERO, CONVEXITYDEFECTS, CALCHIST, CHESSBOARD, CHESSBOARD_SB, MINENCLOSINGTRIANGLE,
          ARUCO_IDS, HOUGHLINES_ACC, CONTOURS_CTRL, GFTT_CTRL, APPROXPOLY_CTRL, HOUGHLINES_CTRL, ARUCO_CORNERS_CTRL]
