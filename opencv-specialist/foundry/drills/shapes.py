"""Drills on output shapes: functions whose 5.0 return value lost an axis, plus one control that did not.

Measured (PLAN.md section 0, item 2): ``HoughLinesP`` (N,1,4) -> (N,4), ``HoughLinesWithAccumulator`` (N,1,3) -> (N,3),
``findNonZero`` (N,1,2) -> (N,2), ``convexityDefects`` (K,1,4) -> (K,4), ``calcHist`` (256,1) -> (256,),
``minEnclosingTriangle`` (3,1,2) -> (3,2), ArUco ``ids`` (N,1) -> (N,). ``findChessboardCorners`` / ``SB`` also changed, but
the chessboard family is held out of training (cvbench/splits.py), so it has no drill. ``HoughCircles`` did not change
and is here as the control that keeps a model from over-applying the rule.

The 4.x idioms the wrong variants reproduce: ``line[0]``, ``lines[:, 0, :]``, ``pts[:, 0, 0]``, ``for s, e, f, d in
defects[:, 0]``, ``hist[i][0]``, ``tri[:, 0, :]``, ``ids[i][0]``. The version-proof habit is ``.reshape(-1, k)``.

Each tolerance below was measured on the reference before it was written down (the numbers sit in the comments, and
tests/test_drills.py re-runs the cheap ones).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import (PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, match_points,
                     point_line_distance, spec)

GROUP = "shapes"
R = Registry()
S = SEED_BASE + 0  # this module's seeds: 61_000 .. 61_999


def _seg_dist(a, b, c, d) -> float:
    """Distance between segments a-b and c-d (0 when they cross)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    if ccw(a, c, d) * ccw(b, c, d) < 0 and ccw(a, b, c) * ccw(a, b, d) < 0:
        return 0.0
    return min(point_line_distance(a, c, d), point_line_distance(b, c, d),
               point_line_distance(c, a, b), point_line_distance(d, a, b))


# ---------------------------------------------------------------- 1. HoughLinesP: the longest stroke

def stroke_scene(seed: int, H: int = 180, W: int = 260):
    """3-5 straight strokes (2 px wide) in free directions, at least 24 px apart. The longest is at least 1.3x the second
    longest. Truth: the longest stroke's two end points."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 6))
    others = rng.uniform(46, 90, k - 1)
    lengths = np.append(others, others.max() * rng.uniform(1.3, 1.5))
    rng.shuffle(lengths)
    segs = []
    for L in lengths:
        for _ in range(500):
            phi = rng.uniform(0, math.pi)
            x1, y1 = rng.uniform(8, W - 8), rng.uniform(8, H - 8)
            x2, y2 = x1 + L * math.cos(phi), y1 + L * math.sin(phi)
            if not (6 <= x2 <= W - 6 and 6 <= y2 <= H - 6):
                continue
            p, q = (round(x1), round(y1)), (round(x2), round(y2))
            if all(_seg_dist(p, q, a, b) >= 24 for a, b in segs):
                segs.append((p, q))
                break
        else:
            raise RuntimeError(f"stroke_scene({seed}): could not place a stroke")
    img = np.zeros((H, W), np.uint8)
    for p, q in segs:
        cv2.line(img, p, q, 255, 2)
    longest = max(segs, key=lambda s: math.dist(*s))
    return img, longest


_LONG_REF = code("edges", '''\
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10).reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return {"segment": lines[int(np.argmax(length))]}
''')
_LONG_HEAD = code("edges", "    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)\n")

# calibrated on 300 scenes x 36 parameter settings (threshold 30..60, minLineLength 20..40, maxLineGap 5..15): the longest
# returned segment lies on the manufactured stroke (3 px) in 10791 of 10800 runs (the misses are maxLineGap = 5 breaking a
# stroke into fragments shorter than another stroke), and then covers at least 0.73 of its length (median 1.0, 1st
# percentile 0.95). The reference's setting (50, 30, 10) covers 0.97 at worst. On 1 px strokes 21 % of the runs fail,
# which is why the strokes are 2 px wide and 24 px apart.
LONG_ON_LINE_PX, LONG_MIN_COVER = 3.0, 0.7


def _long_cases():
    out = []
    for seed in (S + 1, S + 2, S + 3, S + 4):
        img, longest = stroke_scene(seed)
        out.append(Case({"edges": img}, longest))
    return out


def _long_verify(case: Case, out: dict) -> Verdict:
    s = arr(out, "segment").astype(float)
    a, b = case.truth
    for p in (s[:2], s[2:]):
        if point_line_distance(p, a, b) > LONG_ON_LINE_PX:
            return fail(f"end point {p.tolist()} is not on the longest stroke {a}-{b}")
    if math.dist(s[:2], s[2:]) < LONG_MIN_COVER * math.dist(a, b):
        return fail(f"segment is {math.dist(s[:2], s[2:]):.0f} px long, the longest stroke is {math.dist(a, b):.0f} px")
    return PASS


R.add(Check(
    id="drill_hough_longest", group=GROUP, probes=("HoughLinesP",), depends_on_5_0=True, role="gotcha",
    gotcha="HoughLinesP returns (N, 4) in 5.0: picking the longest of lines[i][0] breaks",
    spec=spec("Each edge map holds a few thin straight strokes of different lengths. Find the longest stroke with a "
              "probabilistic Hough transform and return its two end points.",
              {"edges": IOSpec("uint8", "HxW", 2, "0/255 edge map of straight strokes that are far apart; "
                                                  "the longest one is clearly longer than the rest")},
              {"segment": IOSpec("int", "4", 1, "x1, y1, x2, y2 of the longest stroke, in pixels")}),
    make_cases=_long_cases, verify=_long_verify, reference=_LONG_REF,
    wrong={
        "legacy_unpack": Wrong("4.x loop: x1, y1, x2, y2 = line[0] for every detected segment", _LONG_HEAD +
                               '    best, best_len = None, -1.0\n    for line in lines:\n'
                               '        x1, y1, x2, y2 = line[0]\n'
                               '        length = np.hypot(x2 - x1, y2 - y1)\n'
                               '        if length > best_len:\n            best, best_len = (x1, y1, x2, y2), length\n'
                               '    return {"segment": np.array(best)}\n', "TypeError", "cannot unpack"),
        "legacy_index": Wrong("4.x: lines[:, 0, :] to drop the middle axis", _LONG_HEAD +
                              '    lines = lines[:, 0, :]\n'
                              '    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])\n'
                              '    return {"segment": lines[int(np.argmax(length))]}\n', "IndexError", "too many indices"),
        "longest_in_x": Wrong("ranks the segments by horizontal extent only, so a steep stroke loses to a flat one",
                              _LONG_HEAD + '    lines = lines.reshape(-1, 4)\n'
                              '    return {"segment": lines[int(np.argmax(np.abs(lines[:, 2] - lines[:, 0])))]}\n', "verifier"),
        "last_segment": Wrong("returns the last segment Hough reports instead of the longest", _LONG_HEAD +
                              '    return {"segment": lines.reshape(-1, 4)[-1]}\n', "verifier"),
    }),
    "Probabilistic Hough transform on a binary edge map of a few thin straight lines: give the end points "
    "(x1, y1, x2, y2) of the longest line.",
    "Several thin straight strokes were drawn into this edge map. With the progressive probabilistic Hough "
    "transform, find the one stroke that is longest and return where it starts and ends.")


# ---------------------------------------------------------------- 2. HoughLinesWithAccumulator: skew of a ruled sheet

# calibrated on 97 sheets (skew -12..12 deg in 0.25 deg steps) x min_votes 100, 120: the vote-weighted mean angle is within
# 0.75 deg of the drawn skew (the accumulator has 1 deg bins); 1.25 deg keeps a 1.7x margin. min_votes 140 finds no line
# at some skews (a 300 px ruling collects fewer votes when its angle falls between two bins), so the cases stay at <= 130.
SKEW_TOL_DEG = 1.25


def ruled_sheet(skew_deg: float, H: int = 200, W: int = 300, spacing: int = 30, k: int = 6):
    """`k` parallel rulings spanning the sheet, rotated `skew_deg` counter-clockwise (as displayed, y down) from horizontal."""
    img = np.zeros((H, W), np.uint8)
    a = math.radians(skew_deg)
    d = np.array([math.cos(a), -math.sin(a)])      # direction along the rulings in image (x right, y down) coordinates
    n = np.array([math.sin(a), math.cos(a)])       # across the rulings
    c = np.array([W / 2, H / 2])
    for j in range(k):
        mid = c + n * (j - (k - 1) / 2) * spacing
        p, q = mid - d * 400, mid + d * 400
        ok, p1, q1 = cv2.clipLine((0, 0, W, H), (round(p[0]), round(p[1])), (round(q[0]), round(q[1])))
        if ok:
            cv2.line(img, p1, q1, 255, 1)
    return img


_SKEW_REF = code("edges, min_votes", '''\
    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi / 180, min_votes).reshape(-1, 3)
    theta = np.degrees(lines[:, 1])
    return {"skew_deg": float(90.0 - np.average(theta, weights=lines[:, 2]))}
''')
_SKEW_HEAD = code("edges, min_votes",
                  "    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi / 180, min_votes)\n")


def _skew_cases():
    return [Case({"edges": ruled_sheet(a), "min_votes": v}, a) for a, v in ((-9.0, 120), (3.5, 110), (7.0, 130), (-4.0, 120))]


def _skew_verify(case: Case, out: dict) -> Verdict:
    got = out["skew_deg"]
    return PASS if abs(got - case.truth) <= SKEW_TOL_DEG else fail(f"skew {got:.2f} deg, the sheet is rotated {case.truth:.2f} deg")


R.add(Check(
    id="drill_hough_acc_skew", group=GROUP, probes=("HoughLinesWithAccumulator",), depends_on_5_0=True, role="gotcha",
    gotcha="HoughLinesWithAccumulator returns (N, 3) in 5.0: rho, theta, votes = line[0] breaks",
    spec=spec("Estimate how far a scanned ruled sheet is rotated. Find the rulings with a Hough line transform that uses "
              "a 1 px distance step and a 1 degree angle step, discard lines that collect fewer than min_votes votes, "
              "and average the remaining angles weighted by their vote counts. Report the rotation of the rulings in "
              "degrees, counter-clockwise from horizontal as displayed (the image y axis points down).",
              {"edges": IOSpec("uint8", "HxW", 2, "0/255 edge map of parallel rulings spanning the sheet"),
               "min_votes": IOSpec("int", "scalar", None, "ignore lines with fewer votes than this")},
              {"skew_deg": IOSpec("float", "scalar", None, "rotation of the rulings in degrees, within (-45, 45)")}),
    make_cases=_skew_cases, verify=_skew_verify, reference=_SKEW_REF,
    wrong={
        "legacy_unpack": Wrong("4.x: rho, theta, votes = line[0]", _SKEW_HEAD +
                               '    thetas, votes = [], []\n    for line in lines:\n'
                               '        rho, theta, v = line[0]\n        thetas.append(theta)\n        votes.append(v)\n'
                               '    skew = 90.0 - np.average(np.degrees(thetas), weights=votes)\n'
                               '    return {"skew_deg": float(skew)}\n', "TypeError", "cannot unpack"),
        "legacy_index": Wrong("4.x: lines[:, 0, :]", _SKEW_HEAD +
                              '    lines = lines[:, 0, :]\n    theta = np.degrees(lines[:, 1])\n'
                              '    return {"skew_deg": float(90.0 - np.average(theta, weights=lines[:, 2]))}\n',
                              "IndexError", "too many indices"),
        "sign_flipped": Wrong("returns theta - 90 degrees, the rotation with the opposite sign", _SKEW_HEAD +
                              '    lines = lines.reshape(-1, 3)\n    theta = np.degrees(lines[:, 1])\n'
                              '    return {"skew_deg": float(np.average(theta, weights=lines[:, 2]) - 90.0)}\n', "verifier"),
    }),
    "A scan of ruled paper came in crooked. Use cv2.HoughLinesWithAccumulator (1 px, 1 degree) on the edge map, drop "
    "lines under min_votes, and give the vote-weighted mean angle of the rest as the sheet's counter-clockwise "
    "rotation in degrees (image y axis pointing down).",
    "How crooked is this ruled sheet? Run a Hough transform with 1 px and 1 degree bins, keep the lines that get at "
    "least min_votes votes, weight each line's angle by its votes, and return the rulings' counter-clockwise rotation "
    "from horizontal in degrees, as displayed (y down).")


# ---------------------------------------------------------------- 3. findNonZero: extreme points of a blob

def blob_with_specks(seed: int, H: int = 90, W: int = 120):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    n = int(rng.integers(5, 8))
    c = np.array([rng.uniform(40, W - 40), rng.uniform(30, H - 30)])
    R_ = rng.uniform(14, 28)
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    pts = np.round(c + np.c_[R_ * rng.uniform(0.6, 1.0, n) * np.cos(ang), R_ * rng.uniform(0.6, 1.0, n) * np.sin(ang)])
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    for _ in range(int(rng.integers(0, 4))):  # isolated single-pixel specks
        m[int(rng.integers(0, H)), int(rng.integers(0, W))] = 255
    return m


def extremes_truth(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)  # raster order
    left, right = np.lexsort((ys, xs))[0], np.lexsort((ys, -xs))[0]
    top, bottom = np.lexsort((xs, ys))[0], np.lexsort((xs, -ys))[0]
    return np.array([[xs[i], ys[i]] for i in (left, right, top, bottom)], np.int64)


_EXT_REF = code("mask", '''\
    pts = cv2.findNonZero(mask).reshape(-1, 2)
    left = pts[pts[:, 0].argmin()]
    right = pts[pts[:, 0].argmax()]
    top = pts[pts[:, 1].argmin()]
    bottom = pts[pts[:, 1].argmax()]
    return {"extremes": np.array([left, right, top, bottom])}
''')
_EXT_HEAD = code("mask", "    pts = cv2.findNonZero(mask)\n")


def _ext_cases():
    return [Case({"mask": m}, extremes_truth(m)) for m in (blob_with_specks(s) for s in (S + 11, S + 12, S + 13, S + 14))]


def _ext_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "extremes")
    if got.shape != case.truth.shape or not np.array_equal(got, case.truth):
        return fail(f"extreme points {got.tolist()} vs {case.truth.tolist()} (left, right, top, bottom)")
    return PASS


R.add(Check(
    id="drill_nonzero_extremes", group=GROUP, probes=("findNonZero",), depends_on_5_0=True, role="gotcha",
    gotcha="findNonZero returns (N, 2) in 5.0: pts[:, 0, 0] and pts[i][0] break",
    spec=spec("Collect the coordinates of all foreground pixels of the mask with cv2.findNonZero and report its four "
              "extreme points: the leftmost, rightmost, topmost and bottommost foreground pixel. Where several pixels "
              "tie, take the first one in raster order (row by row, left to right).",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255 mask: one blob and possibly a few isolated pixels")},
              {"extremes": IOSpec("int", "4x2", 2, "(x, y) of the leftmost, rightmost, topmost, bottommost pixel, in "
                                                   "that order")}),
    make_cases=_ext_cases, verify=_ext_verify, reference=_EXT_REF,
    wrong={
        "legacy_columns": Wrong("4.x: xs, ys = pts[:, 0, 0], pts[:, 0, 1]", _EXT_HEAD +
                                '    xs, ys = pts[:, 0, 0], pts[:, 0, 1]\n'
                                '    i = [xs.argmin(), xs.argmax(), ys.argmin(), ys.argmax()]\n'
                                '    return {"extremes": np.stack([xs[i], ys[i]], axis=1)}\n', "IndexError", "too many indices"),
        "legacy_nested": Wrong("4.x: tuple(pts[pts[:, :, 0].argmin()][0])", _EXT_HEAD +
                               '    left = pts[pts[:, :, 0].argmin()][0]\n    right = pts[pts[:, :, 0].argmax()][0]\n'
                               '    top = pts[pts[:, :, 1].argmin()][0]\n    bottom = pts[pts[:, :, 1].argmax()][0]\n'
                               '    return {"extremes": np.array([left, right, top, bottom])}\n', "IndexError", "too many indices"),
        "swapped_xy": Wrong("returns (row, column) instead of (x, y)", _EXT_HEAD +
                            '    pts = pts.reshape(-1, 2)[:, ::-1]\n'
                            '    return {"extremes": np.array([pts[pts[:, 1].argmin()], pts[pts[:, 1].argmax()],\n'
                            '                                  pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]])}\n', "verifier"),
    }),
    "Using cv2.findNonZero on the mask, find the leftmost, rightmost, topmost and bottommost foreground pixels and "
    "return their (x, y) coordinates in that order. Ties go to the first pixel in raster order.",
    "The mask holds one blob and maybe a few stray pixels. Get every foreground pixel with cv2.findNonZero and "
    "return the four extreme ones as (x, y) rows: left, right, top, bottom. If several pixels are equally extreme, "
    "take the one met first scanning row by row, left to right.")


# ---------------------------------------------------------------- 4. convexityDefects: chips cut into a plate

def notched_plate(seed: int):
    """A convex plate (5-7 vertices) with 2-3 V-shaped chips cut into different edges. Truth: each chip's bottom point and
    its depth, the distance from the bottom to the edge it was cut from. Depths differ by at least 4 px."""
    rng = np.random.default_rng(seed)
    c = np.array([100.0, 100.0])
    while True:
        n = int(rng.integers(5, 8))
        R_ = rng.uniform(62, 75)
        a0 = rng.uniform(0, 2 * math.pi)
        V = np.round(c + R_ * np.c_[np.cos(a0 + 2 * math.pi * np.arange(n) / n), np.sin(a0 + 2 * math.pi * np.arange(n) / n)])
        k = int(rng.integers(2, 4))
        depths = np.sort(rng.choice([8, 12, 16, 20, 24, 28], size=k, replace=False) + rng.uniform(-1.0, 1.0, k))
        if min(np.diff(depths), default=9) >= 4:
            break
    edges = rng.choice(n, size=k, replace=False)
    rng.shuffle(depths)
    poly, chips = [], []
    for i in range(n):
        poly.append(tuple(int(v) for v in V[i]))
        if i in edges:
            j = int(np.where(edges == i)[0][0])
            A, B = V[i], V[(i + 1) % n]
            e = (B - A) / np.linalg.norm(B - A)
            mid = A + rng.uniform(0.38, 0.62) * (B - A)
            half = rng.uniform(10, 15)
            inward = c - mid
            inward -= e * (inward @ e)
            inward /= np.linalg.norm(inward)
            bottom = mid + inward * depths[j]
            m1, m2 = mid - e * half, mid + e * half
            for q in (m1, bottom, m2):
                poly.append(tuple(int(round(v)) for v in q))
            b_i = np.array(poly[-2])
            chips.append((b_i, point_line_distance(b_i, V[i], V[(i + 1) % n])))
    mask = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    chips.sort(key=lambda t: -t[1])
    return mask, chips


# calibrated on 400 plates x {CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE}: the reference reports exactly the manufactured number
# of chips (no spurious defect deeper than 3 px), its deepest point is within 2 px of the manufactured chip bottom and its
# depth within 2.12 px of the distance from the drawn bottom to the cut edge (rasterising a V-shaped notch fills its last
# pixels, so the measured chip is slightly shallower). Tolerances carry a 1.5x margin over those worst cases.
NOTCH_POINT_TOL, NOTCH_DEPTH_TOL = 3.0, 3.2

_NOTCH_REF = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull).reshape(-1, 4)
    depth = defects[:, 3] / 256.0
    deep = depth > 3.0
    i = int(np.argmax(np.where(deep, depth, -1.0)))
    return {"n_chips": int(deep.sum()), "bottom": cnt.reshape(-1, 2)[defects[i, 2]], "depth": float(depth[i])}
''')
_NOTCH_HEAD = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
''')


def _notch_cases():
    out = []
    for seed in (S + 21, S + 22, S + 23, S + 24):
        mask, chips = notched_plate(seed)
        out.append(Case({"mask": mask}, chips))
    return out


def _notch_verify(case: Case, out: dict) -> Verdict:
    chips = case.truth
    if out["n_chips"] != len(chips):
        return fail(f"{out['n_chips']} chips reported, the plate has {len(chips)} deeper than 3 px")
    bottom, depth = arr(out, "bottom").astype(float), out["depth"]
    if abs(depth - chips[0][1]) > NOTCH_DEPTH_TOL:
        return fail(f"deepest chip is {depth:.1f} px deep, expected {chips[0][1]:.1f} px")
    if np.abs(bottom - chips[0][0]).max() > NOTCH_POINT_TOL:
        return fail(f"deepest chip bottom {bottom.tolist()}, expected {chips[0][0].tolist()}")
    return PASS


R.add(Check(
    id="drill_defect_notches", group=GROUP, probes=("convexityDefects",), depends_on_5_0=True, role="gotcha",
    gotcha="convexityDefects returns (K, 4) in 5.0: defects[:, 0] and defects[i, 0] break",
    spec=spec("A stamped plate has V-shaped chips cut into some of its edges. Use the convexity defects of the plate's "
              "outline to count the chips deeper than 3 px, and report the bottom point and the depth of the deepest "
              "chip.",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255; one plate (convex outline) with 2 or 3 chips cut into it")},
              {"n_chips": IOSpec("int", "scalar", None, "number of chips deeper than 3 px"),
               "bottom": IOSpec("int", "2", 1, "(x, y) of the deepest chip's bottom point"),
               "depth": IOSpec("float", "scalar", None, "depth of the deepest chip in px, measured from the edge it "
                                                         "was cut from")}),
    make_cases=_notch_cases, verify=_notch_verify, reference=_NOTCH_REF,
    wrong={
        "legacy_loop": Wrong("4.x: for s, e, f, d in defects[:, 0]", _NOTCH_HEAD +
                             '    best = None\n    n = 0\n    for s, e, f, d in defects[:, 0]:\n'
                             '        if d / 256.0 > 3.0:\n            n += 1\n'
                             '            if best is None or d > best[1]:\n                best = (f, d)\n'
                             '    return {"n_chips": n, "bottom": cnt[best[0]][0], "depth": best[1] / 256.0}\n',
                             "TypeError", "cannot unpack"),
        "legacy_index": Wrong("4.x: far = defects[:, 0, 2]", _NOTCH_HEAD +
                              '    far, depth = defects[:, 0, 2], defects[:, 0, 3] / 256.0\n'
                              '    i = int(np.argmax(depth))\n'
                              '    return {"n_chips": int((depth > 3.0).sum()), "bottom": cnt[far[i]][0], "depth": float(depth[i])}\n',
                              "IndexError", "too many indices"),
        "fixed_point": Wrong("forgets that the defect depth is fixed-point (x 256)", _NOTCH_HEAD +
                             '    defects = defects.reshape(-1, 4)\n    deep = defects[:, 3] > 3.0\n'
                             '    i = int(np.argmax(defects[:, 3]))\n'
                             '    return {"n_chips": int(deep.sum()), "bottom": cnt.reshape(-1, 2)[defects[i, 2]],\n'
                             '            "depth": float(defects[i, 3])}\n', "verifier"),
    }),
    "From the convexity defects of this plate outline, report how many V-shaped chips are deeper than 3 px, plus the "
    "bottom point (x, y) and the depth in px of the deepest one, measured from the edge it was cut into.",
    "The mask shows a convex plate with two or three V-notches cut in. Find the notches as convexity defects of the "
    "largest contour and return the number deeper than 3 px, the bottom point of the deepest and its depth in "
    "pixels.")


# ---------------------------------------------------------------- 5. calcHist: levels that never occur

def banded_image(seed: int, H: int = 48, W: int = 64):
    """Gray image whose levels sit on a coarse comb: a narrow range stretched to full scale (or gamma-curved), so many
    of the 256 levels never occur. The truth comes from the pixel counts."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3)
    base = (base - base.min()) / (base.max() - base.min())
    mode = seed % 3
    if mode == 0:
        lo, hi = int(rng.integers(20, 60)), int(rng.integers(100, 140))
        small = np.round(lo + base * (hi - lo))
        img = np.round((small - lo) * 255.0 / (hi - lo))
    elif mode == 1:
        img = np.round(255.0 * base ** float(rng.uniform(2.2, 3.0)))
    else:
        img = np.round(base * 60) * 4 + int(rng.integers(0, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def unused_levels(img: np.ndarray):
    counts = np.bincount(img.ravel(), minlength=256)
    unused = counts == 0
    longest = run = 0
    for u in unused:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(unused.sum()), longest


_HIST_REF = code("gray", '''\
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    missing = hist == 0
    longest = run = 0
    for m in missing:
        run = run + 1 if m else 0
        longest = max(longest, run)
    return {"n_missing": int(missing.sum()), "longest_gap": int(longest)}
''')
_HIST_HEAD = code("gray", "    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])\n")
_RUN = ('    longest = run = 0\n    for m in missing:\n        run = run + 1 if m else 0\n'
        '        longest = max(longest, run)\n')


def _hist_cases():
    return [Case({"gray": banded_image(s)}, unused_levels(banded_image(s))) for s in (S + 31, S + 32, S + 33, S + 34, S + 35)]


def _hist_verify(case: Case, out: dict) -> Verdict:
    n, gap = case.truth
    if (out["n_missing"], out["longest_gap"]) != (n, gap):
        return fail(f"{out['n_missing']} missing levels / longest gap {out['longest_gap']}, expected {n} / {gap}")
    return PASS


R.add(Check(
    id="drill_hist_unused_levels", group=GROUP, probes=("calcHist",), depends_on_5_0=True, role="gotcha",
    gotcha="calcHist returns (256,) in 5.0: hist[i][0] and hist[:, 0] break",
    spec=spec("After a contrast stretch some gray levels are never used, which shows up as banding. Build the 256-bin "
              "gray-level histogram of the image and report how many of the 256 levels never occur and the longest run "
              "of consecutive levels that never occur.",
              {"gray": IOSpec("uint8", "HxW", 2, "grayscale image")},
              {"n_missing": IOSpec("int", "scalar", None, "number of levels in 0..255 with zero pixels"),
               "longest_gap": IOSpec("int", "scalar", None, "length of the longest run of consecutive missing levels")}),
    make_cases=_hist_cases, verify=_hist_verify, reference=_HIST_REF,
    wrong={
        "legacy_bin_index": Wrong("4.x: each bin was a length-1 row, so hist[i][0]", _HIST_HEAD +
                                  '    missing = np.array([hist[i][0] == 0 for i in range(256)])\n' + _RUN +
                                  '    return {"n_missing": int(missing.sum()), "longest_gap": int(longest)}\n',
                                  "IndexError", "invalid index to scalar"),
        "legacy_column": Wrong("4.x: hist[:, 0] to get a flat vector", _HIST_HEAD +
                               '    missing = hist[:, 0] == 0\n' + _RUN +
                               '    return {"n_missing": int(missing.sum()), "longest_gap": int(longest)}\n',
                               "IndexError", "too many indices"),
    }),
    "Banding check: compute the 256-bin histogram of this grayscale image, then report the number of gray levels "
    "(0 to 255) that have no pixels and the length of the longest stretch of consecutive empty levels.",
    "A stretched image skips some gray levels. From its histogram with 256 bins, return how many levels are empty and "
    "how long the longest run of neighbouring empty levels is.")


# ---------------------------------------------------------------- 6. minEnclosingTriangle: a triangular sign

# calibrated on 400 triangles (min angle 35 deg, area >= 3000 px^2): the enclosing triangle's vertices are within
# 2.16 px of the drawn vertices and its area is 0.7 % to 4.2 % above the drawn area. 4 px and 8 % keep a 1.9x margin.
TRI_VERTEX_TOL, TRI_AREA_REL = 4.0, 0.08


def _tri_area(t) -> float:
    t = np.asarray(t, np.float64)
    return 0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1]))


def sign_scene(seed: int):
    rng = np.random.default_rng(seed)
    while True:
        v = np.round(rng.uniform(15, 185, (3, 2)))
        sides = [np.linalg.norm(v[(i + 1) % 3] - v[i]) for i in range(3)]
        if min(sides) < 40:
            continue
        angles = [math.degrees(math.acos(np.clip(np.dot(v[(i + 1) % 3] - v[i], v[(i + 2) % 3] - v[i]) /
                                                 (sides[i] * sides[(i + 2) % 3]), -1, 1))) for i in range(3)]
        if min(angles) >= 35 and _tri_area(v) >= 3000:
            break
    mask = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(mask, [v.astype(np.int32)], 255)
    return mask, v


_TRI_REF = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    area, tri = cv2.minEnclosingTriangle(contour)
    return {"vertices": tri.reshape(3, 2), "area": float(area)}
''')
_TRI_HEAD = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    area, tri = cv2.minEnclosingTriangle(contour)
''')


def _tri_cases():
    out = []
    for seed in (S + 41, S + 42, S + 43):
        mask, v = sign_scene(seed)
        out.append(Case({"mask": mask}, v))
    return out


def _tri_verify(case: Case, out: dict) -> Verdict:
    v, area = arr(out, "vertices").astype(float), out["area"]
    if v.shape != (3, 2):
        return fail(f"vertices have shape {v.shape}")
    why = match_points(case.truth, v, TRI_VERTEX_TOL)
    if why:
        return fail("the smallest triangle around a triangular sign is the sign itself: " + why)
    true_area = _tri_area(case.truth)
    if abs(area - true_area) > TRI_AREA_REL * true_area:
        return fail(f"area {area:.0f} px^2, the sign covers about {true_area:.0f}")
    return PASS


R.add(Check(
    id="drill_enclosing_triangle", group=GROUP, probes=("minEnclosingTriangle",), depends_on_5_0=True, role="gotcha",
    gotcha="minEnclosingTriangle returns a (3, 2) array in 5.0: tri[:, 0, :] and tri[i][0] break",
    spec=spec("A triangular road sign is photographed as a binary mask. Fit the smallest-area triangle that encloses "
              "the sign's outline and return its three vertices and its area.",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255 mask with one filled triangle")},
              {"vertices": IOSpec("float", "3x2", 2, "(x, y) of the enclosing triangle's vertices, in any order"),
               "area": IOSpec("float", "scalar", None, "area of the enclosing triangle in px^2")}),
    make_cases=_tri_cases, verify=_tri_verify, reference=_TRI_REF,
    wrong={
        "legacy_index": Wrong("4.x: tri[:, 0, :]", _TRI_HEAD +
                              '    return {"vertices": tri[:, 0, :], "area": float(area)}\n', "IndexError", "too many indices"),
        "legacy_squeeze": Wrong("4.x: tri[:, 0] to get (3, 2); on 5.0 it keeps only the x column", _TRI_HEAD +
                                '    return {"vertices": tri[:, 0], "area": float(area)}\n', "verifier"),
        "legacy_loop": Wrong("4.x loop: vertices = [tuple(p[0]) for p in tri]", _TRI_HEAD +
                             '    return {"vertices": np.array([tuple(p[0]) for p in tri], np.float32), "area": float(area)}\n',
                             "TypeError", "not iterable"),
    }),
    "Find the minimum-area triangle that encloses the outline of the blob in this mask (a triangular sign). Return "
    "the triangle's vertices and its area in px^2.",
    "Given the mask of a triangular sign, compute the smallest enclosing triangle of its contour and report the "
    "three corner points (x, y) together with the triangle's area.")


# ---------------------------------------------------------------- 7. ArUco ids: marker ids and side lengths

# calibrated on 300 sheets (sizes 84..112 px): the detected side length is the drawn size minus one pixel to within 0.1 px;
# 1.5 px leaves room for the corner refinement a model may switch on.
AR_SIDE_TOL = 1.5
AR_DICT = cv2.aruco.DICT_5X5_100


def marker_sheet(seed: int, k: int | None = None):
    """k DICT_5X5_100 markers (k = 0 gives a blank sheet with a dark box), distinct random ids, one common size.
    Truth: ids ascending and each marker's side length in px."""
    rng = np.random.default_rng(seed)
    dic = cv2.aruco.getPredefinedDictionary(AR_DICT)
    k = int(rng.integers(2, 5)) if k is None else k
    size = int(rng.choice([84, 96, 108]))
    H, W = size + 90, max(k, 1) * (size + 36) + 36
    img = np.full((H, W), 255, np.uint8)
    ids = rng.choice(100, size=k, replace=False)
    for i, mid in enumerate(ids):
        x, y = 36 + i * (size + 36), int(rng.integers(20, 70))
        img[y:y + size, x:x + size] = cv2.aruco.generateImageMarker(dic, int(mid), size)
    if k == 0:
        cv2.rectangle(img, (40, 30), (110, 80), 0, -1)
    order = np.argsort(ids)
    return img, np.asarray(ids, np.int64)[order], np.full(k, size - 1.0)


_AR_REF = code("img", '''\
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return {"ids": np.zeros(0, np.int32), "sides": np.zeros(0)}
    ids = ids.reshape(-1)
    sides = np.array([np.linalg.norm(np.roll(c.reshape(4, 2), -1, axis=0) - c.reshape(4, 2), axis=1).mean() for c in corners])
    order = np.argsort(ids)
    return {"ids": ids[order], "sides": sides[order]}
''')
_AR_HEAD = code("img", '''\
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    corners, ids, _ = detector.detectMarkers(img)
''')
_AR_SIDES = ('    sides = [np.linalg.norm(np.roll(c.reshape(4, 2), -1, axis=0) - c.reshape(4, 2), axis=1).mean() for c in corners]\n')


def _ar_cases():
    out = []
    for seed, k in ((S + 51, 3), (S + 52, None), (S + 53, 0), (S + 54, 2)):
        img, ids, sides = marker_sheet(seed, k)
        out.append(Case({"img": img}, (ids, sides)))
    return out


def _ar_verify(case: Case, out: dict) -> Verdict:
    ids_gt, sides_gt = case.truth
    ids, sides = arr(out, "ids"), arr(out, "sides")
    if ids.shape != ids_gt.shape or not np.array_equal(ids, ids_gt):
        return fail(f"ids {ids.tolist()} vs the markers on the sheet {ids_gt.tolist()}")
    if sides.shape != sides_gt.shape or (sides.size and np.abs(sides - sides_gt).max() > AR_SIDE_TOL):
        return fail(f"side lengths {np.round(sides, 1).tolist()} vs {sides_gt.tolist()}")
    return PASS


R.add(Check(
    id="drill_aruco_sizes", group=GROUP, probes=("aruco_detectMarkers",), depends_on_5_0=True, role="gotcha",
    gotcha="ArucoDetector.detectMarkers returns ids as (N,) in 5.0: ids[i][0] and ids[:, 0] break (and ids is None "
           "when nothing is found, in every version)",
    spec=spec("Detect the ArUco markers (DICT_5X5_100 dictionary) on this sheet and report, for each marker in "
              "ascending id order, its id and its side length in px (the mean length of its four detected sides). A "
              "sheet without markers gives two empty arrays.",
              {"img": IOSpec("uint8", "HxW", 2, "grayscale image: square markers of one size on a white sheet; "
                                                "possibly none")},
              {"ids": IOSpec("int", "N", 1, "marker ids, ascending"),
               "sides": IOSpec("float", "N", 1, "side length in px of the marker with the id at the same index")}),
    make_cases=_ar_cases, verify=_ar_verify, reference=_AR_REF,
    wrong={
        "legacy_index": Wrong("4.x: ids[i][0] for every marker", _AR_HEAD +
                              '    if ids is None:\n        return {"ids": np.zeros(0, np.int32), "sides": np.zeros(0)}\n' +
                              _AR_SIDES +
                              '    pairs = sorted((int(ids[i][0]), sides[i]) for i in range(len(ids)))\n'
                              '    return {"ids": np.array([p[0] for p in pairs]), "sides": np.array([p[1] for p in pairs])}\n',
                              "IndexError", "invalid index to scalar"),
        "legacy_column": Wrong("4.x: ids[:, 0]", _AR_HEAD +
                               '    if ids is None:\n        return {"ids": np.zeros(0, np.int32), "sides": np.zeros(0)}\n' +
                               _AR_SIDES +
                               '    order = np.argsort(ids[:, 0])\n'
                               '    return {"ids": ids[:, 0][order], "sides": np.array(sides)[order]}\n',
                               "IndexError", "too many indices"),
        "no_marker_case": Wrong("ids is None on a sheet without markers and the code does not check", _AR_HEAD +
                                _AR_SIDES +
                                '    ids = ids.reshape(-1)\n    order = np.argsort(ids)\n'
                                '    return {"ids": ids[order], "sides": np.array(sides)[order]}\n',
                                "AttributeError", "NoneType"),
    }),
    "Find all ArUco markers (DICT_5X5_100) in the image. For each, in ascending order of id, give the id and the "
    "mean length of its four sides in pixels. If there is no marker, return empty arrays.",
    "Using the DICT_5X5_100 ArUco dictionary, detect the markers on the sheet and return their ids sorted ascending, "
    "plus each marker's side length in px (average of its four detected edges), index-aligned. No markers means "
    "empty results.")


# ---------------------------------------------------------------- 8. control: HoughCircles did not change

# calibrated on 300 coin sheets (2-5 discs, radius 13..29, blur 2 px, noise 2 levels) x 27 settings (param2 20..30, dp 1.0..1.5,
# minDist 25..45): every coin found and nothing extra in all 8100 runs, centre within 1.58 px and radius within 1.32 px at
# worst. 3 px keeps a 1.9x margin. param2 = 40 misses a coin in 0.7-2 % of the runs, so a model that picks it is judged on that.
CIRC_TOL = 3.0


def coin_sheet(seed: int, k: int | None = None):
    rng = np.random.default_rng(seed)
    H, W = 170, 230
    k = int(rng.integers(2, 6)) if k is None else k
    discs = []
    while len(discs) < k:
        r = int(rng.integers(13, 30))
        x, y = int(rng.integers(r + 6, W - r - 6)), int(rng.integers(r + 6, H - r - 6))
        if all(math.hypot(x - a, y - b) >= r + s + 20 for a, b, s in discs):
            discs.append((x, y, r))
    img = np.full((H, W), 40, np.uint8)
    for x, y, r in discs:
        cv2.circle(img, (x, y), r, 200, -1)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img, np.array(sorted(discs), np.float64)


_CIRC_REF = code("gray", '''\
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=100, param2=30, minRadius=10, maxRadius=40)
    circles = circles[0]
    return {"circles": circles[np.argsort(circles[:, 0])]}
''')
_CIRC_HEAD = code("gray", '''\
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=100, param2=30, minRadius=10, maxRadius=40)
''')


def _circ_cases():
    out = []
    for seed, k in ((S + 61, 2), (S + 62, 4), (S + 63, 5), (S + 64, 2)):
        img, discs = coin_sheet(seed, k)
        out.append(Case({"gray": img}, discs))
    return out


def _circ_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "circles").astype(float)
    if got.ndim != 2 or got.shape != case.truth.shape:
        return fail(f"circles have shape {got.shape}, expected {case.truth.shape}")
    if np.any(np.diff(got[:, 0]) < 0):
        return fail("circles are not ordered left to right")
    why = match_points(case.truth[:, :2], got[:, :2], CIRC_TOL)
    if why:
        return fail("centres: " + why)
    nearest = np.linalg.norm(case.truth[:, None, :2] - got[None, :, :2], axis=2).argmin(axis=1)
    if np.abs(case.truth[:, 2] - got[nearest, 2]).max() > CIRC_TOL:
        return fail(f"radii {got[nearest, 2].round(1).tolist()} vs the coins {case.truth[:, 2].tolist()}")
    return PASS


R.add(Check(
    id="drill_coin_circles", group=GROUP, probes=("HoughCircles",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: HoughCircles still returns (1, N, 3); treating it as (N, 3) breaks",
    spec=spec("Find the round coins on the tray with a circle Hough transform and report each coin's centre and "
              "radius, ordered left to right.",
              {"gray": IOSpec("uint8", "HxW", 2, "grayscale image: bright discs on a dark tray; the discs are at "
                                                 "least 20 px apart")},
              {"circles": IOSpec("float", "Nx3", 2, "(x, y, radius) per coin in px, increasing x")}),
    make_cases=_circ_cases, verify=_circ_verify, reference=_CIRC_REF,
    wrong={
        "overapplied_unpack": Wrong("over-applies the lost-axis rule: for x, y, r in circles", _CIRC_HEAD +
                                    '    rows = [(x, y, r) for x, y, r in circles]\n'
                                    '    return {"circles": np.array(sorted(rows))}\n', "ValueError", "values to unpack"),
        "overapplied_asis": Wrong("over-applies the rule: returns the (1, N, 3) array as if it were (N, 3)", _CIRC_HEAD +
                                  '    return {"circles": circles}\n', "verifier"),
    }),
    "Detect the coins in this image with cv2.HoughCircles and return (x, y, radius) for every coin, sorted by x.",
    "The tray image shows bright round coins. Locate them using the Hough gradient circle detector and report their "
    "centres and radii, left to right.")


CHECKS = R.checks
PHRASINGS = R.phrasings
