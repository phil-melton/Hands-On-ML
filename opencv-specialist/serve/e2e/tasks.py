"""The 20 Stage 7 E2E tasks: multi-step, dependent, manufactured from the cvbench families, graded exactly.

Frozen success criterion 4 (LOGBOOK.md, 2026-10-22): "orchestrator + specialist >= orchestrator alone on the Stage 7 E2E
set, with fewer orchestrator tokens"; PLAN.md section 5, Phase 4: 20 multi-step tasks (HuggingGPT-style DAGs).

What a task is
--------------
A prompt in natural language (what to compute, where the images are, the exact JSON answer schema), one to three input
images, and a truth: for every key of the answer, a spec (kind, value, tolerance, and where the tolerance comes from).
Each task chains 2-4 dependent steps (a later step consumes an earlier result: a region found by one step is measured by
the next, a scale measured on one image converts a measurement on another, a corner set defines a rectification). The
truth is known from the construction of the scene, never from running OpenCV on it.

Seeds
-----
Every scene seed lies in ``compose.SEED_RANGE`` = [6_000_000, 7_000_000), disjoint from the dev [3_100_000, 4_000_000),
test [4_000_000, 5_000_000) and foundry [5_000_000, 6_000_000) ranges of ``cvbench.splits`` (tests/test_e2e.py pins it).
Task ``i`` owns the block [6_000_000 + 10_000 i, 6_000_000 + 10_000 (i + 1)); ``_scan`` takes the first seed in the block
that satisfies the task's admissibility predicate (documented per task: a unique "largest", a margin from every
threshold the answer depends on), so a task is a deterministic function of its number.

Tolerance sources (also stored in every spec as ``src``). The rule is the one of CLAUDE.md: derived from the measurement,
never loosened, never chosen after looking at a model's answers.
* quantities that are the families' own quantities import the family's law: ``perspective_rectify.corner_tolerance``,
  ``hough_segments.tolerance``, ``chessboard_corners.tolerance``, ``feature_homography.tolerance`` /
  ``probe_error``, ``contour_area.TOL_C / TOL_C0``, ``components_stats.CENTROID_TOL`` (areas and boxes: 0),
  ``histogram_stats.FLOAT_RTOL`` (counts and the lower median: 0), ``color_segmentation.TOL_PX`` (0), and the
  text_layout verifier itself for the text placement;
* quantities derived from those (a length, an angle, an area, a mean, a scale, a ratio) take the worst-case propagation
  rule of ``compose`` (Lipschitz bound or interval arithmetic) on the family's per-point tolerance;
* quantities transported through a rectification that the reported corners define take ``compose.worst_case`` over all
  corner displacements of the family's corner tolerance, plus the family's own boundary law for the measurement made in
  the rectified frame, plus 1 px where a centroid is read off thresholded pixels (the half-pixel snap of hough_segments'
  law per axis, rounded up).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from cvbench import splits
from cvbench.families import (chessboard_corners as cb, color_segmentation as cs, components_stats as cc,
                              contour_area as ca, feature_homography as fh, histogram_stats as hi,
                              hough_segments as hs, perspective_rectify as pr, text_layout as tl)

from serve.e2e import compose as cp

N_TASKS = 20
BLOCK = 10_000
LINUX_PYTHON = "/home/ptmel/venvs/cv5/bin/python"
DISTRO = "Ubuntu-24.04"


# ---------------------------------------------------------------- truth specs (JSON-able dicts; grade.py consumes them)

def S_int(v, src="exact: integer count/box of a pixel set (zero tolerance in the family law)"):
    return {"kind": "int", "value": int(v), "src": src}


def S_ints(v, src="exact"):
    return {"kind": "ints", "value": [int(x) for x in v], "src": src}


def S_str(v, src="exact"):
    return {"kind": "str", "value": str(v), "src": src}


def S_bool(v, src="exact"):
    return {"kind": "bool", "value": bool(v), "src": src}


def S_num(v, tol, src):
    return {"kind": "num", "value": float(v), "tol": float(tol), "src": src}


def S_rel(v, rtol, src):
    """|got - v| <= rtol * max(1, |v|)."""
    return {"kind": "rel", "value": float(v), "rtol": float(rtol), "src": src}


def S_point(v, tol, src):
    return {"kind": "point", "value": [float(v[0]), float(v[1])], "tol": float(tol), "src": src}


def S_points(v, tol, src):
    return {"kind": "points", "value": [[float(a), float(b)] for a, b in v], "tol": float(tol), "src": src}


def S_angle(v, tol, src, period=180.0):
    return {"kind": "angle", "value": float(v), "tol": float(tol), "period": float(period), "src": src}


def S_sorted_nums(v, tols, src):
    return {"kind": "sorted_nums", "value": [float(x) for x in v], "tols": [float(t) for t in tols], "src": src}


def S_dict(children: dict[str, dict], src="each child carries its own tolerance source"):
    return {"kind": "dict", "children": children, "src": src}


def S_homography(H, pts, tol, src):
    return {"kind": "homography", "value": np.asarray(H, float).tolist(),
            "pts": np.asarray(pts, float).tolist(), "tol": float(tol), "norm_tol": fh.NORM_TOL, "src": src}


def S_present(src="graded together with the key it belongs to"):
    return {"kind": "present", "src": src}


def S_layout(src):
    return {"kind": "layout", "src": src}


# ---------------------------------------------------------------- the task object

@dataclass
class Task:
    id: str
    number: int
    title: str
    steps: tuple[str, ...]
    families: tuple[str, ...]
    prompt_body: str
    images: dict[str, np.ndarray]
    schema: dict[str, str]
    truth: dict[str, dict]
    seeds: dict[str, int]
    extras: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def held_out(self) -> dict[str, bool]:
        return {f: splits.FAMILIES[f].held_out for f in self.families}

    @property
    def uses_held_out(self) -> bool:
        return any(self.held_out.values())

    def meta(self) -> dict:
        """What config.json may say about a task: nothing about the truth."""
        return {"id": self.id, "number": self.number, "title": self.title, "steps": list(self.steps),
                "families": list(self.families), "held_out": self.held_out, "seeds": self.seeds,
                "images": sorted(self.images)}

    def truth_json(self) -> str:
        return json.dumps(self.truth, sort_keys=True)


IMAGE_NOTE = "8-bit PNG files; cv2.imread reads them in BGR channel order"


def render_prompt(task: Task, linux_dir: str, win_dir: str) -> str:
    """The orchestrator's prompt. Identical in both arms apart from the work directory it names."""
    img_lines = []
    for name, im in sorted(task.images.items()):
        h, w = im.shape[:2]
        kind = "single-channel" if im.ndim == 2 else "3-channel BGR"
        img_lines.append(f"  - {name}: {linux_dir}/{name}.png  ({w} px wide, {h} px high, {kind})")
    schema_lines = [f'  - "{k}": {v}' for k, v in task.schema.items()]
    return (
        "You are solving one computer-vision task. Everything you need is in your work directory.\n"
        "\n"
        f"Work directory, as a Linux path inside WSL: {linux_dir}\n"
        f"The same directory as Windows sees it: {win_dir}\n"
        "Python 3 with OpenCV 5.0.0 and NumPy runs inside WSL. Run a script from Git Bash with\n"
        f"    MSYS_NO_PATHCONV=1 wsl.exe -d {DISTRO} -- {LINUX_PYTHON} {linux_dir}/<script>.py\n"
        "(Git Bash rewrites arguments that start with a slash into Windows paths unless MSYS_NO_PATHCONV=1 is set; in "
        "PowerShell leave that prefix out.) Use Linux paths inside scripts. "
        "Keep your scripts and any intermediate files in the work directory. Do not look outside it.\n"
        "\n"
        "TASK\n"
        f"{task.prompt_body.strip()}\n"
        "\n"
        "INPUT IMAGES\n" + "\n".join(img_lines) + f"\n  ({IMAGE_NOTE})\n"
        "\n"
        "ANSWER\n"
        f"Write {linux_dir}/answer.json: one UTF-8 JSON object with exactly these keys and nothing else:\n"
        + "\n".join(schema_lines) + "\n"
        "Pixel coordinates are (x, y) with x to the right and y down, the centre of the top-left pixel being (0, 0). "
        "Numbers are plain JSON numbers at full double precision (write them with json.dump from Python; do not round "
        "unless a key says so). Lists are JSON arrays. The answer is graded automatically against an exact "
        "manufactured truth.\n"
        "When answer.json is written, reply with the single word DONE."
    )


# ---------------------------------------------------------------- registry and seed scan

_BUILDERS: dict[int, Callable[[], Task]] = {}
_CACHE: dict[int, Task] = {}


def _register(n: int):
    def deco(fn):
        _BUILDERS[n] = fn
        return fn
    return deco


def _base(n: int) -> int:
    return cp.SEED_RANGE[0] + BLOCK * n


def _scan(n: int, make, pred=lambda s: True, *, start: int = 0, step: int = 1, span: int = 4000):
    """First scene in task n's seed block that satisfies ``pred``. Returns (seed, scene)."""
    base = _base(n)
    for k in range(start, start + span * step, step):
        seed = base + k
        assert seed < base + BLOCK
        scene = make(seed)
        if pred(scene):
            return seed, scene
    raise RuntimeError(f"task {n}: no admissible seed in {span} candidates from {base + start}")


def _rng(seed: int, n: int) -> np.random.Generator:
    return np.random.default_rng([seed, n])


def _f(x) -> float:
    return float(x)


def _pts(a) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in np.asarray(a, float)]


# ---------------------------------------------------------------- shared prompt pieces

POSITION_RULE = ("TL is the corner with the smallest x+y, BR the largest x+y, TR the smallest y-x and BL the largest "
                 "y-x")
HSV_DEFINITION = (
    "A pixel has a colour when, in 8-bit HSV (cv2.COLOR_BGR2HSV: H 0-179, S and V 0-255), S >= 100 and V >= 80 and its "
    "hue lies in that colour's inclusive range: red 0-10 or 170-179 (red wraps around 0), yellow 20-35, green 40-80, "
    "blue 100-130. Every bound is inclusive. Pixels in no range (greys, pastels, dark pixels, other hues) have no "
    "colour.")
RED_DEFINITION = (
    "A pixel is red when, in 8-bit HSV (cv2.COLOR_BGR2HSV: H 0-179, S and V 0-255), S >= 100 and V >= 80 and its hue "
    "is in 0-10 or 170-179 (red wraps around 0); every bound is inclusive.")
COLOURS = cs.COLOURS


def _check_hsv_text():
    assert cs.S_MIN == 100 and cs.V_MIN == 80 and cs.HUE == {
        "red": ((0, 10), (170, 179)), "yellow": ((20, 35),), "green": ((40, 80),), "blue": ((100, 130),)}


_check_hsv_text()


def _png(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


# ================================================================ T01 card-geometry (perspective_rectify)

@_register(1)
def _t01() -> Task:
    seed, sc = _scan(1, pr.make_scene)
    d = pr.corner_tolerance(sc)
    c = sc.corners
    edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    top = c[1] - c[0]
    ang = math.degrees(math.atan2(top[1], top[0]))
    src_c = "perspective_rectify.corner_tolerance = TOL_A + TOL_B * blur"
    truth = {
        "corners": S_points(c, d, src_c),
        "longest_side_px": S_num(edges.max(), 2 * d, "distance between two points: 2 x " + src_c),
        "quad_area_px2": S_num(cp.poly_area(c), cp.area_tol(c, d), "shoelace expansion in the vertex displacements (compose.area_tol), vertex tolerance = " + src_c),
        "top_edge_angle_deg": S_angle(ang, cp.angle_tol_deg(float(edges[0]), d), "asin(2 d / L), L = |TL TR|, d = " + src_c, period=360.0),
    }
    body = (
        "The photo shows one flat rectangular card lying on a darker surface, seen from an arbitrary angle and rotation, "
        "entirely inside the frame. Find the card's four corners and report them in the order top-left, top-right, "
        f"bottom-right, bottom-left by image position ({POSITION_RULE}). Then, from those four corners, compute the "
        "length in px of the longest of the card's four sides, the area in px^2 enclosed by the four corners, and the "
        "direction of the top edge (from the TL corner to the TR corner) as an angle in degrees, atan2(dy, dx) with y "
        "down, in (-180, 180].")
    schema = {
        "corners": "4 x 2 array [[x, y], ...] in the order TL, TR, BR, BL",
        "longest_side_px": "number, px",
        "quad_area_px2": "number, px^2",
        "top_edge_angle_deg": "number in (-180, 180], degrees",
    }
    return Task("e2e-01-card-geometry", 1, "Card corners, then side, area and direction",
                ("find the card's four corners", "order them by image position", "derive side length, area and edge direction"),
                ("perspective_rectify",), body, {"photo": _png(sc.img)}, schema, truth, {"scene": seed})


# ================================================================ T02 card-blobs (perspective_rectify + components_stats)

def _place_discs(rng: np.random.Generator, cw: float, ch: float):
    """3-4 discs on the printed area of the card (card px, centred), radius >= 18, the largest >= 1.4 x the next, edges >= 24 px apart."""
    hx, hy = 0.4 * cw, 0.4 * ch  # the printed area: 10 % paper margin on every side (perspective_rectify.MU)
    rmax = min(42.0, min(hx, hy) - 14.0)
    for _ in range(4000):
        k = int(rng.integers(3, 5))
        r1 = float(rng.uniform(0.72 * rmax, rmax))
        radii = [r1] + sorted((float(rng.uniform(18.0, min(r1 / 1.4, 34.0))) for _ in range(k - 1)), reverse=True)
        discs = []
        for r in radii:
            for _ in range(300):
                x, y = rng.uniform(-hx + r + 6, hx - r - 6), rng.uniform(-hy + r + 6, hy - r - 6)
                if all(math.hypot(x - a, y - b) >= r + rr + 24.0 for a, b, rr in discs):
                    discs.append((float(x), float(y), r))
                    break
            else:
                break
        if len(discs) == k:
            return discs
    raise RuntimeError("no disc layout")


@_register(2)
def _t02() -> Task:
    seed, sc = _scan(2, pr.make_scene, lambda s: min(s.out_w, s.out_h) >= 150)  # a card big enough for blobs of >= 18 px
    rng = _rng(seed, 2)
    cw, ch = sc.card
    discs = _place_discs(rng, cw, ch)
    paper, ink = rng.uniform(225, 245, 3), rng.uniform(25, 60, 3)
    img = cp.render_card_photo(sc, rng, paper=paper, ink=ink, discs=discs)
    d = pr.corner_tolerance(sc)
    ow, oh = sc.out_w, sc.out_h
    Q = cp.card_to_rect(sc)
    assert np.allclose(Q[2, :2], 0.0), "a rectangle onto a rectangle must be affine"
    s0, s1 = np.linalg.svd(Q[:2, :2])[1]
    big, second = discs[0], discs[1]
    centre_rect = cp.apply_h(Q, [[big[0], big[1]]])[0]
    a1, a2 = (math.pi * r ** 2 * s0 * s1 for r in (big[2], second[2]))
    # registration: the same discs, mapped photo -> rectified by corners displaced by up to d
    th = np.linspace(0, 2 * math.pi, 96, endpoint=False)

    def ring_photo(dc):
        return cp.apply_h(sc.H_ci, np.c_[dc[0] + dc[2] * np.cos(th), dc[1] + dc[2] * np.sin(th)])

    ring1, ring2 = ring_photo(big), ring_photo(second)
    dst = cp.rect_dst(ow, oh)

    def centroid_fn(cn):
        v = cp.apply_h(cp.homography_dlt(cn, dst), ring1)
        x, y = v[:, 0], v[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a = cr.sum() / 2
        return [((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)]

    def ratio_fn(cn):
        Hm = cp.homography_dlt(cn, dst)
        return [cp.poly_area(cp.apply_h(Hm, ring1)) / cp.poly_area(cp.apply_h(Hm, ring2))]

    reg_c = cp.worst_case(sc.corners, d, centroid_fn)
    reg_r = cp.worst_case(sc.corners, d, ratio_fn)
    law1 = ca.TOL_C * cp.ramanujan(big[2] * s0, big[2] * s1) + ca.TOL_C0
    law2 = ca.TOL_C * cp.ramanujan(second[2] * s0, second[2] * s1) + ca.TOL_C0
    src_c = "perspective_rectify.corner_tolerance = TOL_A + TOL_B * blur"
    truth = {
        "corners": S_points(sc.corners, d, src_c),
        "n_blobs": S_int(len(discs), "exact: well separated dark discs (>= 24 card px apart, >= 18 card px radius)"),
        "largest_centroid_xy": S_point(centre_rect, reg_c + 1.0,
                                       "compose.worst_case over corner displacements of " + src_c + " + 1 px (pixel snap of the thresholded blob)"),
        "area_ratio_largest_to_second": S_num(
            a1 / a2, cp.ratio_tol(a1, law1, a2, law2) + reg_r,
            "contour_area law TOL_C * P + TOL_C0 on each rectified ellipse (interval arithmetic for the ratio) + compose.worst_case of the ratio over corner displacements of " + src_c),
    }
    body = (
        "The photo shows one flat rectangular card (bright paper) lying on a darker surface, seen from an arbitrary "
        "angle and rotation. Dark round blobs are printed on the card, away from its edge. Find the card's four "
        f"corners (order them TL, TR, BR, BL by image position: {POSITION_RULE}), then rectify the card: warp it to an "
        f"upright image of {ow} x {oh} px (width x height) so that the TL corner lands on pixel (0, 0), TR on "
        f"({ow - 1}, 0), BR on ({ow - 1}, {oh - 1}) and BL on (0, {oh - 1}). In the rectified image count the dark "
        "blobs, and for the largest blob (by area) report its centroid (x, y) in rectified-image pixels, and the ratio "
        "of its area to the area of the second-largest blob (both measured in the rectified image). Dark pixels along "
        "the border of the rectified image come from the dark surface beyond the card's edge, not from the card: they "
        "are not blobs.")
    schema = {
        "corners": "4 x 2 array [[x, y], ...] of the card corners in the PHOTO, order TL, TR, BR, BL",
        "n_blobs": "integer, number of dark blobs on the card",
        "largest_centroid_xy": "[x, y] in rectified-image pixels",
        "area_ratio_largest_to_second": "number (> 1)",
    }
    return Task("e2e-02-card-blobs", 2, "Rectify the card, then measure its blobs",
                ("find the card corners", "rectify to the stated size", "segment the dark blobs", "count them and measure the largest"),
                ("perspective_rectify", "components_stats", "contour_area"), body, {"photo": _png(img)}, schema, truth,
                {"scene": seed}, extras={"out_size": (ow, oh)},
                notes="photo rendered with the family's pose from compose.render_card_photo")


# ================================================================ T03 strokes-longest (hough_segments)

def _seg_ltol(s: hs.Segment) -> float:
    return 2.0 * hs.tolerance(s)


@_register(3)
def _t03() -> Task:
    def ok(sc):
        if len(sc.segments) < 4:
            return False
        ss = sorted(sc.segments, key=lambda s: -s.length)
        return ss[0].length - ss[1].length > _seg_ltol(ss[0]) + _seg_ltol(ss[1])

    seed, sc = _scan(3, hs.make_scene, ok)
    s = max(sc.segments, key=lambda q: q.length)
    t = hs.tolerance(s)
    src = "hough_segments.tolerance = TOL_A + TOL_B * width, per end point"
    mid = ((s.p1[0] + s.p2[0]) / 2, (s.p1[1] + s.p2[1]) / 2)
    truth = {
        "n_strokes": S_int(len(sc.segments), "exact: one row per stroke is the family's contract"),
        "longest_length_px": S_num(s.length, 2 * t, "distance between the two end points: 2 x " + src),
        "longest_angle_deg": S_angle(cp.direction_deg(s.p1, s.p2), cp.angle_tol_deg(s.length, t),
                                     "asin(2 d / L) with d = " + src, period=180.0),
        "longest_midpoint_xy": S_point(mid, t, "mean of the two end points: " + src),
    }
    body = (
        "The image shows thin bright straight strokes on a dark background (each at least 40 px long; they never touch "
        "or cross). Find every stroke and its two end points. Report how many strokes there are and, for the longest "
        "stroke (the largest distance between its two end points), its length in px, its direction as an angle in "
        "degrees in [0, 180) measured from the +x axis towards +y, i.e. atan2(dy, dx) modulo 180 with y down, and its "
        "midpoint (the mean of its two end points).")
    schema = {
        "n_strokes": "integer",
        "longest_length_px": "number, px",
        "longest_angle_deg": "number in [0, 180), degrees",
        "longest_midpoint_xy": "[x, y]",
    }
    return Task("e2e-03-strokes-longest", 3, "Strokes, then the longest one's length, direction and midpoint",
                ("find every stroke's end points", "pick the longest", "length, direction and midpoint"),
                ("hough_segments",), body, {"strokes": _png(sc.img)}, schema, truth, {"scene": seed})


# ================================================================ T04 strokes-orientation (hough_segments)

AXIS_DEG = 10.0


def _axis_dist(s: hs.Segment) -> float:
    a = cp.direction_deg(s.p1, s.p2)
    return min(a, abs(a - 90.0), 180.0 - a)


@_register(4)
def _t04() -> Task:
    def ok(sc):
        if len(sc.segments) < 4:
            return False
        ss = sorted(sc.segments, key=lambda q: -q.length)
        if any(a.length - b.length <= _seg_ltol(a) + _seg_ltol(b) for a, b in zip(ss, ss[1:])):
            return False
        for q in ss:
            at = cp.angle_tol_deg(q.length, hs.tolerance(q))
            dq = _axis_dist(q)
            if not (dq + at < AXIS_DEG or dq - at > AXIS_DEG):
                return False
        return True

    seed, sc = _scan(4, hs.make_scene, ok)
    ss = sorted(sc.segments, key=lambda q: -q.length)
    src = "hough_segments.tolerance = TOL_A + TOL_B * width, per end point"
    truth = {
        "lengths_desc_px": S_sorted_nums([q.length for q in ss], [2 * hs.tolerance(q) for q in ss],
                                         "each length: 2 x " + src + "; ranks are unambiguous (gaps exceed the sum of the two tolerances, enforced by the seed scan)"),
        "total_length_px": S_num(sum(q.length for q in ss), sum(2 * hs.tolerance(q) for q in ss), "sum of the per-stroke length bounds, each 2 x " + src),
        "n_axis_aligned": S_int(sum(_axis_dist(q) < AXIS_DEG for q in ss),
                                f"exact: every stroke is either within {AXIS_DEG:g} deg - asin(2 d / L) of an image axis or beyond {AXIS_DEG:g} deg + asin(2 d / L) (seed scan)"),
    }
    body = (
        "The image shows thin bright straight strokes on a dark background (each at least 40 px long; they never touch "
        "or cross). Find every stroke and measure its length (the distance between its two end points). Report the "
        "lengths in px sorted from longest to shortest, their sum, and how many strokes are aligned with an image "
        f"axis, meaning that the stroke's direction is within {AXIS_DEG:g} degrees of horizontal or of vertical.")
    schema = {
        "lengths_desc_px": "array of numbers, px, longest first, one per stroke",
        "total_length_px": "number, px",
        "n_axis_aligned": "integer",
    }
    return Task("e2e-04-strokes-orientation", 4, "Strokes, then sorted lengths, total length and axis alignment",
                ("find every stroke's end points", "length of each, sorted", "sum, and count those near an image axis"),
                ("hough_segments",), body, {"strokes": _png(sc.img)}, schema, truth, {"scene": seed})


# ================================================================ T05 component-hist (components_stats + histogram_stats)

BG_LEVEL = 18
FG_MIN = 60


@_register(5)
def _t05() -> Task:
    def ok(sc):
        if sc.seed % 2:
            return False
        comps = sc.truth_for(8)
        a = sorted((c.area for c in comps), reverse=True)
        return len(a) >= 6 and a[0] > a[1] and a[0] % 2 == 0 and 150 <= a[0] <= 2600

    seed, sc = _scan(5, cc.make_scene, ok, step=2)
    labels, comps = cp.label_components(sc.mask, 8)
    truth_c = sc.truth_for(8)
    assert sorted((c.area, c.x, c.y, c.w, c.h) for c in comps) == sorted((c.area, c.x, c.y, c.w, c.h) for c in truth_c)
    rng = _rng(seed, 5)
    k_big = 1 + max(range(len(comps)), key=lambda i: comps[i].area)
    big = comps[k_big - 1]
    H, W = sc.mask.shape
    img = np.full((H, W, 3), BG_LEVEL, np.uint8)
    levels_big = None
    for k, comp in enumerate(comps, 1):
        ys, xs = np.nonzero(labels == k)
        n = comp.area
        if k == k_big:
            v, gap = int(rng.integers(110, 170)), int(rng.integers(3, 12))
            low = v - rng.integers(0, 50, n // 2)
            high = v + gap + rng.integers(0, 70, n - n // 2)
            low[0], high[0] = v, v + gap
            lv = rng.permutation(np.concatenate([low, high]))
            levels_big = lv.astype(np.int64)
        else:
            lv = rng.integers(FG_MIN, 251, n)
        img[ys, xs] = hi._colorize(lv.astype(np.uint8), rng)
    hist = np.bincount(levels_big, minlength=256).astype(np.int64)
    mean, std = hi.exact_stats(hist)
    med = hi.lower_median(hist)
    assert med != hi.upper_median(hist)  # a real tie: np.median-style averaging would be wrong
    rt = hi.FLOAT_RTOL
    truth = {
        "n_components": S_int(len(comps)),
        "largest_area": S_int(big.area),
        "largest_bbox": S_ints(big.bbox),
        "largest_mean_gray": S_rel(mean, rt, "histogram_stats.FLOAT_RTOL (exact integer sums, one rounding)"),
        "largest_std_gray": S_rel(std, rt, "histogram_stats.FLOAT_RTOL, population std (divide by N)"),
        "largest_median_gray": S_int(med, "histogram_stats: the lower median, an integer level, zero tolerance"),
    }
    body = (
        "The image is a micrograph: particles of varying brightness on a uniform dark background (gray level 18). Every "
        "particle pixel is brighter than gray level 59. Group the particle pixels into connected components, where two "
        "pixels are connected when they share an edge or a corner (8-connectivity). Report the number of components. "
        "For the component with the largest pixel area (it is unique) report its area, its tight bounding box "
        "[x, y, w, h] (x and y of the top-left pixel, w and h the number of pixel columns and rows it spans) and the "
        "statistics of the gray levels of its pixels only: the mean, the population standard deviation (divide by N) "
        "and the lower median (the smallest gray level v such that at least half of the component's pixels have gray "
        "level <= v). The gray level of a pixel is its ITU-R BT.601 luma, round(0.299 R + 0.587 G + 0.114 B), the value "
        "cv2.COLOR_BGR2GRAY produces.")
    schema = {
        "n_components": "integer",
        "largest_area": "integer, pixels",
        "largest_bbox": "[x, y, w, h], integers",
        "largest_mean_gray": "number",
        "largest_std_gray": "number",
        "largest_median_gray": "integer gray level",
    }
    return Task("e2e-05-component-hist", 5, "Label the particles, then the gray statistics of the largest",
                ("threshold the particles", "label 8-connected components", "pick the largest, box it", "histogram statistics of its pixels"),
                ("components_stats", "histogram_stats"), body, {"micrograph": _png(img)}, schema, truth, {"scene": seed})


# ================================================================ colour helpers (T06, T10, T20)

_SAFE_HUE = {"red": ((0, 8), (172, 179)), "yellow": ((22, 33),), "green": ((42, 78),), "blue": ((102, 128),)}


def _classify_bgr(bgr) -> str | None:
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    return cs.classify(int(h), int(s), int(v))


def _paint(rng: np.random.Generator, tag: str) -> tuple[int, int, int]:
    """A flat BGR colour. tag: one of the four colour names (comfortably inside the class), 'orange' (full S and V, hue in
    the gap between the ranges) or 'pastel' (a colour's hue, S below 100). Verified against the contract."""
    for _ in range(500):
        if tag == "red":
            lo, hi_ = _SAFE_HUE["red"][int(rng.integers(2))]
        elif tag in _SAFE_HUE:
            lo, hi_ = _SAFE_HUE[tag][0]
        elif tag == "orange":
            lo, hi_ = 12, 18
        elif tag == "pastel":
            lo, hi_ = 0, 179
        else:
            raise ValueError(tag)
        h = int(rng.integers(lo, hi_ + 1))
        s = int(rng.integers(30, 81)) if tag == "pastel" else int(rng.integers(150, 256))
        v = int(rng.integers(172, 255))
        bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])
        want = tag if tag in COLOURS else None
        if _classify_bgr(bgr) == want:
            return bgr
    raise RuntimeError(f"no colour for {tag}")


def _recolour_parts(sc: ca.Scene, tags: list[str], rng: np.random.Generator) -> np.ndarray:
    """The contour_area scene with every shape repainted flat in the colour of its tag."""
    img = sc.img.copy()
    bg = sc.background
    for shape, tag in zip(sc.shapes, tags):
        x0, y0, x1, y1 = shape.box
        sub = img[y0:y1 + 1, x0:x1 + 1]
        m = np.any(sub != bg, axis=2)
        sub[m] = _paint(rng, tag)
    return img


def _part_tags(rng: np.random.Generator, n: int, must: tuple[str, ...]) -> list[str]:
    """Counted colours for most shapes, a distractor (orange, pastel) for about a quarter; every colour in ``must`` at least twice."""
    tags = list(must) * 2
    pool = list(COLOURS) * 2 + ["orange", "pastel"]
    while len(tags) < n:
        tags.append(pool[int(rng.integers(len(pool)))])
    tags = tags[:n]
    return [tags[i] for i in rng.permutation(n)]


def _class_sums(sc: ca.Scene, tags: list[str]) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    area = {c: 0.0 for c in COLOURS}
    tol = {c: 0.0 for c in COLOURS}
    cnt = {c: 0 for c in COLOURS}
    for s, t in zip(sc.shapes, tags):
        if t in COLOURS:
            area[t] += s.area
            tol[t] += ca.tolerance(s)
            cnt[t] += 1
    return area, tol, cnt


def _parts_scene(n: int, *, must: tuple[str, ...], rank_class: str | None = None):
    """Scan for a contour_area scene of 5-7 parts and a colour assignment; returns (seed, scene, tags, img, rng)."""
    base = _base(n)
    for k in range(4000):
        seed = base + k
        sc = ca.make_scene(seed)
        if not 5 <= len(sc.shapes) <= 7:
            continue
        rng = _rng(seed, n)
        tags = _part_tags(rng, len(sc.shapes), must)
        area, tol, cnt = _class_sums(sc, tags)
        if sum(v > 0 for v in cnt.values()) < 3:
            continue
        top = sorted(COLOURS, key=lambda c: -area[c])
        if area[top[0]] - area[top[1]] <= tol[top[0]] + tol[top[1]]:
            continue
        if rank_class is not None and cnt[rank_class] < 2:
            continue
        return seed, sc, tags, _recolour_parts(sc, tags, rng), rng
    raise RuntimeError(f"task {n}: no admissible parts scene")


# ================================================================ T06 parts-colour-area (color_segmentation + contour_area)

@_register(6)
def _t06() -> Task:
    seed, sc, tags, img, _ = _parts_scene(6, must=())
    area, tol, cnt = _class_sums(sc, tags)
    src = "contour_area.tolerance = TOL_C * P + TOL_C0 per shape, summed over the shapes of the colour"
    truth = {
        "area_px2_by_colour": S_dict({c: S_num(area[c], tol[c] if cnt[c] else 0.0, src if cnt[c] else "exact: no part of this colour") for c in COLOURS}),
        "n_parts_by_colour": S_dict({c: S_int(cnt[c], "exact: parts are separated by >= 5 px") for c in COLOURS}),
        "largest_total_colour": S_str(max(COLOURS, key=lambda c: area[c]), "exact: the gap between the two largest totals exceeds the sum of their tolerances (seed scan)"),
    }
    body = (
        "The image shows a few flat-coloured parts lying on a dark belt; a part may be a ring with a hole. "
        + HSV_DEFINITION + " Parts that match none of the four colours must be ignored. For each of red, yellow, green "
        "and blue, count the parts of that colour and add up the area enclosed by each part's outer boundary (a hole "
        "inside a part is NOT subtracted), in px^2. Also report which colour has the largest total area.")
    schema = {
        "area_px2_by_colour": 'object {"red": number, "yellow": number, "green": number, "blue": number}, px^2, 0 for a colour with no part',
        "n_parts_by_colour": 'object {"red": integer, "yellow": integer, "green": integer, "blue": integer}',
        "largest_total_colour": 'one of "red", "yellow", "green", "blue"',
    }
    return Task("e2e-06-parts-colour-area", 6, "Segment parts by colour, then add up their areas",
                ("classify pixels into the four colours", "separate the parts of each colour", "outer-boundary area of each part", "sum per colour"),
                ("color_segmentation", "contour_area"), body, {"parts": _png(img)}, schema, truth, {"scene": seed},
                notes="contour_area scene repainted with colour_segmentation's HSV contract (scene geometry from contour_area)")


# ================================================================ T07 colour-dominant-blob (color_segmentation + components_stats)

@_register(7)
def _t07() -> Task:
    def comps_of(sc, c, connectivity=8):
        ids = [i for i, s in enumerate(sc.shapes, 1) if s.cls == c]
        return cp.label_components(np.isin(sc.labels, ids), connectivity)[1]

    def ok(sc):
        order = sorted(COLOURS, key=lambda c: -sc.truth[c])
        if sc.truth[order[0]] == sc.truth[order[1]]:
            return False
        a = sorted((c.area for c in comps_of(sc, order[0])), reverse=True)
        # the connectivity must matter: 4-connectivity splits a 1-px line or a diagonal into pieces
        return len(a) >= 3 and a[0] > a[1] and len(comps_of(sc, order[0], 4)) != len(a)

    seed, sc = _scan(7, cs.make_scene, ok)
    dom = max(COLOURS, key=lambda c: sc.truth[c])
    comps = comps_of(sc, dom)
    big = max(comps, key=lambda c: c.area)
    truth = {
        "counts": S_dict({c: S_int(sc.truth[c], "color_segmentation.TOL_PX = 0") for c in COLOURS}),
        "dominant": S_str(dom, "exact: unique maximum (seed scan)"),
        "n_components": S_int(len(comps), "exact: 8-connected components of the dominant colour's pixels"),
        "largest_area": S_int(big.area, "components_stats: zero tolerance"),
        "largest_bbox": S_ints(big.bbox, "components_stats: zero tolerance"),
        "largest_centroid": S_point([big.cx, big.cy], cc.CENTROID_TOL, "components_stats.CENTROID_TOL"),
    }
    body = (
        "The image shows flat-coloured shapes on a plain background, plus distractor shapes. " + HSV_DEFINITION +
        " Count the pixels of each of the four colours over the whole image (exactly: no smoothing, resizing or "
        "minimum size). Find the colour with the most pixels. Then group that colour's pixels into connected "
        "components (8-connectivity: pixels sharing an edge or a corner are connected) and report how many components "
        "there are and, for the component with the largest pixel area (it is unique), its area, its tight bounding "
        "box [x, y, w, h] (x, y of the top-left pixel; w, h the pixel columns and rows it spans) and its centroid "
        "(x, y), the mean column and mean row of its pixels.")
    schema = {
        "counts": 'object {"red": integer, "yellow": integer, "green": integer, "blue": integer}, pixels',
        "dominant": 'one of "red", "yellow", "green", "blue"',
        "n_components": "integer",
        "largest_area": "integer, pixels",
        "largest_bbox": "[x, y, w, h], integers",
        "largest_centroid": "[x, y], numbers",
    }
    return Task("e2e-07-colour-dominant-blob", 7, "Count colour pixels, then the largest blob of the dominant colour",
                ("classify pixels into the four colours", "pick the dominant colour", "label its 8-connected components", "area, box and centroid of the largest"),
                ("color_segmentation", "components_stats"), body, {"scene": _png(sc.img)}, schema, truth, {"scene": seed})


# ================================================================ chessboard helpers

def _cb_grid(sc: cb.Scene) -> np.ndarray:
    sx, sy = sc.squares
    return sc.corners.reshape(sy - 1, sx - 1, 2)


def _cb_pitch(sc: cb.Scene) -> tuple[float, float]:
    g = _cb_grid(sc)
    px = float(np.mean(np.linalg.norm(np.diff(g, axis=1), axis=2)))
    py = float(np.mean(np.linalg.norm(np.diff(g, axis=0), axis=2)))
    return px, py


# ================================================================ T08 board-pitch-scale (chessboard_corners)

@_register(8)
def _t08() -> Task:
    seed, sc = _scan(8, cb.make_scene, lambda s: not s.tl_dark)  # a white top-left square: OpenCV lists the corners in reverse
    rng = _rng(seed, 8)
    sq_mm = float(rng.choice([15.0, 20.0, 25.0, 30.0]))
    d = cb.tolerance(sc)
    px, py = _cb_pitch(sc)
    pbar = (px + py) / 2
    src = "chessboard_corners.tolerance(scene) = TOL_M * (alias + noise law)"
    sx, sy = sc.squares
    truth = {
        "n_corners": S_int((sx - 1) * (sy - 1), "exact: (squares_x - 1) * (squares_y - 1)"),
        "first_corner_xy": S_point(_cb_grid(sc)[0, 0], d, src + "; also fixes the corner order"),
        "pitch_x_px": S_num(px, 2 * d, "mean of distances between two corners: 2 x " + src),
        "pitch_y_px": S_num(py, 2 * d, "mean of distances between two corners: 2 x " + src),
        "mm_per_px": S_num(sq_mm / pbar, cp.ratio_tol(sq_mm, 0.0, pbar, 2 * d), "interval arithmetic: square size / mean pitch, pitch within 2 x " + src),
    }
    body = (
        f"The photo shows one chessboard calibration target with {sx} x {sy} squares ({sx} squares along the top edge, "
        f"{sy} along the left edge); each square is {sq_mm:g} mm wide. The board is upright (its top edge is the one "
        "nearest the top of the image, rotated by less than 30 degrees, never mirrored) but seen from an angle, and its "
        "top-left square may be black or white. Find every inner corner (each point where four squares meet) with "
        "sub-pixel accuracy, in row-major order of the board's own frame (the corner nearest the board's top-left "
        "first, then left to right along the top row of corners, then the next row down). Report the number of "
        "corners, the position of the first corner, the mean distance in px between horizontally adjacent corners "
        "(neighbours in the same row of the board, over all rows) and the mean distance between vertically adjacent "
        "corners (neighbours in the same column, over all columns). Then compute the image scale in mm per pixel as the "
        "square size divided by the mean of those two pitches.")
    schema = {
        "n_corners": "integer",
        "first_corner_xy": "[x, y], px",
        "pitch_x_px": "number, px",
        "pitch_y_px": "number, px",
        "mm_per_px": "number",
    }
    return Task("e2e-08-board-pitch-scale", 8, "Chessboard corners, then pitch and image scale",
                ("find the inner corners in board order", "pitch along rows and columns", "scale in mm per pixel"),
                ("chessboard_corners",), body, {"board": _png(sc.img)}, schema, truth, {"scene": seed},
                extras={"squares": tuple(int(v) for v in sc.squares), "square_mm": sq_mm})


# ================================================================ T09 board-corner-lookup (chessboard_corners)

@_register(9)
def _t09() -> Task:
    seed, sc = _scan(9, cb.make_scene, lambda s: s.tl_dark)
    d = cb.tolerance(sc)
    g = _cb_grid(sc)
    sx, sy = sc.squares
    src = "chessboard_corners.tolerance(scene) = TOL_M * (alias + noise law)"
    truth = {
        "corner_row2_col3_xy": S_point(g[2, 3], d, src + "; also fixes the corner order"),
        "grid_centroid_xy": S_point(sc.corners.mean(axis=0), d, "mean of N points each within d: " + src),
        "first_to_last_px": S_num(float(np.linalg.norm(g[-1, -1] - g[0, 0])), 2 * d, "distance between two points: 2 x " + src),
        "top_left_square_is_dark": S_bool(sc.tl_dark, "exact: the top-left square is black in the manufactured board"),
    }
    body = (
        f"The photo shows one chessboard calibration target with {sx} x {sy} squares ({sx} squares along the top edge, "
        f"{sy} along the left edge). The board is upright (its top edge is the one nearest the top of the image, "
        "rotated by less than 30 degrees, never mirrored) but seen from an angle. Find every inner corner (each point "
        "where four squares meet) with sub-pixel accuracy and number them in the board's own frame: row-major, the "
        "corner nearest the board's top-left first, then left to right along the top row of corners, then the next row "
        "down; rows and columns count from 0. Report the corner in row 2, column 3, the centroid (mean position) of "
        "all inner corners, and the distance in px between the first and the last corner. Finally decide from the "
        "image whether the board's top-left square (the square at the top-left end of the board, whose lower-right "
        "corner is the first inner corner) is dark (black) or light (white).")
    schema = {
        "corner_row2_col3_xy": "[x, y], px",
        "grid_centroid_xy": "[x, y], px",
        "first_to_last_px": "number, px",
        "top_left_square_is_dark": "true or false",
    }
    return Task("e2e-09-board-corner-lookup", 9, "Chessboard corners in board order, a lookup and the corner colour",
                ("find the inner corners in board order", "look up row 2 column 3, centroid and span", "classify the top-left square"),
                ("chessboard_corners",), body, {"board": _png(sc.img)}, schema, truth, {"scene": seed},
                extras={"squares": tuple(int(v) for v in sc.squares)})


# ================================================================ T10 board-scale-parts (chessboard_corners + color_segmentation + contour_area)

@_register(10)
def _t10() -> Task:
    n = 10
    seed_b, sc_b = _scan(n, cb.make_scene)
    seed_p, sc_p, tags, img, _ = _parts_scene(n, must=("red",), rank_class="red")
    rng = _rng(seed_b, n)
    sq_mm = float(rng.choice([15.0, 20.0, 25.0, 30.0]))
    d = cb.tolerance(sc_b)
    px, py = _cb_pitch(sc_b)
    pbar = (px + py) / 2
    area, tol, cnt = _class_sums(sc_p, tags)
    a_red, t_red = area["red"], tol["red"]
    s_true = sq_mm / pbar
    s_lo, s_hi = sq_mm / (pbar + 2 * d), sq_mm / (pbar - 2 * d)
    cand = [(a_red + sa * t_red) * ss ** 2 for sa in (-1, 1) for ss in (s_lo, s_hi)]
    tol_mm2 = max(abs(c - a_red * s_true ** 2) for c in cand)
    sx, sy = sc_b.squares
    src_b = "chessboard_corners.tolerance(scene)"
    truth = {
        "mm_per_px": S_num(s_true, cp.ratio_tol(sq_mm, 0.0, pbar, 2 * d), "interval arithmetic: square size / mean pitch, pitch within 2 x " + src_b),
        "n_red_parts": S_int(cnt["red"], "exact: parts are separated by >= 5 px"),
        "red_area_mm2": S_num(a_red * s_true ** 2, tol_mm2,
                              "interval arithmetic on (sum of outer areas +- sum of contour_area.tolerance) x (scale +- its bound from 2 x " + src_b + ")"),
    }
    body = (
        f"Two photos taken with the same camera at the same distance. board.png shows a chessboard calibration target "
        f"with {sx} x {sy} squares ({sx} along the top edge, {sy} along the left edge), each square {sq_mm:g} mm wide; "
        "the board is upright (rotated by less than 30 degrees, never mirrored) but seen from an angle. parts.png shows "
        "flat-coloured parts lying on a dark belt; a part may be a ring with a hole. Calibrate the scale from the "
        "board: find the inner corners with sub-pixel accuracy, take the mean distance in px between adjacent corners "
        "along the rows and the mean along the columns, average those two pitches, and set mm_per_px = square size "
        "/ that mean pitch. Then, in parts.png, " + RED_DEFINITION + " Parts of any other colour (yellow, green, "
        "blue, orange, pastel) must be ignored. Count the red parts, add up the area enclosed by each red part's outer "
        "boundary (a hole inside a part is NOT subtracted) in px^2 and convert the total to mm^2 with the scale: "
        "area_mm2 = area_px2 * mm_per_px^2.")
    schema = {
        "mm_per_px": "number",
        "n_red_parts": "integer",
        "red_area_mm2": "number, mm^2",
    }
    return Task("e2e-10-board-scale-parts", 10, "Calibrate a scale on the board, then measure red parts in mm^2",
                ("chessboard corners in board order", "mean pitch -> mm per pixel", "segment red in parts.png", "outer-boundary areas, converted to mm^2"),
                ("chessboard_corners", "color_segmentation", "contour_area"), body,
                {"board": _png(sc_b.img), "parts": _png(img)}, schema, truth, {"board": seed_b, "parts": seed_p},
                extras={"squares": tuple(int(v) for v in sc_b.squares), "square_mm": sq_mm})


# ================================================================ T11 / T12 view-roi-map, view-pose (feature_homography)

ROI = cp.apply_h(np.eye(3), fh.frame_corners(fh.GUARD))  # img_a's frame inset by GUARD: (32, 32) .. (255, 183)


def _roi_text() -> str:
    return "[[%d, %d], [%d, %d], [%d, %d], [%d, %d]]" % tuple(int(v) for v in ROI.ravel())


@_register(11)
def _t11() -> Task:
    seed, sc = _scan(11, fh.make_scene)
    d = fh.tolerance(sc)
    mapped = fh.project(sc.H, ROI)
    src = "feature_homography.tolerance = TOL_B / sqrt(N) (N = true correspondences); the ROI corners are the family's probe points"
    truth = {
        "H": S_homography(sc.H, ROI, d, "probe error at the ROI corners <= " + src),
        "roi_corners_b": S_points(mapped, d, src),
        "roi_area_b_px2": S_num(cp.poly_area(mapped), cp.area_tol(mapped, d), "shoelace expansion in the vertex displacements (compose.area_tol), vertex tolerance = " + src),
        "roi_centroid_b": S_point(mapped.mean(axis=0), d, "mean of 4 points each within d: " + src),
    }
    body = (
        "img_b shows the same flat, textured surface as img_a from a different viewpoint (img_b is a wider view). "
        "Match local features between the two images and estimate the 3x3 homography H that maps pixel coordinates "
        "(x, y) in img_a to the corresponding pixel coordinates in img_b, normalised so that H[2][2] = 1. A "
        f"rectangular region of interest is marked in img_a by its corners {_roi_text()} (in this order). Map those "
        "four corners into img_b. Report H, the mapped corners, the area in px^2 of the quadrilateral they form in "
        "img_b (shoelace formula), and the centroid of the quadrilateral taken as the mean of its four vertices.")
    schema = {
        "H": "3 x 3 array (list of 3 rows), H[2][2] = 1",
        "roi_corners_b": "4 x 2 array [[x, y], ...] in img_b, same order as the corners given for img_a",
        "roi_area_b_px2": "number, px^2",
        "roi_centroid_b": "[x, y] in img_b",
    }
    return Task("e2e-11-view-roi-map", 11, "Two views: homography, then map a region and measure it",
                ("match features, estimate H", "map the ROI corners to img_b", "area and centroid of the mapped quadrilateral"),
                ("feature_homography",), body, {"img_a": _png(sc.img_a), "img_b": _png(sc.img_b)}, schema, truth, {"scene": seed})


@_register(12)
def _t12() -> Task:
    seed, sc = _scan(12, fh.make_scene)
    d = fh.tolerance(sc)
    mapped = fh.project(sc.H, ROI)
    v_b, v_a = mapped[1] - mapped[0], ROI[1] - ROI[0]
    la, lb = float(np.linalg.norm(v_a)), float(np.linalg.norm(v_b))
    src = "feature_homography.tolerance = TOL_B / sqrt(N) (N = true correspondences); the ROI corners are the family's probe points"
    truth = {
        "rotation_deg": S_angle(math.degrees(math.atan2(v_b[1], v_b[0])), cp.angle_tol_deg(lb, d),
                                "asin(2 d / L), L = mapped top-edge length, d = " + src, period=360.0),
        "edge_scale": S_num(lb / la, 2 * d / la, "ratio of a distance within 2 d to the exact length of the img_a edge: 2 d / L_a, d = " + src),
        "roi_centre_b": S_point(mapped.mean(axis=0), d, "mean of 4 points each within d: " + src),
    }
    body = (
        "img_b shows the same flat, textured surface as img_a from a different viewpoint (img_b is a wider view). "
        "Match local features between the two images and estimate the homography that maps pixel coordinates (x, y) "
        "in img_a to the corresponding coordinates in img_b. A rectangular region of interest is marked in img_a by its "
        f"corners TL, TR, BR, BL = {_roi_text()}. Map the corners into img_b and report: the rotation of the region's "
        "top edge (from the mapped TL corner to the mapped TR corner), as the angle atan2(dy, dx) in degrees (y down, "
        "in (-180, 180]), measured in img_b; the edge scale, the length in px of that mapped top edge divided by the "
        "length of the same edge in img_a; and the centre of the mapped region, the mean of the four mapped corners.")
    schema = {
        "rotation_deg": "number in (-180, 180], degrees",
        "edge_scale": "number",
        "roi_centre_b": "[x, y] in img_b",
    }
    return Task("e2e-12-view-pose", 12, "Two views: homography, then the region's rotation, scale and centre",
                ("match features, estimate H", "map the ROI corners", "rotation, scale and centre of the mapped top edge"),
                ("feature_homography",), body, {"img_a": _png(sc.img_a), "img_b": _png(sc.img_b)}, schema, truth, {"scene": seed})


# ================================================================ T13 / T14 field-label-fit / -shrink (text_layout)

FACE_NAMES = {cv2.FONT_HERSHEY_SIMPLEX: "cv2.FONT_HERSHEY_SIMPLEX", cv2.FONT_HERSHEY_PLAIN: "cv2.FONT_HERSHEY_PLAIN",
              cv2.FONT_HERSHEY_DUPLEX: "cv2.FONT_HERSHEY_DUPLEX", cv2.FONT_HERSHEY_COMPLEX: "cv2.FONT_HERSHEY_COMPLEX",
              cv2.FONT_HERSHEY_TRIPLEX: "cv2.FONT_HERSHEY_TRIPLEX",
              cv2.FONT_HERSHEY_COMPLEX_SMALL: "cv2.FONT_HERSHEY_COMPLEX_SMALL"}


def _field_task(n: int, shrink: bool) -> Task:
    seed, sc = _scan(n, tl.make_scene, lambda s: bool(s.shrink) == shrink, start=1 if shrink else 0, step=2)
    rng = _rng(seed, n)
    x, y, w, h = sc.box
    bg = sc.img[0, 0].astype(int)
    outline = tuple(int(b + rng.integers(70, 121)) if b < 128 else int(b - rng.integers(70, 121)) for b in bg)
    canvas = sc.img.copy()
    canvas[y - 1, x - 1:x + w + 1] = outline
    canvas[y + h, x - 1:x + w + 1] = outline
    canvas[y - 1:y + h + 1, x - 1] = outline
    canvas[y - 1:y + h + 1, x + w] = outline
    scene = tl.Scene(canvas, sc.box, sc.text, sc.face, sc.scale, sc.thickness, sc.color, sc.shrink, sc.s_fit, sc.seed)
    truth = {
        "box": S_ints(sc.box, "exact: the outline's interior, from the construction"),
        "scale": S_present(),
        "text_size": S_present(),
        "layout": S_layout("text_layout.verify_manufactured on out.png: containment, scale rule, glyph size, centring, reported size, colour (the family's own laws)"),
    }
    shrink_rule = (
        "If the text does not fit in the interior at that scale (wider than the interior, or taller than the interior "
        "including the baseline), shrink the font scale to the largest scale at which it does fit; a scale up to 10 % "
        "below that largest scale is accepted. ")
    body = (
        "The image shows a form field: a one-pixel-wide rectangular outline on a plain background. The field's "
        "interior is the set of pixels strictly inside the outline (the outline itself is not part of it). Draw the text "
        f"{json.dumps(sc.text)} on a copy of the image with cv2.putText using the Hershey font {FACE_NAMES[sc.face]} "
        f"(value {sc.face}), requested font scale {sc.scale}, line thickness {sc.thickness} and BGR colour "
        f"{[int(c) for c in sc.color]}, so that the text is centred in the interior, horizontally and vertically. "
        "Centre the box that cv2.getTextSize reports for the text (its width, its height plus the baseline below it). "
        + shrink_rule +
        "Never change a pixel outside the interior (the outline must stay intact). Save the resulting image as "
        "out.png in your work directory (PNG, same size as the input). Report the field's interior box, the font scale "
        "you drew with and the (width, height) that cv2.getTextSize reported at that scale.")
    schema = {
        "box": "[x, y, w, h], integers: the interior covers columns x..x+w-1 and rows y..y+h-1",
        "scale": "number, the font scale used for drawing",
        "text_size": "[width, height], integers, as returned by cv2.getTextSize at that scale and thickness (height without the baseline)",
    }
    kind = "shrinks" if shrink else "fits"
    return Task(f"e2e-{n:02d}-field-label-{'shrink' if shrink else 'fit'}", n,
                f"Find a form field, then centre a label in it (the text {kind})",
                ("find the field's outline and its interior box", "measure the text with getTextSize and fit the scale", "draw it centred, save out.png, report"),
                ("text_layout",), body, {"field": _png(canvas)}, schema, truth, {"scene": seed},
                extras={"tl_scene": scene, "outline": outline})


@_register(13)
def _t13() -> Task:
    return _field_task(13, shrink=False)


@_register(14)
def _t14() -> Task:
    return _field_task(14, shrink=True)


# ================================================================ T15 / T16 exposure (histogram_stats)

def _hist_margin(sc: hi.Scene) -> float:
    lv = np.flatnonzero(sc.hist)
    return float(np.min(np.abs(np.abs(lv - sc.mean) - 2 * sc.std)))


def _n_out(sc: hi.Scene) -> int:
    lv = np.arange(256)
    return int(sc.hist[np.abs(lv - sc.mean) > 2 * sc.std].sum())


@_register(15)
def _t15() -> Task:
    seed, sc = _scan(15, hi.make_scene, lambda s: s.tie and _hist_margin(s) > 1e-3 and _n_out(s) >= 5)
    rt = hi.FLOAT_RTOL
    truth = {
        "mean_gray": S_rel(sc.mean, rt, "histogram_stats.FLOAT_RTOL"),
        "std_gray": S_rel(sc.std, rt, "histogram_stats.FLOAT_RTOL, population std"),
        "median_gray": S_int(sc.median, "histogram_stats: the lower median, zero tolerance (this scene has a real tie)"),
        "n_outliers": S_int(_n_out(sc), "exact: counts are exact and no gray level lies within 1e-3 of mean +- 2 std (seed scan)"),
    }
    body = (
        "Analyse the exposure of the image. The gray level of a pixel is its ITU-R BT.601 luma, round(0.299 R + 0.587 G "
        "+ 0.114 B), the value cv2.COLOR_BGR2GRAY produces. Compute the 256-bin histogram of the gray levels and from it "
        "the mean, the population standard deviation (divide by N, not N-1) and the lower median (the smallest gray "
        "level v such that at least half of the pixels have gray level <= v). Then count the outliers: the pixels whose "
        "gray level differs from the mean by more than two standard deviations, |gray - mean| > 2 * std.")
    schema = {
        "mean_gray": "number",
        "std_gray": "number",
        "median_gray": "integer gray level",
        "n_outliers": "integer, pixels",
    }
    return Task("e2e-15-exposure-outliers", 15, "Gray histogram statistics, then the 2-sigma outliers",
                ("gray histogram", "mean, population std, lower median", "count pixels beyond mean +- 2 std"),
                ("histogram_stats",), body, {"image": _png(sc.img)}, schema, truth, {"scene": seed})


@_register(16)
def _t16() -> Task:
    base = _base(16)
    seed_a = seed_b = None
    for k in range(4000):
        a = hi.make_scene(base + 2 * k)
        b = hi.make_scene(base + 2 * k + 1)
        if abs(a.mean - b.mean) > 3 and abs(a.std - b.std) > 3 and a.median != b.median:
            seed_a, seed_b, sa, sb = base + 2 * k, base + 2 * k + 1, a, b
            break
    else:
        raise RuntimeError("task 16: no admissible pair")
    rt = hi.FLOAT_RTOL
    truth = {
        "mean_a": S_rel(sa.mean, rt, "histogram_stats.FLOAT_RTOL"),
        "mean_b": S_rel(sb.mean, rt, "histogram_stats.FLOAT_RTOL"),
        "median_a": S_int(sa.median, "histogram_stats: the lower median, zero tolerance"),
        "median_b": S_int(sb.median, "histogram_stats: the lower median, zero tolerance"),
        "brighter": S_str("a" if sa.mean > sb.mean else "b", "exact: means differ by more than 3 gray levels (seed scan)"),
        "higher_contrast": S_str("a" if sa.std > sb.std else "b", "exact: stds differ by more than 3 gray levels (seed scan)"),
    }
    body = (
        "Two exposures of the same scene, a.png and b.png. For each image compute the mean and the lower median of its "
        "gray levels. The gray level of a pixel is its ITU-R BT.601 luma, round(0.299 R + 0.587 G + 0.114 B), the value "
        "cv2.COLOR_BGR2GRAY produces; the lower median is the smallest gray level v such that at least half of the "
        "image's pixels have gray level <= v. Then decide which image is brighter (larger mean) and which has the "
        "higher contrast (larger population standard deviation of the gray levels, dividing by N).")
    schema = {
        "mean_a": "number",
        "mean_b": "number",
        "median_a": "integer gray level",
        "median_b": "integer gray level",
        "brighter": 'either "a" or "b"',
        "higher_contrast": 'either "a" or "b"',
    }
    return Task("e2e-16-exposure-compare", 16, "Two exposures: statistics of each, then compare",
                ("statistics of a.png", "statistics of b.png", "compare brightness and contrast"),
                ("histogram_stats",), body, {"a": _png(sa.img), "b": _png(sb.img)}, schema, truth,
                {"a": seed_a, "b": seed_b})


# ================================================================ T17 shape-rank (contour_area)

@_register(17)
def _t17() -> Task:
    def plan(sc):
        if len(sc.shapes) < 5:
            return None
        ss = sorted(sc.shapes, key=lambda s: -s.area)
        tols = [ca.tolerance(s) for s in ss]
        if any(ss[i].area - ss[i + 1].area <= tols[i] + tols[i + 1] for i in range(len(ss) - 1)):
            return None
        best = None
        for T in range(200, int(ss[0].area), 100):
            if all(abs(s.area - T) > t for s, t in zip(ss, tols)):
                n_large = sum(s.area > T for s in ss)
                if best is None or abs(n_large - len(ss) / 2) < abs(best[1] - len(ss) / 2):
                    best = (T, n_large)
        return (ss, tols, best) if best else None

    seed, sc = _scan(17, ca.make_scene, lambda s: plan(s) is not None)
    ss, tols, (T, n_large) = plan(sc)
    areas = [s.area for s in ss]
    share = areas[0] / sum(areas)
    src = "contour_area.tolerance = TOL_C * P + TOL_C0 per shape"
    truth = {
        "n_shapes": S_int(len(ss), "exact: shapes are separated by >= 5 px"),
        "areas_desc_px2": S_sorted_nums(areas, tols, src + "; ranks are unambiguous (adjacent gaps exceed the sum of the two tolerances, seed scan)"),
        "n_large": S_int(n_large, f"exact: no shape's area lies within its tolerance of the {T} px^2 threshold (seed scan)"),
        "largest_share": S_num(share, _share_tol(areas, tols),
                               "interval arithmetic on (largest area) / (total area), each area within " + src),
    }
    body = (
        "The image shows separate bright shapes on a dark background (a shape may have a hole). Find every shape and measure "
        "the area enclosed by its outer boundary in px^2 (a hole inside a shape is NOT subtracted). Report the number "
        "of shapes, the areas sorted from largest to smallest, how many shapes have an area larger than "
        f"{T} px^2, and the share of the largest shape: its area divided by the sum of all the shapes' areas.")
    schema = {
        "n_shapes": "integer",
        "areas_desc_px2": "array of numbers, px^2, largest first, one per shape",
        "n_large": f"integer, shapes with area > {T} px^2",
        "largest_share": "number between 0 and 1",
    }
    return Task("e2e-17-shape-rank", 17, "Shape areas, then rank, threshold count and share",
                ("find every shape", "outer-boundary area of each", "sort, count above a threshold, largest share"),
                ("contour_area",), body, {"shapes": _png(sc.img)}, schema, truth, {"scene": seed},
                extras={"threshold": int(T)})


def _share_tol(areas, tols) -> float:
    a1, t1 = areas[0], tols[0]
    rest, tr = sum(areas[1:]), sum(tols[1:])
    s = a1 / (a1 + rest)
    up = (a1 + t1) / (a1 + t1 + rest - tr)
    dn = (a1 - t1) / (a1 - t1 + rest + tr)
    return max(abs(up - s), abs(dn - s))


# ================================================================ T18 mask-table (components_stats)

@_register(18)
def _t18() -> Task:
    def ok(sc):
        if sc.seed % 2:
            return False
        a = sorted((c.area for c in sc.truth_for(8)), reverse=True)
        return a[0] > a[1]

    seed, sc = _scan(18, cc.make_scene, ok, step=2)
    comps = sc.truth_for(8)
    H, W = sc.mask.shape
    big = comps[0]  # truth_for sorts by (-area, y, x)
    border = sum(1 for c in comps if c.x == 0 or c.y == 0 or c.x + c.w == W or c.y + c.h == H)
    truth = {
        "n_components": S_int(len(comps)),
        "largest_area": S_int(big.area),
        "largest_bbox": S_ints([big.x, big.y, big.w, big.h]),
        "largest_centroid": S_point([big.cx, big.cy], cc.CENTROID_TOL, "components_stats.CENTROID_TOL"),
        "n_touching_border": S_int(border, "exact: components whose bounding box touches an image edge"),
        "n_single_pixel": S_int(sum(c.area == 1 for c in comps), "exact"),
    }
    body = (
        "mask.png is a binary mask (0 = background, 255 = foreground). Group the foreground pixels into connected "
        "components, where two pixels are connected when they share an edge or a corner (8-connectivity); the "
        "background is not a component. Report the number of components; for the component with the largest pixel area "
        "(it is unique) its area, its tight bounding box [x, y, w, h] (x, y of the top-left pixel; w, h the pixel "
        "columns and rows it spans) and its centroid (x, y), the mean column and mean row of its pixels; how many "
        "components touch the image border (any of their pixels lies in the first or last row or column of the mask); "
        "and how many components consist of a single pixel.")
    schema = {
        "n_components": "integer",
        "largest_area": "integer, pixels",
        "largest_bbox": "[x, y, w, h], integers",
        "largest_centroid": "[x, y], numbers",
        "n_touching_border": "integer",
        "n_single_pixel": "integer",
    }
    return Task("e2e-18-mask-table", 18, "Label a mask, then the largest component and two counts",
                ("label 8-connected components", "area, box and centroid of the largest", "count border-touching and single-pixel components"),
                ("components_stats",), body, {"mask": _png(sc.mask.astype(np.uint8) * 255)}, schema, truth, {"scene": seed})


# ================================================================ T19 card-strokes (perspective_rectify + hough_segments)

def _place_strokes(rng: np.random.Generator, cw: float, ch: float):
    """3-4 strokes (card px, centred), each >= 80 px long, centre lines >= 30 px apart, width 2 or 3 card px, inside the printed area."""
    hx, hy = 0.4 * cw - 10.0, 0.4 * ch - 10.0
    for _ in range(2000):
        k = int(rng.integers(3, 5))
        out = []
        for _ in range(60 * k):
            if len(out) == k:
                break
            w = float(rng.integers(2, 4))
            L = float(rng.uniform(80.0, 220.0))
            th = float(rng.uniform(0, math.pi))
            x1, y1 = float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))
            x2, y2 = x1 + L * math.cos(th), y1 + L * math.sin(th)
            if not (abs(x2) <= hx and abs(y2) <= hy):
                continue
            if any(cp.seg_distance((x1, y1), (x2, y2), (a, b), (c, d)) < 30.0 for a, b, c, d, _ in out):
                continue
            out.append((x1, y1, x2, y2, w))
        if len(out) == k:
            return out
    raise RuntimeError("no stroke layout")


@_register(19)
def _t19() -> Task:
    n = 19
    base = _base(n)
    for kk in range(400):
        seed = base + kk
        sc = pr.make_scene(seed)
        rng = _rng(seed, n)
        cw, ch = sc.card
        if min(cw, ch) < 150 or min(sc.out_w, sc.out_h) < 140:
            continue
        strokes = _place_strokes(rng, cw, ch)
        Q = cp.card_to_rect(sc)
        assert np.allclose(Q[2, :2], 0.0)
        s0, s1 = np.linalg.svd(Q[:2, :2])[1]
        d = pr.corner_tolerance(sc)
        dst = cp.rect_dst(sc.out_w, sc.out_h)
        lens = []
        for x1, y1, x2, y2, w in strokes:
            e = cp.apply_h(Q, [[x1, y1], [x2, y2]])
            lens.append((float(np.linalg.norm(e[1] - e[0])), w * max(s0, s1), e, (x1, y1, x2, y2)))
        order = sorted(range(len(strokes)), key=lambda i: -lens[i][0])
        # propagation for the longest and the runner-up
        def reg_len(i):
            ph = cp.apply_h(sc.H_ci, [[strokes[i][0], strokes[i][1]], [strokes[i][2], strokes[i][3]]])
            return cp.worst_case(sc.corners, d, lambda cn: [np.linalg.norm(np.diff(cp.apply_h(cp.homography_dlt(cn, dst), ph), axis=0))])
        i1, i2 = order[0], order[1]
        r1, r2 = reg_len(i1), reg_len(i2)
        t1, t2 = hs.TOL_A + hs.TOL_B * lens[i1][1], hs.TOL_A + hs.TOL_B * lens[i2][1]
        L1, L2 = lens[i1][0], lens[i2][0]
        if L1 < 130.0 or L1 - L2 <= 2 * t1 + r1 + 2 * t2 + r2:
            continue
        break
    else:
        raise RuntimeError("task 19: no admissible seed")
    paper, ink = rng.uniform(225, 245, 3), rng.uniform(25, 60, 3)
    img = cp.render_card_photo(sc, rng, paper=paper, ink=ink, strokes=strokes)
    e = lens[i1][2]
    ph = cp.apply_h(sc.H_ci, [[strokes[i1][0], strokes[i1][1]], [strokes[i1][2], strokes[i1][3]]])

    def dir_fn(cn):
        v = np.diff(cp.apply_h(cp.homography_dlt(cn, dst), ph), axis=0)[0]
        return v / np.linalg.norm(v)

    reg_dir = math.degrees(2 * math.asin(min(1.0, cp.worst_case(sc.corners, d, dir_fn) / 2)))
    src_c = "perspective_rectify.corner_tolerance = TOL_A + TOL_B * blur"
    law = "hough_segments.tolerance = TOL_A + TOL_B * width (width in rectified px), per end point"
    truth = {
        "corners": S_points(sc.corners, d, src_c),
        "n_strokes": S_int(len(strokes), "exact: strokes >= 30 px apart and >= 80 card px long"),
        "longest_length_px": S_num(L1, 2 * t1 + r1, "distance between two end points: 2 x " + law + " + compose.worst_case over corner displacements of " + src_c),
        "longest_angle_deg": S_angle(cp.direction_deg(e[0], e[1]), cp.angle_tol_deg(L1, t1) + reg_dir,
                                     "asin(2 d / L) with d = " + law + " + compose.worst_case direction change over corner displacements of " + src_c, period=180.0),
    }
    ow, oh = sc.out_w, sc.out_h
    body = (
        "The photo shows one flat rectangular card (bright paper) lying on a darker surface, seen from an arbitrary "
        "angle and rotation. Thin dark straight strokes are printed on the card, away from its edge. Find the card's "
        f"four corners (order them TL, TR, BR, BL by image position: {POSITION_RULE}), then rectify the card: warp it "
        f"to an upright image of {ow} x {oh} px (width x height) so that the TL corner lands on pixel (0, 0), TR on "
        f"({ow - 1}, 0), BR on ({ow - 1}, {oh - 1}) and BL on (0, {oh - 1}). In the rectified image find every "
        "stroke and its two end points. Report the number of strokes and, for the longest stroke (the largest distance "
        "between its end points, measured in the rectified image), its length in rectified-image px and its direction "
        "as an angle in degrees in [0, 180), atan2(dy, dx) modulo 180 with y down, in the rectified image. Dark pixels "
        "along the border of the rectified image come from the dark surface beyond the card's edge, not from the card: "
        "they are not strokes.")
    schema = {
        "corners": "4 x 2 array [[x, y], ...] of the card corners in the PHOTO, order TL, TR, BR, BL",
        "n_strokes": "integer",
        "longest_length_px": "number, rectified-image px",
        "longest_angle_deg": "number in [0, 180), degrees, in the rectified image",
    }
    return Task("e2e-19-card-strokes", 19, "Rectify the card, then find its strokes and the longest",
                ("find the card corners", "rectify to the stated size", "find the strokes in the rectified image", "length and direction of the longest"),
                ("perspective_rectify", "hough_segments"), body, {"photo": _png(img)}, schema, truth, {"scene": seed},
                extras={"out_size": (ow, oh)}, notes="photo rendered with the family's pose from compose.render_card_photo")


# ================================================================ T20 palette-luma (color_segmentation + histogram_stats)

@_register(20)
def _t20() -> Task:
    def ok(sc):
        if sc.omitted is not None:
            return False
        shapes = [s for s in sc.shapes if s.cls is not None]
        if any(cp.luma_exact(s.bgr)[1] < 0.02 for s in shapes):  # cvtColor and the exact formula agree away from a tie
            return False
        m = _mean_gray(sc)
        v = sorted(m.values())
        return all(b - a > 0.01 for a, b in zip(v, v[1:]))

    seed, sc = _scan(20, cs.make_scene, ok)
    m = _mean_gray(sc)
    rt = hi.FLOAT_RTOL
    truth = {
        "counts": S_dict({c: S_int(sc.truth[c], "color_segmentation.TOL_PX = 0") for c in COLOURS}),
        "mean_gray_by_colour": S_dict({c: S_rel(m[c], rt, "histogram_stats.FLOAT_RTOL: exact integer sums; no colour within 0.02 of a luma rounding tie (seed scan)") for c in COLOURS}),
        "brightest": S_str(max(COLOURS, key=lambda c: m[c]), "exact: means differ by more than 0.01 gray levels (seed scan)"),
        "darkest": S_str(min(COLOURS, key=lambda c: m[c]), "exact"),
    }
    body = (
        "The image shows flat-coloured shapes on a plain background, plus distractor shapes. " + HSV_DEFINITION +
        " For each of the four colours count its pixels over the whole image (exactly: no smoothing, resizing or "
        "minimum size) and compute the mean gray level of those pixels only. The gray level of a pixel is its ITU-R "
        "BT.601 luma, round(0.299 R + 0.587 G + 0.114 B), the value cv2.COLOR_BGR2GRAY produces. Then name the colour "
        "whose pixels have the highest mean gray level (the brightest) and the one with the lowest (the darkest).")
    schema = {
        "counts": 'object {"red": integer, "yellow": integer, "green": integer, "blue": integer}, pixels',
        "mean_gray_by_colour": 'object {"red": number, "yellow": number, "green": number, "blue": number}',
        "brightest": 'one of "red", "yellow", "green", "blue"',
        "darkest": 'one of "red", "yellow", "green", "blue"',
    }
    return Task("e2e-20-palette-luma", 20, "Segment the four colours, then the gray statistics of each",
                ("classify pixels into the four colours", "count each colour", "mean gray level of each colour's pixels", "rank brightness"),
                ("color_segmentation", "histogram_stats"), body, {"scene": _png(sc.img)}, schema, truth, {"scene": seed})


def _mean_gray(sc: cs.Scene) -> dict[str, float]:
    tot = {c: 0 for c in COLOURS}
    n = {c: 0 for c in COLOURS}
    for s in sc.shapes:
        if s.cls is not None:
            tot[s.cls] += s.area * cp.luma_exact(s.bgr)[0]
            n[s.cls] += s.area
    return {c: tot[c] / n[c] for c in COLOURS}


# ================================================================ public API

TASK_IDS = ("e2e-01-card-geometry", "e2e-02-card-blobs", "e2e-03-strokes-longest", "e2e-04-strokes-orientation",
            "e2e-05-component-hist", "e2e-06-parts-colour-area", "e2e-07-colour-dominant-blob",
            "e2e-08-board-pitch-scale", "e2e-09-board-corner-lookup", "e2e-10-board-scale-parts",
            "e2e-11-view-roi-map", "e2e-12-view-pose", "e2e-13-field-label-fit", "e2e-14-field-label-shrink",
            "e2e-15-exposure-outliers", "e2e-16-exposure-compare", "e2e-17-shape-rank", "e2e-18-mask-table",
            "e2e-19-card-strokes", "e2e-20-palette-luma")
assert len(TASK_IDS) == N_TASKS


def task_number(task_id: str) -> int:
    return TASK_IDS.index(task_id) + 1


def build(task_id_or_number) -> Task:
    """Build (and cache) one task; deterministic in its number."""
    n = task_id_or_number if isinstance(task_id_or_number, int) else task_number(task_id_or_number)
    if n not in _CACHE:
        t = _BUILDERS[n]()
        assert t.id == TASK_IDS[n - 1], (t.id, TASK_IDS[n - 1])
        _CACHE[n] = t
    return _CACHE[n]


def build_all() -> list[Task]:
    return [build(n) for n in range(1, N_TASKS + 1)]


def fresh(task_id_or_number) -> Task:
    """Rebuild without the cache (the determinism test compares it with the cached one)."""
    n = task_id_or_number if isinstance(task_id_or_number, int) else task_number(task_id_or_number)
    return _BUILDERS[n]()


def seed_values(task: Task) -> list[int]:
    return list(task.seeds.values())
