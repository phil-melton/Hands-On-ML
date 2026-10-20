"""Family ``hough_segments``: end points of straight strokes, verified by a manufactured solution.

Method of Manufactured Solutions again: choose the answer first (segments with known
integer end points and a known stroke width), draw the image from it with ``cv2.line``,
then require the code to recover the end points within a tolerance derived from the
physics of the measurement.

Tolerance physics (the law's form was written down before its constants were fitted)
------------------------------------------------------------------------------------
A stroke of width ``t`` drawn by ``cv2.line`` is a band with round caps. A detector sees
the band, not the centre line it was drawn from, so an end point it reports can differ
from the manufactured one (the centre-line end) through four mechanisms:

1. *along the axis*, the round cap extends the stroke past the end point by ceil(t/2) px, except
   that a 1-px line has no cap at all (read off the pixels of 3000 random angles per width in
   OpenCV 5.0.0: at most 0, 1, 2, 2 px for t = 1..4, so t/2 rounded up, not t/2; fitLine finds
   exactly that);
2. *across the axis*, a detector may run along any part of the band: up to t/2;
3. *pixel snapping*: every reported coordinate is rounded, +-0.5 px per axis;
4. *pre-filter*: a pipeline that widens the mask with a 3x3 dilation (the standard remedy
   for thin strokes, see below) and takes the end points from the widened mask reads
   them +1 px per axis too far (``fitLine(dilated)`` in the panel is that, without Hough).

(1) and (2) are orthogonal, so they add in quadrature: t/2 * sqrt(2) ~ 0.7 t. (3) and (4)
do not scale with t. The law is therefore::

    end point error <= TOL_A + TOL_B * t        (px, per end point; t = stroke width)

with TOL_B = 0.7 (geometry) and TOL_A (snapping + pre-filter) fitted on gauge blocks, see
the constants below: 2.6 / 3.3 / 4.0 / 4.7 px for t = 1 / 2 / 3 / 4. Wide strokes get a
looser tolerance because their end is physically more ambiguous.

The Hough angle step is *not* in the law, and the reason is a refutation of the draft.
``HoughLinesP`` quantises the orientation to ``dtheta`` and then *walks* the quantised line
from a voting pixel. The true angle differs from the nearest bin by up to dtheta/2, so the
walk drifts off a stroke laterally by ~s * dtheta/2 after s px, and a bare 1-px stroke is
left after tens of px; whatever remains is a separate piece that is often below the vote
threshold. The union of the fragments of a bare 1-px stroke (mutant ``plain_union``;
bare mask, threshold 10, minimum length 10, gap 5) therefore falls short of the real end
by a heavy-tailed amount: median 0 px, p95 3.0 / 4.5 / 7.1 px, p99 13 / 13 / 16 px,
max 29 / 29 / 30 px and 5.6 / 8.2 / 15 % of the strokes beyond the tolerance, for dtheta =
0.5 / 1 / 2 degrees; with textbook parameters (threshold 20, minimum length 30, gap 5,
1 degree) p95 22 px, p99 66 px, max 148 px, 25 of 1500 strokes missed
(``calibrate()["plain_hough"]``). No choice of threshold or gap removes the tail, so it is a
wrong answer, not a measurement tolerance, and a law ``a + b*t`` cannot cover it. The
reference removes it in two steps: the 3x3 dilation keeps the walk inside the stroke
(alone it leaves 3 of 9148 one-pixel strokes up to 8 px short over 8000 multi-stroke
scenes), and the end points are read off the stroke's own, unwidened pixels along the
fragments' direction, centred on their centroid. (Pivoting that line about one fragment end
instead of the centroid cost a 4.0 px miss on a short 3-px stroke in 1 of 6000 scenes: the
farthest fragment ends straddle the widened band, which tilts the direction.) With both
steps the reference's worst error does not depend on the angle step: 1.0-1.15 / 1.57-1.62 /
2.46-2.54 / 2.44-2.63 px for t = 1 / 2 / 3 / 4 at dtheta = 0.5 / 1 / 2 degrees
(``calibrate()["dtheta_reference"]``), and over 8000 multi-stroke scenes (36 000 strokes) it
is 1.41 / 2.00 / 2.24 / 2.24 px. The reference passed ``verify_manufactured`` on 20 000
scenes and ``verify_metamorphic`` on 3000 without a false reject.

Metric. The draft asked whether to match by end point distance or by Hausdorff distance.
For one collinear segment the two coincide to 1e-3 px on every gauge block of every
estimator (``calibrate()["hausdorff_minus_endpoint"]``), because both are decided by the
along-axis mismatch of the worse end. End point distance is used: it is simpler, it is the
quantity the task asks for, and it does not care which end is which.

Angle. It is not checked separately. Both ends within ``tol`` of the truth bound the angle
error by atan(2 * tol / L), so short segments get more angular slack automatically. The
tighter bound atan(tol / L) rejects 0-0.03 % of the panel's correct answers and
atan(2 * tol / L) none (``calibrate()["angle"]``): a separate angle test adds nothing.

Symmetries (``transforms``): shift, rot90 and mirror move the end points with the image
(covariant: expected = the same map applied to the base end points); end points within a
row and rows within the output are compared as sets, because the contract's canonical
order (smaller x first) is not preserved by rot90 or mirror. Dimming, at x0.45 and again at
x0.25, leaves the output unchanged (invariant); the second, stronger dimming is here because a
low fixed Canny threshold survives the first (see the comment above ``META_TOL``).
``verify_metamorphic`` never sees the scene, so the maps use the
fixed image size ``H`` x ``W`` and the fixed shift below. A segment's direction is an axis,
not a vector (mod 180 degrees): rot90 adds 90 degrees to it and a mirror negates it, which
the end point maps already encode. The order is checked on manufactured scenes only.

Metamorphic checks catch a hard-coded threshold or coordinate; their blind spot is an
answer that is constant or wrong in the same way in every variant (and one whose
dependence on the gain is weaker than ``META_TOL``), which is why they complement the
manufactured gate rather than replace it.

Not in the panel: ``cv2.createLineSegmentDetector`` (LSD). With its edge segments merged
per stroke it lands inside the tolerance for 98.6-99.0 % of the gauge blocks (4 x 6000); its
worst case keeps growing with the sample (4.0 / 3.16 / 5.0 / 4.47 px at 6000 blocks, 2.24 /
2.24 / 4.47 / 4.12 px at 1200), it misses 8-15 thin strokes per 6000, and on 600 multi-stroke
scenes it moves up to 8.5 px under the symmetries and changed the stroke count in 2-3 of
them. Fitting the law to a tail that does not converge would make the tolerance depend on
the sample size. Measured on OpenCV 5.0.0 (``tests`` pin it): ``createLineSegmentDetector().detect(gray)``
returns a 4-tuple ``(lines, width, prec, nfa)`` whose ``lines`` has shape (N, 4) and dtype float32
(11 rows for a 5-stroke scene; two-dimensional, no singleton axis), ``width`` and ``prec`` are (N,)
float64, ``nfa`` is None, and an image without a segment gives None in all four.
"""
from __future__ import annotations

import inspect
import math
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

SPEC = TaskSpec(
    family="hough_segments",
    task=("Find every straight line segment in the image: thin bright strokes on a dark background. Report "
          "the two end points of each stroke, one row per stroke."),
    inputs={"img": IOSpec("uint8", "HxWx3", 3, "BGR; bright straight strokes 1 to 4 px wide, each at least 40 px "
                                               "long, on a dark uniform background; strokes never touch or cross, "
                                               "stay at least 20 px apart and stay clear of the border")},
    outputs={"segments": IOSpec(
        "any", "Nx4", 2,
        "numeric (int or float) pixel coordinates x1, y1, x2, y2 of a stroke's two end points (x to the right, "
        "y down); one row per stroke, so several detections of one stroke must be merged into one; (x1, y1) is "
        "the end point with the smaller x (the smaller y if the x are equal); rows sorted by x1, then y1, "
        "ascending; end points accurate to a few px, the end of a wide stroke being ambiguous by about half "
        "its width")},
)

PHRASINGS = [
    SPEC.task,
    "Detect the straight line segments drawn as bright strokes on this dark image and return their end "
    "points as rows (x1, y1, x2, y2), with no duplicate or fragmentary detections of the same stroke.",
    "The image holds several bright straight strokes on a dark background. For each stroke give the pixel "
    "coordinates of its two ends as one row (x1, y1, x2, y2); a stroke must appear exactly once.",
    "Locate every bright line segment (none touch or cross) and output where each one begins and ends, as an "
    "N x 4 array of x1, y1, x2, y2 pixel coordinates.",
]

H, W = 360, 480  # fixed: the covariant checks of verify_metamorphic never see the scene
WIDTHS = (1, 4)  # stroke widths drawn, px
LMIN, LMAX = 40.0, 220.0  # centre-line length of a stroke, px
BORDER = 14  # px between an end point (plus its cap) and the border; also leaves room for SHIFT
CLEAR = 24.0  # px between the centre lines of two strokes (more than any usual maxLineGap)
N_RANGE = (3, 6)
AXIS_FRAC = 0.25  # share of strokes drawn exactly horizontal or vertical
SHIFT = (7, -5)  # px, well inside BORDER
GAIN = 0.45  # dim factor, shared with contour_area
GAIN_STRONG = 0.25  # stronger dim factor, this family only (why: the comment above META_TOL)

T_SHIFT = f"translate({SHIFT[0]:+d},{SHIFT[1]:+d})"
T_ROT, T_MIRROR, T_DIM, T_DIM_STRONG = "rot90", "mirror", f"dim x{GAIN}", f"dim x{GAIN_STRONG}"

# Calibrated by calibrate() on 4 x 6000 single-stroke gauge blocks (6000 per seed = 1500 per width, OpenCV
# 5.0.0, seeds 12345 / 999 / 2024 / 7). Worst end point error (px) by stroke width t = 1 / 2 / 3 / 4 (seed
# 12345; the panel worst is identical in all four seeds):
#   reference         1.00  2.00  2.24  2.24        fitLine      0.25  1.03  2.01  2.01
#   fitLine(dilated)  1.42  2.24  3.16  3.16        minAreaRect  0.50  1.12  2.06  2.06
#   panel worst       1.42  2.24  3.16  3.16        (the dilated pixel set: cap overshoot + 1 px per axis)
# TOL_B = 0.7 is the geometry (round cap x lateral band, in quadrature, see the docstring). TOL_A is the
# smallest value that puts the tolerance 25 % above the panel worst case at every width:
#   A = max(1.25 * worst - 0.7 * t) = 1.85 at t = 3, rounded up to 1.9
#   -> tolerance 2.6 / 3.3 / 4.0 / 4.7 px = 1.83 / 1.47 / 1.27 / 1.49 x the panel worst case, and 1.7-2.1x
#   the reference's own worst case over 8000 multi-stroke scenes (1.41 / 2.00 / 2.24 / 2.24 px).
TOL_A, TOL_B = 1.9, 0.7
# Metamorphic drift against the covariant prediction (measure_metamorphic_drift, 600 scenes, every stroke):
#   reference            shift / rot90 / mirror: mean 0.5-0.7, p99 2.0, max 2.83 px;  dimming: exactly 0.0
#   fitLine, minAreaRect all four symmetries: exactly 0.0 (they read the pixel set, which the maps preserve)
#   Otsu-driven Canny    shift / rot90 / mirror: max 3.16 / 4.0 / 4.0 px;  dimming: exactly 0.0 (tests/ builds it)
#   median-driven Canny, max-relative Canny, Otsu mask + 7x7 closing (200 scenes): dimming exactly 0.0,
#                        geometric max 3.0 / 4.0 / 4.0 px (the first two), 2.83 px (the third)
#   fixed Canny(100,200) dimming: 9 % of strokes move more than 1 px, one scene changed its stroke count
# Why a second, stronger dimming (GAIN_STRONG = 0.25; Gate 2 review, decision 4). GAIN = 0.45 does not make a
# low fixed Canny threshold move: Canny(30, 100) (the fixed_canny mutant with lower thresholds, now the mutant
# fixed_low_canny) passed all four gates before this transform, both on the pinned scenes and on 10 of 10 fresh
# gate seeds, and its dimming x0.45 drift was exactly 0.0 on 600 scenes (0 rejected; seeds 7_100_000-7_100_599).
# Calibration of dimming x0.25 on the same 600 scenes (2669 strokes, via sandbox.run_solution; the reference through
# measure_metamorphic_drift too), tolerance 1.0 px:
#   reference, Otsu-driven Canny, median-driven Canny, max-relative Canny, Otsu mask + 7x7 closing:
#                        mean = p99 = max = 0.0 px, 0 of 600 scenes rejected, no stroke count changed (also 0.0
#                        at x0.2 and x0.15); on 200 scenes drawn at the weakest the generator allows (1 px, gray 150
#                        on gray 50: contrast 100 -> 26 gray levels after the dimming) all five still find every
#                        stroke and stay at 0.0 px; margin 1.0 >= 1.25 x 0.0 holds, with room to spare
#   Canny(30, 100)       mean 0.19, p99 3.0, max inf (4 scenes changed the stroke count); 7 % of strokes move more
#                        than 1 px, 107 of 600 scenes (17.8 %) fail the check, and 123 of 200 fresh gates (n_meta = 5)
#                        reject it, all of them at x0.25 (before: 0 of 600 scenes, 0 of 10 gates)
#   Canny(100, 200)      at x0.25: 425 of 600 scenes fail the check (at x0.45: 116 of 600)
# What remains open: the stronger dimming is not a general detector of low fixed thresholds. Canny(20, 60) and
# Canny(10, 30) still pass at x0.25 (0 of 300 scenes rejected; Canny(20, 60) starts to fail at x0.2, 11 of 300),
# and Canny(30, 100) is rejected by 62 % of the gates (123 of 200), not by all: 17.8 % of the scenes fail, so
# a gate of 5 scenes rejects it with probability 1 - 0.822**5 = 62 %.
# A geometric symmetry re-samples the Hough accumulator / the Canny edge chains, so an end point can move by
# the estimator's own tail error: 5.0 = 1.25 x the larger of the two pipelines (4.0). Dimming does not touch
# the geometry, so every estimator that thresholds adaptively is exactly invariant: 1.0 px is one pixel of
# snapping, and what a hard-coded threshold does shows up as a larger move.
META_TOL = {T_SHIFT: 5.0, T_ROT: 5.0, T_MIRROR: 5.0, T_DIM: 1.0, T_DIM_STRONG: 1.0}


@dataclass(frozen=True)
class Segment:
    p1: tuple[int, int]  # centre-line end points, integer pixels, as passed to cv2.line
    p2: tuple[int, int]
    width: int

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])


@dataclass
class Scene:
    img: np.ndarray
    segments: list[Segment]
    background: int
    seed: int


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- scene generation

def _point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    s = float(np.clip(np.dot(p - a, v) / np.dot(v, v), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * v)))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segment_distance(s: Segment, t: Segment) -> float:
    """Minimum distance between two centre lines (0 when they cross or touch)."""
    a0, a1, b0, b1 = (np.array(p, float) for p in (s.p1, s.p2, t.p1, t.p2))
    d1, d2 = _cross(a0, a1, b0), _cross(a0, a1, b1)
    d3, d4 = _cross(b0, b1, a0), _cross(b0, b1, a1)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(_point_seg(a0, b0, b1), _point_seg(a1, b0, b1), _point_seg(b0, a0, a1), _point_seg(b1, a0, a1))


def _propose(rng: np.random.Generator) -> Segment | None:
    """Sample one stroke inside the image, or None if it would leave the border margin."""
    width = int(rng.integers(WIDTHS[0], WIDTHS[1] + 1))
    length = float(rng.uniform(LMIN, LMAX))
    axis = rng.random() < AXIS_FRAC
    theta = float(rng.choice([0.0, math.pi / 2])) if axis else float(rng.uniform(0.0, math.pi))
    m = BORDER + width
    p1 = (int(rng.integers(m, W - m)), int(rng.integers(m, H - m)))
    p2 = (int(round(p1[0] + length * math.cos(theta))), int(round(p1[1] + length * math.sin(theta))))
    if not (m <= p2[0] < W - m and m <= p2[1] < H - m):
        return None
    seg = Segment(p1, p2, width)
    return seg if seg.length >= LMIN else None


def make_scene(seed: int) -> Scene:
    """Manufacture a scene: strokes of known end points and width on a dark uniform background."""
    rng = np.random.default_rng(seed)
    bg = int(rng.integers(10, 51))
    img = np.full((H, W, 3), bg, np.uint8)
    segs: list[Segment] = []
    target = int(rng.integers(N_RANGE[0], N_RANGE[1] + 1))
    for _ in range(600):
        if len(segs) == target:
            break
        color = tuple(int(c) for c in rng.integers(150, 256, size=3))
        seg = _propose(rng)
        if seg is None or any(segment_distance(seg, s) < CLEAR for s in segs):
            continue
        cv2.line(img, seg.p1, seg.p2, color, seg.width)
        segs.append(seg)
    return Scene(img, segs, bg, seed)


def inputs(scene: Scene) -> dict:
    return {"img": scene.img}


# ---------------------------------------------------------------- verifiers

def tolerance(seg: Segment) -> float:
    """Allowed distance, px, between a manufactured end point and the reported one."""
    return TOL_A + TOL_B * seg.width


def _endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest end point distance of two segments, pairing the ends the way that makes it smallest."""
    straight = max(math.hypot(*(a[:2] - b[:2])), math.hypot(*(a[2:] - b[2:])))
    swapped = max(math.hypot(*(a[:2] - b[2:])), math.hypot(*(a[2:] - b[:2])))
    return min(straight, swapped)


def _unmatched(truth: np.ndarray, tols: np.ndarray, got: np.ndarray) -> tuple[int | None, np.ndarray]:
    """Bipartite matching (Kuhn): pair every truth segment with a distinct measured row inside its
    tolerance. Returns the index of a truth segment that cannot be paired (or None), and the distance matrix."""
    dist = np.array([[_endpoint_distance(t, g) for g in got] for t in truth]).reshape(len(truth), len(got))
    adj = [list(np.nonzero(dist[i] <= tols[i])[0]) for i in range(len(truth))]
    owner = [-1] * len(got)

    def assign(i: int, seen: set) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] == -1 or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(len(truth)):
        if not assign(i, set()):
            return i, dist
    return None, dist


def _rows(out: dict) -> tuple[np.ndarray | None, str]:
    """The segments as a float (N, 4) array, or (None, reason) when the contract is violated."""
    problems = check_outputs(SPEC, out)
    if problems:
        return None, "contract: " + "; ".join(problems)
    a = np.asarray(out["segments"])
    if a.dtype.kind not in "iuf":
        return None, f"contract: 'segments' should be numeric, got {a.dtype}"
    if a.shape[1] != 4:
        return None, f"contract: 'segments' should have shape Nx4, got {a.shape}"
    a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        return None, "contract: 'segments' has NaN or infinite coordinates"
    return a, ""


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    got, why = _rows(out)
    if got is None:
        return Verdict(False, why)
    n = len(scene.segments)
    if len(got) != n:
        hint = "duplicate or fragmentary detections must be merged" if len(got) > n else "strokes were missed"
        return Verdict(False, f"{len(got)} rows for {n} strokes ({hint})")
    truth = np.array([[*s.p1, *s.p2] for s in scene.segments], dtype=np.float64)
    tols = np.array([tolerance(s) for s in scene.segments])
    bad, dist = _unmatched(truth, tols, got)
    if bad is not None:
        s = scene.segments[bad]
        closest = float(dist[bad].min()) if dist.size else float("nan")
        return Verdict(False, f"stroke {s.p1}->{s.p2} (width {s.width}) has no row within {tols[bad]:.1f} px "
                              f"(closest row is {closest:.1f} px off)")
    first, second = got[:, :2], got[:, 2:]
    for k in range(n):
        if tuple(first[k]) > tuple(second[k]):
            return Verdict(False, f"row {k} {got[k].tolist()}: (x1, y1) must be the end point with the smaller x")
    keys = [tuple(first[k]) for k in range(n)]
    if keys != sorted(keys):
        return Verdict(False, "rows are not sorted by x1, then y1")
    return Verdict(True)


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task: end points are covariant under shift, rot90 and mirror; invariant under dimming."""
    img, bg = scene.img, scene.background
    dx, dy = SHIFT
    shifted = np.full_like(img, bg)
    shifted[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return {
        T_SHIFT: {"img": shifted},
        T_ROT: {"img": np.ascontiguousarray(np.rot90(img))},
        T_MIRROR: {"img": np.ascontiguousarray(img[:, ::-1])},
        T_DIM: {"img": np.round(img.astype(np.float32) * GAIN).astype(np.uint8)},
        T_DIM_STRONG: {"img": np.round(img.astype(np.float32) * GAIN_STRONG).astype(np.uint8)},
    }


def expected_under(name: str, seg: np.ndarray) -> np.ndarray | None:
    """Where the end points ``seg`` (N, 4) must land under symmetry ``name`` (None if unknown)."""
    x1, y1, x2, y2 = seg.T
    if name == T_SHIFT:
        return np.c_[x1 + SHIFT[0], y1 + SHIFT[1], x2 + SHIFT[0], y2 + SHIFT[1]]
    if name == T_ROT:  # np.rot90 turns the image counter-clockwise: pixel (x, y) -> (y, W - 1 - x)
        return np.c_[y1, W - 1 - x1, y2, W - 1 - x2]
    if name == T_MIRROR:
        return np.c_[W - 1 - x1, y1, W - 1 - x2, y2]
    if name in (T_DIM, T_DIM_STRONG):
        return seg
    return None


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    a0, why = _rows(base)
    if a0 is None:
        return Verdict(False, why)
    for name, out in variants.items():
        a, why = _rows(out)
        if a is None:
            return Verdict(False, f"{name}: {why}")
        want = expected_under(name, a0)
        if want is None:
            return Verdict(False, f"{name}: unknown symmetry")
        if len(a) != len(want):
            return Verdict(False, f"{name}: {len(a)} segments, but the original image gave {len(want)}")
        tol = META_TOL[name]
        bad, dist = _unmatched(want, np.full(len(want), tol), a)
        if bad is not None:
            closest = float(dist[bad].min()) if dist.size else float("nan")
            return Verdict(False, f"{name}: segment {want[bad].round(1).tolist()} expected within {tol} px "
                                  f"(closest {closest:.1f} px off)")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # HoughLinesP walks a quantised line and leaves a bare 1-px stroke after ~60-120 px: give it a wider one.
    wide = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(wide)
    lines = cv2.HoughLinesP(wide, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=5)
    if lines is None:
        return {"segments": np.zeros((0, 4), np.int32)}
    segs = lines.reshape(-1, 4)  # (N, 1, 4) on OpenCV 4.x, (N, 4) on 5.0
    stroke = labels[segs[:, 1], segs[:, 0]]  # HoughLinesP returns fragments: group them by stroke
    rows = []
    for k in np.unique(stroke):
        pts = segs[stroke == k].reshape(-1, 2).astype(float)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        u = (pts[j] - pts[i]) / d[i, j]  # the fragments' farthest ends give the stroke's direction
        xy = np.argwhere((labels == k) & (mask > 0))[:, ::-1]  # the stroke's own pixels (x, y), not the widened ones
        c = xy.mean(axis=0)  # their centre lies on the centre line
        s = (xy - c) @ u
        a, b = sorted([tuple(int(round(t)) for t in c + s.min() * u), tuple(int(round(t)) for t in c + s.max() * u)])
        rows.append(a + b)
    return {"segments": np.array(sorted(rows), dtype=np.int32).reshape(-1, 4)}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_THRESH = "_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
_WIDEN = REFERENCE[REFERENCE.index("    # HoughLinesP walks"):REFERENCE.index("    _, labels")]
_RESHAPE = "    segs = lines.reshape(-1, 4)  # (N, 1, 4) on OpenCV 4.x, (N, 4) on 5.0\n"
_GROUP = REFERENCE[REFERENCE.index("    stroke = labels"):REFERENCE.index('    return {"segments": np.array(sorted(rows)')]
_REFINE = REFERENCE[REFERENCE.index("        pts = segs"):REFERENCE.index("        rows.append(a + b)")]
_ENDS = "a, b = sorted([tuple(int(round(t)) for t in c + s.min() * u), tuple(int(round(t)) for t in c + s.max() * u)])"
_FARTHEST = ("        pts = segs[stroke == k].reshape(-1, 2)\n"
             "        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)\n"
             "        i, j = np.unravel_index(np.argmax(d), d.shape)  # the farthest pair of fragment ends\n"
             "        a, b = sorted([tuple(int(t) for t in pts[i]), tuple(int(t) for t in pts[j])])\n")

_HOUGHLINES = '''import cv2
import numpy as np


def solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # (rho, theta) of infinite lines
    rows = []
    if lines is not None:
        for rho, theta in lines.reshape(-1, 2):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            p = (int(x0 - 1000 * b), int(y0 + 1000 * a))
            q = (int(x0 + 1000 * b), int(y0 - 1000 * a))
            rows.append(sorted([p, q])[0] + sorted([p, q])[1])
    return {"segments": np.array(sorted(rows), dtype=np.int32).reshape(-1, 4)}
'''

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "unpack_line0": (
        "The OpenCV 4.x loop `x1, y1, x2, y2 = line[0]`. HoughLinesP rows are (4,) in 5.0, so line[0] is a "
        "scalar and unpacking raises TypeError.",
        _mutate((_RESHAPE, "    segs = []\n    for line in lines:\n        x1, y1, x2, y2 = line[0]\n"
                           "        segs.append((x1, y1, x2, y2))\n    segs = np.array(segs)\n"))),
    "n1x4_layout": (
        "Returns the 4.x (N, 1, 4) layout, which the contract (N, 4) rules out. Runs fine, wrong shape.",
        _mutate(("dtype=np.int32).reshape(-1, 4)}", "dtype=np.int32).reshape(-1, 1, 4)}"))),
    "houghlines_infinite": (
        "Classic tutorial: HoughLines gives (rho, theta) of infinite lines, drawn out to +-1000 px. "
        "Those are not the segments' end points.",
        _HOUGHLINES),
    "no_merge": (
        "Returns every HoughLinesP fragment: a stroke shows up as several overlapping rows.",
        _mutate((_GROUP, "    rows = []\n    for x1, y1, x2, y2 in segs:\n"
                         "        a, b = sorted([(int(x1), int(y1)), (int(x2), int(y2))])\n"
                         "        rows.append(a + b)\n"))),
    "plain_union": (
        "Textbook merge: Hough on the bare mask, end points of the farthest fragments. A 1-px stroke is left by "
        "the quantised walk after ~60-120 px, so the union falls short of the real end.",
        _mutate((_WIDEN, "    wide = mask\n"), (_REFINE, _FARTHEST))),
    "fixed_canny": (
        "Mask from Canny(100, 200) closed with a 7x7 kernel instead of Otsu: absolute edge thresholds.",
        _mutate((_THRESH, "mask = cv2.morphologyEx(cv2.Canny(gray, 100, 200), cv2.MORPH_CLOSE, "
                          "np.ones((7, 7), np.uint8))"))),
    "fixed_low_canny": (
        "As fixed_canny but with Canny(30, 100): low absolute thresholds that survive the x0.45 dimming and only "
        "give way under the x0.25 one.",
        _mutate((_THRESH, "mask = cv2.morphologyEx(cv2.Canny(gray, 30, 100), cv2.MORPH_CLOSE, "
                          "np.ones((7, 7), np.uint8))"))),
    "fixed_threshold_127": (
        "Hard-coded binary threshold: right for these scenes, wrong for a dimmer image.",
        _mutate((_THRESH, "_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)"))),
    "unordered": (
        "Forgets the contract's canonical order: end points as found, rows in stroke-label order.",
        _mutate((_ENDS, "a, b = (tuple(int(round(t)) for t in c + s.min() * u), "
                        "tuple(int(round(t)) for t in c + s.max() * u))"),
                ("np.array(sorted(rows), dtype=np.int32)", "np.array(rows, dtype=np.int32)"))),
}


# ---------------------------------------------------------------- calibration

def _gauge_block(rng: np.random.Generator, width: int) -> tuple[np.ndarray, Segment]:
    """One stroke, white on black, on a canvas just big enough for it."""
    length = float(rng.uniform(LMIN, LMAX))
    theta = float(rng.choice([0.0, math.pi / 2])) if rng.random() < AXIS_FRAC else float(rng.uniform(0.0, math.pi))
    dx, dy = int(round(length * math.cos(theta))), int(round(length * math.sin(theta)))
    pad = BORDER + width
    p1 = (pad + max(-dx, 0), pad + max(-dy, 0))
    p2 = (p1[0] + dx, p1[1] + dy)
    img = np.zeros((abs(dy) + 2 * pad + 1, abs(dx) + 2 * pad + 1, 3), np.uint8)
    cv2.line(img, p1, p2, (255, 255, 255), width)
    return img, Segment(p1, p2, width)


def _otsu_mask(img: np.ndarray) -> np.ndarray:
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _farthest_pair(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i], pts[j]


def _est_fitline(img: np.ndarray, widen: bool = False) -> np.ndarray:
    """Geometry only, no Hough: fit a line to each stroke's pixels and slide to the extreme pixels.
    ``widen`` reads the pixels of the 3x3-dilated mask instead (mechanism 4 of the docstring, without Hough)."""
    mask = _otsu_mask(img)
    n, labels = cv2.connectedComponents(cv2.dilate(mask, np.ones((3, 3), np.uint8)) if widen else mask)
    rows = []
    for k in range(1, n):
        xy = np.argwhere(labels == k)[:, ::-1].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        u, c = np.array([vx, vy]), np.array([x0, y0])
        s = (xy - c) @ u
        rows.append([*(c + s.min() * u), *(c + s.max() * u)])
    return np.array(rows).reshape(-1, 4)


def _est_minarearect(img: np.ndarray) -> np.ndarray:
    """No Hough either: the mid-points of the short sides of each stroke's minimum-area rectangle."""
    rows = []
    for c in cv2.findContours(_otsu_mask(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]:
        box = cv2.boxPoints(cv2.minAreaRect(c))
        if np.linalg.norm(box[0] - box[1]) <= np.linalg.norm(box[1] - box[2]):
            rows.append([*(box[0] + box[1]) / 2, *(box[2] + box[3]) / 2])
        else:
            rows.append([*(box[1] + box[2]) / 2, *(box[3] + box[0]) / 2])
    return np.array(rows).reshape(-1, 4)


def _est_lsd(img: np.ndarray) -> np.ndarray:
    """cv2.createLineSegmentDetector, its edge segments merged per stroke (farthest pair of ends)."""
    n, labels = cv2.connectedComponents(cv2.dilate(_otsu_mask(img), np.ones((3, 3), np.uint8)))
    found = cv2.createLineSegmentDetector().detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]
    if found is None:
        return np.zeros((0, 4))
    segs = np.round(found.reshape(-1, 4)).astype(int)
    segs[:, [0, 2]] = segs[:, [0, 2]].clip(0, img.shape[1] - 1)
    segs[:, [1, 3]] = segs[:, [1, 3]].clip(0, img.shape[0] - 1)
    stroke = labels[segs[:, 1], segs[:, 0]]
    rows = []
    for k in np.unique(stroke[stroke > 0]):
        a, b = _farthest_pair(segs[stroke == k].reshape(-1, 2).astype(float))
        rows.append([*a, *b])
    return np.array(rows).reshape(-1, 4)


def _hough_union(img: np.ndarray, dtheta_deg: float = 1.0, widen: bool = True, threshold: int = 10,
                 min_len: int = 10, max_gap: int = 5, refine: bool = True) -> np.ndarray:
    """The reference pipeline with its knobs exposed (``test_hough_union_is_the_reference`` pins the defaults)."""
    mask = _otsu_mask(img)
    wide = cv2.dilate(mask, np.ones((3, 3), np.uint8)) if widen else mask
    _, labels = cv2.connectedComponents(wide)
    lines = cv2.HoughLinesP(wide, 1, math.radians(dtheta_deg), threshold, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return np.zeros((0, 4))
    segs = lines.reshape(-1, 4)
    stroke = labels[segs[:, 1], segs[:, 0]]
    rows = []
    for k in np.unique(stroke):
        a, b = _farthest_pair(segs[stroke == k].reshape(-1, 2).astype(float))  # the textbook merge
        if refine:  # direction from those ends, position and extent from the stroke's own pixels
            u = (b - a) / np.linalg.norm(b - a)
            xy = np.argwhere((labels == k) & (mask > 0))[:, ::-1]
            c = xy.mean(axis=0)
            s = (xy - c) @ u
            a, b = c + s.min() * u, c + s.max() * u
        rows.append([*a, *b])
    return np.array(rows).reshape(-1, 4)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Hausdorff distance of two segments (a maximum of convex functions sits at the end points)."""
    pa, pb = (a[:2], a[2:]), (b[:2], b[2:])
    return max(max(_point_seg(p, *pb) for p in pa), max(_point_seg(p, *pa) for p in pb))


def _axis_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between the two segments' axes, radians in [0, pi/2]."""
    ta = math.atan2(a[3] - a[1], a[2] - a[0])
    tb = math.atan2(b[3] - b[1], b[2] - b[0])
    return abs((ta - tb + math.pi / 2) % math.pi - math.pi / 2)


# The estimators the law is fitted to: errors are a deterministic function of the stroke geometry.
PANEL = {"reference": lambda img: reference_solve(img)["segments"].astype(float),
         "fitLine": _est_fitline, "fitLine(dilated)": lambda img: _est_fitline(img, widen=True),
         "minAreaRect": _est_minarearect}
# Reported, not fitted: LSD's worst case keeps growing with the sample (region growing luck, like the Hough tail).
INFORMATIONAL = {"LSD": _est_lsd}


def _summ(v: list[float]) -> dict:
    a = np.asarray(v, dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "p50": round(float(np.median(a)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}


def calibrate(n: int = 2400, seed: int = 12345) -> dict:
    """Measure end point errors on single-stroke gauge blocks, ``n`` in total, widths in rotation.

    * ``panel``: per estimator and width, the end point error (px) of four defensible estimators: the
      reference (HoughLinesP), fitLine on the stroke's pixels, fitLine on the 3x3-dilated pixels, and
      minAreaRect (no Hough). TOL_A / TOL_B are fitted to their worst case; ``worst_ratio`` is that worst
      case over the tolerance (< 1 / margin, i.e. < 0.8).
    * ``informational``: the same for LSD, plus the share of its answers inside the tolerance.
    * ``hausdorff_minus_endpoint``: largest |Hausdorff - end point distance| over the panel.
    * ``angle``: how often the panel breaks an angle bound atan(k * tol / L) for k = 1 and k = 2.
    * ``plain_hough``: the reference without the 3x3 dilation and the pixel refinement (mutant
      ``plain_union``), on 1-px strokes, by angle step.
    * ``dtheta_reference``: the reference's worst error per width for three angle steps.
    """
    rng = np.random.default_rng(seed)
    widths = list(range(WIDTHS[0], WIDTHS[1] + 1))
    blocks = [(w, *_gauge_block(rng, w)) for w in (widths[i % len(widths)] for i in range(n))]
    err = {name: {w: [] for w in widths} for name in {**PANEL, **INFORMATIONAL}}
    misses = {name: 0 for name in err}
    inside = {name: 0 for name in INFORMATIONAL}
    gap, rej = 0.0, {1: 0, 2: 0}
    worst_ratio, total = 0.0, 0
    for w, img, seg in blocks:
        truth = np.array([*seg.p1, *seg.p2], dtype=float)
        tol = tolerance(seg)
        for name, est in {**PANEL, **INFORMATIONAL}.items():
            got = est(img)
            if len(got) != 1:
                misses[name] += 1
                continue
            m = _endpoint_distance(truth, got[0])
            err[name][w].append(m)
            if name in INFORMATIONAL:
                inside[name] += m <= tol
                continue
            gap = max(gap, abs(_hausdorff(truth, got[0]) - m))
            worst_ratio = max(worst_ratio, m / tol)
            ang = _axis_error(truth, got[0])
            for k in rej:
                rej[k] += ang > math.atan(k * tol / seg.length)
            total += 1
    summary = {name: {w: _summ(v) for w, v in by_w.items()} for name, by_w in err.items()}
    out: dict = {"n": n, "panel": {k: summary[k] for k in PANEL}, "misses": misses,
                 "informational": {k: {"by_width": summary[k], "inside_tol": round(inside[k] / n, 4)}
                                   for k in INFORMATIONAL}}
    out["panel_worst"] = {w: round(max(max(err[name][w], default=0.0) for name in PANEL), 2) for w in widths}
    out["worst_ratio"] = round(worst_ratio, 3)
    out["hausdorff_minus_endpoint"] = round(gap, 4)
    out["angle"] = {f"reject_atan({k}*tol/L)": round(rej[k] / total, 4) for k in rej}
    thin = [(img, seg) for w, img, seg in blocks if w == 1]
    out["plain_hough"] = {}
    for label, kw in (("0.5deg", dict(dtheta_deg=0.5)), ("1deg", dict(dtheta_deg=1.0)), ("2deg", dict(dtheta_deg=2.0)),
                      ("1deg textbook thr20/len30/gap5", dict(threshold=20, min_len=30, max_gap=5))):
        e = [_endpoint_distance(np.array([*s.p1, *s.p2], float), got[0])
             for img, s in thin if len(got := _hough_union(img, widen=False, refine=False, **kw)) == 1]
        out["plain_hough"][label] = {**_summ(e), "misses": len(thin) - len(e),
                                     "frac_over_tol": round(float(np.mean(np.array(e) > tolerance(thin[0][1]))), 3)}
    out["dtheta_reference"] = {}
    for dth in (0.5, 1.0, 2.0):
        by_w = {w: [] for w in widths}
        for w, img, seg in blocks:
            got = _hough_union(img, dth)
            if len(got) == 1:
                by_w[w].append(_endpoint_distance(np.array([*seg.p1, *seg.p2], float), got[0]))
        out["dtheta_reference"][dth] = {w: round(max(v), 2) for w, v in by_w.items()}
    return out


def measure_metamorphic_drift(n: int = 300, seed: int = 50_000, solve=reference_solve) -> dict:
    """How far ``solve``'s end points sit from the covariant prediction, per symmetry, over all strokes of
    ``n`` scenes: {symmetry: {"max": px, "p99": px, "mean": px}}. ``inf`` if a stroke count changed."""
    moved: dict[str, list[float]] = {name: [] for name in META_TOL}
    for i in range(n):
        scene = make_scene(seed + i)
        base = solve(scene.img)["segments"].astype(float)
        for name, variant in transforms(scene).items():
            got = solve(variant["img"])["segments"].astype(float)
            want = expected_under(name, base)
            if len(got) != len(want):
                moved[name].append(float("inf"))
                continue
            _, dist = _unmatched(want, np.full(len(want), np.inf), got)
            moved[name].extend(dist.min(axis=1).tolist() if dist.size else [])
    return {name: {"max": round(max(v), 2), "p99": round(float(np.percentile(v, 99)), 2),
                   "mean": round(float(np.mean(v)), 3)} for name, v in moved.items()}


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=1))
    print("metamorphic drift of the reference:", measure_metamorphic_drift())
