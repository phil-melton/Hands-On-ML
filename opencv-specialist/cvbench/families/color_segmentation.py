"""Family ``color_segmentation``: pixel counts of named HSV colour classes, verified exactly.

Same Method of Manufactured Solutions as ``contour_area``, but the measurand is different
and so is the tolerance. Choose the answer first, then render the image from it:

* A scene is a *label map* (one integer id per shape, 0 = background) painted with the
  OpenCV drawing primitives. The image is a lookup table over that map,
  ``img = LUT[labels]``, so every pixel is one flat colour. The manufactured answer is
  ``truth[c] = number of label pixels belonging to shapes of colour class c``.

Tolerance law
-------------
    |count_c - truth_c| <= 0 px      (and the metamorphic invariants are exact too)

The contour_area law (error ~ c*P, living on the boundary) does not apply, because this
task does not measure a geometric area. It counts pixels, and a pixel count is an integer
with an exact manufactured value as long as three sources of error are absent by
construction (Gate 2 decision: integer-valued answers get zero tolerance):

1. Boundary blending. There is none: shapes are drawn with the default ``LINE_8`` (no
   anti-aliasing), no resizing, no JPEG, no noise. Evidence that this matters, from
   ``calibrate()['antialias']``: a ``LINE_AA`` disk adds 0.48..0.90 x perimeter extra
   "red" pixels (the blended fringe keeps the hue and often clears S and V), i.e.
   exactness is impossible once the edge is blended, and the error then scales with the
   perimeter P exactly like contour_area's.
2. Conversion rounding. ``cvtColor`` 8-bit hue is fixed-point: it differs from the rounded
   float hue for 1.69 % of all colours (always within 0.14 hue units of a rounding edge,
   max |error| 0.64) and flips the *class* of 0.055 % of random colours
   (``conversion_evidence()``, 2e6 uniform BGR triples, OpenCV 5.0.0). So the generator only
   paints colours whose float HSV (``_hsv_f``, the textbook formula, independent of cv2) is at
   least 1 hue unit, 4 S units and 10 V units inside the class, or sits at an *exact* integer
   hue / saturation, which is 0.5 units from the rounding edge. The decision is then identical
   for cv2's fixed point and for the float formula;
   ``test_scene_colours_classify_identically_everywhere`` pins it.
3. Overlap. Shapes live in disjoint padded boxes, so no later shape overwrites an earlier one.

What the contract fixes (the model must read it from the prompt, nothing is "common
knowledge"): BGR input, ``cv2.COLOR_BGR2HSV`` 8-bit scale (H 0-179, S and V 0-255), every
bound inclusive, S >= S_MIN and V >= V_MIN for every colour, hue ranges in ``HUE`` (red
wraps: 0-10 and 170-179).

What a scene contains
---------------------
* 4-7 *colour shapes*: random colour of a class, drawn from the interior of its HSV box.
  Red shapes alternate between the two lobes (0-8 and 172-179).
* 4-6 *distractors* that count for nothing: greys including black and white (S = 0, hue 0 =
  "red" if S is ignored), pastels (right hue, S below S_MIN), dark colours (right hue and S, V
  below V_MIN), gap hues (orange, cyan, violet: full S and V, hue between the ranges). The
  background is black, white, a grey or a tinted pastel wall (never a counted colour).
* *edge swatches*, a test chart for the contract's bounds, in every scene: each of the 10
  inclusive hue bounds (0, 10, 170, 179, 20, 35, 40, 80, 100, 130) and each of the 8 hues one
  step outside a range (11, 169, 19, 36, 39, 81, 99, 131), S = 100 and S = 99, V = 160 (which the
  dim symmetry lands exactly on V = 80) and V = 79: 22 swatches. Every bound is therefore
  resolved to +-1 in (nearly) every scene: an off-by-one bound or an exclusive comparison
  cannot hide. ``test_every_unit_perturbation_of_a_bound_is_caught`` pins that resolution.
* in 25 % of the scenes one colour is *absent altogether* (no shape, no in-range swatch):
  its answer is the integer 0, and any pixel another colour leaks into it is a failure.
* shapes of 8 kinds: rect, disk, ellipse, rotated rect, triangle, ring (hole = background, not
  counted), speckle (2-6 px) and 1-px line. The measurand is the *pixel count*, not a contour
  polygon area, not holes-filled, with no minimum blob size.

Metamorphic checks
------------------
Each symmetry says how the four counts must move when the image is transformed:

* translate, rot90, mirror: invariant (they permute pixels, so the pixel set of each class
  is carried to the pixel set of the same class).
* dim (every channel halved, ``img >> 1``): invariant *for this scene class*, not in general.
  Halving an even-valued colour keeps hue and saturation exactly (the ratios do not change)
  and halves V, so the class decision is unchanged iff no in-class pixel has V in
  [V_MIN, 2*V_MIN): there it would flip. The generator guarantees that (colour shapes have
  V >= 172, the V edge swatch exactly 2*V_MIN; every in-class colour is even so the halving
  is exact; dark distractors have V <= 66 and stay out; greys, pastels and gap hues stay out
  because H and S are unchanged). So dim is a valid symmetry of the scenes ``make_scene``
  produces, and ``test_dim_is_not_a_symmetry_of_the_contract`` pins that it is not one of the
  contract. What it buys: it probes V in [V_MIN, 127], a band that base scenes cannot contain
  without breaking the symmetry itself, so a copy-pasted ``V >= 100`` (or a strict ``V > 80``,
  via the V = 160 swatch) is caught here and nowhere else.

``verify_metamorphic`` is called with the outputs only (no scene), so every relation must
be expressible on the four counts alone. Blind spot, as always: a constant output (all
zeros, say) is invariant under everything; the manufactured gate is what catches it.
"""
from __future__ import annotations

import inspect
import math
import textwrap
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

# ---------------------------------------------------------------- the contract

COLOURS = ("red", "yellow", "green", "blue")
HUE = {  # inclusive ranges on OpenCV's 8-bit hue scale; red wraps around 0/180
    "red": ((0, 10), (170, 179)),
    "yellow": ((20, 35),),
    "green": ((40, 80),),
    "blue": ((100, 130),),
}
S_MIN, V_MIN = 100, 80  # inclusive, 8-bit scale, shared by all four colours
GAP_HUES = {"orange": (11, 19), "lime": (36, 39), "cyan": (81, 99), "violet": (131, 169)}  # in no range


def _ranges_text() -> str:
    return ", ".join(f"{name} " + " or ".join(f"{lo}-{hi}" for lo, hi in spans) for name, spans in HUE.items())


_RANGES = _ranges_text()
_BOUNDS = f"S >= {S_MIN} and V >= {V_MIN}"

SPEC = TaskSpec(
    family="color_segmentation",
    task=("Segment the image by colour in HSV space and report how many pixels belong to each of four named "
          "colours. Convert with cv2.COLOR_BGR2HSV (8-bit scale: H 0-179, S 0-255, V 0-255). A pixel has a "
          f"colour when {_BOUNDS} and its hue lies in that colour's range, every bound inclusive: {_RANGES} "
          "(red wraps around hue 0). Count every pixel of the image exactly: no smoothing, resizing or minimum "
          "blob size. Pixels in no range (greys, pastels, dark pixels, other hues) count for nothing."),
    inputs={"img": IOSpec("uint8", "HxWx3", 3, "BGR; flat-coloured shapes (no anti-aliasing) on a plain "
                                               "background, plus distractor shapes that match no colour")},
    outputs={c: IOSpec("int", "scalar", None, f"number of {c} pixels in the whole image (0 if none)")
             for c in COLOURS},
)

PHRASINGS = [
    SPEC.task,
    f"Count how many pixels of each of red, yellow, green and blue this BGR image contains. Use HSV in OpenCV's "
    f"8-bit scale (hue 0-179, saturation and value 0-255): a pixel is of a colour when {_BOUNDS} and its hue is "
    f"inside the inclusive range {_RANGES}. Red needs two ranges because it wraps. Give the four pixel totals; "
    "grey, pastel, dark and other-hue pixels belong to no colour, and nothing is filtered or resized.",
    f"Threshold the image in HSV (convert from BGR with cv2.COLOR_BGR2HSV, hue 0-179) and return the number of "
    f"pixels in the mask of each colour. All bounds are inclusive: {_BOUNDS}, plus hue {_RANGES}. Red wraps "
    "around 0/180, so one range is not enough. Every pixel counts, thin lines and tiny specks included.",
    f"For each of four colour classes, report the number of pixels of this image that satisfy its HSV "
    f"definition (8-bit HSV of the BGR image, hue 0-179; all comparisons inclusive). Every class requires "
    f"{_BOUNDS}; the hue ranges are {_RANGES}. Return the counts for red, yellow, green and blue, with 0 for a "
    "colour that is absent. Do not blur, erode or drop small regions.",
]

KINDS = ("rect", "disk", "ellipse", "rotrect", "triangle", "ring", "speckle", "line")
EDGE_KINDS = ("rect", "disk")
DISTRACTORS = ("gray", "pastel", "dark", "gap")
SIZE = (360, 480)
BORDER, GAP = 14, 5  # px kept clear at the image border and between shapes
OMIT_P = 0.25  # share of scenes in which one colour class is absent altogether (its answer is the integer 0)
MARGIN = {"hue": 1.0, "sat": 4.0, "val": 10.0}  # random colour shapes sit this far inside their class (cv2 units);
# the measured cv2-vs-float disagreement is <= 0.14 units of rounding edge, so every margin is several times larger.

# Edge swatches: exact inclusive hue bounds (in) and the first hue outside each range (out).
IN_EDGES = [(0, "red"), (10, "red"), (170, "red"), (179, "red"), (20, "yellow"), (35, "yellow"),
            (40, "green"), (80, "green"), (100, "blue"), (130, "blue")]
OUT_EDGES = [11, 19, 36, 39, 81, 99, 131, 169]
EDGE_S, EDGE_V = 120, 254  # S of an edge swatch (diff 120 over max 254 -> 120.47 -> 120), V = max channel

# Zero tolerance for an integer-valued answer (see "Tolerance law"). Measured by calibrate() on 1200 gauge
# blocks (single shapes: 8 kinds x 4 colours x both red lobes, OpenCV 5.0.0): the reference's error in every
# colour, leaks into the other three colours included, is exactly 0 px in all 1200 (worst |error| = 0 px,
# 0 blocks with any error). Margin: none to add; any non-zero error is a bug in the generator or the reference.
# Whole scenes (make_scene, all 8 kinds, distractors, 22 edge swatches), the reference on 2000 fresh scenes,
# seeds 7_300_000..7_301_999, called in-process (no sandbox): reference_solve(**inputs(scene)) then
# verify_manufactured at TOL_PX = 0 gave 0 failures and a largest |count error| of 0 px over all 8000 counts
# (501 of the 2000 scenes omit one colour class). An earlier report of "2,000 scenes, 0 failures" had no
# recorded method or seeds; this is the re-measurement. The calibration above is the 1200 gauge blocks.
TOL_PX = 0
# measure_metamorphic_drift(): the reference's four counts moved by exactly 0 px under every transform
# (translate, rot90, mirror, dim) in 300 scenes: translate / rot90 / mirror permute pixels, dim is exact on
# this scene class (see "Metamorphic checks"). The same 2000 fresh scenes (seeds above) through transforms()
# and verify_metamorphic: 0 failures, largest count drift 0 px.
META_TOL_PX = 0


@dataclass(frozen=True)
class Shape:
    kind: str
    role: str  # "body" (random colour of a class), "edge" (exact bound swatch) or "distractor"
    cls: str | None  # the colour class whose count this shape feeds; None = counts for nothing
    tag: str
    bgr: tuple[int, int, int]
    area: int  # pixels carrying this shape's label (exact: the image is rendered from the labels)
    analytic: float | None  # closed-form area where one exists (rect, ring, speckle: exact; disk, ellipse: pi r^2)
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 reserved on the canvas


@dataclass
class Scene:
    img: np.ndarray
    labels: np.ndarray  # uint8 id map: 0 background, k = shapes[k-1]
    shapes: list[Shape]
    background: tuple[int, int, int]
    truth: dict[str, int]
    seed: int
    omitted: str | None = None  # a colour class deliberately absent from this scene (its count must be 0)


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- colours

def classify(h: float, s: float, v: float) -> str | None:
    """The contract, as code, for an 8-bit (H, S, V) triple. Used by the generator's own checks only."""
    if s < S_MIN or v < V_MIN:
        return None
    for name, spans in HUE.items():
        if any(lo <= h <= hi for lo, hi in spans):
            return name
    return None


def classify_array(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``classify`` for whole arrays: 0 = no colour, i = COLOURS[i - 1]. The contract again, vectorised."""
    c = np.zeros(h.shape, np.int8)
    ok = (s >= S_MIN) & (v >= V_MIN)
    for i, name in enumerate(COLOURS, 1):
        c[ok & np.any([(h >= lo) & (h <= hi) for lo, hi in HUE[name]], axis=0)] = i
    return c


def _hsv_cv(bgr) -> tuple[int, int, int]:
    return tuple(int(x) for x in cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0])


def _hsv_f(bgr) -> tuple[float, float, float]:
    """Float HSV on OpenCV's 8-bit scale, from the textbook formula (independent of cv2's fixed point)."""
    b, g, r = (float(c) for c in bgr)
    mx, mn = max(b, g, r), min(b, g, r)
    d = mx - mn
    s = 255.0 * d / mx if mx > 0 else 0.0
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 30.0 * ((g - b) / d)
    elif mx == g:
        h = 30.0 * (2.0 + (b - r) / d)
    else:
        h = 30.0 * (4.0 + (r - g) / d)
    return h % 180.0, s, mx


def _even_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    """BGR from HSV with every channel even, so that ``img >> 1`` halves the colour exactly."""
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) & 0xFE for c in bgr)


def _hue_bgr(h: float, hi: int, lo: int) -> tuple[int, int, int]:
    """BGR with max channel hi, min channel lo and the mid channel (rounded to even) that puts the hue near h."""
    k, j = divmod(2.0 * h, 60.0)  # sextant, and degrees into it
    k = int(k)
    step = 2 * int(round(j / 60.0 * (hi - lo) / 2.0))
    return tuple([(lo, lo + step, hi), (lo, hi, hi - step), (lo + step, hi, lo),
                  (hi, hi - step, lo), (hi, lo, lo + step), (hi - step, lo, hi)][k])


def _edge_bgr(h: int) -> tuple[int, int, int]:
    """Colour whose float hue is exactly the integer h: max 254, min 134, so the mid channel steps 4 per hue unit."""
    bgr = _hue_bgr(h, EDGE_V, EDGE_V - 120)
    assert _hsv_cv(bgr) == (h, EDGE_S, EDGE_V), (h, bgr, _hsv_cv(bgr))
    assert abs(_hsv_f(bgr)[0] - h) < 1e-9
    return bgr


def _exact_sat_pair(sat: int) -> tuple[int, int]:
    """Even (max, min) channels with float saturation exactly sat: 255*(max-min)/max == sat, max >= 2*V_MIN."""
    for hi in range(254, 2 * V_MIN - 1, -2):
        for d in range(2, hi, 2):
            if 255 * d == sat * hi:
                return hi, hi - d
    raise RuntimeError(f"no exact pair for S={sat}")


# Float-hue intervals at least MARGIN['hue'] inside each class (the decision edges sit at lo-0.5 and hi+0.5).
_SAFE = {"red": ((0.0, HUE["red"][0][1] - MARGIN["hue"]), (HUE["red"][1][0] + MARGIN["hue"], 180.0)),
         **{c: ((HUE[c][0][0] + MARGIN["hue"], HUE[c][0][1] - MARGIN["hue"]),) for c in ("yellow", "green", "blue")}}
_GAP_SAFE = {"orange": (11.5, 18.5), "cyan": (81.5, 98.5), "violet": (131.5, 168.5)}  # lime is too narrow


def _in(x: float, intervals) -> bool:
    return any(lo <= x <= hi for lo, hi in intervals)


def _class_hue(rng: np.random.Generator, cls: str, lobe: int = 0) -> int:
    """An integer hue at least one unit inside the class (red: lobe 0 is 0-8, lobe 1 is 172-179)."""
    if cls == "red":
        return int(rng.integers(0, 9)) if lobe == 0 else int(rng.integers(172, 180))
    lo, hi = HUE[cls][0]
    return int(rng.integers(lo + 2, hi - 1))


def _body_colour(rng: np.random.Generator, cls: str, lobe: int = 0, *, dark: bool = False,
                 v: int | None = None) -> tuple[int, int, int]:
    """A random colour of class cls, comfortably inside the class (dark=True: inside every bound but V).

    v pins the value channel (the V edge swatch uses 2*V_MIN, which dimming lands exactly on V_MIN)."""
    for _ in range(1000):
        h = _class_hue(rng, cls, lobe)
        if dark:
            s, val = int(rng.integers(120, 256)), int(rng.integers(24, 67))
        else:
            s, val = int(rng.integers(S_MIN + 4, 256)), (v if v is not None else int(rng.integers(172, 255)))
        bgr = _even_bgr(h, s, val)
        hf, sf, vf = _hsv_f(bgr)
        if not _in(hf, _SAFE[cls]):
            continue
        if dark:
            if sf >= S_MIN + 10 and 20 <= vf <= V_MIN - 12 and classify(*_hsv_cv(bgr)) is None:
                return bgr
        elif (sf >= S_MIN + MARGIN["sat"] and (vf == v if v is not None else vf >= 2 * V_MIN + MARGIN["val"])
              and classify(*_hsv_cv(bgr)) == cls):
            return bgr
    raise RuntimeError(f"no colour found for {cls}")


def _sat_edge_colour(rng: np.random.Generator, cls: str, sat: int) -> tuple[int, int, int]:
    """Colour with float saturation exactly sat (so 0.5 from the rounding edge) and the hue family of cls."""
    hi, lo = _exact_sat_pair(sat)
    for _ in range(1000):
        bgr = _hue_bgr(float(_class_hue(rng, cls, int(rng.integers(2)))), hi, lo)
        hf, sf, vf = _hsv_f(bgr)
        if _in(hf, _SAFE[cls]) and abs(sf - sat) < 1e-9 and _hsv_cv(bgr)[1] == sat \
                and classify(*_hsv_cv(bgr)) == (cls if sat >= S_MIN else None):
            return bgr
    raise RuntimeError(f"no S={sat} colour found for {cls}")


def _val_out_colour(rng: np.random.Generator, cls: str) -> tuple[int, int, int]:
    """Colour of class cls in hue and S whose max channel is V_MIN - 1 exactly: out by one unit of V."""
    for _ in range(2000):
        h, s = _class_hue(rng, cls, int(rng.integers(2))), int(rng.integers(120, 256))
        bgr = tuple(int(c) for c in cv2.cvtColor(np.array([[[h, s, V_MIN - 1]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0])
        hf, sf, vf = _hsv_f(bgr)
        if _in(hf, _SAFE[cls]) and sf >= S_MIN + 10 and vf == V_MIN - 1 and classify(*_hsv_cv(bgr)) is None:
            return bgr
    raise RuntimeError(f"no V={V_MIN - 1} colour found for {cls}")


def _distractor_colour(rng: np.random.Generator, tag: str, bg: tuple[int, int, int]) -> tuple[tuple, str]:
    if tag == "gray":
        for _ in range(100):
            g = 2 * int(rng.integers(0, 128))
            if abs(g - max(bg)) >= 40:
                return (g, g, g), "gray"
    if tag == "pastel":  # the right hue family but S below S_MIN: only a missing saturation gate counts it
        for _ in range(1000):
            bgr = _even_bgr(int(rng.integers(0, 180)), int(rng.integers(30, 81)), int(rng.integers(170, 255)))
            hf, sf, vf = _hsv_f(bgr)
            if 25 <= sf <= S_MIN - 15 and vf >= 170 and classify(*_hsv_cv(bgr)) is None:
                return bgr, "pastel"
    if tag == "dark":  # the right hue and S but V below V_MIN: only a missing value gate counts it
        cls = COLOURS[int(rng.integers(len(COLOURS)))]
        return _body_colour(rng, cls, int(rng.integers(2)), dark=True), f"dark {cls}"
    if tag == "gap":  # full S and V but a hue between the ranges
        name = tuple(_GAP_SAFE)[int(rng.integers(len(_GAP_SAFE)))]
        lo, hi = GAP_HUES[name]
        for _ in range(1000):
            bgr = _even_bgr(int(rng.integers(lo + 1, hi)), int(rng.integers(110, 256)), int(rng.integers(172, 255)))
            hf, sf, vf = _hsv_f(bgr)
            if _GAP_SAFE[name][0] <= hf <= _GAP_SAFE[name][1] and sf >= 108 and classify(*_hsv_cv(bgr)) is None:
                return bgr, f"gap {name}"
    raise RuntimeError(f"no colour found for {tag}")


def _background(rng: np.random.Generator) -> tuple[int, int, int]:
    mode = int(rng.integers(3))
    if mode == 0:  # black or white
        g = (0, 254)[int(rng.integers(2))]
        return (g, g, g)
    if mode == 1:  # any even grey
        g = 2 * int(rng.integers(0, 128))
        return (g, g, g)
    for _ in range(1000):  # a tinted pastel wall: right hue family, S far below S_MIN
        bgr = _even_bgr(int(rng.integers(0, 180)), int(rng.integers(20, 71)), int(rng.integers(60, 255)))
        if _hsv_f(bgr)[1] <= S_MIN - 20 and classify(*_hsv_cv(bgr)) is None:
            return bgr
    raise RuntimeError("no background found")


# ---------------------------------------------------------------- shapes

Draw = Callable[[np.ndarray, int, int, int], None]


def _ellipse_area(a: float, b: float) -> float:
    return math.pi * a * b


def _propose(kind: str, rng: np.random.Generator, edge: bool = False) -> tuple[int, int, Draw, float | None]:
    """Sample one shape: (half-width, half-height, draw, analytic area).

    ``draw(canvas, cx, cy, v)`` paints label v with LINE_8 (no anti-aliasing) onto a uint8 label canvas.
    The half extents cover every painted pixel."""
    if kind == "rect":  # w x h pixels exactly: both corners are inclusive in cv2.rectangle
        w, h = (int(x) for x in rng.integers(10, 25, size=2)) if edge else (int(x) for x in rng.integers(8, 61, size=2))

        def draw(c, cx, cy, v):
            x0, y0 = cx - w // 2, cy - h // 2
            cv2.rectangle(c, (x0, y0), (x0 + w - 1, y0 + h - 1), v, -1)
        return w // 2 + 1, h // 2 + 1, draw, float(w * h)

    if kind == "disk":
        r = int(rng.integers(6, 14)) if edge else int(rng.integers(4, 33))

        def draw(c, cx, cy, v):
            cv2.circle(c, (cx, cy), r, v, -1)
        return r + 1, r + 1, draw, math.pi * r * r

    if kind == "ellipse":
        a = int(rng.integers(6, 39))
        b = int(rng.integers(4, a + 1))
        ang = float(rng.uniform(0, 180))

        def draw(c, cx, cy, v):
            cv2.ellipse(c, (cx, cy), (a, b), ang, 0, 360, v, -1)
        return a + 1, a + 1, draw, _ellipse_area(a, b)

    if kind == "rotrect":
        w, h = (float(x) for x in rng.integers(8, 57, size=2))
        ang = float(rng.uniform(0, 90))
        half = int(math.ceil(math.hypot(w, h) / 2)) + 1

        def draw(c, cx, cy, v):
            pts = np.round(cv2.boxPoints(((cx, cy), (w, h), ang))).astype(np.int32)
            cv2.fillPoly(c, [pts], v)
        return half, half, draw, None

    if kind == "triangle":
        R = float(rng.integers(10, 37))
        base = rng.uniform(0, 2 * math.pi)
        ang = base + np.array([0, 2 * math.pi / 3, 4 * math.pi / 3]) + rng.uniform(-0.35, 0.35, 3)
        rad = R * rng.uniform(0.7, 1.0, 3)

        def draw(c, cx, cy, v):
            pts = np.round(np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]).astype(np.int32)
            cv2.fillPoly(c, [pts], v)
        return int(R) + 1, int(R) + 1, draw, None

    if kind == "ring":  # rectangle with a rectangular hole: the hole is background, not part of the colour
        w, h = (int(x) for x in rng.integers(24, 65, size=2))
        m = int(rng.integers(4, min(w, h) // 2 - 3))

        def draw(c, cx, cy, v):
            x0, y0 = cx - w // 2, cy - h // 2
            cv2.rectangle(c, (x0, y0), (x0 + w - 1, y0 + h - 1), v, -1)
            cv2.rectangle(c, (x0 + m, y0 + m), (x0 + w - 1 - m, y0 + h - 1 - m), 0, -1)
        return w // 2 + 1, h // 2 + 1, draw, float(w * h - (w - 2 * m) * (h - 2 * m))

    if kind == "speckle":  # 2-6 px specks: no contour area to speak of, no business being filtered out
        w, h = (int(x) for x in rng.integers(2, 7, size=2))

        def draw(c, cx, cy, v):
            cv2.rectangle(c, (cx, cy), (cx + w - 1, cy + h - 1), v, -1)
        return w + 1, h + 1, draw, float(w * h)

    if kind == "line":  # 1-px wide: contourArea of its mask is ~0, its pixel count is not
        while True:
            dx, dy = (int(x) for x in rng.integers(-30, 31, size=2))
            if max(abs(dx), abs(dy)) >= 8:
                break

        def draw(c, cx, cy, v):
            cv2.line(c, (cx - dx, cy - dy), (cx + dx, cy + dy), v, 1)
        e = max(abs(dx), abs(dy)) + 1
        return e, e, draw, None

    raise ValueError(f"unknown kind {kind!r}")


def _render(labels: np.ndarray, colours: list[tuple[int, int, int]]) -> np.ndarray:
    """img = LUT[labels]: the image is a pure function of the label map (colours[0] is the background)."""
    lut = np.zeros((256, 3), np.uint8)
    lut[:len(colours)] = np.array(colours, np.uint8)
    return lut[labels]


def make_scene(seed: int, size: tuple[int, int] = SIZE) -> Scene:
    """Manufacture a scene: 4-7 colour shapes, 4-6 distractors and 22 edge swatches on a plain background.

    The swatches are a test chart for the contract's bounds: each of the 10 inclusive hue bounds and each of
    the 8 hues one step outside them, S = S_MIN and S_MIN - 1, V = 2*V_MIN (which the dim symmetry lands on
    V_MIN) and V = V_MIN - 1. In OMIT_P of the scenes one colour is left out altogether, together with its
    in-range swatches (the out-of-range ones stay: they must still count for nothing)."""
    rng = np.random.default_rng(seed)
    H, W = size
    bg = _background(rng)
    omitted = COLOURS[int(rng.integers(len(COLOURS)))] if rng.random() < OMIT_P else None
    present = [c for c in COLOURS if c != omitted]
    plan: list[tuple[str, str | None, str, tuple[int, int, int], tuple[str, ...]]] = []  # role, cls, tag, bgr, kinds

    lobe = int(rng.integers(2))  # successive red shapes alternate lobes, so both occur in most scenes
    for _ in range(int(rng.integers(4, 8))):
        cls = present[int(rng.integers(len(present)))]
        if cls == "red":
            lobe ^= 1
        plan.append(("body", cls, f"body {cls}", _body_colour(rng, cls, lobe), KINDS))

    extra = [DISTRACTORS[int(rng.integers(len(DISTRACTORS)))] for _ in range(int(rng.integers(0, 3)))]
    kinds = list(DISTRACTORS) + extra  # every distractor type at least once, in random order
    for t in (kinds[int(i)] for i in rng.permutation(len(kinds))):
        bgr, tag = _distractor_colour(rng, t, bg)
        plan.append(("distractor", None, tag, bgr, KINDS))

    for h, cls in IN_EDGES:
        if cls != omitted:
            plan.append(("edge", cls, f"edge H={h} in {cls}", _edge_bgr(h), EDGE_KINDS))
    for h in OUT_EDGES:
        plan.append(("edge", None, f"edge H={h} out", _edge_bgr(h), EDGE_KINDS))
    c_s_in, c_v_in = (present[int(rng.integers(len(present)))] for _ in range(2))
    c_s_out, c_v_out = (COLOURS[int(rng.integers(len(COLOURS)))] for _ in range(2))  # hue family only
    plan += [
        ("edge", c_s_in, f"edge S={S_MIN} in {c_s_in}", _sat_edge_colour(rng, c_s_in, S_MIN), EDGE_KINDS),
        ("edge", None, f"edge S={S_MIN - 1} out", _sat_edge_colour(rng, c_s_out, S_MIN - 1), EDGE_KINDS),
        ("edge", c_v_in, f"edge V={2 * V_MIN} in {c_v_in} (V={V_MIN} once dimmed)",
         _body_colour(rng, c_v_in, int(rng.integers(2)), v=2 * V_MIN), EDGE_KINDS),
        ("edge", None, f"edge V={V_MIN - 1} out", _val_out_colour(rng, c_v_out), EDGE_KINDS),
    ]

    labels = np.zeros((H, W), np.uint8)
    colours = [bg]
    shapes: list[Shape] = []
    for role, cls, tag, bgr, allowed in plan:
        for _ in range(800):
            kind = allowed[int(rng.integers(len(allowed)))]
            hw, hh, draw, analytic = _propose(kind, rng, edge=role == "edge")
            if W - 2 * (BORDER + hw) <= 0 or H - 2 * (BORDER + hh) <= 0:
                continue
            cx = int(rng.integers(BORDER + hw, W - BORDER - hw))
            cy = int(rng.integers(BORDER + hh, H - BORDER - hh))
            box = (cx - hw, cy - hh, cx + hw, cy + hh)
            if any(box[0] - GAP <= b[2] and b[0] - GAP <= box[2] and box[1] - GAP <= b[3] and b[1] - GAP <= box[3]
                   for b in (s.box for s in shapes)):
                continue
            idx = len(shapes) + 1
            draw(labels, cx, cy, idx)
            colours.append(bgr)
            shapes.append(Shape(kind, role, cls, tag, bgr, int(np.count_nonzero(labels == idx)), analytic, box))
            break
        else:
            raise RuntimeError(f"seed {seed}: no room for {tag!r}")
    truth = {c: sum(s.area for s in shapes if s.cls == c) for c in COLOURS}
    return Scene(_render(labels, colours), labels, shapes, bg, truth, seed, omitted)


def inputs(scene: Scene) -> dict:
    return {"img": scene.img}


# ---------------------------------------------------------------- verifiers

def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = check_outputs(SPEC, out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    wrong = [f"{c}: {out[c]} px, manufactured {scene.truth[c]} px" for c in COLOURS
             if abs(out[c] - scene.truth[c]) > TOL_PX]
    if wrong:
        return Verdict(False, "pixel counts differ from the manufactured ones (tolerance "
                              f"{TOL_PX} px): " + "; ".join(wrong))
    return Verdict(True)


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task: each must leave the four counts unchanged (see the module docstring)."""
    img = scene.img
    H, W = img.shape[:2]
    dx, dy = 7, -5  # well inside the BORDER margin
    shifted = np.empty_like(img)
    shifted[:] = scene.background
    shifted[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return {
        "translate(+7,-5)": {"img": shifted},
        "rot90": {"img": np.ascontiguousarray(np.rot90(img))},
        "mirror": {"img": np.ascontiguousarray(img[:, ::-1])},
        "dim x0.5": {"img": np.ascontiguousarray(img >> 1)},  # exact: every colour here is even
    }


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = check_outputs(SPEC, base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    for name, out in variants.items():
        problems = check_outputs(SPEC, out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        moved = [f"{c} {base[c]} -> {out[c]}" for c in COLOURS if abs(out[c] - base[c]) > META_TOL_PX]
        if moved:
            return Verdict(False, f"{name}: counts changed (allowed {META_TOL_PX} px): " + "; ".join(moved))
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    spans = {
        "red": [(0, 10), (170, 179)],  # red wraps around hue 0/180: two ranges
        "yellow": [(20, 35)],
        "green": [(40, 80)],
        "blue": [(100, 130)],
    }
    out = {}
    for name, ranges in spans.items():
        mask = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, (lo, 100, 80), (hi, 255, 255))
        out[name] = int(cv2.countNonZero(mask))
    return out


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


def _swallow(code: str) -> str:
    """Wrap the body of solve() in try/except Exception, returning zeros: the lint's broad-except target."""
    head, body = code.split("def solve(img):\n")
    zeros = "{name: 0 for name in (\"red\", \"yellow\", \"green\", \"blue\")}"
    return (head + "def solve(img):\n    try:\n" + textwrap.indent(body.rstrip("\n"), "    ")
            + f"\n    except Exception:\n        return {zeros}\n")


_INRANGE = "mask |= cv2.inRange(hsv, (lo, 100, 80), (hi, 255, 255))"
_COUNT = "out[name] = int(cv2.countNonZero(mask))"
_COUNT_3CH = "out[name] = int(cv2.countNonZero(cv2.bitwise_and(img, img, mask=mask)))"

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "rgb2hsv_on_bgr": (
        "COLOR_RGB2HSV on a BGR image: red and blue swap, yellow turns cyan.",
        _mutate(("cv2.COLOR_BGR2HSV", "cv2.COLOR_RGB2HSV"))),
    "red_low_lobe_only": (
        "One hue range for red, 0-10: misses 170-179, the half of red that wraps around.",
        _mutate(('"red": [(0, 10), (170, 179)]', '"red": [(0, 10)]'))),
    "red_wrapped_bounds": (
        "One inRange with lower hue 170 and upper hue 10: an empty range, so red is always 0.",
        _mutate(('"red": [(0, 10), (170, 179)]', '"red": [(170, 10)]'))),
    "hue_0_360": (
        "Hue ranges on the 0-360 degree scale (all doubled): OpenCV's 8-bit hue stops at 179.",
        _mutate(('"red": [(0, 10), (170, 179)]', '"red": [(0, 20), (340, 360)]'),
                ('"yellow": [(20, 35)]', '"yellow": [(40, 70)]'),
                ('"green": [(40, 80)]', '"green": [(80, 160)]'),
                ('"blue": [(100, 130)]', '"blue": [(200, 260)]'))),
    "hsv_full_with_180_bounds": (
        "COLOR_BGR2HSV_FULL (hue 0-255) with the 0-180 bounds of the contract.",
        _mutate(("cv2.COLOR_BGR2HSV)", "cv2.COLOR_BGR2HSV_FULL)"))),
    "no_saturation_value_gate": (
        "Hue only: S and V lower bounds 0, so greys, pastels, dark pixels and black or white walls count.",
        _mutate(("(lo, 100, 80)", "(lo, 0, 0)"))),
    "value_floor_100": (
        "Copy-pasted V >= 100 instead of 80: every colour shape here is brighter, so only dimming sees it.",
        _mutate(("(lo, 100, 80)", "(lo, 100, 100)"))),
    "exclusive_hue_bounds": (
        "numpy masks with strict hue comparisons: the bounds themselves (H = 0, 10, 20, ...) are lost.",
        _mutate((_INRANGE, "keep = (hsv[..., 0] > lo) & (hsv[..., 0] < hi) & (hsv[..., 1] >= 100) "
                           "& (hsv[..., 2] >= 80)\n            mask |= keep.astype(np.uint8) * 255"))),
    "mask_sum_not_count": (
        "mask.sum() adds 255 per pixel instead of counting pixels.",
        _mutate((_COUNT, "out[name] = int(mask.sum())"))),
    "contour_area_sum": (
        "Sums cv2.contourArea over the mask's outer contours: polygon area, not pixel count; holes filled, lines ~0.",
        _mutate((_COUNT, "contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]\n"
                         "        out[name] = int(round(sum(cv2.contourArea(c) for c in contours)))"))),
    "blur_before_hsv": (
        "A habitual GaussianBlur 'to reduce noise' blends the edge pixels into other colours before the HSV test.",
        _mutate(("cv2.cvtColor(img, cv2.COLOR_BGR2HSV)",
                 "cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)"))),
    "countnonzero_3channel": (
        "Counts the masked colour image: countNonZero needs one channel. Raises cv2.error on 4.x and 5.x.",
        _mutate((_COUNT, _COUNT_3CH))),
    "swallowed_countnonzero_3channel": (
        "The same bug wrapped in try/except, returning zeros. Exec-only filtering keeps this one.",
        _swallow(_mutate((_COUNT, _COUNT_3CH)))),
}


# ---------------------------------------------------------------- calibration

def _gauge_block(rng: np.random.Generator, kind: str, cls: str) -> tuple[np.ndarray, int]:
    """One shape of one colour class on a padded plain canvas: (image, manufactured pixel count)."""
    bg = _background(rng)
    bgr = _body_colour(rng, cls, int(rng.integers(2)))
    hw, hh, draw, _ = _propose(kind, rng)
    H, W = 2 * (hh + BORDER), 2 * (hw + BORDER)
    labels = np.zeros((H, W), np.uint8)
    draw(labels, W // 2, H // 2, 1)
    return _render(labels, [bg, bgr]), int(np.count_nonzero(labels))


def antialias_evidence(n: int = 300, seed: int = 4242) -> dict:
    """Why shapes are drawn without anti-aliasing: extra 'red' pixels of a LINE_AA disk, per unit perimeter."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for bg in (0, 100, 254):
        ratios = []
        for _ in range(n):
            r = int(rng.integers(8, 41))
            plain = np.full((120, 120, 3), bg, np.uint8)
            blended = plain.copy()
            cv2.circle(plain, (60, 60), r, (40, 40, 250), -1)
            cv2.circle(blended, (60, 60), r, (40, 40, 250), -1, cv2.LINE_AA)
            ratios.append((reference_solve(blended)["red"] - reference_solve(plain)["red"]) / (2 * math.pi * r))
        out[f"bg{bg}"] = (round(float(min(ratios)), 3), round(float(max(ratios)), 3))
    return out


def conversion_evidence(n: int = 2_000_000, seed: int = 99) -> dict:
    """How cv2's fixed-point BGR2HSV differs from the float formula on uniformly random 8-bit colours."""
    rng = np.random.default_rng(seed)
    px = rng.integers(0, 256, size=(1, n, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0].astype(np.int32)
    b, g, r = (px[0, :, i].astype(np.float64) for i in range(3))
    mx, mn = np.maximum(np.maximum(b, g), r), np.minimum(np.minimum(b, g), r)
    d = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        s_f = np.where(mx > 0, 255 * d / mx, 0.0)
        raw = np.where(mx == r, (g - b) / d, np.where(mx == g, 2 + (b - r) / d, 4 + (r - g) / d))
    h_f = np.where(d > 0, (30 * raw) % 180, 0.0)
    h_r, s_r = np.floor(h_f + 0.5).astype(np.int32) % 180, np.floor(s_f + 0.5).astype(np.int32)
    dh = np.abs(hsv[:, 0] - h_f)
    dh = np.minimum(dh, 180 - dh)
    off = np.abs(((h_f + 0.5) % 1.0 + 0.5) % 1.0 - 0.5)  # distance of h_f from the nearest k+0.5 rounding edge
    flipped = classify_array(hsv[:, 0], hsv[:, 1], hsv[:, 2]) != classify_array(h_r, s_r, mx.astype(np.int32))
    mis = hsv[:, 0] != h_r
    return {"n": n, "max_abs_hue_err": float(dh.max()), "hue_not_rounded_float": float(mis.mean()),
            "mismatch_max_dist_from_rounding_edge": float(off[mis].max()), "class_flipped": float(flipped.mean())}


def hsv_cube() -> np.ndarray:
    """Every 8-bit BGR colour exactly once, as a 4096 x 4096 x 3 image (blue slowest, red fastest)."""
    v = np.arange(256, dtype=np.uint8)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([b, g, r], -1).reshape(4096, 4096, 3)


def hsv_cube_fingerprint() -> dict:
    """sha1 of cv2.cvtColor(hsv_cube(), COLOR_BGR2HSV). Run it under 4.14 and 5.0: equal hashes would mean the
    8-bit BGR2HSV conversion did not change at all, which the one-pixel probe in diff_4.14_vs_5.0.md cannot say."""
    import hashlib
    hsv = cv2.cvtColor(hsv_cube(), cv2.COLOR_BGR2HSV)
    return {"cv2": cv2.__version__, "sha1": hashlib.sha1(hsv.tobytes()).hexdigest(),
            "h_max": int(hsv[..., 0].max()), "s_max": int(hsv[..., 1].max()), "v_max": int(hsv[..., 2].max())}


def calibrate(n: int = 1200, seed: int = 12345, evidence: bool = True) -> dict:
    """Measure (reference count - manufactured count) in every colour on single-shape gauge blocks.

    evidence=True also records why shapes are drawn without anti-aliasing and why colours keep a margin."""
    rng = np.random.default_rng(seed)
    errors, by_kind = [], {}
    for i in range(n):
        kind, cls = KINDS[i % len(KINDS)], COLOURS[(i // len(KINDS)) % len(COLOURS)]
        img, area = _gauge_block(rng, kind, cls)
        out = reference_solve(img)
        err = [out[c] - (area if c == cls else 0) for c in COLOURS]
        errors.append(err)
        by_kind.setdefault(kind, []).extend(err)
    e = np.abs(np.array(errors))
    out = {"n": n, "worst_abs_error_px": int(e.max()), "blocks_with_any_error": int((e.max(axis=1) > 0).sum()),
           "by_kind": {k: (int(min(v)), int(max(v))) for k, v in sorted(by_kind.items())}}
    if evidence:
        out.update(antialias=antialias_evidence(), conversion=conversion_evidence(n=200_000))
    return out


def measure_metamorphic_drift(n: int = 300, seed: int = 50_000) -> int:
    """Largest movement of any of the reference's four counts under the task symmetries."""
    worst = 0
    for i in range(n):
        scene = make_scene(seed + i)
        base = reference_solve(scene.img)
        for t in transforms(scene).values():
            got = reference_solve(t["img"])
            worst = max(worst, max(abs(got[c] - base[c]) for c in COLOURS))
    return worst


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=1))
    print("full-size conversion evidence:", conversion_evidence())
    print("BGR2HSV over the whole 8-bit colour cube:", hsv_cube_fingerprint())
    print("metamorphic drift of the reference:", measure_metamorphic_drift())
