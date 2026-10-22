"""Family ``components_stats``: area, bounding box and centroid of every connected component of a bool mask.

The answer is chosen first (Method of Manufactured Solutions, as in ``contour_area``): the generator
draws objects as *integer pixel sets* whose components, areas, boxes and centroids it knows from the
construction, never from running OpenCV. The code under test must recover that answer.

Tolerance law
-------------
Unlike ``contour_area`` there is no measurement physics here: nothing is estimated from a boundary and
nothing is interpolated. Every quantity is a function of an integer pixel set, so the right answer
is exact, and the tolerance law is "zero, plus the rounding of one division":

* ``area``     integer pixel count. Zero tolerance. A one-pixel error is a defect (a wrong neighbourhood,
               a dropped pixel, a merged or split component), not noise.
* ``box``      (x, y, w, h), integers, w = x_max - x_min + 1 (inclusive extent). Zero tolerance.
* ``centroid`` (x, y) = (sum of pixel columns / area, sum of pixel rows / area), the centre of pixel
               (col, row) being at (col, row). A rational number. The sums are exact integers
               (< 2^53) and the single division is correctly rounded, so the float64 error is
               <= 1/2 ulp of the coordinate = 2^-45 ~ 2.8e-14 px for coordinates < 512 (predicted).
               ``CENTROID_TOL`` = 1e-6 px sits between that floor and the smallest defect the exact
               area and box checks cannot already see: one pixel assigned to the wrong component
               moves a centroid by >= 1/A px, ~4e-4 px for the largest component the generator
               builds. float32 rounding (up to 2^-16 ~ 1.5e-5 px for x in [256, 512)) exceeds the
               tolerance, so the contract demands float64.
* Nothing scales with object size or perimeter. The only size dependence is the floating-point
  floor, which scales with the coordinate magnitude.

What the tolerance cannot absorb must be fixed by the contract instead, so the contract states: the
connectivity (an *input*, 4 or 8, and the scenes are built so that the two give different answers),
the box convention, the centroid origin (pixel centre at integer coordinates, no +0.5), and the order
of the output (area descending; equal areas by the top-left corner of the box, y then x).

Two measured OpenCV 5.0.0 facts shape the family (see ``measure_bool_mask_support`` and the tests)
-----------------------------------------------------------------------------------------------
* "numpy bool masks are now accepted" is true of the binding, not of every function behind it, and "does not
  raise" is not "gives the right answer". Re-measured on OpenCV 5.0.0 / numpy 2.4.6 with
  ``measure_bool_mask_support`` and a throwaway script (an 8x8 mask holding one 3x4 block of True):
    - ``countNonZero(mask)`` works: 12, as for the uint8 mask.
    - ``moments(mask)`` raises cv2.error (-210, unsupported format) with the default ``binaryImage=False``.
      It works only as ``moments(mask, binaryImage=True)``, which equals the uint8 result in all 24 moments.
    - ``bitwise_not(mask)`` does not raise and is a SILENT WRONG ANSWER: it returns a bool array whose bytes
      are 0xFE where the mask is True and 0xFF where it is False (the uint8 operation on the bytes 1 and 0,
      with a bool dtype on top), so every element reads True (``.all()`` is True, ``.sum()`` is 64 where
      ``~mask`` has 52). Use ``~mask`` / ``np.logical_not``, or ``bitwise_not`` on a 0/255 uint8 mask.
    - ``connectedComponents[WithStats]`` (assertion ``iDepth == CV_8U || iDepth == CV_8S``), ``findContours``
      and ``threshold`` (-210, unsupported format, raised inside ``threshold``) and ``dilate`` (-213,
      "Unsupported data type (=9)") still raise: errors inside the algorithm, not at the binding. On 4.14.0 all
      eight functions fail at the binding with "data type = bool is not supported".
  The input is a bool mask so the 4.x workaround (``mask.astype(np.uint8)``) is still required, and the idiom
  "5.0 takes bools, so pass it straight in" fails (or, for ``bitwise_not``, silently corrupts the mask).
* The default CCL algorithm labels components in raster order of their first pixel at connectivity 4,
  but *not* at connectivity 8 (200 of 200 random masks violate it; WU and SAUF are the exceptions among the
  seven algorithms). The pattern fits a block-based scan over pairs of rows: two equal-area components whose
  top rows fall in the same pair of rows are labelled left to right, whichever sits higher. So "sort by
  area, stable" on the raw labels is not the contract's tie-break (it disagrees on ~1.4 % of random
  equal-area pairs). Every scene contains such a stagger pair, so the tie-break is graded.

Metamorphic checks
------------------
The task's symmetries are *covariant*, not invariant: a shift, a rotation by 90 degrees and a mirror
move the boxes and centroids exactly as they move the image, and leave areas alone. Switching the
connectivity from 8 to 4 can only refine the partition: same total area, at least as many components,
each fine component inside a coarse one. ``evaluate_solution`` calls ``verify_metamorphic`` without
the scene, so the covariant maps are computed from the transform name and the fixed image size ``SIZE``.
The shift pads the canvas (``np.pad``) instead of rolling it, so components that touch the border survive
it. Components are compared as sets (sorted by corner): the output order is graded by the manufactured gate only.
Blind spots, as always: a constant output (an empty answer) is invariant under every symmetry, and so is a
solution that ignores an input the symmetry changes (``connectivity_hardcoded_8``). That is why the
manufactured gate stays the main check and the metamorphic one complements it.
"""
from __future__ import annotations

import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

SPEC = TaskSpec(
    family="components_stats",
    task=("Label the connected components of the True (foreground) pixels of a boolean mask, using the given "
          "pixel connectivity (4: pixels sharing an edge are connected; 8: pixels sharing an edge or a corner "
          "are connected). The background is not a component. For each component report its area (pixel "
          "count), bounding box and centroid. List the components largest area first; components of equal "
          "area go by the top-left corner of their bounding box, top to bottom (y), then left to right (x)."),
    inputs={
        "mask": IOSpec("bool", "HxW", 2, "True = foreground; components may touch the image border"),
        "connectivity": IOSpec("int", "scalar", None, "4 = edge neighbours only, 8 = edge and corner neighbours"),
    },
    outputs={
        "count": IOSpec("int", "scalar", None, "number of foreground components (the background is not counted)"),
        "areas": IOSpec("int", "N", 1, "pixels per component, in the order stated in the task"),
        "boxes": IOSpec("int", "Nx4", 2, "(x, y, w, h) per component: x = leftmost column, y = topmost row, "
                                         "w and h = pixel columns and rows spanned, so w = x_max - x_min + 1"),
        "centroids": IOSpec("float64", "Nx2", 2, "(x, y) per component: mean column and mean row of its pixels; "
                                                  "the centre of pixel (col, row) is at (col, row), no +0.5"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Treat the boolean mask as a binary image and label its connected foreground components with the given "
    "connectivity (4: shared edges only; 8: shared edges or corners). For every component give the pixel "
    "count, the bounding box (x, y, width, height) and the centroid (x, y), where the centre of pixel "
    "(col, row) is at (col, row). Skip the background. Largest component first; equal areas go top to "
    "bottom, then left to right, by the top-left corner of the box.",
    "Find the blobs in this True/False mask. Two True pixels belong to the same blob when a chain of "
    "neighbouring True pixels links them: edge neighbours only when connectivity is 4, edge or corner "
    "neighbours when it is 8. Return the number of blobs and, per blob, its area in pixels, its tight "
    "bounding box (x, y, w, h) and its centroid (x, y), sorted by area descending, ties by box top and then "
    "box left. The background is not a blob.",
    "Compute connected-component statistics (area, bounding box, centroid) for the True region(s) of a "
    "boolean mask at 4- or 8-connectivity, background excluded. Order: area descending, then box y, then "
    "box x. Boxes are (x, y, w, h) with w and h counted in pixels; centroids are (x, y) means of pixel "
    "coordinates.",
    "Label the connected True regions of the mask (connectivity 4: edge neighbours only; 8: edge or corner "
    "neighbours) and give each one's area, bounding box and centroid. Sort by area, largest first; equal "
    "areas by the box's top-left corner, y then x. The background is not a component.",
    "In this True/False mask, find the separate clumps of True pixels. With connectivity 4 only pixels "
    "sharing an edge are linked; with 8, pixels sharing an edge or a corner are linked too. For each clump "
    "return its area (pixel count), bounding box and centroid, biggest clump first; if two have the same area, the "
    "one whose box starts higher (then further left) goes first. The background (False) is not a clump.",
    "As a defect-analysis developer, I get a boolean mask of flagged pixels and a connectivity (4 = shared "
    "edges, 8 = shared edges or corners). I need every connected group of flagged pixels with its area, "
    "bounding box and centroid, ordered by area descending and then by bounding-box top-left (y, then x). The "
    "unflagged background is not a group.",
    "How many connected regions of True pixels does the mask have under the given connectivity (4: shared "
    "edges; 8: shared edges or corners), and what are the area, bounding box and centroid of each? List them "
    "largest area first, ties broken by the box's top-left corner (top to bottom, then left to right); the "
    "background is not a region.",
    "(1) Label the connected components of the True pixels, using the given connectivity (4: edge neighbours; "
    "8: edge or corner neighbours), ignoring the background. (2) For each component report area, bounding box "
    "and centroid. (3) Order them by area, largest first, equal areas by the box's top-left corner: y, then "
    "x.",
    "Requirement: for a boolean mask and a connectivity of 4 (edge-adjacent) or 8 (edge- or corner-adjacent), "
    "output the number of foreground components and each component's pixel area, bounding box and centroid. "
    "Components are listed by descending area, ties by the top-left corner of the box (y first, then x). The "
    "background is excluded.",
    "A particle counter thresholds a micrograph into a boolean mask. Using the given connectivity (4: shared "
    "edge; 8: shared edge or corner), report every particle, i.e. every connected group of True pixels "
    "(background excluded), with its area, bounding box and centroid, largest first and ties by the top-left "
    "corner of the box, top to bottom then left to right.",
    "Input: a boolean mask and a connectivity (4 or 8). Output: the count of connected True components "
    "(background not included), and per component its area, bounding box and centroid, sorted by descending "
    "area; equal areas by the box's top-left corner, y then x. Connectivity 4 links edge neighbours, 8 also "
    "corner neighbours.",
    "Connected-component analysis of a boolean mask at connectivity 4 or 8: area, bounding box and centroid "
    "per component, background excluded, largest area first, ties by box top then left.",
    "Could you analyse the connected foreground components of this mask? For each, I want the area, bounding "
    "box and centroid, with 4-connectivity meaning shared edges and 8-connectivity meaning shared edges or "
    "corners. Put the largest first, break ties by the box's top-left corner (y, then x), and leave the "
    "background out.",
    "As a developer measuring objects in a segmentation mask, I want to count the connected groups of True "
    "pixels at 4- or 8-connectivity and get each group's area, bounding box and centroid, biggest area first "
    "(equal areas ordered by box y, then x). The False background is not a group.",
    "1. Group the True pixels of the mask into connected components at the given connectivity (4: edges only, "
    "8: edges and corners); the background is not a component. 2. Measure each component's area, bounding box "
    "and centroid. 3. Return them ordered by area descending, then box y, then box x.",
]

SIZE = (240, 320)  # (H, W) of every manufactured mask; verify_metamorphic relies on it (it never sees the scene)
H0, W0 = SIZE
GAP = 1  # empty pixels kept between the reserved boxes of different objects (so none touch, even diagonally)
P_EDGE = 0.1  # chance that an object is snapped against an image edge (66.6 % of scenes touch the border: make_scene)
PAD = ((5, 2), (7, 3))  # np.pad widths ((top, bottom), (left, right)): the "translate" symmetry
SHIFT = (PAD[1][0], PAD[0][0])  # (dx, dy) = (7, 5) that padding applies to every x, y
TRANSLATE = f"translate(+{SHIFT[0]},+{SHIFT[1]})"

# Tolerances, from the law above, checked by calibrate() on 1200 single-object gauge blocks (12 kinds x 100,
# both connectivities, 4320 components; OpenCV 5.0.0 connectedComponentsWithStats):
#   area error   max 0     box error   max 0      (all seven CCL algorithms return the same set on every block)
#   centroid error against the manufactured truth: 0.0 for the CCL centroids and 0.0 for cv2.moments, i.e.
#   below the predicted floor of 2.8e-14 px; cast to float32 it reaches 1.47e-5 px, 15x CENTROID_TOL (hence the
#   float64 contract). 1/A for the largest component the generator builds (2400 px) is 4.2e-4 px, 420x CENTROID_TOL.
# META_TOL: measure_metamorphic_drift() over 100 scenes x 3 geometric maps found 0 integer mismatches and a
# centroid drift of 5.7e-14 px (1 ulp at 256..512: the covariant map's own subtraction), and 0 failures of the
# connectivity-refinement relation. Both tolerances therefore sit 1.7e7x above the noise and 420x below the defect.
CENTROID_TOL = 1e-6
META_TOL = 1e-6


@dataclass(frozen=True)
class Component:
    area: int
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (-self.area, self.y, self.x)


@dataclass(frozen=True)
class Placed:
    kind: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 reserved on the canvas (inclusive)
    comps8: tuple[Component, ...]  # the object's components at connectivity 8
    comps4: tuple[Component, ...]  # ... and at connectivity 4


@dataclass
class Scene:
    mask: np.ndarray
    connectivity: int
    objects: list[Placed]
    seed: int

    def truth_for(self, connectivity: int) -> list[Component]:
        comps = [c for o in self.objects for c in (o.comps8 if connectivity == 8 else o.comps4)]
        return sorted(comps, key=lambda c: c.order_key)

    @property
    def truth(self) -> list[Component]:
        return self.truth_for(self.connectivity)


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- scene generation

@dataclass
class Obj:
    """An object before placement: a local pixel array and its components at each connectivity."""
    kind: str
    pixels: np.ndarray  # bool, the union
    groups8: list[np.ndarray]  # bool arrays of the same shape: one per component at connectivity 8
    groups4: list[np.ndarray]
    even_row: bool = False  # must be placed with an even top row (the stagger pair)


def _solid(kind: str, pix: np.ndarray) -> Obj:
    pix = np.ascontiguousarray(pix, dtype=bool)
    return Obj(kind, pix, [pix], [pix])


def _singles(pix: np.ndarray) -> list[np.ndarray]:
    out = []
    for r, c in zip(*np.nonzero(pix)):
        one = np.zeros_like(pix)
        one[r, c] = True
        out.append(one)
    return out


def _ints(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _rect(rng):
    return _solid("rect", np.ones((_ints(rng, 1, 40), _ints(rng, 1, 60)), bool))


def _disk(rng):
    r = _ints(rng, 2, 22)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return _solid("disk", xx * xx + yy * yy <= r * r)


def _triangle(rng):
    t = np.tril(np.ones((_ints(rng, 4, 30),) * 2, bool))  # row i has i+1 pixels: centroid != box centre
    return _solid("triangle", t[:, ::-1] if rng.integers(2) else t)


def _lshape(rng):
    w1, h1, w2, h2 = _ints(rng, 2, 8), _ints(rng, 8, 40), _ints(rng, 8, 40), _ints(rng, 2, 8)
    pix = np.zeros((h1 + h2, max(w1, w2)), bool)
    pix[:, :w1] = True
    pix[h1:, :w2] = True
    if rng.integers(2):
        pix = pix[:, ::-1]
    if rng.integers(2):
        pix = pix[::-1]
    return _solid("lshape", pix)


def _blob(rng):
    g = 24
    cells = {(g // 2, g // 2)}
    grown = [(g // 2, g // 2)]
    want = _ints(rng, 12, 90)
    for _ in range(50 * want):
        if len(cells) >= want:
            break
        r, c = grown[int(rng.integers(len(grown)))]
        dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
        p = (r + dr, c + dc)
        if 0 <= p[0] < g and 0 <= p[1] < g and p not in cells:
            cells.add(p)
            grown.append(p)
    rows, cols = zip(*cells)
    pix = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), bool)
    pix[np.array(rows) - min(rows), np.array(cols) - min(cols)] = True
    return _solid("blob", pix)


def _ring_island(rng):
    """A rectangular frame with a separate island floating in its hole (>= 1 empty pixel all round)."""
    w, h, t = _ints(rng, 18, 40), _ints(rng, 16, 36), _ints(rng, 2, 4)
    hw, hh = w - 2 * t, h - 2 * t
    ring = np.ones((h, w), bool)
    ring[t:h - t, t:w - t] = False
    iw, ih = _ints(rng, 1, min(hw - 2, 12)), _ints(rng, 1, min(hh - 2, 12))
    ox, oy = _ints(rng, 1, hw - iw - 1), _ints(rng, 1, hh - ih - 1)
    island = np.zeros((h, w), bool)
    island[t + oy:t + oy + ih, t + ox:t + ox + iw] = True
    return Obj("ring_island", ring | island, [ring, island], [ring, island])


def _pair_gap1(rng):
    """Two rectangles with exactly one empty pixel column between them: a near miss for any closing/dilation."""
    w1, h1, w2, h2 = (_ints(rng, 2, 24) for _ in range(4))
    dy = _ints(rng, -(h2 - 1), h1 - 1)  # B starts dy rows below A's top (at least one row overlaps)
    top = min(0, dy)
    H = max(h1, dy + h2) - top
    a, b = np.zeros((H, w1 + 1 + w2), bool), np.zeros((H, w1 + 1 + w2), bool)
    a[-top:-top + h1, :w1] = True
    b[dy - top:dy - top + h2, w1 + 1:] = True
    if rng.integers(2):  # transpose: the gap becomes a row
        a, b = a.T.copy(), b.T.copy()
    return Obj("pair_gap1", a | b, [a, b], [a, b])


def _stagger_pair(rng):
    """Two equal rectangles; the right one starts one row higher, on an even row. At connectivity 8 OpenCV
    labels the lower, left one first, so a stable sort on raw labels breaks the tie the wrong way."""
    w, h, sep = _ints(rng, 3, 14), _ints(rng, 3, 14), _ints(rng, 1, 3)
    a, b = np.zeros((h + 1, 2 * w + sep), bool), np.zeros((h + 1, 2 * w + sep), bool)
    a[1:, :w] = True  # left, one row lower
    b[:h, w + sep:] = True  # right, higher: its top row is the object's row 0
    return Obj("stagger_pair", a | b, [a, b], [a, b], even_row=True)


def _dot(rng):
    return _solid("dot", np.ones((1, 1), bool))


def _diag_pair(rng):
    """Two rectangles touching only at a corner pixel: one component at connectivity 8, two at 4."""
    w1, h1, w2, h2 = (_ints(rng, 1, 14) for _ in range(4))
    a, b = np.zeros((h1 + h2, w1 + w2), bool), np.zeros((h1 + h2, w1 + w2), bool)
    a[:h1, :w1] = True
    b[h1:, w1:] = True
    if rng.integers(2):
        a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
    return Obj("diag_pair", a | b, [a | b], [a, b])


def _diag_line(rng):
    """A one-pixel-wide diagonal: one component of n pixels at connectivity 8, n single pixels at 4."""
    n = _ints(rng, 3, 12)
    pix = np.eye(n, dtype=bool)
    if rng.integers(2):
        pix = pix[:, ::-1].copy()
    return Obj("diag_line", pix, [pix], _singles(pix))


def _checker(rng):
    """A checkerboard patch: one component at connectivity 8, one per pixel at 4."""
    n, m = _ints(rng, 2, 5), _ints(rng, 2, 5)
    pix = (np.indices((n, m)).sum(axis=0) % 2) == 0
    return Obj("checker", pix, [pix], _singles(pix))


BUILDERS: dict[str, Callable] = {
    "rect": _rect, "disk": _disk, "triangle": _triangle, "lshape": _lshape, "blob": _blob,
    "ring_island": _ring_island, "pair_gap1": _pair_gap1, "stagger_pair": _stagger_pair, "dot": _dot,
    "diag_pair": _diag_pair, "diag_line": _diag_line, "checker": _checker,
}
GENERIC = ("rect", "disk", "triangle", "lshape", "blob", "ring_island", "pair_gap1")
DIAGONAL = ("diag_pair", "diag_line", "checker")  # the connectivity-sensitive kinds
KINDS = tuple(BUILDERS)


def _component(local: np.ndarray, ox: int, oy: int) -> Component:
    """Exact statistics of a local pixel array placed at (ox, oy); integer sums, one correctly rounded division."""
    ys, xs = np.nonzero(local)
    n = int(xs.size)
    x0, y0 = int(xs.min()), int(ys.min())
    return Component(n, ox + x0, oy + y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1,
                     (int(xs.sum()) + ox * n) / n, (int(ys.sum()) + oy * n) / n)


def _placed(obj: Obj, ox: int, oy: int) -> Placed:
    h, w = obj.pixels.shape
    return Placed(obj.kind, (ox, oy, ox + w - 1, oy + h - 1),
                  tuple(_component(g, ox, oy) for g in obj.groups8),
                  tuple(_component(g, ox, oy) for g in obj.groups4))


def _free(box, boxes) -> bool:
    return all(box[0] > b[2] + GAP or b[0] > box[2] + GAP or box[1] > b[3] + GAP or b[1] > box[3] + GAP
               for b in boxes)


def make_scene(seed: int) -> Scene:
    """Manufacture a mask from objects of known components. Even seeds use connectivity 8, odd seeds 4.

    Every scene holds: one connectivity-sensitive object (diag_pair / diag_line / checker), a stagger pair
    (the tie-break), two single pixels (equal areas, the smallest possible), and 3-7 other objects. Some
    objects touch the image border.

    Border contact (a scene with at least one object whose reserved box touches an image edge; on every scene
    measured this is the same set as "some True pixel lies in the first or last row or column of the mask"):
    1332 of 2000 fresh scenes, seeds 7_300_000..7_301_999, = 66.6 % (95 % Wilson interval 64.5-68.6 %; 682 of 1000
    at connectivity 8, 650 of 1000 at 4). The 120 scenes of seeds 0..119 that tests/test_components_stats.py pins
    to 30 < count < 110 hold 83 = 69.2 %, which is where the earlier "about 69 %" reproduces: a 120-scene sample, not
    the population rate (16 disjoint 120-scene batches of the fresh scenes span 71..87, i.e. 59-73 %).
    """
    rng = np.random.default_rng(seed)
    H, W = SIZE
    connectivity = 8 if seed % 2 == 0 else 4
    kinds = [DIAGONAL[int(rng.integers(len(DIAGONAL)))], "stagger_pair", "dot", "dot"]
    kinds += [GENERIC[int(rng.integers(len(GENERIC)))] for _ in range(_ints(rng, 3, 7))]
    objs = [BUILDERS[k](rng) for k in kinds]
    mask = np.zeros(SIZE, bool)
    placed: list[Placed] = []
    boxes: list[tuple[int, int, int, int]] = []
    for obj in sorted(objs, key=lambda o: -o.pixels.size):  # big first: easier packing; stable, so deterministic
        h, w = obj.pixels.shape
        for _ in range(300):
            ox, oy = _ints(rng, 0, W - w), _ints(rng, 0, H - h)
            if rng.random() < P_EDGE:  # snap against one image edge
                side = int(rng.integers(4))
                ox = (0, W - w, ox, ox)[side]
                oy = (oy, oy, 0, H - h)[side]
            if obj.even_row:
                oy -= oy % 2
            box = (ox, oy, ox + w - 1, oy + h - 1)
            if _free(box, boxes):
                mask[oy:oy + h, ox:ox + w] |= obj.pixels
                placed.append(_placed(obj, ox, oy))
                boxes.append(box)
                break
    return Scene(mask, connectivity, placed, seed)


def inputs(scene: Scene) -> dict:
    return {"mask": scene.mask, "connectivity": scene.connectivity}


# ---------------------------------------------------------------- verifiers
# Tables: (N, 7) float64 rows [area, x, y, w, h, cx, cy]. Integers below 2^53 are exact in float64.

def _truth_table(comps: list[Component]) -> np.ndarray:
    return np.array([[c.area, c.x, c.y, c.w, c.h, c.cx, c.cy] for c in comps], dtype=np.float64).reshape(-1, 7)


def _canon(t: np.ndarray) -> np.ndarray:
    """Rows sorted by (x, y, w, h, area, cx, cy): a canonical order, independent of the contract order."""
    return t[np.lexsort((t[:, 6], t[:, 5], t[:, 0], t[:, 4], t[:, 3], t[:, 2], t[:, 1]))]


def _table(out: dict) -> tuple[np.ndarray | None, str]:
    """Rows in the solution's own order, or (None, why they cannot be formed). Assumes check_outputs passed."""
    areas, boxes, cents = np.asarray(out["areas"]), np.asarray(out["boxes"]), np.asarray(out["centroids"])
    n = areas.shape[0]
    if boxes.shape != (n, 4) or cents.shape != (n, 2):
        return None, f"shapes disagree: areas {areas.shape}, boxes {boxes.shape}, centroids {cents.shape}"
    if out["count"] != n:
        return None, f"count {out['count']} but {n} rows"
    return np.column_stack([areas, boxes, cents]).astype(np.float64), ""


def _out_table(out: dict) -> np.ndarray:
    return np.column_stack([out["areas"], out["boxes"], out["centroids"]]).astype(np.float64).reshape(-1, 7)


def _fmt(row: np.ndarray) -> str:
    return f"area {int(row[0])}, box (x={int(row[1])}, y={int(row[2])}, w={int(row[3])}, h={int(row[4])})"


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = check_outputs(SPEC, out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    want = _truth_table(scene.truth)
    n = len(want)
    if out["count"] != n:
        return Verdict(False, f"count {out['count']}, but the mask has {n} components at connectivity "
                              f"{scene.connectivity}")
    got, why = _table(out)
    if got is None:
        return Verdict(False, why)
    w, g = _canon(want), _canon(got)
    bad = np.nonzero(np.any(w[:, :5] != g[:, :5], axis=1))[0]
    if bad.size:  # sets of (area, box) differ; rows sorted by top-left corner, so the first difference is local
        return Verdict(False, f"components differ: expected {_fmt(w[bad[0]])}, got {_fmt(g[bad[0]])} "
                              f"(connectivity {scene.connectivity})")
    err = np.max(np.abs(w[:, 5:] - g[:, 5:]), axis=1) if n else np.zeros(0)
    if n and not err.max() <= CENTROID_TOL:  # written so that a NaN centroid fails
        i = int(np.argmax(err))
        return Verdict(False, f"centroid of the component with {_fmt(w[i])}: expected "
                              f"({w[i, 5]:.6f}, {w[i, 6]:.6f}), got ({g[i, 5]:.6f}, {g[i, 6]:.6f}) "
                              f"(allowed {CENTROID_TOL:g} px)")
    if not np.array_equal(got[:, :5], want[:, :5]):
        k = int(np.nonzero(np.any(got[:, :5] != want[:, :5], axis=1))[0][0])
        return Verdict(False, f"components are right but not in the stated order: position {k} should hold "
                              f"{_fmt(want[k])}, holds {_fmt(got[k])}")
    return Verdict(True)


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task: the first three move boxes and centroids with the image, the last refines the partition."""
    m, c = scene.mask, scene.connectivity
    return {
        TRANSLATE: {"mask": np.pad(m, PAD), "connectivity": c},
        "rot90": {"mask": np.ascontiguousarray(np.rot90(m)), "connectivity": c},
        "mirror": {"mask": np.ascontiguousarray(m[:, ::-1]), "connectivity": c},
        f"connectivity {c}->{12 - c}": {"mask": m, "connectivity": 12 - c},
    }


def _translate(t: np.ndarray) -> np.ndarray:
    out = t.copy()
    out[:, [1, 5]] += SHIFT[0]
    out[:, [2, 6]] += SHIFT[1]
    return out


def _rot90(t: np.ndarray) -> np.ndarray:
    """np.rot90 (counter-clockwise) of a W0-wide image: pixel (x, y) -> (y, W0 - 1 - x)."""
    area, x, y, w, h, cx, cy = t.T
    return np.column_stack([area, y, W0 - x - w, h, w, cy, (W0 - 1) - cx])


def _mirror(t: np.ndarray) -> np.ndarray:
    area, x, y, w, h, cx, cy = t.T
    return np.column_stack([area, W0 - x - w, y, w, h, (W0 - 1) - cx, cy])


COVARIANT = {TRANSLATE: _translate, "rot90": _rot90, "mirror": _mirror}


def _refines(fine: np.ndarray, coarse: np.ndarray) -> str:
    """'' if `fine` (connectivity 4) can be a refinement of `coarse` (connectivity 8), else why not."""
    if fine[:, 0].sum() != coarse[:, 0].sum():
        return f"total foreground area changed: {int(coarse[:, 0].sum())} -> {int(fine[:, 0].sum())}"
    if len(fine) < len(coarse):
        return f"connectivity 4 found fewer components ({len(fine)}) than connectivity 8 ({len(coarse)})"
    for f in fine:
        if not any(c[1] <= f[1] and f[1] + f[3] <= c[1] + c[3] and c[2] <= f[2] and f[2] + f[4] <= c[2] + c[4]
                   and f[0] <= c[0] for c in coarse):
            return f"connectivity-4 component with {_fmt(f)} lies inside no connectivity-8 component"
    return ""


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = check_outputs(SPEC, base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    t0, why = _table(base)
    if t0 is None:
        return Verdict(False, "original image: " + why)
    for name, out in variants.items():
        problems = check_outputs(SPEC, out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        t, why = _table(out)
        if t is None:
            return Verdict(False, f"{name}: {why}")
        rel = re.fullmatch(r"connectivity (\d)->(\d)", name)
        if rel:
            fine, coarse = (t, t0) if rel.group(2) == "4" else (t0, t)
            why = _refines(fine, coarse)
            if why:
                return Verdict(False, f"{name}: {why}")
            continue
        if name not in COVARIANT:
            return Verdict(False, f"{name}: no covariance rule for this transform")
        exp = _canon(COVARIANT[name](t0))
        act = _canon(t)
        if exp.shape != act.shape:
            return Verdict(False, f"{name}: {len(act)} components, expected {len(exp)} (as on the original image)")
        bad = np.nonzero(np.any(exp[:, :5] != act[:, :5], axis=1))[0]
        if bad.size:
            return Verdict(False, f"{name}: expected the original's {_fmt(exp[bad[0]])} to move to this place, "
                                  f"found {_fmt(act[bad[0]])}")
        moved = float(np.max(np.abs(exp[:, 5:] - act[:, 5:]))) if len(exp) else 0.0
        if not moved <= META_TOL:  # written so that a NaN centroid fails
            return Verdict(False, f"{name}: centroids moved {moved:.3g} px away from the covariant position "
                                  f"(allowed {META_TOL:g})")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(mask, connectivity):
    m = mask.astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    stats, cents = stats[1:], cents[1:]
    order = np.lexsort((stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP], -stats[:, cv2.CC_STAT_AREA]))
    stats, cents = stats[order], cents[order]
    cols = [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
    return {"count": len(stats),
            "areas": stats[:, cv2.CC_STAT_AREA].astype(np.int64),
            "boxes": stats[:, cols].astype(np.int64),
            "centroids": cents.astype(np.float64)}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_SLICE = "    stats, cents = stats[1:], cents[1:]\n"
_LEXSORT = "np.lexsort((stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP], -stats[:, cv2.CC_STAT_AREA]))"

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "bool_direct": (
        "Passes the bool mask straight to connectedComponentsWithStats ('5.0 accepts bools now'). The binding "
        "accepts it; the algorithm asserts uint8. Raises cv2.error on 5.0.0.",
        _mutate(("m = mask.astype(np.uint8)", "m = mask"))),
    "include_background": (
        "Keeps label 0, the background, as a component.",
        _mutate((_SLICE, "    stats, cents = stats, cents\n"))),
    "connectivity_hardcoded_8": (
        "Ignores the connectivity argument (OpenCV's default is 8). Right on every 8-scene, wrong on 4-scenes.",
        _mutate(("connectivity=connectivity", "connectivity=8"))),
    "connectivity_swapped": (
        "4 and 8 mixed up: asks for 4 when 8 is wanted and the reverse.",
        _mutate(("connectivity=connectivity", "connectivity=12 - connectivity"))),
    "centroid_row_col": (
        "Returns centroids as (row, col) instead of (x, y).",
        _mutate(("cents.astype(np.float64)", "cents[:, ::-1].astype(np.float64)"))),
    "box_wh_swapped": (
        "CC_STAT_WIDTH and CC_STAT_HEIGHT columns swapped: boxes are (x, y, h, w).",
        _mutate(("cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]", "cv2.CC_STAT_HEIGHT, cv2.CC_STAT_WIDTH]"))),
    "drop_small_noise": (
        "'Noise removal': drops components under 5 pixels. The task wants every component.",
        _mutate((_SLICE, _SLICE + "    keep = stats[:, cv2.CC_STAT_AREA] >= 5\n"
                                  "    stats, cents = stats[keep], cents[keep]\n"))),
    "label_order_ties": (
        "Sorts by area with a stable sort over the raw labels. Ties then follow OpenCV's label order, which at "
        "connectivity 8 is not (y, x).",
        _mutate((_LEXSORT, 'np.argsort(-stats[:, cv2.CC_STAT_AREA], kind="stable")'))),
}


# ---------------------------------------------------------------- calibration

def _gauge(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, list[Component], list[Component]]:
    """A single-object scene at a random position of the full canvas: (mask, truth at 4, truth at 8)."""
    obj = BUILDERS[kind](rng)
    h, w = obj.pixels.shape
    ox, oy = _ints(rng, 0, W0 - w), _ints(rng, 0, H0 - h)
    mask = np.zeros(SIZE, bool)
    mask[oy:oy + h, ox:ox + w] = obj.pixels
    p = _placed(obj, ox, oy)
    return mask, sorted(p.comps4, key=lambda c: c.order_key), sorted(p.comps8, key=lambda c: c.order_key)


def calibrate(n: int = 1200, seed: int = 12345) -> dict:
    """Measure the reference's error against the manufactured truth on single-object gauge blocks.

    Reports the largest error in area, box and centroid for cv2.connectedComponentsWithStats (the reference),
    for the same centroid computed independently with cv2.moments, and for the centroid cast to float32.
    Also whether the seven CCL algorithms agree with each other as sets.
    """
    rng = np.random.default_rng(seed)
    worst = {"area": 0, "box": 0, "centroid_ccl": 0.0, "centroid_moments": 0.0, "centroid_float32": 0.0}
    disagree, n_comps = 0, 0
    algorithms = (cv2.CCL_DEFAULT, cv2.CCL_WU, cv2.CCL_GRANA, cv2.CCL_BOLELLI, cv2.CCL_SAUF, cv2.CCL_BBDT,
                  cv2.CCL_SPAGHETTI)
    for i in range(n):
        mask, t4, t8 = _gauge(KINDS[i % len(KINDS)], rng)
        u8 = mask.astype(np.uint8)
        for conn, truth in ((4, t4), (8, t8)):
            want = _truth_table(truth)
            got = _out_table(reference_solve(mask, conn))
            assert len(got) == len(want), (i, conn, len(got), len(want))
            n_comps += len(want)
            worst["area"] = max(worst["area"], int(np.max(np.abs(got[:, 0] - want[:, 0]), initial=0)))
            worst["box"] = max(worst["box"], int(np.max(np.abs(got[:, 1:5] - want[:, 1:5]), initial=0)))
            worst["centroid_ccl"] = max(worst["centroid_ccl"],
                                        float(np.max(np.abs(got[:, 5:] - want[:, 5:]), initial=0)))
            f32 = got[:, 5:].astype(np.float32).astype(np.float64)
            worst["centroid_float32"] = max(worst["centroid_float32"],
                                            float(np.max(np.abs(f32 - want[:, 5:]), initial=0)))
            # an independent centroid estimator: cv2.moments of each label's own mask
            n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=conn)
            by_key = {(int(r[0]), int(r[1]), int(r[2])): r for r in want}
            for k in range(1, n_lab):
                m = cv2.moments((labels == k).astype(np.uint8), binaryImage=True)
                s = stats[k]
                w = by_key[(int(s[cv2.CC_STAT_AREA]), int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]))]
                worst["centroid_moments"] = max(worst["centroid_moments"],
                                                abs(m["m10"] / m["m00"] - w[5]), abs(m["m01"] / m["m00"] - w[6]))
            sets = []
            for alg in algorithms:
                r = cv2.connectedComponentsWithStatsWithAlgorithm(u8, conn, cv2.CV_32S, alg)
                sets.append(sorted(map(tuple, np.column_stack([r[2], r[3]])[1:].tolist())))
            disagree += sum(s != sets[0] for s in sets)
    return {"gauge_blocks": n, "components": n_comps, "algorithm_disagreements": disagree, **worst}


def measure_metamorphic_drift(n: int = 100, seed: int = 50_000) -> dict:
    """Largest departure of the reference from the covariant map under the three geometric symmetries."""
    ints, cent, refine = 0, 0.0, 0
    for i in range(n):
        s = make_scene(seed + i)
        t0 = _truth_table(sorted(s.truth, key=lambda c: c.order_key))
        for name, var in transforms(s).items():
            t = _out_table(reference_solve(var["mask"], var["connectivity"]))
            if name in COVARIANT:
                exp, act = _canon(COVARIANT[name](t0)), _canon(t)
                if exp.shape != act.shape:
                    return {"int_mismatches": float("inf"), "centroid_drift": float("inf"), "refinement_failures": -1}
                ints += int(np.any(exp[:, :5] != act[:, :5], axis=1).sum())
                cent = max(cent, float(np.max(np.abs(exp[:, 5:] - act[:, 5:]))))
            else:
                base = _out_table(reference_solve(s.mask, s.connectivity))
                fine, coarse = (t, base) if var["connectivity"] == 4 else (base, t)
                refine += bool(_refines(fine, coarse))
    return {"int_mismatches": ints, "centroid_drift": cent, "refinement_failures": refine}


def measure_bool_mask_support() -> dict[str, str]:
    """Which OpenCV functions accept a numpy bool array on this build (the 5.0 'bool masks' claim, itemised).

    "accepted" means only that no cv2.error was raised, not that the result is right. ``moments`` is probed with
    ``binaryImage=True``: without it, a bool array raises. ``bitwise_not`` is "accepted" but returns bytes 0xFE
    (True) and 0xFF (False), so every element reads True: a silent wrong answer (see the module docstring)."""
    m = np.zeros((8, 8), bool)
    m[2:5, 2:5] = True
    probes = {
        "countNonZero": lambda: cv2.countNonZero(m),
        "moments": lambda: cv2.moments(m, binaryImage=True),
        "bitwise_not": lambda: cv2.bitwise_not(m),
        "connectedComponentsWithStats": lambda: cv2.connectedComponentsWithStats(m, connectivity=8),
        "connectedComponents": lambda: cv2.connectedComponents(m, connectivity=8),
        "findContours": lambda: cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
        "threshold": lambda: cv2.threshold(m, 0, 255, cv2.THRESH_BINARY),
        "dilate": lambda: cv2.dilate(m, np.ones((3, 3), np.uint8)),
    }
    out = {}
    for name, fn in probes.items():
        try:
            fn()
            out[name] = "accepted"
        except cv2.error:
            out[name] = "raises cv2.error"
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=1))
    print("metamorphic drift of the reference:", measure_metamorphic_drift())
    print("bool mask support:", json.dumps(measure_bool_mask_support(), indent=1))
