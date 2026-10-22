"""Drills on NumPy 2: the names and rules that old OpenCV tutorials lean on and that a bare ``pip install`` of the 5.0 wheel
(which pulls NumPy 2.x) no longer has. Measured on numpy 2.4.6 in the venv the tests run in.

  ``np.int0``                 removed (the ``np.int0(cv2.boxPoints(rect))`` idiom of every minAreaRect tutorial): np.intp
  ``np.Inf``                  removed in 2.0: np.inf
  ``ndarray.ptp()``           the method is removed (np.ptp(a) stays): max - min
  uint8 array * Python int    NEP 50: a Python int that does not fit the array's dtype raises OverflowError
                              (``img[..., 1] * 587`` on uint8 used to be upcast silently); uint8 * 255 still wraps silently

None of these is an OpenCV 5.0 change, so ``depends_on_5_0`` is False and the probes are marked ``new:``; the group says
what they are. A bare install of OpenCV 5.0 is what makes them matter.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, spec

GROUP = "numpy2"
R = Registry()
S = SEED_BASE + 4_000  # this module's seeds: 65_000 .. 65_999


# ---------------------------------------------------------------- 1. np.int0: outline of the rotated bounding box

def plank_scene(seed: int):
    """A dark BGR image with one bright rotated rectangle (the plank) and a smaller bright block. Truth: the plank's rotated
    rectangle corners (floats)."""
    rng = np.random.default_rng(seed)
    H, W = 150, 200
    img = np.full((H, W, 3), 25, np.uint8)
    cx, cy = rng.uniform(85, 115), rng.uniform(60, 90)
    L, Wd, ang = rng.uniform(70, 100), rng.uniform(26, 38), rng.uniform(15, 75) * rng.choice([-1, 1])
    d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    n = np.array([-d[1], d[0]])
    corners = np.array([[cx, cy] + sx * d * L / 2 + sy * n * Wd / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    cv2.fillPoly(img, [np.round(corners).astype(np.int32)], (200, 200, 200))
    cv2.rectangle(img, (6, 6), (6 + int(rng.integers(14, 20)), 6 + int(rng.integers(10, 16))), (200, 200, 200), -1)
    img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    return img, corners


GREEN = (0, 255, 0)


def _green(img: np.ndarray) -> np.ndarray:
    return (img[..., 0] == 0) & (img[..., 1] == 255) & (img[..., 2] == 0)


# calibrated on 300 planks x {rounded, truncated, intp corners} x {cv2.polylines, cv2.drawContours}: the minAreaRect of the
# rasterised plank has its corners within 1.76 px of the drawn corners, and all 1800 outlines pass (stray 0 %, so the 5 % cap
# is slack; the axis-aligned rectangle strays on 85 % of its pixels).
BOX_TUBE, BOX_MIN_COVER, BOX_MAX_STRAY = np.ones((5, 5), np.uint8), 0.85, 0.05


def _bx_cases():
    out = []
    for seed in (S + 1, S + 2, S + 3, S + 4):
        img, corners = plank_scene(seed)
        truth = np.zeros(img.shape[:2], np.uint8)
        cv2.polylines(truth, [np.round(corners).astype(np.int32)], True, 255, 2)
        out.append(Case({"img": img}, truth))
    return out


def _bx_verify(case: Case, out: dict) -> Verdict:
    got, src = arr(out, "img"), case.inputs["img"]
    if got.shape != src.shape:
        return fail(f"img has shape {got.shape}")
    g = _green(got)
    if (got[~g] != src[~g]).any():
        return fail("pixels other than the green outline were changed")
    if not g.any():
        return fail("no green outline drawn")
    near_truth = cv2.dilate(case.truth, BOX_TUBE) > 0
    near_drawn = cv2.dilate(g.astype(np.uint8) * 255, BOX_TUBE) > 0
    stray = (g & ~near_truth).sum() / g.sum()
    cover = (near_drawn & (case.truth > 0)).sum() / (case.truth > 0).sum()
    if stray > BOX_MAX_STRAY:
        return fail(f"{stray:.0%} of the green pixels are not on the rotated rectangle around the larger blob")
    if cover < BOX_MIN_COVER:
        return fail(f"only {cover:.0%} of the rotated rectangle's outline is drawn")
    return PASS


_BX_FIND = ('    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
            '    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)\n'
            '    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
            '    biggest = max(contours, key=cv2.contourArea)\n')
R.add(Check(
    id="drill_rotated_box_draw", group=GROUP, probes=("new:numpy2_int0_removed",), depends_on_5_0=False, role="gotcha",
    gotcha="np.int0 (the np.int0(cv2.boxPoints(rect)) idiom) was removed in NumPy 2; boxPoints returns float32, which "
           "polylines/drawContours reject; np.intp or .astype(int) is the way",
    spec=spec("Find the largest bright object in the BGR image and draw the outline of its minimum-area rotated rectangle "
              "on a copy of the image, in pure green (0, 255, 0), 2 px thick. Return the copy.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image: bright blocks on a dark ground, one clearly larger and tilted")},
              {"img": IOSpec("uint8", "HxWx3", 3, "the image with the green outline; nothing else changed")}),
    make_cases=_bx_cases, verify=_bx_verify,
    reference=code("img", _BX_FIND + '    box = cv2.boxPoints(cv2.minAreaRect(biggest)).astype(np.intp)\n'
                                     '    out = img.copy()\n    cv2.polylines(out, [box], True, (0, 255, 0), 2)\n    return {"img": out}\n'),
    wrong={
        "np_int0": Wrong("4.x tutorial idiom: box = np.int0(cv2.boxPoints(rect))",
                         code("img", _BX_FIND + '    box = np.int0(cv2.boxPoints(cv2.minAreaRect(biggest)))\n'
                                                '    out = img.copy()\n    cv2.drawContours(out, [box], 0, (0, 255, 0), 2)\n'
                                                '    return {"img": out}\n'), "AttributeError", "int0"),
        "float_box": Wrong("draws with the float32 corner array boxPoints returns",
                           code("img", _BX_FIND + '    box = cv2.boxPoints(cv2.minAreaRect(biggest))\n'
                                                  '    out = img.copy()\n    cv2.polylines(out, [box], True, (0, 255, 0), 2)\n'
                                                  '    return {"img": out}\n'), "error", "CV_32S"),
        "axis_aligned": Wrong("cv2.boundingRect: an axis-aligned rectangle instead of the rotated one",
                              code("img", _BX_FIND + '    x, y, w, h = cv2.boundingRect(biggest)\n'
                                                     '    out = img.copy()\n    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)\n'
                                                     '    return {"img": out}\n'), "verifier"),
    }),
    "Draw the tilted bounding rectangle (minimum area) of the biggest bright block in green, 2 px wide, on a copy of "
    "the BGR image and return it.",
    "Take the largest bright shape in the image, compute its minimum-area rotated rectangle and outline it with a "
    "2-pixel pure-green (0, 255, 0) line on a copy of the picture.")


# ---------------------------------------------------------------- 2. ndarray.ptp is gone

def _cs_cases():
    out = []
    for seed in (S + 11, S + 12, S + 13):
        rng = np.random.default_rng(seed)
        lo = rng.integers(0, 90, 3)
        hi = lo + rng.integers(40, 160, 3)
        img = np.stack([rng.integers(l, h + 1, (20, 30)) for l, h in zip(lo, hi)], axis=2).astype(np.uint8)
        span = img.max(axis=(0, 1)).astype(np.int64) - img.min(axis=(0, 1))
        stretched = np.round((img.astype(np.float64) - img.min(axis=(0, 1))) * 255.0 / span)
        out.append(Case({"img": img}, (stretched.astype(np.uint8), span)))
    return out


def _cs_verify(case: Case, out: dict) -> Verdict:
    stretched, span = case.truth
    if not np.array_equal(arr(out, "spans"), span):
        return fail(f"spans {arr(out, 'spans').tolist()} vs {span.tolist()} (max - min of B, G, R)")
    got = arr(out, "stretched")
    if got.shape != stretched.shape:
        return fail(f"stretched has shape {got.shape}")
    err = np.abs(got.astype(np.int16) - stretched.astype(np.int16)).max()
    return fail(f"stretched differs from the per-channel linear stretch by up to {err} levels") if err > 1 else PASS


R.add(Check(
    id="drill_channel_stretch", group=GROUP, probes=("new:numpy2_ndarray_ptp_removed",), depends_on_5_0=False, role="gotcha",
    gotcha="ndarray.ptp() was removed in NumPy 2 (np.ptp(a) and max - min remain); uint8 arithmetic wraps silently",
    spec=spec("Stretch the contrast of each colour channel separately so that its darkest value becomes 0 and its brightest "
              "255 (linear, rounded). Also report each channel's original range, max minus min.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image whose channels use different parts of the 0..255 range")},
              {"stretched": IOSpec("uint8", "HxWx3", 3, "every channel mapped linearly onto 0..255"),
               "spans": IOSpec("int", "3", 1, "max - min of the B, G and R channels of the input")}),
    make_cases=_cs_cases, verify=_cs_verify,
    reference=code("img", '''\
    lo = img.min(axis=(0, 1)).astype(np.float64)
    spans = np.ptp(img, axis=(0, 1)).astype(np.int64)
    stretched = np.round((img.astype(np.float64) - lo) * 255.0 / spans)
    return {"stretched": stretched.astype(np.uint8), "spans": spans}
'''),
    wrong={
        "ndarray_ptp": Wrong("4.x: img.ptp(axis=(0, 1)) as a method",
                             code("img", '    lo = img.min(axis=(0, 1)).astype(np.float64)\n    spans = img.ptp(axis=(0, 1)).astype(np.int64)\n'
                                         '    stretched = np.round((img.astype(np.float64) - lo) * 255.0 / spans)\n'
                                         '    return {"stretched": stretched.astype(np.uint8), "spans": spans}\n'),
                             "AttributeError", "ptp"),
        "uint8_wrap": Wrong("(img - lo) * 255 stays uint8: 255 fits, so the product wraps silently",
                            code("img", '    lo = img.min(axis=(0, 1))\n    spans = img.max(axis=(0, 1)).astype(np.int64) - lo\n'
                                        '    stretched = (img - lo) * 255 // spans.astype(np.uint8)\n'
                                        '    return {"stretched": stretched, "spans": spans}\n'), "verifier"),
        "one_range_for_all": Wrong("one min and one max over all three channels",
                                   code("img", '    lo, hi = float(img.min()), float(img.max())\n'
                                               '    stretched = np.round((img.astype(np.float64) - lo) * 255.0 / (hi - lo))\n'
                                               '    return {"stretched": stretched.astype(np.uint8),\n'
                                               '            "spans": img.max(axis=(0, 1)).astype(np.int64) - img.min(axis=(0, 1))}\n'),
                                   "verifier"),
    }),
    "Normalise each BGR channel on its own so that its minimum maps to 0 and its maximum to 255 (linear, rounded), and "
    "also return the dynamic range (max minus min) each channel had before.",
    "Per-channel contrast stretching: rescale B, G and R independently to the full 0..255 range, and report the "
    "original peak-to-peak range of each channel.")


# ---------------------------------------------------------------- 3. np.Inf is gone

# calibrated on 40 draws of the check's cases x 3 distance-transform settings: cv2.distanceTransform (DIST_L2) of the inverted
# mask, read at the query pixels, differs from the exact distance by up to 0.0 (DIST_MASK_PRECISE), 1.9 % (mask size 5) and
# 4.5 % (mask size 3) relative. 6 % + 0.15 px accepts all three.
NEAR_REL, NEAR_ABS = 0.06, 0.15


def _nf_cases():
    out = []
    for seed in (S + 21, S + 22, S + 23, S + 24):
        rng = np.random.default_rng(seed)
        mask = np.zeros((50, 70), np.uint8)
        if seed != S + 24:  # the last case has an empty mask
            for _ in range(int(rng.integers(1, 4))):
                cv2.circle(mask, (int(rng.integers(8, 62)), int(rng.integers(8, 42))), int(rng.integers(2, 6)), 255, -1)
        pts = np.c_[rng.integers(0, 70, 8), rng.integers(0, 50, 8)].astype(np.int32)
        ys, xs = np.nonzero(mask)
        if len(xs):
            dist = np.hypot(pts[:, 0, None] - xs[None, :], pts[:, 1, None] - ys[None, :]).min(axis=1)
        else:
            dist = np.full(len(pts), np.inf)
        out.append(Case({"mask": mask, "points": pts}, dist))
    return out


def _nf_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "dist").astype(float)
    if got.shape != case.truth.shape:
        return fail(f"dist has shape {got.shape}")
    inf_gt = np.isinf(case.truth)
    if not np.array_equal(np.isinf(got), inf_gt):
        return fail("distance to an empty mask should be inf, and only then")
    ok = np.abs(got[~inf_gt] - case.truth[~inf_gt]) <= NEAR_REL * case.truth[~inf_gt] + NEAR_ABS
    return PASS if ok.all() else fail(f"distances {got.round(2).tolist()} vs {case.truth.round(2).tolist()}")


_NF_SIG = "mask, points"
R.add(Check(
    id="drill_nearest_foreground", group=GROUP, probes=("new:numpy2_Inf_removed",), depends_on_5_0=False, role="gotcha",
    gotcha="np.Inf was removed in NumPy 2 (np.inf stays); an empty mask has no nearest pixel and the distance transform "
           "then returns a large finite number, not infinity",
    spec=spec("For each query point, report the distance in pixels to the nearest foreground pixel of the mask (pixel "
              "centres, Euclidean). If the mask has no foreground pixel at all, every distance is infinity.",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255 mask; may be empty"),
               "points": IOSpec("int32", "Mx2", 2, "(x, y) query points inside the image")},
              {"dist": IOSpec("float", "M", 1, "distance from each point to the nearest foreground pixel, inf for an empty mask")}),
    make_cases=_nf_cases, verify=_nf_verify,
    reference=code(_NF_SIG, '''\
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"dist": np.full(len(points), np.inf)}
    d = np.hypot(points[:, 0, None] - xs[None, :], points[:, 1, None] - ys[None, :])
    return {"dist": d.min(axis=1)}
'''),
    wrong={
        "np_Inf": Wrong("4.x spelling of infinity: np.Inf",
                        code(_NF_SIG, '    ys, xs = np.nonzero(mask)\n    dist = np.full(len(points), np.Inf)\n'
                                      '    for i, (x, y) in enumerate(points):\n'
                                      '        if len(xs):\n            dist[i] = np.hypot(x - xs, y - ys).min()\n'
                                      '    return {"dist": dist}\n'), "AttributeError", "np.Inf"),
        "distance_transform_only": Wrong("distance transform of the inverted mask with no empty-mask case: the result is a large "
                                         "finite number, not inf",
                                         code(_NF_SIG, '    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)\n'
                                                       '    return {"dist": dt[points[:, 1], points[:, 0]].astype(np.float64)}\n'),
                                         "verifier"),
    }),
    "For every query point find the Euclidean distance (pixels) to the closest foreground pixel in the mask; a mask with "
    "no foreground gives infinity for all points.",
    "Given a 0/255 mask and some (x, y) points, compute each point's distance to the nearest non-zero pixel. If the "
    "mask is completely empty the answer for every point is inf.")


# ---------------------------------------------------------------- 4. uint8 * big Python int raises OverflowError

def _il_cases():
    out = []
    for seed in (S + 31, S + 32, S + 33):
        img = np.random.default_rng(seed).integers(0, 256, (16, 24, 3), dtype=np.uint8)
        b, g, r = (img[..., i].astype(np.int64) for i in range(3))
        out.append(Case({"img": img}, ((114 * b + 587 * g + 299 * r) // 1000).astype(np.uint8)))
    return out


def _il_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "luma")
    if got.shape != case.truth.shape:
        return fail(f"luma has shape {got.shape}")
    diff = int((got != case.truth).sum())
    return fail(f"{diff} of {got.size} luma values differ from (114 B + 587 G + 299 R) // 1000") if diff else PASS


R.add(Check(
    id="drill_int_luma", group=GROUP, probes=("new:numpy2_nep50_uint8_overflow",), depends_on_5_0=False, role="gotcha",
    gotcha="NEP 50 (NumPy 2): uint8_array * 587 raises OverflowError (a Python int that does not fit the array dtype); under "
           "NumPy 1 it was upcast; staying in uint16 wraps silently above 65535",
    spec=spec("Compute an integer luma image with fixed-point weights: luma = (114 * B + 587 * G + 299 * R) // 1000, from "
              "a BGR uint8 image, using integer arithmetic only.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image")},
              {"luma": IOSpec("uint8", "HxW", 2, "(114 * B + 587 * G + 299 * R) // 1000")}),
    make_cases=_il_cases, verify=_il_verify,
    reference=code("img", '''\
    b, g, r = (img[..., i].astype(np.int32) for i in range(3))
    return {"luma": ((114 * b + 587 * g + 299 * r) // 1000).astype(np.uint8)}
'''),
    wrong={
        "uint8_times_int": Wrong("img[..., 1] * 587 on the uint8 array (fine under NumPy 1, OverflowError under NumPy 2)",
                                 code("img", '    luma = (img[..., 0] * 114 + img[..., 1] * 587 + img[..., 2] * 299) // 1000\n'
                                             '    return {"luma": luma.astype(np.uint8)}\n'), "OverflowError", "out of bounds for uint8"),
        "uint16_wrap": Wrong("widens to uint16 only: 255 * 587 does not fit and wraps silently",
                             code("img", '    b, g, r = (img[..., i].astype(np.uint16) for i in range(3))\n'
                                         '    return {"luma": ((b * 114 + g * 587 + r * 299) // 1000).astype(np.uint8)}\n'), "verifier"),
    }),
    "Convert the BGR image to a grayscale image with integer arithmetic: gray = (114 * B + 587 * G + 299 * R) // 1000.",
    "Fixed-point luma: for each pixel of the BGR uint8 image compute (114 * blue + 587 * green + 299 * red) // 1000 "
    "exactly and return the uint8 result.")


CHECKS = R.checks
PHRASINGS = R.phrasings
