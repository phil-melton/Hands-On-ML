"""Drills on element types: which dtype each cv2 function takes, which new dtype 5.0 accepts where, and which it still rejects.

5.0 subset (PLAN.md section 0, item 1, with its 2026-10-17 correction that the new dtypes are accepted per function):
  bool     accepted by ``countNonZero``, ``mask=`` arguments, ``moments(binaryImage=True)``, INTER_NEAREST ``resize``;
           rejected by ``findContours``, ``dilate``/``erode``, ``threshold``, the blurs, ``cvtColor``; ``bitwise_not`` runs
           but returns bytes 0xFE/0xFF, i.e. every element True. A mask handed to ``cv2.mean(mask=...)`` keeps its meaning.
  uint32   ``minMaxLoc`` is exact (a float32 cast loses the low bits above 2**24, an int32 cast wraps above 2**31)
  float16  ``resize`` still rejects it (so do GaussianBlur, Sobel, ...): go through float32 and cast back

Timeless (same on 4.14 and 5.0):
  Canny wants uint8, findContours wants a single-channel uint8 image, cvtColor wants uint8/uint16/float32 (not float64)
  and the channel order you name, threshold with THRESH_OTSU wants one channel and returns (value, image).

The wrong variants reproduce the 4.x idiom (``mask.astype(np.uint8)`` then ``bitwise_not``, a float32 cast) or the
over-generalisation "5.0 takes bool, so it takes bool everywhere".
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, mask_iou, spec

GROUP = "dtypes"
R = Registry()
S = SEED_BASE + 1_000  # this module's seeds: 62_000 .. 62_999


def blobs(seed: int, shape=(48, 64), k: int = 3, smooth: float = 3.0, level: float = 0.55) -> np.ndarray:
    """Boolean mask of a few smooth blobs (a thresholded blurred random field)."""
    rng = np.random.default_rng(seed)
    f = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), smooth)
    f = (f - f.min()) / (f.max() - f.min())
    return f > np.quantile(f, level)


# ---------------------------------------------------------------- 1. bool mask in cv2.mean(mask=...), and its complement

def _bm_cases():
    out = []
    for seed in (S + 1, S + 2, S + 3):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (36, 48, 3), dtype=np.uint8)
        mask = blobs(seed, (36, 48), smooth=2.5, level=0.6)
        out.append(Case({"img": img, "mask": mask},
                        (img[mask].mean(axis=0), img[~mask].mean(axis=0))))
    return out


def _bm_verify(case: Case, out: dict) -> Verdict:
    fg_gt, bg_gt = case.truth
    fg, bg = arr(out, "fg_mean").astype(float), arr(out, "bg_mean").astype(float)
    if fg.shape != (3,) or np.abs(fg - fg_gt).max() > 1e-3:
        return fail(f"fg_mean {fg.round(2).tolist()} vs {fg_gt.round(2).tolist()} (BGR)")
    if bg.shape != (3,) or np.abs(bg - bg_gt).max() > 1e-3:
        return fail(f"bg_mean {bg.round(2).tolist()} vs {bg_gt.round(2).tolist()} (BGR): "
                    "it should average only the pixels outside the mask")
    return PASS


_BM_HEAD = code("img, mask", "")
R.add(Check(
    id="drill_bool_background_mean", group=GROUP, probes=("bool_array_input", "bitwise_and_bool_mask", "new:bitwise_not_bool"),
    depends_on_5_0=True, role="gotcha",
    gotcha="5.0 takes a bool array as cv2.mean(mask=...), but cv2.bitwise_not(bool_mask) returns bytes 0xFE/0xFF, so the "
           "'inverse mask' selects every pixel",
    spec=spec("Report the mean BGR colour of the object (the pixels where the mask is True) and the mean BGR colour of the "
              "background (the pixels where it is False).",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
               "mask": IOSpec("bool", "HxW", 2, "True on the object, False on the background; both are non-empty")},
              {"fg_mean": IOSpec("float", "3", 1, "mean (B, G, R) over the object pixels"),
               "bg_mean": IOSpec("float", "3", 1, "mean (B, G, R) over the background pixels")}),
    make_cases=_bm_cases, verify=_bm_verify,
    reference=code("img, mask", '    fg = cv2.mean(img, mask=mask)[:3]\n    bg = cv2.mean(img, mask=np.logical_not(mask))[:3]\n'
                                '    return {"fg_mean": np.array(fg), "bg_mean": np.array(bg)}\n'),
    note="the reference hands the bool mask to cv2.mean directly (4.14 raises there); only bitwise_not is avoided",
    wrong={
        "bitwise_not_bool": Wrong("bg mask = cv2.bitwise_not(mask) on the bool mask: every byte is 0xFE/0xFF, so all True",
                                  _BM_HEAD + '    fg = cv2.mean(img, mask=mask)[:3]\n'
                                             '    bg = cv2.mean(img, mask=cv2.bitwise_not(mask))[:3]\n'
                                             '    return {"fg_mean": np.array(fg), "bg_mean": np.array(bg)}\n', "verifier"),
        "bitwise_not_01": Wrong("4.x habit: 0/1 uint8 mask, then cv2.bitwise_not -> 254/255, still every pixel", _BM_HEAD +
                                '    m = mask.astype(np.uint8)\n    fg = cv2.mean(img, mask=m)[:3]\n'
                                '    bg = cv2.mean(img, mask=cv2.bitwise_not(m))[:3]\n'
                                '    return {"fg_mean": np.array(fg), "bg_mean": np.array(bg)}\n', "verifier"),
    }),
    "A segmentation mask marks an object (True) against its background (False). Compute the average BGR colour inside "
    "the object and the average BGR colour of everything else.",
    "Given a BGR image and a boolean object mask, return two (B, G, R) means: one over the masked object pixels and one "
    "over the unmasked background pixels.")


# ---------------------------------------------------------------- 2. dilate is not available for bool

def _gr_cases():
    out = []
    for seed, r in ((S + 11, 3), (S + 12, 5), (S + 13, 2)):
        m = blobs(seed, (40, 56), smooth=2.0, level=0.9)
        grown = cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        out.append(Case({"mask": m, "radius": r}, grown))
    return out


def _gr_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "grown")
    if got.shape != case.truth.shape:
        return fail(f"grown has shape {got.shape}")
    diff = int((got.astype(bool) != case.truth).sum())
    return fail(f"{diff} pixels differ from the (2 * radius + 1) square dilation of the mask") if diff else PASS


R.add(Check(
    id="drill_bool_grow", group=GROUP, probes=("bool_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="COUNTER-CASE: 5.0 still rejects a bool mask in dilate/erode/morphologyEx (-213 Unsupported data type); it needs "
           ".astype(np.uint8) and the result needs .astype(bool)",
    spec=spec("Grow the object: set to True every pixel that lies within `radius` pixels of a True pixel in x and in y (a "
              "square neighbourhood of side 2 * radius + 1), and return the grown boolean mask.",
              {"mask": IOSpec("bool", "HxW", 2, "True on the object"),
               "radius": IOSpec("int", "scalar", None, "px")},
              {"grown": IOSpec("bool", "HxW", 2, "True within radius px (square neighbourhood) of the object, "
                                                 "including the object")}),
    make_cases=_gr_cases, verify=_gr_verify,
    reference=code("mask, radius", '    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)\n'
                                   '    grown = cv2.dilate(mask.astype(np.uint8), k)\n    return {"grown": grown.astype(bool)}\n'),
    wrong={
        "dilate_bool": Wrong("over-generalizes 'bool is accepted': cv2.dilate(mask, kernel) on the bool array",
                             code("mask, radius", '    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)\n'
                                                  '    return {"grown": cv2.dilate(mask, k)}\n'),
                             "error", "Unsupported data type"),
        "returns_uint8": Wrong("dilates the uint8 copy but returns it as 0/1 uint8 instead of a bool mask",
                               code("mask, radius", '    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)\n'
                                                    '    return {"grown": cv2.dilate(mask.astype(np.uint8), k)}\n'), "verifier"),
        "radius_as_size": Wrong("kernel of side `radius` instead of 2 * radius + 1",
                                code("mask, radius", '    k = np.ones((radius, radius), np.uint8)\n'
                                                     '    return {"grown": cv2.dilate(mask.astype(np.uint8), k).astype(bool)}\n'),
                                "verifier"),
    }),
    "Dilate the boolean object mask by `radius` pixels: a pixel becomes True when any pixel within `radius` in x and y "
    "(a (2 * radius + 1) square) is True. Return a boolean mask of the same size.",
    "Expand a boolean mask outward by `radius` pixels in every direction, using a square neighbourhood of side "
    "2 * radius + 1. Return a boolean array the same shape as the input.")


# ---------------------------------------------------------------- 3. findContours is not available for bool

def lump_scene(seed: int):
    """Two solid blobs (a rectangle and an ellipse) and one thin bar whose outline is longer than the biggest blob's.
    Truth: (x, y, w, h) of the blob with the largest area."""
    rng = np.random.default_rng(seed)
    H, W = 90, 180
    m = np.zeros((H, W), np.uint8)
    big_w, big_h = int(rng.integers(26, 38)), int(rng.integers(26, 38))
    bx, by = int(rng.integers(6, 30)), int(rng.integers(6, H - big_h - 6))
    cv2.rectangle(m, (bx, by), (bx + big_w - 1, by + big_h - 1), 1, -1)
    bar_len = big_w + big_h + 6  # outline 2 * (bar_len + 3) exceeds 2 * (w + h); area 3 * bar_len stays far below w * h
    x0 = bx + big_w + 8 + int(rng.integers(0, 4))
    cv2.rectangle(m, (x0, H - 12), (x0 + bar_len - 1, H - 10), 1, -1)
    ex, ey = int(rng.integers(W - 36, W - 22)), int(rng.integers(16, 40))
    cv2.ellipse(m, (ex, ey), (int(rng.integers(8, 12)), int(rng.integers(8, 12))), 0, 0, 360, 1, -1)
    return m.astype(bool), (bx, by, big_w, big_h)


def _lb_cases():
    return [Case({"mask": m}, box) for m, box in (lump_scene(s) for s in (S + 21, S + 22, S + 23, S + 24))]


def _lb_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "box")
    if got.shape != (4,) or tuple(int(v) for v in got) != case.truth:
        return fail(f"box {got.tolist()} vs {list(case.truth)}: it should enclose the blob with the largest area")
    return PASS


_LB_CTRS = ('contours = cv2.findContours(%s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n')
R.add(Check(
    id="drill_bool_blob_box", group=GROUP, probes=("bool_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="COUNTER-CASE: 5.0 still rejects a bool mask in findContours (-210 Unsupported format); it needs .astype(np.uint8)",
    spec=spec("Several solid blobs and one thin long bar are marked True in the boolean mask. Find the blob with the "
              "largest area (not the longest outline) and return its bounding box.",
              {"mask": IOSpec("bool", "HxW", 2, "True on the blobs; they do not touch each other or the border")},
              {"box": IOSpec("int", "4", 1, "x, y, width, height of the bounding box of the largest-area blob")}),
    make_cases=_lb_cases, verify=_lb_verify,
    reference=code("mask", '    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                           '    biggest = max(contours, key=cv2.contourArea)\n'
                           '    return {"box": np.array(cv2.boundingRect(biggest))}\n'),
    wrong={
        "findcontours_bool": Wrong("over-generalizes 'bool is accepted': findContours on the bool mask",
                                   code("mask", '    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                                '    biggest = max(contours, key=cv2.contourArea)\n'
                                                '    return {"box": np.array(cv2.boundingRect(biggest))}\n'),
                                   "error", "Unsupported format"),
        "bounding_rect_bool": Wrong("over-generalizes: cv2.boundingRect on the bool mask",
                                    code("mask", '    return {"box": np.array(cv2.boundingRect(mask))}\n'),
                                    "error", "npoints >= 0"),
        "longest_outline": Wrong("takes the blob with the longest outline, which is the thin bar",
                                 code("mask", '    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                              '    longest = max(contours, key=lambda c: cv2.arcLength(c, True))\n'
                                              '    return {"box": np.array(cv2.boundingRect(longest))}\n'), "verifier"),
    }),
    "Locate the largest solid blob (by area, not by perimeter) in the boolean mask and give its bounding box as "
    "(x, y, width, height). A thin long bar is also present and must not be chosen.",
    "The boolean mask contains a few separate solid shapes and a thin bar. Return the bounding rectangle "
    "(x, y, w, h) of the shape that covers the most pixels.")


# ---------------------------------------------------------------- 4. moments: bool needs binaryImage=True

def _bc_cases():
    out = []
    for seed in (S + 31, S + 32, S + 33):
        m = blobs(seed, (50, 70), smooth=3.0, level=0.8)
        ys, xs = np.nonzero(m)
        out.append(Case({"mask": m}, (np.array([xs.mean(), ys.mean()]), int(m.sum()))))
    return out


def _bc_verify(case: Case, out: dict) -> Verdict:
    c_gt, a_gt = case.truth
    c = arr(out, "centroid").astype(float)
    if c.shape != (2,) or np.abs(c - c_gt).max() > 1e-6:
        return fail(f"centroid {c.round(3).tolist()} vs {c_gt.round(3).tolist()}: (x, y) of the pixel centres, mean over the True pixels")
    if out["area"] != a_gt:
        return fail(f"area {out['area']} vs {a_gt} True pixels")
    return PASS


R.add(Check(
    id="drill_bool_centroid", group=GROUP, probes=("bool_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="5.0 takes a bool array in cv2.moments only with binaryImage=True (plain moments(mask) raises -210); "
           "scaling a 0/1 mask to 0/255 makes m00 255 times the area",
    spec=spec("Find the centre of mass and the pixel area of the object in the boolean mask, using image moments. The "
              "centre of pixel (column, row) is at (column, row).",
              {"mask": IOSpec("bool", "HxW", 2, "True on the object")},
              {"centroid": IOSpec("float", "2", 1, "(x, y) mean position of the True pixels"),
               "area": IOSpec("int", "scalar", None, "number of True pixels")}),
    make_cases=_bc_cases, verify=_bc_verify,
    reference=code("mask", '    m = cv2.moments(mask, binaryImage=True)\n'
                           '    return {"centroid": np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), "area": int(round(m["m00"]))}\n'),
    wrong={
        "moments_bool": Wrong("cv2.moments(mask) on the bool mask without binaryImage",
                              code("mask", '    m = cv2.moments(mask)\n'
                                           '    return {"centroid": np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), "area": int(m["m00"])}\n'),
                              "error", "Unsupported format"),
        "scaled_area": Wrong("moments of mask * 255: the centroid is right, but m00 is 255 times the area",
                             code("mask", '    m = cv2.moments(mask.astype(np.uint8) * 255)\n'
                                         '    return {"centroid": np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), "area": int(round(m["m00"]))}\n'),
                             "verifier"),
        "swapped_axes": Wrong("centroid as (m01 / m00, m10 / m00), i.e. (row, column)",
                              code("mask", '    m = cv2.moments(mask, binaryImage=True)\n'
                                           '    return {"centroid": np.array([m["m01"] / m["m00"], m["m10"] / m["m00"]]), "area": int(round(m["m00"]))}\n'),
                              "verifier"),
    }),
    "Using cv2.moments, compute where the object in this boolean mask is centred, as (x, y), and how many pixels it covers.",
    "From the image moments of the boolean mask, report the object's centroid (x, y) in pixel-centre coordinates and its "
    "area in pixels.")


# ---------------------------------------------------------------- 5. float16: resize is rejected

# float16 has 11 significand bits: the 2x2 mean computed in float32 and rounded once to float16 is the reference; the
# reference solution reproduces it exactly in all 60 scenes measured, a float16-accumulated mean is off by up to 3 ulp.
F16_ULP = 1.5


def _fd_cases():
    out = []
    for seed in (S + 41, S + 42, S + 43):
        rng = np.random.default_rng(seed)
        f = cv2.GaussianBlur(rng.random((48, 64)).astype(np.float32), (0, 0), 2.0) * rng.uniform(2, 9)
        f16 = f.astype(np.float16)
        avg = f16.astype(np.float32).reshape(24, 2, 32, 2).mean(axis=(1, 3))
        out.append(Case({"feat": f16}, avg.astype(np.float16)))
    return out


def _fd_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "small")
    if got.shape != case.truth.shape:
        return fail(f"small has shape {got.shape}, expected {case.truth.shape}")
    ulp = np.spacing(np.abs(case.truth)).astype(np.float64)
    err = (np.abs(got.astype(np.float64) - case.truth.astype(np.float64)) / np.maximum(ulp, 1e-9)).max()
    return fail(f"small differs from the 2x2 block means by {err:.1f} float16 ulp (allowed {F16_ULP})") if err > F16_ULP else PASS


R.add(Check(
    id="drill_f16_downscale", group=GROUP, probes=("float16_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="COUNTER-CASE: 5.0 accepts float16 in cv2.add/mean but cv2.resize still rejects it (func != 0); go through "
           "float32 and cast back to float16",
    spec=spec("Halve the resolution of the half-precision feature map by averaging every 2x2 block of pixels (area "
              "interpolation). Keep the result in float16.",
              {"feat": IOSpec("float16", "HxW", 2, "feature map; H and W are even")},
              {"small": IOSpec("float16", "H/2xW/2", 2, "mean of each 2x2 block")}),
    make_cases=_fd_cases, verify=_fd_verify,
    reference=code("feat", '    h, w = feat.shape\n'
                           '    small = cv2.resize(feat.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA)\n'
                           '    return {"small": small.astype(np.float16)}\n'),
    wrong={
        "resize_float16": Wrong("over-generalizes 'float16 is accepted': cv2.resize on the float16 array",
                                code("feat", '    h, w = feat.shape\n'
                                             '    return {"small": cv2.resize(feat, (w // 2, h // 2), interpolation=cv2.INTER_AREA)}\n'),
                                "error", "func != 0"),
        "returns_float32": Wrong("goes through float32 and forgets to cast back",
                                 code("feat", '    h, w = feat.shape\n'
                                              '    return {"small": cv2.resize(feat.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA)}\n'),
                                 "verifier"),
        "dsize_swapped": Wrong("dsize = (h // 2, w // 2) instead of (w // 2, h // 2)",
                               code("feat", '    h, w = feat.shape\n'
                                            '    small = cv2.resize(feat.astype(np.float32), (h // 2, w // 2), interpolation=cv2.INTER_AREA)\n'
                                            '    return {"small": small.astype(np.float16)}\n'), "verifier"),
    }),
    "Downscale a float16 feature map by a factor of two using area averaging, so each output value is the mean of a 2x2 "
    "block, and return it as float16.",
    "A network produced a half-precision feature map. Shrink it to half its height and width by averaging 2x2 blocks "
    "(INTER_AREA style) and hand back a float16 array.")


# ---------------------------------------------------------------- 6. uint32: minMaxLoc is exact

def _ue_cases():
    out = []
    for seed in (S + 51, S + 52, S + 53):
        rng = np.random.default_rng(seed)
        d = rng.integers(2 ** 31, 2 ** 32 - 1, (20, 30), dtype=np.uint64)
        d[rng.integers(0, 20), rng.integers(0, 30)] = rng.integers(2 ** 24 + 1, 2 ** 25)  # a minimum not exact in float32
        d[rng.integers(0, 20), rng.integers(0, 30)] = 2 ** 32 - 1 - rng.integers(0, 1000)  # the maximum
        d = d.astype(np.uint32)
        lo, hi = int(d.min()), int(d.max())
        assert (d == lo).sum() == 1 and (d == hi).sum() == 1
        (ly, lx), (hy, hx) = np.argwhere(d == lo)[0], np.argwhere(d == hi)[0]
        out.append(Case({"depth": d}, (lo, hi, np.array([lx, ly]), np.array([hx, hy]))))
    return out


def _ue_verify(case: Case, out: dict) -> Verdict:
    lo, hi, lo_loc, hi_loc = case.truth
    if (out["min_value"], out["max_value"]) != (lo, hi):
        return fail(f"min / max {out['min_value']} / {out['max_value']}, expected {lo} / {hi} (exact)")
    if not (np.array_equal(arr(out, "min_loc"), lo_loc) and np.array_equal(arr(out, "max_loc"), hi_loc)):
        return fail("min_loc / max_loc are not the (x, y) of the extreme values")
    return PASS


R.add(Check(
    id="drill_u32_extrema", group=GROUP, probes=("uint32_array_input",), depends_on_5_0=True, role="gotcha",
    gotcha="5.0 accepts uint32 in cv2.minMaxLoc and returns exact values; the 4.x workarounds (float32 cast loses bits above "
           "2**24, int32 cast wraps above 2**31) return wrong extremes",
    spec=spec("Find the nearest and the farthest point of a 32-bit depth map (values in micrometres, up to 2**32 - 1): "
              "the smallest and the largest value, exactly, and where each occurs.",
              {"depth": IOSpec("uint32", "HxW", 2, "depth in micrometres; the minimum and the maximum are unique")},
              {"min_value": IOSpec("int", "scalar", None, "smallest value"),
               "max_value": IOSpec("int", "scalar", None, "largest value"),
               "min_loc": IOSpec("int", "2", 1, "(x, y) of the smallest value"),
               "max_loc": IOSpec("int", "2", 1, "(x, y) of the largest value")}),
    make_cases=_ue_cases, verify=_ue_verify,
    reference=code("depth", '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(depth)\n'
                            '    return {"min_value": int(lo), "max_value": int(hi),\n'
                            '            "min_loc": np.array(lo_loc), "max_loc": np.array(hi_loc)}\n'),
    wrong={
        "float32_cast": Wrong("4.x workaround: depth.astype(np.float32) before cv2.minMaxLoc (24-bit mantissa)",
                              code("depth", '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(depth.astype(np.float32))\n'
                                            '    return {"min_value": int(lo), "max_value": int(hi),\n'
                                            '            "min_loc": np.array(lo_loc), "max_loc": np.array(hi_loc)}\n'),
                              "verifier"),
        "int32_cast": Wrong("4.x workaround: depth.astype(np.int32) (values above 2**31 turn negative)",
                            code("depth", '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(depth.astype(np.int32))\n'
                                          '    return {"min_value": int(lo), "max_value": int(hi),\n'
                                          '            "min_loc": np.array(lo_loc), "max_loc": np.array(hi_loc)}\n'),
                            "verifier"),
        "row_col_order": Wrong("minMaxLoc already returns (x, y); the code flips it to (row, column)",
                               code("depth", '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(depth)\n'
                                             '    return {"min_value": int(lo), "max_value": int(hi),\n'
                                             '            "min_loc": np.array(lo_loc)[::-1], "max_loc": np.array(hi_loc)[::-1]}\n'),
                               "verifier"),
    }),
    "For a uint32 depth map, return the exact minimum and maximum values and the (x, y) position of each. Both extremes "
    "are unique.",
    "A 32-bit depth image stores micrometres. Report its smallest and largest value without any rounding, together with "
    "the (x, y) coordinates where they occur.")


# ---------------------------------------------------------------- 7. Canny wants uint8

# calibrated on 40 scenes x 8 Canny settings (low 20..100, high 60..250, on the 8-bit rescaling of the image): no edge pixel
# falls farther than 2 px from an outline (stray 0 %), and the outlines are covered to 0.83 at worst (Canny(100, 250), whose
# high threshold misses the weaker sides); 0.75 keeps a 1.1x margin and still rejects an empty or a random edge map.
EDGE_MIN_COVER, EDGE_MAX_STRAY = 0.75, 0.05


def lit_boxes(seed: int, shape=(70, 100)):
    """A float32 image in [0, 1]: 1-2 bright boxes on a dark ground, blurred and slightly noisy. Truth: the outline pixels."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 0.12, np.float32)
    outline = np.zeros(shape, np.uint8)
    for i in range(int(rng.integers(1, 3))):
        w, h = int(rng.integers(18, 34)), int(rng.integers(14, 28))
        x, y = int(rng.integers(6, shape[1] // 2 - 6)) + i * (shape[1] // 2 - 6), int(rng.integers(8, shape[0] - h - 8))
        cv2.rectangle(img, (x, y), (x + w, y + h), 0.85, -1)
        cv2.rectangle(outline, (x, y), (x + w, y + h), 1, 1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 0.004, shape).astype(np.float32)
    return img.astype(np.float32), outline


def _fe_cases():
    return [Case({"img": im}, ol) for im, ol in (lit_boxes(s) for s in (S + 61, S + 62, S + 63))]


def _fe_verify(case: Case, out: dict) -> Verdict:
    edges = arr(out, "edges")
    if edges.shape != case.truth.shape:
        return fail(f"edges has shape {edges.shape}")
    e = edges > 0
    if not e.any():
        return fail("no edge pixels")
    k = np.ones((5, 5), np.uint8)
    near_outline = cv2.dilate(case.truth, k) > 0
    near_edges = cv2.dilate(e.astype(np.uint8), k) > 0
    stray = (e & ~near_outline).sum() / e.sum()
    cover = (near_edges & (case.truth > 0)).sum() / (case.truth > 0).sum()
    if stray > EDGE_MAX_STRAY:
        return fail(f"{stray:.0%} of the edge pixels are farther than 2 px from a box outline")
    if cover < EDGE_MIN_COVER:
        return fail(f"only {cover:.0%} of the box outlines have an edge pixel within 2 px")
    return PASS


R.add(Check(
    id="drill_float_edges", group=GROUP, probes=("Canny_float_input",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.Canny raises on a float image (-215 depth == CV_8U); casting float [0, 1] to uint8 without scaling gives zeros",
    spec=spec("Find the outlines of the bright boxes in the float32 image (values in 0..1) with the Canny edge "
              "detector and return the edge map.",
              {"img": IOSpec("float32", "HxW", 2, "grayscale image with values in 0..1: bright boxes on a dark ground")},
              {"edges": IOSpec("uint8", "HxW", 2, "255 on edge pixels, 0 elsewhere")}),
    make_cases=_fe_cases, verify=_fe_verify,
    reference=code("img", '    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)\n'
                          '    return {"edges": cv2.Canny(img8, 50, 150)}\n'),
    wrong={
        "canny_float": Wrong("cv2.Canny on the float32 image", code("img", '    return {"edges": cv2.Canny(img, 50, 150)}\n'),
                             "error", "depth() == CV_8U"),
        "unscaled_cast": Wrong("img.astype(np.uint8) without the x255 scaling: every value in 0..1 becomes 0",
                               code("img", '    return {"edges": cv2.Canny(img.astype(np.uint8), 50, 150)}\n'), "verifier"),
    }),
    "The image is float32 with values between 0 and 1. Run Canny edge detection on it and return a 0/255 uint8 edge "
    "map showing the outlines of the bright boxes.",
    "Detect the edges of the light boxes in this normalized (0..1, float32) grayscale image using cv2.Canny. The output "
    "is a uint8 map with 255 on edges.")


# ---------------------------------------------------------------- 8. findContours on a soft (float) mask

def soft_scene(seed: int):
    """A float32 probability map with 2-3 Gaussian blobs whose regions above 0.5 differ in area by at least 15 %. Truth:
    the regions' bounding boxes, largest first (computed on the thresholded map itself)."""
    rng = np.random.default_rng(seed)
    H, W = 80, 120
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    while True:
        k = int(rng.integers(2, 4))
        prob = np.zeros((H, W), np.float32)
        for i, (sx, sy) in enumerate(zip(rng.uniform(5, 13, k), rng.uniform(5, 13, k))):
            cx, cy = 18 + i * (W - 36) / max(k - 1, 1) + rng.uniform(-3, 3), rng.uniform(22, H - 22)
            prob = np.maximum(prob, np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)) * rng.uniform(0.8, 1.0))
        n, _, st, _ = cv2.connectedComponentsWithStats((prob > 0.5).astype(np.uint8), connectivity=8)
        areas = sorted((st[i, cv2.CC_STAT_AREA] for i in range(1, n)), reverse=True)
        if n - 1 == k and all(a >= 1.15 * b for a, b in zip(areas, areas[1:])) and areas[-1] >= 60:
            break
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    return prob.astype(np.float32), np.array([st[i, :4] for i in order])


def _sb_cases():
    return [Case({"prob": p}, boxes) for p, boxes in (soft_scene(s) for s in (S + 71, S + 72, S + 73, S + 74))]


def _sb_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "boxes")
    if got.shape != case.truth.shape or not np.array_equal(got, case.truth):
        return fail(f"boxes {got.tolist()} vs {case.truth.tolist()} (largest region first)")
    return PASS


R.add(Check(
    id="drill_soft_mask_boxes", group=GROUP, probes=("findContours_float_input",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.findContours raises on a float32 map (and on bool): threshold to 0/1 or 0/255 uint8 first",
    spec=spec("A network outputs a float32 probability map. Take the regions where the probability is above 0.5 and "
              "return one bounding box (x, y, width, height) per region, the region with the largest area first.",
              {"prob": IOSpec("float32", "HxW", 2, "probability map in 0..1; the regions above 0.5 are separate, with "
                                                   "different areas")},
              {"boxes": IOSpec("int", "Kx4", 2, "(x, y, w, h) per region, largest area first")}),
    make_cases=_sb_cases, verify=_sb_verify,
    reference=code("prob", '    mask = (prob > 0.5).astype(np.uint8)\n'
                           '    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                           '    contours = sorted(contours, key=cv2.contourArea, reverse=True)\n'
                           '    return {"boxes": np.array([cv2.boundingRect(c) for c in contours])}\n'),
    wrong={
        "contours_of_float": Wrong("cv2.findContours on the float32 map itself",
                                   code("prob", '    contours = cv2.findContours(prob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                                '    contours = sorted(contours, key=cv2.contourArea, reverse=True)\n'
                                                '    return {"boxes": np.array([cv2.boundingRect(c) for c in contours])}\n'),
                                   "error", "ContourScanner"),
        "contours_of_bool": Wrong("cv2.findContours on the bool array prob > 0.5",
                                  code("prob", '    contours = cv2.findContours(prob > 0.5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                              '    contours = sorted(contours, key=cv2.contourArea, reverse=True)\n'
                                              '    return {"boxes": np.array([cv2.boundingRect(c) for c in contours])}\n'),
                                  "error", "Unsupported format"),
        "unscaled_cast": Wrong("prob.astype(np.uint8) without thresholding: every value in 0..1 becomes 0, nothing is found",
                               code("prob", '    contours = cv2.findContours(prob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                            '    contours = sorted(contours, key=cv2.contourArea, reverse=True)\n'
                                            '    return {"boxes": np.array([cv2.boundingRect(c) for c in contours]).reshape(-1, 4)}\n'),
                               "verifier"),
    }),
    "Segment a float32 probability map at 0.5 and report the bounding box (x, y, w, h) of every connected region above "
    "it, ordered from the region with the largest area to the smallest.",
    "Threshold the probability map (float32, 0..1) at 0.5 and, using contours, return the bounding rectangles of the "
    "regions that remain, biggest region first.")


# ---------------------------------------------------------------- 9. cvtColor: float64 and the order of the channels

def _gf_cases():
    out = []
    for seed in (S + 81, S + 82, S + 83):
        rng = np.random.default_rng(seed)
        rgb = rng.random((20, 28, 3))
        rgb[:, :14, 0] = np.clip(rgb[:, :14, 0] + 0.5, 0, 1)  # strongly red on the left half, so RGB vs BGR matters
        gray = np.round(255 * (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]))
        out.append(Case({"rgb": rgb}, gray.astype(np.uint8)))
    return out


def _gf_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "gray")
    if got.shape != case.truth.shape:
        return fail(f"gray has shape {got.shape}")
    err = np.abs(got.astype(np.int16) - case.truth.astype(np.int16)).max()
    return fail(f"gray is off by up to {err} levels from the luma 0.299 R + 0.587 G + 0.114 B scaled to 0..255") if err > 1 else PASS


R.add(Check(
    id="drill_rgb_float_gray", group=GROUP, probes=("new:cvtColor_float64_input",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.cvtColor raises on a float64 image (depth must be uint8, uint16 or float32), and COLOR_BGR2GRAY on an RGB "
           "array swaps the red and blue weights",
    spec=spec("Convert the RGB image (float64, values 0..1, red channel first) to an 8-bit grayscale image with the "
              "usual luma weights, scaled to 0..255.",
              {"rgb": IOSpec("float64", "HxWx3", 3, "RGB image (channel 0 is red), values in 0..1")},
              {"gray": IOSpec("uint8", "HxW", 2, "round(255 * (0.299 R + 0.587 G + 0.114 B))")}),
    make_cases=_gf_cases, verify=_gf_verify,
    reference=code("rgb", '    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)\n'
                          '    return {"gray": np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)}\n'),
    wrong={
        "cvtcolor_float64": Wrong("cv2.cvtColor(rgb, COLOR_RGB2GRAY) on the float64 array",
                                  code("rgb", '    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)\n'
                                              '    return {"gray": np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)}\n'),
                                  "error", "Unsupported depth"),
        "bgr_flag_on_rgb": Wrong("COLOR_BGR2GRAY on an RGB array: red and blue weights swapped",
                                 code("rgb", '    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_BGR2GRAY)\n'
                                             '    return {"gray": np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)}\n'),
                                 "verifier"),
        "forgot_scale": Wrong("casts the 0..1 gray to uint8 without scaling by 255",
                              code("rgb", '    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)\n'
                                          '    return {"gray": gray.astype(np.uint8)}\n'), "verifier"),
    }),
    "Turn a float64 RGB image with values between 0 and 1 into an 8-bit grayscale image (luma weights 0.299, 0.587, "
    "0.114; result 0..255). The first channel is red.",
    "Grayscale conversion: the input is an RGB array (red first) of float64 values in [0, 1]. Return a uint8 image whose "
    "pixels are the luma scaled to 0..255 and rounded.")


# ---------------------------------------------------------------- 10. Otsu: one channel in, (value, image) out

# calibrated on the 3 images below: Otsu on the BT.601 gray is 55 / 77 / 40; the channel mean and the RGB-flipped weights land
# within 3 levels of it, the channel maximum (not a gray conversion) within 12. 4 levels accepts the first two and rejects
# the third; the fixed threshold 127 is 50+ levels away.
OTSU_TOL = 4.0


def _ot_cases():
    out = []
    for seed, (dark, light) in ((S + 91, (40, 150)), (S + 92, (60, 200)), (S + 93, (25, 110))):
        rng = np.random.default_rng(seed)
        base = np.full((40, 56), dark, np.float32)
        for _ in range(3):
            x, y = int(rng.integers(4, 40)), int(rng.integers(4, 26))
            cv2.rectangle(base, (x, y), (x + int(rng.integers(8, 16)), y + int(rng.integers(6, 12))), light, -1)
        img = np.clip(base[..., None] + rng.normal(0, 7, (40, 56, 3)), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(Case({"img": img}, (float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]), gray)))
    return out


def _ot_verify(case: Case, out: dict) -> Verdict:
    thr_gt, gray = case.truth
    thr, mask = out["threshold"], arr(out, "mask")
    if abs(thr - thr_gt) > OTSU_TOL:
        return fail(f"threshold {thr}, the Otsu threshold of the gray image is {thr_gt}")
    if mask.shape != gray.shape or not set(np.unique(mask).tolist()) <= {0, 255}:
        return fail("mask is not a 0/255 image of the same size")
    if ((gray > thr) != (mask > 0)).mean() > 0.005:
        return fail("mask does not match (gray > threshold) for the returned threshold")
    return PASS


_OT_HEAD = code("img", "")
R.add(Check(
    id="drill_otsu_binarize", group=GROUP, probes=("threshold_otsu_value", "new:threshold_otsu_3channel"),
    depends_on_5_0=False, role="gotcha",
    gotcha="cv2.threshold returns (value, image) and THRESH_OTSU needs a single-channel image (-2 on BGR); a fixed threshold "
           "is not Otsu",
    spec=spec("Binarize the colour image with Otsu's method on its grayscale version: report the threshold Otsu picked and "
              "the 0/255 mask of the pixels brighter than it.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image: bright patches on a darker ground, with noise")},
              {"threshold": IOSpec("float", "scalar", None, "the threshold chosen by Otsu's method"),
               "mask": IOSpec("uint8", "HxW", 2, "255 where gray > threshold, else 0")}),
    make_cases=_ot_cases, verify=_ot_verify,
    reference=code("img", '    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                          '    thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n'
                          '    return {"threshold": float(thr), "mask": mask}\n'),
    wrong={
        "otsu_on_bgr": Wrong("THRESH_OTSU on the 3-channel image",
                             code("img", '    thr, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n'
                                         '    return {"threshold": float(thr), "mask": mask[..., 0]}\n'),
                             "error", "src_type == CV_8UC1"),
        "fixed_threshold": Wrong("threshold 127 instead of Otsu's value",
                                 code("img", '    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                                             '    thr, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)\n'
                                             '    return {"threshold": float(thr), "mask": mask}\n'), "verifier"),
        "swapped_return": Wrong("mask, thr = cv2.threshold(...): the return order is (value, image)",
                                code("img", '    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                                            '    mask, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n'
                                            '    return {"threshold": thr, "mask": mask}\n'), "verifier"),
    }),
    "Apply Otsu thresholding to this BGR image (via its grayscale conversion) and return the chosen threshold together "
    "with the binary mask, 255 where the gray value exceeds it.",
    "Pick the binarization threshold automatically with Otsu's method on the gray version of the colour image; return "
    "that threshold value and the resulting 0/255 mask.")


CHECKS = R.checks
PHRASINGS = R.phrasings
