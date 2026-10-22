"""Drills on calling conventions that bite on any version: what a function returns and in which order or unit.

All timeless (same on 4.14 and 5.0), none a copy of a gotcha-suite task:
  * ``findContours`` returns two values (contours, hierarchy) and the hierarchy is (1, N, 4), not (N, 4): the row of a
    contour is ``hierarchy[0][i]``, its parent is column 3, its first child column 2 (the 3.x unpack ``_, cnts, h`` fails);
  * ``matchTemplate`` scores are read at the minimum for the SQDIFF methods and at the maximum for the others, and the
    location is the template's top-left corner, not its centre;
  * ``minAreaRect`` reports ``(w, h)`` and an angle that depend on which side the library calls the width, so the long axis
    needs the box points, not the raw angle;
  * a rotation that keeps the whole picture needs the translated matrix, a canvas computed from |cos| and |sin|, and the
    sign convention of ``getRotationMatrix2D`` (positive = counter-clockwise);
  * ``kmeans`` takes float32 data only and returns labels as (N, 1).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, spec

GROUP = "calling"
R = Registry()
S = SEED_BASE + 2_000  # this module's seeds: 63_000 .. 63_999


# ---------------------------------------------------------------- 1. RETR_CCOMP hierarchy: shapes with holes

def washer_scene(seed: int):
    """2-4 solid shapes (rectangles or ellipses) on a black ground, each with 0-2 round holes, at least one shape with a
    hole and one without. Truth: (shapes, holes, shapes with at least one hole)."""
    rng = np.random.default_rng(seed)
    H, W = 110, 200
    while True:
        k = int(rng.integers(2, 5))
        m = np.zeros((H, W), np.uint8)
        holes = []
        slot = (W - 20) // k
        for i in range(k):
            cx, cy = 10 + slot * i + slot // 2 + int(rng.integers(-3, 4)), H // 2 + int(rng.integers(-8, 9))
            a, b = int(rng.integers(14, min(slot // 2 - 2, 34) + 1)), int(rng.integers(26, 42))
            if rng.random() < 0.5:
                cv2.ellipse(m, (cx, cy), (a, b), 0, 0, 360, 255, -1)
            else:
                cv2.rectangle(m, (cx - a, cy - b), (cx + a, cy + b), 255, -1)
            n_holes = int(rng.integers(0, 3))
            holes.append(n_holes)
            for j in range(n_holes):
                hx, hy = cx + (j * 2 - n_holes + 1) * (a // 2), cy + int(rng.integers(-8, 9))
                cv2.circle(m, (hx, hy), int(rng.integers(3, min(6, a // 2 - 2) + 1)), 0, -1)
        if any(holes) and not all(holes):
            return m, (k, sum(holes), sum(1 for h in holes if h))


_HR_REF = code("mask", '''\
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rows = hierarchy[0]
    outer = rows[:, 3] == -1
    with_holes = outer & (rows[:, 2] != -1)
    return {"n_shapes": int(outer.sum()), "n_holes": int((~outer).sum()), "n_with_holes": int(with_holes.sum())}
''')


def _hr_cases():
    return [Case({"mask": m}, t) for m, t in (washer_scene(s) for s in (S + 1, S + 2, S + 3, S + 4))]


def _hr_verify(case: Case, out: dict) -> Verdict:
    got = (out["n_shapes"], out["n_holes"], out["n_with_holes"])
    return PASS if got == case.truth else fail(f"(shapes, holes, shapes with holes) = {got}, expected {case.truth}")


R.add(Check(
    id="drill_hierarchy_holes", group=GROUP, probes=("findContours_tree",), depends_on_5_0=False, role="gotcha",
    gotcha="findContours returns (contours, hierarchy) in every 4.x/5.0 version, hierarchy is (1, N, 4): hierarchy[i][3] "
           "indexes the wrong axis, and `_, cnts, h = cv2.findContours` is the 3.x unpack",
    spec=spec("Inspect the washers in the mask: count the separate solid shapes, the holes in them, and how many shapes "
              "have at least one hole. Use the contour hierarchy.",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255; solid shapes (some with round holes) that touch neither each "
                                                 "other nor the border; no shape lies inside a hole")},
              {"n_shapes": IOSpec("int", "scalar", None, "number of separate shapes"),
               "n_holes": IOSpec("int", "scalar", None, "total number of holes"),
               "n_with_holes": IOSpec("int", "scalar", None, "number of shapes that have at least one hole")}),
    make_cases=_hr_cases, verify=_hr_verify, reference=_HR_REF,
    wrong={
        "three_value_unpack": Wrong("3.x unpack: _, contours, hierarchy = cv2.findContours(...)",
                                    code("mask", '    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)\n'
                                                 '    rows = hierarchy[0]\n    outer = rows[:, 3] == -1\n'
                                                 '    return {"n_shapes": int(outer.sum()), "n_holes": int((~outer).sum()),\n'
                                                 '            "n_with_holes": int((outer & (rows[:, 2] != -1)).sum())}\n'),
                                    "ValueError", "expected 3, got 2"),
        "hierarchy_row_index": Wrong("hierarchy[i][3] for contour i: the contour index is the second axis of a (1, N, 4) array",
                                     code("mask", '    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)\n'
                                                  '    parents = [hierarchy[i][3] for i in range(len(contours))]\n'
                                                  '    n_holes = sum(1 for p in parents if p != -1)\n'
                                                  '    return {"n_shapes": len(contours) - n_holes, "n_holes": n_holes,\n'
                                                  '            "n_with_holes": len({p for p in parents if p != -1})}\n'),
                                     "IndexError", "out of bounds"),
        "retr_list": Wrong("RETR_LIST has no nesting: every contour has parent -1, so the holes count as shapes",
                           code("mask", '    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)\n'
                                        '    rows = hierarchy[0]\n    outer = rows[:, 3] == -1\n'
                                        '    return {"n_shapes": int(outer.sum()), "n_holes": int((~outer).sum()),\n'
                                        '            "n_with_holes": int((outer & (rows[:, 2] != -1)).sum())}\n'),
                           "verifier"),
    }),
    "A mask shows washers and plates. Using findContours with a two-level hierarchy, report how many separate shapes there "
    "are, how many holes they contain in total, and how many of the shapes have at least one hole.",
    "Count outer outlines and inner outlines in the 0/255 mask with the contour hierarchy: the number of shapes, the "
    "number of holes, and the number of shapes that contain at least one hole.")


# ---------------------------------------------------------------- 2. matchTemplate: which extreme, and where the centre is

def textured(seed: int, shape=(96, 144)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = sum(cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), s) * w for s, w in ((1.5, 1.0), (4.0, 2.5)))
    f = (f - f.min()) / (f.max() - f.min())
    img = np.round(255 * f).astype(np.uint8)
    for _ in range(8):
        x, y, w, h = int(rng.integers(0, shape[1] - 20)), int(rng.integers(0, shape[0] - 20)), int(rng.integers(6, 18)), int(rng.integers(6, 18))
        cv2.rectangle(img, (x, y), (x + w, y + h), int(rng.integers(0, 256)), -1)
    return img


def _tc_cases():
    out = []
    for seed, method in ((S + 11, "sqdiff"), (S + 12, "ccoeff"), (S + 13, "sqdiff"), (S + 14, "ccoeff")):
        img = textured(seed)
        rng = np.random.default_rng(seed)
        w, h = int(rng.integers(16, 31)), int(rng.integers(14, 27))
        x, y = int(rng.integers(0, img.shape[1] - w)), int(rng.integers(0, img.shape[0] - h))
        out.append(Case({"img": img, "templ": img[y:y + h, x:x + w].copy(), "method": method},
                        (np.array([x + w // 2, y + h // 2]), 0.0 if method == "sqdiff" else 1.0)))
    return out


def _tc_verify(case: Case, out: dict) -> Verdict:
    centre_gt, score_gt = case.truth
    c = arr(out, "center")
    if c.shape != (2,) or not np.array_equal(c, centre_gt):
        return fail(f"centre {c.tolist()}, the template was cut from the image around {centre_gt.tolist()}")
    if abs(out["score"] - score_gt) > 0.02:
        return fail(f"score {out['score']:.3f}: the best match of a patch cut from the image scores {score_gt}")
    return PASS


_TC_REF = code("img, templ, method", '''\
    flag = cv2.TM_SQDIFF_NORMED if method == "sqdiff" else cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(img, templ, flag)
    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(res)
    x, y = lo_loc if method == "sqdiff" else hi_loc
    h, w = templ.shape
    return {"center": np.array([x + w // 2, y + h // 2]), "score": float(lo if method == "sqdiff" else hi)}
''')
R.add(Check(
    id="drill_template_center", group=GROUP, probes=("new:matchTemplate_extreme_by_method",), depends_on_5_0=False, role="gotcha",
    gotcha="matchTemplate: the best SQDIFF score is the minimum, the best CCOEFF score the maximum, and the location is the "
           "template's top-left corner; a model that always reads maxLoc finds the worst match for SQDIFF",
    spec=spec("Find where the template sits in the image. `method` names the score: 'sqdiff' is the normalized squared "
              "difference (best match = lowest score), 'ccoeff' the normalized correlation coefficient (best = highest). "
              "Return the centre of the best match, (x + w // 2, y + h // 2) for a match whose top-left corner is "
              "(x, y), and its score.",
              {"img": IOSpec("uint8", "HxW", 2, "grayscale image"),
               "templ": IOSpec("uint8", "hxw", 2, "a patch cut out of the image"),
               "method": IOSpec("str", "scalar", None, "'sqdiff' or 'ccoeff'")},
              {"center": IOSpec("int", "2", 1, "(x, y) of the centre of the best match"),
               "score": IOSpec("float", "scalar", None, "the matching score at the best match")}),
    make_cases=_tc_cases, verify=_tc_verify, reference=_TC_REF,
    wrong={
        "always_max": Wrong("reads maxLoc for both scores: for 'sqdiff' that is the worst match",
                            code("img, templ, method", '    flag = cv2.TM_SQDIFF_NORMED if method == "sqdiff" else cv2.TM_CCOEFF_NORMED\n'
                                                       '    res = cv2.matchTemplate(img, templ, flag)\n'
                                                       '    _, hi, _, (x, y) = cv2.minMaxLoc(res)\n    h, w = templ.shape\n'
                                                       '    return {"center": np.array([x + w // 2, y + h // 2]), "score": float(hi)}\n'),
                            "verifier"),
        "top_left": Wrong("returns the top-left corner of the match instead of its centre",
                          code("img, templ, method", '    flag = cv2.TM_SQDIFF_NORMED if method == "sqdiff" else cv2.TM_CCOEFF_NORMED\n'
                                                     '    res = cv2.matchTemplate(img, templ, flag)\n'
                                                     '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(res)\n'
                                                     '    loc, score = (lo_loc, lo) if method == "sqdiff" else (hi_loc, hi)\n'
                                                     '    return {"center": np.array(loc), "score": float(score)}\n'), "verifier"),
        "swapped_arguments": Wrong("cv2.matchTemplate(templ, img, ...): image and template swapped",
                                   code("img, templ, method", '    flag = cv2.TM_SQDIFF_NORMED if method == "sqdiff" else cv2.TM_CCOEFF_NORMED\n'
                                                              '    res = cv2.matchTemplate(templ, img, flag)\n'
                                                              '    lo, hi, lo_loc, hi_loc = cv2.minMaxLoc(res)\n'
                                                              '    return {"center": np.array(hi_loc), "score": float(hi)}\n'),
                                   "verifier"),
    }),
    "Locate a cut-out patch in the image with template matching and return the centre of the best match and its score. "
    "For method 'sqdiff' use the normalized squared difference (lower is better); for 'ccoeff' use the normalized "
    "correlation coefficient (higher is better). The centre is (x + w // 2, y + h // 2).",
    "Template matching: the patch was cut from the image. Score it with TM_SQDIFF_NORMED when method is 'sqdiff' and "
    "with TM_CCOEFF_NORMED when it is 'ccoeff'; report the best match's centre (x + w // 2, y + h // 2) and score.")


# ---------------------------------------------------------------- 3. minAreaRect: the long axis of a rod

# calibrated on 600 rods (length 60..110, width 12..24, angle 0..180 in 1 degree steps, drawn with fillPoly): the long axis
# measured from the box points is within 1.0 degree of the drawn angle, the length and the width within 1.78 px;
# 2.5 degrees / 3 px keep a 1.7-2.5x margin.
ROD_ANGLE_TOL, ROD_SIZE_TOL = 2.5, 3.0


def rod_scene(seed: int, angle=None):
    rng = np.random.default_rng(seed)
    L, Wd = float(rng.integers(60, 111)), float(rng.integers(12, 25))
    theta = float(rng.uniform(0, 180)) if angle is None else float(angle)
    c = np.array([rng.uniform(60, 90), rng.uniform(60, 90)])
    d = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
    n = np.array([-d[1], d[0]])
    corners = np.array([c + sx * d * L / 2 + sy * n * Wd / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    m = np.zeros((150, 150), np.uint8)
    cv2.fillPoly(m, [np.round(corners).astype(np.int32)], 255)
    return m, (theta, L, Wd)


def _ra_cases():
    return [Case({"mask": m}, t) for m, t in (rod_scene(S + 21 + i, a) for i, a in enumerate((20.0, 75.0, 112.0, 160.0)))]


def _ra_verify(case: Case, out: dict) -> Verdict:
    theta, L, Wd = case.truth
    a = out["angle_deg"]
    diff = abs((a - theta + 90.0) % 180.0 - 90.0)
    if not 0.0 <= a < 180.0:
        return fail(f"angle {a:.1f} is not in [0, 180)")
    if diff > ROD_ANGLE_TOL:
        return fail(f"long axis at {a:.1f} deg, the rod was drawn at {theta:.1f} deg")
    if abs(out["length"] - L) > ROD_SIZE_TOL or abs(out["width"] - Wd) > ROD_SIZE_TOL:
        return fail(f"length / width {out['length']:.1f} / {out['width']:.1f}, the rod is {L:.0f} x {Wd:.0f} px")
    return PASS


_RA_REF = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    rod = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(rod))
    e0, e1 = box[1] - box[0], box[2] - box[1]
    long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    sides = sorted([float(np.linalg.norm(e0)), float(np.linalg.norm(e1))], reverse=True)
    return {"angle_deg": float(np.degrees(np.arctan2(long_edge[1], long_edge[0])) % 180.0),
            "length": sides[0], "width": sides[1]}
''')
_RA_HEAD = code("mask", '''\
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    rod = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(rod)
''')
R.add(Check(
    id="drill_rod_axis", group=GROUP, probes=("minAreaRect",), depends_on_5_0=False, role="gotcha",
    gotcha="minAreaRect's (w, h) and angle depend on which side the library calls the width: using rect[2] as the long-axis "
           "angle, or (w, h) as (length, width), is wrong for half of all orientations",
    spec=spec("A rod lies on the table. Measure it: the direction of its long axis, as the angle from the positive x axis "
              "towards the positive y axis (the image y axis points down) in degrees in [0, 180), and its length and "
              "width in px.",
              {"mask": IOSpec("uint8", "HxW", 2, "0/255 mask with one filled rectangle (the rod), at least 60 x 12 px")},
              {"angle_deg": IOSpec("float", "scalar", None, "angle of the long axis in [0, 180)"),
               "length": IOSpec("float", "scalar", None, "the longer side in px"),
               "width": IOSpec("float", "scalar", None, "the shorter side in px")}),
    make_cases=_ra_cases, verify=_ra_verify, reference=_RA_REF,
    wrong={
        "raw_angle": Wrong("angle = rect[2], taken as the long-axis angle", _RA_HEAD +
                           '    return {"angle_deg": float(angle) % 180.0, "length": max(w, h), "width": min(w, h)}\n', "verifier"),
        "unsorted_sides": Wrong("length, width = rect[1] in the order the library returns them (angle itself is right)",
                                code("mask", '    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                             '    rod = max(contours, key=cv2.contourArea)\n'
                                             '    rect = cv2.minAreaRect(rod)\n    box = cv2.boxPoints(rect)\n'
                                             '    e0, e1 = box[1] - box[0], box[2] - box[1]\n'
                                             '    long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1\n'
                                             '    (w, h) = rect[1]\n'
                                             '    return {"angle_deg": float(np.degrees(np.arctan2(long_edge[1], long_edge[0])) % 180.0),\n'
                                             '            "length": float(w), "width": float(h)}\n'), "verifier"),
        "unwrapped_angle": Wrong("arctan2 of the long edge without the modulo: negative angles for edges pointing up",
                                 code("mask", '    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]\n'
                                              '    rod = max(contours, key=cv2.contourArea)\n'
                                              '    box = cv2.boxPoints(cv2.minAreaRect(rod))\n'
                                              '    e0, e1 = box[1] - box[0], box[2] - box[1]\n'
                                              '    long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1\n'
                                              '    sides = sorted([float(np.linalg.norm(e0)), float(np.linalg.norm(e1))], reverse=True)\n'
                                              '    return {"angle_deg": float(np.degrees(np.arctan2(long_edge[1], long_edge[0]))),\n'
                                              '            "length": sides[0], "width": sides[1]}\n'), "verifier"),
    }),
    "Find the rod in the mask with its minimum-area bounding rectangle and report the orientation of its long axis "
    "(degrees from the +x axis towards +y, in [0, 180)), its length (longer side) and its width (shorter side) in pixels.",
    "From the mask of a single rectangular rod, get the angle of its long axis in [0, 180) degrees, measured from the "
    "positive x axis towards positive y (image coordinates, y down), plus the lengths of its long and short sides in px.")


# ---------------------------------------------------------------- 4. rotate clockwise without cropping

def flag_image(seed: int):
    """A white sheet with a red 12 x 12 marker near its top-left corner (so rotation direction is visible)."""
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(50, 70)), int(rng.integers(90, 120))
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (6, 6), (17, 17), (0, 0, 255), -1)
    return img


def _marker_offset(rot: np.ndarray):
    """Centroid of the red marker relative to the centre of the non-black region's bounding box, plus that box's size."""
    red = (rot[..., 2] > 180) & (rot[..., 1] < 80) & (rot[..., 0] < 80)
    lit = rot.max(axis=2) > 20
    if not red.any() or not lit.any():
        return None
    ys, xs = np.nonzero(red)
    ly, lx = np.nonzero(lit)
    centre = np.array([(lx.min() + lx.max()) / 2, (ly.min() + ly.max()) / 2])
    return np.array([xs.mean(), ys.mean()]) - centre, (lx.max() - lx.min() + 1, ly.max() - ly.min() + 1), int(lit.sum())


# calibrated on 60 sheets x 18 angles (-170..170): the reference's bounding box is within 1.5 px of h|sin| + w|cos| (and the
# transposed formula), its lit area within 2.5 % of h * w, and its marker offset within 1.0 px of the rotated marker offset;
# all 1080 runs pass.
ROT_TOL_PX, ROT_AREA_REL = 3.0, 0.06


def _rc_cases():
    out = []
    for seed, angle in ((S + 31, 30.0), (S + 32, 75.0), (S + 33, -25.0), (S + 34, 140.0)):
        img = flag_image(seed)
        out.append(Case({"img": img, "angle": angle}, None))
    return out


def _rc_verify(case: Case, out: dict) -> Verdict:
    img, angle = case.inputs["img"], case.inputs["angle"]
    h, w = img.shape[:2]
    rot = arr(out, "rotated")
    if rot.ndim != 3 or rot.shape[2] != 3:
        return fail(f"rotated has shape {rot.shape}")
    a = math.radians(angle)
    want_w, want_h = h * abs(math.sin(a)) + w * abs(math.cos(a)), h * abs(math.cos(a)) + w * abs(math.sin(a))
    got = _marker_offset(rot)
    if got is None:
        return fail("the red marker or the sheet is missing from the rotated image")
    off, (bw, bh), lit = got
    if abs(bw - want_w) > ROT_TOL_PX or abs(bh - want_h) > ROT_TOL_PX:
        return fail(f"the rotated sheet spans {bw} x {bh} px, expected about {want_w:.0f} x {want_h:.0f} (nothing cropped)")
    if abs(lit - h * w) > ROT_AREA_REL * h * w:
        return fail(f"{lit} lit pixels, the sheet has {h * w}: part of it is cropped")
    mx, my = 11.5 - (w - 1) / 2, 11.5 - (h - 1) / 2  # marker centre relative to the sheet centre
    want_off = np.array([mx * math.cos(a) - my * math.sin(a), mx * math.sin(a) + my * math.cos(a)])  # y down: clockwise
    if np.abs(off - want_off).max() > ROT_TOL_PX:
        return fail(f"the marker is at {off.round(1).tolist()} from the sheet centre, expected {want_off.round(1).tolist()}: "
                    "is the rotation clockwise?")
    return PASS


_RC_REF = code("img, angle", '''\
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return {"rotated": cv2.warpAffine(img, M, (new_w, new_h))}
''')
R.add(Check(
    id="drill_rotate_clockwise", group=GROUP, probes=("new:getRotationMatrix2D_sign_and_canvas",), depends_on_5_0=False,
    role="gotcha",
    gotcha="getRotationMatrix2D's positive angle is counter-clockwise, warpAffine crops to dsize, and dsize is (width, height)",
    spec=spec("Rotate the picture clockwise by `angle` degrees (a negative angle turns it counter-clockwise) about its centre. "
              "Nothing may be cropped: make the output canvas just large enough to hold the whole rotated picture, "
              "filled with black where the picture is not.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR picture, wider than tall"),
               "angle": IOSpec("float", "scalar", None, "degrees, clockwise as displayed")},
              {"rotated": IOSpec("uint8", "H'xW'x3", 3, "the rotated picture on the smallest canvas that holds it")}),
    make_cases=_rc_cases, verify=_rc_verify, reference=_RC_REF,
    wrong={
        "counter_clockwise": Wrong("passes +angle to getRotationMatrix2D, which turns counter-clockwise",
                                   code("img, angle", '    h, w = img.shape[:2]\n    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)\n'
                                                      '    cos, sin = abs(M[0, 0]), abs(M[0, 1])\n'
                                                      '    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))\n'
                                                      '    M[0, 2] += new_w / 2 - w / 2\n    M[1, 2] += new_h / 2 - h / 2\n'
                                                      '    return {"rotated": cv2.warpAffine(img, M, (new_w, new_h))}\n'), "verifier"),
        "cropped": Wrong("keeps the input size: the corners of the rotated sheet are cut off",
                         code("img, angle", '    h, w = img.shape[:2]\n    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)\n'
                                           '    return {"rotated": cv2.warpAffine(img, M, (w, h))}\n'), "verifier"),
        "dsize_swapped": Wrong("canvas passed as (height, width) instead of (width, height)",
                               code("img, angle", '    h, w = img.shape[:2]\n    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)\n'
                                                  '    cos, sin = abs(M[0, 0]), abs(M[0, 1])\n'
                                                  '    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))\n'
                                                  '    M[0, 2] += new_w / 2 - w / 2\n    M[1, 2] += new_h / 2 - h / 2\n'
                                                  '    return {"rotated": cv2.warpAffine(img, M, (new_h, new_w))}\n'), "verifier"),
    }),
    "Turn the picture clockwise by `angle` degrees around its centre and return the whole rotated picture on a black "
    "canvas just big enough to contain it (no cropping). A negative angle means counter-clockwise.",
    "Rotate a BGR picture clockwise (as displayed) by `angle` degrees without losing any corner: the output canvas "
    "must be the smallest one that holds the entire rotated image, with black fill.")


# ---------------------------------------------------------------- 5. kmeans: float32 in, (N, 1) labels out

def banded_colours(seed: int, k: int):
    """Vertical bands of k well separated colours with clearly different widths, plus Gaussian noise (sigma 6)."""
    rng = np.random.default_rng(seed)
    H, W = 36, 72
    while True:
        cols = rng.integers(20, 236, (k, 3))
        if min(np.linalg.norm(cols[i] - cols[j]) for i in range(k) for j in range(i)) > 90:
            break
    while True:
        widths = [int(v) for v in np.sort(rng.choice(np.arange(8, 22), size=k, replace=False))]
        if sum(widths) <= W - 6:
            break
    widths[-1] += W - sum(widths)
    clean = np.concatenate([np.tile(cols[i].astype(np.uint8), (H, w, 1)) for i, w in enumerate(widths)], axis=1)
    noisy = np.clip(clean + rng.normal(0, 6, clean.shape), 0, 255).astype(np.uint8)
    return noisy, clean, sorted((w * H for w in widths), reverse=True)


def _pk_cases():
    out = []
    for seed, k in ((S + 41, 3), (S + 42, 4), (S + 43, 3), (S + 44, 5)):
        noisy, clean, counts = banded_colours(seed, k)
        out.append(Case({"img": noisy, "k": k}, (clean, np.array(counts))))
    return out


def _pk_verify(case: Case, out: dict) -> Verdict:
    clean, counts_gt = case.truth
    q, counts = arr(out, "quantized"), arr(out, "counts")
    if q.shape != clean.shape:
        return fail(f"quantized has shape {q.shape}")
    if len(np.unique(q.reshape(-1, 3), axis=0)) > case.inputs["k"]:
        return fail(f"more than {case.inputs['k']} different colours in the quantized image")
    err = np.abs(q.astype(np.int16) - clean.astype(np.int16)).mean()
    if err > 5.0:
        return fail(f"the palette is far from the true colours (mean absolute error {err:.1f})")
    if counts.shape != counts_gt.shape or not np.array_equal(counts, counts_gt):
        return fail(f"pixel counts {counts.tolist()} vs {counts_gt.tolist()} (largest first)")
    return PASS


_PK_REF = code("img, k", '''\
    h, w = img.shape[:2]
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    quantized = np.clip(np.round(centers), 0, 255).astype(np.uint8)[labels].reshape(h, w, 3)
    return {"quantized": quantized, "counts": np.sort(np.bincount(labels, minlength=k))[::-1]}
''')
R.add(Check(
    id="drill_palette_kmeans", group=GROUP, probes=("kmeans",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: cv2.kmeans needs float32 samples (uint8 and float64 raise) and its labels are still (N, 1): "
           "np.bincount(labels) and labels used as an index need .flatten()",
    spec=spec("Reduce the picture to a palette of k colours with k-means clustering of the pixel colours. Return the "
              "palette image (every pixel replaced by the centre of its cluster) and the number of pixels in each "
              "cluster, largest first.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR picture made of k roughly flat colours plus noise"),
               "k": IOSpec("int", "scalar", None, "number of colours")},
              {"quantized": IOSpec("uint8", "HxWx3", 3, "the picture with at most k distinct colours"),
               "counts": IOSpec("int", "k", 1, "pixels per cluster, largest first")}),
    make_cases=_pk_cases, verify=_pk_verify, reference=_PK_REF,
    wrong={
        "kmeans_uint8": Wrong("cv2.kmeans on the uint8 pixel array",
                              code("img, k", '    h, w = img.shape[:2]\n    data = img.reshape(-1, 3)\n'
                                             '    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)\n'
                                             '    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)\n'
                                             '    labels = labels.flatten()\n'
                                             '    return {"quantized": centers.astype(np.uint8)[labels].reshape(h, w, 3),\n'
                                             '            "counts": np.sort(np.bincount(labels, minlength=k))[::-1]}\n'),
                              "error", "CV_32F"),
        "bincount_of_column": Wrong("np.bincount(labels) on the (N, 1) label column",
                                    code("img, k", '    h, w = img.shape[:2]\n    data = img.reshape(-1, 3).astype(np.float32)\n'
                                                   '    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)\n'
                                                   '    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)\n'
                                                   '    quantized = centers.astype(np.uint8)[labels.flatten()].reshape(h, w, 3)\n'
                                                   '    return {"quantized": quantized, "counts": np.sort(np.bincount(labels))[::-1]}\n'),
                                    "ValueError", "too deep"),
        "float_palette": Wrong("returns the float32 cluster centres as the image",
                               code("img, k", '    h, w = img.shape[:2]\n    data = img.reshape(-1, 3).astype(np.float32)\n'
                                              '    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)\n'
                                              '    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)\n'
                                              '    labels = labels.flatten()\n'
                                              '    return {"quantized": centers[labels].reshape(h, w, 3),\n'
                                              '            "counts": np.sort(np.bincount(labels, minlength=k))[::-1]}\n'),
                               "verifier"),
    }),
    "Colour quantization: cluster the pixel colours of the BGR picture into k groups with cv2.kmeans, give back the picture "
    "recoloured with the k cluster centres, and the pixel count of every cluster in descending order.",
    "Use k-means on the pixel colours to build a k-colour version of the image. Return that image and how many pixels "
    "each of the k colours covers, largest group first.")


CHECKS = R.checks
PHRASINGS = R.phrasings
