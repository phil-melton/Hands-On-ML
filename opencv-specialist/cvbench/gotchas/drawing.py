"""Group 3, views and drawing: what the cv2 drawing functions accept as target and as coordinates.

All six are timeless (identical on 4.14 and 5.0):
  * drawing needs a target cv2 can wrap as a Mat: ``img[:, :, ::-1]`` (negative stride) and ``img.T``
    (swapped strides) raise "Layout of the output array img is incompatible with cv::Mat";
    a row-strided view such as ``img[::2]`` and a plain ROI slice are fine;
  * an ROI slice is a VIEW, so drawing into it writes through to the parent: copying it first and drawing
    on the copy leaves the parent untouched;
  * coordinates must be Python/numpy INTEGERS: a float centre or endpoint raises "Can't parse 'center'"
    (even an integral one such as ``np.round(x)``, which is float64), while numpy int32/int64 are fine.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail, mask_iou

GROUP = "drawing"


def _spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="gotcha", task=task, inputs=inputs, outputs=outputs)


def _code(sig: str, body: str) -> str:
    return f"import cv2\nimport numpy as np\n\n\ndef solve({sig}):\n{body}"


def _exact(case: Case, out: dict, key: str = "img") -> Verdict:
    got = arr(out, key)
    if got.shape != case.truth.shape:
        return fail(f"{key} has shape {got.shape}, expected {case.truth.shape}")
    if not np.array_equal(got, case.truth):
        return fail(f"{int((got != case.truth).sum())} values differ from the expected drawing")
    return PASS


# ---------------------------------------------------------------- 1. drawing on a channel-reversed view

def _rv_cases():
    out = []
    for seed in (401, 402):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
        x0, y0 = int(rng.integers(2, 20)), int(rng.integers(2, 12))
        x1, y1 = x0 + int(rng.integers(10, 30)), y0 + int(rng.integers(8, 20))
        rgb = rng.integers(0, 256, 3).astype(np.int32)
        exp = img.copy()
        cv2.rectangle(exp, (x0, y0), (x1, y1), (int(rgb[2]), int(rgb[1]), int(rgb[0])), -1)
        out.append(Case({"img": img, "box": np.array([x0, y0, x1, y1], np.int32), "color_rgb": rgb}, exp))
    return out


_RV_SIG = "img, box, color_rgb"
_RV_PRE = ('    out = img.copy()\n    p0, p1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))\n'
           '    color = tuple(int(c) for c in color_rgb)\n')
REVERSED_VIEW = Check(
    id="draw_reversed_view", group=GROUP, probes=("draw_on_reversed_channel_view",), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2 drawing functions raise on a negative-stride view such as img[:, :, ::-1]",
    spec=_spec("Draw a filled rectangle with corners (x0, y0) and (x1, y1) on the image. The colour is given in "
               "RGB order, but the image is BGR (OpenCV order).",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
                "box": IOSpec("int", "4", 1, "x0, y0, x1, y1"),
                "color_rgb": IOSpec("int", "3", 1, "(r, g, b), each 0..255")},
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image with the rectangle drawn")}),
    make_cases=_rv_cases, verify=_exact,
    reference=_code(_RV_SIG, _RV_PRE + '    r, g, b = color\n    cv2.rectangle(out, p0, p1, (b, g, r), -1)\n'
                                       '    return {"img": out}\n'),
    wrong={
        "reversed_view": Wrong("draws on the RGB view out[:, :, ::-1] hoping it writes through to the BGR image",
                               _code(_RV_SIG, _RV_PRE + '    rgb_view = out[:, :, ::-1]\n'
                                                        '    cv2.rectangle(rgb_view, p0, p1, color, -1)\n'
                                                        '    return {"img": out}\n'),
                               "error", "Layout of the output array img is incompatible"),
        "copy_not_swapped_back": Wrong("draws on a copy of the reversed view and returns it still in RGB order",
                                       _code(_RV_SIG, _RV_PRE + '    rgb = out[:, :, ::-1].copy()\n'
                                                                '    cv2.rectangle(rgb, p0, p1, color, -1)\n'
                                                                '    return {"img": rgb}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 2. drawing on a transposed view

def _tv_cases():
    out = []
    for seed in (411, 412):
        rng = np.random.default_rng(seed)
        H, W = 36, 52
        img = rng.integers(0, 200, (H, W), dtype=np.uint8)
        x, y = int(rng.integers(12, W - 12)), int(rng.integers(10, H - 10))
        if x == y:
            x += 1
        exp = img.copy()
        cv2.circle(exp, (x, y), 6, 255, -1)
        out.append(Case({"img_t": np.ascontiguousarray(img.T), "x": x, "y": y, "radius": 6}, exp))
    return out


_TV_SIG = "img_t, x, y, radius"
TRANSPOSED_VIEW = Check(
    id="draw_transposed_view", group=GROUP, probes=("draw_on_transposed_view",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2 drawing functions raise on img.T (a transposed view has swapped strides)",
    spec=_spec("The image was saved transposed: img_t[x, y] is the pixel in column x, row y. Draw a filled white "
               "(255) circle centred on column x, row y, and return the image in normal orientation.",
               {"img_t": IOSpec("uint8", "WxH", 2, "transposed grayscale image"),
                "x": IOSpec("int", "scalar", None, "column"), "y": IOSpec("int", "scalar", None, "row"),
                "radius": IOSpec("int", "scalar", None, "px")},
               {"img": IOSpec("uint8", "HxW", 2, "upright image with the circle")}),
    make_cases=_tv_cases, verify=_exact,
    reference=_code(_TV_SIG, '    img = np.ascontiguousarray(img_t.T)\n    cv2.circle(img, (x, y), radius, 255, -1)\n'
                             '    return {"img": img}\n'),
    wrong={
        "draw_on_T": Wrong("cv2.circle(img_t.T, ...)",
                           _code(_TV_SIG, '    img = img_t.T\n    cv2.circle(img, (x, y), radius, 255, -1)\n'
                                          '    return {"img": img}\n'),
                           "error", "Layout of the output array img is incompatible"),
        "draw_then_transpose": Wrong("draws on the transposed array at (x, y), then transposes: circle lands at (y, x)",
                                     _code(_TV_SIG, '    img = img_t.copy()\n    cv2.circle(img, (x, y), radius, 255, -1)\n'
                                                    '    return {"img": np.ascontiguousarray(img.T)}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 3. ROI slices are views: drawing writes through

def _roi_cases():
    out = []
    for seed in (421, 422):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (60, 90, 3), dtype=np.uint8)
        rois = []
        for i in range(3):
            w, h = int(rng.integers(20, 28)), int(rng.integers(20, 40))
            rois.append((4 + i * 29, int(rng.integers(2, 60 - h - 2)), w, h))
        exp = img.copy()
        for x, y, w, h in rois:
            cv2.circle(exp, (x + w // 2, y + h // 2), min(w, h) // 3, (0, 255, 0), -1)
        out.append(Case({"img": img, "rois": np.array(rois, np.int32)}, exp))
    return out


ROI_WRITE_THROUGH = Check(
    id="draw_roi_writethrough", group=GROUP, probes=("draw_on_roi_view",), depends_on_5_0=False, role="gotcha",
    gotcha="an ROI slice is a view: drawing into it writes through to the parent, and drawing into .copy() does not",
    spec=_spec("Mark each window: draw a filled green (0, 255, 0) circle of radius min(w, h) // 3 at the centre of "
               "every (x, y, w, h) window and return the full image.",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
                "rois": IOSpec("int", "Kx4", 2, "windows (x, y, w, h); they do not overlap")},
               {"img": IOSpec("uint8", "HxWx3", 3, "the image with every circle drawn")}),
    make_cases=_roi_cases, verify=_exact,
    reference=_code("img, rois", '    out = img.copy()\n    for x, y, w, h in rois:\n'
                                 '        roi = out[y:y + h, x:x + w]\n'
                                 '        cv2.circle(roi, (w // 2, h // 2), min(w, h) // 3, (0, 255, 0), -1)\n'
                                 '    return {"img": out}\n'),
    wrong={
        "copy_and_forget": Wrong("defensive roi = out[...].copy(): the circle is drawn on a copy nobody pastes back",
                                 _code("img, rois", '    out = img.copy()\n    for x, y, w, h in rois:\n'
                                                    '        roi = out[y:y + h, x:x + w].copy()\n'
                                                    '        cv2.circle(roi, (w // 2, h // 2), min(w, h) // 3, (0, 255, 0), -1)\n'
                                                    '    return {"img": out}\n'), "verifier"),
        "wrong_offset_paste": Wrong("draws in window coordinates on the full image (forgets the window offset)",
                                    _code("img, rois", '    out = img.copy()\n    for x, y, w, h in rois:\n'
                                                       '        cv2.circle(out, (w // 2, h // 2), min(w, h) // 3, (0, 255, 0), -1)\n'
                                                       '    return {"img": out}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 4. CONTROL: numpy int coordinates are fine

def _ni_cases():
    out = []
    for seed in (431, 432):
        rng = np.random.default_rng(seed)
        img = np.zeros((50, 70), np.uint8)
        centers = np.c_[rng.integers(8, 62, 4), rng.integers(8, 42, 4)].astype(np.int64)
        exp = img.copy()
        for cx, cy in centers:
            cv2.circle(exp, (int(cx), int(cy)), 5, 255, -1)
        out.append(Case({"img": img, "centers": centers, "radius": 5}, exp))
    return out


_NI_SIG = "img, centers, radius"
NUMPY_INT_CTRL = Check(
    id="draw_numpy_int_ctrl", group=GROUP, probes=("circle_numpy_int_center",), depends_on_5_0=False, role="control",
    gotcha="CONTROL: numpy int32/int64 centres are accepted as they are (tuple(c), or the array itself); only floats raise",
    spec=_spec("Draw a filled white (255) circle at each given centre.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image"),
                "centers": IOSpec("int64", "Nx2", 2, "(x, y) centres, as np.array([[x, y], ...]) gives them"),
                "radius": IOSpec("int", "scalar", None, "px")},
               {"img": IOSpec("uint8", "HxW", 2, "image with the circles")}),
    make_cases=_ni_cases, verify=_exact,
    reference=_code(_NI_SIG, '    out = img.copy()\n    for c in centers:\n'
                             '        cv2.circle(out, tuple(c), radius, 255, -1)\n    return {"img": out}\n'),
    wrong={
        "overcast_float32": Wrong("over-cautious: 'OpenCV wants float32 points', so tuple(c.astype(np.float32))",
                                  _code(_NI_SIG, '    out = img.copy()\n    for c in centers.astype(np.float32):\n'
                                                 '        cv2.circle(out, tuple(c), radius, 255, -1)\n'
                                                 '    return {"img": out}\n'),
                                  "error", "Can't parse 'center'"),
        "overcast_float": Wrong("over-cautious: tuple(float(v) for v in c)",
                                _code(_NI_SIG, '    out = img.copy()\n    for c in centers:\n'
                                               '        cv2.circle(out, tuple(float(v) for v in c), radius, 255, -1)\n'
                                               '    return {"img": out}\n'),
                                "error", "Can't parse 'center'"),
    })


# ---------------------------------------------------------------- 5. float circle centre

# calibrated on 300 random p/q pairs (r = 14): a circle centred 0 or 1 px off in x and/or y (floor, ceil or round of a
# half-integer midpoint) overlaps the expected one with IoU >= 0.880; a circle at p instead of the midpoint scores at
# most 0.272. 0.80 sits between them.
FC_IOU = 0.80


def _fc_cases():
    out = []
    for seed in (441, 442, 443):
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 200, (80, 120, 3), dtype=np.uint8)  # R < 255 everywhere, so the red disk is the only change
        p = np.array([rng.integers(15, 45), rng.integers(15, 65)], np.int32)
        q = np.array([rng.integers(70, 105), rng.integers(15, 65)], np.int32)
        if (p + q)[0] % 2 == 0 and (p + q)[1] % 2 == 0:
            q[0] += 1  # at least one half-integer coordinate
        disk = np.zeros((80, 120), np.uint8)
        cv2.circle(disk, (int((p[0] + q[0]) // 2), int((p[1] + q[1]) // 2)), 14, 255, -1)
        out.append(Case({"img": img, "p": p, "q": q, "radius": 14}, (disk, img)))
    return out


def _fc_verify(case: Case, out: dict) -> Verdict:
    disk, base = case.truth
    got = arr(out, "img")
    if got.shape != base.shape:
        return fail(f"img has shape {got.shape}")
    changed = (got != base).any(axis=2)
    if not changed.any():
        return fail("nothing was drawn")
    if not (got[changed] == (0, 0, 255)).all():
        return fail("the drawn pixels are not pure red (0, 0, 255) in BGR")
    iou = mask_iou(changed, disk)
    return fail(f"the circle overlaps the expected one with IoU {iou:.2f}") if iou < FC_IOU else PASS


_FC_SIG = "img, p, q, radius"
FLOAT_CENTER = Check(
    id="draw_float_center", group=GROUP, probes=("circle_float_center",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.circle raises on a float centre, even an integral one such as np.round(...) (float64)",
    spec=_spec("Draw a filled red circle centred on the midpoint of the segment from p to q.",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
                "p": IOSpec("int", "2", 1, "(x, y)"), "q": IOSpec("int", "2", 1, "(x, y)"),
                "radius": IOSpec("int", "scalar", None, "px")},
               {"img": IOSpec("uint8", "HxWx3", 3, "image with a filled pure-red (0, 0, 255) circle")}),
    make_cases=_fc_cases, verify=_fc_verify,
    reference=_code(_FC_SIG, '    out = img.copy()\n'
                             '    center = (int(round((p[0] + q[0]) / 2)), int(round((p[1] + q[1]) / 2)))\n'
                             '    cv2.circle(out, center, radius, (0, 0, 255), -1)\n    return {"img": out}\n'),
    wrong={
        "float_midpoint": Wrong("center = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)",
                                _code(_FC_SIG, '    out = img.copy()\n    center = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)\n'
                                               '    cv2.circle(out, center, radius, (0, 0, 255), -1)\n    return {"img": out}\n'),
                                "error", "Can't parse 'center'"),
        "rounded_but_float": Wrong("tuple(np.round((p + q) / 2)): rounded, but still float64",
                                   _code(_FC_SIG, '    out = img.copy()\n    center = tuple(np.round((p + q) / 2))\n'
                                                  '    cv2.circle(out, center, radius, (0, 0, 255), -1)\n    return {"img": out}\n'),
                                   "error", "Can't parse 'center'"),
        "draws_at_p": Wrong("ignores q and draws at p", _code(_FC_SIG, '    out = img.copy()\n'
                                                                       '    cv2.circle(out, (int(p[0]), int(p[1])), radius, (0, 0, 255), -1)\n'
                                                                       '    return {"img": out}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 6. float line endpoint

def _fe_cases():
    out = []
    for seed in (451, 452):
        rng = np.random.default_rng(seed)
        poly = np.array([[30, 20], [110, 25], [125, 70], [70, 95], [25, 65]], np.int32) + rng.integers(-4, 5, (5, 2))
        mid = (poly + np.roll(poly, -1, axis=0)) // 2
        exp = np.zeros((110, 150), np.uint8)
        for i in range(len(mid)):
            cv2.line(exp, tuple(int(v) for v in mid[i]), tuple(int(v) for v in mid[(i + 1) % len(mid)]), 255, 1)
        out.append(Case({"img": np.zeros((110, 150), np.uint8), "poly": poly}, exp))
    return out


def _fe_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "img")
    if got.shape != case.truth.shape:
        return fail(f"img has shape {got.shape}")
    drawn = got > 0
    tube = np.ones((5, 5), np.uint8)  # +-2 px: floor vs round of a half-integer endpoint moves a line by <= 1 px
    near_exp = cv2.dilate(case.truth, tube) > 0
    near_got = cv2.dilate(drawn.astype(np.uint8) * 255, tube) > 0
    if (drawn & ~near_exp).any():
        return fail(f"{int((drawn & ~near_exp).sum())} drawn pixels are more than 2 px from the expected lines")
    if ((case.truth > 0) & ~near_got).any():
        return fail("part of the expected midpoint polygon is missing")
    return PASS


_FE_SIG = "img, poly"
FLOAT_ENDPOINT = Check(
    id="draw_float_endpoint", group=GROUP, probes=("line_float_endpoint",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.line raises on float endpoints, such as the (a + b) / 2 midpoint of two integer vertices",
    spec=_spec("Draw the closed polygon through the midpoints of the polygon's edges, as 1 px white (255) lines.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image (all zeros)"),
                "poly": IOSpec("int", "Nx2", 2, "polygon vertices (x, y), in order")},
               {"img": IOSpec("uint8", "HxW", 2, "image with the midpoint polygon drawn")}),
    make_cases=_fe_cases, verify=_fe_verify,
    reference=_code(_FE_SIG, '    out = img.copy()\n    mid = (poly + np.roll(poly, -1, axis=0)) // 2\n'
                             '    for i in range(len(mid)):\n'
                             '        cv2.line(out, tuple(mid[i]), tuple(mid[(i + 1) % len(mid)]), 255, 1)\n'
                             '    return {"img": out}\n'),
    wrong={
        "float_midpoints": Wrong("mid = (poly + next) / 2 then cv2.line(tuple(mid[i]), ...)",
                                 _code(_FE_SIG, '    out = img.copy()\n    mid = (poly + np.roll(poly, -1, axis=0)) / 2\n'
                                                '    for i in range(len(mid)):\n'
                                                '        cv2.line(out, tuple(mid[i]), tuple(mid[(i + 1) % len(mid)]), 255, 1)\n'
                                                '    return {"img": out}\n'),
                                 "error", "Can't parse 'pt1'"),
        "rounded_but_float": Wrong("np.round(mid): integral values, still float64",
                                   _code(_FE_SIG, '    out = img.copy()\n    mid = np.round((poly + np.roll(poly, -1, axis=0)) / 2)\n'
                                                  '    for i in range(len(mid)):\n'
                                                  '        cv2.line(out, tuple(mid[i]), tuple(mid[(i + 1) % len(mid)]), 255, 1)\n'
                                                  '    return {"img": out}\n'),
                                   "error", "Can't parse 'pt1'"),
        "vertices_not_midpoints": Wrong("connects the vertices themselves",
                                        _code(_FE_SIG, '    out = img.copy()\n'
                                                       '    for i in range(len(poly)):\n'
                                                       '        cv2.line(out, tuple(poly[i]), tuple(poly[(i + 1) % len(poly)]), 255, 1)\n'
                                                       '    return {"img": out}\n'), "verifier"),
    })


CHECKS = [REVERSED_VIEW, TRANSPOSED_VIEW, ROI_WRITE_THROUGH, NUMPY_INT_CTRL, FLOAT_CENTER, FLOAT_ENDPOINT]
