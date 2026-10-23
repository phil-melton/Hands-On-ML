"""Reference pipelines for the 20 E2E tasks: what a competent orchestrator does, built from the families' REFERENCE
solutions run through ``sandbox.run_solution`` (CLAUDE.md Rule 4) plus a few lines of glue.

They are the proof that every task is solvable by composing the verified family solutions, that each truth is right
(the pipeline recovers it from the pixels alone) and that every derived tolerance admits a correct pipeline.
tests/test_e2e.py runs all of them through the grader. They are test and validation tools: nothing in run.py uses them.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from cvbench.families import (chessboard_corners as cb, color_segmentation as cs, components_stats as cc,
                              contour_area as ca, feature_homography as fh, histogram_stats as hi,
                              hough_segments as hs, perspective_rectify as pr, text_layout as tl)
from sandbox.run import run_solution

COLOURS = cs.COLOURS


def run_ref(fam, **kwargs) -> dict:
    """The family's REFERENCE solution, in the sandbox, on one input set."""
    res = run_solution(fam.REFERENCE, [kwargs], timeout_s=120)
    assert res.ok, res.summary()
    return res.outputs[0]


def _L(a) -> list:
    return np.asarray(a).tolist()


def _gray(img) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _otsu(gray, inv=False) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    return cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)[1]


def _class_masks(img) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cls = cs.classify_array(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    return {c: cls == i for i, c in enumerate(COLOURS, 1)}


def _areas_of_mask(mask: np.ndarray) -> dict:
    img = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    return run_ref(ca, img=img)


def _stats_of_pixels(img: np.ndarray, sel: np.ndarray) -> dict:
    """histogram_stats' REFERENCE on the selected pixels only (an n x 1 image of them)."""
    px = np.ascontiguousarray(img[sel].reshape(-1, 1, 3))
    return run_ref(hi, img=px)


def _poly_area(c) -> float:
    c = np.asarray(c, float)
    return 0.5 * abs(float(np.dot(c[:, 0], np.roll(c[:, 1], -1)) - np.dot(c[:, 1], np.roll(c[:, 0], -1))))


def _longest(segs: np.ndarray):
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    k = int(np.argmax(L))
    return segs[k].astype(float), float(L[k])


def _dir(s) -> float:
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180.0


def _board(task, name="board"):
    sx, sy = task.extras["squares"]
    out = run_ref(cb, img=task.images[name], squares_x=sx, squares_y=sy)
    return np.asarray(out["corners"], float), sx, sy


def _pitches(corners, sx, sy):
    g = corners.reshape(sy - 1, sx - 1, 2)
    return float(np.mean(np.linalg.norm(np.diff(g, axis=1), axis=2))), float(np.mean(np.linalg.norm(np.diff(g, axis=0), axis=2)))


def _card_blobs(task, name="photo"):
    ow, oh = task.extras["out_size"]
    out = run_ref(pr, img=task.images[name], out_w=ow, out_h=oh)
    return np.asarray(out["corners"], float), out["warped"]


def reference_answer(task, workdir=None) -> dict:
    """The answer a correct pipeline gives for ``task`` (also writes out.png into ``workdir`` for the text tasks)."""
    return _PIPELINES[task.number](task, workdir)


# ---------------------------------------------------------------- pipelines

def _p01(task, wd):
    img = task.images["photo"]
    c = np.asarray(run_ref(pr, img=img, out_w=300, out_h=200)["corners"], float)
    e = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    top = c[1] - c[0]
    return {"corners": _L(c), "longest_side_px": float(e.max()), "quad_area_px2": _poly_area(c),
            "top_edge_angle_deg": math.degrees(math.atan2(top[1], top[0]))}


def _p02(task, wd):
    corners, warped = _card_blobs(task)
    g = _gray(warped)
    ink = _otsu(g, inv=True) > 0
    out = run_ref(cc, mask=ink, connectivity=8)
    H, W = g.shape
    keep = [i for i, (x, y, w, h) in enumerate(out["boxes"])
            if x > 0 and y > 0 and x + w < W and y + h < H and out["areas"][i] >= 20]
    areas = [int(out["areas"][i]) for i in keep]
    order = sorted(range(len(keep)), key=lambda j: -areas[j])
    big = keep[order[0]]
    return {"corners": _L(corners), "n_blobs": len(keep), "largest_centroid_xy": _L(out["centroids"][big]),
            "area_ratio_largest_to_second": areas[order[0]] / areas[order[1]]}


def _hs_summary(segs):
    return segs.reshape(-1, 4)


def _p03(task, wd):
    segs = _hs_summary(run_ref(hs, img=task.images["strokes"])["segments"])
    s, L = _longest(segs)
    return {"n_strokes": int(len(segs)), "longest_length_px": L, "longest_angle_deg": _dir(s),
            "longest_midpoint_xy": [float((s[0] + s[2]) / 2), float((s[1] + s[3]) / 2)]}


def _p04(task, wd):
    segs = _hs_summary(run_ref(hs, img=task.images["strokes"])["segments"])
    L = np.sort(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))[::-1]
    ang = np.array([_dir(s) for s in segs])
    axis = np.minimum(np.minimum(ang, np.abs(ang - 90.0)), 180.0 - ang)
    return {"lengths_desc_px": _L(L), "total_length_px": float(L.sum()), "n_axis_aligned": int(np.sum(axis < 10.0))}


def _p05(task, wd):
    img = task.images["micrograph"]
    g = _gray(img)
    mask = g > 40
    out = run_ref(cc, mask=mask, connectivity=8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big_box = [int(v) for v in out["boxes"][0]]
    k = next(i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) == int(out["areas"][0])
             and [int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)] == big_box)
    st = _stats_of_pixels(img, labels == k)
    return {"n_components": int(out["count"]), "largest_area": int(out["areas"][0]), "largest_bbox": big_box,
            "largest_mean_gray": float(st["mean"]), "largest_std_gray": float(st["std"]), "largest_median_gray": int(st["median"])}


def _p06(task, wd):
    masks = _class_masks(task.images["parts"])
    area, cnt = {}, {}
    for c in COLOURS:
        o = _areas_of_mask(masks[c])
        area[c], cnt[c] = float(np.sum(o["areas"])), int(o["count"])
    return {"area_px2_by_colour": area, "n_parts_by_colour": cnt, "largest_total_colour": max(COLOURS, key=lambda c: area[c])}


def _p07(task, wd):
    img = task.images["scene"]
    counts = run_ref(cs, img=img)
    counts = {c: int(counts[c]) for c in COLOURS}
    dom = max(COLOURS, key=lambda c: counts[c])
    out = run_ref(cc, mask=_class_masks(img)[dom], connectivity=8)
    return {"counts": counts, "dominant": dom, "n_components": int(out["count"]), "largest_area": int(out["areas"][0]),
            "largest_bbox": [int(v) for v in out["boxes"][0]], "largest_centroid": _L(out["centroids"][0])}


def _p08(task, wd):
    corners, sx, sy = _board(task)
    px, py = _pitches(corners, sx, sy)
    sq_mm = task.extras["square_mm"]
    return {"n_corners": int(len(corners)), "first_corner_xy": _L(corners[0]), "pitch_x_px": px, "pitch_y_px": py,
            "mm_per_px": sq_mm / ((px + py) / 2)}


def _p09(task, wd):
    corners, sx, sy = _board(task)
    g = corners.reshape(sy - 1, sx - 1, 2)
    right, down = g[0, 1] - g[0, 0], g[1, 0] - g[0, 0]
    gray = _gray(task.images["board"])

    def at(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(gray[y - 1:y + 2, x - 1:x + 2].mean())

    tl_dark = at(g[0, 0] - 0.5 * (right + down)) < at(g[0, 0] + 0.5 * (right - down))
    return {"corner_row2_col3_xy": _L(g[2, 3]), "grid_centroid_xy": _L(corners.mean(axis=0)),
            "first_to_last_px": float(np.linalg.norm(g[-1, -1] - g[0, 0])), "top_left_square_is_dark": bool(tl_dark)}


def _p10(task, wd):
    corners, sx, sy = _board(task)
    px, py = _pitches(corners, sx, sy)
    sq_mm = task.extras["square_mm"]
    s = sq_mm / ((px + py) / 2)
    o = _areas_of_mask(_class_masks(task.images["parts"])["red"])
    return {"mm_per_px": s, "n_red_parts": int(o["count"]), "red_area_mm2": float(np.sum(o["areas"])) * s * s}


def _fh_H(task):
    H = np.asarray(run_ref(fh, img_a=task.images["img_a"], img_b=task.images["img_b"])["H"], float)
    return H / H[2, 2]


def _p11(task, wd):
    from serve.e2e.tasks import ROI
    H = _fh_H(task)
    m = fh.project(H, ROI)
    return {"H": _L(H), "roi_corners_b": _L(m), "roi_area_b_px2": _poly_area(m), "roi_centroid_b": _L(m.mean(axis=0))}


def _p12(task, wd):
    from serve.e2e.tasks import ROI
    m = fh.project(_fh_H(task), ROI)
    va, vb = ROI[1] - ROI[0], m[1] - m[0]
    return {"rotation_deg": math.degrees(math.atan2(vb[1], vb[0])), "edge_scale": float(np.linalg.norm(vb) / np.linalg.norm(va)),
            "roi_centre_b": _L(m.mean(axis=0))}


def _field(task, wd):
    sc = task.extras["tl_scene"]
    img = task.images["field"]
    ys, xs = np.nonzero(np.any(img != img[0, 0], axis=2))
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    box = [int(x0 + 1), int(y0 + 1), int(x1 - x0 - 1), int(y1 - y0 - 1)]
    out = run_ref(tl, img=img, box=box, text=sc.text, face=int(sc.face), scale=float(sc.scale),
                  thickness=int(sc.thickness), color=[int(c) for c in sc.color])
    if wd is not None:
        cv2.imwrite(str(Path(wd) / "out.png"), out["img"])
    return {"box": box, "scale": float(out["scale"]), "text_size": [int(v) for v in out["text_size"]]}


def _p15(task, wd):
    st = run_ref(hi, img=task.images["image"])
    h = np.asarray(st["hist"], np.int64)
    lv = np.arange(256)
    return {"mean_gray": float(st["mean"]), "std_gray": float(st["std"]), "median_gray": int(st["median"]),
            "n_outliers": int(h[np.abs(lv - st["mean"]) > 2 * st["std"]].sum())}


def _p16(task, wd):
    a, b = run_ref(hi, img=task.images["a"]), run_ref(hi, img=task.images["b"])
    return {"mean_a": float(a["mean"]), "mean_b": float(b["mean"]), "median_a": int(a["median"]), "median_b": int(b["median"]),
            "brighter": "a" if a["mean"] > b["mean"] else "b", "higher_contrast": "a" if a["std"] > b["std"] else "b"}


def _p17(task, wd):
    out = run_ref(ca, img=task.images["shapes"])
    a = np.sort(np.asarray(out["areas"], float))[::-1]
    return {"n_shapes": int(out["count"]), "areas_desc_px2": _L(a), "n_large": int(np.sum(a > task.extras["threshold"])),
            "largest_share": float(a[0] / a.sum())}


def _p18(task, wd):
    mask = task.images["mask"] > 0
    out = run_ref(cc, mask=mask, connectivity=8)
    H, W = mask.shape
    boxes = np.asarray(out["boxes"])
    border = sum(1 for x, y, w, h in boxes if x == 0 or y == 0 or x + w == W or y + h == H)
    return {"n_components": int(out["count"]), "largest_area": int(out["areas"][0]), "largest_bbox": [int(v) for v in boxes[0]],
            "largest_centroid": _L(out["centroids"][0]), "n_touching_border": int(border),
            "n_single_pixel": int(np.sum(np.asarray(out["areas"]) == 1))}


def _p19(task, wd):
    corners, warped = _card_blobs(task)
    g = _gray(warped).copy()
    b = int(0.05 * min(g.shape))
    g[:b], g[-b:], g[:, :b], g[:, -b:] = 255, 255, 255, 255  # the paper margin: nothing the bg leaked in survives
    strokes = cv2.cvtColor(_otsu(g, inv=True), cv2.COLOR_GRAY2BGR)
    segs = _hs_summary(run_ref(hs, img=strokes)["segments"])
    s, L = _longest(segs)
    return {"corners": _L(corners), "n_strokes": int(len(segs)), "longest_length_px": L, "longest_angle_deg": _dir(s)}


def _p20(task, wd):
    img = task.images["scene"]
    counts = run_ref(cs, img=img)
    masks = _class_masks(img)
    mean = {c: float(_stats_of_pixels(img, masks[c])["mean"]) for c in COLOURS}
    return {"counts": {c: int(counts[c]) for c in COLOURS}, "mean_gray_by_colour": mean,
            "brightest": max(COLOURS, key=lambda c: mean[c]), "darkest": min(COLOURS, key=lambda c: mean[c])}


_PIPELINES = {1: _p01, 2: _p02, 3: _p03, 4: _p04, 5: _p05, 6: _p06, 7: _p07, 8: _p08, 9: _p09, 10: _p10, 11: _p11,
              12: _p12, 13: _field, 14: _field, 15: _p15, 16: _p16, 17: _p17, 18: _p18, 19: _p19, 20: _p20}
