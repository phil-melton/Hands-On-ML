"""Shared helpers for the Stage 7 E2E tasks: exact labelling, homography algebra, card photos with known content,
and tolerance propagation.

Nothing here calls OpenCV to compute a truth. OpenCV is used for rendering only (Gaussian PSF blur), exactly as the
families do; every answer the tasks grade is a function of integers or of the manufactured geometry.

Tolerance propagation (used by tasks.py, one rule per quantity; each is a worst-case bound, not a statistic)
-------------------------------------------------------------------------------------------------------------
* A quantity that is a function of reported points inherits the family's per-point tolerance ``d`` by the
  Lipschitz constant of the function:
    mean of points                       <= d
    distance between two points          <= 2 d
    mean of such distances                <= 2 d
    direction (mod 180) of a segment L px <= asin(2 d / L)               (|e| <= 2 d on the vector, angle <= asin(|e| / L))
    shoelace area of a polygon            <= d/2 * sum_i |v[i+1] - v[i-1]| + n d^2 / 2   (exact expansion of the
                                             shoelace sum in the displacements e_i: the cross terms are bounded by |e|^2)
    ratio / quotient                      <= interval arithmetic on the numerator and denominator ranges
* Where a point is *transported* through a map that the reported corners define (rectification), the bound is the
  worst case of the transported quantity over all corner displacements of size <= d. ``worst_case`` evaluates the
  vertices of the regular octagon that circumscribes the disc of radius d at each of the 4 corners (8**4 = 4096
  combinations). The transported quantity is smooth and almost linear in the displacements over d << the card size,
  so its extreme over the disc is at the boundary and the circumscribed octagon bounds it.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from cvbench.families import perspective_rectify as pr

SEED_RANGE = (6_000_000, 7_000_000)  # the E2E scene seeds: disjoint from dev, test and foundry (tests pin it)


# ---------------------------------------------------------------- exact labelling

@dataclass(frozen=True)
class Comp:
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

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


def label_components(mask, connectivity: int = 8) -> tuple[np.ndarray, list[Comp]]:
    """Flood-fill labelling of a bool mask (no OpenCV). Labels are 1.. in raster order of the first pixel; the
    components come back in the same order. Area, box and centroid are exact integer sums with one division."""
    m = np.asarray(mask, bool)
    H, W = m.shape
    labels = np.zeros((H, W), np.int32)
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if connectivity == 8 else ())
    comps: list[Comp] = []
    ys0, xs0 = np.nonzero(m)
    for y0, x0 in zip(ys0.tolist(), xs0.tolist()):
        if labels[y0, x0]:
            continue
        k = len(comps) + 1
        labels[y0, x0] = k
        stack, xs, ys = [(y0, x0)], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = k
                    stack.append((yy, xx))
        n = len(xs)
        comps.append(Comp(n, min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, sum(xs) / n, sum(ys) / n))
    return labels, comps


# ---------------------------------------------------------------- gray level (BT.601), exact

def luma_exact(bgr) -> tuple[int, float]:
    """round-half-up of 0.299 R + 0.587 G + 0.114 B in exact integer arithmetic, and the distance of the fractional
    part from the rounding tie (0.5): colours with a distance below ~0.003 are where cv2.cvtColor can differ."""
    b, g, r = (int(v) for v in bgr)
    num = 299 * r + 587 * g + 114 * b
    return (num + 500) // 1000, abs((num % 1000) / 1000.0 - 0.5)


# ---------------------------------------------------------------- planar geometry

def homography_dlt(src, dst) -> np.ndarray:
    """3x3 H with H @ src_i ~ dst_i for four point pairs (direct linear solve, float64)."""
    A, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(Hm, pts) -> np.ndarray:
    q = np.asarray(pts, float) @ np.asarray(Hm, float)[:, :2].T + np.asarray(Hm, float)[:, 2]
    return q[..., :2] / q[..., 2:3]


def poly_area(v) -> float:
    v = np.asarray(v, float)
    return 0.5 * abs(float(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1))))


def poly_perimeter(v) -> float:
    v = np.asarray(v, float)
    return float(np.sum(np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)))


def area_tol(v, d: float) -> float:
    """Bound on |shoelace(v') - shoelace(v)| when every vertex moves by <= d (see the module docstring)."""
    v = np.asarray(v, float)
    n = len(v)
    return 0.5 * d * float(np.sum(np.linalg.norm(np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0), axis=1))) + 0.5 * n * d * d


def direction_deg(p1, p2) -> float:
    """Direction of the segment p1-p2 as an axis, degrees in [0, 180) (x right, y down)."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0


def angle_tol_deg(length: float, d: float) -> float:
    """Direction error bound, degrees, for a segment of ``length`` whose two ends are each off by <= d."""
    return math.degrees(math.asin(min(1.0, 2.0 * d / length)))


def seg_distance(a0, a1, b0, b1) -> float:
    """Minimum distance between the segments a0-a1 and b0-b1 (0 when they cross)."""
    a0, a1, b0, b1 = (np.asarray(p, float) for p in (a0, a1, b0, b1))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_seg(p, a, b):
        v = b - a
        s = float(np.clip(np.dot(p - a, v) / np.dot(v, v), 0.0, 1.0))
        return float(np.linalg.norm(p - (a + s * v)))

    if cross(a0, a1, b0) * cross(a0, a1, b1) < 0 and cross(b0, b1, a0) * cross(b0, b1, a1) < 0:
        return 0.0
    return min(point_seg(a0, b0, b1), point_seg(a1, b0, b1), point_seg(b0, a0, a1), point_seg(b1, a0, a1))


def ramanujan(a: float, b: float) -> float:
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def ratio_tol(num: float, num_tol: float, den: float, den_tol: float) -> float:
    """Largest |n'/d' - n/d| for n' in [n -+ num_tol], d' in [d -+ den_tol] (interval arithmetic, d' > 0)."""
    lo_d, hi_d = max(den - den_tol, 1e-12), den + den_tol
    cand = [(num + sn * num_tol) / dd for sn in (-1, 1) for dd in (lo_d, hi_d)]
    return max(abs(c - num / den) for c in cand)


# ---------------------------------------------------------------- corner-displacement propagation

def octagon(d: float) -> np.ndarray:
    """Vertices of the regular octagon that circumscribes the disc of radius d."""
    R = d / math.cos(math.pi / 8.0)
    ang = math.pi / 8.0 + np.arange(8) * math.pi / 4.0
    return R * np.c_[np.cos(ang), np.sin(ang)]


def worst_case(corners, d: float, fn) -> float:
    """max over corner displacements of size <= d of || fn(displaced corners) - fn(corners) ||."""
    corners = np.asarray(corners, float)
    ref = np.asarray(fn(corners), float).ravel()
    offs = octagon(d)
    worst = 0.0
    for combo in itertools.product(range(8), repeat=4):
        v = np.asarray(fn(corners + offs[list(combo)]), float).ravel()
        worst = max(worst, float(np.linalg.norm(v - ref)))
    return worst


# ---------------------------------------------------------------- card photos with known content

def card_quad(scene) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Card-frame corner coordinates C (centred, card px), their photo positions P, and ``idx`` such that P[idx] are
    the position-ordered (TL, TR, BR, BL) corners the family reports."""
    cw, ch = scene.card
    C = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    P = apply_h(scene.H_ci, C)
    idx, _ = pr.position_order(P)
    assert np.allclose(P[idx], scene.corners), "card geometry disagrees with the family's corners"
    return C, P, idx


def rect_dst(ow: int, oh: int) -> np.ndarray:
    return np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], float)


def card_to_rect(scene) -> np.ndarray:
    """Homography from card-frame (centred, card px) coordinates to the rectified image the contract defines:
    photo corners TL, TR, BR, BL (by image position) go to the corner pixels of the out_w x out_h output."""
    C, _, idx = card_quad(scene)
    return homography_dlt(C[idx], rect_dst(scene.out_w, scene.out_h))


def ink_mask(X, Y, discs=(), strokes=()) -> np.ndarray:
    """True where the card coordinates (X, Y) lie on ink: inside a disc (cx, cy, r) or within w/2 of a stroke
    segment (x1, y1, x2, y2, w), i.e. a capsule, the shape cv2.line draws with round caps."""
    m = np.zeros(np.shape(X), bool)
    for cx, cy, r in discs:
        m |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    for x1, y1, x2, y2, w in strokes:
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((X - x1) * dx + (Y - y1) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        m |= (X - (x1 + t * dx)) ** 2 + (Y - (y1 + t * dy)) ** 2 <= (w / 2.0) ** 2
    return m


def render_card_photo(scene, rng, *, paper, ink, discs=(), strokes=()) -> np.ndarray:
    """The family's photo of the card (same pose, blur, noise level, size) carrying ``paper`` with ``ink`` shapes
    printed on it. 4x4 area-coverage rendering, then the lens PSF and sensor noise, as ``perspective_rectify._render``."""
    H, W = pr.IMG_H, pr.IMG_W
    _, P, _ = card_quad(scene)
    Hinv = np.linalg.inv(scene.H_ci)
    paper, ink = np.asarray(paper, float), np.asarray(ink, float)
    bg0, tint = rng.uniform(18.0, 42.0), rng.uniform(-4.0, 4.0, 3)
    gx, gy = rng.uniform(-14.0, 14.0, 2)
    noise_field = rng.standard_normal((H, W, 3))
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    img = (bg0 + gx * (xs / W - 0.5) + gy * (ys / H - 0.5))[..., None] + tint
    x0, x1 = max(int(math.floor(P[:, 0].min())) - 3, 0), min(int(math.ceil(P[:, 0].max())) + 3, W - 1)
    y0, y1 = max(int(math.floor(P[:, 1].min())) - 3, 0), min(int(math.ceil(P[:, 1].max())) + 3, H - 1)
    X, Y = xs[y0:y1 + 1, x0:x1 + 1], ys[y0:y1 + 1, x0:x1 + 1]
    sgn = 1.0 if np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)) > 0 else -1.0
    offs = (np.arange(4) + 0.5) / 4 - 0.5
    cov, inkc = np.zeros(X.shape), np.zeros(X.shape)
    for oy in offs:
        for ox in offs:
            inside = np.ones(X.shape, bool)
            for i in range(4):
                (ax, ay), (bx, by) = P[i], P[(i + 1) % 4]
                inside &= sgn * ((bx - ax) * (Y + oy - ay) - (by - ay) * (X + ox - ax)) >= 0
            q = apply_h(Hinv, np.stack([X + ox, Y + oy], axis=-1))
            cov += inside
            inkc += inside & ink_mask(q[..., 0], q[..., 1], discs, strokes)
    cov /= 16.0
    inkc /= 16.0
    frac = (inkc / np.where(cov > 0, cov, 1.0))[..., None]
    card = paper * (1.0 - frac) + ink * frac
    c3 = cov[..., None]
    img[y0:y1 + 1, x0:x1 + 1] = c3 * card + (1.0 - c3) * img[y0:y1 + 1, x0:x1 + 1]
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), scene.blur, borderType=cv2.BORDER_REPLICATE)
    return np.clip(np.rint(img.astype(np.float64) + scene.noise * noise_field), 0, 255).astype(np.uint8)
