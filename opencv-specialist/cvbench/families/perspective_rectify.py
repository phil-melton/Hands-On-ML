"""Family ``perspective_rectify``: find a card's four corners and warp it flat, verified by a manufactured solution.

The answer is chosen first: a flat rectangular card of known size carrying a known (analytic) texture,
seen by a pinhole camera under a known pose. The camera pose gives the homography, the homography gives
the four true corners (continuous, sub-pixel, pixel centres at integer coordinates) and the true rectified
image (the texture sampled on the output grid); the photo is rendered from them (4x4 area coverage, lens
PSF blur, sensor noise). The code under test must recover both within a tolerance derived from the physics
of the measurement.

Contract choices
----------------
* Corner order is fixed by *image position*, not by what is printed on the card: TL has the smallest x+y,
  BR the largest x+y, TR the smallest y-x, BL the largest y-x. The rule is unambiguous: generation keeps
  GAP = 30 px between the winner and the runner-up of each of the four tests. A corner error e moves x+y and
  y-x of a corner by at most sqrt(2)*e, so it can close a gap by at most 2*sqrt(2)*e; for the largest corner
  tolerance, TOL_A + TOL_B*BLUR_RANGE[1] = 6.4 + 0.75*2.2 = 8.05 px, that is 22.8 px < 30 px, so a corner
  error cannot flip a label (the smallest gap generated on 300 fresh scenes, seeds 7_300_000..7_300_299, is
  35.4 px). It is also what a document scanner does. Scenes roll the card by a multiple of 90 degrees plus
  up to 25 degrees, so the printed top of the card is anywhere and the rectified image is "as the photo
  sees it".
* ``out_w`` and ``out_h`` are inputs (never equal: |out_w - out_h| >= 24, so swapping them shows). The four
  corners map to the four corner pixels of the output: corners[0] -> warped[0, 0], corners[1] ->
  warped[0, out_w-1], corners[2] -> warped[out_h-1, out_w-1], corners[3] -> warped[out_h-1, 0].
* The card has a bright paper margin (MU = 10 % of each side) around a coloured, asymmetric texture, so the
  silhouette is a clean bright quad on a dark background while the interior still tells right from
  mirrored, rotated or channel-swapped.

Tolerance physics (written before calibrating; the measured outcome is in the comment at TOL_A)
-----------------------------------------------------------------------------------------------
Corners. The image of an ideal edge is a step convolved with the pixel footprint and the lens PSF
(Gaussian, sigma). Its 50 % crossing sits on the true edge, so edges are unbiased; corners are not:
(a) a contour traces whole pixels, centred up to 1/2 px inside the boundary; (b) the iso-contour of a
blurred wedge is rounded, so the vertex of a polygon fitted to the contour is shaved inwards by a length
proportional to sigma (a Gaussian-blurred right-angle quadrant is 1/4 at its vertex; its 1/2 level reaches the
bisector 0.77 sigma inside); (c) Otsu puts the threshold at 35-50 % of the edge profile, not 50 %, which moves
every edge by up to ~sigma/2; (d) a polygon approximation may put the vertex anywhere on the rounded arc.
Predicted law: contour-polygon estimators err by up to a + b*sigma (a from (a), b from (b)-(d)); an estimator
that intersects fitted edge lines never sees (b)-(d) and sits far inside that band. The corner tolerance is
TOL_A + TOL_B*sigma, sigma being the manufactured blur.

Rectified image. It is a warp of the same pixels, so a registration error d (output px) costs a normalised
RMS error NRMSE ~ k_rms * d, where k_rms is the RMS spatial frequency of the texture in output pixels
(measured by calibrate()), on top of a floor from sensor noise and from the blur that softens the texture.
Registration has two legitimate sources: the corner error above, and the output-pixel convention (corners onto
pixel centres, 0..W-1, or onto pixel edges, 0..W: up to 1 px at the far edge). The corner error is judged
once, on the corners; the image is judged against the texture rectified *from the reported corners*, so that
it measures the warp (convention, interpolation, order, channels, shape) and does not charge the corner error
twice. NRMSE is computed on the central 80 % x 80 % of the output, the texture region: the outermost rows and
columns sample the card edge, where the blurred silhouette mixes in the background, which is the corner
error's business. Pixel-exact comparison is useless: 5.0 changed warpPerspective/warpAffine by up to 4 gray
levels on ~3 % of pixels (probes/results/diff_4.14_vs_5.0.md), and nearest-neighbour is a legitimate choice.

Metamorphic checks
------------------
Shifting the photo moves the corners by the shift and leaves the rectified image unchanged; rotating by 90
degrees maps the corner set by (x, y) -> (y, W-1-x) and rotates the rectified image the same way (out_w and
out_h swap); mirroring maps x -> W-1-x and mirrors the rectified image; dimming leaves the corners unchanged
and scales the rectified image. Corner sets are compared *as sets*; the labels follow the position rule, which
the check verifies separately on every variant. The photo size is a fixed constant (IMG_H x IMG_W), which is
all the covariant checks need: evaluate_solution calls verify_metamorphic without the scene. Blind spot: a
constant output (a zero warp) is invariant under all of them, which is why the manufactured gate stays.
"""
from __future__ import annotations

import inspect
import itertools
import math
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

SPEC = TaskSpec(
    family="perspective_rectify",
    task=("The photo shows one flat rectangular card (bright paper with a printed picture) lying on a darker "
          "surface, seen from an arbitrary angle and rotation. Find the card's four corners and rectify the "
          "card: warp it to an upright out_w x out_h image. Report the corners and the rectified image."),
    inputs={
        "img": IOSpec("uint8", "HxWx3", 3, "BGR photo; one bright card on a dark background, entirely inside "
                                           "the frame, in any in-plane rotation"),
        "out_w": IOSpec("int", "scalar", None, "width of the rectified image in px (never equal to out_h)"),
        "out_h": IOSpec("int", "scalar", None, "height of the rectified image in px"),
    },
    outputs={
        "corners": IOSpec("float", "4x2", 2, "card corners as (x, y) px in the photo, pixel centres at integer "
                                             "coordinates; row order TL, TR, BR, BL by IMAGE position: TL has "
                                             "the smallest x+y, BR the largest x+y, TR the smallest y-x, BL "
                                             "the largest y-x"),
        "warped": IOSpec("uint8", "out_hxout_wx3", 3, "BGR; the card rectified so that corners[0] lands on "
                                                      "warped[0, 0], corners[1] on warped[0, out_w-1], "
                                                      "corners[2] on warped[out_h-1, out_w-1] and corners[3] "
                                                      "on warped[out_h-1, 0]"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Document-scanner step. A bright rectangular sheet lies on a dark desk in this BGR photo. Locate the "
    "sheet's four corners in image coordinates and warp the sheet into a flat, upright out_w x out_h picture.",
    "Given a photo of a rectangular card on a dark background, return where its four corners are and the card "
    "rectified (perspective-corrected) to out_w x out_h pixels, corners mapped onto the output's corners.",
    "Straighten the card: detect its quadrilateral outline, order the corners top-left, top-right, "
    "bottom-right, bottom-left by their position in the photo, and warp the quadrilateral onto an "
    "out_w x out_h rectangle.",
]

IMG_H, IMG_W = 360, 480  # fixed photo size of every manufactured scene (the covariant checks rely on it)
BORDER = 16.0  # px: the true corners stay this far inside the frame
MU = 0.10  # paper margin and unjudged band, as a fraction of each side
RAMP = 0.04  # width of the smooth margin -> picture transition, fraction of a side
ROLL_MAX = 25.0  # degrees of in-plane roll on top of a multiple of 90
GAP = 30.0  # px: best vs second-best corner for each position test (>= 2*sqrt(2) * the largest corner tolerance, 22.8 px)
ANGLE_RANGE = (55.0, 125.0)  # interior angles of the card as photographed, degrees
MIN_EDGE = 100.0  # px: shortest side of the photographed card
AREA_RANGE = (0.15, 0.70)  # card area as a fraction of the photo
TILT_MAX = 40.0  # degrees of out-of-plane tilt
BLUR_RANGE = (0.6, 2.2)  # Gaussian PSF sigma, px
NOISE_RANGE = (0.0, 4.0)  # sensor noise sigma, gray levels

# --- calibrated constants ----------------------------------------------------------------------------------
# calibrate(n=2000, seed=500_000): 2000 single-card gauge blocks, 8000 corners, OpenCV 5.0.0.
#   corner error, px      mean   p99   max    max by sigma bin [0.6-0.9, 0.9-1.2, 1.2-1.5, 1.5-1.8, 1.8-2.2]
#   reference (4-gon)     1.29   2.82  5.92   [3.34, 3.46, 5.92, 4.27, 4.60]
#   extreme contour px    1.44   3.66  5.68   [5.32, 3.47, 5.11, 4.84, 5.68]
#   line fit              0.24   0.63  1.15   [1.15, 1.10, 0.83, 0.87, 0.84]
#   (info) approxPolyDP at a fixed 2 % epsilon: identical, except one vertex slid 16.5 px (a 107-degree corner
#   whose protrusion is smaller than epsilon): an algorithmic error, correctly flagged.
# The mean error of the contour estimators does grow with sigma (OLS slope 0.59 px per px of sigma: 0.95 px at
# sigma 0.75 to 1.6 px at 2.0), but the worst case does not follow it: it is a sigma-independent a ~ 5.1 px
# (vertex slide along a rounded, obtuse corner; staircase on a near-45 degree edge), so the draft law a + b*sigma
# holds with a large a. With the 25 % margin used throughout: TOL_A = 1.25 * 5.06, TOL_B = 1.25 * 0.59.
TOL_A, TOL_B = 6.4, 0.75
# Rectified image, NRMSE against the texture rectified from the *reported* corners (judged region only). Worst
# over all gauge blocks: true corners 0.137, same with the pixel-edge convention 0.179; reference 0.134, line fit
# 0.137, extremes 0.132 (pixel-edge convention: 0.178 / 0.178 / 0.176); reference with INTER_NEAREST 0.157 (and
# pixel-edge convention 0.188, the worst case), INTER_CUBIC 0.139. Means 0.06-0.12. Registration costs
# k_rms * d with k_rms = 0.124 rad per output px (measured), i.e. ~0.12 per px. 0.188 * 1.25 = 0.235.
TOL_IMG = 0.24
# measure_metamorphic_drift(n=300, seed=50_000): the standard estimators are not exactly covariant (Otsu's
# threshold moves by a gray level when the histogram or the quantisation changes; polygon vertices depend on where
# the contour starts). Largest corner-set movement against the symmetry: reference 2.24 (shift), 4.24 (rot90),
# 5.00 (mirror), 2.24 (dim); extremes 2.24 / 2.83 / 4.24 / 2.24; line fit <= 0.25. Largest image drift (NRMSE):
# reference 0.214 (rot90), extremes 0.231 (mirror), line fit 0.027. No estimator ever broke the position order.
# 5.0 * 1.25 = 6.25 and 0.231 * 1.25 = 0.289.
META_TOL_CORNER = 6.3
META_TOL_IMG = 0.29


@dataclass
class Scene:
    img: np.ndarray  # (360, 480, 3) uint8 BGR
    corners: np.ndarray  # (4, 2) float64: true corners, TL TR BR BL by image position
    out_w: int
    out_h: int
    blur: float  # sigma of the Gaussian PSF, px
    noise: float  # sigma of the sensor noise, gray levels
    card: tuple[float, float]  # physical card size (width, height) in card px: the texture's units
    H_ci: np.ndarray  # 3x3 homography: card-centred card px -> photo px
    tex: dict  # texture parameters
    tilt_deg: float
    roll_deg: float
    seed: int


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


CORNER_NAMES = ("TL", "TR", "BR", "BL")


# ---------------------------------------------------------------- geometry

def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with H @ src_i ~ dst_i for four point pairs (direct linear solve, float64)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def _apply(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.asarray(pts, float) @ Hm[:, :2].T + Hm[:, 2]
    return q[..., :2] / q[..., 2:3]


def position_order(pts: np.ndarray) -> tuple[list[int], float]:
    """Indices of the TL, TR, BR, BL corners by the contract's image-position rule, and the smallest gap
    between a winner and the runner-up over the four tests."""
    pts = np.asarray(pts, float)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    idx = [int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    ss, dd = np.sort(s), np.sort(d)
    return idx, float(min(ss[1] - ss[0], ss[3] - ss[2], dd[1] - dd[0], dd[3] - dd[2]))


def _rodrigues(axis, ang: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def interior_angles(P: np.ndarray) -> list[float]:
    out = []
    for i in range(4):
        a, b = P[i - 1] - P[i], P[(i + 1) % 4] - P[i]
        out.append(math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))))
    return out


# ---------------------------------------------------------------- texture: the manufactured picture

def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _texture(tex: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Card colour (BGR, float) at normalised card coordinates a (along the width), b (along the height)."""
    cw, ch = tex["card"]
    X, Y = (a - 0.5) * cw, (b - 0.5) * ch
    out = np.empty(a.shape + (3,))
    J = tex["freq"].shape[1]
    w = (_smooth((a - MU) / RAMP) * _smooth((1 - a - MU) / RAMP)
         * _smooth((b - MU) / RAMP) * _smooth((1 - b - MU) / RAMP))
    for c in range(3):
        g = np.zeros(a.shape)
        for j in range(J):
            fx, fy = tex["freq"][c, j]
            g += np.cos(2 * math.pi * (fx * X + fy * Y) + tex["phase"][c, j])
        g *= math.sqrt(2.0 / J)
        for amp, (bx, by), s in zip(tex["blob_amp"][c], tex["blob_pos"], tex["blob_sigma"]):
            g += amp * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * s * s))
        out[..., c] = tex["paper"][c] + w * (165.0 + 70.0 * np.tanh(0.9 * g) - tex["paper"][c])
    return out


def _make_texture(rng, cw: float, ch: float) -> dict:
    J = 10
    ang = rng.uniform(0, 2 * math.pi, (3, J))
    lam = rng.uniform(36.0, 140.0, (3, J))  # wavelengths in card px
    return {
        "card": (cw, ch),
        "freq": np.stack([np.cos(ang) / lam, np.sin(ang) / lam], axis=-1),
        "phase": rng.uniform(0, 2 * math.pi, (3, J)),
        "blob_amp": rng.choice([-1.0, 1.0], (3, 2)) * rng.uniform(0.8, 1.6, (3, 2)),
        "blob_pos": np.c_[rng.uniform(-0.3, 0.3, 2) * cw, rng.uniform(-0.3, 0.3, 2) * ch],
        "blob_sigma": rng.uniform(18.0, 30.0, 2),
        "paper": rng.uniform(225.0, 248.0, 3),
    }


# ---------------------------------------------------------------- scene generation

def make_scene(seed: int, *, blur: float | None = None, noise: float | None = None,
               tilt_max: float = TILT_MAX) -> Scene:
    """Manufacture a photo of one card. ``blur``/``noise`` override the sampled values (gauge-block sweeps)."""
    rng = np.random.default_rng(seed)
    blur_d, noise_d = rng.uniform(*BLUR_RANGE), rng.uniform(*NOISE_RANGE)
    blur = blur_d if blur is None else float(blur)
    noise = noise_d if noise is None else float(noise)
    cx, cy = (IMG_W - 1) / 2, (IMG_H - 1) / 2

    for _ in range(20000):
        long_ = rng.uniform(190.0, 300.0)
        short = long_ * rng.uniform(0.46, 0.8)
        if short < 130.0:
            continue
        cw, ch = (long_, short) if rng.random() < 0.5 else (short, long_)
        F = rng.uniform(380.0, 720.0)  # focal length, px
        Z = F / rng.uniform(0.7, 1.25)  # distance such that the card is 0.7-1.25 photo px per card px
        tilt = math.radians(rng.uniform(0.0, tilt_max))
        phi = rng.uniform(0, 2 * math.pi)  # azimuth of the tilt axis
        roll = math.radians(int(rng.integers(4)) * 90.0 + rng.uniform(-ROLL_MAX, ROLL_MAX))
        dx, dy = rng.uniform(-70, 70), rng.uniform(-45, 45)
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        R = Rz @ _rodrigues([math.cos(phi), math.sin(phi), 0], tilt)
        t = np.array([dx * Z / F, dy * Z / F, Z])
        H_ci = np.array([[F, 0, cx], [0, F, cy], [0, 0, 1.0]]) @ np.column_stack([R[:, 0], R[:, 1], t])
        P = _apply(H_ci, np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]]))
        if P.min() < BORDER or P[:, 0].max() > IMG_W - 1 - BORDER or P[:, 1].max() > IMG_H - 1 - BORDER:
            continue
        edges = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        area = 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))
        ang = interior_angles(P)
        idx, gap = position_order(P)
        if (edges.min() < MIN_EDGE or not (AREA_RANGE[0] <= area / (IMG_W * IMG_H) <= AREA_RANGE[1])
                or min(ang) < ANGLE_RANGE[0] or max(ang) > ANGLE_RANGE[1]
                or len(set(idx)) < 4 or gap < GAP):
            continue
        # photo-oriented size: which card side runs from the photo's TL corner to its TR corner?
        w_phys, h_phys = (cw, ch) if {idx[0], idx[1]} in ({0, 1}, {2, 3}) else (ch, cw)
        r = rng.uniform(0.6, 1.15)
        out_w, out_h = int(round(r * w_phys)), int(round(r * h_phys))
        if abs(out_w - out_h) < 24:
            continue
        break
    else:  # pragma: no cover
        raise RuntimeError(f"no valid geometry for seed {seed}")

    tex = _make_texture(rng, cw, ch)
    bg0, bg_tint = rng.uniform(18.0, 42.0), rng.uniform(-4.0, 4.0, 3)
    gx, gy = rng.uniform(-14.0, 14.0, 2)
    noise_field = rng.standard_normal((IMG_H, IMG_W, 3))
    img = _render(P, H_ci, tex, bg0, bg_tint, gx, gy, blur, noise, noise_field)
    return Scene(img, P[idx], out_w, out_h, blur, noise, (cw, ch), H_ci, tex,
                 math.degrees(tilt), math.degrees(roll), seed)


def _render(P, H_ci, tex, bg0, bg_tint, gx, gy, blur, noise, noise_field) -> np.ndarray:
    """Area-coverage (4x4 supersampled) rendering of the card over a dark gradient, then PSF blur and noise."""
    cw, ch = tex["card"]
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W].astype(float)
    img = (bg0 + gx * (xs / IMG_W - 0.5) + gy * (ys / IMG_H - 0.5))[..., None] + bg_tint
    x0, x1 = max(int(math.floor(P[:, 0].min())) - 3, 0), min(int(math.ceil(P[:, 0].max())) + 3, IMG_W - 1)
    y0, y1 = max(int(math.floor(P[:, 1].min())) - 3, 0), min(int(math.ceil(P[:, 1].max())) + 3, IMG_H - 1)
    X, Y = xs[y0:y1 + 1, x0:x1 + 1], ys[y0:y1 + 1, x0:x1 + 1]
    sgn = 1.0 if np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)) > 0 else -1.0
    offs = (np.arange(4) + 0.5) / 4 - 0.5
    cov = np.zeros(X.shape)
    for oy in offs:
        for ox in offs:
            inside = np.ones(X.shape, bool)
            for i in range(4):
                (ax, ay), (bx, by) = P[i], P[(i + 1) % 4]
                inside &= sgn * ((bx - ax) * (Y + oy - ay) - (by - ay) * (X + ox - ax)) >= 0
            cov += inside
    cov = cov[..., None] / 16.0
    q = _apply(np.linalg.inv(H_ci), np.stack([X, Y], axis=-1))
    card = _texture(tex, q[..., 0] / cw + 0.5, q[..., 1] / ch + 0.5)
    img[y0:y1 + 1, x0:x1 + 1] = cov * card + (1 - cov) * img[y0:y1 + 1, x0:x1 + 1]
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur, borderType=cv2.BORDER_REPLICATE)
    return np.clip(np.rint(img.astype(np.float64) + noise * noise_field), 0, 255).astype(np.uint8)


def truth_warp(scene: Scene, corners: np.ndarray | None = None) -> np.ndarray:
    """The manufactured rectified image: the texture on the output grid, float64 (out_h, out_w, 3).
    Rectifies the quadrilateral ``corners`` (default: the true corners); the verifier passes the reported
    corners to judge the warp against what *those* corners delimit."""
    ow, oh = scene.out_w, scene.out_h
    uv = np.stack(np.meshgrid(np.arange(ow) / (ow - 1), np.arange(oh) / (oh - 1)), axis=-1)
    G = _homography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
                    scene.corners if corners is None else np.asarray(corners, np.float64))
    q = _apply(np.linalg.inv(scene.H_ci), _apply(G, uv))
    cw, ch = scene.card
    return _texture(scene.tex, q[..., 0] / cw + 0.5, q[..., 1] / ch + 0.5)


def inputs(scene: Scene) -> dict:
    return {"img": scene.img, "out_w": scene.out_w, "out_h": scene.out_h}


# ---------------------------------------------------------------- verifiers

def _crop(a: np.ndarray) -> np.ndarray:
    """The judged region: central (1 - 2*MU) of each side, i.e. the texture without the paper margin."""
    h, w = a.shape[:2]
    return a[math.ceil(MU * (h - 1)):math.floor((1 - MU) * (h - 1)) + 1,
             math.ceil(MU * (w - 1)):math.floor((1 - MU) * (w - 1)) + 1]


def nrmse(out: np.ndarray, ref: np.ndarray) -> float:
    """RMS difference over the judged region, normalised by the contrast (std) of the reference."""
    o, r = _crop(np.asarray(out, np.float64)), _crop(np.asarray(ref, np.float64))
    return float(np.sqrt(np.mean((o - r) ** 2)) / max(float(r.std()), 1.0))


def corner_tolerance(scene: Scene) -> float:
    return TOL_A + TOL_B * scene.blur


def _best_assignment(got: np.ndarray, want: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Smallest achievable max-distance between two corner sets over all pairings, and the pairing."""
    best = (float("inf"), (0, 1, 2, 3))
    for perm in itertools.permutations(range(4)):
        worst = float(np.max(np.linalg.norm(got[list(perm)] - want, axis=1)))
        if worst < best[0]:
            best = (worst, perm)
    return best


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = check_outputs(SPEC, out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    corners = np.asarray(out["corners"], np.float64)
    if corners.shape != (4, 2):
        return Verdict(False, f"contract: corners should have shape (4, 2), got {corners.shape}")
    warped = np.asarray(out["warped"])
    if warped.shape != (scene.out_h, scene.out_w, 3):
        return Verdict(False, f"contract: warped should have shape (out_h, out_w, 3) = "
                              f"{(scene.out_h, scene.out_w, 3)}, got {warped.shape}")
    if not np.all(np.isfinite(corners)):
        return Verdict(False, "corners contain NaN or inf")

    tol = corner_tolerance(scene)
    dist = np.linalg.norm(corners - scene.corners, axis=1)
    if np.any(dist > tol):
        i = int(np.argmax(dist))
        hint = ""
        worst, perm = _best_assignment(corners, scene.corners)
        if worst <= tol:
            hint = f"; the four points are right but listed in the wrong order (perm {perm})"
        return Verdict(False, f"corner {CORNER_NAMES[i]} is {dist[i]:.1f} px from the manufactured "
                              f"{tuple(np.round(scene.corners[i], 1))} (tolerance {tol:.1f} px){hint}")
    err = nrmse(warped, truth_warp(scene, corners))
    if err > TOL_IMG:
        return Verdict(False, f"rectified image differs from the manufactured texture rectified from the "
                              f"reported corners: NRMSE {err:.2f} (tolerance {TOL_IMG:.2f})")
    return Verdict(True)


# ---------------------------------------------------------------- symmetries

_SHIFT = (9, -7)  # px: well inside BORDER
DIM = 0.5
T_SHIFT, T_ROT90, T_MIRROR, T_DIM = f"translate({_SHIFT[0]:+d},{_SHIFT[1]:+d})", "rot90", "mirror", f"dim x{DIM}"


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task. Corners and the rectified image must follow them (see verify_metamorphic)."""
    img, ow, oh = scene.img, scene.out_w, scene.out_h
    H, W = img.shape[:2]
    dx, dy = _SHIFT
    kept = img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    shifted = np.pad(kept, ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0)), (0, 0)), mode="edge")
    return {
        T_SHIFT: {"img": shifted, "out_w": ow, "out_h": oh},
        T_ROT90: {"img": np.ascontiguousarray(np.rot90(img)), "out_w": oh, "out_h": ow},
        T_MIRROR: {"img": np.ascontiguousarray(img[:, ::-1]), "out_w": ow, "out_h": oh},
        T_DIM: {"img": np.rint(img.astype(np.float32) * DIM).astype(np.uint8), "out_w": ow, "out_h": oh},
    }


def _expected_corners(name: str, c: np.ndarray) -> np.ndarray:
    if name == T_SHIFT:
        return c + np.array(_SHIFT, float)
    if name == T_ROT90:  # np.rot90: pixel (x, y) -> (y, W-1-x)
        return np.stack([c[:, 1], IMG_W - 1 - c[:, 0]], axis=1)
    if name == T_MIRROR:
        return np.stack([IMG_W - 1 - c[:, 0], c[:, 1]], axis=1)
    return c


def _expected_warp(name: str, w: np.ndarray) -> np.ndarray:
    if name == T_ROT90:
        return np.rot90(w)
    if name == T_MIRROR:
        return w[:, ::-1]
    if name == T_DIM:
        return np.rint(DIM * w)
    return w


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = check_outputs(SPEC, base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    c0, w0 = np.asarray(base["corners"], np.float64), np.asarray(base["warped"], np.float64)
    if c0.shape != (4, 2):
        return Verdict(False, f"contract: corners should have shape (4, 2), got {c0.shape}")
    for name in (T_SHIFT, T_ROT90, T_MIRROR, T_DIM):
        if name not in variants:
            return Verdict(False, f"missing variant {name!r}")
        out = variants[name]
        problems = check_outputs(SPEC, out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        c = np.asarray(out["corners"], np.float64)
        if c.shape != (4, 2):
            return Verdict(False, f"{name}: contract: corners should have shape (4, 2), got {c.shape}")
        # corners: the set moves with the photo (compared as a set, labels are checked right below)
        moved, _ = _best_assignment(c, _expected_corners(name, c0))
        if moved > META_TOL_CORNER:
            return Verdict(False, f"{name}: corner set is {moved:.1f} px from where the symmetry puts it "
                                  f"(allowed {META_TOL_CORNER})")
        idx, _ = position_order(c)
        if idx != [0, 1, 2, 3]:
            return Verdict(False, f"{name}: corners are not in TL, TR, BR, BL order by image position")
        # rectified image: covariant (rotated, mirrored, scaled) or unchanged
        want = _expected_warp(name, w0)
        w = np.asarray(out["warped"], np.float64)
        if w.shape != want.shape:
            return Verdict(False, f"{name}: rectified image has shape {w.shape}, expected {want.shape}")
        drift = nrmse(w, want)
        if drift > META_TOL_IMG:
            return Verdict(False, f"{name}: rectified image moved: NRMSE {drift:.2f} against the "
                                  f"symmetry-transformed original (allowed {META_TOL_IMG})")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img, out_w, out_h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(card)
    peri = cv2.arcLength(hull, True)
    for frac in np.arange(0.005, 0.2, 0.005):
        poly = cv2.approxPolyDP(hull, frac * peri, True)
        if len(poly) == 4:
            break
    pts = poly.reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return {"corners": corners, "warped": warped}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_CORNERS = "corners = np.array([tl, tr, br, bl], dtype=np.float32)"
_THRESH = "cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
_WARP = "warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)"
_PERSPECTIVE = "M = cv2.getPerspectiveTransform(corners, dst)\n    " + _WARP

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "float64_points": (
        "getPerspectiveTransform needs float32 points; float64 raises cv2.error (same on 4.14 and 5.0).",
        _mutate((_CORNERS, "corners = np.array([tl, tr, br, bl], dtype=np.float64)"))),
    "int32_points": (
        "approxPolyDP returns int32; passing those points on to getPerspectiveTransform raises.",
        _mutate((_CORNERS, "corners = np.array([tl, tr, br, bl], dtype=np.int32)"))),
    "dsize_hw": (
        "dsize is (width, height); (h, w) builds a transposed canvas and a wrongly shaped output.",
        _mutate(("(out_w, out_h), flags", "(out_h, out_w), flags"))),
    "contour_order": (
        "Skips the TL/TR/BR/BL ordering: corners stay in contour order, so the rectified card is mirrored.",
        _mutate((_CORNERS, "corners = pts.astype(np.float32)"))),
    "first_four_of_polygon": (
        "Polygon fit with a tiny epsilon is not reduced to four points; takes the first four vertices.",
        _mutate(("for frac in np.arange(0.005, 0.2, 0.005):", "for frac in (0.002,):"),
                ("pts = poly.reshape(4, 2)", "pts = poly.reshape(-1, 2)[:4]"))),
    "bgr_to_rgb_output": (
        "Converts the rectified image to RGB before returning it: channels swapped.",
        _mutate(('return {"corners": corners, "warped": warped}',
                 'warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)\n    return {"corners": corners, "warped": warped}'))),
    "fixed_threshold_127": (
        "Hard-coded threshold: right for these photos, wrong for a dimmer one.",
        _mutate((_THRESH, "cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)"))),
    "min_area_rect": (
        "minAreaRect/boxPoints gives a rotated rectangle, not the perspective quadrilateral.",
        _mutate(("pts = poly.reshape(4, 2)", "pts = np.rint(cv2.boxPoints(cv2.minAreaRect(card))).astype(np.int32)"))),
    "swallowed_float64": (
        "The float64 bug wrapped in try/except with an empty fallback. Exec-only filtering keeps this one.",
        _mutate((_CORNERS, "corners = np.array([tl, tr, br, bl], dtype=np.float64)"),
                (_PERSPECTIVE, "try:\n        M = cv2.getPerspectiveTransform(corners, dst)\n        "
                               + _WARP + "\n    except Exception:\n        warped = np.zeros((out_h, out_w, 3), np.uint8)"))),
    "inter_nearest (harmless)": (
        "Nearest-neighbour sampling moves the texture by at most 1/2 px: inside the registration tolerance.",
        _mutate(("flags=cv2.INTER_LINEAR", "flags=cv2.INTER_NEAREST"))),
    "dst_wh_convention (harmless)": (
        "Maps the corners onto (0..W, 0..H) instead of (0..W-1, 0..H-1): at most 1 px at the far edge.",
        _mutate(("[out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]", "[out_w, 0], [out_w, out_h], [0, out_h]"))),
    "extremes_of_contour (harmless)": (
        "No polygon fit at all: the position tests pick the extreme contour pixels, which sit at the corners.",
        _mutate(("pts = poly.reshape(4, 2)", "pts = card.reshape(-1, 2)"))),
}


# ---------------------------------------------------------------- the other standard estimator

def refine_corners(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Line-fit corner estimator: fit a line to the straight middle of each contour edge near ``quad`` (TL, TR,
    BR, BL, from any polygon fit) and intersect neighbouring lines. It has none of the contour estimators'
    rounding bias, so it brackets the physics from below (calibrate() measures both)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    P = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    q = np.asarray(quad, np.float64)
    nearest, owner, along = np.full(len(P), np.inf), np.zeros(len(P), int), np.zeros(len(P))
    for i in range(4):
        a, ab = q[i], q[(i + 1) % 4] - q[i]
        t = ((P - a) @ ab) / (ab @ ab)
        dist = np.linalg.norm(P - (a + np.clip(t, 0, 1)[:, None] * ab), axis=1)
        closer = dist < nearest
        nearest[closer], owner[closer], along[closer] = dist[closer], i, t[closer]
    lines = [cv2.fitLine(P[(owner == i) & (along > 0.15) & (along < 0.85)].astype(np.float32),
                         cv2.DIST_L2, 0, 0.01, 0.01).ravel().astype(np.float64) for i in range(4)]
    out = []
    for i in range(4):
        (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.array([[vx1, -vx2], [vy1, -vy2]]), np.array([x2 - x1, y2 - y1]))
        out.append([x1 + t[0] * vx1, y1 + t[0] * vy1])
    return np.array(out)


def warp_with(img: np.ndarray, corners: np.ndarray, out_w: int, out_h: int, *, pixel_edges: bool = False,
              flags: int = cv2.INTER_LINEAR) -> np.ndarray:
    """cv2 rectification used by the calibration: corners onto the output's corner pixel centres (default)
    or onto its outer pixel edges (0..W, 0..H)."""
    w, h = (out_w, out_h) if pixel_edges else (out_w - 1, out_h - 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=flags)


def _card_contour(img: np.ndarray, approx: int = cv2.CHAIN_APPROX_SIMPLE) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, approx)
    return max(contours, key=cv2.contourArea)


def position_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts).reshape(-1, 2)
    idx, _ = position_order(pts)
    return pts[idx].astype(np.float64)


def extremes_solve(img, out_w, out_h):
    """Standard estimator 2: the extreme contour pixels under the position tests, no polygon fit."""
    corners = position_corners(_card_contour(img))
    return {"corners": corners.astype(np.float32), "warped": warp_with(img, corners, out_w, out_h)}


def linefit_solve(img, out_w, out_h):
    """Standard estimator 3: the reference's quadrilateral refined by edge-line intersection."""
    corners = refine_corners(img, reference_solve(img, out_w, out_h)["corners"])
    return {"corners": corners.astype(np.float32), "warped": warp_with(img, corners, out_w, out_h)}


def dp2pct_solve(img, out_w, out_h):
    """Informational estimator (not in the envelope): fixed 2 % epsilon, the tutorial default."""
    card = _card_contour(img)
    poly = cv2.approxPolyDP(card, 0.02 * cv2.arcLength(card, True), True)
    corners = position_corners(poly)
    return {"corners": corners.astype(np.float32), "warped": warp_with(img, corners, out_w, out_h)}


STANDARD = {"reference": reference_solve, "extremes": extremes_solve, "line fit": linefit_solve}


# ---------------------------------------------------------------- calibration

def _gauge(seed: int) -> dict:
    s = make_scene(seed)
    ow, oh = s.out_w, s.out_h
    rec: dict = {"blur": s.blur, "corner": {}, "img": {}}
    sets = {}
    for name, fn in STANDARD.items():
        c = np.asarray(fn(s.img, ow, oh)["corners"], np.float64)
        sets[name] = c
        rec["corner"][name] = np.linalg.norm(c - s.corners, axis=1)
    c2 = np.asarray(dp2pct_solve(s.img, ow, oh)["corners"], np.float64)
    rec["corner"]["dp 2% (info)"] = np.linalg.norm(c2 - s.corners, axis=1)
    # rectified image against the manufactured texture rectified from the corners that produced it
    cases = {"true corners": (s.corners, {}), "true corners, pixel-edge dst": (s.corners, {"pixel_edges": True})}
    for name, c in sets.items():
        cases[name] = (c, {})
        cases[name + ", pixel-edge dst"] = (c, {"pixel_edges": True})
    cases["reference, nearest"] = (sets["reference"], {"flags": cv2.INTER_NEAREST})
    cases["reference, nearest, pixel-edge dst"] = (sets["reference"], {"flags": cv2.INTER_NEAREST, "pixel_edges": True})
    cases["reference, cubic"] = (sets["reference"], {"flags": cv2.INTER_CUBIC})
    truths: dict = {}
    for name, (c, kw) in cases.items():
        key = c.tobytes()
        if key not in truths:
            truths[key] = truth_warp(s, c)
        rec["img"][name] = nrmse(warp_with(s.img, c, ow, oh, **kw), truths[key])
    return rec


def texture_k_rms(n: int = 60, seed: int = 700_000) -> float:
    """Mean RMS spatial frequency of the manufactured texture, rad per output pixel per axis."""
    ks = []
    for i in range(n):
        t = _crop(truth_warp(make_scene(seed + i)))
        gy, gx = np.gradient(t, axis=(0, 1))
        ks.append(math.sqrt(float(np.mean(gx ** 2 + gy ** 2)) / 2 / float(t.var())))
    return float(np.mean(ks))


def calibrate(n: int = 2000, seed: int = 500_000, processes: int = 1, k_scenes: int = 60) -> dict:
    """Measure corner error (px) and rectified-image NRMSE on single-card gauge blocks, OpenCV 5.0.
    ``processes > 1`` forks a pool: only use it from a fresh process (``python -m ...``), not after OpenCV
    has already run threads in the caller."""
    if processes > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            recs = pool.map(_gauge, range(seed, seed + n), chunksize=8)
    else:
        recs = [_gauge(seed + i) for i in range(n)]
    blur = np.array([r["blur"] for r in recs])
    edges = [0.6, 0.9, 1.2, 1.5, 1.8, BLUR_RANGE[1] + 1e-9]
    out: dict = {"n": n, "corner": {}, "image": {}}
    errs = {}
    for name in recs[0]["corner"]:
        e = np.array([r["corner"][name] for r in recs])  # (n, 4)
        errs[name] = e
        by_bin = [round(float(e[(blur >= lo) & (blur < hi)].max()), 2) for lo, hi in zip(edges[:-1], edges[1:])]
        mean_bin = [round(float(e[(blur >= lo) & (blur < hi)].mean()), 2) for lo, hi in zip(edges[:-1], edges[1:])]
        out["corner"][name] = {"mean": round(float(e.mean()), 2), "p99": round(float(np.percentile(e, 99)), 2),
                               "max": round(float(e.max()), 2), "max_by_blur_bin": by_bin,
                               "mean_by_blur_bin": mean_bin,
                               "over_tolerance": int((e > (TOL_A + TOL_B * blur)[:, None]).sum())}
    sig = np.repeat(blur, 4)
    # slope: OLS of the error on sigma for the two contour estimators (the line fit has no rounding bias);
    # intercept: what the worst point of any standard estimator needs on top of that slope
    contour = np.concatenate([errs[k].ravel() for k in ("reference", "extremes")])
    slope = float(np.polyfit(np.tile(sig, 2), contour, 1)[0])
    intercept = max(float((errs[k].ravel() - slope * sig).max()) for k in STANDARD)
    worst = max(float((errs[k].ravel() / (TOL_A + TOL_B * sig)).max()) for k in STANDARD)
    out["corner_fit"] = {"slope_px_per_sigma": round(slope, 3), "intercept_px": round(intercept, 2),
                         "suggested_TOL_A_TOL_B_at_25pct_margin": (round(1.25 * intercept, 2), round(1.25 * slope, 2)),
                         "bin_edges_sigma": edges, "worst_ratio_to_tolerance": round(worst, 3)}
    worst_img = 0.0
    for name in recs[0]["img"]:
        v = np.array([r["img"][name] for r in recs])
        out["image"][name] = {"mean": round(float(v.mean()), 3), "max": round(float(v.max()), 3)}
        worst_img = max(worst_img, float(v.max()))
    out["image_worst_max"] = round(worst_img, 3)
    out["image_worst_ratio_to_tolerance"] = round(worst_img / TOL_IMG, 3)
    out["texture_k_rms"] = round(texture_k_rms(k_scenes), 3)
    return out


def _drift_one(seed: int) -> dict:
    scene = make_scene(seed)
    base_in = inputs(scene)
    variants = transforms(scene)
    rec: dict = {}
    for est, fn in STANDARD.items():
        base = fn(**base_in)
        c0, w0 = np.asarray(base["corners"], np.float64), np.asarray(base["warped"], np.float64)
        for name, kw in variants.items():
            out = fn(**kw)
            c, w = np.asarray(out["corners"], np.float64), np.asarray(out["warped"], np.float64)
            moved = _best_assignment(c, _expected_corners(name, c0))[0]
            want = _expected_warp(name, w0)
            drift = nrmse(w, want) if w.shape == want.shape else float("inf")
            ordered = position_order(c)[0] == [0, 1, 2, 3]
            rec[(est, name)] = (moved, drift, ordered)
    return rec


def measure_metamorphic_drift(n: int = 300, seed: int = 50_000, processes: int = 1) -> dict:
    """Largest movement of each standard estimator's corners (px) and rectified image (NRMSE) under the symmetries."""
    if processes > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            recs = pool.map(_drift_one, range(seed, seed + n), chunksize=4)
    else:
        recs = [_drift_one(seed + i) for i in range(n)]
    out: dict = {"n": n, "by": {}}
    for key in recs[0]:
        est, name = key
        vals = [r[key] for r in recs]
        out["by"][f"{est} / {name}"] = {"corner_max": round(max(v[0] for v in vals), 2),
                                        "image_max": round(max(v[1] for v in vals), 3),
                                        "order_violations": sum(not v[2] for v in vals)}
    out["corner_worst"] = max(v["corner_max"] for v in out["by"].values())
    out["image_worst"] = max(v["image_max"] for v in out["by"].values())
    return out


if __name__ == "__main__":
    import json
    cv2.setNumThreads(1)  # the worker pool forks: keep OpenCV's own thread pool out of the parent
    print(json.dumps(calibrate(processes=12), indent=1))
    print(json.dumps(measure_metamorphic_drift(processes=12), indent=1))
