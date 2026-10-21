"""Family ``chessboard_corners``: sub-pixel inner corners of a photographed chessboard.

Choose the answer first (method of manufactured solutions): the corners are the image of the
grid points (i, j) of a flat board under a known homography (a pinhole camera looking at the
board with pitch and yaw each drawn uniformly in +-32 degrees and a roll of up to 22 degrees; the
board normal then lies up to ~43 degrees off the optical axis, see ``_pose``). The photo is rendered
*from* that homography. Every grid line is a straight image line, and a straight edge seen through a
Gaussian PSF has the exact profile erf(d / (sigma sqrt 2)) (d = signed distance to the line),
so the board is the product of these edge profiles and its saddle points sit exactly on the
manufactured corners: no supersampling, no phase error in the truth. Pixel (col, row) is
sampled at (col, row), OpenCV's convention. Two independent confirmations that the truth is
right: the reference below has a mean signed error of +0.0002 / -0.0001 px over 300 scenes, and
the rendered noise-free board is point-symmetric about each manufactured corner (a test pins it).

Tolerance physics
-----------------
A sub-pixel corner refinement (``cornerSubPix``: the point where all window gradients are
orthogonal to the offset) has two error mechanisms. The first-principles draft had only the
second, as ``a + b * noise / contrast`` with constants independent of blur; calibration refuted
that (the "constant" a falls from 0.20 px to 0.03 px across blur 0..1.5 px, and the slope b
grows ~4x between blur 0.3 and 2.0), hence both terms below:

1. Sampling bias on sharp edges. An edge through a PSF of Gaussian width ``sigma_e`` is
   sampled once per pixel; its spectrum decays as exp(-2 pi^2 sigma_e^2 f^2), and the part
   that folds back across Nyquist is what biases the gradient estimate. It vanishes quickly
   with blur: alias(sigma_e) = A * exp(-sigma_e^2 / (2 W^2)).
   sigma_e^2 = blur^2 + 1/12 adds the pixel's own box footprint to the optical blur.
2. Noise. The offset is a least-squares solution built from gradient pixels, so its jitter
   scales as (gray-level noise / edge contrast) times a length that grows with the width of
   the gradient band: K * sigma_e * sigma_n_eff / C. 8-bit rounding adds uniform noise of
   variance 1/12 gray level^2 whatever the camera does, so sigma_n_eff^2 = sigma_n^2 + 1/12
   (this is what makes the error of a noise-free blurred, low-contrast board non-zero).

    tol(scene) = TOL_M * [ A exp(-sigma_e^2 / (2 W^2)) + K sigma_e sqrt(sigma_n^2 + 1/12) / C ]

Two measured facts that shaped the law (see the numbers at the constants):

* Perspective, board size and corner count do not enter the reference class: the worst corner
  error is 0.034 / 0.035 / 0.036 / 0.032 px at 0 / 0.3 / 0.6 / 1.0 of full tilt (blur 1, no
  noise), and |corr(error/law, n corners or square size)| < 0.14 on 6000 scenes. The law bounds
  "every corner of the board", a max over N corners.
* The error depends on noise and contrast only through the ratio: at sigma_n / C = 4 / 140 the
  worst error is 0.153 / 0.158 / 0.154 / 0.160 px for contrast 60 / 100 / 140 / 190.

What the law is calibrated on (and what it is not)
--------------------------------------------------
The law is the upper envelope of the standard *refined* estimator: ``findChessboardCorners``
followed by ``cornerSubPix`` with a window of half-size 8..13 and either termination
criterion. Per-scene pass rates at TOL_M over 1500 scenes by half-window: 8, 11, 13: 100 %;
6: 99.9 %; 5: 99.8 %; 4: 98.5 %; 15: 99.3 %; but 3: 72 % (window narrower than the gradient
band), 2: 44 %, and 20: 79 % (window wider than two squares).

Not covered, deliberately: the *unrefined* detectors. ``findChessboardCorners`` alone has an
error that grows with blur (worst corner 0.2 px on sharp boards, 1.5-2.5 px at blur 2; passes
42 % of scenes): the refinement step is the point of the task. ``findChessboardCornersSB`` alone has a different, estimator-specific error
that does not follow this law: it is ~1/(local square size) (median 0.12 px for 15-25 px squares,
0.02 px above 50 px) and grows with foreshortening (worst corner 0.080 px frontal, 0.239 px at full
tilt, noise-free), and its ACCURACY/EXHAUSTIVE flags change nothing in 5.0.0; with a
``cornerSubPix`` pass on top it joins the reference class (99.7 % of scenes). Under this law SB
alone passes ~87 % of scenes, so it passes a whole batch of 20 only ~6 % of the time. That is a
judgment about what "sub-pixel accurate" means here, not a calibration artefact: the other choice
is a pattern-scale term c / s_min in the tolerance, at the price of also admitting the unrefined
``findChessboardCorners`` on sharp scenes. The constants make either easy to revisit.

Ordering
--------
The contract fixes the order (board frame, row-major, upright board) so that the manufactured
gate can compare corner i with corner i. OpenCV 5.0.0 does not: it returns the upright
row-major order only when the board's top-left square is black, and the *reverse* when it is
white (300 of 300 scenes, no exception). A correct solution has to canonicalize; that trap is
a mutant. Under rotation the order is not covariant, so the metamorphic gate compares point
*sets*.

Metamorphic checks
------------------
Symmetries and what they imply for the output (``verify_metamorphic`` sees only the outputs,
so every mapping is a function of the transform name and the fixed image size):

* translate(+7,-5): covariant, corners + (7, -5)
* rot90 (np.rot90, counter-clockwise): covariant, (x, y) -> (y, W-1-x); squares_x/y swap
* mirror (columns reversed): covariant, (x, y) -> (W-1-x, y)
* dim x0.5 (all gray levels halved, noise too): invariant, which catches hard-coded
  thresholds; it costs ~0.09 px of drift (the rounding noise doubles in relative terms)
* NOT used: contrast inversion. The board is no longer a white-bordered black-and-white
  target (the quiet zone turns black) and the reference finds it in only 10 % of scenes.

A solution that returns the same corners for every image would pass these (a stuck gauge);
the manufactured gate is what catches it.
"""
from __future__ import annotations

import functools
import inspect
import math
import textwrap
from dataclasses import dataclass, replace

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

IMG_H, IMG_W = 480, 640  # fixed: the covariant metamorphic checks need the image size without the scene
SQ_MIN = 20.0            # px; smallest local square size anywhere on the board
MARGIN = 9               # px kept clear between the quiet zone and the image border (room for the shift)
QUIET = 1.0              # white margin around the board, in squares

SPEC = TaskSpec(
    family="chessboard_corners",
    task=("The image shows one chessboard calibration target. Find every inner corner (each point where "
          "four squares meet) with sub-pixel accuracy and return them in row-major order."),
    inputs={
        "img": IOSpec("uint8", "HxWx3", 3, "BGR photo of one chessboard on a plain background. The whole board is "
                                           "visible and upright (its top edge is the one nearest the top of the "
                                           "image, rotated by less than 30 degrees, never mirrored), seen from up "
                                           "to ~30 degrees off-axis. It may be blurred or noisy, and its top-left "
                                           "square may be black or white"),
        "squares_x": IOSpec("int", "scalar", None, "number of squares along the board's top edge"),
        "squares_y": IOSpec("int", "scalar", None, "number of squares along the board's left edge"),
    },
    outputs={
        "corners": IOSpec("float", "Nx2", 2,
                          "(x, y) pixel coordinates, x to the right and y down, the centre of the top-left pixel "
                          "being (0, 0); N = (squares_x - 1) * (squares_y - 1). Row-major in the board's own frame: "
                          "first the corner nearest the board's top-left, then left to right along the board's "
                          "top row of corners, then the next row down, and so on"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Locate the inner corners of the chessboard in this photo (the squares_x by squares_y board). Give their "
    "pixel coordinates to sub-pixel accuracy, ordered row by row starting at the board's top-left corner.",
    "This is a picture of a chessboard used for camera calibration. Detect all of its inner corners, refine "
    "them to sub-pixel precision, and return the (x, y) positions row-major: left to right along the top "
    "row of corners, then each row below.",
    "Find where the squares of the chessboard meet. Report all (squares_x - 1) * (squares_y - 1) interior "
    "corners as sub-pixel (x, y) positions in the board's reading order, top row first.",
]

# ---------------------------------------------------------------------------------- tolerance law
# alias(sigma_e) = A exp(-sigma_e^2 / (2 W^2)); noise = K sigma_e sqrt(sigma_n^2 + 1/12) / C;
# tol = TOL_M * (alias + noise).
#
# A, W, K: upper-envelope fit (least squares on law - error with a heavy penalty on law < error,
# numpy Nelder-Mead, `fit_envelope`) to the per-scene worst corner error of the refined estimator
# over the 3840 gauge blocks of calibrate() (12 blurs 0..2.5 px x 8 noise/contrast ratios 0..0.08,
# 40 boards each, contrast 60..200; OpenCV 5.0.0; max over cornerSubPix windows 8 and 11 and both
# termination criteria). The fit recovered (A, W, K) = (0.2601, 0.5022, 7.314) from two independent
# runs; the noise exponent of sigma_e, left free, came out 1.00 (0.9-1.1). Without the 8-bit
# rounding term the fit needed a constant floor of ~0.03 px; with it the floor is 0.000, so the
# rounding noise accounts for the whole low-contrast, high-blur floor.
# On the grid the law's constants, rounded, bound the worst gauge block at 1.062x (2 of 3840 blocks
# excluded: a corner mislocated by whole pixels by findChessboardCorners itself, 12 px, which no
# window repairs; 0.05 %). On 6000 fresh scenes drawn from make_scene's own prior the worst ratio
# is 0.927 for the reference (window 11) and 1.163 for the loosest standard variant (window 8).
#
# TOL_M = 1.5: the margin rule is >= 25 % over the worst ratio seen anywhere (1.163 * 1.25 = 1.45).
# At 1.5 the reference peaks at 0.62 of the tolerance and the worst standard variant at 0.78.
# Mechanism-by-mechanism, for the record: the alias term dominates for blur < 0.6 px (0.2 px for a
# sharp, noise-free board), the noise term for blur > 1 px (at the prior's corner, blur 2, ratio
# 0.057, the law is 0.4 px).
A_ALIAS, W_ALIAS, K_NOISE = 0.26, 0.50, 7.3
TOL_M = 1.5

# META_TOL: set distance between the mapped corners of the original and those found in a transformed
# image (measure_metamorphic_drift(), 600 scenes x 4 transforms, seeds 800_000..800_599; the board was found
# every time). Re-run on OpenCV 5.0.0 with n=600: the reference
# (findChessboardCorners + cornerSubPix(11, 11), eps 0.001) moves by <= 0.0007 px under
# translate/rot90/mirror and <= 0.094 px (0.0939) when the image is dimmed x0.5 (the rounding noise doubles
# in relative terms); window 8 gives 0.0009 and 0.086 (an earlier record said 0.077, which did not reproduce
# on these 600 scenes); the loose termination eps = 0.1, a legitimate setting, moves by <= 0.074 px (0.0735)
# under the exact symmetries and <= 0.130 px (0.1298) under dimming. META_TOL = 0.2 is ~1.5x the worst of
# those, and far below the 0.7-1.0 px that a pixel-centre convention slip produces under mirror/rot90.
META_TOL = 0.2

# What calibrate() / validate_on_prior() / measure_metamorphic_drift() recorded (OpenCV 5.0.0,
# numpy 2.4.6); tests/test_chessboard_corners.py pins the constants against these.
RECORDED = {
    "gauge_blocks": 3840,               # calibrate(reps=40, seed=100000)
    "gauge_worst_over_law": 1.062,      # law-class estimators, 2 detector failures excluded
    "refit_A_W_K": (0.2601, 0.5022, 7.314),
    "prior_scenes": 6000,               # validate_on_prior(6000, seed=500000)
    "prior_worst_over_law": 1.163,      # window 8; the reference's own worst is 0.927
}


@dataclass
class Scene:
    img: np.ndarray            # BGR uint8, IMG_H x IMG_W x 3
    corners: np.ndarray        # (N, 2) float64: manufactured truth, row-major in the board frame
    squares: tuple[int, int]   # (squares_x, squares_y)
    blur: float                # Gaussian PSF sigma, px
    noise: float               # luminance noise sigma, gray levels (the same draw in all three channels)
    contrast: float            # gray-level difference white - black squares (BT.601 gray of the BGR levels)
    surround: tuple[float, float, float]  # BGR level outside the quiet zone
    min_square_px: float       # smallest local square size on the board, px
    tl_dark: bool              # is the board's top-left square black?
    seed: int


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- scene generation

_T = np.linspace(-6.0, 6.0, 12001)
_ERF = np.array([math.erf(t) for t in _T])
_SQRT2 = math.sqrt(2.0)


def _step(d: np.ndarray, sigma_eff: float) -> np.ndarray:
    """A straight edge seen through a Gaussian PSF: erf(d / (sigma sqrt 2)), d = signed distance in px."""
    return np.interp(d / (sigma_eff * _SQRT2), _T, _ERF)


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _pose(rng: np.random.Generator, sx: int, sy: int, tilt: float = 1.0) -> tuple[np.ndarray, float]:
    """Homography board -> image (board units = squares, origin at the board's top-left corner).

    Returns it with the smallest local square size (px per square) over the board.

    Pose draws: pitch and yaw each uniform in +-32 degrees and roll uniform in +-22 degrees (all times ``tilt``);
    an attempt is kept only if every local square is >= SQ_MIN px and the quiet zone clears the MARGIN. Over 600
    accepted scenes (seeds 7_300_000..7_300_599) the largest |pitch|, |yaw|, |roll| are 31.8, 32.0, 22.0 degrees,
    and the angle between the board normal and the optical axis, acos(cos(pitch) cos(yaw)), has a median of 23.8,
    a 99th percentile of 41.1 and a maximum of 43.1 degrees (25 % of the scenes exceed 30 degrees)."""
    for _ in range(500):
        pitch, yaw = np.radians(tilt * rng.uniform(-32, 32, 2))
        roll = math.radians(tilt * rng.uniform(-22, 22))
        f = rng.uniform(450, 800)
        R = _rot(pitch, yaw, roll)
        u0 = IMG_W / 2 + rng.uniform(-20, 20)
        v0 = IMG_H / 2 + rng.uniform(-16, 16)
        fill = rng.uniform(0.72, 0.95)
        K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1.0]])
        shift = np.array([[1, 0, -sx / 2], [0, 1, -sy / 2], [0, 0, 1.0]])
        quad = np.array([[-QUIET, -QUIET, 1], [sx + QUIET, -QUIET, 1],
                         [sx + QUIET, sy + QUIET, 1], [-QUIET, sy + QUIET, 1.0]]).T
        Z, ok = 20.0, True
        for _ in range(10):  # scale the distance until the quiet zone fills `fill` of the image
            t = np.array([(u0 - IMG_W / 2) * Z / f, (v0 - IMG_H / 2) * Z / f, Z])
            Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ shift
            p = Hm @ quad
            if np.any(p[2] <= 0):
                ok = False
                break
            p = p[:2] / p[2]
            Z *= max(np.ptp(p[0]) / (fill * IMG_W), np.ptp(p[1]) / (fill * IMG_H))
        if not ok:
            continue
        scales = []
        for X, Y in ((0, 0), (sx, 0), (sx, sy), (0, sy)):  # Jacobian columns = px per square
            v = Hm @ np.array([X, Y, 1.0])
            J = (Hm[:2, :2] * v[2] - np.outer(v[:2], Hm[2, :2])) / v[2] ** 2
            scales += list(np.linalg.norm(J, axis=0))
        if (min(scales) >= SQ_MIN and p[0].min() >= MARGIN and p[0].max() <= IMG_W - 1 - MARGIN
                and p[1].min() >= MARGIN and p[1].max() <= IMG_H - 1 - MARGIN):
            return Hm, float(min(scales))
    raise RuntimeError("no admissible board pose")


def _gray(bgr) -> float:
    return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])


def _board_pairs() -> list[tuple[int, int]]:
    # (squares_x, squares_y) of opposite parity, so the inner-corner grid is never 180-degree symmetric
    return [(a, b) for a in range(6, 11) for b in range(5, 10) if (a - b) % 2 == 1]


def _build(seed: int, blur: float | None = None, noise: float | None = None, contrast: float | None = None,
           squares: tuple[int, int] | None = None, tilt: float = 1.0) -> Scene:
    """Manufacture a board: draw the corners' pose first, then render the image from it.

    The keyword overrides pin one factor (gauge blocks); every random draw is made regardless, so
    pinning blur leaves the geometry and the noise pattern of that seed unchanged."""
    rng = np.random.default_rng(seed)
    pairs = _board_pairs()
    sq = pairs[int(rng.integers(len(pairs)))]
    tl_dark = bool(rng.integers(2))
    pose_seed = rng.integers(1 << 31)
    b_ = float(rng.uniform(0.0, 2.0))
    n_ = float(rng.uniform(0.0, 4.0))
    c_ = float(rng.uniform(70.0, 190.0))
    mid_u, gains, surround_u = rng.uniform(0, 1), rng.uniform(0.92, 1.08, 3), rng.uniform(50, 190)
    sx, sy = squares or sq
    blur = b_ if blur is None else float(blur)
    noise = n_ if noise is None else float(noise)
    C = c_ if contrast is None else float(contrast)
    Hm, smin = _pose(np.random.default_rng(int(pose_seed)), sx, sy, tilt)

    lo, hi = 14 + C / 2, 234 - C / 2
    mid = lo + mid_u * (hi - lo)
    white = np.clip((mid + C / 2) * gains, 0, 255)
    black = np.clip((mid - C / 2) * gains, 0, 255)
    surround = np.clip(surround_u * gains, 0, 255)

    sig = math.sqrt(blur ** 2 + 1.0 / 12.0)  # PSF plus the pixel's own box footprint
    Hinv = np.linalg.inv(Hm)
    if Hinv[2] @ np.array([IMG_W / 2, IMG_H / 2, 1.0]) < 0:
        Hinv = -Hinv

    def dist(axis: int, k: float, xs, ys):
        a, b, c = Hinv[axis] - k * Hinv[2]
        return (a * xs + b * ys + c) / math.hypot(a, b)

    pad = int(math.ceil(6 * sig)) + 2
    full = np.array([[-QUIET, -QUIET, 1], [sx + QUIET, -QUIET, 1], [sx + QUIET, sy + QUIET, 1],
                     [-QUIET, sy + QUIET, 1.0]]).T
    pq = Hm @ full
    pq = pq[:2] / pq[2]
    x0, x1 = max(0, int(pq[0].min()) - pad), min(IMG_W, int(pq[0].max()) + pad + 1)
    y0, y1 = max(0, int(pq[1].min()) - pad), min(IMG_H, int(pq[1].max()) + pad + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)

    par = np.ones_like(xs)  # product of the signs of all interior grid lines = the checker pattern
    for k in range(1, sx):
        par *= _step(dist(0, k, xs, ys), sig)
    for k in range(1, sy):
        par *= _step(dist(1, k, xs, ys), sig)
    t0 = (-1.0) ** (sx + sy)  # the product's value inside the top-left square
    g = -t0 if tl_dark else t0
    chk = 0.5 * (1.0 + g * par)  # 0 on black squares, 1 on white ones

    def ind(axis, k, sign):
        return 0.5 * (1.0 + _step(sign * dist(axis, k, xs, ys), sig))

    m_b = ind(0, 0, +1) * ind(0, sx, -1) * ind(1, 0, +1) * ind(1, sy, -1)
    m_q = ind(0, -QUIET, +1) * ind(0, sx + QUIET, -1) * ind(1, -QUIET, +1) * ind(1, sy + QUIET, -1)
    board = black + (white - black) * chk[..., None]
    inner = white * (1.0 - m_b[..., None]) + board * m_b[..., None]
    val = surround * (1.0 - m_q[..., None]) + inner * m_q[..., None]

    img = np.broadcast_to(surround, (IMG_H, IMG_W, 3)).astype(np.float64).copy()
    img[y0:y1, x0:x1] = val
    # luminance noise: one draw shared by the three channels, so the gray image any solver derives
    # (cvtColor or a single channel) carries sigma = `noise`
    img = img + (rng.normal(0.0, noise, img.shape[:2])[..., None] if noise > 0 else 0.0)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    ii, jj = np.meshgrid(np.arange(1, sx), np.arange(1, sy))
    pts = Hm @ np.stack([ii.ravel(), jj.ravel(), np.ones(ii.size)])
    corners = (pts[:2] / pts[2]).T
    return Scene(img, corners, (sx, sy), blur, noise, _gray(white) - _gray(black),
                 tuple(float(v) for v in surround), smin, tl_dark, seed)


@functools.lru_cache(maxsize=64)  # ~0.9 MB per scene
def _cached(seed, blur, noise, contrast, squares, tilt) -> Scene:
    return _build(seed, blur, noise, contrast, squares, tilt)


def make_scene(seed: int, blur: float | None = None, noise: float | None = None, contrast: float | None = None,
               squares: tuple[int, int] | None = None, tilt: float = 1.0) -> Scene:
    """Deterministic in `seed`. Rendering is cached (a gate matrix regenerates the same scenes for
    every candidate); callers get their own copy of the arrays."""
    s = _cached(seed, blur, noise, contrast, squares, tilt)
    return replace(s, img=s.img.copy(), corners=s.corners.copy())


def inputs(scene: Scene) -> dict:
    return {"img": scene.img, "squares_x": int(scene.squares[0]), "squares_y": int(scene.squares[1])}


# ---------------------------------------------------------------- verifiers

def _law(blur: float, noise: float, contrast: float, a=None, w=None, k=None) -> float:
    """The fitted envelope of the refined estimator's worst corner error (px), before the margin."""
    a, w, k = A_ALIAS if a is None else a, W_ALIAS if w is None else w, K_NOISE if k is None else k
    se2 = blur ** 2 + 1.0 / 12.0
    return (a * math.exp(-se2 / (2.0 * w ** 2))
            + k * math.sqrt(se2) * math.sqrt(noise ** 2 + 1.0 / 12.0) / contrast)


def tolerance(scene: Scene) -> float:
    """Allowed distance (px) between a returned corner and the manufactured one, from the scene's own
    blur, noise and contrast: the physics law in the module docstring, times the margin."""
    return TOL_M * _law(scene.blur, scene.noise, scene.contrast)


def _set_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff distance between two point sets: order-free, and unambiguous here because the corner
    spacing (>= 15 px) is far above every tolerance used."""
    if len(a) == 0 or len(b) == 0:  # equal shapes are checked first, so both are empty: identical sets
        return 0.0 if len(a) == len(b) else float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _order_diagnosis(got: np.ndarray, truth: np.ndarray, squares: tuple[int, int], tol: float) -> str:
    """The positions are right: say how the order is wrong."""
    cols, rows = squares[0] - 1, squares[1] - 1
    grid = truth.reshape(rows, cols, 2)
    readings = {
        "reversed (OpenCV starts at the far corner when the board's top-left square is white)": grid[::-1, ::-1],
        "column-major (the grid read column by column)": grid.transpose(1, 0, 2),
        "the grid read after a quarter turn (rows and columns exchanged, e.g. a swapped patternSize)": np.rot90(grid, 1),
        "the grid read after a quarter turn the other way": np.rot90(grid, -1),
        "the grid read column by column from the far side": np.rot90(grid, 2).transpose(1, 0, 2),
    }
    for what, g in readings.items():
        if np.max(np.linalg.norm(got - g.reshape(-1, 2), axis=1)) <= tol:
            return what
    return "not a symmetry of the grid at all (e.g. sorted by image coordinates)"


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = check_outputs(SPEC, out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    got = np.asarray(out["corners"], dtype=np.float64)
    truth = scene.corners
    sx, sy = scene.squares
    if got.shape != truth.shape:
        return Verdict(False, f"{len(got)} corners for a {sx}x{sy}-square board "
                              f"({sx - 1}x{sy - 1} = {len(truth)} inner corners expected)")
    if not np.all(np.isfinite(got)):
        return Verdict(False, "non-finite corner coordinates")
    tol = tolerance(scene)
    err = np.linalg.norm(got - truth, axis=1)
    if err.max() <= tol:
        return Verdict(True)
    if _set_distance(got, truth) <= tol:
        return Verdict(False, "the corner positions are right but the order is wrong: "
                              + _order_diagnosis(got, truth, scene.squares, tol))
    i = int(np.argmax(err))
    return Verdict(False, f"corner {i} of {len(truth)} is {err[i]:.3f} px from the manufactured position "
                          f"(allowed {tol:.3f} px at blur {scene.blur:.2f}, noise {scene.noise:.2f}, "
                          f"contrast {scene.contrast:.0f})")


_DX, _DY = 7, -5  # translation, well inside MARGIN
_COVARIANT = {
    "translate(+7,-5)": lambda p: p + np.array([_DX, _DY], dtype=np.float64),
    "rot90": lambda p: np.stack([p[:, 1], IMG_W - 1 - p[:, 0]], axis=1),  # np.rot90: (x, y) -> (y, W-1-x)
    "mirror": lambda p: np.stack([IMG_W - 1 - p[:, 0], p[:, 1]], axis=1),
    "dim x0.5": lambda p: p,
}


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task. Corner sets must follow the image (see ``_COVARIANT``)."""
    img = scene.img
    sx, sy = scene.squares
    H, W = img.shape[:2]
    shifted = np.empty_like(img)
    shifted[:] = np.round(np.array(scene.surround)).astype(np.uint8)
    shifted[max(_DY, 0):H + min(_DY, 0), max(_DX, 0):W + min(_DX, 0)] = \
        img[max(-_DY, 0):H - max(_DY, 0), max(-_DX, 0):W - max(_DX, 0)]
    return {
        "translate(+7,-5)": {"img": shifted, "squares_x": sx, "squares_y": sy},
        "rot90": {"img": np.ascontiguousarray(np.rot90(img)), "squares_x": sy, "squares_y": sx},
        "mirror": {"img": np.ascontiguousarray(img[:, ::-1]), "squares_x": sx, "squares_y": sy},
        "dim x0.5": {"img": np.round(img.astype(np.float32) * 0.5).astype(np.uint8),
                     "squares_x": sx, "squares_y": sy},
    }


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = check_outputs(SPEC, base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    c0 = np.asarray(base["corners"], dtype=np.float64)
    for name, out in variants.items():
        problems = check_outputs(SPEC, out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        if name not in _COVARIANT:
            return Verdict(False, f"{name}: unknown transform")
        c = np.asarray(out["corners"], dtype=np.float64)
        if c.shape != c0.shape:
            return Verdict(False, f"{name}: {len(c)} corners, but {len(c0)} on the original image")
        moved = _set_distance(_COVARIANT[name](c0), c)
        if not moved <= META_TOL:  # also catches NaN
            return Verdict(False, f"{name}: corners moved by {moved:.3f} px relative to the original image "
                                  f"(allowed {META_TOL})")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img, squares_x, squares_y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (squares_x - 1, squares_y - 1))
    if not found:
        raise ValueError("chessboard not found")
    corners = corners.reshape(-1, 2).astype(np.float32)  # (N, 1, 2) on 4.x, (N, 2) on 5.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)
    if corners[0].sum() > corners[-1].sum():  # a white top-left square makes OpenCV start at the far corner
        corners = corners[::-1]
    return {"corners": np.ascontiguousarray(corners, dtype=np.float64)}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_RESHAPE = "    corners = corners.reshape(-1, 2).astype(np.float32)  # (N, 1, 2) on 4.x, (N, 2) on 5.0\n"
_SUBPIX = ("    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)\n"
           "    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)\n")
_FLIP = ("    if corners[0].sum() > corners[-1].sum():  # a white top-left square makes OpenCV start at the far corner\n"
         "        corners = corners[::-1]\n")
_FCC = "cv2.findChessboardCorners(gray, (squares_x - 1, squares_y - 1))"
_RET = '    return {"corners": np.ascontiguousarray(corners, dtype=np.float64)}'

# Plausible model mistakes (and two correct alternatives, marked). Each runs unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "rgb2gray (harmless here)": (
        "Channel order cannot matter: the noise is shared by the channels and the gains differ by <8 %. A true negative.",
        _mutate(("COLOR_BGR2GRAY", "COLOR_RGB2GRAY"))),
    "SB + cornerSubPix (correct)": (
        "findChessboardCornersSB for detection, refined with cornerSubPix: joins the reference class. A true negative.",
        _mutate((_FCC, "cv2.findChessboardCornersSB(gray, (squares_x - 1, squares_y - 1))"))),
    "index_corners_0": (
        "The 4.x idiom corners[:, 0, :]: findChessboardCorners returns (N, 2) in 5.0, so this raises IndexError.",
        _mutate((_RESHAPE, "    corners = corners[:, 0, :].astype(np.float32)\n"))),
    "no_cornersubpix": (
        "Returns findChessboardCorners' own corners. They are good to ~0.1-0.3 px on sharp images and degrade "
        "to 1-2 px as the blur grows.",
        _mutate((_SUBPIX, ""))),
    "sb_unrefined": (
        "findChessboardCornersSB without cornerSubPix. SB's own error (~1/square size, up to 0.3 px on tilted "
        "boards) is above the refined-estimator law in about a quarter of the scenes.",
        _mutate((_FCC, "cv2.findChessboardCornersSB(gray, (squares_x - 1, squares_y - 1))"), (_SUBPIX, ""))),
    "pattern_in_squares": (
        "patternSize is the number of INNER corners; this passes the squares. Detection fails and the "
        "solution raises.",
        _mutate((_FCC, "cv2.findChessboardCorners(gray, (squares_x, squares_y))"))),
    "swapped_rows_cols": (
        "patternSize is (columns, rows) of inner corners; this passes (rows, columns). OpenCV usually still "
        "finds the board, labelled column-major: the right points in the wrong order.",
        _mutate((_FCC, "cv2.findChessboardCorners(gray, (squares_y - 1, squares_x - 1))"))),
    "no_order_fix": (
        "Keeps OpenCV's order. It is the reverse of row-major whenever the board's top-left square is white.",
        _mutate((_FLIP, ""))),
    "sorted_by_y_then_x": (
        "'Row-major' implemented as a lexicographic sort on (y, x): wrong as soon as the board is rolled "
        "more than a fraction of a square.",
        _mutate((_FLIP, _FLIP + "    corners = corners[np.lexsort((corners[:, 0], corners[:, 1]))]\n"))),
    "xy_swapped": (
        "Returns (y, x) instead of (x, y).",
        _mutate((_RET, '    return {"corners": np.ascontiguousarray(corners[:, ::-1], dtype=np.float64)}'))),
    "half_pixel_offset": (
        "Adds 0.5: pixel centres taken at (i + 0.5, j + 0.5). OpenCV's convention puts them at integers.",
        _mutate((_RET, '    return {"corners": np.ascontiguousarray(corners + 0.5, dtype=np.float64)}'))),
    "integer_pixels": (
        "Rounds the refined corners to whole pixels (uniform +-0.5 px error).",
        _mutate((_RET, '    return {"corners": np.ascontiguousarray(np.round(corners), dtype=np.float64)}'))),
    "subpix_on_color": (
        "Passes the BGR image to cornerSubPix, which needs one channel. Raises cv2.error.",
        _mutate(("cv2.cornerSubPix(gray, corners,", "cv2.cornerSubPix(img, corners,"))),
    "tiny_subpix_window": (
        "winSize (2, 2): a 5x5 pixel window is narrower than the gradient band of a blurred edge.",
        _mutate(("(11, 11)", "(2, 2)"))),
    "huge_subpix_window": (
        "winSize (20, 20): a 41x41 window is wider than two squares and pulls in neighbouring corners.",
        _mutate(("(11, 11)", "(20, 20)"))),
    "fixed_threshold_127": (
        "Binarizes at a hard-coded 127 before detection: right for some bright scenes, wrong for a dimmer image.",
        _mutate((_FCC, "cv2.findChessboardCorners(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], "
                       "(squares_x - 1, squares_y - 1))"))),
}


# ---------------------------------------------------------------- calibration

GRID_BLUR = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)  # px, PSF sigma
GRID_RATIO = (0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08)                # sigma_n / contrast
LAW_CLASS = ("R8", "R11", "R11l")  # the estimators the law envelopes (see the docstring)
_CRIT = lambda eps: (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, eps)  # noqa: E731


def _upright(c: np.ndarray) -> np.ndarray:
    return c[::-1] if c[0].sum() > c[-1].sum() else c


def _refined(img: np.ndarray, sx: int, sy: int, win: int = 11, eps: float = 0.001):
    """findChessboardCorners + cornerSubPix(win, eps), upright row-major; None if not found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, c = cv2.findChessboardCorners(gray, (sx - 1, sy - 1))
    if not found:
        return None
    c = cv2.cornerSubPix(gray, c.reshape(-1, 2).astype(np.float32), (win, win), (-1, -1), _CRIT(eps))
    return _upright(c.reshape(-1, 2).astype(np.float64))


def _estimates(scene: Scene) -> dict[str, np.ndarray | None]:
    """Corner estimates (upright, row-major) of the usual OpenCV recipes on one scene."""
    sx, sy = scene.squares
    gray = cv2.cvtColor(scene.img, cv2.COLOR_BGR2GRAY)
    out: dict[str, np.ndarray | None] = dict.fromkeys(
        ("raw", "R5", "R8", "R11", "R11l", "sb", "sbR11"))
    found, c = cv2.findChessboardCorners(gray, (sx - 1, sy - 1))
    if found:
        c = c.reshape(-1, 2).astype(np.float32)
        out["raw"] = _upright(c.astype(np.float64))
        for name, win, eps in (("R5", 5, 0.001), ("R8", 8, 0.001), ("R11", 11, 0.001), ("R11l", 11, 0.1)):
            r = cv2.cornerSubPix(gray, c.copy(), (win, win), (-1, -1), _CRIT(eps))
            out[name] = _upright(r.reshape(-1, 2).astype(np.float64))
    found, c = cv2.findChessboardCornersSB(gray, (sx - 1, sy - 1))
    if found:
        c = c.reshape(-1, 2).astype(np.float32)
        out["sb"] = _upright(c.astype(np.float64))
        r = cv2.cornerSubPix(gray, c.copy(), (11, 11), (-1, -1), _CRIT(0.001))
        out["sbR11"] = _upright(r.reshape(-1, 2).astype(np.float64))
    return out


def _measure(scene: Scene) -> dict:
    est = _estimates(scene)
    err = {k: None if v is None else float(np.linalg.norm(v - scene.corners, axis=1).max()) for k, v in est.items()}
    return {"blur": scene.blur, "noise": scene.noise, "contrast": scene.contrast,
            "n": len(scene.corners), "smin": scene.min_square_px, "err": err}


def _gauge_block(task: tuple[int, float, float]) -> dict:
    """One single-board gauge block: pinned blur and noise/contrast ratio, random contrast and pose."""
    seed, blur, ratio = task
    cv2.setNumThreads(1)
    contrast = float(np.random.default_rng(seed + 7).uniform(60.0, 200.0))
    return _measure(_build(seed, blur=blur, noise=ratio * contrast, contrast=contrast))


def _prior_block(seed: int) -> dict:
    cv2.setNumThreads(1)
    return _measure(_build(seed))


def _map(fn, tasks, workers: int):
    if workers <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing
    # "spawn", not fork: a parent that has already run OpenCV's thread pool deadlocks forked children
    with multiprocessing.get_context("spawn").Pool(workers) as p:
        return p.map(fn, list(tasks), chunksize=8)


def fit_envelope(rows: np.ndarray) -> tuple[float, float, float]:
    """Upper-envelope fit of (A, W, K) to rows of (blur, noise, contrast, worst corner error).

    Least squares on (law - error) with a heavy penalty on law < error, minimised by a small
    numpy Nelder-Mead (no scipy)."""
    blur, noise, contrast, err = rows.T

    def cost(p):
        if p[1] <= 0.05 or min(p[0], p[2]) < 0:
            return 1e9
        law = np.array([_law(b, n, c, *p) for b, n, c in zip(blur, noise, contrast)])
        return float(np.mean((law - err) ** 2) + 1e5 * np.mean(np.maximum(0.0, err - law) ** 2))

    best = None
    for w0 in (0.3, 0.5):
        pts = [np.array([0.25, w0, 7.0])] + [np.array([0.25, w0, 7.0]) + d for d in np.diag([0.1, 0.15, 3.0])]
        vals = [cost(p) for p in pts]
        for _ in range(500):
            o = np.argsort(vals)
            pts, vals = [pts[i] for i in o], [vals[i] for i in o]
            c0 = np.mean(pts[:-1], axis=0)
            xr = c0 + (c0 - pts[-1])
            fr = cost(xr)
            if fr < vals[0]:
                xe = c0 + 2 * (c0 - pts[-1])
                fe = cost(xe)
                pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < vals[-2]:
                pts[-1], vals[-1] = xr, fr
            else:
                xc = c0 + 0.5 * (pts[-1] - c0)
                fc = cost(xc)
                if fc < vals[-1]:
                    pts[-1], vals[-1] = xc, fc
                else:
                    pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                    vals = [vals[0]] + [cost(p) for p in pts[1:]]
        i = int(np.argmin(vals))
        if best is None or vals[i] < best[1]:
            best = (pts[i], vals[i])
    return tuple(float(v) for v in best[0])


def _summarize(blocks: list[dict], refit: bool) -> dict:
    """Ratios of every estimator's worst corner error to the bare law and to the tolerance."""
    summary: dict = {"n_blocks": len(blocks), "estimators": {}}
    for est in ("R11", "R8", "R11l", "R5", "raw", "sb", "sbR11"):
        ratios, missing = [], 0
        for b in blocks:
            e = b["err"][est]
            if e is None:
                missing += 1
            else:
                ratios.append(e / _law(b["blur"], b["noise"], b["contrast"]))
        r = np.array(ratios)
        summary["estimators"][est] = {
            "not_found": missing,
            "worst_over_law": float(r.max()),
            "q99_over_law": float(np.quantile(r, 0.99)),
            "median_over_law": float(np.median(r)),
            "pass_fraction_at_tolerance": float(np.mean(r <= TOL_M)),
        }
    # detector failures (a corner mislocated by whole pixels) are not a property of the law
    ok = [b for b in blocks if all(b["err"][e] is not None and b["err"][e] < 3.0 for e in LAW_CLASS)]
    ratios = [max(b["err"][e] for e in LAW_CLASS) / _law(b["blur"], b["noise"], b["contrast"]) for b in ok]
    summary["law_class_worst_over_law"] = float(max(ratios))
    summary["law_class_blocks_excluded_as_detector_failures"] = len(blocks) - len(ok)
    summary["worst_over_tolerance"] = float(max(ratios)) / TOL_M
    if refit:
        rows = np.array([(b["blur"], b["noise"], b["contrast"], max(b["err"][e] for e in LAW_CLASS)) for b in ok])
        summary["refit_A_W_K"] = fit_envelope(rows)
    return summary


def calibrate(reps: int = 40, seed: int = 100_000, workers: int = 1, refit: bool = False) -> dict:
    """Gauge blocks: single-board scenes over the blur x (noise/contrast) grid (reps per cell;
    reps=40 is the recorded 3840-block measurement), every OpenCV recipe measured against the
    manufactured corners. ``refit=True`` re-derives A, W, K from the data."""
    tasks, s = [], seed
    for blur in GRID_BLUR:
        for ratio in GRID_RATIO:
            for _ in range(reps):
                tasks.append((s, blur, ratio))
                s += 1
    return _summarize(_map(_gauge_block, tasks, workers), refit)


def validate_on_prior(n: int = 6000, seed: int = 500_000, workers: int = 1) -> dict:
    """The same ratios on scenes drawn from make_scene's own distribution (fresh seeds)."""
    return _summarize(_map(_prior_block, range(seed, seed + n), workers), refit=False)


def window_sweep(n: int = 1500, seed: int = 700_000, windows=(2, 3, 4, 5, 6, 8, 11, 13, 15, 20)) -> dict:
    """Per-scene pass fraction at the tolerance of findChessboardCorners + cornerSubPix(win) per half-window."""
    hits = {w: 0 for w in windows}
    for sd in range(seed, seed + n):
        sc = _build(sd)
        gray = cv2.cvtColor(sc.img, cv2.COLOR_BGR2GRAY)
        found, c0 = cv2.findChessboardCorners(gray, (sc.squares[0] - 1, sc.squares[1] - 1))
        if not found:
            continue
        for w in windows:
            c = cv2.cornerSubPix(gray, c0.reshape(-1, 2).astype(np.float32), (w, w), (-1, -1), _CRIT(0.001))
            c = _upright(c.reshape(-1, 2).astype(np.float64))
            hits[w] += int(float(np.linalg.norm(c - sc.corners, axis=1).max()) <= tolerance(sc))
    return {w: hits[w] / n for w in windows}


def measure_metamorphic_drift(n: int = 600, seed: int = 800_000, win: int = 11, eps: float = 0.001) -> dict:
    """Largest set distance between the mapped base corners and the corners of each transformed image
    for findChessboardCorners + cornerSubPix(win, eps), plus how often the board was not found."""
    worst: dict[str, float] = {}
    not_found: dict[str, int] = {}
    for sd in range(seed, seed + n):
        sc = _build(sd)
        base = _refined(sc.img, *sc.squares, win=win, eps=eps)
        for name, inp in transforms(sc).items():
            c = _refined(inp["img"], inp["squares_x"], inp["squares_y"], win=win, eps=eps)
            if c is None or base is None:
                not_found[name] = not_found.get(name, 0) + 1
                continue
            worst[name] = max(worst.get(name, 0.0), _set_distance(_COVARIANT[name](base), c))
    return {"worst_drift_px": worst, "not_found": not_found}


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(workers=8, refit=True), indent=1))
    print(json.dumps(validate_on_prior(workers=8), indent=1))
    print(json.dumps(window_sweep(), indent=1))
    print(json.dumps(measure_metamorphic_drift(), indent=1))
    print(json.dumps(measure_metamorphic_drift(win=11, eps=0.1), indent=1))
