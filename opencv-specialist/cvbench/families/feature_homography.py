"""Family ``feature_homography``: match keypoints between two views of a flat scene and recover H.

Manufactured solution
---------------------
The answer is drawn first: a random similarity (rotation, scale, shift) composed with a perspective
distortion of img_b's frame gives the 3x3 homography H that maps img_a to img_b. Both images are then
rendered from one texture canvas (random shapes on a smooth colour field, optics blur): img_a is a crop of
the canvas, img_b is the canvas warped through H (bilinear), and each image gets independent Gaussian sensor
noise. img_b is larger than img_a and contains it, so img_a's inset corners always land inside img_b. The
pixel-centre convention is the one of ``cv2.warpPerspective`` (integer coordinates are pixel centres); a
test checks that warping img_a by H reproduces img_b to within 2 grey levels.

Tolerance physics
-----------------
The quantity compared is the *probe error* e: the largest distance, in img_b pixels, between where the
returned H and the true H send the four points of img_a's frame inset by GUARD = 32 px. A homography has 8
free parameters and the data constrain it only where keypoints exist, so the error has three sources:

1. Keypoint localization noise. Each correspondence is off by sigma_kp; a least-squares fit over N of them
   moves a probe point by about sigma_kp * sqrt(2 (1 + leverage) / N), so the error averages down as
   1/sqrt(N). N is the verifier-side count of true correspondences (``Scene.n_corr``): SIFT keypoints of
   img_a that img_b repeats within 1.5 px of where H sends them. A solution cannot see N; the verifier uses
   it to scale the tolerance with how much information the scene holds.
2. Extrapolation. Detectors reject keypoints near the border (ORB: edgeThreshold = 31 px; the measured ORB
   keypoint hull starts 31-44 px inside the frame), so the frame corners lie outside the data and a
   homography extrapolated there amplifies noise. Probing the corners of the frame inset by GUARD = 32 px
   (ORB's border) measures what the data constrain.
3. Estimator inefficiency, which does not average down. OpenCV's RANSAC refits on the consensus set of its
   winning hypothesis, and that set trims high-leverage points at the periphery. On the 12 worst ORB scenes
   of an early 3000-pair sweep the returned H was 2.7-6.0 px off at the probes, while a plain least-squares
   fit on the same truth-consistent matches was 0.2-1.9 px off.

The law is e <= TOL_A + TOL_B / sqrt(N). The draft had the same form, probed at the corners of the frame.
Calibration kept the form, moved the probes inward (source 2), found no use for an intercept (TOL_A = 0: at the same
false-reject budget a free fit gives a = 0.20 px, b = 55, i.e. 4.10 vs 4.17 px at N = 201), and showed that TOL_B
depends on which pipelines are accepted (below).

Calibration
-----------
``calibrate()`` (its defaults: n_bench=24000, n_wide=2000, n_floor=60, seed=2_000_000; ``python -m
cvbench.families.feature_homography`` runs it, about 10 minutes on 18 cores) on OpenCV 5.0.0 renders gauge blocks
and runs both reference pipelines (``reference_solve``: SIFT, ratio test, RANSAC, re-fit on the 1.5 px
inliers; ``orb_solve``: ORB(2000), Hamming, ratio test, plain RANSAC). Results (probe error e in px):

* Benchmark scenes (24000 pairs, N = 60-494, median 201; noise 0-6 grey levels, blur 0.5-1.0 px, rotation
  +-20 deg, scale 1/1.2-1.2, perspective 4-12 % of img_b's height).
  SIFT: e median 0.166, p95 0.32, max 1.32. e*sqrt(N): median 2.38, p95 4.14, p99.9 7.1, max 15.75; the
  median is 2.39 / 2.36 / 2.34 / 2.44 / 2.50 in the N bins 60-100 / 100-150 / 150-250 / 250-400 / 400+:
  the draft's 1/sqrt(N) law holds, with intercept 0.
  ORB: e median 0.656, p95 2.02, max 11.9 (35 pairs above 5 px). e*sqrt(N): median 9.2, p95 28.0, p99.9 68,
  max 125. Its median error falls only from 0.92 px (N ~ 80) to 0.54 px (N ~ 420) where 1/sqrt(N) predicts
  a factor 2.3: ORB's keypoints are about 4x noisier (integer positions on a pyramid) and the RANSAC fit
  adds an N-independent tail (source 3).
* Floor series (360 noiseless dense pairs, N = 90-892, median 603): SIFT e median 0.113, e*sqrt(N) median
  2.45; ORB e median 0.59, max 3.1. Neither falls toward zero as N grows.
* Warp size: rank correlation of e*sqrt(N) with rotation is +0.35 for SIFT, 0.08 for ORB; with noise +0.25
  (SIFT) and +0.09 (ORB); |rho| <= 0.12 with blur, perspective, log-scale, density. The SIFT rotation effect
  has a measured cause: OpenCV 5.0 SIFT reports every keypoint +0.246 px off in x and in y (a keypoint and
  its mirror image sit exactly 0.500 px apart, n ~ 2250 per axis; ORB: -0.09 to -0.12 px, level dependent).
  The offset is fixed in each image's own frame, so it stops cancelling once H rotates or scales one image
  against the other. Its size (0.25 px x |R - I| ~ 0.1 px at 20 degrees) is consistent with the ~0.1 px
  floor of the SIFT errors; the floor was not isolated any further. (2026-10-20: the offset is not new in 5.0;
  4.14 shows it too. It is the default ``enable_precise_upscale=False`` (opencv/opencv issue #23123); with
  ``SIFT_create(enable_precise_upscale=True)`` it falls below 0.03 px. All calibration here uses the default.)
* Draft probes (the true frame corners): SIFT e median 0.224 (vs 0.166 inset), max 2.23 (vs 1.32); ORB e
  median 1.07 (vs 0.66), max 18.9 (vs 11.9), 313 pairs above 5 px (vs 35). Hence the inset.
* Beyond the served scenes (2000 pairs: N down to 18, noise to 12, blur to 1.6 px, rotation to 30 deg):
  SIFT stays within 4.5 px but its e*sqrt(N) maximum doubles (29.6, at N = 43) below N = 60, which is why scenes need
  N >= N_MIN = 60; ORB returns no model in 6 of 2000 and exceeds 5 px in 11 %.
* Tails are heavy. The maximum keeps growing with the sample: SIFT e*sqrt(N) 7.0 (3000 pairs), 9.6 (12000),
  12.9 (9000, other seed), 15.75 (24000). A fresh 9000-pair sample exceeded a b = 8.8 envelope fitted on
  3000 pairs in 2 pairs. The recipe "worst case x 1.25" therefore only holds for an estimator whose tail is
  bounded, and the profiles below use it for SIFT alone.

Accuracy classes
----------------
Which pipelines count as correct is a decision, not a measurement. Both profiles are kept in
``TOLERANCE_PROFILES`` as (TOL_A, TOL_B, META_TOL); ``PROFILE`` picks the active one.

* ``sift+orb`` (active): TOL_B = 60 is the 99.8th percentile of plain ORB's e*sqrt(N) (59): 0.18 % of ORB
  pairs and none of the 24000 SIFT pairs exceed it, so a 20-scene gate rejects plain ORB with probability
  4 % (a false-reject budget of 0.2 % per scene, chosen; 3000 further make_scene scenes: 0.30 %). Tolerance 7.7 / 4.2 / 2.7 px at N = 60 / 201 / 494.
  Tail re-check, 2026-10-20 (96000 fresh pairs, seeds 7200000-7295999, 4x the calibration sample): SIFT 0
  rejects (95 % CI <= 0.004 %); ORB with Hamming 176 = 0.183 % [0.158, 0.212] (Wilson), within the budget, and
  stable across four 24000-pair blocks (51 / 33 / 45 / 47). Its maximum keeps growing with n (e*sqrt(N) 70 at
  1500 pairs, 401 at 48000; Hill tail index ~5.2) while its 99.8th percentile stays at 59 [57.5, 60.7], so the
  budget must be stated as a rate, never as a worst case. Mutants for comparison: ``orb_norm_l2`` 0.40 % per
  scene (2x the budget; 7.7 % of 20-scene gates), ``best_10_matches`` 1.8 % (31 % of gates; tail index ~1.6).
  On 200 scenes (seeds 7297000-7297199) the reference, ``no_ratio_test`` and ``rgb2gray`` pass every scene,
  ORB with Hamming 199, ``best_10_matches`` 197, ``orb_norm_l2`` 198.
  This matches the approved design, in which ORB with Hamming matching is a valid pipeline and ORB with
  NORM_L2 a mutant. It still separates gross mistakes (wrong direction, no RANSAC, affine instead of
  projective) by the 20-scene "all must pass" rule, but a single scene separates them only weakly.
* ``sift-only``: TOL_B = 20 is 1.27 x the largest SIFT e*sqrt(N) (15.75), tolerance 2.6 / 1.4 / 0.9 px; no
  SIFT pair exceeds it. Plain ORB exceeds it in 12.9 % of scenes and passes a 20-scene gate with probability
  6 %. It makes every ORB solution wrong and detects fewer-matches mistakes (``best_10_matches``).

META_TOL follows the same two rules: 1.25 x the largest SIFT discrepancy (1.50 px over 9000 scenes x 4
symmetries; 0.97 in a second run) for ``sift-only``, and the 99.8th percentile of ORB's discrepancy for
``sift+orb`` (5.64 px over 36000 comparisons, rounded up to 6.0).
The contract (``ACCURACY``) states the active accuracy to the model.

Metamorphic checks
------------------
Four symmetries, each with an H that is predictable from the base H and fixed constants (the verifier is not
given the scene): ``swap`` (exchange the images: H inverts), ``mirror`` (flip both left-right: H becomes
F_b H F_a), ``rot90`` (rotate both by 90 degrees) and ``crop`` (cut a different border off each image: H is
conjugated by the two translations). The variant's H must match the prediction at the probe points to
within META_TOL. A homography estimated by a randomized pipeline is not exactly equivariant (SIFT: median
discrepancy 0.06-0.17 px; ORB: 0.8-0.9 px, p95 2.4-2.7, max 11-28 over two runs), hence a calibrated constant. Blind
spots: it is a consistency check between two noisy estimates, so at the active tolerance it cannot see an
error smaller than META_TOL (the affine mutant passes it), and a consistent wrong answer passes any symmetry
it respects. The different sizes of img_a and img_b are what make mirror, rot90 and crop expose a constant
(identity) answer and a reversed direction: swapping the roles changes which frame each flip refers to.
"""
from __future__ import annotations

import inspect
import math
import textwrap
from dataclasses import dataclass, field

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

# The contract states the accuracy the verifier enforces (it is a toleranced drawing, not a trap): the four
# points 32 px inside img_a's corners must be mapped to within TOL_B / sqrt(N) px = 60 / sqrt(N), i.e. 2.7-7.7 px
# over the scenes served (N = 60-494 true keypoint correspondences; 4.2 px at the median N = 201).
ACCURACY = ("Accuracy requirement: the four points of img_a that lie 32 px inside its corners must be mapped to "
            "within about 4 px of their true positions in img_b (3 px for scenes rich in features, up to 8 px "
            "for sparse ones).")

SPEC = TaskSpec(
    family="feature_homography",
    task=("img_b shows the same flat, textured surface as img_a from a different viewpoint. Match local "
          "features between the two images and estimate the 3x3 homography H that maps pixel coordinates "
          "(x, y) in img_a to the corresponding pixel coordinates in img_b, normalized so that H[2, 2] = 1. "
          + ACCURACY),
    inputs={
        "img_a": IOSpec("uint8", "HxWx3", 3, "BGR, reference picture of a flat textured surface, about 216x288 px"),
        "img_b": IOSpec("uint8", "HxWx3", 3, "BGR, wider view (about 288x384 px) that contains that surface after a "
                                             "perspective change; independent sensor noise"),
    },
    outputs={
        "H": IOSpec("float", "3x3", 2, "maps (x, y) in img_a to (x, y) in img_b; x to the right, y down, pixel "
                                       "centres at integer coordinates; normalized so H[2, 2] = 1"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Two photos show the same planar, textured scene from different positions. Detect and match keypoints "
    "between img_a and img_b, then return the 3x3 homography H (with H[2, 2] = 1) that sends pixel "
    "coordinates in img_a to pixel coordinates in img_b. " + ACCURACY,
    "Estimate the projective mapping between two views of a flat textured surface: find the 3x3 matrix H, "
    "scaled so its bottom-right entry is 1, such that H applied to a pixel (x, y) of img_a gives the "
    "matching pixel of img_b. " + ACCURACY,
    "Register img_a onto img_b with feature matching. Return the 3x3 homography (float, H[2, 2] = 1) taking "
    "img_a pixel coordinates to img_b pixel coordinates. Both inputs are BGR uint8 images of one planar scene. "
    + ACCURACY,
    "Recover the planar homography that relates two images of the same textured scene. Return H with "
    "H[2, 2] = 1 so that the point (x, y) in img_a lands on the corresponding point in img_b. " + ACCURACY,
]

A_H, A_W = 216, 288  # img_a: a reference picture of the flat object
B_H, B_W = 288, 384  # img_b: a wider view that contains the object, seen from another viewpoint
CANVAS_H, CANVAS_W = 520, 600  # the plane's texture; both views are renderings of it
GUARD = 32  # px: probe points sit this far inside img_a's frame (see tolerance physics)
CORR_RADIUS = 1.5  # px: a keypoint pair counts as a true correspondence inside this radius
N_MIN = 60  # scenes need at least this many true correspondences to be well posed

# (TOL_A, TOL_B, META_TOL) per accuracy class; PROFILE picks the active one. Measured numbers are in the
# docstring ("Calibration", "Accuracy classes"):
#   sift+orb  TOL_B = 60   99.8th percentile of plain ORB's e*sqrt(N) = 59.0 (24000 pairs; 0.30 % of 3000 more
#                          make_scene scenes exceed it); 0 of 24000 SIFT pairs exceed it.
#             META_TOL = 6.0   99.8th percentile of ORB's symmetry discrepancy = 5.64 (36000 comparisons).
#   sift-only TOL_B = 20   1.27 x the largest SIFT e*sqrt(N) = 15.75 (24000 pairs).
#             META_TOL = 1.9   1.25 x the largest SIFT discrepancy = 1.50 (36000 comparisons, other seed 0.97).
TOLERANCE_PROFILES = {"sift+orb": (0.0, 60.0, 6.0), "sift-only": (0.0, 20.0, 1.9)}
PROFILE = "sift+orb"
TOL_A, TOL_B, META_TOL = TOLERANCE_PROFILES[PROFILE]
SYMMETRIES = ("swap", "mirror", "rot90", "crop")
NORM_TOL = 1e-6  # |H[2, 2] - 1|: float representation, not a measurement tolerance

# Crop windows for the "crop" symmetry: pixels removed from each side of img_a / img_b.
CROP_A, CROP_B = (16, 12), (24, 20)  # (x, y)


@dataclass
class Scene:
    img_a: np.ndarray
    img_b: np.ndarray
    H: np.ndarray  # 3x3 float64, H[2, 2] = 1: maps img_a pixel coordinates to img_b pixel coordinates
    n_corr: int  # SIFT keypoints of img_a whose true image has a SIFT keypoint of img_b within CORR_RADIUS px
    knobs: dict = field(default_factory=dict)
    seed: int = 0


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- geometry helpers

def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points (float64, homogeneous division)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def frame_corners(inset: float = 0.0, h: int = A_H, w: int = A_W) -> np.ndarray:
    return np.float64([[inset, inset], [w - 1 - inset, inset], [w - 1 - inset, h - 1 - inset], [inset, h - 1 - inset]])


def probe_error(H_est: np.ndarray, H_true: np.ndarray, pts: np.ndarray | None = None) -> float:
    """Largest distance (px) between the two mappings of the probe points (default: img_a's inset corners)."""
    pts = frame_corners(GUARD) if pts is None else pts
    d = np.linalg.norm(project(H_est, pts) - project(H_true, pts), axis=1)
    return float(np.max(d)) if np.all(np.isfinite(d)) else float("inf")


# ---------------------------------------------------------------- scene generation

def _texture(rng: np.random.Generator, h: int, w: int, density: float, blur: float) -> np.ndarray:
    """A flat 'poster': smooth colour field plus ``density`` random shapes per 1000 px^2, then optics blur."""
    low = rng.normal(0, 1, (h // 24 + 2, w // 24 + 2)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    tint = rng.uniform(0.6, 1.0, 3)
    img = np.clip(125 + 28 * low[..., None] * tint[None, None, :], 0, 255).astype(np.uint8)
    for _ in range(int(density * h * w / 1000.0)):
        kind = int(rng.integers(5))
        col = tuple(int(c) for c in rng.integers(15, 241, 3))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = float(np.exp(rng.uniform(math.log(4), math.log(36))))
        if kind == 0:
            cv2.rectangle(img, (cx, cy), (cx + int(s), cy + int(s * rng.uniform(0.4, 1.6))), col, -1)
        elif kind == 1:
            cv2.ellipse(img, (cx, cy), (max(2, int(s / 2)), max(2, int(s / 2 * rng.uniform(0.4, 1)))),
                        float(rng.uniform(0, 180)), 0, 360, col, -1)
        elif kind == 2:
            pts = np.array([[cx + s * rng.uniform(-1, 1), cy + s * rng.uniform(-1, 1)] for _ in range(3)])
            cv2.fillPoly(img, [np.round(pts).astype(np.int32)], col)
        elif kind == 3:
            a = rng.uniform(0, 2 * math.pi)
            cv2.line(img, (cx, cy), (int(cx + s * math.cos(a)), int(cy + s * math.sin(a))), col, int(rng.integers(1, 4)))
        else:
            cv2.circle(img, (cx, cy), max(2, int(s / 3)), col, int(rng.choice([-1, 2])))
    return cv2.GaussianBlur(img, (0, 0), blur) if blur > 0 else img


def _count_correspondences(img_a: np.ndarray, img_b: np.ndarray, H: np.ndarray) -> int:
    """Verifier-side information content N: SIFT keypoints of img_a that img_b repeats where H says they go."""
    sift = cv2.SIFT_create()
    ka = sift.detect(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None)
    kb = sift.detect(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None)
    if not ka or not kb:
        return 0
    pa = project(H, np.array([k.pt for k in ka]))
    pb = np.array([k.pt for k in kb])
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min(axis=1)
    return int(np.sum(d <= CORR_RADIUS))


def _similarity(theta_deg: float, scale: float, t: tuple[float, float]) -> np.ndarray:
    """Canvas -> img_b: rotate by theta and scale about the canvas centre, put that centre at img_b's centre + t."""
    th = math.radians(theta_deg)
    c0 = np.array([(CANVAS_W - 1) / 2, (CANVAS_H - 1) / 2])
    b0 = np.array([(B_W - 1) / 2 + t[0], (B_H - 1) / 2 + t[1]])
    R = scale * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    S = np.eye(3)
    S[:2, :2] = R
    S[:2, 2] = b0 - R @ c0
    return S


def render(seed: int, density: float, noise: float, blur: float, rot_max: float, scale_max: float,
           shift_max: float, persp: float) -> Scene:
    """Manufacture one pair: the answer H is drawn first, then img_b is the texture seen through it.

    rot_max (deg), scale_max (> 1) and shift_max (fraction of img_b's size) bound the similarity part of H;
    persp (fraction of img_b's height) moves the four corners of img_b's frame independently, which adds the
    projective part. Draws that put img_a's probe points outside img_b, or img_b's footprint outside the
    texture, are redrawn.
    """
    rng = np.random.default_rng(seed)
    canvas = _texture(rng, CANVAS_H, CANVAS_W, density, blur)
    oy, ox = (CANVAS_H - A_H) // 2, (CANVAS_W - A_W) // 2
    img_a = canvas[oy:oy + A_H, ox:ox + A_W].copy()
    bframe = frame_corners(0, B_H, B_W)
    for _ in range(500):
        th = rng.uniform(-rot_max, rot_max)
        s = math.exp(rng.uniform(-math.log(scale_max), math.log(scale_max)))
        t = (rng.uniform(-shift_max, shift_max) * B_W, rng.uniform(-shift_max, shift_max) * B_H)
        quad = bframe + rng.uniform(-persp, persp, (4, 2)) * B_H
        edges = np.roll(quad, -1, axis=0) - quad
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)  # turn at each corner
        if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
            continue
        G = cv2.getPerspectiveTransform(bframe.astype(np.float32), quad.astype(np.float32)).astype(np.float64)
        H = G @ _similarity(th, s, t) @ np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1.0]])  # img_a -> canvas -> img_b
        H /= H[2, 2]
        Hc = H @ np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1.0]])  # canvas -> img_b, exactly consistent with H
        back = project(np.linalg.inv(Hc), bframe)
        probes = project(H, frame_corners(GUARD))
        inside = bool(np.all((probes > 6) & (probes < [B_W - 7, B_H - 7])))
        if inside and back.min() > 3 and back[:, 0].max() < CANVAS_W - 4 and back[:, 1].max() < CANVAS_H - 4:
            break
    else:
        raise RuntimeError("no admissible homography found")
    img_b = cv2.warpPerspective(canvas, Hc, (B_W, B_H), flags=cv2.INTER_LINEAR)
    if noise > 0:
        img_a = np.clip(img_a.astype(np.float32) + rng.normal(0, noise, img_a.shape), 0, 255).round().astype(np.uint8)
        img_b = np.clip(img_b.astype(np.float32) + rng.normal(0, noise, img_b.shape), 0, 255).round().astype(np.uint8)
    return Scene(img_a, img_b, H, _count_correspondences(img_a, img_b, H),
                 dict(density=density, noise=noise, blur=blur, rot=float(th), scale=float(s), persp=persp), seed)


def _draw_knobs(rng: np.random.Generator, domain: str) -> dict:
    if domain == "bench":  # the scenes the benchmark serves
        return dict(density=float(np.exp(rng.uniform(math.log(1.5), math.log(6.0)))), noise=float(rng.uniform(0, 6)),
                    blur=float(rng.uniform(0.5, 1.0)), rot_max=20.0, scale_max=1.2, shift_max=0.05,
                    persp=float(rng.uniform(0.04, 0.12)))
    if domain == "wide":  # calibration also probes beyond them: sparser, noisier, blurrier, bigger warps
        return dict(density=float(np.exp(rng.uniform(math.log(0.3), math.log(12.0)))), noise=float(rng.uniform(0, 12)),
                    blur=float(rng.uniform(0.5, 1.6)), rot_max=30.0, scale_max=1.35, shift_max=0.08,
                    persp=float(rng.uniform(0.01, 0.12)))
    raise ValueError(domain)


def make_scene(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        knobs = _draw_knobs(rng, "bench")
        scene = render(int(rng.integers(2 ** 31)), **knobs)
        scene.seed = seed
        if scene.n_corr >= N_MIN:
            return scene
    raise RuntimeError(f"seed {seed}: no well-posed scene in 20 attempts")


def inputs(scene: Scene) -> dict:
    return {"img_a": scene.img_a, "img_b": scene.img_b}


# ---------------------------------------------------------------- verifiers

def tolerance(scene: Scene, profile: str | None = None) -> float:
    a, b, _ = TOLERANCE_PROFILES[profile or PROFILE]
    return a + b / math.sqrt(scene.n_corr)


def _matrix(out: dict, what: str = "H") -> tuple[np.ndarray | None, str]:
    """The returned homography as a float64 3x3, or (None, reason)."""
    problems = check_outputs(SPEC, out)
    if problems:
        return None, "contract: " + "; ".join(problems)
    H = np.asarray(out["H"], dtype=np.float64)
    if H.shape != (3, 3):
        return None, f"contract: {what} should be 3x3, got shape {H.shape}"
    if not np.all(np.isfinite(H)):
        return None, f"{what} contains NaN or inf"
    return H, ""


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    H, why = _matrix(out)
    if H is None:
        return Verdict(False, why)
    if abs(H[2, 2] - 1.0) > NORM_TOL:
        return Verdict(False, f"contract: H[2, 2] = {H[2, 2]:.6g}, but the homography must be normalized to 1")
    err, tol = probe_error(H, scene.H), tolerance(scene)
    if not err <= tol:
        return Verdict(False, f"inset-corner error {err:.2f} px exceeds {tol:.2f} px "
                              f"(N = {scene.n_corr} true correspondences)")
    return Verdict(True)


def _flip_x(w: int) -> np.ndarray:
    return np.array([[-1.0, 0, w - 1], [0, 1, 0], [0, 0, 1]])


def _rot90(w: int) -> np.ndarray:
    """np.rot90 on an image of width w: pixel (x, y) moves to (y, w - 1 - x)."""
    return np.array([[0.0, 1, 0], [-1, 0, w - 1], [0, 0, 1]])


def _shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0, -dx], [0, 1, -dy], [0, 0, 1]])


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task. Each variant's H is predictable from the base H and constants alone."""
    a, b = scene.img_a, scene.img_b
    ca, cb = CROP_A, CROP_B
    return {
        "swap": {"img_a": b, "img_b": a},
        "mirror": {"img_a": np.ascontiguousarray(a[:, ::-1]), "img_b": np.ascontiguousarray(b[:, ::-1])},
        "rot90": {"img_a": np.ascontiguousarray(np.rot90(a)), "img_b": np.ascontiguousarray(np.rot90(b))},
        "crop": {"img_a": np.ascontiguousarray(a[ca[1]:A_H - ca[1], ca[0]:A_W - ca[0]]),
                 "img_b": np.ascontiguousarray(b[cb[1]:B_H - cb[1], cb[0]:B_W - cb[0]])},
    }


def discrepancy(name: str, H_base: np.ndarray, H_var: np.ndarray) -> float:
    """Largest distance (px) between the variant's H and the H the symmetry predicts from the base H,
    at the inset corners of the variant's first image."""
    if name == "swap":  # roles exchanged: H_var should invert H_base. Probe where H_base sends img_a's probes.
        try:
            inverse = np.linalg.inv(H_base)
        except np.linalg.LinAlgError:  # a singular answer is inconsistent with everything
            return float("inf")
        return probe_error(H_var, inverse, project(H_base, frame_corners(GUARD)))
    if name == "mirror":
        Ma, Mb, shape = _flip_x(A_W), _flip_x(B_W), (A_H, A_W)
    elif name == "rot90":
        Ma, Mb, shape = _rot90(A_W), _rot90(B_W), (A_W, A_H)
    elif name == "crop":
        Ma, Mb = _shift(*CROP_A), _shift(*CROP_B)
        shape = (A_H - 2 * CROP_A[1], A_W - 2 * CROP_A[0])
    else:
        raise KeyError(name)
    H_pred = Mb @ H_base @ np.linalg.inv(Ma)
    return probe_error(H_var, H_pred, frame_corners(GUARD, *shape))


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    H0, why = _matrix(base)
    if H0 is None:
        return Verdict(False, why)
    for name, out in variants.items():
        if name not in SYMMETRIES:
            return Verdict(False, f"{name}: unknown symmetry")
        Hv, why = _matrix(out)
        if Hv is None:
            return Verdict(False, f"{name}: {why}")
        d = discrepancy(name, H0, Hv)
        if not d <= META_TOL:
            return Verdict(False, f"{name}: H differs from the symmetry's prediction by {d:.2f} px "
                                  f"(allowed {META_TOL})")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img_a, img_b):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    err = np.linalg.norm(cv2.perspectiveTransform(src, H) - dst, axis=2).ravel()
    keep = err < 1.5
    if keep.sum() >= 8:
        H, _ = cv2.findHomography(src[keep], dst[keep], 0)
    return {"H": H / H[2, 2]}


def orb_solve(img_a, img_b):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2000)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return {"H": H / H[2, 2]}


def _as_solve(fn) -> str:
    return ("import cv2\nimport numpy as np\n\n\n"
            + textwrap.dedent(inspect.getsource(fn)).replace(f"def {fn.__name__}(", "def solve("))


REFERENCE = _as_solve(reference_solve)
ORB_ALTERNATIVE = _as_solve(orb_solve)


def _mutate(code: str, *edits: tuple[str, str]) -> str:
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


def _swallow(code: str) -> str:
    """Wrap solve()'s body in try/except Exception that returns the identity homography."""
    head, body = code.split("def solve(img_a, img_b):\n")
    body = "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True))
    return head + ("def solve(img_a, img_b):\n    try:\n" + body.rstrip("\n")
                   + "\n    except Exception:\n        return {\"H\": np.eye(3)}\n")


_MATCH = ("    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)\n"
          "    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]\n")
_SRC = "    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)\n"
_DST = "    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)\n"
_FIT = ("    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)\n"
        "    err = np.linalg.norm(cv2.perspectiveTransform(src, H) - dst, axis=2).ravel()\n"
        "    keep = err < 1.5\n"
        "    if keep.sum() >= 8:\n"
        "        H, _ = cv2.findHomography(src[keep], dst[keep], 0)\n")

_AKAZE = _mutate(REFERENCE, ("cv2.SIFT_create()", "cv2.AKAZE_create()"), ("cv2.NORM_L2", "cv2.NORM_HAMMING"))

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "akaze_removed": (
        "4.x idiom: cv2.AKAZE_create() is absent from the 5.0 main wheel. Raises AttributeError.",
        _AKAZE),
    "swallowed_akaze_identity": (
        "The same bug wrapped in try/except, falling back to the identity homography. Exec-only keeps it.",
        _swallow(_AKAZE)),
    "orb_norm_l2 (harmless here)": (
        "ORB's binary descriptors matched with NORM_L2 instead of NORM_HAMMING. L2 on the descriptor bytes ranks "
        "neighbours almost as well, and the ratio test and RANSAC do the rest: inside the ORB accuracy class.",
        _mutate(ORB_ALTERNATIVE, ("cv2.NORM_HAMMING", "cv2.NORM_L2"))),
    "no_ratio_test (harmless here)": (
        "Nearest-neighbour matches with no Lowe ratio test: more wrong matches reach RANSAC, and RANSAC with the "
        "1.5 px re-fit removes them again.",
        _mutate(REFERENCE, (_MATCH, "    good = cv2.BFMatcher(cv2.NORM_L2).match(des_a, des_b)\n"))),
    "no_ransac": (
        "findHomography with its default method (0 = least squares on every match, outliers included).",
        _mutate(REFERENCE, (_FIT, "    H, _ = cv2.findHomography(src, dst)\n"))),
    "perspective_transform_Nx2": (
        "Keypoints left as (N, 2): findHomography accepts them but perspectiveTransform needs (N, 1, 2). "
        "Raises cv2.error.",
        _mutate(REFERENCE, (_SRC, "    src = np.float32([kp_a[m.queryIdx].pt for m in good])\n"),
                (_DST, "    dst = np.float32([kp_b[m.trainIdx].pt for m in good])\n"))),
    "swapped_direction": (
        "Fits the mapping from img_b to img_a (src and dst exchanged): the inverse of the requested H.",
        _mutate(REFERENCE, ("kp_a[m.queryIdx]", "kp_TMP"), ("kp_b[m.trainIdx]", "kp_a[m.queryIdx]"),
                ("kp_TMP", "kp_b[m.trainIdx]"))),
    "affine_instead_of_homography": (
        "estimateAffine2D (6 degrees of freedom) padded to 3x3: no perspective terms.",
        _mutate(REFERENCE, (_FIT, "    M, _ = cv2.estimateAffine2D(src, dst)\n    H = np.vstack([M, [0.0, 0.0, 1.0]])\n"))),
    "best_10_matches (within accepted accuracy)": (
        "Keeps only the 10 best matches: right idea, but 10 points cannot average the keypoint noise down. Its "
        "error is ORB-sized, so it passes the active tolerance and fails the sift-only one.",
        _mutate(REFERENCE, (_SRC, "    good = sorted(good, key=lambda m: m.distance)[:10]\n" + _SRC))),
    "orb_hamming (valid alternative)": (
        "ORB + Hamming + ratio test + RANSAC: a different, correct pipeline, about 4x noisier than SIFT's. Must "
        "pass (it sets the tolerance: see the module docstring).",
        ORB_ALTERNATIVE),
    "rgb2gray (harmless here)": (
        "Channel order cannot matter: both images go through the same conversion.",
        _mutate(REFERENCE, ("COLOR_BGR2GRAY", "COLOR_RGB2GRAY"))),
}


# ---------------------------------------------------------------- calibration

PIPELINES = {"sift": reference_solve, "orb": orb_solve}
FLOOR_DENSITIES = (2, 4, 8, 16, 32, 64)  # textures so dense that N is large and keypoint noise is the only error
REJECT_BUDGET = 0.002  # per scene: the share of plain-ORB pairs the active tolerance may reject (a policy choice)


def _pool(processes: int):
    """A worker pool that starts fresh interpreters: forking after OpenCV has spun up its threads can deadlock."""
    import multiprocessing
    return multiprocessing.get_context("spawn").Pool(processes)


def _pair_worker(args):
    """One gauge block: render a pair, run both reference pipelines, record the probe error and N."""
    seed, knobs = args
    cv2.setNumThreads(1)
    sc = render(seed, **knobs)
    rec = {"N": sc.n_corr, **{k: float(v) for k, v in knobs.items()},
           "rot": abs(sc.knobs["rot"]), "log_scale": abs(math.log(sc.knobs["scale"]))}
    for name, fn in PIPELINES.items():
        try:
            H = fn(sc.img_a, sc.img_b)["H"]
            rec[name] = probe_error(H, sc.H)  # at the inset corners (the verifier's metric)
            rec[name + "_frame"] = probe_error(H, sc.H, frame_corners(0))  # at the frame corners (the draft law)
        except (cv2.error, TypeError):  # TypeError: findHomography found no model and returned None
            rec[name] = rec[name + "_frame"] = float("inf")
    return rec


def fit_budget_envelope(N: np.ndarray, e_sift: np.ndarray, e_orb: np.ndarray,
                        budget: float = REJECT_BUDGET) -> tuple[float, float]:
    """Intercept and slope of the tolerance a + b / sqrt(N) with the smallest mean value over these N such that
    no SIFT pair and at most ``budget`` of the ORB pairs lie above it. Used to check that TOL_A = 0 loses nothing."""
    best = (np.inf, 0.0, 0.0)
    root = np.sqrt(N)
    for a in np.linspace(0.0, 6.0, 241):
        b = max(0.0, float(np.quantile((e_orb - a) * root, 1 - budget)), float(np.max((e_sift - a) * root)))
        cost = float(np.mean(a + b / root))
        if cost < best[0]:
            best = (cost, a, b)
    return best[1], best[2]


def _summ(N: np.ndarray, e: np.ndarray) -> dict:
    x = e * np.sqrt(N)
    return {"err_med": float(np.median(e)), "err_p95": float(np.percentile(e, 95)), "err_max": float(e.max()),
            "e_sqrtN_med": float(np.median(x)), "e_sqrtN_p95": float(np.percentile(x, 95)),
            "e_sqrtN_max": float(x.max()), "n_above_5px": int(np.sum(e > 5))}


def calibrate(n_bench: int = 24_000, n_wide: int = 2_000, n_floor: int = 60, seed: int = 2_000_000,
              processes: int = 1, keep_records: bool = False) -> dict:
    """Measure the reference pipelines' probe error on gauge blocks (the defaults are the recorded run).

    bench: the scenes the benchmark serves. wide: sparser, noisier, blurrier, bigger warps. floor: noiseless
    dense textures (``n_floor`` for each of FLOOR_DENSITIES), to expose any N-independent floor. Both pipelines
    (SIFT, ORB) run on every pair; ``rules`` re-derives the two TOL_B values from the bench pairs.
    """
    rng = np.random.default_rng(seed)
    jobs = {"bench": [(seed + i, _draw_knobs(rng, "bench")) for i in range(n_bench)],
            "wide": [(seed + 10_000 + i, _draw_knobs(rng, "wide")) for i in range(n_wide)],
            "floor": [(seed + 20_000 + i, dict(density=float(d), noise=0.0, blur=0.6, rot_max=20.0, scale_max=1.2,
                                                   shift_max=0.05, persp=0.08))
                      for i, d in enumerate(np.repeat(FLOOR_DENSITIES, n_floor))]}
    if processes > 1:
        with _pool(processes) as pool:
            recs = {k: pool.map(_pair_worker, v, chunksize=4) for k, v in jobs.items()}
    else:
        recs = {k: [_pair_worker(j) for j in v] for k, v in jobs.items()}
    out: dict = {}
    for dom, rs in recs.items():
        if not rs:
            continue
        N = np.array([r["N"] for r in rs], float)
        out[dom] = {"pairs": len(rs), "N": (float(N.min()), float(np.median(N)), float(N.max()))}
        for name in PIPELINES:
            ok = np.array([np.isfinite(r[name]) for r in rs])
            out[dom][name] = {"failures": int((~ok).sum()),
                              "inset32": _summ(N[ok], np.array([r[name] for r in rs])[ok]),
                              "frame_corners": _summ(N[ok], np.array([r[name + "_frame"] for r in rs])[ok])}
    bench = recs["bench"]
    N = np.array([r["N"] for r in bench], float)
    e = {nm: np.array([r[nm] for r in bench]) for nm in PIPELINES}
    x = {nm: e[nm] * np.sqrt(N) for nm in PIPELINES}
    a_free, b_free = fit_budget_envelope(N, e["sift"], e["orb"])
    out["rules"] = {"sift-only TOL_B = 1.25 x max SIFT e*sqrt(N)": 1.25 * float(x["sift"].max()),
                    f"sift+orb TOL_B = {1 - REJECT_BUDGET:.1%} quantile of ORB e*sqrt(N)": float(np.quantile(x["orb"], 1 - REJECT_BUDGET)),
                    "free-intercept fit at the same budget (a, b)": (a_free, b_free)}
    wide = [r for r in recs["wide"] if r["N"] >= N_MIN]
    out["share_above_profile"] = {}
    for prof, (a, b, _) in TOLERANCE_PROFILES.items():
        out["share_above_profile"][prof] = {
            nm: {"bench": float(np.mean(e[nm] > a + b / np.sqrt(N))),
                 "wide": float(np.mean([r[nm] > a + b / math.sqrt(r["N"]) for r in wide])) if wide else None}
            for nm in PIPELINES}
    # evidence for the scaling law: error against N, and against the size of the warp
    out["by_N"] = {}
    for lo, hi in ((60, 100), (100, 150), (150, 250), (250, 400), (400, 10_000)):
        m = (N >= lo) & (N < hi)
        if m.sum() >= 5:
            out["by_N"][f"[{lo},{hi})"] = {"pairs": int(m.sum()), **{
                nm: {"err_med": float(np.median(e[nm][m])), "err_max": float(e[nm][m].max()),
                     "e_sqrtN_med": float(np.median(x[nm][m])), "e_sqrtN_max": float(x[nm][m].max())}
                for nm in PIPELINES}}

    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)
    out["rank_corr_with_error"] = {
        nm: {k: float(np.corrcoef(rank([r[k] for r in bench]), rank(x[nm]))[0, 1])
             for k in ("noise", "blur", "persp", "rot", "log_scale", "density")} for nm in PIPELINES}
    if keep_records:
        out["records"] = recs
    return out


def measure_metamorphic_drift(n: int = 9_000, seed: int = 3_000_000, processes: int = 1, keep_rows: bool = False) -> dict:
    """Discrepancy between each reference pipeline's variant H and the symmetry's prediction, over n scenes
    (the defaults are the recorded run). ``pooled`` has one comparison per scene and symmetry."""
    if processes > 1:
        with _pool(processes) as pool:
            rows = pool.map(_drift_worker, range(seed, seed + n), chunksize=2)
    else:
        rows = [_drift_worker(s) for s in range(seed, seed + n)]
    out: dict = {}
    for name in PIPELINES:
        out[name] = {}
        for t in SYMMETRIES:
            d = np.array([r[name][t] for r in rows])
            out[name][t] = {"med": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
        pooled = np.array([r[name][t] for r in rows for t in SYMMETRIES])
        out[name]["pooled"] = {"q99.8": float(np.quantile(pooled, 1 - REJECT_BUDGET)),
                               "q99.9": float(np.quantile(pooled, 0.999)),
                               "max": float(pooled.max()), "comparisons": int(pooled.size)}
        if keep_rows:
            out[name]["pooled"]["values"] = pooled.tolist()
    out["worst"] = max(out[name]["pooled"]["max"] for name in PIPELINES)
    return out


def _drift_worker(seed: int) -> dict:
    cv2.setNumThreads(1)
    sc = make_scene(seed)
    variants = transforms(sc)
    row = {}
    for name, fn in PIPELINES.items():
        try:
            H0 = fn(**inputs(sc))["H"]
            row[name] = {t: discrepancy(t, H0, fn(**v)["H"]) for t, v in variants.items()}
        except (cv2.error, TypeError):  # a pipeline that finds no model counts as an infinite discrepancy
            row[name] = {t: float("inf") for t in variants}
    return row


if __name__ == "__main__":  # about 15 minutes on 18 cores
    import json
    import os
    workers = os.cpu_count() or 1
    print(json.dumps(calibrate(processes=workers), indent=1))
    print(json.dumps(measure_metamorphic_drift(processes=workers), indent=1))
