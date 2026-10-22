"""Drills on APIs that are gone from the OpenCV 5.0 wheel, each with the replacement that still exists.

Measured (probes/results/diff_4.14_vs_5.0.md, "existence"; replacements from cvbench/api_check.SUGGESTIONS):
  KCF / CSRT / GOTURN trackers and ``cv2.legacy``  -> ``cv2.TrackerMIL_create`` (the others need an ONNX file)
  ``cv2.estimateRigidTransform``                   -> ``cv2.estimateAffinePartial2D`` (``fullAffine=False``)
  ``cv2.ml`` (KNearest, SVM, ...)                  -> numpy (or scikit-learn)
  ``cv2.BOWKMeansTrainer``                         -> ``cv2.kmeans`` and a histogram built by hand
  ``cv2.createBackgroundSubtractorMOG`` / bgsegm   -> ``cv2.createBackgroundSubtractorMOG2``
  ``cv2.CV_AA``                                    -> ``cv2.LINE_AA``
  AKAZE / KAZE / BRISK, SURF (``xfeatures2d``)     -> ``cv2.ORB_create`` or ``cv2.SIFT_create``

Not drilled, and why: ``CascadeClassifier`` / Haar XML, ``readNetFromCaffe`` / ``Darknet`` and the GOTURN / Nano / Vit
models need files the sandbox does not have; ``HOGDescriptor`` has no confirmed replacement and a verified numpy HOG would
be a task of its own; ``cv2.gapi`` has no use in a one-function task; ``AgastFeatureDetector`` is the one removed detector
whose name a task cannot mention without naming the API the answer must avoid.

A prompt never names the removed API. The wrong variants are the plausible 4.x answers; each raises AttributeError.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import (PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, spec)

GROUP = "removed"
R = Registry()
S = SEED_BASE + 3_000  # this module's seeds: 64_000 .. 64_999


def _iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# ---------------------------------------------------------------- 1. trackers: KCF / CSRT / GOTURN are gone, MIL is not

def drift_clip(seed: int, T: int = 12, H: int = 120, W: int = 160):
    """T gray frames: a 40-52 px square patch of high-contrast blocks slides 1-3 px per frame (and up to 1 px vertically)
    over a smooth, noisy background. Truth: the patch's box in the first and the last frame."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2.0)
    bg = 75 + 30 * (bg - bg.min()) / (bg.max() - bg.min())
    s = int(rng.integers(40, 53))
    patch = cv2.resize(np.round(rng.integers(0, 2, (4, 4)).astype(np.float32)) * 120 + 130, (s, s), interpolation=cv2.INTER_NEAREST)
    x0, y0 = int(rng.integers(10, 30)), int(rng.integers(20, H - s - 20))
    vx, vy = int(rng.integers(1, 4)), int(rng.integers(-1, 2))
    frames = []
    for t in range(T):
        f = bg.copy()
        x, y = x0 + vx * t, y0 + vy * t
        f[y:y + s, x:x + s] = patch
        frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
    return np.stack(frames), (x0, y0, s, s), (x0 + vx * (T - 1), y0 + vy * (T - 1), s, s)


# The reference (cv2.TrackerMIL_create) is not a strong tracker: on 119 clips of this kind it ends above IoU 0.70 on 36 of them
# and a box that never moved overlaps the truth with IoU 0.07..0.25 on those. The cases are three seeds (measured once, at
# build time) on which MIL ends at IoU 1.00, 0.92 and 0.95, a tracker that never moves would score 0.13, 0.14 and 0.13, and
# whose final boxes overlap each other by at most 0.28, so the first case's box is no answer to the others. The verifier asks
# for IoU >= 0.40, between the two.
TRACK_MIN_IOU = 0.40
TRACK_SEEDS = (S + 37, S + 59, S + 83)

_TR_REF = code("frames, box0", '''\
    tracker = cv2.TrackerMIL_create()
    tracker.init(frames[0], tuple(int(v) for v in box0))
    box = tuple(int(v) for v in box0)
    for frame in frames[1:]:
        ok, box = tracker.update(frame)
    return {"box": np.array([int(round(v)) for v in box])}
''')


def _tr_cases():
    out = []
    for seed in TRACK_SEEDS:
        frames, box0, box_last = drift_clip(seed)
        out.append(Case({"frames": frames, "box0": np.array(box0, np.int32)}, box_last))
    return out


def _tr_verify(case: Case, out: dict) -> Verdict:
    box = [int(v) for v in arr(out, "box")]
    iou = _iou(box, case.truth)
    return PASS if iou >= TRACK_MIN_IOU else fail(f"final box {box} overlaps the object's true box {list(case.truth)} with IoU {iou:.2f}")


def _tr_wrong(creator: str) -> str:
    return code("frames, box0", f'    tracker = {creator}\n    tracker.init(frames[0], tuple(int(v) for v in box0))\n'
                                '    for frame in frames[1:]:\n        ok, box = tracker.update(frame)\n'
                                '    return {"box": np.array([int(round(v)) for v in box])}\n')


R.add(Check(
    id="drill_track_box", group=GROUP, probes=("has cv2.TrackerGOTURN_create", "has cv2.TrackerKCF_create"),
    depends_on_5_0=True, role="gotcha",
    gotcha="TrackerKCF / TrackerCSRT / TrackerGOTURN and cv2.legacy are not in the 5.0 wheel: cv2.TrackerMIL_create is the one "
           "tracker that needs no model file",
    spec=spec("Follow a bright square patch through a short grayscale clip with one of OpenCV's trackers: initialise it on "
              "the first frame with the given box and report the object's box in the last frame.",
              {"frames": IOSpec("uint8", "TxHxW", 3, "T grayscale frames; the patch moves a few pixels per frame"),
               "box0": IOSpec("int", "4", 1, "x, y, width, height of the patch in the first frame")},
              {"box": IOSpec("int", "4", 1, "x, y, width, height of the patch in the last frame")}),
    make_cases=_tr_cases, verify=_tr_verify, reference=_TR_REF,
    lint=True,
    note="TrackerMIL is the only built-in tracker that needs no model file, which is why the cases are chosen where it holds on",
    wrong={
        "kcf": Wrong("cv2.TrackerKCF_create(): the classic tutorial tracker, absent from the 5.0 wheel", _tr_wrong("cv2.TrackerKCF_create()"),
                     "AttributeError", "TrackerKCF_create"),
        "csrt": Wrong("cv2.TrackerCSRT_create()", _tr_wrong("cv2.TrackerCSRT_create()"), "AttributeError", "TrackerCSRT_create"),
        "goturn": Wrong("cv2.TrackerGOTURN_create(): removed from the 5.0 wheel (and needs a model file)",
                        _tr_wrong("cv2.TrackerGOTURN_create()"), "AttributeError", "TrackerGOTURN_create"),
        "legacy": Wrong("cv2.legacy.TrackerMedianFlow_create(): the legacy module is not in the wheel",
                        _tr_wrong("cv2.legacy.TrackerMedianFlow_create()"), "AttributeError", "legacy"),
        "no_tracking": Wrong("returns the first frame's box: the patch has moved", code("frames, box0", '    return {"box": np.array(box0)}\n'),
                             "verifier"),
    }),
    "Track the patch with an OpenCV tracker: initialise on frames[0] with box0 (x, y, w, h), update it on every later "
    "frame, and return its (x, y, w, h) in the last frame.",
    "A bright square moves slowly across a grayscale clip. Initialise a tracker from box0 on the first frame, feed it "
    "the remaining frames, and report where the square is in the final frame as (x, y, w, h).")


# ---------------------------------------------------------------- 2. estimateRigidTransform -> estimateAffinePartial2D

# calibrated on 300 point sets (40 points, 8 outliers, noise 0.3 px, angle +-40 deg, scale 0.7..1.4): estimateAffinePartial2D
# (RANSAC defaults) recovers the angle to 0.12 deg, the scale to 0.26 % and the shift to 0.42 px at worst; 1 deg, 2 % and
# 2 px leave 8x, 8x and 5x. The first three matches alone (outliers included) are off by tens of degrees.
SIM_ANGLE_TOL, SIM_SCALE_REL, SIM_SHIFT_TOL = 1.0, 0.02, 2.0


def similarity_pairs(seed: int, n: int = 40, outliers: int = 8):
    rng = np.random.default_rng(seed)
    theta, s = float(rng.uniform(-40, 40)), float(rng.uniform(0.7, 1.4))
    t = rng.uniform(-30, 30, 2)
    src = rng.uniform(0, 200, (n, 2))
    a = math.radians(theta)
    R_ = s * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = src @ R_.T + t + rng.normal(0, 0.3, (n, 2))
    bad = rng.choice(n, outliers, replace=False)
    dst[bad] = rng.uniform(0, 250, (outliers, 2))
    return src.astype(np.float32), dst.astype(np.float32), (theta, s, t)


def _sf_cases():
    out = []
    for seed in (S + 11, S + 12, S + 13, S + 14):
        src, dst, truth = similarity_pairs(seed)
        out.append(Case({"src": src, "dst": dst}, truth))
    return out


def _sf_verify(case: Case, out: dict) -> Verdict:
    theta, s, t = case.truth
    if abs(out["angle_deg"] - theta) > SIM_ANGLE_TOL:
        return fail(f"angle {out['angle_deg']:.2f} deg, the points were rotated by {theta:.2f}")
    if abs(out["scale"] / s - 1) > SIM_SCALE_REL:
        return fail(f"scale {out['scale']:.3f}, the points were scaled by {s:.3f}")
    shift = arr(out, "shift").astype(float)
    if shift.shape != (2,) or np.abs(shift - t).max() > SIM_SHIFT_TOL:
        return fail(f"shift {shift.round(1).tolist()}, expected {t.round(1).tolist()}")
    return PASS


_SF_REF = code("src, dst", '''\
    M, inliers = cv2.estimateAffinePartial2D(src, dst)
    return {"angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            "scale": float(np.hypot(M[0, 0], M[1, 0])),
            "shift": M[:, 2].copy()}
''')
R.add(Check(
    id="drill_similarity_fit", group=GROUP, probes=("has cv2.estimateRigidTransform",), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2.estimateRigidTransform is not in the 4.14 or 5.0 wheel: estimateAffinePartial2D replaces it (RANSAC, so it "
           "also survives the outliers)",
    spec=spec("Matched points (src[i] <-> dst[i]) were produced by a rotation, a uniform scale and a shift, "
              "dst = s * R(angle) * src + shift, with R(angle) = [[cos, -sin], [sin, cos]] acting on (x, y), and a few of "
              "the matches are wrong. Recover the angle in degrees, the scale and the shift.",
              {"src": IOSpec("float32", "Nx2", 2, "(x, y) points"),
               "dst": IOSpec("float32", "Nx2", 2, "the matched points; about a fifth of the rows are bad matches")},
              {"angle_deg": IOSpec("float", "scalar", None, "rotation in degrees"),
               "scale": IOSpec("float", "scalar", None, "uniform scale factor"),
               "shift": IOSpec("float", "2", 1, "(tx, ty)")}),
    make_cases=_sf_cases, verify=_sf_verify, reference=_SF_REF,
    wrong={
        "estimate_rigid_transform": Wrong("4.x: cv2.estimateRigidTransform(src, dst, fullAffine=False)",
                                          code("src, dst", '    M = cv2.estimateRigidTransform(src, dst, False)\n'
                                                           '    return {"angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),\n'
                                                           '            "scale": float(np.hypot(M[0, 0], M[1, 0])), "shift": M[:, 2].copy()}\n'),
                                          "AttributeError", "estimateRigidTransform"),
        "sign_flipped": Wrong("angle from arctan2(M[0, 1], M[0, 0]): the rotation with the opposite sign",
                              code("src, dst", '    M, inliers = cv2.estimateAffinePartial2D(src, dst)\n'
                                               '    return {"angle_deg": float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))),\n'
                                               '            "scale": float(np.hypot(M[0, 0], M[1, 0])), "shift": M[:, 2].copy()}\n'),
                              "verifier"),
        "first_three_points": Wrong("cv2.getAffineTransform on the first three matches (outliers included)",
                                    code("src, dst", '    M = cv2.getAffineTransform(src[:3], dst[:3])\n'
                                                     '    return {"angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),\n'
                                                     '            "scale": float(np.hypot(M[0, 0], M[1, 0])), "shift": M[:, 2].copy()}\n'),
                                    "verifier"),
    }),
    "Estimate a similarity transform (rotation angle in degrees, uniform scale, translation (tx, ty)) from the matched "
    "points, where dst = s * R(angle) * src + (tx, ty). Some matches are outliers.",
    "Fit rotation, uniform scale and translation between two point sets with a robust estimator. The convention is "
    "dst = s * [[cos a, -sin a], [sin a, cos a]] @ src + t; roughly one match in five is wrong.")


# ---------------------------------------------------------------- 3. cv2.ml.KNearest -> numpy

def knn_truth(train, labels, queries, k):
    d = np.linalg.norm(queries[:, None] - train[None], axis=2)
    nearest = np.argsort(d, axis=1)[:, :k]
    votes = np.apply_along_axis(lambda r: np.bincount(r, minlength=3), 1, labels[nearest])
    return votes.argmax(axis=1), votes, labels[nearest[:, 0]]


def knn_scene(seed: int):
    """Three overlapping Gaussian classes; 12 queries. Redrawn until no query has a tied vote and at least two queries have a
    1-nearest-neighbour label that differs from the k-nearest majority (so that ignoring k is visible)."""
    rng = np.random.default_rng(seed)
    while True:
        centres = np.array([[0.0, 0.0], [5.5, 1.0], [2.5, 5.5]]) + rng.uniform(-0.8, 0.8, (3, 2))
        train, labels = [], []
        for c, n in enumerate((22, 18, 20)):
            train.append(centres[c] + rng.normal(0, 1.6, (n, 2)))
            labels += [c] * n
        train, labels = np.vstack(train).astype(np.float32), np.array(labels, np.int32)
        queries = (centres[rng.integers(0, 3, 12)] + rng.normal(0, 1.6, (12, 2))).astype(np.float32)
        want, votes, first = knn_truth(train, labels, queries, 5)
        srt = np.sort(votes, axis=1)
        if (srt[:, -1] > srt[:, -2]).all() and (want != first).sum() >= 2:
            return train, labels, queries, 5, want


def _kn_cases():
    out = []
    for seed in (S + 21, S + 22, S + 23):
        train, labels, queries, k, want = knn_scene(seed)
        out.append(Case({"train": train, "train_labels": labels, "queries": queries, "k": k}, want))
    return out


def _kn_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "labels")
    if got.shape != case.truth.shape or not np.array_equal(got, case.truth):
        return fail(f"labels {got.tolist()} vs {case.truth.tolist()} (majority of the k nearest training points)")
    return PASS


_KN_SIG = "train, train_labels, queries, k"
R.add(Check(
    id="drill_knn_labels", group=GROUP, probes=("has cv2.ml",), depends_on_5_0=True, role="gotcha",
    gotcha="cv2.ml (KNearest, SVM, ...) is not in the 5.0 wheel: plain numpy (or scikit-learn) does k-nearest-neighbour voting",
    spec=spec("Classify each query point by a majority vote among its k nearest training points (Euclidean distance) and "
              "return the predicted labels.",
              {"train": IOSpec("float32", "Nx2", 2, "training points"),
               "train_labels": IOSpec("int32", "N", 1, "class of each training point, 0..2"),
               "queries": IOSpec("float32", "Mx2", 2, "points to classify"),
               "k": IOSpec("int", "scalar", None, "number of neighbours, odd; the vote never ties")},
              {"labels": IOSpec("int", "M", 1, "predicted class of each query point")}),
    make_cases=_kn_cases, verify=_kn_verify,
    reference=code(_KN_SIG, '''\
    d = np.linalg.norm(queries[:, None, :] - train[None, :, :], axis=2)
    nearest = np.argsort(d, axis=1)[:, :k]
    votes = train_labels[nearest]
    labels = np.array([np.bincount(row).argmax() for row in votes])
    return {"labels": labels}
'''),
    wrong={
        "cv2_ml_knearest": Wrong("4.x: cv2.ml.KNearest_create(), train, findNearest",
                                 code(_KN_SIG, '    knn = cv2.ml.KNearest_create()\n'
                                               '    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)\n'
                                               '    _, results, _, _ = knn.findNearest(queries, k)\n'
                                               '    return {"labels": results.flatten().astype(np.int64)}\n'),
                                 "AttributeError", "no attribute 'ml'"),
        "cv2_ml_svm": Wrong("4.x: a cv2.ml.SVM_create() classifier instead of k nearest neighbours",
                            code(_KN_SIG, '    svm = cv2.ml.SVM_create()\n    svm.setKernel(cv2.ml.SVM_RBF)\n'
                                          '    svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)\n'
                                          '    return {"labels": svm.predict(queries)[1].flatten().astype(np.int64)}\n'),
                            "AttributeError", "no attribute 'ml'"),
        "nearest_only": Wrong("1-nearest neighbour: ignores k", code(_KN_SIG, '''\
    d = np.linalg.norm(queries[:, None, :] - train[None, :, :], axis=2)
    return {"labels": train_labels[d.argmin(axis=1)]}
'''), "verifier"),
    }),
    "k-nearest-neighbour classification: for every query point, find its k nearest training points by Euclidean distance "
    "and return the class that most of them have.",
    "Given labelled 2-D training points and some query points, predict each query's label as the majority class among "
    "its k closest training points. k is odd, so the vote never ties.")


# ---------------------------------------------------------------- 4. BOWKMeansTrainer -> cv2.kmeans

def word_scene(seed: int, k: int):
    rng = np.random.default_rng(seed)
    dim = 8
    while True:
        centres = rng.uniform(0, 12, (k, dim))
        if min(np.linalg.norm(centres[i] - centres[j]) for i in range(k) for j in range(i)) > 9.0:
            break
    sizes = sorted(rng.choice(np.arange(20, 80), size=k, replace=False), reverse=True)
    data = np.vstack([centres[i] + rng.normal(0, 0.5, (n, dim)) for i, n in enumerate(sizes)])
    return rng.permutation(data).astype(np.float32), np.array(sizes, np.int64)


def _vw_cases():
    return [Case({"descriptors": d, "k": k}, sizes) for d, sizes, k in ((*word_scene(s, k), k) for s, k in ((S + 31, 3), (S + 32, 4), (S + 33, 5)))]


def _vw_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "counts")
    if got.shape != case.truth.shape or not np.array_equal(got, case.truth):
        return fail(f"counts {got.tolist()} vs the cluster sizes {case.truth.tolist()}")
    return PASS


_VW_HEAD = code("descriptors, k", "")
R.add(Check(
    id="drill_visual_words", group=GROUP, probes=("has cv2.BOWKMeansTrainer",), depends_on_5_0=True, role="gotcha",
    gotcha="cv2.BOWKMeansTrainer is not in the 5.0 wheel: cv2.kmeans on float32 descriptors (labels are (N, 1)) replaces it",
    spec=spec("Build a visual vocabulary: cluster the local descriptors into k visual words and report how many "
              "descriptors fall into each word, largest word first.",
              {"descriptors": IOSpec("float32", "Nx8", 2, "local feature descriptors from many images; they form k "
                                                          "well separated groups of different sizes"),
               "k": IOSpec("int", "scalar", None, "vocabulary size")},
              {"counts": IOSpec("int", "k", 1, "descriptors per visual word, largest first")}),
    make_cases=_vw_cases, verify=_vw_verify,
    reference=code("descriptors, k", '''\
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, centers = cv2.kmeans(descriptors, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return {"counts": np.sort(np.bincount(labels.flatten(), minlength=k))[::-1]}
'''),
    wrong={
        "bow_trainer": Wrong("4.x: cv2.BOWKMeansTrainer(k).add(descriptors), cluster()",
                             code("descriptors, k", '    trainer = cv2.BOWKMeansTrainer(k)\n    trainer.add(descriptors)\n'
                                                    '    vocabulary = trainer.cluster()\n'
                                                    '    d = np.linalg.norm(descriptors[:, None] - vocabulary[None], axis=2)\n'
                                                    '    return {"counts": np.sort(np.bincount(d.argmin(axis=1), minlength=k))[::-1]}\n'),
                             "AttributeError", "BOWKMeansTrainer"),
        "float64_descriptors": Wrong("cv2.kmeans on float64 data",
                                     code("descriptors, k", '    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)\n'
                                                            '    _, labels, centers = cv2.kmeans(descriptors.astype(np.float64), k, None, criteria, 5,\n'
                                                            '                                    cv2.KMEANS_PP_CENTERS)\n'
                                                            '    return {"counts": np.sort(np.bincount(labels.flatten(), minlength=k))[::-1]}\n'),
                                     "error", "CV_32F"),
        "bincount_of_column": Wrong("np.bincount(labels) on the (N, 1) label column",
                                    code("descriptors, k", '    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)\n'
                                                           '    _, labels, centers = cv2.kmeans(descriptors, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)\n'
                                                           '    return {"counts": np.sort(np.bincount(labels))[::-1]}\n'),
                                    "ValueError", "too deep"),
    }),
    "Cluster the descriptors into k visual words (bag-of-visual-words vocabulary) and give the number of descriptors "
    "assigned to each word, in descending order.",
    "A bag-of-words pipeline needs a codebook. Group the feature descriptors into k clusters and return the cluster "
    "sizes from largest to smallest.")


# ---------------------------------------------------------------- 5. background subtraction: MOG -> MOG2

def conveyor_clip(seed: int, T: int = 24, H: int = 64, W: int = 96, enters: int = 8):
    """A static smooth background; from frame `enters` on a flat rectangle (bright or dark) moves 5-6 px per frame to the
    right. Truth: its mask in the last frame."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3.0)
    bg = 60 + 80 * (bg - bg.min()) / (bg.max() - bg.min())
    w, h = int(rng.integers(14, 19)), int(rng.integers(12, 18))
    y, x0, v = int(rng.integers(8, H - h - 8)), int(rng.integers(0, 6)), int(rng.integers(5, 7))
    level = 235 if seed % 2 == 0 else 15
    frames = []
    for t in range(T):
        f = bg.copy()
        if t >= enters:
            x = x0 + v * (t - enters)
            f[y:y + h, x:x + w] = level
        frames.append(np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255).astype(np.uint8))
    x = x0 + v * (T - 1 - enters)
    truth = np.zeros((H, W), bool)
    truth[y:y + h, x:x + w] = True
    return np.stack(frames), truth


# calibrated on 60 clips x 4 settings (history 50..500, varThreshold 16..25, shadows on or off): MOG2's foreground (value 255)
# overlaps the moving rectangle with IoU 1.00 in all 240 runs at 5-6 px per frame (at 3-4 px per frame the trailing edge is
# partly learned as background and the IoU drops to 0.57, which is why the clip moves fast). A mask of the whole frame
# scores 0.04 and a subtractor fed the last frame only 0.01. 0.6 leaves room for other subtractors and thresholds.
BG_MIN_IOU = 0.6


def _bg_cases():
    return [Case({"frames": f}, t) for f, t in (conveyor_clip(s) for s in (S + 41, S + 42, S + 43))]


def _bg_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "mask")
    if got.shape != case.truth.shape:
        return fail(f"mask has shape {got.shape}")
    pred = got == 255
    iou = (pred & case.truth).sum() / max((pred | case.truth).sum(), 1)
    return PASS if iou >= BG_MIN_IOU else fail(f"the mask (255 = moving) overlaps the moving rectangle with IoU {iou:.2f}")


R.add(Check(
    id="drill_moving_mask", group=GROUP, probes=("has cv2.createBackgroundSubtractorMOG",), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2.createBackgroundSubtractorMOG (and cv2.bgsegm) are not in the wheel: createBackgroundSubtractorMOG2 / KNN are",
    spec=spec("Detect the moving object in the last frame of a fixed-camera clip with a background subtractor fed every "
              "frame in order, and return its foreground mask (255 moving, 0 background).",
              {"frames": IOSpec("uint8", "TxHxW", 3, "T grayscale frames of a static scene; a rectangle enters part-way "
                                                     "through and moves right")},
              {"mask": IOSpec("uint8", "HxW", 2, "255 where the last frame is foreground, 0 elsewhere")}),
    make_cases=_bg_cases, verify=_bg_verify,
    reference=code("frames", '''\
    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
    for frame in frames:
        mask = subtractor.apply(frame)
    return {"mask": mask}
'''),
    wrong={
        "mog": Wrong("4.x contrib: cv2.createBackgroundSubtractorMOG()",
                     code("frames", '    subtractor = cv2.createBackgroundSubtractorMOG()\n    for frame in frames:\n'
                                    '        mask = subtractor.apply(frame)\n    return {"mask": mask}\n'),
                     "AttributeError", "createBackgroundSubtractorMOG"),
        "bgsegm_mog": Wrong("4.x contrib: cv2.bgsegm.createBackgroundSubtractorMOG()",
                            code("frames", '    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()\n    for frame in frames:\n'
                                           '        mask = subtractor.apply(frame)\n    return {"mask": mask}\n'),
                            "AttributeError", "bgsegm"),
        "last_frame_only": Wrong("applies the subtractor to the last frame only: a fresh model calls everything foreground",
                                 code("frames", '    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)\n'
                                                '    return {"mask": subtractor.apply(frames[-1])}\n'), "verifier"),
    }),
    "Background subtraction: feed the frames one after another to an OpenCV background subtractor and return the "
    "foreground mask it gives for the last frame, 255 on the moving object and 0 elsewhere.",
    "The camera is fixed and an object slides through the scene. Learn the background from the clip with a "
    "Gaussian-mixture background subtractor and return the moving object's mask (255/0) for the final frame.")


# ---------------------------------------------------------------- 6. CV_AA -> LINE_AA

def _aa_truth(img, p, q, radius):
    out = img.copy()
    cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, tuple(int(v) for v in q), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def _aa_cases():
    out = []
    for seed, radius in ((S + 51, 7), (S + 52, 11), (S + 53, 5)):
        rng = np.random.default_rng(seed)
        img = np.clip(np.linspace(0, 70, 90)[None, :, None] + rng.integers(0, 12, (60, 90, 3)), 0, 255).astype(np.uint8)
        p, q = rng.integers(10, 80, 2), rng.integers(12, 78, 2)
        p, q = np.array([p[0], rng.integers(8, 52)], np.int32), np.array([q[0], rng.integers(8, 52)], np.int32)
        if abs(p[0] - q[0]) < 15:
            q[0] = min(q[0] + 20, 82)
        out.append(Case({"img": img, "p": p, "q": q, "radius": radius}, _aa_truth(img, p, q, radius)))
    return out


def _aa_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "img")
    if got.shape != case.truth.shape:
        return fail(f"img has shape {got.shape}")
    diff = int((got != case.truth).sum())
    return fail(f"{diff} values differ from the anti-aliased drawing (an aliased line has no intermediate grays)") if diff else PASS


_AA_SIG = "img, p, q, radius"
R.add(Check(
    id="drill_aa_overlay", group=GROUP, probes=("has cv2.CV_AA",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.CV_AA is gone since OpenCV 4.0 (not in 4.14 or 5.0): the anti-aliasing flag is cv2.LINE_AA",
    spec=spec("Mark a path on the image: draw a white, 2 px wide, anti-aliased line from p to q, then a filled white "
              "anti-aliased circle of the given radius centred on q. Return the image.",
              {"img": IOSpec("uint8", "HxWx3", 3, "BGR image"),
               "p": IOSpec("int", "2", 1, "(x, y) start of the line"),
               "q": IOSpec("int", "2", 1, "(x, y) end of the line and centre of the circle"),
               "radius": IOSpec("int", "scalar", None, "circle radius in px")},
              {"img": IOSpec("uint8", "HxWx3", 3, "the image with the line and the circle drawn on a copy")}),
    make_cases=_aa_cases, verify=_aa_verify,
    reference=code(_AA_SIG, '''\
    out = img.copy()
    cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, tuple(int(v) for v in q), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return {"img": out}
'''),
    wrong={
        "cv_aa": Wrong("OpenCV 2/3 spelling: cv2.CV_AA",
                       code(_AA_SIG, '    out = img.copy()\n'
                                     '    cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 2, cv2.CV_AA)\n'
                                     '    cv2.circle(out, tuple(int(v) for v in q), radius, (255, 255, 255), -1, cv2.CV_AA)\n'
                                     '    return {"img": out}\n'), "AttributeError", "CV_AA"),
        "aliased": Wrong("default lineType (LINE_8): jagged edges",
                         code(_AA_SIG, '    out = img.copy()\n'
                                       '    cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 2)\n'
                                       '    cv2.circle(out, tuple(int(v) for v in q), radius, (255, 255, 255), -1)\n'
                                       '    return {"img": out}\n'), "verifier"),
        "line_only": Wrong("anti-aliased line but no end circle",
                           code(_AA_SIG, '    out = img.copy()\n'
                                         '    cv2.line(out, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 2, cv2.LINE_AA)\n'
                                         '    return {"img": out}\n'), "verifier"),
    }),
    "On a copy of the image, draw a smooth (anti-aliased) white line two pixels thick from p to q and a smooth filled "
    "white circle of the given radius at q.",
    "Overlay a route marker: an anti-aliased white 2 px line from point p to point q, ending in a filled anti-aliased "
    "white disc of the given radius centred on q. Do not modify the input image.")


# ---------------------------------------------------------------- 7. AKAZE / BRISK / KAZE -> ORB / SIFT (rotation)

def feature_texture(seed: int, shape=(240, 320)) -> np.ndarray:
    """Rich in corners and blobs at several scales (blurred noise + random rectangles), so ORB and SIFT both find plenty."""
    rng = np.random.default_rng(seed)
    f = sum(cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), s) * w for s, w in ((1.2, 1.0), (3.0, 1.6), (7.0, 2.4)))
    f = (f - f.min()) / (f.max() - f.min())
    img = np.round(255 * f).astype(np.uint8)
    for _ in range(25):
        x, y = int(rng.integers(0, shape[1] - 30)), int(rng.integers(0, shape[0] - 30))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 25)), y + int(rng.integers(5, 25))), int(rng.integers(0, 256)), -1)
    return img


# calibrated on 30 pairs (rotation +-40 deg): ORB(1500) + Hamming cross-check + estimateAffinePartial2D recovers the angle to
# 0.07 deg at worst and SIFT to 0.003 deg; scale 0.6..1.7: ORB to 0.4 % and SIFT to 0.013 %. 1.0 deg and 3 % keep a wide margin.
ROT_TOL_DEG, SCALE_REL_TOL = 1.0, 0.03


def _rf_cases():
    out = []
    for seed, theta in ((S + 61, 28.0), (S + 62, -17.5), (S + 63, 36.0), (S + 64, -33.0)):
        a = feature_texture(seed)
        h, w = a.shape
        M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), theta, 1.0)
        out.append(Case({"img_a": a, "img_b": cv2.warpAffine(a, M, (w, h))}, theta))
    return out


def _rf_verify(case: Case, out: dict) -> Verdict:
    got = out["angle_deg"]
    return PASS if abs(got - case.truth) <= ROT_TOL_DEG else fail(f"angle {got:.2f} deg, img_b is img_a rotated by {case.truth:.2f}")


_FEAT_BODY = '''\
    ka, da = {det}.detectAndCompute(img_a, None)
    kb, db = {det}.detectAndCompute(img_b, None)
    matches = cv2.BFMatcher({norm}, crossCheck=True).match(da, db)
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
'''


def _feat(det: str, norm: str, tail: str) -> str:
    return code("img_a, img_b", _FEAT_BODY.format(det=det, norm=norm) + tail)


_ROT_TAIL = '    return {"angle_deg": float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))}\n'
R.add(Check(
    id="drill_rotation_from_features", group=GROUP,
    probes=("has cv2.AKAZE_create", "has cv2.KAZE_create", "has cv2.BRISK_create"), depends_on_5_0=True, role="gotcha",
    gotcha="AKAZE, KAZE and BRISK are not in the 5.0 wheel: ORB (binary descriptors, NORM_HAMMING) and SIFT remain",
    spec=spec("img_b is img_a rotated counter-clockwise by an unknown angle about the image centre (the convention of "
              "cv2.getRotationMatrix2D with a positive angle). Match local features between the two images and report "
              "the rotation angle in degrees.",
              {"img_a": IOSpec("uint8", "HxW", 2, "grayscale textured image"),
               "img_b": IOSpec("uint8", "HxW", 2, "img_a rotated about its centre, black where nothing maps")},
              {"angle_deg": IOSpec("float", "scalar", None, "counter-clockwise rotation from img_a to img_b in degrees")}),
    make_cases=_rf_cases, verify=_rf_verify, reference=_feat("cv2.ORB_create(1500)", "cv2.NORM_HAMMING", _ROT_TAIL),
    wrong={
        "akaze": Wrong("4.x favourite for binary descriptors: cv2.AKAZE_create()", _feat("cv2.AKAZE_create()", "cv2.NORM_HAMMING", _ROT_TAIL),
                       "AttributeError", "AKAZE_create"),
        "brisk": Wrong("cv2.BRISK_create()", _feat("cv2.BRISK_create()", "cv2.NORM_HAMMING", _ROT_TAIL), "AttributeError", "BRISK_create"),
        "kaze": Wrong("cv2.KAZE_create()", _feat("cv2.KAZE_create()", "cv2.NORM_L2", _ROT_TAIL), "AttributeError", "KAZE_create"),
        "sign_flipped": Wrong("angle from arctan2(M[1, 0], M[0, 0]): the rotation with the opposite sign",
                              _feat("cv2.ORB_create(1500)", "cv2.NORM_HAMMING",
                                    '    return {"angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))}\n'), "verifier"),
    }),
    "Find the in-plane rotation between two views of the same textured image: detect and match local features, fit a "
    "similarity transform and report the angle in degrees (counter-clockwise positive, as in cv2.getRotationMatrix2D).",
    "img_b was made by turning img_a counter-clockwise about its centre. Estimate by how many degrees, using matched "
    "keypoints between the two images; positive means counter-clockwise like getRotationMatrix2D.")


# ---------------------------------------------------------------- 8. SURF / KAZE -> SIFT (scale)

def _sc_cases():
    out = []
    for seed, s in ((S + 71, 0.62), (S + 72, 0.78), (S + 73, 1.45), (S + 74, 1.72)):
        a = feature_texture(seed)
        b = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        out.append(Case({"img_a": a, "img_b": b}, s))
    return out


def _sc_verify(case: Case, out: dict) -> Verdict:
    got = out["scale"]
    return PASS if abs(got / case.truth - 1) <= SCALE_REL_TOL else fail(f"scale {got:.3f}, img_b is img_a resized by {case.truth:.3f}")


_SC_TAIL = '    return {"scale": float(np.hypot(M[0, 0], M[1, 0]))}\n'
R.add(Check(
    id="drill_scale_from_features", group=GROUP,
    probes=("has cv2.KAZE_create", "has cv2.xfeatures2d"), depends_on_5_0=True, role="gotcha",
    gotcha="SURF (cv2.xfeatures2d, contrib) and KAZE are not in the 5.0 wheel: SIFT is the scale-invariant detector left, "
           "and ORB's pyramid also copes with moderate scale changes",
    spec=spec("img_b is img_a resized by an unknown factor (both axes). Match scale-invariant local features between the "
              "two images and report that factor, the size of img_b relative to img_a.",
              {"img_a": IOSpec("uint8", "HxW", 2, "grayscale textured image"),
               "img_b": IOSpec("uint8", "H'xW'", 2, "img_a resized by one factor between 0.6 and 1.8")},
              {"scale": IOSpec("float", "scalar", None, "size of img_b divided by size of img_a")}),
    make_cases=_sc_cases, verify=_sc_verify, reference=_feat("cv2.SIFT_create()", "cv2.NORM_L2", _SC_TAIL),
    wrong={
        "surf": Wrong("4.x contrib: cv2.xfeatures2d.SURF_create(400)", _feat("cv2.xfeatures2d.SURF_create(400)", "cv2.NORM_L2", _SC_TAIL),
                      "AttributeError", "xfeatures2d"),
        "kaze": Wrong("cv2.KAZE_create()", _feat("cv2.KAZE_create()", "cv2.NORM_L2", _SC_TAIL), "AttributeError", "KAZE_create"),
        "transform_not_similarity": Wrong("takes the determinant of the 2x2 block: the area scale, not the length scale",
                                          _feat("cv2.SIFT_create()", "cv2.NORM_L2",
                                                '    return {"scale": float(abs(np.linalg.det(M[:, :2])))}\n'), "verifier"),
    }),
    "Estimate how much larger or smaller img_b is than img_a (a uniform resize) from matched scale-invariant keypoints, "
    "and return that factor.",
    "Two images show the same scene at different sizes. Using matched local features, determine the scale factor from "
    "img_a to img_b.")


CHECKS = R.checks
PHRASINGS = R.phrasings
