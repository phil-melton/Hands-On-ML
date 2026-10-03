#!/usr/bin/env python
"""Differential API probe: run the same gauge blocks through any OpenCV build.

Run once per environment (one venv per OpenCV version), then diff the two files:

    python probes/api_diff.py              # writes probes/results/probe_<cv2 version>.json
    python probes/diff_report.py probes/results/probe_4.14.0.json \
        probes/results/probe_5.0.0.json --out probes/results/diff_4.14_vs_5.0.md

Each probe is a tiny deterministic experiment on a synthetic image. It records
*what came back* (types, shapes, dtypes, small values) or *what was raised*.
The diff between two versions is the curriculum for the fine-tune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROBES: list[dict] = []
ARRAYS: dict[str, np.ndarray] = {}  # full outputs of fingerprinted probes, for cross-version |Δ|
_CURRENT = {"name": ""}


def probe(category: str, idiom: str = ""):
    """Register a probe. `idiom` names the user code that breaks if the result changes."""
    def deco(fn):
        PROBES.append({"name": fn.__name__, "category": category, "idiom": idiom, "fn": fn})
        return fn
    return deco


# ---------------------------------------------------------------- describers

def arr(a) -> str | None:
    """Compact array signature, e.g. 'int32[52, 1, 2]'."""
    if a is None:
        return None
    a = np.asarray(a)
    return f"{a.dtype}{list(a.shape)}"


def describe(x, depth: int = 0):
    """JSON-able summary of a return value: structure, shapes, dtypes, small values."""
    if isinstance(x, np.ndarray):
        if 0 < x.size <= 16:
            vals = np.round(x.astype(np.float64), 3).tolist() if x.dtype.kind == "f" else x.tolist()
            return {"array": arr(x), "values": vals}
        return arr(x)
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return round(x, 3)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, (tuple, list)):
        if depth >= 3:
            return f"{type(x).__name__}[{len(x)}]"
        return {type(x).__name__: [describe(v, depth + 1) for v in x[:4]], "len": len(x)}
    return f"<{type(x).__name__}>"


def fingerprint(a: np.ndarray) -> dict:
    """Pixel-exact identity of a larger array (detects interpolation/rendering changes).

    The full array is kept too, so diff_report.py can measure *how much* it changed.
    """
    a = np.ascontiguousarray(a)
    ARRAYS[_CURRENT["name"]] = a
    return {"array": arr(a), "sha1": hashlib.sha1(a.tobytes()).hexdigest()[:12],
            "sum": round(float(a.sum()), 3)}


def short_error(e: BaseException) -> str:
    """Version-independent error text: drop OpenCV version, file:line and function names."""
    s = str(e)
    s = re.sub(r"OpenCV\(\d[\w.\-]*\)\s*", "", s)
    s = re.sub(r"\S+\.(?:cpp|hpp|h|cc):\d+:\s*", "", s)
    s = re.sub(r":-1:\s*", "", s)
    s = re.sub(r"in function '[^']*'", "", s)
    s = re.sub(r"\s*\n>?\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^error:\s*", "", s)
    return f"{type(e).__name__}: {s[:220]}"


# ---------------------------------------------------------------- fixtures

def blobs() -> np.ndarray:
    """Binary uint8 scene: a rectangle with a hole, a disk and a triangle."""
    img = np.zeros((120, 160), np.uint8)
    cv2.rectangle(img, (10, 10), (60, 70), 255, -1)
    cv2.rectangle(img, (25, 25), (45, 50), 0, -1)
    cv2.circle(img, (110, 40), 20, 255, -1)
    cv2.fillPoly(img, [np.array([[80, 110], [150, 110], [115, 75]], np.int32)], 255)
    return img


def largest_contour() -> np.ndarray:
    cs = cv2.findContours(blobs(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return max(cs, key=cv2.contourArea)


def concave_contour() -> np.ndarray:
    img = np.zeros((100, 100), np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), 255, -1)
    cv2.rectangle(img, (40, 0), (60, 60), 0, -1)  # notch from the top: a U shape
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2][0]


def lines_img() -> np.ndarray:
    img = np.zeros((120, 120), np.uint8)
    cv2.line(img, (10, 10), (110, 90), 255, 2)
    cv2.line(img, (10, 110), (110, 20), 255, 2)
    return img


def circles_img() -> np.ndarray:
    img = np.zeros((120, 120), np.uint8)
    cv2.circle(img, (40, 40), 18, 255, 2)
    cv2.circle(img, (85, 80), 22, 255, 2)
    return cv2.GaussianBlur(img, (5, 5), 1.5)


def texture(seed: int = 0, shape=(160, 200)) -> np.ndarray:
    """Deterministic gray image full of corners (for features, flow, matching)."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 30, np.uint8)
    for _ in range(25):
        x, y = rng.integers(10, shape[1] - 30), rng.integers(10, shape[0] - 30)
        w, h = rng.integers(8, 25, size=2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(80, 255)), -1)
    return img


def chessboard(pattern=(6, 4), sq=20, margin=20) -> np.ndarray:
    cols, rows = pattern[0] + 1, pattern[1] + 1
    img = np.full((rows * sq + 2 * margin, cols * sq + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * sq, margin + c * sq
                img[y:y + sq, x:x + sq] = 0
    return img


K = np.array([[200.0, 0, 80], [0, 200.0, 60], [0, 0, 1]])
DIST = np.array([0.1, -0.05, 0, 0, 0])
QUAD_SRC = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
QUAD_DST = np.float32([[5, 3], [96, 8], [102, 85], [2, 78]])


# ---------------------------------------------------------------- A. existence

EXISTENCE = {
    "CascadeClassifier": "Haar face detection: cv2.CascadeClassifier(cv2.data.haarcascades + ...)",
    "HOGDescriptor": "HOG people detector: cv2.HOGDescriptor_getDefaultPeopleDetector()",
    "ml": "cv2.ml.KNearest_create / SVM_create / RTrees_create",
    "gapi": "G-API graphs",
    "dnn.readNetFromCaffe": "Caffe .prototxt/.caffemodel loading",
    "dnn.readNetFromDarknet": "YOLO .cfg/.weights loading",
    "dnn.readNetFromTensorflow": "TensorFlow .pb loading",
    "dnn.readNetFromONNX": "ONNX loading (the 5.0 path)",
    "dnn.readNet": "generic dnn loader",
    "FaceDetectorYN": "YuNet face detector (the 5.0 replacement for Haar)",
    "FaceDetectorYN.create": "",
    "FaceDetectorYN_create": "",
    "FaceRecognizerSF": "",
    "FontFace": "new text API in 5.0",
    "SIFT_create": "",
    "SIFT.create": "",
    "ORB_create": "",
    "AKAZE_create": "AKAZE keypoints + binary descriptors",
    "KAZE_create": "",
    "BRISK_create": "",
    "AgastFeatureDetector_create": "",
    "FastFeatureDetector_create": "",
    "GFTTDetector_create": "",
    "MSER_create": "",
    "SimpleBlobDetector_create": "blob counting with SimpleBlobDetector_Params",
    "FlannBasedMatcher": "",
    "BFMatcher_create": "",
    "drawKeypoints": "",
    "drawMatchesKnn": "",
    "BOWKMeansTrainer": "bag-of-visual-words",
    "xfeatures2d": "contrib-only (SURF etc.)",
    "SURF_create": "SURF (patented, contrib-only since 3.x)",
    "cv": "OpenCV 2 API: cv2.cv.CV_*",
    "CV_AA": "OpenCV 2 constant (now cv2.LINE_AA)",
    "LINE_AA": "",
    "VideoWriter_fourcc": "",
    "samples.findFile": "tutorial data helper",
    "aruco": "",
    "QRCodeDetector": "",
    "aruco.ArucoDetector": "",
    "aruco.generateImageMarker": "",
    "QRCodeEncoder": "",
    "dnn.NMSBoxes": "",
    "dnn.DetectionModel": "high-level dnn model API",
    "TrackerMIL_create": "",
    "TrackerGOTURN_create": "GOTURN tracker",
    "TrackerNano_create": "",
    "TrackerVit_create": "",
    "TrackerDaSiamRPN_create": "",
    "TrackerKCF_create": "",
    "TrackerCSRT_create": "",
    "legacy": "contrib legacy trackers",
    "createBackgroundSubtractorMOG2": "",
    "createBackgroundSubtractorKNN": "",
    "createBackgroundSubtractorMOG": "contrib bgsegm",
    "findEssentialMat": "",
    "recoverPose": "",
    "stereoRectify": "",
    "triangulatePoints": "",
    "initUndistortRectifyMap": "",
    "fisheye": "",
    "estimateRigidTransform": "removed in 4.0 (use estimateAffinePartial2D)",
    "createCLAHE": "",
    "Stitcher_create": "",
    "StereoBM_create": "",
    "StereoSGBM_create": "",
    "solvePnP": "",
    "calibrateCamera": "",
    "findChessboardCorners": "",
    "findChessboardCornersSB": "",
    "undistort": "",
    "inpaint": "",
    "fastNlMeansDenoising": "",
    "seamlessClone": "",
    "createLineSegmentDetector": "",
    "ximgproc": "contrib-only",
    "INTER_NEAREST_EXACT": "Pillow-style nearest (4.x name)",
    "CV_16BF": "new dtype", "CV_Bool": "new dtype", "CV_32U": "new dtype",
    "CV_64U": "new dtype", "CV_64S": "new dtype", "CV_16F": "",
    "Mat": "",
    "data.haarcascades": "path to bundled Haar XML files",
}


def _has(path: str) -> bool:
    obj = cv2
    for part in path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


for _path, _idiom in EXISTENCE.items():
    PROBES.append({"name": f"has cv2.{_path}", "category": "existence", "idiom": _idiom,
                   "fn": (lambda p=_path: _has(p))})


@probe("existence", "cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'")
def haarcascade_xml_shipped():
    base = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if base is None:
        return {"haarcascades_dir": None}
    return {"xml_present": Path(base, "haarcascade_frontalface_default.xml").exists()}


# ---------------------------------------------------------------- B. return shapes

@probe("shapes", "contours, hierarchy = cv2.findContours(...)  (3-value unpack is OpenCV 3)")
def findContours_tree():
    r = cv2.findContours(blobs(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cs, h = r[-2], r[-1]
    return {"n_returns": len(r), "container": type(cs).__name__, "n": len(cs),
            "contour0": arr(cs[0]), "hierarchy": arr(h)}


@probe("shapes", "cnt[:, 0, :] to get (N, 2) points")
def findContours_approx_none():
    cs = cv2.findContours(blobs(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    return {"n": len(cs), "contour0": arr(cs[0])}


@probe("shapes", "len(approx) == 4 to detect quads")
def approxPolyDP():
    return arr(cv2.approxPolyDP(largest_contour(), 3.0, True))


@probe("shapes")
def convexHull_points():
    return arr(cv2.convexHull(largest_contour()))


@probe("shapes")
def convexHull_indices():
    return arr(cv2.convexHull(concave_contour(), returnPoints=False))


@probe("shapes", "for s, e, f, d in defects[:, 0]: ...")
def convexityDefects():
    c = concave_contour()
    return arr(cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False)))


@probe("shapes", "for line in lines: x1, y1, x2, y2 = line[0]")
def HoughLinesP():
    return arr(cv2.HoughLinesP(lines_img(), 1, np.pi / 180, 30, minLineLength=30, maxLineGap=5))


@probe("shapes", "for line in lines: rho, theta = line[0]")
def HoughLines():
    return arr(cv2.HoughLines(lines_img(), 1, np.pi / 180, 60))


@probe("shapes", "for x, y, r in circles[0]: ...")
def HoughCircles():
    return arr(cv2.HoughCircles(circles_img(), cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=100, param2=15, minRadius=10, maxRadius=40))


@probe("shapes", "for c in corners: x, y = c.ravel()  /  c[0]")
def goodFeaturesToTrack():
    return arr(cv2.goodFeaturesToTrack(texture(), 20, 0.01, 5))


@probe("shapes")
def cornerSubPix():
    g = texture()
    c = cv2.goodFeaturesToTrack(g, 20, 0.01, 5)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return arr(cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit))


@probe("shapes", "pts = cv2.findNonZero(mask); xs = pts[:, 0, 0]")
def findNonZero():
    return arr(cv2.findNonZero(blobs()))


@probe("shapes", "vx, vy, x0, y0 = line  (vs line.ravel())")
def fitLine():
    pts = np.float32([[0, 0], [10, 10.5], [20, 19.5], [30, 30.2]])
    return arr(cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01))


@probe("shapes", "(cx, cy), (w, h), angle = cv2.minAreaRect(cnt)")
def minAreaRect():
    return describe(cv2.minAreaRect(largest_contour()))


@probe("shapes")
def boxPoints():
    return arr(cv2.boxPoints(cv2.minAreaRect(largest_contour())))


@probe("shapes")
def minEnclosingCircle():
    return describe(cv2.minEnclosingCircle(largest_contour()))


@probe("shapes")
def fitEllipse():
    return describe(cv2.fitEllipse(largest_contour()))


@probe("shapes")
def boundingRect():
    return describe(cv2.boundingRect(largest_contour()))


@probe("shapes", "n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)")
def connectedComponentsWithStats():
    n, labels, stats, cents = cv2.connectedComponentsWithStats(blobs(), connectivity=8)
    return {"n": n, "labels": arr(labels), "stats": arr(stats), "centroids": arr(cents)}


@probe("shapes")
def connectedComponents():
    n, labels = cv2.connectedComponents(blobs())
    return {"n": n, "labels": arr(labels)}


@probe("shapes", "m['m10'] / m['m00']")
def moments():
    m = cv2.moments(largest_contour())
    return {"type": type(m).__name__, "n_keys": len(m), "m00": round(m["m00"], 1)}


@probe("shapes")
def HuMoments():
    return arr(cv2.HuMoments(cv2.moments(largest_contour())))


@probe("shapes", "plt.plot(hist) / hist.ravel()")
def calcHist():
    return arr(cv2.calcHist([blobs()], [0], None, [256], [0, 256]))


@probe("shapes", "labels.ravel() == k  (vs labels[:, 0])")
def kmeans():
    cv2.setRNGSeed(0)
    rng = np.random.default_rng(0)
    data = np.float32(np.r_[rng.normal(0, 1, (20, 2)), rng.normal(5, 1, (20, 2))])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    compact, labels, centers = cv2.kmeans(data, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return {"compactness": type(compact).__name__, "labels": arr(labels), "centers": arr(centers)}


@probe("shapes", "good_new = p1[st == 1]")
def calcOpticalFlowPyrLK():
    a = texture(0)
    b = np.roll(a, (2, 3), axis=(0, 1))
    p0 = cv2.goodFeaturesToTrack(a, 20, 0.01, 5)
    p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
    return {"p0": arr(p0), "p1": arr(p1), "status": arr(st), "err": arr(err)}


@probe("shapes")
def calcOpticalFlowFarneback():
    a = texture(0)
    b = np.roll(a, 2, axis=1)
    return arr(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0))


@probe("shapes", "matchesMask = mask.ravel().tolist()")
def findHomography():
    src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50], [20, 70]]).reshape(-1, 1, 2)
    h_true = np.array([[1.1, 0.05, 5], [0.02, 0.95, -3], [1e-4, 2e-4, 1]])
    dst = cv2.perspectiveTransform(src, h_true)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return {"H": arr(H), "mask": arr(mask), "perspectiveTransform(Nx1x2)": arr(dst)}


@probe("shapes")
def estimateAffine2D():
    src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]])
    dst = src @ np.float32([[0.9, 0.1], [-0.1, 0.9]]).T + np.float32([5, 3])
    M, inliers = cv2.estimateAffine2D(src, dst)
    return {"M": arr(M), "inliers": arr(inliers)}


@probe("shapes")
def getPerspectiveTransform_float32():
    return arr(cv2.getPerspectiveTransform(QUAD_SRC, QUAD_DST))


@probe("shapes")
def getRotationMatrix2D():
    return arr(cv2.getRotationMatrix2D((50.0, 40.0), 30, 1.0))


@probe("shapes")
def getAffineTransform():
    return arr(cv2.getAffineTransform(QUAD_SRC[:3], QUAD_DST[:3]))


@probe("shapes", "ret, corners = cv2.findChessboardCorners(gray, (nx, ny))")
def findChessboardCorners():
    ok, corners = cv2.findChessboardCorners(chessboard(), (6, 4))
    return {"found": bool(ok), "corners": arr(corners)}


@probe("shapes")
def undistortPoints():
    pts = np.float32([[10, 10], [50, 60], [100, 90]]).reshape(-1, 1, 2)
    return arr(cv2.undistortPoints(pts, K, DIST))


@probe("shapes", "imgpts, _ = cv2.projectPoints(...); tuple(imgpts[i].ravel())")
def projectPoints():
    obj = np.float32([[0, 0, 5], [1, 0, 5], [0, 1, 5]])
    pts, jac = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, DIST)
    return {"points": arr(pts), "jacobian": arr(jac)}


@probe("shapes")
def Rodrigues():
    R, J = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))
    rvec, _ = cv2.Rodrigues(R)
    return {"R": arr(R), "J": arr(J), "rvec_back": arr(rvec)}


@probe("shapes")
def solvePnP():
    obj = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.2], [0.2, 0.8, 0.1]])
    img_pts, _ = cv2.projectPoints(obj, np.array([0.1, -0.2, 0.05]), np.array([0.1, 0.2, 4.0]), K, None)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None)
    return {"ok": bool(ok), "rvec": arr(rvec), "tvec": arr(tvec)}


@probe("shapes")
def ORB_detectAndCompute():
    kps, des = cv2.ORB_create(nfeatures=100).detectAndCompute(texture(), None)
    return {"keypoints": f"{type(kps).__name__}[{len(kps)}]", "descriptors": arr(des)}


@probe("shapes")
def SIFT_detectAndCompute():
    kps, des = cv2.SIFT_create().detectAndCompute(texture(), None)
    return {"keypoints": f"{type(kps).__name__}[{len(kps)}]", "descriptors": arr(des)}


@probe("shapes", "for m, n in matches: if m.distance < 0.75 * n.distance")
def BFMatcher_knnMatch():
    orb = cv2.ORB_create(200)
    a = texture(0)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(np.roll(a, 3, axis=1), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    return {"outer": type(m).__name__, "inner": type(m[0]).__name__,
            "inner_len": len(m[0]), "item": type(m[0][0]).__name__}


@probe("shapes", "min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)")
def minMaxLoc():
    return describe(cv2.minMaxLoc(texture()))


@probe("shapes")
def mean():
    return describe(cv2.mean(texture()))


@probe("shapes", "b, g, r = cv2.split(img)")
def split():
    return describe(cv2.split(np.zeros((4, 4, 3), np.uint8)))


@probe("shapes", "ok, buf = cv2.imencode('.png', img); buf.tobytes()")
def imencode():
    ok, buf = cv2.imencode(".png", texture())
    return {"ok": bool(ok), "buf": arr(buf)}


@probe("shapes")
def matchTemplate():
    t = texture()
    return arr(cv2.matchTemplate(t, t[20:50, 30:70], cv2.TM_CCOEFF_NORMED))


@probe("shapes")
def distanceTransform():
    return arr(cv2.distanceTransform(blobs(), cv2.DIST_L2, 5))


@probe("shapes")
def dnn_blobFromImage():
    return arr(cv2.dnn.blobFromImage(np.zeros((64, 64, 3), np.uint8), 1 / 255, (32, 32)))


@probe("shapes", "retval, points = cv2.QRCodeDetector().detect(img)")
def QRCodeDetector_detect():
    return describe(cv2.QRCodeDetector().detect(np.zeros((60, 60), np.uint8)))


@probe("shapes", "(w, h), baseline = cv2.getTextSize(...)")
def getTextSize():
    return describe(cv2.getTextSize("OpenCV 5", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2))


def dot_grid(pattern=(4, 3), step=30, r=6, margin=30) -> np.ndarray:
    img = np.full((margin * 2 + step * (pattern[1] - 1), margin * 2 + step * (pattern[0] - 1)), 255, np.uint8)
    for j in range(pattern[1]):
        for i in range(pattern[0]):
            cv2.circle(img, (margin + i * step, margin + j * step), r, 0, -1)
    return img


@probe("shapes", "ret, centers = cv2.findCirclesGrid(...)")
def findCirclesGrid():
    ok, centers = cv2.findCirclesGrid(dot_grid(), (4, 3), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    return {"found": bool(ok), "centers": arr(centers)}


@probe("shapes")
def findChessboardCornersSB():
    ok, corners = cv2.findChessboardCornersSB(chessboard(), (6, 4))
    return {"found": bool(ok), "corners": arr(corners)}


@probe("shapes", "corners, ids, _ = detector.detectMarkers(img); ids.flatten()")
def aruco_detectMarkers():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, 7, 100), 50, 50, 50, 50,
                             cv2.BORDER_CONSTANT, value=255)
    corners, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
    return {"corners": describe(corners), "ids": arr(ids)}


@probe("shapes", "data, points, _ = cv2.QRCodeDetector().detectAndDecode(img)")
def QRCode_detectAndDecode():
    qr = cv2.QRCodeEncoder.create().encode("opencv5")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    data, pts, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return {"data": data, "points": arr(pts)}


@probe("shapes", "for i in indices: box = boxes[i]  (old: i = i[0])")
def dnn_NMSBoxes():
    boxes = [[10, 10, 50, 50], [12, 12, 50, 50], [100, 100, 30, 30]]
    return arr(cv2.dnn.NMSBoxes(boxes, [0.9, 0.8, 0.7], 0.5, 0.4))


@probe("shapes")
def KeyPoint_convert():
    kps = cv2.ORB_create(50).detect(texture(), None)
    return arr(cv2.KeyPoint_convert(kps))


@probe("shapes", "regions, bboxes = mser.detectRegions(gray)")
def MSER_detectRegions():
    regions, bboxes = cv2.MSER_create().detectRegions(texture())
    return {"regions": f"{type(regions).__name__}[{len(regions)}]",
            "region0": arr(regions[0]) if len(regions) else None, "bboxes": arr(bboxes)}


def text_img() -> np.ndarray:
    img = np.full((120, 300), 230, np.uint8)
    cv2.putText(img, "MSER 123", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 5)
    return img


@probe("pixels", "MSER text/blob detection with default parameters")
def MSER_text_default_params():
    return {"regions": len(cv2.MSER_create().detectRegions(text_img())[0])}


@probe("pixels")
def MSER_text_min_diversity_0():
    m = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25, min_diversity=0.0)
    return {"regions": len(m.detectRegions(text_img())[0])}


@probe("shapes", "mean, std = cv2.meanStdDev(img); mean[0][0]")
def meanStdDev():
    m, s = cv2.meanStdDev(texture())
    return {"mean": arr(m), "std": arr(s)}


@probe("shapes", "col_sums = cv2.reduce(img, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]")
def reduce_rows():
    return arr(cv2.reduce(texture(), 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S))


@probe("shapes", "w, u, vt = cv2.SVDecomp(A)")
def SVDecomp():
    w, u, vt = cv2.SVDecomp(np.arange(12, dtype=np.float64).reshape(4, 3) + np.eye(4, 3))
    return {"w": arr(w), "u": arr(u), "vt": arr(vt)}


@probe("shapes")
def eigen():
    ok, vals, vecs = cv2.eigen(np.array([[2.0, 1.0], [1.0, 3.0]]))
    return {"ok": bool(ok), "vals": arr(vals), "vecs": arr(vecs)}


@probe("shapes", "mean, eigvecs, eigvals = cv2.PCACompute2(data, None)")
def PCACompute2():
    data = np.float32(np.random.default_rng(0).normal(size=(30, 3)))
    mean_, vecs, vals = cv2.PCACompute2(data, None)
    return {"mean": arr(mean_), "eigenvectors": arr(vecs), "eigenvalues": arr(vals)}


@probe("shapes")
def minEnclosingTriangle():
    area, tri = cv2.minEnclosingTriangle(largest_contour())
    return {"area": type(area).__name__, "triangle": arr(tri)}


@probe("shapes")
def HoughLinesWithAccumulator():
    return arr(cv2.HoughLinesWithAccumulator(lines_img(), 1, np.pi / 180, 60))


@probe("shapes")
def calcHist_2d():
    hsv = cv2.cvtColor(cv2.merge([texture(0), texture(1), texture(2)]), cv2.COLOR_BGR2HSV)
    return arr(cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]))


@probe("shapes")
def findEssentialMat_recoverPose():
    obj = np.random.default_rng(1).uniform(-1, 1, (20, 3)) + [0, 0, 6]
    p1, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, None)
    p2, _ = cv2.projectPoints(obj, np.array([0.0, 0.1, 0.0]), np.array([0.5, 0.0, 0.0]), K, None)
    E, mask = cv2.findEssentialMat(p1, p2, K)
    n, R, t, mask2 = cv2.recoverPose(E, p1, p2, K)
    return {"E": arr(E), "mask": arr(mask), "R": arr(R), "t": arr(t)}


@probe("shapes")
def triangulatePoints():
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), [[-0.5], [0], [0]]])
    pts = np.float64([[80, 90], [60, 70]]).T
    return arr(cv2.triangulatePoints(P1, P2, pts, pts + [[5], [0]]))


@probe("shapes", "mapx, mapy = cv2.initUndistortRectifyMap(...)")
def initUndistortRectifyMap():
    mx, my = cv2.initUndistortRectifyMap(K, DIST, None, K, (160, 120), cv2.CV_32FC1)
    return {"mapx": arr(mx), "mapy": arr(my)}


@probe("shapes", "newK, roi = cv2.getOptimalNewCameraMatrix(...)")
def getOptimalNewCameraMatrix():
    return describe(cv2.getOptimalNewCameraMatrix(K, DIST, (160, 120), 1))


# ---------------------------------------------------------------- C. pixels

@probe("pixels", "pixel-exact tests of nearest-neighbour resize")
def resize_nearest_up_1x4_to_1x6():
    return cv2.resize(np.uint8([[0, 10, 20, 30]]), (6, 1), interpolation=cv2.INTER_NEAREST).tolist()


@probe("pixels", "pixel-exact tests of nearest-neighbour resize")
def resize_nearest_down_1x6_to_1x4():
    row = np.uint8([[0, 10, 20, 30, 40, 50]])
    return cv2.resize(row, (4, 1), interpolation=cv2.INTER_NEAREST).tolist()


@probe("pixels", "pixel-exact tests of nearest-neighbour resize")
def resize_nearest_3x3_to_5x5():
    a = np.arange(9, dtype=np.uint8).reshape(3, 3)
    return cv2.resize(a, (5, 5), interpolation=cv2.INTER_NEAREST).tolist()


@probe("pixels")
def resize_linear_1x4_to_1x7():
    row = np.float32([[0, 10, 20, 30]])
    return np.round(cv2.resize(row, (7, 1), interpolation=cv2.INTER_LINEAR), 3).tolist()


@probe("pixels")
def resize_cubic_fingerprint():
    return fingerprint(cv2.resize(texture(), (123, 77), interpolation=cv2.INTER_CUBIC))


@probe("pixels")
def resize_area_fingerprint():
    return fingerprint(cv2.resize(texture(), (50, 40), interpolation=cv2.INTER_AREA))


@probe("pixels", "sub-pixel warps compared against stored baselines")
def warpAffine_half_pixel_shift():
    row = np.float32([[0, 10, 20, 30, 40, 50, 60, 70]])
    M = np.float32([[1, 0, 0.5], [0, 1, 0]])
    out = cv2.warpAffine(row, M, (8, 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.round(out, 3).tolist()


@probe("pixels")
def warpAffine_rotate_fingerprint():
    t = texture()
    M = cv2.getRotationMatrix2D((100.0, 80.0), 17.0, 1.0)
    return fingerprint(cv2.warpAffine(t, M, (200, 160), flags=cv2.INTER_LINEAR))


@probe("pixels")
def warpPerspective_fingerprint():
    H = cv2.getPerspectiveTransform(QUAD_SRC, QUAD_DST)
    return fingerprint(cv2.warpPerspective(texture(), H, (200, 160), flags=cv2.INTER_LINEAR))


@probe("pixels")
def remap_quarter_pixel():
    row = np.float32([[0, 10, 20, 30, 40, 50, 60, 70]])
    mx = (np.arange(8, dtype=np.float32) + 0.25)[None, :]
    my = np.zeros((1, 8), np.float32)
    return np.round(cv2.remap(row, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 3).tolist()


@probe("pixels")
def GaussianBlur_fingerprint():
    return fingerprint(cv2.GaussianBlur(texture(), (7, 7), 1.7))


@probe("pixels")
def Canny_fingerprint():
    return fingerprint(cv2.Canny(texture(), 50, 150))


@probe("pixels", "text overlays compared against stored baselines")
def putText_hershey_pixels():
    img = np.zeros((60, 240), np.uint8)
    cv2.putText(img, "OpenCV 5", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_8)
    return {"nonzero": int(cv2.countNonZero(img)), **fingerprint(img)}


@probe("pixels", "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
def cvtColor_BGR2GRAY_primaries():
    px = np.uint8([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])  # pure B, G, R
    return cv2.cvtColor(px, cv2.COLOR_BGR2GRAY).ravel().tolist()


@probe("pixels")
def cvtColor_BGR2HSV_red():
    return cv2.cvtColor(np.uint8([[[0, 0, 255]]]), cv2.COLOR_BGR2HSV).ravel().tolist()


@probe("pixels", "img + 50 (wraps) vs cv2.add(img, 50) (saturates)")
def add_saturation_vs_numpy():
    a, b = np.uint8([[250]]), np.uint8([[10]])
    return {"cv2.add": int(cv2.add(a, b)[0, 0]), "numpy +": int((a + b)[0, 0])}


@probe("pixels")
def threshold_otsu_value():
    img = np.zeros((40, 40), np.uint8)
    img[:, 20:] = 200
    img = cv2.GaussianBlur(img, (5, 5), 2)
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


@probe("pixels", "contourArea traces boundary-pixel centres: area = pixels - perimeter/2 + 1")
def contourArea_square_20px():
    img = np.zeros((40, 40), np.uint8)
    cv2.rectangle(img, (10, 10), (29, 29), 255, -1)
    c = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2][0]
    return {"pixels": int(cv2.countNonZero(img)), "contourArea": cv2.contourArea(c),
            "arcLength": round(cv2.arcLength(c, True), 3)}


@probe("pixels")
def contourArea_disk_r20():
    img = np.zeros((60, 60), np.uint8)
    cv2.circle(img, (30, 30), 20, 255, -1)
    c = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2][0]
    return {"pixels": int(cv2.countNonZero(img)), "contourArea": cv2.contourArea(c),
            "pi_r2": round(np.pi * 20 ** 2, 1)}


def _nearest_convention(flag: int) -> dict:
    """Classify a nearest-neighbour mode over all 1-D size pairs 2..16 -> 2..32.

    legacy:  src = floor(dst * in/out)          (OpenCV INTER_NEAREST, 2.x-4.x)
    centre:  src = floor((dst + 0.5) * in/out)  (Pillow / INTER_NEAREST_EXACT)
    """
    counts = {"legacy": 0, "centre": 0, "both": 0, "neither": 0}
    for n_in in range(2, 17):
        row = np.arange(n_in, dtype=np.uint8)[None, :]
        for n_out in range(2, 33):
            if n_out == n_in:
                continue
            got = cv2.resize(row, (n_out, 1), interpolation=flag)[0].astype(int)
            d = np.arange(n_out)
            legacy = np.minimum((d * n_in) // n_out, n_in - 1)
            centre = np.minimum(((2 * d + 1) * n_in) // (2 * n_out), n_in - 1)
            is_l, is_c = np.array_equal(got, legacy), np.array_equal(got, centre)
            counts["both" if is_l and is_c else "legacy" if is_l else "centre" if is_c else "neither"] += 1
    return counts


@probe("pixels", "migration wiki: INTER_NEAREST 'now follows the Pillow convention'")
def resize_nearest_convention_sweep():
    return _nearest_convention(cv2.INTER_NEAREST)


@probe("pixels")
def resize_nearest_exact_convention_sweep():
    return _nearest_convention(cv2.INTER_NEAREST_EXACT)


@probe("pixels")
def resize_nearest_2d_fingerprint():
    return fingerprint(cv2.resize(texture(), (317, 233), interpolation=cv2.INTER_NEAREST))


# ---------------------------------------------------------------- D. gotchas & errors

@probe("gotchas", "cv2.rectangle(img[:, :, ::-1], ...)  (negative-stride view)")
def draw_on_reversed_channel_view():
    bgr = np.zeros((40, 40, 3), np.uint8)
    cv2.rectangle(bgr[:, :, ::-1], (5, 5), (20, 20), (255, 0, 0), 2)
    return "drawn"


@probe("gotchas", "drawing into an ROI view writes through to the parent")
def draw_on_roi_view():
    bgr = np.zeros((40, 40, 3), np.uint8)
    cv2.rectangle(bgr[10:30, 10:30], (2, 2), (10, 10), (0, 255, 0), -1)
    return {"parent_modified": bool(bgr.any())}


@probe("gotchas", "cv2.circle(img.T, ...)  (transposed view)")
def draw_on_transposed_view():
    g = np.zeros((40, 30), np.uint8)
    cv2.circle(g.T, (10, 10), 5, 255, -1)
    return {"parent_modified": bool(g.any())}


@probe("gotchas", "cv2.circle(img, (x / 2, y / 2), ...)  (float centre)")
def circle_float_center():
    cv2.circle(np.zeros((30, 30), np.uint8), (10.5, 10.5), 3, 255, -1)
    return "drawn"


@probe("gotchas", "cv2.circle(img, tuple(np.int64 values), ...)")
def circle_numpy_int_center():
    cv2.circle(np.zeros((30, 30), np.uint8), (np.int64(10), np.int64(12)), 3, 255, -1)
    return "drawn"


@probe("gotchas", "img = cv2.imread(path)  # silently None on a bad path")
def imread_missing_file():
    return describe(cv2.imread("/nonexistent/definitely_missing.png"))


@probe("gotchas", "cv2.imshow in a headless sandbox")
def imshow_headless():
    cv2.imshow("probe", np.zeros((10, 10), np.uint8))
    return "shown"


@probe("gotchas")
def waitKey_headless():
    return cv2.waitKey(1)


@probe("gotchas", "cvtColor(gray, COLOR_BGR2GRAY) on an already-gray image")
def cvtColor_gray_input():
    return arr(cv2.cvtColor(np.zeros((10, 10), np.uint8), cv2.COLOR_BGR2GRAY))


@probe("gotchas", "GaussianBlur(img, (4, 4), 0)  (even kernel)")
def GaussianBlur_even_kernel():
    return arr(cv2.GaussianBlur(np.zeros((10, 10), np.uint8), (4, 4), 0))


@probe("gotchas", "Canny on a float image")
def Canny_float_input():
    return arr(cv2.Canny(np.zeros((10, 10), np.float32), 50, 150))


@probe("gotchas", "findContours on a float mask")
def findContours_float_input():
    return describe(cv2.findContours(np.zeros((10, 10), np.float32), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2])


@probe("gotchas", "findContours on a 3-channel image")
def findContours_3channel():
    return describe(cv2.findContours(np.zeros((10, 10, 3), np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2])


@probe("gotchas", "getPerspectiveTransform with float64 points")
def getPerspectiveTransform_float64():
    return arr(cv2.getPerspectiveTransform(QUAD_SRC.astype(np.float64), QUAD_DST.astype(np.float64)))


@probe("gotchas", "getPerspectiveTransform with int32 points")
def getPerspectiveTransform_int32():
    return arr(cv2.getPerspectiveTransform(QUAD_SRC.astype(np.int32), QUAD_DST.astype(np.int32)))


@probe("gotchas", "cv2.perspectiveTransform(pts_Nx2, H)  (forgot the (N, 1, 2) reshape)")
def perspectiveTransform_Nx2_input():
    H = cv2.getPerspectiveTransform(QUAD_SRC, QUAD_DST)
    return arr(cv2.perspectiveTransform(QUAD_SRC, H))


@probe("gotchas", "cv2.resize(img, (w, h)): dsize is (width, height)")
def resize_dsize_order():
    return list(cv2.resize(np.zeros((10, 20), np.uint8), (5, 8)).shape)


@probe("gotchas", "if cap.get(prop) == 0: unsupported  (5.0 returns -1)")
def VideoCapture_get_unopened():
    return cv2.VideoCapture().get(cv2.CAP_PROP_FPS)


@probe("gotchas")
def VideoCapture_open_missing():
    return cv2.VideoCapture("/nonexistent/clip.mp4").isOpened()


@probe("gotchas", "passing a numpy bool mask straight to OpenCV")
def bool_array_input():
    m = np.zeros((8, 8), bool)
    m[2:5, 2:5] = True
    return describe(cv2.countNonZero(m))


@probe("gotchas", "int64 arrays (numpy's default int) passed to OpenCV")
def int64_array_input():
    a = np.arange(16, dtype=np.int64).reshape(4, 4)
    return arr(cv2.add(a, a))


@probe("gotchas", "int64 counts/accumulators passed through cv2 arithmetic (silent int32 narrowing)")
def int64_overflow_narrowing():
    a = np.array([[2 ** 40, 5]], np.int64)
    out = cv2.add(a, a)
    return {"dtype": str(out.dtype), "values": out.tolist(), "expected": (2 * a).tolist()}


@probe("gotchas", "uint32 arrays passed to OpenCV")
def uint32_array_input():
    a = np.arange(16, dtype=np.uint32).reshape(4, 4)
    return arr(cv2.add(a, a))


@probe("gotchas", "float16 arrays passed to OpenCV")
def float16_array_input():
    a = np.ones((4, 4), np.float16)
    return arr(cv2.add(a, a))


@probe("gotchas", "cv2.threshold on int64 input")
def threshold_int64_input():
    return arr(cv2.threshold(np.arange(16, dtype=np.int64).reshape(4, 4), 7, 255, cv2.THRESH_BINARY)[1])


TRI_INT64 = np.array([[2, 2], [25, 3], [12, 20]])  # numpy's default int is int64


@probe("gotchas", "cv2.fillPoly(img, [np.array([[x, y], ...])], 255)  (numpy default int64)")
def fillPoly_int64_points():
    img = np.zeros((30, 30), np.uint8)
    cv2.fillPoly(img, [TRI_INT64], 255)
    return {"filled_pixels": int(cv2.countNonZero(img))}


@probe("gotchas", "cv2.polylines(img, [pts_int64], True, 255)")
def polylines_int64_points():
    img = np.zeros((30, 30), np.uint8)
    cv2.polylines(img, [TRI_INT64], True, 255)
    return {"drawn_pixels": int(cv2.countNonZero(img))}


@probe("gotchas", "cv2.fillPoly with float32 points")
def fillPoly_float32_points():
    img = np.zeros((30, 30), np.uint8)
    cv2.fillPoly(img, [TRI_INT64.astype(np.float32)], 255)
    return {"filled_pixels": int(cv2.countNonZero(img))}


@probe("gotchas", "cv2.drawContours(img, [cnt_Nx2], -1, 255, -1)  (contour without the middle axis)")
def drawContours_Nx2_int32():
    img = np.zeros((30, 30), np.uint8)
    cv2.drawContours(img, [TRI_INT64.astype(np.int32)], -1, 255, -1)
    return {"filled_pixels": int(cv2.countNonZero(img))}


@probe("gotchas", "cv2.contourArea(np.array([[x, y], ...]))  (numpy default int64)")
def contourArea_int64_points():
    return cv2.contourArea(TRI_INT64)


@probe("gotchas", "cv2.line(img, (x / 2, y), ...)  (float endpoint)")
def line_float_endpoint():
    cv2.line(np.zeros((30, 30), np.uint8), (1.5, 2.0), (20.0, 20.0), 255)
    return "drawn"


@probe("gotchas", "cv2.bitwise_and(img, img, mask=bool_mask)")
def bitwise_and_bool_mask():
    img = texture()
    mask = np.zeros(img.shape, bool)
    mask[20:60, 30:90] = True
    return {"kept_pixels": int(cv2.countNonZero(cv2.bitwise_and(img, img, mask=mask)))}


@probe("gotchas", "cv2.addWeighted(uint8_img, 0.5, float32_img, 0.5, 0)")
def addWeighted_mixed_dtypes():
    a = np.zeros((10, 10), np.uint8)
    return arr(cv2.addWeighted(a, 0.5, a.astype(np.float32), 0.5, 0))


@probe("gotchas", "cv2.remap with float64 maps")
def remap_float64_maps():
    mx = np.tile(np.arange(20, dtype=np.float64), (10, 1))
    my = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 20))
    return arr(cv2.remap(np.zeros((10, 20), np.uint8), mx, my, cv2.INTER_LINEAR))


@probe("gotchas", "cv2.countNonZero(bgr_img)")
def countNonZero_3channel():
    return cv2.countNonZero(np.zeros((10, 10, 3), np.uint8))


@probe("gotchas", "cv2.imdecode(bad_bytes)  # None, not an exception")
def imdecode_garbage():
    return describe(cv2.imdecode(np.frombuffer(b"not an image at all", np.uint8), cv2.IMREAD_COLOR))


# ---------------------------------------------------------------- runner

def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: describe(o)))


def run_all() -> dict:
    out = {}
    for p in PROBES:
        _CURRENT["name"] = p["name"]
        rec = {"category": p["category"], "idiom": p["idiom"]}
        try:
            rec.update(ok=True, obs=jsonable(p["fn"]()))
        except Exception as e:  # recording failures is the point of a probe
            rec.update(ok=False, error=short_error(e))
        out[p["name"]] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(__file__).parent / "results" / f"probe_{cv2.__version__}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": {"cv2": cv2.__version__, "numpy": np.__version__,
                "python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}"},
        "probes": run_all(),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), **ARRAYS)
    n_fail = sum(not r["ok"] for r in payload["probes"].values())
    print(f"cv2 {cv2.__version__}: {len(payload['probes'])} probes, {n_fail} raised -> {out}")


if __name__ == "__main__":
    main()
