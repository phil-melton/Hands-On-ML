"""Repro 06: SIFT keypoints are offset by about +0.25 px in x and y unless enable_precise_upscale=True.

Needs only numpy and cv2. Runs in about a second. An analytic Gaussian blob is rendered with its centre at known
pixel-centre coordinates (pixel (row i, col j) is sampled at x=j, y=i, the convention of cv2.resize and cv2.warp*).
SIFT is asked for the keypoint nearest to the true centre. The result is the same on 4.14.0 and 5.0.0.
The documentation of SIFT::create says enable_precise_upscale "prevents localization bias. The option is
disabled by default." Exit code 1 if the default SIFT is off by more than 0.1 px.
"""
import sys

import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)


def blob(cx, cy, shape=(128, 160), sigma=6.0, amp=200.0, base=30.0):
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    g = base + amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


centres = [(80.0, 64.0), (80.3, 64.7), (60.5, 50.25), (40.0, 90.0)]
worst = {}
for precise in (False, True):
    sift = cv2.SIFT_create(enable_precise_upscale=precise)
    print("\nSIFT_create(enable_precise_upscale=%s)" % precise)
    errs = []
    for cx, cy in centres:
        kps = sift.detect(blob(cx, cy), None)
        k = min(kps, key=lambda k: (k.pt[0] - cx) ** 2 + (k.pt[1] - cy) ** 2)
        errs.append((k.pt[0] - cx, k.pt[1] - cy))
        print("  true centre (%5.2f, %5.2f)  keypoint (%7.3f, %7.3f)  error (%+.3f, %+.3f)"
              % (cx, cy, k.pt[0], k.pt[1], errs[-1][0], errs[-1][1]))
    worst[precise] = float(np.abs(errs).max())

# reflection check: a keypoint and its mirror image should be W-1 apart in x
rng = np.random.default_rng(3)
img = np.full((160, 200), 40, np.uint8)
for _ in range(40):
    x, y = rng.integers(10, 170), rng.integers(10, 130)
    w, h = rng.integers(8, 25, size=2)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(80, 255)), -1)
img = cv2.GaussianBlur(img, (0, 0), 1.2)
W = img.shape[1]
for precise in (False, True):
    sift = cv2.SIFT_create(enable_precise_upscale=precise)
    k1 = sift.detect(img, None)
    k2 = sift.detect(np.ascontiguousarray(img[:, ::-1]), None)
    p1 = np.array([k.pt for k in k1]); s1 = np.array([k.size for k in k1])
    dx = []
    for k in k2:
        xm = W - 1 - k.pt[0]
        d = np.hypot(p1[:, 0] - xm, p1[:, 1] - k.pt[1])
        j = int(np.argmin(d))
        if d[j] < 2 and abs(s1[j] - k.size) < 1e-3:
            dx.append(p1[j, 0] - xm)
    print("mirror test, enable_precise_upscale=%-5s: %d keypoint pairs, mean(x - (W-1-x_mirrored)) = %+.3f px"
          % (precise, len(dx), np.mean(dx)))

bad = worst[False] > 0.1
print("\nlargest |error| default: %.3f px   with enable_precise_upscale=True: %.3f px" % (worst[False], worst[True]))
print("RESULT:", "default SIFT keypoints are offset by about +0.25 px" if bad else "not reproduced")
sys.exit(1 if bad else 0)
