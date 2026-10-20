# 06. SIFT keypoints are offset by about +0.25 px in x and y

| | |
|---|---|
| Classification | **Known and documented; has an opt-in fix** (`enable_precise_upscale=True`). Not a 5.0 change: identical on 4.14.0 |
| Recommendation | **Do not file.** Duplicate of open issue [#23123](https://github.com/opencv/opencv/issues/23123) (label: bug); the fix [#23124](https://github.com/opencv/opencv/pull/23124) is merged but disabled by default ([#23270](https://github.com/opencv/opencv/pull/23270), "temporarily") |
| Repro | [`repro_06_sift_keypoint_offset.py`](repro_06_sift_keypoint_offset.py) |
| Verified on | opencv-python-headless 5.0.0.93 and 4.14.0.94 (identical output) |

## What I found

- An analytic Gaussian blob centred at known pixel-centre coordinates (convention of `resize`/`warp*`: pixel (row i, col j)
  is at x=j, y=i): the nearest SIFT keypoint is at +0.22 to +0.24 px in both x and y for four different centres,
  including integer and sub-pixel ones, on both versions. With `SIFT_create(enable_precise_upscale=True)` the error is
  0.000 to 0.030 px.
- Reflection check on a random textured image: a keypoint in the image and its counterpart in the left-right mirrored
  image sit 0.501 px apart from where mirror symmetry puts them (11 pairs); with `enable_precise_upscale=True` the
  mean offset is +0.000 px (67 pairs). Two offsets of +0.25 px in opposite directions give 0.5 px, consistent
  with the +0.246 px figure measured in `feature_homography` (n ~ 2250 per axis).
- Cause (from the source and from #23123): SIFT doubles the image first. `cv::resize(..., INTER_LINEAR)` maps
  destination pixel u to source u/2 - 0.25 in the pixel-centre convention, but keypoint coordinates are scaled back as
  `pt *= 0.5` (`kpt.pt.x = (c + xc) * (1 << octv)` in
  [sift.simd.hpp L389](https://github.com/opencv/opencv/blob/5.0.0/modules/features/src/sift.simd.hpp#L389), then
  `kpt.pt *= scale` in
  [sift.dispatch.cpp L566](https://github.com/opencv/opencv/blob/5.0.0/modules/features/src/sift.dispatch.cpp#L566)),
  which leaves +0.25 px. With `enable_precise_upscale=True` the upscale is a `warpAffine` with a pure 0.5 scale
  (`dst(u) = src(u/2)`, [sift.dispatch.cpp L196-L202](https://github.com/opencv/opencv/blob/5.0.0/modules/features/src/sift.dispatch.cpp#L196-L202)),
  which matches the `pt *= 0.5` mapping.
- Documentation (5.0.0,
  [features.hpp L376-L378](https://github.com/opencv/opencv/blob/5.0.0/modules/features/include/opencv2/features.hpp#L376-L378)):
  "`enable_precise_upscale` Whether to enable precise upscaling in the scale pyramid, which maps index x to 2x. This
  prevents localization bias. The option is disabled by default."
- ORB: the same project measurement gave about -0.1 px, level dependent. I did not isolate it. My mirror test gives
  -0.040 px (sd 0.189, 260 pairs) in x, which cannot separate a 0.1 px bias from detector asymmetry, so I make no claim about ORB.

## Duplicate search

| Item | State | Covers this case? |
|---|---|---|
| [#23123](https://github.com/opencv/opencv/issues/23123) SIFT detector has bias of 0.25 pixels due to chosen interpolation scheme in scale pyramid | **open** (2023-01-10), label `bug`, 5 comments | **Yes, exactly** |
| [#23124](https://github.com/opencv/opencv/pull/23124) different interpolation by double image | merged to 4.x 2023-02-17 | The fix (adds `enable_precise_upscale`) |
| [#23270](https://github.com/opencv/opencv/pull/23270) temporarily set "enable_precise_upscale=false" by default | merged 2023-02-17 | Why it is off: sporadic regression-test failures because `warpAffine()` is not bit-exact |

Searches (GitHub, `repo:opencv/opencv`): "enable_precise_upscale", "SIFT keypoint pixel offset bias", "SIFT keypoints
shifted 0.5 pixel", "SIFT localization bias upscale", "SIFT 0.25 pixel".

## Consequence for this project

For `feature_homography`: pass `enable_precise_upscale=True` to `cv2.SIFT_create` in the reference pipeline if the aim is
an unbiased detector, or keep the default and keep the offset in the calibration. Either way the offset is a documented
parameter choice rather than a defect to wait on. Whether the default should change in 5.x is a maintainers' question that
#23123 and #23270 leave open ("temporarily"); a comment on #23123 asking that would be the right form, not a new issue.

---

## Draft (not recommended for filing; at most a comment on #23123)

**Title:** SIFT: keypoints are offset by about +0.25 px with the default `enable_precise_upscale=False` (5.0.0)

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`)
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

With the default `enable_precise_upscale=False`, SIFT keypoints are 0.22-0.24 px too far right and down on an analytic
Gaussian blob, and mirrored images give keypoints 0.5 px apart from where symmetry puts them. With
`enable_precise_upscale=True` the offset is below 0.03 px. Output is identical on 4.14.0 and 5.0.0. This is the issue
described in #23123, whose fix (#23124) is disabled by default (#23270, "temporarily"). Question for the maintainers: is it
planned to change the default in 5.x?

```
OpenCV 5.0.0 | numpy 2.4.6

SIFT_create(enable_precise_upscale=False)
  true centre (80.00, 64.00)  keypoint ( 80.236,  64.236)  error (+0.236, +0.236)
  true centre (80.30, 64.70)  keypoint ( 80.533,  64.924)  error (+0.233, +0.224)
  true centre (60.50, 50.25)  keypoint ( 60.721,  50.482)  error (+0.221, +0.232)
  true centre (40.00, 90.00)  keypoint ( 40.236,  90.236)  error (+0.236, +0.236)

SIFT_create(enable_precise_upscale=True)
  true centre (80.00, 64.00)  keypoint ( 80.000,  64.000)  error (+0.000, +0.000)
  true centre (80.30, 64.70)  keypoint ( 80.286,  64.670)  error (-0.014, -0.030)
  true centre (60.50, 50.25)  keypoint ( 60.475,  50.239)  error (-0.025, -0.011)
  true centre (40.00, 90.00)  keypoint ( 40.000,  90.000)  error (+0.000, +0.000)
mirror test, enable_precise_upscale=False: 11 keypoint pairs, mean(x - (W-1-x_mirrored)) = +0.501 px
mirror test, enable_precise_upscale=True : 67 keypoint pairs, mean(x - (W-1-x_mirrored)) = +0.000 px

largest |error| default: 0.236 px   with enable_precise_upscale=True: 0.030 px
RESULT: default SIFT keypoints are offset by about +0.25 px
```

### Steps to reproduce

```python
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
```

### Issue submission checklist

- [x] I report the issue, it's not a question
- [ ] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution (**false: #23123 exists and the docs describe the option**)
- [x] I updated to the latest OpenCV version and the issue is still there
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
