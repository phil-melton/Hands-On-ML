# 03. cv2.remap accepts a map2 of any dtype when map1 is float32

| | |
|---|---|
| Classification | **Bug in 5.0.0** (missing input validation; 4.14.0 rejects these calls with an assertion). **Already fixed on `5.x`** (yesterday) |
| Recommendation | **Do not file.** Fixed by [#30286](https://github.com/opencv/opencv/pull/30286) "Fix code according to fuzzer findings" (merged 2026-10-19), not yet in any release |
| Repro | [`repro_03_remap_map2_dtype.py`](repro_03_remap_map2_dtype.py) |
| Verified on | opencv-python-headless 5.0.0.93 (affected), 4.14.0.94 (rejects), `opencv_imgproc` built from `5.x` commit `9123a3c209` (rejects, C++) |

## What I found

- `cv2.remap(img, map1, map2, interp)` with `map1` float32 and `map2` of dtype float64, int32, int16, uint16, uint8, int8,
  float16 or int64 (numerically the identity y-map) returns an image on 5.0.0 with no error. The result is wrong: mean
  absolute difference from the identity image of 70.6 for float64 and 100-102 for the others. Evidently the bytes of
  `map2` are read as float32. 4.14.0 raises `(-215:Assertion failed) ((map1.type() == CV_32FC2 || ...` for every one of them.
  The recorded probe `remap_float64_maps` (both maps float64) raises on both versions, which is why it did not show this.
- For one-byte dtypes the result is not even stable: in a separate loop (20 calls per dtype, in one process) the mean
  difference for `uint8` and `int8` `map2` came out as either 100.0 or 101.25, depending on the run, while the other dtypes
  were stable. That would fit the kernel reading past the end of the smaller `map2` buffer. I did not confirm this with a
  memory checker, and the shipped repro often prints a single value (101.2) because the outcome depends on heap contents.
- Source (5.0.0 tag, `cv::remap` starts at
  [imgwarp.cpp L1407](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/imgwarp.cpp#L1407)): the
  `CV_Assert` on the map types is at
  [L1658](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/imgwarp.cpp#L1658), after several earlier
  paths (`CALL_HAL` blocks and a call to `genericWarp(...)` that returns when it succeeds). I did not trace which path
  accepts these inputs.
- `5.x` branch: the type check was moved to the top of `cv::remap`:
  `CV_Assert( (map1.type() == CV_16SC2 && (map2.empty() || map2.type() == CV_16UC1 || map2.type() == CV_16SC1)) || (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) || (map1.type() == CV_32FC2 && map2.empty()) )`
  ([5.x imgwarp.cpp](https://github.com/opencv/opencv/blob/5.x/modules/imgproc/src/imgwarp.cpp), added by #30286).
  Checked on `5.x`: with `opencv_core` and `opencv_imgproc` built from commit [`9123a3c209`](https://github.com/opencv/opencv/commit/9123a3c209b48ffa38397a889152b2e1f81b656d) (2026-10-20, `5.1.0-dev`), the C++ `cv::remap(src, dst, float32 map1, float64 map2)` and `(..., uint8 map2)` both throw `cv::Exception` code -215 (see [`check_5x_fixes.cpp`](check_5x_fixes.cpp)).

## Duplicate search

| Item | State | Covers this case? |
|---|---|---|
| [#30286](https://github.com/opencv/opencv/pull/30286) Fix code according to fuzzer findings (adds the `remap` map-type assertion, `convertMaps` and `warpPolar` checks) | merged to 5.x 2026-10-19 | Yes, as a fix; it does not name an issue |
| [#30160](https://github.com/opencv/opencv/pull/30160) imgproc: reject N-D input in remap, warpAffine and warpPerspective (fixes #29589) | merged 2026-10-06 | No (N-D input) |
| [#30155](https://github.com/opencv/opencv/issues/30155) remap() with INTER_LANCZOS4 reads past lanczos4_tab for new depths | closed | No (different defect, same function) |
| [#30296](https://github.com/opencv/opencv/issues/30296) Python stubs reject scalar arithmetic operands and empty remap map2 | closed | No (typing stubs) |

Searches (GitHub, `repo:opencv/opencv`): "remap map2 type float32 assertion", "remap map1 map2 depth reinterpret",
"remap map2 CV_32FC1 type check", "remap python 5.x" (newest first). I found no issue for this specific input.

---

## Draft (bug-report template; not recommended for filing, it is fixed on `5.x`)

**Title:** `cv2.remap` accepts a `map2` of any dtype when `map1` is float32 and reinterprets its bytes (5.0.0; 4.14.0 raises)

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`)
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

With `map1` of type float32 (`CV_32FC1`) and `map2` of any other element type, `cv2.remap` returns an image on 5.0.0
instead of raising. The maps are numerically the identity, but the result is not the identity image, so the bytes of
`map2` seem to be read as float32. 4.14.0 raises an assertion for the same calls.

4.14.0:

```
OpenCV 4.14.0 | numpy 2.4.6
float32 / float32 maps give the identity image: True
map2 float64  -> raised cv2.error (-215:Assertion failed)
map2 int32    -> raised cv2.error (-215:Assertion failed)
map2 int16    -> raised cv2.error (-215:Assertion failed)
map2 uint16   -> raised cv2.error (-215:Assertion failed)
map2 uint8    -> raised cv2.error (-215:Assertion failed)
map2 int8     -> raised cv2.error (-215:Assertion failed)
map2 float16  -> raised cv2.error (-215:Assertion failed)
map2 int64    -> raised cv2.error (-215:Assertion failed)

expected: every non-float32 map2 is rejected (as in 4.14.0), or converted so the result is the identity image.
RESULT: not reproduced
```

5.0.0:

```
OpenCV 5.0.0 | numpy 2.4.6
float32 / float32 maps give the identity image: True
map2 float64  -> returned an image; mean |difference| from the identity image: 70.6
map2 int32    -> returned an image; mean |difference| from the identity image: 100.0
map2 int16    -> returned an image; mean |difference| from the identity image: 100.0
map2 uint16   -> returned an image; mean |difference| from the identity image: 100.0
map2 uint8    -> returned an image; mean |difference| from the identity image: 101.2
map2 int8     -> returned an image; mean |difference| from the identity image: 101.2
map2 float16  -> returned an image; mean |difference| from the identity image: 102.1
map2 int64    -> returned an image; mean |difference| from the identity image: 100.0

expected: every non-float32 map2 is rejected (as in 4.14.0), or converted so the result is the identity image.
RESULT: REPRODUCED, accepted map2 dtypes: float64, int32, int16, uint16, uint8, int8, float16, int64
```

The documented combinations are two `CV_32FC1` maps, or one `CV_32FC2` / `CV_16SC2` map with an empty (or
`CV_16UC1` / `CV_16SC1`) `map2`. In a separate loop the results for `uint8` and `int8` `map2` varied between calls (mean
difference 100.0 or 101.25), which would fit reads past the end of the buffer; I did not confirm that with a memory checker.

### Steps to reproduce

```python
"""Repro 03: cv2.remap accepts a map2 of any dtype when map1 is float32 and reinterprets its bytes.

Needs only numpy and cv2. Runs in well under a second. map1 is a float32 identity x-map. map2 is the identity
y-map stored in different dtypes. The documented combination is two CV_32FC1 maps (or one CV_32FC2 / CV_16SC2 map
plus an empty or CV_16UC1 / CV_16SC1 map2). OpenCV 4.14.0 rejects every other map2 with an assertion. OpenCV
5.0.0 returns an image, silently reading the bytes of map2 as float32. Each call is repeated 20 times to show
whether the result is stable. Exit code 1 if any non-float32 map2 is accepted.
"""
import re
import sys

import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)

img = (np.arange(6 * 8, dtype=np.uint8).reshape(6, 8) * 5)
map_x, map_y = np.meshgrid(np.arange(8, dtype=np.float32), np.arange(6, dtype=np.float32))
identity = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
print("float32 / float32 maps give the identity image:", np.array_equal(identity, img))

accepted = []
for dtype in (np.float64, np.int32, np.int16, np.uint16, np.uint8, np.int8, np.float16, np.int64):
    name = np.dtype(dtype).name
    map2 = np.ascontiguousarray(map_y.astype(dtype))        # same numeric values 0..5, different storage type
    diffs = set()
    try:
        for _ in range(20):
            out = cv2.remap(img, map_x, map2, cv2.INTER_LINEAR)
            diffs.add(round(float(np.abs(out.astype(int) - identity.astype(int)).mean()), 1))
    except cv2.error as e:
        code = re.search(r"\(-\d+:[A-Za-z ]+\)", str(e))
        print("map2 %-8s -> raised cv2.error %s" % (name, code.group(0) if code else ""))
        continue
    accepted.append(name)
    note = "  <- differs between calls" if len(diffs) > 1 else ""
    print("map2 %-8s -> returned an image; mean |difference| from the identity image: %s%s"
          % (name, " or ".join("%.1f" % d for d in sorted(diffs)), note))

print("\nexpected: every non-float32 map2 is rejected (as in 4.14.0), or converted so the result is the identity image.")
print("RESULT:", ("REPRODUCED, accepted map2 dtypes: " + ", ".join(accepted)) if accepted else "not reproduced")
sys.exit(1 if accepted else 0)
```

### Issue submission checklist

- [x] I report the issue, it's not a question
- [x] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution
- [ ] I updated to the latest OpenCV version and the issue is still there (**false for `5.x` HEAD after #30286; true for the latest release**)
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
