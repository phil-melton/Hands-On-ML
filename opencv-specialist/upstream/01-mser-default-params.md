# 01. MSER_create() with default parameters returns 0 regions on 5.0.0

| | |
|---|---|
| Classification | Bug in 5.0.0 (confirmed; root cause is one changed line). **Already reported and fixed upstream.** |
| Recommendation | **Do not file.** Duplicate of [#29652](https://github.com/opencv/opencv/issues/29652) (closed, completed); fixed on `5.x` by [#29653](https://github.com/opencv/opencv/pull/29653) |
| Repro | [`repro_01_mser_default_params.py`](repro_01_mser_default_params.py) |
| Verified on | opencv-python-headless 5.0.0.93 and 4.14.0.94 (numpy 2.4.6, Python 3.12.3, Ubuntu 24.04 on WSL2) |

## What I found

- On 5.0.0 `cv2.MSER_create().detectRegions(img)` returns no regions on the clean-evidence image (the 25-rectangle
  texture of `probes/api_diff.py`: 22 regions on 4.14.0, 0 on 5.0.0) and on the image used in #29652 (2 on
  4.14.0, 0 on 5.0.0). `min_diversity=0.0` restores 22 and 2. The defaults (`getMinDiversity() == 0.2` etc.) are
  identical in both versions.
- On 4.14.0 the count does not depend on `min_diversity` at all (2 / 22 / 116 for every value I tried). On 5.0.0
  it does, and the default 0.2 prunes everything on the two synthetic images and 116 -> 17 on uniform noise.
- **Source diff, `mser.cpp` 4.14.0 vs 5.0.0: one line.** In `CompHistory::checkAndCapture` (the grayscale path):

  ```
  4.14.0 (features2d/src/mser.cpp:267):  ... || var < 0.f                   || var > wp.p.maxVariation )
  5.0.0  (features/src/mser.cpp:267):    ... || var < wp.p.minDiversity     || var > wp.p.maxVariation )
  ```

  So on 5.0.0 the region *variation* is compared with `min_diversity` (default 0.2) as a lower bound. In 4.14.0
  `min_diversity` is used only in the colour path. The 5.0.0 docstring still says
  "`min_diversity` for color image, trace back to cut off mser with diversity less than min_diversity"
  ([features.hpp L541](https://github.com/opencv/opencv/blob/5.0.0/modules/features/include/opencv2/features.hpp#L541)).
- Origin of the change: PR [#20170](https://github.com/opencv/opencv/pull/20170) "Handle minDiversity in Grayscale
  MSER", merged 2021-05-29 into `next` (the development branch of what is now 5.x; I did not trace the branch history).
  Neither 4.14.0 nor the current `4.x` branch has the change (`var < 0.f` at line 267 in both).
- Fix: commit [`02466fac6b`](https://github.com/opencv/opencv/commit/02466fac6bad93a5934e27c79d5cffd0bbcac7ec)
  (PR #29653, merged 2026-08-04) restores `var < 0.f` and adds a regression test. It is on the `5.x` branch only.
  As of 2026-10-20 the latest release is 5.0.0 (tag 2026-06-06; PyPI wheel 5.0.0.93, 2026-07-02), so no wheel contains the fix.
  I checked it: `opencv_core`, `opencv_imgproc` and `opencv_features` built from `5.x` commit
  [`9123a3c209`](https://github.com/opencv/opencv/commit/9123a3c209b48ffa38397a889152b2e1f81b656d) (2026-10-20, `5.1.0-dev`),
  C++ in [`check_5x_fixes.cpp`](check_5x_fixes.cpp): default `MSER::create()` finds 2 regions on the #29652 image
  (`getMinDiversity()` is still 0.20) and 22 on a 25-rectangle image, the same as with `min_diversity=0`.
- The "7 -> 0" figure in PLAN.md section 0 mixes a changed `putText` fixture with the detector change; the texture
  probe (22 -> 0, identical input) is the clean evidence, as already noted there.
- #29652 also says that `bboxes` is an empty *tuple* on 5.0.0 "rather than an ndarray (0, 4) as on OpenCV 4". I could not
  confirm the second half: on 4.14.0 `detectRegions` of an 8x8 zero image also returns `bboxes == ()`
  (`type(bboxes)` is `tuple`; an ndarray only appears when there is at least one region). So that part is
  long-standing binding behaviour, not a 5.0 change.

## Duplicate search

| Item | State | Covers this case? |
|---|---|---|
| [#29652](https://github.com/opencv/opencv/issues/29652) MSER::detectRegions() finds 0 regions on OpenCV 5.0.0 where 4.13.0 finds 2, with identical default parameters | closed, completed (2026-08-04) | Yes, same symptom, same version (5.0.0.93 wheel) |
| [#29653](https://github.com/opencv/opencv/pull/29653) features: fix MSER discarding stable regions via minDiversity | merged to 5.x (2026-08-04) | Yes, the fix |
| [#20170](https://github.com/opencv/opencv/pull/20170) Handle minDiversity in Grayscale MSER | merged to `next` (2021-05-29) | Introduced the behaviour |
| [#29752](https://github.com/opencv/opencv/pull/29752) features: re-enable & triage DISABLED tests | merged (2026-08-25) | Related test housekeeping |

Searches: GitHub issue/PR search `repo:opencv/opencv` for "MSER", "MSER minDiversity", "MSER 5.0 regions"; web search for
"opencv 5.0.0 MSER detectRegions returns 0 regions minDiversity regression" (nothing beyond the GitHub items).

## Consequence for this project

`cvbench/gotchas/drift.py` (`drift_mser_default`) is correct for the pinned 5.0.0.93 and will stop being a gotcha once
a release contains #29653. PLAN.md section 0 item 3 says "I found no upstream issue yet": the issue exists
(#29652) and is fixed on `5.x`.

---

## Draft (kept for completeness; **not recommended for filing**, it would duplicate #29652)

**Title:** MSER_create() with default parameters returns 0 regions on 5.0.0 (min_diversity used as lower bound for variation in the grayscale path)

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`)
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

With default parameters `MSER_create().detectRegions()` returns no regions on 5.0.0 for images on which 4.14.0 returns
regions. `getDelta()`, `getMinArea()`, `getMaxArea()`, `getMaxVariation()` and `getMinDiversity()` are identical in both
versions. On 4.14.0 the number of regions does not depend on `min_diversity`; on 5.0.0 it does.

```
OpenCV 4.14.0 | numpy 2.4.6
defaults: delta=5 min_area=60 max_area=14400 max_variation=0.25 min_diversity=0.20

regions returned by detectRegions:
image                         default   div=0     div=0.001 div=0.01  div=0.05  div=0.1   div=0.2   
rectangle with hole (#29652)  2         2         2         2         2         2         2         
25 random rectangles          22        22        22        22        22        22        22        
uniform noise 120x160         116       116       116       116       116       116       116       

expected: the same counts as 4.x (4.14.0 returns 2 / 22 / 116 for every column, because the
grayscale code path ignores min_diversity there).
RESULT: not reproduced
```

```
OpenCV 5.0.0 | numpy 2.4.6
defaults: delta=5 min_area=60 max_area=14400 max_variation=0.25 min_diversity=0.20

regions returned by detectRegions:
image                         default   div=0     div=0.001 div=0.01  div=0.05  div=0.1   div=0.2   
rectangle with hole (#29652)  0         2         0         0         0         0         0         
25 random rectangles          0         22        0         0         0         0         0         
uniform noise 120x160         17        116       115       115       85        58        17        

expected: the same counts as 4.x (4.14.0 returns 2 / 22 / 116 for every column, because the
grayscale code path ignores min_diversity there).
RESULT: REPRODUCED (defaults find 0 regions where min_diversity=0.0 finds some)
```

Comparing `mser.cpp` of the 4.14.0 and 5.0.0 tags, one line differs (`CompHistory::checkAndCapture`):
`var < 0.f` became `var < wp.p.minDiversity`. I did not test other changes. This was already reported as #29652 and
fixed on `5.x` by #29653.

### Steps to reproduce

```python
"""Repro 01: cv2.MSER_create() with default parameters finds no regions on OpenCV 5.0.0.

Needs only numpy and cv2. Runs in about a second. Prints, for the same images and the same
parameter values, how many regions MSER returns. Exit code 1 if the defaults return 0 regions on an
image where min_diversity=0.0 (or the 4.x behaviour) returns some.
"""
import sys

import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)


def rect_with_hole():
    """The image of opencv/opencv#29652: a filled rectangle with a circular hole."""
    img = np.zeros((200, 200), np.uint8)
    cv2.rectangle(img, (40, 40), (160, 160), 255, -1)
    cv2.circle(img, (100, 100), 30, 0, -1)
    return img


def random_rectangles(seed=0, shape=(160, 200)):
    """25 filled rectangles of random gray levels on a dark background."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 30, np.uint8)
    for _ in range(25):
        x, y = rng.integers(10, shape[1] - 30), rng.integers(10, shape[0] - 30)
        w, h = rng.integers(8, 25, size=2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(80, 255)), -1)
    return img


def uniform_noise(seed=0, shape=(120, 160)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


m = cv2.MSER_create()
print("defaults: delta=%d min_area=%d max_area=%d max_variation=%.2f min_diversity=%.2f"
      % (m.getDelta(), m.getMinArea(), m.getMaxArea(), m.getMaxVariation(), m.getMinDiversity()))

images = {"rectangle with hole (#29652)": rect_with_hole(),
          "25 random rectangles": random_rectangles(),
          "uniform noise 120x160": uniform_noise()}
diversities = (None, 0.0, 0.001, 0.01, 0.05, 0.1, 0.2)   # None = library default (0.2)

print("\nregions returned by detectRegions:")
print("%-30s" % "image" + "".join("%-10s" % ("default" if d is None else "div=%g" % d) for d in diversities))
reproduced = False
for name, img in images.items():
    counts = []
    for d in diversities:
        mser = cv2.MSER_create() if d is None else cv2.MSER_create(min_diversity=d)
        counts.append(len(mser.detectRegions(img)[0]))
    print("%-30s" % name + "".join("%-10d" % c for c in counts))
    if counts[0] == 0 and counts[1] > 0:
        reproduced = True

print("\nexpected: the same counts as 4.x (4.14.0 returns 2 / 22 / 116 for every column, because the")
print("grayscale code path ignores min_diversity there).")
print("RESULT:", "REPRODUCED (defaults find 0 regions where min_diversity=0.0 finds some)" if reproduced
      else "not reproduced")
sys.exit(1 if reproduced else 0)
```

### Issue submission checklist

- [x] I report the issue, it's not a question
- [x] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution (**false: #29652 exists**)
- [ ] I updated to the latest OpenCV version and the issue is still there (**false for `5.x` HEAD, which contains the fix; true for the latest release**)
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
