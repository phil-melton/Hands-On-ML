# 05. connectedComponents label order is not raster order at connectivity 8

| | |
|---|---|
| Classification | **Documented behaviour** (unspecified order for Spaghetti and BBDT, explicitly row-major only for SAUF). Not a 5.0 change: identical on 4.14.0 |
| Recommendation | **Do not file.** The docs already state it |
| Repro | [`repro_05_ccl_label_order.py`](repro_05_ccl_label_order.py) |
| Verified on | opencv-python-headless 5.0.0.93 and 4.14.0.94 (identical output) |

## What I found

- Over 200 random 40x40 binary images (density 0.3), the number of images whose labels are **not** numbered in raster
  order of each component's first pixel:

  | connectivity | `CCL_DEFAULT` | `CCL_SAUF` (= `CCL_WU`) | `CCL_BBDT` (= `CCL_GRANA`) | `CCL_SPAGHETTI` (= `CCL_BOLELLI`) |
  |---|---|---|---|---|
  | 4 | 0 | 0 | 0 | 0 |
  | 8 | 200 | 0 | 200 | 200 |

  Same table on 4.14.0 and 5.0.0. Smallest example: two isolated pixels at (row 0, col 3) and (row 1, col 0), connectivity 8,
  default algorithm: the pixel in row 0 gets label 2.
- The documentation of both `connectedComponents` and `connectedComponentsWithStats` says: "Note that SAUF algorithm
  forces a row major ordering of labels while Spaghetti and BBDT do not."
  ([imgproc.hpp L3680-L3681, 5.0.0](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/include/opencv2/imgproc.hpp#L3680-L3681);
  the same sentence is at [L4018-L4019 in 4.14.0](https://github.com/opencv/opencv/blob/4.14.0/modules/imgproc/include/opencv2/imgproc.hpp#L4018-L4019)).
  `CCL_DEFAULT` is documented as Spaghetti for 8-way and Spaghetti4C for 4-way
  ([enum, L396](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/include/opencv2/imgproc.hpp#L396)).
- So the default is not documented to give raster order at either connectivity. In my sample it did at connectivity 4
  (0 of 200 violations) but that is an observation, not a promise.

## Duplicate search

No issue found, and none needed. Items that touch the topic:

| Item | State | Covers this case? |
|---|---|---|
| [#6823](https://github.com/opencv/opencv/pull/6823) Add Grana's connected components algorithm for 8-way connectivity | merged 2016 | Introduced BBDT |
| [#26049](https://github.com/opencv/opencv/issues/26049) different result while invoking connectedComponentsWithStats with multi thread | open | Not checked in detail; title suggests run-to-run differences, not label order |

Searches (GitHub, `repo:opencv/opencv`): "connectedComponents label order", "connectedComponents labels raster order",
"connectedComponents labeling order different connectivity", "connected components label ordering row major".

## Consequence for this project

Keep the `components_stats` contract as it is (sort explicitly by area and top-left corner; do not rely on label order).
If label order matters, use `cv2.CCL_SAUF`. Note also that on 5.x `HEAD` `connectedComponents` accepts bool masks
([#29622](https://github.com/opencv/opencv/pull/29622), merged 2026-07-28); in 5.0.0 it does not.

---

## Draft (not recommended for filing)

**Title:** connectedComponents: label order at connectivity 8 is not raster order for the default algorithm

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`)
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

At connectivity 8 the default algorithm (and `CCL_BBDT`, `CCL_SPAGHETTI`) numbers components in a block-scan order, not in
raster order of their first pixel. This is documented ("SAUF algorithm forces a row major ordering of labels while
Spaghetti and BBDT do not") and identical on 4.14.0, so this is not a bug report; it is kept as a record of the
measurement.

```
OpenCV 5.0.0 | numpy 2.4.6

images (of 200) whose label order is NOT raster order of first pixels:
connectivity  DEFAULT     SAUF        BBDT        SPAGHETTI   
4             0           0           0           0           
8             200         0           200         200         

two isolated foreground pixels, (row 0, col 3) and (row 1, col 0), connectivity 8, default algorithm:
[[0 0 0 2 0 0]
 [1 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]]
raster order would label the pixel in row 0 as 1; the default algorithm labels it 2
```

### Steps to reproduce

```python
"""Repro 05: connectedComponents label order is not raster order at 8-connectivity for the default algorithm.

Needs only numpy and cv2. Runs in about a second. Counts, over 200 random binary images, how many have labels
that are not numbered in raster order of each component's first pixel, for every algorithm and both
connectivities. Also prints the smallest example. The result is the same on 4.14.0 and 5.0.0.
The documentation of connectedComponents says: "SAUF algorithm forces a row major ordering of labels while
Spaghetti and BBDT do not" (modules/imgproc/include/opencv2/imgproc.hpp), so the observation below is the
documented behaviour. Exit code 0 always (this script is a measurement, not a failure test).
"""
import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)


def raster_ordered(labels):
    """True if labels 1..n appear in raster order of their first pixel."""
    flat = labels.ravel()
    idx = np.nonzero(flat)[0]
    lab = flat[idx]
    order = np.argsort(lab, kind="stable")
    lab_s, idx_s = lab[order], idx[order]
    firsts = idx_s[np.r_[0, np.nonzero(np.diff(lab_s))[0] + 1]]      # first pixel index of label 1, 2, ...
    return bool(np.all(np.diff(firsts) > 0))


algos = (("DEFAULT", cv2.CCL_DEFAULT), ("SAUF", cv2.CCL_SAUF), ("BBDT", cv2.CCL_BBDT), ("SPAGHETTI", cv2.CCL_SPAGHETTI))
rng = np.random.default_rng(0)
images = [(rng.random((40, 40)) < 0.3).astype(np.uint8) for _ in range(200)]

print("\nimages (of 200) whose label order is NOT raster order of first pixels:")
print("%-14s" % "connectivity" + "".join("%-12s" % n for n, _ in algos))
for conn in (4, 8):
    row = []
    for _, a in algos:
        bad = sum(not raster_ordered(cv2.connectedComponentsWithAlgorithm(im, conn, cv2.CV_32S, a)[1]) for im in images)
        row.append(bad)
    print("%-14d" % conn + "".join("%-12d" % b for b in row))

# the smallest example: two isolated pixels in consecutive rows
im = np.zeros((4, 6), np.uint8)
im[0, 3] = 1          # first pixel in raster order
im[1, 0] = 1
n, labels = cv2.connectedComponents(im, connectivity=8)
print("\ntwo isolated foreground pixels, (row 0, col 3) and (row 1, col 0), connectivity 8, default algorithm:")
print(labels)
print("raster order would label the pixel in row 0 as 1; the default algorithm labels it", labels[0, 3])
```

### Issue submission checklist

- [ ] I report the issue, it's not a question (**not an issue: documented behaviour**)
- [x] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution (**the documentation answers it**)
- [x] I updated to the latest OpenCV version and the issue is still there
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
