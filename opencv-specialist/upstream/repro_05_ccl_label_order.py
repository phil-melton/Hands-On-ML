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
