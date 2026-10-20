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
