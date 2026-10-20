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
