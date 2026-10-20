"""Repro 02: cv2.bitwise_not on a numpy bool array returns bytes 0xFE/0xFF, not a logical negation.

Needs only numpy and cv2. Runs instantly. On OpenCV 4.x a bool array is rejected ("data type = bool
is not supported"); on 5.0.0 the call is accepted and returns a bool array whose bytes are the bitwise
complement of 0/1, i.e. 0xFF/0xFE, so every element reads as True and the result is not a valid bool array.
Exit code 1 if the result differs from the logical negation.
"""
import sys

import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)

mask = np.array([[True, False, True],
                 [False, False, True]])
expected = ~mask                      # logical negation (numpy)

try:
    result = cv2.bitwise_not(mask)
except cv2.error as e:
    print("cv2.bitwise_not(bool array) raised:", str(e).strip().splitlines()[-2:])
    print("RESULT: bool input is rejected on this version (4.x behaviour), nothing to compare")
    sys.exit(0)

print("input mask            :", mask.tolist())
print("expected (np.invert)  :", expected.tolist())
print("cv2.bitwise_not dtype :", result.dtype)
print("cv2.bitwise_not values:", result.tolist())
print("cv2.bitwise_not bytes :", result.view(np.uint8).tolist())
print("expected bytes        :", expected.view(np.uint8).tolist())
print("True count            : got", int(result.sum()), "expected", int(expected.sum()))

# Same call for the other bool bitwise ops, which do return valid 0/1 bytes on 5.0.0:
a = np.array([[True, True, False, False]])
b = np.array([[True, False, True, False]])
for name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
    r = getattr(cv2, name)(a, b)
    print("%-12s bytes   :" % name, r.view(np.uint8).tolist())

ok = np.array_equal(result.view(np.uint8), expected.view(np.uint8))
print("RESULT:", "ok" if ok else "REPRODUCED (not a logical negation)")
sys.exit(0 if ok else 1)
