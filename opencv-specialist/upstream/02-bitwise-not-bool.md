# 02. cv2.bitwise_not on a bool array returns bytes 0xFE/0xFF instead of a logical not

| | |
|---|---|
| Classification | **Bug (silent wrong result)**; arguably also a documentation gap, because the docs describe `bitwise_not` only as a "bit-wise inversion" and do not say what that means for `CV_Bool` |
| Recommendation | **File.** It is the only one of the six with no duplicate found, no fix upstream, and the behaviour is still present on a current `5.x` build (checked in C++, see below) |
| Repro | [`repro_02_bitwise_not_bool.py`](repro_02_bitwise_not_bool.py) |
| Verified on | opencv-python-headless 5.0.0.93 (affected), 4.14.0.94 (bool input rejected), `opencv_core` built from `5.x` commit `9123a3c209` (affected) |

## What I found

- 4.14.0 rejects a numpy bool array in every `bitwise_*` call (`src data type = bool is not supported`).
- 5.0.0 accepts it. `bitwise_and`, `bitwise_or` and `bitwise_xor` return correct 0/1 bytes. **`bitwise_not` returns a
  `bool` array whose bytes are 0xFE where the input was True and 0xFF where it was False**, with or without `dst`
  or `mask`. Every element then reads as `True` (`result.tolist()` is all `True`, `result.sum()` is 6 where the
  input had 3 `False`), and the array is not a valid bool array. A 100x100 random input (larger arrays, SIMD path) gives only the bytes 254 and 255 as well.
- `bitwise_not` on `uint8` 0/1 gives 254/255 and on 0/255 gives 255/0 in both versions, which is the documented
  bit-wise behaviour. The bool result is the same operation applied to the one-byte storage of a type whose valid values are 0 and 1.
- Documentation (5.0.0 tag,
  [core.hpp L1358-L1373](https://github.com/opencv/opencv/blob/5.0.0/modules/core/include/opencv2/core.hpp#L1358-L1373)):
  "per-element bit-wise inversion"; the `mask` parameter is documented as "CV_8U, CV_8S or CV_Bool single channel
  array". Nothing says that a `CV_Bool` source is unsupported or what its inversion should be.
- Why it is easy to hit: 5.0 accepts bool in some calls where 4.x raised, so `cv2.bitwise_not(mask)` on a numpy bool
  mask no longer raises, and `~mask` / `np.logical_not(mask)` look equivalent.
- **Current `5.x`** (checked 2026-10-20): I built `opencv_core` from the `5.x` branch at commit
  [`9123a3c209`](https://github.com/opencv/opencv/commit/9123a3c209b48ffa38397a889152b2e1f81b656d) (version string `5.1.0-dev`) and
  ran `cv::bitwise_not` on a `CV_Bool` Mat holding 1 0 1 0 0 1 in C++: the result is 254 255 254 255 255 254, the same as in
  5.0.0. `cv::bitwise_and` of two `CV_Bool` Mats gives correct 0/1 bytes. This agrees with the source: the rewritten element-wise engine on `5.x`
  (`modules/core/src/arithm.simd.hpp`, `getNotFunc_`) picks `notKernel<uint8_t,...>` by element size under the comment
  "A bit-pattern op ignores the operand's semantic type".

## Duplicate search

No issue or PR found that reports this. Closest items (all different defects of the same area, `CV_Bool` on 5.x):

| Item | State | Covers this case? |
|---|---|---|
| [#29880](https://github.com/opencv/opencv/issues/29880) / [#29883](https://github.com/opencv/opencv/pull/29883) `addWeighted` null kernel crash for `CV_Bool` inputs | closed / merged 2026-09-07 | No |
| [#30153](https://github.com/opencv/opencv/issues/30153) / [#30154](https://github.com/opencv/opencv/pull/30154) `inRange` segfault for `CV_Bool`, fixed by rejecting `CV_Bool` | closed 2026-10-06 | No, but shows the maintainers' usual remedy (reject with a clear error) |
| [#29622](https://github.com/opencv/opencv/pull/29622) accept `CV_Bool` masks in connected components (and similar #29676, #29677, #29679, #29681) | merged to 5.x, July-August 2026 | No |
| [#29798](https://github.com/opencv/opencv/issues/29798) `cv::dilate` asserts on input mask of type `CV_Bool` | open | No |

Searches (GitHub, `repo:opencv/opencv`): "bitwise_not bool", "bitwise_not CV_Bool", "CV_Bool bitwise", "bool numpy array
bitwise_not python 5.x", "bitwise_not bool mask result 255", `bitwise_not` sorted by newest, `CV_Bool` sorted by newest,
"in:title bool". Web search "OpenCV 5.0 cv2.bitwise_not bool array CV_Bool wrong result 254 255": nothing relevant.
GitHub search matches words anywhere in an issue, so a report phrased differently could still exist; the owner may want to
run `is:issue bitwise_not` once more just before filing.

## Not verified

- Whether the maintainers consider logical negation or a clear error the right outcome. The draft asks for either.
- The `5.x` check used the C++ API on `opencv_core` only; the Python binding was not built from `5.x` for this function.
  The binding passes the numpy bool array as a `CV_Bool` Mat in 5.0.0, which is what the C++ check used.

---

## Draft (bug-report template)

**Title:** `cv2.bitwise_not` on a numpy bool array returns bytes 0xFE/0xFF (every element True) instead of the negation (5.0.0)

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`), where bool input is rejected
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

OpenCV 5.0.0 accepts numpy `bool` arrays (`CV_Bool`) in `bitwise_and`, `bitwise_or` and `bitwise_xor`, and these return
correct 0/1 results. `bitwise_not` accepts them too, but returns an array whose bytes are 0xFE (input True) and 0xFF
(input False). As a numpy `bool` array every element then reads as `True`, and the array contains values other than 0
and 1. 4.14.0 raises `src data type = bool is not supported` for all four functions.

Observed on 5.0.0:

```
OpenCV 5.0.0 | numpy 2.4.6
input mask            : [[True, False, True], [False, False, True]]
expected (np.invert)  : [[False, True, False], [True, True, False]]
cv2.bitwise_not dtype : bool
cv2.bitwise_not values: [[True, True, True], [True, True, True]]
cv2.bitwise_not bytes : [[254, 255, 254], [255, 255, 254]]
expected bytes        : [[0, 1, 0], [1, 1, 0]]
True count            : got 6 expected 3
bitwise_and  bytes   : [[1, 0, 0, 0]]
bitwise_or   bytes   : [[1, 1, 1, 0]]
bitwise_xor  bytes   : [[0, 1, 1, 0]]
RESULT: REPRODUCED (not a logical negation)
```

Expected: either the logical negation (`[[False, True, False], [True, True, False]]`, as `~mask` gives), or an error
saying that `CV_Bool` is not supported by `bitwise_not`, as for other functions that do not support `CV_Bool`.

The documentation of `bitwise_not` says "per-element bit-wise inversion" and lists `CV_Bool` among the mask types, but
does not say what the inversion of a `CV_Bool` array is.

The same happens on the `5.x` branch (commit 9123a3c209, `5.1.0-dev`): in C++, `cv::bitwise_not` of a `CV_Bool` Mat holding
1 0 1 0 0 1 returns bytes 254 255 254 255 255 254.

### Steps to reproduce

```python
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
```

C++ check used for the `5.x` branch (also in `repro_02_bitwise_not_bool.cpp`):

```cpp
// C++ check for repro 02: cv::bitwise_not on a CV_Bool Mat.
// Used to confirm the behaviour on the 5.x branch (opencv_core only), where the Python wheel is not available.
// Build against an OpenCV 5 tree, e.g.:  g++ -std=c++17 repro_02_bitwise_not_bool.cpp -I<opencv>/modules/core/include \
//     -I<build> -I<build>/modules/core -L<build>/lib -lopencv_core -Wl,-rpath,<build>/lib -o bool_not
#include <opencv2/core.hpp>
#include <cstdint>
#include <cstdio>

int main()
{
    printf("OpenCV %s\n", cv::getVersionString().c_str());
    uint8_t in_bytes[6] = {1, 0, 1, 0, 0, 1};
    cv::Mat src(2, 3, CV_Bool, in_bytes);
    cv::Mat dst;
    cv::bitwise_not(src, dst);
    printf("bitwise_not: type=%d (CV_Bool=%d) bytes:", dst.type(), (int)CV_Bool);
    for (int i = 0; i < 6; i++) printf(" %u", (unsigned)dst.data[i]);
    printf("   (a logical not would give 0 1 0 1 1 0)\n");

    uint8_t b_bytes[6] = {1, 1, 0, 0, 1, 0};
    cv::Mat b(2, 3, CV_Bool, b_bytes), r;
    cv::bitwise_and(src, b, r);
    printf("bitwise_and: bytes:");
    for (int i = 0; i < 6; i++) printf(" %u", (unsigned)r.data[i]);
    printf("   (expected 1 0 0 0 0 0)\n");
    return 0;
}
```

### Issue submission checklist

- [x] I report the issue, it's not a question
- [x] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution
- [x] I updated to the latest OpenCV version and the issue is still there (latest release 5.0.0, and `5.x` at 9123a3c209 in C++)
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
