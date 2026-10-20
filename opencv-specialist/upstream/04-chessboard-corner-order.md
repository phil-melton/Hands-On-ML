# 04. findChessboardCorners returns the corners starting at the bottom-right when the top-left square is white

| | |
|---|---|
| Classification | **Expected, long-standing behaviour; documentation gap** (now addressed in the `5.x` docs). Not a 5.0 change: identical on 4.14.0 |
| Recommendation | **Do not file.** Discussed in open issue [#22083](https://github.com/opencv/opencv/issues/22083); docs expanded by merged PR [#30208](https://github.com/opencv/opencv/pull/30208) (2026-10-12, `5.x`, not in 5.0.0) |
| Repro | [`repro_04_chessboard_corner_order.py`](repro_04_chessboard_corner_order.py) |
| Verified on | opencv-python-headless 5.0.0.93 and 4.14.0.94 |

## What I found

- Board of 7x6 inner corners (8x7 squares, white margin), rendered with the top-left square black and with it white (the
  same board rotated by 180 degrees). `findChessboardCorners` returns `corners[0]` at the top-left inner corner for the
  black case and at the **bottom-right** inner corner for the white case. Output is identical on 4.14.0 and 5.0.0.
  `findChessboardCornersSB` returns the opposite order in both cases, also identical on both versions.
- I swept ten patterns (`(7,6) (6,4) (5,5) (6,5) (5,6) (8,6) (9,6) (4,3) (3,3) (4,4)`, both colourings) with
  `findChessboardCorners` on both versions. With a black top-left square all ten came out in image raster order. With a
  white top-left square, the five patterns whose width and height have different parity (`(7,6) (6,5) (5,6) (9,6) (4,3)`)
  and `(3,3)` came out reversed; `(6,4) (5,5) (8,6) (4,4)` did not. For every pattern the 4.14.0 and 5.0.0
  `findChessboardCorners` results agree. (The only version difference in the sweep is `findChessboardCornersSB` on
  `(6,4)` with a white top-left square: 4.14.0 raster order, 5.0.0 reversed.)
- This matches how the detector works, as far as I read the source: it builds the grid from dark squares, and the end
  that is returned first is tied to the board, not to the image. The two places in the 5.0.0 source that reorder the
  result are a swap for patterns with even width and even height that makes the first corner the one with the smaller y
  ([calibinit.cpp L837-L848](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/calibinit.cpp#L837-L848))
  and a handedness fix
  ([L1755](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/calibinit.cpp#L1755)); I found no rule that
  makes the first corner the image top-left. I did not read the whole file.
- 5.0.0 documentation says only: "True if all of the corners are found and placed in a certain order (row by row,
  left to right in every row)"
  ([objdetect.hpp L357](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/include/opencv2/objdetect.hpp#L357)).
  It does not say that "left to right" refers to the board and not to the image.
- `5.x` documentation (merged #30208) adds a note that says the order is repeatable across views only when width and
  height of `patternSize` differ in parity, is ambiguous up to a 180 degree rotation when they have the same parity,
  and up to 90 degrees when they are equal, and that "which one you get depends on how the board happens to lie in the image".

## Duplicate search

| Item | State | Covers this case? |
|---|---|---|
| [#22083](https://github.com/opencv/opencv/issues/22083) Difference in corners between findChessboardCorners() and findChessboardCornersSB() | open (2022), 10 comments | Yes for the order difference and the missing normalisation; no 5.0-specific claim |
| [#30208](https://github.com/opencv/opencv/pull/30208) calib document fix | merged to 5.x 2026-10-12 | Yes, it is the documentation answer |
| [#15712](https://github.com/opencv/opencv/issues/15712) Incorrect chessboard corners returned by findChessboardCornersSB | closed (2019) | Related (SB), not this |
| [#22288](https://github.com/opencv/opencv/issues/22288) Subpixel coordinates of findChessboardCorners are unintuitive | open | No |

Searches (GitHub, `repo:opencv/opencv`): "findChessboardCorners order reversed", "findChessboardCorners corners order
orientation", "findChessboardCorners first corner black white square", "chessboard origin ambiguity orientation",
"findChessboardCornersSB order", "chessboard corners upside down order flipped", "findChessboardCorners reverse order white".
Web search for the same topic found only forum threads (answers.opencv.org "Order of corners in findChessboardCorners ?",
which I could not open: HTTP 403) and library code that flips the order itself.

## Consequence for this project

Nothing to change. The `chessboard_corners` family generates only boards whose square counts have opposite parity
(`_board_pairs()`), which is the case where the origin is unique, so the 300/300 measurement is what the sweep above predicts.
Its contract fixes the order in the board frame and a "reversed" mutant covers the trap. The property belongs to the
board's dark-square layout and holds on 4.14.0 as well; it is not a 5.0 drift.

---

## Draft (not recommended for filing; if the owner wants to add to #22083, the useful content is the table above)

**Title:** findChessboardCorners: corner order starts at the bottom-right when the top-left square is white (documentation does not say the order is relative to the board)

### System Information

OpenCV python version: 5.0.0.93 (`opencv-python-headless` wheel from PyPI; `cv2.__version__` is `5.0.0`)
Compared against: 4.14.0.94 (`opencv-python-headless`)
Operating System / Platform: Ubuntu 24.04.5 LTS under WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), x86_64
Python version: 3.12.3
NumPy version: 2.4.6

### Detailed description

The documentation says the corners are "placed in a certain order (row by row, left to right in every row)". For a 7x6
board whose top-left square is white, `findChessboardCorners` returns the corner list starting at the bottom-right inner
corner. The same board with a black top-left square (the board rotated by 180 degrees) is returned top-left first.
Behaviour is identical on 4.14.0.

```
OpenCV 5.0.0 | numpy 2.4.6

board 7x6 inner corners, top-left square black
  findChessboardCorners  : corners[0]=(69.5, 69.5) corners[-1]=(249.5, 219.5)  -> image order: first corner = top-left
  findChessboardCornersSB: corners[0]=(249.5, 219.5) corners[-1]=(69.5, 69.5)  -> image order: first corner = BOTTOM-RIGHT (reversed)

board 7x6 inner corners, top-left square WHITE
  findChessboardCorners  : corners[0]=(249.5, 219.5) corners[-1]=(69.5, 69.5)  -> image order: first corner = BOTTOM-RIGHT (reversed)
  findChessboardCornersSB: corners[0]=(69.5, 69.5) corners[-1]=(249.5, 219.5)  -> image order: first corner = top-left

The documented return value says the corners are 'placed in a certain order (row by row, left to right
in every row)'. The two boards above differ only by a 180 degree rotation.
RESULT: findChessboardCorners starts at the bottom-right when the top-left square is white
```

This appears to be intended (the origin is tied to the dark squares), so this is a documentation request: say that the
order is relative to the board, not to the image. The `5.x` docs (#30208) already do.

### Steps to reproduce

```python
"""Repro 04: findChessboardCorners returns its corners starting at the bottom-right when the board's top-left square is white.

Needs only numpy and cv2. Runs in a second or two. Renders a 7x6 inner-corner board (8x7 squares, white
border) twice: top-left square black, and top-left square white (the same board turned by 180 degrees).
Prints where corners[0] and corners[-1] are, for findChessboardCorners and findChessboardCornersSB.
The same output is produced by 4.14.0 and 5.0.0 for findChessboardCorners. Exit code 1 if the order flips.
"""
import sys

import cv2
import numpy as np

print("OpenCV", cv2.__version__, "| numpy", np.__version__)


def board(cols, rows, white_top_left, sq=30, margin=40):
    """(cols+1) x (rows+1) squares -> cols x rows inner corners, with a white margin of `margin` px."""
    img = np.full(((rows + 1) * sq + 2 * margin, (cols + 1) * sq + 2 * margin), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if ((r + c) % 2 == 0) != white_top_left:          # square is black
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    # true inner-corner positions, row by row, left to right (pixel-centre coordinates)
    truth = np.array([[margin + (c + 1) * sq - 0.5, margin + (r + 1) * sq - 0.5]
                      for r in range(rows) for c in range(cols)], np.float32)
    return img, truth


def describe(corners, truth):
    c = np.asarray(corners, np.float32).reshape(-1, 2)
    if np.abs(c - truth).max() < 2:
        return "image order: first corner = top-left"
    if np.abs(c - truth[::-1]).max() < 2:
        return "image order: first corner = BOTTOM-RIGHT (reversed)"
    return "other"


cols, rows = 7, 6
flipped = False
for white_tl in (False, True):
    img, truth = board(cols, rows, white_tl)
    print("\nboard %dx%d inner corners, top-left square %s" % (cols, rows, "WHITE" if white_tl else "black"))
    for name, fn in (("findChessboardCorners  ", lambda im: cv2.findChessboardCorners(im, (cols, rows))),
                     ("findChessboardCornersSB", lambda im: cv2.findChessboardCornersSB(im, (cols, rows)))):
        ok, corners = fn(img)
        if not ok:
            print("  %s: not found" % name)
            continue
        c = corners.reshape(-1, 2)
        verdict = describe(c, truth)
        print("  %s: corners[0]=(%.1f, %.1f) corners[-1]=(%.1f, %.1f)  -> %s"
              % (name, c[0, 0], c[0, 1], c[-1, 0], c[-1, 1], verdict))
        if name.strip() == "findChessboardCorners" and white_tl and "BOTTOM-RIGHT" in verdict:
            flipped = True

print("\nThe documented return value says the corners are 'placed in a certain order (row by row, left to right")
print("in every row)'. The two boards above differ only by a 180 degree rotation.")
print("RESULT:", "findChessboardCorners starts at the bottom-right when the top-left square is white"
      if flipped else "not reproduced")
sys.exit(1 if flipped else 0)
```

### Issue submission checklist

- [x] I report the issue, it's not a question
- [ ] I checked the problem with documentation, FAQ, open issues, forum.opencv.org, Stack Overflow, etc and have not found any solution (**false: #22083 and #30208 exist**)
- [ ] I updated to the latest OpenCV version and the issue is still there (**the `5.x` docs already describe it**)
- [x] There is reproducer code and related data files (videos, images, onnx, etc)
