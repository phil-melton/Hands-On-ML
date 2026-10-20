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
