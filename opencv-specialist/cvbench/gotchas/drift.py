"""Group 5, behaviour drift: results that changed between 4.14 and 5.0 without any exception.

Four checks, all in the 5.0 subset:
  * ``VideoCapture().get(prop)`` on a capture that is not open returns -1.0 (was 0.0): ``== 0`` as "unsupported" breaks;
  * ``MSER_create()`` finds 0 regions on 5.0.0 (the probe's text image: 7 on 4.14). Measured here on seven images (five
    text images, the inverted probe image, a rectangle texture): 0 regions with the defaults on ALL seven, and on the
    probe image the count falls with min_diversity: 170 at 0.0, 151 at 0.001, 51 at 0.01, 3 at 0.05, 0 at 0.1 and at the
    default 0.2. So it is not specific to text, and tuning delta / area / max_variation does not help;
  * ``warpAffine`` / ``warpPerspective`` outputs moved by up to 4 gray levels on ~3 % of the pixels: a verifier has to
    compare by correlation plus a measured bound, not by equality. The check pins the verifier's tolerance against the
    4.14 arrays recorded in ``probes/results/probe_4.14.0.npz``;
  * ``getTextSize`` changed ((155, 22) base 10 -> (136, 27) base 6 for "OpenCV 5"), and so did the glyph pixels (measured
    here: 5.0 renders Hershey text anti-aliased even with LINE_8, ink pixels span 128..255 over a gray box where 4.x
    drew a single colour): code that hard-codes text metrics is off. The check asks for a tight label box, judged by
    containment of the rendered ink, not by pixel baselines. A model that calls getTextSize passes on both versions.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail

GROUP = "drift"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="gotcha", task=task, inputs=inputs, outputs=outputs)


def _code(sig: str, body: str) -> str:
    return f"import cv2\nimport numpy as np\n\n\ndef solve({sig}):\n{body}"


# ---------------------------------------------------------------- 1. VideoCapture.get on an unopened capture

def _vc_cases():
    return [Case({"default_fps": fps}, fps) for fps in (25.0, 30.0)]


def _vc_verify(case: Case, out: dict) -> Verdict:
    fps = out["fps"]
    return PASS if fps == case.truth else fail(f"fps {fps}: an unopened capture reports no frame rate "
                                               f"(-1.0 on 5.0), so the fallback {case.truth} should have been used")


VIDEOCAPTURE_GET = Check(
    id="drift_videocapture_get", group=GROUP, probes=("VideoCapture_get_unopened",), depends_on_5_0=True,
    role="gotcha",
    gotcha="VideoCapture.get(prop) on an unopened capture returns -1.0 in 5.0, was 0.0: `if fps == 0` / `fps or default` miss it",
    spec=_spec("The video source could not be opened, so create an unopened `cv2.VideoCapture()` and read its frame "
               "rate (CAP_PROP_FPS). If it does not report a usable (positive) frame rate, use default_fps.",
               {"default_fps": IOSpec("float", "scalar", None, "fallback frame rate")},
               {"fps": IOSpec("float", "scalar", None, "the frame rate to use")}),
    make_cases=_vc_cases, verify=_vc_verify,
    reference=_code("default_fps", '    fps = cv2.VideoCapture().get(cv2.CAP_PROP_FPS)\n'
                                   '    if not np.isfinite(fps) or fps <= 0:\n        fps = default_fps\n'
                                   '    return {"fps": float(fps)}\n'),
    lint=False,
    note="lint=False: the production lint bans cv2.VideoCapture (device/file I/O). This task needs the constructor only; "
         "it opens no device and no file.",
    wrong={
        "eq_zero": Wrong("4.x: `if fps == 0: fps = default_fps` (4.14 returned 0.0 for an unsupported property)",
                         _code("default_fps", '    fps = cv2.VideoCapture().get(cv2.CAP_PROP_FPS)\n'
                                             '    if fps == 0:\n        fps = default_fps\n    return {"fps": float(fps)}\n'),
                         "verifier"),
        "falsy": Wrong("4.x: `fps = cap.get(...) or default_fps` (0.0 is falsy, -1.0 is not)",
                       _code("default_fps", '    fps = cv2.VideoCapture().get(cv2.CAP_PROP_FPS) or default_fps\n'
                                           '    return {"fps": float(fps)}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 2. MSER finds nothing with default parameters

def text_image(text: str = "MSER 123", scale: float = 2.0, thickness: int = 5, org=(10, 80), shape=(120, 300)):
    img = np.full(shape, 230, np.uint8)
    cv2.putText(img, text, org, FONT, scale, 20, thickness)
    return img


def _glyph_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, st, _ = cv2.connectedComponentsWithStats((img < 125).astype(np.uint8), connectivity=8)
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if st[i, 4] >= 30]


def _box_iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# calibrated: with min_diversity=0 the reference box set contains every glyph with IoU 1.00 on all five text images
MSER_IOU = 0.5


def _mser_cases():
    imgs = [text_image(), text_image("OPEN 4567", 1.6, 4, (10, 70), (100, 320)),
            text_image("HELLO", 1.8, 4, (10, 70), (100, 260))]
    return [Case({"img": im}, _glyph_boxes(im)) for im in imgs]


def _mser_verify(case: Case, out: dict) -> Verdict:
    boxes = arr(out, "boxes").astype(np.int64)
    if len(boxes) == 0:
        return fail("no boxes returned")
    for g in case.truth:
        best = max(_box_iou(g, tuple(b)) for b in boxes)
        if best < MSER_IOU:
            return fail(f"glyph at {g} (x, y, w, h) has no returned box with IoU >= {MSER_IOU} (best {best:.2f}); "
                        f"{len(boxes)} boxes returned")
    return PASS


_MS_TAIL = ('    regions, bboxes = mser.detectRegions(img)\n'
            '    return {"boxes": np.array(bboxes, np.int32).reshape(-1, 4)}\n')
MSER_DEFAULT = Check(
    id="drift_mser_default", group=GROUP, probes=("MSER_text_default_params", "MSER_text_min_diversity_0"),
    depends_on_5_0=True, role="gotcha",
    gotcha="cv2.MSER_create() with default parameters returns 0 regions on 5.0.0 (7 on 4.14); only min_diversity=0.0 "
           "brings them back",
    spec=_spec("Detect the text glyphs in this image with MSER (cv2.MSER_create) and return the bounding box of every "
               "region it finds. There must be a box for each glyph; extra nested boxes are fine.",
               {"img": IOSpec("uint8", "HxW", 2, "dark text on a light background")},
               {"boxes": IOSpec("int", "Nx4", 2, "(x, y, w, h) per MSER region")}),
    make_cases=_mser_cases, verify=_mser_verify,
    reference=_code("img", '    mser = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25,\n'
                           '                           min_diversity=0.0)\n' + _MS_TAIL),
    note="the reference passes min_diversity=0.0 explicitly: that is the workaround measured on 5.0.0",
    wrong={
        "default_params": Wrong("cv2.MSER_create() with the library defaults (the idiom in every 4.x tutorial)",
                                _code("img", '    mser = cv2.MSER_create()\n' + _MS_TAIL), "verifier"),
        "tuned_other_params": Wrong("tunes delta / area / variation but leaves min_diversity at its default",
                                    _code("img", '    mser = cv2.MSER_create(delta=5, min_area=60, max_area=14400,\n'
                                                 '                           max_variation=0.5)\n' + _MS_TAIL),
                                    "verifier"),
        "small_diversity": Wrong("min_diversity=0.1 instead of the default 0.2: still nothing",
                                 _code("img", '    mser = cv2.MSER_create(min_diversity=0.1)\n' + _MS_TAIL), "verifier"),
    })


# ---------------------------------------------------------------- 3. warp outputs moved by up to 4 gray levels

def texture(seed: int = 0, shape=(160, 200)) -> np.ndarray:
    """Same generator as probes/api_diff.py, so the arrays can be compared with the recorded 4.14 outputs."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 30, np.uint8)
    for _ in range(25):
        x, y = rng.integers(10, shape[1] - 30), rng.integers(10, shape[0] - 30)
        w, h = rng.integers(8, 25, size=2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(80, 255)), -1)
    return img


QUAD_SRC = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
QUAD_DST = np.float32([[5, 3], [96, 8], [102, 85], [2, 78]])

# Measured on the probes' own scene (200x160 texture, 17 degree rotation about (100, 80), the QUAD_SRC -> QUAD_DST
# homography; bilinear): 5.0.0 vs the 4.14.0 arrays recorded in probe_4.14.0.npz differ by max 4 gray levels on 3.1 % /
# 3.3 % of the pixels, Pearson correlation 0.999990 / 0.999989. Genuinely different warps on the same scene:
#   nearest 125-149 / corr 0.9908,  bicubic 35-36 / 0.9989,  angle off by 0.5 deg 140 / 0.9899,  half a pixel of shift
#   118-124 / 0.981-0.988,  border replicate 30 / 0.990,  wrong direction or centre 200+ / <= 0.54.
# So: max abs difference <= 6 (1.5x the version drift) and correlation >= 0.9999 (10x tighter than the closest genuine
# difference's distance from 1, 9x looser than the drift's). Pixel-exact equality would reject the 4.14 arrays.
WARP_MAX_ABS, WARP_MIN_CORR = 6, 0.9999


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def warp_agrees(got: np.ndarray, truth: np.ndarray) -> str | None:
    if got.shape != truth.shape:
        return f"shape {got.shape}, expected {truth.shape}"
    worst = int(np.abs(got.astype(np.int16) - truth.astype(np.int16)).max())
    corr = pearson(got, truth)
    if worst > WARP_MAX_ABS or corr < WARP_MIN_CORR:
        return f"max abs difference {worst} (allowed {WARP_MAX_ABS}), correlation {corr:.5f} (required {WARP_MIN_CORR})"
    return None


def _warp_cases():
    img = texture()
    M = cv2.getRotationMatrix2D((100.0, 80.0), 17.0, 1.0)
    H = cv2.getPerspectiveTransform(QUAD_SRC, QUAD_DST)
    truth = {"affine": cv2.warpAffine(img, M, (200, 160), flags=cv2.INTER_LINEAR),
             "perspective": cv2.warpPerspective(img, H, (200, 160), flags=cv2.INTER_LINEAR)}
    return [Case({"img": img, "M": M, "H": H}, truth)]


def _warp_verify(case: Case, out: dict) -> Verdict:
    for key in ("affine", "perspective"):
        why = warp_agrees(arr(out, key), case.truth[key])
        if why:
            return fail(f"{key}: {why}")
    return PASS


_WP_SIG = "img, M, H"
WARP_TOLERANCE = Check(
    id="drift_warp_tolerance", group=GROUP,
    probes=("warpAffine_rotate_fingerprint", "warpPerspective_fingerprint"), depends_on_5_0=True, role="gotcha",
    gotcha="5.0 warps differ from 4.14 by up to 4 gray levels on ~3 % of pixels: the verifier must compare by "
           "correlation and a measured bound, and a solution must get the geometry exactly right",
    spec=_spec("Warp the image twice with bilinear interpolation, keeping the input size and filling outside "
               "pixels with 0: once with the 2x3 affine matrix M (cv2.warpAffine convention) and once with the 3x3 "
               "homography H (cv2.warpPerspective convention).",
               {"img": IOSpec("uint8", "160x200", 2, "grayscale image, 160 rows x 200 columns"),
                "M": IOSpec("float64", "2x3", 2, "forward affine matrix"),
                "H": IOSpec("float64", "3x3", 2, "forward homography")},
               {"affine": IOSpec("uint8", "160x200", 2, "img warped by M"),
                "perspective": IOSpec("uint8", "160x200", 2, "img warped by H")}),
    make_cases=_warp_cases, verify=_warp_verify,
    reference=_code(_WP_SIG, '    h, w = img.shape\n'
                             '    return {"affine": cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR),\n'
                             '            "perspective": cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)}\n'),
    wrong={
        "nearest": Wrong("INTER_NEAREST instead of bilinear",
                         _code(_WP_SIG, '    h, w = img.shape\n'
                                        '    return {"affine": cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST),\n'
                                        '            "perspective": cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_NEAREST)}\n'),
                         "verifier"),
        "inverse_map": Wrong("WARP_INVERSE_MAP: treats the forward matrices as destination-to-source",
                             _code(_WP_SIG, '    h, w = img.shape\n    f = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP\n'
                                            '    return {"affine": cv2.warpAffine(img, M, (w, h), flags=f),\n'
                                            '            "perspective": cv2.warpPerspective(img, H, (w, h), flags=f)}\n'),
                             "verifier"),
        "dsize_swapped": Wrong("dsize=(h, w) instead of (w, h)",
                               _code(_WP_SIG, '    h, w = img.shape\n'
                                              '    return {"affine": cv2.warpAffine(img, M, (h, w)),\n'
                                              '            "perspective": cv2.warpPerspective(img, H, (h, w))}\n'),
                               "verifier"),
    })


# ---------------------------------------------------------------- 4. getTextSize and glyph metrics changed

TEXT_CANVAS = (100, 360)
TEXT_ORG = (20, 60)

# The reference box (from cv2.getTextSize) contains the rendered ink in all 180 (6 texts x 5 scales x 6 thicknesses)
# combinations measured, with per-side slack of at most 4 px left, 5 right, 13 top (the cap-height gap above the ink),
# 1 bottom; (slack - 2 px) / ink height <= 0.262 on every side (tests/test_gotchas.py re-runs this sweep). The allowance
# is 2 px + 0.40 x ink height per side (1.5x margin on the 0.262). The 4.x rule-of-thumb metrics are 21-51 px too wide
# on the check's own cases, i.e. 1.6-3.7x the allowance.
TB_SLACK_PX, TB_SLACK_FRAC = 2.0, 0.40


def ink(text: str, scale: float, thickness: int):
    canvas = np.zeros(TEXT_CANVAS, np.uint8)
    cv2.putText(canvas, text, TEXT_ORG, FONT, scale, 255, thickness)
    ys, xs = np.nonzero(canvas)
    return canvas > 0, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _tb_cases():
    out = []
    for text, scale, thickness in (("OpenCV 5", 1.0, 2), ("gjpqy yaq", 1.2, 2), ("Quality 42", 0.8, 1)):
        out.append(Case({"img": np.zeros(TEXT_CANVAS, np.uint8), "text": text, "scale": scale, "thickness": thickness,
                         "x": TEXT_ORG[0], "y": TEXT_ORG[1]}, ink(text, scale, thickness)))
    return out


def text_box_problem(truth, out_img: np.ndarray, box: np.ndarray) -> str | None:
    mask, (ix0, iy0, ix1, iy1) = truth
    if out_img.shape != mask.shape:
        return f"img has shape {out_img.shape}"
    x0, y0, x1, y1 = (int(v) for v in box)
    if not (x0 <= ix0 and y0 <= iy0 and x1 >= ix1 and y1 >= iy1):
        return f"box {[x0, y0, x1, y1]} does not contain the text ink {[ix0, iy0, ix1, iy1]}"
    slack = TB_SLACK_PX + TB_SLACK_FRAC * (iy1 - iy0 + 1)
    sides = {"left": ix0 - x0, "top": iy0 - y0, "right": x1 - ix1, "bottom": y1 - iy1}
    loose = {k: v for k, v in sides.items() if v > slack}
    if loose:
        return f"box is loose by {loose} px (allowed {slack:.1f} per side)"
    # 5.0 renders Hershey text with coverage-blended (anti-aliased) pixels even with LINE_8, so on a gray 128 box the
    # glyph pixels are 128..255, not exactly 255: require every ink pixel to be at least the box gray and nearly all of
    # them brighter, and the full-white core to exist.
    if out_img[mask].min() < 128 or (out_img[mask] > 128).mean() < 0.9 or out_img.max() != 255:
        return "the text is not drawn in white (255) with the requested font, scale and thickness"
    inside = np.zeros(mask.shape, bool)
    inside[y0:y1 + 1, x0:x1 + 1] = True
    background = inside & ~mask
    if (out_img[background] != 128).any() or (out_img[~inside] != 0).any():
        return "the box is not filled with gray 128 behind the text (or something was drawn outside it)"
    return None


def _tb_verify(case: Case, out: dict) -> Verdict:
    why = text_box_problem(case.truth, arr(out, "img"), arr(out, "box"))
    return fail(why) if why else PASS


_TB_SIG = "img, text, scale, thickness, x, y"
_TB_TAIL = ('    cv2.rectangle(out, (x0, y0), (x1, y1), 128, -1)\n'
            '    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)\n'
            '    return {"img": out, "box": np.array([x0, y0, x1, y1])}\n')
GETTEXTSIZE_BOX = Check(
    id="drift_gettextsize_box", group=GROUP, probes=("getTextSize", "putText_hershey_pixels"), depends_on_5_0=True,
    role="gotcha",
    gotcha="getTextSize / Hershey glyph metrics changed in 5.0 ((155, 22) base 10 -> (136, 27) base 6): hard-coded "
           "4.x metrics give a loose box, and ignoring the baseline clips descenders",
    spec=_spec("Draw a label: white (255) text in cv2.FONT_HERSHEY_SIMPLEX with its bottom-left corner (baseline start) "
               "at (x, y), on a filled gray (128) rectangle that fits the text snugly and still contains its "
               "descenders. Return the image and the rectangle's corners.",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale canvas, all zeros"),
                "text": IOSpec("str", "scalar", None, "the label"),
                "scale": IOSpec("float", "scalar", None, "font scale"),
                "thickness": IOSpec("int", "scalar", None, "stroke thickness in px"),
                "x": IOSpec("int", "scalar", None, "text origin, bottom-left"),
                "y": IOSpec("int", "scalar", None, "text origin, bottom-left")},
               {"img": IOSpec("uint8", "HxW", 2, "canvas with the rectangle and the text"),
                "box": IOSpec("int", "4", 1, "x0, y0, x1, y1 of the filled rectangle")}),
    make_cases=_tb_cases, verify=_tb_verify,
    reference=_code(_TB_SIG, '    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)\n'
                             '    out = img.copy()\n    x0, y0, x1, y1 = x, y - h, x + w, y + baseline\n' + _TB_TAIL),
    note="a model that calls getTextSize passes on 4.14 and 5.0 alike: this check only catches hard-coded metrics",
    wrong={
        "hardcoded_4x_metrics": Wrong("4.x rule of thumb for the box: width 19.4 px per char, height 22 and baseline 10, times scale",
                                      _code(_TB_SIG, '    w, h, baseline = int(19.4 * len(text) * scale), int(22 * scale), int(10 * scale)\n'
                                                     '    out = img.copy()\n    x0, y0, x1, y1 = x, y - h, x + w, y + baseline\n' + _TB_TAIL),
                                      "verifier"),
        "ignores_baseline": Wrong("box height from getTextSize's h only: descenders (g, j, p, q, y) poke out below",
                                  _code(_TB_SIG, '    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)\n'
                                                 '    out = img.copy()\n    x0, y0, x1, y1 = x, y - h, x + w, y\n' + _TB_TAIL),
                                  "verifier"),
        "top_left_anchor": Wrong("treats (x, y) as the top-left corner of the text",
                                 _code(_TB_SIG, '    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)\n'
                                                '    out = img.copy()\n    x0, y0, x1, y1 = x, y, x + w, y + h + baseline\n' + _TB_TAIL),
                                 "verifier"),
    })


CHECKS = [VIDEOCAPTURE_GET, MSER_DEFAULT, WARP_TOLERANCE, GETTEXTSIZE_BOX]
