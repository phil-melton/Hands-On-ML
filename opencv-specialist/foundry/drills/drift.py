"""Drills on behaviour that changed between 4.14 and 5.0 without raising an exception.

Measured (PLAN.md section 0, item 3, and probes/results/diff_4.14_vs_5.0.md):
  * ``getTextSize`` changed ((155, 22) base 10 -> (136, 27) base 6 for one string): code that hard-codes 4.x text metrics
    breaks, code that asks getTextSize does not. The drill asks for greedy word wrapping to a pixel width, which depends on the
    widths getTextSize reports (measured here: a line's width is the sum of its words' widths plus the spaces to within 2 px, so
    the verifier measures whole lines, as a model would).
  * ``INTER_NEAREST_EXACT``: its mapping changed and is not the wiki's "same as INTER_NEAREST" (measured here on the 1,890
    size pairs 5..39 -> 5..59: it equals INTER_NEAREST on 75, the pixel-centre formula floor((i + 0.5) * n_in / n_out) on 1,586,
    and neither the centre nor the legacy formula on 302). A task that names the flag is checked against the flag itself.
  * ``VideoCapture.get`` on a property the capture does not report returns -1.0 (was 0.0): measured on an unopened capture,
    all eight properties tried return -1.0. ``lint=False``: the production lint bans ``cv2.VideoCapture`` (device and file I/O),
    and this drill needs the constructor only; it opens no device and no file. The consequence for the foundry: a verified
    answer to this drill cannot pass the lint gate, so it can never be Tier A; use it only as a repair pair or a dev probe.
  * ``MSER_create()`` with default parameters finds no regions on 5.0.0 (opencv/opencv issue #29652, fixed on the 5.x branch by
    PR #29653, not in any release yet): ``min_diversity=0.0`` brings them back. Measured here on 20 scenes of dark discs: 0.3
    boxes per scene with the defaults, 87 with min_diversity 0, 2.3 with 0.1. If the pinned wheel is ever replaced by a release
    that carries the fix, the ``default_params`` wrong variant starts to pass and tests/test_drills.py fails on purpose.

  * ``putText`` with a Hershey font ignores ``lineType`` in 5.0 and always anti-aliases (13 % of the glyph pixels differ from 4.x;
    measured here: the ink pixels span 1..255 on a black canvas, where 4.x drew only the colour you asked for): code that finds
    the text pixels with ``== 255`` loses the soft edge.
  * ``cv2.FontFace`` is new in 5.0 (the probe "has cv2.FontFace" flips false -> true): ``putText(img, text, org, color, fface,
    size)`` and ``getTextSize(imgsize, text, org, fface, size)`` take the colour before the font and return a ``Rect``. No
    model trained before June 2026 has seen it. Measured quirk (5.0.0): after any Hershey text call (``getTextSize`` or ``putText``)
    in a process, a FontFace ``putText`` at a different size than the previous FontFace call renders differently (1,755 of 36,000
    values for size 18 after size 24); Hershey text is unaffected by earlier FontFace calls. The drill's truth is therefore
    rendered in a clean sandbox process (``_ff_truth``), never in the test process.

The wrong variants reproduce the 4.x habit (``== 0`` as the unsupported marker, hard-coded text metrics, INTER_NEAREST for
"exact", the library's default MSER parameters).
"""
from __future__ import annotations

import functools
import math

import cv2
import numpy as np
from sandbox.run import run_solution

from .common import PASS, SEED_BASE, Case, Check, IOSpec, Registry, Verdict, Wrong, arr, code, fail, match_points, spec

GROUP = "drift"
R = Registry()
S = SEED_BASE + 5_000  # this module's seeds: 66_000 .. 66_999
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- 1. getTextSize: wrapping a caption to a pixel width

def text_width(line: str, scale: float, thickness: int) -> int:
    return cv2.getTextSize(line, FONT, scale, thickness)[0][0]


_SENTENCES = (
    "the quick brown fox jumps over the lazy dog near the old river bank at dawn",
    "calibration of the stereo rig failed twice before the technician replaced the loose lens mount",
    "bright stripes appear whenever the projector and the shutter are out of phase by a quarter turn",
    "Hough accumulators vote for every line through every edge pixel so thin lines can still win",
)


def wrap_problem(text: str, max_width: int, scale: float, thickness: int, lines) -> str | None:
    lines = [str(x) for x in lines]
    words = text.split()
    if " ".join(lines).split() != words or any(not ln.strip() for ln in lines):
        return "the lines do not contain exactly the words of the text, in order"
    for i, ln in enumerate(lines):
        if len(ln.split()) > 1 and text_width(ln, scale, thickness) > max_width:
            return f"line {i + 1} {ln!r} is {text_width(ln, scale, thickness)} px wide, the limit is {max_width}"
    for i in range(len(lines) - 1):
        grown = lines[i] + " " + lines[i + 1].split()[0]
        if text_width(grown, scale, thickness) <= max_width:
            return f"line {i + 1} could take the next word ({text_width(grown, scale, thickness)} px <= {max_width}): wrapping is greedy"
    return None


def _cw_cases():
    out = []
    for text, width, scale, th in ((_SENTENCES[0], 260, 0.7, 1), (_SENTENCES[1], 330, 0.8, 1), (_SENTENCES[2], 240, 0.6, 1),
                                   (_SENTENCES[3], 300, 1.0, 2)):
        out.append(Case({"text": text, "max_width": width, "scale": scale, "thickness": th}, None))
    return out


def _cw_verify(case: Case, out: dict) -> Verdict:
    i = case.inputs
    why = wrap_problem(i["text"], i["max_width"], i["scale"], i["thickness"], arr(out, "lines").tolist())
    return fail(why) if why else PASS


_CW_SIG = "text, max_width, scale, thickness"
R.add(Check(
    id="drill_caption_wrap", group=GROUP, probes=("getTextSize",), depends_on_5_0=True, role="gotcha",
    gotcha="getTextSize's metrics changed in 5.0: wrapping to a pixel width with hard-coded 4.x character widths (or a "
           "characters-per-line guess) gives lines that are too short or too long",
    spec=spec("Wrap a caption for display. Split `text` at its single spaces into lines for cv2.putText with "
              "cv2.FONT_HERSHEY_SIMPLEX at the given font scale and thickness, as greedily as possible: put as many words as fit "
              "on a line, where a line fits when its rendered width is at most max_width pixels. A single word wider than "
              "max_width gets a line to itself.",
              {"text": IOSpec("str", "scalar", None, "words separated by single spaces"),
               "max_width": IOSpec("int", "scalar", None, "px"),
               "scale": IOSpec("float", "scalar", None, "font scale"),
               "thickness": IOSpec("int", "scalar", None, "stroke thickness in px")},
              {"lines": IOSpec("any", "L", 1, "the wrapped lines, top to bottom; joining them with single spaces gives "
                                              "back text")}),
    make_cases=_cw_cases, verify=_cw_verify,
    reference=code(_CW_SIG, '''\
    def width(s):
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]

    lines, current = [], ""
    for word in text.split():
        trial = word if not current else current + " " + word
        if current and width(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return {"lines": lines}
'''),
    wrong={
        "hardcoded_metrics": Wrong("4.x rule of thumb: 19.4 px per character at scale 1.0, no call to getTextSize",
                                   code(_CW_SIG, '''\
    def width(s):
        return int(19.4 * scale * len(s))

    lines, current = [], ""
    for word in text.split():
        trial = word if not current else current + " " + word
        if current and width(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return {"lines": lines}
'''), "verifier"),
        "characters_per_line": Wrong("textwrap.wrap with max_width // 12 characters per line: ignores the font",
                                     code(_CW_SIG, '    return {"lines": textwrap.wrap(text, max(1, max_width // 12))}\n',
                                          extra_imports="import textwrap\n"), "verifier"),
        "words_without_spaces": Wrong("measures each word alone and adds the widths: the spaces are never counted",
                                      code(_CW_SIG, '''\
    def width(s):
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]

    lines, current, used = [], [], 0
    for word in text.split():
        w = width(word)
        if current and used + w > max_width:
            lines.append(" ".join(current))
            current, used = [], 0
        current.append(word)
        used += w
    lines.append(" ".join(current))
    return {"lines": lines}
'''), "verifier"),
    }),
    "Greedy word wrap for putText: break `text` into lines so that each line's rendered width in cv2.FONT_HERSHEY_SIMPLEX "
    "(given scale and thickness) is at most max_width px, with as many words per line as fit. Return the list of lines.",
    "A caption has to fit a column max_width pixels wide when drawn with cv2.FONT_HERSHEY_SIMPLEX at the given scale and "
    "thickness. Break it at the spaces into lines, filling each line with as many words as the width allows; an "
    "over-long single word keeps its own line.")


# ---------------------------------------------------------------- 2. INTER_NEAREST_EXACT

# size pairs (in -> out) on which INTER_NEAREST_EXACT is neither INTER_NEAREST nor floor((i + 0.5) * n_in / n_out) nor
# floor(i * n_in / n_out) in 5.0.0: 426 pairs in 12..40 -> 15..89; these are among the ones that differ from the centre formula
# in 2-4 places.
EXACT_PAIRS = (((12, 21), (14, 21)), ((12, 42), (12, 33)), ((12, 18), (12, 22)), ((14, 21), (12, 21)))


def label_map(seed: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # one random class id per pixel: a source index that is off by one then shows up in the output
    return rng.integers(0, 8, (h, w), dtype=np.uint8)


def _le_cases():
    out = []
    for i, (rows, cols) in enumerate(EXACT_PAIRS):
        lm = label_map(S + 21 + i, rows[0], cols[0])
        truth = cv2.resize(lm, (cols[1], rows[1]), interpolation=cv2.INTER_NEAREST_EXACT)
        out.append(Case({"labels": lm, "out_h": rows[1], "out_w": cols[1]}, truth))
    return out


def _le_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "resized")
    if got.shape != case.truth.shape:
        return fail(f"resized has shape {got.shape}, expected {case.truth.shape}")
    diff = int((got != case.truth).sum())
    return fail(f"{diff} of {got.size} labels differ from cv2.resize(..., interpolation=cv2.INTER_NEAREST_EXACT)") if diff else PASS


_LE_SIG = "labels, out_h, out_w"
R.add(Check(
    id="drill_label_map_resize", group=GROUP, probes=("resize_nearest_exact_convention_sweep",), depends_on_5_0=True,
    role="gotcha",
    gotcha="INTER_NEAREST_EXACT's index mapping changed in 5.0 and differs from both INTER_NEAREST and the pixel-centre "
           "formula; dsize is (width, height)",
    spec=spec("Resize the segmentation label map to out_h rows and out_w columns with OpenCV's exact nearest-neighbour "
              "interpolation (cv2.INTER_NEAREST_EXACT), so that no new label values appear.",
              {"labels": IOSpec("uint8", "HxW", 2, "class ids 0..7"),
               "out_h": IOSpec("int", "scalar", None, "output rows"),
               "out_w": IOSpec("int", "scalar", None, "output columns")},
              {"resized": IOSpec("uint8", "out_hxout_w", 2, "the resized label map")}),
    make_cases=_le_cases, verify=_le_verify,
    reference=code(_LE_SIG, '    return {"resized": cv2.resize(labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST_EXACT)}\n'),
    wrong={
        "plain_nearest": Wrong("INTER_NEAREST: the wiki says it is now the same mapping; it is not",
                               code(_LE_SIG, '    return {"resized": cv2.resize(labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST)}\n'),
                               "verifier"),
        "numpy_centre_formula": Wrong("hand-rolled floor((i + 0.5) * n_in / n_out) indexing",
                                      code(_LE_SIG, '''\
    h, w = labels.shape
    rows = np.minimum(((np.arange(out_h) + 0.5) * h / out_h).astype(int), h - 1)
    cols = np.minimum(((np.arange(out_w) + 0.5) * w / out_w).astype(int), w - 1)
    return {"resized": labels[rows][:, cols]}
'''), "verifier"),
        "linear": Wrong("default INTER_LINEAR: invents label values between classes",
                        code(_LE_SIG, '    return {"resized": cv2.resize(labels, (out_w, out_h))}\n'), "verifier"),
        "dsize_swapped": Wrong("dsize = (out_h, out_w)",
                               code(_LE_SIG, '    return {"resized": cv2.resize(labels, (out_h, out_w), interpolation=cv2.INTER_NEAREST_EXACT)}\n'),
                               "verifier"),
    }),
    "Rescale a uint8 class-id image to out_h x out_w using cv2.resize with the INTER_NEAREST_EXACT flag; the output must "
    "only contain ids that were in the input.",
    "Resize the label image to the requested height and width with nearest-neighbour sampling, specifically OpenCV's "
    "INTER_NEAREST_EXACT mode, keeping the class ids intact.")


# ---------------------------------------------------------------- 3. VideoCapture.get on a property the capture does not report

PROPS = {"width": cv2.CAP_PROP_FRAME_WIDTH, "height": cv2.CAP_PROP_FRAME_HEIGHT, "fps": cv2.CAP_PROP_FPS,
         "frames": cv2.CAP_PROP_FRAME_COUNT, "pos": cv2.CAP_PROP_POS_FRAMES, "brightness": cv2.CAP_PROP_BRIGHTNESS,
         "fourcc": cv2.CAP_PROP_FOURCC, "buffer": cv2.CAP_PROP_BUFFERSIZE}


def _cp_cases():
    out = []
    for names in (("width", "height", "fps"), ("frames", "fps", "brightness", "fourcc"), ("pos", "buffer"),
                  ("width", "height", "fps", "frames", "pos")):
        out.append(Case({"props": np.array([PROPS[n] for n in names], np.int32)}, len(names)))
    return out


def _cp_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "values").astype(float)
    if got.shape != (case.truth,):
        return fail(f"values has shape {got.shape}, expected ({case.truth},)")
    if not np.isnan(got).all():
        return fail(f"values {got.tolist()}: an unopened capture reports no property, so every entry should be NaN")
    return PASS


R.add(Check(
    id="drill_capture_properties", group=GROUP, probes=("VideoCapture_get_unopened",), depends_on_5_0=True, role="gotcha",
    gotcha="VideoCapture.get(prop) returns -1.0 in 5.0 for a property the capture does not report (was 0.0): `v if v != 0 else "
           "nan` and `v or nan` pass -1.0 through",
    spec=spec("Report the properties of a camera that could not be opened. Create an unopened cv2.VideoCapture() and, for "
              "each property id in `props`, read it with get(); a property counts as reported only when the value is a "
              "positive number. Return the reported values, with NaN for every property that is not reported.",
              {"props": IOSpec("int32", "P", 1, "cv2.CAP_PROP_* ids")},
              {"values": IOSpec("float", "P", 1, "the value of each property, NaN where not reported")}),
    make_cases=_cp_cases, verify=_cp_verify,
    reference=code("props", '''\
    cap = cv2.VideoCapture()
    values = []
    for prop in props:
        v = cap.get(int(prop))
        values.append(v if np.isfinite(v) and v > 0 else np.nan)
    return {"values": np.array(values)}
'''),
    lint=False,
    note="lint=False: the production lint bans cv2.VideoCapture (device/file I/O). The task needs the constructor only; "
         "it opens no device and no file. An unopened capture reports nothing, so a constant NaN array of the right length "
         "also passes: the cases only vary its length. A verified answer cannot be Tier A (the lint gate rejects it).",
    wrong={
        "eq_zero": Wrong("4.x: `v if v != 0 else nan` (4.14 returned 0.0 for an unsupported property)",
                         code("props", '    cap = cv2.VideoCapture()\n    values = []\n    for prop in props:\n'
                                       '        v = cap.get(int(prop))\n        values.append(v if v != 0 else np.nan)\n'
                                       '    return {"values": np.array(values)}\n'), "verifier"),
        "falsy": Wrong("4.x: `cap.get(prop) or nan` (0.0 is falsy, -1.0 is not)",
                       code("props", '    cap = cv2.VideoCapture()\n'
                                     '    return {"values": np.array([cap.get(int(p)) or np.nan for p in props])}\n'), "verifier"),
        "raw_values": Wrong("returns whatever get() says",
                            code("props", '    cap = cv2.VideoCapture()\n'
                                          '    return {"values": np.array([cap.get(int(p)) for p in props])}\n'), "verifier"),
    }),
    "The camera failed to open. Make an unopened cv2.VideoCapture(), query each property id in `props` with get(), and "
    "return the values that are positive, using NaN for the ones that are not.",
    "Build a property report for a capture device that is not open: ask a bare cv2.VideoCapture() for every id in `props` "
    "and keep a value only if it is a positive number; otherwise put NaN.")


# ---------------------------------------------------------------- 4. MSER with its default parameters

def bubble_scene(seed: int, k: int | None = None, H: int = 160, W: int = 240):
    """Dark bubbles on a bright, unevenly lit liquid. Truth: the bubble centres, sorted by x."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 7)) if k is None else k
    discs = []
    while len(discs) < k:
        r = int(rng.integers(9, 21))
        x, y = int(rng.integers(r + 8, W - r - 8)), int(rng.integers(r + 8, H - r - 8))
        if all(math.hypot(x - a, y - b) >= r + s + 16 for a, b, s in discs):
            discs.append((x, y, r))
    yy, xx = np.mgrid[0:H, 0:W]
    img = 205 + 12 * np.sin(xx / 40.0) + 8 * np.cos(yy / 30.0)
    img = img.astype(np.float32)
    for x, y, r in discs:
        cv2.circle(img, (x, y), r, 70, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.array(sorted((x, y) for x, y, r in discs), np.float64)


# calibrated on 150 scenes (3-6 bubbles) x 4 MSER settings with min_diversity <= 0.01 (delta 5..8, min_area 60 default): the greedy
# merge at 6 px recovers exactly the manufactured number of bubbles in all 600 runs and the merged centres are within 0.71 px of
# the bubble centres (precision of the raw boxes 1.0 on 20 scenes); with the defaults 0.3 boxes are returned per scene. 3 px
# keeps a 4x margin.
BUB_TOL = 3.0

_MS_REF = code("gray", '''\
    mser = cv2.MSER_create(min_diversity=0.0)
    regions, boxes = mser.detectRegions(gray)
    centres = []
    for x, y, w, h in np.asarray(boxes).reshape(-1, 4):
        c = np.array([x + (w - 1) / 2.0, y + (h - 1) / 2.0])
        if not any(np.linalg.norm(c - p) < 6.0 for p in centres):
            centres.append(c)
    centres = np.array(sorted(centres, key=lambda p: p[0])).reshape(-1, 2)
    return {"count": len(centres), "centers": centres}
''')


def _bc_cases():
    return [Case({"gray": g}, c) for g, c in (bubble_scene(s, k) for s, k in ((S + 31, 4), (S + 32, 6), (S + 33, 3), (S + 34, None)))]


def _bc_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "centers").astype(float)
    if out["count"] != len(case.truth):
        return fail(f"count {out['count']}, there are {len(case.truth)} bubbles")
    if got.ndim != 2 or got.shape != case.truth.shape:
        return fail(f"centers have shape {got.shape}, expected {case.truth.shape}")
    why = match_points(case.truth, got, BUB_TOL)
    return fail(why) if why else PASS


_MS_TAIL = '''\
    regions, boxes = mser.detectRegions(gray)
    centres = []
    for x, y, w, h in np.asarray(boxes).reshape(-1, 4):
        c = np.array([x + (w - 1) / 2.0, y + (h - 1) / 2.0])
        if not any(np.linalg.norm(c - p) < 6.0 for p in centres):
            centres.append(c)
    centres = np.array(sorted(centres, key=lambda p: p[0])).reshape(-1, 2)
    return {"count": len(centres), "centers": centres}
'''
R.add(Check(
    id="drill_bubble_count_mser", group=GROUP,
    probes=("MSER_detectRegions", "MSER_text_default_params", "MSER_text_min_diversity_0"),
    depends_on_5_0=True, role="gotcha",
    gotcha="cv2.MSER_create() with default parameters returns no regions on 5.0.0 (upstream issue #29652, fixed on the 5.x "
           "branch, not in a release); min_diversity=0.0 brings them back, tuning delta / area / variation does not",
    spec=spec("Count the dark bubbles in the image with MSER (cv2.MSER_create). MSER reports nested regions for one "
              "bubble: merge regions whose box centres are less than 6 px apart, and return one centre per bubble, "
              "sorted by x, with their number.",
              {"gray": IOSpec("uint8", "HxW", 2, "grayscale image: dark round bubbles on a bright, unevenly lit liquid; the "
                                                 "bubbles are at least 16 px apart")},
              {"count": IOSpec("int", "scalar", None, "number of bubbles"),
               "centers": IOSpec("float", "Kx2", 2, "(x, y) centre of each bubble, increasing x")}),
    make_cases=_bc_cases, verify=_bc_verify, reference=_MS_REF,
    note="the reference passes min_diversity=0.0 explicitly: the workaround measured on 5.0.0",
    wrong={
        "default_params": Wrong("cv2.MSER_create() with the library defaults (the idiom in every 4.x tutorial)",
                                code("gray", '    mser = cv2.MSER_create()\n' + _MS_TAIL), "verifier"),
        "tuned_other_params": Wrong("tunes delta / area / variation but leaves min_diversity at its default",
                                    code("gray", '    mser = cv2.MSER_create(delta=5, min_area=100, max_area=14400, max_variation=0.5)\n' + _MS_TAIL),
                                    "verifier"),
        "small_diversity": Wrong("min_diversity=0.1 instead of the default 0.2: still almost nothing",
                                 code("gray", '    mser = cv2.MSER_create(min_diversity=0.1)\n' + _MS_TAIL), "verifier"),
    }),
    "Find the dark bubbles with cv2.MSER_create and detectRegions, collapse the nested regions of one bubble (box centres "
    "closer than 6 px), and report the number of bubbles and their centres in increasing x.",
    "MSER on a grayscale picture of bubbles: take the bounding box of every region it detects, treat boxes whose centres are "
    "within 6 px as the same bubble, and return how many bubbles there are plus one (x, y) centre each, sorted by x.")


# ---------------------------------------------------------------- 5. putText always anti-aliases: the ink mask

def _ink_truth(canvas, text, x, y, scale, thickness):
    out = canvas.copy()
    cv2.putText(out, text, (x, y), FONT, scale, 255, thickness)
    return out != canvas


def _ik_cases():
    out = []
    for text, x, y, scale, th, shape in (("Level 7", 12, 52, 1.0, 2, (70, 220)), ("rgb 255", 8, 40, 0.7, 1, (56, 160)),
                                         ("Gamma", 14, 60, 1.4, 3, (80, 200)), ("x = 3.5", 6, 36, 0.6, 1, (48, 140))):
        canvas = np.zeros(shape, np.uint8)
        mask = _ink_truth(canvas, text, x, y, scale, th)
        out.append(Case({"canvas": canvas, "text": text, "x": x, "y": y, "scale": scale, "thickness": th}, mask))
    return out


def _ik_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "ink")
    if got.shape != case.truth.shape:
        return fail(f"ink has shape {got.shape}")
    diff = int((got.astype(bool) != case.truth).sum())
    if diff:
        return fail(f"{diff} pixels differ from the set of pixels the rendering changed ({int(case.truth.sum())} of them)")
    if out["count"] != int(case.truth.sum()):
        return fail(f"count {out['count']}, the text touches {int(case.truth.sum())} pixels")
    return PASS


_IK_SIG = "canvas, text, x, y, scale, thickness"
R.add(Check(
    id="drill_text_ink_mask", group=GROUP, probes=("putText_hershey_pixels",), depends_on_5_0=True, role="gotcha",
    gotcha="putText with a Hershey font anti-aliases in 5.0 whatever lineType says: the glyph edge pixels are not 255, so "
           "`canvas == 255` (right for 4.x LINE_8 text) and a fixed threshold lose them",
    spec=spec("Render `text` with cv2.putText (cv2.FONT_HERSHEY_SIMPLEX, white 255, bottom-left corner at (x, y), the given "
              "scale and thickness) onto a copy of the black canvas, and return the mask of every pixel the rendering "
              "changed, together with the number of such pixels.",
              {"canvas": IOSpec("uint8", "HxW", 2, "black grayscale canvas, all zeros"),
               "text": IOSpec("str", "scalar", None, "the string to render"),
               "x": IOSpec("int", "scalar", None, "x of the bottom-left corner of the text"),
               "y": IOSpec("int", "scalar", None, "y of the bottom-left corner (the baseline)"),
               "scale": IOSpec("float", "scalar", None, "font scale"),
               "thickness": IOSpec("int", "scalar", None, "stroke thickness in px")},
              {"ink": IOSpec("bool", "HxW", 2, "True on every pixel whose value changed"),
               "count": IOSpec("int", "scalar", None, "number of True pixels")}),
    make_cases=_ik_cases, verify=_ik_verify,
    reference=code(_IK_SIG, '''\
    out = canvas.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)
    ink = out != canvas
    return {"ink": ink, "count": int(ink.sum())}
'''),
    wrong={
        "equals_255": Wrong("4.x: the text pixels are the ones equal to the colour drawn (LINE_8 text had no other values)",
                            code(_IK_SIG, '''\
    out = canvas.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)
    ink = out == 255
    return {"ink": ink, "count": int(ink.sum())}
'''), "verifier"),
        "threshold_127": Wrong("binarizes the rendering at 127: the faint anti-aliased edge pixels are dropped",
                               code(_IK_SIG, '''\
    out = canvas.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)
    ink = out > 127
    return {"ink": ink, "count": int(ink.sum())}
'''), "verifier"),
        "origin_top_left": Wrong("treats (x, y) as the top-left corner of the text and draws below it",
                                 code(_IK_SIG, '''\
    out = canvas.copy()
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(out, text, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)
    ink = out != canvas
    return {"ink": ink, "count": int(ink.sum())}
'''), "verifier"),
    }),
    "Draw the text on a copy of the black canvas with cv2.putText (Hershey simplex font, colour 255, origin at the "
    "bottom-left corner (x, y)) and report which pixels it modified, as a boolean mask, and how many there are.",
    "Compute the footprint of a text string: render it with cv2.putText in cv2.FONT_HERSHEY_SIMPLEX at the given scale "
    "and thickness, white on black, baseline origin (x, y), and return a boolean mask of the pixels that end up "
    "non-zero plus their count.")


# ---------------------------------------------------------------- 6. cv2.FontFace: the text API that is new in 5.0

_FF_INPUTS = (("Inspection OK", 12, 50, 24, (80, 260)), ("Lot 42-B", 10, 40, 18, (60, 200)),
              ("Torque 12 Nm", 16, 70, 32, (96, 300)), ("a jig, pegs", 8, 34, 16, (50, 220)))
_FF_SIG = "img, text, x, y, size"
_FF_REF = code(_FF_SIG, '''\
    out = img.copy()
    face = cv2.FontFace("sans")
    cv2.putText(out, text, (x, y), (0, 0, 0), face, size)
    rect = cv2.getTextSize((img.shape[1], img.shape[0]), text, (x, y), face, size)
    return {"img": out, "rect": np.array(rect)}
''')


def _ff_inputs() -> list[dict]:
    return [{"img": np.full((*shape, 3), 255, np.uint8), "text": text, "x": x, "y": y, "size": size}
            for text, x, y, size, shape in _FF_INPUTS]


@functools.lru_cache(maxsize=1)
def _ff_truth() -> tuple:
    """What the FontFace calls render, from a CLEAN process. Measured on 5.0.0: once any Hershey text call (getTextSize or
    putText) has run in a process, a later FontFace putText at a different size than the previous FontFace call renders
    differently (1,755 of 36,000 values at size 18 after size 24), so the truth must not be computed in this process,
    where earlier checks have called Hershey text functions. The sandbox starts a fresh interpreter."""
    r = run_solution(_FF_REF, _ff_inputs())
    if not r.ok:
        raise RuntimeError(f"could not render the FontFace truth: {r.summary()}")
    return tuple((o["img"], np.asarray(o["rect"], np.int64)) for o in r.outputs)


def _ff_cases():
    return [Case(inputs, truth) for inputs, truth in zip(_ff_inputs(), _ff_truth())]


def _ff_verify(case: Case, out: dict) -> Verdict:
    img_gt, rect_gt = case.truth
    got, rect = arr(out, "img"), arr(out, "rect")
    if got.shape != img_gt.shape or not np.array_equal(got, img_gt):
        return fail("the rendered text differs from the built-in 'sans' FontFace rendering at that size and position")
    if rect.shape != (4,) or not np.array_equal(rect, rect_gt):
        return fail(f"rect {rect.tolist()}, the bounding rectangle of the text is {rect_gt.tolist()} (x, y, w, h)")
    return PASS


R.add(Check(
    id="drill_fontface_label", group=GROUP, probes=("has cv2.FontFace",), depends_on_5_0=True, role="gotcha",
    gotcha="cv2.FontFace is new in 5.0: putText(img, text, org, color, fface, size) puts the colour before the font and "
           "getTextSize(imgsize, text, org, fface, size) returns a Rect; neither takes the Hershey (fontFace, fontScale, "
           "thickness) arguments",
    spec=spec("Draw a label with OpenCV's FontFace text API: the built-in 'sans' font (cv2.FontFace('sans')), `size` "
              "pixels tall, black, with its origin at (x, y). Return the image and the bounding rectangle "
              "(x, y, width, height) that OpenCV reports for that text.",
              {"img": IOSpec("uint8", "HxWx3", 3, "white BGR canvas"),
               "text": IOSpec("str", "scalar", None, "the label"),
               "x": IOSpec("int", "scalar", None, "x of the text origin"),
               "y": IOSpec("int", "scalar", None, "y of the text origin (bottom-left, baseline side)"),
               "size": IOSpec("int", "scalar", None, "font size in pixels")},
              {"img": IOSpec("uint8", "HxWx3", 3, "the canvas with the black label (drawn on a copy)"),
               "rect": IOSpec("int", "4", 1, "x, y, width, height of the text's bounding rectangle")}),
    make_cases=_ff_cases, verify=_ff_verify,
    reference=_FF_REF,
    wrong={
        "hershey_arguments": Wrong("old argument order with the FontFace object: putText(img, text, org, fontFace, fontScale, color)",
                                   code(_FF_SIG, '    out = img.copy()\n    face = cv2.FontFace("sans")\n'
                                                 '    cv2.putText(out, text, (x, y), face, size, (0, 0, 0))\n'
                                                 '    return {"img": out, "rect": np.array([x, y, 0, 0])}\n'),
                                   "error", "fontFace"),
        "hershey_font": Wrong("a Hershey font at size / 30 and the 4.x getTextSize box",
                              code(_FF_SIG, '''\
    out = img.copy()
    scale = size / 30.0
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return {"img": out, "rect": np.array([x, y - h, w, h + baseline])}
'''), "verifier"),
        "rect_from_old_gettextsize": Wrong("draws with FontFace but takes the box from the Hershey getTextSize (on 5.0.0 that "
                                           "call also changes how later FontFace text renders in the same process)",
                                           code(_FF_SIG, '''\
    out = img.copy()
    cv2.putText(out, text, (x, y), (0, 0, 0), cv2.FontFace("sans"), size)
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 1)
    return {"img": out, "rect": np.array([x, y - h, w, h + baseline])}
'''), "verifier"),
    }),
    "Using cv2.FontFace('sans'), write the label in black at the given origin in a font `size` pixels tall on a copy of "
    "the canvas, and return that image together with the label's bounding rectangle (x, y, w, h) as OpenCV computes it.",
    "Label a white canvas with OpenCV's FontFace text API: font 'sans', `size` px, black, origin (x, y). Also report "
    "the text's bounding Rect (x, y, width, height) from OpenCV, not an estimate.")


CHECKS = R.checks
PHRASINGS = R.phrasings
