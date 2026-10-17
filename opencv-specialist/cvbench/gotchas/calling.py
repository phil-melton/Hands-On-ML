"""Group 4, calling conventions: argument order, accepted values and failure modes that bite regardless of version.

All five are timeless (identical on 4.14 and 5.0):
  * ``cv2.resize`` takes ``dsize=(width, height)``; ``img.shape[:2]`` is (height, width), so passing it
    straight in silently transposes the output;
  * ``GaussianBlur`` needs an odd, positive kernel: ``int(6 * sigma) + 1`` is even for half-integer sigma;
    ``ksize=(0, 0)`` derives it from sigma;
  * ``cvtColor(gray, COLOR_BGR2GRAY)`` raises "Bad number of channels" on an image that is already gray;
  * ``countNonZero`` accepts one channel only;
  * ``imdecode`` (and ``imread``) return ``None`` on undecodable data instead of raising.

``imread`` itself is not a check: the production lint bans file I/O and the task footer says "no file access".
The ``None`` return is exercised through ``imdecode`` on in-memory bytes, which has identical semantics.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import PASS, Case, Check, IOSpec, TaskSpec, Verdict, Wrong, arr, fail
from .drift import pearson
from .dtypes import smooth_image

GROUP = "calling"


def _spec(task: str, inputs: dict, outputs: dict) -> TaskSpec:
    return TaskSpec(family="gotcha", task=task, inputs=inputs, outputs=outputs)


def _code(sig: str, body: str) -> str:
    return f"import cv2\nimport numpy as np\n\n\ndef solve({sig}):\n{body}"


# ---------------------------------------------------------------- 1. resize dsize is (width, height)

# The shape is the gotcha; the content check only asks "is this the resized src". Calibrated on 50 seeds x the check's two
# scale factors: Pearson correlation with the INTER_LINEAR result is >= 0.9886 for AREA, 0.9995 CUBIC, 0.9998 LANCZOS4 and
# 0.843 for NEAREST, while content that is merely the right shape (a constant, another image, a crop, shuffled rows,
# a flipped or shifted result) scores between -0.78 and 0.15. 0.70 sits in that gap.
RS_MIN_CORR = 0.70


def _color_smooth(seed: int, h: int, w: int) -> np.ndarray:
    return np.dstack([smooth_image(seed + 10 * c, h, w) for c in range(3)])


def _rs_cases():
    out = []
    for seed, (h1, w1), (h2, w2) in ((501, (48, 64), (60, 90)), (502, (90, 70), (40, 100))):
        src, ref = _color_smooth(seed, h1, w1), np.zeros((h2, w2, 3), np.uint8)
        out.append(Case({"src": src, "ref": ref}, cv2.resize(src, (w2, h2), interpolation=cv2.INTER_LINEAR)))
    return out


def _rs_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "resized")
    if got.shape != case.truth.shape:
        return fail(f"resized has shape {got.shape}, ref has shape {case.truth.shape}")
    corr = pearson(got, case.truth) if got.std() > 0 else 0.0
    return fail(f"resized content does not match a resize of src (correlation {corr:.2f})") if corr < RS_MIN_CORR else PASS


RESIZE_ORDER = Check(
    id="call_resize_order", group=GROUP, probes=("resize_dsize_order",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.resize wants dsize=(width, height) but img.shape[:2] is (height, width): the result is transposed",
    spec=_spec("Resize src so that it has exactly the same height and width as ref.",
               {"src": IOSpec("uint8", "H1xW1x3", 3, "image to resize"),
                "ref": IOSpec("uint8", "H2xW2x3", 3, "only its shape matters; H2 != W2")},
               {"resized": IOSpec("uint8", "H2xW2x3", 3, "src resized to ref's height and width")}),
    make_cases=_rs_cases, verify=_rs_verify,
    reference=_code("src, ref", '    return {"resized": cv2.resize(src, (ref.shape[1], ref.shape[0]))}\n'),
    wrong={
        "shape_slice": Wrong("cv2.resize(src, ref.shape[:2]): (h, w) where (w, h) is expected",
                             _code("src, ref", '    return {"resized": cv2.resize(src, ref.shape[:2])}\n'), "verifier"),
        "explicit_hw": Wrong("h, w = ref.shape[:2]; cv2.resize(src, (h, w))",
                             _code("src, ref", '    h, w = ref.shape[:2]\n    return {"resized": cv2.resize(src, (h, w))}\n'),
                             "verifier"),
    })


# ---------------------------------------------------------------- 2. GaussianBlur needs an odd kernel

# calibrated on 30 noisy images x sigma 1.5/2.5/3.5: kernels truncated at 3 sigma (2*ceil(3s)+1) and at 4 sigma, and the
# odd-ified int(6s)+1 kernel stay within 0 gray levels of cv2.GaussianBlur(img, (0, 0), sigma), a float32 pipeline within 1;
# a 5x5 box blur (a wrong substitute) is off by up to 87. 2 keeps a 2x margin over the legitimate worst case.
GB_TOL = 2


def _gb_cases():
    out = []
    for seed, sigma in ((511, 1.5), (512, 2.5), (513, 3.5)):
        rng = np.random.default_rng(seed)
        img = np.clip(smooth_image(seed, 48, 64).astype(np.float64) + rng.normal(0, 40, (48, 64)), 0, 255).astype(np.uint8)
        out.append(Case({"img": img, "sigma": sigma}, cv2.GaussianBlur(img, (0, 0), sigma)))
    return out


def _gb_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "blurred")
    if got.shape != case.truth.shape:
        return fail(f"blurred has shape {got.shape}")
    err = np.abs(got.astype(np.int16) - case.truth.astype(np.int16)).max()
    return fail(f"blurred image differs from a Gaussian of that sigma by up to {err} gray levels") if err > GB_TOL else PASS


GAUSSIAN_EVEN = Check(
    id="call_gaussian_even", group=GROUP, probes=("GaussianBlur_even_kernel",), depends_on_5_0=False, role="gotcha",
    gotcha="GaussianBlur raises on an even kernel size; int(6 * sigma) + 1 is even for sigma = 1.5, 2.5, 3.5",
    spec=_spec("Smooth the image with a Gaussian blur of standard deviation sigma (in px).",
               {"img": IOSpec("uint8", "HxW", 2, "grayscale image"),
                "sigma": IOSpec("float", "scalar", None, "px; may be fractional, e.g. 1.5")},
               {"blurred": IOSpec("uint8", "HxW", 2, "blurred image")}),
    make_cases=_gb_cases, verify=_gb_verify,
    reference=_code("img, sigma", '    return {"blurred": cv2.GaussianBlur(img, (0, 0), sigma)}\n'),
    wrong={
        "six_sigma_plus_one": Wrong("k = int(6 * sigma) + 1; GaussianBlur(img, (k, k), sigma): k = 10 for sigma 1.5",
                                    _code("img, sigma", '    k = int(6 * sigma) + 1\n'
                                                        '    return {"blurred": cv2.GaussianBlur(img, (k, k), sigma)}\n'),
                                    "error", "ksize.width % 2 == 1"),
        "even_by_construction": Wrong("k = 2 * int(3 * sigma): always even",
                                      _code("img, sigma", '    k = 2 * int(3 * sigma)\n'
                                                          '    return {"blurred": cv2.GaussianBlur(img, (k, k), sigma)}\n'),
                                      "error", "ksize.width % 2 == 1"),
    })


# ---------------------------------------------------------------- 3. cvtColor on an already-gray image

def _cg_cases():
    out = []
    rng = np.random.default_rng(521)
    bgr1 = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    gray1 = rng.integers(0, 256, (20, 30), dtype=np.uint8)
    bgr2 = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    gray2 = rng.integers(0, 256, (16, 24), dtype=np.uint8)
    for img in (bgr1, gray1, bgr2, gray2):  # the BGR image comes first, so code that always converts gets that far
        truth = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(Case({"img": img}, truth))
    return out


def _cg_verify(case: Case, out: dict) -> Verdict:
    got = arr(out, "gray")
    if got.shape != case.truth.shape:
        return fail(f"gray has shape {got.shape}, expected {case.truth.shape}")
    err = np.abs(got.astype(np.int16) - case.truth.astype(np.int16)).max()
    return fail(f"gray differs from the expected luminance by up to {err} levels") if err > 1 else PASS


CVTCOLOR_GRAY = Check(
    id="call_cvtcolor_gray", group=GROUP, probes=("cvtColor_gray_input",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.cvtColor(img, COLOR_BGR2GRAY) raises 'Bad number of channels' when img is already single-channel",
    spec=_spec("Return the image as a single-channel grayscale image. The loader hands over either a BGR image "
               "or one that is already gray.",
               {"img": IOSpec("uint8", "HxWx3 or HxW", None, "BGR, or already grayscale")},
               {"gray": IOSpec("uint8", "HxW", 2, "grayscale; an already-gray input comes back unchanged")}),
    make_cases=_cg_cases, verify=_cg_verify,
    reference=_code("img", '    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                           '    return {"gray": gray}\n'),
    wrong={
        "always_convert": Wrong("cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) on every input",
                                _code("img", '    return {"gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}\n'),
                                "error", "Bad number of channels"),
    })


# ---------------------------------------------------------------- 4. countNonZero takes one channel

def _cn_cases():
    out = []
    for seed in (531, 532):
        rng = np.random.default_rng(seed)
        img = np.zeros((30, 40, 3), np.uint8)
        box = rng.random((30, 40)) < 0.3
        img[box] = rng.integers(0, 256, (int(box.sum()), 3))
        dim = rng.random((30, 40)) < 0.1  # pixels whose only non-zero channel is B or R at value 1:
        img[dim] = 0                       # they are not black, but their gray value rounds to 0
        img[dim, 0 if seed % 2 else 2] = 1
        out.append(Case({"img": img}, int(img.any(axis=2).sum())))
    return out


def _cn_verify(case: Case, out: dict) -> Verdict:
    return PASS if out["count"] == case.truth else fail(f"count {out['count']}, expected {case.truth} non-black pixels")


COUNTNONZERO_3CH = Check(
    id="call_countnonzero_3ch", group=GROUP, probes=("countNonZero_3channel",), depends_on_5_0=False, role="gotcha",
    gotcha="cv2.countNonZero raises on a 3-channel image (cn == 1); counting through grayscale misses dim pixels",
    spec=_spec("Count the pixels that are not pure black, i.e. that have at least one non-zero channel.",
               {"img": IOSpec("uint8", "HxWx3", 3, "BGR image; some pixels are only very slightly non-black")},
               {"count": IOSpec("int", "scalar", None, "number of pixels with any channel > 0")}),
    make_cases=_cn_cases, verify=_cn_verify,
    reference=_code("img", '    return {"count": cv2.countNonZero(img.max(axis=2))}\n'),
    wrong={
        "direct": Wrong("cv2.countNonZero(img) on the 3-channel image",
                        _code("img", '    return {"count": cv2.countNonZero(img)}\n'), "error", "cn == 1"),
        "via_gray": Wrong("counts non-zero pixels of the grayscale conversion (a pixel (1, 0, 0) converts to 0)",
                          _code("img", '    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n'
                                       '    return {"count": cv2.countNonZero(gray)}\n'), "verifier"),
    })


# ---------------------------------------------------------------- 5. imdecode returns None on bad data

def _id_cases():
    out = []
    rng = np.random.default_rng(541)
    good1 = smooth_image(541, 24, 32)
    good1 = np.dstack([good1, good1[::-1], good1[:, ::-1]])
    good2 = rng.integers(0, 256, (18, 20, 3), dtype=np.uint8)
    garbage = np.frombuffer(b"this is a text file, not an image", np.uint8)
    noise = rng.integers(0, 256, 500, dtype=np.uint8)
    for img, buf in ((good1, None), (None, garbage), (good2, None), (None, noise)):
        if img is not None:
            buf = cv2.imencode(".png", img)[1].reshape(-1)
            truth = (np.array(img.shape, np.int64), float(img.mean()))
        else:
            truth = (np.zeros(3, np.int64), 0.0)
        out.append(Case({"buf": buf}, truth))
    return out


def _id_verify(case: Case, out: dict) -> Verdict:
    shape, mean = case.truth
    if not np.array_equal(arr(out, "shape"), shape):
        return fail(f"shape {arr(out, 'shape').tolist()}, expected {shape.tolist()}")
    if abs(out["mean"] - mean) > 1e-6:
        return fail(f"mean {out['mean']}, expected {mean}")
    return PASS


_ID_NONE = '        return {"shape": np.zeros(3, np.int64), "mean": 0.0}\n'
IMDECODE_NONE = Check(
    id="call_imdecode_none", group=GROUP, probes=("imdecode_garbage", "imread_missing_file"), depends_on_5_0=False,
    role="gotcha",
    gotcha="cv2.imdecode (like imread) returns None on undecodable data instead of raising: img.shape then fails",
    spec=_spec("Decode the encoded image bytes as a colour image. Some of the buffers are not valid images; for those "
               "report a shape of [0, 0, 0] and a mean of 0.0.",
               {"buf": IOSpec("uint8", "N", 1, "bytes of an encoded image file (PNG), or arbitrary bytes")},
               {"shape": IOSpec("int", "3", 1, "[height, width, channels] of the decoded image, or [0, 0, 0]"),
                "mean": IOSpec("float", "scalar", None, "mean of all decoded values, or 0.0")}),
    make_cases=_id_cases, verify=_id_verify,
    reference=_code("buf", '    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)\n    if img is None:\n' + _ID_NONE +
                           '    return {"shape": np.array(img.shape, np.int64), "mean": float(img.mean())}\n'),
    wrong={
        "no_none_check": Wrong("uses the result without checking for None",
                               _code("buf", '    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)\n'
                                            '    return {"shape": np.array(img.shape, np.int64), "mean": float(img.mean())}\n'),
                               "AttributeError", "NoneType"),
        "expects_cv2_error": Wrong("wraps the call in try/except cv2.error, assuming bad data raises",
                                   _code("buf", '    try:\n        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)\n'
                                                '    except cv2.error:\n' + _ID_NONE +
                                                '    return {"shape": np.array(img.shape, np.int64), "mean": float(img.mean())}\n'),
                                   "AttributeError", "NoneType"),
    })


CHECKS = [RESIZE_ORDER, GAUSSIAN_EVEN, CVTCOLOR_GRAY, COUNTNONZERO_3CH, IMDECODE_NONE]
