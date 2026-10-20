"""Family ``histogram_stats``: gray-level histogram and summary statistics, verified by a manufactured solution.

The answer is chosen first: a vector of 256 integer counts. The image is then built from it
(a multiset of gray levels, shuffled in space, each level painted with a random colour of
exactly that luma), so the verifier knows the truth without running any OpenCV code.

Tolerance law
-------------
No measurement noise exists here: the image is the answer, rearranged. Two kinds of output,
two laws.

* Counts (``hist``, ``median``) are integers, and ``cvtColor`` + ``calcHist`` are integer
  operations, so a correct solution reproduces them exactly: COUNT_TOL = 0. Any slack would
  hide precisely the mistakes this family exists to catch: a dropped saturated pixel is an
  error of 1 in bin 255 (``[0, 255]`` instead of ``[0, 256]``), a truncated luma moves
  half of the pixels by one bin.
* Floats (``mean``, ``std``): the truth is computed in exact integer arithmetic
  (sum v*h and sum v^2*h as Python ints, one rounding at the end). A correct solution's own
  error is float64 roundoff, ~eps * O(log N) (measured by ``calibrate()``). The smallest
  *convention* difference is population vs sample std: relative sqrt(N/(N-1)) - 1 ~ 1/(2N). The
  generator builds N = H*W <= 120*160 = 19200, so it is >= 1/(2*19200) = 2.6e-5 for every scene it can
  build (3.1e-4 at the smallest N = 1600); on the 400 ``calibrate()`` scenes, whose largest N is 18564,
  the smallest effect is 2.69e-5.
  FLOAT_RTOL sits between: |got - true| <= FLOAT_RTOL * max(1, |true|).

Gray levels are defined as BT.601 luma, ``round(0.299 R + 0.587 G + 0.114 B)``. That is what
``cv2.COLOR_BGR2GRAY`` computes, but its integer arithmetic and the exact formula disagree on
colours whose luma lies within ~0.003 of a rounding tie (``calibrate()`` sweeps all 2^24
colours; 20753 disagree, never by more than one level). Scenes never use a colour within
TIE_MARGIN of a tie, so the manufactured gray plane is unambiguous for any reasonable
implementation, and 255 - img maps gray v to exactly 255 - v.

Which integer formula that is, measured over the same domain (all 2^24 BGR colours through
``cv2.cvtColor(BGR2GRAY)``, OpenCV 5.0.0; 4.14.0 gives identical counts), against each candidate:
the 15-bit fixed-point formula ``(9798 R + 19235 G + 3735 B + 16384) >> 15`` has 0 mismatches, i.e.
cvtColor *is* that formula; the classic 14-bit formula ``(4899 R + 9617 G + 1868 B + 8192) >> 14``
(coefficients summing to 2^14) has 43,864 mismatches (never by more than one level); the exact
round-half-up luma ``(299 R + 587 G + 114 B + 500) // 1000`` has 20,753 (the figure above); float64
``np.rint(0.299 R + 0.587 G + 0.114 B)`` has 22,906. So each of those other candidates disagrees with
cvtColor on 0.12-0.26 % of the colours (20,753 to 43,864 of 16,777,216), always by one level.

Median convention (discrete data): the *lower* median, i.e. the ceil(N/2)-th smallest pixel =
the smallest level v with cumulative count >= N/2. For even N it is the lower of the two middle
pixels; ``np.median`` (which averages them) differs whenever the cumulative count hits N/2
exactly at a level boundary. 70 % of scenes are built so that the cumulative count hits
floor(N/2) exactly at a level boundary (``P_TIE``), with a gap of 1..12 empty levels above it.
For even N (about 3/4 of the scenes) that is a real tie, where the lower, upper and averaged
medians all differ (54 % of scenes measured); for odd N it separates ``n // 2`` from ``n / 2``.

Symmetries
----------
Any rearrangement of the pixels keeps the multiset, so everything is *invariant* under rot90,
mirror, transpose and a circular shift. Three more:

* ``recolor (same luma)``: new colours, same gray plane: invariant (the task is defined on
  luma, so any dependence on chroma is a bug).
* ``invert``: 255 - img maps gray v to 255 - v (luma weights sum to 1; ties are excluded).
  The histogram is *reversed*, mean -> 255 - mean, std invariant, and the lower median ->
  255 - upper median (the ceil(N/2)-th smallest becomes the (floor(N/2)+1)-th largest).
* ``tile 2x2``: every count x4 (extensive); mean, std and median unchanged (intensive).
  Sample std (ddof=1) is not intensive.

Blind spot: a constant answer is invariant under every pure permutation, like a stuck gauge.
The reversal and x4 relations break that for histograms that are not palindromic, and the
manufactured gate catches the rest.
"""
from __future__ import annotations

import inspect
import math
import textwrap
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

SPEC = TaskSpec(
    family="histogram_stats",
    task=("Compute the 256-bin gray-level histogram of the image, and the mean, standard deviation and median "
          "of its gray levels. The gray level of a pixel is its ITU-R BT.601 luma, "
          "round(0.299 R + 0.587 G + 0.114 B), the value cv2.COLOR_BGR2GRAY produces."),
    inputs={"img": IOSpec("uint8", "HxWx3", 3, "BGR colour image; any pixel values 0..255 may occur")},
    outputs={
        "hist": IOSpec("int", "256", 1, "hist[v] = number of pixels whose gray level is exactly v, v = 0..255; "
                                        "integer counts that sum to H*W"),
        "mean": IOSpec("float", "scalar", None, "arithmetic mean of the gray levels over all H*W pixels"),
        "std": IOSpec("float", "scalar", None, "population standard deviation of the gray levels "
                                               "(divide by N, not N-1)"),
        "median": IOSpec("int", "scalar", None, "lower median: the smallest gray level v such that at least "
                                                "half of the pixels have gray level <= v"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Build the gray-level histogram of this BGR image: 256 integer counts, one per level 0..255, with gray "
    "taken as BT.601 luma (what cv2.COLOR_BGR2GRAY computes). Also report the mean and the population "
    "standard deviation of the gray levels, and their lower median (the smallest level v with at least "
    "half of the pixels at or below v).",
    "Convert the BGR image to gray (BT.601 luma, rounded to the nearest integer level) and return how many "
    "pixels sit at each level 0..255, together with the mean, the population standard deviation (divide by "
    "N) and the lower median of the gray levels.",
    "For the colour image img, tabulate how often each of the 256 gray levels occurs (gray = BT.601 luma, as "
    "cv2.COLOR_BGR2GRAY gives it), as integer counts that sum to the number of pixels. Then give the mean, "
    "the population standard deviation and the lower median (the first level whose cumulative count reaches "
    "half of the pixels) of the gray levels.",
]

# ---------------------------------------------------------------- constants

# Counts are exact integers: no slack (Gate 2 decision). Enforced by array_equal, kept as a named constant
# so the law is visible and the tests can pin it.
COUNT_TOL = 0

# Counts, measured by calibrate() (OpenCV 5.0.0, numpy 2.4.6): 768 constant-luma gauge images and 400 scenes
# (3 155 582 pixels): max |count error| = 0 for cv2.calcHist and np.bincount, and cvtColor reproduces the
# manufactured gray plane on every pixel. Slack would cost: an absolute slack of 1 count hides the
# range_255 mistake in 9.8 % of scenes, a relative slack of 1 % of the pixels in 71.5 % of them.
#
# Floats, worst relative error over the 400 scenes, |got - true| / max(1, |true|):
#   numpy float64 mean/std                                    2.2e-16
#   histogram moments, float64                                3.1e-16
#   one-pass E[x^2] - E[x]^2, float64                         3.1e-14   (cancellation when std << mean)
#   cv2.meanStdDev                                            1.2e-13
#   numpy float32                                             1.1e-7
#   population vs sample std (ddof=1), smallest effect        2.7e-5   (see the note below)
# The ddof=1 effect is sqrt(N/(N-1)) - 1 ~ 1/(2N): 2.69e-5 at N = 18564, the largest N among these 400 scenes
# (hence "2.7e-5"). It is NOT 1/(2N) at N = 19200: 19200 = 120*160 is the largest N the generator can build and
# gives 2.60e-5, the smallest effect any scene can have.
# FLOAT_RTOL = 1e-9 is 8.1e3 above the worst float64 error and 2.7e4 below the smallest convention effect on
# these scenes (2.6e4 at the largest possible N; about the log-midpoint of the gap). It therefore also rejects
# float32 arithmetic (1.1e-7): see the open question in the report.
FLOAT_RTOL = 1e-9
# The symmetries are exact on the multiset, so counts and medians must match exactly and only float64 roundoff
# can move the floats. measure_metamorphic_drift(), reference over 300 scenes x 7 symmetries: max count drift 0,
# max median drift 0, max relative drift of mean 5.1e-16 and of std 2.3e-16.
META_RTOL = FLOAT_RTOL

# Scenes never use a colour whose luma is within TIE_MARGIN gray levels of a rounding tie.
# calibrate(): cvtColor disagrees with the exact round-half-up luma on 20753 of 2^24 colours, all of them
# within 0.003 of a tie; 0 disagreements at distance >= 0.05. TIE_MARGIN = 0.05 is ~17x the worst case.
# (cvtColor is the 15-bit fixed-point formula exactly, 0 mismatches over the 2^24 colours; the classic 14-bit
# formula disagrees with it on 43,864 colours: the module docstring has the formulas and the other candidates.)
TIE_MARGIN = 0.05
_MARGIN_MILLI = round(TIE_MARGIN * 1000)

KINDS = ("bell", "sparse", "comb", "flat")  # histogram shapes
H_RANGE, W_RANGE = (40, 120), (40, 160)  # inclusive; 1600..19200 pixels, odd and even N
P_TIE = 0.7  # share of scenes whose cumulative count hits floor(N/2) exactly at a level boundary


@dataclass
class Scene:
    img: np.ndarray  # BGR uint8, HxWx3
    gray: np.ndarray  # the manufactured gray plane, uint8 HxW
    hist: np.ndarray  # the manufactured answer: int64 (256,)
    mean: float
    std: float
    median: int  # lower median
    median_upper: int
    kind: str
    seed: int

    @property
    def n(self) -> int:
        return int(self.hist.sum())

    @property
    def tie(self) -> bool:
        """True when np.median-style averaging of the two middle pixels differs from the lower median."""
        return self.n % 2 == 0 and self.median != self.median_upper


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- exact statistics of a histogram

def lower_median(hist) -> int:
    """The ceil(N/2)-th smallest pixel, found by materialising the sorted pixels (no cumsum shortcut)."""
    h = np.asarray(hist, dtype=np.int64)
    return int(np.repeat(np.arange(256), h)[(int(h.sum()) + 1) // 2 - 1])


def upper_median(hist) -> int:
    """The (floor(N/2)+1)-th smallest pixel: the upper of the two middle pixels for even N."""
    h = np.asarray(hist, dtype=np.int64)
    return int(np.repeat(np.arange(256), h)[int(h.sum()) // 2])


def exact_stats(hist) -> tuple[float, float]:
    """Mean and population std of the gray levels, from exact integer sums and a single rounding."""
    h = [int(c) for c in hist]
    n = sum(h)
    s1 = sum(v * c for v, c in enumerate(h))
    s2 = sum(v * v * c for v, c in enumerate(h))
    return s1 / n, math.sqrt((n * s2 - s1 * s1) / (n * n))


# ---------------------------------------------------------------- scene generation

@lru_cache(maxsize=1)
def _chroma_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chroma 'shapes' e (BGR, lowest channel 0, highest m) with their luma rounding k, away from ties.

    Adding an integer t to all three channels adds exactly t to the luma (the weights sum to 1),
    so e + t has gray level k + t whenever every channel stays in 0..255 (t >= 0, m + t <= 255).
    """
    rng = np.random.default_rng(0xC0102)
    K = 60000
    m = rng.integers(0, 256, size=K)
    e = np.floor(rng.random((K, 3)) * (m[:, None] + 1)).astype(np.int64)
    rows = np.arange(K)
    e[rows, e.argmin(1)] = 0
    e[rows, e.argmax(1)] = m
    e[0] = 0  # a neutral gray is always available
    num = 114 * e[:, 0] + 587 * e[:, 1] + 299 * e[:, 2]  # BGR weights, thousandths of a gray level
    keep = np.abs(num % 1000 - 500) >= _MARGIN_MILLI
    e, num = e[keep], num[keep]
    return e, (num + 500) // 1000, e.max(1)


@lru_cache(maxsize=None)
def _feasible(level: int) -> np.ndarray:
    _, k, m = _chroma_bank()
    idx = np.flatnonzero((k <= level) & (level - k <= 255 - m))
    assert idx.size, level
    return idx


def _colorize(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint every pixel with a random BGR colour whose gray level is exactly the pixel's value."""
    e, k, _ = _chroma_bank()
    flat = gray.ravel()
    order = np.argsort(flat, kind="stable")
    out = np.empty((flat.size, 3), np.uint8)
    start = 0
    for level, cnt in enumerate(np.bincount(flat, minlength=256)):
        if cnt:
            feas = _feasible(level)
            pick = feas[rng.integers(feas.size, size=cnt)]
            out[order[start:start + cnt]] = e[pick] + (level - k[pick])[:, None]
        start += int(cnt)
    return out.reshape(gray.shape + (3,))


def _profile(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Non-negative weights over the interior levels 1..254 (zero = unoccupied)."""
    x = np.arange(256, dtype=np.float64)

    def bump():
        mu, sd = rng.uniform(15, 240), rng.uniform(4, 45)
        return rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((x - mu) / sd) ** 2)

    if kind == "bell":
        w = sum(bump() for _ in range(int(rng.integers(1, 4))))
        w[w < 1e-4 * w.max()] = 0.0  # truncated tails: empty bins
    elif kind == "sparse":
        w = np.zeros(256)
        lv = rng.choice(np.arange(1, 255), size=int(rng.integers(3, 25)), replace=False)
        w[lv] = rng.uniform(0.2, 1.0, lv.size)
    elif kind == "comb":  # a contrast-stretched look: every s-th level
        s, off = int(rng.choice([2, 3, 4, 5, 8])), int(rng.integers(0, 8))
        w = bump() * ((x - off) % s == 0)
    elif kind == "flat":
        w = np.ones(256)
        for lv in rng.choice(np.arange(1, 255), size=int(rng.integers(1, 5)), replace=False):
            w[lv] *= rng.uniform(5, 40)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    w[0] = w[255] = 0.0  # the two extreme levels are added separately
    if w[1:254].sum() == 0:
        w[128] = 1.0
    return w


def _apportion(w: np.ndarray, total: int, anchors: tuple[int, ...] = ()) -> np.ndarray:
    """Integer counts proportional to w that sum to ``total`` (largest remainder); anchors get >= 1."""
    q = np.asarray(w, dtype=np.float64) / np.sum(w) * total
    c = np.floor(q).astype(np.int64)
    short = int(total - c.sum())
    if short:
        c[np.argsort(-(q - c), kind="stable")[:short]] += 1
    for a in anchors:
        if c[a] == 0:
            c[int(np.argmax(c))] -= 1
            c[a] = 1
    return c


def _boundary(w: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Pick the level v where the cumulative count will hit floor(N/2), and the gap to the next occupied level."""
    occ = np.flatnonzero(w[:254] > 0)
    occ = occ[occ >= 1]
    mid = int(np.searchsorted(np.cumsum(w) / w.sum(), 0.5))
    j = int(np.argmin(np.abs(occ - mid))) + int(rng.integers(-3, 4))
    v = int(occ[min(max(j, 0), occ.size - 1)])
    gap = 1 if rng.random() < 0.3 else int(rng.integers(2, 13))
    return v, min(gap, 254 - v)


def make_scene(seed: int) -> Scene:
    """Manufacture an image together with its histogram: counts first, pixels second."""
    rng = np.random.default_rng(seed)
    H, W = int(rng.integers(H_RANGE[0], H_RANGE[1] + 1)), int(rng.integers(W_RANGE[0], W_RANGE[1] + 1))
    n = H * W
    kind = KINDS[int(rng.integers(len(KINDS)))]
    w = _profile(kind, rng)
    cap = max(2, n // 20)  # saturated levels hold 1 .. N/20 pixels, log-uniform: single stray pixels happen
    e0, e255 = (max(1, int(math.exp(rng.uniform(0.0, math.log(cap))))) for _ in range(2))
    hist = np.zeros(256, np.int64)
    hist[0], hist[255] = e0, e255
    if rng.random() < P_TIE:
        v, gap = _boundary(w, rng)
        w[v + 1:v + gap] = 0.0
        fill = w[w > 0].mean()
        w[v], w[v + gap] = max(w[v], fill), max(w[v + gap], fill)
        c = n // 2  # cumulative count at level v
        hist[1:v + 1] = _apportion(w[1:v + 1], c - e0, anchors=(v - 1,))
        hist[v + 1:255] = _apportion(w[v + 1:255], n - c - e255, anchors=(gap - 1,))
    else:
        hist[1:255] = _apportion(w[1:255], n - e0 - e255)
    assert hist.sum() == n and hist.min() >= 0
    flat = np.repeat(np.arange(256, dtype=np.uint8), hist)
    rng.shuffle(flat)
    gray = flat.reshape(H, W)
    mean, std = exact_stats(hist)
    return Scene(_colorize(gray, rng), gray, hist, mean, std, lower_median(hist), upper_median(hist), kind, seed)


def inputs(scene: Scene) -> dict:
    return {"img": scene.img}


# ---------------------------------------------------------------- verifiers

def _contract(out: dict) -> list[str]:
    problems = check_outputs(SPEC, out)
    if not problems and np.asarray(out["hist"]).shape != (256,):
        problems.append(f"'hist' should have shape (256,), got {np.asarray(out['hist']).shape}")
    return problems


def _close(got: float, want: float, rtol: float) -> bool:
    return abs(got - want) <= rtol * max(1.0, abs(want))  # False for NaN


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = _contract(out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    h = np.asarray(out["hist"]).astype(np.int64)
    if np.any(np.abs(h - scene.hist) > COUNT_TOL):
        diff = np.flatnonzero(h != scene.hist)
        shown = ", ".join(f"bin {i}: got {h[i]}, expected {scene.hist[i]}" for i in diff[:3])
        return Verdict(False, f"hist differs in {diff.size} bins ({shown}); counts sum to {int(h.sum())} "
                              f"for {scene.n} pixels")
    for name, want in (("mean", scene.mean), ("std", scene.std)):
        if not _close(float(out[name]), want, FLOAT_RTOL):
            return Verdict(False, f"{name} {out[name]!r}, expected {want!r} (rel. tol {FLOAT_RTOL:g})")
    if out["median"] != scene.median:
        return Verdict(False, f"median {out['median']}, expected the lower median {scene.median} "
                              f"(upper median {scene.median_upper}, N = {scene.n})")
    return Verdict(True)


T_ROT90, T_MIRROR, T_TRANSPOSE = "rot90", "mirror", "transpose"
T_ROLL, T_RECOLOR = "roll(+13,-7)", "recolor (same luma)"
T_TILE, T_INVERT = "tile 2x2", "invert 255-x"
_INVARIANT = (T_ROT90, T_MIRROR, T_TRANSPOSE, T_ROLL, T_RECOLOR)


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task. Relations to the base output are in ``_expected``."""
    img = scene.img
    return {
        T_ROT90: {"img": np.ascontiguousarray(np.rot90(img))},
        T_MIRROR: {"img": np.ascontiguousarray(img[:, ::-1])},
        T_TRANSPOSE: {"img": np.ascontiguousarray(img.transpose(1, 0, 2))},
        T_ROLL: {"img": np.roll(img, (13, -7), axis=(0, 1))},
        T_RECOLOR: {"img": _colorize(scene.gray, np.random.default_rng([scene.seed, 1]))},
        T_TILE: {"img": np.tile(img, (2, 2, 1))},
        T_INVERT: {"img": np.uint8(255) - img},
    }


def _expected(name: str, base: dict) -> dict | None:
    """What a symmetry implies for the output, given the (already validated) base output."""
    h = np.asarray(base["hist"]).astype(np.int64)
    if name in _INVARIANT:
        return {"hist": h, "mean": base["mean"], "std": base["std"], "median": base["median"]}
    if name == T_TILE:
        return {"hist": 4 * h, "mean": base["mean"], "std": base["std"], "median": base["median"]}
    if name == T_INVERT:
        if h.sum() == 0:
            return None
        return {"hist": h[::-1], "mean": 255.0 - base["mean"], "std": base["std"],
                "median": 255 - upper_median(h)}
    return None


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = _contract(base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    for name, out in variants.items():
        problems = _contract(out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        want = _expected(name, base)
        if want is None:
            return Verdict(False, f"{name}: no relation is defined for this symmetry")
        h = np.asarray(out["hist"]).astype(np.int64)
        if np.any(np.abs(h - want["hist"]) > COUNT_TOL):
            i = int(np.flatnonzero(h != want["hist"])[0])
            return Verdict(False, f"{name}: hist bin {i} is {h[i]}, the symmetry implies {want['hist'][i]}")
        for key in ("mean", "std"):
            if not _close(float(out[key]), float(want[key]), META_RTOL):
                return Verdict(False, f"{name}: {key} {out[key]!r}, the symmetry implies {want[key]!r}")
        if out["median"] != want["median"]:
            return Verdict(False, f"{name}: median {out['median']}, the symmetry implies {want['median']}")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    n = int(hist.sum())
    median = int(np.searchsorted(np.cumsum(hist), n / 2))
    return {"hist": hist, "mean": float(gray.mean()), "std": float(gray.std()), "median": median}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_GRAY = "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
_MEDIAN = "np.searchsorted(np.cumsum(hist), n / 2)"

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "hist_col0": (
        "4.x idiom hist[:, 0]: calcHist returns (256, 1) on 4.14 but (256,) on 5.0, so this raises IndexError.",
        _mutate((".ravel().astype(np.int64)", "[:, 0].astype(np.int64)"))),
    "range_255": (
        "ranges [0, 255]: the upper bound is exclusive, so gray level 255 falls outside and is dropped.",
        _mutate(("[0, 256]", "[0, 255]"))),
    "bins_255": (
        "histSize 255: neighbouring levels are merged and the histogram has 255 entries.",
        _mutate(("None, [256]", "None, [255]"))),
    "no_gray_conversion": (
        "Skips the gray conversion and histograms the blue channel of the BGR image.",
        _mutate((_GRAY, "gray = np.ascontiguousarray(img[:, :, 0])"))),
    "float_counts": (
        "Returns calcHist's float32 counts as they are; the contract asks for integer counts.",
        _mutate((".ravel().astype(np.int64)", ".ravel()"))),
    "rgb2gray": (
        "COLOR_RGB2GRAY on a BGR image swaps the red and blue weights. Harmless on gray pixels, "
        "wrong on coloured ones (unlike contour_area, where every shape is bright in every channel).",
        _mutate(("COLOR_BGR2GRAY", "COLOR_RGB2GRAY"))),
    "gray_truncate": (
        "Hand-rolled luma with astype(uint8), which truncates instead of rounding: ~half the pixels land one "
        "level low. A tolerance of a count or two would not notice.",
        _mutate((_GRAY, "gray = (img[:, :, ::-1] @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)"))),
    "std_ddof1": (
        "Sample standard deviation (N-1). Off by a relative 1/(2N), 2.6e-5 to 3.1e-4 here: far above FLOAT_RTOL.",
        _mutate(("gray.std()", "gray.std(ddof=1)"))),
    "median_np_int": (
        "int(np.median(...)): np.median averages the two middle pixels of an even-sized image and int() "
        "truncates the half. Differs from the lower median when they sit 2+ levels apart.",
        _mutate(("int(" + _MEDIAN + ")", "int(np.median(gray))"))),
    "median_strict": (
        "First level with cumulative count > N/2 (searchsorted side='right'): the upper median.",
        _mutate((_MEDIAN, _MEDIAN[:-1] + ', side="right")'))),
}


# ---------------------------------------------------------------- calibration

def _sweep_cvtcolor() -> dict:
    """cv2.cvtColor(BGR2GRAY) against the exact round-half-up luma, on all 2^24 colours."""
    G, R = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    base = 587 * G + 299 * R
    total = beyond = worst_level = worst_tie = 0
    for b in range(256):
        plane = np.stack([np.full((256, 256), b), G, R], axis=-1).astype(np.uint8)
        got = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY).astype(np.int64)
        num = base + 114 * b
        bad = got != (num + 500) // 1000
        if bad.any():
            tie = np.abs(num % 1000 - 500)  # thousandths of a gray level from the rounding tie
            total += int(bad.sum())
            beyond += int((bad & (tie >= _MARGIN_MILLI)).sum())
            worst_level = max(worst_level, int(np.abs(got - (num + 500) // 1000)[bad].max()))
            worst_tie = max(worst_tie, int(tie[bad].max()))
    return {"colours": 256 ** 3, "mismatches": total, "max_level_error": worst_level,
            "max_tie_distance_of_a_mismatch": worst_tie / 1000, "mismatches_at_or_beyond_TIE_MARGIN": beyond}


def _estimates(gray: np.ndarray) -> dict[str, tuple[float, float]]:
    """(mean, std) by several plausible float implementations."""
    h = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p, v = h / h.sum(), np.arange(256, dtype=np.float64)
    mu = float((v * p).sum())
    m, s = cv2.meanStdDev(gray)
    g64 = gray.astype(np.float64)
    g32 = gray.astype(np.float32)
    return {
        "numpy float64": (float(gray.mean()), float(gray.std())),
        "cv2.meanStdDev": (float(np.asarray(m).ravel()[0]), float(np.asarray(s).ravel()[0])),
        "histogram moments float64": (mu, math.sqrt(float((((v - mu) ** 2) * p).sum()))),
        "one-pass E[x^2]-E[x]^2 float64": (float(g64.mean()), math.sqrt(max(float((g64 * g64).mean() - g64.mean() ** 2), 0.0))),
        "numpy float32": (float(g32.mean()), float(g32.std())),
    }


def calibrate(n: int = 400, seed: int = 10_000) -> dict:
    """Show that counts need zero tolerance and measure the float error of the statistics."""
    out: dict = {"cvtColor_vs_exact_luma": _sweep_cvtcolor()}

    # gauge blocks: constant-luma images, one per gray level and three colourings each
    rng = np.random.default_rng(seed)
    gauge_err = 0
    for level in range(256):
        for _ in range(3):
            gray = np.full((8, 12), level, np.uint8)
            img = _colorize(gray, rng)
            h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]).ravel()
            want = np.zeros(256)
            want[level] = gray.size
            gauge_err = max(gauge_err, int(np.abs(h - want).max()))
    out["gauge_constant_images"] = {"levels": 256, "images": 768, "max_abs_count_error": gauge_err}

    count_err = plane_err = pixels = 0
    rel: dict[str, float] = {}
    ddof_effect = np.inf
    hidden_1, hidden_1pct = 0, 0
    for i in range(n):
        s = make_scene(seed + i)
        gray = cv2.cvtColor(s.img, cv2.COLOR_BGR2GRAY)
        plane_err += int(np.count_nonzero(gray != s.gray))
        h = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        count_err = max(count_err, int(np.abs(h - s.hist).max()),
                        int(np.abs(np.bincount(gray.ravel(), minlength=256) - s.hist).max()))
        pixels += s.n
        hidden_1 += int(s.hist[255] <= 1)  # range_255 drops exactly hist[255] pixels
        hidden_1pct += int(s.hist[255] <= 0.01 * s.n)
        for name, (m, sd) in _estimates(gray).items():
            e = max(abs(m - s.mean) / max(1.0, s.mean), abs(sd - s.std) / max(1.0, s.std))
            rel[name] = max(rel.get(name, 0.0), e)
        ddof_effect = min(ddof_effect, math.sqrt(s.n / (s.n - 1)) - 1)
    out["scenes"] = {"n": n, "pixels": pixels, "gray_plane_mismatches_vs_cvtColor": plane_err,
                     "max_abs_count_error": count_err}
    out["float_rel_error"] = rel
    out["worst_float64_rel_error"] = max(v for k, v in rel.items() if k != "numpy float32")
    out["ddof1_min_rel_effect"] = float(ddof_effect)
    # what slack would cost: share of scenes in which the range_255 mistake (error = hist[255] counts) stays
    # inside an absolute slack of 1 count, or a relative slack of 1 % of the pixels
    out["range_255_hidden_by_slack"] = {"abs_1_count": hidden_1 / n, "rel_1_percent_of_pixels": hidden_1pct / n}
    return out


def measure_metamorphic_drift(n: int = 300, seed: int = 50_000) -> dict:
    """Largest movement of the reference's outputs from what each symmetry implies."""
    worst = {"count": 0, "median": 0, "mean": 0.0, "std": 0.0}
    for i in range(n):
        scene = make_scene(seed + i)
        base = reference_solve(scene.img)
        for name, t in transforms(scene).items():
            got, want = reference_solve(t["img"]), _expected(name, base)
            worst["count"] = max(worst["count"], int(np.abs(got["hist"] - want["hist"]).max()))
            worst["median"] = max(worst["median"], abs(got["median"] - want["median"]))
            for key in ("mean", "std"):
                worst[key] = max(worst[key], abs(got[key] - want[key]) / max(1.0, abs(want[key])))
    return worst


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=1))
    print("metamorphic drift of the reference:", measure_metamorphic_drift())
