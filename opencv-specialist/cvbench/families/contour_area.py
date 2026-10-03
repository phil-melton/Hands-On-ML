"""Family ``contour_area``: outer-boundary areas of blobs, verified by a manufactured solution.

The verifier borrows the Method of Manufactured Solutions from CFD code verification
(Roache 2002): choose the answer first (shapes of known area), render the image from
it, then require the code to recover the answer within a tolerance derived from the
physics of the measurement rather than picked by taste.

Tolerance physics
-----------------
A blob's area error lives on its boundary, not in its bulk. The two standard
estimators bracket the truth:

* ``cv2.contourArea`` traces the polygon through the *centres* of the boundary pixels,
  so it can sit up to ~1/2 px inside the true boundary all the way round: error ~ -P/2
  (disk r = 20: 1257 px filled, contourArea 1200; see probes/results).
* pixel counting (``countNonZero``, connected-component stats) can sit up to ~1/2 px
  outside: error ~ +P/2 (20x20-px square drawn corner-to-corner on pixel centres:
  400 px for a 361 px^2 polygon).

So |error| <= c*P + c0 with c just above 1/2, and the *relative* error ~ c*P/A = 2c/r
for a disk: 2 % at r = 50 px, 20 % at r = 5 px. A fixed-percentage tolerance is too
tight for small blobs and too loose for big ones. ``calibrate()`` measures c on
single-shape "gauge blocks"; ``TOL_C`` carries a safety margin over the worst case.

Metamorphic checks
------------------
Where no exact answer exists, symmetries of the input imply invariants of the output:
translating, rotating by 90 degrees, mirroring or dimming the image must not move the
areas. These catch hidden assumptions (a hard-coded threshold) that the base scenes
never exercise. Their blind spot: a constant output is invariant under everything,
like a stuck gauge. That is why they complement the manufactured solution rather than
replace it.
"""
from __future__ import annotations

import inspect
import math
import textwrap
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from cvbench.contract import IOSpec, TaskSpec, check_outputs

SPEC = TaskSpec(
    family="contour_area",
    task=("Find every separate bright shape on the dark background. Report the area enclosed by each "
          "shape's outer boundary (holes are not subtracted) and the number of shapes."),
    inputs={"img": IOSpec("uint8", "HxWx3", 3, "BGR; bright solid shapes on a dark uniform background; "
                                               "shapes touch neither each other nor the border")},
    outputs={
        "areas": IOSpec("float", "N", 1, "px^2, one per shape, area inside the outer boundary, largest first"),
        "count": IOSpec("int", "scalar", None, "number of shapes"),
    },
)

PHRASINGS = [
    SPEC.task,
    "Count the bright blobs on the dark background and give each blob's area in px^2 (everything inside "
    "its outer boundary, holes included), largest first.",
    "For each light object in this BGR image, measure the area inside its outer contour. Return the areas "
    "in descending order and how many objects there are.",
    "Segment the bright shapes from the dark background and report their outer-contour areas, biggest "
    "first, with the shape count. Holes inside a shape count towards its area.",
]

KINDS = ("rect", "disk", "ellipse", "rotrect", "triangle", "ring", "annulus")
BORDER, GAP = 14, 5  # px kept clear at the image border and between shapes

# Calibrated by calibrate() on 3500 single-shape gauge blocks (OpenCV 5.0.0), error / P:
#   contourArea  in [-0.589, +0.120]  (disks/annuli -0.40..-0.59: the half-pixel inward shave;
#                                     rect/ring/rotrect exactly 0: vertices sit on pixel centres)
#   pixel count  in [-0.151, +0.598]  (rects +0.50..+0.53: the half-pixel outward rim)
# Worst |error| / P = 0.598, so TOL_C = 0.75 carries ~25 % margin. TOL_C0 covers the corner
# quantization of the smallest shapes. META_TOL: the reference's sorted areas moved by exactly
# 0.0 px^2 under every transform in 300 scenes (the symmetries preserve the pixel set).
TOL_C, TOL_C0 = 0.75, 2.0
META_TOL = 1.0


@dataclass(frozen=True)
class Shape:
    kind: str
    area: float  # analytic area enclosed by the outer boundary, px^2
    perimeter: float  # analytic outer perimeter, px
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 reserved on the canvas


@dataclass
class Scene:
    img: np.ndarray
    shapes: list[Shape]
    background: int
    seed: int


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- scene generation

def _polygon(pts: np.ndarray) -> tuple[float, float]:
    """Shoelace area and perimeter of a closed integer polygon."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))))
    return area, perim


def _ramanujan(a: float, b: float) -> float:
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


Draw = Callable[[np.ndarray, int, int, tuple, int], tuple[float, float]]


def _propose(kind: str, rng: np.random.Generator) -> tuple[int, int, Draw]:
    """Sample one shape: (half-width, half-height, draw). draw() paints it and returns (area, perimeter)."""
    if kind == "rect":
        w, h = (int(v) for v in rng.integers(6, 91, size=2))

        def draw(img, cx, cy, color, bg):
            x0, y0 = cx - w // 2, cy - h // 2
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, -1)
            return w * h, 2.0 * (w + h)
        return w // 2 + 1, h // 2 + 1, draw

    if kind == "disk":
        r = int(rng.integers(5, 46))

        def draw(img, cx, cy, color, bg):
            cv2.circle(img, (cx, cy), r, color, -1)
            return math.pi * r * r, 2 * math.pi * r
        return r + 1, r + 1, draw

    if kind == "ellipse":
        a = int(rng.integers(6, 51))
        b = int(rng.integers(5, a + 1))
        ang = float(rng.uniform(0, 180))

        def draw(img, cx, cy, color, bg):
            cv2.ellipse(img, (cx, cy), (a, b), ang, 0, 360, color, -1)
            return math.pi * a * b, _ramanujan(a, b)
        return a + 1, a + 1, draw

    if kind == "rotrect":
        w, h = (float(v) for v in rng.integers(8, 81, size=2))
        ang = float(rng.uniform(0, 90))
        half = int(math.ceil(math.hypot(w, h) / 2)) + 1

        def draw(img, cx, cy, color, bg):
            pts = np.round(cv2.boxPoints(((cx, cy), (w, h), ang))).astype(np.int32)
            cv2.fillPoly(img, [pts], color)
            return _polygon(pts)
        return half, half, draw

    if kind == "triangle":
        R = float(rng.integers(10, 46))
        base = rng.uniform(0, 2 * math.pi)
        ang = base + np.array([0, 2 * math.pi / 3, 4 * math.pi / 3]) + rng.uniform(-0.35, 0.35, 3)
        rad = R * rng.uniform(0.7, 1.0, 3)

        def draw(img, cx, cy, color, bg):
            pts = np.round(np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]).astype(np.int32)
            cv2.fillPoly(img, [pts], color)
            return _polygon(pts)
        return int(R) + 1, int(R) + 1, draw

    if kind == "ring":  # rectangle with a rectangular hole
        w, h = (int(v) for v in rng.integers(24, 91, size=2))
        m = int(rng.integers(4, min(w, h) // 2 - 3))

        def draw(img, cx, cy, color, bg):
            x0, y0 = cx - w // 2, cy - h // 2
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, -1)
            cv2.rectangle(img, (x0 + m, y0 + m), (x0 + w - m, y0 + h - m), (bg, bg, bg), -1)
            return w * h, 2.0 * (w + h)
        return w // 2 + 1, h // 2 + 1, draw

    if kind == "annulus":
        R = int(rng.integers(12, 46))
        r_in = int(rng.integers(4, R - 4))

        def draw(img, cx, cy, color, bg):
            cv2.circle(img, (cx, cy), R, color, -1)
            cv2.circle(img, (cx, cy), r_in, (bg, bg, bg), -1)
            return math.pi * R * R, 2 * math.pi * R
        return R + 1, R + 1, draw

    raise ValueError(f"unknown kind {kind!r}")


def make_scene(seed: int, size: tuple[int, int] = (360, 480), n_range: tuple[int, int] = (3, 7),
               kinds: tuple[str, ...] = KINDS) -> Scene:
    """Manufacture a scene: shapes of known area, placed apart, on a dark uniform background."""
    rng = np.random.default_rng(seed)
    H, W = size
    bg = int(rng.integers(10, 51))
    img = np.full((H, W, 3), bg, np.uint8)
    shapes: list[Shape] = []
    target = int(rng.integers(n_range[0], n_range[1] + 1))
    for _ in range(400):
        if len(shapes) == target:
            break
        kind = kinds[int(rng.integers(len(kinds)))]
        hw, hh, draw = _propose(kind, rng)
        if W - 2 * (BORDER + hw) <= 0 or H - 2 * (BORDER + hh) <= 0:
            continue
        cx = int(rng.integers(BORDER + hw, W - BORDER - hw))
        cy = int(rng.integers(BORDER + hh, H - BORDER - hh))
        box = (cx - hw, cy - hh, cx + hw, cy + hh)
        if any(box[0] - GAP <= b[2] and b[0] - GAP <= box[2] and box[1] - GAP <= b[3] and b[1] - GAP <= box[3]
               for b in (s.box for s in shapes)):
            continue
        color = tuple(int(c) for c in rng.integers(150, 256, size=3))
        area, perim = draw(img, cx, cy, color, bg)
        shapes.append(Shape(kind, float(area), float(perim), box))
    return Scene(img, shapes, bg, seed)


def inputs(scene: Scene) -> dict:
    return {"img": scene.img}


# ---------------------------------------------------------------- verifiers

def tolerance(shape: Shape) -> float:
    return TOL_C * shape.perimeter + TOL_C0


def _unmatched(truth: list[float], tols: list[float], got: np.ndarray) -> int | None:
    """Bipartite matching (Kuhn): pair every manufactured shape with a distinct measured area
    inside its tolerance. Returns the index of a shape that cannot be paired, or None."""
    adj = [[j for j, g in enumerate(got) if abs(g - t) <= tol] for t, tol in zip(truth, tols)]
    owner = [-1] * len(got)

    def assign(i: int, seen: set) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] == -1 or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(len(truth)):
        if not assign(i, set()):
            return i
    return None


def verify_manufactured(scene: Scene, out: dict) -> Verdict:
    problems = check_outputs(SPEC, out)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    got = np.asarray(out["areas"], dtype=np.float64).ravel()
    n = len(scene.shapes)
    if out["count"] != n:
        return Verdict(False, f"count {out['count']}, but the scene has {n} shapes")
    if got.size != n:
        return Verdict(False, f"{got.size} areas for {n} shapes")
    if np.any(np.diff(got) > 0):
        return Verdict(False, "areas are not sorted largest-first")
    truth, tols = [s.area for s in scene.shapes], [tolerance(s) for s in scene.shapes]
    bad = _unmatched(truth, tols, got)
    if bad is not None:
        s = scene.shapes[bad]
        closest = got[np.argmin(np.abs(got - s.area))]
        return Verdict(False, f"{s.kind} with manufactured area {s.area:.1f} ± {tols[bad]:.1f} px^2 has no match "
                              f"(closest measured {closest:.1f})")
    return Verdict(True)


def transforms(scene: Scene) -> dict[str, dict]:
    """Symmetries of the task: each must leave the sorted areas and the count unchanged."""
    img, bg = scene.img, scene.background
    H, W = img.shape[:2]
    dx, dy = 7, -5  # well inside the BORDER margin
    shifted = np.full_like(img, bg)
    shifted[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return {
        "translate(+7,-5)": {"img": shifted},
        "rot90": {"img": np.ascontiguousarray(np.rot90(img))},
        "mirror": {"img": np.ascontiguousarray(img[:, ::-1])},
        "dim x0.45": {"img": np.round(img.astype(np.float32) * 0.45).astype(np.uint8)},
    }


def verify_metamorphic(base: dict, variants: dict[str, dict]) -> Verdict:
    problems = check_outputs(SPEC, base)
    if problems:
        return Verdict(False, "contract: " + "; ".join(problems))
    a0 = np.sort(np.asarray(base["areas"], dtype=np.float64).ravel())
    for name, out in variants.items():
        problems = check_outputs(SPEC, out)
        if problems:
            return Verdict(False, f"{name}: contract: " + "; ".join(problems))
        a = np.sort(np.asarray(out["areas"], dtype=np.float64).ravel())
        if out["count"] != base["count"] or a.size != a0.size:
            return Verdict(False, f"{name}: count {out['count']} vs {base['count']} on the original image")
        moved = float(np.max(np.abs(a - a0))) if a.size else 0.0
        if moved > META_TOL:
            return Verdict(False, f"{name}: areas moved by up to {moved:.1f} px^2 (allowed {META_TOL})")
    return Verdict(True)


# ---------------------------------------------------------------- reference and mutants

def reference_solve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    return {"areas": np.array(areas, dtype=np.float64), "count": len(contours)}


REFERENCE = ("import cv2\nimport numpy as np\n\n\n"
             + textwrap.dedent(inspect.getsource(reference_solve)).replace("def reference_solve(", "def solve("))


def _mutate(*edits: tuple[str, str]) -> str:
    code = REFERENCE
    for old, new in edits:
        assert old in code, f"mutation target not found: {old!r}"
        code = code.replace(old, new)
    return code


_FIND = "contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"

# Plausible model mistakes. Each runs without raising unless noted.
MUTANTS: dict[str, tuple[str, str]] = {
    "rgb2gray (harmless here)": (
        "Channel order cannot matter: every shape is bright in every channel. A true negative.",
        _mutate(("COLOR_BGR2GRAY", "COLOR_RGB2GRAY"))),
    "retr_tree": (
        "RETR_TREE also returns hole contours, so holes are counted as shapes.",
        _mutate(("cv2.RETR_EXTERNAL", "cv2.RETR_TREE"))),
    "points_as_area": (
        "Uses the number of contour points as the 'area'.",
        _mutate(("cv2.contourArea(c)", "float(len(c))"))),
    "inverted_threshold": (
        "THRESH_BINARY_INV makes the background the blob.",
        _mutate(("cv2.THRESH_BINARY + cv2.THRESH_OTSU", "cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU"))),
    "fixed_threshold_127": (
        "Hard-coded threshold: right for these scenes, wrong for a dimmer image.",
        _mutate(("cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
                 "cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)"))),
    "component_pixel_areas": (
        "connectedComponentsWithStats areas: holes get subtracted.",
        _mutate((_FIND + "\n    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)",
                 "n, _, stats, _ = cv2.connectedComponentsWithStats(mask)\n"
                 "    contours = range(n - 1)\n"
                 "    areas = sorted(stats[1:, cv2.CC_STAT_AREA].astype(float), reverse=True)"))),
    "unsorted": (
        "Forgets to sort the areas largest-first.",
        _mutate(("sorted((cv2.contourArea(c) for c in contours), reverse=True)",
                 "[cv2.contourArea(c) for c in contours]"))),
    "opencv3_unpack": (
        "OpenCV 3 three-value unpack of findContours. Raises ValueError on 4.x and 5.x.",
        _mutate((_FIND, "_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"))),
    "swallowed_opencv3_unpack": (
        "The same bug wrapped in try/except, returning an empty result. Exec-only filtering keeps this one.",
        _mutate(("    gray = cv2.cvtColor", "    try:\n        gray = cv2.cvtColor"),
                ("    _, mask = cv2", "        _, mask = cv2"),
                ("    " + _FIND, "        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, "
                                   "cv2.CHAIN_APPROX_SIMPLE)\n"
                                   "    except Exception:\n"
                                   "        return {\"areas\": np.zeros(0), \"count\": 0}"))),
}


# ---------------------------------------------------------------- calibration

def calibrate(n: int = 3500, seed: int = 12345) -> dict:
    """Measure (estimate - truth) / perimeter on single-shape gauge blocks, for both estimators."""
    rng = np.random.default_rng(seed)
    out: dict = {"contourArea": [], "pixel_count": []}
    by_kind: dict = {}
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        hw, hh, draw = _propose(kind, rng)
        H, W = 2 * (hh + BORDER), 2 * (hw + BORDER)
        img = np.zeros((H, W, 3), np.uint8)
        area, perim = draw(img, W // 2, H // 2, (255, 255, 255), 0)
        mask = img[..., 0]
        c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [c], -1, 255, -1)
        for est, val in (("contourArea", cv2.contourArea(c)), ("pixel_count", cv2.countNonZero(filled))):
            ratio = (val - area) / perim
            out[est].append(ratio)
            by_kind.setdefault((kind, est), []).append(ratio)
    summary = {est: {"min": float(np.min(v)), "max": float(np.max(v))} for est, v in out.items()}
    summary["by_kind"] = {f"{k}/{e}": (round(float(np.min(v)), 3), round(float(np.max(v)), 3))
                          for (k, e), v in sorted(by_kind.items())}
    summary["worst_abs"] = max(max(abs(s["min"]), abs(s["max"])) for s in summary.values() if "min" in s)
    return summary


def measure_metamorphic_drift(n: int = 300, seed: int = 50_000) -> float:
    """Largest movement of the reference's sorted areas under the task symmetries."""
    worst = 0.0
    for i in range(n):
        scene = make_scene(seed + i)
        base = np.sort(reference_solve(scene.img)["areas"])
        for t in transforms(scene).values():
            got = np.sort(reference_solve(t["img"])["areas"])
            if got.size != base.size:
                return float("inf")
            if got.size:
                worst = max(worst, float(np.max(np.abs(got - base))))
    return worst


if __name__ == "__main__":
    import json
    print(json.dumps(calibrate(), indent=1))
    print("metamorphic drift of the reference:", measure_metamorphic_drift())
