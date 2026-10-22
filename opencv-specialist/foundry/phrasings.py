"""Teacher-written task phrasings (Gate 4a decision 2): up to 40 new wordings per trainable family.

Every family has 16 hand-written phrasings of its task line: 0-3 dev, 4-11 test, 12-15 foundry. Training data written from
only four wordings per family would teach the model four prompts, so the teacher is asked for more.

    python -m foundry.run prepare-phrasings --run NAME      prompt rows (kind "phrasing") in the run's prompts.jsonl
    (the Colab job answers them: foundry/colab_job.py, k samples each)
    python -m foundry.run screen-phrasings  --run NAME      parse, screen, write foundry/phrasings/<family>.json
    python -m foundry.run calibrate-phrasings               the numbers behind DUP_THRESHOLD and RUN_LIMIT

What the teacher sees, per family and per request: ``SPEC.task`` (the thing to reword) and the four foundry phrasings
(12-15) as style examples; never the dev/test phrasings 1-11. The rules it is given: same task and outputs, no API names
beyond those in ``SPEC.task``, no implementation hints, no tolerance or benchmark words, and a register that varies from
request to request (REGISTERS). The shared accuracy sentence of ``feature_homography`` is taken out of everything the
teacher sees and put back by the screen, exactly as the 16 hand-written phrasings carry it.

The screen keeps a candidate only if it

  (a) is not a near-duplicate of the family's 16 phrasings or of a candidate already kept,
  (b) does not overlap any dev/test phrasing (any family) or gotcha task line, and its rendered prompt is not a scored prompt,
  (c) adds no ``cv2.`` name beyond ``SPEC.task`` and none of HINT_WORDS (the checks of tests/test_splits.py) or EXTRA_HINT_WORDS
      (more algorithm and library words), and is plain ASCII like the hand-written phrasings, and
  (d) still carries the family's required content (CONTENT_CHECKS, mirroring each tests/test_<family>.py check).

"Near-duplicate" and "overlap" are one measure: the Jaccard similarity of the two texts' word 5-gram sets at or above
DUP_THRESHOLD, or a shared run of RUN_LIMIT consecutive words or more. Both limits are calibrated on the 16 hand-written
phrasings of the 7 trainable families (``calibrate``): their largest pairwise 5-gram Jaccard is 0.233 (hough_segments 4 vs
7), so DUP_THRESHOLD = 0.25 lets nothing through that is closer to an existing phrasing than the existing phrasings are to each
other; their longest shared run is 16 words (feature_homography 5 vs 13, components_stats 4 vs 8), so RUN_LIMIT = 17.
tests/test_foundry.py recomputes both and fails if the phrasings change under them.

Kept phrasings are written to ``foundry/phrasings/<family>.json`` with their provenance (teacher model, revision and license,
run, date); specs use them as phrasing indices 16, 17, ... (``foundry.specs``). Screening again keeps the phrasings already
in the file (so indices stay stable) and fills up to MAX_PER_FAMILY from the new candidates.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import re
from collections import Counter, namedtuple
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from cvbench.splits import (DEV_PHRASINGS, FAMILIES, FOUNDRY_PHRASINGS, PHRASINGS_PER_FAMILY, TEST_PHRASINGS,
                            foundry_phrasings)

PHRASINGS_DIR = Path(__file__).resolve().parent / "phrasings"
GENERATED_BASE = PHRASINGS_PER_FAMILY   # phrasing index of the first teacher-written phrasing of a family
MAX_PER_FAMILY = 40
NGRAM = 5
DUP_THRESHOLD = 0.25   # calibrated, see the module docstring
RUN_LIMIT = 17         # calibrated, see the module docstring

# Words that turn a task statement into a recipe or leak how it is graded (mirrors HINT_WORDS in tests/test_splits.py;
# a test compares the two).
HINT_WORDS = ("ransac", "sift", "orb", "akaze", "brisk", "hough", "houghlinesp", "findcontours", "contourarea", "inrange",
              "calchist", "connectedcomponents", "warpperspective", "getperspectivetransform", "findhomography",
              "approxpolydp", "cornersubpix", "findchessboardcorners", "tolerance", "gate", "seed", "benchmark",
              "metamorphic", "version", "deprecated", "4.14", "countnonzero", "minarearect", "boundingrect")

# Algorithm and library words no hand-written phrasing of a trainable family uses (a test checks): a wording that adds one is
# turning the task into a recipe. The screen refuses them in addition to HINT_WORDS.
EXTRA_HINT_WORDS = ("otsu", "canny", "threshold", "thresholding", "morphology", "morphological", "dilate", "erode", "gaussian", "blur",
                    "kmeans", "sobel", "laplacian", "binarize", "binarise", "numpy", "adaptive", "watershed", "opencv")

_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[a-z_][a-z0-9_]*)*")
_ASCII_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ", "−": "-",
                             "×": "x", "…": "...", " ": " ", "→": "->"})


# ---------------------------------------------------------------- text measures

def words(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(word_list: list[str], n: int = NGRAM) -> frozenset:
    return frozenset(tuple(word_list[i:i + n]) for i in range(len(word_list) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def longest_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of consecutive words that both lists contain."""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


Line = namedtuple("Line", "label text words grams")


def make_line(label: str, text: str) -> Line:
    w = words(text)
    return Line(label, text, w, ngrams(w))


def closeness(cand: Line, other: Line, *, dup_threshold: float = DUP_THRESHOLD, run_limit: int = RUN_LIMIT) -> str | None:
    """Why ``cand`` is too close to ``other`` (a reason string), or None."""
    j = jaccard(cand.grams, other.grams)
    if j >= dup_threshold:
        return f"5-gram Jaccard {j:.2f} with {other.label}"
    if cand.grams & other.grams:       # a shared run of 17 words needs shared 5-grams: skip the quadratic scan otherwise
        r = longest_run(cand.words, other.words)
        if r >= run_limit:
            return f"shares a run of {r} words with {other.label}"
    return None


# ---------------------------------------------------------------- family facts

def family_module(name: str):
    return importlib.import_module(f"cvbench.families.{name}")


def shared_text(family: str) -> str:
    """Text every phrasing of the family carries verbatim (feature_homography's accuracy sentence), else ''."""
    return getattr(family_module(family), "ACCURACY", "")


def squash(text: str) -> str:
    return " ".join(text.split())


def core_of(family: str, text: str) -> str:
    """``text`` without the family's shared text, on one line."""
    shared = shared_text(family)
    return squash(text.replace(shared, "")) if shared else squash(text)


def with_shared(family: str, core: str) -> str:
    shared = shared_text(family)
    return f"{core} {shared}" if shared else core


def spec_api(family: str) -> set[str]:
    return set(re.findall(r"cv2\.\w+", family_module(family).SPEC.task))


def new_cv2_names(family: str, text: str) -> list[str]:
    return sorted(set(re.findall(r"cv2\.\w+", text)) - spec_api(family))


def hint_words_in(text: str) -> list[str]:
    low = text.lower()
    return [w for w in (*HINT_WORDS, *EXTRA_HINT_WORDS) if re.search(rf"\b{re.escape(w)}\b", low)]


# Required content, mirroring the phrasing checks of tests/test_<family>.py (the 7 trainable families).

def _need(ok: bool, what: str) -> str | None:
    return None if ok else what


def _contour_area(text: str, low: str) -> str | None:
    if not ("area" in low and "outer" in low and ("boundary" in low or "edge" in low or "contour" in low)):
        return "must ask for the area inside the outer boundary"
    if not ("bright" in low or "light" in low):
        return "must say the shapes are bright (or light)"
    if not any(w in low for w in ("count", "number", "how many")):
        return "must ask for the number of shapes"
    return _need("hole" in low, "must state the hole rule")   # the test exempts index 2 only, which predates the rule


def _hough_segments(text: str, low: str) -> str | None:
    if not ("stroke" in low or "line" in low or "segment" in low):
        return "must name the strokes, lines or segments"
    if "end" not in low:
        return "must ask for the end points"
    return _need("row" in low or "N x 4" in text, "must ask for rows")


def _perspective_rectify(text: str, low: str) -> str | None:
    if not ("card" in low or "sheet" in low):
        return "must name the card"
    if not ("corner" in low and "out_w" in text and "out_h" in text):
        return "must name the corners and the out_w x out_h target"
    return _need(any(w in low for w in ("warp", "rectif", "flat", "straighten", "upright")), "must ask for the rectified image")


def _feature_homography(text: str, low: str) -> str | None:
    if not ("H[2, 2]" in text or "bottom-right entry is 1" in text):
        return "must state the H[2, 2] = 1 normalisation"
    return _need(family_module("feature_homography").ACCURACY in text, "must carry the accuracy sentence")


def _histogram_stats(text: str, low: str) -> str | None:
    if not ("BT.601" in text and "median" in low and "mean" in low):
        return "must name BT.601, the mean and the median"
    return _need(("256" in text or "0..255" in text) and "standard deviation" in low, "must name the 256 levels and the standard deviation")


def _components_stats(text: str, low: str) -> str | None:
    if not ("connect" in low and "box" in low and "centroid" in low and "background" in low):
        return "must name connectivity, the box, the centroid and the background"
    if not ("4" in text and "8" in text and "area" in low):
        return "must name both connectivities and the area"
    if not any(w in low for w in ("largest", "biggest", "descending")):
        return "must state the largest-first order"
    return _need(any(w in low for w in ("top-left", "box top", "box y", "starts higher", "top to bottom")), "must state the tie-break order")


def _text_layout(text: str, low: str) -> str | None:
    if not all(w in text for w in ("scale", "thickness", "color", "getTextSize")):
        return "must name scale, thickness, color and getTextSize"
    if not any(w in text for w in ("box", "rectangle", "region")):
        return "must name the box"
    if not ("baseline" in low and ("centre" in low or "middle" in low)):
        return "must ask to centre the getTextSize box, baseline included"
    return _need(any(w in low for w in ("largest", "biggest", "shrink", "reduce", "lower")), "must state the shrink-to-fit rule")


CONTENT_CHECKS = {
    "contour_area": _contour_area, "hough_segments": _hough_segments, "perspective_rectify": _perspective_rectify,
    "feature_homography": _feature_homography, "histogram_stats": _histogram_stats, "components_stats": _components_stats,
    "text_layout": _text_layout,
}


def content_problem(family: str, text: str) -> str | None:
    """Why ``text`` lacks what a phrasing of ``family`` must say, or None."""
    if family not in CONTENT_CHECKS:
        raise ValueError(f"no content check for {family!r} (trainable families: {sorted(CONTENT_CHECKS)})")
    return CONTENT_CHECKS[family](text, text.lower())


# ---------------------------------------------------------------- reference lines (everything a candidate is measured against)

@lru_cache(maxsize=None)
def own_lines(family: str) -> tuple[Line, ...]:
    """The family's 16 hand-written phrasings (shared text removed)."""
    ph = family_module(family).PHRASINGS
    return tuple(make_line(f"{family}[{i}]", core_of(family, ph[i])) for i in range(PHRASINGS_PER_FAMILY))


@lru_cache(maxsize=None)
def scored_lines() -> tuple[Line, ...]:
    """Every dev/test task line of every family (indices 0-11; all 16 for the held-out families, which are never trained)
    and every gotcha task line."""
    from cvbench.gotchas import CHECKS

    out = []
    for name, row in FAMILIES.items():
        ph = family_module(name).PHRASINGS
        idx = range(PHRASINGS_PER_FAMILY) if row.held_out else [*DEV_PHRASINGS, *TEST_PHRASINGS]
        out += [make_line(f"{name}[{i}]", core_of(name, ph[i])) for i in idx]
    out += [make_line(f"gotcha:{cid}", c.spec.task) for cid, c in CHECKS.items()]
    return tuple(out)


def hard_problems(family: str, text: str, *, dup_threshold: float = DUP_THRESHOLD, run_limit: int = RUN_LIMIT) -> list[str]:
    """The checks (b), (c) and (d) on one finished phrasing (shared text included). [] when it passes."""
    problems = []
    c = content_problem(family, text)
    if c:
        problems.append(f"content: {c}")
    names = new_cv2_names(family, text)
    if names:
        problems.append(f"adds cv2 names {names}")
    hints = hint_words_in(core_of(family, text))
    if hints:
        problems.append(f"hint words {hints}")
    cand = make_line("candidate", core_of(family, text))
    for other in scored_lines():
        why = closeness(cand, other, dup_threshold=dup_threshold, run_limit=run_limit)
        if why:
            problems.append(f"overlaps a scored task: {why}")
            break
    return problems


# ---------------------------------------------------------------- calibration

def calibrate() -> dict:
    """Pairwise similarity of the 16 hand-written phrasings of each trainable family, and the limits it implies."""
    import itertools

    per = {}
    for name, row in FAMILIES.items():
        if row.held_out:
            continue
        lines = own_lines(name)
        pairs = list(itertools.combinations(range(PHRASINGS_PER_FAMILY), 2))
        jac = [(jaccard(lines[i].grams, lines[j].grams), i, j) for i, j in pairs]
        runs = [(longest_run(lines[i].words, lines[j].words), i, j) for i, j in pairs]
        mj, mr = max(jac), max(runs)
        per[name] = {"pairs": len(pairs), "max_jaccard": round(mj[0], 4), "max_jaccard_pair": [mj[1], mj[2]],
                     "mean_jaccard": round(sum(x[0] for x in jac) / len(jac), 4), "max_run": mr[0], "max_run_pair": [mr[1], mr[2]]}
    top_j = max(v["max_jaccard"] for v in per.values())
    top_r = max(v["max_run"] for v in per.values())
    return {"per_family": per, "max_jaccard": top_j, "max_run": top_r,
            "recommended_threshold": math.ceil(top_j * 20 - 1e-9) / 20, "recommended_run_limit": top_r + 1}


# ---------------------------------------------------------------- the teacher's requests

SYSTEM = ("You are a careful technical editor. You reword task statements for a computer-vision coding task without changing "
          "what they ask for, and you follow the output format exactly.")

REGISTERS = (
    "terse, like a line in a specification: one short sentence or two clipped ones",
    "a question put by a colleague (\"Can you ...\", \"How many ...\", \"Where is ...\")",
    "a first-person request that describes a realistic situation (\"I have ...\", \"I need ...\")",
    "a short role-play that opens with \"As a <job>\" and states the practical need",
    "numbered steps written inside the single line, such as \"(1) ... (2) ... (3) ...\"",
    "formal requirement wording (\"Requirement: ...\", or \"Input: ... Output: ...\")",
    "a plain imperative instruction of medium length, two or three sentences",
    "long and explicit: every property of the result stated in its own clause",
)

REQUEST_TEMPLATE = """Original task statement:
{original}

Other wordings of the same task, shown for their style only:
{examples}

Write {n} new wordings of the original task statement.

Rules:
- Same task, same outputs, same meaning as the original. Add no requirement and drop none.
- Keep every parameter and output name exactly as the original writes it.
- Name no OpenCV function or constant that the original does not name, and give no implementation hints: no algorithm, library call, threshold value or processing step that the original does not state.
- Do not mention tolerances, accuracy margins, grading, benchmarks, tests, seeds or software versions.{extra}
- Do not copy sentences from the original or from the examples; reword and restructure.
- Register for this batch: {register}.
- Each wording is one paragraph on a single line.

Output exactly {n} lines, numbered "1." to "{n}.", and nothing before or after them."""


def request_messages(family: str, variant: int, n: int) -> list[dict]:
    fam = family_module(family)
    original = core_of(family, fam.SPEC.task)
    examples = "\n".join(f"- {core_of(family, t)}" for t in foundry_phrasings(family))   # indices 12-15, never 0-11
    extra = ("\n- The sentence about the required accuracy is added separately: do not write any accuracy statement."
             if shared_text(family) else "")
    user = REQUEST_TEMPLATE.format(original=original, examples=examples, n=n, extra=extra,
                                   register=REGISTERS[variant % len(REGISTERS)])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def trainable() -> list[str]:
    return [name for name, row in FAMILIES.items() if not row.held_out]


def build_requests(families: list[str] | None = None, *, variants: int = 8, per_request: int = 6, k: int = 2,
                   max_tokens: int = 1500, temperature: float = 0.9, top_p: float = 0.95) -> list[dict]:
    """Prompt rows for colab_job.py: ``variants`` requests per family, each asking for ``per_request`` wordings in a different
    register, each sampled ``k`` times, thinking off."""
    fams = families or trainable()
    bad = [f for f in fams if f not in CONTENT_CHECKS]
    if bad:
        raise ValueError(f"not trainable families: {bad}")
    rows = []
    for f in fams:
        for v in range(variants):
            rid = f"phr-{f}-{v:02d}"
            rows.append({"id": rid, "spec_id": rid, "family": f, "kind": "phrasing", "origin": "phrasings", "variant": v,
                         "n": per_request, "k": k, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                         "chat_template_kwargs": {"enable_thinking": False}, "messages": request_messages(f, v, per_request)})
    return rows


# ---------------------------------------------------------------- reading the replies

_ITEM = re.compile(r"^\s*(?:[-*]\s+)?\**\(?(\d{1,2})[.)]\**\s+(.*\S)\s*$")

def clean_item(text: str) -> str:
    """One item on one line, without markdown bold or a surrounding pair of quotes or backticks, typographic characters folded
    to ASCII (the hand-written phrasings are all ASCII)."""
    text = squash(text.replace("**", "").translate(_ASCII_FOLD))
    for q in ('"', "'", "`"):
        if len(text) > 1 and text[0] == q and text[-1] == q and q not in text[1:-1]:
            text = text[1:-1].strip()
    return squash(text)


def parse_candidates(reply: str) -> list[str]:
    """The numbered items of one reply, in order. An item continues on following non-blank lines; numbering must be 1, 2, 3 ...
    (so "(2)" inside a wording is not mistaken for an item)."""
    items: list[str] = []
    cur: str | None = None
    open_ = False
    for line in reply.splitlines():
        m = _ITEM.match(line)
        if m and int(m.group(1)) == len(items) + (1 if cur is None else 2):
            if cur is not None:
                items.append(cur)
            cur, open_ = m.group(2), True
        elif not line.strip():
            open_ = False
        elif cur is not None and open_:
            cur += " " + line.strip()
    if cur is not None:
        items.append(cur)
    return [c for c in (clean_item(i) for i in items) if c]


# ---------------------------------------------------------------- the screen

def candidates_from(rows: list[dict], family: str) -> tuple[list[dict], Counter]:
    """Candidates of one family from the completions of its phrasing requests, in round-robin order (every request's first
    wording, then every request's second ...), so the quota is not used up by one register. A reply that was cut off loses
    its last item."""
    out, stats = [], Counter()
    for r in rows:
        if r.get("kind") != "phrasing" or not str(r.get("spec_id", r["id"])).startswith(f"phr-{family}-"):
            continue
        reply = r.get("answer", r.get("text", ""))
        items = parse_candidates(reply)
        stats["replies"] += 1
        if r.get("finish_reason") == "length" and items:
            items = items[:-1]
            stats["cut_off_replies"] += 1
        if not items:
            stats["empty_replies"] += 1
        variant = int(str(r["spec_id"]).rsplit("-", 1)[1])
        for pos, text in enumerate(items):
            out.append({"core": text, "request": r["spec_id"], "variant": variant, "sample": r.get("sample", 0), "position": pos})
    out.sort(key=lambda c: (c["position"], c["variant"], c["sample"]))
    stats["candidates"] = len(out)
    return out, stats


def screen_family(family: str, candidates: list[dict], *, existing: list[str] = (), max_keep: int = MAX_PER_FAMILY,
                  dup_threshold: float = DUP_THRESHOLD, run_limit: int = RUN_LIMIT) -> tuple[list[dict], Counter, dict]:
    """Keep the candidates that pass (a)-(d). ``existing`` are phrasings already accepted (kept as they are, they count against
    the quota and against duplicates). Returns (newly accepted candidates with ``text``, rejection counts, examples)."""
    from foundry.specs import reserved_prompt_hashes   # lazy: specs imports this module

    reserved = reserved_prompt_hashes()
    spec = family_module(family).SPEC
    own = list(own_lines(family))
    accepted_lines = [make_line(f"kept[{i}]", core_of(family, t)) for i, t in enumerate(existing)]
    lens = [len(l.text) for l in own]
    lo, hi = 0.5 * min(lens), 1.5 * max(lens)
    rejected: Counter = Counter()
    examples: dict[str, list[str]] = {}
    new: list[dict] = []

    def reject(reason: str, text: str) -> None:
        key = re.split(r"[:\[]", reason)[0].strip()   # "content: ..." -> "content"; "adds cv2 names [...]" -> "adds cv2 names"
        rejected[key] += 1
        examples.setdefault(key, [])
        if len(examples[key]) < 3:
            examples[key].append(f"{text[:160]}  [{reason}]")

    for c in candidates:
        if len(existing) + len(new) >= max_keep:
            rejected["quota"] += 1
            continue
        core = core_of(family, c["core"])
        if not core.isascii():
            reject("non-ASCII characters", core)
            continue
        if not lo <= len(core) <= hi:
            reject("length out of range", core)
            continue
        final = with_shared(family, core)
        probs = hard_problems(family, final, dup_threshold=dup_threshold, run_limit=run_limit)
        if probs:
            reject(probs[0], core)
            continue
        if sha1(dataclasses.replace(spec, task=final).to_prompt()) in reserved:
            reject("prompt equals a scored prompt", core)
            continue
        line = make_line("candidate", core)
        why = next((w for w in (closeness(line, o, dup_threshold=dup_threshold, run_limit=run_limit) for o in own) if w), None)
        if why:
            reject(f"near-duplicate of the family's phrasings: {why}", core)
            continue
        why = next((w for w in (closeness(line, o, dup_threshold=dup_threshold, run_limit=run_limit) for o in accepted_lines) if w), None)
        if why:
            reject(f"near-duplicate of a kept candidate: {why}", core)
            continue
        accepted_lines.append(make_line(f"kept[{len(accepted_lines)}]", core))
        new.append({**c, "text": final})
    return new, rejected, examples


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- files

def family_file(family: str) -> Path:
    return PHRASINGS_DIR / f"{family}.json"


@lru_cache(maxsize=64)
def _load(path_str: str, mtime_ns: int, size: int) -> tuple:
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    return tuple(p["text"] for p in data["phrasings"])


def generated_phrasings(family: str) -> list[str]:
    """The family's teacher-written phrasings (phrasing indices 16, 17, ...), [] when there is no file."""
    path = family_file(family)
    if not path.is_file():
        return []
    stat = path.stat()
    return list(_load(str(path), stat.st_mtime_ns, stat.st_size))


def load_record(family: str) -> dict | None:
    path = family_file(family)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_family(family: str, existing_record: dict | None, new: list[dict], *, teacher: dict, run: str, stats: dict,
                 rejected: dict, dup_threshold: float, run_limit: int) -> dict:
    """Write foundry/phrasings/<family>.json: the phrasings already there, then the new ones. Returns the record."""
    if family not in CONTENT_CHECKS:
        raise ValueError(f"{family} is not a trainable family")
    kept = list((existing_record or {}).get("phrasings", []))
    for c in new:
        kept.append({"text": c["text"], "sha1": sha1(c["text"]), "request": c["request"], "sample": c["sample"], "position": c["position"],
                     "run": run})
    record = {"family": family, "teacher": teacher, "created": datetime.now(timezone.utc).date().isoformat(), "run": run,
              "screen": {"dup_threshold": dup_threshold, "run_limit": run_limit, "ngram": NGRAM, "max_keep": MAX_PER_FAMILY,
                         "candidates": stats.get("candidates", 0), "replies": stats.get("replies", 0), "new": len(new),
                         "rejected": dict(rejected)},
              "phrasings": kept}
    PHRASINGS_DIR.mkdir(parents=True, exist_ok=True)
    family_file(family).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return record
