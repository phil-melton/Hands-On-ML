"""Why did a reply fail? A rule-based classifier over a grade record, keeping the evidence string.

PLAN.md picks the lever from the error class: removed API / wrong convention -> fine-tune (prior), wrong
algorithm -> a bigger model (capacity), anything measurable -> a repair loop (feedback). So the classes are:

    format      no code, a syntax error, or no top-level solve()
    api         a hallucinated or removed cv2 name or call signature
    lint        forbidden behaviour announced or attempted (imports, file/GUI calls, blanket except)
    timeout     the sandbox killed it
    convention  it ran or crashed on an output/shape/dtype/ordering/signature contract (e.g. (N,1,4) vs (N,4))
    algorithm   it ran, met the contract, and the manufactured or metamorphic gate rejected the numbers
    other       none of the above (NameError, ZeroDivisionError, a killed child, a crashed grader ...)

Every signal found becomes a candidate (class, rule, evidence); the primary class is the one earliest in
PRIORITY, the rest are kept in ``also``. The order puts the most specific root cause first: a removed API
usually *causes* the crash that follows, and a blanket ``except`` hides whatever failed underneath it. The
stored ``taxonomy`` field is a convenience: ``report`` re-classifies from the raw gate details, so a rule
change applies to old grades without regrading.

Input is the grade record written by bakeoff.grade. Two shapes: a family grade carries ``gates``
({name: {passed, detail}} from cvbench.evaluate), a gotcha grade carries ``stage`` and ``detail`` (cvbench.gotchas).
The gate details are text built by sandbox.run.RunResult.summary() ("exception on input 0: IndexError ...") or by
a family verifier ("scene 4000003: <reason>"); the patterns below were read off those producers and checked against
every family mutant and every gotcha wrong variant (236 graded snippets, see tests/test_bakeoff.py).

Known ambiguity: the graders hand back text, not a typed reason, so the split between `convention` and `algorithm` for a
verifier rejection rests on the wording ("contract:", "has shape", "not sorted", "wrong order" ...). A gotcha check whose
wrong idiom gives right-shaped but wrong values (uint8 wrap, int64 overflow, a view's write-through, a changed default)
lands in `algorithm` under the rule name `gotcha-value`; read gotcha counts separately from family counts.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

CLASSES = ("format", "api", "lint", "timeout", "convention", "algorithm", "other")
PRIORITY = {c: i for i, c in enumerate(CLASSES)}  # earlier = reported first when several signals fire

RUN_GATES = ("exec-only", "manufactured", "metamorphic")  # the gates that execute the code, in grading order


@dataclass(frozen=True)
class Classification:
    cls: str
    rule: str      # which rule fired (stable names, for audits)
    evidence: str  # the grader text that decided it, at most 300 characters
    also: tuple[str, ...] = field(default=())  # "class:rule" of every other signal that fired

    def as_dict(self) -> dict:
        return {"class": self.cls, "rule": self.rule, "evidence": self.evidence, "also": list(self.also)}


# ---------------------------------------------------------------- patterns read off the graders

_RUN = re.compile(r"\A(?P<status>exception|violation|bad_output|timeout|killed)(?: on input (?P<item>\d+))?:\s*(?P<rest>.*)\Z", re.S)
_CV2_ATTR = re.compile(r"module 'cv2[\w.]*' has no attribute|'cv2\.[\w.]+' object has no attribute")
_SOLVE_SIG = re.compile(r"\bsolve\(\) (?:got an unexpected keyword argument|missing \d+ required|takes )")
_NOT_IMPLEMENTED = re.compile(r"\(-213|not implemented", re.I)
# a wrong cv2 call signature (names/arity), as opposed to a wrong argument *type* ("Can't parse 'org'. Sequence item
# with index 0 has a wrong type", "Layout of the output array img is incompatible"), which is a dtype/layout convention
_CALL_SIG = re.compile(r"unexpected keyword argument|invalid keyword argument|Required argument|given by name|"
                       r"takes (?:at most|at least|exactly) \d+|got multiple values for")
# IndexError, KeyError and AttributeError (on a non-cv2 object, e.g. None or a tuple) are shape/type confusion by
# nature; ValueError and TypeError only when the message is about shapes, unpacking, dtypes or indexing
# (ValueError("chessboard not found") is the code giving up, not a convention)
_ALWAYS_CONVENTION = {"IndexError", "KeyError", "AttributeError", "AxisError"}
_SHAPE_MESSAGE = re.compile(
    r"unpack|broadcast|reshape|shape|dimension|\baxis\b|\baxes\b|\bindex\b|indices|non-iterable|not iterable|subscriptable|"
    r"has no len|not callable|inhomogeneous|setting an array element|\bcast\b|dtype|\bscalar\b|NoneType|out of range|"
    r"out of bounds|too many|not enough|operand|concatenat|mismatch|same (?:number|size|shape)", re.I)
# verifier reasons about the contract: shapes, dtypes, key names, ordering, labels (all families use these words)
_CONVENTION_REASON = re.compile(
    r"contract:|\bnot sorted\b|\bnot ordered\b|\bsorted by\b|stated order|order is wrong|wrong order|"
    r"TL, TR, BR, BL order|must be the end point|has shape|\bshape\b.*\bexpected\b|shapes disagree|"
    r"text_size has \d+ values|\d+ but \d+ rows|not a 0/255 mask|\bdtype\b", re.I)
_LINT_FORMAT_RULES = {"syntax", "contract"}  # lint rules that mean "there is no usable function", not "forbidden behaviour"


def parse_run_detail(detail: str) -> dict | None:
    """Split RunResult.summary() text into status, input index, exception class and message; None if not one."""
    m = _RUN.match((detail or "").strip())
    if not m:
        return None
    status, rest = m["status"], m["rest"].strip()
    etype, msg = None, rest
    if status in ("exception", "violation", "bad_output"):
        parts = rest.split(None, 1)
        etype, msg = (parts[0] if parts else None), (parts[1] if len(parts) > 1 else "")
    return {"status": status, "item": int(m["item"]) if m["item"] else None, "etype": etype, "msg": msg}


def _clip(text: str, n: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def classify_exception(etype: str | None, msg: str) -> tuple[str, str]:
    """(class, rule) for a Python exception raised by the code under test."""
    if etype in ("SyntaxError", "IndentationError", "TabError"):
        return "format", "syntax-at-runtime"
    if etype in ("ImportError", "ModuleNotFoundError"):
        return "api", "import"
    if etype == "AttributeError" and _CV2_ATTR.search(msg):
        return "api", "cv2-attribute"
    if _SOLVE_SIG.search(msg):
        return "convention", "solve-signature"
    if etype == "error" or msg.startswith("OpenCV("):  # cv2.error
        if _NOT_IMPLEMENTED.search(msg):
            return "api", "cv2-not-implemented"
        if _CALL_SIG.search(msg):
            return "api", "cv2-call-signature"
        return "convention", "cv2-assertion"  # -215 assertion, -210 unsupported format, -5 bad argument type, -15 channels ...
    if etype == "TypeError" and _CALL_SIG.search(msg):
        return "api", "call-signature"
    if etype in _ALWAYS_CONVENTION or (etype in ("ValueError", "TypeError") and _SHAPE_MESSAGE.search(msg)):
        return "convention", f"{etype}"
    return "other", f"exception:{etype}"


def classify_run(run: dict) -> tuple[str, str]:
    """(class, rule) for a parsed RunResult summary."""
    status = run["status"]
    if status == "timeout":
        return "timeout", "timeout"
    if status == "violation":
        return "lint", "sandbox-violation"
    if status == "bad_output":
        if "no callable solve" in run["msg"]:
            return "format", "no-solve"
        return "convention", "bad-output"
    if status == "killed":
        return "other", "killed"
    return classify_exception(run["etype"], run["msg"])


def classify_reason(reason: str) -> tuple[str, str]:
    """(class, rule) for a verifier's rejection text: a contract problem is a convention, wrong numbers are the algorithm."""
    if _CONVENTION_REASON.search(reason):
        return "convention", "verifier-contract"
    return "algorithm", "verifier-value"


def _detail_candidate(detail: str) -> tuple[str, str, str]:
    run = parse_run_detail(detail)
    if run is not None:
        cls, rule = classify_run(run)
        return cls, rule, _clip(detail)
    cls, rule = classify_reason(re.sub(r"\Ascene \d+:\s*", "", detail or ""))
    return cls, rule, _clip(detail)


def _static_candidates(rec: dict, lint_failed: bool) -> list[tuple[str, str, str]]:
    out = []
    for f in rec.get("api_findings") or []:
        out.append(("api", f"removed-api:{f['label']}", _clip(f"line {f['line']}: {f['name']} -> {f['replacement'] or 'no confirmed 5.0 replacement'}")))
    if lint_failed:
        for f in rec.get("lint_findings") or []:
            cls = "format" if f["rule"] in _LINT_FORMAT_RULES else "lint"
            out.append((cls, f"lint:{f['rule']}", _clip(f"line {f['line']}: [{f['rule']}] {f['message']}")))
    return out


def classify(rec: dict) -> Classification | None:
    """Classify one failing grade record (None if it passed)."""
    if rec.get("passed"):
        return None
    cands: list[tuple[str, str, str]] = []
    if rec.get("code_empty"):
        cands.append(("format", "no-code", "the reply contained no code"))
    elif rec.get("grader_error"):
        cands.append(("other", "grader-error", _clip(rec["grader_error"].strip().splitlines()[-1])))
    elif "gates" in rec and rec["gates"]:  # a family grade
        gates = rec["gates"]
        cands += _static_candidates(rec, lint_failed=not gates.get("lint", {}).get("passed", True))
        first = next((g for g in RUN_GATES if not gates.get(g, {}).get("passed", True)), None)
        if first:
            c = _detail_candidate(gates[first]["detail"])
            cands.append((c[0], c[1], f"{first}: {c[2]}"))
    else:  # a gotcha grade
        stage = rec.get("stage")
        detail = rec.get("detail", "")
        cands += _static_candidates(rec, lint_failed=stage == "lint")
        if stage in ("run", "contract", "verify", "malformed"):
            if stage == "run":
                cands.append(_detail_candidate(detail))
            elif stage == "verify":
                reason = re.sub(r"\Aitem \d+:\s*", "", detail)
                cls, rule = classify_reason(reason)
                # a gotcha check is built so that the wrong idiom (uint8 wrap, a view, a changed default) gives wrong
                # *values* with the right shape: `algorithm` here may really be a dtype or behaviour convention
                cands.append((cls, "gotcha-value" if rule == "verifier-value" else rule, _clip(detail)))
            else:  # contract problems, or output the verifier could not read at all
                cands.append(("convention", f"stage:{stage}", _clip(detail)))
    if not cands:
        cands.append(("other", "unexplained", _clip(rec.get("detail") or "failed with no recorded reason")))
    cands.sort(key=lambda c: PRIORITY[c[0]])  # stable: ties keep discovery order
    first, rest = cands[0], cands[1:]
    return Classification(first[0], first[1], first[2], tuple(dict.fromkeys(f"{c}:{r}" for c, r, _ in rest)))


def sample_failures(records: list[dict], k: int = 3, seed: int = 0) -> dict[str, list[tuple[dict, Classification]]]:
    """Up to k failing records per class, drawn at random with a fixed seed, for a human to audit the rules."""
    by_class: dict[str, list[tuple[dict, Classification]]] = {c: [] for c in CLASSES}
    for rec in sorted(records, key=lambda r: (r["rung"], r["task_id"], r["mode"], r["sample"])):
        c = classify(rec)
        if c is not None:
            by_class[c.cls].append((rec, c))
    rng = random.Random(seed)
    return {cls: sorted(rng.sample(rows, min(k, len(rows))), key=lambda t: (t[0]["rung"], t[0]["task_id"], t[0]["sample"]))
            for cls, rows in by_class.items()}
