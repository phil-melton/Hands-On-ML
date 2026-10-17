"""Removed-API gate for cvbench: labels OpenCV names that do not exist in 5.0.0.

Report only. The gate never blocks execution: the code still runs through the sandbox,
and this gate names the error class for the taxonomy and gives repair prompts a hint.

The catalogue is generated at import time from the two probe files, so it follows the
measurements rather than a hand-typed list:

    removed_in_5    existence probe true in 4.14.0, false in 5.0.0
    absent_in_both  existence probe false in both versions
    removed_in_5    haarcascade_xml_shipped flipped to false: the bundled Haar XML files are
                    not in the 5.0 wheel. Keyed on cv2.data.haarcascades, the path they load through.

Replacements come from the hand-written SUGGESTIONS map. Each cv2 symbol in it is checked
with hasattr against cv2 5.0 in tests/test_api_check.py. Where no replacement is confirmed,
the finding says so instead of guessing.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from cvbench.evaluate import Gate

PROBES = Path(__file__).resolve().parent.parent / "probes" / "results"
_HAAR_FILE = re.compile(r"haarcascade_\w+\.xml")


@dataclass(frozen=True)
class Suggestion:
    symbols: tuple[str, ...]  # cv2 names; the tests check each exists in cv2 5.0 (empty if none)
    note: str = ""

    @property
    def text(self) -> str:
        head = " or ".join(self.symbols)
        if head and self.note:
            return f"{head} ({self.note})"
        return head or self.note


_TRACKERS = Suggestion(
    ("cv2.TrackerMIL_create", "cv2.TrackerNano_create", "cv2.TrackerVit_create"),
    "Nano and Vit need ONNX model files; MIL needs none",
)
_FACE = Suggestion(("cv2.FaceDetectorYN.create",), "YuNet; needs an ONNX model file")
_ORB_SIFT = Suggestion(("cv2.ORB_create", "cv2.SIFT_create"))
_ONNX = Suggestion(("cv2.dnn.readNetFromONNX",), "after converting the model to ONNX")

# Hand-written. Keys are catalogue names; a key missing from the catalogue fails the tests.
SUGGESTIONS: dict[str, Suggestion] = {
    "cv2.CascadeClassifier": _FACE,
    "cv2.data.haarcascades": _FACE,
    "cv2.ml": Suggestion((), "scikit-learn or numpy"),
    "cv2.gapi": Suggestion((), "no G-API equivalent; call the cv2 functions directly"),
    "cv2.dnn.readNetFromCaffe": _ONNX,
    "cv2.dnn.readNetFromDarknet": _ONNX,
    "cv2.AKAZE_create": _ORB_SIFT,
    "cv2.KAZE_create": _ORB_SIFT,
    "cv2.BRISK_create": _ORB_SIFT,
    "cv2.AgastFeatureDetector_create": Suggestion(("cv2.FastFeatureDetector_create",)),
    "cv2.BOWKMeansTrainer": Suggestion(("cv2.kmeans",), "cluster the descriptors, then build the histogram yourself"),
    "cv2.TrackerGOTURN_create": _TRACKERS,
    "cv2.TrackerKCF_create": _TRACKERS,
    "cv2.TrackerCSRT_create": _TRACKERS,
    "cv2.xfeatures2d": Suggestion(("cv2.SIFT_create",), "for xfeatures2d.SURF_create"),
    "cv2.SURF_create": Suggestion(("cv2.SIFT_create",)),
    "cv2.cv": Suggestion((), "OpenCV 2 names; use the current cv2.* name, e.g. CV_AA became LINE_AA"),
    "cv2.CV_AA": Suggestion(("cv2.LINE_AA",)),
    "cv2.legacy": Suggestion(("cv2.TrackerMIL_create",), "legacy trackers; MIL needs no model file"),
    "cv2.createBackgroundSubtractorMOG": Suggestion(("cv2.createBackgroundSubtractorMOG2",)),
    "cv2.estimateRigidTransform": Suggestion(
        ("cv2.estimateAffinePartial2D", "cv2.estimateAffine2D"),
        "first for fullAffine=False, second for fullAffine=True"),
}


@dataclass(frozen=True)
class Entry:
    key: str          # dotted name as code spells it, e.g. "cv2.ml"
    label: str        # "removed_in_5" or "absent_in_both"
    probe: str        # the probe the label was read from
    replacement: str  # rendered suggestion, "" when none is confirmed


def _render(key: str) -> str:
    s = SUGGESTIONS.get(key)
    return s.text if s else ""


def _load_probes(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["probes"]


def _build_catalogue(p414: dict, p500: dict) -> dict[str, Entry]:
    cat: dict[str, Entry] = {}
    for probe, now in p500.items():
        if not probe.startswith("has cv2."):
            continue
        before = p414[probe]
        if now["ok"] is not True or before["ok"] is not True:
            raise ValueError(f"probe {probe!r} errored; cannot label it")
        if not isinstance(now["obs"], bool) or not isinstance(before["obs"], bool):
            raise ValueError(f"probe {probe!r} is not a boolean existence probe")
        if now["obs"]:
            continue  # exists in 5.0: never flagged
        key = probe[len("has "):]
        label = "removed_in_5" if before["obs"] else "absent_in_both"
        cat[key] = Entry(key, label, probe, _render(key))
    h4 = p414["haarcascade_xml_shipped"]["obs"]["xml_present"]
    h5 = p500["haarcascade_xml_shipped"]["obs"]["xml_present"]
    if h4 is True and h5 is False:
        key = "cv2.data.haarcascades"
        cat[key] = Entry(key, "removed_in_5", "haarcascade_xml_shipped", _render(key))
    return cat


CATALOGUE: dict[str, Entry] = _build_catalogue(
    _load_probes(PROBES / "probe_4.14.0.json"), _load_probes(PROBES / "probe_5.0.0.json"))
_LONGEST_FIRST = sorted(CATALOGUE, key=len, reverse=True)


def _match(dotted: str) -> Entry | None:
    """Longest catalogue entry the dotted name equals or sits under (cv2.ml.SVM_create -> cv2.ml)."""
    for key in _LONGEST_FIRST:
        if dotted == key or dotted.startswith(key + "."):
            return CATALOGUE[key]
    return None


@dataclass(frozen=True)
class Finding:
    line: int
    name: str         # the name as the code resolves it, e.g. "cv2.ml.SVM_create"
    api: str          # the catalogue entry it matched, e.g. "cv2.ml"
    label: str        # "removed_in_5" or "absent_in_both"
    replacement: str  # "" when no replacement is confirmed

    def __str__(self) -> str:
        target = self.replacement or "no confirmed 5.0 replacement"
        return f"line {self.line}: [{self.label}] {self.name} -> {target}"


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.names: dict[str, str] = {"cv2": "cv2"}  # local name -> dotted name it stands for
        self._seen: set[tuple[int, str]] = set()     # one finding per line per catalogue entry

    def _report(self, node: ast.AST, name: str, entry: Entry) -> None:
        line = getattr(node, "lineno", 0)
        if (line, entry.key) in self._seen:
            return
        self._seen.add((line, entry.key))
        self.findings.append(Finding(line, name, entry.key, entry.label, entry.replacement))

    def _check_dotted(self, node: ast.AST, dotted: str) -> None:
        if dotted.split(".")[0] != "cv2":
            return
        entry = _match(dotted)
        if entry is not None:
            self._report(node, dotted, entry)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.asname:
                self.names[a.asname] = a.name
            else:
                root = a.name.split(".")[0]
                self.names[root] = root
            self._check_dotted(node, a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            return
        for a in node.names:
            if a.name == "*":
                continue
            dotted = f"{node.module}.{a.name}"
            self.names[a.asname or a.name] = dotted
            self._check_dotted(node, dotted)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attrs: list[str] = []
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            attrs.append(root.attr)
            root = root.value
        if isinstance(root, ast.Name) and root.id in self.names:
            dotted = ".".join([self.names[root.id], *reversed(attrs)])
            if dotted.split(".")[0] == "cv2":
                entry = _match(dotted)
                if entry is not None:
                    self._report(node, dotted, entry)
                    return  # the whole chain is one API use; do not re-report its prefixes
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.names:
            self._check_dotted(node, self.names[node.id])

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return  # a bare string is a docstring or comment, not code
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "cv2.data.haarcascades" in CATALOGUE:
            m = _HAAR_FILE.search(node.value)
            if m:
                self._report(node, m.group(0), CATALOGUE["cv2.data.haarcascades"])


def check_removed_api(code: str) -> list[Finding]:
    """Findings sorted by line. Unparseable code returns none: the lint gate reports syntax."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    v = _Visitor()
    v.visit(tree)
    return sorted(v.findings, key=lambda f: (f.line, f.api))


def api_gate(code: str) -> Gate:
    findings = check_removed_api(code)
    return Gate(not findings, "; ".join(map(str, findings)) or "clean")
