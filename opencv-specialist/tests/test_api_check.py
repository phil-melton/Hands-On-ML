import json

import cv2
import pytest

from cvbench.api_check import CATALOGUE, PROBES, SUGGESTIONS, api_gate, check_removed_api
from cvbench.evaluate import Gate
from cvbench.families import contour_area as fam

REMOVED = [
    "cv2.CascadeClassifier", "cv2.HOGDescriptor", "cv2.ml", "cv2.gapi",
    "cv2.dnn.readNetFromCaffe", "cv2.dnn.readNetFromDarknet", "cv2.AKAZE_create", "cv2.KAZE_create",
    "cv2.BRISK_create", "cv2.AgastFeatureDetector_create", "cv2.BOWKMeansTrainer", "cv2.TrackerGOTURN_create",
]
ABSENT = [
    "cv2.xfeatures2d", "cv2.SURF_create", "cv2.CV_AA", "cv2.TrackerKCF_create", "cv2.TrackerCSRT_create",
    "cv2.legacy", "cv2.createBackgroundSubtractorMOG", "cv2.estimateRigidTransform", "cv2.ximgproc",
]
PRESENT = [
    "cv2.ORB_create", "cv2.SIFT_create", "cv2.FaceDetectorYN", "cv2.FaceDetectorYN.create", "cv2.findContours",
    "cv2.cvtColor", "cv2.dnn.readNetFromONNX", "cv2.dnn.readNet", "cv2.LINE_AA",
    "cv2.createBackgroundSubtractorMOG2", "cv2.FastFeatureDetector_create", "cv2.TrackerMIL_create",
    "cv2.TrackerVit_create",
]

AKAZE = """\
import cv2
import numpy as np

def solve(img):
    akaze = cv2.AKAZE_create()
    kp = akaze.detect(img, None)
    return {"count": len(kp)}
"""


def _exists(dotted: str) -> bool:
    obj = cv2
    for part in dotted.split(".")[1:]:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def test_catalogue_is_derived_from_the_probe_files():
    p414 = json.loads((PROBES / "probe_4.14.0.json").read_text(encoding="utf-8"))["probes"]
    p500 = json.loads((PROBES / "probe_5.0.0.json").read_text(encoding="utf-8"))["probes"]
    expected = {}
    for probe, now in p500.items():
        if probe.startswith("has cv2.") and now["obs"] is False:
            expected[probe[4:]] = "removed_in_5" if p414[probe]["obs"] is True else "absent_in_both"
    xml4 = p414["haarcascade_xml_shipped"]["obs"]["xml_present"]
    xml5 = p500["haarcascade_xml_shipped"]["obs"]["xml_present"]
    if xml4 and not xml5:
        expected["cv2.data.haarcascades"] = "removed_in_5"
    assert {k: e.label for k, e in CATALOGUE.items()} == expected
    assert sum(e.label == "removed_in_5" for e in CATALOGUE.values()) == 13  # 12 API removals + Haar XML
    assert sum(e.label == "absent_in_both" for e in CATALOGUE.values()) == 11


@pytest.mark.parametrize("api", REMOVED)
def test_known_removals_are_labelled_removed_in_5(api):
    assert CATALOGUE[api].label == "removed_in_5"


@pytest.mark.parametrize("api", ABSENT)
def test_absent_in_both_are_labelled(api):
    assert CATALOGUE[api].label == "absent_in_both"


def test_haar_xml_loading_is_flagged():
    assert CATALOGUE["cv2.data.haarcascades"].label == "removed_in_5"
    fs = check_removed_api("import cv2\npath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'\n")
    assert [(f.line, f.api) for f in fs] == [(2, "cv2.data.haarcascades")]
    fs = check_removed_api("import cv2\npath = '/opt/haarcascade_eye.xml'\n")
    assert [(f.line, f.api) for f in fs] == [(2, "cv2.data.haarcascades")]


def test_suggestion_keys_are_all_catalogued():
    assert set(SUGGESTIONS) <= set(CATALOGUE)


def test_every_suggested_symbol_exists_in_cv2_5():
    assert cv2.__version__.split(".")[0] == "5", cv2.__version__
    missing = {k: [s for s in sug.symbols if not _exists(s)] for k, sug in SUGGESTIONS.items()}
    assert {k: v for k, v in missing.items() if v} == {}


def test_replacements_are_not_themselves_flagged():
    for key, sug in SUGGESTIONS.items():
        for s in sug.symbols:
            assert check_removed_api(f"import cv2\nf = {s}\n") == [], (key, s)


@pytest.mark.parametrize("api", PRESENT)
def test_present_apis_are_not_flagged(api):
    assert check_removed_api(f"import cv2\nf = {api}\n") == []


@pytest.mark.parametrize("code, api", [
    ("import cv2 as cv\nx = cv.AKAZE_create()\n", "cv2.AKAZE_create"),
    ("from cv2 import AKAZE_create as mk\nx = mk()\n", "cv2.AKAZE_create"),
    ("from cv2 import ml\nm = ml.SVM_create()\n", "cv2.ml"),
    ("import cv2.ml as m\nk = m.KNearest_create()\n", "cv2.ml"),
    ("import cv2 as c\nk = c.ml.SVM_create()\n", "cv2.ml"),
    ("from cv2.dnn import readNetFromDarknet as load\nn = load('a.cfg', 'a.weights')\n",
     "cv2.dnn.readNetFromDarknet"),
    ("import cv2\nm = cv2.ml.SVM_create().train(x, 0, y)\n", "cv2.ml"),
    ("import cv2\nx = cv2.xfeatures2d.SURF_create(400)\n", "cv2.xfeatures2d"),
    ("import cv2\nx = cv2.cv.CV_AA\n", "cv2.cv"),
    ("import cv2\nx = cv2.CV_AA\n", "cv2.CV_AA"),
    ("import cv2\nd = cv2.dnn.readNetFromCaffe('a.prototxt', 'a.caffemodel')\n", "cv2.dnn.readNetFromCaffe"),
    ("import cv2\nt = cv2.TrackerGOTURN_create()\n", "cv2.TrackerGOTURN_create"),
    ("import cv2\nd = cv2.createBackgroundSubtractorMOG()\n", "cv2.createBackgroundSubtractorMOG"),
])
def test_aliases_and_from_imports_are_caught(code, api):
    apis = {f.api for f in check_removed_api(code)}
    assert api in apis, check_removed_api(code)


def test_from_import_is_reported_at_import_and_at_use():
    fs = check_removed_api("from cv2 import AKAZE_create\nx = AKAZE_create()\n")
    assert [(f.line, f.api) for f in fs] == [(1, "cv2.AKAZE_create"), (2, "cv2.AKAZE_create")]


def test_finding_carries_line_api_label_and_replacement():
    fs = check_removed_api(AKAZE)
    assert len(fs) == 1
    f = fs[0]
    assert (f.line, f.api, f.label, f.replacement) == (
        5, "cv2.AKAZE_create", "removed_in_5", "cv2.ORB_create or cv2.SIFT_create")


def test_no_confirmed_replacement_is_said_so():
    (f,) = check_removed_api("import cv2\nh = cv2.HOGDescriptor()\n")
    assert f.replacement == "" and "no confirmed 5.0 replacement" in str(f)


def test_docstring_and_syntax_error_are_not_flagged():
    assert check_removed_api('"""Loads haarcascade_frontalface_default.xml."""\ndef solve():\n    return {}\n') == []
    assert check_removed_api("def solve(:\n    cv2.AKAZE_create()\n") == []


def test_api_gate_passes_contour_area_reference():
    g = api_gate(fam.REFERENCE)
    assert isinstance(g, Gate) and g.passed and g.detail == "clean"


def test_api_gate_fails_on_akaze_and_names_it():
    g = api_gate(AKAZE)
    assert isinstance(g, Gate) and not g.passed and str(g) == "FAIL"
    assert "line 5" in g.detail and "cv2.AKAZE_create" in g.detail and "cv2.ORB_create" in g.detail
