"""Stage 7: the MCP server (serve/server.py) and its runtime metamorphic checks (serve/metamorphic.py).

Ollama is never called: the HTTP layer is an httpx.MockTransport. The sandbox is real (generated code runs through
``sandbox.run_solution`` as in production). Needs the `mcp` SDK, so the module skips itself in ~/venvs/cv5.
"""
import asyncio
import json
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402

from bakeoff.generate import build_request as eval_build_request  # noqa: E402
from bakeoff.rungs import Rung, request_options  # noqa: E402
from cvbench.contract import IOSpec, TaskSpec  # noqa: E402
from foundry.answer import format_answer  # noqa: E402
from foundry.repair import FEEDBACK  # noqa: E402
from serve import metamorphic as mm  # noqa: E402
from serve import server  # noqa: E402
from train.config import SYSTEM  # noqa: E402

TASK = "Count the bright blobs and give the area of each, largest first."
OUTPUTS = {"areas": "float, shape N. px^2, largest first", "count": "int, shape scalar"}

GOOD = textwrap.dedent('''\
    import cv2
    import numpy as np

    def solve(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        return {"areas": np.sort(areas)[::-1], "count": int(n - 1)}
    ''')

CRASH = textwrap.dedent('''\
    import numpy as np

    def solve(img):
        return {"areas": np.zeros(3)[5], "count": 1}
    ''')

WRONG_DTYPE = textwrap.dedent('''\
    import numpy as np

    def solve(img):
        return {"areas": np.array([4, 2]), "count": 2}
    ''')


def fence(code: str) -> str:
    return "```python\n" + code.rstrip("\n") + "\n```"


def make_image(h=60, w=80) -> np.ndarray:
    """Three bright rectangles (areas 200, 100, 36) on black: asymmetric, so rot90 / mirror move everything."""
    img = np.zeros((h, w, 3), np.uint8)
    img[5:15, 10:30] = 255      # 10 x 20 = 200 px^2
    img[30:40, 50:60] = 255     # 10 x 10 = 100
    img[45:51, 12:18] = 255     # 6 x 6 = 36
    return img


@pytest.fixture
def img_path(tmp_path) -> str:
    p = tmp_path / "blobs.png"
    cv2.imwrite(str(p), make_image())
    return str(p)


@pytest.fixture
def ollama(monkeypatch, tmp_path):
    """Replace the HTTP layer. `ollama.replies` is the list of assistant texts served in order; `ollama.calls` the
    decoded /api/chat request bodies."""
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(server, "RETRY_BACKOFF_S", 0.0)

    class Fake:
        replies: list = []
        calls: list = []
        chat_status = 200

    fake = Fake()
    fake.replies, fake.calls = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "other:latest", "digest": "sha256:000"},
                {"name": "opencv-expert:latest", "digest": "sha256:abc123", "size": 6_000_000_000,
                 "details": {"quantization_level": "Q5_K_M"}}]})
        assert request.url.path == "/api/chat"
        fake.calls.append(json.loads(request.content))
        if fake.chat_status != 200:
            return httpx.Response(fake.chat_status, json={"error": "model 'opencv-expert' not found"})
        text = fake.replies.pop(0) if fake.replies else "no more scripted replies"
        return httpx.Response(200, json={"model": "opencv-expert", "message": {"role": "assistant", "content": text},
                                         "done": True, "done_reason": "stop", "eval_count": 7})

    monkeypatch.setattr(server, "HTTP_TRANSPORT", httpx.MockTransport(handler))
    return fake


def solve(img_path, ollama, replies, *, outputs=None, **kw):
    ollama.replies[:] = list(replies)
    kw.setdefault("metamorphic", False)
    return server.opencv_solve(task=TASK, images={"img": img_path}, outputs=outputs or OUTPUTS, **kw)


# ------------------------------------------------------------------------------------------- prompt and request

def test_prompt_is_taskspec_to_prompt(img_path, ollama):
    r = solve(img_path, ollama, [fence(GOOD)])
    expected = TaskSpec("serve", TASK, {"img": IOSpec("uint8", "HxWx3", 3, "BGR colour image")},
                        {"areas": IOSpec("float", "N", 1, "px^2, largest first"),
                         "count": IOSpec("int", "scalar", None, "")}).to_prompt()
    sent = ollama.calls[0]["messages"]
    assert sent[1] == {"role": "user", "content": expected}
    assert sent[0] == {"role": "system", "content": SYSTEM}
    assert expected == (
        "Task: Count the bright blobs and give the area of each, largest first.\n"
        "OpenCV: 5.0 (Python cv2, headless)\n"
        "Inputs:\n"
        "  - img: uint8, shape HxWx3. BGR colour image\n"
        "Return a dict with:\n"
        "  - 'areas': float, shape N. px^2, largest first\n"
        "  - 'count': int, shape scalar\n"
        "Write one function, `def solve(img) -> dict`, with the imports it needs. No file, GUI or network access; "
        "do not catch exceptions to hide failures.")
    assert (Path(r["run_dir"]) / "prompt.txt").read_text() == expected


def test_two_images_and_params_make_the_signature(tmp_path, ollama):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(a), make_image())
    cv2.imwrite(str(b), make_image(), )
    ollama.replies[:] = ["no code"]
    server.opencv_solve(task="Align b to a.", images={"img_a": str(a), "img_b": str(b)},
                        outputs={"H": "float, shape 3x3"}, params={"ratio": 0.75, "n": 5, "mode": "fast"},
                        metamorphic=False)
    prompt = ollama.calls[0]["messages"][1]["content"]
    assert "  - img_a: uint8, shape HxWx3. BGR colour image\n  - img_b: uint8, shape HxWx3. BGR colour image\n" in prompt
    assert "  - ratio: float, shape scalar\n  - n: int, shape scalar\n  - mode: str, shape scalar\n" in prompt
    assert "`def solve(img_a, img_b, ratio, n, mode) -> dict`" in prompt


def test_grayscale_image_is_described_as_such(tmp_path, ollama):
    p = tmp_path / "g.png"
    cv2.imwrite(str(p), make_image()[:, :, 0])
    ollama.replies[:] = ["no code"]
    server.opencv_solve(task="t", images={"gray": str(p)}, outputs={"n": "int, shape scalar"}, metamorphic=False)
    assert "  - gray: uint8, shape HxW. grayscale image\n" in ollama.calls[0]["messages"][1]["content"]


def test_request_is_the_evaluated_request(img_path, ollama):
    solve(img_path, ollama, [fence(GOOD)])
    body = ollama.calls[0]
    assert "format" not in body                      # no constrained decoding: the model answers with a fenced block
    assert body["think"] is False and body["stream"] is False and body["model"] == "opencv-expert"
    assert body["options"] == request_options("greedy", num_predict=2048, num_ctx=8192, seed=body["options"]["seed"])
    assert body["options"]["temperature"] == 0.0 and body["options"]["repeat_penalty"] == 1.0
    # byte-for-byte what bakeoff.generate sends for a Stage 6 rung with the same prompt
    rung = Rung("s6-q5", "opencv-expert", think=False)
    assert body == eval_build_request(rung, SYSTEM, body["messages"][1]["content"], body["options"], server.KEEP_ALIVE)


# --------------------------------------------------------------------------------------------- the main path

def test_correct_reply_is_verified_with_measurements(img_path, ollama):
    r = solve(img_path, ollama, [fence(GOOD)])
    assert r["verified"] is True and r["attempts"] == 1 and r["log"] == [] and r["error"] is None
    assert r["measurements"]["count"] == 3
    assert r["measurements"]["areas"] == [200.0, 100.0, 36.0]
    assert r["code"].strip() == GOOD.strip()
    assert r["model"]["tag"] == "opencv-expert" and r["model"]["digest"] == "sha256:abc123"
    assert r["timing"]["total_s"] > 0 and r["timing"]["sandbox_s"] > 0
    run_dir = Path(r["run_dir"])
    assert (run_dir / "attempt1.py").read_text().strip() == GOOD.strip()
    assert json.loads((run_dir / "result.json").read_text())["verified"] is True
    assert r["metamorphic"] == {"skipped": "metamorphic=False"}
    assert len(ollama.calls) == 1


def test_bare_code_without_a_fence_is_accepted(img_path, ollama):
    r = solve(img_path, ollama, [GOOD])
    assert r["verified"] is True and r["attempts"] == 1


def test_metamorphic_runs_after_success_by_default(img_path, ollama):
    r = solve(img_path, ollama, [fence(GOOD)], metamorphic=True)
    m = r["metamorphic"]
    assert r["verified"] and m["stable_fraction"] == 1.0 and m["verdict"].startswith("stable")
    assert {t: c["areas"]["class"] for t, c in m["checks"].items()} == {
        "rot90": "invariant", "mirror": "invariant", "dim": "invariant"}


def test_image_and_large_outputs_are_saved_and_summarised(img_path, ollama):
    code = textwrap.dedent('''\
        import numpy as np

        def solve(img):
            mask = (img[:, :, 0] > 100).astype(np.uint8) * 255
            return {"mask": mask, "big": np.arange(100, dtype=np.float32), "pts": np.array([[1.5, 2.5]]),
                    "flag": np.ones((20, 20), bool), "name": "x"}
        ''')
    r = solve(img_path, ollama, [fence(code)], outputs={"mask": "uint8, shape HxW", "big": "float, shape N",
                                                        "pts": "float, shape Nx2", "flag": "bool, shape HxW",
                                                        "name": "str, shape scalar. a label"})
    assert r["verified"], r["log"]
    m = r["measurements"]
    assert Path(m["mask"]["image"]).is_file() and cv2.imread(m["mask"]["image"], 0).shape == (60, 80)
    assert m["mask"]["shape"] == [60, 80] and m["mask"]["dtype"] == "uint8"
    assert m["big"] == {"shape": [100], "dtype": "float32", "min": 0.0, "max": 99.0, "mean": 49.5}
    assert m["pts"] == [[1.5, 2.5]] and m["name"] == "x"
    assert m["flag"]["true_count"] == 400 and Path(m["flag"]["image"]).is_file()
    assert any("cannot test it" in w for w in r["warnings"])    # dtype str is shown to the model, only the key is checked


def test_floats_come_back_at_full_precision(img_path, ollama):
    # the orchestrator may have no other copy of a value (E2E forced delegation, 2026-10-23: 8 digits failed 1e-9 checks)
    code = textwrap.dedent('''\
        import numpy as np

        def solve(img):
            return {"mean": 46.541373155555675, "pair": [1 / 3, 2 / 3], "f32": np.float32(0.1)}
        ''')
    r = solve(img_path, ollama, [fence(code)], outputs={"mean": "float, shape scalar", "pair": "float, shape N",
                                                        "f32": "any, shape scalar"})
    assert r["verified"], r["log"]
    m = r["measurements"]
    assert m["mean"] == 46.541373155555675 and m["pair"] == [1 / 3, 2 / 3]
    assert m["f32"] == float(np.float32(0.1))          # a float32 result is passed on exactly, not re-rounded


def test_save_dir_copies_image_outputs(img_path, ollama, tmp_path):
    code = textwrap.dedent('''\
        import numpy as np

        def solve(img):
            return {"out": (img[:, :, 0] > 100).astype(np.uint8) * 255, "n": 1}
        ''')
    dest = tmp_path / "work"
    dest.mkdir()
    r = solve(img_path, ollama, [fence(code)], outputs={"out": "uint8, shape HxW", "n": "int, shape scalar"},
              save_dir=str(dest))
    assert r["verified"], r["log"]
    assert r["measurements"]["out"]["saved_to"] == str(dest / "out.png")
    assert (cv2.imread(str(dest / "out.png"), 0) == cv2.imread(r["measurements"]["out"]["image"], 0)).all()
    r = solve(img_path, ollama, [fence(code)], outputs={"out": "uint8, shape HxW", "n": "int, shape scalar"},
              save_dir=str(tmp_path / "missing"))
    assert r["verified"] and "saved_to" not in r["measurements"]["out"]
    assert any("not an existing directory" in w for w in r["warnings"])


# --------------------------------------------------------------------------------------------------- repairs

def test_repair_round_uses_the_training_repair_format(img_path, ollama):
    r = solve(img_path, ollama, [fence(CRASH), fence(GOOD)])
    assert r["verified"] is True and r["attempts"] == 2
    assert len(r["log"]) == 1 and r["log"][0].startswith("attempt 1: exception on input 0: IndexError")
    first, second = ollama.calls
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][:2] == first["messages"]                              # same system and task prompt
    assert second["messages"][2] == {"role": "assistant", "content": format_answer(CRASH)}  # the failed code, fenced
    message = r["log"][0][len("attempt 1: "):]
    assert second["messages"][3] == {"role": "user", "content": FEEDBACK.format(message=message)}
    assert second["messages"][3]["content"].startswith("That failed: exception on input 0: IndexError")
    assert second["messages"][3]["content"].endswith("\nReturn the corrected full function.")
    assert "format" not in second and second["think"] is False


def test_contract_violation_is_reported_as_contract_feedback(img_path, ollama):
    r = solve(img_path, ollama, [fence(WRONG_DTYPE), fence(GOOD)])
    assert r["verified"] and r["attempts"] == 2
    assert ollama.calls[1]["messages"][3]["content"] == (
        "That failed: contract: 'areas' should be float, got int64\nReturn the corrected full function.")


def test_removed_api_hint_is_appended_like_in_training(img_path, ollama):
    code = "import cv2\n\ndef solve(img):\n    knn = cv2.ml.KNearest_create()\n    return {'areas': 1.0, 'count': 1}\n"
    r = solve(img_path, ollama, [fence(code), fence(GOOD)])
    fb = ollama.calls[1]["messages"][3]["content"]
    assert "exception on input 0: AttributeError" in fb
    assert "(`cv2.ml.KNearest_create` is not in OpenCV 5.0; use scikit-learn or numpy)" in fb
    assert r["verified"]


def test_lint_failure_feedback_is_the_lint_summary(img_path, ollama):
    code = "import cv2\n\ndef solve(img):\n    try:\n        x = 1 / 0\n    except:\n        x = 0\n    return {'areas': x, 'count': 1}\n"
    solve(img_path, ollama, [fence(code), fence(GOOD)])
    assert ollama.calls[1]["messages"][3]["content"].startswith("That failed: lint: line 6: [broad-except]")


def test_repair_stops_when_the_model_repeats_itself(img_path, ollama):
    r = solve(img_path, ollama, [fence(CRASH), fence(CRASH), fence(GOOD)])
    assert r["verified"] is False and r["attempts"] == 2 and len(ollama.calls) == 2
    assert "identical code to attempt 1" in r["log"][-1]


def test_max_repairs_zero_makes_one_attempt(img_path, ollama):
    r = solve(img_path, ollama, [fence(CRASH), fence(GOOD)], max_repairs=0)
    assert r["verified"] is False and r["attempts"] == 1 and len(ollama.calls) == 1 and len(r["log"]) == 1
    assert r["code"].strip() == CRASH.strip() and r["measurements"] == {}


def test_every_attempt_failing_returns_a_clean_result(img_path, ollama):
    codes = [CRASH.replace("[5]", f"[{i + 4}]") for i in range(4)]
    r = solve(img_path, ollama, [fence(c) for c in codes])
    assert r["verified"] is False and r["attempts"] == 4 and len(r["log"]) == 4   # 1 + max_repairs(3)
    assert r["metamorphic"] == {"skipped": "the code was not verified"}


# -------------------------------------------------------------------------------------------- failure results

def test_reply_without_a_code_block(img_path, ollama):
    r = solve(img_path, ollama, ["I am sorry, I can only describe the image in words."])
    assert r["verified"] is False and r["attempts"] == 1 and r["code"] == ""
    assert "no code" in r["log"][0] and "no Python code" in r["error"]
    assert len(ollama.calls) == 1            # not repaired: the training never repaired prose


def test_empty_reply(img_path, ollama):
    r = solve(img_path, ollama, [""])
    assert r["verified"] is False and "no code" in r["log"][0]


@pytest.mark.parametrize("make, expect", [
    (lambda tmp: str(tmp / "missing.png"), "no such file"),
    (lambda tmp: (tmp / "notimg.png").write_text("not an image") and str(tmp / "notimg.png"), "cannot decode"),
    (lambda tmp: "", "path is empty"),
])
def test_unreadable_image_is_a_clean_error(tmp_path, ollama, make, expect):
    r = server.opencv_solve(task=TASK, images={"img": make(tmp_path)}, outputs=OUTPUTS)
    assert r["verified"] is False and r["attempts"] == 0 and expect in r["error"]
    assert r["log"] == [r["error"]] and r["measurements"] == {} and ollama.calls == []


def test_sixteen_bit_image_is_refused(tmp_path, ollama):
    p = tmp_path / "deep.png"
    cv2.imwrite(str(p), (make_image()[:, :, 0].astype(np.uint16)) * 257)
    r = server.opencv_solve(task=TASK, images={"img": str(p)}, outputs=OUTPUTS)
    assert "not 8-bit" in r["error"] and ollama.calls == []


@pytest.mark.parametrize("kwargs, expect", [
    ({"task": " "}, "task must be"),
    ({"images": {}}, "images must map"),
    ({"outputs": {}}, "outputs must map"),
    ({"images": {"class": "x.png"}}, "not a valid Python argument name"),
    ({"params": {"img": 3}}, "both images and params"),
    ({"params": {"k": [1, 2]}}, "not supported"),
    ({"max_repairs": "many"}, "max_repairs must be an integer"),
])
def test_bad_arguments_are_clean_errors(img_path, ollama, kwargs, expect):
    args = {"task": TASK, "images": {"img": img_path}, "outputs": OUTPUTS, **kwargs}
    r = server.opencv_solve(**args)
    assert r["verified"] is False and expect in r["error"] and ollama.calls == []


def test_ollama_unreachable_is_reported_not_raised(img_path, ollama, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(server, "HTTP_TRANSPORT", httpx.MockTransport(refuse))
    r = solve(img_path, ollama, [])
    assert r["verified"] is False and r["attempts"] == 0 and r["error"].startswith("model request failed")
    assert "connection refused" in r["error"] and r["model"]["digest"] is None


def test_model_not_found_is_reported_without_retrying(img_path, ollama):
    ollama.chat_status = 404
    r = solve(img_path, ollama, [])
    assert r["verified"] is False and "not found" in r["error"] and len(ollama.calls) == 1


def test_internal_errors_become_results(img_path, ollama, monkeypatch):
    monkeypatch.setattr(server, "_solve", lambda *a, **k: 1 / 0)
    r = solve(img_path, ollama, [])
    assert r["verified"] is False and r["error"].startswith("internal error: ZeroDivisionError")


# ------------------------------------------------------------------------ contract parsing and paths, schema

def test_output_descriptions_parse_in_the_training_style():
    assert server.parse_output("float, shape N. px^2, largest first") == (
        IOSpec("float", "N", 1, "px^2, largest first"),) * 2 + (True,)
    assert server.parse_output("int, shape scalar")[0] == IOSpec("int", "scalar", None, "")
    assert server.parse_output("float, shape Nx2. (x, y) per point. sorted")[0] == IOSpec(
        "float", "Nx2", 2, "(x, y) per point. sorted")
    assert server.parse_output("uint8, shape HxWx3")[0] == IOSpec("uint8", "HxWx3", 3, "")
    assert server.parse_output("float, shape (N, 2). px")[0] == IOSpec("float", "Nx2", 2, "px")   # spelled the trained way
    assert server.parse_output("int, shape (N,)")[0] == IOSpec("int", "N", 1, "")
    assert server.parse_output("float, shape out_hxout_w")[0].ndim is None                          # ndim only when certain


def test_unparsable_descriptions_pass_through_as_the_note(img_path, ollama):
    shown, checked, ok = server.parse_output("a list of blob areas")
    assert not ok and shown == checked == IOSpec("any", "unspecified", None, "a list of blob areas")
    assert server.parse_output("int")[2] is False                  # a bare dtype has no shape: not parsed
    assert server.parse_output("float, shape ???")[2] is False
    r = solve(img_path, ollama, ['```python\ndef solve(img):\n    return {"blobs": [1, 2]}\n```'],
              outputs={"blobs": "a list of blob areas"})
    assert "  - 'blobs': any, shape unspecified. a list of blob areas\n" in ollama.calls[0]["messages"][1]["content"]
    assert r["verified"] is True                                    # only the key is checked
    assert any("free text" in w for w in r["warnings"])
    r = solve(img_path, ollama, ['```python\ndef solve(img):\n    return {"other": 1}\n```'] * 2,
              outputs={"blobs": "a list of blob areas"}, max_repairs=1)
    assert not r["verified"] and "missing output 'blobs'" in r["log"][0]


def test_windows_paths_are_translated_for_a_linux_server(monkeypatch):
    assert server.resolve_path("C:\\Users\\me\\a b\\x.png") == Path("/mnt/c/Users/me/a b/x.png")
    assert server.resolve_path("d:/data/x.png") == Path("/mnt/d/data/x.png")
    assert server.resolve_path("\\\\wsl.localhost\\Ubuntu-24.04\\home\\ptmel\\x.png") == Path("/home/ptmel/x.png")
    assert server.resolve_path("/tmp/x.png") == Path("/tmp/x.png")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-24.04")
    assert server.windows_path("/home/ptmel/.opencv-expert/runs/a/b.png") == (
        "\\\\wsl.localhost\\Ubuntu-24.04\\home\\ptmel\\.opencv-expert\\runs\\a\\b.png")
    assert server.windows_path("/mnt/c/Users/me/x.png") == "C:\\Users\\me\\x.png"
    monkeypatch.delenv("WSL_DISTRO_NAME")
    assert server.windows_path("/home/ptmel/x.png") is None


def test_the_tool_is_registered_with_its_schema():
    tools = asyncio.run(server.mcp.list_tools())
    assert [t.name for t in tools] == ["opencv_solve"]
    t = tools[0]
    schema = t.input_schema
    assert set(schema["properties"]) == {"task", "images", "outputs", "params", "max_repairs", "metamorphic", "save_dir"}
    assert set(schema["required"]) == {"task", "images", "outputs"}
    assert schema["properties"]["max_repairs"]["default"] == 3 and schema["properties"]["metamorphic"]["default"] is True
    assert "<dtype>, shape <shape>" in t.description and "float, shape N. px^2" in t.description


# ------------------------------------------------------------------------------------------ metamorphic checks

def run_meta(code: str, img=None, **inputs):
    from sandbox.run import run_solution
    inputs = {"img": make_image() if img is None else img, **inputs}
    first = run_solution(code, [inputs])
    assert first.ok, first.summary()
    return mm.run_metamorphic(code, inputs, first.outputs[0]), first.outputs[0]


def classes(report, key):
    return {t: c[key]["class"] for t, c in report["checks"].items()}


def test_transforms_move_pixels_where_the_coordinate_maps_say():
    img = np.zeros((10, 20, 3), np.uint8)
    img[2, 5] = 255                                                   # col x=5, row y=2
    for name in ("rot90", "mirror"):
        tf = mm.TRANSFORMS[name]
        moved = tf.apply(img)
        (x, y), = tf.points(np.array([[5.0, 2.0]]), 10, 20)
        assert moved[int(y), int(x), 0] == 255 and moved.sum() == 255 * 3
        assert np.array_equal(tf.inverse(moved), img)
    dim = mm.TRANSFORMS["dim"].apply(np.array([[0, 100, 255]], np.uint8))
    assert dim.tolist() == [[0, 60, 153]] and dim.dtype == np.uint8


def test_counter_is_invariant_under_all_three_transforms():
    code = textwrap.dedent('''\
        import cv2
        import numpy as np

        def solve(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            n, labels, stats, cents = cv2.connectedComponentsWithStats((g > 100).astype(np.uint8))
            return {"count": n - 1, "areas": np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]}
        ''')
    rep, _ = run_meta(code)
    assert classes(rep, "count") == {"rot90": "invariant", "mirror": "invariant", "dim": "invariant"}
    assert classes(rep, "areas") == {"rot90": "invariant", "mirror": "invariant", "dim": "invariant"}
    assert rep["stable_fraction"] == 1.0 and rep["counts"]["invariant"] == 6
    assert rep["verdict"].startswith("stable: all 6 checks")


def test_centroid_finder_is_covariant_under_rot90_and_mirror():
    code = textwrap.dedent('''\
        import numpy as np

        def solve(img):
            ys, xs = np.nonzero(img[:, :, 0] > 100)
            return {"centroid": np.array([[xs.mean(), ys.mean()]])}
        ''')
    img = np.zeros((60, 80, 3), np.uint8)
    img[5:15, 10:30] = 255
    rep, _ = run_meta(code, img)
    assert classes(rep, "centroid") == {"rot90": "covariant", "mirror": "covariant", "dim": "invariant"}
    assert rep["stable_fraction"] == 1.0


def test_boxes_and_segments_are_covariant():
    code = textwrap.dedent('''\
        import cv2
        import numpy as np

        def solve(img):
            mask = (img[:, :, 0] > 100).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            pts = np.argwhere(mask)[:, ::-1].astype(np.int64)          # (x, y) of every foreground pixel
            d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(d.argmax(), d.shape)
            return {"boxes": np.array([[x, y, w, h]]), "segment": np.array([[*pts[i], *pts[j]]])}
        ''')
    img = np.zeros((60, 80, 3), np.uint8)
    cv2.line(img, (12, 8), (47, 33), (255, 255, 255), 1)
    rep, _ = run_meta(code, img)
    assert classes(rep, "boxes") == {"rot90": "covariant", "mirror": "covariant", "dim": "invariant"}
    assert classes(rep, "segment") == {"rot90": "covariant", "mirror": "covariant", "dim": "invariant"}


def test_a_wrong_translation_is_changed_not_covariant():
    code = ("import numpy as np\n\ndef solve(img):\n    ys, xs = np.nonzero(img[:, :, 0] > 100)\n"
            "    return {'c': np.array([[xs.mean() + 9.0, ys.mean()]])}\n")
    rep, _ = run_meta(code)
    # a +9 px shift is no covariant map under rot90 / mirror; under dim the same wrong answer is merely unchanged
    assert classes(rep, "c") == {"rot90": "changed", "mirror": "changed", "dim": "invariant"}


def test_mean_intensity_changes_under_dim_only():
    code = "import cv2\n\ndef solve(img):\n    return {'mean': float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())}\n"
    rep, out = run_meta(code)
    assert classes(rep, "mean") == {"rot90": "invariant", "mirror": "invariant", "dim": "changed"}
    assert "0.6" not in rep["checks"]["dim"]["mean"]["detail"] and "->" in rep["checks"]["dim"]["mean"]["detail"]
    assert rep["stable_fraction"] == 0.667 and "dim: mean" in rep["verdict"] and "Confidence signal only" in rep["verdict"]


def test_a_rerun_that_crashes_is_an_error_not_a_tool_failure():
    code = textwrap.dedent('''\
        def solve(img):
            h, w = img.shape[:2]
            if h > w:
                raise ValueError("needs a landscape image")
            return {"count": 1, "wide": w > h}
        ''')
    rep, _ = run_meta(code)                               # 60 x 80 is landscape; rot90 makes it portrait
    assert classes(rep, "count") == {"rot90": "error", "mirror": "invariant", "dim": "invariant"}
    assert classes(rep, "wide")["rot90"] == "error"
    assert rep["reruns"]["rot90"]["ok"] is False and "ValueError" in rep["reruns"]["rot90"]["error"]
    assert "needs a landscape image" in rep["checks"]["rot90"]["count"]["detail"]
    assert rep["counts"]["error"] == 2 and rep["stable_fraction"] == 0.667


def test_all_reruns_failing_gives_no_signal_verdict():
    code = "def solve(img, calls=[]):\n    calls.append(1)\n    return {'k': 1}\n"
    from sandbox.run import run_solution

    def failing(code, inputs, **kw):                       # a sandbox that always refuses the rerun
        return run_solution("def solve(img):\n    raise RuntimeError('no')\n", inputs, **kw)

    rep = mm.run_metamorphic(code, {"img": make_image()}, {"k": 1}, run=failing)
    assert rep["counts"]["error"] == 3 and rep["stable_fraction"] == 0.0 and rep["verdict"].startswith("no signal")


def test_images_in_the_outputs_are_mapped_back_before_comparing():
    code = textwrap.dedent('''\
        import numpy as np

        def solve(img):
            return {"mask": (img[:, :, 0] > 100).astype(np.uint8) * 255, "gray": img[:, :, 0].copy(),
                    "thumb": img[::2, ::2, :].copy()}
        ''')
    rep, _ = run_meta(code)
    assert classes(rep, "mask") == {"rot90": "covariant", "mirror": "covariant", "dim": "invariant"}
    assert classes(rep, "gray") == {"rot90": "covariant", "mirror": "covariant", "dim": "changed"}   # 0.6 x intensity
    assert "IoU" in rep["checks"]["rot90"]["mask"]["detail"] and "mean abs diff" in rep["checks"]["dim"]["gray"]["detail"]


def test_no_image_input_is_skipped_not_failed():
    rep = mm.run_metamorphic("def solve(k):\n    return {'k': k}\n", {"k": 3}, {"k": 3})
    assert rep == {"skipped": "no uint8 image among the inputs"}


def test_value_classification_unit_cases():
    tf, dims = mm.TRANSFORMS["rot90"], [(60, 80)]
    cmp = lambda a, b, t=tf: mm.compare_value(a, b, t, dims)[0]
    assert cmp(7, 7) == "invariant" and cmp(7, 8) == "changed" and cmp(1000, 1004) == "invariant"   # 0.4 % < 1 %
    assert cmp([3.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == "invariant"                                     # same multiset
    assert cmp("a", "a") == "invariant" and cmp("a", "b") == "changed"
    assert cmp(np.zeros((0, 2)), np.zeros((0, 2))) == "invariant"
    assert cmp(np.zeros((2, 2)), np.zeros((3, 2))) == "changed"
    pts = np.array([[10.0, 5.0], [20.0, 30.0]])
    moved = tf.points(pts, 60, 80)[::-1]                                                              # unordered set
    assert cmp(pts, moved) == "covariant" and cmp(pts, moved + 1.5) == "covariant" and cmp(pts, moved + 5) == "changed"
    assert cmp({"a": 1}, {"a": 1}) == "invariant" and cmp({"a": 1}, {"b": 1}) == "changed"
    assert cmp([np.zeros(3), np.zeros(5)], [np.zeros(5), np.zeros(3)]) == "invariant"               # ragged: shallow
    assert cmp(float("nan"), float("nan")) == "invariant"
