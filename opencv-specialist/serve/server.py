"""opencv-expert MCP server: one tool, ``opencv_solve``, backed by the fine-tuned model in Ollama.

    python serve/server.py            # stdio; register it with `claude mcp add`, see serve/README.md

Flow of one call (PLAN.md Phase 4): build the task contract with ``TaskSpec.to_prompt()`` (the exact training
format, Rule 7) -> ask the model exactly as the Stage 6 evaluation did -> extract the code with the bake-off's own
extractor -> run it through ``sandbox.run_solution`` (Rule 4) -> check the typed contract -> on failure repair, up to
``max_repairs`` rounds, in the format of the training repair pairs -> on success rerun it on rot90 / mirror / dimmed
copies (serve/metamorphic.py) as a confidence signal. Model and sandbox failures come back in the result dict; the
tool does not raise for them.

The request is the evaluation's request (bakeoff.generate.build_request): /api/chat, system message
``train.config.SYSTEM``, ``think: false``, ``stream: false``, greedy options from ``bakeoff.rungs.request_options``.
There is NO ``format`` / JSON-schema constrained decoding: the model was trained to answer with one fenced Python
block, and constraining the output would change its distribution.

Repair turns. The training repair pairs (foundry.repair.build_pairing_pairs) are 5-message chats:
    system, user = task prompt, assistant = the failed code (one fenced block), user = feedback, assistant = fix
    feedback = "That failed: <message>\\nReturn the corrected full function."
Serving sends the first four. ``<message>`` is built like the training one: the sandbox's own summary
(``RunResult.summary()``: "exception on input 0: TypeError ...", "lint: line 3: [rule] ..."), or
"contract: " + the ``check_outputs`` problems, plus the removed-in-5.0 API hints, whitespace-collapsed and cut at
``foundry.repair.MAX_MESSAGE_CHARS``. Only the latest failed attempt is replayed (always the trained shape of
exactly one failed turn); earlier failures stay in the log. A history of several failed turns was never trained,
and three of them plus the prompt would not fit the 8192-token context.
"""
from __future__ import annotations

import hashlib
import json
import keyword
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `python serve/server.py` puts serve/ on sys.path, not the repo root
    sys.path.insert(0, str(ROOT))

from bakeoff.extract import extract_code  # noqa: E402
from bakeoff.rungs import request_options, sample_seed  # noqa: E402
from cvbench.api_check import check_removed_api  # noqa: E402
from cvbench.contract import IOSpec, TaskSpec, check_outputs  # noqa: E402
from foundry.answer import format_answer  # noqa: E402
from foundry.repair import MAX_MESSAGE_CHARS, feedback_text  # noqa: E402
from sandbox.lint import lint as lint_code  # noqa: E402
from sandbox.run import run_solution  # noqa: E402
from serve import metamorphic as meta_checks  # noqa: E402
from train.config import OLLAMA, SYSTEM  # noqa: E402

from mcp.server.mcpserver import MCPServer  # noqa: E402

# --- configuration -----------------------------------------------------------------------------------------------
MODEL = os.environ.get("OPENCV_EXPERT_MODEL", "opencv-expert")
OLLAMA_URL = os.environ.get("OPENCV_EXPERT_OLLAMA_URL", OLLAMA).rstrip("/")
RUNS_DIR = Path(os.environ.get("OPENCV_EXPERT_RUNS", str(Path.home() / ".opencv-expert" / "runs")))
NUM_PREDICT, NUM_CTX = 2048, 8192      # what every Stage 6 evaluation request sent, and the Modelfile's values
KEEP_ALIVE = "5m"                      # Ollama's own default, sent explicitly; the GPU is shared
HTTP_TIMEOUT_S = float(os.environ.get("OPENCV_EXPERT_HTTP_TIMEOUT", "300"))  # model load + up to 2048 tokens
RETRIES, RETRY_BACKOFF_S = 1, 2.0
SOLVE_TIMEOUT_S, SOLVE_MEM_MB = 30.0, 4096
MAX_REPAIRS_CAP = 5
MAX_PIXELS = 36_000_000
INLINE_ELEMS = 64                      # arrays up to this many elements are returned as lists
HTTP_TRANSPORT: httpx.BaseTransport | None = None  # tests put an httpx.MockTransport here

mcp = MCPServer("opencv-expert")


class OllamaError(RuntimeError):
    pass


# --- paths -------------------------------------------------------------------------------------------------------
_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")
_WSL_UNC = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/]+[^\\/]+[\\/]*(.*)$", re.I)


def resolve_path(p: str) -> Path:
    """A path as given by the orchestrator. Windows forms (C:\\Users\\x.png, \\\\wsl.localhost\\Distro\\home\\x.png)
    are translated when the server runs on Linux, because the Windows ``claude.exe`` starts it inside WSL."""
    s = os.path.expanduser(str(p).strip().strip('"').strip("'"))
    if os.name == "posix":
        m = _WSL_UNC.match(s)
        if m:
            return Path("/" + m.group(1).replace("\\", "/"))
        m = _WIN_DRIVE.match(s)
        if m:
            return Path(f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/"))
    return Path(s)


def windows_path(p: str | Path) -> str | None:
    """The path as Windows sees it (for the orchestrator's file viewer), or None when not running in WSL."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    s = str(p)
    if os.name != "posix" or not distro or not s.startswith("/"):
        return None
    m = re.match(r"^/mnt/([a-z])(/.*)?$", s)
    if m:
        return f"{m.group(1).upper()}:" + (m.group(2) or "\\").replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}" + s.replace("/", "\\")


# --- the contract ------------------------------------------------------------------------------------------------
_KNOWN_DTYPES = {"float", "int", "bool", "any", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32",
                 "int64", "float16", "float32", "float64"}
_OUT_RE = re.compile(r"^\s*(?P<dtype>[A-Za-z][A-Za-z0-9_]*)\s*,\s*shape\s+(?P<shape>.+?)\s*(?:\.\s+(?P<note>.*))?$", re.S)
_SHAPE_WORD = re.compile(r"^[A-Za-z0-9_]+$")
_SHAPE_PAREN = re.compile(r"^\(\s*([A-Za-z0-9_]+\s*(?:,\s*[A-Za-z0-9_]*\s*)*)?\)$")
_NDIM_SHAPE = re.compile(r"^(?:[A-Za-z]|\d+)(?:x(?:[A-Za-z]|\d+))*$")


def _normalise_shape(raw: str) -> str | None:
    """'(N, 2)' -> 'Nx2', '(N,)' -> 'N', '()' -> 'scalar'; training-style shapes are kept; anything else is None."""
    s = raw.strip().rstrip(".").strip()
    if s.lower() == "scalar":
        return "scalar"
    if _SHAPE_WORD.match(s):
        return s
    m = _SHAPE_PAREN.match(s)
    if m:
        dims = [d.strip() for d in (m.group(1) or "").split(",") if d.strip()]
        return "x".join(dims) if dims else "scalar"
    return None


def parse_output(description: str) -> tuple[IOSpec, IOSpec, bool]:
    """(spec for the prompt, spec for the check, parsed) of one output description.

    Training style: ``<dtype>, shape <shape>[. <note>]``. When it parses, the prompt shows it in exactly that form
    ('(N, 2)' is rewritten to the training spelling 'Nx2'). A dtype that ``cvbench.contract.check_outputs`` cannot
    test (e.g. 'str') is still shown to the model but checked as 'any'. When it does not parse, the whole description
    is passed through as the note ('any, shape unspecified. <description>') and only the key's presence is checked.
    """
    desc = str(description).strip()
    m = _OUT_RE.match(desc)
    shape = _normalise_shape(m.group("shape")) if m else None
    if m and shape is not None:
        dtype, note = m.group("dtype"), (m.group("note") or "").strip()
        if dtype.lower() in _KNOWN_DTYPES:
            dtype = dtype.lower()  # the trained spelling
        ndim = None if shape == "scalar" else (len(shape.split("x")) if _NDIM_SHAPE.match(shape) else None)
        shown = IOSpec(dtype, shape, ndim, note)
        # check_outputs reads "scalar" as a Python int/float and compares unknown dtypes by name: a bool, str or "any"
        # scalar would fail it for no reason, so those are checked for presence only
        if dtype not in _KNOWN_DTYPES:
            checked = IOSpec("any", "unspecified" if shape == "scalar" else shape, ndim, note)
        elif shape == "scalar" and dtype in ("bool", "any"):
            checked = IOSpec("any", "unspecified", None, note)
        else:
            checked = shown
        return shown, checked, True
    spec = IOSpec("any", "unspecified", None, desc)
    return spec, spec, False


def _param_spec(v: Any) -> IOSpec | None:
    for typ, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if isinstance(v, typ):
            return IOSpec(name, "scalar", None, "")
    return None


def _image_spec(img: np.ndarray) -> IOSpec:
    if img.ndim == 2:
        return IOSpec("uint8", "HxW", 2, "grayscale image")
    return IOSpec("uint8", "HxWx3", 3, "BGR colour image")


def load_image(path: str) -> tuple[np.ndarray | None, str]:
    """(uint8 image, "") or (None, reason). Colour comes back as BGR HxWx3 (alpha dropped), gray as HxW."""
    p = resolve_path(path)
    if not p.is_file():
        return None, f"no such file: {p}"
    try:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    except cv2.error as e:  # pragma: no cover - imread normally returns None
        return None, f"cannot decode {p}: {e}"
    if img is None:
        return None, f"cannot decode {p} as an image (PNG/JPG expected)"
    if img.dtype != np.uint8:
        return None, f"{p} is {img.dtype}, not 8-bit; convert it to an 8-bit PNG/JPG first"
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim not in (2, 3) or img.size == 0:
        return None, f"{p} has an unsupported shape {img.shape}"
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        return None, f"{p} is {img.shape[1]}x{img.shape[0]}; the limit is {MAX_PIXELS // 1_000_000} megapixels"
    return np.ascontiguousarray(img), ""


def _bad_name(name: Any) -> bool:
    return not (isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name))


# --- Ollama ------------------------------------------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(transport=HTTP_TRANSPORT, trust_env=False, timeout=HTTP_TIMEOUT_S)  # local: ignore proxies


def build_request(messages: list[dict], seed: int) -> dict:
    """The body bakeoff.generate.build_request sends for the Stage 6 rungs (think False), with ``messages`` given."""
    return {"model": MODEL, "stream": False, "keep_alive": KEEP_ALIVE,
            "options": request_options("greedy", num_predict=NUM_PREDICT, num_ctx=NUM_CTX, seed=seed),
            "messages": messages, "think": False}


def ollama_chat(body: dict) -> dict:
    """POST /api/chat. Raises OllamaError after RETRIES retries."""
    last = ""
    for attempt in range(RETRIES + 1):
        try:
            with _client() as c:
                r = c.post(OLLAMA_URL + "/api/chat", json=body)
            if r.status_code >= 400:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code < 500:
                    break  # model missing, bad request: a retry cannot help
            else:
                data = r.json()
                if data.get("error"):
                    raise OllamaError(str(data["error"])[:300])
                return data
        except OllamaError:
            raise
        except (httpx.HTTPError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < RETRIES:
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise OllamaError(f"Ollama at {OLLAMA_URL}: {last}")


def model_info() -> dict:
    """Tag and digest of the served model from /api/tags; never raises."""
    info: dict[str, Any] = {"tag": MODEL, "digest": None}
    try:
        with _client() as c:
            r = c.get(OLLAMA_URL + "/api/tags", timeout=10.0)
        for m in r.json().get("models", []):
            if m.get("name") in (MODEL, MODEL + ":latest") or m.get("model") in (MODEL, MODEL + ":latest"):
                d = m.get("details") or {}
                info.update(digest=m.get("digest"), size_bytes=m.get("size"), modified_at=m.get("modified_at"),
                            quantization=d.get("quantization_level"), parameter_size=d.get("parameter_size"))
                break
        else:
            info["error"] = f"{MODEL} is not in `ollama list`"
    except (httpx.HTTPError, ValueError) as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


# --- repair feedback ---------------------------------------------------------------------------------------------

def failure_message(message: str, code: str) -> str:
    """The text after 'That failed: ' for a training-format repair turn (foundry.verify.failure_of + build_pairing_pairs):
    the grader's message, then the removed-in-5.0 hints (at most 3), whitespace collapsed, cut at MAX_MESSAGE_CHARS."""
    hints = [f"`{f.name}` is not in OpenCV 5.0" + (f"; use {f.replacement}" if f.replacement else "")
             for f in check_removed_api(code)[:3]]
    if hints:
        message += " (" + "; ".join(hints) + ")"
    message = " ".join(message.split())
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[:MAX_MESSAGE_CHARS - 3] + "..."
    return message


def contract_problems(spec: TaskSpec, out: dict) -> list[str]:
    try:
        return check_outputs(spec, out)
    except Exception as e:  # e.g. np.asarray on a ragged list
        return [f"outputs could not be checked: {type(e).__name__}: {e}"]


# --- measurements ------------------------------------------------------------------------------------------------

def _py(x: Any) -> Any:
    """JSON-safe python value: numpy scalars unwrapped, floats shortened, nan/inf as strings."""
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return float(f"{x:.8g}") if np.isfinite(x) else str(x)
    if isinstance(x, (bool, int)) or x is None:
        return x
    return str(x)[:2000]


def _stats(a: np.ndarray) -> dict:
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and a.dtype == bool:
        out["true_count"] = int(a.sum())
    elif a.size and a.dtype.kind in "iuf":
        f = a.astype(np.float64)
        if np.isfinite(f).any():
            out.update(min=_py(float(np.nanmin(f[np.isfinite(f)]))), max=_py(float(np.nanmax(f[np.isfinite(f)]))),
                       mean=_py(float(np.mean(f[np.isfinite(f)]))))
        if not np.isfinite(f).all():
            out["non_finite"] = int((~np.isfinite(f)).sum())
    return out


def describe(value: Any, run_dir: Path, name: str) -> Any:
    """Small values inline; arrays above INLINE_ELEMS summarised; images (uint8 HxW / HxWx{1,3,4}, bool masks) saved as
    PNG under ``run_dir`` and returned as paths so the orchestrator can look at them."""
    if isinstance(value, dict):
        return {str(k): describe(v, run_dir, f"{name}.{k}") for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)) and not (value and all(np.isscalar(v) for v in value)):
        try:
            arr = np.asarray(value)
            if arr.dtype == object:
                raise ValueError
            value = arr
        except ValueError:  # ragged: describe the first few items
            items = [describe(v, run_dir, f"{name}[{i}]") for i, v in enumerate(value[:8])]
            return {"type": type(value).__name__, "length": len(value), "items": items,
                    **({"truncated": True} if len(value) > 8 else {})}
    if isinstance(value, (list, tuple)):
        return _py(list(value)) if len(value) <= INLINE_ELEMS else {"length": len(value)}
    if not isinstance(value, np.ndarray):
        return _py(value)
    a = value
    is_image = (a.dtype == np.uint8 and a.ndim in (2, 3) and (a.ndim == 2 or a.shape[2] in (1, 3, 4))
                and min(a.shape[:2]) > 8)
    is_mask = a.dtype == bool and a.ndim == 2 and min(a.shape) > 8
    if is_image or is_mask:
        path = run_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".png")
        cv2.imwrite(str(path), (a.astype(np.uint8) * 255) if is_mask else a)
        return {"image": str(path), "windows_path": windows_path(path), **_stats(a),
                **({"note": "bool mask saved as 0/255"} if is_mask else {})}
    if a.size <= INLINE_ELEMS:
        return _py(a.tolist())
    return _stats(a)


# --- the tool ----------------------------------------------------------------------------------------------------

def _new_run_dir() -> Path:
    d = RUNS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _empty_result(error: str | None = None) -> dict:
    return {"verified": False, "attempts": 0, "code": "", "log": [], "measurements": {}, "run_dir": None,
            "run_dir_windows": None, "model": {"tag": MODEL, "digest": None},
            "timing": {"total_s": 0.0, "model_s": 0.0, "sandbox_s": 0.0, "metamorphic_s": 0.0},
            "metamorphic": {"skipped": "not run"}, "warnings": [], "error": error}


def _fail(error: str, **extra: Any) -> dict:
    r = _empty_result(error)
    r["log"] = [error]
    r.update(extra)
    return r


@mcp.tool()
def opencv_solve(task: str, images: dict[str, str], outputs: dict[str, str], params: dict | None = None,
                 max_repairs: int = 3, metamorphic: bool = True) -> dict:
    """Write, run and check OpenCV 5.0 (Python cv2) code for an image-measurement task on local images.

    A small local model writes `solve(...) -> dict`; the server runs it in a sandbox on your image(s), checks the
    result against the declared outputs, and asks the model to repair it (up to `max_repairs` times) when it fails.
    Use it for well-specified measurements: counting, areas, contours, edges, line segments, colour segmentation,
    histograms, homography, perspective warps. State exactly what each output means (units, order, coordinate
    convention): the model only knows what the task text and the output descriptions say.

    Args:
      task: what to compute, in plain words, e.g. "Count the bright blobs and give the area of each, largest first".
      images: {solve() argument name: local PNG/JPG path}. Usually {"img": "/path/x.png"}; two images for e.g.
        homography: {"img_a": "...", "img_b": "..."}. Colour images arrive as uint8 BGR HxWx3, gray ones as HxW.
        Windows paths (C:\\Users\\me\\x.png) are accepted.
      outputs: {output name: description}, one entry per key `solve()` must return. Describe each as
        "<dtype>, shape <shape>[. <note>]" (the style the model was trained on). dtype: float, int, bool, uint8,
        float32, ... or any. shape: scalar, N, Nx2, Nx4, HxWx3, 3x3, 256, ... A note says units/order/convention.
        Examples:
          {"areas": "float, shape N. px^2, one per blob, largest first", "count": "int, shape scalar"}
          {"corners": "float, shape Nx2. (x, y) pixel coordinates, sorted by x then y",
           "mask": "uint8, shape HxW. 255 inside the object, 0 elsewhere"}
        A description that does not fit this form is passed to the model as free text and only the presence of
        the key is checked.
      params: optional extra scalar inputs of solve() (int, float, bool or str), e.g. {"connectivity": 8}.
      max_repairs: repair rounds after a failed attempt (0-5, default 3).
      metamorphic: after success, rerun the code on rotated / mirrored / dimmed copies of the image(s) and report
        how the outputs behave (confidence signal only).

    Returns a dict:
      verified: True if the code ran in the sandbox and every declared output has the declared dtype and shape.
        This does NOT prove the values are right: look at `measurements` (and the saved images) before trusting them.
      attempts, code (the final solve() source), log (one line per failed attempt), error (set when no attempt
      could run: unreadable image, bad arguments, Ollama unreachable),
      measurements: small values inline; larger arrays as {shape, dtype, min, max, mean}; uint8 images and masks are
        saved as PNG and given as {"image": path, "windows_path": ...},
      run_dir (everything: prompt, each reply, each attempt's code, result.json), model {tag, digest}, timing,
      metamorphic {checks per transform and key (invariant / covariant / changed / error), stable_fraction,
        verdict}: some tasks are legitimately not invariant, so read it as a hint, never as a failure.
    """
    t_start = time.monotonic()
    try:
        return _solve(task, images, outputs, params, max_repairs, metamorphic, t_start)
    except Exception as e:  # a bug here must still reach the client as a result, with enough to debug it
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        r = _fail(f"internal error: {type(e).__name__}: {e}")
        r["traceback"] = traceback.format_exc()[-1500:]
        r["timing"]["total_s"] = round(time.monotonic() - t_start, 2)
        return r


def _solve(task, images, outputs, params, max_repairs, do_metamorphic, t_start) -> dict:
    # ---- arguments
    if not isinstance(task, str) or not task.strip():
        return _fail("task must be a non-empty string")
    if not isinstance(images, dict) or not images:
        return _fail("images must map each solve() image argument to a file path, e.g. {\"img\": \"/path/x.png\"}")
    if not isinstance(outputs, dict) or not outputs:
        return _fail("outputs must map each output name to a description, e.g. {\"count\": \"int, shape scalar\"}")
    params = params or {}
    if not isinstance(params, dict):
        return _fail("params must be an object of scalar values")
    for n in [*images, *params, *outputs]:
        if not isinstance(n, str) or not n:
            return _fail(f"names must be non-empty strings, got {n!r}")
    for n in [*images, *params]:
        if _bad_name(n):
            return _fail(f"{n!r} is not a valid Python argument name")
    clash = sorted(set(images) & set(params))
    if clash:
        return _fail(f"{clash} appear in both images and params")
    try:
        max_repairs = max(0, min(int(max_repairs), MAX_REPAIRS_CAP))
    except (TypeError, ValueError):
        return _fail("max_repairs must be an integer")

    # ---- inputs
    arrays: dict[str, np.ndarray] = {}
    for name, path in images.items():
        img, why = load_image(path) if isinstance(path, str) and path.strip() else (None, "the path is empty")
        if img is None:
            return _fail(f"image {name!r}: {why}")
        arrays[name] = img
    in_specs = {n: _image_spec(a) for n, a in arrays.items()}
    for n, v in params.items():
        s = _param_spec(v)
        if s is None:
            return _fail(f"param {n!r}: {type(v).__name__} values are not supported; use int, float, bool or str")
        in_specs[n] = s
    inputs = {**arrays, **params}

    # ---- contract and prompt
    shown, checked, warnings = {}, {}, []
    for n, desc in outputs.items():
        shown[n], checked[n], ok = parse_output(desc)
        if not ok:
            warnings.append(f"output {n!r}: description is not in '<dtype>, shape <shape>[. <note>]' form; sent as "
                            "free text, only the presence of the key is checked")
        elif shown[n] != checked[n]:
            warnings.append(f"output {n!r}: '{shown[n].dtype}, shape {shown[n].shape}' is sent as written but the "
                            "contract check cannot test it; only the presence of the key is checked")
    prompt_spec = TaskSpec("serve", task.strip(), in_specs, shown)
    check_spec = TaskSpec("serve", task.strip(), in_specs, checked)
    prompt = prompt_spec.to_prompt()
    task_id = hashlib.sha1(prompt.encode()).hexdigest()[:12]

    run_dir = _new_run_dir()
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    result = _empty_result()
    result.update(run_dir=str(run_dir), run_dir_windows=windows_path(run_dir), warnings=warnings,
                  model=model_info())
    timing = result["timing"]
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]

    # ---- generate, run, repair
    log: list[str] = result["log"]
    seen: dict[str, int] = {}
    prev: tuple[str, str] | None = None  # (failed code, feedback message)
    res = None
    code = ""
    verified = False
    for attempt in range(1, max_repairs + 2):
        messages = list(base)
        if prev is not None:
            messages += [{"role": "assistant", "content": format_answer(prev[0])},
                         {"role": "user", "content": feedback_text(prev[1])}]
        t0 = time.monotonic()
        try:
            reply = ollama_chat(build_request(messages, sample_seed(MODEL, task_id, "greedy", attempt - 1)))
        except OllamaError as e:
            timing["model_s"] += time.monotonic() - t0
            result["error"] = f"model request failed: {e}"
            log.append(f"attempt {attempt}: {e}")
            break
        timing["model_s"] += time.monotonic() - t0
        text = (reply.get("message") or {}).get("content") or ""
        ex = extract_code(text)
        result["attempts"] = attempt
        (run_dir / f"attempt{attempt}.reply.txt").write_text(text, encoding="utf-8")
        (run_dir / f"attempt{attempt}.py").write_text(ex.code, encoding="utf-8")
        cut = " (the reply hit the token limit)" if reply.get("done_reason") == "length" else ""

        # prose or nothing: the training never repaired these (foundry.verify.failure_of), and at temperature 0 a
        # second request would repeat the first
        if ex.path == "empty" or (ex.path == "bare" and any(f.rule == "syntax" for f in lint_code(ex.code))):
            log.append(f"attempt {attempt}: the reply contained no code{cut}")
            result["error"] = "the model's reply contained no Python code"
            break
        digest = hashlib.sha1(ex.code.strip().encode()).hexdigest()
        if digest in seen:
            log.append(f"attempt {attempt}: identical code to attempt {seen[digest]}; stopping, the model repeats itself")
            break
        seen[digest] = attempt
        code = ex.code

        t0 = time.monotonic()
        res = run_solution(ex.code, [inputs], timeout_s=SOLVE_TIMEOUT_S, mem_mb=SOLVE_MEM_MB)
        timing["sandbox_s"] += time.monotonic() - t0
        if res.ok:
            problems = contract_problems(check_spec, res.outputs[0])
            if not problems:
                verified = True
                break
            message = "contract: " + "; ".join(problems)
        else:
            message = res.summary()
        message = failure_message(message, ex.code)
        log.append(f"attempt {attempt}: {message}{cut}")
        prev = (ex.code, message)

    # ---- report
    result["verified"] = verified
    result["code"] = code
    if res is not None and res.ok:
        result["measurements"] = {k: describe(v, run_dir, k) for k, v in res.outputs[0].items()}
    if verified and do_metamorphic:
        t0 = time.monotonic()
        result["metamorphic"] = meta_checks.run_metamorphic(
            code, inputs, res.outputs[0], keys=list(outputs),
            timeout_s=min(60.0, max(SOLVE_TIMEOUT_S, 3 * res.seconds + 5)), mem_mb=SOLVE_MEM_MB)
        timing["metamorphic_s"] += time.monotonic() - t0
    elif verified:
        result["metamorphic"] = {"skipped": "metamorphic=False"}
    else:
        result["metamorphic"] = {"skipped": "the code was not verified"}
    timing["total_s"] = time.monotonic() - t_start
    result["timing"] = {k: round(v, 2) for k, v in timing.items()}
    try:
        (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


if __name__ == "__main__":
    mcp.run()
