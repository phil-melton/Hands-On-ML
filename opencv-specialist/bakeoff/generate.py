"""Generation: ask an Ollama model every task, one reply at a time, and append each reply to generations.jsonl.

For each rung, in this order:

    1. load the model (empty-message request, keep_alive held for the whole rung)
    2. one warm-up request, discarded (Stage 1 measured a 3.8x slower first request after model creation);
       it is written to warmups.jsonl, never to generations.jsonl, so it cannot enter a statistic
    3. the greedy pass (T=0, one reply per task), then the sampling passes (T=0.8, sample 0 of every task,
       then sample 1 ...). Sample-major order means an interrupted run still has n complete samples per task
    4. unload (keep_alive 0), also on an error or Ctrl-C, because the next rung needs the VRAM

Rerunning the same command skips every (rung, task, mode, sample) already in the file, so a crash costs at most
the reply in flight. The options a resume must agree on are fixed in config.json; a mismatch stops the run
rather than mixing two protocols in one file.

Transport is httpx (as in train/hydro_eval.py) when it is installed and urllib otherwise, so the cv5 venv works
unchanged. ``chat`` is the single function that touches the network, which is what the tests replace.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from bakeoff.extract import extract_code
from bakeoff.rungs import MODES, Rung, notes_token_estimate, request_options, sample_seed, system_prompt
from bakeoff.store import JsonlWriter, RunDir, keys, read_json, read_jsonl, write_json
from bakeoff.tasks import Task

WARMUP_NUM_PREDICT = 64


class OllamaError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- transport

def _http(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    try:
        import httpx
    except ImportError:
        httpx = None
    # Ollama is local: ignore proxy environment variables (a WSL shell may set them) in both clients
    if httpx is not None:
        try:
            r = httpx.request(method, url, json=payload, timeout=timeout, trust_env=False)
            if r.status_code >= 400:
                raise OllamaError(f"{method} {url}: HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise OllamaError(f"{method} {url}: {e!r}") from e
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise OllamaError(f"{method} {url}: HTTP {e.code}: {e.read()[:300]!r}") from e
    except (OSError, ValueError) as e:  # URLError, timeouts, bad JSON
        raise OllamaError(f"{method} {url}: {e!r}") from e


def chat(base_url: str, body: dict, timeout: float) -> dict:
    """POST /api/chat. The only call that sends a generation request."""
    return _http("POST", base_url.rstrip("/") + "/api/chat", body, timeout)


def _chat_retry(base_url: str, body: dict, st: "GenSettings") -> dict:
    for attempt in range(st.retries + 1):
        try:
            return chat(base_url, body, st.http_timeout)
        except OllamaError as e:
            if attempt == st.retries:
                raise
            _log(f"  request failed ({e}); retry {attempt + 1}/{st.retries} in {st.backoff_s * (attempt + 1):.0f}s")
            time.sleep(st.backoff_s * (attempt + 1))
    raise AssertionError("unreachable")


def _info(base_url: str, path: str, payload: dict | None, timeout: float = 30.0) -> dict:
    try:
        return _http("GET" if payload is None else "POST", base_url.rstrip("/") + path, payload, timeout)
    except OllamaError as e:
        _log(f"  warning: {e}")
        return {}


# ---------------------------------------------------------------- settings and request

@dataclass(frozen=True)
class GenSettings:
    base_url: str
    samples: int = 5            # sampled replies per task at T=0.8 (0: greedy only)
    num_predict: int = 1024
    num_ctx: int = 4096
    keep_alive: str = "30m"
    http_timeout: float = 900.0
    retries: int = 2
    backoff_s: float = 5.0
    dedupe_greedy: bool = False  # reuse one greedy reply for tasks with an identical prompt (saves GPU time)


def build_request(rung: Rung, system: str, prompt: str, options: dict, keep_alive: str) -> dict:
    body = {"model": rung.model, "stream": False, "keep_alive": keep_alive, "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if rung.think is not None:
        body["think"] = rung.think
    return body


def plan(tasks: list[Task], samples: int):
    """(task, mode, sample index) in generation order: greedy for all tasks, then sample-major sampling."""
    for t in tasks:
        yield t, "greedy", 0
    for s in range(samples):
        for t in tasks:
            yield t, "sample", s


def make_record(rung: Rung, task: Task, mode: str, sample: int, seed: int, options: dict, resp: dict,
                wall_s: float, st: GenSettings) -> dict:
    msg = resp.get("message") or {}
    reply = msg.get("content") or ""
    ex = extract_code(reply)
    done_reason = resp.get("done_reason")
    eval_count = int(resp.get("eval_count") or 0)
    return {
        "rung": rung.name, "model": rung.model, **task.meta(), "mode": mode, "sample": sample,
        "temperature": options["temperature"], "seed": seed,
        "reply": reply, "code": ex.code, "extract": ex.path, "think_block": ex.think,
        "thinking_chars": len(msg.get("thinking") or ""),
        "eval_count": eval_count, "eval_duration": int(resp.get("eval_duration") or 0),
        "prompt_eval_count": int(resp.get("prompt_eval_count") or 0),
        "prompt_eval_duration": int(resp.get("prompt_eval_duration") or 0),
        "load_duration": int(resp.get("load_duration") or 0), "total_duration": int(resp.get("total_duration") or 0),
        "done_reason": done_reason, "truncated": done_reason == "length" or eval_count >= st.num_predict,
        "wall_s": round(wall_s, 3), "reused_from": None, "t": _now(),
    }


def _generate_one(rung, system, task, mode, sample, st) -> dict:
    seed = sample_seed(rung.name, task.id, mode, sample)
    options = request_options(mode, num_predict=st.num_predict, num_ctx=st.num_ctx, seed=seed)
    body = build_request(rung, system, task.prompt, options, st.keep_alive)
    t0 = time.perf_counter()
    resp = _chat_retry(st.base_url, body, st)
    return make_record(rung, task, mode, sample, seed, options, resp, time.perf_counter() - t0, st)


# ---------------------------------------------------------------- config

def decoding_config(st: GenSettings) -> dict:
    """What every request carried, minus the per-request seed."""
    opts = {m: request_options(m, num_predict=st.num_predict, num_ctx=st.num_ctx, seed=0) for m in MODES}
    for o in opts.values():
        del o["seed"]
    return {**opts, "keep_alive": st.keep_alive,
            "seed": "sha256('rung|task_id|mode|sample')[:4] & 0x7fffffff, sent as options.seed",
            "top_k_min_p": "not set: the model's Modelfile / Ollama defaults apply"}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prepare_config(run: RunDir, rungs: list[Rung], systems: dict[str, str], st: GenSettings,
                   notes_text: str | None, invocation: dict) -> dict:
    """Write config.json, or check that a resume agrees with the one already there."""
    dec = decoding_config(st)
    cfg = read_json(run.config, None)
    if cfg is None:
        cfg = {"created": _now(), "ollama": st.base_url, "decoding": dec, "rungs": {}, "invocations": []}
    else:
        cmp = lambda d: {k: v for k, v in d.items() if k != "keep_alive"}
        if cmp(cfg["decoding"]) != cmp(dec):
            raise SystemExit(f"{run.config} was made with different decoding options:\n  have {cfg['decoding']}\n  asked {dec}\n"
                             "Use a new --run name; one file must hold one protocol.")
    for r in rungs:
        entry = {"model": r.model, "think": r.think, "system_sha1": _sha1(systems[r.name]), "system": systems[r.name]}
        if r.notes:
            entry.update(notes_sha1=_sha1(notes_text or ""), notes_chars=len((notes_text or "").strip()),
                         notes_est_tokens=notes_token_estimate(notes_text or ""))
        old = cfg["rungs"].get(r.name)
        if old and (old["model"], old["think"], old["system_sha1"]) != (entry["model"], entry["think"], entry["system_sha1"]):
            raise SystemExit(f"rung {r.name} in {run.config} used a different model/think/system prompt; use a new --run name")
        cfg["rungs"][r.name] = {**(old or {}), **entry}
    cfg["invocations"].append({"t": _now(), **invocation})
    write_json(run.config, cfg)
    return cfg


# ---------------------------------------------------------------- the loop

def _serve_rung(run, rung, system, tasks, todo, st, writers, greedy_cache, log) -> int:
    out, warm_out = writers
    cfg = read_json(run.config)
    log(f"[{rung.name}] {rung.model}: {len(todo)} replies to generate")
    t_load = time.perf_counter()
    load = _chat_retry(st.base_url, {"model": rung.model, "messages": [], "keep_alive": st.keep_alive}, st)
    show = _info(st.base_url, "/api/show", {"model": rung.model})
    caps = show.get("capabilities") or []
    if "thinking" in caps and rung.think is None:
        log(f"  warning: {rung.model} reports the 'thinking' capability but the rung sends no think flag")
    cfg["rungs"][rung.name]["show"] = {"capabilities": caps, "parameters": show.get("parameters"), "details": show.get("details")}
    cfg["ollama_version"] = _info(st.base_url, "/api/version", None).get("version")
    write_json(run.config, cfg)

    # warm-up: a real request, same system prompt, discarded
    first = tasks[0]
    wopts = request_options("greedy", num_predict=WARMUP_NUM_PREDICT, num_ctx=st.num_ctx,
                            seed=sample_seed(rung.name, first.id, "warmup", 0))
    t0 = time.perf_counter()
    wresp = _chat_retry(st.base_url, build_request(rung, system, first.prompt, wopts, st.keep_alive), st)
    ps = next((m for m in _info(st.base_url, "/api/ps", None).get("models", []) if m.get("name", "").startswith(rung.model)), {})
    warm_out.write({"rung": rung.name, "model": rung.model, "task_id": first.id, "wall_s": round(time.perf_counter() - t0, 3),
                    "load_wall_s": round(t0 - t_load, 3), "load_duration_load_request": load.get("load_duration"),
                    "load_duration": wresp.get("load_duration"), "eval_count": wresp.get("eval_count"),
                    "eval_duration": wresp.get("eval_duration"), "prompt_eval_count": wresp.get("prompt_eval_count"),
                    "prompt_eval_duration": wresp.get("prompt_eval_duration"), "size": ps.get("size"),
                    "size_vram": ps.get("size_vram"), "t": _now()})
    log(f"  warm-up discarded ({time.perf_counter() - t0:.1f}s); model on GPU: {ps.get('size_vram')} of {ps.get('size')} bytes")

    n = 0
    t_start = time.perf_counter()
    for task, mode, sample in todo:
        if mode == "greedy" and st.dedupe_greedy and task.prompt_sha1 in greedy_cache:
            src = greedy_cache[task.prompt_sha1]
            rec = {**src, **task.meta(), "sample": 0, "seed": sample_seed(rung.name, task.id, mode, 0), "wall_s": 0.0,
                   "reused_from": src["task_id"], "t": _now()}
        else:
            rec = _generate_one(rung, system, task, mode, sample, st)
            if mode == "greedy":
                greedy_cache.setdefault(task.prompt_sha1, rec)
        out.write(rec)
        n += 1
        if n % 25 == 0 or n == len(todo):
            rate = n / (time.perf_counter() - t_start)
            log(f"  [{rung.name}] {n}/{len(todo)} replies, {1 / rate:.1f}s each, about {(len(todo) - n) / rate / 60:.0f} min left")
    return n


def run_generation(run: RunDir, rungs: list[Rung], tasks: list[Task], st: GenSettings, *,
                   notes_text: str | None = None, invocation: dict | None = None, log=_log,
                   system: str | None = None) -> dict:
    """Generate every missing reply for every rung. Returns {rung: replies written}.

    ``system`` replaces the OpenCV system prompt for every rung (the forgetting check sends one neutral prompt
    to both arms, bakeoff/forgetting.py); the default None keeps ``rungs.system_prompt``.
    """
    if not tasks:
        raise ValueError("no tasks selected")
    run.path.mkdir(parents=True, exist_ok=True)
    run.done.unlink(missing_ok=True)
    systems = {r.name: (system if system is not None else system_prompt(r, notes_text)) for r in rungs}
    prepare_config(run, rungs, systems, st, notes_text, invocation or {})
    existing = read_jsonl(run.generations)
    done = keys(existing)
    written: dict[str, int] = {}
    with JsonlWriter(run.generations) as out, JsonlWriter(run.warmups) as warm_out:
        for rung in rungs:
            todo = [(t, m, s) for t, m, s in plan(tasks, st.samples) if (rung.name, t.id, m, s) not in done]
            if not todo:
                log(f"[{rung.name}] nothing to do")
                written[rung.name] = 0
                continue
            greedy_cache = {r["prompt_sha1"]: r for r in existing
                            if r["rung"] == rung.name and r["mode"] == "greedy" and not r.get("reused_from")}
            try:
                written[rung.name] = _serve_rung(run, rung, systems[rung.name], tasks, todo, st, (out, warm_out), greedy_cache, log)
            finally:  # the next rung needs the VRAM, also after an error or Ctrl-C
                try:
                    chat(st.base_url, {"model": rung.model, "messages": [], "keep_alive": 0}, 60.0)
                except OllamaError as e:
                    log(f"  warning: could not unload {rung.model}: {e}")
    run.done.write_text(_now() + "\n")
    return written
