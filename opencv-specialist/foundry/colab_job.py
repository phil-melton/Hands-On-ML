#!/usr/bin/env python3
"""Offline batch generation with vLLM on a Colab VM: prompts.jsonl in, completions.jsonl and timing.json out.

STANDALONE. This file is copied to the VM and run there with the VM's Python; it imports nothing from this repository
(tests pin that), only the standard library at import time. vLLM, and torch for the GPU name, are imported inside
``main`` so the file can be imported and tested anywhere.

    python colab_job.py --prompts /content/foundry/generate/prompts.jsonl --out-dir /content/foundry/generate/out \\
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct --quantization awq --max-model-len 8192 -k 4

Input, one JSON object per line:  {"id": "...", "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}
(optional "spec_id", which defaults to "id"). The prompt is rendered with the model's own chat template
(``tokenizer.apply_chat_template(..., add_generation_prompt=True)``), then generated with ``n = k`` samples.

A prompt row may override the job's settings for itself: ``k`` (samples), ``temperature``, ``top_p``, ``top_k``,
``max_tokens`` and ``chat_template_kwargs``; and carry the labels ``kind``, ``mode`` and ``origin``, which are copied to its
completions. So one job (one model load, the expensive part) can serve several kinds of prompt, e.g. teacher-written
phrasings, thinking and non-thinking solution samples.

Output ``completions.jsonl``, one row per sample, flushed after every chunk of prompts:

    {"id", "spec_id", "sample" (0..k-1), "text" (the raw completion), "thinking", "answer", "think_status",
     "thinking_tokens", "answer_tokens", "prompt_tokens", "completion_tokens", "finish_reason", "model",
     and the prompt's "kind" / "mode" / "origin" when it has them}

finish_reason is vLLM's ("stop", "length") or "prompt_too_long" (the rendered prompt leaves no room; nothing generated).

Thinking. With ``--chat-template-kwargs '{"enable_thinking": true}'`` a Qwen3-family model writes its reasoning first and
closes it with ``</think>``. The Qwen3.5/3.8 template already opens the block in the prompt (it ends in ``<think>``), so the
completion then holds only the closing tag. ``split_thinking`` splits the raw text at the first ``</think>``: ``thinking`` is
what comes before (a leading ``<think>`` removed), ``answer`` what comes after. ``think_status`` is "none" (no think block),
"empty" (a closed, empty block), "closed", or "truncated" (a block that was opened, in the prompt or by the model, and never
closed: the budget ran out inside the reasoning, so ``answer`` is empty). ``thinking_tokens`` counts the reasoning including
the closing tag and ``answer_tokens`` the rest; they add up to ``completion_tokens``. Everything downstream (code extraction,
grading, training examples) must use ``answer`` only; ``text`` is kept for provenance.

Output ``timing.json``: load_s (vLLM start-up and weights), gen_s (the sum of the generate() calls), output_tokens,
thinking_tokens, answer_tokens, output_tokens_per_s = output_tokens / gen_s, token totals per group ("groups", labelled by
the prompt's mode, else its kind), and the versions and GPU. Resumable: a prompt whose k samples are already in
completions.jsonl is skipped, and timing.json accumulates over invocations (load_s is paid again by each).

Sampling: temperature 0.8, top_p 0.95, max_tokens 2048 by default (``--top-k`` is off by default). The seed goes to
``LLM(seed=...)``; vLLM's per-request seed is not used (with n > 1 it is version-dependent whether the n samples differ), so a
rerun with the same seed and batch shapes is repeatable but not bit-exact. Thinking needs room: for example
``--max-model-len 16384 --max-tokens 12288``.

The Colab CLI's ``exec -f`` runs a file inside the Jupyter kernel and passes no argv, so arguments can come from the
environment: FOUNDRY_JOB_ARGS="--model ... -k 4" is used (and sys.argv ignored) when it is set. foundry/colab.py instead
runs this file in a subprocess from a small launcher, which gives vLLM a clean interpreter and its own process tree.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
LABELS = ("kind", "mode", "origin")  # copied from a prompt row to its completions


def _log(msg: str) -> None:
    print(f"[colab_job] {msg}", flush=True)


# ---------------------------------------------------------------- arguments

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default="/content/foundry/prompts.jsonl", help="input JSONL (id, messages)")
    ap.add_argument("--out-dir", default="/content/foundry/out", help="completions and timing are written here")
    ap.add_argument("--completions-name", default="completions.jsonl")
    ap.add_argument("--timing-name", default="timing.json")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or a local path")
    ap.add_argument("--revision", default=None, help="model repo commit to load (pin it: CLAUDE.md Rule 5 provenance)")
    ap.add_argument("--quantization", default=None, help="vLLM quantization method of the checkpoint (awq, gptq, fp8, ...); "
                    "omit for an unquantized or self-describing checkpoint")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=8192, help="prompt + completion tokens a sequence may use")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=0, help="0: vLLM's default")
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (saves memory on a tight card)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("-k", "--samples", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0: off (vLLM's default)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50, help="prompts per generate() call; output is saved after each (0: all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test; 0: all)")
    ap.add_argument("--chat-template-kwargs", default="{}", help="JSON object passed to apply_chat_template")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """argv, else $FOUNDRY_JOB_ARGS (shell-split), else sys.argv[1:]."""
    if argv is None:
        env = os.environ.get("FOUNDRY_JOB_ARGS", "").strip()
        argv = shlex.split(env) if env else sys.argv[1:]
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        a.chat_template_kwargs = json.loads(a.chat_template_kwargs)
    except json.JSONDecodeError as e:
        ap.error(f"--chat-template-kwargs is not JSON: {e}")
    if not isinstance(a.chat_template_kwargs, dict):
        ap.error("--chat-template-kwargs must be a JSON object")
    if a.samples < 1:
        ap.error("-k must be at least 1")
    if not 0.0 < a.gpu_memory_utilization <= 1.0:
        ap.error("--gpu-memory-utilization must be in (0, 1]")
    if a.temperature < 0 or not 0.0 < a.top_p <= 1.0 or a.max_tokens < 1 or a.max_model_len < 2 or a.top_k < 0:
        ap.error("bad sampling settings (temperature >= 0, 0 < top_p <= 1, top_k >= 0, max_tokens >= 1, max_model_len >= 2)")
    if a.chunk < 0 or a.limit < 0:
        ap.error("--chunk and --limit must be >= 0")
    return a


# ---------------------------------------------------------------- files

def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file ([] if missing). A torn final line is skipped; a bad line elsewhere raises."""
    if not path.is_file():
        return []
    lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise ValueError(f"{path}:{i + 1}: not valid JSON")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def read_prompts(path: Path, limit: int = 0) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path}: missing or empty")
    seen = set()
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not isinstance(r.get("id"), str) or not isinstance(msgs, list) or not msgs:
            raise SystemExit(f"{path}:{i + 1}: a prompt row needs a string 'id' and a non-empty 'messages' list")
        if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
               or not isinstance(m.get("content"), str) for m in msgs):
            raise SystemExit(f"{path}:{i + 1}: every message needs a role (system, user, assistant) and string content")
        if r["id"] in seen:
            raise SystemExit(f"{path}:{i + 1}: duplicate id {r['id']!r}")
        seen.add(r["id"])
        bad = _bad_override(r)
        if bad:
            raise SystemExit(f"{path}:{i + 1}: {bad}")
    return rows[:limit] if limit else rows


def _bad_override(row: dict) -> str | None:
    """Why a prompt row's per-prompt settings are unusable, else None."""
    for key in ("k", "max_tokens", "top_k"):
        v = row.get(key, 1)
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "top_k" else 1):
            return f"{key} must be a {'non-negative' if key == 'top_k' else 'positive'} integer, got {v!r}"
    t = row.get("temperature", 0.0)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
        return f"temperature must be >= 0, got {t!r}"
    tp = row.get("top_p", 1.0)
    if isinstance(tp, bool) or not isinstance(tp, (int, float)) or not 0 < tp <= 1:
        return f"top_p must be in (0, 1], got {tp!r}"
    if not isinstance(row.get("chat_template_kwargs", {}), dict):
        return "chat_template_kwargs must be an object"
    return None


def row_settings(p: dict, a: argparse.Namespace) -> dict:
    """The sampling settings of one prompt: its own overrides, else the job's."""
    return {"k": int(p.get("k", a.samples)), "temperature": float(p.get("temperature", a.temperature)),
            "top_p": float(p.get("top_p", a.top_p)), "top_k": int(p.get("top_k", a.top_k)),
            "max_tokens": int(p.get("max_tokens", a.max_tokens)),
            "chat_template_kwargs": dict(p.get("chat_template_kwargs", a.chat_template_kwargs))}


# ---------------------------------------------------------------- thinking

def split_thinking(text: str, prompt_open: bool = False) -> dict:
    """Split a raw completion into its reasoning and its answer (module docstring: "Thinking").

    ``prompt_open``: the rendered prompt already ends inside an open ``<think>`` block, so the completion holds only the
    closing tag. Returns {"thinking", "answer", "status"}; status is none / empty / closed / truncated."""
    close = text.find(THINK_CLOSE)
    if close >= 0:
        head = text[:close].lstrip()
        if head.startswith(THINK_OPEN):
            head = head[len(THINK_OPEN):]
        thinking = head.strip()
        return {"thinking": thinking, "answer": text[close + len(THINK_CLOSE):].strip(), "status": "closed" if thinking else "empty"}
    body = text.lstrip()
    if prompt_open or body.startswith(THINK_OPEN):
        if body.startswith(THINK_OPEN):
            body = body[len(THINK_OPEN):]
        return {"thinking": body.strip(), "answer": "", "status": "truncated"}
    return {"thinking": "", "answer": text.strip(), "status": "none"}


def think_close_id(tok) -> int | None:
    """Token id of ``</think>`` when the tokenizer has it as one token, else None."""
    fn = getattr(tok, "convert_tokens_to_ids", None)
    if fn is None:
        return None
    try:
        i = fn(THINK_CLOSE)
        unk = getattr(tok, "unk_token_id", None)
    except Exception:  # noqa: BLE001 - a tokenizer without the token: count by re-encoding instead
        return None
    return i if isinstance(i, int) and not isinstance(i, bool) and i >= 0 and i != unk else None


def count_split(token_ids: list[int], close_id: int | None, status: str, text: str, tok) -> tuple[int, int]:
    """(thinking tokens, answer tokens) of one completion; they add up to ``len(token_ids)``."""
    n = len(token_ids)
    if status == "none":
        return 0, n
    if status == "truncated":
        return n, 0
    if close_id is not None and close_id in token_ids:
        i = token_ids.index(close_id)
        return i + 1, n - i - 1
    head = text[: text.find(THINK_CLOSE) + len(THINK_CLOSE)]
    t = min(len(tok.encode(head, add_special_tokens=False)), n)
    return t, n - t


# ---------------------------------------------------------------- vLLM

def load_llm(a: argparse.Namespace):
    try:
        from vllm import LLM
    except ImportError as e:
        raise SystemExit("vllm is not installed on this machine (pip install vllm); this job needs a GPU VM") from e
    kw = {"model": a.model, "dtype": a.dtype, "max_model_len": a.max_model_len,
          "gpu_memory_utilization": a.gpu_memory_utilization, "seed": a.seed}
    if a.revision:
        kw["revision"] = kw["tokenizer_revision"] = a.revision
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.enforce_eager:
        kw["enforce_eager"] = True
    if a.trust_remote_code:
        kw["trust_remote_code"] = True
    return LLM(**kw)


def sampling_params(**kw):
    try:
        from vllm import SamplingParams
    except ImportError as e:
        raise SystemExit("vllm is not installed on this machine (pip install vllm)") from e
    return SamplingParams(**kw)


def render_prompt(tokenizer, messages: list[dict], template_kwargs: dict) -> tuple[str, list[int]]:
    """The prompt text under the model's own chat template, and its token ids. Tokenized here with
    add_special_tokens=False because the template already contains whatever special tokens the model wants."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    return text, list(tokenizer.encode(text, add_special_tokens=False))


def environment() -> dict:
    env = {"python": sys.version.split()[0]}
    for mod in ("vllm", "torch", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - absent or broken: record that, never fail the job over a version string
            env[mod] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        env["gpu"] = None
    return env


# ---------------------------------------------------------------- the job

def _timing_args(a: argparse.Namespace) -> dict:
    keep = ("model", "revision", "quantization", "dtype", "max_model_len", "gpu_memory_utilization", "samples", "temperature",
            "top_p", "top_k", "max_tokens", "seed", "chunk", "limit", "enforce_eager", "chat_template_kwargs", "prompts")
    return {k: getattr(a, k) for k in keep}


def run(a: argparse.Namespace, *, llm=None, make_params=None, log=_log) -> dict:
    """Generate every missing prompt; return the timing record. ``llm`` and ``make_params`` are for tests (a fake engine)."""
    prompts = read_prompts(Path(a.prompts), a.limit)
    out_dir = Path(a.out_dir)
    comp_path, timing_path = out_dir / a.completions_name, out_dir / a.timing_name
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {p["id"]: row_settings(p, a) for p in prompts}

    prior = read_jsonl(comp_path)
    have: dict[str, set[int]] = {}
    for r in prior:
        have.setdefault(r["id"], set()).add(r["sample"])
    complete = {i for i, s in have.items() if s >= set(range(settings[i]["k"] if i in settings else a.samples))}
    if len(complete) != len(have):
        prior = [r for r in prior if r["id"] in complete]   # drop prompts left half-written by a killed run
        write_jsonl(comp_path, prior)
    todo = [p for p in prompts if p["id"] not in complete]
    ks = sorted({settings[p["id"]]["k"] for p in todo})
    log(f"{len(prompts)} prompts, {len(complete)} already complete, {len(todo)} to generate, k = {ks[0] if len(ks) == 1 else ks}")

    old = {}
    if timing_path.is_file():
        try:
            old = json.loads(timing_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
    tot = {key: old.get(key, 0) for key in ("load_s", "gen_s", "output_tokens", "thinking_tokens", "answer_tokens", "prompt_tokens",
                                            "n_samples", "n_prompts")}
    tot["n_invocations"] = old.get("n_invocations", 0) + 1
    chunks = list(old.get("chunks", []))
    groups: dict[str, dict] = {g: dict(v) for g, v in (old.get("groups") or {}).items()}
    env: dict = {}

    def write_timing(done: bool) -> dict:
        t = {**tot, "chunks": chunks, "groups": groups, "complete": done, "model": a.model, "args": _timing_args(a), "env": env,
             "output_tokens_per_s": (tot["output_tokens"] / tot["gen_s"]) if tot["gen_s"] else None,
             "wall_s": round(time.perf_counter() - t_start, 3),
             "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        out_dir.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(t, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return t

    t_start = time.perf_counter()
    if not todo:
        log("nothing to generate")
        return write_timing(True)

    t0 = time.perf_counter()
    env.update(environment())  # imports vllm and torch, so their import time counts as start-up, like the weights
    llm = llm or load_llm(a)
    tot["load_s"] += round(time.perf_counter() - t0, 3)
    log(f"model loaded in {time.perf_counter() - t0:.1f} s")
    tok = llm.get_tokenizer()
    close_id = think_close_id(tok)
    make = make_params or sampling_params
    params_cache: dict[tuple, object] = {}

    def params_of(st: dict):
        key = (st["k"], st["temperature"], st["top_p"], st["top_k"], st["max_tokens"])
        if key not in params_cache:
            kw = {"n": st["k"], "temperature": st["temperature"], "top_p": st["top_p"], "max_tokens": st["max_tokens"]}
            if st["top_k"] > 0:
                kw["top_k"] = st["top_k"]
            params_cache[key] = make(**kw)
        return params_cache[key]

    size = a.chunk or len(todo)
    n_chunks = -(-len(todo) // size)
    with comp_path.open("a", encoding="utf-8", newline="\n") as out:
        for ci in range(n_chunks):
            batch = todo[ci * size:(ci + 1) * size]
            rendered = [render_prompt(tok, p["messages"], settings[p["id"]]["chat_template_kwargs"]) for p in batch]
            fits = [len(ids) < a.max_model_len - 1 for _, ids in rendered]
            live = [{"prompt_token_ids": ids} for (_, ids), ok in zip(rendered, fits) if ok]
            live_params = [params_of(settings[p["id"]]) for p, ok in zip(batch, fits) if ok]
            call_params = live_params[0] if len({id(x) for x in live_params}) == 1 else live_params  # one object, or one per prompt
            t1 = time.perf_counter()
            outs = iter(llm.generate(live, call_params, use_tqdm=False) if live else [])
            gen_s = time.perf_counter() - t1
            out_tokens = think_tokens = prompt_tokens = n_rows = 0
            for p, (prompt_text, ids), ok in zip(batch, rendered, fits):
                st = settings[p["id"]]
                spec_id = p.get("spec_id", p["id"])
                prompt_tokens += len(ids)  # once per prompt: the n samples share it
                if ok:
                    o = next(outs)
                    samples = sorted(o.outputs, key=lambda c: getattr(c, "index", 0))
                    rows = [(c.text, list(c.token_ids), c.finish_reason) for c in samples]
                else:
                    rows = [("", [], "prompt_too_long")] * st["k"]
                label = p.get("mode") or p.get("kind") or "generate"
                g = groups.setdefault(label, {"n_prompts": 0, "n_samples": 0, "output_tokens": 0, "thinking_tokens": 0,
                                              "answer_tokens": 0, "truncated_thinking": 0, "cut_off": 0})
                g["n_prompts"] += 1
                prompt_open = prompt_text.rstrip().endswith(THINK_OPEN)
                for s, (text, token_ids, reason) in enumerate(rows):
                    sp = split_thinking(text, prompt_open)
                    n_think, n_answer = count_split(token_ids, close_id, sp["status"], text, tok)
                    row = {"id": p["id"], "spec_id": spec_id, "sample": s, "text": text, "thinking": sp["thinking"],
                           "answer": sp["answer"], "think_status": sp["status"], "thinking_tokens": n_think,
                           "answer_tokens": n_answer, "prompt_tokens": len(ids), "completion_tokens": len(token_ids),
                           "finish_reason": reason, "model": a.model}
                    row.update({key: p[key] for key in LABELS if key in p})
                    out.write(json.dumps(row, sort_keys=True) + "\n")
                    out_tokens += len(token_ids)
                    think_tokens += n_think
                    n_rows += 1
                    g["n_samples"] += 1
                    g["output_tokens"] += len(token_ids)
                    g["thinking_tokens"] += n_think
                    g["answer_tokens"] += n_answer
                    g["truncated_thinking"] += sp["status"] == "truncated"
                    g["cut_off"] += reason == "length"
            out.flush()
            os.fsync(out.fileno())
            tot["gen_s"] += round(gen_s, 3)
            tot["output_tokens"] += out_tokens
            tot["thinking_tokens"] += think_tokens
            tot["answer_tokens"] += out_tokens - think_tokens
            tot["prompt_tokens"] += prompt_tokens
            tot["n_samples"] += n_rows
            tot["n_prompts"] += len(batch)
            chunks.append({"n_prompts": len(batch), "n_samples": n_rows, "gen_s": round(gen_s, 3), "output_tokens": out_tokens,
                           "thinking_tokens": think_tokens})
            write_timing(False)
            log(f"chunk {ci + 1}/{n_chunks}: {len(batch)} prompts, {n_rows} samples, {out_tokens} output tokens "
                f"({think_tokens} thinking) in {gen_s:.1f} s ({out_tokens / gen_s if gen_s else 0:.0f} tok/s)")
    t = write_timing(True)
    log(f"done: {tot['n_samples']} samples, {tot['output_tokens']} output tokens ({tot['thinking_tokens']} thinking), "
        f"load {tot['load_s']:.1f} s, generate {tot['gen_s']:.1f} s, {t['output_tokens_per_s'] or 0:.0f} tok/s")
    return t


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
