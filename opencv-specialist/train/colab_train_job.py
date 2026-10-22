#!/usr/bin/env python3
"""Stage 5 training job: 16-bit LoRA on Qwen3.5-9B with a dev pass@1 generation at every half-epoch.

STANDALONE. This file is copied to the Colab VM and run there; it imports nothing from this repository (tests pin that)
and only the standard library at import time. unsloth, torch, transformers, trl and datasets are imported inside the
functions that need them, so the data, subset, template and mask logic can be tested anywhere. ``train/stage5.py`` is the
local driver (predict, Colab commands, grading, report); ``train/train_lora.py`` is the Stage 1 script this one extends.

    python colab_train_job.py prefetch --repo unsloth/Qwen3.5-9B --revision SHA          # download the weights, print speed
    python colab_train_job.py check    --data-dir DIR                                      # template, markers, lengths (no GPU)
    python colab_train_job.py train    --data-dir DIR --out-dir OUT --run r100 --frac 1.0  # one training run
    python colab_train_job.py gen      --data-dir DIR --out-dir OUT --adapter A.tar ...    # dev answers from a saved adapter

Inputs in --data-dir: ``train.jsonl`` and ``dev.jsonl`` (rows with ``messages``, ``kind``, ``family``; the Stage 4 files) and,
for generation, ``dev_prompts.jsonl`` (``id`` and ``messages`` = [system, user] of each distinct cvbench dev prompt).

What one ``train`` run does (CLAUDE.md Rule 7: the base's own chat template, non-thinking, loss on the final answer only)
  1. Pre-flight, before any GPU time: every train and dev row rendered by the tokenizer's own chat template must equal
     ``render_row`` (the text ``train.config.render_turn`` defines for serving), carry exactly one response marker (the
     Qwen3.5 template puts the empty think block only in front of the LAST assistant turn, so a repair row trains only on its
     fix), and fit in --max-len (never truncate). The dev generation prompt is checked the same way.
  2. Load the model in 16-bit (--qlora: 4-bit, for the local smoke test), attach LoRA r=16 alpha=16 to every language-model
     projection (unsloth's Qwen3.5 pattern also covers the linear-attention in_proj_* / out_proj of 24 of the 32 layers),
     build the SFTTrainer, mask the loss with unsloth's ``train_on_responses_only`` and verify the labels of EVERY row: the
     labelled tokens must decode to exactly the final answer plus <|im_end|>.
  3. Log the step-0 dev loss (and the loss on a fixed probe of training rows, for the train-validation gap), optionally
     generate the step-0 dev answers, then train. A nested, stratified subset (--frac) of train.jsonl is used.
  4. At every --ckpt-every epochs (half-epochs by default): dev loss, probe loss, the mean training loss of the window, the
     adapter (a tar), and one greedy answer for every distinct dev prompt (T = 0, no repetition penalty, at most
     --dev-max-new-tokens, stops at <|im_end|>). The answers are graded on a CPU elsewhere; nothing here executes model code.

Outputs, all flat files named ``<run>.<what>`` in --out-dir (``colab download`` takes one file at a time), written as they
are produced so a disconnect loses at most the checkpoint in flight; ``<run>.manifest.json`` lists them with sizes and sha256:

    <run>.log.jsonl             events: env, preflight, load, eval0, one row per optimizer step, one per checkpoint, done
    <run>.<tag>.adapter.tar     the LoRA adapter (adapter_model.safetensors + adapter_config.json), tag = step0 | e0.5 | e1 | ...
    <run>.<tag>.dev_answers.jsonl   {prompt_id, text, n_tokens, finish: stop|length, ...} per distinct dev prompt
    <run>.summary.json          configuration, versions, GPU, timings, tokens/s, peak memory, the curve, state
    <run>.manifest.json

Resuming a half-finished run is not implemented (the optimizer state is not saved): a failed run keeps the checkpoints it
finished and is rerun from the start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "unsloth/Qwen3.5-9B"
DEFAULT_REVISION = "005429cee5cb648998cf2b70eebdd83175989c9a"  # the snapshot in the local HF cache (Stage 1 / 3 / 4 used it)

# Rule 7. Identical to train.config.BASES["qwen3.5-9b"]; tests/test_stage5.py pins that and the equality of render_row with
# the foundry's render_chat over every real row.
SYSTEM_START = "<|im_start|>system\n"
USER_MARK = "<|im_start|>user\n"
END = "<|im_end|>"
ANSWER_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
SEED = 3407
LORA_R = 16
LORA_ALPHA = 16


def _log(msg: str) -> None:
    print(f"[stage5_job] {msg}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- files

def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a missing file is an error here (the inputs must be there)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path}: missing")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i + 1}: not valid JSON ({e})")
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Outputs:
    """The flat output files of one run and their manifest (rewritten after every registered file)."""

    def __init__(self, out_dir: Path, run: str):
        self.dir, self.run = Path(out_dir), run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = {}
        self.state = "running"
        self.write_manifest()

    def path(self, what: str) -> Path:
        return self.dir / f"{self.run}.{what}"

    def event(self, kind: str, **kw) -> None:
        with self.path("log.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"t": _now(), "event": kind, **kw}, sort_keys=True) + "\n")

    def register(self, what: str, kind: str) -> None:
        p = self.path(what)
        self.files[p.name] = {"name": p.name, "kind": kind, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        self.write_manifest()

    def write_manifest(self) -> None:
        write_json_atomic(self.path("manifest.json"), {"run": self.run, "state": self.state, "updated": _now(),
                                                       "files": sorted(self.files.values(), key=lambda f: f["name"])})

    def finish(self, state: str) -> None:
        self.state = state
        self.write_manifest()


# ---------------------------------------------------------------- rows, subsets

def row_problems(row: dict) -> list[str]:
    """Why a training row is not a (system, user, [assistant, user]*, assistant) conversation of strings."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return ["messages is not a list of at least 3"]
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in msgs):
        return ["a message has no string content"]
    roles = [m.get("role") for m in msgs]
    ok = (roles[0] == "system" and roles[1] == "user" and roles[-1] == "assistant" and len(roles) % 2 == 1
          and all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles[1:])))
    return [] if ok else [f"roles {roles} are not system, user, [assistant, user]*, assistant"]


def validate_rows(rows: list[dict], label: str = "rows") -> str:
    """The one system prompt of the rows; raises SystemExit naming the first bad row."""
    systems = set()
    for i, r in enumerate(rows):
        bad = row_problems(r)
        if bad:
            raise SystemExit(f"{label}[{i}] ({r.get('id')}): {bad[0]}")
        systems.add(r["messages"][0]["content"])
    if len(systems) != 1:
        raise SystemExit(f"{label}: {len(systems)} different system prompts; Rule 7 wants exactly one")
    return systems.pop()


def stratum(row: dict) -> tuple[str, str]:
    return (str(row.get("kind") or "?"), str(row.get("family") or "(none)"))


def subset_order(rows: list[dict], seed: int = 0) -> list[int]:
    """Row indices in a nested, stratified order: every prefix is a balanced sample.

    Rows are grouped by (kind, family); each group is shuffled by a seed derived from the group's name (a string seed:
    identical in every process, unlike ``hash``), and its j-th row gets the key (j + 0.5) / group size. Sorting all rows by
    that key interleaves the groups in proportion to their sizes, so the first m rows hold each group's share of m to within
    one row, for every m. A 25 % subset is therefore inside the 50 % subset, which is inside the 100 % one, and all three have
    the composition of the full set (70 % solutions, 20 % repairs, 10 % replay; the same families).
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(stratum(r), []).append(i)
    ranked = []
    for key in sorted(groups):
        idx = sorted(groups[key])
        random.Random(f"{seed}|{key[0]}|{key[1]}").shuffle(idx)
        n = len(idx)
        ranked.extend(((j + 0.5) / n, key, i) for j, i in enumerate(idx))
    ranked.sort()
    return [i for _, _, i in ranked]


def subset_size(n: int, frac: float) -> int:
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    return max(1, min(n, int(n * frac + 0.5)))


def select_subset(rows: list[dict], frac: float, seed: int = 0) -> list[int]:
    """Indices (in file order) of the nested subset of ``frac`` of the rows."""
    return sorted(subset_order(rows, seed)[:subset_size(len(rows), frac)])


def probe_indices(rows: list[dict], n: int, seed: int = 0) -> list[int]:
    """The first ``n`` rows of the nested order: inside every subset (>= n rows), so all runs share one probe."""
    return sorted(subset_order(rows, seed)[:min(n, len(rows))])


# ---------------------------------------------------------------- the template (Rule 7)

def render_row(messages: list[dict]) -> str:
    """The text Qwen3.5's chat template makes of a training conversation (system, user, [assistant, user]*, assistant).

    The template trims every content; earlier assistant turns are plain, the last carries the empty think block
    (ANSWER_PREFIX). Equal to the tokenizer's own ``apply_chat_template`` (checked in pre-flight)."""
    text = f"{SYSTEM_START}{messages[0]['content'].strip()}{END}\n"
    for m in messages[1:-1]:
        text += f"<|im_start|>{m['role']}\n{m['content'].strip()}{END}\n"
    return text + ANSWER_PREFIX + messages[-1]["content"].strip() + END + "\n"


def render_prompt(system: str, user: str) -> str:
    """The serve-time prompt: system, user, and the assistant turn opened behind the empty think block."""
    return f"{SYSTEM_START}{system.strip()}{END}\n{USER_MARK}{user.strip()}{END}\n{ANSWER_PREFIX}"


def expected_response(messages: list[dict]) -> str:
    """What the loss may see: the final answer and its end token."""
    return messages[-1]["content"].strip() + END


def _input_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    enc = tokenizer(texts, add_special_tokens=False)
    return [list(x) for x in (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids)]


def preflight(tokenizer, rows: list[dict], max_len: int, label: str = "rows") -> dict:
    """Template equality, one response marker, and length, for every row. Raises SystemExit on any failure."""
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    mismatches = [i for i, (t, r) in enumerate(zip(texts, rows)) if t != render_row(r["messages"])]
    if mismatches:
        i = mismatches[0]
        raise SystemExit(f"{label}: {len(mismatches)} rows render differently under the tokenizer's chat template and "
                         f"render_row (Rule 7); first is row {i}:\n--- tokenizer\n{texts[i][:600]!r}\n"
                         f"--- render_row\n{render_row(rows[i]['messages'])[:600]!r}")
    markers = [i for i, t in enumerate(texts) if t.count(ANSWER_PREFIX) != 1]
    if markers:
        raise SystemExit(f"{label}: {len(markers)} rows without exactly one response marker; first is row {markers[0]} "
                         f"({rows[markers[0]].get('id')}): the loss mask would be wrong")
    lengths = [len(x) for x in _input_ids(tokenizer, texts)]
    over = [i for i, n in enumerate(lengths) if n > max_len]
    if over:
        raise SystemExit(f"{label}: {len(over)} rows are longer than --max-len {max_len} (longest {max(lengths)}): raise it, "
                         "never truncate")
    s = sorted(lengths)
    return {"label": label, "rows": len(rows), "mean": round(sum(s) / len(s), 1), "p50": s[len(s) // 2],
            "p99": s[min(len(s) - 1, int(0.99 * len(s)))], "max": s[-1], "template_mismatches": 0,
            "rows_without_one_marker": 0, "texts": texts, "lengths": lengths}


def check_prompt_template(tokenizer, system: str, user: str) -> None:
    """The dev generation prompt under the tokenizer's own template (non-thinking) must equal render_prompt."""
    got = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    want = render_prompt(system, user)
    if got != want:
        raise SystemExit(f"dev prompt drift (Rule 7):\n--- tokenizer\n{got[-300:]!r}\n--- render_prompt\n{want[-300:]!r}")


def check_labels(tokenizer, rows: list[dict], input_ids: list[list[int]], labels: list[list[int]], label: str = "rows") -> dict:
    """The labelled tokens of EVERY row must decode to exactly the final answer plus <|im_end|> (a trailing newline token is
    tolerated), and each labelled token must be the input token. Raises SystemExit on the first failure."""
    if not (len(rows) == len(input_ids) == len(labels)):
        raise SystemExit(f"{label}: {len(rows)} rows but {len(input_ids)} tokenized / {len(labels)} labelled "
                         "(the trainer dropped or added rows)")
    trained = 0
    for i, (r, ids, lab) in enumerate(zip(rows, input_ids, labels)):
        if len(ids) != len(lab):
            raise SystemExit(f"{label}[{i}]: {len(ids)} input tokens, {len(lab)} labels")
        kept = [t for t, y in zip(ids, lab) if y != -100]
        if any(y != -100 and y != t for t, y in zip(ids, lab)):
            raise SystemExit(f"{label}[{i}]: a label differs from its input token")
        seen = tokenizer.decode(kept)
        if seen.strip() != expected_response(r["messages"]).strip():
            raise SystemExit(f"{label}[{i}] ({r.get('id')}): the loss would see the wrong span.\n--- sees\n{seen[:500]!r}\n"
                             f"--- should see\n{expected_response(r['messages'])[:500]!r}")
        trained += len(kept)
    return {"label": label, "rows": len(rows), "trained_tokens": trained, "mean_trained_tokens": round(trained / max(1, len(rows)), 1)}


# ---------------------------------------------------------------- checkpoints

def ckpt_plan(max_steps: int, epochs: float, every: float) -> dict[int, str]:
    """{optimizer step: tag} of the checkpoints: every ``every`` epochs, the last at ``max_steps``. The step of epoch e is
    round(max_steps * e / epochs); the tag is e0.5, e1, e1.5, e2 ..."""
    if max_steps < 1 or epochs <= 0 or every <= 0:
        raise ValueError("need max_steps >= 1, epochs > 0, every > 0")
    plan: dict[int, str] = {}
    k = 1
    while k * every <= epochs + 1e-9:
        e = k * every
        step = min(max_steps, max(1, int(max_steps * e / epochs + 0.5)))
        plan.setdefault(step, f"e{e:g}")
        k += 1
    plan[max_steps] = f"e{epochs:g}"                # the final step is always the end of training
    return dict(sorted(plan.items()))


def adapter_tag(tar) -> str:
    """The checkpoint tag of ``<run>.<tag>.adapter.tar`` (the tag may hold a dot: r025.e0.5.adapter.tar -> e0.5)."""
    name = Path(tar).name
    stem = name[:-len(".adapter.tar")] if name.endswith(".adapter.tar") else Path(tar).stem
    return stem.split(".", 1)[1] if "." in stem else stem


def mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def window_stats(log_history: list[dict], lo: int, hi: int) -> dict:
    """Mean training loss of the logged optimizer steps lo < step <= hi."""
    losses = [h["loss"] for h in log_history if "loss" in h and lo < h.get("step", -1) <= hi]
    return {"train_loss_window": mean(losses), "steps_in_window": len(losses)}


def stop_cut(ids: list[int], stop_ids: set[int]) -> tuple[list[int], str]:
    """A generated id list cut at the first stop token (dropped); finish is 'stop' if one was found, else 'length'."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i], "stop"
    return ids, "length"


# ---------------------------------------------------------------- the model side (heavy imports inside)

def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "datasets", "bitsandbytes", "triton", "fla",
                "causal_conv1d", "xformers", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception:  # noqa: BLE001 - absent or broken: record that, never fail over a version string
            out[mod] = None
    return out


def gpu_info() -> dict:
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "total_gib": round(p.total_memory / 2**30, 2), "bf16": bool(torch.cuda.is_bf16_supported())}
    except Exception:  # noqa: BLE001
        return {"name": None}


def cuda_mem_gib() -> dict:
    import torch
    return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def stop_token_ids(tokenizer) -> set[int]:
    ids = set()
    for tok in (END, "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(i, int) and i >= 0 and i != getattr(tokenizer, "unk_token_id", None):
            ids.add(i)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def generate_dev(model, tokenizer, prompts: list[dict], *, max_new_tokens: int, batch_size: int = 0, log=_log) -> list[dict]:
    """One greedy answer per prompt (T = 0, no repetition penalty, stop at <|im_end|>), left-padded batches, longest prompts
    first. The model is put in inference mode for the call and back in training mode after it. On a CUDA OOM the batch is
    halved. Returns rows {prompt_id, text, n_tokens, finish, prompt_tokens}."""
    import torch
    from unsloth import FastModel

    stops = stop_token_ids(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(stops)
    texts = {p["id"]: tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for p in prompts}
    order = sorted(prompts, key=lambda p: -len(texts[p["id"]]))
    results: dict[str, dict] = {}
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    FastModel.for_inference(model)
    try:
        bs = batch_size or len(order)
        pos = 0
        while pos < len(order):
            batch = order[pos:pos + bs]
            enc = out = None
            try:
                enc = tokenizer([texts[p["id"]] for p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                         top_k=None, repetition_penalty=1.0, eos_token_id=sorted(stops), pad_token_id=pad,
                                         use_cache=True)
            except torch.cuda.OutOfMemoryError:
                enc = out = None
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log(f"dev generation: CUDA out of memory, retrying with batches of {bs}")
                continue
            width = enc["input_ids"].shape[1]
            n_prompt = enc["attention_mask"].sum(dim=1).tolist()
            for j, (p, row) in enumerate(zip(batch, out[:, width:].tolist())):
                ids, finish = stop_cut(row, stops)
                results[p["id"]] = {"prompt_id": p["id"], "text": tokenizer.decode(ids, skip_special_tokens=False),
                                    "n_tokens": len(ids), "finish": finish, "prompt_tokens": int(n_prompt[j])}
            pos += len(batch)
            del enc, out
    finally:
        tokenizer.padding_side = old_side
        FastModel.for_training(model, use_gradient_checkpointing="unsloth")
    return [results[p["id"]] for p in prompts]


def save_adapter_tar(model, dest_tar: Path, work: Path) -> None:
    """model.save_pretrained(work) (adapter weights and config only), then one uncompressed tar (safetensors do not compress)."""
    work.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(work))
    with tarfile.open(dest_tar, "w") as tf:
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            tf.add(work / name, arcname=name)
    shutil.rmtree(work, ignore_errors=True)             # the tar is the artifact; do not keep a second copy on the VM disk


def load_model(a, max_len: int):
    """unsloth FastModel in 16-bit (or 4-bit with --qlora) with LoRA on the language model. Returns (model, tokenizer)."""
    import torch  # noqa: F401
    import unsloth  # noqa: F401  (before transformers, trl and peft)
    from unsloth import FastModel

    kw = {"model_name": a.repo, "max_seq_length": max_len, "load_in_4bit": bool(a.qlora), "dtype": None}
    if a.revision:
        kw["revision"] = a.revision
    model, processor = FastModel.from_pretrained(**kw)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True, r=a.lora_r,
                                     lora_alpha=a.lora_alpha, lora_dropout=0, bias="none",
                                     use_gradient_checkpointing="unsloth", random_state=SEED)
    return model, processor, tokenizer


# ---------------------------------------------------------------- the commands

def cmd_check(a) -> int:
    from transformers import AutoTokenizer
    kw = {"revision": a.revision} if a.revision else {}
    tok = AutoTokenizer.from_pretrained(a.repo, **kw)
    rows = read_jsonl(Path(a.data_dir) / "train.jsonl")
    dev = read_jsonl(Path(a.data_dir) / "dev.jsonl")
    system = validate_rows(rows, "train")
    if validate_rows(dev, "dev") != system:
        raise SystemExit("train and dev rows carry different system prompts")
    for name, rs in (("train", rows), ("dev", dev)):
        s = preflight(tok, rs, a.max_len, name)
        print(json.dumps({k: v for k, v in s.items() if k not in ("texts", "lengths")}))
    prompts = read_jsonl(Path(a.data_dir) / "dev_prompts.jsonl") if (Path(a.data_dir) / "dev_prompts.jsonl").is_file() else []
    for p in prompts:
        check_prompt_template(tok, p["messages"][0]["content"], p["messages"][1]["content"])
    print(f"ok: {len(rows)} train rows, {len(dev)} dev rows, {len(prompts)} dev prompts under the tokenizer's own template")
    return 0


def cmd_prefetch(a) -> int:
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(a.repo, revision=a.revision or None)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    dt = time.time() - t0
    _log(f"prefetch: {a.repo}@{a.revision} -> {path}: {size / 2**30:.1f} GiB in {dt:.0f} s ({size / 2**20 / max(dt, 1e-9):.0f} MiB/s)")
    return 0


class Job:
    """The state of one training run (everything the callback needs)."""

    def __init__(self, a, out: Outputs, rows, sub_idx, dev_rows, prompts, probe_idx):
        self.a, self.out = a, out
        self.rows, self.sub_idx, self.dev_rows, self.prompts, self.probe_idx = rows, sub_idx, dev_rows, prompts, probe_idx
        self.t_job = time.perf_counter()
        self.summary: dict = {"run": a.run, "frac": a.frac, "n_train": len(sub_idx), "n_dev": len(dev_rows),
                              "n_dev_prompts": len(prompts), "epochs": a.epochs, "ckpt_every": a.ckpt_every,
                              "batch": a.batch, "grad_accum": a.grad_accum, "lr": a.lr, "warmup_ratio": a.warmup_ratio,
                              "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "max_len": a.max_len, "qlora": bool(a.qlora),
                              "repo": a.repo, "revision": a.revision, "seed": SEED, "state": "running", "checkpoints": []}
        self.step_s_total = 0.0
        self.step_t0 = 0.0
        self.last_ckpt_step = 0
        self.last_ckpt_tokens = 0
        self.last_ckpt_train_s = 0.0
        self.plan: dict[int, str] = {}
        self.in_checkpoint = False
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.probe_ds = None

    def save_summary(self) -> None:
        self.summary["wall_s"] = round(time.perf_counter() - self.t_job, 1)
        write_json_atomic(self.out.path("summary.json"), self.summary)
        self.out.register("summary.json", "summary")
        self.out.register("log.jsonl", "log")

    # -- losses
    def eval_losses(self) -> dict:
        t0 = time.perf_counter()
        res = {"dev_loss": round(float(self.trainer.evaluate()["eval_loss"]), 5)}
        if self.probe_ds is not None:
            res["probe_loss"] = round(float(self.trainer.evaluate(eval_dataset=self.probe_ds, metric_key_prefix="probe")["probe_loss"]), 5)
        res["eval_s"] = round(time.perf_counter() - t0, 1)
        return res

    # -- dev answers
    def dev_answers(self, tag: str) -> dict:
        if self.a.dev_gen == "off":
            return {}
        t0 = time.perf_counter()
        import torch
        torch.cuda.reset_peak_memory_stats()
        rows = generate_dev(self.model, self.tokenizer, self.prompts, max_new_tokens=self.a.dev_max_new_tokens,
                            batch_size=self.a.dev_batch)
        gen_s = time.perf_counter() - t0
        for r in rows:
            r.update(run=self.a.run, ckpt=tag)
        p = self.out.path(f"{tag}.dev_answers.jsonl")
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        self.out.register(f"{tag}.dev_answers.jsonl", "answers")
        toks = sum(r["n_tokens"] for r in rows)
        return {"gen_s": round(gen_s, 1), "gen_tokens": toks, "gen_tok_s": round(toks / max(gen_s, 1e-9), 1),
                "gen_longest": max(r["n_tokens"] for r in rows), "gen_cut_at_cap": sum(r["finish"] == "length" for r in rows),
                "gen_peak_gib": cuda_mem_gib()["peak_allocated_gib"]}

    def save_adapter(self, tag: str) -> float:
        t0 = time.perf_counter()
        save_adapter_tar(self.model, self.out.path(f"{tag}.adapter.tar"), self.out.dir / f"{self.a.run}.{tag}.adapter")
        self.out.register(f"{tag}.adapter.tar", "adapter")
        return round(time.perf_counter() - t0, 1)

    def step0(self) -> None:
        losses = self.eval_losses()
        rec = {"tag": "step0", "step": 0, "epoch": 0.0, **losses}
        _log(f"step 0: {losses}")
        self.out.event("eval0", **losses)
        if self.a.step0_gen:
            rec.update(self.dev_answers("step0"))
            rec["adapter_s"] = self.save_adapter("step0")
        rec.update(cuda_mem_gib())
        self.summary["step0"] = rec
        self.summary["dev_loss_step0"] = losses["dev_loss"]
        self.save_summary()

    # -- called by the callback
    def checkpoint(self, state, tag: str) -> None:
        import torch
        self.in_checkpoint = True
        try:
            t0 = time.perf_counter()
            mem_train = cuda_mem_gib()
            win = window_stats(state.log_history, self.last_ckpt_step, state.global_step)
            tokens = int(state.num_input_tokens_seen)
            train_s = self.step_s_total - self.last_ckpt_train_s
            rec = {"tag": tag, "step": int(state.global_step), "epoch": round(float(state.epoch or 0.0), 3), **win,
                   "tokens_window": tokens - self.last_ckpt_tokens, "train_s_window": round(train_s, 1),
                   "tok_s_window": round((tokens - self.last_ckpt_tokens) / max(train_s, 1e-9), 1),
                   "peak_train_gib": mem_train["peak_allocated_gib"], "peak_train_reserved_gib": mem_train["peak_reserved_gib"]}
            rec.update(self.eval_losses())
            rec["adapter_s"] = self.save_adapter(tag)
            rec.update(self.dev_answers(tag))
            rec["checkpoint_s"] = round(time.perf_counter() - t0, 1)
            torch.cuda.reset_peak_memory_stats()
            self.summary["checkpoints"].append(rec)
            self.out.event("checkpoint", **rec)
            _log(f"checkpoint {tag}: " + json.dumps({k: v for k, v in rec.items() if k != "tag"}, sort_keys=True))
            self.last_ckpt_step, self.last_ckpt_tokens, self.last_ckpt_train_s = state.global_step, tokens, self.step_s_total
            self.save_summary()
        finally:
            self.in_checkpoint = False


def make_callback(job: Job):
    from transformers import TrainerCallback

    class Stage5Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            job.plan = ckpt_plan(state.max_steps, job.a.epochs, job.a.ckpt_every)
            job.summary["max_steps"] = state.max_steps
            job.summary["ckpt_plan"] = {str(k): v for k, v in job.plan.items()}
            _log(f"{state.max_steps} optimizer steps; checkpoints at {job.plan}")

        def on_step_begin(self, args, state, control, **kw):
            job.step_t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            job.step_s_total += time.perf_counter() - job.step_t0

        def on_log(self, args, state, control, logs=None, **kw):
            if job.in_checkpoint or not logs or "loss" not in logs:
                return                                    # an evaluation's own log row, or the final summary row
            step = int(state.global_step)
            job.out.event("step", step=step, epoch=round(float(state.epoch or 0.0), 4), loss=logs["loss"],
                          lr=logs.get("learning_rate"), grad_norm=logs.get("grad_norm"),
                          tokens_seen=int(state.num_input_tokens_seen), train_s=round(job.step_s_total, 2))
            if step % 5 == 0 or step in job.plan:
                _log(f"step {step}/{state.max_steps} loss {logs['loss']:.4f} lr {logs.get('learning_rate', 0):.2e} "
                     f"tokens {int(state.num_input_tokens_seen)} train {job.step_s_total:.0f} s")
            if step in job.plan:
                job.checkpoint(state, job.plan[step])

    return Stage5Callback()


def cmd_train(a) -> int:
    t_start = time.perf_counter()
    data = Path(a.data_dir)
    rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")
    system = validate_rows(rows, "train")
    if validate_rows(dev_rows, "dev") != system:
        raise SystemExit("train and dev rows carry different system prompts")
    sub_idx = select_subset(rows, a.frac, a.seed)
    sub = [rows[i] for i in sub_idx]
    probe_idx = probe_indices(rows, a.probe_rows, a.seed)
    probe = [rows[i] for i in probe_idx]
    prompts = read_jsonl(data / "dev_prompts.jsonl") if a.dev_gen != "off" else []
    if a.dev_limit:
        prompts = prompts[:a.dev_limit]
    out = Outputs(a.out_dir, a.run)
    job = Job(a, out, rows, sub_idx, dev_rows, prompts, probe_idx)
    job.summary.update(versions=versions(), gpu=gpu_info(), args={k: v for k, v in vars(a).items() if k != "fn"})
    out.event("env", **{k: job.summary[k] for k in ("versions", "gpu")})
    _log(f"run {a.run}: {len(sub)} of {len(rows)} train rows (frac {a.frac}), {len(dev_rows)} dev rows, {len(prompts)} dev prompts")

    try:
        import unsloth  # noqa: F401  (before transformers, trl and peft, so every unsloth patch applies)
        # 1. pre-flight on the tokenizer alone: nothing is loaded onto the GPU if the template, markers or lengths are wrong
        from transformers import AutoTokenizer
        tk_kw = {"revision": a.revision} if a.revision else {}
        tok0 = AutoTokenizer.from_pretrained(a.repo, **tk_kw)
        pf = {n: preflight(tok0, rs, a.max_len, n) for n, rs in (("train", rows), ("dev", dev_rows))}
        for p in prompts:
            check_prompt_template(tok0, p["messages"][0]["content"], p["messages"][1]["content"])
        sub_lengths = [pf["train"]["lengths"][i] for i in sub_idx]
        job.summary["preflight"] = {n: {k: v for k, v in s.items() if k not in ("texts", "lengths")} for n, s in pf.items()}
        job.summary["subset"] = {"rows": len(sub), "tokens": sum(sub_lengths), "mean_tokens": round(sum(sub_lengths) / len(sub), 1),
                                 "kinds": {k: sum(1 for r in sub if r.get("kind") == k) for k in sorted({r.get("kind") for r in sub}, key=str)}}
        out.event("preflight", **job.summary["preflight"], subset=job.summary["subset"])
        _log(f"pre-flight ok: {job.summary['subset']}")
        texts = pf["train"]["texts"]
        train_texts = [texts[i] for i in sub_idx]
        dev_texts = pf["dev"]["texts"]
        probe_texts = [texts[i] for i in probe_idx]

        # 2. model, LoRA, trainer
        import torch
        t = time.perf_counter()
        model, processor, tokenizer = load_model(a, a.max_len)
        job.model, job.tokenizer = model, tokenizer
        job.summary["dtype"] = str(next(p for p in model.parameters() if p.is_floating_point()).dtype)
        job.summary["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        job.summary["load_s"] = round(time.perf_counter() - t, 1)
        job.summary["load_mem"] = cuda_mem_gib()
        out.event("load", **{k: job.summary[k] for k in ("dtype", "trainable_params", "load_s", "load_mem")})
        _log(f"model loaded in {job.summary['load_s']} s; {job.summary['trainable_params']:,} trainable parameters; {job.summary['load_mem']}")

        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth.chat_templates import train_on_responses_only
        os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(Path(a.out_dir) / f"{a.run}.tb"))
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,                       # as train_lora.py (Stage 1): the text tokenizer
            train_dataset=Dataset.from_dict({"text": train_texts}), eval_dataset=Dataset.from_dict({"text": dev_texts}),
            args=SFTConfig(
                dataset_text_field="text", max_length=a.max_len, dataset_num_proc=1,
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, per_device_eval_batch_size=1,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=a.warmup_ratio, num_train_epochs=a.epochs,
                max_steps=a.max_steps, logging_steps=1, eval_strategy="no", save_strategy="no",
                # without this every evaluation keeps the logits of every row on the host: 248,320 vocabulary entries x ~1,100
                # tokens x 4 bytes = 1.1 GB per dev row (measured on the 3060: host memory grew 170 MB/s until the watchdog fired)
                prediction_loss_only=True,
                bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                output_dir=str(Path(a.out_dir) / f"{a.run}.trainer"), optim="adamw_8bit", seed=SEED, report_to="none",
                disable_tqdm=True, include_num_input_tokens_seen="all"))
        mask_kw = dict(instruction_part=USER_MARK, response_part=ANSWER_PREFIX)
        try:
            trainer = train_on_responses_only(trainer, last_response_only=True, **mask_kw)
            job.summary["last_response_only"] = True
        except TypeError:                                   # an unsloth without the argument: the marker alone selects the last turn
            trainer = train_on_responses_only(trainer, **mask_kw)
            job.summary["last_response_only"] = False
        job.trainer = trainer

        # 3. the loss mask, for every row
        tr_ds, ev_ds = trainer.train_dataset, trainer.eval_dataset
        m_train = check_labels(tokenizer, sub, [list(x) for x in tr_ds["input_ids"]], [list(x) for x in tr_ds["labels"]], "train")
        m_dev = check_labels(tokenizer, dev_rows, [list(x) for x in ev_ds["input_ids"]], [list(x) for x in ev_ds["labels"]], "dev")
        job.summary["loss_mask"] = {"train": m_train, "dev": m_dev, "ok": True}
        out.event("loss_mask", **job.summary["loss_mask"])
        _log(f"loss mask ok on all {m_train['rows']} train and {m_dev['rows']} dev rows "
             f"({m_train['mean_trained_tokens']} trained tokens per train row)")

        # the probe: fixed training rows evaluated like the dev rows (the train-validation gap)
        try:
            mask_fn = train_on_responses_only(trainer, return_function=True, **mask_kw)
            enc = tokenizer(probe_texts, add_special_tokens=False)
            probe_ds = Dataset.from_dict({"input_ids": [list(x) for x in enc["input_ids"]],
                                          "attention_mask": [list(x) for x in enc["attention_mask"]]}).map(mask_fn, batched=True)
            check_labels(tokenizer, probe, [list(x) for x in probe_ds["input_ids"]], [list(x) for x in probe_ds["labels"]], "probe")
            job.probe_ds = probe_ds
        except Exception as e:  # noqa: BLE001 - a diagnostic extra must not sink a billed run; recorded, not hidden
            job.summary["probe_error"] = repr(e)
            _log(f"WARNING: the training-row probe is disabled: {e!r}")

        trainer.add_callback(make_callback(job))
        job.summary["prepare_s"] = round(time.perf_counter() - t_start - job.summary["load_s"], 1)

        # 4. step 0, then train
        job.step0()
        torch.cuda.reset_peak_memory_stats()
        t_train = time.perf_counter()
        result = trainer.train()
        job.summary["train"] = {
            "runtime_s": round(time.perf_counter() - t_train, 1), "step_s": round(job.step_s_total, 1),
            "steps": int(trainer.state.global_step), "tokens_seen": int(trainer.state.num_input_tokens_seen),
            "tokens_per_s": round(int(trainer.state.num_input_tokens_seen) / max(job.step_s_total, 1e-9), 1),
            "final_train_loss": round(float(result.metrics.get("train_loss", float("nan"))), 5)}
        missing = [t for s, t in job.plan.items() if not any(c["tag"] == t for c in job.summary["checkpoints"])]
        if missing:
            raise RuntimeError(f"training ended without the checkpoints {missing}")
        job.summary["state"] = "complete"
        job.save_summary()
        out.event("done", **job.summary["train"])
        out.finish("complete")
        _log("done: " + json.dumps(job.summary["train"]))
        return 0
    except BaseException as e:
        job.summary["state"] = "failed"
        job.summary["error"] = repr(e)
        try:
            job.save_summary()
            out.event("failed", error=repr(e))
            out.register("log.jsonl", "log")
        finally:
            out.finish("failed")
        raise


def cmd_gen(a) -> int:
    """Dev answers from saved adapters (``--adapter RUN.TAG.adapter.tar`` ...) without training: for a run whose generation failed."""
    data = Path(a.data_dir)
    prompts = read_jsonl(data / "dev_prompts.jsonl")
    if a.dev_limit:
        prompts = prompts[:a.dev_limit]
    out = Outputs(a.out_dir, a.run)
    model, processor, tokenizer = load_model(a, a.max_len)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    for tar in a.adapter:
        tag = adapter_tag(tar)
        work = Path(a.out_dir) / f"{a.run}.{tag}.adapter_in"
        with tarfile.open(tar) as tf:
            tf.extractall(work)
        res = set_peft_model_state_dict(model, load_file(str(work / "adapter_model.safetensors")))
        _log(f"loaded adapter {tag}: {len(getattr(res, 'unexpected_keys', []) or [])} unexpected keys")
        t0 = time.perf_counter()
        rows = generate_dev(model, tokenizer, prompts, max_new_tokens=a.dev_max_new_tokens, batch_size=a.dev_batch)
        for r in rows:
            r.update(run=a.run, ckpt=tag)
        out.path(f"{tag}.dev_answers.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        out.register(f"{tag}.dev_answers.jsonl", "answers")
        _log(f"{tag}: {len(rows)} answers in {time.perf_counter() - t0:.0f} s")
    out.finish("complete")
    return 0


# ---------------------------------------------------------------- arguments

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        p.add_argument("--repo", default=DEFAULT_REPO)
        p.add_argument("--revision", default=DEFAULT_REVISION, help="model repo commit (CLAUDE.md Rule 5); '' for the default branch")
        if data:
            p.add_argument("--data-dir", default="/content/stage5_data", help="train.jsonl, dev.jsonl, dev_prompts.jsonl")
            p.add_argument("--max-len", type=int, default=4608, help="tokens per example; the dataset max is 4,390 and rows are never truncated")

    c = sub.add_parser("check", help="template, marker and length pre-flight on the tokenizer alone (no GPU)")
    common(c)
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("prefetch", help="download the model snapshot")
    common(f, data=False)
    f.set_defaults(fn=cmd_prefetch)

    def model_args(p):
        p.add_argument("--out-dir", default="/content/stage5_out")
        p.add_argument("--run", required=True, help="run name, e.g. r100; prefix of every output file")
        p.add_argument("--qlora", action="store_true", help="4-bit base (the local smoke test); the Colab runs are 16-bit")
        p.add_argument("--lora-r", type=int, default=LORA_R)
        p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
        p.add_argument("--dev-gen", choices=("on", "off"), default="on")
        p.add_argument("--dev-max-new-tokens", type=int, default=2048)
        p.add_argument("--dev-batch", type=int, default=0, help="prompts per generate() call; 0: all at once")
        p.add_argument("--dev-limit", type=int, default=0, help="only the first N dev prompts (smoke test)")

    t = sub.add_parser("train", help="one training run")
    common(t)
    model_args(t)
    t.add_argument("--frac", type=float, default=1.0, help="nested stratified subset of train.jsonl")
    t.add_argument("--seed", type=int, default=0, help="subset seed (the trainer seed is fixed at 3407)")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--max-steps", type=int, default=-1, help="override the epochs (smoke test)")
    t.add_argument("--ckpt-every", type=float, default=0.5, help="epochs between checkpoints")
    t.add_argument("--batch", type=int, default=1)
    t.add_argument("--grad-accum", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup-ratio", type=float, default=0.05)
    t.add_argument("--probe-rows", type=int, default=64, help="training rows evaluated at every checkpoint (0: none)")
    t.add_argument("--step0-gen", action="store_true", help="also generate and save the dev answers of the untrained model")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("gen", help="dev answers from saved adapters")
    common(g)
    model_args(g)
    g.add_argument("--adapter", action="append", required=True, help="RUN.TAG.adapter.tar, repeatable")
    g.set_defaults(fn=cmd_gen)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "frac", 1.0) <= 0 or getattr(a, "frac", 1.0) > 1:
        raise SystemExit("--frac must be in (0, 1]")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
