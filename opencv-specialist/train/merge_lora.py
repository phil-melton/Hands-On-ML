"""Merge a PEFT LoRA adapter into safetensors checkpoint shards, one tensor at a time (Stage 6).

Why not ``PeftModel.merge_and_unload``: it needs the whole 9B in memory (19 GB bf16), and this machine's binding
limit is Windows commit charge (LOGBOOK.md; WSL is capped at 14 GB). This module reads a shard, adds the LoRA delta to
each targeted weight in float32, rounds back to bf16 (round to nearest even, the cast torch does) and writes the same
shard layout. Peak memory is the adapter (173 MB, float32) plus one weight and a few row blocks (measured on
Qwen3.5-9B: 1.5 GiB peak RSS for the merge with a full verification, 1.7 GiB for a verify-only pass). It needs numpy
only, so the tests run in every venv.

bf16 cannot hold a small LoRA delta exactly: after rounding, the realized delta differs from B @ A by 15-32 % of its
norm (cosine 0.95-0.99), a zero-mean rounding noise comparable to the weights' own bf16 resolution. ``verify`` reports
it per module type.

The output shard is the input shard with the data bytes of the targeted tensors replaced: the JSON header (names,
dtypes, shapes, offsets) is copied byte for byte, and every untouched tensor is a byte-identical copy.

    W' = W + scale * (B @ A),   scale = lora_alpha / r   (lora_alpha / sqrt(r) with rsLoRA, which is refused here)

Key mapping (PEFT adapter key -> base weight key), printed by the CLI:

    base_model.model.<module>.lora_A.weight   ->  <module>.weight      (A: r x in)
    base_model.model.<module>.lora_B.weight   ->  <module>.weight      (B: out x r)

The module path is kept as it is: Unsloth/PEFT saved ``base_model.model.model.language_model.layers.N.<sub>`` and the
base checkpoint names the same module ``model.language_model.layers.N.<sub>``. Only if that exact key is missing does
the loader try ``model.layers.N`` -> ``model.language_model.layers.N`` (a text-only adapter on a multimodal base).

Run from opencv-specialist/:

    python -m train.merge_lora BASE_SNAPSHOT ADAPTER(.tar|dir) OUT_DIR [--verify sample|all|none]
    python -m train.merge_lora BASE_SNAPSHOT ADAPTER OUT_DIR --verify-only --verify all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAPPING_RULE = ("base_model.model.<module>.lora_{A,B}.weight -> <module>.weight  "
                "(fallback: model.layers.N.* -> model.language_model.layers.N.*, only if the exact key is absent)")

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
MERGEABLE = ("BF16", "F16", "F32")  # dtypes a targeted weight may have
CHUNK = 64 << 20
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
# Files of the base snapshot that the GGUF converter and the tokenizer need; weights and the index are handled apart.
AUX_SKIP = {"README.md", ".gitattributes"}


class MergeError(RuntimeError):
    pass


# ------------------------------------------------------------------ safetensors, by hand

@dataclass
class Header:
    raw: bytes                       # the 8-byte length plus the JSON, exactly as in the file
    tensors: dict[str, dict]         # name -> {"dtype", "shape", "data_offsets"}
    data_start: int
    file_size: int

    def ordered(self) -> list[tuple[str, dict]]:
        return sorted(self.tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])


def read_header(path: Path) -> Header:
    """Parse and validate a safetensors header: contiguous, in-bounds data ranges whose sizes match dtype and shape."""
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise MergeError(f"{path}: shorter than a safetensors header")
        (n,) = struct.unpack("<Q", head)
        if n > 256 << 20 or 8 + n > size:
            raise MergeError(f"{path}: implausible header length {n}")
        js = f.read(n)
    meta = json.loads(js)
    meta.pop("__metadata__", None)
    h = Header(raw=head + js, tensors=meta, data_start=8 + n, file_size=size)
    pos = 0
    for name, t in h.ordered():
        b, e = t["data_offsets"]
        if t["dtype"] not in DTYPE_BYTES:
            raise MergeError(f"{path}: tensor {name} has unknown dtype {t['dtype']}")
        want = DTYPE_BYTES[t["dtype"]] * math.prod(t["shape"])
        if b != pos or e - b != want:
            raise MergeError(f"{path}: tensor {name} range {b}:{e} is not contiguous or does not match shape {t['shape']}")
        pos = e
    if h.data_start + pos != size:
        raise MergeError(f"{path}: data ends at {h.data_start + pos}, file has {size} bytes")
    return h


def read_exact(f, n: int) -> np.ndarray:
    """Read exactly n bytes from the current position of a binary file into a fresh uint8 array."""
    buf = np.empty(n, dtype=np.uint8)
    mv, got = memoryview(buf), 0
    while got < n:
        k = f.readinto(mv[got:])
        if not k:
            raise MergeError(f"unexpected end of file ({got} of {n} bytes)")
        got += k
    return buf


def read_tensor_raw(f, h: Header, name: str) -> np.ndarray:
    """The raw bytes of one tensor as uint8 (f is an open binary file of the shard)."""
    b, e = h.tensors[name]["data_offsets"]
    f.seek(h.data_start + b)
    return read_exact(f, e - b)


_NP = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "BF16": np.uint16}


def decode(raw: np.ndarray, dtype: str, shape: list[int]) -> np.ndarray:
    """Raw bytes -> float32 array (bf16 by shifting its 16 bits into the top of a float32)."""
    if dtype == "BF16":
        u = raw.view(np.uint16).astype(np.uint32)
        u <<= np.uint32(16)
        return u.view(np.float32).reshape(shape)
    if dtype in _NP:
        return raw.view(_NP[dtype]).reshape(shape).astype(np.float32, copy=False)
    raise MergeError(f"cannot decode dtype {dtype}")


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), round to nearest even; NaN stays NaN, overflow goes to inf."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    b = x.view(np.uint32)
    rounded = b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = ((b[nan] >> np.uint32(16)) | np.uint32(0x40)).astype(np.uint16)  # keep the sign, force a quiet NaN
    return out


def encode(x: np.ndarray, dtype: str) -> np.ndarray:
    """float32 array -> array whose bytes are the tensor data in ``dtype``."""
    if dtype == "BF16":
        return f32_to_bf16_bits(x)
    if dtype in _NP:
        return np.ascontiguousarray(x.astype(_NP[dtype]))
    raise MergeError(f"cannot encode dtype {dtype}")


def _drop_cache(fd: int) -> None:
    """Hand the file's pages back (Linux page cache counts against the VM; no sudo needed for this)."""
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def sha256_file(path: Path, *, drop_cache: bool = True) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            if drop_cache:
                _drop_cache(f.fileno())
    return h.hexdigest()


# ------------------------------------------------------------------ the adapter

@dataclass
class Lora:
    a: np.ndarray                    # r x in, float32
    b: np.ndarray                    # out x r, float32
    keys: tuple[str, str] = ("", "")


@dataclass
class Adapter:
    config: dict
    scale: float
    r: int
    loras: dict[str, Lora]           # module path (no ".weight") -> pair
    sha256_safetensors: str = ""
    sha256_config: str = ""


_PEFT_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")


def parse_peft_key(key: str) -> tuple[str, str]:
    """('module path', 'A' or 'B') of a PEFT adapter tensor name."""
    m = _PEFT_KEY.match(key)
    if m is None:
        raise MergeError(f"adapter tensor {key!r} is not a LoRA A/B weight (DoRA magnitudes, biases and modules_to_save "
                         "are not supported)")
    return m["module"], m["ab"]


def check_config(cfg: dict) -> float:
    """Refuse every adapter variant this module does not implement; return scale = alpha / r."""
    if cfg.get("peft_type") != "LORA":
        raise MergeError(f"peft_type {cfg.get('peft_type')!r} is not LORA")
    for name in ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out", "modules_to_save", "target_parameters",
                 "trainable_token_indices", "layer_replication"):
        if cfg.get(name) not in (None, False, [], {}):
            raise MergeError(f"adapter_config {name}={cfg.get(name)!r} is not supported")
    if cfg.get("bias", "none") != "none":
        raise MergeError(f"adapter_config bias={cfg['bias']!r} is not supported")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise MergeError("rank_pattern / alpha_pattern are not supported")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if not (isinstance(r, int) and r > 0):
        raise MergeError(f"bad rank {r!r}")
    return float(alpha) / float(r)


def load_adapter(source: Path) -> Adapter:
    """Read an adapter from a directory (adapter_model.safetensors + adapter_config.json) or a .tar holding both."""
    source = Path(source)
    tmp = None
    try:
        if source.is_file():
            tmp = tempfile.TemporaryDirectory(prefix="adapter-")
            with tarfile.open(source) as tf:
                names = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
                for want in ADAPTER_FILES:
                    if want not in names:
                        raise MergeError(f"{source} has no {want}")
                    with tf.extractfile(names[want]) as src, open(Path(tmp.name) / want, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            d = Path(tmp.name)
        else:
            d = source
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        scale = check_config(cfg)
        st = d / "adapter_model.safetensors"
        h = read_header(st)
        loras: dict[str, dict[str, np.ndarray]] = {}
        seen: set[str] = set()
        with open(st, "rb") as f:
            for key, meta in h.ordered():
                if key in seen:
                    raise MergeError(f"duplicate adapter tensor {key}")
                seen.add(key)
                module, ab = parse_peft_key(key)
                if ab in loras.setdefault(module, {}):
                    raise MergeError(f"module {module}: lora_{ab} appears twice")
                arr = decode(read_tensor_raw(f, h, key), meta["dtype"], meta["shape"])
                if arr.ndim != 2:
                    raise MergeError(f"{key}: expected a 2-D matrix, got shape {meta['shape']}")
                loras[module][ab] = arr
        out: dict[str, Lora] = {}
        for module, ab in sorted(loras.items()):
            if set(ab) != {"A", "B"}:
                raise MergeError(f"module {module}: has only lora_{'/'.join(ab)}; an A/B pair is needed")
            a, b = ab["A"], ab["B"]
            if a.shape[0] != cfg["r"] or b.shape[1] != cfg["r"]:
                raise MergeError(f"module {module}: A {a.shape} / B {b.shape} do not have rank r={cfg['r']}")
            out[module] = Lora(a=a, b=b)
        ad = Adapter(config=cfg, scale=scale, r=cfg["r"], loras=out)
        ad.sha256_safetensors = sha256_file(st)
        ad.sha256_config = hashlib.sha256((d / "adapter_config.json").read_bytes()).hexdigest()
        return ad
    finally:
        if tmp is not None:
            tmp.cleanup()


# ------------------------------------------------------------------ the plan

@dataclass
class Plan:
    scale: float
    targets: dict[str, str]                       # base weight key -> module path
    by_shard: dict[str, list[str]] = field(default_factory=dict)   # shard file -> base weight keys
    notes: list[str] = field(default_factory=list)


def resolve_key(module: str, base_keys: set[str]) -> str:
    exact = module + ".weight"
    if exact in base_keys:
        return exact
    if module.startswith("model.layers."):
        alt = "model.language_model." + module[len("model."):] + ".weight"
        if alt in base_keys:
            return alt
    raise MergeError(f"adapter module {module!r}: base checkpoint has no weight {exact!r}")


def config_target_modules(cfg: dict, weights: dict[str, list[int]]) -> set[str]:
    """The 2-D weights the adapter config's target_modules would have wrapped (PEFT: regex fullmatch on the module name
    for a string, suffix match for a list)."""
    tm = cfg.get("target_modules")
    mods = [k[:-len(".weight")] for k, shape in weights.items() if k.endswith(".weight") and len(shape) == 2]
    if isinstance(tm, str):
        pat = re.compile(tm)
        return {m for m in mods if pat.fullmatch(m)}
    names = set(tm or [])
    return {m for m in mods if m in names or any(m.endswith("." + t) for t in names)}


def build_plan(adapter: Adapter, weight_map: dict[str, str], shapes: dict[str, list[int]]) -> Plan:
    """Map every adapter pair onto its base weight; assert 1:1, existence, shapes, and agreement with target_modules."""
    base_keys = set(weight_map)
    targets: dict[str, str] = {}
    for module, lo in adapter.loras.items():
        key = resolve_key(module, base_keys)
        if key in targets:
            raise MergeError(f"two adapter modules ({targets[key]}, {module}) map to {key}")
        shape = shapes[key]
        want = [lo.b.shape[0], lo.a.shape[1]]
        if shape != want:
            raise MergeError(f"{key}: base shape {shape} but B@A is {want} (B {list(lo.b.shape)}, A {list(lo.a.shape)})")
        targets[key] = module
    notes = []
    wanted = config_target_modules(adapter.config, shapes)
    have = {k[:-len(".weight")] for k in targets}
    if wanted != have:
        raise MergeError(f"adapter_config target_modules selects {len(wanted)} base weights, the adapter holds {len(have)}; "
                         f"only in config: {sorted(wanted - have)[:5]}, only in adapter: {sorted(have - wanted)[:5]}")
    notes.append(f"target_modules regex selects exactly the {len(have)} modules the adapter holds")
    by_shard: dict[str, list[str]] = {}
    for key in sorted(targets):
        by_shard.setdefault(weight_map[key], []).append(key)
    return Plan(scale=adapter.scale, targets=targets, by_shard=by_shard, notes=notes)


# ------------------------------------------------------------------ the merge

def merge_array(w: np.ndarray, dtype: str, lo: Lora, scale: float, block_rows: int = 2048) -> np.ndarray:
    """``w`` is the weight as float32 (out x in); returns the bytes-ready array of ``dtype``: round(W + scale * (B @ A))."""
    out_f = w.shape[0]
    res = np.empty(w.shape, dtype=np.uint16 if dtype == "BF16" else _NP[dtype])
    for r0 in range(0, out_f, block_rows):
        r1 = min(out_f, r0 + block_rows)
        delta = lo.b[r0:r1] @ lo.a
        if scale != 1.0:
            delta *= np.float32(scale)
        delta += w[r0:r1]
        res[r0:r1] = encode(delta, dtype)
    return res


def merge_shard(src: Path, dst: Path, keys: list[str], plan: Plan, adapter: Adapter, *, drop_cache: bool = True,
                log=print) -> dict:
    """Write dst = src with the data of ``keys`` replaced. Returns {"sha256", "bytes", "merged", "seconds"}."""
    t0 = time.time()
    h = read_header(src)
    missing = [k for k in keys if k not in h.tensors]
    if missing:
        raise MergeError(f"{src.name}: index lists {missing[:3]} here but the shard does not hold them")
    todo = set(keys)
    sha = hashlib.sha256()
    part = dst.with_name(dst.name + ".partial")
    written_since_sync = 0
    with open(src, "rb") as fi, open(part, "wb") as fo:
        fo.write(h.raw)
        sha.update(h.raw)
        for name, meta in h.ordered():
            b, e = meta["data_offsets"]
            fi.seek(h.data_start + b)
            if name in todo:
                if meta["dtype"] not in MERGEABLE:
                    raise MergeError(f"{name}: dtype {meta['dtype']} cannot take a LoRA delta")
                lo = adapter.loras[plan.targets[name]]
                w = decode(read_exact(fi, e - b), meta["dtype"], meta["shape"])
                if drop_cache:
                    _drop_cache(fi.fileno())
                data = merge_array(w, meta["dtype"], lo, plan.scale).tobytes()
                del w
                if len(data) != e - b:
                    raise MergeError(f"{name}: merged data is {len(data)} bytes, expected {e - b}")
                fo.write(data)
                sha.update(data)
                written_since_sync += len(data)
                todo.discard(name)
            else:
                left = e - b
                while left:
                    chunk = fi.read(min(CHUNK, left))
                    if not chunk:
                        raise MergeError(f"{src.name}: unexpected end of file in {name}")
                    fo.write(chunk)
                    sha.update(chunk)
                    left -= len(chunk)
                    written_since_sync += len(chunk)
                    if drop_cache:
                        _drop_cache(fi.fileno())
            if drop_cache and written_since_sync >= (256 << 20):
                fo.flush()
                os.fsync(fo.fileno())
                _drop_cache(fo.fileno())
                written_since_sync = 0
        fo.flush()
        os.fsync(fo.fileno())
        if drop_cache:
            _drop_cache(fo.fileno())
    if todo:
        raise MergeError(f"{src.name}: tensors not merged: {sorted(todo)[:3]}")
    os.replace(part, dst)
    out = {"sha256": sha.hexdigest(), "bytes": dst.stat().st_size, "merged": len(keys), "seconds": round(time.time() - t0, 1)}
    log(f"  {dst.name}: {len(keys)} tensors merged, {out['bytes'] / 2**30:.2f} GiB, sha256 {out['sha256'][:16]}, {out['seconds']} s")
    return out


def copy_aux(base: Path, out: Path) -> list[str]:
    """Copy config, tokenizer, template, processor files and the index (real files, not links)."""
    copied = []
    for p in sorted(base.iterdir()):
        if p.is_dir() or p.suffix == ".safetensors" or p.name in AUX_SKIP:
            continue
        shutil.copyfile(p, out / p.name)    # follows the symlinks of a Hugging Face cache snapshot
        copied.append(p.name)
    return copied


def load_base(base: Path) -> tuple[dict[str, str], dict[str, list[int]], dict[str, Header]]:
    idx = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    headers = {sh: read_header(base / sh) for sh in sorted(set(wm.values()))}
    shapes = {}
    for sh, hd in headers.items():
        for k, t in hd.tensors.items():
            if wm.get(k) == sh:
                shapes[k] = t["shape"]
    if set(shapes) != set(wm):
        raise MergeError("the index and the shard headers disagree on tensor names")
    return wm, shapes, headers


def merge(base: Path, adapter_src: Path, out: Path, *, drop_cache: bool = True, log=print) -> dict:
    base, out = Path(base), Path(out)
    t0 = time.time()
    adapter = load_adapter(adapter_src)
    wm, shapes, _ = load_base(base)
    plan = build_plan(adapter, wm, shapes)
    log(f"mapping rule: {MAPPING_RULE}")
    log(f"adapter: r {adapter.r}, alpha {adapter.config['lora_alpha']}, scale {plan.scale}, {len(adapter.loras)} A/B pairs "
        f"({2 * len(adapter.loras)} tensors); sha256 {adapter.sha256_safetensors[:16]}")
    for n in plan.notes:
        log(f"  {n}")
    out.mkdir(parents=True, exist_ok=True)
    shards = {}
    for sh in sorted(set(wm.values())):
        keys = plan.by_shard.get(sh, [])
        shards[sh] = merge_shard(base / sh, out / sh, keys, plan, adapter, drop_cache=drop_cache, log=log)
    copied = copy_aux(base, out)
    used = sum(s["merged"] for s in shards.values())
    if used != len(adapter.loras):
        raise MergeError(f"{used} base weights merged but the adapter holds {len(adapter.loras)} pairs")
    manifest = {
        "tool": "train/merge_lora.py", "mapping_rule": MAPPING_RULE, "base": str(base),
        "base_index_sha256": hashlib.sha256((base / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "adapter": {"source": str(adapter_src), "sha256_safetensors": adapter.sha256_safetensors,
                    "sha256_config": adapter.sha256_config, "r": adapter.r, "lora_alpha": adapter.config["lora_alpha"],
                    "scale": plan.scale, "pairs": len(adapter.loras), "tensors": 2 * len(adapter.loras)},
        "rounding": "float32 add of scale*(B@A), cast to bf16 round-to-nearest-even",
        "tensors_merged": used, "shards": shards, "aux_files": copied, "seconds": round(time.time() - t0, 1)}
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log(f"merged {used} weights into {len(shards)} shards in {manifest['seconds']} s")
    return manifest


# ------------------------------------------------------------------ verification

def _half_ulp(x: np.ndarray, dtype: str) -> np.ndarray:
    """Half the spacing of ``dtype`` numbers at |x| (bf16: 7 explicit mantissa bits; fp16: 10; fp32: 23)."""
    bits = {"BF16": 7, "F16": 10, "F32": 23}[dtype]
    _, e = np.frexp(np.abs(x).astype(np.float64))
    return np.ldexp(1.0, e - 1 - bits - 1)


# The merge evaluates B @ A in float32 (a sum of r products): its error is below r * 2**-24 * sum|b_i a_i|, i.e. 2**-20 of
# that sum for r = 16. Where W and the delta nearly cancel, the result is far smaller than that error, so the check allows
# it on top of the half-ulp of the output (measured: ratio 28 against half an ulp, 1.00001 against the allowance, at a
# weight of -1.09e-4 plus a delta of +1.09e-4 whose sum is 2e-11).
FP32_DOT_ALLOWANCE = 2.0 ** -20


def verify_tensor(base_w: np.ndarray, out_w: np.ndarray, lo: Lora, scale: float, dtype: str, block_rows: int = 1024) -> dict:
    """Compare a merged tensor with W + scale * (B @ A) computed in float64 from the same inputs.

    ``max_err_over_half_ulp``: worst |merged - ideal| in half-ulps of the output dtype at the merged value.
    ``max_err_over_tol``: the same error over (half ulp + FP32_DOT_ALLOWANCE * scale * sum|b a|); the merge is correct
    iff this stays <= 1."""
    worst = 0.0
    worst_tol = 0.0
    over_half_ulp = 0
    changed = 0
    num = den = dot = dd = 0.0
    abs_delta = abs_w = 0.0
    a64, b64 = lo.a.astype(np.float64), lo.b.astype(np.float64)
    a_abs, b_abs = np.abs(a64), np.abs(b64)
    for r0 in range(0, base_w.shape[0], block_rows):
        r1 = min(base_w.shape[0], r0 + block_rows)
        w, m = base_w[r0:r1].astype(np.float64), out_w[r0:r1].astype(np.float64)
        delta = (b64[r0:r1] @ a64) * scale
        err = np.abs(m - (w + delta))
        hu = _half_ulp(m, dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(err == 0, 0.0, err / hu)
            ratio_tol = np.where(err == 0, 0.0, err / (hu + FP32_DOT_ALLOWANCE * scale * (b_abs[r0:r1] @ a_abs)))
        worst = max(worst, float(np.nanmax(ratio)) if ratio.size else 0.0)
        worst_tol = max(worst_tol, float(np.nanmax(ratio_tol)) if ratio_tol.size else 0.0)
        over_half_ulp += int((ratio > 1.0001).sum())
        real = m - w
        changed += int((real != 0).sum())
        num += float(((real - delta) ** 2).sum())
        den += float((delta ** 2).sum())
        dot += float((real * delta).sum())
        dd += float((real ** 2).sum())
        abs_delta += float(np.abs(delta).sum())
        abs_w += float(np.abs(w).sum())
    n = base_w.size
    return {"max_err_over_half_ulp": worst, "max_err_over_tol": worst_tol, "elements_over_half_ulp": over_half_ulp,
            "elements": n, "changed_fraction": changed / n,
            "rel_fro_err_of_realized_delta": math.sqrt(num / den) if den else 0.0,
            "cosine_realized_vs_ideal": dot / math.sqrt(dd * den) if dd and den else 1.0,
            "mean_abs_delta_over_mean_abs_w": abs_delta / abs_w if abs_w else 0.0}


def module_type(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[-3:-1]) if "layers" in parts else key


def pick_sample(keys: list[str]) -> list[str]:
    """First, middle and last layer of every module type (deterministic)."""
    by_type: dict[str, list[tuple[int, str]]] = {}
    for k in keys:
        m = re.search(r"layers\.(\d+)\.", k)
        by_type.setdefault(module_type(k), []).append((int(m[1]) if m else -1, k))
    chosen = []
    for items in by_type.values():
        items.sort()
        picks = {items[0], items[len(items) // 2], items[-1]}
        chosen += [k for _, k in sorted(picks)]
    return sorted(chosen)


def verify(base: Path, adapter_src: Path, out: Path, *, which: str = "sample", drop_cache: bool = True, log=print) -> dict:
    """Re-read base and merged shards and check: identical headers; every untouched tensor byte-identical; merged tensors equal
    the float64 reference W + scale*(B@A) within half a bf16 ulp plus the float32 allowance (``which`` = sample | all)."""
    base, out = Path(base), Path(out)
    adapter = load_adapter(adapter_src)
    wm, shapes, headers = load_base(base)
    plan = build_plan(adapter, wm, shapes)
    result = {"which": which, "header_identical": True, "untouched_tensors": 0, "untouched_identical": True,
              "merged_checked": 0, "max_err_over_half_ulp": 0.0, "max_err_over_tol": 0.0, "elements_over_half_ulp": 0,
              "elements_checked": 0, "by_type": {}}
    all_keys = sorted(plan.targets)
    check = set(all_keys if which == "all" else pick_sample(all_keys))
    stats_by_type: dict[str, list[dict]] = {}
    for sh, bh in headers.items():
        oh = read_header(out / sh)
        if oh.raw != bh.raw:
            result["header_identical"] = False
        with open(base / sh, "rb") as fb, open(out / sh, "rb") as fo:
            for name, meta in bh.ordered():
                b, e = meta["data_offsets"]
                if name in plan.targets:
                    if name not in check:
                        continue
                    w = decode(read_tensor_raw(fb, bh, name), meta["dtype"], meta["shape"])
                    m = decode(read_tensor_raw(fo, oh, name), meta["dtype"], meta["shape"])
                    st = verify_tensor(w, m, adapter.loras[plan.targets[name]], plan.scale, meta["dtype"])
                    st["key"] = name
                    stats_by_type.setdefault(module_type(name), []).append(st)
                    result["merged_checked"] += 1
                    result["max_err_over_half_ulp"] = max(result["max_err_over_half_ulp"], st["max_err_over_half_ulp"])
                    result["max_err_over_tol"] = max(result["max_err_over_tol"], st["max_err_over_tol"])
                    result["elements_over_half_ulp"] += st["elements_over_half_ulp"]
                    result["elements_checked"] += st["elements"]
                    if drop_cache:
                        _drop_cache(fb.fileno())
                        _drop_cache(fo.fileno())
                else:
                    fb.seek(bh.data_start + b)
                    fo.seek(oh.data_start + b)
                    left = e - b
                    while left:
                        n = min(CHUNK, left)
                        if fb.read(n) != fo.read(n):
                            result["untouched_identical"] = False
                            log(f"  UNTOUCHED TENSOR DIFFERS: {name}")
                            break
                        left -= n
                        if drop_cache:
                            _drop_cache(fb.fileno())
                            _drop_cache(fo.fileno())
                    result["untouched_tensors"] += 1
    for t, lst in sorted(stats_by_type.items()):
        result["by_type"][t] = {
            "n": len(lst), "max_err_over_half_ulp": max(s["max_err_over_half_ulp"] for s in lst),
            "max_err_over_tol": max(s["max_err_over_tol"] for s in lst),
            "changed_fraction": float(np.mean([s["changed_fraction"] for s in lst])),
            "rel_fro_err_of_realized_delta": float(np.mean([s["rel_fro_err_of_realized_delta"] for s in lst])),
            "cosine_realized_vs_ideal": float(np.mean([s["cosine_realized_vs_ideal"] for s in lst])),
            "mean_abs_delta_over_mean_abs_w": float(np.mean([s["mean_abs_delta_over_mean_abs_w"] for s in lst]))}
    result["ok"] = (result["header_identical"] and result["untouched_identical"]
                    and result["max_err_over_tol"] <= 1.0001 and result["merged_checked"] > 0)
    return result


# ------------------------------------------------------------------ command line

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="folder with the base model's safetensors shards, index and config")
    ap.add_argument("adapter", type=Path, help="adapter folder, or the .tar that holds adapter_model.safetensors + adapter_config.json")
    ap.add_argument("out", type=Path, help="output folder (created)")
    ap.add_argument("--verify", choices=("sample", "all", "none"), default="sample")
    ap.add_argument("--verify-only", action="store_true", help="do not merge; check an existing output folder")
    ap.add_argument("--keep-cache", action="store_true", help="do not hand file pages back to the OS while streaming")
    a = ap.parse_args(argv)
    drop = not a.keep_cache
    manifest = None
    if not a.verify_only:
        manifest = merge(a.base, a.adapter, a.out, drop_cache=drop)
    rc = 0
    if a.verify != "none":
        t0 = time.time()
        v = verify(a.base, a.adapter, a.out, which=a.verify, drop_cache=drop)
        v["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(v, indent=1, sort_keys=True))
        mp = a.out / "merge_manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            m["verification"] = v
            mp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        rc = 0 if v["ok"] else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
