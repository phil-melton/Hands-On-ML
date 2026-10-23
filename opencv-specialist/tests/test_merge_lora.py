"""Tests for train/merge_lora.py on a tiny fake multimodal checkpoint (no 9B, numpy only; torch is used as an oracle when installed)."""
from __future__ import annotations

import hashlib
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

from train import merge_lora as ml

R = 4
ALPHA = 8                    # scale = alpha / r = 2.0, so a missing scale shows up
CFG_REGEX = r".*language.*\.(?:gate_proj|up_proj|down_proj|in_proj_a)"
L = "model.language_model.layers"

# base weight -> (shard, dtype, shape); only the first group is targeted by the adapter
TARGETS = {
    f"{L}.0.mlp.gate_proj.weight": (1, "BF16", (12, 8)),
    f"{L}.0.mlp.up_proj.weight": (1, "BF16", (12, 8)),
    f"{L}.0.mlp.down_proj.weight": (1, "BF16", (8, 12)),
    f"{L}.0.linear_attn.in_proj_a.weight": (1, "BF16", (4, 8)),
    f"{L}.1.mlp.gate_proj.weight": (2, "BF16", (12, 8)),
    f"{L}.1.mlp.up_proj.weight": (2, "BF16", (12, 8)),
    f"{L}.1.mlp.down_proj.weight": (2, "BF16", (8, 12)),
}
OTHERS = {
    "model.language_model.embed_tokens.weight": (1, "BF16", (16, 8)),
    f"{L}.0.linear_attn.conv1d.weight": (1, "BF16", (8, 1, 4)),         # 3-D: never a LoRA target
    f"{L}.0.linear_attn.A_log": (1, "F32", (4,)),
    f"{L}.0.input_layernorm.weight": (1, "BF16", (8,)),
    f"{L}.0.mlp.gate_proj.bias": (1, "BF16", (12,)),
    "model.visual.blocks.0.mlp.linear_fc1.weight": (2, "BF16", (6, 8)),  # 2-D but not in the language model
    "mtp.layers.0.mlp.gate_proj.weight": (2, "BF16", (12, 8)),
    f"{L}.1.self_attn.o_proj.weight": (2, "BF16", (8, 8)),               # 2-D language-model weight the config does not name
    "model.language_model.norm.weight": (2, "BF16", (8,)),
    "lm_head.weight": (2, "BF16", (16, 8)),
}


# ------------------------------------------------------------------ helpers

def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Independent of the module under test: truncate-with-round-to-nearest-even via python ints."""
    out = np.empty(x.shape, dtype=np.uint16)
    flat = np.asarray(x, dtype=np.float32).ravel()
    res = out.reshape(-1)
    for i, v in enumerate(flat):
        b = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        res[i] = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return out


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def write_st(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    js = json.dumps(header, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def make_array(rng, dtype, shape):
    if dtype == "BF16":
        return bf16_bits(rng.normal(size=shape).astype(np.float32))
    return rng.normal(size=shape).astype(np.float32)


def make_fixture(tmp_path: Path, *, seed=0, zero_b=False, adapter_prefix="base_model.model.", cfg_overrides=None,
                 drop_base_key=None, skip_b_for=None, base_key_alias=False):
    rng = np.random.default_rng(seed)
    base = tmp_path / "base"
    base.mkdir()
    shards = {1: {}, 2: {}}
    values = {}
    for name, (sh, dtype, shape) in {**TARGETS, **OTHERS}.items():
        if name == drop_base_key:
            continue
        arr = make_array(rng, dtype, shape)
        shards[sh][name] = (dtype, arr)
        values[name] = (dtype, arr)
    wm = {}
    for sh, tensors in shards.items():
        fn = f"model.safetensors-0000{sh}-of-00002.safetensors"
        write_st(base / fn, tensors)
        wm.update({k: fn for k in tensors})
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": wm}))
    (base / "config.json").write_text('{"model_type": "fake"}')
    (base / "tokenizer.json").write_text("{}")
    (base / "README.md").write_text("not copied")

    ad = tmp_path / "adapter"
    ad.mkdir()
    lora = {}
    pairs = {}
    for name, (sh, dtype, shape) in TARGETS.items():
        module = name[:-len(".weight")]
        a = rng.normal(scale=0.3, size=(R, shape[1])).astype(np.float32)
        b = (np.zeros((shape[0], R)) if zero_b else rng.normal(scale=0.3, size=(shape[0], R))).astype(np.float32)
        pairs[name] = (a, b)
        pk = module if not base_key_alias else module.replace("model.language_model.layers", "model.layers")
        lora[f"{adapter_prefix}{pk}.lora_A.weight"] = ("F32", a)
        if module != skip_b_for:
            lora[f"{adapter_prefix}{pk}.lora_B.weight"] = ("F32", b)
    write_st(ad / "adapter_model.safetensors", lora)
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": ALPHA, "use_dora": False, "use_rslora": False, "lora_bias": False,
           "bias": "none", "fan_in_fan_out": False, "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {},
           "target_parameters": None, "target_modules": CFG_REGEX}
    cfg.update(cfg_overrides or {})
    (ad / "adapter_config.json").write_text(json.dumps(cfg))
    return {"base": base, "adapter": ad, "values": values, "pairs": pairs, "scale": ALPHA / R, "weight_map": wm}


def read_all(path: Path) -> dict[str, np.ndarray]:
    h = ml.read_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, meta in h.ordered():
            out[name] = ml.read_tensor_raw(f, h, name)
    return out


def merged_u16(fx, out, name):
    sh = fx["weight_map"][name]
    return read_all(out / sh)[name].view(np.uint16).reshape(TARGETS[name][2])


@pytest.fixture
def merged(tmp_path):
    fx = make_fixture(tmp_path)
    out = tmp_path / "out"
    manifest = ml.merge(fx["base"], fx["adapter"], out, log=lambda *_: None)
    return fx, out, manifest


# ------------------------------------------------------------------ key mapping

def test_parse_peft_key():
    assert ml.parse_peft_key("base_model.model.model.language_model.layers.3.mlp.up_proj.lora_A.weight") == (
        "model.language_model.layers.3.mlp.up_proj", "A")
    assert ml.parse_peft_key("base_model.model.model.language_model.layers.3.linear_attn.in_proj_a.lora_B.weight") == (
        "model.language_model.layers.3.linear_attn.in_proj_a", "B")
    assert ml.parse_peft_key("x.q_proj.lora_A.default.weight") == ("x.q_proj", "A")        # adapter name kept by some savers
    for bad in ("base_model.model.x.lora_magnitude_vector", "base_model.model.x.weight", "base_model.model.x.lora_A.bias"):
        with pytest.raises(ml.MergeError):
            ml.parse_peft_key(bad)


def test_mapping_is_one_to_one_and_complete(tmp_path):
    fx = make_fixture(tmp_path)
    ad = ml.load_adapter(fx["adapter"])
    assert len(ad.loras) == len(TARGETS)
    wm, shapes, _ = ml.load_base(fx["base"])
    plan = ml.build_plan(ad, wm, shapes)
    assert set(plan.targets) == set(TARGETS)                     # every adapter pair has its base weight, none twice
    assert sorted(k for ks in plan.by_shard.values() for k in ks) == sorted(TARGETS)
    assert plan.scale == 2.0


def test_text_only_adapter_prefix_falls_back_to_language_model(tmp_path):
    fx = make_fixture(tmp_path, base_key_alias=True)             # adapter says model.layers.N, base model.language_model.layers.N
    m = ml.merge(fx["base"], fx["adapter"], tmp_path / "out", log=lambda *_: None)
    assert m["tensors_merged"] == len(TARGETS)


def test_adapter_without_base_weight_is_refused(tmp_path):
    fx = make_fixture(tmp_path, drop_base_key=f"{L}.1.mlp.up_proj.weight")
    with pytest.raises(ml.MergeError, match="no weight"):
        ml.merge(fx["base"], fx["adapter"], tmp_path / "out", log=lambda *_: None)


def test_lonely_lora_a_is_refused(tmp_path):
    fx = make_fixture(tmp_path, skip_b_for=f"{L}.0.mlp.up_proj")
    with pytest.raises(ml.MergeError, match="needed"):
        ml.load_adapter(fx["adapter"])


def test_shape_mismatch_is_refused(tmp_path):
    fx = make_fixture(tmp_path)
    ad = ml.load_adapter(fx["adapter"])
    key = f"{L}.0.mlp.gate_proj"
    ad.loras[key].b = ad.loras[key].b[:-1]                        # 11 x r instead of 12 x r
    wm, shapes, _ = ml.load_base(fx["base"])
    with pytest.raises(ml.MergeError, match="base shape"):
        ml.build_plan(ad, wm, shapes)


def test_config_target_modules_must_equal_adapter_modules(tmp_path):
    # the config names o_proj as well, the adapter has no pair for it: a half-saved adapter must not pass
    fx = make_fixture(tmp_path, cfg_overrides={"target_modules": CFG_REGEX.replace("in_proj_a", "in_proj_a|o_proj")})
    with pytest.raises(ml.MergeError, match="target_modules"):
        ml.merge(fx["base"], fx["adapter"], tmp_path / "out", log=lambda *_: None)


@pytest.mark.parametrize("override", [{"use_dora": True}, {"use_rslora": True}, {"lora_bias": True}, {"bias": "all"},
                                      {"modules_to_save": ["lm_head"]}, {"rank_pattern": {"x": 2}}, {"peft_type": "IA3"}])
def test_unsupported_adapter_variants_are_refused(tmp_path, override):
    fx = make_fixture(tmp_path, cfg_overrides=override)
    with pytest.raises(ml.MergeError):
        ml.load_adapter(fx["adapter"])


# ------------------------------------------------------------------ rounding

def test_f32_to_bf16_round_to_nearest_even():
    bits = lambda u32: np.array([u32], dtype=np.uint32).view(np.float32)
    # exactly half way between 0x3F80 and 0x3F81: ties go to the even mantissa
    assert ml.f32_to_bf16_bits(bits(0x3F808000))[0] == 0x3F80
    assert ml.f32_to_bf16_bits(bits(0x3F818000))[0] == 0x3F82
    # just above / below the tie
    assert ml.f32_to_bf16_bits(bits(0x3F808001))[0] == 0x3F81
    assert ml.f32_to_bf16_bits(bits(0x3F807FFF))[0] == 0x3F80
    assert ml.f32_to_bf16_bits(bits(0xBF808001))[0] == 0xBF81               # sign kept
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    got = ml.f32_to_bf16_bits(x)
    assert list(got[:4]) == [0x0000, 0x8000, 0x7F80, 0xFF80]
    assert np.isnan(bf16_to_f32(got[4:])).all()
    assert ml.f32_to_bf16_bits(bits(0x7F7FFFFF))[0] == 0x7F80               # float32 max rounds up to inf, as torch does


def test_f32_to_bf16_matches_reference_and_torch():
    x = np.random.default_rng(1).normal(scale=3.0, size=5000).astype(np.float32)
    assert np.array_equal(ml.f32_to_bf16_bits(x), bf16_bits(x))
    assert np.array_equal(ml.f32_to_bf16_bits(bf16_to_f32(ml.f32_to_bf16_bits(x))), ml.f32_to_bf16_bits(x))   # idempotent
    torch = pytest.importorskip("torch")
    want = torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(ml.f32_to_bf16_bits(x), want)


def test_decode_bf16_is_exact():
    u = np.arange(0, 65536, 7, dtype=np.uint16)
    f = ml.decode(u.view(np.uint8), "BF16", [u.size])
    ok = ~np.isnan(f)
    assert np.array_equal(ml.f32_to_bf16_bits(f[ok]), u[ok])


# ------------------------------------------------------------------ the merge

def test_merged_tensors_equal_w_plus_scale_ba(merged):
    fx, out, _ = merged
    for name, (a, b) in fx["pairs"].items():
        base = bf16_to_f32(fx["values"][name][1]).astype(np.float64)
        ideal = base + fx["scale"] * (b.astype(np.float64) @ a.astype(np.float64))
        got = bf16_to_f32(merged_u16(fx, out, name)).astype(np.float64)
        _, e = np.frexp(np.abs(got))
        half_ulp = np.ldexp(1.0, e - 9)                                   # bf16: 7 explicit mantissa bits
        assert (np.abs(got - ideal) <= half_ulp * 1.0001).all(), name
        assert not np.array_equal(got, base), f"{name}: the delta had no effect"
        # and the rounding is the one torch / numpy float32 give: float32 sum, then round-to-nearest-even
        ref = bf16_bits(bf16_to_f32(fx["values"][name][1]) + np.float32(fx["scale"]) * (b @ a))
        assert (merged_u16(fx, out, name) != ref).mean() < 0.02, name        # float32 matmul order may flip a rare tie


def test_scale_is_alpha_over_r(tmp_path):
    fx = make_fixture(tmp_path)
    ad = ml.load_adapter(fx["adapter"])
    assert ad.scale == ALPHA / R == 2.0
    # with the same A/B and alpha = r the delta is half as large
    cfg = json.loads((fx["adapter"] / "adapter_config.json").read_text())
    cfg["lora_alpha"] = R
    (fx["adapter"] / "adapter_config.json").write_text(json.dumps(cfg))
    assert ml.load_adapter(fx["adapter"]).scale == 1.0


def test_untouched_tensors_and_headers_are_byte_identical(merged):
    fx, out, _ = merged
    for fn in sorted(set(fx["weight_map"].values())):
        hb, ho = ml.read_header(fx["base"] / fn), ml.read_header(out / fn)
        assert hb.raw == ho.raw                                         # same names, dtypes, shapes, offsets
        b, o = read_all(fx["base"] / fn), read_all(out / fn)
        for name in b:
            assert (b[name].tobytes() == o[name].tobytes()) == (name not in TARGETS), name


def test_zero_b_gives_identical_files(tmp_path):
    fx = make_fixture(tmp_path, zero_b=True)
    out = tmp_path / "out"
    ml.merge(fx["base"], fx["adapter"], out, log=lambda *_: None)
    for fn in set(fx["weight_map"].values()):
        assert (out / fn).read_bytes() == (fx["base"] / fn).read_bytes()


def test_manifest_hashes_and_copied_files(merged):
    fx, out, manifest = merged
    assert manifest["tensors_merged"] == len(TARGETS)
    assert manifest["adapter"]["scale"] == 2.0 and manifest["adapter"]["pairs"] == len(TARGETS)
    for fn, s in manifest["shards"].items():
        assert s["sha256"] == hashlib.sha256((out / fn).read_bytes()).hexdigest()
        assert s["bytes"] == (out / fn).stat().st_size
    assert sum(s["merged"] for s in manifest["shards"].values()) == len(TARGETS)
    assert (out / "model.safetensors.index.json").read_bytes() == (fx["base"] / "model.safetensors.index.json").read_bytes()
    assert (out / "config.json").exists() and (out / "tokenizer.json").exists()
    assert not (out / "README.md").exists()
    assert not list(out.glob("*.partial"))
    assert json.loads((out / "merge_manifest.json").read_text())["tensors_merged"] == len(TARGETS)


def test_adapter_as_tar(tmp_path):
    fx = make_fixture(tmp_path)
    tar = tmp_path / "ad.tar"
    with tarfile.open(tar, "w") as tf:
        for fn in ml.ADAPTER_FILES:
            tf.add(fx["adapter"] / fn, arcname=fn)
    a1, a2 = ml.load_adapter(fx["adapter"]), ml.load_adapter(tar)
    assert a1.sha256_safetensors == a2.sha256_safetensors
    assert all(np.array_equal(a1.loras[k].a, a2.loras[k].a) and np.array_equal(a1.loras[k].b, a2.loras[k].b) for k in a1.loras)


def test_verify_accepts_a_good_merge_and_rejects_damage(merged):
    fx, out, _ = merged
    v = ml.verify(fx["base"], fx["adapter"], out, which="all", log=lambda *_: None)
    assert v["ok"] and v["merged_checked"] == len(TARGETS) and v["header_identical"] and v["untouched_identical"]
    assert v["max_err_over_tol"] <= 1.0001
    assert 0.0 < v["by_type"]["mlp.gate_proj"]["changed_fraction"] <= 1.0
    s = ml.verify(fx["base"], fx["adapter"], out, which="sample", log=lambda *_: None)
    assert s["ok"] and 0 < s["merged_checked"] <= len(TARGETS)

    name = f"{L}.0.mlp.up_proj.weight"                           # corrupt one merged weight by a large amount
    fn = out / fx["weight_map"][name]
    h = ml.read_header(fn)
    b, _ = h.tensors[name]["data_offsets"]
    raw = bytearray(fn.read_bytes())
    raw[h.data_start + b + 1] ^= 0x40
    fn.write_bytes(bytes(raw))
    bad = ml.verify(fx["base"], fx["adapter"], out, which="all", log=lambda *_: None)
    assert not bad["ok"] and bad["max_err_over_tol"] > 1.0001

    name = "lm_head.weight"                                      # corrupt an untouched tensor
    fn = out / fx["weight_map"][name]
    h = ml.read_header(fn)
    b, _ = h.tensors[name]["data_offsets"]
    raw = bytearray(fn.read_bytes())
    raw[h.data_start + b] ^= 0x01
    fn.write_bytes(bytes(raw))
    bad2 = ml.verify(fx["base"], fx["adapter"], out, which="all", log=lambda *_: None)
    assert not bad2["untouched_identical"] and not bad2["ok"]


def test_read_header_rejects_truncated_and_gapped_files(tmp_path):
    p = tmp_path / "x.safetensors"
    write_st(p, {"a": ("BF16", np.zeros((4, 4), dtype=np.uint16))})
    assert ml.read_header(p).tensors["a"]["shape"] == [4, 4]
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ml.MergeError):
        ml.read_header(p)


def test_pick_sample_covers_every_module_type():
    keys = [f"model.language_model.layers.{i}.mlp.{n}.weight" for i in range(8) for n in ("up_proj", "down_proj")]
    keys += [f"model.language_model.layers.{i}.linear_attn.in_proj_a.weight" for i in (0, 1, 2)]
    got = ml.pick_sample(keys)
    assert {ml.module_type(k) for k in got} == {"mlp.up_proj", "mlp.down_proj", "linear_attn.in_proj_a"}
    assert "model.language_model.layers.0.mlp.up_proj.weight" in got and "model.language_model.layers.7.mlp.up_proj.weight" in got


def test_other_libraries_read_the_merged_shards_and_agree(merged):
    fx, out, _ = merged
    st = pytest.importorskip("safetensors")
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    for fn in sorted(set(fx["weight_map"].values())):
        with safe_open(str(out / fn), "pt") as f, safe_open(str(fx["base"] / fn), "pt") as g:
            assert sorted(f.keys()) == sorted(g.keys())
            for name in f.keys():
                assert f.get_tensor(name).dtype == g.get_tensor(name).dtype
                if name in TARGETS:
                    a, b = (torch.from_numpy(x) for x in fx["pairs"][name])
                    want = (g.get_tensor(name).float() + fx["scale"] * (b @ a)).to(torch.bfloat16)
                    diff = (f.get_tensor(name).view(torch.int16).int() - want.view(torch.int16).int()).abs()
                    assert int(diff.max()) <= 1 and float((diff > 0).float().mean()) < 0.02, name


def test_verify_tensor_allows_float32_error_only_where_w_and_delta_cancel():
    one = np.array([[1.0]], dtype=np.float32)
    lo = ml.Lora(a=np.array([[1.0 + 2.0 ** -21]], dtype=np.float32), b=one)       # delta = 1 + 2^-21 (exact)
    w = np.array([[-1.0]], dtype=np.float32)                                          # sum = 2^-21 = 4.8e-7
    ideal = 2.0 ** -21
    # a result off by 2^-24 (a float32 rounding of the delta): 32 half-ulps of the output, but 1/16 of the allowance
    m = np.array([[ideal + 2.0 ** -24]], dtype=np.float32)
    st = ml.verify_tensor(w, m, lo, 1.0, "BF16")
    assert st["max_err_over_half_ulp"] > 10 and st["max_err_over_tol"] < 0.1 and st["elements_over_half_ulp"] == 1
    # a real error (2^-12, a thousandth of the weight) is caught by both
    bad = ml.verify_tensor(w, np.array([[ideal + 2.0 ** -12]], dtype=np.float32), lo, 1.0, "BF16")
    assert bad["max_err_over_tol"] > 1.0
    # the exactly rounded value passes both
    exact = ml.verify_tensor(w, bf16_to_f32(ml.f32_to_bf16_bits(np.array([[ideal]], dtype=np.float32))), lo, 1.0, "BF16")
    assert exact["max_err_over_half_ulp"] == 0.0 and exact["max_err_over_tol"] == 0.0


def test_a_wrong_scale_is_caught(tmp_path):
    """A merge made with scale 1 instead of alpha/r = 2 must fail verification (a half-ulp check alone would miss small deltas)."""
    fx = make_fixture(tmp_path)
    out = tmp_path / "out"
    ml.merge(fx["base"], fx["adapter"], out, log=lambda *_: None)
    cfg = json.loads((fx["adapter"] / "adapter_config.json").read_text())
    cfg["lora_alpha"] = R                                         # verify against scale 1.0
    (fx["adapter"] / "adapter_config.json").write_text(json.dumps(cfg))
    v = ml.verify(fx["base"], fx["adapter"], out, which="all", log=lambda *_: None)
    assert not v["ok"] and v["max_err_over_tol"] > 1.0
