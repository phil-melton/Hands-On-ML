"""The Stage 4 data-foundry machinery: specs and their invariants, the prompts file, the Colab job script with vLLM
mocked (thinking split, per-prompt settings), the Colab driver with the CLI mocked (dry run, guards, and the stop in
``finally``), verification and tiers on fake completions built from the families' REFERENCE solutions and mutants, AST
dedup, repair prompts, the report, and the Gate 4a additions: teacher-written phrasings and their screen, the full-generation
specs, drills (a fake ``foundry.drills``), the thinking-mode test, repair pairs by pairing, and the dataset builder.

No GPU, no network and no Colab command is run. The one place the real CLI is touched is ``colab help <cmd>`` (read-only),
to check every flag the driver uses against the installed CLI; that test skips when the CLI is absent.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from bakeoff.rungs import RUNGS, system_prompt
from cvbench.families import contour_area as ca
from cvbench.families import histogram_stats as hs
from cvbench.splits import (FAMILIES, FOUNDRY_PHRASINGS, FOUNDRY_RANGE, N_SCENES, SCENE_STRIDE, build_splits,
                            foundry_phrasings)
from foundry import colab as colab_mod
from foundry import colab_job, repair as repair_mod, report as report_mod, verify as verify_mod
from foundry import dataset as ds
from foundry import phrasings as phr
from foundry import session as session_mod
from foundry import specs as specs_mod
from foundry.answer import answer_of, answer_problems, format_answer, row_problems
from foundry.astnorm import ast_hash, normalize_code
from foundry.run import main, predict_mix, predict_text, prompt_rows
from foundry.specs import (FOUNDRY_BLOCK, NOTES_PATH, SpecError, check_specs, family_counts, full_specs, pilot_specs,
                           reserved_prompt_hashes, sha1, sizing, teacher_system, think_test_specs, trainable_families)
from foundry.store import FoundryRun, read_json, read_jsonl
from train.config import BASES, SYSTEM, render_turn


# ================================================================ specs

@pytest.fixture(scope="module")
def specs200():
    return pilot_specs(200)


def test_pilot_specs_are_balanced_over_the_seven_trainable_families(specs200):
    fams = trainable_families()
    assert len(fams) == 7 and not any(FAMILIES[f].held_out for f in fams)
    counts = {f: sum(s.family == f for s in specs200) for f in fams}
    assert len(specs200) == 200 and sum(counts.values()) == 200
    assert max(counts.values()) - min(counts.values()) <= 1                       # balanced: 4 x 29 + 3 x 28
    assert sorted(counts.values()) == [28, 28, 28, 29, 29, 29, 29]
    assert {s.family for s in specs200} == set(fams)
    held_out = {f for f, row in FAMILIES.items() if row.held_out}
    assert held_out == {"chessboard_corners", "color_segmentation"} and not held_out & {s.family for s in specs200}
    assert family_counts(200) == counts
    assert len({s.id for s in specs200}) == 200


def test_specs_use_foundry_phrasings_only_and_the_contract_prompt(specs200):
    for s in specs200:
        assert s.phrasing_index in FOUNDRY_PHRASINGS
        fam_spec = importlib.import_module("cvbench.families." + s.family)
        task = foundry_phrasings(s.family)[s.phrasing_index - FOUNDRY_PHRASINGS.start]
        assert s.prompt == dataclasses.replace(fam_spec.SPEC, task=task).to_prompt()    # TaskSpec.to_prompt(), Rule 7
        assert s.prompt.startswith("Task: " + task)
        assert task not in fam_spec.PHRASINGS[:FOUNDRY_PHRASINGS.start]                  # no dev or test task line
        assert s.system == SYSTEM
    # all four foundry phrasings are used by every family
    for f in trainable_families():
        assert {s.phrasing_index for s in specs200 if s.family == f} == set(FOUNDRY_PHRASINGS)


def test_no_spec_prompt_equals_a_dev_test_or_gotcha_prompt(specs200):
    dev, test = build_splits()
    scored = {r["prompt_sha1"] for r in dev + test}
    assert len(scored) == 36 + 72                                                       # the disjoint-phrasings fix is in place
    spec_hashes = {s.prompt_sha1 for s in specs200}
    assert not spec_hashes & scored
    assert not spec_hashes & set(reserved_prompt_hashes())                              # gotcha prompts included
    assert len(spec_hashes) == 7 * 4                                                    # 4 foundry prompts per family


def test_seed_windows_lie_in_the_foundry_range_and_never_overlap(specs200):
    windows = sorted(s.window for s in specs200)
    for lo, hi in windows:
        assert FOUNDRY_RANGE[0] <= lo and hi <= FOUNDRY_RANGE[1] and hi - lo == N_SCENES
    for (lo1, hi1), (lo2, _) in zip(windows, windows[1:]):
        assert hi1 <= lo2
    by_family = {}
    for s in specs200:
        by_family.setdefault(s.family, []).append(s)
    for f, mine in by_family.items():
        seeds = [s.seed for s in mine]
        assert [b - a for a, b in zip(seeds, seeds[1:])] == [SCENE_STRIDE] * (len(seeds) - 1)   # stepping by SCENE_STRIDE
        assert seeds[0] == FOUNDRY_RANGE[0] + FOUNDRY_BLOCK * list(FAMILIES).index(f)
    # no dev or test task graded on a scene a foundry spec uses
    dev, test = build_splits()
    assert all(r["seed"] + N_SCENES <= FOUNDRY_RANGE[0] for r in dev + test)


def test_check_specs_rejects_each_violation(specs200):
    s = specs200[0]
    reserved = reserved_prompt_hashes()
    check_specs([s], reserved=reserved)
    chess = dataclasses.replace(s, family="chessboard_corners", id="x")
    with pytest.raises(SpecError, match="held-out"):
        check_specs([chess], reserved=reserved)
    with pytest.raises(SpecError, match="not a foundry phrasing"):
        check_specs([dataclasses.replace(s, phrasing_index=3)], reserved=reserved)
    with pytest.raises(SpecError, match="outside FOUNDRY_RANGE"):
        check_specs([dataclasses.replace(s, seed=FOUNDRY_RANGE[1] - 5)], reserved=reserved)
    with pytest.raises(SpecError, match="outside FOUNDRY_RANGE"):
        check_specs([dataclasses.replace(s, seed=4_999_990)], reserved=reserved)
    with pytest.raises(SpecError, match="overlap"):
        check_specs([s, dataclasses.replace(s, id="other", seed=s.seed + N_SCENES - 1)], reserved=reserved)
    with pytest.raises(SpecError, match="duplicate"):
        check_specs([s, s], reserved=reserved)
    with pytest.raises(SpecError, match="is the prompt of scored task"):
        check_specs([s], reserved={s.prompt_sha1: "test-contour_area-000"})
    with pytest.raises(SpecError, match="not TaskSpec.to_prompt"):
        check_specs([dataclasses.replace(s, prompt=s.prompt + " ")], reserved=reserved)
    with pytest.raises(SpecError, match="train.config.SYSTEM"):
        check_specs([dataclasses.replace(s, system=SYSTEM + "!")], reserved=reserved)


def test_specs_are_prefix_stable_and_the_teacher_prompt_is_system_plus_notes(specs200):
    small = pilot_specs(70)
    for f in trainable_families():
        a, b = [s for s in small if s.family == f], [s for s in specs200 if s.family == f]
        assert a == b[:len(a)]
    notes = NOTES_PATH.read_text(encoding="utf-8")
    assert specs200[0].teacher_system == SYSTEM + "\n\n" + notes.strip() == teacher_system()
    assert specs200[0].teacher_system == system_prompt(RUNGS["r4-7b-notes"], notes)     # the bake-off's notes rung
    assert [m["role"] for m in specs200[0].messages()] == ["system", "user"] and specs200[0].messages()[1]["content"] == specs200[0].prompt
    with pytest.raises(ValueError):
        teacher_system("  ")


# ================================================================ prepare: the prompts file

def test_prepare_writes_the_prompts_file_the_job_reads(tmp_path):
    assert main(["prepare", "--run", "p", "--root", str(tmp_path), "--n", "14"]) == 0
    run = FoundryRun.of("p", tmp_path)
    rows, prompts = read_jsonl(run.specs), read_jsonl(run.stage("generate").prompts)
    assert len(rows) == len(prompts) == 14 and [r["id"] for r in rows] == [p["id"] for p in prompts]
    for r, p in zip(rows, prompts):
        assert p["spec_id"] == p["id"] == r["id"] and p["kind"] == "generate"
        assert [m["role"] for m in p["messages"]] == ["system", "user"]
        assert p["messages"][1]["content"] == r["prompt"] and sha1(r["prompt"]) == r["prompt_sha1"]
        assert p["messages"][0]["content"].startswith(SYSTEM + "\n\n## OpenCV 5.0 notes")
        assert r["phrasing_index"] in FOUNDRY_PHRASINGS and FOUNDRY_RANGE[0] <= r["seed"] < FOUNDRY_RANGE[1]
    cfg = read_json(run.config)
    assert cfg["n_specs"] == 14 and cfg["notes_sha1"] == sha1(NOTES_PATH.read_text(encoding="utf-8"))
    colab_job.read_prompts(run.stage("generate").prompts)                                # the job accepts the file as written
    before = run.stage("generate").prompts.read_bytes()
    assert main(["prepare", "--run", "p", "--root", str(tmp_path), "--n", "14"]) == 0   # idempotent
    assert run.stage("generate").prompts.read_bytes() == before
    run.stage("generate").completions.write_text("{}\n")                               # a run with output must not be re-prepared differently
    with pytest.raises(SystemExit, match="different prompts"):
        main(["prepare", "--run", "p", "--root", str(tmp_path), "--n", "21"])


# ================================================================ the Colab job script, vLLM mocked

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert tokenize is False and add_generation_prompt is True
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        tail = {False: "<think></think>", True: "<think>\n"}.get(kw.get("enable_thinking"), "")   # Qwen3.5 opens the block in the prompt
        return body + "<|im_start|>assistant\n" + tail

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


CLOSE_ID = 99


class FakeIdsTokenizer(FakeTokenizer):
    """A tokenizer that has ``</think>`` as one token; a "token" of a completion is a whitespace-separated word."""

    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return CLOSE_ID if token == "</think>" else -1


class FakeWordsTokenizer(FakeTokenizer):
    """No ``convert_tokens_to_ids``; ``encode`` counts words, like the completions' fake token ids."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


class FakeLLM:
    """Stands in for vllm.LLM: prompts arrive as token ids, replies come from ``answer(prompt_text, sample, prompt_no)``.
    ``sampling_params`` is one object, or a list with one per prompt (what colab_job passes when the prompts differ)."""

    def __init__(self, answer, finish="stop", tokenizer=None, close_id=None):
        self.answer, self.finish, self.calls, self.n_prompts = answer, finish, [], 0
        self.tokenizer, self.close_id = tokenizer or FakeTokenizer(), close_id

    def get_tokenizer(self):
        return self.tokenizer

    def _ids(self, text):
        words = text.split()
        return [self.close_id if (self.close_id is not None and w == "</think>") else i for i, w in enumerate(words)]

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.calls.append((prompts, sampling_params, use_tqdm))
        outs = []
        for j, p in enumerate(prompts):
            params = sampling_params[j] if isinstance(sampling_params, list) else sampling_params
            text = "" if isinstance(self.tokenizer, FakeWordsTokenizer) else bytes(p["prompt_token_ids"]).decode("utf-8")
            reps = [self.answer(text, i, self.n_prompts) for i in range(params.n)]
            self.n_prompts += 1
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(index=i, text=t, token_ids=self._ids(t), finish_reason=self.finish)
                                                 for i, t in enumerate(reps)]))
        return outs


def _params(**kw):
    return SimpleNamespace(**kw)


def _job_args(tmp_path, prompts_path, **over):
    argv = ["--prompts", str(prompts_path), "--out-dir", str(tmp_path / "out"), "-k", "3"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return colab_job.parse_args(argv)


def _write_prompts(path, n, content="solve it"):
    rows = [{"id": f"s{i}", "spec_id": f"spec{i}", "messages": [{"role": "system", "content": "SYS"},
                                                                {"role": "user", "content": f"{content} {i}"}]} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_job_script_is_standalone_and_imports_cleanly_without_vllm():
    src = Path(colab_job.__file__).read_text(encoding="utf-8")
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not mods & {"foundry", "cvbench", "bakeoff", "train", "sandbox", "numpy", "cv2"}   # nothing from this repo
    top = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else n.module.split(".")[0]
           for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert not top & {"vllm", "torch", "transformers"}                                   # vllm only inside functions
    assert "vllm" not in sys.modules


def test_job_arguments_defaults_and_environment(monkeypatch):
    a = colab_job.parse_args(["--prompts", "p.jsonl"])
    assert (a.temperature, a.top_p, a.max_tokens, a.samples, a.seed) == (0.8, 0.95, 2048, 4, 0)
    assert (a.max_model_len, a.gpu_memory_utilization, a.quantization, a.chat_template_kwargs) == (8192, 0.90, None, {})
    assert a.out_dir == "/content/foundry/out" and a.completions_name == "completions.jsonl" and a.timing_name == "timing.json"
    b = colab_job.parse_args(["--model", "m/x", "--quantization", "awq", "--max-model-len", "4096", "--gpu-memory-utilization", "0.8",
                              "-k", "2", "--temperature", "0.5", "--top-p", "0.9", "--max-tokens", "100", "--seed", "7",
                              "--chat-template-kwargs", '{"enable_thinking": false}', "--enforce-eager"])
    assert (b.model, b.quantization, b.max_model_len, b.gpu_memory_utilization, b.samples) == ("m/x", "awq", 4096, 0.8, 2)
    assert (b.temperature, b.top_p, b.max_tokens, b.seed, b.enforce_eager) == (0.5, 0.9, 100, 7, True)
    assert b.chat_template_kwargs == {"enable_thinking": False}
    monkeypatch.setenv("FOUNDRY_JOB_ARGS", "--model env/model -k 5")                  # `colab exec -f` has no argv
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "kernel.json"])
    c = colab_job.parse_args()
    assert (c.model, c.samples) == ("env/model", 5)
    for bad in (["-k", "0"], ["--gpu-memory-utilization", "1.5"], ["--chat-template-kwargs", "[1]"], ["--chat-template-kwargs", "{"],
                ["--top-p", "0"]):
        with pytest.raises(SystemExit):
            colab_job.parse_args(bad)


def test_job_output_format_timing_and_chat_template(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 5)
    a = _job_args(tmp_path, prompts, chunk=2, chat_template_kwargs='{"enable_thinking": false}', model="m/x", seed=11)
    llm = FakeLLM(lambda text, i, n: f"```python\ncode {n} {i}\n```")
    logs = []
    t = colab_job.run(a, llm=llm, make_params=_params, log=logs.append)

    rows = colab_job.read_jsonl(tmp_path / "out" / "completions.jsonl")
    assert len(rows) == 5 * 3 and {r["sample"] for r in rows} == {0, 1, 2}
    r0 = next(r for r in rows if r["id"] == "s0" and r["sample"] == 1)
    assert set(r0) == {"id", "spec_id", "sample", "text", "prompt_tokens", "completion_tokens", "finish_reason", "model",
                       "thinking", "answer", "think_status", "thinking_tokens", "answer_tokens"}
    assert r0["spec_id"] == "spec0" and r0["text"] == "```python\ncode 0 1\n```" and r0["finish_reason"] == "stop" and r0["model"] == "m/x"
    assert r0["answer"] == r0["text"] and r0["thinking"] == "" and r0["think_status"] == "none"         # no think block: all answer
    assert (r0["thinking_tokens"], r0["answer_tokens"]) == (0, r0["completion_tokens"])
    assert r0["completion_tokens"] == len(r0["text"].split()) and r0["prompt_tokens"] > 0

    prompts_sent, params, use_tqdm = llm.calls[0]                                      # the model's own chat template was applied
    text = bytes(prompts_sent[0]["prompt_token_ids"]).decode()
    assert text.startswith("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nsolve it 0<|im_end|>\n<|im_start|>assistant\n<think></think>")
    assert (params.n, params.temperature, params.top_p, params.max_tokens) == (3, 0.8, 0.95, 2048) and use_tqdm is False
    assert [len(c[0]) for c in llm.calls] == [2, 2, 1]                                  # chunk = 2 prompts per generate() call

    timing = json.loads((tmp_path / "out" / "timing.json").read_text())
    assert timing == t and timing["complete"] is True and timing["model"] == "m/x"
    assert timing["n_samples"] == 15 and timing["n_prompts"] == 5 and len(timing["chunks"]) == 3
    assert timing["output_tokens"] == sum(r["completion_tokens"] for r in rows)
    assert timing["prompt_tokens"] == sum(r["prompt_tokens"] for r in rows if r["sample"] == 0)   # counted once per prompt
    assert timing["gen_s"] >= 0 and timing["load_s"] >= 0 and "output_tokens_per_s" in timing
    assert timing["args"]["seed"] == 11 and {"vllm", "torch", "gpu", "python"} <= set(timing["env"])
    assert any("chunk 3/3" in m for m in logs)


def test_job_resumes_accumulates_timing_and_flags_a_prompt_that_cannot_fit(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 4)
    a = _job_args(tmp_path, prompts, limit=2)
    colab_job.run(a, llm=FakeLLM(lambda t, i, n: "a b c"), make_params=_params, log=lambda m: None)
    comp = tmp_path / "out" / "completions.jsonl"
    first = colab_job.read_jsonl(comp)
    assert len(first) == 6
    with comp.open("a") as fh:                                                          # a kill left s2 half written
        fh.write(json.dumps({"id": "s2", "spec_id": "spec2", "sample": 0, "text": "x", "prompt_tokens": 1,
                             "completion_tokens": 1, "finish_reason": "stop", "model": "m"}) + "\n" + '{"id": "s2", "sam')
    llm = FakeLLM(lambda t, i, n: "d e")
    t = colab_job.run(_job_args(tmp_path, prompts), llm=llm, make_params=_params, log=lambda m: None)
    rows = colab_job.read_jsonl(comp)
    assert len(rows) == 12 and {(r["id"], r["sample"]) for r in rows} == {(f"s{i}", j) for i in range(4) for j in range(3)}
    assert rows[:6] == first and sum(len(c[0]) for c in llm.calls) == 2                 # s0 and s1 were not asked again
    assert t["n_invocations"] == 2 and t["n_prompts"] == 4 and t["complete"] is True    # timing accumulates
    again = FakeLLM(lambda t, i, n: "never")
    colab_job.run(_job_args(tmp_path, prompts), llm=again, make_params=_params, log=lambda m: None)
    assert again.calls == [] and len(colab_job.read_jsonl(comp)) == 12                  # nothing to do: no model call

    long_prompts = tmp_path / "long.jsonl"
    _write_prompts(long_prompts, 1, content="x" * 400)
    b = colab_job.parse_args(["--prompts", str(long_prompts), "--out-dir", str(tmp_path / "out2"), "-k", "2", "--max-model-len", "100"])
    llm = FakeLLM(lambda t, i, n: "never")
    colab_job.run(b, llm=llm, make_params=_params, log=lambda m: None)
    rows = colab_job.read_jsonl(tmp_path / "out2" / "completions.jsonl")
    assert [r["finish_reason"] for r in rows] == ["prompt_too_long"] * 2 and llm.calls == []


def test_job_without_vllm_exits_with_an_instruction(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 1)
    monkeypatch.setitem(sys.modules, "vllm", None)                                      # import vllm -> ImportError
    with pytest.raises(SystemExit, match="vllm is not installed"):
        colab_job.main(["--prompts", str(prompts), "--out-dir", str(tmp_path / "o")])
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "messages": [{"role": "robot", "content": "x"}]}\n')
    with pytest.raises(SystemExit, match="role"):
        colab_job.read_prompts(bad)


# ================================================================ the driver, CLI mocked

def _argv_sub(argv):
    """The colab subcommand of an argv (the global --auth takes a value and precedes it)."""
    rest = argv[1:]
    if rest and rest[0] == "--auth":
        rest = rest[2:]
    return rest[0]


class FakeRunner:
    """Records every command; ``fail`` maps a subcommand to a return code or an exception to raise."""

    def __init__(self, cfg, fail=None, usage="12.5 CU left"):
        self.cfg, self.fail, self.usage, self.calls = cfg, dict(fail or {}), usage, []

    def names(self):
        return [_argv_sub(c) for c in self.calls]

    def __call__(self, argv, *, timeout, log_path=None):
        self.calls.append(list(argv))
        sub = _argv_sub(argv)
        f = self.fail.get(sub)
        if isinstance(f, BaseException):
            raise f
        if f:
            return colab_mod.CmdResult(f, "boom")
        if sub == "download" and "completions" in argv[-2]:                              # the job produced everything
            prompts = read_jsonl(self.cfg.run.stage(self.cfg.stage).prompts)
            Path(argv[-1]).write_text("".join(json.dumps({"id": p["id"], "spec_id": p["spec_id"], "sample": s}) + "\n"
                                              for p in prompts for s in range(self.cfg.samples)))
        elif sub == "download":
            Path(argv[-1]).write_text("{}")
        return colab_mod.CmdResult(0, self.usage if sub == "usage" else "")


@pytest.fixture
def run14(tmp_path):
    assert main(["prepare", "--run", "d", "--root", str(tmp_path), "--n", "14"]) == 0
    return FoundryRun.of("d", tmp_path)


def _cfg(run, **kw):
    base = dict(run=run, stage="generate", gpu="L4", predicted_minutes=45.0, predicted_cu=1.3, teacher_license="apache-2.0",
                colab_bin="colab", quantization="awq")
    return colab_mod.JobConfig(**{**base, **kw})


def test_job_arguments_of_the_driver_parse_with_the_job_parser(run14):
    cfg = _cfg(run14, chat_template_kwargs={"enable_thinking": False}, enforce_eager=True, limit=3, max_num_seqs=64)
    a = colab_job.parse_args(colab_mod.job_args(cfg))
    assert (a.model, a.quantization, a.samples, a.max_model_len, a.chunk, a.limit, a.max_num_seqs) == (
        cfg.model, "awq", 4, 8192, 50, 3, 64)
    assert a.chat_template_kwargs == {"enable_thinking": False} and a.enforce_eager is True
    assert a.prompts == "/content/generate_prompts.jsonl" and a.completions_name == "generate_completions.jsonl"
    assert colab_job.parse_args(colab_mod.job_args(_cfg(run14, stage="repair"))).samples == 1   # repair: asked once


def test_dry_run_prints_the_commands_and_runs_nothing(run14):
    runner = FakeRunner(_cfg(run14))
    out = []
    rec = colab_mod.run_job(_cfg(run14, install_via="colab", packages=("vllm==0.11.0",)), dry_run=True, runner=runner, log=out.append)
    assert runner.calls == [] and rec["dry_run"] is True
    assert [_argv_sub(c) for c in rec["commands"]] == ["usage", "new", "install", "upload", "upload", "exec", "download", "download", "stop", "usage"]
    s = "colab"
    assert rec["commands"][1] == [s, "new", "-s", "foundry-d-generate", "--gpu", "L4"]
    assert rec["commands"][2] == [s, "install", "-s", "foundry-d-generate", "vllm==0.11.0"]
    assert rec["commands"][3][:3] == [s, "upload", "-s"] and rec["commands"][3][-1] == "/content/colab_job.py"
    assert rec["commands"][4][-1] == "/content/generate_prompts.jsonl"
    ex = rec["commands"][5]
    assert ex[:4] == [s, "exec", "-s", "foundry-d-generate"] and "-f" in ex and ex[-2:] == ["--timeout", "5400"]   # 2 x 45 min
    assert rec["commands"][-2] == [s, "stop", "-s", "foundry-d-generate"]
    text = "\n".join(out)
    assert "launcher" in text and "colab_job.py" in text and "[finally] stop" in text
    assert not (run14.path / "launch_generate.py").exists() and not run14.stage("generate").session.exists()   # a dry run writes nothing
    assert colab_mod.run_job(_cfg(run14, install_via="pip"), dry_run=True, log=lambda m: None)["steps"].count("install") == 0
    assert "'vllm'" in colab_mod.launcher_source(_cfg(run14)) and "PIP = []" in colab_mod.launcher_source(_cfg(run14, install_via="colab"))


def test_guards_nothing_runs_without_approval_prediction_or_a_valid_gpu(run14):
    runner = FakeRunner(_cfg(run14))
    with pytest.raises(colab_mod.ApprovalRequired):
        colab_mod.run_job(_cfg(run14), runner=runner, log=lambda m: None)
    with pytest.raises(colab_mod.ConfigError, match="predicted_minutes"):
        colab_mod.run_job(_cfg(run14, predicted_minutes=None), approved=True, runner=runner, log=lambda m: None)
    with pytest.raises(colab_mod.ConfigError, match="predicted_cu"):
        colab_mod.run_job(_cfg(run14, predicted_cu=0), approved=True, runner=runner, log=lambda m: None)
    with pytest.raises(colab_mod.ConfigError, match="teacher_license"):
        colab_mod.run_job(_cfg(run14, teacher_license=None), approved=True, runner=runner, log=lambda m: None)
    for gpu in ("l4", "A10", "H200", ""):
        with pytest.raises(colab_mod.ConfigError, match="unknown GPU"):                  # the CLI would silently bill an A100
            colab_mod.run_job(_cfg(run14, gpu=gpu), dry_run=True, log=lambda m: None)
    assert runner.calls == []
    # through the CLI: refused with exit code 2 and no command run
    base = ["colab", "--run", "d", "--root", str(run14.path.parent), "--colab-bin", "/nonexistent/colab"]
    assert main([*base, "--i-have-approval"]) == 2
    assert main([*base, "--i-have-approval", "--predicted-minutes", "30", "--predicted-cu", "1", "--teacher-license", "apache-2.0",
                 "--gpu", "L40"]) == 2


def test_real_run_sequence_records_usage_and_stops(run14):
    cfg = _cfg(run14)
    runner = FakeRunner(cfg)
    rec = colab_mod.run_job(cfg, approved=True, runner=runner, log=lambda m: None)
    assert runner.names() == ["usage", "new", "upload", "upload", "exec", "download", "download", "stop", "usage"]
    assert rec["state"] == "ok" and rec["stop_ok"] is True and rec["error"] is None
    assert rec["usage_before"] == rec["usage_after"] == "12.5 CU left"
    session = read_json(run14.stage("generate").session)
    assert session["model"] == cfg.model and session["teacher_license"] == "apache-2.0" and session["predicted_cu"] == 1.3
    assert [c["name"] for c in session["commands"]][-2:] == ["stop", "usage-after"]
    launcher = (run14.path / "launch_generate.py").read_text()
    compile(launcher, "launch_generate.py", "exec")                                      # the generated launcher is valid Python
    assert read_jsonl(run14.stage("generate").completions) and colab_mod.missing_prompts(cfg) == 0
    runner2 = FakeRunner(cfg)                                                            # complete locally: no VM at all
    assert colab_mod.run_job(cfg, approved=True, runner=runner2, log=lambda m: None) == {"skipped": True, "missing_prompts": 0}
    assert runner2.calls == []


@pytest.mark.parametrize("fail", [{"exec": 1}, {"exec": subprocess.TimeoutExpired("colab", 1)}, {"upload": 1}, {"install": 1},
                                  {"download": 1}, {"new": 1}, {"new": subprocess.TimeoutExpired("colab", 1)},
                                  {"exec": KeyboardInterrupt()}])
def test_stop_runs_in_finally_whatever_fails(run14, fail):
    cfg = _cfg(run14, install_via="colab")
    runner = FakeRunner(cfg, fail=fail)
    exc = (KeyboardInterrupt, subprocess.TimeoutExpired, colab_mod.ColabError)
    with pytest.raises(exc):
        colab_mod.run_job(cfg, approved=True, runner=runner, log=lambda m: None)
    names = runner.names()
    assert names.count("stop") == 1 and names[-2:] == ["stop", "usage"]                  # stop, then the usage after
    assert names.index("stop") > names.index("new")                                      # even when `new` itself failed
    session = read_json(cfg.run.stage("generate").session)
    assert session["state"] == "failed" and session["stop_ok"] is True and session["error"]
    if "exec" in names and "download" not in fail:                                       # a failed job: salvage what it wrote first
        assert names[names.index("exec") + 1:names.index("stop")] == ["download", "download"]


def test_a_failed_stop_is_retried_and_reported_loudly(run14):
    cfg = _cfg(run14)
    runner = FakeRunner(cfg, fail={"stop": 1})
    logs = []
    rec = colab_mod.run_job(cfg, approved=True, runner=runner, log=logs.append)
    assert runner.names().count("stop") == 2 and rec["stop_ok"] is False
    assert any("DID NOT SUCCEED" in m for m in logs) and any("colab sessions" in m for m in logs)


def test_the_cli_dry_run_exits_zero_without_touching_the_cli(run14, capsys):
    assert main(["colab", "--run", "d", "--root", str(run14.path.parent), "--stage", "generate", "--dry-run",
                 "--colab-bin", "/nonexistent/colab"]) == 0
    assert "/nonexistent/colab new -s foundry-d-generate --gpu L4" in capsys.readouterr().out


def test_exec_timeout_is_twice_the_prediction_and_resume_uploads_partial_output(run14):
    cfg = _cfg(run14, predicted_minutes=10.0)
    assert cfg.timeout_s == 1200.0 and _cfg(run14, predicted_minutes=10.0, exec_timeout_s=99).timeout_s == 99.0
    exec_step = next(s for s in colab_mod.plan(cfg) if s.name == "exec")
    assert exec_step.argv[-2:] == ["--timeout", "1200"] and exec_step.timeout_s > 1200      # the local kill comes after the remote one
    st = run14.stage("generate")
    st.completions.write_text(json.dumps({"id": "foundry-contour_area-0000", "spec_id": "x", "sample": 0}) + "\n")
    st.timing.write_text("{}")
    names = [s.name for s in colab_mod.plan(cfg)]
    assert names[names.index("upload-prompts") + 1:names.index("exec")] == ["upload-partial-completions", "upload-partial-timing"]
    assert colab_mod.missing_prompts(cfg) == 14                                          # k = 4 samples needed, one present
    auth = colab_mod.plan(_cfg(run14, colab_auth="adc"))
    assert all(s.argv[1:3] == ["--auth", "adc"] for s in auth)                           # the global flag precedes the subcommand


def test_job_script_runs_standalone_as_a_script(tmp_path):
    cmd = [sys.executable, "-I", colab_job.__file__]                                     # -I: no cwd, no PYTHONPATH, so no repo imports
    r = subprocess.run([*cmd, "--help"], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert r.returncode == 0 and "--chat-template-kwargs" in r.stdout and "--quantization" in r.stdout
    assert subprocess.run([*cmd, "-k", "0"], capture_output=True, text=True, cwd=tmp_path, timeout=60).returncode == 2


def test_launcher_runs_the_job_in_a_subprocess_and_fails_loudly(run14):
    src = colab_mod.launcher_source(_cfg(run14, install_via="colab"))

    def launch(args):
        code = src.replace(repr(colab_mod.REMOTE_JOB), repr(colab_job.__file__))
        code = re.sub(r"^ARGS = .*$", f"ARGS = {args!r}", code, flags=re.M)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)

    ok = launch(["--help"])
    assert ok.returncode == 0 and "launcher: job finished OK" in ok.stdout and "usage:" in ok.stdout
    bad = launch(["-k", "0"])
    assert bad.returncode != 0 and "colab_job.py exited with status 2" in bad.stderr      # the kernel would show this traceback


def test_subprocess_runner_tees_output_and_kills_on_timeout(tmp_path):
    log = tmp_path / "job.log"
    res = colab_mod.subprocess_runner([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], timeout=30, log_path=log)
    assert res.returncode == 3 and res.stdout.strip() == "hello" and log.read_text().strip() == "hello"
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        colab_mod.subprocess_runner([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    assert time.monotonic() - t0 < 20


def test_specs_and_predict_commands(run14, capsys):
    assert main(["specs", "--n", "21"]) == 0
    out = capsys.readouterr().out
    assert "21 specs" in out and "contour_area" in out and "chessboard" not in out and "color_segmentation" not in out
    text = predict_text(200, 4, 700, 600, 300, 600, 1.7, 0.1)                            # 560,000 tokens at 600 tok/s
    assert "560000 output tokens" in text and "generation 15.6 min" in text and "predicted 30.6 min" in text and "0.87 CU" in text
    assert "kill at 2x = 61 min" in text
    assert main(["predict", "--run", "d", "--root", str(run14.path.parent), "--tok-s", "600", "--tokens", "700"]) == 0
    assert "prompts 14 x k 4 = 56 samples" in capsys.readouterr().out


FAKE_COLAB = '''#!{python}
import json, os, sys
from pathlib import Path

state = Path(os.environ["FAKE_COLAB_STATE"])
argv = sys.argv[1:]
with (state / "calls.log").open("a") as fh:
    fh.write(json.dumps(argv) + "\\n")
if argv[0] == "--auth":
    argv = argv[2:]
sub, args = argv[0], argv[1:]
if "-s" in args:
    i = args.index("-s")
    del args[i:i + 2]
if sub == "usage":
    print("Usage rate 1.7 CU/h, balance 100.0 CU")
elif sub == "upload":
    local, remote = args
    (state / Path(remote).name).write_bytes(Path(local).read_bytes())
elif sub == "exec":
    if os.environ.get("FAKE_COLAB_FAIL") == "exec":
        print("kernel error")
        sys.exit(1)
    print("launcher ran")
elif sub == "download":
    remote, local = args
    name = Path(remote).name
    if name.endswith("_completions.jsonl"):
        prompts = [json.loads(line) for line in (state / (name.split("_")[0] + "_prompts.jsonl")).read_text().splitlines()]
        Path(local).write_text("".join(json.dumps({{"id": p["id"], "spec_id": p["spec_id"], "sample": s, "text": "x"}}) + "\\n"
                                       for p in prompts for s in range(4)))
    else:
        Path(local).write_text("{{}}")
else:
    print(sub, "ok")
'''


def _fake_colab(tmp_path, monkeypatch):
    state = tmp_path / "fake_state"
    state.mkdir()
    exe = tmp_path / "colab"
    exe.write_text(FAKE_COLAB.format(python=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state))
    return exe, state


def _approved(run, exe):
    return ["colab", "--run", run.name, "--root", str(run.path.parent), "--i-have-approval", "--colab-bin", str(exe),
            "--predicted-minutes", "30", "--predicted-cu", "1", "--teacher-license", "apache-2.0"]


def test_cli_approved_run_through_the_real_runner_and_a_fake_colab_executable(run14, tmp_path, monkeypatch):
    exe, state = _fake_colab(tmp_path, monkeypatch)
    assert main(_approved(run14, exe)) == 0
    calls = [json.loads(line) for line in (state / "calls.log").read_text().splitlines()]
    assert [c[0] for c in calls] == ["usage", "new", "upload", "upload", "exec", "download", "download", "stop", "usage"]
    assert calls[1] == ["new", "-s", "foundry-d-generate", "--gpu", "L4"] and "--timeout" in calls[4] and calls[4][-1] == "3600"
    st = run14.stage("generate")
    assert len(read_jsonl(st.completions)) == 14 * 4 and read_json(st.timing) == {}
    session = read_json(st.session)
    assert session["state"] == "ok" and session["stop_ok"] is True and "balance 100.0 CU" in session["usage_after"]
    assert "launcher ran" in st.log.read_text()                                          # the exec output is kept next to the run
    assert main(_approved(run14, exe)) == 0                                              # complete: no second VM
    assert len((state / "calls.log").read_text().splitlines()) == 9


def test_cli_exec_failure_exits_one_and_still_stops(tmp_path, monkeypatch):
    assert main(["prepare", "--run", "f", "--root", str(tmp_path), "--n", "7"]) == 0
    run = FoundryRun.of("f", tmp_path)
    exe, state = _fake_colab(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_COLAB_FAIL", "exec")
    assert main(_approved(run, exe)) == 1
    subs = [json.loads(line)[0] for line in (state / "calls.log").read_text().splitlines()]
    assert subs.count("stop") == 1 and subs[-2:] == ["stop", "usage"] and "exec" in subs
    assert read_json(run.stage("generate").session)["state"] == "failed"


def _colab_cli():
    return shutil.which("colab") or (str(p) if (p := Path.home() / "venvs" / "colab" / "bin" / "colab").is_file() else None)


@pytest.mark.skipif(_colab_cli() is None, reason="the Colab CLI is not installed")
def test_every_flag_the_driver_uses_exists_in_the_installed_cli_help(run14):
    cli = _colab_cli()
    cfg = _cfg(run14, install_via="colab", colab_auth="oauth2", colab_bin=cli)
    st = run14.stage("generate")
    st.completions.write_text("{}\n")
    st.timing.write_text("{}")
    helps = {}
    for step in colab_mod.plan(cfg):
        sub = _argv_sub(step.argv)
        if sub not in helps:
            r = subprocess.run([cli, "help", sub], capture_output=True, text=True, timeout=60)   # read-only: prints usage
            assert r.returncode == 0, r.stderr
            helps[sub] = r.stdout + r.stderr
        flags = {a for a in step.argv[1:] if a.startswith("-") and a != "-" and not a.lstrip("-").isdigit()}
        flags -= {"--auth"}                                                              # global flag, checked against the top-level help
        for f in flags:
            assert f in helps[sub], f"`colab {sub}` has no {f}"
    top = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
    assert "--auth" in top.stdout + top.stderr


# ================================================================ AST normalization and dedup

RENAMED = '''import numpy as np
import cv2


def solve(img: "np.ndarray") -> dict:
    """Areas of the bright shapes, largest first."""
    # grey, then Otsu
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res: list = sorted((cv2.contourArea(c) for c in cs), reverse=True)
    return {'areas': np.array(res, dtype=np.float64), 'count': len(cs)}
'''


def test_ast_hash_ignores_comments_docstrings_names_annotations_and_import_order():
    assert ast_hash(RENAMED) == ast_hash(ca.REFERENCE)
    assert normalize_code(RENAMED) == normalize_code(ca.REFERENCE)
    assert normalize_code(ca.REFERENCE).startswith("import cv2 as _v0\nimport numpy as _v1\n\ndef solve(_v2):")
    assert ast_hash(RENAMED.replace("import numpy as np", "import numpy as numpy").replace("np.", "numpy.")) == ast_hash(ca.REFERENCE)
    assert ast_hash("import cv2, numpy as np\n" + RENAMED.split("import cv2\n", 1)[1]) == ast_hash(ca.REFERENCE)   # one import line
    assert ast_hash(RENAMED.replace("'areas'", '"areas"')) == ast_hash(ca.REFERENCE)                              # quote style
    helper = "def solve(a):\n    def f(x):\n        'doc'\n        return x + 1\n    return {'v': f(a)}\n"
    renamed = "def solve(b: int) -> dict:\n    def g(y):\n        'another docstring'\n        return y + 1\n    return {'v': g(b)}\n"
    assert ast_hash(helper) == ast_hash(renamed)                                          # helper, parameter and docstring differ
    assert ast_hash(helper) != ast_hash(renamed.replace("y + 1", "y + 2"))
    assert ast_hash(helper) != ast_hash(renamed.replace("{'v':", "{'w':"))                # dict keys are the contract: kept


def test_ast_hash_separates_different_code():
    base = ast_hash(ca.REFERENCE)
    for name, (_, code) in ca.MUTANTS.items():
        assert ast_hash(code) != base, name                                              # every mutant differs from the reference
    assert ast_hash(RENAMED.replace("reverse=True", "reverse=False")) != base            # a literal changes the code
    assert ast_hash(RENAMED.replace("cv2.THRESH_OTSU", "cv2.THRESH_TRIANGLE")) != base   # so does an attribute name
    assert ast_hash(RENAMED.replace("COLOR_BGR2GRAY", "COLOR_RGB2GRAY")) != base
    assert ast_hash(ca.REFERENCE.replace("def solve", "def helper")) != base             # `solve` is the contract: it is never renamed
    broken = "def solve(:\n  pass"
    assert ast_hash(broken) == ast_hash("def solve(:\n    pass") and ast_hash(broken) != base   # unparsable: whitespace-normalized text


def _row(spec, sample, code, tier="A", **kw):
    return {"id": spec, "spec_id": spec, "sample": sample, "family": "contour_area", "tier": tier, "code": code,
            "ast_sha1": ast_hash(code), **kw}


def test_select_examples_dedups_by_ast_and_caps_at_two_per_spec():
    alt1 = ca.MUTANTS["rgb2gray (harmless here)"][1]
    alt2 = ca.REFERENCE.replace("CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE")
    rows = [_row("a", 3, alt2), _row("a", 1, RENAMED), _row("a", 0, ca.REFERENCE), _row("a", 2, alt1),    # unsorted on purpose
            _row("b", 0, ca.REFERENCE), _row("b", 1, ca.REFERENCE), _row("b", 2, "x = 1", tier="B"),
            _row("c", 0, "y = 2", tier="fail")]
    kept, stats = verify_mod.select_examples(rows, cap=2)
    assert [(r["spec_id"], r["sample"], r["rank"]) for r in kept] == [("a", 0, 0), ("a", 2, 1), ("b", 0, 0)]
    assert stats == {"candidates": 6, "duplicates": 2, "over_cap": 1}                    # a1 and b1 duplicates, a3 over the cap
    assert [r["sample"] for r in verify_mod.select_examples(rows, cap=3)[0] if r["spec_id"] == "a"] == [0, 2, 3]
    b_rows, _ = verify_mod.select_examples(rows, cap=2, tier="B")
    assert [r["spec_id"] for r in b_rows] == ["b"]


# ================================================================ tiers and failure messages (unit)

def _gates(**over):
    g = {n: {"passed": True, "detail": "ok"} for n in verify_mod.FOUR}
    g["api"] = {"passed": True, "detail": "clean"}
    for k, v in over.items():
        g[k] = {"passed": False, "detail": v}
    return {"gates": g, "passed": all(x["passed"] for n, x in g.items() if n != "api")}


@pytest.mark.parametrize("over, tier", [
    ({}, "A"),
    ({"manufactured": "scene 5000003: median 119, expected the lower median 114 (upper median 124, N = 11060)"}, "B"),
    ({"manufactured": "scene 5000000: count 12, but the scene has 7 shapes"}, "B"),
    ({"manufactured": "scene 5000000: contract: 'hist' should have shape (256,), got (255,)"}, "fail"),     # a contract problem
    ({"manufactured": "scene 5000000: areas are not sorted largest-first"}, "fail"),                        # the contract's order
    ({"manufactured": "exception on input 3: ValueError not enough values to unpack"}, "fail"),              # did not run to the end
    ({"manufactured": "scene 1: count 0, but the scene has 7 shapes", "metamorphic": "scene 1: dim x0.45: count 0 vs 7"}, "fail"),
    ({"metamorphic": "scene 1: dim x0.45: count 0 vs 7 on the original image"}, "fail"),                    # passes manufactured only
    ({"exec-only": "exception on input 0: IndexError", "manufactured": "scene 1: x"}, "fail"),
    ({"lint": "line 10: [broad-except] hides failures", "manufactured": "scene 1: count 0, but the scene has 7 shapes"}, "fail"),
])
def test_tier_rules(over, tier):
    rec = _gates(**over)
    assert verify_mod.tier_of(rec) == tier
    assert (verify_mod.tier_of(rec) == "A") == rec["passed"]                              # Tier A is exactly the bake-off's pass
    assert verify_mod.tier_of({"code_empty": True, "passed": False}) == "fail"
    assert verify_mod.tier_of({**rec, "grader_error": "Traceback"}) == "fail"


def test_failure_message_names_the_first_failing_gate_without_the_scene_seed():
    rec = {**_gates(manufactured="scene 5000003: count 12, but the scene has 7 shapes"), "lint_findings": [],
           "api_findings": [{"line": 4, "name": "cv2.AKAZE_create", "api": "AKAZE_create", "label": "removed_in_5",
                             "replacement": "cv2.ORB_create or cv2.SIFT_create"}]}
    rec["tier"] = verify_mod.tier_of(rec)
    f = verify_mod.failure_of(rec)
    assert f["gate"] == "manufactured" and f["repairable"] is True
    assert f["message"].startswith("count 12, but the scene has 7 shapes") and "5000003" not in f["message"]
    assert "`cv2.AKAZE_create` is not in OpenCV 5.0; use cv2.ORB_create or cv2.SIFT_create" in f["message"]
    both = {**_gates(**{"exec-only": "exception on input 0: ValueError x", "lint": "line 1: [import] os"}), "lint_findings": [], "api_findings": []}
    assert verify_mod.failure_of(both)["gate"] == "exec-only"                             # gate order: exec-only first
    assert verify_mod.failure_of({"passed": False, "code_empty": True, "tier": "fail"})["repairable"] is False
    assert verify_mod.failure_of({"passed": False, "grader_error": "tb", "tier": "fail"})["repairable"] is False
    assert verify_mod.failure_of({**_gates(), "tier": "A"}) is None


# ================================================================ the pipeline on fake completions (real graders)

def code_of(code: str) -> str:
    """What extract_code returns for a fenced block: the code with exactly one trailing newline."""
    return code.rstrip("\n") + "\n"


def fence(code: str) -> str:
    return "Here is the function.\n\n```python\n" + code.rstrip("\n") + "\n```\n\nIt follows the contract."


CA, HS = ca.MUTANTS, hs.MUTANTS
CHAIN_NONE = ca.REFERENCE.replace("CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE")
# (spec id, sample) -> the teacher's reply. Expected tiers: see test_pipeline below.
TEACHER = {
    "foundry-contour_area-0000": [fence(ca.REFERENCE), fence(RENAMED), fence(CA["rgb2gray (harmless here)"][1]), fence(CHAIN_NONE)],
    "foundry-contour_area-0001": [fence(CA["retr_tree"][1]), fence(CA["unsorted"][1]), fence(CA["fixed_threshold_127"][1]),
                                  fence(CA["swallowed_opencv3_unpack"][1])],
    "foundry-histogram_stats-0000": [fence(hs.REFERENCE), fence(HS["no_gray_conversion"][1]), fence(HS["hist_col0"][1]), "I cannot help with that."],
    "foundry-histogram_stats-0001": [fence(HS["bins_255"][1]), fence(HS["range_255"][1]), fence(hs.REFERENCE), fence(HS["float_counts"][1])],
}
EXPECTED = {
    "foundry-contour_area-0000": ["A", "A", "A", "A"],
    "foundry-contour_area-0001": ["B", "fail", "fail", "fail"],
    "foundry-histogram_stats-0000": ["A", "fail", "fail", "fail"],
    "foundry-histogram_stats-0001": ["fail", "fail", "A", "fail"],
}


def test_verify_never_uses_more_than_four_workers(tmp_path, monkeypatch):
    run, chosen = _make_run(tmp_path)
    run.stage("generate").completions.write_text(json.dumps({"id": chosen[0].id, "spec_id": chosen[0].id, "sample": 0, "text": fence(ca.REFERENCE),
                                                             "completion_tokens": 10, "finish_reason": "stop"}) + "\n")
    seen = []

    class Pool(verify_mod.ProcessPoolExecutor):
        def __init__(self, max_workers=None, **kw):
            seen.append(max_workers)
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr(verify_mod, "ProcessPoolExecutor", Pool)
    counts = verify_mod.grade_stage(run, "generate", verify_mod.VerifySettings(workers=16), log=lambda m: None)
    assert seen == [4] and counts["A"] == 1


def _make_run(tmp_path):
    chosen = [s for s in pilot_specs(14) if s.family in ("contour_area", "histogram_stats")]
    assert [s.id for s in chosen] == [f"foundry-{f}-000{j}" for f in ("contour_area", "histogram_stats") for j in (0, 1)]
    run = FoundryRun.of("pipe", tmp_path)
    run.path.mkdir(parents=True)
    rows = [s.to_row() for s in chosen]
    run.specs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run.stage("generate").prompts.write_text("".join(json.dumps(r) + "\n" for r in prompt_rows(chosen)))
    return run, chosen


def _fake_teacher(run, stage, chosen, answers, k):
    """Run the real job script against a fake vLLM; ``answers(prompt_text, sample, prompt_no)`` is the teacher."""
    st = run.stage(stage)
    a = colab_job.parse_args(["--prompts", str(st.prompts), "--out-dir", str(run.path), "-k", str(k), "--model", "fake/teacher",
                              "--completions-name", st.completions.name, "--timing-name", st.timing.name])
    colab_job.run(a, llm=FakeLLM(answers), make_params=_params, log=lambda m: None)


def test_pipeline_generate_verify_repair_report(tmp_path):
    run, chosen = _make_run(tmp_path)
    by_prompt = {s.prompt: s.id for s in chosen}

    def teacher(text, i, n):
        spec_id = next(sid for p, sid in by_prompt.items() if p in text)
        return TEACHER[spec_id][i]

    _fake_teacher(run, "generate", chosen, teacher, k=4)
    st = run.stage("generate")
    assert len(read_jsonl(st.completions)) == 16
    argv = ["--run", "pipe", "--root", str(tmp_path)]

    # ---- verify: real graders, 20 scenes and 5 metamorphic scenes at each spec's seed
    assert main(["verify", *argv, "--workers", "2"]) == 0
    grades = {(r["spec_id"], r["sample"]): r for r in read_jsonl(st.grades)}
    assert len(grades) == 16
    for sid, tiers in EXPECTED.items():
        assert [grades[(sid, i)]["tier"] for i in range(4)] == tiers, sid
    for r in grades.values():
        assert (r["tier"] == "A") == r["passed"] and r["family"] in ("contour_area", "histogram_stats")
        assert r["tier"] == verify_mod.tier_of(r) and (r["failure"] is None) == (r["tier"] == "A")
    g = grades
    assert g[("foundry-contour_area-0000", 0)]["extract"] == "fence" and g[("foundry-contour_area-0000", 0)]["code"] == code_of(ca.REFERENCE)
    assert g[("foundry-contour_area-0000", 0)]["ast_sha1"] == g[("foundry-contour_area-0000", 1)]["ast_sha1"]   # the renamed copy
    assert g[("foundry-histogram_stats-0000", 3)]["extract"] == "bare" and g[("foundry-histogram_stats-0000", 3)]["tier"] == "fail"
    # the failure messages say what the grader said
    f = lambda sid, i: g[(sid, i)]["failure"]  # noqa: E731
    assert f("foundry-contour_area-0001", 0)["gate"] == "manufactured" and "shapes" in f("foundry-contour_area-0001", 0)["message"]
    assert f("foundry-contour_area-0001", 1)["message"] == "areas are not sorted largest-first"
    assert f("foundry-contour_area-0001", 2)["gate"] == "metamorphic" and "dim" in f("foundry-contour_area-0001", 2)["message"]
    assert f("foundry-contour_area-0001", 3)["gate"] == "lint" and "broad-except" in f("foundry-contour_area-0001", 3)["message"]
    assert f("foundry-histogram_stats-0000", 2)["gate"] == "exec-only" and "IndexError" in f("foundry-histogram_stats-0000", 2)["message"]
    assert f("foundry-histogram_stats-0001", 0)["message"].startswith("contract: 'hist' should have shape (256,)")
    assert f("foundry-histogram_stats-0000", 3)["repairable"] is False

    kept = read_jsonl(run.kept)                                                                              # at most 2, distinct AST
    assert [(r["spec_id"], r["sample"]) for r in kept] == [("foundry-contour_area-0000", 0), ("foundry-contour_area-0000", 2),
                                                           ("foundry-histogram_stats-0000", 0), ("foundry-histogram_stats-0001", 2)]
    assert main(["verify", *argv, "--workers", "2"]) == 0                                                   # resumable: nothing regraded
    assert len(read_jsonl(st.grades)) == 16
    with pytest.raises(SystemExit, match="existing grades used"):
        main(["verify", *argv, "--n-scenes", "5"])                                                          # one grades file, one grader setting

    # ---- repair prompts
    assert main(["repair", *argv]) == 0
    rp = read_jsonl(run.stage("repair").prompts)
    assert [r["id"] for r in rp] == ["foundry-contour_area-0001/s0", "foundry-contour_area-0001/s1", "foundry-histogram_stats-0000/s1",
                                     "foundry-histogram_stats-0000/s2", "foundry-histogram_stats-0001/s0", "foundry-histogram_stats-0001/s1"]
    # (contour_area-0000 already has 2 Tier-A examples; max 2 per spec; the no-code reply is not sent)
    p0 = rp[0]
    assert [m["role"] for m in p0["messages"]] == ["system", "user", "assistant", "user"]
    assert p0["messages"][:2] == read_jsonl(st.prompts)[1]["messages"]                                       # the original messages first
    assert p0["messages"][2]["content"].startswith("```python\nimport cv2") and p0["failed_code"] in p0["messages"][2]["content"]
    assert p0["messages"][3]["content"] == f"That failed: {p0['failure']['message']}\nReturn the corrected full function."
    assert p0["tier_before"] == "B" and p0["spec_id"] == "foundry-contour_area-0001"
    colab_job.read_prompts(run.stage("repair").prompts)                                                      # the same job reads it

    # ---- the repair job (k = 1) and its verification
    def repair_teacher(text, i, n):                                                                          # n: position in repair_prompts.jsonl
        if n == 4:
            return fence(HS["bins_255"][1])                                                                   # one repair stays broken
        return fence(ca.REFERENCE if n < 2 else hs.REFERENCE)

    _fake_teacher(run, "repair", chosen, repair_teacher, k=1)
    assert main(["verify", *argv, "--stage", "repair", "--workers", "2"]) == 0
    rg = read_jsonl(run.stage("repair").grades)
    assert len(rg) == 6 and sorted(r["tier"] for r in rg) == ["A"] * 5 + ["fail"]
    pairs = read_jsonl(run.repair_pairs)
    assert len(pairs) == 5 and {p["tier"] for p in pairs} == {"A"} and "foundry-histogram_stats-0001/s0#0" not in {p["id"] for p in pairs}
    pr = next(p for p in pairs if p["id"] == "foundry-contour_area-0001/s1#0")
    assert pr["failed_code"] == code_of(CA["unsorted"][1]) and pr["fixed_code"] == code_of(ca.REFERENCE)
    assert pr["feedback"].startswith("That failed: areas are not sorted") and pr["provenance"]["teacher"]["model"] == "fake/teacher"
    assert [m["role"] for m in pr["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert pr["messages"][0]["content"] == SYSTEM and SYSTEM in pr["messages"][0]["content"] and "notes" not in pr["messages"][0]["content"]

    # ---- the report, with a measured GPU time of one hour on an L4
    (run.path / "timing.json").write_text(json.dumps({"load_s": 600.0, "gen_s": 3000.0, "output_tokens": 1, "model": "fake/teacher"}))
    (run.path / "repair_timing.json").write_text(json.dumps({"load_s": 100.0, "gen_s": 800.0}))
    assert main(["report", *argv, "--gpu-rate", "1.7"]) == 0
    s = read_json(run.summary)
    t = s["totals"]
    assert (t["samples"], t["tier_A"], t["tier_B"], t["fail"], t["kept"], t["specs"]) == (16, 6, 1, 9, 4, 4)
    assert t["kept_per_spec"] == 1.0 and t["solved_pct"] == 75.0 and t["specs_solved"] == 3
    assert t["gate_pass"] == {"exec-only": 14, "lint": 14, "manufactured": 7, "metamorphic": 9}
    assert t["repair_prompts"] == 6 and t["repair_to_A"] == 5 and t["repair_pairs_A"] == 5
    assert t["gpu_s"] == pytest.approx(3600.0) and t["cu"] == pytest.approx(1.7) and t["usd"] == pytest.approx(0.17)
    assert t["cu_per_A"] == pytest.approx(1.7 / 4) and t["usd_per_A"] == pytest.approx(0.17 / 4)
    assert t["repair_cu"] == pytest.approx(900 / 3600 * 1.7) and t["cu_per_A_incl_repair"] == pytest.approx((1.7 + 900 / 3600 * 1.7) / 9)
    rows = s["rows"]
    assert sum(rows[f]["cu"] for f in rows) == pytest.approx(1.7)                                              # the split adds up
    assert rows["contour_area"]["kept"] == 2 and rows["histogram_stats"]["kept"] == 2
    assert s["teacher"]["model"] == "fake/teacher" and s["selection"] == {"candidates": 6, "duplicates": 1, "over_cap": 1, "kept": 4, "cap": 2,
                                                                         "distinct_prompt_ast": 4, "distinct_prompts": 4, "kept_with_api_findings": 0}
    md = run.report_md.read_text()
    assert "| contour_area |" in md and "| **all** | | 4 | 16 | 6 | 1 | 9 | 4 | 1.00 | 4 | 75.0 |" in md
    assert " 3600 | 1.700 | 0.17 | 4 | 0.425 | 0.0425 | 0.425 | 0.236 |" in md                                  # CU, $, CU / Tier-A, / distinct, incl. repair

    ex = read_jsonl(run.examples)
    a_rows = [e for e in ex if e["tier"] == "A"]
    assert len(a_rows) == 4 and [e["tier"] for e in ex].count("B") == 1                                       # B only where no Tier A: contour_area-0001
    b = next(e for e in ex if e["tier"] == "B")
    assert b["spec_id"] == "foundry-contour_area-0001" and b["provenance"]["sample"] == 0
    e0 = next(e for e in ex if e["id"] == "foundry-contour_area-0000#0")
    spec = chosen[0]
    assert [m["role"] for m in e0["messages"]] == ["system", "user", "assistant"]
    assert e0["messages"][0]["content"] == SYSTEM and e0["messages"][1]["content"] == spec.prompt                # Rule 7, no notes
    assert e0["raw_reply"].startswith("Here is the function.") and e0["code"] == code_of(ca.REFERENCE)
    assert e0["messages"][2]["content"] == e0["code"] and e0["answer_format"] == "code"
    prov = e0["provenance"]
    assert prov["teacher"]["model"] == "fake/teacher" and prov["spec_id"] == spec.id and prov["tier"] == "A"
    assert set(prov["gates"]) >= set(verify_mod.FOUR) and all(prov["gates"][g] for g in verify_mod.FOUR) and prov["scene_seed"] == spec.seed
    assert main(["report", *argv, "--answer-format", "raw", "--no-tier-b"]) == 0
    ex2 = read_jsonl(run.examples)
    assert len(ex2) == 4 and ex2[0]["messages"][2]["content"] == ex2[0]["raw_reply"] and ex2[0]["code"] != ex2[0]["raw_reply"]


# ================================================================ repair prompts and the report on hand-made grades

def _tiny_run(tmp_path):
    run = FoundryRun.of("tiny", tmp_path)
    run.path.mkdir(parents=True)
    specs = [{"id": f"foundry-contour_area-000{j}", "family": "contour_area", "index": j, "phrasing_index": 12, "seed": 5_000_000 + 40 * j,
              "prompt": f"PROMPT {j}", "subset": "timeless"} for j in range(3)]
    specs.append({"id": "foundry-text_layout-0000", "family": "text_layout", "index": 0, "phrasing_index": 12, "seed": 5_800_000,
                  "prompt": "PROMPT T", "subset": "5.0"})
    run.specs.write_text("".join(json.dumps(r) + "\n" for r in specs))
    gen = run.stage("generate")
    gen.prompts.write_text("".join(json.dumps({"id": s["id"], "spec_id": s["id"], "messages": [
        {"role": "system", "content": "TEACHER SYSTEM"}, {"role": "user", "content": s["prompt"]}]}) + "\n" for s in specs))
    return run, specs


def _g(spec, sample, tier, code, family="contour_area", msg="count 0, but the scene has 3 shapes", cls="algorithm", **kw):
    fail = None if tier == "A" else {"gate": "manufactured", "message": msg, "class": cls, "rule": "verifier-value", "repairable": True}
    return {"id": spec, "spec_id": spec, "sample": sample, "family": family, "tier": tier, "code": code, "ast_sha1": ast_hash(code),
            "code_sha1": sha1(code), "failure": fail, "completion_tokens": 100, "finish_reason": "stop", "extract": "fence",
            "gates": {g: {"passed": tier == "A" or g != "manufactured", "detail": ""} for g in verify_mod.FOUR}, **kw}


def test_repair_prompt_construction(tmp_path):
    run, specs = _tiny_run(tmp_path)
    s0, s1, s2, t0 = (s["id"] for s in specs)
    rows = [
        _g(s0, 0, "A", "a = 1"), _g(s0, 1, "A", "a = 2"), _g(s0, 2, "fail", "a = 3"),                  # solved (2 kept): skipped
        _g(s1, 0, "fail", "b = 1"), _g(s1, 1, "fail", "b = 1  # same code, other comment"),             # AST duplicate of sample 0
        _g(s1, 2, "B", "b = 3", msg="median 5, expected 4"), _g(s1, 3, "fail", "b = 4"),                # cap 2: samples 0 and 2
        _g(s2, 0, "fail", "", code_empty=True), _g(s2, 1, "fail", "c = 1", failure={"repairable": False, "gate": None,
                                                                                    "message": "grader error", "class": "other", "rule": "x"}),
        _g(t0, 0, "A", "t = 1", family="text_layout"), _g(t0, 1, "fail", "t = 2", family="text_layout", msg="exception on input 0: KeyError 'x'",
                                                          cls="convention"),
    ]
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = repair_mod.build_repair_prompts(run)
    assert [r["id"] for r in out] == [f"{s1}/s0", f"{s1}/s2", f"{t0}/s1"]                              # nothing for s0, s2 or empty code
    r = out[1]
    assert r["tier_before"] == "B" and r["sample_of"] == 2 and r["failed_code"] == "b = 3"
    assert r["messages"] == [{"role": "system", "content": "TEACHER SYSTEM"}, {"role": "user", "content": "PROMPT 1"},
                             {"role": "assistant", "content": "```python\nb = 3\n```"},
                             {"role": "user", "content": "That failed: median 5, expected 4\nReturn the corrected full function."}]
    assert [x["id"] for x in repair_mod.build_repair_prompts(run, max_per_spec=1)] == [f"{s1}/s0", f"{t0}/s1"]
    assert f"{s0}/s2" in [x["id"] for x in repair_mod.build_repair_prompts(run, include_solved=True)]
    repair_mod.write_repair_prompts(run, out)
    assert main(["repair", "--run", "tiny", "--root", str(tmp_path)]) == 0 and len(read_jsonl(run.stage("repair").prompts)) == 3


def test_report_arithmetic_on_a_tiny_run(tmp_path):
    run, specs = _tiny_run(tmp_path)
    s0, s1, s2, t0 = (s["id"] for s in specs)
    ca_code = [f"x = {i}" for i in range(6)]
    rows = [_g(s0, 0, "A", ca_code[0]), _g(s0, 1, "A", ca_code[1]), _g(s0, 2, "A", ca_code[2]), _g(s0, 3, "fail", ca_code[3], finish_reason="length"),   # 3 A -> 2 kept
            _g(s1, 0, "B", ca_code[4]), _g(s1, 1, "fail", ca_code[5], cls="api"),
            _g(t0, 0, "A", "t = 1", family="text_layout", completion_tokens=300),
            _g(t0, 1, "fail", "t = 2", family="text_layout", completion_tokens=300, cls="convention")]
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run.path / "timing.json").write_text(json.dumps({"load_s": 100.0, "gen_s": 700.0, "model": "q/coder"}))
    (run.path / "session_generate.json").write_text(json.dumps({"model": "q/coder", "teacher_license": "apache-2.0", "wall_s": 1800.0,
                                                                "predicted_minutes": 30, "predicted_cu": 0.9}))
    s = report_mod.summarize_run(run, gpu_rate=1.8, usd_per_cu=0.1, overhead_s=200.0)
    t, ca_row, tl = s["totals"], s["rows"]["contour_area"], s["rows"]["text_layout"]
    assert (t["samples"], t["tier_A"], t["tier_B"], t["fail"], t["kept"], t["specs"]) == (8, 4, 1, 3, 3, 3)
    assert ca_row["kept"] == 2 and ca_row["specs"] == 2 and ca_row["kept_per_spec"] == 1.0 and ca_row["solved_pct"] == 50.0
    assert ca_row["failure_classes"] == {"algorithm": 2, "api": 1} and tl["failure_classes"] == {"convention": 1}
    assert ca_row["truncated"] == 1 and t["truncated"] == 1 and tl["truncated"] == 0   # finish_reason "length": the answer was cut off
    # GPU seconds = 100 + 700 + 200 overhead = 1000 s; 1.8 CU/h -> 0.5 CU; $0.05; split by output tokens (600 contour, 600 text_layout)
    assert t["gpu_s"] == 1000.0 and t["cu"] == pytest.approx(0.5) and t["usd"] == pytest.approx(0.05)
    assert ca_row["output_tokens"] == 600 and tl["output_tokens"] == 600 and ca_row["cu"] == pytest.approx(0.25) and tl["cu"] == pytest.approx(0.25)
    assert ca_row["cu_per_A"] == pytest.approx(0.125) and tl["cu_per_A"] == pytest.approx(0.25) and t["cu_per_A"] == pytest.approx(0.5 / 3)
    assert t["cu_per_A_incl_repair"] == pytest.approx(0.5 / 3)                                                      # no repair stage yet
    assert s["cost"]["session_wall_cu"] == pytest.approx(1800 / 3600 * 1.8) and s["cost"]["decode_only_cu"] == pytest.approx(700 / 3600 * 1.8)
    md = report_mod.render_markdown(s)
    assert "q/coder" in md and "apache-2.0" in md and "No repair stage yet" in md and "| text_layout |" in md
    assert "chessboard" not in md and "color_segmentation" not in md and "hough_segments" not in md       # families without specs are absent
    # without timing the cost columns say n/a rather than inventing a number
    (run.path / "timing.json").unlink()
    s2 = report_mod.summarize_run(run)
    assert s2["totals"]["cu"] is None and s2["totals"]["cu_per_A"] is None and "n/a" in report_mod.render_markdown(s2)
    # a family with no Tier-A example has no finite cost per example
    solo = [r for r in rows if r["family"] == "contour_area" and r["spec_id"] == s1]
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in solo))
    (run.path / "timing.json").write_text(json.dumps({"load_s": 0, "gen_s": 360.0}))
    s3 = report_mod.summarize_run(run)
    assert s3["rows"]["contour_area"]["kept"] == 0 and s3["rows"]["contour_area"]["cu_per_A"] is None and "none" in report_mod.render_markdown(s3)


def test_specs_sharing_a_prompt_count_once_in_distinct_and_in_examples(tmp_path):
    """Specs of one family and phrasing differ only by a seed the prompt does not show: same prompt, same code = one example."""
    run, specs = _tiny_run(tmp_path)
    s0, s1 = specs[0]["id"], specs[1]["id"]
    twin = {**specs[0], "id": "foundry-contour_area-0004", "index": 4, "seed": 5_000_160}          # the same prompt as s0
    run.specs.write_text("".join(json.dumps(r) + "\n" for r in [*specs, twin]))
    rows = [_g(s0, 0, "A", "a = 1"), _g(s0, 1, "A", "a = 2"), _g(twin["id"], 0, "A", "a = 1  # again"), _g(twin["id"], 1, "A", "a = 3"),
            _g(s1, 0, "A", "a = 1", api_findings=[{"line": 2, "name": "cv2.AKAZE_create", "api": "AKAZE_create", "label": "removed_in_5",
                                                   "replacement": "cv2.ORB_create"}])]                  # s1: another prompt, same code
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run.stage("generate").completions.write_text("".join(json.dumps({"id": r["id"], "sample": r["sample"], "text": r["code"]}) + "\n" for r in rows))
    s = report_mod.summarize_run(run)
    assert s["totals"]["kept"] == 5 and s["totals"]["kept_distinct"] == 4 and s["selection"]["distinct_prompt_ast"] == 4
    assert s["selection"]["distinct_prompts"] == 4                                                  # 5 specs, but s0 and its twin share a prompt
    ex = report_mod.build_examples(run)
    assert [e["id"] for e in ex] == [f"{s0}#0", f"{s0}#1", f"{s1}#0", f"{twin['id']}#1"]
    assert len({e["provenance"]["prompt_sha1"] for e in ex}) == 2
    assert len(report_mod.build_examples(run, cross_spec_dedup=False)) == 5
    assert s["selection"]["kept_with_api_findings"] == 1                                            # reported, not blocked
    assert [e["provenance"]["api_findings"] for e in ex].count([]) == 3 and "removed" in report_mod.render_markdown(s)


# ================================================================ Gate 4a: thinking mode in the job script

@pytest.mark.parametrize("text, prompt_open, status, thinking, answer", [
    ("reasoning here\n</think>\n\n```python\nx = 1\n```", True, "closed", "reasoning here", "```python\nx = 1\n```"),   # Qwen3.5: opened in the prompt
    ("<think>\nreasoning\n</think>\n\nthe answer", False, "closed", "reasoning", "the answer"),                         # Qwen3: the model opens it
    ("  <think>reasoning</think>answer", False, "closed", "reasoning", "answer"),
    ("</think>\n\nanswer only", True, "empty", "", "answer only"),
    ("<think>\n\n</think>\n\nanswer", False, "empty", "", "answer"),
    ("still thinking about the shapes", True, "truncated", "still thinking about the shapes", ""),                    # never closed
    ("<think>\nstill thinking", False, "truncated", "still thinking", ""),
    ("```python\nx = 1\n```", False, "none", "", "```python\nx = 1\n```"),                                              # no think block at all
    ("a </think> b </think> c", True, "closed", "a", "b </think> c"),                                                  # the first close wins
])
def test_split_thinking(text, prompt_open, status, thinking, answer):
    assert colab_job.split_thinking(text, prompt_open) == {"thinking": thinking, "answer": answer, "status": status}


def _answers_by_sample(replies):
    return lambda text, i, n: replies[i]


def test_job_in_thinking_mode_splits_each_completion_and_counts_both_parts(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 2)
    a = _job_args(tmp_path, prompts, chat_template_kwargs='{"enable_thinking": true}', max_model_len=16384, max_tokens=12288, model="m/t")
    replies = ["think a b\n</think>\n\n```python\ncode\n```",      # words: think a b </think> ```python code ``` -> 4 + 3 tokens
               "still thinking forever and ever",                    # opened in the prompt, never closed: all 5 tokens are reasoning
               "</think>\n\nanswer"]                                  # an empty block
    llm = FakeLLM(_answers_by_sample(replies), tokenizer=FakeIdsTokenizer(), close_id=CLOSE_ID)
    t = colab_job.run(a, llm=llm, make_params=_params, log=lambda m: None)
    rows = {(r["id"], r["sample"]): r for r in colab_job.read_jsonl(tmp_path / "out" / "completions.jsonl")}
    assert bytes(llm.calls[0][0][0]["prompt_token_ids"]).decode().endswith("<|im_start|>assistant\n<think>\n")   # the template opened it
    assert llm.calls[0][1].max_tokens == 12288

    r0, r1, r2 = (rows["s0", i] for i in range(3))
    assert r0["text"] == replies[0] and r0["think_status"] == "closed" and r0["thinking"] == "think a b" and r0["answer"] == "```python\ncode\n```"
    assert (r0["thinking_tokens"], r0["answer_tokens"], r0["completion_tokens"]) == (4, 3, 7)      # the close tag counts as reasoning
    assert r1["think_status"] == "truncated" and r1["answer"] == "" and r1["thinking"] == replies[1]
    assert (r1["thinking_tokens"], r1["answer_tokens"]) == (5, 0)
    assert r2["think_status"] == "empty" and r2["answer"] == "answer" and (r2["thinking_tokens"], r2["answer_tokens"]) == (1, 1)
    for r in rows.values():
        assert r["thinking_tokens"] + r["answer_tokens"] == r["completion_tokens"]
        assert answer_of(r) == r["answer"]                                                         # downstream reads the answer
    assert t["thinking_tokens"] == 2 * (4 + 5 + 1) and t["answer_tokens"] == 2 * (3 + 0 + 1)
    assert t["output_tokens"] == t["thinking_tokens"] + t["answer_tokens"]
    g = t["groups"]["generate"]
    assert (g["n_samples"], g["truncated_thinking"], g["thinking_tokens"]) == (6, 2, 20) and t["chunks"][0]["thinking_tokens"] == 20
    assert t["args"]["chat_template_kwargs"] == {"enable_thinking": True} and t["args"]["max_model_len"] == 16384


def test_thinking_tokens_are_counted_by_re_encoding_when_the_tokenizer_has_no_close_token(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 1)
    a = _job_args(tmp_path, prompts, chat_template_kwargs='{"enable_thinking": true}')
    llm = FakeLLM(lambda text, i, n: "think a b </think> answer text", tokenizer=FakeWordsTokenizer())
    colab_job.run(a, llm=llm, make_params=_params, log=lambda m: None)
    r = colab_job.read_jsonl(tmp_path / "out" / "completions.jsonl")[0]
    assert (r["thinking_tokens"], r["answer_tokens"]) == (4, 2) and r["think_status"] == "closed"
    assert colab_job.think_close_id(FakeWordsTokenizer()) is None and colab_job.think_close_id(FakeIdsTokenizer()) == CLOSE_ID


def test_without_thinking_nothing_is_split_and_a_stray_open_block_is_flagged(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, 1)
    a = _job_args(tmp_path, prompts, chat_template_kwargs='{"enable_thinking": false}')
    llm = FakeLLM(lambda text, i, n: ["```python\nx\n```", "<think>\nit began to think anyway", "plain prose"][i])
    colab_job.run(a, llm=llm, make_params=_params, log=lambda m: None)
    rows = sorted(colab_job.read_jsonl(tmp_path / "out" / "completions.jsonl"), key=lambda r: r["sample"])
    assert [r["think_status"] for r in rows] == ["none", "truncated", "none"]
    assert rows[0]["answer"] == "```python\nx\n```" and rows[1]["answer"] == "" and rows[2]["answer"] == "plain prose"


def test_prompt_rows_can_override_sampling_and_carry_labels(tmp_path):
    rows = [{"id": "p0", "messages": [{"role": "user", "content": "think about it"}], "k": 2, "max_tokens": 100, "temperature": 0.3,
             "top_p": 0.5, "top_k": 7, "chat_template_kwargs": {"enable_thinking": True}, "mode": "think", "origin": "o1", "kind": "generate"},
            {"id": "p1", "messages": [{"role": "user", "content": "plain"}]},
            {"id": "p2", "messages": [{"role": "user", "content": "wordings"}], "kind": "phrasing", "k": 3, "spec_id": "phr-x-00"}]
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    a = colab_job.parse_args(["--prompts", str(path), "--out-dir", str(tmp_path / "out"), "-k", "1"])
    llm = FakeLLM(lambda text, i, n: f"reply {n} {i}")
    t = colab_job.run(a, llm=llm, make_params=_params, log=lambda m: None)

    prompts_sent, params, _ = llm.calls[0]
    assert isinstance(params, list) and len(params) == 3                                    # the prompts differ: one object each
    assert [(p.n, p.max_tokens, p.temperature, p.top_p, getattr(p, "top_k", None)) for p in params] == [
        (2, 100, 0.3, 0.5, 7), (1, 2048, 0.8, 0.95, None), (3, 2048, 0.8, 0.95, None)]       # top_k only when asked for
    texts = [bytes(p["prompt_token_ids"]).decode() for p in prompts_sent]
    assert texts[0].endswith("<think>\n") and not texts[1].endswith("<think>\n")             # per-prompt chat template kwargs
    got = colab_job.read_jsonl(tmp_path / "out" / "completions.jsonl")
    by_id = {}
    for r in got:
        by_id.setdefault(r["id"], []).append(r)
    assert {k: len(v) for k, v in by_id.items()} == {"p0": 2, "p1": 1, "p2": 3}
    assert all((r["mode"], r["origin"], r["kind"]) == ("think", "o1", "generate") for r in by_id["p0"])
    assert not {"mode", "origin", "kind"} & set(by_id["p1"][0]) and by_id["p2"][0]["kind"] == "phrasing" and by_id["p2"][0]["spec_id"] == "phr-x-00"
    assert set(t["groups"]) == {"think", "generate", "phrasing"} and t["groups"]["phrasing"]["n_samples"] == 3   # label: mode, else kind

    again = FakeLLM(lambda text, i, n: "never")                                              # complete: nothing is asked again
    colab_job.run(a, llm=again, make_params=_params, log=lambda m: None)
    assert again.calls == []
    comp = tmp_path / "out" / "completions.jsonl"
    comp.write_text("".join(json.dumps(r) + "\n" for r in got if not (r["id"] == "p0" and r["sample"] == 1)))   # p0 lost a sample
    redo = FakeLLM(lambda text, i, n: "again")
    colab_job.run(a, llm=redo, make_params=_params, log=lambda m: None)
    assert [len(c[0]) for c in redo.calls] == [1] and sum(1 for r in colab_job.read_jsonl(comp) if r["id"] == "p0") == 2   # only p0, k = 2


@pytest.mark.parametrize("override", [{"k": 0}, {"max_tokens": -1}, {"top_p": 1.5}, {"temperature": -0.1}, {"top_k": 1.5},
                                      {"chat_template_kwargs": ["x"]}, {"k": True}])
def test_bad_per_prompt_settings_are_refused(tmp_path, override):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "x"}], **override}) + "\n")
    with pytest.raises(SystemExit):
        colab_job.read_prompts(path)


def test_driver_passes_top_k_and_counts_missing_samples_per_prompt(run14):
    cfg = _cfg(run14, top_k=20, max_tokens=12288, max_model_len=16384, chat_template_kwargs={"enable_thinking": True})
    a = colab_job.parse_args(colab_mod.job_args(cfg))
    assert (a.top_k, a.max_tokens, a.max_model_len, a.chat_template_kwargs) == (20, 12288, 16384, {"enable_thinking": True})
    assert "--top-k" not in colab_mod.job_args(_cfg(run14))                                  # off by default
    st = run14.stage("generate")
    rows = read_jsonl(st.prompts)
    for r in rows[:3]:
        r["k"] = 2
    st.prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    st.completions.write_text("".join(json.dumps({"id": r["id"], "spec_id": r["spec_id"], "sample": s}) + "\n"
                                      for r in rows[:3] for s in range(2)))
    assert colab_mod.missing_prompts(cfg) == len(rows) - 3                                   # three prompts need only their own 2 samples
    assert colab_job.parse_args(colab_mod.job_args(_cfg(run14, top_k=0))).top_k == 0


# ================================================================ Gate 4a: the training answer

CLEAN = "import cv2\nimport numpy as np\n\nK = np.ones((3, 3), np.uint8)\n\n\ndef helper(x):\n    return x\n\n\ndef solve(img):\n    \"\"\"doc\"\"\"\n    return {'v': 1}\n"


@pytest.mark.parametrize("code, expect", [
    (CLEAN, []),
    ('"""module doc"""\n' + CLEAN, []),
    (CLEAN + "\nif __name__ == '__main__':\n    print(solve(None))\n", ["top-level If statement (line 15)"]),
    (CLEAN + "\nprint(solve(None))\n", ["top-level Expr statement (line 15)"]),
    ("import cv2\n\ndef helper():\n    pass\n", ["no top-level solve()"]),
    ("", ["empty"]),
    (CLEAN + "# ```\n", ["contains a code fence"]),
])
def test_answer_problems(code, expect):
    assert answer_problems(code) == expect
    assert answer_problems("def solve(:\n  pass")[0].startswith("syntax error")


def test_answer_format_and_what_a_grade_row_needs_to_become_a_target():
    assert format_answer("import cv2\n\n\ndef solve(img):\n    return {}\n") == "```python\nimport cv2\n\n\ndef solve(img):\n    return {}\n```"
    assert format_answer("x = 1\n\n") == "```python\nx = 1\n```"
    ok = {"code": CLEAN, "finish_reason": "stop", "extract": "fence"}
    assert row_problems(ok) == [] and row_problems({"code": CLEAN}) == []
    assert row_problems({**ok, "finish_reason": "length"}) == ["finish_reason length"]
    assert row_problems({**ok, "extract": "fence-unclosed"}) == ["unclosed code fence"]
    assert answer_of({"text": "<think>x</think>\n\nanswer"}) == "answer"                       # an older row: split from its text
    assert answer_of({"text": "a", "answer": "b"}) == "b"
    assert answer_of({"text": "plain reply"}) == "plain reply"


# ================================================================ Gate 4a: teacher-written phrasings

def test_hint_words_are_the_ones_the_splits_test_uses():
    tree = ast.parse((Path(__file__).with_name("test_splits.py")).read_text(encoding="utf-8"))
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HINT_WORDS" for t in n.targets))
    assert phr.HINT_WORDS == ast.literal_eval(node)


def test_phrasing_thresholds_are_calibrated_on_the_hand_written_phrasings():
    c = phr.calibrate()
    assert (c["recommended_threshold"], c["recommended_run_limit"]) == (phr.DUP_THRESHOLD, phr.RUN_LIMIT) == (0.25, 17)
    assert set(c["per_family"]) == set(trainable_families())
    for name, v in c["per_family"].items():
        assert v["pairs"] == 120 and v["max_jaccard"] < phr.DUP_THRESHOLD and v["max_run"] < phr.RUN_LIMIT, name
    assert c["max_jaccard"] == pytest.approx(0.2333, abs=1e-4) and c["per_family"]["hough_segments"]["max_jaccard_pair"] == [4, 7]
    assert c["max_run"] == 16 and c["per_family"]["feature_homography"]["max_run"] == 16
    # no pair of existing phrasings is flagged, but a reworded copy of one is (one word changed)
    lines = phr.own_lines("contour_area")
    assert all(phr.closeness(a, b) is None for i, a in enumerate(lines) for b in lines[i + 1:])
    copy = phr.make_line("copy", lines[13].text.replace("tell", "show").replace("Can you", "Could you"))
    assert "5-gram Jaccard" in phr.closeness(copy, lines[13])
    long_run = phr.make_line("run", "unrelated words first " + " ".join(lines[13].text.split()[:18]) + " and more unrelated words after")
    assert phr.closeness(long_run, lines[13]) is not None
    assert phr.make_line("short", "a b c d").grams == frozenset() and phr.jaccard(frozenset(), frozenset()) == 0.0


def test_content_checks_pass_every_hand_written_phrasing_and_reject_a_dropped_contract():
    for f in trainable_families():
        ph = importlib.import_module("cvbench.families." + f).PHRASINGS
        bad = [i for i, p in enumerate(ph) if phr.content_problem(f, p)]
        assert bad == ([2] if f == "contour_area" else []), f       # index 2 predates the hole-rule check in tests/test_contour_area.py
        assert not [i for i, p in enumerate(ph) if phr.hint_words_in(phr.core_of(f, p)) or not p.isascii()], f   # the extra hint words are safe
    with pytest.raises(ValueError):
        phr.content_problem("chessboard_corners", "x")
    ph = {f: importlib.import_module("cvbench.families." + f).PHRASINGS[12] for f in trainable_families()}
    fh = importlib.import_module("cvbench.families.feature_homography")
    broken = {
        "contour_area": ph["contour_area"].replace("holes", "gaps"),
        "hough_segments": ph["hough_segments"].replace("one row", "one entry"),
        "perspective_rectify": ph["perspective_rectify"].replace("out_h", "the height"),
        "feature_homography": ph["feature_homography"].replace(fh.ACCURACY, ""),
        "histogram_stats": ph["histogram_stats"].replace("BT.601", "BT601"),
        "components_stats": ph["components_stats"].replace("centroid", "centre"),
        "text_layout": ph["text_layout"].replace("baseline", "descender"),
    }
    for f, text in broken.items():
        assert text != ph[f] and phr.content_problem(f, text), f
    assert phr.content_problem("feature_homography", ph["feature_homography"].replace("H[2, 2]", "H22")) == "must state the H[2, 2] = 1 normalisation"


def test_phrasing_requests_show_the_teacher_the_foundry_phrasings_and_never_the_scored_ones(tmp_path):
    rows = phr.build_requests()
    assert len(rows) == 7 * 8 and {r["family"] for r in rows} == set(trainable_families())
    assert {r["id"] for r in rows} == {f"phr-{f}-{v:02d}" for f in trainable_families() for v in range(8)}
    for r in rows:
        f = r["family"]
        fam = importlib.import_module("cvbench.families." + f)
        user = r["messages"][1]["content"]
        assert [m["role"] for m in r["messages"]] == ["system", "user"] and r["messages"][0]["content"] == phr.SYSTEM
        assert phr.core_of(f, fam.SPEC.task) in user                                         # the thing to reword
        for t in fam.PHRASINGS[12:16]:
            assert phr.core_of(f, t) in user                                                 # the style examples: foundry only
        for t in fam.PHRASINGS[1:12]:
            assert phr.core_of(f, t) not in user                                             # never a dev or test phrasing
        assert "Accuracy requirement" not in user and "tolerance" in user                    # the rules forbid tolerance talk
        assert (r["kind"], r["origin"], r["k"]) == ("phrasing", "phrasings", 2) and r["chat_template_kwargs"] == {"enable_thinking": False}
    fh_user = next(r for r in rows if r["family"] == "feature_homography")["messages"][1]["content"]
    assert "do not write any accuracy statement" in fh_user
    registers = {r["messages"][1]["content"].split("Register for this batch: ")[1].split("\n")[0] for r in rows if r["family"] == "contour_area"}
    assert len(registers) == 8                                                                 # a different register per request
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(colab_job.read_prompts(path)) == 56                                            # the job accepts the file as written
    with pytest.raises(ValueError):
        phr.build_requests(["chessboard_corners"])


def test_parse_candidates_reads_numbered_lists_and_ignores_the_rest():
    reply = ("Here you go:\n\n1. First wording, with (1) inside it.\n2) \"Second wording\"\n3. **Third** wording that wraps\n"
             "   onto a second line.\n\nI hope these help.\n")
    assert phr.parse_candidates(reply) == ["First wording, with (1) inside it.", "Second wording", "Third wording that wraps onto a second line."]
    assert phr.parse_candidates("1. a\n\n3. b") == ["a"]                                      # numbering must run 1, 2, 3 ...
    assert phr.parse_candidates("1. “Quoted” wording — with a dash and a × sign\n2. `Only code quotes`") == [
        '"Quoted" wording - with a dash and a x sign', "Only code quotes"]                      # folded to ASCII; only a whole-item pair is unwrapped
    assert phr.parse_candidates("no list at all") == [] and phr.parse_candidates("") == []
    cut = phr.candidates_from([{"id": "x", "spec_id": "phr-contour_area-03", "kind": "phrasing", "sample": 1,
                                "text": "1. one\n2. two\n3. thr", "finish_reason": "length"},
                               {"id": "y", "spec_id": "phr-text_layout-00", "kind": "phrasing", "sample": 0, "text": "1. other", "finish_reason": "stop"},
                               {"id": "z", "spec_id": "s", "kind": "generate", "sample": 0, "text": "1. code"}], "contour_area")
    assert [c["core"] for c in cut[0]] == ["one", "two"] and cut[1]["cut_off_replies"] == 1      # the cut-off last item is dropped
    assert (cut[0][0]["variant"], cut[0][0]["sample"]) == (3, 1)


GOOD = {
    "contour_area": [
        "A camera over a conveyor belt sees bright parts on a dark belt. Tell me how many parts are in view and, for each part, the "
        "area it covers inside its outer edge, with any hole counted as covered. List the areas from largest to smallest.",
        "Report the number of separate light objects against the black background, together with the area bounded by the outermost "
        "contour of each one (a hole does not reduce the area), sorted biggest first.",
        "I need two results for this image of bright blobs on a dark ground: the count of blobs, and each blob's area measured inside "
        "its outer boundary, so that interior holes still count as area.",
    ],
    "perspective_rectify": [
        "A photo of a rectangular card on a dark table is given. Return the card's four corners as seen in the photo and a rectified, "
        "front-on out_w x out_h picture of the card.",
        "Detect where the four corners of the bright card lie in the image, then flatten the card into an out_w by out_h picture and "
        "return the corners together with that picture.",
        "Straighten the card seen in this snapshot: give me its four corner positions and the upright out_w x out_h image obtained by "
        "undoing the perspective.",
    ],
    "feature_homography": [
        "Two views of one flat, textured surface are given as img_a and img_b. Work out the 3x3 matrix H, scaled so that H[2, 2] = 1, "
        "that carries a pixel (x, y) of img_a onto the same surface point in img_b, using local feature correspondences.",
        "Return the 3x3 transform H (normalised to H[2, 2] = 1) relating the flat textured scene in img_a to its appearance in img_b, so "
        "that H applied to a pixel of img_a gives the matching pixel of img_b; base it on matched local features.",
        "Given a reference image img_a and a second view img_b of the same textured plane, estimate H, the 3x3 homography whose "
        "bottom-right entry is 1, mapping img_a pixel coordinates to img_b pixel coordinates by way of matched local features.",
    ],
}


def _cand(text, pos=0, variant=0, sample=0):
    return {"core": text, "request": "r", "variant": variant, "sample": sample, "position": pos}


def test_screen_keeps_good_wordings_and_gives_each_rejection_its_reason():
    fam = "contour_area"
    ph = importlib.import_module("cvbench.families.contour_area").PHRASINGS
    good = GOOD[fam]
    cands = [_cand(good[0]),
             _cand(ph[12].replace("Measure", "Gauge")),                                    # a reworded foundry phrasing
             _cand(ph[5]),                                                                 # a test phrasing, verbatim
             _cand(good[1] + " Use cv2.contourArea."),                                     # an API name the task does not give
             _cand(good[1] + " Mind the tolerance."),                                      # a hint word
             _cand(good[2].replace("interior holes", "interior gaps")),                    # drops the hole rule
             _cand(good[0]),                                                               # a copy of the first candidate
             _cand("Areas of bright shapes, please."),                                     # far too short
             _cand(good[1] + " Apply Otsu first."),                                        # an algorithm word
             _cand(good[1].replace("Report", "Réport")),                              # not ASCII
             _cand(good[1]), _cand(good[2])]
    new, rejected, examples = phr.screen_family(fam, cands)
    assert [c["text"] for c in new] == [good[0], good[1], good[2]]
    assert dict(rejected) == {"near-duplicate of the family's phrasings": 1, "overlaps a scored task": 1, "adds cv2 names": 1,
                              "hint words": 2, "content": 1, "near-duplicate of a kept candidate": 1, "length out of range": 1,
                              "non-ASCII characters": 1}
    assert "cv2.contourArea" in examples["adds cv2 names"][0] and "5-gram Jaccard" in examples["overlaps a scored task"][0]
    # the quota counts the phrasings already kept, and the kept ones are never picked again
    new2, rej2, _ = phr.screen_family(fam, cands, existing=[good[0]], max_keep=2)
    assert [c["text"] for c in new2] == [good[1]] and rej2["quota"] >= 1
    assert phr.screen_family(fam, [_cand(good[0])], existing=[good[0]])[0] == []              # kept already: a near-duplicate of itself


def test_the_screen_puts_the_shared_accuracy_sentence_back():
    fh = importlib.import_module("cvbench.families.feature_homography")
    new, rejected, _ = phr.screen_family("feature_homography", [_cand(GOOD["feature_homography"][0]),
                                                                _cand(GOOD["feature_homography"][1] + " " + fh.ACCURACY)])
    assert len(new) == 2 and not rejected
    assert all(c["text"].endswith(fh.ACCURACY) and c["text"].count(fh.ACCURACY) == 1 for c in new)   # added once, whether or not it was sent
    assert not phr.hard_problems("feature_homography", new[0]["text"])


def _good_teacher(prompt_text, spec_prompts=None):
    for fam, texts in GOOD.items():
        task = phr.core_of(fam, importlib.import_module("cvbench.families." + fam).SPEC.task)
        if "Original task statement:\n" + task in prompt_text:
            return "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    return None


REFS = {f: importlib.import_module("cvbench.families." + f) for f in trainable_families()}
FAMILY_WRONG = {"contour_area": ca.MUTANTS["fixed_threshold_127"][1], "histogram_stats": hs.MUTANTS["bins_255"][1]}


def _session_answer(by_prompt):
    """The fake teacher of the combined session: phrasing requests get GOOD wordings, solution requests get code, in the
    mode the prompt asked for. think mode: contour_area sample 1 is wrong; hough sample 0 never finishes thinking; histogram sample 0
    thinks the right code but answers a wrong one and sample 1 the reverse; everything else answers the reference."""
    def answer(text, i, n):
        said = _good_teacher(text)
        if said is not None:
            return said
        spec_id = next(sid for p, sid in by_prompt.items() if p in text)
        fam = spec_id.split("-")[1]
        correct, wrong = REFS[fam].REFERENCE, FAMILY_WRONG.get(fam)
        if text.endswith("<think>\n"):
            if fam == "contour_area" and i == 1:
                return "reasoning\n</think>\n\n" + fence(wrong)
            if fam == "hough_segments" and i == 0:
                return "I keep going and going without closing the block."
            if fam == "histogram_stats":
                swapped = (correct, wrong) if i == 0 else (wrong, correct)   # (what the reasoning drafts, what the answer says)
                return f"Draft:\n```python\n{swapped[0]}```\n</think>\n\n" + fence(swapped[1])
            return "reasoning about the task\n</think>\n\n" + fence(correct)
        return fence(correct)
    return answer


@pytest.fixture
def phrasing_dir(tmp_path, monkeypatch):
    d = tmp_path / "foundry_phrasings"
    monkeypatch.setattr(phr, "PHRASINGS_DIR", d)
    return d


def test_combined_session_phrasings_and_thinking_test_end_to_end(tmp_path, phrasing_dir, capsys):
    root = ["--root", str(tmp_path)]
    fams = "contour_area,perspective_rectify,feature_homography"
    assert main(["prepare-phrasings", "--run", "s1", *root, "--families", fams, "--variants", "3", "--per-request", "3", "-k", "2"]) == 0
    assert main(["prepare-think-test", "--run", "s1", *root, "--per-family", "1", "--drills", "0", "-k", "2"]) == 0
    run = FoundryRun.of("s1", tmp_path)
    st = run.stage("generate")
    rows = read_jsonl(st.prompts)
    assert [r["origin"] for r in rows] == ["phrasings"] * 9 + ["think-test"] * 14                # blocks by origin
    assert len({r["id"] for r in rows}) == 23 and len(read_jsonl(run.specs)) == 7
    think = [r for r in rows if r.get("mode") == "think"]
    nothink = [r for r in rows if r.get("mode") == "nothink"]
    assert len(think) == len(nothink) == 7
    assert think[0]["id"].endswith("@think") and think[0]["spec_id"] == nothink[0]["spec_id"] and think[0]["k"] == 2
    assert (think[0]["chat_template_kwargs"], think[0]["max_tokens"], think[0]["temperature"], think[0]["top_p"], think[0]["top_k"]) == (
        {"enable_thinking": True}, 12288, 0.6, 0.95, 20)
    assert (nothink[0]["chat_template_kwargs"], nothink[0]["max_tokens"], nothink[0]["temperature"], nothink[0]["top_p"]) == (
        {"enable_thinking": False}, 2048, 0.7, 0.8)                                            # the pilot's settings
    cfg = read_json(run.config)
    assert cfg["phrasings"]["requests"] == 9 and cfg["think_test"]["n_specs"] == 7
    assert main(["prepare-think-test", "--run", "s1", *root, "--per-family", "1", "--drills", "0", "-k", "2"]) == 0   # idempotent
    assert read_jsonl(st.prompts) == rows

    # ---- the job: one run of colab_job over the merged file
    specs = think_test_specs(1, 0, 2)
    by_prompt = {s.prompt: s.id for s in specs}
    a = colab_job.parse_args(["--prompts", str(st.prompts), "--out-dir", str(run.path), "-k", "4", "--model", "fake/teacher", "--chunk", "10",
                              "--max-model-len", "16384", "--chat-template-kwargs", '{"enable_thinking": false}'])
    llm = FakeLLM(_session_answer(by_prompt), tokenizer=FakeIdsTokenizer(), close_id=CLOSE_ID)
    timing = colab_job.run(a, llm=llm, make_params=_params, log=lambda m: None)
    comps = read_jsonl(st.completions)
    assert len(comps) == 9 * 2 + 14 * 2 and {c["kind"] for c in comps} == {"phrasing", "generate"}
    assert set(timing["groups"]) == {"phrasing", "nothink", "think"} and timing["groups"]["think"]["truncated_thinking"] == 1
    assert len(llm.calls) == 3 and [isinstance(c[1], list) for c in llm.calls] == [True, True, False]   # chunks that mix settings: one params object per prompt

    # ---- verify: only solutions are graded, only their answers
    assert main(["verify", "--run", "s1", *root, "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "Thinking vs non-thinking" in out and "| **all** |" in out
    grades = read_jsonl(st.grades)
    assert len(grades) == 28 and all(not g["id"].startswith("phr-") for g in grades)
    tiers = {(g["spec_id"].split("-")[1], g["mode"], g["sample"]): g["tier"] for g in grades}
    assert all(t == "A" for (f, m, s), t in tiers.items() if m == "nothink")
    assert {k: v for k, v in tiers.items() if k[1] == "think" and v != "A"} == {
        ("contour_area", "think", 1): "fail", ("hough_segments", "think", 0): "fail", ("histogram_stats", "think", 0): "fail"}
    by = {(g["spec_id"].split("-")[1], g["mode"], g["sample"]): g for g in grades}
    h0, h1 = by["histogram_stats", "think", 0], by["histogram_stats", "think", 1]
    assert h0["tier"] == "fail" and h1["tier"] == "A"                                           # the answer decides, not the reasoning
    assert h1["code"].strip() == hs.REFERENCE.strip() and h1["think_status"] == "closed" and h1["thinking_tokens"] > 0
    hough = by["hough_segments", "think", 0]
    assert hough["think_status"] == "truncated" and hough["code_empty"] is True
    assert hough["failure"]["rule"] == "thinking-truncated" and hough["failure"]["repairable"] is False
    assert all(g["thinking_tokens"] == 0 for g in grades if g["mode"] == "nothink")

    # ---- the comparison and the report
    table = verify_mod.mode_table(grades)
    assert table["modes"] == ["nothink", "think"]
    assert (table["all"]["nothink"]["tier_A"], table["all"]["think"]["tier_A"]) == (14, 11)
    assert table["all"]["think"]["truncated_thinking"] == 1 and table["all"]["nothink"]["truncated_thinking"] == 0
    assert table["paired_all"]["specs"] == 7 and table["paired"]["contour_area"]["think"]["tier_A_pct"] == 50.0
    lo, hi = table["all"]["think"]["ci"]
    assert lo < 100 * 11 / 14 < hi and table["all"]["think"]["tokens_per_A"] > 0
    assert verify_mod.mode_table([g for g in grades if g["mode"] == "think"]) is None           # one mode only: no comparison
    assert main(["report", "--run", "s1", *root, "--gpu-rate", "5.4", "--answer-format", "fenced"]) == 0
    s = read_json(run.summary)
    assert s["modes"]["all"]["think"]["tier_A"] == 11 and s["totals"]["truncated_thinking"] == 1
    assert "## Thinking vs non-thinking" in run.report_md.read_text()
    ex = read_jsonl(run.examples)
    assert ex and all(e["messages"][2]["content"] == format_answer(e["code"]) for e in ex)
    assert not any("</think>" in e["raw_reply"] or "Draft:" in e["raw_reply"] for e in ex)         # the reasoning never reaches an example

    # ---- the phrasings: screened into foundry/phrasings/<family>.json with provenance
    run.stage("generate").session.write_text(json.dumps({"model": "fake/teacher", "revision": "rev123", "teacher_license": "apache-2.0",
                                                         "quantization": None, "gpu": "A100"}))
    assert main(["screen-phrasings", "--run", "s1", *root]) == 0
    assert sorted(p.name for p in phrasing_dir.iterdir()) == ["contour_area.json", "feature_homography.json", "perspective_rectify.json"]
    rec = json.loads((phrasing_dir / "contour_area.json").read_text())
    assert [p["text"] for p in rec["phrasings"]] == GOOD["contour_area"]
    assert rec["teacher"] == {"model": "fake/teacher", "license": "apache-2.0", "revision": "rev123", "quantization": None, "gpu": "A100"}
    assert rec["family"] == "contour_area" and rec["run"] == "s1" and re.fullmatch(r"\d{4}-\d\d-\d\d", rec["created"])
    assert rec["screen"]["dup_threshold"] == 0.25 and rec["screen"]["run_limit"] == 17 and rec["screen"]["replies"] == 6
    assert rec["screen"]["candidates"] == 18 and rec["screen"]["new"] == 3 and rec["screen"]["rejected"] == {"near-duplicate of a kept candidate": 15}
    fh = importlib.import_module("cvbench.families.feature_homography")
    rec_fh = json.loads((phrasing_dir / "feature_homography.json").read_text())
    assert all(p["text"].endswith(fh.ACCURACY) for p in rec_fh["phrasings"]) and rec_fh["phrasings"][0]["sha1"] == sha1(rec_fh["phrasings"][0]["text"])
    before = (phrasing_dir / "contour_area.json").read_bytes()
    assert main(["screen-phrasings", "--run", "s1", *root, "--dry-run"]) == 0
    assert (phrasing_dir / "contour_area.json").read_bytes() == before                           # a dry run writes nothing
    assert main(["screen-phrasings", "--run", "s1", *root]) == 0                                  # again: the kept ones stay, indices are stable
    assert [p["text"] for p in json.loads((phrasing_dir / "contour_area.json").read_text())["phrasings"]] == GOOD["contour_area"]

    # ---- specs use them as phrasing indices 16, 17, ...
    assert phr.generated_phrasings("contour_area") == GOOD["contour_area"] and phr.generated_phrasings("text_layout") == []
    full = full_specs(100, include_drills=False)
    by_fam = {f: [s for s in full if s.family == f] for f in trainable_families()}
    assert {s.phrasing_index for s in by_fam["contour_area"]} == {12, 13, 14, 15, 16, 17, 18}
    assert {s.phrasing_index for s in by_fam["text_layout"]} == {12, 13, 14, 15}                 # no file: foundry phrasings only
    assert {s.phrasing_index for s in full_specs(100, include_drills=False, use_generated=False) if s.family == "contour_area"} == {12, 13, 14, 15}
    s16 = next(s for s in by_fam["contour_area"] if s.phrasing_index == 16)
    assert s16.prompt.startswith("Task: " + GOOD["contour_area"][0]) and s16.prompt_sha1 not in reserved_prompt_hashes()
    check_specs(full)
    # a generated phrasing that is a scored one is refused by the invariants, whatever file it came from
    rec["phrasings"][0]["text"] = ph5 = importlib.import_module("cvbench.families.contour_area").PHRASINGS[5]
    (phrasing_dir / "contour_area.json").write_text(json.dumps(rec))
    with pytest.raises(SpecError, match="fails the screen"):
        full_specs(100, include_drills=False)
    assert phr.hard_problems("contour_area", ph5)[0].startswith("overlaps a scored task")


# ================================================================ Gate 4a: the full-generation specs and the prompts file

def test_sizing_follows_the_target_the_rates_and_the_weights():
    plan = sizing(2100, include_drills=False)
    assert set(plan) == set(trainable_families())
    for g, p in plan.items():
        assert p["kept_target"] == pytest.approx(300) and p["specs"] == math.ceil(300 / specs_mod.DEFAULT_RATES[g] - 1e-9), g
    assert plan["contour_area"]["specs"] == 150 and plan["hough_segments"]["specs"] == 231 and plan["text_layout"]["specs"] == 375
    assert specs_mod.PILOT_KEPT_PER_SPEC["contour_area"] == 2.0 and specs_mod.PILOT_KEPT_PER_SPEC["text_layout"] == 0.54   # the pilot's rates
    with_drills = sizing(2100, drill_weight=3.0)
    assert with_drills["drills"]["kept_target"] == pytest.approx(630) and with_drills["contour_area"]["kept_target"] == pytest.approx(210)
    assert with_drills["drills"]["specs"] == math.ceil(630 / 1.5)
    rates = sizing(2100, include_drills=False, rates={"contour_area": 1.0})                  # the rates are an argument
    assert rates["contour_area"]["specs"] == 300 and rates["hough_segments"] == plan["hough_segments"]
    weighted = sizing(2100, include_drills=False, weights={"text_layout": 0.5, "contour_area": 2.0})
    assert weighted["text_layout"]["kept_target"] < plan["text_layout"]["kept_target"] < weighted["contour_area"]["kept_target"]
    assert sum(p["kept_target"] for p in weighted.values()) == pytest.approx(2100)
    with pytest.raises(ValueError, match="positive"):
        sizing(10, rates={"contour_area": 0})
    with pytest.raises(ValueError, match="unknown group"):
        sizing(10, weights={"chessboard_corners": 1})
    with pytest.raises(ValueError):
        sizing(0)


def test_full_specs_k_ids_seeds_prefix_stability_and_invariants(phrasing_dir):
    small, big = full_specs(210, include_drills=False), full_specs(420, include_drills=False)
    check_specs(big)
    assert {s.family for s in big} == set(trainable_families())
    assert all(s.id == f"full-{s.family}-{s.index:04d}" and s.kind == "family" for s in big)
    for s in big:
        assert s.k == (8 if s.family in ("hough_segments", "text_layout") else 4), s.id        # twice the samples where the teacher fails
        assert s.phrasing_index in FOUNDRY_PHRASINGS                                          # no teacher-written file in this checkout
        assert FOUNDRY_RANGE[0] <= s.seed and s.window[1] <= FOUNDRY_RANGE[1]
    for f in trainable_families():                                                            # spec j never changes with the target
        a, b = [s for s in small if s.family == f], [s for s in big if s.family == f]
        assert a == b[:len(a)] and len(b) > len(a)
    assert full_specs(210, include_drills=False, k=3)[0].k == 3 and full_specs(210, include_drills=False, k={"contour_area": 5})[0].k == 5
    assert len(full_specs(210, include_drills=False, rates={"contour_area": 1.0})) > len(small)
    rows = summarize_specs_rows(big)
    assert {r["group"] for r in rows} == set(trainable_families()) and all(r["prompts"] == 4 for r in rows)
    assert sum(r["samples"] for r in rows) == sum(s.k for s in big)
    # not trainable, a scored prompt, an overlapping window ...: refused, as for the pilot
    s = big[0]
    with pytest.raises(SpecError, match="held-out"):
        check_specs([dataclasses.replace(s, family="color_segmentation", id="x")])
    with pytest.raises(SpecError, match="unknown spec kind"):
        check_specs([dataclasses.replace(s, kind="other")])


def summarize_specs_rows(specs):
    return specs_mod.summarize_specs(specs)


def _fake_drills(monkeypatch, phrasings=None):
    """A stand-in for foundry.drills (written by another agent): two drills built on real gotcha checks, with their own task lines."""
    from cvbench.gotchas import CHECKS
    base = {"fake_hist": CHECKS["shape_calchist"], "fake_nonzero": CHECKS["shape_findnonzero"]}
    texts = phrasings or {
        "fake_hist": ["Give back how often each of the 256 gray values occurs in this one-channel picture, and the value that is most common.",
                      "For a single-channel image, count the pixels at every one of the 256 intensity levels and name the commonest level.",
                      "Which intensity is most frequent in this grayscale frame? Return all 256 per-level pixel counts as well as that level."],
        "fake_nonzero": ["Return the (x, y) position of every lit pixel in the mask.",
                         "Which pixels of this binary mask are switched on? List their column and row.",
                         "Collect the coordinates of the foreground pixels, one row per pixel."]}
    drills = {name: dataclasses.replace(c, id=name, spec=dataclasses.replace(c.spec, task=texts[name][0])) for name, c in base.items()}
    mod = SimpleNamespace(DRILLS=drills, DRILL_PHRASINGS={n: list(t) for n, t in texts.items()})
    monkeypatch.setitem(sys.modules, "foundry.drills", mod)
    return mod, base


def test_drill_specs_in_the_full_set_and_their_invariants(monkeypatch, phrasing_dir):
    mod, base = _fake_drills(monkeypatch)
    specs = full_specs(200)                                                                    # 60 kept / 1.5 = 40 drill specs, 20 per drill
    drills = [s for s in specs if s.kind == "drill"]
    assert len(drills) == 40 and {s.drill for s in drills} == {"fake_hist", "fake_nonzero"}
    assert all(s.family == "drills" and s.seed == 0 and s.k == 4 and s.subset == "drill" for s in drills)
    d0 = [s for s in drills if s.drill == "fake_hist"]
    assert [s.phrasing_index for s in d0[:6]] == [0, 1, 2, 0, 1, 2] and d0[0].id == "full-fake_hist-000"
    assert d0[1].prompt == dataclasses.replace(base["fake_hist"].spec, task=mod.DRILL_PHRASINGS["fake_hist"][1]).to_prompt()
    assert len({s.prompt_sha1 for s in drills}) == 6 and d0[0].window == (0, 0)
    assert specs_mod.drill_names() == ["fake_hist", "fake_nonzero"]
    assert Counter_of(summarize_specs_rows(specs))["drills"] == (40, 6)
    tt = think_test_specs(per_family=2, drills=1, k=3)                                          # spread over the sorted drill names
    assert [s.drill for s in tt if s.kind == "drill"] == ["fake_hist"] and len([s for s in tt if s.kind == "family"]) == 14
    assert all(s.id.startswith("tt-") and s.k == 3 for s in tt) and all(s.phrasing_index in FOUNDRY_PHRASINGS for s in tt if s.kind == "family")
    # the drill invariants
    s = d0[0]
    with pytest.raises(SpecError, match="unknown drill"):
        check_specs([dataclasses.replace(s, drill="nope")])
    with pytest.raises(SpecError, match="not TaskSpec.to_prompt"):
        check_specs([dataclasses.replace(s, prompt=s.prompt + " ")])
    with pytest.raises(SpecError, match="phrasing 7 outside"):
        check_specs([dataclasses.replace(s, phrasing_index=7)])
    gotcha = base["fake_hist"]
    mod.DRILL_PHRASINGS["fake_hist"][1] = gotcha.spec.task                                       # a drill that IS a gotcha prompt
    with pytest.raises(SpecError, match="is the prompt of scored task shape_calchist"):
        check_specs([dataclasses.replace(s, phrasing_index=1, prompt=specs_mod._render_drill("fake_hist", 1))])
    mod.DRILL_PHRASINGS["fake_hist"][1] = gotcha.spec.task.replace("most often", "most frequently")   # a near copy of one
    with pytest.raises(SpecError, match="overlaps a scored task"):
        check_specs([dataclasses.replace(s, phrasing_index=1, prompt=specs_mod._render_drill("fake_hist", 1))])
    with pytest.raises(SpecError, match="foundry.drills"):
        monkeypatch.setitem(sys.modules, "foundry.drills", None)                                 # import of the missing module
        full_specs(50)


def Counter_of(rows):
    return {r["group"]: (r["specs"], r["prompts"]) for r in rows}


def test_prepare_full_writes_prompts_with_their_k_and_the_chosen_thinking_families(tmp_path, capsys, phrasing_dir):
    root = ["--root", str(tmp_path)]
    assert main(["specs", "--full", "--target-kept", "70", "--no-drills"]) == 0
    assert "hough_segments" in capsys.readouterr().out
    assert main(["prepare-full", "--run", "f", *root, "--target-kept", "70", "--no-drills", "--think-families", "hough_segments,text_layout"]) == 0
    run = FoundryRun.of("f", tmp_path)
    rows, specs = read_jsonl(run.stage("generate").prompts), read_jsonl(run.specs)
    assert [r["id"] for r in rows] == [s["id"] for s in specs] and {r["origin"] for r in rows} == {"full"}
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    for f, rs in by_fam.items():
        if f in ("hough_segments", "text_layout"):
            assert all(r["mode"] == "think" and r["k"] == 8 and r["chat_template_kwargs"] == {"enable_thinking": True}
                       and r["max_tokens"] == 12288 and r["top_k"] == 20 for r in rs), f
        else:
            assert all(r["mode"] == "nothink" and r["k"] == 4 and "max_tokens" not in r for r in rs), f   # the job's own settings apply
    assert len(colab_job.read_prompts(run.stage("generate").prompts)) == len(rows)
    cfg = read_json(run.config)["full"]
    assert cfg["think_families"] == ["hough_segments", "text_layout"] and cfg["n_specs"] == len(specs) and cfg["plan"]["contour_area"]["specs"] == 5
    before = run.stage("generate").prompts.read_bytes()
    assert main(["prepare-full", "--run", "f", *root, "--target-kept", "70", "--no-drills", "--think-families", "hough_segments,text_layout"]) == 0
    assert run.stage("generate").prompts.read_bytes() == before                                     # idempotent
    run.stage("generate").completions.write_text("{}\n")
    with pytest.raises(SystemExit, match="different 'full' prompts"):
        main(["prepare-full", "--run", "f", *root, "--target-kept", "140", "--no-drills"])         # a run with output must not change
    with pytest.raises(SystemExit, match="no specs"):
        main(["prepare-full", "--run", "g", *root, "--target-kept", "70", "--no-drills", "--think-families", "color_segmentation"])
    assert main(["prepare-full", "--run", "h", *root, "--no-drills", "--no-generated", "--rates", '{"contour_area": 1.0}', "--weights",
                 '{"text_layout": 0.5}', "-k", "2", "--target-kept", "70"]) == 0
    h = read_jsonl(FoundryRun.of("h", tmp_path).stage("generate").prompts)
    assert {r["k"] for r in h} == {2} and not any("mode" in r for r in h)                           # no thinking families: no mode labels


def test_session_merge_replaces_only_its_own_origin(tmp_path):
    run = FoundryRun.of("m", tmp_path)
    run.path.mkdir()
    spec_a = think_test_specs(1, 0, 1)[0]
    row = lambda i, origin: {"id": i, "spec_id": i, "origin": origin, "kind": "generate", "messages": [{"role": "user", "content": i}]}  # noqa: E731
    assert session_mod.merge_prompts(run, "full", [row("f1", "full"), row("f2", "full")]) == 2
    assert session_mod.merge_prompts(run, "phrasings", [row("p1", "phrasings")]) == 3
    assert [r["id"] for r in read_jsonl(run.stage("generate").prompts)] == ["p1", "f1", "f2"]       # blocks in a fixed order
    assert session_mod.merge_prompts(run, "full", [row("f3", "full")]) == 2                          # replaced, the other origin kept
    assert [r["id"] for r in read_jsonl(run.stage("generate").prompts)] == ["p1", "f3"]
    with pytest.raises(SystemExit, match="already used"):
        session_mod.merge_prompts(run, "think-test", [row("p1", "think-test")])
    with pytest.raises(ValueError):
        session_mod.merge_prompts(run, "bogus", [])
    run.stage("generate").grades.write_text("{}\n")
    with pytest.raises(SystemExit, match="already has completions or grades"):
        session_mod.merge_prompts(run, "full", [row("f9", "full")])
    assert session_mod.merge_prompts(run, "full", [row("f9", "full")], force=True) == 2
    assert session_mod.merge_specs(run, [spec_a]) == 1 and session_mod.merge_specs(run, [spec_a]) == 1
    with pytest.raises(SystemExit, match="different content"):
        session_mod.merge_specs(run, [dataclasses.replace(spec_a, seed=spec_a.seed + 1)])
    assert session_mod.label_of({"mode": "think", "kind": "generate"}) == "think" and session_mod.label_of({"kind": "phrasing"}) == "phrasing"
    assert session_mod.label_of({}) == "generate"


def test_predict_mix_prices_each_label_at_its_own_tokens_and_speed(tmp_path, capsys):
    rows = [{"id": "a", "mode": "think", "k": 8}, {"id": "b", "mode": "think", "k": 8}, {"id": "c", "mode": "nothink"},
            {"id": "d", "kind": "phrasing", "k": 2}]
    text = predict_mix(rows, 4, {"think": 5000, "nothink": 800, "phrasing": 450}, {"think": 300}, None, 800, 600, 600, 5.4, 0.1)
    assert re.search(r"think\s+2 prompts,\s+16 samples x\s+5000 tokens =\s+80000 output tokens; at 300 tok/s\s+4\.4 min", text)
    assert re.search(r"nothink\s+1 prompts,\s+4 samples x\s+800 tokens =\s+3200 output tokens; at 800 tok/s\s+0\.1 min", text)
    assert "total 84100 output tokens" in text and "generation 4.5 min" in text and "model load 10.0 min" in text
    # (266.7 + 4.0 + 1.1) s of decode + 600 s load + 600 s setup = 1471.8 s = 24.5 min -> 2.21 CU at 5.4 CU/h
    assert "predicted 24.5 min at 5.4 CU/h = 2.21 CU" in text and "kill at 2x = 49 min" in text
    with pytest.raises(SystemExit, match="no mean output tokens"):
        predict_mix(rows, 4, {"think": 1}, {}, None, 800, 600, 600, 5.4, 0.1)
    assert main(["prepare-think-test", "--run", "p", "--root", str(tmp_path), "--per-family", "1", "--drills", "0", "-k", "2"]) == 0
    assert main(["prepare-phrasings", "--run", "p", "--root", str(tmp_path), "--families", "contour_area", "--variants", "2"]) == 0
    assert main(["predict", "--run", "p", "--root", str(tmp_path), "--tok-s", "600", "--tokens-by-label",
                 '{"phrasing": 450, "nothink": 800, "think": 4500}', "--tok-s-by-label", '{"think": 300}']) == 0
    out = capsys.readouterr().out
    assert "phrasing" in out and "nothink" in out and "think" in out and "LOGBOOK line" in out
    with pytest.raises(SystemExit, match="no mean output tokens"):
        main(["predict", "--run", "p", "--root", str(tmp_path), "--tok-s", "600"])


# ================================================================ Gate 4a: drills are graded by gotchas.core.grade

class ThreadPool(ThreadPoolExecutor):
    """Stands in for the spawn-based process pool so a fake foundry.drills (in this process only) can be imported by the workers."""

    def __init__(self, max_workers=None, mp_context=None):
        super().__init__(max_workers=max_workers)


def _drill_run(tmp_path, monkeypatch, name="drillrun"):
    mod, base = _fake_drills(monkeypatch)
    spec = specs_mod._drill_spec("fake_hist", 0, 0, prefix="t", system=teacher_system(), k=4)
    run = FoundryRun.of(name, tmp_path)
    run.path.mkdir(parents=True)
    run.specs.write_text(json.dumps(spec.to_row()) + "\n")
    run.stage("generate").prompts.write_text("".join(json.dumps(r) + "\n" for r in prompt_rows([spec])))
    wrong = base["fake_hist"].wrong["legacy_bin_index"].code
    comps = [{"id": spec.id, "spec_id": spec.id, "sample": 0, "text": fence(base["fake_hist"].reference), "finish_reason": "stop", "completion_tokens": 50},
             {"id": spec.id, "spec_id": spec.id, "sample": 1, "text": fence(wrong), "finish_reason": "stop", "completion_tokens": 40},
             {"id": spec.id, "spec_id": spec.id, "sample": 2, "text": "I cannot help with that.", "finish_reason": "stop", "completion_tokens": 5},
             {"id": spec.id, "spec_id": spec.id, "sample": 3, "text": "thinking...", "thinking": "thinking...", "answer": "",
              "think_status": "truncated", "thinking_tokens": 9, "answer_tokens": 0, "finish_reason": "length", "completion_tokens": 9}]
    run.stage("generate").completions.write_text("".join(json.dumps(c) + "\n" for c in comps))
    monkeypatch.setattr(verify_mod, "ProcessPoolExecutor", ThreadPool)
    return run, spec, base


def test_drill_tier_a_is_a_passed_grade(tmp_path, monkeypatch):
    run, spec, base = _drill_run(tmp_path, monkeypatch)
    counts = verify_mod.grade_stage(run, "generate", verify_mod.VerifySettings(workers=2), log=lambda m: None)
    assert counts["A"] == 1 and counts["fail"] == 3 and counts["B"] == 0 and counts["grader_errors"] == 0
    g = {r["sample"]: r for r in read_jsonl(run.stage("generate").grades)}
    assert [g[i]["tier"] for i in range(4)] == ["A", "fail", "fail", "fail"] and all(g[i]["kind"] == "drill" for i in g)
    assert g[0]["passed"] is True and g[0]["stage"] == "ok" and "gates" not in g[0] and g[0]["failure"] is None
    f1 = g[1]["failure"]                                                                           # the grader's own words, no "item 0:" prefix
    assert f1["repairable"] is True and f1["gate"] in ("run", "verify") and not f1["message"].startswith("item") and f1["class"] == "convention"
    assert g[2]["failure"]["repairable"] is False and g[2]["failure"]["rule"] == "no-code"
    assert g[3]["code_empty"] is True and g[3]["failure"]["rule"] == "thinking-truncated" and g[3]["think_status"] == "truncated"
    s = report_mod.summarize_run(run)
    assert s["families"] == ["drills"] and s["rows"]["drills"]["subset"] == "drill" and s["totals"]["tier_A"] == 1
    assert s["totals"]["truncated_thinking"] == 1 and "| drills | drill |" in report_mod.render_markdown(s)
    # pairing and the dataset take the drill like any other spec
    pairs = repair_mod.build_pairing_pairs(run)
    assert len(pairs) == 1 and pairs[0]["failed_sample"] == 1 and pairs[0]["failure"]["class"] == "convention"
    assert pairs[0]["fixed_code"].strip() == base["fake_hist"].reference.strip()
    sol, rep, st = ds.collect_run(run)
    assert (len(sol), len(rep)) == (1, 1) and sol[0].meta["drill"] == "fake_hist" and sol[0].meta["scene_seed"] is None
    assert sol[0].meta["grades"] == "drill-grade:ok" and sol[0].meta["family"] == "drills"
    assert verify_mod.tier_of({"kind": "drill", "passed": False}) == "fail" and verify_mod.tier_of({"kind": "drill", "passed": True}) == "A"


# ================================================================ Gate 4a: repair examples by pairing

def good_code(n):
    return f"import cv2\nimport numpy as np\n\n\ndef solve(img):\n    return {{'v': {n}}}\n"


def _pairing_run(tmp_path, name="pairrun"):
    run = FoundryRun.of(name, tmp_path)
    run.path.mkdir(parents=True)
    specs = [{"id": f"foundry-contour_area-{j:04d}", "family": "contour_area", "index": j, "phrasing_index": 12, "seed": 5_000_000 + 40 * j,
              "prompt": f"PROMPT {j}", "subset": "timeless"} for j in range(5)]
    specs.append({"id": "foundry-text_layout-0000", "family": "text_layout", "index": 0, "phrasing_index": 13, "seed": 5_800_000,
                  "prompt": "PROMPT T", "subset": "5.0"})
    run.specs.write_text("".join(json.dumps(r) + "\n" for r in specs))
    return run, [s["id"] for s in specs]


def test_pairing_builds_one_repair_example_per_spec_preferring_convention_and_api(tmp_path):
    run, ids = _pairing_run(tmp_path)
    s0, s1, s2, s3, s4, t0 = ids
    long_msg = "x" * 900
    rows = [
        _g(s0, 0, "fail", good_code(1), msg="count 1, but the scene has 7 shapes", cls="algorithm"),
        _g(s0, 1, "fail", good_code(2), msg="contract: 'areas' should be float, got int64", cls="convention"),
        _g(s0, 2, "A", good_code(3)),
        _g(s0, 3, "fail", good_code(4), msg="`cv2.AKAZE_create` is not in OpenCV 5.0", cls="api"),
        _g(s1, 0, "fail", good_code(5), cls="algorithm"), _g(s1, 1, "fail", good_code(6), cls="convention"),   # no verified answer: no pair
        _g(s2, 0, "A", good_code(7)),                                                                          # nothing failed: no pair
        _g(s3, 0, "A", good_code(8)), _g(s3, 1, "fail", "", code_empty=True),                                  # an empty reply is no lesson
        _g(s3, 2, "fail", good_code(9), finish_reason="length"),                                               # cut off by the token budget
        _g(s3, 3, "fail", good_code(10), failure={"repairable": False, "gate": None, "message": "grader error", "class": "other", "rule": "x"}),
        _g(s3, 4, "fail", good_code(11), extract="fence-unclosed"),
        _g(s4, 0, "A", good_code(12)), _g(s4, 1, "fail", good_code(12) + "# same code, another comment\n", cls="algorithm"),   # same AST as the answer
        _g(t0, 0, "fail", good_code(13), family="text_layout", msg=long_msg, cls="algorithm"),
        _g(t0, 1, "A", good_code(14), family="text_layout"),
    ]
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pairs = repair_mod.build_pairing_pairs(run)
    assert [p["spec_id"] for p in pairs] == [s0, t0]                                                       # at most one per spec, only where possible
    p0, pt = pairs
    assert (p0["failed_sample"], p0["fixed_sample"], p0["failure"]["class"]) == (1, 2, "convention")      # convention over the earlier algorithm failure
    assert p0["failed_code"] == good_code(2) and p0["fixed_code"] == good_code(3) and p0["tier"] == "A" and p0["kind"] == "repair"
    assert [m["role"] for m in p0["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert p0["messages"][0]["content"] == SYSTEM and p0["messages"][1]["content"] == "PROMPT 0"
    assert p0["messages"][2]["content"] == format_answer(good_code(2)) and p0["messages"][4]["content"] == format_answer(good_code(3))
    assert p0["messages"][3]["content"] == "That failed: contract: 'areas' should be float, got int64\nReturn the corrected full function."
    assert p0["feedback"] == p0["messages"][3]["content"] and p0["failed_ast_sha1"] == ast_hash(good_code(2))
    assert pt["failure"]["class"] == "algorithm" and pt["failed_sample"] == 0                              # the only failure there is
    assert len(pt["feedback"]) <= len("That failed: \nReturn the corrected full function.") + 800 and "..." in pt["feedback"]
    # api ranks with convention: with no convention failure the earlier of the preferred ones wins, whatever its sample number
    rows[1] = _g(s0, 1, "fail", good_code(2), cls="algorithm")
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert repair_mod.build_pairing_pairs(run)[0]["failed_sample"] == 3 and repair_mod.build_pairing_pairs(run)[0]["failure"]["class"] == "api"
    assert main(["pair", "--run", run.name, "--root", str(tmp_path)]) == 0
    assert len(read_jsonl(run.path / "pairing_pairs.jsonl")) == 2
    # an unclean verified answer (a demo call at module level) is not used as the target
    rows[2] = _g(s0, 2, "A", good_code(3) + "print(solve(None))\n")
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [p["spec_id"] for p in repair_mod.build_pairing_pairs(run)] == [t0]
    with pytest.raises(SystemExit, match="run `verify` first"):
        repair_mod.build_pairing_pairs(FoundryRun.of("empty", tmp_path))


# ================================================================ Gate 4a: the dataset builder

@pytest.fixture(scope="module")
def decon():
    return ds.Decontaminator()


def _ex(i, prompt="P", kind="solution", **meta):
    return ds.Example(f"e{i}", kind, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                                      {"role": "assistant", "content": f"```python\nx = {i}\n```"}], (kind, prompt, i), dict(meta))


def _ds_run(tmp_path, name="dsrun"):
    """A verified run, written by hand: 10 solutions and 6 repair pairs worth keeping, 3 solutions and 2 pairs that must not be."""
    run = FoundryRun.of(name, tmp_path)
    run.path.mkdir(parents=True)
    dev_prompt = dataclasses.replace(ca.SPEC, task=ca.PHRASINGS[1]).to_prompt()                       # a dev prompt, verbatim
    test_copy = dataclasses.replace(ca.SPEC, task=ca.PHRASINGS[5].replace("Tell me", "Give me")).to_prompt()   # a test prompt, reworded
    specs, rows = [], []

    def spec(sid, family, prompt, j, **kw):
        specs.append({"id": sid, "family": family, "index": j, "phrasing_index": 12 + j % 4, "seed": 5_000_000 + 40 * j, "prompt": prompt,
                      "subset": "timeless", **kw})
        return sid

    for j in range(4):
        s = spec(f"foundry-contour_area-{j:04d}", "contour_area", f"Fake prompt number {j} about measuring bright objects in a picture", j)
        rows += [_g(s, 0, "A", good_code(10 + j)), _g(s, 1, "A", good_code(20 + j)), _g(s, 2, "fail", good_code(30 + j))]
    t = spec("foundry-text_layout-0000", "text_layout", "Fake text prompt about drawing a label in a box", 0)
    rows += [_g(t, 0, "A", good_code(50), family="text_layout"), _g(t, 1, "fail", good_code(51), family="text_layout", cls="convention")]
    d = spec("full-fake_hist-000", "drills", "Fake drill prompt about counting gray levels", 0, kind="drill", drill="fake_hist", seed=0)
    rows += [{**_g(d, 0, "A", good_code(60), family="drills"), "kind": "drill", "stage": "ok"},
             {**_g(d, 1, "fail", good_code(61), family="drills", msg="wrong values", cls="algorithm"), "kind": "drill", "stage": "verify"}]
    bad = spec("foundry-contour_area-0004", "contour_area", "Fake prompt four with unclean answers", 4)       # clean code is required
    rows += [_g(bad, 0, "A", good_code(70) + "\nif __name__ == '__main__':\n    print(solve(None))\n"), _g(bad, 1, "A", good_code(71), finish_reason="length")]
    for sid, j, prompt in (("foundry-contour_area-0005", 5, dev_prompt), ("foundry-contour_area-0006", 6, test_copy)):
        s = spec(sid, "contour_area", prompt, j)
        rows += [_g(s, 0, "A", good_code(80 + j)), _g(s, 1, "fail", good_code(90 + j))] + ([_g(s, 2, "A", good_code(85 + j))] if j == 5 else [])
    run.specs.write_text("".join(json.dumps(r) + "\n" for r in specs))
    run.stage("generate").grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run.stage("generate").session.write_text(json.dumps({"model": "fake/teacher", "revision": "r1", "teacher_license": "apache-2.0"}))
    return run


def _replay(tmp_path, rows=None):
    rows = rows or [
        {"messages": [{"role": "user", "content": "Write a function that reverses a string."},
                      {"role": "assistant", "content": "```python\ndef reverse(s):\n    return s[::-1]\n```"}], "source": "unit-test", "license": "mit"},
        {"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Sum the numbers of a list."},
                      {"role": "assistant", "content": "```python\ndef total(xs):\n    return sum(xs)\n```"}], "source": "unit-test", "license": "mit"},
        {"messages": [{"role": "user", "content": "Find the chessboard corners in an image."}, {"role": "assistant", "content": "ok"}],
         "source": "unit-test", "license": "mit"}]
    path = tmp_path / "replay.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_collect_keeps_clean_tier_a_answers_in_the_fenced_format_and_counts_what_it_rejects(tmp_path):
    run = _ds_run(tmp_path)
    sol, rep, st = ds.collect_run(run)
    assert (len(sol), len(rep)) == (13, 8) and st["grades"] == 23 and st["tier_A"] == 15
    assert st["format_rejects"] == {"top-level If statement": 1, "finish_reason length": 1}
    s0 = next(e for e in sol if e.id == "dsrun/foundry-contour_area-0000#0")
    assert [m["role"] for m in s0.messages] == ["system", "user", "assistant"] and s0.messages[0]["content"] == SYSTEM
    assert s0.messages[2]["content"] == format_answer(good_code(10)) == "```python\n" + good_code(10).strip("\n") + "\n```"
    assert s0.meta["family"] == "contour_area" and s0.meta["tier"] == "A" and s0.meta["grades"] == "exec-only+lint+manufactured+metamorphic"
    assert (s0.meta["teacher"], s0.meta["teacher_revision"], s0.meta["teacher_license"]) == ("fake/teacher", "r1", "apache-2.0")
    assert s0.meta["scene_seed"] == 5_000_000 and s0.meta["phrasing"] == 12 and s0.meta["thinking"] is False
    assert {e.id for e in sol if e.meta["spec_id"] == "foundry-contour_area-0000"} == {"dsrun/foundry-contour_area-0000#0", "dsrun/foundry-contour_area-0000#1"}
    pair = next(e for e in rep if e.meta["spec_id"] == "foundry-contour_area-0000")
    assert [m["role"] for m in pair.messages] == ["system", "user", "assistant", "user", "assistant"]
    assert pair.meta["failed_ast_sha1"] == ast_hash(good_code(30)) and pair.meta["failure_class"] == "algorithm"
    assert pair.key == ("fix", sha1("Fake prompt number 0 about measuring bright objects in a picture"), ast_hash(good_code(30)), ast_hash(good_code(10)))
    drill = next(e for e in sol if e.meta["family"] == "drills")
    assert drill.meta["drill"] == "fake_hist" and drill.meta["scene_seed"] is None and drill.meta["grades"] == "drill-grade:ok"
    held = FoundryRun.of("heldout", tmp_path)                                                       # a held-out family is an error, not a filter
    held.path.mkdir()
    held.specs.write_text(json.dumps({"id": "x", "family": "chessboard_corners", "index": 0, "phrasing_index": 12, "seed": 1, "prompt": "p"}) + "\n")
    held.stage("generate").grades.write_text(json.dumps(_g("x", 0, "A", good_code(1), family="chessboard_corners")) + "\n")
    with pytest.raises(SystemExit, match="held-out"):
        ds.collect_run(held)


def test_decontamination_drops_scored_prompts_and_near_copies(decon, tmp_path):
    sol, rep, _ = ds.collect_run(_ds_run(tmp_path))
    kept, st = ds.decontaminate(sol + rep, decon)
    gone = {e.meta["spec_id"] for e in sol + rep} - {e.meta["spec_id"] for e in kept}
    assert gone == {"foundry-contour_area-0005", "foundry-contour_area-0006"}
    assert st["dropped"] == {"solution": 3, "repair": 2} and st["checked"] == 21 and st["dup_threshold"] == 0.25 and st["run_limit"] == 17
    why = {e.meta["spec_id"]: decon.reason(e) for e in sol if e.meta["spec_id"] in gone}
    assert why["foundry-contour_area-0005"].startswith("is the prompt of dev-contour_area-")
    assert "5-gram Jaccard" in why["foundry-contour_area-0006"] and "contour_area[5]" in why["foundry-contour_area-0006"]
    # the held-out families' own prompts, and a replay prompt that contains a gotcha task line
    from cvbench.families import chessboard_corners as cb
    from cvbench.gotchas import CHECKS
    held = _ex(1, dataclasses.replace(cb.SPEC, task=cb.PHRASINGS[13]).to_prompt())
    assert decon.reason(held).startswith("is the prompt of chessboard_corners[13]")
    line = CHECKS["shape_houghlinesp"].spec.task
    buried = _ex(2, "Here is a long unrelated preamble about writing a function and so on. " + line + " Thank you for your help with this.", kind="replay")
    assert "gotcha:shape_houghlinesp" in decon.reason(buried)
    assert decon.reason(_ex(3, "Write a function that reverses a string and returns the result.", kind="replay")) is None


def test_balanced_pick_serves_every_prompt_before_any_gets_a_second(tmp_path):
    items = [_ex(i, prompt=f"P{i // 5}") for i in range(10)] + [_ex(10 + i, prompt=f"Q{i}") for i in range(3)]    # P0 and P1 have 5 each
    got = ds.balanced_pick(items, 4, seed=0)
    assert len(got) == 4 and len({e.prompt_sha1 for e in got}) == 4                                  # four prompts, one example each
    assert len({e.prompt_sha1 for e in ds.balanced_pick(items, 5, seed=0)}) == 5                     # all five prompts before a repeat
    assert sorted(e.id for e in ds.balanced_pick(items, 99, seed=0)) == sorted(e.id for e in items)
    assert [e.id for e in ds.balanced_pick(items, 6, seed=3)] == [e.id for e in ds.balanced_pick(items, 6, seed=3)]
    assert [e.id for e in ds.balanced_pick(items, 6, seed=3)] != [e.id for e in ds.balanced_pick(items, 6, seed=4)]
    train, dev = ds.split_dev(items, 0.25, seed=1)
    assert {e.prompt_sha1 for e in train}.isdisjoint({e.prompt_sha1 for e in dev}) and len(train) + len(dev) == len(items)
    assert 1 <= len(dev) <= 0.25 * len(items) * 1.25 + 1


def test_replay_rows_are_single_turn_licensed_and_given_the_training_system_prompt(tmp_path):
    ex, st = ds.load_replay(_replay(tmp_path))
    assert st == {"rows": 3, "kept": 2, "held_out_topic": 1}
    assert all(e.kind == "replay" and e.messages[0] == {"role": "system", "content": SYSTEM} for e in ex)
    assert [e.messages[1]["content"] for e in ex] == ["Write a function that reverses a string.", "Sum the numbers of a list."]
    assert {(e.meta["source"], e.meta["license"]) for e in ex} == {("unit-test", "mit")} and ex[0].id.startswith("replay/")
    turn = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
                         {"role": "assistant", "content": "d"}], "source": "s", "license": "l"}
    think = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>x</think> b"}], "source": "s", "license": "l"}
    ok = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "source": "s", "license": "l"}
    ex, st = ds.load_replay(_replay(tmp_path, [turn, think, ok]))
    assert st == {"rows": 3, "kept": 1, "not_single_turn": 1, "has_reasoning": 1}
    for bad in ({"messages": ok["messages"], "source": "s"}, {"messages": ok["messages"], "license": "l"}, {"source": "s", "license": "l"}):
        with pytest.raises(SystemExit, match="license"):
            ds.load_replay(_replay(tmp_path, [bad]))


def test_build_dataset_end_to_end(tmp_path, decon, monkeypatch):
    run = _ds_run(tmp_path)
    replay = _replay(tmp_path)
    out = tmp_path / "data"
    m = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=replay, out_dir=out, total=None, dev_frac=0.2, seed=0, tokens="no",
                         decontaminator=decon, log=lambda x: None)
    st = m["stages"]
    assert st["collected"] == {"solution": 13, "repair": 8, "replay": 2} and st["decontamination"]["solution"]["dropped"] == {"solution": 3}
    assert st["after_dedup"] == {"solution": 10, "repair": 6, "replay": 2}
    train, dev = read_jsonl(out / "train.jsonl"), read_jsonl(out / "dev.jsonl")
    rows = train + dev
    assert len(rows) == 18 and m["composition"]["all"]["by_kind"] == {"solution": 10, "repair": 6, "replay": 2}
    assert {r["prompt_sha1"] for r in train}.isdisjoint({r["prompt_sha1"] for r in dev}) and dev                      # no prompt in both files
    assert m["dev"]["rows"] == len(dev) and m["dev"]["distinct_prompts"] == len({r["prompt_sha1"] for r in dev})
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 18
    assert {"id", "kind", "messages", "train_on", *ds.ROW_KEYS} <= set(rows[0])
    assert {frozenset(r) for r in rows} == {frozenset({"id", "kind", "messages", "train_on", *ds.ROW_KEYS})}          # one flat schema
    for r in rows:
        roles = [msg["role"] for msg in r["messages"]]
        assert r["messages"][0] == {"role": "system", "content": SYSTEM} and r["train_on"] == "last_assistant"        # Rule 7: never the notes
        assert roles == {"solution": ["system", "user", "assistant"], "repair": ["system", "user", "assistant", "user", "assistant"],
                         "replay": ["system", "user", "assistant"]}[r["kind"]]
        assert "OpenCV 5.0 notes" not in json.dumps(r["messages"]) and r["prompt_sha1"] == sha1(r["messages"][1]["content"])
        if r["kind"] != "replay":
            assert r["messages"][-1]["content"].startswith("```python\n") and r["messages"][-1]["content"].endswith("\n```")
            assert r["messages"][-1]["content"].count("```") == 2                                                      # no prose around it
            assert (r["teacher"], r["teacher_revision"], r["tier"], r["thinking"]) == ("fake/teacher", "r1", "A", False)
        else:
            assert (r["source"], r["license"]) == ("unit-test", "mit") and r["spec_id"] is None
    by_id = {r["id"]: r for r in rows}
    fix = by_id["dsrun/foundry-contour_area-0001/pair"]
    assert fix["messages"][2]["content"] == format_answer(good_code(31)) and fix["messages"][4]["content"] == format_answer(good_code(11))
    assert fix["failure_class"] == "algorithm" and fix["failed_ast_sha1"] == ast_hash(good_code(31)) and fix["ast_sha1"] == ast_hash(good_code(11))
    assert {r["family"] for r in rows if r["kind"] != "replay"} == {"contour_area", "text_layout", "drills"}
    # no scored prompt, no unclean answer, nothing from the rejected specs
    assert not [r for r in rows if r["spec_id"] in ("foundry-contour_area-0004", "foundry-contour_area-0005", "foundry-contour_area-0006")]
    # the manifest
    man = json.loads((out / "manifest.json").read_text())
    for name in ("train.jsonl", "dev.jsonl"):
        f = man["files"][name]
        assert f["sha256"] == ds.sha256_file(out / name) and f["bytes"] == (out / name).stat().st_size < ds.MAX_FILE_BYTES
        assert f["rows"] == len(read_jsonl(out / name))
    assert man["inputs"]["replay"]["sha256"] == ds.sha256_file(replay) and man["inputs"]["runs"][0]["teacher"]["model"] == "fake/teacher"
    assert man["builder"]["system_sha1"] == sha1(SYSTEM) and man["composition"]["all"]["rows"] == 18 and "tokens" not in man
    assert man["composition"]["all"]["sources"] == {"foundry-teacher (apache-2.0)": 16, "unit-test (mit)": 2}
    assert "18 examples" in ds.render_summary(m)

    # a second copy of the run adds nothing: global dedup by (prompt, normalised code)
    shutil.copytree(run.path, FoundryRun.of("dsrun2", tmp_path).path)
    m2 = ds.build_dataset(["dsrun", "dsrun2"], root=str(tmp_path), replay=replay, out_dir=tmp_path / "data2", total=None, dev_frac=0.2, seed=0,
                          tokens="no", decontaminator=decon, log=lambda x: None)
    assert m2["stages"]["collected"] == {"solution": 26, "repair": 16, "replay": 2}
    assert m2["stages"]["dedup"]["dropped"] == {"solution": 10, "repair": 6} and m2["stages"]["after_dedup"] == m["stages"]["after_dedup"]
    assert (tmp_path / "data2" / "train.jsonl").read_bytes() == (out / "train.jsonl").read_bytes()                    # and the same files

    # the mix: 70 / 20 / 10 of a total, chosen round-robin over prompts, and what is short is reported
    m3 = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=replay, out_dir=tmp_path / "data3", total=10, dev_frac=0.2, seed=0,
                          tokens="no", decontaminator=decon, log=lambda x: None)
    assert {k: v["selected"] for k, v in m3["stages"]["mix"].items()} == {"solution": 7, "repair": 2, "replay": 1}
    assert all(v["shortfall"] == 0 for v in m3["stages"]["mix"].values())
    sol3 = [r for r in read_jsonl(tmp_path / "data3" / "train.jsonl") + read_jsonl(tmp_path / "data3" / "dev.jsonl") if r["kind"] == "solution"]
    assert len({r["prompt_sha1"] for r in sol3}) == 6                                                                  # all 6 prompts before a 2nd example
    m4 = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=replay, out_dir=tmp_path / "data4", total=40, tokens="no",
                          decontaminator=decon, log=lambda x: None)
    assert {k: v["shortfall"] for k, v in m4["stages"]["mix"].items()} == {"solution": 18, "repair": 2, "replay": 2}
    assert "SHORT by 18" in ds.render_summary(m4)
    again = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=replay, out_dir=tmp_path / "data5", total=10, dev_frac=0.2, seed=0,
                             tokens="no", decontaminator=decon, log=lambda x: None)
    assert (tmp_path / "data5" / "train.jsonl").read_bytes() == (tmp_path / "data3" / "train.jsonl").read_bytes()      # deterministic
    ds.build_dataset(["dsrun"], root=str(tmp_path), replay=replay, out_dir=tmp_path / "data6", total=10, dev_frac=0.2, seed=1,
                     tokens="no", decontaminator=decon, log=lambda x: None)
    assert (tmp_path / "data6" / "train.jsonl").read_bytes() != (tmp_path / "data3" / "train.jsonl").read_bytes()      # a seed matters
    # a dry run writes nothing; a file over 50 MB is refused
    m7 = ds.build_dataset(["dsrun"], root=str(tmp_path), out_dir=tmp_path / "data7", total=None, tokens="no", dry_run=True,
                          decontaminator=decon, log=lambda x: None)
    assert not (tmp_path / "data7").exists() and m7["composition"]["all"]["by_kind"] == {"solution": 10, "repair": 6}
    monkeypatch.setattr(ds, "MAX_FILE_BYTES", 100)
    with pytest.raises(SystemExit, match="50 MB"):
        ds.build_dataset(["dsrun"], root=str(tmp_path), out_dir=tmp_path / "data8", total=None, tokens="no", decontaminator=decon, log=lambda x: None)
    assert ds.parse_shares("0.5,0.3,0.2") == {"solution": 0.5, "repair": 0.3, "replay": 0.2} and ds.parse_shares(None) == ds.DEFAULT_SHARES
    with pytest.raises(SystemExit, match="sum to 1"):
        ds.parse_shares("0.5,0.5,0.5")


def test_build_dataset_command(tmp_path, capsys):
    _ds_run(tmp_path)
    replay = _replay(tmp_path)
    base = ["build-dataset", "--run", "dsrun", "--root", str(tmp_path), "--replay", str(replay), "--out", str(tmp_path / "out"), "--tokens", "no",
            "--dev-frac", "0.2"]
    assert main([*base, "--all"]) == 0
    out = capsys.readouterr().out
    assert "18 examples" in out and "solution 10 (55.6 %)" in out and (tmp_path / "out" / "manifest.json").is_file()
    assert main([*base, "--total", "40"]) == 3                                                                   # a shortfall is a non-zero exit
    with pytest.raises(SystemExit, match="tokenizer cannot be loaded"):
        ds.build_dataset(["dsrun"], root=str(tmp_path), out_dir=tmp_path / "o2", total=None, tokens="yes", dry_run=True,
                         tokenizer=None, decontaminator=ds.Decontaminator(), log=lambda x: None) if not importlib.util.find_spec("transformers") \
            else (_ for _ in ()).throw(SystemExit("tokenizer cannot be loaded"))


# ---------------------------------------------------------------- token lengths under the base's own chat template

@pytest.mark.parametrize("base", sorted(BASES))
def test_render_chat_equals_render_turn_for_a_single_turn(base):
    for answer in ("```python\nx = 1\n```", "  padded  \n"):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "U"}, {"role": "assistant", "content": answer}]
        assert ds.render_chat(base, msgs) == render_turn(base, SYSTEM, "U", answer)


def test_render_chat_leaves_earlier_assistant_turns_plain():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}, {"role": "assistant", "content": "FAILED"},
            {"role": "user", "content": "U2"}, {"role": "assistant", "content": "FIXED"}]
    # measured with the Qwen3.5-9B tokenizer's chat template on 2026-10-23: only the last assistant turn gets the empty think block
    assert ds.render_chat("qwen3.5-9b", msgs) == ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nFAILED<|im_end|>\n"
                                                  "<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nFIXED<|im_end|>\n")
    for bad in (msgs[:-1], msgs[1:], [msgs[0], msgs[2]], [msgs[0], msgs[1], msgs[1]]):
        with pytest.raises(ValueError):
            ds.render_chat("qwen3.5-9b", bad)


class FakeHFTokenizer:
    """Tokens are words. ``apply_chat_template`` is what render_chat should equal, plus ``drift`` characters when asked to disagree."""

    def __init__(self, drift=""):
        self.drift = drift

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return ds.render_chat("qwen3.5-9b", messages) + self.drift

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=text.split())


def test_token_report_lengths_and_template_agreement(tmp_path):
    def row(kind, n):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": " ".join(["w"] * n)}]
        if kind == "repair":
            msgs[2:2] = [{"role": "assistant", "content": "bad"}, {"role": "user", "content": "again"}]
        return {"kind": kind, "messages": msgs}

    rows = [row("solution", n) for n in range(1, 101)] + [row("repair", 50)]
    rep = ds.token_report(rows, FakeHFTokenizer())
    lengths = [len(ds.render_chat("qwen3.5-9b", r["messages"]).split()) for r in rows]
    assert rep["n"] == 101 and rep["max"] == max(lengths) and rep["template_mismatches"] == 0 and rep["tokenizer"] == "Qwen/Qwen3.5-9B"
    s = sorted(lengths)
    assert rep["p50"] == s[50] and rep["p99"] == s[int(0.99 * 101)] and rep["mean"] == pytest.approx(sum(lengths) / 101, abs=0.05)
    assert rep["by_kind"]["repair"]["n"] == 1 and rep["by_kind"]["solution"]["n"] == 100 and "MAX_LEN" in rep["note"]
    assert ds.token_report(rows, FakeHFTokenizer(drift="!"))["template_mismatches"] == 101           # a drifting template is counted
    assert rep["rows_without_exactly_one_response_marker"] == 0                                       # loss on the last assistant turn only
    odd = {"kind": "solution", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                                            {"role": "assistant", "content": "<|im_start|>assistant\n<think>\n\n</think>\n\nsneaky"}]}
    assert ds.token_report([odd], FakeHFTokenizer())["rows_without_exactly_one_response_marker"] == 1
    assert ds.length_stats([]) == {"n": 0}
    data = tmp_path / "d"
    data.mkdir()
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "dev.jsonl").write_text("")
    out = ds.tokens_for_files(data, tokenizer=FakeHFTokenizer())
    assert out["train"]["max"] == rep["max"] and out["dev"] is None
    man = json.loads((data / "manifest.json").read_text())
    assert man["tokens"]["train"]["p99"] == rep["p99"] and man["files"]["train.jsonl"]["rows"] == 101


def test_the_real_qwen35_template_renders_what_render_chat_says():
    pytest.importorskip("transformers")
    try:
        tok = ds.load_tokenizer("qwen3.5-9b")
    except Exception as e:  # noqa: BLE001 - not cached on this machine
        pytest.skip(f"Qwen/Qwen3.5-9B tokenizer not in the local cache: {e!r}")
    single = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "U"}, {"role": "assistant", "content": format_answer(good_code(1))}]
    multi = [*single[:2], {"role": "assistant", "content": format_answer(good_code(2))}, {"role": "user", "content": "That failed: x"}, single[2]]
    for msgs in (single, multi):
        assert tok.apply_chat_template(msgs, tokenize=False) == ds.render_chat("qwen3.5-9b", msgs)
    rep = ds.token_report([{"kind": "solution", "messages": single}, {"kind": "repair", "messages": multi}], tok)
    assert rep["template_mismatches"] == 0 and rep["rows_without_exactly_one_response_marker"] == 0


def test_think_test_specs_can_give_the_weak_families_more_specs():
    specs = think_test_specs(per_family=1, drills=0, k=2, more={"hough_segments": 3, "text_layout": 2})
    assert [sum(s.family == f for s in specs) for f in trainable_families()] == [1, 3, 1, 1, 1, 1, 2]
    assert [s.id for s in specs if s.family == "hough_segments"] == [f"tt-hough_segments-000{j}" for j in range(3)]
    assert {s.phrasing_index for s in specs if s.family == "hough_segments"} == {12, 13, 14}


def test_predict_can_price_a_full_generation_family_by_family(tmp_path, capsys, phrasing_dir):
    rows = [{"id": "a", "family": "hough_segments", "mode": "think", "k": 8}, {"id": "b", "family": "contour_area", "k": 4},
            {"id": "c", "family": "contour_area", "k": 4}]
    text = predict_mix(rows, 4, {"hough_segments/think": 6000, "contour_area": 300}, {"hough_segments/think": 400}, None, 800, 0, 0, 5.4, 0.1,
                       group_by="family")
    assert re.search(r"hough_segments/think\s+1 prompts,\s+8 samples x\s+6000 tokens =\s+48000 output tokens; at 400 tok/s\s+2\.0 min", text)
    assert re.search(r"contour_area\s+2 prompts,\s+8 samples x\s+300 tokens =\s+2400 output tokens; at 800 tok/s", text)
    assert main(["prepare-full", "--run", "pf", "--root", str(tmp_path), "--target-kept", "70", "--no-drills", "--think-families", "text_layout"]) == 0
    assert main(["predict", "--run", "pf", "--root", str(tmp_path), "--tok-s", "800", "--group-by", "family", "--tokens", "500",
                 "--tokens-by-label", '{"text_layout/think": 6000}']) == 0
    out = capsys.readouterr().out
    assert "text_layout/think" in out and "hough_segments/nothink" in out and "6000 tokens" in out


REAL_REPLAY = ds.DEFAULT_REPLAY


@pytest.mark.skipif(not REAL_REPLAY.is_file(), reason="the replay file is not on this machine")
def test_the_real_replay_file_loads_and_goes_through_the_builder(tmp_path, decon):
    lines = REAL_REPLAY.read_text(encoding="utf-8").splitlines()
    raw = [json.loads(x) for x in lines]
    assert all(set(r) == {"messages", "source", "license", "id"} and [m["role"] for m in r["messages"]] == ["user", "assistant"] for r in raw)
    assert {r["license"] for r in raw} == {"CC-BY-4.0"} and all(r["source"].startswith("nvidia/OpenCodeInstruct@") for r in raw)
    assert len({r["id"] for r in raw}) == len(raw)
    ex, st = ds.load_replay(REAL_REPLAY)
    assert st["rows"] == len(raw) and st["kept"] + st.get("held_out_topic", 0) == len(raw) and st["kept"] >= len(raw) - 5
    assert set(st) <= {"rows", "kept", "held_out_topic"}                                           # nothing is multi-turn, nothing has reasoning
    for e in ex:
        assert e.kind == "replay" and e.messages[0] == {"role": "system", "content": SYSTEM} and e.meta["license"] == "CC-BY-4.0"
        assert e.messages[2]["content"].startswith("```python\n") and e.messages[2]["content"].count("```") == 2
    assert len({e.id for e in ex}) == len(ex)
    _ds_run(tmp_path)
    out = tmp_path / "data"
    m = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=REAL_REPLAY, out_dir=out, total=None, tokens="no", decontaminator=decon,
                         log=lambda x: None)
    assert m["stages"]["collected"]["replay"] == st["kept"] and m["stages"]["decontamination"]["replay"]["dropped"] == {}   # nothing is scored
    assert m["stages"]["after_dedup"]["replay"] == st["kept"] and m["inputs"]["replay"]["sha256"] == ds.sha256_file(REAL_REPLAY)
    rows = read_jsonl(out / "train.jsonl") + read_jsonl(out / "dev.jsonl")
    rep = [r for r in rows if r["kind"] == "replay"]
    assert len(rep) == st["kept"] and {(r["source"].split("@")[0], r["license"]) for r in rep} == {("nvidia/OpenCodeInstruct", "CC-BY-4.0")}
    assert all(r["messages"][0]["content"] == SYSTEM and r["family"] is None and r["spec_id"] is None for r in rep)
    assert {r["prompt_sha1"] for r in read_jsonl(out / "train.jsonl")}.isdisjoint({r["prompt_sha1"] for r in read_jsonl(out / "dev.jsonl")})
    # the mix asks for 10 % of 3,000 = 300 of them
    m2 = ds.build_dataset(["dsrun"], root=str(tmp_path), replay=REAL_REPLAY, out_dir=tmp_path / "d2", total=3000, tokens="no",
                          decontaminator=decon, log=lambda x: None)
    assert m2["stages"]["mix"]["replay"] == {"supply": st["kept"], "target": 300, "selected": 300, "shortfall": 0, "total": 3000}


def test_build_dataset_command_uses_the_default_replay_file_unless_told_not_to(tmp_path, monkeypatch, capsys):
    _ds_run(tmp_path)
    fake = _replay(tmp_path)
    monkeypatch.setattr(ds, "DEFAULT_REPLAY", fake)
    assert ds.default_replay() == fake
    base = ["build-dataset", "--run", "dsrun", "--root", str(tmp_path), "--tokens", "no", "--dev-frac", "0.2", "--all"]
    assert main([*base, "--out", str(tmp_path / "with")]) == 0
    assert json.loads((tmp_path / "with" / "manifest.json").read_text())["builder"]["replay"] == str(fake)
    assert "replay 2" in capsys.readouterr().out
    assert main([*base, "--out", str(tmp_path / "without"), "--no-replay"]) == 0
    assert json.loads((tmp_path / "without" / "manifest.json").read_text())["builder"]["replay"] is None
    monkeypatch.setattr(ds, "DEFAULT_REPLAY", tmp_path / "missing.jsonl")
    assert ds.default_replay() is None
    assert main([*base, "--out", str(tmp_path / "none")]) == 0
    assert "no replay file" in capsys.readouterr().err


def test_keep_shares_shrinks_the_total_instead_of_letting_one_kind_dominate(tmp_path, decon):
    _ds_run(tmp_path)
    replay = _replay(tmp_path)
    kw = dict(root=str(tmp_path), replay=replay, tokens="no", decontaminator=decon, log=lambda x: None, dev_frac=0.2)
    plain = ds.build_dataset(["dsrun"], out_dir=tmp_path / "a", total=40, **kw)
    assert {k: v["selected"] for k, v in plain["stages"]["mix"].items()} == {"solution": 10, "repair": 6, "replay": 2}
    kept = ds.build_dataset(["dsrun"], out_dir=tmp_path / "b", total=40, keep_shares=True, **kw)
    mix = kept["stages"]["mix"]
    # supply 10 / 6 / 2 at shares 0.7 / 0.2 / 0.1 allows totals of 14.3 / 30 / 20: the total shrinks to 14
    assert {k: v["selected"] for k, v in mix.items()} == {"solution": 10, "repair": 3, "replay": 1} and all(v["total"] == 14 for v in mix.values())
    assert all(v["shortfall"] == 0 for v in mix.values()) and kept["builder"]["keep_shares"] is True


def test_a_drill_graded_without_the_lint_never_yields_a_training_target():
    # drill_capture_properties needs lint=False (I/O the lint bans); its passing answers must not become Tier A.
    from foundry.verify import tier_of
    from foundry.specs import drill_names
    from foundry.drills import DRILLS
    assert tier_of({"kind": "drill", "passed": True, "lint_enforced": False}) == "fail"
    assert tier_of({"kind": "drill", "passed": True, "lint_enforced": True}) == "A"
    assert all(DRILLS[n].lint for n in drill_names())
    assert {n for n, c in DRILLS.items() if not c.lint} and not {n for n, c in DRILLS.items() if not c.lint} & set(drill_names())
