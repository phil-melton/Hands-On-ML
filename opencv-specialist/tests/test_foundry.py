"""The Stage 4 teacher-pilot machinery: specs and their invariants, the prompts file, the Colab job script with vLLM
mocked, the Colab driver with the CLI mocked (dry run, guards, and the stop in ``finally``), verification and tiers on
fake completions built from the families' REFERENCE solutions and mutants, AST dedup, repair prompts, and the report.

No GPU, no network and no Colab command is run. The one place the real CLI is touched is ``colab help <cmd>`` (read-only),
to check every flag the driver uses against the installed CLI; that test skips when the CLI is absent.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import re
import shutil
import subprocess
import sys
import time
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
from foundry.astnorm import ast_hash, normalize_code
from foundry.run import main, predict_text, prompt_rows
from foundry.specs import (FOUNDRY_BLOCK, NOTES_PATH, SpecError, check_specs, family_counts, pilot_specs,
                           reserved_prompt_hashes, sha1, teacher_system, trainable_families)
from foundry.store import FoundryRun, read_json, read_jsonl
from train.config import SYSTEM


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
        return body + "<|im_start|>assistant\n" + ("<think></think>" if kw.get("enable_thinking") is False else "")

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class FakeLLM:
    """Stands in for vllm.LLM: prompts arrive as token ids, replies come from ``answer(prompt_text, sample, prompt_no)``."""

    def __init__(self, answer, finish="stop"):
        self.answer, self.finish, self.calls, self.n_prompts = answer, finish, [], 0

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.calls.append((prompts, sampling_params, use_tqdm))
        outs = []
        for p in prompts:
            text = bytes(p["prompt_token_ids"]).decode("utf-8")
            reps = [self.answer(text, i, self.n_prompts) for i in range(sampling_params.n)]
            self.n_prompts += 1
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(index=i, text=t, token_ids=list(range(len(t.split()))),
                                                                 finish_reason=self.finish) for i, t in enumerate(reps)]))
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
    assert set(r0) == {"id", "spec_id", "sample", "text", "prompt_tokens", "completion_tokens", "finish_reason", "model"}
    assert r0["spec_id"] == "spec0" and r0["text"] == "```python\ncode 0 1\n```" and r0["finish_reason"] == "stop" and r0["model"] == "m/x"
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
