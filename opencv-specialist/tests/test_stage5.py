"""The Stage 5 training machinery: the Colab job's data, subset, template and loss-mask logic (a fake tokenizer, and the real
rows against the foundry's and train.config's renderers), the predict arithmetic, the driver with the Colab CLI mocked (dry
run, guards, stop in ``finally``, exec timeout = 2x, the tokens/s check), and the dev-answer path (download, ingest, grade in
the real sandbox, report) with fake answers built from the families' REFERENCE solutions and mutants.

No GPU, no network and no Colab command is run. The one place the real CLI is touched is ``colab help <cmd>`` (read-only), to
check every flag the driver uses against the installed CLI; that test skips when the CLI is absent. The GPU side of the job
(unsloth, the callback, the generation) is exercised by the local smoke test recorded in LOGBOOK.md, not here.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvbench.families import contour_area, histogram_stats, hough_segments
from foundry import colab as colab_mod
from foundry.dataset import render_chat
from train import colab_train_job as job
from train import stage5 as s5
from train.config import BASES, END, SYSTEM, USER_MARK, render_turn

TRAIN, DEV = s5.TRAIN_PATH, s5.DEV_PATH


@pytest.fixture(scope="module")
def train_rows():
    return s5.read_jsonl(TRAIN)


@pytest.fixture(scope="module")
def dev_rows():
    return s5.read_jsonl(DEV)


# ================================================================ a fake tokenizer: characters are tokens

class FakeTok:
    """Qwen3.5's chat template, written independently of ``job.render_row``: every content trimmed, the empty think block only
    in front of assistant turns after the last user turn, the generation prompt closed behind it when thinking is off."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        out = ""
        for i, m in enumerate(messages):
            c = m["content"].strip()
            if m["role"] == "assistant" and i > last_user:
                out += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{c}<|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
        return out

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return SimpleNamespace(input_ids=[ord(c) for c in texts])
        return SimpleNamespace(input_ids=[[ord(c) for c in t] for t in texts])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def fake_labels(text: str) -> list[int]:
    """What a correct response mask is for one rendered row: the final answer and <|im_end|>, nothing else."""
    start = text.rindex(job.ANSWER_PREFIX) + len(job.ANSWER_PREFIX)
    end = text.rindex(END) + len(END)
    return [ord(c) if start <= i < end else -100 for i, c in enumerate(text)]


def tokenized(rows):
    tok = FakeTok()
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    return texts, [[ord(c) for c in t] for t in texts]


def mini_row(user="U", first="print('wrong')", fix=None, kind="solution", family="f"):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": user}, {"role": "assistant", "content": first}]
    if fix is not None:
        msgs += [{"role": "user", "content": "That failed: nope"}, {"role": "assistant", "content": fix}]
    return {"id": f"{kind}-{user}", "kind": kind, "family": family, "messages": msgs}


# ================================================================ rows and the template (Rule 7)

def test_the_job_constants_are_train_config(train_rows):
    cfg = BASES["qwen3.5-9b"]
    assert job.ANSWER_PREFIX == cfg["answer_prefix"] and job.USER_MARK == USER_MARK and job.END == END
    assert job.DEFAULT_REPO == cfg["repo"] and job.SEED == 3407 and (job.LORA_R, job.LORA_ALPHA) == (16, 16)
    assert job.DEFAULT_REVISION == "005429cee5cb648998cf2b70eebdd83175989c9a"
    assert job.render_prompt(SYSTEM, "do it") == render_turn("qwen3.5-9b", SYSTEM, "do it")
    assert job.render_row([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "do it"},
                           {"role": "assistant", "content": "  code \n"}]) == render_turn("qwen3.5-9b", SYSTEM, "do it", "code")


def test_every_real_row_is_valid_and_renders_like_the_foundry(train_rows, dev_rows):
    manifest = json.loads((TRAIN.parent / "manifest.json").read_text())
    assert len(train_rows) == manifest["composition"]["train"]["rows"] == 2847 and len(dev_rows) == 153
    assert job.validate_rows(train_rows, "train") == SYSTEM == job.validate_rows(dev_rows, "dev")
    for r in train_rows + dev_rows:
        text = job.render_row(r["messages"])
        assert text == render_chat("qwen3.5-9b", r["messages"]), r["id"]
        assert text.count(job.ANSWER_PREFIX) == 1, r["id"]                       # one response marker, also in the repair rows
        assert FakeTok().apply_chat_template(r["messages"]) == text, r["id"]     # an independent template agrees


def test_repair_rows_train_only_on_the_fix(train_rows):
    repair = next(r for r in train_rows if len(r["messages"]) == 5)
    text = job.render_row(repair["messages"])
    labels = fake_labels(text)
    seen = "".join(chr(t) for t in labels if t != -100)
    assert seen == repair["messages"][-1]["content"].strip() + END
    assert repair["messages"][2]["content"].strip()[:60] not in seen             # the failed first answer is masked


def test_validate_rows_rejects_bad_structure():
    good = mini_row(fix="ok")
    assert job.row_problems(good) == []
    bad_roles = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "assistant", "content": "z"}]}
    assert job.row_problems(bad_roles)
    assert job.row_problems({"messages": good["messages"][:2]})                   # no answer
    assert job.row_problems({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": 3}]})
    with pytest.raises(SystemExit, match="different system prompts"):
        other = mini_row(user="V")
        other["messages"][0]["content"] = "T"
        job.validate_rows([good, other])
    with pytest.raises(SystemExit, match="roles"):
        job.validate_rows([bad_roles])


def test_preflight_passes_real_rows_and_fails_a_drifted_template_a_second_marker_and_a_long_row(train_rows, dev_rows):
    tok = FakeTok()
    s = job.preflight(tok, train_rows, 100_000, "train")
    assert s["rows"] == 2847 and s["lengths"][0] == len(job.render_row(train_rows[0]["messages"])) and s["max"] == max(s["lengths"])
    assert job.preflight(tok, dev_rows, 100_000, "dev")["rows"] == 153

    class Drifted(FakeTok):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **kw).replace("<|im_end|>\n", "<|im_end|>\n\n", 1)

    with pytest.raises(SystemExit, match="render differently"):
        job.preflight(Drifted(), [mini_row()], 1000)
    twice = mini_row(first="a", fix="b " + job.ANSWER_PREFIX + " c")             # the answer itself contains the marker
    with pytest.raises(SystemExit, match="exactly one response marker"):
        job.preflight(tok, [twice], 1000)
    with pytest.raises(SystemExit, match="never truncate"):
        job.preflight(tok, [mini_row(first="x" * 500)], 100)


def test_dev_prompt_template_check():
    tok = FakeTok()
    job.check_prompt_template(tok, SYSTEM, "task text")

    class Thinks(FakeTok):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **{**kw, "enable_thinking": None})

    with pytest.raises(SystemExit, match="dev prompt drift"):
        job.check_prompt_template(Thinks(), SYSTEM, "task text")


def test_the_loss_mask_check_accepts_the_right_mask_and_catches_every_wrong_one(train_rows):
    rows = [mini_row(), mini_row(user="V", first="old", fix="new fix", kind="repair"), train_rows[0]]
    tok = FakeTok()
    texts, ids = tokenized(rows)
    good = [fake_labels(t) for t in texts]
    stats = job.check_labels(tok, rows, ids, good, "rows")
    assert stats["rows"] == 3 and stats["trained_tokens"] == sum(sum(y != -100 for y in lab) for lab in good)

    def run(labels, rows=rows, ids=ids):
        return job.check_labels(tok, rows, ids, labels, "rows")

    whole = [list(i) for i in ids]                                                # trained on everything, prompt included
    with pytest.raises(SystemExit, match="wrong span"):
        run(whole)
    first_turn = texts[1].index("old")                                            # the failed answer of the repair row is trained too
    leaky = [list(x) for x in good]
    for k in range(first_turn, first_turn + 3):
        leaky[1][k] = ids[1][k]
    with pytest.raises(SystemExit, match="wrong span"):
        run(leaky)
    short = [list(x) for x in good]
    for i in [i for i, y in enumerate(short[0]) if y != -100][-4:]:               # an answer whose last tokens, <|im_end|> included, are masked
        short[0][i] = -100
    with pytest.raises(SystemExit, match="wrong span"):
        run(short)
    with pytest.raises(SystemExit, match="dropped or added rows"):
        run(good[:2])
    shifted = [list(x) for x in good]
    k = next(i for i, y in enumerate(shifted[0]) if y != -100)
    shifted[0][k] = shifted[0][k] + 1
    with pytest.raises(SystemExit, match="differs from its input token"):
        run(shifted)
    with pytest.raises(SystemExit, match="labels"):
        run([x[:-1] for x in good])
    assert run([fake_labels(t) for t in texts], rows, ids)["mean_trained_tokens"] > 0
    nl = [list(x) for x in good]                                                  # a trailing newline token after <|im_end|> is tolerated
    nl[0][len(nl[0]) - 1] = ids[0][-1]
    assert run(nl)["rows"] == 3


# ================================================================ nested stratified subsets

def test_subsets_are_nested_deterministic_and_stratified(train_rows):
    n = len(train_rows)
    sizes = [job.subset_size(n, f) for f in (0.25, 0.5, 1.0)]
    assert sizes == [712, 1424, 2847]
    subs = [job.select_subset(train_rows, f) for f in (0.25, 0.5, 1.0)]
    assert [len(s) for s in subs] == sizes
    assert set(subs[0]) < set(subs[1]) < set(subs[2]) == set(range(n))
    assert subs == [job.select_subset(train_rows, f) for f in (0.25, 0.5, 1.0)]    # deterministic in-process
    assert all(s == sorted(s) for s in subs)
    full: dict = {}
    for r in train_rows:
        full[job.stratum(r)] = full.get(job.stratum(r), 0) + 1
    for f, s in zip((0.25, 0.5), subs):
        got: dict = {}
        for i in s:
            got[job.stratum(train_rows[i])] = got.get(job.stratum(train_rows[i]), 0) + 1
        for key, total in full.items():                                           # every kind x family keeps its share, within one row
            assert abs(got.get(key, 0) - f * total) <= 1.0 + 1e-9, (f, key, got.get(key, 0), f * total)
        kinds = {k: sum(v for (kk, _), v in got.items() if kk == k) / len(s) for k in ("solution", "repair", "replay")}
        fk = {k: sum(v for (kk, _), v in full.items() if kk == k) / n for k in kinds}
        assert all(abs(kinds[k] - fk[k]) < 0.01 for k in kinds)
    probe = job.probe_indices(train_rows, 64)
    assert len(probe) == 64 and set(probe) <= set(subs[0])                          # one probe, inside every subset
    other = job.select_subset(train_rows, 0.25, seed=1)
    assert other != subs[0] and set(job.select_subset(train_rows, 0.25, seed=1)) < set(job.select_subset(train_rows, 0.5, seed=1))
    with pytest.raises(ValueError):
        job.subset_size(10, 0.0)
    with pytest.raises(ValueError):
        job.subset_size(10, 1.5)
    assert job.subset_size(3, 0.01) == 1 and job.select_subset([mini_row()], 1.0) == [0]


def test_subset_order_does_not_depend_on_the_process_hash_seed():
    code = ("import json, sys; sys.path.insert(0, %r); from train import colab_train_job as j; "
            "rows = [json.loads(l) for l in open(%r)]; print(hash(tuple(j.select_subset(rows, 0.25))))" % (str(Path(s5.ROOT)), str(TRAIN)))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout.strip() for seed in ("1", "2", "random")}
    # hash() of a tuple of ints is not salted, so equal output means equal selections despite different string-hash seeds
    assert len(outs) == 1 and outs != {""}


def test_checkpoint_plan_and_window_helpers():
    assert job.ckpt_plan(90, 2.0, 0.5) == {23: "e0.5", 45: "e1", 68: "e1.5", 90: "e2"}
    assert job.ckpt_plan(4, 1.0, 0.5) == {2: "e0.5", 4: "e1"}
    assert job.ckpt_plan(356, 2.0, 1.0) == {178: "e1", 356: "e2"}
    assert list(job.ckpt_plan(89, 2.0, 0.5)) == [22, 45, 67, 89] and list(job.ckpt_plan(89, 2.0, 0.5).values())[-1] == "e2"
    assert job.ckpt_plan(1, 2.0, 0.5) == {1: "e2"}                                    # degenerate: everything on the last step
    with pytest.raises(ValueError):
        job.ckpt_plan(0, 2.0, 0.5)
    log = [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}, {"eval_loss": 9.0, "step": 2}, {"step": 3, "loss": 3.0}]
    assert job.window_stats(log, 0, 2) == {"train_loss_window": 1.5, "steps_in_window": 2}
    assert job.window_stats(log, 2, 3) == {"train_loss_window": 3.0, "steps_in_window": 1}
    assert job.window_stats(log, 3, 9)["train_loss_window"] is None
    assert job.stop_cut([5, 6, 99, 7], {99}) == ([5, 6], "stop") and job.stop_cut([5, 6], {99}) == ([5, 6], "length")


def test_outputs_manifest_lists_files_with_hashes_and_states(tmp_path):
    out = job.Outputs(tmp_path / "o", "r025")
    assert json.loads(out.path("manifest.json").read_text())["state"] == "running"
    out.event("env", gpu="x")
    out.path("e1.dev_answers.jsonl").write_text('{"a": 1}\n')
    out.register("e1.dev_answers.jsonl", "answers")
    out.register("log.jsonl", "log")
    out.finish("complete")
    m = json.loads(out.path("manifest.json").read_text())
    assert m["state"] == "complete" and [f["name"] for f in m["files"]] == ["r025.e1.dev_answers.jsonl", "r025.log.jsonl"]
    f = m["files"][0]
    assert f["bytes"] == 9 and f["sha256"] == hashlib.sha256(b'{"a": 1}\n').hexdigest() and f["kind"] == "answers"
    assert not list((tmp_path / "o").glob("*.tmp"))                                   # atomic: no temp file left
    assert json.loads(out.path("log.jsonl").read_text().splitlines()[0])["event"] == "env"


def test_the_job_script_is_standalone_and_its_arguments_parse(tmp_path):
    tree = ast.parse(Path(job.__file__).read_text())
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert top <= set(sys.stdlib_module_names) | {"__future__"}, top                  # nothing from the repo, nothing heavy at import
    r = subprocess.run([sys.executable, "-I", job.__file__, "--help"], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert r.returncode == 0 and "prefetch" in r.stdout and "train" in r.stdout
    r = subprocess.run([sys.executable, "-I", job.__file__, "train", "--run", "x", "--frac", "2"], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert r.returncode != 0 and "frac" in r.stderr
    cfg = s5.StudyConfig(name="t", root=tmp_path)
    a = job.build_parser().parse_args(s5.job_args(cfg, "r050", first=False))
    assert (a.run, a.frac, a.epochs, a.ckpt_every, a.batch, a.grad_accum, a.lr, a.max_len, a.repo, a.revision, a.qlora, a.step0_gen) == (
        "r050", 0.5, 2.0, 0.5, 1, 16, 2e-4, 4608, job.DEFAULT_REPO, job.DEFAULT_REVISION, False, False)
    assert (a.data_dir, a.out_dir, a.dev_max_new_tokens, a.probe_rows) == (s5.REMOTE["data"], s5.REMOTE["out"], 2048, 64)
    first = job.build_parser().parse_args(s5.job_args(dataclasses.replace(cfg, qlora=True), "r025", first=True))
    assert first.step0_gen is True and first.qlora is True and first.frac == 0.25


def test_the_pinned_requirements_are_the_validated_stack():
    lines = [ln for ln in s5.REQUIREMENTS.read_text().splitlines() if ln.strip()]
    pins = dict(ln.split("==") for ln in lines)
    assert len(pins) == len(lines) and all("==" in ln for ln in lines)
    assert {k: pins[k] for k in ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "bitsandbytes", "accelerate", "datasets")} == {
        "torch": "2.12.1", "transformers": "5.5.0", "trl": "0.24.0", "peft": "0.21.2", "unsloth": "2026.9.14", "unsloth_zoo": "2026.9.9",
        "bitsandbytes": "0.50.2", "accelerate": "1.15.0", "datasets": "4.3.0"}          # the stack Stage 1 trained the 4B on
    assert not {"pytest", "opencv-python-headless", "pluggy", "iniconfig"} & set(pins)
    assert s5.StudyConfig(name="t").packages() == lines               # unsloth_zoo vendors fla 0.5.1: nothing extra is installed


# ================================================================ predict

def flat_tokens(n_rows=100, per_row=1000, dev=10):
    rows = [{"kind": "solution", "family": "a", "messages": []} for _ in range(n_rows)]
    return s5.RowTokens([per_row] * n_rows, [per_row] * dev, "tokenizer"), rows


def test_predict_arithmetic_against_hand_values():
    tokens, rows = flat_tokens()
    as_ = s5.assumptions_for("A100", tok_s=1000.0)
    r = s5.predict_run(tokens, rows, 50, as_, epochs=2.0, ckpt_every=0.5, step0_gen=False)
    assert r["rows"] == 50 and r["epoch_tokens"] == 50_000 and r["train_tokens"] == 100_000 and r["steps"] == 7 and r["checkpoints"] == 4
    p = r["parts"]
    assert p["train"] == pytest.approx(100_000 / 1000 / 60)                                    # COMPUTE.md: tokens / (tokens/s)
    assert p["eval"] == pytest.approx(5 * (10_000 + 64_000) / (3 * 1000) / 60)                 # step 0 and 4 checkpoints; dev + probe rows
    assert p["generate"] == pytest.approx(4 * (2048 * 0.09 + 15) / 60)
    assert p["download"] == pytest.approx((4 * 175 / 10 + 11 * 10) / 60)                       # 4 adapters; 8 + 3 files
    assert p["load"] == 3 and p["prep"] == 1.5 and p["save"] == pytest.approx(4 * 15 / 60) and p["teardown"] == 0.5
    assert r["minutes"] == pytest.approx(sum(p.values())) and r["cu"] == pytest.approx(r["minutes"] / 60 * 5.4)
    first = s5.predict_run(tokens, rows, 50, as_, epochs=2.0, ckpt_every=0.5, step0_gen=True)
    assert first["generations"] == 5 and first["parts"]["generate"] == pytest.approx(5 * (2048 * 0.09 + 15) / 60)
    study = s5.predict_study(tokens, rows, (25, 50, 100), as_)
    assert [x["rows"] for x in study["runs"]] == [25, 50, 100] and study["runs"][0]["generations"] == 5 and study["runs"][1]["generations"] == 4
    assert study["setup"]["minutes"] == 19 and study["minutes"] == pytest.approx(19 + sum(x["minutes"] for x in study["runs"]))
    assert study["cu"] == pytest.approx(study["minutes"] / 60 * 5.4) and study["train_tok_s"] == 1000.0


def test_the_derived_throughput_is_the_compute_md_formula():
    a = s5.assumptions_for("A100")
    assert a.train_tok_s == pytest.approx(312e12 * 0.20 / (6 * 9.65e9)) and 1070 < a.train_tok_s < 1085
    assert s5.assumptions_for("A100", mfu=0.35).train_tok_s == pytest.approx(312e12 * 0.35 / (6 * 9.65e9))
    assert s5.assumptions_for("L4").cu_per_hour == 1.7 and s5.assumptions_for("H100", cu_per_hour=9.0).cu_per_hour == 9.0
    with pytest.raises(colab_mod.ConfigError, match="cu-per-hour"):
        s5.assumptions_for("H100")
    assert s5.n_checkpoints(2.0, 0.5) == 4 and s5.n_checkpoints(2.0, 1.0) == 2


def test_the_prediction_text_shows_the_cap_the_extra_round_and_the_kill_points(capsys):
    assert s5.main(["predict", "--tok-s", "1000"]) == 0
    out = capsys.readouterr().out
    for needle in ("STUDY:", "killed at 2x", "cap 60 CU", "one extra round", "r025", "r050", "r100", "LOGBOOK line", "1,000 training tokens/s",
                   "35 % utilization", "10 % utilization", "spent + 2 x its predicted CU", "in two sessions", "A (runs 25)", "B (runs 50,100"):
        assert needle in out, needle
    assert s5.main(["predict", "--tok-s", "1000", "--spent-cu", "55", "--extra-round", "100"]) == 0
    assert "DOES NOT FIT" in capsys.readouterr().out
    assert s5.main(["predict", "--tok-s", "1000", "--json", "--extra-round", "0"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["extra"] is None and data["cap_cu"] == 60.0 and [r["run"] for r in data["study"]["runs"]] == ["r025", "r050", "r100"]


def test_estimated_row_tokens_when_there_is_no_cache(tmp_path, train_rows, dev_rows):
    t = s5.load_row_tokens(train_rows[:5], dev_rows[:3], cache=tmp_path / "none.json")
    assert t.source == "estimate" and len(t.train) == 5 and all(x > 100 for x in t.train)
    cache = tmp_path / "c.json"
    cache.write_text(json.dumps({"train": [7] * 2847, "dev": [8] * 153, "train_sha256": s5.sha256_file(TRAIN), "dev_sha256": s5.sha256_file(DEV)}))
    t = s5.load_row_tokens(train_rows, dev_rows, cache=cache)
    assert t.source == "tokenizer" and t.train[0] == 7
    cache.write_text(json.dumps({"train": [7] * 2847, "dev": [8] * 153, "train_sha256": "stale", "dev_sha256": s5.sha256_file(DEV)}))
    assert s5.load_row_tokens(train_rows, dev_rows, cache=cache).source == "estimate"      # a changed data file invalidates the cache


# ================================================================ files a run leaves on the VM

def make_run_files(run: str, tags: list[str], *, tok_s=1000.0, state="complete", answers=None, frac=0.25, losses=None) -> dict[str, bytes]:
    """The flat files colab_train_job.py would have written (manifest last), as {name: bytes}."""
    files: dict[str, bytes] = {}
    cks, kinds = [], {}
    for i, tag in enumerate(tags):
        if tag != "step0":
            cks.append({"tag": tag, "step": 10 * (i + 1), "epoch": job.mean([float(tag[1:])]), "train_loss_window": 1.0 / (i + 1),
                        "probe_loss": 0.9 / (i + 1), "dev_loss": 0.8 / (i + 1), "tok_s_window": tok_s, "gen_s": 100.0, "eval_s": 20.0,
                        "peak_train_gib": 30.0, "adapter_s": 3.0, **(losses or {}).get(tag, {})})
        if answers is not None and tag in answers:
            files[f"{run}.{tag}.dev_answers.jsonl"] = "".join(json.dumps(a) + "\n" for a in answers[tag]).encode()
            kinds[f"{run}.{tag}.dev_answers.jsonl"] = "answers"
        files[f"{run}.{tag}.adapter.tar"] = f"adapter {run} {tag}".encode()
        kinds[f"{run}.{tag}.adapter.tar"] = "adapter"
    summary = {"run": run, "frac": frac, "state": state, "checkpoints": cks, "max_steps": 10 * len(tags), "n_train": int(frac * 100),
               "step0": {"tag": "step0", "step": 0, "epoch": 0.0, "dev_loss": 1.5, "probe_loss": 1.4}, "dev_loss_step0": 1.5, "load_s": 60.0,
               "train": {"tokens_per_s": tok_s, "step_s": 500.0, "steps": 10 * len(tags), "tokens_seen": int(tok_s * 500), "runtime_s": 600.0}}
    files[f"{run}.summary.json"] = json.dumps(summary).encode()
    files[f"{run}.log.jsonl"] = b"".join(json.dumps({"event": "step", "step": k, "loss": 2.0 / k}).encode() + b"\n" for k in range(1, 4))
    kinds[f"{run}.summary.json"], kinds[f"{run}.log.jsonl"] = "summary", "log"
    man = {"run": run, "state": state, "files": [{"name": n, "kind": kinds[n], "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                                for n, b in sorted(files.items())]}
    files[f"{run}.manifest.json"] = json.dumps(man).encode()
    return files


def write_run(study_dir: Path, run: str, files: dict[str, bytes]) -> None:
    d = study_dir / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (d / n).write_bytes(b)


# ================================================================ the driver, CLI mocked

def _sub(argv):
    rest = argv[1:]
    if rest and rest[0] == "--auth":
        rest = rest[2:]
    return rest[0]


class FakeColab:
    """A fake `colab`: records every command; `exec` of a run's launcher produces that run's files on a fake remote disk."""

    def __init__(self, tags=("e0.5", "e1", "e1.5", "e2"), fail=None, tok_s=None, state="complete", usage="balance 100.0 CU", answers=None):
        self.calls, self.timeouts, self.remote, self.fail, self.tok_s, self.state, self.usage = [], [], {}, dict(fail or {}), tok_s, state, usage
        self.tags, self.answers, self.launchers = tags, answers, {}

    def names(self):
        out = []
        for c in self.calls:
            s = _sub(c)
            out.append(s if s != "exec" else "exec:" + Path(c[c.index("-f") + 1]).stem.replace("launch_", ""))
        return out

    def __call__(self, argv, *, timeout, log_path=None):
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        sub = _sub(argv)
        key = sub if sub != "exec" else "exec:" + Path(argv[argv.index("-f") + 1]).stem.replace("launch_", "")
        f = self.fail.get(key, self.fail.get(sub))
        if isinstance(f, BaseException):
            raise f
        if f:
            return colab_mod.CmdResult(f, "boom")
        if sub == "upload":
            self.remote[argv[-1]] = Path(argv[-2]).read_bytes()
        elif sub == "exec":
            src = Path(argv[argv.index("-f") + 1]).read_text()
            compile(src, "launcher", "exec")
            if "launch_setup" not in argv[argv.index("-f") + 1]:
                args = ast.literal_eval(next(ln for ln in src.splitlines() if ln.startswith("ARGS = "))[len("ARGS = "):])
                run, frac = args[args.index("--run") + 1], float(args[args.index("--frac") + 1])
                self.launchers[run] = args
                tags = (["step0"] if "--step0-gen" in args else []) + list(self.tags)
                for n, b in make_run_files(run, tags, tok_s=self.tok_s or 1078.0, state=self.state, frac=frac, answers=self.answers).items():
                    self.remote[f"{s5.REMOTE['out']}/{n}"] = b
        elif sub == "download":
            if argv[-2] not in self.remote:
                return colab_mod.CmdResult(1, "no such file")
            Path(argv[-1]).write_bytes(self.remote[argv[-2]])
        return colab_mod.CmdResult(0, self.usage if sub == "usage" else "")


def make_cfg(tmp_path, **kw):
    tokens, rows = flat_tokens(n_rows=2847, per_row=1100, dev=153)
    cfg = s5.StudyConfig(name="t", root=tmp_path, percents=(25, 50), colab_bin="colab", base_license="apache-2.0", **kw)
    as_ = s5.assumptions_for(cfg.gpu, cfg.cu_per_hour)
    study = s5.predict_study(tokens, rows, cfg.percents, as_, epochs=cfg.epochs, ckpt_every=cfg.ckpt_every, step0_gen=cfg.step0_gen)
    cfg.predicted_minutes, cfg.predicted_cu = round(study["minutes"], 1), round(study["cu"], 2)
    return cfg, tokens, rows, study


def run(cfg, tokens, rows, runner=None, **kw):
    return s5.run_study(cfg, tokens=tokens, train_rows=rows, runner=runner, log=lambda m: None, **kw)


def test_dry_run_lists_every_command_with_timeouts_of_twice_the_prediction_and_writes_nothing(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    out = []
    rec = s5.run_study(cfg, dry_run=True, tokens=tokens, train_rows=rows, log=out.append)
    assert rec["dry_run"] is True
    assert rec["steps"] == ["usage-before", "new", "upload-job", "upload-train", "upload-dev", "upload-dev-prompts", "exec-setup", "exec-r025",
                            "download-r025", "exec-r050", "download-r050", "stop", "usage-after"]
    cmds = {s: c for s, c in zip(rec["steps"], rec["commands"])}
    assert cmds["new"] == ["colab", "new", "-s", "stage5-t", "--gpu", "A100"] and cmds["stop"] == ["colab", "stop", "-s", "stage5-t"]
    assert cmds["upload-job"][-1] == "/content/stage5_job.py" and cmds["upload-train"][-1] == "/content/stage5_train.jsonl"
    scale = cfg.predicted_minutes / study["minutes"]
    for name, mins in (("exec-setup", study["setup"]["minutes"]), ("exec-r025", study["runs"][0]["minutes"]), ("exec-r050", study["runs"][1]["minutes"])):
        argv = cmds[name]
        assert argv[argv.index("--timeout") + 1] == str(int(2 * 60 * mins * scale)), name       # Rule 2: twice the prediction
    text = "\n".join(out)
    assert "[finally] stop" in text and "uv" in text and "launch_r025.py" in text and "--step0-gen" in text
    assert not (cfg.path).exists()                                                              # a dry run writes nothing
    assert rec["budget"]["total_min"] == pytest.approx(cfg.predicted_minutes)
    half = dataclasses.replace(cfg, predicted_minutes=cfg.predicted_minutes / 2)
    rec2 = s5.run_study(half, dry_run=True, tokens=tokens, train_rows=rows, log=lambda m: None)
    a, b = rec["commands"][rec["steps"].index("exec-r025")], rec2["commands"][rec2["steps"].index("exec-r025")]
    assert int(b[b.index("--timeout") + 1]) == pytest.approx(int(a[a.index("--timeout") + 1]) / 2, abs=1)   # the timeouts follow the prediction


def test_guards_nothing_runs_without_approval_a_prediction_a_license_a_fitting_gpu_or_room_under_the_cap(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    fake = FakeColab()
    with pytest.raises(colab_mod.ApprovalRequired):
        run(cfg, tokens, rows, fake)
    for field in ("predicted_minutes", "predicted_cu"):
        with pytest.raises(colab_mod.ConfigError, match=field):
            run(dataclasses.replace(cfg, **{field: None}), tokens, rows, fake, approved=True)
    with pytest.raises(colab_mod.ConfigError, match="base_license"):
        run(dataclasses.replace(cfg, base_license=None), tokens, rows, fake, approved=True)
    for gpu in ("a100", "A10", ""):
        with pytest.raises(colab_mod.ConfigError, match="unknown GPU"):
            run(dataclasses.replace(cfg, gpu=gpu), tokens, rows, fake, approved=True)
    for gpu in ("L4", "T4"):
        with pytest.raises(colab_mod.ConfigError, match="16-bit LoRA"):
            run(dataclasses.replace(cfg, gpu=gpu), tokens, rows, fake, approved=True)
    s5.validate(dataclasses.replace(cfg, gpu="T4", qlora=True), real_run=False)                # the QLoRA fallback is allowed
    with pytest.raises(colab_mod.ConfigError, match="implies"):                                  # a wrong CU number would weaken the 2x kill
        run(dataclasses.replace(cfg, predicted_cu=cfg.predicted_cu / 5), tokens, rows, fake, approved=True)
    with pytest.raises(colab_mod.ConfigError, match="cap"):                                      # spent + 2 x predicted must stay under 60
        run(dataclasses.replace(cfg, spent_cu=40.0), tokens, rows, fake, approved=True)
    with pytest.raises(colab_mod.ConfigError, match="increasing percents"):
        s5.validate(dataclasses.replace(cfg, percents=(50, 25)), real_run=False)
    with pytest.raises(colab_mod.ConfigError, match="cu-per-hour"):
        s5.run_study(dataclasses.replace(cfg, gpu="H100"), dry_run=True, tokens=tokens, train_rows=rows, log=lambda m: None)
    assert fake.calls == [] and not cfg.path.exists()
    base = ["colab", "--study", "g", "--root", str(tmp_path), "--colab-bin", "/nonexistent/colab"]
    assert s5.main([*base, "--i-have-approval"]) == 2
    assert s5.main([*base, "--i-have-approval", "--predicted-minutes", "30", "--predicted-cu", "2.7", "--gpu", "L40"]) == 2
    assert s5.main([*base, "--i-have-approval", "--predicted-minutes", "30", "--predicted-cu", "2.7", "--base-license", ""]) == 2


def test_a_real_run_sequence_downloads_checks_and_stops(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    fake = FakeColab(usage="balance 100.0 CU")
    rec = run(cfg, tokens, rows, fake, approved=True)
    assert rec["state"] == "ok" and rec["stop_ok"] is True and rec["error"] is None and rec["usage_before"] == rec["usage_after"] == "balance 100.0 CU"
    names = fake.names()
    assert names[:8] == ["usage", "new", "upload", "upload", "upload", "upload", "exec:setup", "exec:r025"]
    assert names.count("exec:r050") == 1 and names.index("exec:r050") > names.index("exec:r025")
    assert names[-2:] == ["stop", "usage"] and names.count("stop") == 1
    assert fake.launchers["r025"][fake.launchers["r025"].index("--step0-gen")] == "--step0-gen" and "--step0-gen" not in fake.launchers["r050"]
    assert fake.launchers["r050"][fake.launchers["r050"].index("--frac") + 1] == "0.5" and "--qlora" not in fake.launchers["r050"]
    # the remote paths of the uploads are the ones the setup launcher copies from
    ups = {c[-1] for c in fake.calls if _sub(c) == "upload"}
    assert ups == {s5.REMOTE[k] for k in ("job", "train", "dev", "prompts")}
    setup_src = (cfg.path / "launch_setup.py").read_text()
    assert all(u in setup_src for u in ups - {s5.REMOTE["job"]}) and "flash-linear-attention" not in setup_src and "unsloth==2026.9.14" in setup_src
    # every manifest file of both runs is local with its hash
    for r in cfg.runs:
        assert s5.run_complete(cfg, r)
        assert (cfg.run_dir(r) / f"{r}.e2.adapter.tar").read_bytes() == f"adapter {r} e2".encode()
    assert (cfg.run_dir("r025") / "r025.step0.adapter.tar").is_file() and not (cfg.run_dir("r050") / "r050.step0.adapter.tar").exists()
    session = json.loads((cfg.path / "session_stage5.json").read_text())
    assert session["state"] == "ok" and session["base_license"] == "apache-2.0" and session["predicted_cu"] == cfg.predicted_cu
    assert session["per_run"]["r025"]["measured_tok_s"] == 1078.0 and session["per_run"]["r025"]["predicted_tok_s"] == pytest.approx(1077.7, abs=0.2)
    assert [c["name"] for c in session["commands"]][-2:] == ["stop", "usage-after"]
    assert (cfg.path / "study.done").is_file() and json.loads((cfg.path / "config.json").read_text())["percents"] == [25, 50]
    assert len(s5.read_jsonl(cfg.path / "dev_prompts.jsonl")) == 36
    fake2 = FakeColab()                                                                          # complete locally: no VM at all
    assert run(cfg, tokens, rows, fake2, approved=True) == {"skipped": True, "todo": []} and fake2.calls == []


@pytest.mark.parametrize("fail", [{"exec:setup": 1}, {"exec:r025": 1}, {"exec:r050": 1}, {"upload": 1}, {"new": 1}, {"exec:r025": KeyboardInterrupt()},
                                  {"exec:r025": subprocess.TimeoutExpired("colab", 1)}, {"new": subprocess.TimeoutExpired("colab", 1)},
                                  {"download": 1}])
def test_stop_runs_in_finally_whatever_fails(tmp_path, fail):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    fake = FakeColab(fail=fail)
    with pytest.raises((KeyboardInterrupt, subprocess.TimeoutExpired, colab_mod.ColabError)):
        run(cfg, tokens, rows, fake, approved=True)
    names = fake.names()
    assert names.count("stop") == 1 and names[-2:] == ["stop", "usage"]                          # stop, then the usage after
    assert names.index("stop") > names.index("new")                                              # even when `new` itself failed
    session = json.loads((cfg.path / "session_stage5.json").read_text())
    assert session["state"] == "failed" and session["stop_ok"] is True and session["error"]
    if fail == {"exec:r025": 1}:                                                                 # a failed run: what it wrote is salvaged first
        after = names[names.index("exec:r025") + 1:names.index("stop")]
        assert after and all(n == "download" for n in after)                                     # the salvage downloads, then the stop
    if fail == {"exec:r050": 1}:
        assert s5.run_complete(cfg, "r025") and not s5.run_complete(cfg, "r050")                 # the finished run is kept


def test_a_rerun_after_a_failure_only_redoes_the_incomplete_run(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    with pytest.raises(colab_mod.ColabError):
        run(cfg, tokens, rows, FakeColab(fail={"exec:r050": 1}), approved=True)
    assert s5.run_complete(cfg, "r025") and not s5.run_complete(cfg, "r050")
    fake = FakeColab()
    rec = run(cfg, tokens, rows, fake, approved=True)
    names = fake.names()
    assert rec["state"] == "ok" and "exec:r025" not in names and names.count("exec:r050") == 1
    assert s5.run_complete(cfg, "r050")
    dry = s5.run_study(cfg, dry_run=True, tokens=tokens, train_rows=rows, log=lambda m: None)
    assert dry["todo"] == []


def test_a_run_slower_than_a_third_of_the_prediction_stops_the_study_but_a_much_faster_one_only_warns(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    slow = FakeColab(tok_s=1078.0 / 3.5)
    with pytest.raises(colab_mod.ColabError, match="3.5x BELOW the prediction"):
        run(cfg, tokens, rows, slow, approved=True)
    assert "exec:r050" not in slow.names() and slow.names()[-2:] == ["stop", "usage"]             # the second run was never started
    assert s5.run_complete(cfg, "r025")                                                            # but the first run's files are kept
    cfg2, tokens, rows, _ = make_cfg(tmp_path / "b")
    logs = []
    fast = FakeColab(tok_s=1078.0 * 3.5)
    s5.run_study(cfg2, approved=True, runner=fast, tokens=tokens, train_rows=rows, log=logs.append)
    assert "exec:r050" in fast.names() and any("ABOVE the prediction" in m for m in logs)
    cfg3, tokens, rows, _ = make_cfg(tmp_path / "c", abort_when_slow=False)
    logs = []
    s5.run_study(cfg3, approved=True, runner=FakeColab(tok_s=1078.0 / 4), tokens=tokens, train_rows=rows, log=logs.append)
    assert any("BELOW the prediction" in m for m in logs)


def test_an_incomplete_manifest_or_missing_files_fail_the_run(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    with pytest.raises(colab_mod.ColabError, match="not complete"):
        run(cfg, tokens, rows, FakeColab(state="failed"), approved=True)
    cfg2, tokens, rows, _ = make_cfg(tmp_path / "x")

    class Lossy(FakeColab):
        def __call__(self, argv, *, timeout, log_path=None):
            if _sub(argv) == "download" and argv[-2].endswith(".e1.adapter.tar"):
                return colab_mod.CmdResult(1, "network")
            return super().__call__(argv, timeout=timeout, log_path=log_path)

    with pytest.raises(colab_mod.ColabError, match="not complete"):
        run(cfg2, tokens, rows, Lossy(), approved=True)
    assert not s5.run_complete(cfg2, "r025")


def test_the_session_deadline_is_twice_the_prediction(tmp_path, monkeypatch):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    clock = {"t": 1000.0}
    monkeypatch.setattr(s5.time, "monotonic", lambda: clock["t"])

    class Slow(FakeColab):
        def __call__(self, argv, *, timeout, log_path=None):
            if _sub(argv) == "exec":
                clock["t"] += 2.1 * 60 * cfg.predicted_minutes / 2                               # the first exec eats just over a quarter
            return super().__call__(argv, timeout=timeout, log_path=log_path)

    fake = Slow()
    with pytest.raises(colab_mod.ColabError, match="2x its predicted"):
        run(cfg, tokens, rows, fake, approved=True)
    assert fake.names().count("stop") == 1


def test_launchers_are_valid_python_and_carry_the_pins_and_run_arguments(tmp_path):
    cfg, tokens, rows, study = make_cfg(tmp_path)
    src = s5.setup_launcher_source(cfg)
    compile(src, "setup", "exec")
    assert "unsloth==2026.9.14" in src and '"uv", "venv", "--python", "3.12"' in src
    assert src.count("prefetch") == 1 and "check" in src and "nvidia-smi" in src
    run_src = s5.run_launcher_source(cfg, "r100", first=False)
    compile(run_src, "run", "exec")
    assert "expandable_segments:True" in run_src and "'--frac', '1.0'" in run_src and "colab_train_job.py exited" in run_src
    exec_ns: dict = {}
    pins_line = next(ln for ln in src.splitlines() if ln.startswith("PINS = "))
    exec(pins_line, exec_ns)
    assert exec_ns["PINS"] == cfg.packages() and "torch==2.12.1" in exec_ns["PINS"]


def test_exec_launcher_runs_the_job_and_fails_loudly(tmp_path):
    cfg, tokens, rows, _ = make_cfg(tmp_path)
    src = s5.run_launcher_source(cfg, "r025", first=True)
    src = src.replace(repr(s5.REMOTE["venv"] + "/bin/python"), repr(sys.executable)).replace(repr(s5.REMOTE["job"]), repr(job.__file__))
    src = src.replace(repr(s5.REMOTE["work"]), repr(str(tmp_path)))
    ok = subprocess.run([sys.executable, "-c", src.replace(next(ln for ln in src.splitlines() if ln.startswith("ARGS = ")), "ARGS = ['--help']")],
                        capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0 and "launcher: job finished OK" in ok.stdout
    bad = subprocess.run([sys.executable, "-c", src.replace(next(ln for ln in src.splitlines() if ln.startswith("ARGS = ")), "ARGS = ['train', '--run', 'x', '--frac', '9']")],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and "colab_train_job.py exited with status" in bad.stderr


def test_the_cli_dry_run_exits_zero_and_prints_the_new_command(tmp_path, capsys):
    assert s5.main(["colab", "--study", "d", "--root", str(tmp_path), "--dry-run", "--colab-bin", "/nonexistent/colab"]) == 0
    out = capsys.readouterr().out
    assert "/nonexistent/colab new -s stage5-d --gpu A100" in out and "exec-r100" in out and "setup launcher" in out
    assert not (tmp_path / "d").exists()


def _colab_cli():
    return shutil.which("colab") or (str(p) if (p := Path.home() / "venvs" / "colab" / "bin" / "colab").is_file() else None)


@pytest.mark.skipif(_colab_cli() is None, reason="the Colab CLI is not installed")
def test_every_flag_the_driver_uses_exists_in_the_installed_cli_help(tmp_path):
    cli = _colab_cli()
    cfg, tokens, rows, study = make_cfg(tmp_path)
    cfg = dataclasses.replace(cfg, colab_bin=cli, colab_auth="oauth2")
    steps = s5.plan(cfg, s5.make_budget(cfg, study))
    helps = {}
    for step in steps:
        sub = _sub(step.argv)
        if sub not in helps:
            try:
                r = subprocess.run([cli, "help", sub], capture_output=True, text=True, timeout=60)     # read-only: prints usage
            except subprocess.TimeoutExpired:  # the CLI sometimes stalls on a network check (2 min seen 2026-10-23): environment, not code
                pytest.skip(f"`colab help {sub}` did not answer within 60 s")
            assert r.returncode == 0, r.stderr
            helps[sub] = r.stdout + r.stderr
        flags = {a for a in step.argv[1:] if a.startswith("-") and a != "-" and not a.lstrip("-").isdigit()} - {"--auth"}
        for f in flags:
            assert f in helps[sub], f"`colab {sub}` has no {f}"
    top = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
    assert "--auth" in top.stdout + top.stderr


# ================================================================ the dev-answer path: download, ingest, grade, report

def _fenced(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


@pytest.fixture(scope="module")
def dev_subset():
    """A handful of real dev tasks (4 contour phrasings, 2 histogram, 1 hough) with their real prompts."""
    from bakeoff.tasks import load_tasks
    ids = ["dev-contour_area-000", "dev-contour_area-001", "dev-contour_area-004", "dev-contour_area-005", "dev-histogram_stats-000",
           "dev-histogram_stats-001", "dev-hough_segments-000"]
    tasks = {t.id: t for t in load_tasks(("dev",), only="^(" + "|".join(ids) + ")$")}
    assert set(tasks) == set(ids)
    return tasks


def _answers_for(prompts, verdict):
    """{prompt id: answer text}: verdict(task id) is 'ref' (REFERENCE) or a mutant name or 'prose' (no code)."""
    fam = {"contour_area": contour_area, "histogram_stats": histogram_stats, "hough_segments": hough_segments}
    out = {}
    for p in prompts:
        tid = p["task_ids"][0]
        f = fam[tid.split("-")[1]]
        v = verdict(tid)
        out[p["id"]] = ("I cannot help with that." if v == "prose" else _fenced(f.REFERENCE if v == "ref" else f.MUTANTS[v][1]))
    return out


def _answer_rows(by_prompt, tokens=120):
    return [{"prompt_id": pid, "text": text, "n_tokens": tokens, "finish": "stop", "prompt_tokens": 400} for pid, text in by_prompt.items()]


def test_answers_become_bakeoff_rows_one_per_dev_task(dev_subset):
    prompts = [dict(p, task_ids=[t for t in p["task_ids"] if t in dev_subset]) for p in s5.build_dev_prompts() if any(t in dev_subset for t in p["task_ids"])]
    assert sum(len(p["task_ids"]) for p in prompts) == 7 and len(prompts) == 5          # contour 000/004 and 001/005 share prompts
    by = _answers_for(prompts, lambda t: "ref")
    rows = s5.generation_rows(prompts, _answer_rows(by), dev_subset, "r025@e1", "r025/e1")
    assert len(rows) == 7 and {r["task_id"] for r in rows} == set(dev_subset)
    pair = [r for r in rows if r["task_id"] in ("dev-contour_area-000", "dev-contour_area-004")]
    assert {r["code"] for r in pair} == {contour_area.REFERENCE.strip() + "\n"} or len({r["code"] for r in pair}) == 1
    assert sorted(r["reused_from"] is None for r in pair) == [False, True]              # the second task reuses the first's answer
    r0 = rows[0]
    assert (r0["rung"], r0["mode"], r0["sample"], r0["extract"], r0["truncated"], r0["kind"], r0["held_out"]) == ("r025@e1", "greedy", 0, "fence", False, "family", False)
    assert {"task_seed", "cluster", "subset", "role", "family"} <= set(r0)
    cut = s5.generation_rows(prompts, [{**_answer_rows(by)[0], "finish": "length"}], dev_subset, "x", "m")
    assert all(r["truncated"] for r in cut)
    assert s5.rung_of("r050", "e1.5") == "r050@e1.5" and s5.rung_of("r025", "step0") == "step0"


def test_downloaded_answers_are_found_by_run_and_tag(tmp_path):
    d = tmp_path / "s"
    write_run(d, "r025", make_run_files("r025", ["step0", "e0.5", "e1"], answers={"step0": [], "e0.5": [], "e1": []}))
    write_run(d, "r100", make_run_files("r100", ["e1.5"], answers={"e1.5": []}))
    assert [(r, t) for r, t, _ in s5.answer_files(d)] == [("r025", "e0.5"), ("r025", "e1"), ("r025", "step0"), ("r100", "e1.5")]


def test_grade_and_report_the_fake_answers_references_pass_and_mutants_fail(tmp_path, dev_subset):
    study = tmp_path / "S"
    prompts = [dict(p, task_ids=[t for t in p["task_ids"] if t in dev_subset]) for p in s5.build_dev_prompts() if any(t in dev_subset for t in p["task_ids"])]
    s5.write_jsonl(study / "dev_prompts.jsonl", prompts)
    mut = {"contour": "opencv3_unpack", "hist": next(iter(histogram_stats.MUTANTS)), "hough": "n1x4_layout"}

    def untrained(tid):                                    # the untrained model: only the histogram tasks right, one reply with no code
        return "ref" if "histogram" in tid else ("prose" if tid.endswith("-000") and "contour" in tid else mut["contour"] if "contour" in tid else mut["hough"])

    def e05(tid):                                          # contour learned
        return "ref" if ("contour" in tid or "histogram" in tid) else mut["hough"]

    def e1(tid):                                           # everything right
        return "ref"

    def e2(tid):                                           # histogram regressed to a mutant
        return mut["hist"] if "histogram" in tid else "ref"

    tags = {"step0": untrained, "e0.5": e05, "e1": e1}
    answers = {t: _answer_rows(_answers_for(prompts, v)) for t, v in tags.items()}
    write_run(study, "r025", make_run_files("r025", ["step0", "e0.5", "e1"], answers=answers, frac=0.25))
    answers2 = {"e1": _answer_rows(_answers_for(prompts, e1)), "e2": _answer_rows(_answers_for(prompts, e2))}
    write_run(study, "r050", make_run_files("r050", ["e1", "e2"], answers=answers2, frac=0.5, losses={"e2": {"dev_loss": 0.7}}))

    totals = s5.grade_study(study, workers=2, tasks=dev_subset, log=lambda m: None)
    assert totals["grader_errors"] == 0 and totals["ingested"] == 7 * 5 and totals["graded"] == 7 * 5
    again = s5.grade_study(study, workers=2, tasks=dev_subset, log=lambda m: None)
    assert again["ingested"] == 0 and again["graded"] == 0                                   # resumable: nothing new, nothing regraded
    from bakeoff.store import RunDir
    grades = s5.read_jsonl(RunDir(study / "grading").grades)
    passed = {(g["rung"], g["task_id"]): g["passed"] for g in grades}
    assert passed[("step0", "dev-histogram_stats-000")] and not passed[("step0", "dev-contour_area-001")]
    assert not passed[("step0", "dev-hough_segments-000")] and not passed[("step0", "dev-contour_area-000")]
    assert all(passed[("r025@e1", t)] for t in dev_subset)                                    # the REFERENCE passes every gate
    assert [g["taxonomy"]["class"] for g in grades if g["rung"] == "step0" and g["task_id"] == "dev-contour_area-000"] == ["format"]

    s = s5.write_report(study, "S")
    by = {r["rung"]: r for r in s["rows"]}
    assert by["step0"]["pass1"] == pytest.approx(2 / 7) and by["r025@e0.5"]["pass1"] == pytest.approx(6 / 7) and by["r025@e1"]["pass1"] == 1.0
    assert by["r050@e2"]["pass1"] == pytest.approx(5 / 7) and by["r050@e1"]["pass1"] == 1.0
    assert by["step0"]["pass1_trained"] == pytest.approx(2 / 7) and by["step0"]["pass1_held_out"] is None
    assert by["r025@e1"]["dev_loss"] == pytest.approx(0.8 / 3) and by["r050@e2"]["dev_loss"] == 0.7 and by["r025@e0.5"]["train_loss"] == 0.5
    assert by["step0"]["fenced"] == by["step0"]["answers"] - 1 and by["step0"]["cut_at_cap"] == 0
    assert s["best"] == {"r025": "r025@e1", "r050": "r050@e1"}                                 # best by pass@1; a tie goes to the lower dev loss, then earlier
    pv = {p["a"]: p for p in s["paired_vs_step0"]}
    assert pv["r025@e1"]["a_only"] == 5 and pv["r025@e1"]["b_only"] == 0 and pv["r025@e1"]["diff"] == pytest.approx(5 / 7)
    assert pv["r025@e1"]["p"] == pytest.approx(2 / 32)
    conv = {(c["which"], c["a"], c["b"]): c for c in s["convergence"]}
    same = conv[("best", "r050@e1", "r025@e1")]
    assert same["diff"] == 0.0 and same["moves_more_than_noise"] is False                        # both are 7/7: nothing moved
    worse = conv[("final", "r050@e2", "r025@e1")]
    assert worse["diff"] == pytest.approx(-2 / 7) and worse["noise_se"] > 0
    # the rule is 2 SE (PLAN.md "Eval noise"), not 1: on 7 fake tasks a 2-task drop is within it
    assert worse["moves_more_than_noise"] is (abs(worse["diff"]) > 2 * worse["noise_se"])
    md = (study / "report.md").read_text()
    for needle in ("Convergence rule", "r025@e0.5", "**best**", "Paired against the untrained model", "probe loss", "held-out"):
        assert needle in md, needle
    assert (study / "summary.json").is_file() and (study / "curves.csv").read_text().splitlines()[0].startswith("rung,run,tag")
    assert s5.main(["report", "--study", "S", "--root", str(tmp_path)]) == 0


def test_pick_best_breaks_ties_by_dev_loss_then_by_the_earlier_checkpoint():
    rows = [{"tag": "step0", "pass1": 0.9, "epoch": 0.0, "dev_loss": 0.1},
            {"tag": "e0.5", "pass1": 0.5, "epoch": 0.5, "dev_loss": 0.9},
            {"tag": "e1", "pass1": 0.6, "epoch": 1.0, "dev_loss": 0.7},
            {"tag": "e1.5", "pass1": 0.6, "epoch": 1.5, "dev_loss": 0.6},
            {"tag": "e2", "pass1": 0.6, "epoch": 2.0, "dev_loss": 0.6}]
    assert s5.pick_best(rows)["tag"] == "e1.5"                                                  # step 0 is never a candidate; tie -> lower loss -> earlier
    assert s5.pick_best([{"tag": "step0", "pass1": 1.0, "epoch": 0.0}]) is None and s5.pick_best([]) is None


def test_follow_stops_once_the_driver_has_written_study_done(tmp_path, dev_subset):
    study = tmp_path / "F"
    prompts = [dict(p, task_ids=[t for t in p["task_ids"] if t in dev_subset]) for p in s5.build_dev_prompts() if any(t in dev_subset for t in p["task_ids"])][:1]
    s5.write_jsonl(study / "dev_prompts.jsonl", prompts)
    write_run(study, "r025", make_run_files("r025", ["e1"], answers={"e1": _answer_rows(_answers_for(prompts, lambda t: "ref"))}))
    (study / "study.done").write_text("x\n")
    t = s5.grade_study(study, workers=1, follow=True, poll_s=0.01, tasks=dev_subset, log=lambda m: None)
    assert t["ingested"] == len(prompts[0]["task_ids"]) and t["grader_errors"] == 0


def test_build_dev_prompts_covers_the_99_dev_tasks_with_the_serve_time_system_prompt():
    prompts = s5.build_dev_prompts()
    assert len(prompts) == 36 and sum(len(p["task_ids"]) for p in prompts) == 99
    assert len({p["id"] for p in prompts}) == 36 and all(p["messages"][0] == {"role": "system", "content": SYSTEM} for p in prompts)
    assert all(p["messages"][1]["role"] == "user" and p["messages"][1]["content"].startswith("Task:") for p in prompts)
    for p in prompts:
        job.check_prompt_template(FakeTok(), p["messages"][0]["content"], p["messages"][1]["content"])


def test_the_adapter_tar_holds_the_two_adapter_files_and_the_work_folder_is_removed(tmp_path):
    import tarfile

    class FakeModel:
        def save_pretrained(self, path):
            (Path(path) / "adapter_model.safetensors").write_bytes(b"weights")
            (Path(path) / "adapter_config.json").write_text("{}")
            (Path(path) / "README.md").write_text("not shipped")

    dest = tmp_path / "r025.e1.adapter.tar"
    job.save_adapter_tar(FakeModel(), dest, tmp_path / "work")
    with tarfile.open(dest) as tf:
        assert sorted(tf.getnames()) == ["adapter_config.json", "adapter_model.safetensors"]
        assert tf.extractfile("adapter_model.safetensors").read() == b"weights"
    assert not (tmp_path / "work").exists()
    assert [job.adapter_tag(f"/x/{n}") for n in ("r025.e1.adapter.tar", "r025.e0.5.adapter.tar", "r100.step0.adapter.tar")] == ["e1", "e0.5", "step0"]


def test_an_unhashed_live_log_does_not_make_a_finished_run_look_incomplete(tmp_path):
    # Found in Stage 5 session A (2026-10-22): the job lists log.jsonl with no hash, and the driver called the
    # finished run incomplete, which would stop a multi-run session after its first run.
    import json
    from train import stage5
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "x.log.jsonl").write_text('{"event": "done"}\n')
    assert stage5._have(tmp_path / "x.log.jsonl", {"name": "x.log.jsonl", "sha256": None})
    assert stage5._have(tmp_path / "a.json", {"name": "a.json", "sha256": stage5.sha256_file(tmp_path / "a.json")})
    assert not stage5._have(tmp_path / "a.json", {"name": "a.json", "sha256": "0" * 64})
    assert not stage5._have(tmp_path / "missing.log.jsonl", {"name": "missing.log.jsonl", "sha256": None})
    # the real case: the manifest hashed the log before the job appended its "done" line
    assert stage5._have(tmp_path / "x.log.jsonl", {"name": "x.log.jsonl", "kind": "log", "sha256": "0" * 64})
