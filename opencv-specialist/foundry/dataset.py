"""The training set: verified runs + a replay file -> data/train.jsonl and data/dev.jsonl (Gate 4a decisions 3-7).

    python -m foundry.run build-dataset --run full1 [--run other ...] [--replay FILE | --no-replay] [--out data] [--total 3000]
    python -m foundry.run tokens        [--data data]       token lengths only (needs transformers: ~/venvs/train/bin/python)

Three kinds of example, mixed to a target (default 70 % solutions, 20 % repairs, 10 % replay of a total of 3,000):

    solution   system (``train.config.SYSTEM``, never the notes), the task prompt, the verified answer. Tier A only: every gate
               of the family graders, or the drill's grade (foundry.specs: a drill has no metamorphic gate). At most 2 per
               spec, distinct by AST (foundry.verify.select_examples), taken from the answer part of the reply (never the
               reasoning of a thinking teacher).
    repair     the same prompt, the failed code, the grader's message, the verified code (foundry.repair.build_pairing_pairs: one
               per spec, by pairing, no teacher repair round). Train on the last assistant turn only (``train_on`` says so). Under
               the Qwen3.5 template only the LAST assistant turn carries the empty think block that train_lora's response marker
               (``answer_prefix``) starts with, so ``train_on_responses_only`` skips the failed code by itself; the token report
               counts the rows in which the marker is not there exactly once (it must be none).
    replay    rows of a general-code file (``messages`` as a chat, ``source``, ``license``), single-turn only, the system prompt
               replaced by SYSTEM (Rule 7: the system prompt is the same in every example).

The answer of solution and repair examples is the fenced format of foundry.answer.format_answer: a fenced python block with
the imports and ``solve``, no prose. A completion whose code is not a clean module (a demo call, an ``if __name__`` block, a
reply cut off by the token budget) is not used (foundry.answer.row_problems): training targets are the teacher's own output.

Pipeline, in this order, each stage counted in the manifest:

    collect   Tier-A solutions and pairs of every run, replay rows; a held-out family anywhere is an error, not a filter
    decontaminate   drop an example whose first user message (its task line) is a scored prompt (sha1) or overlaps a dev/test/gotcha
              task line or a held-out family's phrasing at the phrasing screen's measure (foundry.phrasings: word 5-gram Jaccard
              >= DUP_THRESHOLD, or a shared run of RUN_LIMIT words); replay rows that mention a held-out family's topic
    dedup     globally by AST-normalized code (foundry.astnorm): a solution by (prompt, code), a repair by (prompt, failed code,
              fixed code), a replay row by (prompt, answer); the first in run order wins
    mix       per kind, a seeded round-robin over distinct prompts, so a prompt that many specs share does not crowd out the rest
    split     about ``dev_frac`` of the examples go to dev, chosen by distinct prompt: no prompt is in both files, whatever its kinds

Rows are flat: ``id``, ``kind``, ``messages`` (system, user, assistant[, user, assistant]), ``train_on``, and scalar provenance
(``source``, ``license``, ``run``, ``spec_id``, ``family``, ``drill``, ``tier``, ``grades``, ``teacher``, ``teacher_revision``,
``teacher_license``, ``thinking``, ``sample``, ``scene_seed``, ``phrasing``, ``prompt_sha1``, ``ast_sha1``, ``failed_ast_sha1``,
``failure_class``, ``failure_rule``, ``completion_tokens``) with None where a field does not apply, so ``datasets`` reads one
schema. ``manifest.json`` holds the SHA-256, size and row count of each file, the counts of every stage, the mix actually
reached, and the token lengths under the base's own chat template; each file is asserted under 50 MB (Rule 8).

Token lengths use ``train.config.render_turn`` conventions (the base's answer prefix, the answer stripped, the end token): a
repair example is rendered with its earlier assistant turn plain, as the Qwen3.5 template and the Ollama TEMPLATE do (measured
2026-10-23: only the last assistant turn carries the empty think block). The p99 of the lengths sets Stage 5's MAX_LEN.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from foundry.answer import format_answer, row_problems
from foundry.astnorm import ast_hash
from foundry.store import FoundryRun, read_jsonl, teacher_info
from foundry.verify import FOUR, select_examples
from train.config import BASES, END, SYSTEM, WORK

REPO_DATA = Path(__file__).resolve().parent.parent / "data"
# The general-code replay set (600 rows of nvidia/OpenCodeInstruct, CC-BY-4.0, answers by Qwen2.5-Coder-32B-Instruct, Apache-2.0):
# {"messages": [user, assistant], "source": "nvidia/OpenCodeInstruct@<commit>", "license": "CC-BY-4.0", "id"}; its README.md next to it
# records the provenance and the filters. ``build-dataset`` uses it unless told otherwise.
DEFAULT_REPLAY = WORK / "replay" / "replay.jsonl"


def default_replay() -> Path | None:
    return DEFAULT_REPLAY if DEFAULT_REPLAY.is_file() else None
MAX_FILE_BYTES = 50_000_000      # CLAUDE.md Rule 8
KINDS = ("solution", "repair", "replay")
DEFAULT_SHARES = {"solution": 0.70, "repair": 0.20, "replay": 0.10}
HELD_OUT_TOPICS = re.compile(r"chessboard|\binrange\b|[a-z]*2hsv|\bhsv\b|\bhue\b", re.I)   # color_segmentation and chessboard_corners
ROW_KEYS = ("source", "license", "run", "spec_id", "family", "drill", "tier", "grades", "teacher", "teacher_revision",
            "teacher_license", "thinking", "sample", "scene_seed", "phrasing", "prompt_sha1", "ast_sha1", "failed_ast_sha1",
            "failure_class", "failure_rule", "completion_tokens")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Example:
    id: str
    kind: str                 # solution | repair | replay
    messages: list[dict]      # system, user, assistant[, user, assistant]
    key: tuple                # global dedup identity
    meta: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.messages[1]["content"]

    @property
    def prompt_sha1(self) -> str:
        return sha1(self.prompt)

    def row(self) -> dict:
        meta = {k: self.meta.get(k) for k in ROW_KEYS}
        meta["prompt_sha1"] = self.prompt_sha1
        return {"id": self.id, "kind": self.kind, "messages": self.messages, "train_on": "last_assistant", **meta}


# ---------------------------------------------------------------- collect

def grades_summary(g: dict) -> str:
    if g.get("kind") == "drill":
        return f"drill-grade:{g.get('stage') or 'ok'}"
    gates = g.get("gates") or {}
    return "+".join(name for name in FOUR if gates.get(name, {}).get("passed")) or "none"


def _meta(run: FoundryRun, teacher: dict, spec: dict, g: dict) -> dict:
    return {"source": "foundry-teacher", "license": teacher.get("license"), "run": run.name, "spec_id": g["spec_id"],
            "family": g["family"], "drill": spec.get("drill") or None, "tier": g["tier"], "grades": grades_summary(g),
            "teacher": teacher.get("model"), "teacher_revision": teacher.get("revision"), "teacher_license": teacher.get("license"),
            "thinking": (g.get("thinking_tokens") or 0) > 0, "sample": g["sample"],
            "scene_seed": None if spec.get("kind") == "drill" else spec.get("seed"), "phrasing": spec.get("phrasing_index"),
            "ast_sha1": g["ast_sha1"], "completion_tokens": g.get("completion_tokens")}


def collect_run(run: FoundryRun, *, cap: int = 2) -> tuple[list[Example], list[Example], dict]:
    """(solutions, repairs, stats) of one verified run. A held-out family in the run is an error."""
    from cvbench.splits import FAMILIES
    from foundry.repair import build_pairing_pairs

    gen = run.stage("generate")
    grades = read_jsonl(gen.grades)
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    if not grades or not specs:
        raise SystemExit(f"{run.path}: grades.jsonl or specs.jsonl missing: run `verify` first")
    for s in specs.values():
        if s.get("kind", "family") == "family" and FAMILIES[s["family"]].held_out:
            raise SystemExit(f"{run.path}: spec {s['id']} is of held-out family {s['family']}: held-out families never enter the data")
    teacher = teacher_info(run)
    stats = {"run": run.name, "grades": len(grades), "tier_A": sum(1 for g in grades if g["tier"] == "A"),
             "format_rejects": Counter(), "teacher": teacher}
    ok = []
    for g in grades:
        if g["tier"] != "A":
            continue
        problems = row_problems(g)
        if problems:
            stats["format_rejects"][problems[0].split(" (")[0]] += 1
        else:
            ok.append(g)
    kept, sel = select_examples(ok, cap, "A")
    stats["selection"] = sel
    solutions = []
    for g in kept:
        spec = specs[g["spec_id"]]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec["prompt"]},
                {"role": "assistant", "content": format_answer(g["code"])}]
        meta = _meta(run, teacher, spec, g)
        solutions.append(Example(f"{run.name}/{g['spec_id']}#{g['rank']}", "solution", msgs,
                                 ("sol", sha1(spec["prompt"]), g["ast_sha1"]), meta))
    repairs = []
    for p in build_pairing_pairs(run):
        spec = specs[p["spec_id"]]
        good = next(g for g in grades if g["spec_id"] == p["spec_id"] and g["sample"] == p["fixed_sample"] and g["tier"] == "A")
        meta = _meta(run, teacher, spec, good)
        meta.update(failed_ast_sha1=p["failed_ast_sha1"], failure_class=p["failure"]["class"], failure_rule=p["failure"]["rule"],
                    ast_sha1=p["fixed_ast_sha1"], sample=p["fixed_sample"])
        repairs.append(Example(f"{run.name}/{p['spec_id']}/pair", "repair", p["messages"],
                               ("fix", sha1(spec["prompt"]), p["failed_ast_sha1"], p["fixed_ast_sha1"]), meta))
    stats["solutions"], stats["repairs"] = len(solutions), len(repairs)
    stats["format_rejects"] = dict(stats["format_rejects"])
    return solutions, repairs, stats


def load_replay(path: Path) -> tuple[list[Example], dict]:
    """Single-turn replay rows ({"messages", "source", "license"}) as examples with the training system prompt."""
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path}: missing or empty")
    stats = Counter()
    out = []
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        src, lic = r.get("source"), r.get("license")
        if not isinstance(msgs, list) or not src or not lic:
            raise SystemExit(f"{path}:{i + 1}: a replay row needs `messages`, `source` and `license` (CLAUDE.md Rule 5 records the "
                             "license of every dataset)")
        chat = [m for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
        if [m.get("role") for m in chat] != ["user", "assistant"] or not all(isinstance(m.get("content"), str) and m["content"].strip() for m in chat):
            stats["not_single_turn"] += 1
            continue
        if "<think>" in chat[1]["content"] or "</think>" in chat[1]["content"]:
            stats["has_reasoning"] += 1
            continue
        text = chat[0]["content"] + "\n" + chat[1]["content"]
        if HELD_OUT_TOPICS.search(text):
            stats["held_out_topic"] += 1
            continue
        answer = chat[1]["content"]
        code = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
        digest = ast_hash(code.group(1)) if code else sha1(" ".join(answer.split()))
        msgs2 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": chat[0]["content"]},
                 {"role": "assistant", "content": answer}]
        meta = {"source": src, "license": lic, "run": None, "spec_id": None, "family": None, "tier": None, "grades": None,
                "ast_sha1": digest, "teacher": r.get("teacher")}
        out.append(Example(f"replay/{sha1(chat[0]['content'] + chr(0) + answer)[:12]}", "replay", msgs2,
                           ("rep", sha1(chat[0]["content"]), digest), meta))
    stats["rows"], stats["kept"] = len(rows), len(out)
    return out, dict(stats)


# ---------------------------------------------------------------- decontaminate

TASK_LINE = re.compile(r"\ATask: (.*?)\nOpenCV:", re.S)


def task_text(prompt: str) -> str:
    m = TASK_LINE.match(prompt)
    return m.group(1) if m else prompt


class Decontaminator:
    """Drops examples whose task is a scored one. Built once (it renders every dev/test/gotcha/held-out prompt)."""

    def __init__(self, dup_threshold: float | None = None, run_limit: int | None = None):
        import dataclasses

        from cvbench.splits import FAMILIES, PHRASINGS_PER_FAMILY
        from foundry import phrasings as phr
        from foundry.specs import reserved_prompt_hashes

        self.phr = phr
        self.dup_threshold = phr.DUP_THRESHOLD if dup_threshold is None else dup_threshold
        self.run_limit = phr.RUN_LIMIT if run_limit is None else run_limit
        self.lines = phr.scored_lines()
        self.hashes = dict(reserved_prompt_hashes())      # dev, test and gotcha prompts
        for name, row in FAMILIES.items():                # every prompt of a held-out family: never trained on
            if row.held_out:
                fam = phr.family_module(name)
                for i in range(PHRASINGS_PER_FAMILY):
                    self.hashes[sha1(dataclasses.replace(fam.SPEC, task=fam.PHRASINGS[i]).to_prompt())] = f"{name}[{i}]"

    def reason(self, e: Example) -> str | None:
        prompt = e.prompt
        h = sha1(prompt)
        if h in self.hashes:
            return f"is the prompt of {self.hashes[h]}"
        fam = e.meta.get("family")
        task = task_text(prompt)
        if fam in self.phr.CONTENT_CHECKS:
            task = self.phr.core_of(fam, task)         # the shared accuracy sentence is not part of the comparison
        cand = self.phr.make_line("example", task)
        for other in self.lines:
            why = self.phr.closeness(cand, other, dup_threshold=self.dup_threshold, run_limit=self.run_limit)
            if why:
                return why
        return None


def decontaminate(examples: list[Example], dec: Decontaminator) -> tuple[list[Example], dict]:
    kept, dropped = [], Counter()
    sample: dict[str, str] = {}
    for e in examples:
        why = dec.reason(e)
        if why:
            dropped[e.kind] += 1
            sample.setdefault(e.kind, f"{e.id}: {why}")
        else:
            kept.append(e)
    return kept, {"checked": len(examples), "dropped": dict(dropped), "first_drop": sample, "dup_threshold": dec.dup_threshold,
                  "run_limit": dec.run_limit}


# ---------------------------------------------------------------- dedup, mix, split

def dedup(examples: list[Example]) -> tuple[list[Example], dict]:
    seen, kept, dropped = set(), [], Counter()
    for e in examples:
        if e.key in seen:
            dropped[e.kind] += 1
        else:
            seen.add(e.key)
            kept.append(e)
    return kept, {"dropped": dict(dropped)}


def balanced_pick(items: list[Example], n: int, seed: int) -> list[Example]:
    """``n`` of ``items``: a round-robin over distinct prompts (their order shuffled by ``seed``), the r-th example of each
    prompt in round r, so no prompt supplies a second example before every prompt has supplied its first."""
    if n >= len(items):
        return list(items)
    by_prompt: dict[str, list[Example]] = defaultdict(list)
    for e in items:
        by_prompt[e.prompt_sha1].append(e)
    for v in by_prompt.values():
        v.sort(key=lambda e: e.id)
    order = sorted(by_prompt)
    random.Random(seed).shuffle(order)
    picked: list[Example] = []
    r = 0
    while len(picked) < n:
        took = False
        for p in order:
            if r < len(by_prompt[p]):
                picked.append(by_prompt[p][r])
                took = True
                if len(picked) == n:
                    break
        if not took:
            break
        r += 1
    return picked


def select_mix(by_kind: dict[str, list[Example]], total: int | None, shares: dict[str, float], seed: int,
               keep_shares: bool = False) -> tuple[list[Example], dict]:
    """Per kind, ``total * share`` examples. ``keep_shares``: when a kind is short, shrink ``total`` until every kind can meet its
    share (so the mix stays 70 / 20 / 10 at a smaller size, instead of replay taking a larger share of what is left)."""
    if keep_shares and total is not None:
        fit = min((len(by_kind.get(k, [])) / shares[k] for k in KINDS if shares.get(k, 0) > 0), default=total)
        total = min(total, int(fit))
    if total is None:
        return [e for k in KINDS for e in by_kind.get(k, [])], {k: {"supply": len(by_kind.get(k, [])), "target": None,
                                                                   "selected": len(by_kind.get(k, []))} for k in KINDS}
    out, report = [], {}
    for k in KINDS:
        items = sorted(by_kind.get(k, []), key=lambda e: e.id)
        target = round(total * shares.get(k, 0.0))
        got = balanced_pick(items, target, seed)
        report[k] = {"supply": len(items), "target": target, "selected": len(got), "shortfall": max(0, target - len(got)), "total": total}
        out += got
    return out, report


def split_dev(examples: list[Example], frac: float, seed: int) -> tuple[list[Example], list[Example]]:
    """(train, dev): whole prompts go to dev until it holds about ``frac`` of the examples; a prompt never straddles the split."""
    by_prompt: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_prompt[e.prompt_sha1].append(e)
    order = sorted(by_prompt)
    random.Random(seed).shuffle(order)
    quota = round(frac * len(examples))
    dev_prompts, n = set(), 0
    for p in order:
        if n >= quota:
            break
        if n + len(by_prompt[p]) <= quota * 1.25 + 1:      # do not overshoot the quota by a big prompt group
            dev_prompts.add(p)
            n += len(by_prompt[p])
    train = [e for e in examples if e.prompt_sha1 not in dev_prompts]
    dev = [e for e in examples if e.prompt_sha1 in dev_prompts]
    return train, dev


# ---------------------------------------------------------------- tokens

def render_chat(base: str, messages: list[dict]) -> str:
    """The text the base's chat template makes of a training conversation (system, user, [assistant, user]*, assistant):
    earlier assistant turns plain, the last one behind the base's answer prefix, stripped when the template strips."""
    roles = [m["role"] for m in messages]
    ok = roles[:2] == ["system", "user"] and roles[-1] == "assistant" and len(roles) % 2 == 1 and \
        all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles[1:]))
    if not ok:
        raise ValueError(f"not a system/user/assistant conversation: {roles}")
    text = f"<|im_start|>system\n{messages[0]['content']}{END}\n"
    for m in messages[1:-1]:
        text += f"<|im_start|>{m['role']}\n{m['content']}{END}\n"
    last = messages[-1]["content"]
    return text + BASES[base]["answer_prefix"] + (last.strip() if BASES[base]["trim_answer"] else last) + END + "\n"


def _pct(sorted_lengths: list[int], q: float) -> int:
    return sorted_lengths[min(len(sorted_lengths) - 1, int(q * len(sorted_lengths)))]


def length_stats(lengths: list[int]) -> dict:
    s = sorted(lengths)
    return {"n": len(s), "mean": round(sum(s) / len(s), 1), "p50": _pct(s, 0.5), "p99": _pct(s, 0.99), "max": s[-1]} if s else {"n": 0}


def token_report(rows: list[dict], tokenizer, base: str = "qwen3.5-9b") -> dict:
    """Token lengths of ``rows`` under the base's chat template: overall and per kind, and how many rows the tokenizer's own
    ``apply_chat_template`` renders differently from ``render_chat`` (must be 0: Rule 7)."""
    lengths, by_kind, mismatches, marker = [], defaultdict(list), 0, 0
    prefix = BASES[base]["answer_prefix"]
    for r in rows:
        text = render_chat(base, r["messages"])
        if tokenizer.apply_chat_template(r["messages"], tokenize=False) != text:
            mismatches += 1
        if text.count(prefix) != 1:
            marker += 1
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        lengths.append(n)
        by_kind[r["kind"]].append(n)
    return {"base": base, "tokenizer": BASES[base]["upstream"], **length_stats(lengths), "template_mismatches": mismatches,
            "rows_without_exactly_one_response_marker": marker,
            "by_kind": {k: length_stats(v) for k, v in sorted(by_kind.items())},
            "note": "p99 sets Stage 5's MAX_LEN (train_lora refuses to truncate: it stops when the longest example exceeds it)"}


def load_tokenizer(base: str = "qwen3.5-9b"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASES[base]["upstream"], local_files_only=True)


# ---------------------------------------------------------------- write

def write_rows(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    size = path.stat().st_size
    if size >= MAX_FILE_BYTES:
        raise SystemExit(f"{path} is {size} bytes, over the 50 MB limit of CLAUDE.md Rule 8: commit a manifest with SHA-256 hashes instead")
    return {"sha256": sha256_file(path), "bytes": size, "rows": len(rows)}


def composition(rows: list[dict]) -> dict:
    kinds = Counter(r["kind"] for r in rows)
    n = len(rows) or 1
    return {"rows": len(rows), "by_kind": dict(kinds), "share_by_kind": {k: round(v / n, 4) for k, v in kinds.items()},
            "by_family": dict(Counter(r["family"] or "(replay)" for r in rows)),
            "by_kind_family": {k: dict(Counter(r["family"] or "(replay)" for r in rows if r["kind"] == k)) for k in kinds},
            "distinct_prompts": len({r["prompt_sha1"] for r in rows}), "thinking_rows": sum(1 for r in rows if r.get("thinking")),
            "sources": dict(Counter(f"{r['source']} ({r['license']})" for r in rows))}


def write_manifest(out_dir: Path, extra: dict | None = None) -> Path:
    """manifest.json: SHA-256, size and row count of every data file in ``out_dir`` we wrote, plus ``extra`` (counts, tokens)."""
    path = out_dir / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    files = {}
    for name in ("train.jsonl", "dev.jsonl"):
        f = out_dir / name
        if f.is_file():
            size = f.stat().st_size
            if size >= MAX_FILE_BYTES:
                raise SystemExit(f"{f} is {size} bytes, over the 50 MB limit of CLAUDE.md Rule 8")
            files[name] = {"sha256": sha256_file(f), "bytes": size, "rows": sum(1 for _ in f.open("rb"))}
    manifest = {**old, **(extra or {}), "files": files, "max_file_bytes": MAX_FILE_BYTES}
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- the build

def parse_shares(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_SHARES)
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3 or any(p < 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise SystemExit("--shares must be three non-negative numbers that sum to 1: solutions,repairs,replay")
    return dict(zip(KINDS, parts))


def build_dataset(runs: list[str], *, root: str | None = None, replay: Path | None = None, out_dir: Path = REPO_DATA,
                  total: int | None = 3000, shares: dict[str, float] | None = None, dev_frac: float = 0.05, seed: int = 0,
                  cap: int = 2, base: str = "qwen3.5-9b", tokens: str = "auto", dry_run: bool = False, keep_shares: bool = False,
                  decontaminator: Decontaminator | None = None, tokenizer=None, log=lambda m: print(m, file=sys.stderr)) -> dict:
    """Build the training set (module docstring). Returns the manifest dict (written unless ``dry_run``)."""
    shares = shares or dict(DEFAULT_SHARES)
    cands: dict[str, list[Example]] = {k: [] for k in KINDS}
    run_stats = []
    for name in runs:
        sol, rep, st = collect_run(FoundryRun.of(name, root), cap=cap)
        cands["solution"] += sol
        cands["repair"] += rep
        run_stats.append(st)
        log(f"{name}: {len(sol)} solutions, {len(rep)} repair pairs; format rejects {st['format_rejects'] or 0}")
    replay_stats = None
    if replay is not None:
        rep_ex, replay_stats = load_replay(replay)
        cands["replay"] += rep_ex
        log(f"replay {replay}: {replay_stats}")
    collected = {k: len(v) for k, v in cands.items()}

    dec = decontaminator or Decontaminator()
    dec_stats = {}
    for k in KINDS:
        cands[k], dec_stats[k] = decontaminate(cands[k], dec)
    everything, dedup_stats = dedup([e for k in KINDS for e in cands[k]])     # global: across kinds and runs, keys carry the kind
    cands = {k: [e for e in everything if e.kind == k] for k in KINDS}
    after_dedup = {k: len(v) for k, v in cands.items()}

    selected, mix = select_mix(cands, total, shares, seed, keep_shares)
    train, dev = split_dev(selected, dev_frac, seed)
    rnd = random.Random(seed + 1)
    train_rows, dev_rows = [e.row() for e in train], [e.row() for e in dev]
    rnd.shuffle(train_rows)
    rnd.shuffle(dev_rows)
    assert not {r["prompt_sha1"] for r in train_rows} & {r["prompt_sha1"] for r in dev_rows}, "a prompt is in both train and dev"

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": {"runs": runs, "root": root, "replay": str(replay) if replay else None, "total": total, "shares": shares,
                    "dev_frac": dev_frac, "seed": seed, "cap": cap, "base": base, "keep_shares": keep_shares, "system_sha1": sha1(SYSTEM)},
        "inputs": {"runs": run_stats,
                   "replay": None if replay is None else {"path": str(replay), "sha256": sha256_file(replay), **(replay_stats or {})}},
        "stages": {"collected": collected, "decontamination": dec_stats, "dedup": dedup_stats, "after_dedup": after_dedup, "mix": mix},
        "composition": {"all": composition(train_rows + dev_rows), "train": composition(train_rows), "dev": composition(dev_rows)},
        "dev": {"rows": len(dev_rows), "share": round(len(dev_rows) / max(1, len(train_rows) + len(dev_rows)), 4),
                "distinct_prompts": len({r["prompt_sha1"] for r in dev_rows})},
    }
    if tokens != "no":
        tk = tokenizer
        if tk is None:
            try:
                tk = load_tokenizer(base)
            except Exception as e:  # noqa: BLE001 - transformers missing or the tokenizer not cached
                if tokens == "yes":
                    raise SystemExit(f"--tokens yes, but the tokenizer cannot be loaded: {e!r}")
                log(f"token lengths skipped ({e!r}); run `python -m foundry.run tokens` with ~/venvs/train/bin/python")
        if tk is not None:
            manifest["tokens"] = {"train": token_report(train_rows, tk, base), "dev": token_report(dev_rows, tk, base) if dev_rows else None}
    if dry_run:
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "train.jsonl", train_rows)
    write_rows(out_dir / "dev.jsonl", dev_rows)
    write_manifest(out_dir, manifest)
    return manifest


def tokens_for_files(data_dir: Path, base: str = "qwen3.5-9b", tokenizer=None) -> dict:
    """Token lengths of data/train.jsonl and dev.jsonl, written into manifest.json (for a build made without transformers)."""
    tk = tokenizer or load_tokenizer(base)
    report = {}
    for name in ("train", "dev"):
        rows = read_jsonl(data_dir / f"{name}.jsonl")
        report[name] = token_report(rows, tk, base) if rows else None
    write_manifest(data_dir, {"tokens": report})
    return report


def render_summary(m: dict) -> str:
    """Console summary of a build."""
    L = []
    st = m["stages"]
    L.append(f"collected {st['collected']} -> decontaminated {({k: v['checked'] - sum(v['dropped'].values()) for k, v in st['decontamination'].items()})} "
             f"-> dedup {st['after_dedup']}")
    for k, v in st["mix"].items():
        L.append(f"  {k:<9} supply {v['supply']:>5}  target {v['target'] if v['target'] is not None else '-':>5}  selected {v['selected']:>5}"
                 + (f"  SHORT by {v['shortfall']}" if v.get("shortfall") else ""))
    c = m["composition"]["all"]
    L.append(f"{c['rows']} examples: " + ", ".join(f"{k} {v} ({100 * c['share_by_kind'][k]:.1f} %)" for k, v in c["by_kind"].items())
             + f"; {c['distinct_prompts']} distinct prompts; {c['thinking_rows']} from thinking-mode samples")
    L.append(f"dev {m['dev']['rows']} rows ({100 * m['dev']['share']:.1f} %), {m['dev']['distinct_prompts']} prompts; train {m['composition']['train']['rows']} rows")
    L.append("by family: " + ", ".join(f"{k} {v}" for k, v in sorted(c["by_family"].items())))
    t = (m.get("tokens") or {}).get("train")
    if t:
        L.append(f"tokens (train, {t['tokenizer']} template): n {t['n']} mean {t['mean']} p50 {t['p50']} p99 {t['p99']} max {t['max']}; "
                 f"template mismatches {t['template_mismatches']}; rows without exactly one response marker "
                 f"{t['rows_without_exactly_one_response_marker']}")
    return "\n".join(L)
