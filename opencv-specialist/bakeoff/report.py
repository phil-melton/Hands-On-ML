"""Report: generations + grades -> summary.json, report.md and audit.md.

What it computes (PLAN.md "Baseline bake-off", "Go/no-go", "Eval noise"):

    pass@1 at T=0           greedy reply, binomial SE
    unbiased pass@k, T=0.8  Chen et al. 2021 from the n sampled replies per task, bootstrap SE over tasks
    groups                  overall, families vs gotchas, subset 5.0 vs timeless, trained vs held-out families
    per family              the 9 families and the 5 gotcha groups
    error taxonomy          bakeoff.taxonomy over the failing greedy and sampled replies
    paired comparison       exact McNemar on pass@1 and the paired difference with its SE, for every rung pair
    throughput              decode tok/s (median, p10, p90; warm-up excluded), truncations, wall time

Beside each standard SE it gives a cluster-aware one. Tasks of one family share prompts (72 distinct prompts in the
198 test tasks), and at T=0 identical prompts get identical replies, so the plain binomial SE treats about three
copies of one outcome as independent draws. The cluster SE resamples whole prompts and is the one to read when the
two disagree.

Taxonomy is recomputed here from the raw gate details, so a rule change applies to grades made earlier.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from bakeoff.rungs import RUNGS
from bakeoff.stats import (binomial_se, bootstrap_se, cluster_se, pass_at_k, paired_compare, percentile)
from bakeoff.store import RunDir, read_json, read_jsonl, write_json
from bakeoff.taxonomy import CLASSES, classify, sample_failures

GROUPS: list[tuple[str, str, object]] = [  # (id, description, predicate over task meta)
    ("all", "every task", lambda m: True),
    ("families", "the 9 cvbench families", lambda m: m["kind"] == "family"),
    ("gotchas", "the 40 gotcha checks (incl. controls)", lambda m: m["kind"] == "gotcha"),
    ("5.0", "subset 5.0 (families and gotchas)", lambda m: m["subset"] == "5.0"),
    ("timeless", "subset timeless (families and gotchas)", lambda m: m["subset"] == "timeless"),
    ("families/5.0", "families, subset 5.0", lambda m: m["kind"] == "family" and m["subset"] == "5.0"),
    ("families/timeless", "families, subset timeless", lambda m: m["kind"] == "family" and m["subset"] == "timeless"),
    ("gotchas/5.0", "gotchas (role gotcha), subset 5.0", lambda m: m["role"] == "gotcha" and m["subset"] == "5.0"),
    ("gotchas/timeless", "gotchas (role gotcha), subset timeless", lambda m: m["role"] == "gotcha" and m["subset"] == "timeless"),
    ("controls", "gotcha controls (behaviour that did not change)", lambda m: m["role"] == "control"),
    ("trained families", "families that may be training data", lambda m: m["kind"] == "family" and not m["held_out"]),
    ("held-out families", "families never trained on", lambda m: m["kind"] == "family" and m["held_out"]),
]
PAIRED_GROUPS = ("all", "5.0", "timeless", "families", "gotchas", "held-out families")


def _nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def clean(obj):
    """NaN -> None so the summary is valid JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return None if _nan(obj) else obj


class Outcomes:
    """Per-rung, per-task pass/fail from the grade rows."""

    def __init__(self, grades: list[dict]):
        self.meta: dict[str, dict] = {}
        self.greedy: dict[str, dict[str, bool]] = {}
        self.samples: dict[str, dict[str, list[bool]]] = {}
        for r in grades:
            self.meta.setdefault(r["task_id"], {k: r[k] for k in ("kind", "family", "subset", "held_out", "role", "cluster")})
            if r["mode"] == "greedy":
                self.greedy.setdefault(r["rung"], {}).setdefault(r["task_id"], bool(r["passed"]))
            else:
                self.samples.setdefault(r["rung"], {}).setdefault(r["task_id"], []).append(bool(r["passed"]))
        known = [n for n in RUNGS if n in self.greedy or n in self.samples]
        self.rungs = known + sorted((set(self.greedy) | set(self.samples)) - set(known))

    def ids(self, pred) -> list[str]:
        return [t for t, m in self.meta.items() if pred(m)]


def stat_block(out: Outcomes, rung: str, ids: list[str], k: int, B: int, seed: int) -> dict:
    g = [t for t in ids if t in out.greedy.get(rung, {})]
    v1 = [float(out.greedy[rung][t]) for t in g]
    c1 = [out.meta[t]["cluster"] for t in g]
    p1 = sum(v1) / len(v1) if v1 else float("nan")
    s = [t for t in ids if len(out.samples.get(rung, {}).get(t, [])) >= k]
    vk = [pass_at_k(len(out.samples[rung][t]), sum(out.samples[rung][t]), k) for t in s]
    ck = [out.meta[t]["cluster"] for t in s]
    return {"n1": len(g), "pass1": p1, "se1": binomial_se(p1, len(g)) if g else float("nan"),
            "se1_cluster": cluster_se(v1, c1) if len(g) > 1 else float("nan"),
            "nk": len(s), "passk": sum(vk) / len(vk) if vk else float("nan"),
            "sek": bootstrap_se(vk, B=B, seed=seed), "sek_cluster": bootstrap_se(vk, ck, B=B, seed=seed)}


def taxonomy_counts(grades: list[dict], rungs: list[str]) -> dict:
    out: dict = {}
    for rung in rungs:
        rows = [r for r in grades if r["rung"] == rung and not r["passed"]]
        buckets = {"greedy": [r for r in rows if r["mode"] == "greedy"],
                   "greedy/5.0": [r for r in rows if r["mode"] == "greedy" and r["subset"] == "5.0"],
                   "greedy/timeless": [r for r in rows if r["mode"] == "greedy" and r["subset"] == "timeless"],
                   "greedy/families": [r for r in rows if r["mode"] == "greedy" and r["kind"] == "family"],
                   "greedy/gotchas": [r for r in rows if r["mode"] == "greedy" and r["kind"] == "gotcha"],
                   "sample": [r for r in rows if r["mode"] == "sample"]}
        entry: dict = {}
        for name, rs in buckets.items():
            cls = [classify(r) for r in rs]
            entry[name] = {"n": len(rs), "classes": {c: sum(x.cls == c for x in cls) for c in CLASSES}}
        rules = Counter(f"{x.cls}:{x.rule}" for x in (classify(r) for r in rows))
        entry["rules"] = dict(rules.most_common())
        out[rung] = entry
    return out


def throughput(gens: list[dict], warmups: list[dict], config: dict, rungs: list[str]) -> dict:
    out: dict = {}
    for rung in rungs:
        rs = [g for g in gens if g["rung"] == rung]
        timed = [g for g in rs if not g.get("reused_from")]
        tps = [g["eval_count"] / (g["eval_duration"] / 1e9) for g in timed if g["eval_duration"] > 0 and g["eval_count"] > 0]
        w = next((x for x in warmups if x["rung"] == rung), None)
        greedy_by_prompt: dict[str, list] = {}
        for g in timed:
            if g["mode"] == "greedy":
                greedy_by_prompt.setdefault(g["prompt_sha1"], []).append(g["reply"])
        multi = {p: rl for p, rl in greedy_by_prompt.items() if len(rl) > 1}
        out[rung] = {
            "model": config.get("rungs", {}).get(rung, {}).get("model"),
            "replies": len(rs), "timed_replies": len(timed), "reused_greedy": len(rs) - len(timed),
            "decode_tps": {"median": percentile(tps, 50), "p10": percentile(tps, 10), "p90": percentile(tps, 90), "n": len(tps)},
            "tokens_per_answer": {"mean": sum(g["eval_count"] for g in timed) / len(timed) if timed else float("nan"),
                                  "median": percentile([g["eval_count"] for g in timed], 50)},
            "prompt_tokens_median": percentile([g["prompt_eval_count"] for g in timed], 50),
            "truncated": sum(bool(g.get("truncated")) for g in rs),
            "extract_paths": dict(Counter(g["extract"] for g in rs)),
            "think_blocks": dict(Counter(g["think_block"] or "none" for g in rs)),
            "thinking_chars_total": sum(g.get("thinking_chars", 0) for g in rs),
            "gen_wall_s": sum(g["wall_s"] for g in rs),
            "warmup": None if w is None else {k: w.get(k) for k in ("wall_s", "load_wall_s", "load_duration", "eval_count", "eval_duration",
                                                                     "prompt_eval_count", "size", "size_vram")},
            "greedy_repeat_groups": len(multi),
            "greedy_repeat_identical": sum(len(set(rl)) == 1 for rl in multi.values()),
        }
    for rung, entry in out.items():  # measured size of the notes block: warm-up prompt tokens minus the plain twin's
        spec = RUNGS.get(rung)
        w = entry["warmup"]
        if spec and spec.notes and w and w.get("prompt_eval_count"):
            twin = next((r for r in out if r != rung and RUNGS.get(r) and RUNGS[r].model == spec.model and not RUNGS[r].notes), None)
            tw = out[twin]["warmup"] if twin else None
            if tw and tw.get("prompt_eval_count"):
                entry["notes_prompt_tokens"] = w["prompt_eval_count"] - tw["prompt_eval_count"]
    return out


def summarize(gens: list[dict], grades: list[dict], warmups: list[dict], config: dict, grading_s: dict | None = None, *,
              k: int = 5, boot: int = 2000, seed: int = 0) -> dict:
    out = Outcomes(grades)
    rungs = out.rungs
    gen_keys = {(g["rung"], g["task_id"], g["mode"], g["sample"]) for g in gens}
    graded_keys = {(g["rung"], g["task_id"], g["mode"], g["sample"]) for g in grades}
    groups = {}
    for gid, desc, pred in GROUPS:
        ids = out.ids(pred)
        if ids:
            groups[gid] = {"description": desc, "n_tasks": len(ids), "n_clusters": len({out.meta[t]["cluster"] for t in ids}),
                           "rungs": {r: stat_block(out, r, ids, k, boot, seed) for r in rungs}}
    families: dict = {}
    for fam in dict.fromkeys(m["family"] for m in out.meta.values()):
        ids = [t for t, m in out.meta.items() if m["family"] == fam]
        held = out.meta[ids[0]]["held_out"]
        subsets = {out.meta[t]["subset"] for t in ids}  # a gotcha group mixes 5.0 checks, timeless checks and controls
        families[fam] = {"held_out": held, "subset": subsets.pop() if len(subsets) == 1 else "mixed", "n_tasks": len(ids),
                         "rungs": {r: stat_block(out, r, ids, k, boot, seed) for r in rungs}}
    paired: dict = {}
    for gid in PAIRED_GROUPS:
        pred = dict((g[0], g[2]) for g in GROUPS)[gid]
        ids = out.ids(pred)
        rows = []
        for a, b in combinations(rungs, 2):
            both = [t for t in ids if t in out.greedy.get(a, {}) and t in out.greedy.get(b, {})]
            if both:
                res = paired_compare([out.greedy[a][t] for t in both], [out.greedy[b][t] for t in both],
                                     [out.meta[t]["cluster"] for t in both])
                rows.append({"a": a, "b": b, **res})
        if rows:
            paired[gid] = rows
    return clean({
        "k": k, "rungs": rungs, "n_tasks": len(out.meta), "groups": groups, "families": families, "paired": paired,
        "taxonomy": taxonomy_counts(grades, rungs),
        "throughput": throughput(gens, warmups, config, rungs),
        "counts": {"generations": len(gen_keys), "graded": len(graded_keys), "ungraded": len(gen_keys - graded_keys),
                   "grader_errors": sum(bool(g.get("grader_error")) for g in grades),
                   "grade_cpu_s": round(sum(g.get("grade_s", 0.0) or 0.0 for g in grades), 1),
                   "cached_grades": sum(bool(g.get("cached")) for g in grades)},
        "decoding": config.get("decoding"), "grading": grading_s, "ollama_version": config.get("ollama_version"),
    })


# ---------------------------------------------------------------- markdown

_NO_SE = object()


def pct(p, se=_NO_SE, digits: int = 1) -> str:
    """Percent with an optional ± SE; an undefined SE (None, e.g. one task) prints as '± -'."""
    if p is None:
        return "-"
    s = f"{100 * p:.{digits}f}"
    if se is _NO_SE:
        return s
    return f"{s} ± -" if se is None else f"{s} ± {100 * se:.{digits}f}"


def num(x, fmt: str = ".1f") -> str:
    return "-" if x is None else format(x, fmt)


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(s: dict, run_name: str) -> str:
    rungs, k = s["rungs"], s["k"]
    md = [f"# Bake-off report: {run_name}", ""]
    c = s["counts"]
    pending = f", **{c['ungraded']} not yet graded**" if c["ungraded"] else ""
    errors = " (**a grader raised: look at these before trusting any number**)" if c["grader_errors"] else ""
    md.append(f"{s['n_tasks']} tasks, {len(rungs)} rungs ({', '.join(rungs)}); {c['generations']} replies, {c['graded']} graded"
              f"{pending}; {c['grader_errors']} grader errors{errors}. Ollama {s.get('ollama_version') or '?'}. "
              f"Grader CPU time {c['grade_cpu_s']:.0f} s ({c['cached_grades']} grades reused from an identical (task, code) pair).")
    if s.get("decoding"):
        d = s["decoding"]
        md += ["", f"Decoding: greedy `{d['greedy']}`; sampling `{d['sample']}`; {d['top_k_min_p']}."]
    md += ["", "## 1. Pass rates (percent, ± standard error)", "",
           f"pass@1 is one greedy reply at T=0 (binomial SE). pass@{k} is the unbiased estimator from the sampled replies at T=0.8 "
           "(bootstrap SE over tasks). `n` is tasks in the group; `prompts` is how many distinct prompts they use.", ""]
    g = s["groups"]
    md.append("**pass@1, T=0**\n")
    md.append(table(["group", "n", "prompts"] + rungs,
                    [[gid, v["n_tasks"], v["n_clusters"]] + [pct(v["rungs"][r]["pass1"], v["rungs"][r]["se1"]) for r in rungs]
                     for gid, v in g.items()]))
    md.append(f"\n**pass@{k}, T=0.8**\n")
    md.append(table(["group", "n", "prompts"] + rungs,
                    [[gid, v["n_tasks"], v["n_clusters"]] + [pct(v["rungs"][r]["passk"], v["rungs"][r]["sek"]) if v["rungs"][r]["nk"] else "-" for r in rungs]
                     for gid, v in g.items()]))
    md.append(f"\nA `-` in the pass@{k} table means no task in the group has {k} graded sampled replies yet. "
              "Groups: " + "; ".join(f"`{gid}` {v['description']}" for gid, v in g.items()) + ".")
    if "all" in g:
        md += ["", "**Clustering check (group `all`)**: the SE if tasks sharing a prompt are not treated as independent.", ""]
        md.append(table(["rung", "pass@1 SE binomial", "pass@1 SE by prompt", f"pass@{k} SE tasks", f"pass@{k} SE by prompt"],
                        [[r] + [pct(None) if x is None else f"{100 * x:.1f}" for x in (g["all"]["rungs"][r]["se1"], g["all"]["rungs"][r]["se1_cluster"],
                                                                                     g["all"]["rungs"][r]["sek"], g["all"]["rungs"][r]["sek_cluster"])]
                         for r in rungs]))

    md += ["", "## 2. Per family (pass@1 / pass@%d, percent)" % k, ""]
    md.append(table(["family", "subset", "n"] + rungs,
                    [[f"{name}{' †' if v['held_out'] else ''}", v["subset"], v["n_tasks"]] +
                     [f"{pct(v['rungs'][r]['pass1'])} / {pct(v['rungs'][r]['passk']) if v['rungs'][r]['nk'] else '-'}" for r in rungs]
                     for name, v in s["families"].items()]))
    md.append("\n† held-out family (never training data).")

    md += ["", "## 3. Error taxonomy of failing replies", "",
           "Classes (first matching signal wins; the rest are kept per reply as `also`): " + ", ".join(f"`{x}`" for x in CLASSES) +
           ". `api` = hallucinated or removed cv2 name or call signature (prior -> fine-tune); `convention` = shape, dtype, "
           "ordering, signature or contract mismatch (prior -> fine-tune); `algorithm` = met the contract, wrong numbers "
           "(capacity -> bigger model); `lint`, `format`, `timeout`, `other` as in `bakeoff/taxonomy.py`.", ""]
    for bucket, title in (("greedy", "T=0 failures, all tasks"), ("greedy/5.0", "T=0 failures, subset 5.0"),
                          ("greedy/timeless", "T=0 failures, subset timeless"),
                          ("greedy/families", "T=0 failures, families only (the clean `convention` vs `algorithm` split)"),
                          ("greedy/gotchas", "T=0 failures, gotchas only (`algorithm` here may be a dtype or behaviour convention)"),
                          ("sample", "T=0.8 sampled failures")):
        md.append(f"**{title}**\n")
        md.append(table(["rung", "failures"] + list(CLASSES),
                        [[r, s["taxonomy"][r][bucket]["n"]] + [s["taxonomy"][r][bucket]["classes"][c] for c in CLASSES] for r in rungs]))
        md.append("")
    md.append("Rules fired (all failing replies, `class:rule`):\n")
    allrules = Counter()
    for r in rungs:
        allrules.update(s["taxonomy"][r]["rules"])
    md.append(table(["class:rule"] + rungs, [[rule] + [s["taxonomy"][r]["rules"].get(rule, 0) for r in rungs] for rule, _ in allrules.most_common()]))

    md += ["", "## 4. Paired comparison on pass@1 (exact McNemar)", "",
           "A - B is the paired difference in percentage points with its SE; `A only` / `B only` are the discordant tasks; `p` is the "
           "exact two-sided McNemar p-value. `p by prompt` is an exact sign test over prompts (one vote per distinct prompt), the "
           "conservative companion when many tasks share a prompt.", ""]
    for gid, rows in s["paired"].items():
        md.append(f"**group `{gid}`**\n")
        md.append(table(["A", "B", "n", "A only", "B only", "A - B (pts)", "SE", "SE by prompt", "p", "p by prompt"],
                        [[x["a"], x["b"], x["n"], x["a_only"], x["b_only"], f"{100 * x['diff']:+.1f}",
                          "-" if x["se"] is None else f"{100 * x['se']:.1f}",
                          "-" if x.get("se_cluster") is None else f"{100 * x['se_cluster']:.1f}",
                          f"{x['p']:.3g}", "-" if x.get("p_sign_cluster") is None else f"{x['p_sign_cluster']:.3g}"] for x in rows]))
        md.append("")
    for gid in ("all", "5.0"):
        if gid in s["paired"] and len(rungs) > 1:
            md.append(f"**A - B matrix, group `{gid}`** (row A, column B; points, `*` = McNemar p < 0.05)\n")
            cell = {(x["a"], x["b"]): x for x in s["paired"][gid]}
            rows = []
            for a in rungs:
                row = [a]
                for b in rungs:
                    if (a, b) in cell:
                        x = cell[(a, b)]
                        row.append(f"{100 * x['diff']:+.1f}{'*' if x['p'] < 0.05 else ''}")
                    elif (b, a) in cell:
                        x = cell[(b, a)]
                        row.append(f"{-100 * x['diff'] + 0.0:+.1f}{'*' if x['p'] < 0.05 else ''}")  # + 0.0: no "-0.0"
                    else:
                        row.append("")
                rows.append(row)
            md.append(table(["A \\ B"] + rungs, rows) + "\n")

    md += ["## 5. Throughput and cost", "",
           "Decode tok/s is `eval_count / eval_duration` per reply, warm-up excluded (it is in `warmups.jsonl`). A reply is truncated "
           "when it stopped on `num_predict`. Prompt tokens can undercount when Ollama reuses a cached prompt prefix; the warm-up "
           "figure (first request after load) is the full prompt.", ""]
    t = s["throughput"]
    md.append(table(["rung", "model", "replies", "tok/s median", "p10", "p90", "tokens/answer mean", "prompt tokens median",
                     "truncated", "generation wall (min)"],
                    [[r, t[r]["model"], t[r]["replies"], num(t[r]['decode_tps']['median']), num(t[r]['decode_tps']['p10']),
                      num(t[r]['decode_tps']['p90']), num(t[r]['tokens_per_answer']['mean'], '.0f'), num(t[r]['prompt_tokens_median'], '.0f'),
                      t[r]["truncated"], f"{t[r]['gen_wall_s'] / 60:.1f}"] for r in rungs]))
    md.append(f"\nTotal generation wall time: {sum(t[r]['gen_wall_s'] for r in rungs) / 3600:.2f} h "
              f"(plus {sum((t[r]['warmup'] or {}).get('wall_s') or 0 for r in rungs):.0f} s of discarded warm-ups).\n")
    md.append(table(["rung", "extraction path", "think block", "thinking chars", "warm-up s", "load s", "warm-up prompt tokens", "on GPU (bytes)",
                     "greedy repeats identical"],
                    [[r, t[r]["extract_paths"], t[r]["think_blocks"], t[r]["thinking_chars_total"],
                      (t[r]["warmup"] or {}).get("wall_s", "-"), (t[r]["warmup"] or {}).get("load_wall_s", "-"),
                      (t[r]["warmup"] or {}).get("prompt_eval_count", "-"),
                      f"{(t[r]['warmup'] or {}).get('size_vram', '-')} of {(t[r]['warmup'] or {}).get('size', '-')}",
                      f"{t[r]['greedy_repeat_identical']} of {t[r]['greedy_repeat_groups']}"] for r in rungs]))
    for r in rungs:
        if "notes_prompt_tokens" in t[r]:
            md.append(f"\nMeasured size of the notes block in `{r}`: {t[r]['notes_prompt_tokens']} prompt tokens "
                      f"(warm-up prompt minus the plain 7B's); the budget is 1500.")
    md.append("")
    return "\n".join(md)


def render_audit(samples: dict, gens_by_key: dict, k: int) -> str:
    md = ["# Taxonomy audit", "",
          f"Up to {k} randomly drawn failing replies per class (fixed seed). Read each: does the class match what went wrong? "
          "`also` lists the other signals that fired for the same reply.", ""]
    for cls, rows in samples.items():
        md += [f"## {cls} ({len(rows)} shown)", ""]
        if not rows:
            md += ["(none)", ""]
        for rec, c in rows:
            gen = gens_by_key.get((rec["rung"], rec["task_id"], rec["mode"], rec["sample"]), {})
            code = gen.get("code", "")
            head = "\n".join(code.splitlines()[:40])
            md += [f"### {rec['rung']} / {rec['task_id']} / {rec['mode']} {rec['sample']}", "",
                   f"- rule `{c.rule}`; also {list(c.also) or 'nothing'}", f"- evidence: {c.evidence}", "", "```python", head, "```", ""]
    return "\n".join(md)


def write_report(run: RunDir, *, k: int = 5, boot: int = 2000, seed: int = 0, audit_k: int = 3) -> dict:
    gens, grades = read_jsonl(run.generations), read_jsonl(run.grades)
    if not grades:
        raise SystemExit(f"{run.grades} has no grades; run the grade command first")
    summary = summarize(gens, grades, read_jsonl(run.warmups), read_json(run.config, {}), read_json(run.grading), k=k, boot=boot, seed=seed)
    write_json(run.summary, summary)
    run.report_md.write_text(render_markdown(summary, run.path.name), encoding="utf-8")
    samples = sample_failures(grades, audit_k, seed)
    run.audit_md.write_text(render_audit(samples, {(g["rung"], g["task_id"], g["mode"], g["sample"]): g for g in gens}, audit_k), encoding="utf-8")
    return summary
