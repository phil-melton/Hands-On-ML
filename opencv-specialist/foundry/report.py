"""The pilot's yield table and the training-example candidates.

``summarize_run`` reads a run folder (specs, completions, grades, timing, and the repair stage when present) and returns
one dict; ``render_markdown`` turns it into report.md. ``write_examples`` writes examples.jsonl.

Per family (the 7 trainable families):

    specs, samples            specs graded; completions graded (k per spec)
    gate pass %               share of samples passing each gate on its own: exec-only, lint, manufactured, metamorphic
    Tier A / Tier B           samples at each tier (foundry.verify); Tier-A samples can repeat a solution
    kept                      Tier-A examples kept: at most 2 per spec, distinct by AST-normalized code
    A / spec                  kept / specs: the yield. ``solved``: share of specs with at least one Tier-A sample
    distinct                  kept examples distinct by (prompt, AST-normalized code) across all specs
    repair                    prompts sent, and the share whose repair reached Tier A

A spec differs from another spec of the same family and phrasing only by its scene seed, which the prompt does not
show, so such specs share one prompt (the 200-spec pilot has 28 distinct prompts). Two kept solutions with the same
prompt and the same normalized code are one training example, which is why ``distinct`` is reported next to ``kept``
and why examples.jsonl drops cross-spec duplicates by default.

Cost. The measured figure is the GPU job's own time from timing.json: ``load_s + gen_s`` (+ ``--overhead-s`` for install and
download, which the job cannot see; the session record's wall clock is shown beside it). Seconds become CU at the hourly
rate (``--gpu-rate``, L4 1.7 CU/h by default, COMPUTE.md) and dollars at ``--usd-per-cu`` (0.10). The job is one batch, so
the seconds are split over families in proportion to their output tokens: decoding is bandwidth-bound and costs about the
same per token whatever the family, and the load time is a fixed cost that every family's tokens share. Cost per Tier-A
example = the family's CU / its kept examples; a family with none has no finite cost (shown as "none"). The last column
adds the repair job: (generation CU + repair CU) / (kept + repair pairs at Tier A).

``examples.jsonl``: one candidate training example per kept Tier-A solution (and, only for specs with no Tier-A sample,
up to 2 distinct Tier-B ones, tier-labelled; ``include_tier_b=False`` drops them): messages = [training SYSTEM (not the
notes), the task prompt, the assistant answer], plus the raw reply, the extracted code and the provenance (teacher model id
and license, spec id, tier, gates, sample, seed). The assistant answer format is not decided here: ``answer_format``
"code" puts the extracted code in the assistant turn, "raw" the reply as the teacher wrote it; both are stored in every row.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from cvbench.splits import FAMILIES
from foundry.store import FoundryRun, read_json, read_jsonl, teacher_info
from foundry.verify import FOUR, select_examples
from train.config import SYSTEM

DEFAULT_RATE = 1.7      # CU/h, L4 (COMPUTE.md)
DEFAULT_USD_PER_CU = 0.10
CLASSES = ("api", "convention", "algorithm", "lint", "format", "timeout", "other")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _pct(n: int, d: int) -> float | None:
    return 100.0 * n / d if d else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def gpu_seconds(timing: dict | None, overhead_s: float = 0.0) -> float | None:
    """Load + generation seconds of a job (plus ``overhead_s``), or None without a timing record."""
    if not timing:
        return None
    return float(timing.get("load_s", 0.0)) + float(timing.get("gen_s", 0.0)) + overhead_s


def _stage_cost(timing: dict | None, tokens_by_family: dict[str, int], rate: float, usd_per_cu: float, overhead_s: float) -> dict:
    """GPU seconds, CU and dollars of one job, split over families by output tokens."""
    total = sum(tokens_by_family.values())
    secs = gpu_seconds(timing, overhead_s)
    out = {"gpu_s": secs, "cu": None if secs is None else secs / 3600.0 * rate, "by_family": {}}
    out["usd"] = None if out["cu"] is None else out["cu"] * usd_per_cu
    for f, t in tokens_by_family.items():
        share = t / total if total else 0.0
        out["by_family"][f] = {"share": share, "gpu_s": None if secs is None else secs * share,
                               "cu": None if out["cu"] is None else out["cu"] * share}
    return out


def distinct_examples(kept: list[dict], prompt_hash: dict[str, str]) -> list[dict]:
    """The kept rows, first occurrence of each (prompt, AST-normalized code) in (spec id, rank) order."""
    seen, out = set(), []
    for r in sorted(kept, key=lambda r: (r["spec_id"], r["rank"])):
        key = (prompt_hash[r["spec_id"]], r["ast_sha1"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def summarize_run(run: FoundryRun, *, gpu_rate: float = DEFAULT_RATE, usd_per_cu: float = DEFAULT_USD_PER_CU,
                  overhead_s: float = 0.0, cap: int = 2) -> dict:
    gen, rep = run.stage("generate"), run.stage("repair")
    specs = read_jsonl(run.specs)
    grades = read_jsonl(gen.grades)
    if not grades:
        raise SystemExit(f"{gen.grades} is missing or empty: run `verify` first")
    prompt_hash = {s["id"]: _sha1(s["prompt"]) for s in specs}
    fams = [f for f, row in FAMILIES.items() if not row.held_out and any(s["family"] == f for s in specs)]
    planned = Counter(s["family"] for s in specs)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for g in grades:
        by_family[g["family"]].append(g)
    kept, sel = select_examples(grades, cap)
    distinct = distinct_examples(kept, prompt_hash)
    kept_by_family = Counter(r["family"] for r in kept)
    distinct_by_family = Counter(r["family"] for r in distinct)
    rgrades = read_jsonl(rep.grades)
    rprompts = {r["id"]: r for r in read_jsonl(rep.prompts)}
    rgrades_by_family: dict[str, list[dict]] = defaultdict(list)
    for g in rgrades:
        rgrades_by_family[g["family"]].append(g)
    prompts_by_family = Counter(r["family"] for r in rprompts.values())
    repaired_ids = {g["id"] for g in rgrades if g["tier"] == "A"}
    repaired = {pid: pid in repaired_ids for pid in rprompts}
    pairs = read_jsonl(run.repair_pairs)
    pairs_a_by_family = Counter(p["family"] for p in pairs if p["tier"] == "A")

    tokens = {f: sum(g.get("completion_tokens") or 0 for g in by_family[f]) for f in fams}
    rtokens = {f: sum(g.get("completion_tokens") or 0 for g in rgrades_by_family[f]) for f in fams}
    gen_timing, rep_timing = read_json(gen.timing), read_json(rep.timing)
    gen_cost = _stage_cost(gen_timing, tokens, gpu_rate, usd_per_cu, overhead_s)
    rep_cost = _stage_cost(rep_timing, rtokens, gpu_rate, usd_per_cu, 0.0)
    session = read_json(gen.session, {}) or {}

    rows = {}
    for f in fams:
        gs = by_family[f]
        n = len(gs)
        n_specs = len({g["spec_id"] for g in gs})
        tiers = Counter(g["tier"] for g in gs)
        solved = len({g["spec_id"] for g in gs if g["tier"] == "A"})
        fail_cls = Counter((g["failure"] or {}).get("class") for g in gs if g["tier"] != "A")
        kept_n, distinct_n = kept_by_family[f], distinct_by_family[f]
        cu = gen_cost["by_family"][f]["cu"]
        rcu = rep_cost["by_family"][f]["cu"]
        r_prompts = prompts_by_family[f]
        r_ok = sum(1 for pid, ok in repaired.items() if ok and rprompts[pid]["family"] == f)
        rows[f] = {
            "subset": FAMILIES[f].subset, "specs_planned": planned[f], "specs": n_specs, "samples": n,
            "gate_pass": {g: sum(1 for x in gs if (x.get("gates") or {}).get(g, {}).get("passed")) for g in FOUR},
            "tier_A": tiers["A"], "tier_B": tiers["B"], "fail": tiers["fail"],
            "kept": kept_n, "kept_per_spec": _ratio(kept_n, n_specs), "kept_distinct": distinct_n,
            "specs_solved": solved, "solved_pct": _pct(solved, n_specs), "pass_rate_A": _pct(tiers["A"], n),
            "no_code": sum(1 for x in gs if x.get("code_empty")), "truncated": sum(1 for x in gs if x.get("finish_reason") == "length"),
            "grader_errors": sum(1 for x in gs if x.get("grader_error")), "failure_classes": dict(fail_cls),
            "output_tokens": tokens[f], "mean_tokens": _ratio(tokens[f], n),
            "gpu_s": gen_cost["by_family"][f]["gpu_s"], "cu": cu, "usd": None if cu is None else cu * usd_per_cu,
            "cu_per_A": _ratio(cu, kept_n), "usd_per_A": None if cu is None or not kept_n else cu * usd_per_cu / kept_n,
            "cu_per_distinct": _ratio(cu, distinct_n),
            "repair_prompts": r_prompts, "repair_samples": len(rgrades_by_family[f]), "repair_to_A": r_ok,
            "repair_rate": _pct(r_ok, r_prompts), "repair_pairs_A": pairs_a_by_family[f],
            "repair_cu": rcu,
            "cu_per_A_incl_repair": _ratio(None if cu is None else cu + (rcu or 0.0), kept_n + pairs_a_by_family[f]),
        }

    def total(key):
        return sum(rows[f][key] for f in fams)

    t_cu, r_cu = gen_cost["cu"], rep_cost["cu"]
    t_kept, t_pairs = total("kept"), total("repair_pairs_A")
    totals = {
        "specs_planned": total("specs_planned"), "specs": total("specs"), "samples": total("samples"),
        "gate_pass": {g: sum(rows[f]["gate_pass"][g] for f in fams) for g in FOUR},
        "tier_A": total("tier_A"), "tier_B": total("tier_B"), "fail": total("fail"), "kept": t_kept,
        "kept_per_spec": _ratio(t_kept, total("specs")), "kept_distinct": total("kept_distinct"),
        "specs_solved": total("specs_solved"), "solved_pct": _pct(total("specs_solved"), total("specs")),
        "pass_rate_A": _pct(total("tier_A"), total("samples")),
        "no_code": total("no_code"), "truncated": total("truncated"), "grader_errors": total("grader_errors"),
        "output_tokens": total("output_tokens"), "mean_tokens": _ratio(total("output_tokens"), total("samples")),
        "repair_prompts": total("repair_prompts"), "repair_samples": total("repair_samples"), "repair_to_A": total("repair_to_A"),
        "repair_rate": _pct(total("repair_to_A"), total("repair_prompts")), "repair_pairs_A": t_pairs,
        "gpu_s": gen_cost["gpu_s"], "cu": t_cu, "usd": gen_cost["usd"], "cu_per_A": _ratio(t_cu, t_kept),
        "usd_per_A": None if gen_cost["usd"] is None or not t_kept else gen_cost["usd"] / t_kept,
        "cu_per_distinct": _ratio(t_cu, total("kept_distinct")),
        "repair_gpu_s": rep_cost["gpu_s"], "repair_cu": r_cu,
        "cu_per_A_incl_repair": _ratio(None if t_cu is None else t_cu + (r_cu or 0.0), t_kept + t_pairs),
        "failure_classes": dict(sum((Counter(rows[f]["failure_classes"]) for f in fams), Counter())),
    }
    rclass: dict[str, dict] = {}
    for pid, p in rprompts.items():
        d = rclass.setdefault(p["failure"]["class"], {"prompts": 0, "repaired_to_A": 0})
        d["prompts"] += 1
        d["repaired_to_A"] += bool(repaired[pid])
    return {
        "run": run.name, "families": fams, "rows": rows, "totals": totals,
        "selection": {**sel, "kept": len(kept), "cap": cap, "distinct_prompt_ast": len(distinct),
                      "distinct_prompts": len(set(prompt_hash[s["id"]] for s in specs)),
                      "kept_with_api_findings": sum(1 for r in kept if r.get("api_findings"))},
        "repair_by_class": rclass, "teacher": teacher_info(run), "repair_teacher": teacher_info(run, "repair") if rprompts else None,
        "cost": {"gpu_rate_cu_h": gpu_rate, "usd_per_cu": usd_per_cu, "overhead_s": overhead_s,
                 "generate": {k: v for k, v in gen_cost.items() if k != "by_family"},
                 "repair": {k: v for k, v in rep_cost.items() if k != "by_family"},
                 "decode_only_cu": None if not gen_timing else float(gen_timing.get("gen_s", 0.0)) / 3600.0 * gpu_rate,
                 "output_tokens_per_s": (gen_timing or {}).get("output_tokens_per_s"),
                 "session_wall_s": session.get("wall_s"),
                 "session_wall_cu": None if not session.get("wall_s") else session["wall_s"] / 3600.0 * gpu_rate,
                 "predicted_minutes": session.get("predicted_minutes"), "predicted_cu": session.get("predicted_cu")},
    }


# ---------------------------------------------------------------- markdown

def _f(x, nd=1, none="-"):
    return none if x is None else f"{x:.{nd}f}"


def _cost_cell(x, nd=3):
    return "n/a" if x is None else _f(x, nd)


def _per_example(x, n, nd=3):
    """A cost per example: n/a without a measured cost, "none" when the denominator is zero."""
    if x is None and n:
        return "n/a"
    return "none" if not n else _f(x, nd)


def render_markdown(s: dict) -> str:
    rows, tot, cost, sel = s["rows"], s["totals"], s["cost"], s["selection"]
    has_repair = bool(tot["repair_prompts"])
    L = [f"# Teacher pilot yield: run `{s['run']}`", ""]
    t = s["teacher"]
    L += [f"Teacher: `{t['model']}` (license {t['license']}; quantization {t['quantization'] or 'none'}; GPU {t['gpu'] or 'unrecorded'}). "
          f"{tot['specs']} of {tot['specs_planned']} planned specs graded, {tot['samples']} samples; the specs share "
          f"{sel['distinct_prompts']} distinct prompts.", ""]
    L += ["## Yield per family", "",
          "| family | subset | specs | samples | Tier A | Tier B | fail | kept | A / spec | distinct | solved % | no code | cut off |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in s["families"]:
        r = rows[f]
        L.append(f"| {f} | {r['subset']} | {r['specs']} | {r['samples']} | {r['tier_A']} | {r['tier_B']} | {r['fail']} | {r['kept']} "
                 f"| {_f(r['kept_per_spec'], 2)} | {r['kept_distinct']} | {_f(r['solved_pct'])} | {r['no_code']} | {r['truncated']} |")
    L.append(f"| **all** | | {tot['specs']} | {tot['samples']} | {tot['tier_A']} | {tot['tier_B']} | {tot['fail']} | {tot['kept']} "
             f"| {_f(tot['kept_per_spec'], 2)} | {tot['kept_distinct']} | {_f(tot['solved_pct'])} | {tot['no_code']} | {tot['truncated']} |")
    L += ["", "A / spec = Tier-A examples kept (at most 2 per spec, distinct AST-normalized code) per spec. "
          f"{sel['candidates']} Tier-A samples gave {sel['kept']} kept; {sel['duplicates']} were AST duplicates within a spec, "
          f"{sel['over_cap']} over the cap of {sel['cap']}. distinct = kept examples distinct by (prompt, normalized code) across "
          f"all specs ({sel['distinct_prompt_ast']}): specs of one family and phrasing share a prompt."
          + (f" {sel['kept_with_api_findings']} kept examples name an API that 5.0 removed (the api gate reports and never blocks; "
             "see provenance.api_findings in examples.jsonl)." if sel["kept_with_api_findings"] else ""), ""]
    L += ["## Pass rate by gate (% of samples; each gate on its own)", "",
          "| family | exec-only | lint | manufactured | metamorphic | all four (Tier A) |", "|---|---:|---:|---:|---:|---:|"]
    for f in s["families"]:
        r = rows[f]
        L.append(f"| {f} | " + " | ".join(_f(_pct(r["gate_pass"][g], r["samples"])) for g in FOUR) + f" | {_f(r['pass_rate_A'])} |")
    L.append("| **all** | " + " | ".join(_f(_pct(tot["gate_pass"][g], tot["samples"])) for g in FOUR) + f" | {_f(tot['pass_rate_A'])} |")
    L += ["", "## Failure classes of the non-Tier-A samples", "", "| family | " + " | ".join(CLASSES) + " |", "|---|" + "---:|" * len(CLASSES)]
    for f in s["families"]:
        c = rows[f]["failure_classes"]
        L.append(f"| {f} | " + " | ".join(str(c.get(k, 0)) for k in CLASSES) + " |")
    L += ["", "## Cost", ""]
    if cost["generate"]["gpu_s"] is None:
        L.append("No timing.json for the generation job: cost columns are n/a.")
    else:
        L += [f"GPU seconds = load + generation from timing.json (+ {cost['overhead_s']:.0f} s overhead given): "
              f"{cost['generate']['gpu_s']:.0f} s at {cost['gpu_rate_cu_h']} CU/h = {cost['generate']['cu']:.3f} CU = "
              f"${cost['generate']['usd']:.2f} at ${cost['usd_per_cu']}/CU. Decode alone: {_f(cost['decode_only_cu'], 3)} CU; "
              f"{_f(cost['output_tokens_per_s'], 0, 'n/a')} output tokens/s. "
              + (f"Session wall clock (new to stop): {cost['session_wall_s']:.0f} s = {cost['session_wall_cu']:.3f} CU; "
                 f"predicted {cost['predicted_minutes']} min, {cost['predicted_cu']} CU. " if cost["session_wall_s"] else ""),
              "Seconds are split over families by output tokens.", ""]
    L += ["| family | output tokens | tokens / sample | GPU s | CU | $ | kept | CU / Tier-A | $ / Tier-A | CU / distinct | CU / Tier-A incl. repair |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def cost_row(name, r):
        incl = _per_example(r["cu_per_A_incl_repair"], r["kept"] + r["repair_pairs_A"]) if has_repair else "-"
        return (f"| {name} | {r['output_tokens']} | {_f(r['mean_tokens'], 0)} | {_f(r['gpu_s'], 0, 'n/a')} | {_cost_cell(r['cu'])} "
                f"| {_cost_cell(r['usd'], 2)} | {r['kept']} | {_per_example(r['cu_per_A'], r['kept'])} "
                f"| {_per_example(r['usd_per_A'], r['kept'], 4)} | {_per_example(r['cu_per_distinct'], r['kept_distinct'])} | {incl} |")

    for f in s["families"]:
        L.append(cost_row(f, rows[f]))
    L.append(cost_row("**all**", tot))
    L += ["", "## Repair (one round)", ""]
    if not has_repair:
        L.append("No repair stage yet.")
    else:
        L += ["| family | repair prompts | samples | repaired to Tier A | success % | repair pairs (Tier A) | repair CU |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for f in s["families"]:
            r = rows[f]
            L.append(f"| {f} | {r['repair_prompts']} | {r['repair_samples']} | {r['repair_to_A']} | {_f(r['repair_rate'])} "
                     f"| {r['repair_pairs_A']} | {_cost_cell(r['repair_cu'])} |")
        L.append(f"| **all** | {tot['repair_prompts']} | {tot['repair_samples']} | {tot['repair_to_A']} | {_f(tot['repair_rate'])} "
                 f"| {tot['repair_pairs_A']} | {_cost_cell(tot['repair_cu'])} |")
        L += ["", "| failure class | prompts | repaired to Tier A | success % |", "|---|---:|---:|---:|"]
        for c, d in sorted(s["repair_by_class"].items()):
            L.append(f"| {c} | {d['prompts']} | {d['repaired_to_A']} | {_f(_pct(d['repaired_to_A'], d['prompts']))} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- examples

def build_examples(run: FoundryRun, *, answer_format: str = "code", include_tier_b: bool = True, cap: int = 2,
                   cross_spec_dedup: bool = True) -> list[dict]:
    if answer_format not in ("code", "raw"):
        raise ValueError("answer_format must be 'code' or 'raw'")
    grades = read_jsonl(run.stage("generate").grades)
    specs = {r["id"]: r for r in read_jsonl(run.specs)}
    texts = {(r["id"], r["sample"]): r["text"] for r in read_jsonl(run.stage("generate").completions)}
    teacher = teacher_info(run)
    chosen, _ = select_examples(grades, cap, "A")
    if include_tier_b:
        solved = {r["spec_id"] for r in chosen}
        b_rows, _ = select_examples([g for g in grades if g["spec_id"] not in solved], cap, "B")
        chosen += b_rows
    seen, out = set(), []
    for r in sorted(chosen, key=lambda r: (r["tier"], r["spec_id"], r["rank"])):
        spec = specs[r["spec_id"]]
        key = (_sha1(spec["prompt"]), r["ast_sha1"])
        if cross_spec_dedup and key in seen:
            continue                                  # same prompt, same normalized code as an earlier example
        seen.add(key)
        raw = texts[(r["id"], r["sample"])]
        out.append({
            "id": f"{r['spec_id']}#{r['rank']}", "spec_id": r["spec_id"], "family": r["family"], "tier": r["tier"],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec["prompt"]},
                         {"role": "assistant", "content": r["code"] if answer_format == "code" else raw}],
            "answer_format": answer_format, "raw_reply": raw, "code": r["code"],
            "provenance": {"teacher": teacher, "run": run.name, "spec_id": r["spec_id"], "sample": r["sample"], "tier": r["tier"],
                           "gates": {k: v["passed"] for k, v in (r.get("gates") or {}).items()}, "ast_sha1": r["ast_sha1"],
                           "code_sha1": r["code_sha1"], "extract": r["extract"], "finish_reason": r.get("finish_reason"),
                           "completion_tokens": r.get("completion_tokens"), "scene_seed": spec["seed"],
                           "phrasing_index": spec["phrasing_index"], "prompt_sha1": key[0], "system_prompt": "train.config.SYSTEM",
                           "api_findings": r.get("api_findings") or []},
        })
    return out


def write_examples(run: FoundryRun, **kw) -> int:
    rows = build_examples(run, **kw)
    with run.examples.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)
