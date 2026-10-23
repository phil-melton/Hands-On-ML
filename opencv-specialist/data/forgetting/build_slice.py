"""Step 2 of the slice build: filters, 13-gram overlap, reference validation, seeded selection, files.

Run with the cv5 venv in isolated mode (the repo is added to sys.path by hand because -I drops the script directory):
    ~/venvs/cv5/bin/python -I ~/opencv-expert/scratch/forgetting/build_slice.py RAW.jsonl OUT_DIR [--n 300] [--workers 2]

Input RAW.jsonl comes from parquet_to_jsonl.py (data only). The test scripts in it are parsed with ast and run
only inside the sandbox (bakeoff.forgetting.validate_reference); nothing from the download is executed in this process.

Pipeline (counts go to OUT_DIR/build_report.json):
    378 problems
      - drop: a word in IMAGE_WORDS (OpenCV / image processing) in the text, reference solution or asserts
      - drop: a 13-gram (lowercased \\w+ tokens) of the text, the prompt as shown to the model, the reference
              solution or the asserts that also occurs in a row of data/replay/replay.jsonl or data/train.jsonl
      - drop: the reference solution does not pass its own tests through the sandbox path
      = pool; slice = the first N ids of the pool sorted by sha256(seed|id)
"""
import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO = Path("/home/ptmel/Hands-On-ML/opencv-specialist")
sys.path.insert(0, str(REPO))

from bakeoff import forgetting as F  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)  # regex strings in the dataset code


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_rows(path):
    for r in read_jsonl(path):
        yield r["id"], F.corpus_text(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=F.SLICE_SIZE)
    ap.add_argument("--seed", type=int, default=F.SLICE_SEED)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    raw = read_jsonl(a.raw)
    recs = {}
    for r in raw:
        entry = F.entry_point_of(r["test"])
        assert F.defines(r["code"], entry), (r["task_id"], entry)
        assert entry in r["test_list"][0], (r["task_id"], entry)
        recs[r["task_id"]] = {
            "task_id": f"Mbpp/{r['task_id']}", "mbpp_id": r["task_id"], "prompt": r["prompt"].strip(), "entry_point": entry,
            "test_list": r["test_list"], "test_imports": r["test_imports"], "test": r["test"], "reference": r["code"].strip("\n") + "\n"}
    print(len(recs), "problems read")

    # ---- 1. OpenCV / image words
    kw = {i: F.exclusion_reason({"prompt": r["prompt"], "code": r["reference"], "test_list": r["test_list"]}) for i, r in recs.items()}
    kw_dropped = {i: w for i, w in kw.items() if w}

    # ---- 2. 13-gram overlap against every row of replay.jsonl and train.jsonl
    queries = {}
    for i, r in recs.items():
        queries[(i, "text")] = r["prompt"]
        queries[(i, "shown")] = F.user_prompt(r["prompt"], r["entry_point"], r["test_list"][0])
        queries[(i, "reference")] = r["reference"]
        queries[(i, "tests")] = "\n".join(r["test_list"])
    corpora = {"replay": corpus_rows(REPO / "data/replay/replay.jsonl"), "train": corpus_rows(REPO / "data/train.jsonl")}
    scan = F.overlap_scan(queries, corpora, n=F.NGRAM)
    n_replay = len(read_jsonl(REPO / "data/replay/replay.jsonl"))
    n_train = len(read_jsonl(REPO / "data/train.jsonl"))
    overlap_dropped = {}
    for i in recs:
        hits = [(f, c, scan[(i, f)][c]) for f in ("text", "shown", "reference", "tests") for c in ("replay", "train") if scan[(i, f)][c]["shared"] > 0]
        if hits:
            overlap_dropped[i] = [{"field": f, "corpus": c, **h} for f, c, h in hits]
    has_grams = {f: sum(1 for i in recs if scan[(i, f)]["replay"]["n_grams"] > 0) for f in ("text", "shown", "reference", "tests")}

    # informational 8-gram sensitivity (not used to drop)
    scan8 = F.overlap_scan({k: v for k, v in queries.items() if k[1] in ("text", "reference")},
                           {"replay": corpus_rows(REPO / "data/replay/replay.jsonl"), "train": corpus_rows(REPO / "data/train.jsonl")}, n=8)
    shared8 = {f: sum(1 for i in recs if max(scan8[(i, f)][c]["shared"] for c in ("replay", "train")) > 0) for f in ("text", "reference")}

    # ---- 3. the reference solution must pass its own tests through the sandbox path
    cand = [r for i, r in recs.items() if i not in kw_dropped and i not in overlap_dropped]
    print(len(cand), "problems left; validating their reference solutions in the sandbox")
    val = F.validate_reference(cand, workers=a.workers, timeout_s=60.0)
    ref_failed = {v["mbpp_id"]: {k: v.get(k) for k in ("cls", "error_type", "error")} for v in val if not v["passed"]}
    times = [v["run_s"] for v in val if v.get("run_s") is not None]

    pool = sorted(i for i in recs if i not in kw_dropped and i not in overlap_dropped and i not in ref_failed)
    chosen = F.select_slice(pool, a.n, a.seed)
    print(f"pool {len(pool)}, slice {len(chosen)}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = f"{F.DATASET}@{F.REVISION}"
    with open(out / "mbppplus_slice.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in chosen:
            row = {**recs[i], "source": src, "license": "tests: Apache-2.0 (EvalPlus); problems: CC-BY-4.0 (MBPP)", "ngram13_shared": 0}
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with open(out / "overlap_report.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(recs):
            status = ("dropped: image word" if i in kw_dropped else "dropped: 13-gram overlap" if i in overlap_dropped
                      else "dropped: reference fails" if i in ref_failed else "in slice" if i in chosen else "pool, not sampled")
            row = {"task_id": recs[i]["task_id"], "status": status,
                   "fields": {fld: {"n_13grams": scan[(i, fld)]["replay"]["n_grams"],
                                    **{c: {k: scan[(i, fld)][c][k] for k in ("shared", "row", "rows")} for c in ("replay", "train")}}
                              for fld in ("text", "shown", "reference", "tests")}}
            f.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "dataset": src, "seed": a.seed, "n_requested": a.n, "raw": len(recs), "image_word_dropped": {str(i): w for i, w in kw_dropped.items()},
        "ngram": F.NGRAM, "corpus_rows": {"replay": n_replay, "train": n_train},
        "problems_with_a_13gram": has_grams,
        "overlap_dropped": {str(i): v for i, v in overlap_dropped.items()},
        "n_overlap_dropped": len(overlap_dropped),
        "info_8gram_problems_sharing_with_replay_or_train": shared8,
        "reference_failed": {str(i): v for i, v in ref_failed.items()},
        "validated": len(val), "sandbox_s": {"mean": sum(times) / len(times), "max": max(times), "p90": F.percentile(times, 90)},
        "pool": len(pool), "slice": len(chosen), "slice_ids": chosen,
        "pool_not_sampled": [i for i in pool if i not in chosen],
        "slice_bytes": (out / "mbppplus_slice.jsonl").stat().st_size,
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("slice_ids", "overlap_dropped", "pool_not_sampled")}, indent=1)[:3000])
    print("dropped by overlap:", sorted(overlap_dropped))


main()
