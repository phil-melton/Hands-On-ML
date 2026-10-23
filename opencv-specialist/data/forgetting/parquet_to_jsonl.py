"""Step 1 of the slice build: the pinned MBPP+ parquet -> one JSON object per line, in the scratch folder.

Data only: pyarrow reads the file, nothing in it is executed or imported. Run with the train venv, in isolated mode:
    ~/venvs/train/bin/python -I ~/opencv-expert/scratch/forgetting/parquet_to_jsonl.py PARQUET OUT.jsonl
"""
import json
import sys

import pyarrow.parquet as pq

src, dst = sys.argv[1], sys.argv[2]
rows = pq.read_table(src).to_pylist()
rows.sort(key=lambda r: r["task_id"])
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps({k: r[k] for k in ("task_id", "prompt", "code", "test_imports", "test_list", "test")}, sort_keys=True) + "\n")
print(len(rows), "rows ->", dst)
