# opencv-specialist

This project specializes an open code model as an **OpenCV 5.0 worker** for an orchestrator
(Claude Code via MCP). Every answer it gives comes back with an inspection report.

- **The plan, findings and templates:** [PLAN.md](PLAN.md)
- **GPU logistics** (Colab via Google AI Pro, CU budget, runbook): [COMPUTE.md](COMPUTE.md)

## What's here: the Phase 0 kit (CPU-only, tested)

| Path | What it is |
|---|---|
| `probes/api_diff.py`, `probes/diff_report.py` | Differential API probe: 213 gauge-block experiments run in each OpenCV version |
| `probes/results/diff_4.14_vs_5.0.md` | The measured 4.14 → 5.0 diff: 40 behaviour changes and 31 timeless gotchas. This is the curriculum |
| `sandbox/` | Lint gate and subprocess runner: timeout with process-group kill, rlimits, GUI/network/write guards |
| `cvbench/contract.py` | The task contract: prompt rendering and output checks |
| `cvbench/families/contour_area.py` | Worked task family: manufactured-solution and metamorphic verifiers, tolerance calibrated on 3,500 gauge blocks |
| `cvbench/families/README.md` | How to write the next families, and which ones to write |
| `cvbench/evaluate.py` | Four-gate grader (exec-only, lint, manufactured, metamorphic) for any family |
| `cvbench/demo_exec_vs_mms.py` | Why `exec()` isn't a verifier: it accepts 7 of 8 wrong solutions |
| `tests/` | 33 tests; they pass on OpenCV 5.0.0 and 4.14.0 |

## Quickstart

Linux or macOS (on Windows, use WSL2: the sandbox relies on POSIX rlimits and process groups).

```bash
cd opencv-specialist
python3 -m venv .venv-cv5 && . .venv-cv5/bin/activate
pip install -r requirements-cv5.txt

python -m cvbench.demo_exec_vs_mms   # the gate matrix, about 5 s
pytest                               # 33 tests, about 10 s
```

Reproduce the version diff with a second environment:

```bash
python3 -m venv .venv-cv4 && .venv-cv4/bin/pip install -r requirements-cv4.txt
.venv-cv4/bin/python probes/api_diff.py      # writes probes/results/probe_4.14.0.{json,npz}
.venv-cv5/bin/python probes/api_diff.py      # writes probes/results/probe_5.0.0.{json,npz}
.venv-cv5/bin/python probes/diff_report.py probes/results/probe_4.14.0.json \
    probes/results/probe_5.0.0.json --out probes/results/diff_4.14_vs_5.0.md
```

## Using the sandbox directly

```python
import numpy as np
from sandbox.run import run_solution

code = '''
import cv2

def solve(img):
    return {"edges": cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)}
'''
res = run_solution(code, [{"img": np.zeros((64, 64, 3), np.uint8)}], timeout_s=10)
print(res.summary(), res.outputs[0]["edges"].shape if res.ok else "")
```

The sandbox is a seatbelt, not a vault. Its Python-level guards stop accidents, not adversaries.
For bulk generation, run the same runner inside a container with `--network none`.
