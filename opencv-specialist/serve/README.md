# opencv-expert MCP server (Stage 7)

`serve/server.py` is a stdio MCP server with one tool, `opencv_solve`. The orchestrator (Claude Code) hands it an
image task; the fine-tuned model served by Ollama (`opencv-expert`, `serve/Modelfile`) writes `solve(...)`; the server
runs it in the sandbox, checks the declared outputs, repairs failures, and reports measurements it can be trusted
to have *run*, not to have *got right*.

## What one call does

1. Loads the images (`images`: argument name -> PNG/JPG path; Windows paths are translated), adds `params`.
2. Builds the user message with `cvbench.contract.TaskSpec(...).to_prompt()`: the training format. Inputs are
   described as `uint8, shape HxWx3. BGR colour image` (`HxW`, `grayscale image` for gray files). Outputs are the
   caller's `"<dtype>, shape <shape>[. <note>]"` strings; `(N, 2)` is respelled `Nx2` as in training, and a
   description that does not parse is sent as free text (`any, shape unspecified. <text>`) with only key presence checked.
3. Asks the model exactly as the Stage 6 evaluation did: `/api/chat`, system = `train.config.SYSTEM`, `think: false`,
   `stream: false`, greedy options (`temperature 0`, `repeat_penalty 1.0`, `num_predict 2048`, `num_ctx 8192`).
   No `format`/JSON-schema decoding: the model answers with one fenced Python block.
4. Extracts the code with `bakeoff.extract.extract_code`, runs it with `sandbox.run_solution`, applies
   `cvbench.contract.check_outputs`.
5. On failure, up to `max_repairs` times (default 3): sends `[system, task prompt, assistant: the failed code in one
   fenced block, user: "That failed: <message>\nReturn the corrected full function."]`, the shape of the
   training repair pairs (`foundry.repair`). `<message>` is the sandbox summary or `contract: ...` plus
   removed-in-5.0 API hints. A reply with no code, or code identical to an earlier attempt, ends the loop.
6. After success, reruns the code on rot90 / mirror / dimmed copies of the image(s) (below).

Everything lands in `~/.opencv-expert/runs/<timestamp>-<id>/` (prompt, each reply and attempt, `result.json`, output
PNGs). Under WSL the result also carries `run_dir_windows` / `windows_path` (`\\wsl.localhost\...`) so a Windows
viewer can open the images. `save_dir` (an existing directory, e.g. the caller's working directory) also copies each
image output there as `<name>.png`. Floats come back at full precision: the caller may have no other copy of a value.

`verified: true` means the code ran and every declared output has the declared dtype/shape. It does not mean the
numbers are right. Model and sandbox failures (and unreadable images, unreachable Ollama) come back in the result
(`verified`, `error`, `log`); the tool does not raise for them.

Configuration (environment): `OPENCV_EXPERT_MODEL` (default `opencv-expert`), `OPENCV_EXPERT_OLLAMA_URL`
(default `http://127.0.0.1:11434`), `OPENCV_EXPERT_RUNS`, `OPENCV_EXPERT_HTTP_TIMEOUT` (seconds, default 300).

## Environment

```bash
python3 -m venv ~/venvs/serve
~/venvs/serve/bin/python -m pip install -r ~/Hands-On-ML/opencv-specialist/serve/requirements-serve.txt
~/venvs/serve/bin/python -m pytest tests/test_serve.py -q      # from the repo root; skips itself without `mcp`
```

`requirements-serve.txt` is the pinned freeze of `requirements-cv5.txt` + `mcp` + `httpx` (mcp 2.3.0, MIT; it needs
the 2.x `MCPServer` API). Do not install `mcp` into `~/venvs/cv5`, the grading environment.

## Register with Claude Code (Windows `claude.exe` -> WSL)

The first `--` ends `claude`'s own options; the second belongs to `wsl.exe`.

```powershell
claude mcp add opencv-expert --scope user -- wsl.exe -d Ubuntu-24.04 -- /home/ptmel/venvs/serve/bin/python /home/ptmel/Hands-On-ML/opencv-specialist/serve/server.py
```

Equivalent `--mcp-config` file (`claude --mcp-config opencv-expert.mcp.json`):

```json
{
  "mcpServers": {
    "opencv-expert": {
      "command": "wsl.exe",
      "args": ["-d", "Ubuntu-24.04", "--",
               "/home/ptmel/venvs/serve/bin/python",
               "/home/ptmel/Hands-On-ML/opencv-specialist/serve/server.py"]
    }
  }
}
```

Ollama must be running on Windows (reachable from WSL at `127.0.0.1:11434` through mirrored networking) with the
`opencv-expert` tag created (`ollama create opencv-expert -f serve/Modelfile`). A call can take minutes (model load,
up to 2048 tokens per attempt, four attempts): if Claude Code times out, raise `MCP_TOOL_TIMEOUT`.

## The metamorphic signal (`serve/metamorphic.py`)

At serve time there is no manufactured answer, but the task's symmetries still exist. After a verified run the same
code is rerun (through `run_solution`) on copies of every input image: `rot90` (clockwise), `mirror` (horizontal
flip), `dim` (pixels x 0.6, rounded, uint8). Per transform and output key the result is:

| class | meaning |
|---|---|
| `invariant` | unchanged: scalars/counts/areas within a relative tolerance (1 % for rot90/mirror, 5 % for dim), sorted values, point/box/segment sets within 2 px, images by IoU (0.98 / 0.90) or mean abs diff (3 gray levels) |
| `covariant` | moved with the input: the original (N,2) points, (N,4) segments or boxes pushed through the same transform match as an unordered set within 2 px; a mask/image matches after the inverse transform |
| `changed` | the rerun worked and the value differs |
| `error` | the rerun failed or lacks the key |

The result has `checks` (per transform, per key, with a short `detail`), `reruns`, `counts`, `stable_fraction`
((invariant + covariant) / all checks) and a one-line `verdict`. The tolerances are named constants at the top of
the module, each with the reason for its value; they come from pixel-grid arguments, not from labelled data.

It is a confidence signal, not a gate, and nothing fails because of it. Some tasks are legitimately not invariant:

- `dim` changes intensity statistics: mean gray level, histograms, fixed-threshold counts, gray image outputs.
- `rot90` swaps height and width and moves ordering conventions ("sorted by x", "first corner is top-left"),
  angles, (w, h) pairs; a homography between two images changes. `mirror` flips signed angles and orientation.
- A high fraction shows robustness, not correctness: a function that returns a constant is invariant under everything.
- Coordinates are read in pixels of the first image; (N,2)/(N,4) arrays are assumed to be points, segments or boxes.
  Lists of unequal arrays and dicts are compared shallowly.
