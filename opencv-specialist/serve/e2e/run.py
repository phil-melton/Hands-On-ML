#!/usr/bin/env python
"""Stage 7 E2E runner: build the work dirs, run ``claude.exe -p`` once per task and arm, grade, report, predict.

    python serve/e2e/run.py build   --run NAME [--root DIR] [--force]
    python serve/e2e/run.py predict --run NAME --tokens-per-task N --s-per-task S [--arm ...]
    python serve/e2e/run.py run     --run NAME --arm alone|specialist --model MODEL [--only REGEX] [--limit N]
                                    [--max-turns N] [--cwd task|PATH] [--dry-run]
    python serve/e2e/run.py grade   --run NAME
    python serve/e2e/run.py report  --run NAME

Run it in WSL with the cv5 environment (``~/venvs/cv5/bin/python``). It starts the Windows ``claude.exe`` through WSL
interop. Nothing here talks to a model except the ``claude.exe -p`` sessions of ``run`` (never started by --dry-run).

Layout under ``ROOT/NAME`` (ROOT defaults to ~/opencv-expert/e2e-runs)
    config.json              the run: tasks (families, held-out flags, seeds, input hashes), flags, paths. No truth.
    mcp.json                 the --mcp-config file of the specialist arm
    prompts/<arm>/<id>.txt   the prompt sent on stdin (identical in both arms apart from the work dir it names)
    <arm>/<id>/              the orchestrator's work dir: the input PNGs only, separate per arm; answer.json (and
                             out.png for the text tasks) appear here
    logs/<arm>/<id>/         transcript.jsonl (stream-json), result.json, stderr.txt (outside the work dir)
    grades/<arm>/<id>.json   per-key pass/fail with errors
    report.md, summary.json  the comparison

Flags of ``claude.exe`` (2.1.289), why
    -p --output-format stream-json --verbose   headless, full event stream (usage, tool calls, results)
    --model M                                  the same orchestrator model in both arms
    --strict-mcp-config                        both arms: no MCP server except the ones in --mcp-config (the specialist
                                               arm passes one file with one server, the alone arm passes none)
    --tools Bash,Read,Write                    the same three built-in tools in both arms (no web, no sub-agents)
    --allowedTools ...                         pre-approves those tools (+ mcp__opencv-expert__opencv_solve in the
                                               specialist arm); --permission-mode dontAsk denies anything else, so a
                                               headless session can never block on a prompt
    --add-dir <work dir>                       the work dir is readable/writable even when --cwd points elsewhere
    --no-session-persistence                   no session files in the user's ~/.claude
    --max-turns N                              hidden in --help but a parsed option of 2.1.289 (validated as a number);
                                               run.py also counts assistant messages and stops the process itself
    isolation "flags" (default), to keep the user's own settings, CLAUDE.md, memory, skills and plugins out:
      --setting-sources ""                     load no user/project/local settings (and so no CLAUDE.md/plugins/hooks)
      --disable-slash-commands                 no skills
      --no-chrome                              no browser integration
      CLAUDE_CODE_DISABLE_AUTO_MEMORY=1, CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 in the environment (forwarded through WSLENV)
    alternatives: --isolation safe-mode (--safe-mode: disables every customization, but may also disable --mcp-config
    servers: check before using it for the specialist arm), bare (--bare: needs ANTHROPIC_API_KEY, OAuth is never
    read, so it cannot use the subscription), none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARMS = ("alone", "specialist")
MCP_NAME = "opencv-expert"
MCP_TOOL = f"mcp__{MCP_NAME}__opencv_solve"
BUILTIN_TOOLS = ("Bash", "Read", "Write")
DEFAULT_ROOT = Path.home() / "opencv-expert" / "e2e-runs"
DEFAULT_CLAUDE = "/mnt/c/Users/ptmel/.local/bin/claude.exe"
SERVE_PYTHON = "/home/ptmel/venvs/serve/bin/python"
SERVER_PY = "/home/ptmel/Hands-On-ML/opencv-specialist/serve/server.py"
DEFAULT_TIMEOUT_S = 1800
TOKEN_METRICS = ("total", "fresh", "output")  # every counter | input + cache_creation + output | output only

ISOLATION_ARGS = {
    "flags": ["--setting-sources", "", "--disable-slash-commands", "--no-chrome"],
    "safe-mode": ["--safe-mode"],
    "bare": ["--bare"],
    "none": [],
}
ISOLATION_ENV = {
    "flags": {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"},
    "safe-mode": {},
    "bare": {},
    "none": {},
}


# ---------------------------------------------------------------- paths

def distro_name() -> str:
    return os.environ.get("WSL_DISTRO_NAME") or "Ubuntu-24.04"


def to_windows_path(path, distro: str | None = None) -> str:
    """The path as a Windows process sees it: /mnt/c/x -> C:\\x, anything else -> \\\\wsl.localhost\\<distro>\\..."""
    p = str(path)
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", p)
    if m:
        return f"{m.group(1).upper()}:" + (m.group(2) or "/").replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro or distro_name()}" + p.replace("/", "\\")


class Layout:
    def __init__(self, root: Path, name: str):
        self.root, self.name = Path(root), name
        self.run = self.root / name

    @property
    def config(self) -> Path:
        return self.run / "config.json"

    @property
    def mcp(self) -> Path:
        return self.run / "mcp.json"

    def prompt(self, arm: str, tid: str) -> Path:
        return self.run / "prompts" / arm / f"{tid}.txt"

    def work(self, arm: str, tid: str) -> Path:
        return self.run / arm / tid

    def log(self, arm: str, tid: str) -> Path:
        return self.run / "logs" / arm / tid

    def grade(self, arm: str, tid: str) -> Path:
        return self.run / "grades" / arm / f"{tid}.json"


def load_config(lay: Layout) -> dict:
    if not lay.config.is_file():
        sys.exit(f"no config.json in {lay.run}: run `build --run {lay.name}` first")
    return json.loads(lay.config.read_text())


# ---------------------------------------------------------------- the claude.exe command (pure)

def mcp_config(distro: str = "Ubuntu-24.04", python: str = SERVE_PYTHON, server: str = SERVER_PY) -> dict:
    return {"mcpServers": {MCP_NAME: {"type": "stdio", "command": "wsl.exe",
                                      "args": ["-d", distro, "--", python, server], "env": {}}}}


def allowed_tools(arm: str) -> list[str]:
    return list(BUILTIN_TOOLS) + ([MCP_TOOL] if arm == "specialist" else [])


def build_command(*, arm: str, model: str, claude_exe: str = DEFAULT_CLAUDE, mcp_config_win: str | None = None,
                  workdir_win: str | None = None, isolation: str = "flags", max_turns: int | None = None,
                  max_budget_usd: float | None = None, effort: str | None = None) -> list[str]:
    """The exact argv of one headless session. The prompt goes on stdin. The two arms differ in exactly two places:
    ``--mcp-config <file>`` (specialist only) and the MCP tool in ``--allowedTools``."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    cmd = [claude_exe, "-p", "--output-format", "stream-json", "--verbose", "--model", model,
           "--permission-mode", "dontAsk", "--tools", ",".join(BUILTIN_TOOLS),
           "--allowedTools", ",".join(allowed_tools(arm)), "--strict-mcp-config", "--no-session-persistence"]
    cmd += ISOLATION_ARGS[isolation]
    if arm == "specialist":
        if not mcp_config_win:
            raise ValueError("the specialist arm needs mcp_config_win")
        cmd += ["--mcp-config", mcp_config_win]
    if workdir_win:
        cmd += ["--add-dir", workdir_win]
    if max_turns:
        cmd += ["--max-turns", str(int(max_turns))]
    if max_budget_usd:
        cmd += ["--max-budget-usd", f"{max_budget_usd:g}"]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def build_env(base: dict, *, arm: str, isolation: str = "flags", extra: dict | None = None) -> dict:
    """Environment for claude.exe. Variables set in WSL reach a Windows process only when listed in WSLENV."""
    env = dict(base)
    add = dict(ISOLATION_ENV[isolation])
    if arm == "specialist":
        add.setdefault("MCP_TIMEOUT", "60000")  # ms for the MCP server to start (wsl.exe cold start + imports)
    add.update(extra or {})
    env.update(add)
    names = [n for n in env.get("WSLENV", "").split(":") if n]
    for k in add:
        if not any(n.split("/")[0] == k for n in names):
            names.append(f"{k}/u")
    env["WSLENV"] = ":".join(names)
    return env


def shell_line(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) if c != "" else "''" for c in cmd)


# ---------------------------------------------------------------- transcript -> result (pure)

FORBIDDEN_FRAGMENTS = ("opencv-specialist", "serve/e2e", "serve\\e2e")
_RUN_PATH = re.compile(r"e2e-runs[/\\]+([^/\\\s\"']+)[/\\]+([^/\\\s\"']+)[/\\]+([^/\\\s\"']+)")


def _texts(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _verified(content) -> bool | None:
    """The ``verified`` field of an opencv_solve result, however the MCP client wrapped it."""
    text = _texts(content)
    try:
        obj = json.loads(text)
        while isinstance(obj, dict) and "verified" not in obj:
            obj = next((v for v in obj.values() if isinstance(v, dict)), None)
        if isinstance(obj, dict) and isinstance(obj.get("verified"), bool):
            return obj["verified"]
    except (json.JSONDecodeError, TypeError, StopIteration):
        pass
    m = re.search(r'"verified"\s*:\s*(true|false)', text)
    return None if not m else m.group(1) == "true"


def _usage_dict(u: dict | None) -> dict:
    u = u or {}
    d = {"input": int(u.get("input_tokens") or 0), "cache_creation": int(u.get("cache_creation_input_tokens") or 0),
         "cache_read": int(u.get("cache_read_input_tokens") or 0), "output": int(u.get("output_tokens") or 0)}
    d["fresh"] = d["input"] + d["cache_creation"] + d["output"]
    d["total"] = d["fresh"] + d["cache_read"]
    return d


def summarize_events(events: list[dict], *, arm: str, task_id: str, run_name: str = "") -> dict:
    """Everything result.json records that can be read off the stream-json events."""
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    final = next((e for e in reversed(events) if e.get("type") == "result"), None)
    per_msg: dict[str, dict] = {}
    calls: dict[str, str] = {}
    counts: dict[str, int] = {}
    n_err = 0
    solve = {"calls": 0, "verified": 0, "unverified": 0, "unknown": 0, "errors": 0}
    peeked: list[str] = []
    n_assistant = 0
    for e in events:
        t = e.get("type")
        msg = e.get("message") or {}
        if t == "assistant":
            mid = msg.get("id") or f"anon{len(per_msg)}"
            if mid not in per_msg:
                n_assistant += 1
            cur = _usage_dict(msg.get("usage"))
            old = per_msg.setdefault(mid, {k: 0 for k in cur})
            for k in ("input", "cache_creation", "cache_read", "output"):
                old[k] = max(old[k], cur[k])
            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name = b.get("name", "?")
                    calls[b.get("id", "")] = name
                    counts[name] = counts.get(name, 0) + 1
                    if name == MCP_TOOL:
                        solve["calls"] += 1
                    blob = json.dumps(b.get("input", {}))
                    for frag in FORBIDDEN_FRAGMENTS:
                        if frag in blob:
                            peeked.append(f"{name}: {frag}")
                    for m in _RUN_PATH.finditer(blob.replace("\\\\", "/")):
                        if (m.group(1), m.group(2), m.group(3)) != (run_name, arm, task_id) and m.group(1) != "":
                            peeked.append(f"{name}: {m.group(0)}")
        elif t == "user":
            c = msg.get("content")
            for b in c if isinstance(c, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    if b.get("is_error"):
                        n_err += 1
                    if calls.get(b.get("tool_use_id")) == MCP_TOOL:
                        if b.get("is_error"):
                            solve["errors"] += 1
                        v = _verified(b.get("content"))
                        solve["verified" if v is True else "unverified" if v is False else "unknown"] += 1
    sums = {k: sum(m[k] for m in per_msg.values()) for k in ("input", "cache_creation", "cache_read", "output")}
    usage_msgs = _usage_dict({"input_tokens": sums["input"], "cache_creation_input_tokens": sums["cache_creation"],
                              "cache_read_input_tokens": sums["cache_read"], "output_tokens": sums["output"]})
    usage_final = _usage_dict(final.get("usage")) if final and final.get("usage") else None
    tools = list(init.get("tools") or [])
    servers = init.get("mcp_servers") or []
    notes = []
    if arm == "alone":
        if any(str(t).startswith("mcp__") for t in tools) or servers:
            notes.append("the alone arm has MCP tools or servers")
    else:
        if MCP_TOOL not in tools:
            notes.append(f"{MCP_TOOL} is not among the session's tools")
        bad = [s for s in servers if s.get("name") != MCP_NAME or s.get("status") != "connected"]
        if bad or not servers:
            notes.append(f"MCP servers: {servers!r}")
    extra_builtin = sorted(t for t in tools if not str(t).startswith("mcp__") and t not in BUILTIN_TOOLS)
    if extra_builtin:
        notes.append(f"built-in tools beyond {BUILTIN_TOOLS}: {extra_builtin}")
    return {
        "usage": usage_final or usage_msgs,
        "usage_source": "result" if usage_final else "messages",
        "usage_messages": usage_msgs,
        "cost_usd": (final or {}).get("total_cost_usd"),
        "num_turns": (final or {}).get("num_turns"),
        "n_assistant_messages": n_assistant,
        "is_error": (final or {}).get("is_error"),
        "permission_denials": [d.get("tool_name", "?") if isinstance(d, dict) else str(d) for d in (final or {}).get("permission_denials") or []],
        "result_subtype": (final or {}).get("subtype"),
        "result_text": str((final or {}).get("result", ""))[:2000],
        "tool_calls": counts,
        "n_tool_calls": sum(counts.values()),
        "n_tool_errors": n_err,
        "n_solve": solve["calls"],
        "n_solve_verified": solve["verified"],
        "solve": solve,
        "init": {k: init.get(k) for k in ("model", "cwd", "session_id", "tools", "mcp_servers", "permissionMode",
                                          "claude_code_version", "apiKeySource", "slash_commands", "skills", "plugins")
                 if k in init},
        "isolation_ok": not notes,
        "isolation_notes": notes,
        "peeked": sorted(set(peeked)),
        "has_result_event": final is not None,
    }


# ---------------------------------------------------------------- statistics (pure)

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test_p(b: int, c: int) -> float:
    """Exact two-sided sign test on the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = p * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _desc(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "total": 0.0}
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "total": float(sum(xs))}


def summarize(results: dict[str, dict[str, dict]], grades: dict[str, dict[str, dict]], tasks: list[dict]) -> dict:
    """results[arm][task_id] = result.json, grades[arm][task_id] = grade dict, tasks = config task metas.
    Pure: the report is a function of these three."""
    ids = [t["id"] for t in tasks]
    held = {t["id"]: any(t["held_out"].values()) for t in tasks}
    out: dict = {"n_tasks": len(ids), "arms": {}, "paired": {}, "tokens": {}, "wall": {}, "per_task": []}
    for arm in ARMS:
        done = [i for i in ids if i in results.get(arm, {})]
        passed = [i for i in done if grades.get(arm, {}).get(i, {}).get("passed")]
        lo, hi = wilson(len(passed), len(done))
        ho = [i for i in done if held[i]]
        a = {"n_run": len(done), "n_pass": len(passed), "rate": len(passed) / len(done) if done else None,
             "wilson95": [lo, hi],
             "held_out_family_tasks": {"n": len(ho), "n_pass": sum(1 for i in ho if grades.get(arm, {}).get(i, {}).get("passed"))},
             "status": {}}
        for i in done:
            s = grades.get(arm, {}).get(i, {}).get("status", "ungraded")
            a["status"][s] = a["status"].get(s, 0) + 1
        res = [results[arm][i] for i in done]
        a["isolation_problems"] = [i for i in done if not results[arm][i].get("isolation_ok", True)]
        a["peeked"] = [i for i in done if results[arm][i].get("peeked")]
        a["permission_denials"] = {i: results[arm][i]["permission_denials"] for i in done if results[arm][i].get("permission_denials")}
        a["stopped_by"] = {}
        for r in res:
            a["stopped_by"][r.get("stopped_by", "?")] = a["stopped_by"].get(r.get("stopped_by", "?"), 0) + 1
        a["tools"] = {"calls_per_task": _desc([float(r.get("n_tool_calls", 0)) for r in res]),
                      "bash_per_task": _desc([float(r.get("tool_calls", {}).get("Bash", 0)) for r in res]),
                      "turns_per_task": _desc([float(r.get("num_turns") or r.get("n_assistant_messages") or 0) for r in res]),
                      "tool_errors_per_task": _desc([float(r.get("n_tool_errors", 0)) for r in res]),
                      "solve_calls_per_task": _desc([float(r.get("n_solve", 0)) for r in res]),
                      "solve_calls_total": sum(r.get("n_solve", 0) for r in res),
                      "solve_verified_total": sum(r.get("n_solve_verified", 0) for r in res),
                      "tasks_using_solve": sum(1 for r in res if r.get("n_solve", 0) > 0)}
        out["arms"][arm] = a
        out["tokens"][arm] = {m: _desc([float(r["usage"][m]) for r in res if r.get("usage")]) for m in TOKEN_METRICS}
        out["tokens"][arm]["parts"] = {k: _desc([float(r["usage"][k]) for r in res if r.get("usage")])
                                       for k in ("input", "cache_creation", "cache_read", "output")}
        out["tokens"][arm]["cost_usd"] = _desc([float(r["cost_usd"]) for r in res if r.get("cost_usd") is not None])
        out["wall"][arm] = _desc([float(r["wall_s"]) for r in res if r.get("wall_s") is not None])

    both = [i for i in ids if i in results.get("alone", {}) and i in results.get("specialist", {})]
    ok = lambda arm, i: bool(grades.get(arm, {}).get(i, {}).get("passed"))  # noqa: E731
    cells = {"both": [], "alone_only": [], "specialist_only": [], "neither": []}
    for i in both:
        a_, s_ = ok("alone", i), ok("specialist", i)
        cells["both" if a_ and s_ else "alone_only" if a_ else "specialist_only" if s_ else "neither"].append(i)
    out["paired"] = {"n": len(both), **{k: len(v) for k, v in cells.items()}, "tasks": cells,
                     "sign_test_p": sign_test_p(len(cells["alone_only"]), len(cells["specialist_only"]))}
    ratios: dict = {}
    for m in TOKEN_METRICS:
        rs = [(i, results["specialist"][i]["usage"][m] / results["alone"][i]["usage"][m]) for i in both
              if results["alone"][i].get("usage", {}).get(m) and results["specialist"][i].get("usage")]
        vals = [r for _, r in rs]
        fewer = sum(1 for v in vals if v < 1)
        more = sum(1 for v in vals if v > 1)
        ratios[m] = {"n": len(vals), "median": statistics.median(vals) if vals else None,
                     "iqr": [_q(vals, 0.25), _q(vals, 0.75)] if vals else None,
                     "mean_of_ratios": statistics.fmean(vals) if vals else None,
                     "ratio_of_means": (out["tokens"]["specialist"][m]["mean"] / out["tokens"]["alone"][m]["mean"]
                                        if out["tokens"]["alone"][m]["mean"] and out["tokens"]["specialist"][m]["mean"] is not None else None),
                     "specialist_fewer": fewer, "specialist_more": more, "sign_test_p": sign_test_p(fewer, more)}
    out["token_ratio_specialist_over_alone"] = ratios
    wr = [results["specialist"][i]["wall_s"] / results["alone"][i]["wall_s"] for i in both
          if results["alone"][i].get("wall_s") and results["specialist"][i].get("wall_s") is not None]
    out["wall_ratio_specialist_over_alone"] = {"n": len(wr), "median": statistics.median(wr) if wr else None,
                                               "iqr": [_q(wr, 0.25), _q(wr, 0.75)] if wr else None}
    ra, rs_ = out["arms"]["alone"]["rate"], out["arms"]["specialist"]["rate"]
    tr = ratios["total"]
    out["criterion4_inputs"] = {
        "specialist_success_ge_alone": None if ra is None or rs_ is None else rs_ >= ra,
        "paired_n": len(both),
        "median_token_ratio_total": tr["median"],
        "mean_tokens_specialist_lt_alone": (out["tokens"]["specialist"]["total"]["mean"] < out["tokens"]["alone"]["total"]["mean"]
                                           if out["tokens"]["specialist"]["total"]["mean"] is not None and out["tokens"]["alone"]["total"]["mean"] is not None else None),
    }
    for t in tasks:
        i = t["id"]
        row = {"id": i, "families": t["families"], "held_out": held[i], "held_out_map": t["held_out"], "steps": len(t["steps"])}
        for arm in ARMS:
            r, g = results.get(arm, {}).get(i), grades.get(arm, {}).get(i)
            row[arm] = None if r is None else {
                "passed": bool(g and g.get("passed")), "keys": f"{g['n_pass']}/{g['n_keys']}" if g else "-",
                "status": g.get("status") if g else "ungraded", "tokens_total": (r.get("usage") or {}).get("total"),
                "tokens_fresh": (r.get("usage") or {}).get("fresh"), "tokens_output": (r.get("usage") or {}).get("output"),
                "wall_s": r.get("wall_s"), "tool_calls": r.get("n_tool_calls"), "solve": r.get("n_solve"),
                "solve_verified": r.get("n_solve_verified"), "stopped_by": r.get("stopped_by")}
        out["per_task"].append(row)
    return out


def _fmt(x, nd=0):
    return "-" if x is None else (f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_report(name: str, s: dict) -> str:
    L = [f"# E2E report: run `{name}`", ""]
    L += ["## Success rate (task passes when every key passes)", "",
          "| arm | tasks run | passed | rate | Wilson 95 % CI | held-out-family tasks passed | answer status |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        a = s["arms"][arm]
        ci = a["wilson95"]
        L.append(f"| {arm} | {a['n_run']} | {a['n_pass']} | {_fmt(a['rate'] and 100 * a['rate'], 1)} % | "
                 f"{_fmt(100 * ci[0], 1)}-{_fmt(100 * ci[1], 1)} % | {a['held_out_family_tasks']['n_pass']}/{a['held_out_family_tasks']['n']} | "
                 + ", ".join(f"{k} {v}" for k, v in sorted(a["status"].items())) + " |")
    p = s["paired"]
    L += ["", "## Paired comparison over tasks run in both arms", "",
          f"n = {p['n']}: both pass {p['both']}, alone only {p['alone_only']}, specialist only {p['specialist_only']}, neither {p['neither']}. "
          f"Exact two-sided sign test on the {p['alone_only'] + p['specialist_only']} discordant tasks: p = {p['sign_test_p']:.3f}.", ""]
    L += ["## Orchestrator tokens", "",
          "`total` = input + cache_creation + cache_read + output (every token processed or produced), `fresh` = input + "
          "cache_creation + output, `output` = generated tokens. Per task, from the final `result` event.", "",
          "| arm | metric | mean | median | total |", "|---|---|---|---|---|"]
    for arm in ARMS:
        for m in TOKEN_METRICS:
            d = s["tokens"][arm][m]
            L.append(f"| {arm} | {m} | {_fmt(d['mean'])} | {_fmt(d['median'])} | {_fmt(d['total'])} |")
    L += ["", "Paired per-task ratio specialist / alone:", "", "| metric | n | median | IQR | mean of ratios | ratio of means | tasks with fewer / more | sign test p |",
          "|---|---|---|---|---|---|---|---|"]
    for m in TOKEN_METRICS:
        r = s["token_ratio_specialist_over_alone"][m]
        iqr = "-" if not r["iqr"] else f"{r['iqr'][0]:.2f}-{r['iqr'][1]:.2f}"
        L.append(f"| {m} | {r['n']} | {_fmt(r['median'], 2)} | {iqr} | {_fmt(r['mean_of_ratios'], 2)} | {_fmt(r['ratio_of_means'], 2)} | "
                 f"{r['specialist_fewer']} / {r['specialist_more']} | {r['sign_test_p']:.3f} |")
    L += ["", "## Wall time (seconds per task)", "", "| arm | mean | median | total |", "|---|---|---|---|"]
    for arm in ARMS:
        d = s["wall"][arm]
        L.append(f"| {arm} | {_fmt(d['mean'], 1)} | {_fmt(d['median'], 1)} | {_fmt(d['total'], 0)} |")
    w = s["wall_ratio_specialist_over_alone"]
    L += ["", f"Paired wall-time ratio specialist / alone: median {_fmt(w['median'], 2)}" + ("" if not w["iqr"] else f", IQR {w['iqr'][0]:.2f}-{w['iqr'][1]:.2f}") + f" (n = {w['n']}).", ""]
    L += ["## Tool use", "", "| arm | tool calls / task | Bash / task | turns / task | tool errors / task | opencv_solve calls (total) | verified | tasks using it |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        t = s["arms"][arm]["tools"]
        L.append(f"| {arm} | {_fmt(t['calls_per_task']['mean'], 1)} | {_fmt(t['bash_per_task']['mean'], 1)} | {_fmt(t['turns_per_task']['mean'], 1)} | "
                 f"{_fmt(t['tool_errors_per_task']['mean'], 2)} | {t['solve_calls_total']} | {t['solve_verified_total']} | {t['tasks_using_solve']} |")
    L += ["", "## Run hygiene", ""]
    for arm in ARMS:
        a = s["arms"][arm]
        L.append(f"- {arm}: stopped_by {a['stopped_by']}; isolation problems in {a['isolation_problems'] or 'none'}; transcripts that touched the repo or other runs: {a['peeked'] or 'none'}; permission denials: {a['permission_denials'] or 'none'}.")
    c = s["criterion4_inputs"]
    L += ["", "## Criterion 4 inputs (frozen 2026-10-22: specialist >= alone on success, with fewer orchestrator tokens)", "",
          f"- specialist success rate >= alone: {c['specialist_success_ge_alone']}",
          f"- median paired token ratio (total): {_fmt(c['median_token_ratio_total'], 2)}; mean tokens specialist < alone: {c['mean_tokens_specialist_lt_alone']}",
          "- these are the mechanical inputs; the decision stays with the project owner (CLAUDE.md rule 10).", ""]
    L += ["## Per task", "",
          "| task | steps | families (* held out) | alone | tokens | s | specialist | tokens | s | solve calls (verified) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["per_task"]:
        fams = ", ".join(f + ("*" if r["held_out_map"].get(f) else "") for f in r["families"])

        def cell(a):
            return ("-", "-", "-") if a is None else (("PASS" if a["passed"] else "FAIL") + f" {a['keys']}", _fmt(a["tokens_total"]), _fmt(a["wall_s"], 0))

        a, b = cell(r["alone"]), cell(r["specialist"])
        sv = "-" if r["specialist"] is None else f"{r['specialist']['solve']} ({r['specialist']['solve_verified']})"
        L.append(f"| {r['id']} | {r['steps']} | {fams} | {a[0]} | {a[1]} | {a[2]} | {b[0]} | {b[1]} | {b[2]} | {sv} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- commands

def _tasks_from(only: str | None, limit: int | None, ids: list[str]) -> list[str]:
    sel = [i for i in ids if not only or re.search(only, i)]
    return sel[:limit] if limit else sel


def cmd_build(a) -> None:
    from serve.e2e import tasks as T
    import cv2

    lay = Layout(Path(a.root), a.run)
    if lay.run.exists() and not a.force:
        sys.exit(f"{lay.run} exists; pass --force to rewrite the inputs and config (logs and grades are kept)")
    dist = a.distro or distro_name()
    metas, hashes = [], {}
    tasks = [T.build(i) for i in T.TASK_IDS if not a.only or re.search(a.only, i)]
    for arm in ARMS:
        for t in tasks:
            wd = lay.work(arm, t.id)
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            for name, im in t.images.items():
                path = wd / f"{name}.png"
                assert cv2.imwrite(str(path), im), path
                hashes.setdefault(t.id, {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            p = lay.prompt(arm, t.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(T.render_prompt(t, str(wd), to_windows_path(wd, dist)), encoding="utf-8")
    metas = [t.meta() for t in tasks]
    lay.mcp.write_text(json.dumps(mcp_config(dist, a.serve_python, a.server), indent=2))
    cfg = {"name": a.run, "created": datetime.now(timezone.utc).isoformat(), "root": str(lay.root), "distro": dist,
           "claude": a.claude, "mcp_config": str(lay.mcp), "mcp_server": {"python": a.serve_python, "server": a.server},
           "arms": list(ARMS), "builtin_tools": list(BUILTIN_TOOLS), "tasks": metas, "input_sha256": hashes,
           "seed_range": list(T.cp.SEED_RANGE)}
    lay.config.write_text(json.dumps(cfg, indent=2))
    print(f"built {len(tasks)} tasks x {len(ARMS)} arms under {lay.run}")
    print(f"  config  {lay.config}\n  mcp     {lay.mcp}")


def cmd_predict(a) -> None:
    lay = Layout(Path(a.root), a.run)
    cfg = load_config(lay)
    ids = _tasks_from(a.only, a.limit, [t["id"] for t in cfg["tasks"]])
    n = len(ids)
    tok = {"alone": a.tokens_per_task, "specialist": a.spec_tokens_per_task or a.tokens_per_task}
    sec = {"alone": a.s_per_task, "specialist": a.spec_s_per_task or a.s_per_task}
    print(f"prediction for run {a.run}: {n} tasks per arm, one `claude.exe -p` session each (Rule 1: predict before spending)")
    tot_t = tot_s = 0.0
    for arm in ARMS:
        t, s = n * tok[arm], n * sec[arm]
        tot_t += t
        tot_s += s
        print(f"  {arm:<10} {n} x {tok[arm]:>10,.0f} tokens = {t:>14,.0f} tokens   {n} x {sec[arm]:>6,.0f} s = {s / 60:>7.1f} min ({s / 3600:.2f} h)")
    print(f"  {'both arms':<10} {2 * n} sessions: {tot_t:,.0f} tokens, {tot_s / 60:.1f} min ({tot_s / 3600:.2f} h) sequential wall time")
    print("  tokens are the per-session `total` counter (input + cache_creation + cache_read + output); record this line in LOGBOOK.md before `run`")


def _cwd_for(a, workdir: Path) -> str:
    return str(workdir) if a.cwd == "task" else a.cwd


def _stream(proc: subprocess.Popen, transcript, on_event, deadline: float, max_assistant: int | None) -> str:
    """Copy stdout lines to the transcript, feed events to on_event; returns why it stopped."""
    why = {"v": "completed"}

    def pump():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            transcript.write(line + "\n")
            transcript.flush()
            try:
                on_event(json.loads(line))
            except json.JSONDecodeError:
                pass

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while th.is_alive():
        th.join(timeout=1.0)
        if time.monotonic() > deadline:
            why["v"] = "timeout"
            break
        if max_assistant and on_event.n_assistant > max_assistant:
            why["v"] = "max_turns"
            break
    if why["v"] != "completed":
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    th.join(timeout=10)
    return why["v"]


class _Counter:
    def __init__(self):
        self.ids: set = set()
        self.n_assistant = 0

    def __call__(self, ev: dict) -> None:
        if ev.get("type") == "assistant":
            mid = (ev.get("message") or {}).get("id")
            if mid not in self.ids:
                self.ids.add(mid)
                self.n_assistant += 1


def run_one(a, cfg: dict, lay: Layout, arm: str, tid: str) -> dict | None:
    work, logd = lay.work(arm, tid), lay.log(arm, tid)
    prompt = lay.prompt(arm, tid).read_text(encoding="utf-8")
    dist = cfg["distro"]
    cmd = build_command(arm=arm, model=a.model, claude_exe=a.claude or cfg["claude"],
                        mcp_config_win=to_windows_path(lay.mcp, dist), workdir_win=to_windows_path(work, dist),
                        isolation=a.isolation, max_turns=a.max_turns, max_budget_usd=a.max_budget_usd, effort=a.effort)
    env = build_env(os.environ, arm=arm, isolation=a.isolation, extra=dict(kv.split("=", 1) for kv in a.env))
    cwd = _cwd_for(a, work)
    if a.dry_run:
        print(f"[dry-run] {arm} {tid}")
        print(f"  cwd:   {cwd}   (a Windows process sees {to_windows_path(cwd, dist)})")
        added = {k: env[k] for k in env if k.startswith("CLAUDE_CODE_") or k in ("MCP_TIMEOUT", "WSLENV")}
        print(f"  env+:  {added}")
        print(f"  cmd:   {shell_line(cmd)}")
        print(f"  stdin: the prompt, {len(prompt)} chars, sha1 {hashlib.sha1(prompt.encode()).hexdigest()[:12]}")
        if a.show_prompt:
            print("  ---- prompt ----\n" + prompt + "\n  ---- end ----")
        return None
    if (logd / "result.json").exists() and not a.rerun:
        print(f"skip {arm} {tid}: result exists")
        return None
    # a clean work dir: inputs only, so nothing from an earlier aborted attempt can grade as this run's answer
    for stale in ("answer.json", "out.png"):
        (work / stale).unlink(missing_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    counter = _Counter()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    with open(logd / "transcript.jsonl", "w", encoding="utf-8") as tf, open(logd / "stderr.txt", "wb") as ef:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ef)
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
        why = _stream(proc, tf, counter, t0 + a.timeout, (a.max_turns + 3) if a.max_turns else None)
        rc = proc.wait()
    wall = time.monotonic() - t0
    events = []
    for line in (logd / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    res = summarize_events(events, arm=arm, task_id=tid, run_name=lay.name)
    if why == "completed":
        if res["result_subtype"] == "error_max_turns":
            why = "max_turns"
        elif res["result_subtype"] == "error_max_budget_usd":
            why = "max_budget"
        elif rc != 0 or not res["has_result_event"] or res["is_error"]:
            why = "error"
    res.update({"task": tid, "arm": arm, "model": a.model, "exit_code": rc, "wall_s": wall, "stopped_by": why,
                "started": started, "cwd": cwd, "isolation": a.isolation, "cmd": cmd,
                "stderr_tail": (logd / "stderr.txt").read_bytes()[-2000:].decode("utf-8", "replace")})
    (logd / "result.json").write_text(json.dumps(res, indent=2))
    u = res["usage"]
    print(f"{arm} {tid}: {why} rc={rc} {wall:.0f}s tokens total {u['total']:,} (fresh {u['fresh']:,}, out {u['output']:,}) "
          f"tools {res['n_tool_calls']} solve {res['n_solve']}({res['n_solve_verified']}) isolation_ok={res['isolation_ok']}")
    return res


def cmd_run(a) -> None:
    lay = Layout(Path(a.root), a.run)
    cfg = load_config(lay)
    ids = _tasks_from(a.only, a.limit, [t["id"] for t in cfg["tasks"]])
    if not ids:
        sys.exit("no task matches --only")
    if not a.dry_run and a.cwd != "task" and not Path(a.cwd).is_dir():
        sys.exit(f"--cwd {a.cwd} is not a directory")
    print(f"{'dry run: ' if a.dry_run else ''}{a.arm} arm, {len(ids)} tasks, model {a.model}")
    for tid in ids:
        run_one(a, cfg, lay, a.arm, tid)


def _load_results(lay: Layout, cfg: dict) -> tuple[dict, dict]:
    res = {arm: {} for arm in ARMS}
    gr = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for t in cfg["tasks"]:
            r = lay.log(arm, t["id"]) / "result.json"
            if r.is_file():
                res[arm][t["id"]] = json.loads(r.read_text())
            g = lay.grade(arm, t["id"])
            if g.is_file():
                gr[arm][t["id"]] = json.loads(g.read_text())
    return res, gr


def cmd_grade(a) -> None:
    from serve.e2e import grade as G
    from serve.e2e import tasks as T

    lay = Layout(Path(a.root), a.run)
    cfg = load_config(lay)
    results, _ = _load_results(lay, cfg)
    n = 0
    for arm in ARMS:
        for t in cfg["tasks"]:
            tid = t["id"]
            if tid not in results[arm] or (a.arm and arm != a.arm):
                continue
            res = G.grade_workdir(T.build(tid), lay.work(arm, tid))
            lay.grade(arm, tid).parent.mkdir(parents=True, exist_ok=True)
            lay.grade(arm, tid).write_text(json.dumps(res, indent=2))
            bad = [k for k, v in res["keys"].items() if not v["ok"]]
            print(f"{arm:<10} {tid:<32} {'PASS' if res['passed'] else 'FAIL'} {res['n_pass']}/{res['n_keys']} [{res['status']}]"
                  + ("" if res["passed"] else f"  failed: {', '.join(bad) or res['detail']}"))
            n += 1
    if not n:
        print("nothing to grade: no finished run")


def cmd_report(a) -> None:
    lay = Layout(Path(a.root), a.run)
    cfg = load_config(lay)
    results, grades = _load_results(lay, cfg)
    s = summarize(results, grades, cfg["tasks"])
    (lay.run / "summary.json").write_text(json.dumps(s, indent=2))
    (lay.run / "report.md").write_text(render_report(a.run, s))
    print((lay.run / "report.md").read_text())
    print(f"wrote {lay.run / 'report.md'} and {lay.run / 'summary.json'}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--run", required=True, help="run name")
        p.add_argument("--root", default=str(DEFAULT_ROOT), help="parent directory of the runs")

    p = sub.add_parser("build", help="write the work dirs, prompts, mcp.json and config.json")
    common(p)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", help="regex on task ids")
    p.add_argument("--distro")
    p.add_argument("--claude", default=DEFAULT_CLAUDE)
    p.add_argument("--serve-python", default=SERVE_PYTHON)
    p.add_argument("--server", default=SERVER_PY)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("predict", help="predicted totals for both arms (write them in the logbook before `run`)")
    common(p)
    p.add_argument("--tokens-per-task", type=float, required=True)
    p.add_argument("--s-per-task", type=float, required=True)
    p.add_argument("--spec-tokens-per-task", type=float)
    p.add_argument("--spec-s-per-task", type=float)
    p.add_argument("--only")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("run", help="run claude.exe -p once per task (skips tasks with a result)")
    common(p)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--only", help="regex on task ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-turns", type=int)
    p.add_argument("--max-budget-usd", type=float)
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="wall-clock limit per task, seconds")
    p.add_argument("--cwd", default="task", help="'task' (the work dir; a UNC path to Windows) or a Linux path used for every task "
                   "(e.g. /mnt/c/Users/ptmel/e2e-cwd if a UNC cwd does not work)")
    p.add_argument("--isolation", choices=sorted(ISOLATION_ARGS), default="flags")
    p.add_argument("--claude", help="path of claude.exe (default: the one recorded at build)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment for claude.exe")
    p.add_argument("--rerun", action="store_true", help="run again even if a result exists")
    p.add_argument("--dry-run", action="store_true", help="print the command lines; write nothing, start nothing")
    p.add_argument("--show-prompt", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("grade", help="grade every finished run")
    common(p)
    p.add_argument("--arm", choices=ARMS)
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("report", help="write report.md and summary.json")
    common(p)
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
