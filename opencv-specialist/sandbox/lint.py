"""Static gate for generated solutions, run before anything executes.

The lint rejects code that *announces* a forbidden behaviour. It is a cheap first
gate, not a security boundary: the runtime guards in run.py catch what the lint
misses (best effort), and a container with ``--network none`` is the real wall
when the data foundry runs thousands of untrusted snippets.

Rules
-----
syntax         the code does not parse
contract       no top-level ``def solve(...)``
import         modules a worker never needs (network, processes, FFI, files, plotting)
builtin        open/eval/exec/compile/__import__/input/breakpoint/exit/quit
gui            cv2 window/GUI calls: they hang or fail in a headless worker
io             cv2 file/device I/O: the contract passes arrays in and out
broad-except   bare ``except:`` or ``except (Base)Exception``: it hides failures from the
               verifier, and exec-only filtering would *select for* it
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

BANNED_IMPORTS = {
    "os", "sys", "socket", "ssl", "urllib", "requests", "http", "ftplib", "smtplib",
    "subprocess", "multiprocessing", "threading", "concurrent", "asyncio", "signal",
    "ctypes", "cffi", "resource", "importlib", "shutil", "pathlib", "tempfile", "glob",
    "pickle", "marshal", "shelve", "builtins", "matplotlib", "PIL", "IPython",
}
BANNED_BUILTINS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "exit", "quit"}
CV2_FORBIDDEN = {
    **{n: ("gui", "GUI calls hang or fail in a headless worker; return arrays instead")
       for n in ("imshow", "waitKey", "waitKeyEx", "pollKey", "namedWindow", "destroyWindow",
                 "destroyAllWindows", "setMouseCallback", "createTrackbar", "selectROI", "selectROIs")},
    **{n: ("io", "the contract passes arrays in and out; no file or device I/O")
       for n in ("imread", "imreadmulti", "imwrite", "imwritemulti", "VideoCapture", "VideoWriter")},
}


@dataclass(frozen=True)
class Finding:
    rule: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.message}"


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.aliases: dict[str, str] = {}  # local name -> root module

    def flag(self, rule: str, node: ast.AST, message: str) -> None:
        self.findings.append(Finding(rule, getattr(node, "lineno", 0), message))

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            self.aliases[a.asname or root] = root
            if root in BANNED_IMPORTS:
                self.flag("import", node, f"`import {a.name}` is not allowed in a solution")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in BANNED_IMPORTS:
            self.flag("import", node, f"`from {node.module} import ...` is not allowed in a solution")
        if root == "cv2":
            for a in node.names:
                if a.name == "*":
                    self.flag("import", node, "`from cv2 import *` hides which API is used")
                elif a.name in CV2_FORBIDDEN:
                    rule, msg = CV2_FORBIDDEN[a.name]
                    self.flag(rule, node, f"cv2.{a.name}: {msg}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (isinstance(node.value, ast.Name) and self.aliases.get(node.value.id) == "cv2"
                and node.attr in CV2_FORBIDDEN):
            rule, msg = CV2_FORBIDDEN[node.attr]
            self.flag(rule, node, f"cv2.{node.attr}: {msg}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
            self.flag("builtin", node, f"`{node.func.id}()` is not allowed in a solution")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.flag("broad-except", node, "bare `except:` hides failures from the verifier")
        else:
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            for t in types:
                if isinstance(t, ast.Name) and t.id in ("Exception", "BaseException"):
                    self.flag("broad-except", node, f"`except {t.id}` hides failures from the verifier")
        self.generic_visit(node)


def lint(code: str, required_func: str = "solve") -> list[Finding]:
    """Return findings sorted by line; an empty list means the code may run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding("syntax", e.lineno or 0, e.msg)]
    v = _Visitor()
    v.visit(tree)
    if not any(isinstance(n, ast.FunctionDef) and n.name == required_func for n in tree.body):
        v.findings.append(Finding("contract", 1, f"no top-level `def {required_func}(...)`"))
    return sorted(v.findings, key=lambda f: f.line)
