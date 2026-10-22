"""AST-normalized identity of a solution, for dedup (PLAN.md Phase 1 hygiene: "AST-normalized dedup").

Two samples that differ only in comments, docstrings, whitespace, quoting, parentheses, type annotations, the names the
author gave to variables and helpers, or the order and spelling of the imports are the same training example. This
module reduces such pairs to the same text, ``normalize_code``, and hashes it, ``ast_hash``.

What is normalized (and nothing else):

  * comments, blank lines, formatting, quote style, redundant parentheses: lost by parsing and ``ast.unparse``
  * docstrings and any other bare string statement (a string used as a comment): removed
  * annotations: parameter and return annotations dropped; ``x: T = v`` becomes ``x = v``; ``x: T`` is dropped
  * names bound anywhere in the code (variables, parameters, loop and ``with`` targets, comprehension variables,
    ``except ... as e``, helper functions and classes, import aliases, the targets of ``global``/``nonlocal``) are
    renamed ``_v0, _v1, ...`` in order of first binding. ``solve`` keeps its name (it is the contract). Names that are
    never bound in the code (builtins, ``cv2`` used without an import) and attribute or keyword-argument names
    (``arr.shape``, ``dtype=``) are left alone, so ``cv2.threshold`` stays ``_v0.threshold`` only through the alias of
    its import, and the module paths of the imports are kept.
  * imports: ``import a, b`` is split into one statement per module, from-import names are sorted, and the module-level
    import statements are sorted among themselves (by module and imported names, not by alias) before names are
    numbered, so the order of the import block does not change the numbering.

What is not: statement order, inlining of temporaries, constant folding, the order of operands (``a + b`` vs ``b + a``),
literal values, default arguments, which cv2 function is called. Two solutions that compute the same thing in a
different way stay distinct, which is the point: dedup removes copies, it does not judge algorithms.

Code that does not parse falls back to a whitespace-normalized text hash (prefixed so it cannot collide with an AST one).
"""
from __future__ import annotations

import ast
import hashlib
from typing import Iterator

KEEP_NAMES = frozenset({"solve"})
_STMT_LISTS = ("body", "orelse", "finalbody")


def _preorder(node: ast.AST) -> Iterator[ast.AST]:
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _preorder(child)


def _import_key(node: ast.stmt) -> tuple:
    if isinstance(node, ast.Import):
        return (0, node.names[0].name, "", 0)
    return (1, node.module or "", ",".join(a.name for a in node.names), node.level)


def _canonical_imports(body: list[ast.stmt]) -> list[ast.stmt]:
    """Split ``import a, b``, sort from-import names, then sort the import statements into the slots imports occupy."""
    expanded: list[ast.stmt] = []
    for st in body:
        if isinstance(st, ast.Import) and len(st.names) > 1:
            expanded += [ast.copy_location(ast.Import(names=[a]), st) for a in st.names]
        else:
            if isinstance(st, ast.ImportFrom):
                st.names = sorted(st.names, key=lambda a: (a.name, a.asname or ""))
            expanded.append(st)
    slots = [i for i, st in enumerate(expanded) if isinstance(st, (ast.Import, ast.ImportFrom))]
    imports = sorted((expanded[i] for i in slots), key=_import_key)
    for i, st in zip(slots, imports):
        expanded[i] = st
    return expanded


def _bound_names(tree: ast.AST) -> dict[str, str]:
    """old name -> canonical name, numbered by first binding in a source-order walk."""
    order: dict[str, str] = {}

    def add(name: str | None) -> None:
        if name and name not in KEEP_NAMES and name not in order:
            order[name] = f"_v{len(order)}"

    for node in _preorder(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add(node.id)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.alias):
            if node.name != "*":
                add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                add(n)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            add(node.name)
        elif isinstance(node, ast.MatchMapping):
            add(node.rest)
    return order


def _is_string_statement(st: ast.stmt) -> bool:
    return isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str)


class _Canon(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]):
        self.names = names

    def _rename(self, name: str | None) -> str | None:
        return self.names.get(name, name) if name else name

    def _clean_lists(self, node: ast.AST) -> None:
        for field in _STMT_LISTS:
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and all(isinstance(s, ast.stmt) for s in stmts):
                kept = [s for s in stmts if not _is_string_statement(s)]
                setattr(node, field, kept or ([ast.Pass()] if field == "body" else []))

    def generic_visit(self, node: ast.AST) -> ast.AST:
        self._clean_lists(node)
        node = super().generic_visit(node)
        if isinstance(getattr(node, "body", None), list) and not node.body and isinstance(node, ast.stmt):
            node.body = [ast.Pass()]  # a bare annotation was the whole body
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg, node.annotation, node.type_comment = self._rename(node.arg), None, None
        return node

    def _func(self, node):
        node.name, node.returns, node.type_comment = self._rename(node.name), None, None
        return self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = _func

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._rename(node.name)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.name != "*":
            bound = node.asname or node.name.split(".")[0]
            node.asname = self._rename(bound)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        node.name = self._rename(node.name)
        return self.generic_visit(node)

    def _names(self, node):
        node.names = [self._rename(n) for n in node.names]
        return node

    visit_Global = visit_Nonlocal = _names

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        node.name = self._rename(node.name)
        return self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        node.rest = self._rename(node.rest)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return None  # a bare annotation binds nothing
        return self.visit(ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node))


def normalize_code(code: str) -> str:
    """The canonical text of ``code`` (see the module docstring). Raises SyntaxError if it does not parse."""
    tree = ast.parse(code)
    tree.body = _canonical_imports(tree.body)
    tree = _Canon(_bound_names(tree)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def ast_hash(code: str) -> str:
    """sha1 of the normalized code, or of its whitespace-normalized text when it does not parse."""
    try:
        text = normalize_code(code)
    except (SyntaxError, ValueError, RecursionError):
        text = "RAW:" + " ".join(code.split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
