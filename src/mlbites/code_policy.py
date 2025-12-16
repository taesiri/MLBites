from __future__ import annotations

"""
Best-effort policy gate for user-submitted Python code.

IMPORTANT: This is NOT a complete security sandbox.
Python-level restrictions can be bypassed by determined attackers.
Treat this as a defense-in-depth guardrail for an interview practice app,
and combine it with OS/container sandboxing in production.
"""

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    # Hard caps to reduce DoS / spam
    max_code_bytes: int = 150_000

    # Allowed imports (prefix-based). Keep intentionally small.
    allowed_import_prefixes_pytorch: tuple[str, ...] = (
        "__future__",
        "torch",
        "math",
        "typing",
        "dataclasses",
        "collections",
        "functools",
        "itertools",
        "operator",
        "statistics",
        "random",
    )
    allowed_import_prefixes_numpy: tuple[str, ...] = (
        "__future__",
        "numpy",
        "math",
        "typing",
        "dataclasses",
        "collections",
        "functools",
        "itertools",
        "operator",
        "statistics",
        "random",
    )

    # Top-level statements we allow (keep candidate side-effect light).
    # Note: `if __name__ == "__main__": ...` is allowed, but won't execute on import.
    allowed_toplevel_nodes: tuple[type[ast.AST], ...] = (
        ast.Module,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,  # docstring or harmless expression (validated below)
        ast.If,  # only allowed for __main__ guard
        ast.Pass,
    )

    # Always forbidden module prefixes (defense-in-depth).
    forbidden_import_prefixes: tuple[str, ...] = (
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "tempfile",
        "importlib",
        "runpy",
        "types",
        "inspect",
        "ctypes",
        "resource",
        "signal",
        "multiprocessing",
        "threading",
        "asyncio",
        "selectors",
        "ssl",
        "http",
        "urllib",
        "requests",
        "ftplib",
        "telnetlib",
        "pickle",
        "marshal",
    )

    # Builtins / helpers we do not allow candidates to call.
    # This list is intentionally conservative to reduce escape hatches.
    forbidden_call_names: tuple[str, ...] = (
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "breakpoint",
        "memoryview",
    )

    # Some dunder attributes are common/necessary in normal ML code (e.g. super().__init__()).
    # Keep this list extremely small.
    allowed_dunder_attribute_names: tuple[str, ...] = ("__init__",)


def _is_prefix_match(name: str, prefixes: tuple[str, ...]) -> bool:
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return True
    return False


def _framework_allowed_import_prefixes(framework: str, cfg: PolicyConfig) -> tuple[str, ...]:
    fw = (framework or "").strip().lower()
    if fw == "numpy":
        return cfg.allowed_import_prefixes_numpy
    return cfg.allowed_import_prefixes_pytorch


def validate_candidate_source(
    code: str, *, framework: str, cfg: PolicyConfig | None = None
) -> list[str]:
    """
    Return a list of policy violations. Empty list means "allowed".

    This is intentionally strict: candidates should mostly define functions/classes.
    """
    cfg = cfg or PolicyConfig()
    issues: list[str] = []

    if "\x00" in code:
        return ["Code contains NUL bytes (not allowed)."]

    code_bytes = len(code.encode("utf-8", errors="replace"))
    if code_bytes > cfg.max_code_bytes:
        issues.append(
            f"Code too large ({code_bytes} bytes). Max is {cfg.max_code_bytes} bytes."
        )
        return issues

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        # Surface syntax errors as-is; they aren't policy violations per se.
        issues.append(f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})")
        return issues

    allowed_imports = _framework_allowed_import_prefixes(framework, cfg)

    # --- top-level restrictions (reduce side effects on import) ---
    for node in getattr(tree, "body", []):
        if not isinstance(node, cfg.allowed_toplevel_nodes):
            issues.append(
                f"Top-level statement {node.__class__.__name__} is not allowed. "
                "Define functions/classes instead (no loops/with/try at top-level)."
            )
            continue

        if isinstance(node, ast.Expr):
            # Allow only module docstring at top-level.
            v = getattr(node, "value", None)
            if not (
                isinstance(v, ast.Constant)
                and isinstance(getattr(v, "value", None), str)
            ):
                issues.append(
                    "Only a module docstring string literal is allowed as a top-level expression."
                )

        if isinstance(node, ast.If):
            # Allow only: if __name__ == "__main__": ...
            test = node.test
            ok = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )
            if not ok:
                issues.append(
                    "Only `if __name__ == \"__main__\": ...` is allowed at top-level."
                )

    # --- deeper walk: imports, dunder access, dangerous calls ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _is_prefix_match(name, cfg.forbidden_import_prefixes):
                    issues.append(f"Import not allowed: `{name}`")
                elif not _is_prefix_match(name, allowed_imports):
                    issues.append(
                        f"Import not allowed: `{name}`. Allowed prefixes: {sorted(set(allowed_imports))}"
                    )

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                if _is_prefix_match(mod, cfg.forbidden_import_prefixes):
                    issues.append(f"Import not allowed: `from {mod} import ...`")
                elif not _is_prefix_match(mod, allowed_imports):
                    issues.append(
                        f"Import not allowed: `from {mod} import ...`. Allowed prefixes: {sorted(set(allowed_imports))}"
                    )

        elif isinstance(node, ast.Attribute):
            # Block dunder attribute access (`obj.__class__`, etc.)
            attr = node.attr or ""
            if attr.startswith("__") or attr.endswith("__"):
                if attr not in cfg.allowed_dunder_attribute_names:
                    issues.append(f"Dunder attribute access is not allowed: `.{attr}`")

        elif isinstance(node, ast.Name):
            # Block direct references to dunder-ish globals.
            if node.id in {"__builtins__", "__loader__", "__spec__", "__package__", "__file__"}:
                issues.append(f"Access to `{node.id}` is not allowed.")

        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                # Allow getattr/setattr/hasattr/delattr for the common "ctx" pattern,
                # but only with a constant, non-dunder attribute name.
                if fn.id in {"getattr", "setattr", "hasattr", "delattr"}:
                    if len(node.args) < 2:
                        issues.append(f"Calling `{fn.id}()` is not allowed (invalid usage).")
                        continue
                    name_arg = node.args[1]
                    if not (
                        isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)
                    ):
                        issues.append(
                            f"Calling `{fn.id}()` is only allowed with a constant attribute name string."
                        )
                        continue
                    s = name_arg.value
                    if s.startswith("__") or s.endswith("__"):
                        issues.append(
                            f"Calling `{fn.id}()` with dunder attribute names is not allowed: {s!r}"
                        )
                    continue

                if fn.id in cfg.forbidden_call_names:
                    issues.append(f"Calling `{fn.id}()` is not allowed.")

    # Dedupe but keep stable order
    seen: set[str] = set()
    out: list[str] = []
    for msg in issues:
        if msg not in seen:
            out.append(msg)
            seen.add(msg)
    return out


