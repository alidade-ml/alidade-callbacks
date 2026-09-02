#!/usr/bin/env python3
"""Refuse callback code that imports alidade.* (other than the vendored contract).

The callback library lives in researcher training environments.
Importing the engine would force every research repo to install the
entire orchestration stack — exactly the failure mode the three-package
split exists to prevent.

Allowed exception: nothing. Even the contract is named
``alidade_callbacks.contract`` locally (the vendor script writes
to that path), not ``alidade.contract``. There is no legitimate
reason for callback code to ``import alidade.*``.

Run from CI as ``python tools/check-no-engine-import.py``. Exits 1 on
any violation; exits 0 if clean.

See ``alidade/plans/package-audit.md`` (Check 2: Boundary check).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SKIP_PARTS = frozenset({"tests", "__pycache__", ".venv", "venv", "build", "dist"})

CALLBACK_ROOT = Path("src/alidade_callbacks")


def _find_engine_imports(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, statement), ...] for any alidade.* import in
    ``path``. Distinguished from alidade_callbacks.* — only bare
    ``alidade`` and its submodules are flagged."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return []
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "alidade" or (
                    alias.name.startswith("alidade.")
                    and not alias.name.startswith("alidade_")
                ):
                    violations.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "alidade" or (
                mod.startswith("alidade.") and not mod.startswith("alidade_")
            ):
                names = ", ".join(a.name for a in node.names)
                violations.append((node.lineno, f"from {mod} import {names}"))
    return violations


def main() -> int:
    if not CALLBACK_ROOT.is_dir():
        print(f"✗ {CALLBACK_ROOT} not found (run from repo root)", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for py in CALLBACK_ROOT.rglob("*.py"):
        if any(part in SKIP_PARTS for part in py.parts):
            continue
        for lineno, stmt in _find_engine_imports(py):
            all_violations.append(f"  {py}:{lineno}: {stmt}")

    if all_violations:
        print(
            "✗ Callback code imports alidade.* (boundary violation):",
            file=sys.stderr,
        )
        for v in all_violations:
            print(v, file=sys.stderr)
        print(
            "\nThe callback library cannot depend on the engine package — "
            "researcher training environments would have to install the "
            "entire orchestration stack. If you need a name from the "
            "contract, use alidade_callbacks.contract (vendored locally).",
            file=sys.stderr,
        )
        return 1

    print(f"✓ src/alidade_callbacks/ contains no alidade.* imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
