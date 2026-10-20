"""Fail when a Workbench adapter imports a private PanorAi path."""

from __future__ import annotations

import ast
from pathlib import Path


WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PACKAGE = WORKBENCH_ROOT / "backend" / "panorai_workbench"


def _private_part(path: str) -> str | None:
    if path == "panorai":
        return None
    return next(
        (part for part in path.split(".")[1:] if part.startswith("_")),
        None,
    )


def main() -> int:
    violations: list[str] = []
    imports: list[str] = []
    for source in sorted(BACKEND_PACKAGE.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                candidates = [node.module]
            elif isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            else:
                continue
            for candidate in candidates:
                if not candidate.startswith("panorai"):
                    continue
                imports.append(candidate)
                private = _private_part(candidate)
                if private is not None:
                    violations.append(
                        f"{source.relative_to(WORKBENCH_ROOT)}:{node.lineno}: "
                        f"private segment {private!r} in {candidate!r}"
                    )
    if not imports:
        raise SystemExit("no PanorAi imports were found in Workbench adapters")
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"public-import audit passed: {len(imports)} PanorAi imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
