#!/usr/bin/env python3
"""Synchronize the canonical public catalog into installed package resources."""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
PACKAGE = ROOT / "panorai" / "catalog"
RESOURCES = (
    "public-api-surfaces-v2.json",
    "public-api-surfaces-v2.schema.json",
)


def _module_path(module_name: str) -> Path:
    relative = module_name.removeprefix("panorai").strip(".").replace(".", "/")
    directory = ROOT / "panorai" / relative
    return (
        directory / "__init__.py"
        if directory.is_dir()
        else directory.with_suffix(".py")
    )


def _literal_all(module_name: str) -> list[str]:
    path = _module_path(module_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            exports = ast.literal_eval(value)
            if not isinstance(exports, (list, tuple)) or not all(
                isinstance(item, str) for item in exports
            ):
                raise ValueError(f"{module_name}.__all__ must be a literal string list")
            return list(exports)
    raise ValueError(f"{module_name} does not declare __all__")


def _apply_namespace_defaults(
    manifest: dict, module_exports: dict[str, list[str]]
) -> tuple[dict, dict[str, list[str]]]:
    """Add only previously unclassified exports to Experimental catchalls."""

    generated = copy.deepcopy(manifest)
    discovery = generated["policy"]["public_namespace_discovery"]
    defaults = discovery.get("default_surfaces", {})
    surfaces = {item["id"]: item for item in generated["surfaces"]}
    additions: dict[str, list[str]] = {}
    for module_name, target_id in defaults.items():
        if module_name not in module_exports:
            continue
        classified = {
            name
            for surface in generated["surfaces"]
            if surface.get("module") == module_name
            for key in ("exports_exact", "classified_exports", "required_symbols")
            for name in surface.get(key, ())
        }
        missing = [
            name for name in module_exports[module_name] if name not in classified
        ]
        if not missing:
            continue
        target = surfaces.get(target_id)
        if target is None:
            assignments = generated["documentation"]["assignments"]
            sources = [
                surface
                for surface in generated["surfaces"]
                if surface.get("module") == module_name
                and surface.get("canonical_import") == module_name
                and surface.get("visibility") != "internal"
            ]
            if not sources:
                raise ValueError(
                    f"cannot derive an automatic surface for {module_name!r}"
                )
            source_ids = {surface["id"] for surface in sources}
            source = next(
                (
                    surface
                    for surface in sources
                    if surface.get("visibility") == "primary"
                ),
                sources[0],
            )
            target = {
                "id": target_id,
                "domain": source["domain"],
                "family": source["family"],
                "kind": "component",
                "visibility": "advanced",
                "canonical_import": module_name,
                "replacement": None,
                "contract": f"{module_name}/auto-experimental",
                "tier": "experimental",
                "module": module_name,
                "classified_exports": [],
                "promotion_gates": [
                    "deliberate assignment to a versioned public surface",
                    "scientific, documentation, and installed-wheel evidence appropriate to the symbol",
                ],
            }
            generated["surfaces"].append(target)
            surfaces[target_id] = target
            roles = [
                role
                for role, surface_ids in assignments.items()
                if source_ids.intersection(surface_ids)
            ]
            if not roles:
                raise ValueError(
                    f"cannot derive a documentation role for {module_name!r}"
                )
            for role in roles:
                assignments[role].append(target_id)
        if target["tier"] != "experimental":
            raise ValueError(f"default surface {target_id!r} must remain Experimental")
        target.setdefault("classified_exports", []).extend(missing)
        additions[module_name] = missing
    return generated, additions


def _register_application_docs(manifest: dict, additions: dict[str, list[str]]) -> None:
    names = additions.get("panorai.applications", [])
    if not names:
        return
    from panorai.applications import Application
    import panorai.applications as applications

    components = manifest["documentation"]["application_components"]
    surface_id = manifest["policy"]["public_namespace_discovery"]["default_surfaces"][
        "panorai.applications"
    ]
    for name in names:
        value = getattr(applications, name)
        if not inspect.isclass(value) or not issubclass(value, Application):
            continue
        if inspect.isabstract(value):
            continue
        components[name] = {
            "surface": surface_id,
            "documentation": "reference/applications",
            "documentation_token": ".. panorai-api-components:: application",
        }


def generate_canonical(*, check: bool) -> bool:
    path = CONTRACTS / RESOURCES[0]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    defaults = manifest["policy"]["public_namespace_discovery"].get(
        "default_surfaces", {}
    )
    exports = {module: _literal_all(module) for module in defaults}
    generated, additions = _apply_namespace_defaults(manifest, exports)
    _register_application_docs(generated, additions)
    if generated == manifest:
        return False
    if not check:
        path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return True


def synchronize(*, check: bool) -> tuple[str, ...]:
    mismatches: list[str] = []
    for name in RESOURCES:
        source = CONTRACTS / name
        target = PACKAGE / name
        payload = source.read_bytes()
        if target.is_file() and target.read_bytes() == payload:
            continue
        mismatches.append(name)
        if not check:
            target.write_bytes(payload)
    return tuple(mismatches)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when packaged resources differ",
    )
    arguments = parser.parse_args(argv)
    generated_mismatch = generate_canonical(check=arguments.check)
    mismatches = synchronize(check=arguments.check)
    if arguments.check and (generated_mismatch or mismatches):
        if generated_mismatch:
            print("out of sync: generated public exports", file=sys.stderr)
        for name in mismatches:
            print(f"out of sync: {name}", file=sys.stderr)
        return 1
    if mismatches:
        for name in mismatches:
            print(f"synchronized: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
