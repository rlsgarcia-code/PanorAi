#!/usr/bin/env python3
"""Verify that every documentation route in the public catalog was built."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contracts" / "public-api-surfaces-v2.json"


def declared_docnames(manifest: dict) -> tuple[str, ...]:
    documentation = manifest["documentation"]
    site = documentation["site"]
    names = [site["home"], site["catalog"], site["public_surfaces"]]
    for route in documentation["roles"].values():
        names.extend(route[key] for key in ("catalog", "reference", "concept", "guide"))
    names.extend(
        component["documentation"]
        for component in documentation["application_components"].values()
    )
    return tuple(dict.fromkeys(names))


def missing_routes(site_root: Path, manifest: dict) -> tuple[str, ...]:
    return tuple(
        docname
        for docname in declared_docnames(manifest)
        if not (site_root / f"{docname}.html").is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_root", type=Path)
    arguments = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = missing_routes(arguments.site_root, manifest)
    if missing:
        for docname in missing:
            print(f"missing built documentation route: {docname}.html", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
