#!/usr/bin/env python3
"""Synchronize the canonical public catalog into installed package resources."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
PACKAGE = ROOT / "panorai" / "catalog"
RESOURCES = (
    "public-api-surfaces-v2.json",
    "public-api-surfaces-v2.schema.json",
)


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
    mismatches = synchronize(check=arguments.check)
    if arguments.check and mismatches:
        for name in mismatches:
            print(f"out of sync: {name}", file=sys.stderr)
        return 1
    if mismatches:
        for name in mismatches:
            print(f"synchronized: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
