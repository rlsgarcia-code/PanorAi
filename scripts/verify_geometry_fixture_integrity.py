#!/usr/bin/env python3
"""Verify the raw-byte geometry-v1 fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_ROOT = REPOSITORY_ROOT / "tests/fixtures/geometry/v1"


def verify_fixture_integrity(root: Path) -> int:
    """Return the verified file count or raise on the first mismatch."""

    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != metadata["sha256"]:
            raise AssertionError(f"checksum mismatch: {name}")
    return len(manifest["files"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-root", type=Path, default=DEFAULT_FIXTURES_ROOT)
    args = parser.parse_args()
    count = verify_fixture_integrity(args.fixtures_root)
    print(f"geometry-v1 fixture integrity: OK files={count}")


if __name__ == "__main__":
    main()
