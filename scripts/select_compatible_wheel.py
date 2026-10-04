#!/usr/bin/env python3
"""Select the single wheel compatible with the running Python interpreter."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename


def compatible_wheels(directory: Path) -> list[Path]:
    supported = set(sys_tags())
    compatible: list[Path] = []
    for path in sorted(directory.glob("*.whl")):
        try:
            _, _, _, tags = parse_wheel_filename(path.name)
        except InvalidWheelFilename as error:
            raise ValueError(f"invalid wheel filename: {path.name}") from error
        if supported.intersection(tags):
            compatible.append(path)
    return compatible


def select_compatible_wheel(directory: Path) -> Path:
    matches = compatible_wheels(directory)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one compatible wheel in {directory}, found "
            f"{len(matches)}: {[path.name for path in matches]}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(select_compatible_wheel(args.directory).resolve())


if __name__ == "__main__":
    main()
