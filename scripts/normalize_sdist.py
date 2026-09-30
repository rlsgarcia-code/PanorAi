#!/usr/bin/env python3
"""Normalize sdist archive metadata for byte-reproducible release builds."""

from __future__ import annotations

import argparse
import copy
import gzip
import os
from pathlib import Path
import tarfile


def normalize_sdist(path: Path, *, epoch: int) -> None:
    """Rewrite ``path`` atomically with deterministic tar and gzip metadata."""

    path = path.resolve()
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if not path.name.endswith(".tar.gz"):
        raise ValueError("sdist path must end with .tar.gz")
    temporary = path.with_name(f".{path.name}.normalized")
    try:
        with (
            tarfile.open(path, "r:gz") as source,
            temporary.open("wb") as raw_output,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=epoch,
            ) as compressed,
            tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.PAX_FORMAT,
            ) as target,
        ):
            for member in source.getmembers():
                normalized = copy.copy(member)
                normalized.mtime = epoch
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.pax_headers = {}
                payload = source.extractfile(member) if member.isfile() else None
                target.addfile(normalized, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    normalize_sdist(args.artifact, epoch=args.epoch)


if __name__ == "__main__":
    main()
