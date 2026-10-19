#!/usr/bin/env python3
"""Emit one installed-wheel matrix record for the reviewed DoG profile."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray


PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _texture() -> NDArray[np.float32]:
    y, x = np.indices((96, 192), dtype=np.float32)
    base = 0.5 + 0.22 * np.sin(x / 7.0) + 0.18 * np.cos(y / 5.0)
    return np.stack((base, np.roll(base, 9, axis=1), base), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--platform-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    if len(args.source_commit) != 40:
        raise ValueError("source_commit must be a complete Git SHA")
    if not args.wheel.is_file():
        raise FileNotFoundError(args.wheel)

    import panorai
    from panorai.applications import SphericalDoGFeatures

    origin = Path(panorai.__file__).resolve()
    try:
        origin.relative_to(args.source_root.resolve())
    except ValueError:
        isolated = True
    else:
        isolated = False
    if not isolated:
        raise AssertionError(f"panorai resolved inside source checkout: {origin}")

    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    result = method(_texture(), panorama_id="installed-reviewed-dog")
    restored = SphericalDoGFeatures.from_config(method.to_config())
    if not len(result) or result.descriptors.shape != (len(result), 128):
        raise AssertionError("reviewed DoG application produced no real descriptors")
    if restored != method or restored.profile_id != PROFILE_ID:
        raise AssertionError("reviewed DoG profile identity did not round-trip")
    if SphericalDoGFeatures().stability != "experimental":
        raise AssertionError("generic DoG constructor was promoted unexpectedly")

    record = {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": "installed-wheel",
        "profile_id": PROFILE_ID,
        "source_commit": args.source_commit,
        "wheel_sha256": _sha256(args.wheel),
        "artifact_version": version("panorai"),
        "platform": args.platform_id
        or f"{platform.system().lower()}-{platform.machine().lower()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "metrics": {
            "exact_commit_and_wheel_sha_bound": True,
            "source_checkout_isolated": True,
            "profile_identity_roundtrip": True,
            "real_application_execution": True,
            "generic_constructor_remains_experimental": True,
            "feature_count": len(result),
        },
        "origin": str(origin),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output} sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
