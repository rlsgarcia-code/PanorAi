#!/usr/bin/env python3
"""Benchmark reviewed-v1 from an installed wheel with raw timing evidence."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray


PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _texture(shape_hw: tuple[int, int]) -> NDArray[np.float32]:
    y, x = np.indices(shape_hw, dtype=np.float32)
    base = 0.5 + 0.22 * np.sin(x / 17.0) + 0.20 * np.cos(y / 13.0)
    return cast(
        NDArray[np.float32],
        np.stack(
            (base, np.roll(base, 11, axis=1), np.roll(base, 7, axis=0)), axis=-1
        ).astype(np.float32),
    )


def _load_image(path: Path) -> NDArray[Any]:
    if path.suffix.lower() == ".npy":
        result = np.asarray(np.load(path, allow_pickle=False))
    else:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"could not decode {path}")
        result = (
            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB) if decoded.ndim == 3 else decoded
        )
    result = np.asarray(result)
    if result.ndim not in {2, 3}:
        raise ValueError("benchmark image must use HW or HWC layout")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", choices=("512x1024", "2048x4096", "native"), required=True
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark evidence: {args.output}"
        )
    if args.warmup < 1 or args.repetitions < 3:
        raise ValueError("reviewed benchmark requires >=1 warmup and >=3 repetitions")
    if len(args.source_commit) != 40:
        raise ValueError("source_commit must be a complete Git SHA")

    import panorai
    from panorai.applications import SphericalDoGFeatures

    origin = Path(panorai.__file__).resolve()
    try:
        origin.relative_to(args.source_root.resolve())
    except ValueError:
        pass
    else:
        raise AssertionError(f"panorai resolved inside the source checkout: {origin}")
    if not args.wheel.is_file():
        raise FileNotFoundError(args.wheel)

    if args.image is None:
        if args.resolution == "native":
            raise ValueError("native resolution requires --image")
        height, width = (int(item) for item in args.resolution.split("x"))
        shape = (height, width)
        image = _texture(shape)
        image_identity = f"generated-periodic-texture/{args.resolution}"
    else:
        image = _load_image(args.image)
        image_identity = _sha256(args.image)
    expected = {
        "512x1024": (512, 1024),
        "2048x4096": (2048, 4096),
    }.get(args.resolution)
    if expected is not None and image.shape[:2] != expected:
        raise ValueError(f"{args.resolution} requires image shape {expected}")

    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="auto")
    for _ in range(args.warmup):
        method(image, panorama_id="benchmark-warmup")
    samples: list[float] = []
    counts: list[int] = []
    for index in range(args.repetitions):
        started = time.perf_counter()
        result = method(image, panorama_id=f"benchmark-{index}")
        samples.append(time.perf_counter() - started)
        counts.append(len(result))
    if len(set(counts)) != 1:
        raise AssertionError(
            "reviewed profile produced nondeterministic feature counts"
        )

    sorted_samples = sorted(samples)
    p95_index = min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)
    raw = {
        "seconds": samples,
        "feature_counts": counts,
        "shape_hw": list(image.shape[:2]),
        "dtype": str(image.dtype),
        "image_identity": image_identity,
    }
    raw_sha = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": "performance",
        "profile_id": PROFILE_ID,
        "resolution": args.resolution,
        "source_commit": args.source_commit,
        "wheel_sha256": _sha256(args.wheel),
        "hardware_id": args.hardware_id,
        "raw_samples_sha256": raw_sha,
        "metrics": {
            "installed_wheel_isolated": True,
            "warmup_recorded": True,
            "median_recorded": True,
            "p95_recorded": True,
            "peak_rss_recorded": True,
            "raw_samples_recorded": True,
            "median_seconds": statistics.median(samples),
            "p95_seconds": sorted_samples[p95_index],
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }
    payload = {
        "schema": "panorai-spherical-dog-performance/v1",
        "artifact_version": version("panorai"),
        "origin": str(origin),
        "record": record,
        "raw": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output} sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
