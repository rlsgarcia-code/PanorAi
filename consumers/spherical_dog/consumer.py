#!/usr/bin/env python3
"""Installed-wheel consumer for descriptor-free spherical DoG detection."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


def _texture() -> np.ndarray:
    y, x = np.indices((64, 128))
    checker = ((x // 8 + y // 8) % 2) * 150
    waves = (np.sin(x / 3.0) + np.cos(y / 2.5) + 2.0) * 25
    base = np.clip(checker + waves, 0, 255).astype(np.uint8)
    return np.stack((base, np.roll(base, 5, axis=1), np.roll(base, 7, axis=0)), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.features import (
        SphericalDoGDetector,
        SphericalDoGDetectorConfig,
        SphericalKeypointSet,
    )

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    detector = SphericalDoGDetector(
        SphericalDoGDetectorConfig(
            octaves=2,
            levels_per_octave=3,
            max_keypoints=40,
            contrast_threshold=0.003,
            convolution_backend="numpy",
        )
    )
    result = detector.detect(_texture(), panorama_id="installed-dog")

    assert isinstance(result, SphericalKeypointSet)
    assert len(result) > 0
    assert result.panorama_id == "installed-dog"
    assert result.bearings.shape == (len(result), 3)
    if len(result):
        np.testing.assert_allclose(
            np.linalg.norm(result.bearings, axis=1), 1.0, atol=1e-6
        )
    assert result.interface == "panorai-spherical-dog-detector/v2"
    assert result.stability == "experimental"
    assert result.config.to_dict()["descriptor"] is None

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-dog-detector",
                "version": version("panorai"),
                "origin": str(origin),
                "keypoint_count": len(result),
                "interface": result.interface,
                "stability": result.stability,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
