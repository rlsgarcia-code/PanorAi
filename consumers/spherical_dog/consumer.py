#!/usr/bin/env python3
"""Installed-wheel consumer for spherical DoG features."""

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
    from panorai.applications import SphericalDoGFeatures
    from panorai.features import SphericalDoGSIFTConfig, SphericalFeatureSet

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    application = SphericalDoGFeatures(
        SphericalDoGSIFTConfig(
            octaves=1,
            levels_per_octave=2,
            max_features=20,
            patch_size=24,
            convolution_backend="numpy",
        )
    )
    result = application(_texture(), panorama_id="installed-dog")

    assert isinstance(result, SphericalFeatureSet)
    assert result.panorama_id == "installed-dog"
    assert result.descriptors.ndim == 2
    assert result.descriptors.shape[0] == len(result)
    assert result.bearings.shape == (len(result), 3)
    if len(result):
        np.testing.assert_allclose(
            np.linalg.norm(result.bearings, axis=1), 1.0, atol=1e-6
        )
    assert application.as_pipeline().stability == "experimental"

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-dog",
                "version": version("panorai"),
                "origin": str(origin),
                "feature_count": len(result),
                "descriptor_shape": list(result.descriptors.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
