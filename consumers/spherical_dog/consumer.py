#!/usr/bin/env python3
"""Installed-wheel consumer for the exact reviewed DoG candidate profile."""

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
    from panorai.features import SphericalFeatureSet

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    result = method(_texture(), panorama_id="installed-dog")

    assert isinstance(result, SphericalFeatureSet)
    assert len(result) > 0
    assert result.panorama_id == "installed-dog"
    assert result.bearings.shape == (len(result), 3)
    assert result.descriptors.shape == (len(result), 128)
    assert result.descriptor_type == "root-sift-float32"
    if len(result):
        np.testing.assert_allclose(
            np.linalg.norm(result.bearings, axis=1), 1.0, atol=1e-6
        )
    assert method.interface == "panorai-spherical-dog-features/v1"
    assert method.profile_id == "panorai-spherical-dog-features/reviewed-v1"
    assert method.stability == "experimental"
    restored = SphericalDoGFeatures.from_config(method.to_config())
    assert restored == method

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-dog-reviewed-profile",
                "version": version("panorai"),
                "origin": str(origin),
                "keypoint_count": len(result),
                "interface": method.interface,
                "profile_id": method.profile_id,
                "stability": method.stability,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
