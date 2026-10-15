#!/usr/bin/env python3
"""Installed-wheel consumer for direct spherical stereo depth."""

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
    y, x = np.indices((24, 48), dtype=np.float32)
    return np.stack(
        (
            (np.sin(x / 2.0) + 1.0) / 2.0,
            (np.cos(y / 2.0) + 1.0) / 2.0,
            ((x // 4 + y // 4) % 2).astype(np.float32),
        ),
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.applications import SphericalStereoDepth, SphericalStereoInput
    from panorai.stereo import SphericalStereoOptions, SphericalStereoResult

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    reference = _texture()
    source = np.roll(reference, 1, axis=1)
    rotation = np.eye(3)
    translation = np.asarray((0.05, 0.0, 0.0))
    application = SphericalStereoDepth(
        SphericalStereoOptions(
            min_range=0.5,
            max_range=2.0,
            num_hypotheses=3,
            window_size=3,
            bidirectional_consistency=False,
            filter_backend="numpy",
        )
    )
    direct = application(reference, source, rotation, translation)
    composed = application.as_pipeline()(
        SphericalStereoInput(reference, source, rotation, translation)
    )

    assert isinstance(direct, SphericalStereoResult)
    assert direct.quantity == "radial_range"
    assert direct.range.shape == reference.shape[:2]
    assert direct.validity_mask.dtype == np.bool_
    assert np.all(np.isnan(direct.range[~direct.validity_mask]))
    np.testing.assert_array_equal(direct.range, composed.range)
    np.testing.assert_array_equal(direct.validity_mask, composed.validity_mask)

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-stereo",
                "version": version("panorai"),
                "origin": str(origin),
                "shape_hw": list(direct.range.shape),
                "valid_fraction": float(direct.validity_mask.mean()),
                "quantity": direct.quantity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
