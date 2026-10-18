#!/usr/bin/env python3
"""Installed-wheel consumer for Stable intrinsic spherical morphology."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.data import RasterFieldSpec
    from panorai.ops import spherical as S
    from panorai.pipeline import OnSphere, Pipeline, RasterField

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    mask = np.zeros((24, 48), dtype=bool)
    mask[10:15, -2:] = True
    mask[10:15, :2] = True
    mask[12, 0] = False
    panorama = panorai.EquirectangularImage.from_field(
        "mask",
        mask,
        spec=RasterFieldSpec.categorical(),
    )
    operation = S.Closing(S.GeodesicDisk(8.0))

    filtered = panorama.filter(operation, field="mask")
    pipeline = Pipeline(OnSphere(operation, input="mask", output="mask"))
    result = pipeline(panorama)
    assert isinstance(result, RasterField)
    np.testing.assert_array_equal(filtered.field("mask"), result.values)
    assert result.values[12, 0]
    assert result.values[12, -1]
    assert result.grid.is_canonical_erp
    assert result.grid.transforms[-1].operation == "closing"

    with tempfile.TemporaryDirectory(prefix="panorai-morphology-") as directory:
        restored = Pipeline.load(pipeline.save(Path(directory) / "pipeline"))
        actual = restored(panorama)
        np.testing.assert_array_equal(actual.values, result.values)

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-morphology",
                "version": version("panorai"),
                "origin": str(origin),
                "shape_hw": list(result.values.shape),
                "true_pixels": int(result.values.sum()),
                "seam_closed": True,
                "pipeline_round_trip": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
