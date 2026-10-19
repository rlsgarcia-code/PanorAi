#!/usr/bin/env python3
"""Installed-wheel consumer for Stable spherical region analysis."""

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
    from panorai.data import RasterFieldSpec, RegionAnalysisResult
    from panorai.ops import spherical as S
    from panorai.pipeline import AnalyzeRegions, OnSphere, Pipeline

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    height, width = 24, 48
    mask = np.zeros((height, width), dtype=bool)
    mask[7:18, :6] = True
    mask[7:18, -5:] = True
    panorama = panorai.EquirectangularImage.constant(
        0,
        height=height,
        width=width,
    ).with_field(
        "mask",
        mask,
        spec=RasterFieldSpec.categorical(),
    )

    labels = S.connected_components(mask)
    contour_set = S.trace_contours(labels)
    skeleton = S.skeletonize(mask)
    assert labels.max() == 1
    assert len(contour_set.contours) == 1
    assert contour_set.contours[0].closed
    assert np.all(~skeleton | mask)
    assert np.any(skeleton[:, 0]) and np.any(skeleton[:, -1])

    skeleton_panorama = panorama.apply_spherical(
        S.Skeletonize(),
        input="mask",
        output="mask",
        spec=RasterFieldSpec.categorical(),
    )
    np.testing.assert_array_equal(skeleton_panorama.field("mask"), skeleton)

    label_stage = OnSphere(S.ConnectedComponents(8), input="mask", output="labels")
    analysis_stage = AnalyzeRegions(include_contours=True)
    pipeline = Pipeline(label_stage, analysis_stage)
    analysis = pipeline(panorama)
    assert isinstance(analysis, RegionAnalysisResult)
    assert analysis.interface == "panorai-spherical-region-analysis/v1"
    assert analysis.contours is not None
    assert len(analysis.measurements.regions) == 1

    with tempfile.TemporaryDirectory(prefix="panorai-regions-") as directory:
        root = Path(directory)
        restored = Pipeline.load(pipeline.save(root / "pipeline"))
        restored_result = restored(panorama)
        assert restored_result.contours is not None
        np.testing.assert_array_equal(
            restored_result.contours.contours[0].bearings_xyz,
            analysis.contours.contours[0].bearings_xyz,
        )

        cached = Pipeline(label_stage, analysis_stage, memory=root / "cache")
        cold = cached.run(panorama)
        warm = cached.run(panorama)
        assert cold.manifest()["cache"]["hit"] is False
        assert warm.manifest()["cache"]["hit"] is True
        assert isinstance(warm.output, RegionAnalysisResult)

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-region-analysis",
                "version": version("panorai"),
                "origin": str(origin),
                "region_count": len(analysis.measurements.regions),
                "contour_count": len(contour_set.contours),
                "skeleton_pixels": int(skeleton.sum()),
                "pipeline_round_trip": True,
                "typed_cache_hit": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
