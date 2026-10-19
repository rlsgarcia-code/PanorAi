#!/usr/bin/env python3
"""Installed-wheel consumer for Experimental central panorama stitching."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
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


def _yaw(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )


def _texture(world_rays: np.ndarray) -> np.ndarray:
    flat = np.asarray(world_rays, dtype=np.float64).reshape(-1, 3)
    random = np.random.default_rng(4207)
    centres = random.normal(size=(150, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    colours = random.uniform(0.15, 1.0, size=(150, 3))
    dots = flat @ centres.T
    weights = np.exp(np.clip((dots - 1.0) / 0.0012, -80.0, 0.0))
    blobs = weights @ colours / np.maximum(weights.sum(axis=1, keepdims=True), 0.35)
    longitude = np.arctan2(flat[:, 0], flat[:, 2])
    latitude = np.arcsin(np.clip(flat[:, 1], -1.0, 1.0))
    waves = np.stack(
        (
            np.sin(17.0 * longitude + 7.0 * latitude),
            np.sin(23.0 * longitude - 11.0 * latitude + 0.7),
            np.sin(13.0 * longitude + 19.0 * latitude + 1.4),
        ),
        axis=1,
    )
    values = 42.0 + 185.0 * blobs + 28.0 * waves
    return (
        np.rint(np.clip(values, 0.0, 255.0))
        .astype(np.uint8)
        .reshape(world_rays.shape[:-1] + (3,))
    )


def _render_view(
    camera_to_world: np.ndarray,
    *,
    shape_hw: tuple[int, int] = (180, 240),
    horizontal_fov_deg: float = 92.0,
) -> np.ndarray:
    from panorai.calibration import PinholeIntrinsics, pinhole_pixels_to_rays

    intrinsics = PinholeIntrinsics.from_horizontal_fov(shape_hw, horizontal_fov_deg)
    rows, columns = np.indices(shape_hw, dtype=np.float64)
    pixels = np.stack((columns, rows), axis=-1)
    camera_rays = pinhole_pixels_to_rays(pixels, intrinsics)
    return _texture(camera_rays @ camera_to_world.T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.applications import Application, PanoramaStitcher
    from panorai.pipeline import Pipeline
    from panorai.stitching import PanoramaStitchingResult, StitchingOptions

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    rotations = (_yaw(18.0), _yaw(-18.0))
    # Deliberately reverse the natural left/right order: input order is not an
    # overlap prior, although the first image still fixes the world gauge.
    images = tuple(_render_view(rotation) for rotation in rotations)
    application = PanoramaStitcher(
        StitchingOptions(
            max_features=1800,
            ratio_test=0.8,
            min_pair_matches=10,
            min_pair_inliers=7,
            max_transfer_error_px=4.0,
            max_angular_error_deg=1.5,
            output_shape_hw=(96, 192),
            random_seed=17,
        )
    )
    result = application(
        images,
        image_ids=("right", "left"),
        horizontal_fov_deg=92.0,
    )
    assert isinstance(result, PanoramaStitchingResult)
    assert result.panorama.shape[:2] == (96, 192)
    assert result.diagnostics.accepted_pair_count == 1
    assert result.diagnostics.coverage > 0.1
    assert result.diagnostics.estimated_fov_count == 0
    assert np.any(result.panorama.support_mask)
    assert np.all(result.panorama.validity("image") <= result.panorama.support_mask)

    pipeline = application.as_pipeline()
    restored_pipeline = Pipeline.from_config(pipeline.to_config())
    assert restored_pipeline.to_config() == pipeline.to_config()
    with tempfile.TemporaryDirectory(prefix="panorai-stitching-") as directory:
        restored_application = Application.load(
            application.save(Path(directory) / "application.json")
        )
        assert restored_application.to_config() == application.to_config()

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "panorama-stitching",
                "version": version("panorai"),
                "origin": str(origin),
                "input_count": result.diagnostics.input_image_count,
                "accepted_pair_count": result.diagnostics.accepted_pair_count,
                "coverage": result.diagnostics.coverage,
                "configuration_round_trip": True,
                "pipeline_round_trip": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
