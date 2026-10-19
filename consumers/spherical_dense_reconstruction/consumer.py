#!/usr/bin/env python3
"""Installed-wheel consumer for registered ranges, TSDF, and surface output."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.applications import SphericalDenseReconstruction
    from panorai.geometry import erp_pixels_to_rays
    from panorai.pipeline import Pipeline
    from panorai.reconstruction import (
        ColorizeSurface,
        DenseReconstructionWorkspace,
        ExtractSurface,
        FuseTSDF,
        MultiViewRangeMap,
        MultiViewRangeSet,
        RegisteredSphericalScene,
        RegisteredSphericalView,
        SphericalCameraPose,
        TSDFSpec,
        ValidateDenseReconstruction,
    )

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    height, width = 24, 48
    yy, xx = np.indices((height, width), dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((xx, yy), axis=-1), (height, width))
    valid = rays[..., 2] > 0.2
    radial_range = np.full((height, width), np.nan, dtype=np.float32)
    radial_range[valid] = (1.0 / rays[..., 2][valid]).astype(np.float32)

    views = []
    maps = []
    for index, center_x in enumerate((-0.15, 0.0, 0.15)):
        panorama_id = f"camera-{index}"
        pose = SphericalCameraPose(
            panorama_id=panorama_id,
            rotation_world_to_panorama=np.eye(3),
            center_world=np.asarray((center_x, 0.0, 0.0)),
        )
        image = panorai.EquirectangularImage.constant(
            (0.0, 0.0, 0.0), height=height, width=width
        )
        views.append(RegisteredSphericalView(image=image, pose=pose))
        count = valid.astype(np.uint16) * 2
        maps.append(
            MultiViewRangeMap(
                reference_id=panorama_id,
                pose=pose,
                radial_range=radial_range,
                validity=valid,
                sampling_support=valid,
                observable=valid,
                confidence=valid.astype(np.float32),
                observable_source_count=count,
                source_count=count,
                source_consensus_count=count,
                contribution_count=count,
                source_ids=tuple(
                    f"camera-{other}" for other in range(3) if other != index
                ),
            )
        )

    scene = RegisteredSphericalScene(views=tuple(views), units="m")
    ranges = MultiViewRangeSet(scene=scene, maps=tuple(maps))
    spec = TSDFSpec(
        voxel_size=0.1,
        bounds=((-0.5, -0.5, 0.5), (0.5, 0.5, 1.5)),
        truncation_distance=0.4,
        min_surface_views=2,
    )
    route = Pipeline(
        FuseTSDF(spec),
        ExtractSurface(),
        ColorizeSurface(),
        ValidateDenseReconstruction("all"),
    )
    result = route(DenseReconstructionWorkspace(scene=scene, ranges=ranges))
    surface = result.require_success().surface
    assert surface is not None
    assert len(surface.faces) > 0
    assert surface.color_validity is not None
    assert np.all(surface.color_validity)
    assert np.all(surface.colors_rgb == 0.0)
    assert np.all(surface.vertex_view_count >= 2)

    application = SphericalDenseReconstruction.reviewed_profile(
        voxel_size=0.1,
        bounds=((-0.5, -0.5, 0.5), (0.5, 0.5, 1.5)),
        filter_backend="numpy",
    )
    restored = SphericalDenseReconstruction.from_config(application.to_config())
    assert restored == application
    assert len(application.as_pipeline().named_steps) == 6
    assert application.describe()["stability"] == "experimental"

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-dense-reconstruction",
                "version": version("panorai"),
                "origin": str(origin),
                "surface_vertices": len(surface.vertices_xyz),
                "surface_faces": len(surface.faces),
                "valid_black_color": True,
                "minimum_view_evidence": int(surface.vertex_view_count.min()),
                "application_profile": application.profile_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
