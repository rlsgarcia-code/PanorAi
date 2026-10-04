#!/usr/bin/env python3
"""Execute the code shown in the canonical PanorAi documentation."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--source-checkout", action="store_true")
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    if args.source_checkout and args.require_installed:
        parser.error("--source-checkout and --require-installed are mutually exclusive")
    if args.source_checkout and str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    return args


def _assert_origin(args: argparse.Namespace, package) -> Path:
    origin = Path(package.__file__).resolve()
    distribution_version = version("panorai")
    assert package.__version__ == distribution_version
    if args.expected_version is not None:
        assert distribution_version == args.expected_version
    if args.require_installed:
        try:
            origin.relative_to(args.source_root.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError(f"documentation examples imported checkout: {origin}")
    return origin


def main() -> None:
    args = _parse_args()

    # DOCS_FUNCTIONAL_START = None
    import numpy as np
    from panorai.geometry import GnomonicSpec, equirectangular_to_gnomonic

    height, width = 16, 32
    rgb = np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32)
    rgb = rgb.reshape(height, width, 3)  # HWC, float RGB in [0, 1]
    view_spec = GnomonicSpec(
        center_lat_deg=15.0,
        center_lon_deg=30.0,
        hfov_deg=90.0,
        vfov_deg=60.0,
        output_shape_hw=(8, 12),
    )
    view = equirectangular_to_gnomonic(rgb, view_spec, interpolation="bilinear")
    assert view.data.shape == (8, 12, 3)
    assert view.data.dtype == np.float32
    assert view.support_mask.shape == (8, 12)
    # DOCS_FUNCTIONAL_END = None

    # DOCS_PROJECTOR_START = None
    from panorai.geometry import GnomonicProjector

    projector = GnomonicProjector(view_spec, interpolation="bilinear")
    projected = projector.project(rgb)
    restored = projector.back_project(projected, (height, width))
    assert projected.data.shape == (8, 12, 3)
    assert restored.data.shape == rgb.shape
    assert restored.support_mask.any()  # support is geometry, never ``data != 0``
    # DOCS_PROJECTOR_END = None

    # DOCS_CONTAINER_START = None
    from panorai.data import EquirectangularImage

    panorama = EquirectangularImage(rgb)
    face = panorama.to_gnomonic(spec=view_spec, interpolation="bilinear")
    assert face.data.shape == (8, 12, 3)
    assert face.data.dtype == np.float32
    assert face.support_mask.all()
    # DOCS_CONTAINER_END = None

    # DOCS_WORKFLOW_START = None
    import panorai as pa

    workflow_panorama = pa.EquirectangularImage(rgb)
    faces = workflow_panorama.views("cube", size=8)
    processed = faces.map(lambda image: np.clip(image, 0.0, 1.0))
    reconstructed = processed.reconstruct()
    shortcut = workflow_panorama.process_views(
        lambda image: np.clip(image, 0.0, 1.0),
        layout="cube",
        size=8,
    )
    assert len(faces) == 6
    assert reconstructed.image.shape == rgb.shape
    assert np.allclose(reconstructed.image, shortcut.image, equal_nan=True)
    assert faces.describe()["contract"] == "geometry-v1"
    # DOCS_WORKFLOW_END = None

    # DOCS_MODALITIES_START = None
    labels = (np.arange(height * width) % 7).reshape(height, width).astype(np.int16)
    label_view = equirectangular_to_gnomonic(labels, view_spec, interpolation="nearest")
    assert label_view.data.dtype == np.int16

    valid = np.ones((height, width), dtype=bool)
    valid[:, :4] = False
    mask_view = equirectangular_to_gnomonic(valid, view_spec, interpolation="nearest")
    assert mask_view.data.dtype == np.bool_

    radial_range_m = np.full((height, width), 2.5, dtype=np.float32)
    radial_range_m[0, 0] = np.nan  # invalid data stays distinct from support
    depth_is_valid = np.isfinite(radial_range_m)
    depth_view = equirectangular_to_gnomonic(
        radial_range_m,
        view_spec,
        interpolation="bilinear",
        invalid_policy="renormalize",
        validity_mask=depth_is_valid,
        min_valid_weight=0.5,
    )
    assert depth_view.data.dtype == np.float32
    assert depth_view.support_mask.all()
    assert depth_view.validity_mask.shape == view_spec.output_shape_hw
    assert depth_view.valid_weight.shape == view_spec.output_shape_hw
    # DOCS_MODALITIES_END = None

    # DOCS_FEATURES_START = None
    from panorai.features import SphericalFeaturePipeline

    feature_height, feature_width = 128, 256
    fy, fx = np.indices((feature_height, feature_width))
    texture = (((fx // 9) + (fy // 11)) % 2 * 180).astype(np.uint8)
    feature_rgb = np.stack(
        (texture, np.roll(texture, 5, axis=1), np.roll(texture, 7, axis=0)),
        axis=-1,
    )
    feature_pipeline = SphericalFeaturePipeline.from_preset(
        "sift-flann",
        face_sampler="cube",
        face_fov_deg=100.0,
        face_shape_hw=(96, 96),
        edge_margin_px=4,
        max_features=120,
    )
    feature_matches = feature_pipeline.extract_and_match(feature_rgb, feature_rgb)
    assert len(feature_matches) > 0
    assert feature_matches.bearings_a.shape[1] == 3
    assert feature_pipeline.describe()["interface"] == "panorai-spherical-features/v1"
    # DOCS_FEATURES_END = None

    # DOCS_RELATIVE_POSE_START = None
    from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator

    pose_estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            min_num_trials=8,
            max_num_trials=8,
            local_optimization_steps=0,
            minimal_solver_starts=8,
            minimal_solver_max_nfev=10,
            refinement_max_nfev=10,
            stability_trials=0,
            model_competition_trials=8,
            random_seed=7,
        )
    )
    pairwise_pose = pose_estimator.estimate(
        feature_matches.to_bearing_correspondences()
    )
    if pairwise_pose is not None:
        assert pairwise_pose.R.shape == (3, 3)
        assert pairwise_pose.t.shape == (3,)
        assert pairwise_pose.describe()["translation"] == "unit-direction-only"
    # DOCS_RELATIVE_POSE_END = None

    # DOCS_RECONSTRUCTION_START = None
    from panorai.reconstruction import SphericalGlobalMapper

    mapper = SphericalGlobalMapper(relative_pose_estimator=pose_estimator)
    insufficient = mapper.reconstruct(
        matches=(), panorama_ids=("pano-a", "pano-b", "pano-c")
    )
    assert not insufficient.success
    assert insufficient.points_xyz.shape == (0, 3)
    assert insufficient.failure_reasons
    # DOCS_RECONSTRUCTION_END = None

    # DOCS_SLAM_START = None
    from panorai.slam import (
        SphericalIncrementalSLAM,
        SphericalIncrementalSLAMOptions,
    )

    first_features = feature_pipeline.extract(feature_rgb, panorama_id="frame-000")
    slam = SphericalIncrementalSLAM(
        feature_pipeline=feature_pipeline,
        relative_pose_estimator=pose_estimator,
        options=SphericalIncrementalSLAMOptions(
            min_frame_features=1,
            global_refine_on_finish=False,
        ),
    )
    first_tracking = slam.add_features(first_features, timestamp_s=0.0)
    assert first_tracking.state == "initializing"
    assert first_tracking.pose is not None
    partial_trajectory = slam.finish()
    assert partial_trajectory.scale == "arbitrary"
    assert not partial_trajectory.success  # one frame cannot define a trajectory
    # DOCS_SLAM_END = None

    # DOCS_BLENDER_START = None
    from panorai.blenders import AverageBlender

    black = np.zeros((4, 6, 3), dtype=np.float32)
    white = np.ones((4, 6, 3), dtype=np.float32)
    black_is_valid = np.ones((4, 6), dtype=bool)
    white_is_invalid = np.zeros((4, 6), dtype=bool)
    blended, blend_support = AverageBlender().blend(
        [black, white],
        [black_is_valid, white_is_invalid],
        return_mask=True,
    )
    assert np.array_equal(blended, black)  # black is data, not missingness
    assert blend_support.all()
    # DOCS_BLENDER_END = None

    # DOCS_CUBEMAP_START = None
    from panorai.geometry import (
        CUBE_FACE_ORDER,
        cubemap_to_equirectangular,
        equirectangular_to_cubemap,
    )

    cubemap = equirectangular_to_cubemap(
        rgb, face_shape_hw=(8, 8), interpolation="bilinear"
    )
    assert tuple(cubemap) == CUBE_FACE_ORDER
    rebuilt = cubemap_to_equirectangular(
        {name: result.data for name, result in cubemap.items()},
        output_shape_hw=(height, width),
        interpolation="bilinear",
    )
    assert rebuilt.data.shape == rgb.shape
    assert rebuilt.support_mask.all()
    # DOCS_CUBEMAP_END = None

    if args.torch:
        # DOCS_TORCH_START = None
        import torch

        batch = torch.linspace(
            0.0, 1.0, 2 * 3 * height * width, dtype=torch.float32
        ).reshape(2, 3, height, width)
        batch.requires_grad_(True)  # NCHW batch on the current device
        torch_view = equirectangular_to_gnomonic(
            batch, view_spec, interpolation="bilinear"
        )
        assert torch_view.data.shape == (2, 3, 8, 12)
        torch_view.data.sum().backward()
        assert batch.grad is not None and torch.isfinite(batch.grad).all()
        # DOCS_TORCH_END = None

    import panorai

    origin = _assert_origin(args, panorai)
    print(f"documentation examples: OK version={panorai.__version__} origin={origin}")


if __name__ == "__main__":
    main()
