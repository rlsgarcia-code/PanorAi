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

    # DOCS_SPHERICAL_PROCESSING_START = None
    from panorai.image_processing import (
        spherical_canny,
        spherical_equalize_histogram,
        spherical_filter2d,
        spherical_gaussian_blur,
        spherical_gaussian_pyramid,
        spherical_gradient,
        spherical_rotate,
    )

    process_height, process_width = 48, 96
    py, px = np.indices((process_height, process_width))
    low_contrast = (88 + 24 * (((px // 8) + (py // 8)) % 2)).astype(np.uint8)
    equalized = spherical_equalize_histogram(low_contrast)
    signal = equalized.astype(np.float32) / 255.0
    sharpen_kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
    sharpened = spherical_filter2d(signal, sharpen_kernel)
    smoothed = spherical_gaussian_blur(signal, ksize=5, sigma=1.2)
    gradient = spherical_gradient(smoothed, operator="scharr")
    edges = spherical_canny(signal, 0.04, 0.10, gaussian_ksize=3)
    yaw = np.deg2rad(20.0)
    rotation = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    rotated = spherical_rotate(signal, rotation)
    pyramid = spherical_gaussian_pyramid(signal, levels=3)
    assert sharpened.shape == smoothed.shape == rotated.shape == signal.shape
    assert gradient.east.shape == gradient.north.shape == signal.shape
    assert edges.dtype == np.uint8 and np.count_nonzero(edges) > 0
    assert [level.shape for level in pyramid] == [(48, 96), (24, 48), (12, 24)]
    # DOCS_SPHERICAL_PROCESSING_END = None

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

    # DOCS_SPHERICAL_PREPROCESSING_START = None
    from panorai.image_processing import spherical_equalize_histogram

    enhanced_texture = spherical_equalize_histogram(texture)
    enhanced_features = feature_pipeline.extract(
        enhanced_texture, panorama_id="equalized-frame"
    )
    assert enhanced_texture.dtype == np.uint8
    assert len(enhanced_features) > 0
    # DOCS_SPHERICAL_PREPROCESSING_END = None

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

    # DOCS_TRIANGULATION_START = None
    def triangulate_bearings(bearing_a, bearing_b, R_b_from_a, t_b_from_a):
        """Educational closest-rays triangulation in camera-A coordinates."""
        bearing_a = np.asarray(bearing_a, dtype=np.float64)
        bearing_b = np.asarray(bearing_b, dtype=np.float64)
        R_b_from_a = np.asarray(R_b_from_a, dtype=np.float64)
        t_b_from_a = np.asarray(t_b_from_a, dtype=np.float64)
        bearing_a /= np.linalg.norm(bearing_a)
        bearing_b /= np.linalg.norm(bearing_b)
        center_b_in_a = -R_b_from_a.T @ t_b_from_a
        ray_b_in_a = R_b_from_a.T @ bearing_b
        system = np.column_stack((bearing_a, -ray_b_in_a))
        depths, *_ = np.linalg.lstsq(system, center_b_in_a, rcond=None)
        point_on_a = depths[0] * bearing_a
        point_on_b = center_b_in_a + depths[1] * ray_b_in_a
        point_a = 0.5 * (point_on_a + point_on_b)
        ray_gap = float(np.linalg.norm(point_on_a - point_on_b))
        return point_a, depths, ray_gap

    known_point_a = np.array([0.5, 0.2, 3.0])
    known_center_b_a = np.array([1.0, 0.0, 0.0])
    synthetic_R_ba = np.eye(3)
    synthetic_t_ba = -known_center_b_a
    synthetic_bearing_a = known_point_a / np.linalg.norm(known_point_a)
    synthetic_bearing_b = known_point_a - known_center_b_a
    synthetic_bearing_b /= np.linalg.norm(synthetic_bearing_b)
    triangulated, depths, ray_gap = triangulate_bearings(
        synthetic_bearing_a,
        synthetic_bearing_b,
        synthetic_R_ba,
        synthetic_t_ba,
    )
    np.testing.assert_allclose(triangulated, known_point_a, atol=1e-12)
    assert np.all(depths > 0.0) and ray_gap < 1e-12
    # DOCS_TRIANGULATION_END = None

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
