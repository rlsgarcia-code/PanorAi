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

    # DOCS_STEREO_START = None
    from panorai.stereo import SphericalStereoOptions, estimate_spherical_range

    # Dense stereo starts after R and metric t have already been obtained.
    stereo_options = SphericalStereoOptions(
        min_range=0.5,
        max_range=4.0,
        num_hypotheses=8,
        window_size=3,
        pole_margin_fraction=0.0,
        min_texture_std=0.0,
        min_confidence=0.0,
        max_matching_cost=1.0,
        bidirectional_consistency=False,
    )
    stereo_result = estimate_spherical_range(
        rgb,
        np.roll(rgb, 1, axis=1),
        np.eye(3),
        np.array([-0.25, 0.0, 0.0]),  # X_b = R_ba @ X_a + t_ba; metres here
        options=stereo_options,
    )
    assert stereo_result.range.shape == rgb.shape[:2]
    assert stereo_result.validity_mask.dtype == np.bool_
    assert stereo_result.quantity == "radial_range"
    # DOCS_STEREO_END = None

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

    # DOCS_METRIC_FLOOR_START = None
    def intersect_floor(bearing, up_camera, camera_height_m, horizon_margin=1e-3):
        """Intersect one camera-frame unit bearing with its metric floor plane."""
        bearing = np.asarray(bearing, dtype=np.float64)
        up_camera = np.asarray(up_camera, dtype=np.float64)
        bearing /= np.linalg.norm(bearing)
        up_camera /= np.linalg.norm(up_camera)
        downward_component = float(up_camera @ bearing)
        if downward_component >= -horizon_margin:
            raise ValueError("bearing is above or too close to the floor horizon")
        radial_range_m = -float(camera_height_m) / downward_component
        return radial_range_m * bearing

    def metric_scale_from_floor_match(
        floor_point_a, bearing_b, R_b_from_a, unit_t_b_from_a
    ):
        """Recover baseline scale from one verified floor correspondence."""
        bearing_b = np.asarray(bearing_b, dtype=np.float64)
        bearing_b /= np.linalg.norm(bearing_b)
        q_b = np.asarray(R_b_from_a, dtype=np.float64) @ floor_point_a
        unit_t = np.asarray(unit_t_b_from_a, dtype=np.float64)
        tangent_projector = np.eye(3) - np.outer(bearing_b, bearing_b)
        projected_t = tangent_projector @ unit_t
        denominator = float(projected_t @ projected_t)
        if denominator <= 1e-12:
            raise ValueError("floor match does not constrain translation scale")
        return -float(projected_t @ (tangent_projector @ q_b)) / denominator

    up_a = np.array([0.0, 1.0, 0.0])
    height_a_m = height_b_m = 1.60
    center_b_a_m = np.array([1.20, 0.0, 0.35])
    floor_point_a_m = np.array([2.0, -height_a_m, 4.0])
    floor_bearing_a = floor_point_a_m / np.linalg.norm(floor_point_a_m)
    floor_bearing_b = floor_point_a_m - center_b_a_m
    floor_bearing_b /= np.linalg.norm(floor_bearing_b)
    unit_t_ba = -center_b_a_m / np.linalg.norm(center_b_a_m)

    metric_floor_point = intersect_floor(floor_bearing_a, up_a, height_a_m)
    baseline_m = metric_scale_from_floor_match(
        metric_floor_point, floor_bearing_b, np.eye(3), unit_t_ba
    )
    metric_t_ba = baseline_m * unit_t_ba
    recovered_center_b_a = -metric_t_ba

    np.testing.assert_allclose(metric_floor_point, floor_point_a_m, atol=1e-12)
    np.testing.assert_allclose(recovered_center_b_a, center_b_a_m, atol=1e-12)
    assert np.isclose(up_a @ recovered_center_b_a, height_b_m - height_a_m)
    # Equal known heights validate a horizontal baseline; without an observed
    # floor point they do not determine the baseline length.
    # DOCS_METRIC_FLOOR_END = None

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
