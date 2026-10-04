from __future__ import annotations

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
)
from panorai.features import (
    FaceSetSpec,
    FeatureProvenance,
    SphericalFeature,
    SphericalFeaturePipeline,
    SphericalFeatureSet,
)
from panorai.geometry import GnomonicSpec, rays_to_erp_pixels
from panorai.reconstruction import SphericalGlobalMapperOptions
from panorai.reconstruction._math import rotation_exp
from panorai.slam import (
    SphericalIncrementalSLAM,
    SphericalIncrementalSLAMOptions,
)
from panorai.slam._incremental import _KeyframeRecord, _PairEvaluation
from panorai.slam._local_map import (
    LocalBundleAdjuster,
    LocalSphericalMap,
    ObservationState,
    PoseState,
    triangulate_bearings,
)


def _oracle_estimator() -> SphericalRelativePoseEstimator:
    return SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.15,
            min_num_trials=8,
            max_num_trials=40,
            min_inliers=8,
            local_optimization_steps=1,
            stability_trials=0,
            model_competition_trials=16,
            random_seed=91,
        ),
        quality_policy=RelativePoseAcceptancePolicy(
            min_inliers=8,
            min_inlier_ratio=0.2,
            min_occupied_cells=1,
            min_coverage_entropy=0.0,
            max_median_residual_deg=1.0,
            min_median_parallax_deg=0.0,
            min_cheirality_ratio=0.0,
            min_translation_orientation_margin=0.0,
            require_stability=False,
            require_essential_preferred=False,
        ),
    )


class _AcceptedQuality:
    accepted = True


class _OraclePose:
    def __init__(self, rotation, translation, count):
        self.R = np.asarray(rotation)
        self.t = np.asarray(translation)
        self.inlier_mask = np.ones(count, dtype=bool)
        self.num_inliers = count
        self.quality_report = _AcceptedQuality()


class _KnownPoseEstimator:
    """Independent scene oracle used only to exercise SLAM state transitions."""

    def __init__(self, feature_sets, rotations, centers):
        self._row_owner = {}
        for frame_id, features in feature_sets.items():
            for bearing in features.bearings:
                self._row_owner[np.round(bearing, 12).tobytes()] = frame_id
        self._rotations = rotations
        self._centers = centers

    def estimate(self, correspondences):
        if len(correspondences.bearings_a) == 0:
            return None
        frame_a = self._row_owner[np.round(correspondences.bearings_a[0], 12).tobytes()]
        frame_b = self._row_owner[np.round(correspondences.bearings_b[0], 12).tobytes()]
        rotation_a = self._rotations[frame_a]
        rotation_b = self._rotations[frame_b]
        center_a = self._centers[frame_a]
        center_b = self._centers[frame_b]
        rotation = rotation_b @ rotation_a.T
        translation = rotation_b @ (center_a - center_b)
        translation /= np.linalg.norm(translation)
        return _OraclePose(rotation, translation, len(correspondences.bearings_a))


def _scene(group_size: int = 24):
    rng = np.random.default_rng(20261004)
    groups = {}
    for name, offset in zip("ABCD", (-2.4, -0.8, 0.8, 2.4)):
        groups[name] = rng.uniform(
            (offset - 0.7, -1.8, 4.0),
            (offset + 0.7, 1.8, 8.0),
            size=(group_size, 3),
        )
    points = np.concatenate(tuple(groups.values()), axis=0)
    descriptors = rng.normal(size=(len(points), 32)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    rotations = [
        rotation_exp(np.asarray((0.005 * index, -0.025 * index, 0.008 * index)))
        for index in range(5)
    ]
    centers = [np.asarray((0.35 * index, 0.03 * index, 0.0)) for index in range(5)]
    slices = {
        name: np.arange(index * group_size, (index + 1) * group_size)
        for index, name in enumerate("ABCD")
    }
    return points, descriptors, rotations, centers, slices


def _feature_set(
    frame_id: str,
    point_indices: np.ndarray,
    points: np.ndarray,
    descriptors: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
) -> SphericalFeatureSet:
    vectors = (points[point_indices] - center) @ rotation.T
    bearings = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pixels = rays_to_erp_pixels(bearings, (128, 256)).pixels_xy
    spec = GnomonicSpec(output_shape_hw=(64, 64))
    features = []
    for row, (bearing, pixel) in enumerate(zip(bearings, pixels)):
        features.append(
            SphericalFeature(
                feature_id=f"{frame_id}:feature-{row:04d}",
                panorama_id=frame_id,
                face_id="synthetic-sphere",
                pixel_xy=np.asarray((32.0, 32.0)),
                source_erp_xy=np.asarray(pixel),
                bearing_xyz=np.asarray(bearing),
                response=float(1.0 - row * 1e-4),
                scale=1.0,
                angle_deg=0.0,
                octave=0,
                descriptor_index=row,
                valid=True,
                projection_spec=spec,
                provenance=FeatureProvenance(
                    interface="test-independent-spherical-scene/v1",
                    source_panorama_checksum=frame_id,
                    projection_backend="analytic-bearing-oracle",
                    projection_backend_version="1",
                    face_id="synthetic-sphere",
                ),
            )
        )
    return SphericalFeatureSet(
        panorama_id=frame_id,
        features=features,
        descriptors=descriptors[point_indices].copy(),
        descriptor_type="sift-float32",
        descriptor_metric="l2",
        extractor_name="analytic-oracle",
        extractor_config={},
        backend_name="analytic-oracle",
        backend_version="1",
        face_set_spec=FaceSetSpec(
            sampler="custom", shape_hw=(64, 64), fov_deg=(90.0, 90.0)
        ),
        projection_backend="analytic-bearing-oracle",
        projection_backend_version="1",
        panorama_checksum=frame_id,
    )


def test_triangulation_matches_independent_line_intersection_oracle():
    point = np.asarray((0.4, -0.3, 4.5))
    poses = (
        PoseState("a", 0.0, np.eye(3), np.asarray((0.0, 0.0, 0.0))),
        PoseState("b", 1.0, np.eye(3), np.asarray((1.0, 0.0, 0.0))),
        PoseState("c", 2.0, np.eye(3), np.asarray((0.1, 0.8, 0.0))),
    )
    observations = []
    for pose in poses:
        bearing = point - pose.center
        bearing /= np.linalg.norm(bearing)
        observations.append((pose, bearing))
    recovered = triangulate_bearings(
        observations,
        min_angle_deg=0.1,
        max_error_deg=1e-6,
    )
    assert recovered is not None
    np.testing.assert_allclose(recovered, point, atol=1e-11, rtol=1e-11)
    assert (
        triangulate_bearings(observations[:1], min_angle_deg=0.1, max_error_deg=1.0)
        is None
    )


def test_local_spherical_ba_reduces_direct_angular_cost():
    rng = np.random.default_rng(77)
    true_poses = {
        f"k{index}": PoseState(
            f"k{index}",
            float(index),
            rotation_exp(np.asarray((0.01 * index, -0.02 * index, 0.005 * index))),
            np.asarray((0.45 * index, 0.03 * index, 0.0)),
        )
        for index in range(4)
    }
    working_poses = {
        key: PoseState(
            pose.frame_id,
            pose.timestamp_s,
            pose.rotation.copy(),
            pose.center.copy(),
        )
        for key, pose in true_poses.items()
    }
    working_poses["k2"].rotation = (
        rotation_exp(np.asarray((0.02, -0.01, 0.015))) @ working_poses["k2"].rotation
    )
    working_poses["k2"].center += np.asarray((0.08, -0.05, 0.03))
    working_poses["k3"].rotation = (
        rotation_exp(np.asarray((-0.015, 0.018, -0.01))) @ working_poses["k3"].rotation
    )
    working_poses["k3"].center += np.asarray((-0.07, 0.04, -0.02))
    points = rng.uniform((-1.5, -1.2, 3.5), (2.5, 1.2, 7.0), size=(28, 3))
    local_map = LocalSphericalMap()
    for index, point in enumerate(points):
        observations = []
        for frame_id, pose in true_poses.items():
            bearing = pose.rotation @ (point - pose.center)
            bearing /= np.linalg.norm(bearing)
            observations.append(ObservationState(frame_id, index, bearing))
        local_map.add_point(point + rng.normal(scale=0.04, size=3), observations)
    options = SphericalIncrementalSLAMOptions(
        local_window_size=4,
        local_ba_max_points=64,
        local_ba_max_nfev=100,
        max_reprojection_error_deg=5.0,
    )
    report = LocalBundleAdjuster(options).optimize(
        working_poses, local_map, tuple(true_poses)
    )
    assert report is not None
    assert report.variable_pose_count == 2
    assert report.observation_count == 4 * len(points)
    assert report.final_cost < report.initial_cost * 0.1
    assert report.removed_observation_count == 0


def test_incremental_slam_tracks_maps_relocalizes_and_closes_loop():
    pytest.importorskip("cv2")
    points, descriptors, rotations, centers, slices = _scene()
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=(64, 64),
        max_features=128,
        ratio_test=0.8,
    )
    options = SphericalIncrementalSLAMOptions(
        local_window_size=2,
        min_frame_features=16,
        min_pair_matches=12,
        min_map_correspondences=8,
        keyframe_min_interval_s=0.0,
        keyframe_max_interval_s=0.0,
        keyframe_min_parallax_deg=0.0,
        min_triangulation_angle_deg=0.1,
        max_reprojection_error_deg=1.0,
        local_ba_max_nfev=35,
        local_ba_max_points=96,
        loop_min_keyframe_separation=3,
        loop_min_matches=12,
        max_loop_candidates=2,
        max_tracking_pose_candidates=1,
        optimize_global_on_loop=False,
        global_refine_on_finish=False,
        mapper=SphericalGlobalMapperOptions(
            min_active_tracks_per_panorama=2,
            bundle_max_nfev=30,
            max_refinement_rounds=1,
        ),
    )
    visible = (
        np.concatenate((slices["A"], slices["B"])),
        np.concatenate((slices["A"], slices["B"])),
        np.concatenate((slices["B"], slices["C"])),
        np.concatenate((slices["C"], slices["D"])),
        slices["A"],
    )
    feature_sets = {}
    for index, selected in enumerate(visible):
        feature_sets[f"frame-{index}"] = _feature_set(
            f"frame-{index}",
            selected,
            points,
            descriptors,
            rotations[index],
            centers[index],
        )
    rotations_by_id = {
        f"frame-{index}": rotation for index, rotation in enumerate(rotations)
    }
    centers_by_id = {f"frame-{index}": center for index, center in enumerate(centers)}
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_KnownPoseEstimator(
            feature_sets, rotations_by_id, centers_by_id
        ),
        options=options,
    )
    states = []
    outputs = []
    for index, feature_set in enumerate(feature_sets.values()):
        output = session.add_features(feature_set, timestamp_s=0.2 * index)
        outputs.append(output)
        states.append(output.state)
    assert states[0] == "initializing"
    assert states[1:4] == ["keyframe", "keyframe", "keyframe"]
    assert states[4] == "relocalized", outputs[4]
    assert len(session.local_map) >= 3 * len(slices["A"])
    result = session.finish()
    assert result.success
    assert len(result.trajectory) == 5
    assert result.diagnostics.relocalization_count == 1
    assert result.diagnostics.loop_closure_count >= 1
    assert result.diagnostics.active_map_point_count > 0
    assert result.diagnostics.local_ba_reports
    assert result.frames[0].local_map_point_count == 0
    assert result.frames[-1].local_map_point_count == outputs[-1].local_map_point_count
    assert result.frames[-1].local_map_point_count > 0
    assert all(
        report.final_cost <= report.initial_cost
        for report in result.diagnostics.local_ba_reports
    )
    assert result.describe()["interface"] == "panorai-spherical-incremental-slam/v1"


def test_incremental_pair_initializes_with_real_five_point_estimator():
    pytest.importorskip("cv2")
    points, descriptors, rotations, centers, slices = _scene(group_size=16)
    selected = np.concatenate((slices["A"], slices["B"]))
    feature_sets = [
        _feature_set(
            f"real-{index}",
            selected,
            points,
            descriptors,
            rotations[index],
            centers[index],
        )
        for index in range(2)
    ]
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=(64, 64),
        max_features=64,
        ratio_test=0.8,
    )
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_oracle_estimator(),
        options=SphericalIncrementalSLAMOptions(
            local_window_size=2,
            min_frame_features=16,
            min_pair_matches=12,
            min_map_correspondences=8,
            keyframe_min_interval_s=0.0,
            keyframe_max_interval_s=0.0,
            min_triangulation_angle_deg=0.1,
            max_reprojection_error_deg=1.0,
            max_tracking_pose_candidates=1,
            global_refine_on_finish=False,
            loop_min_keyframe_separation=10,
            local_ba_max_points=64,
            local_ba_max_nfev=20,
        ),
    )
    first = session.add_features(feature_sets[0], timestamp_s=0.0)
    second = session.add_features(feature_sets[1], timestamp_s=0.2)
    assert first.state == "initializing"
    assert second.state == "keyframe"
    assert second.inlier_count >= 12
    assert len(session.local_map) >= 12
    assert session.finish().success


def test_tracking_recovers_scale_change_from_existing_map():
    pytest.importorskip("cv2")
    points, descriptors, rotations, _, slices = _scene(group_size=16)
    centers = [
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.35, 0.0, 0.0)),
        np.asarray((0.4375, 0.0, 0.0)),
    ]
    selected = np.concatenate((slices["A"], slices["B"]))
    feature_sets = {
        f"scale-{index}": _feature_set(
            f"scale-{index}",
            selected,
            points,
            descriptors,
            rotations[index],
            centers[index],
        )
        for index in range(3)
    }
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf",
        face_sampler="cube",
        face_shape_hw=(64, 64),
        max_features=64,
        ratio_test=0.8,
    )
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_KnownPoseEstimator(
            feature_sets,
            {f"scale-{index}": rotations[index] for index in range(3)},
            {f"scale-{index}": centers[index] for index in range(3)},
        ),
        options=SphericalIncrementalSLAMOptions(
            local_window_size=3,
            min_frame_features=16,
            min_pair_matches=12,
            min_map_correspondences=8,
            keyframe_min_interval_s=0.0,
            keyframe_max_interval_s=0.0,
            min_triangulation_angle_deg=0.1,
            max_reprojection_error_deg=1.0,
            max_tracking_pose_candidates=1,
            global_refine_on_finish=False,
            loop_min_keyframe_separation=10,
            local_ba_max_points=64,
            local_ba_max_nfev=30,
        ),
    )
    outputs = [
        session.add_features(features, timestamp_s=float(index))
        for index, features in enumerate(feature_sets.values())
    ]
    assert [item.state for item in outputs] == [
        "initializing",
        "keyframe",
        "keyframe",
    ]
    assert outputs[-1].map_correspondence_count >= 8
    assert "spherical-2d3d-refined" in outputs[-1].reasons
    assert session.finish().success


def test_map_correspondences_exclude_relative_pose_outliers():
    pytest.importorskip("cv2")
    points, descriptors, rotations, centers, slices = _scene(group_size=16)
    feature_sets = [
        _feature_set(
            f"filter-{index}",
            slices["A"],
            points,
            descriptors,
            rotations[index],
            centers[index],
        )
        for index in range(2)
    ]
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf", face_sampler="cube", face_shape_hw=(64, 64), max_features=32
    )
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_oracle_estimator(),
    )
    reference_pose = PoseState("filter-0", 0.0, rotations[0], centers[0])
    for feature_index, bearing in enumerate(feature_sets[0].bearings):
        session._map.add_point(
            centers[0] + 5.0 * bearing,
            (
                ObservationState("filter-0", feature_index, bearing),
                ObservationState("support", feature_index, bearing),
            ),
        )
    matches = pipeline.match(feature_sets[0], feature_sets[1])
    relative = _OraclePose(np.eye(3), np.asarray((-1.0, 0.0, 0.0)), len(matches))
    relative.inlier_mask[1::2] = False
    evaluation = _PairEvaluation(
        _KeyframeRecord(feature_sets[0], reference_pose),
        matches,
        relative,
    )
    correspondences = session._map_correspondences((evaluation,))
    assert len(correspondences) == int(relative.inlier_mask.sum())
    assert {item[2] for item in correspondences} == set(
        matches.feature_indices_b[relative.inlier_mask].tolist()
    )


def test_add_frame_uses_real_erp_projection_and_opencv_without_mutation():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(19)
    panorama = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
    original = panorama.copy()
    pipeline = SphericalFeaturePipeline.from_preset(
        "orb-hamming",
        face_sampler="cube",
        face_fov_deg=100.0,
        face_shape_hw=(96, 96),
        max_features=256,
        edge_margin_px=4,
        ratio_test=0.8,
    )
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_oracle_estimator(),
        options=SphericalIncrementalSLAMOptions(min_frame_features=8),
    )
    output = session.add_frame(panorama, timestamp_s=0.0, frame_id="erp-000")
    assert output.state == "initializing"
    assert output.pose is not None
    assert output.feature_count >= 8
    np.testing.assert_array_equal(panorama, original)
    assert session.describe()["input"] == "central-equirectangular-image"


def test_incremental_input_validation_and_explicit_failure():
    pipeline = SphericalFeaturePipeline.from_preset(
        "sift-bf", face_sampler="cube", face_shape_hw=(32, 32), max_features=32
    )
    session = SphericalIncrementalSLAM(
        feature_pipeline=pipeline,
        relative_pose_estimator=_oracle_estimator(),
        options=SphericalIncrementalSLAMOptions(min_frame_features=4),
    )
    points, descriptors, rotations, centers, slices = _scene(group_size=3)
    first = _feature_set(
        "first", slices["A"], points, descriptors, rotations[0], centers[0]
    )
    output = session.add_features(first, timestamp_s=1.0)
    assert output.state == "lost"
    assert output.reasons == ("insufficient-features",)
    with pytest.raises(ValueError, match="duplicate"):
        session.add_features(first, timestamp_s=2.0)
    second = _feature_set(
        "second", slices["B"], points, descriptors, rotations[1], centers[1]
    )
    with pytest.raises(ValueError, match="increase strictly"):
        session.add_features(second, timestamp_s=0.5)
    result = session.finish()
    assert not result.success
    assert "insufficient-keyframes" in result.failure_reasons
    assert "no-active-map-points" in result.failure_reasons
    with pytest.raises(RuntimeError, match="only once"):
        session.finish()


def test_incremental_option_validation():
    with pytest.raises(ValueError, match="local_window_size"):
        SphericalIncrementalSLAMOptions(local_window_size=1)
    with pytest.raises(ValueError, match="keyframe_max_interval_s"):
        SphericalIncrementalSLAMOptions(
            keyframe_min_interval_s=1.0,
            keyframe_max_interval_s=0.5,
        )
    with pytest.raises(ValueError, match="tracked_ratio"):
        SphericalIncrementalSLAMOptions(keyframe_min_tracked_ratio=1.1)
    with pytest.raises(ValueError, match="edge_admission"):
        SphericalIncrementalSLAMOptions(edge_admission="anything")
