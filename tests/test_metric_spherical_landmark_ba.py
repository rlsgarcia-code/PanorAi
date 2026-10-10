from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.reconstruction.metric_landmark_ba import (
    METRIC_SPHERICAL_LANDMARK_BA_INTERFACE,
    MetricBearingObservation,
    MetricLandmarkBAOptions,
    MetricRadialRangePrior,
    MetricScaleGauge,
    MetricSphericalCamera,
    MetricSphericalLandmark,
    refine_metric_spherical_landmarks,
)
from panorai.reconstruction._math import rotation_exp, rotation_log
from panorai.reconstruction._metric_landmark_solver import (
    CameraState,
    LandmarkState,
    finite_difference_observation_jacobians,
    observation_residual_jacobians,
)


def _bearing(rotation: np.ndarray, center: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = rotation @ (point - center)
    return value / np.linalg.norm(value)


def _rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(rotation_log(estimated @ truth.T))))


def _metric_scene(seed: int = 41042, point_count: int = 48):
    rng = np.random.default_rng(seed)
    truth_rotations = (
        np.eye(3),
        rotation_exp(np.asarray([0.01, 0.03, -0.02])),
        rotation_exp(np.asarray([-0.02, 0.04, 0.01])),
    )
    truth_centers = (
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.4, 0.1, 0.8]),
    )
    points = rng.uniform((-2.0, -1.0, 3.0), (2.0, 1.0, 7.0), (point_count, 3))
    cameras = []
    for index, (rotation, center) in enumerate(zip(truth_rotations, truth_centers)):
        rotation_noise = np.zeros(3) if index == 0 else rng.normal(0.0, 0.008, 3)
        center_noise = np.zeros(3) if index == 0 else rng.normal(0.0, 0.03, 3)
        cameras.append(
            MetricSphericalCamera(
                f"c{index}",
                rotation_exp(rotation_noise) @ rotation,
                center + center_noise,
            )
        )
    landmarks = []
    observations = []
    priors = []
    for index, point in enumerate(points):
        landmark_id = f"p{index}"
        landmarks.append(
            MetricSphericalLandmark(
                landmark_id,
                point + rng.normal(0.0, 0.04, 3),
            )
        )
        for camera_index in range(3):
            observations.append(
                MetricBearingObservation(
                    f"c{camera_index}",
                    landmark_id,
                    _bearing(
                        truth_rotations[camera_index],
                        truth_centers[camera_index],
                        point,
                    ),
                )
            )
        priors.append(
            MetricRadialRangePrior(
                "c0",
                landmark_id,
                float(np.linalg.norm(point - truth_centers[0])),
                0.05,
            )
        )
    return (
        cameras,
        landmarks,
        observations,
        priors,
        truth_rotations,
        truth_centers,
        points,
    )


def test_metric_observation_jacobians_match_central_difference() -> None:
    camera = CameraState(
        "camera",
        rotation_exp(np.asarray([0.1, -0.05, 0.02])),
        np.asarray([0.2, -0.1, 0.3]),
    )
    landmark = LandmarkState("point", np.asarray([1.0, 0.4, 4.0]))
    measured = np.asarray([0.1, 0.2, 0.97])
    measured /= np.linalg.norm(measured)

    _, analytic_pose, analytic_point = observation_residual_jacobians(
        camera, landmark, measured
    )
    numeric_pose, numeric_point = finite_difference_observation_jacobians(
        camera, landmark, measured
    )

    np.testing.assert_allclose(analytic_pose, numeric_pose, rtol=2e-6, atol=5e-9)
    np.testing.assert_allclose(analytic_point, numeric_point, rtol=2e-6, atol=5e-9)


def test_metric_log_map_keeps_near_antipodal_error() -> None:
    camera = CameraState("camera", np.eye(3), np.zeros(3))
    angle = math.radians(179.0)
    landmark = LandmarkState(
        "point", np.asarray([math.sin(angle), 0.0, math.cos(angle)]) * 4.0
    )

    residual, _, _ = observation_residual_jacobians(
        camera, landmark, np.asarray([0.0, 0.0, 1.0])
    )

    np.testing.assert_allclose(np.linalg.norm(residual), angle, rtol=1e-9)


def test_public_metric_landmark_ba_recovers_seeded_scene() -> None:
    (
        cameras,
        landmarks,
        observations,
        priors,
        truth_rotations,
        truth_centers,
        points,
    ) = _metric_scene()
    input_camera = cameras[1].center_world_m.copy()
    input_landmark = landmarks[0].position_world_m.copy()

    result = refine_metric_spherical_landmarks(
        cameras,
        landmarks,
        observations,
        radial_range_priors=priors,
        fixed_camera_ids=("c0",),
        scale_gauge=MetricScaleGauge("c0", "c1", 1.0, 0.01),
        options=MetricLandmarkBAOptions(max_iterations=40, initial_damping=1e-5),
    )

    assert result.interface == METRIC_SPHERICAL_LANDMARK_BA_INTERFACE
    assert result.support == "supplied-landmarks-only"
    assert result.report.final_cost < result.report.initial_cost * 1e-6
    assert result.report.final_mean_error_deg < 1e-5
    assert result.report.landmark_count == len(points)
    assert result.report.variable_camera_count == 2
    for camera_index in (1, 2):
        camera = result.camera_by_id[f"c{camera_index}"]
        assert (
            np.linalg.norm(camera.center_world_m - truth_centers[camera_index]) < 1e-5
        )
        assert (
            _rotation_error_deg(
                camera.rotation_world_to_camera,
                truth_rotations[camera_index],
            )
            < 1e-4
        )
    mean_point_error = np.mean(
        [
            np.linalg.norm(result.landmark_by_id[f"p{index}"].position_world_m - point)
            for index, point in enumerate(points)
        ]
    )
    assert mean_point_error < 1e-5
    np.testing.assert_array_equal(cameras[1].center_world_m, input_camera)
    np.testing.assert_array_equal(landmarks[0].position_world_m, input_landmark)
    assert not result.cameras[0].center_world_m.flags.writeable
    assert not result.landmarks[0].position_world_m.flags.writeable


def test_metric_landmark_ba_requires_two_views_per_landmark() -> None:
    cameras = (
        MetricSphericalCamera("a", np.eye(3), np.zeros(3)),
        MetricSphericalCamera("b", np.eye(3), np.asarray([1.0, 0.0, 0.0])),
    )
    landmarks = (MetricSphericalLandmark("p", np.asarray([0.0, 0.0, 4.0])),)
    observations = (MetricBearingObservation("a", "p", np.asarray([0.0, 0.0, 1.0])),)

    with pytest.raises(ValueError, match="at least two views"):
        refine_metric_spherical_landmarks(
            cameras,
            landmarks,
            observations,
            fixed_camera_ids=("a",),
            scale_gauge=MetricScaleGauge("a", "b", 1.0),
        )


def test_metric_landmark_ba_rejects_missing_pose_gauge() -> None:
    cameras = (
        MetricSphericalCamera("a", np.eye(3), np.zeros(3)),
        MetricSphericalCamera("b", np.eye(3), np.asarray([1.0, 0.0, 0.0])),
    )
    landmark = MetricSphericalLandmark("p", np.asarray([0.0, 0.0, 4.0]))
    observations = (
        MetricBearingObservation("a", "p", np.asarray([0.0, 0.0, 1.0])),
        MetricBearingObservation(
            "b", "p", np.asarray([-1.0, 0.0, 4.0]) / math.sqrt(17.0)
        ),
    )

    with pytest.raises(ValueError, match="reference camera must be fixed"):
        refine_metric_spherical_landmarks(
            cameras,
            (landmark,),
            observations,
            fixed_camera_ids=("b",),
            scale_gauge=MetricScaleGauge("a", "b", 1.0),
        )


def test_radial_range_prior_does_not_accept_zero_as_invalid_sentinel() -> None:
    with pytest.raises(ValueError, match="radial_range_m"):
        MetricRadialRangePrior("a", "p", 0.0, 0.1)
