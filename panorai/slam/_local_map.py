"""Numerical local-map operations for Experimental spherical SLAM."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from panorai.reconstruction._math import (
    angular_error,
    rotation_exp,
)
from panorai.reconstruction._mapper import _spherical_log_residual_batch

from ._models import (
    SphericalIncrementalSLAMOptions,
    SphericalLocalBAReport,
    SphericalMapObservation,
    SphericalMapPoint,
)


@dataclass(slots=True)
class PoseState:
    frame_id: str
    timestamp_s: float
    rotation: np.ndarray
    center: np.ndarray


@dataclass(slots=True)
class ObservationState:
    frame_id: str
    feature_index: int
    bearing: np.ndarray
    active: bool = True
    residual_deg: float = 0.0


@dataclass(slots=True)
class MapPointState:
    point_id: str
    position: np.ndarray
    observations: dict[str, ObservationState] = field(default_factory=dict)
    active: bool = True


@dataclass(frozen=True, slots=True)
class AbsolutePoseEstimate:
    rotation: np.ndarray
    center: np.ndarray
    initial_cost: float
    final_cost: float
    median_error_deg: float
    p90_error_deg: float
    success: bool


class LocalSphericalMap:
    """Mutable internal map with immutable public snapshots."""

    def __init__(self) -> None:
        self.points: dict[str, MapPointState] = {}
        self.feature_to_point: dict[tuple[str, int], str] = {}
        self._next_point = 0

    def point_for_feature(self, frame_id: str, feature_index: int) -> str | None:
        point_id = self.feature_to_point.get((frame_id, int(feature_index)))
        if point_id is None:
            return None
        point = self.points.get(point_id)
        return point_id if point is not None and point.active else None

    def add_point(
        self,
        position: np.ndarray,
        observations: Iterable[ObservationState],
    ) -> str | None:
        items = tuple(observations)
        if len(items) < 2 or len({item.frame_id for item in items}) != len(items):
            return None
        if any(
            (item.frame_id, int(item.feature_index)) in self.feature_to_point
            for item in items
        ):
            return None
        point_id = f"point-{self._next_point:08d}"
        self._next_point += 1
        point = MapPointState(
            point_id=point_id,
            position=np.asarray(position, dtype=np.float64).copy(),
            observations={item.frame_id: item for item in items},
        )
        self.points[point_id] = point
        for item in items:
            self.feature_to_point[(item.frame_id, int(item.feature_index))] = point_id
        return point_id

    def add_observation(
        self,
        point_id: str,
        observation: ObservationState,
    ) -> bool:
        point = self.points.get(point_id)
        if (
            point is None
            or not point.active
            or observation.frame_id in point.observations
        ):
            return False
        key = (observation.frame_id, int(observation.feature_index))
        existing = self.feature_to_point.get(key)
        if existing is not None and existing != point_id:
            return False
        point.observations[observation.frame_id] = observation
        self.feature_to_point[key] = point_id
        return True

    def deactivate_observation(self, point_id: str, frame_id: str) -> None:
        point = self.points[point_id]
        observation = point.observations[frame_id]
        observation.active = False
        self.feature_to_point.pop((frame_id, observation.feature_index), None)
        if sum(item.active for item in point.observations.values()) < 2:
            point.active = False

    @property
    def active_point_count(self) -> int:
        return sum(point.active for point in self.points.values())

    def snapshot(self) -> tuple[SphericalMapPoint, ...]:
        result = []
        for point_id in sorted(self.points):
            point = self.points[point_id]
            observations = tuple(
                SphericalMapObservation(
                    frame_id=item.frame_id,
                    feature_index=item.feature_index,
                    bearing_xyz=item.bearing,
                    active=item.active,
                    residual_deg=item.residual_deg,
                )
                for item in sorted(
                    point.observations.values(), key=lambda value: value.frame_id
                )
            )
            result.append(
                SphericalMapPoint(
                    point_id=point.point_id,
                    position_world=point.position,
                    observations=observations,
                    active=point.active,
                )
            )
        return tuple(result)


def triangulate_bearings(
    observations: Iterable[tuple[PoseState, np.ndarray]],
    *,
    min_angle_deg: float,
    max_error_deg: float,
) -> np.ndarray | None:
    """Intersect world-frame rays with explicit positive-range checks."""

    items = tuple(observations)
    if len(items) < 2:
        return None
    centers = []
    rays = []
    for pose, bearing in items:
        value = np.asarray(bearing, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if value.shape != (3,) or not np.isfinite(value).all() or norm <= 0.0:
            return None
        centers.append(np.asarray(pose.center, dtype=np.float64))
        ray = pose.rotation.T @ (value / norm)
        rays.append(ray / np.linalg.norm(ray))
    greatest_angle = max(
        math.degrees(math.acos(float(np.clip(rays[left] @ rays[right], -1.0, 1.0))))
        for left in range(len(rays))
        for right in range(left + 1, len(rays))
    )
    if greatest_angle < min_angle_deg:
        return None
    identity = np.eye(3)
    matrix = sum((identity - np.outer(ray, ray) for ray in rays), np.zeros((3, 3)))
    vector = sum(
        (
            (identity - np.outer(ray, ray)) @ center
            for center, ray in zip(centers, rays)
        ),
        np.zeros(3),
    )
    if np.linalg.matrix_rank(matrix, tol=1e-10) < 3:
        return None
    point = np.linalg.solve(matrix, vector)
    if not np.isfinite(point).all():
        return None
    for (pose, bearing), center, ray in zip(items, centers, rays):
        if float(np.dot(point - center, ray)) <= 1e-9:
            return None
        predicted = pose.rotation @ (point - center)
        predicted /= np.linalg.norm(predicted)
        measured = np.asarray(bearing, dtype=np.float64)
        measured /= np.linalg.norm(measured)
        if math.degrees(angular_error(measured, predicted)) > max_error_deg:
            return None
    return point


def refine_absolute_pose(
    initial: PoseState,
    correspondences: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    loss: str,
    loss_scale_deg: float,
    max_nfev: int,
) -> AbsolutePoseEstimate:
    """Refine a world-to-camera pose from spherical 2D-3D correspondences."""

    items = tuple(
        (
            np.asarray(point, dtype=np.float64),
            _unit(np.asarray(bearing, dtype=np.float64)),
        )
        for point, bearing in correspondences
    )
    if len(items) < 3:
        return AbsolutePoseEstimate(
            initial.rotation.copy(),
            initial.center.copy(),
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            False,
        )
    rotation0 = np.asarray(initial.rotation, dtype=np.float64)
    center0 = np.asarray(initial.center, dtype=np.float64)
    points = np.stack([item[0] for item in items])
    bearings = np.stack([item[1] for item in items])

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rotation_exp(parameters[:3]) @ rotation0, center0 + parameters[3:]

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation, center = decode(parameters)
        camera = (points - center) @ rotation.T
        norms = np.linalg.norm(camera, axis=1)
        valid = np.isfinite(camera).all(axis=1) & (norms > 1e-12)
        predicted = camera / np.where(valid, norms, 1.0)[:, None]
        values = _spherical_log_residual_batch(bearings, predicted)
        values[~valid] = math.pi
        return values.ravel()

    zero = np.zeros(6, dtype=np.float64)
    initial_residual = residual(zero)
    initial_cost = 0.5 * float(initial_residual @ initial_residual)
    solution = least_squares(
        residual,
        zero,
        loss=loss,
        f_scale=math.radians(loss_scale_deg),
        max_nfev=max_nfev,
        method="trf",
    )
    final_residual = residual(solution.x)
    final_cost = 0.5 * float(final_residual @ final_residual)
    if not np.isfinite(solution.x).all() or final_cost > initial_cost + 1e-12:
        parameters = zero
        final_residual = initial_residual
        final_cost = initial_cost
        successful = False
    else:
        parameters = solution.x
        successful = bool(solution.success)
    rotation, center = decode(parameters)
    angular = np.linalg.norm(final_residual.reshape(-1, 2), axis=1)
    return AbsolutePoseEstimate(
        rotation=rotation,
        center=center,
        initial_cost=initial_cost,
        final_cost=final_cost,
        median_error_deg=float(np.degrees(np.median(angular))),
        p90_error_deg=float(np.degrees(np.percentile(angular, 90))),
        success=successful,
    )


class LocalBundleAdjuster:
    """Robust spherical BA over a bounded keyframe window."""

    def __init__(self, options: SphericalIncrementalSLAMOptions) -> None:
        self.options = options

    def optimize(
        self,
        poses: dict[str, PoseState],
        local_map: LocalSphericalMap,
        keyframe_ids: Iterable[str],
    ) -> SphericalLocalBAReport | None:
        frame_ids = tuple(item for item in keyframe_ids if item in poses)
        if len(frame_ids) < 2:
            return None
        frame_set = set(frame_ids)
        candidates = []
        for point in local_map.points.values():
            local_observations = [
                item
                for item in point.observations.values()
                if item.active and item.frame_id in frame_set
            ]
            if point.active and len(local_observations) >= 2:
                candidates.append((point, local_observations))
        candidates.sort(key=lambda item: (-len(item[1]), item[0].point_id))
        candidates = candidates[: self.options.local_ba_max_points]
        if not candidates:
            return None

        # Two fixed poses remove global SE(3) and monocular scale gauge.
        variable_ids = frame_ids[2:]
        pose_offsets = {
            frame_id: 6 * index for index, frame_id in enumerate(variable_ids)
        }
        pose_parameter_count = 6 * len(variable_ids)
        point_offsets = {
            point.point_id: pose_parameter_count + 3 * index
            for index, (point, _) in enumerate(candidates)
        }
        parameter_count = pose_parameter_count + 3 * len(candidates)
        initial = np.zeros(parameter_count, dtype=np.float64)
        observations = [
            (point.point_id, observation)
            for point, point_observations in candidates
            for observation in point_observations
        ]
        base_rotations = {
            frame_id: poses[frame_id].rotation.copy() for frame_id in variable_ids
        }
        base_centers = {
            frame_id: poses[frame_id].center.copy() for frame_id in variable_ids
        }
        base_points = {point.point_id: point.position.copy() for point, _ in candidates}
        observation_frame_ids = tuple(item.frame_id for _, item in observations)
        observation_point_ids = tuple(point_id for point_id, _ in observations)
        measured_bearings = np.stack([item.bearing for _, item in observations])

        def decoded_pose(
            frame_id: str, parameters: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            offset = pose_offsets.get(frame_id)
            if offset is None:
                return poses[frame_id].rotation, poses[frame_id].center
            rotation = (
                rotation_exp(parameters[offset : offset + 3]) @ base_rotations[frame_id]
            )
            center = base_centers[frame_id] + parameters[offset + 3 : offset + 6]
            return rotation, center

        def decoded_point(point_id: str, parameters: np.ndarray) -> np.ndarray:
            offset = point_offsets[point_id]
            return base_points[point_id] + parameters[offset : offset + 3]

        def residual(parameters: np.ndarray) -> np.ndarray:
            decoded_poses = {
                frame_id: decoded_pose(frame_id, parameters) for frame_id in frame_ids
            }
            decoded_points = {
                point.point_id: decoded_point(point.point_id, parameters)
                for point, _ in candidates
            }
            rotations = np.stack(
                [decoded_poses[frame_id][0] for frame_id in observation_frame_ids]
            )
            centers = np.stack(
                [decoded_poses[frame_id][1] for frame_id in observation_frame_ids]
            )
            points = np.stack(
                [decoded_points[point_id] for point_id in observation_point_ids]
            )
            camera = np.einsum("nij,nj->ni", rotations, points - centers)
            norms = np.linalg.norm(camera, axis=1)
            valid = np.isfinite(camera).all(axis=1) & (norms > 1e-12)
            predicted = camera / np.where(valid, norms, 1.0)[:, None]
            values = _spherical_log_residual_batch(measured_bearings, predicted)
            values[~valid] = math.pi
            return values.ravel()

        sparsity = lil_matrix((2 * len(observations), parameter_count), dtype=np.int8)
        for observation_index, (point_id, observation) in enumerate(observations):
            rows = slice(2 * observation_index, 2 * observation_index + 2)
            pose_offset = pose_offsets.get(observation.frame_id)
            if pose_offset is not None:
                sparsity[rows, pose_offset : pose_offset + 6] = 1
            point_offset = point_offsets[point_id]
            sparsity[rows, point_offset : point_offset + 3] = 1

        initial_residual = residual(initial)
        initial_cost = 0.5 * float(initial_residual @ initial_residual)
        solution = least_squares(
            residual,
            initial,
            loss=self.options.local_ba_loss,
            f_scale=math.radians(self.options.local_ba_loss_scale_deg),
            max_nfev=self.options.local_ba_max_nfev,
            method="trf",
            jac_sparsity=sparsity.tocsr(),
        )
        final_residual = residual(solution.x)
        final_cost = 0.5 * float(final_residual @ final_residual)
        accepted = (
            np.isfinite(solution.x).all()
            and np.isfinite(final_cost)
            and final_cost <= initial_cost + 1e-12
        )
        parameters = solution.x if accepted else initial
        if not accepted:
            final_residual = initial_residual
            final_cost = initial_cost

        for frame_id in variable_ids:
            rotation, center = decoded_pose(frame_id, parameters)
            poses[frame_id].rotation = rotation
            poses[frame_id].center = center
        for point, _ in candidates:
            point.position = decoded_point(point.point_id, parameters)

        errors = np.degrees(np.linalg.norm(final_residual.reshape(-1, 2), axis=1))
        removed = 0
        for (point_id, observation), error in zip(observations, errors):
            observation.residual_deg = float(error)
            if error > self.options.max_reprojection_error_deg:
                local_map.deactivate_observation(point_id, observation.frame_id)
                removed += 1
        return SphericalLocalBAReport(
            keyframe_ids=frame_ids,
            point_count=len(candidates),
            observation_count=len(observations),
            variable_pose_count=len(variable_ids),
            initial_cost=initial_cost,
            final_cost=final_cost,
            removed_observation_count=removed,
            success=bool(accepted and solution.success),
        )


def observation_error_deg(
    pose: PoseState,
    point: np.ndarray,
    bearing: np.ndarray,
) -> float:
    camera = pose.rotation @ (np.asarray(point) - pose.center)
    norm = float(np.linalg.norm(camera))
    if norm <= 1e-12 or not math.isfinite(norm):
        return math.inf
    return math.degrees(angular_error(_unit(bearing), camera / norm))


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm <= 0.0:
        raise ValueError("bearing must be a finite non-zero vector with shape (3,)")
    return value / norm


__all__ = [
    "AbsolutePoseEstimate",
    "LocalBundleAdjuster",
    "LocalSphericalMap",
    "MapPointState",
    "ObservationState",
    "PoseState",
    "observation_error_deg",
    "refine_absolute_pose",
    "triangulate_bearings",
]
