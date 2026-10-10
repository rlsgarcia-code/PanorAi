"""Internal metric spherical landmark BA with point Schur elimination."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from panorai.reconstruction._math import rotation_exp, skew, tangent_basis


@dataclass(slots=True)
class CameraState:
    camera_id: str
    rotation_world_to_camera: np.ndarray
    center_world: np.ndarray

    def __post_init__(self) -> None:
        self.rotation_world_to_camera = np.array(
            self.rotation_world_to_camera, dtype=np.float64, copy=True
        ).reshape(3, 3)
        self.center_world = np.array(
            self.center_world, dtype=np.float64, copy=True
        ).reshape(3)


@dataclass(slots=True)
class LandmarkState:
    landmark_id: str
    position_world: np.ndarray

    def __post_init__(self) -> None:
        self.position_world = np.array(
            self.position_world, dtype=np.float64, copy=True
        ).reshape(3)


@dataclass(frozen=True, slots=True)
class BearingObservation:
    camera_id: str
    landmark_id: str
    bearing_camera: np.ndarray
    confidence: float = 1.0

    def __post_init__(self) -> None:
        bearing = _unit(self.bearing_camera)
        object.__setattr__(self, "bearing_camera", bearing)
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence <= 1.0:
            raise ValueError("observation confidence must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class MonocularRangePrior:
    camera_id: str
    landmark_id: str
    range_m: float
    sigma_log_range: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        for name in ("range_m", "sigma_log_range", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.confidence > 1.0:
            raise ValueError("confidence must not exceed 1")


@dataclass(slots=True)
class ObservationGraph:
    cameras: dict[str, CameraState] = field(default_factory=dict)
    landmarks: dict[str, LandmarkState] = field(default_factory=dict)
    observations: list[BearingObservation] = field(default_factory=list)
    depth_priors: list[MonocularRangePrior] = field(default_factory=list)

    def add_camera(self, camera: CameraState) -> None:
        if camera.camera_id in self.cameras:
            raise ValueError(f"duplicate camera: {camera.camera_id}")
        self.cameras[camera.camera_id] = camera

    def add_landmark(
        self,
        landmark: LandmarkState,
        observations: Sequence[BearingObservation],
        *,
        depth_prior: MonocularRangePrior | None = None,
    ) -> None:
        if landmark.landmark_id in self.landmarks:
            raise ValueError(f"duplicate landmark: {landmark.landmark_id}")
        camera_ids = {item.camera_id for item in observations}
        if len(observations) < 2 or len(camera_ids) < 2:
            raise ValueError("a landmark must be validated in at least two views")
        if any(item.landmark_id != landmark.landmark_id for item in observations):
            raise ValueError("observations must identify the promoted landmark")
        if not camera_ids <= self.cameras.keys():
            raise ValueError("all observation cameras must already exist")
        self.landmarks[landmark.landmark_id] = landmark
        self.observations.extend(observations)
        if depth_prior is not None:
            if depth_prior.landmark_id != landmark.landmark_id:
                raise ValueError("depth prior must identify the promoted landmark")
            self.depth_priors.append(depth_prior)

    def add_observation(self, observation: BearingObservation) -> None:
        if observation.camera_id not in self.cameras:
            raise ValueError("observation camera is unknown")
        if observation.landmark_id not in self.landmarks:
            raise ValueError("observation landmark is unknown")
        key = (observation.camera_id, observation.landmark_id)
        if any((item.camera_id, item.landmark_id) == key for item in self.observations):
            raise ValueError("duplicate camera-landmark observation")
        self.observations.append(observation)

    def observation_matrix(self) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
        camera_ids = tuple(sorted(self.cameras))
        landmark_ids = tuple(sorted(self.landmarks))
        camera_index = {value: index for index, value in enumerate(camera_ids)}
        landmark_index = {value: index for index, value in enumerate(landmark_ids)}
        matrix = np.zeros((len(camera_ids), len(landmark_ids)), dtype=bool)
        for item in self.observations:
            matrix[camera_index[item.camera_id], landmark_index[item.landmark_id]] = (
                True
            )
        return matrix, camera_ids, landmark_ids


@dataclass(frozen=True, slots=True)
class ScaleGauge:
    reference_camera_id: str
    anchor_camera_id: str
    baseline_m: float
    sigma_m: float = 1e-3


@dataclass(frozen=True, slots=True)
class BundleOptions:
    angular_sigma_deg: float = 0.25
    angular_huber_delta_deg: float = 1.0
    depth_huber_delta_log: float = 0.10
    max_iterations: int = 20
    initial_damping: float = 1e-3
    minimum_damping: float = 1e-9
    maximum_damping: float = 1e9
    step_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "angular_sigma_deg",
            "angular_huber_delta_deg",
            "depth_huber_delta_log",
            "initial_damping",
            "minimum_damping",
            "maximum_damping",
            "step_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_damping < self.minimum_damping:
            raise ValueError("maximum_damping must not be below minimum_damping")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class BundleReport:
    initial_cost: float
    final_cost: float
    iterations: int
    accepted_steps: int
    observation_count: int
    landmark_count: int
    variable_camera_count: int
    schur_point_count: int
    initial_mean_error_deg: float
    final_mean_error_deg: float
    converged: bool
    gauge: str


def observation_residual_jacobians(
    camera: CameraState,
    landmark: LandmarkState,
    measured_bearing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent residual and analytic pose/point Jacobian blocks."""

    measured = _unit(measured_bearing)
    delta_world = landmark.position_world - camera.center_world
    vector = camera.rotation_world_to_camera @ delta_world
    distance = float(np.linalg.norm(vector))
    if not math.isfinite(distance) or distance <= 1e-12:
        raise ValueError("landmark projects to an invalid camera vector")
    predicted = vector / distance
    basis = tangent_basis(measured)
    cosine = float(np.clip(measured @ predicted, -1.0, 1.0))
    tangent_coordinates = basis.T @ predicted
    sine = float(np.linalg.norm(tangent_coordinates))
    angle = math.atan2(sine, cosine)
    if sine < 1e-8 and cosine > 0.0:
        scale = 1.0
        scale_derivative_cosine = -1.0 / 3.0
    else:
        safe_sine = max(sine, 1e-8)
        scale = angle / safe_sine
        scale_derivative_cosine = (angle * cosine - safe_sine) / safe_sine**3
    residual = scale * tangent_coordinates
    log_map_jacobian = scale * basis.T + np.outer(
        tangent_coordinates, scale_derivative_cosine * measured
    )
    normalizer = (np.eye(3) - np.outer(predicted, predicted)) / distance
    point_jacobian = log_map_jacobian @ normalizer @ camera.rotation_world_to_camera
    rotation_jacobian = log_map_jacobian @ normalizer @ (-skew(vector))
    center_jacobian = -point_jacobian
    pose_jacobian = np.column_stack((rotation_jacobian, center_jacobian))
    return residual, pose_jacobian, point_jacobian


def finite_difference_observation_jacobians(
    camera: CameraState,
    landmark: LandmarkState,
    measured_bearing: np.ndarray,
    *,
    step: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    pose_jacobian = np.empty((2, 6), dtype=np.float64)
    point_jacobian = np.empty((2, 3), dtype=np.float64)
    for axis in range(6):
        delta = np.zeros(6, dtype=np.float64)
        delta[axis] = step
        plus = _perturbed_camera(camera, delta)
        minus = _perturbed_camera(camera, -delta)
        pose_jacobian[:, axis] = (
            observation_residual_jacobians(plus, landmark, measured_bearing)[0]
            - observation_residual_jacobians(minus, landmark, measured_bearing)[0]
        ) / (2.0 * step)
    for axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = step
        plus = LandmarkState(landmark.landmark_id, landmark.position_world + delta)
        minus = LandmarkState(landmark.landmark_id, landmark.position_world - delta)
        point_jacobian[:, axis] = (
            observation_residual_jacobians(camera, plus, measured_bearing)[0]
            - observation_residual_jacobians(camera, minus, measured_bearing)[0]
        ) / (2.0 * step)
    return pose_jacobian, point_jacobian


class SchurBundleAdjuster:
    """Robust LM bundle adjustment eliminating independent 3D point blocks."""

    def __init__(self, options: BundleOptions | None = None) -> None:
        self.options = options or BundleOptions()

    def optimize(
        self,
        graph: ObservationGraph,
        *,
        fixed_camera_ids: Iterable[str],
        scale_gauge: ScaleGauge,
    ) -> BundleReport:
        fixed = frozenset(fixed_camera_ids)
        if scale_gauge.reference_camera_id not in fixed:
            raise ValueError("the scale-gauge reference camera must be fixed")
        variable_ids = tuple(sorted(set(graph.cameras) - fixed))
        pose_offsets = {value: 6 * index for index, value in enumerate(variable_ids)}
        landmark_ids = tuple(sorted(graph.landmarks))
        observations = tuple(graph.observations)
        _validate_graph(graph, scale_gauge)
        initial_cost, initial_errors = _graph_cost(graph, self.options, scale_gauge)
        damping = self.options.initial_damping
        accepted_steps = 0
        converged = False
        iterations = 0
        for iteration in range(self.options.max_iterations):
            iterations = iteration + 1
            linear = _linearize(
                graph,
                variable_ids,
                pose_offsets,
                landmark_ids,
                self.options,
                scale_gauge,
            )
            pose_step, point_steps = _schur_step(linear, damping)
            step_norm = math.sqrt(
                float(pose_step @ pose_step)
                + sum(float(value @ value) for value in point_steps.values())
            )
            if step_norm < self.options.step_tolerance:
                converged = True
                break
            backup = _snapshot_graph(graph)
            _apply_step(graph, variable_ids, pose_offsets, pose_step, point_steps)
            candidate_cost, _ = _graph_cost(graph, self.options, scale_gauge)
            if math.isfinite(candidate_cost) and candidate_cost < linear.cost:
                accepted_steps += 1
                damping = max(self.options.minimum_damping, damping * 0.35)
                if abs(linear.cost - candidate_cost) < 1e-12:
                    converged = True
                    break
            else:
                _restore_graph(graph, backup)
                damping = min(self.options.maximum_damping, damping * 10.0)
                if damping >= self.options.maximum_damping:
                    break
        final_cost, final_errors = _graph_cost(graph, self.options, scale_gauge)
        return BundleReport(
            initial_cost=initial_cost,
            final_cost=final_cost,
            iterations=iterations,
            accepted_steps=accepted_steps,
            observation_count=len(observations),
            landmark_count=len(landmark_ids),
            variable_camera_count=len(variable_ids),
            schur_point_count=len(landmark_ids),
            initial_mean_error_deg=float(np.degrees(np.mean(initial_errors))),
            final_mean_error_deg=float(np.degrees(np.mean(final_errors))),
            converged=converged,
            gauge=(
                f"fixed:{','.join(sorted(fixed))}; baseline:"
                f"{scale_gauge.reference_camera_id}-{scale_gauge.anchor_camera_id}="
                f"{scale_gauge.baseline_m:.9g}m"
            ),
        )


@dataclass(slots=True)
class _Linearization:
    hpp: np.ndarray
    bp: np.ndarray
    hll: dict[str, np.ndarray]
    bl: dict[str, np.ndarray]
    hpl: dict[str, np.ndarray]
    cost: float


def _linearize(
    graph: ObservationGraph,
    variable_ids: tuple[str, ...],
    pose_offsets: dict[str, int],
    landmark_ids: tuple[str, ...],
    options: BundleOptions,
    scale_gauge: ScaleGauge,
) -> _Linearization:
    pose_count = 6 * len(variable_ids)
    hpp = np.zeros((pose_count, pose_count), dtype=np.float64)
    bp = np.zeros(pose_count, dtype=np.float64)
    hll = {value: np.zeros((3, 3), dtype=np.float64) for value in landmark_ids}
    bl = {value: np.zeros(3, dtype=np.float64) for value in landmark_ids}
    hpl = {value: np.zeros((pose_count, 3), dtype=np.float64) for value in landmark_ids}
    cost = 0.0
    angular_sigma = math.radians(options.angular_sigma_deg)
    delta = options.angular_huber_delta_deg / options.angular_sigma_deg
    for observation in graph.observations:
        camera = graph.cameras[observation.camera_id]
        landmark = graph.landmarks[observation.landmark_id]
        residual, jp, jl = observation_residual_jacobians(
            camera, landmark, observation.bearing_camera
        )
        residual = residual / angular_sigma
        jp = jp / angular_sigma
        jl = jl / angular_sigma
        norm = float(np.linalg.norm(residual))
        robust_weight, robust_cost = _huber(norm, delta)
        weight = observation.confidence * robust_weight
        root = math.sqrt(weight)
        residual = root * residual
        jl = root * jl
        cost += observation.confidence * robust_cost
        hll[observation.landmark_id] += jl.T @ jl
        bl[observation.landmark_id] += jl.T @ residual
        offset = pose_offsets.get(observation.camera_id)
        if offset is not None:
            jp = root * jp
            block = slice(offset, offset + 6)
            hpp[block, block] += jp.T @ jp
            bp[block] += jp.T @ residual
            hpl[observation.landmark_id][block] += jp.T @ jl
    for prior in graph.depth_priors:
        camera = graph.cameras[prior.camera_id]
        landmark = graph.landmarks[prior.landmark_id]
        vector = landmark.position_world - camera.center_world
        distance = float(np.linalg.norm(vector))
        residual = math.log(distance / prior.range_m) / prior.sigma_log_range
        weight, prior_cost = _huber(residual, options.depth_huber_delta_log)
        weight *= prior.confidence
        root = math.sqrt(weight)
        jl = root * vector / (distance * distance * prior.sigma_log_range)
        value = root * residual
        cost += prior.confidence * prior_cost
        hll[prior.landmark_id] += np.outer(jl, jl)
        bl[prior.landmark_id] += jl * value
        offset = pose_offsets.get(prior.camera_id)
        if offset is not None:
            jp = np.zeros(6, dtype=np.float64)
            jp[3:] = -jl
            block = slice(offset, offset + 6)
            hpp[block, block] += np.outer(jp, jp)
            bp[block] += jp * value
            hpl[prior.landmark_id][block] += np.outer(jp, jl)
    _add_scale_gauge(graph, pose_offsets, scale_gauge, hpp, bp)
    scale_residual = _scale_residual(graph, scale_gauge)
    cost += 0.5 * scale_residual**2
    return _Linearization(hpp, bp, hll, bl, hpl, cost)


def _schur_step(
    linear: _Linearization, damping: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pose_count = linear.hpp.shape[0]
    schur = linear.hpp.copy()
    if pose_count:
        schur += damping * np.diag(np.maximum(np.diag(schur), 1.0))
    rhs = -linear.bp.copy()
    inverses: dict[str, np.ndarray] = {}
    for landmark_id, hessian in linear.hll.items():
        damped = hessian + damping * np.diag(np.maximum(np.diag(hessian), 1.0))
        inverse = np.linalg.inv(damped)
        inverses[landmark_id] = inverse
        cross = linear.hpl[landmark_id]
        if pose_count:
            schur -= cross @ inverse @ cross.T
            rhs += cross @ inverse @ linear.bl[landmark_id]
    if pose_count:
        pose_step = np.linalg.solve(schur, rhs)
    else:
        pose_step = np.empty(0, dtype=np.float64)
    point_steps = {
        landmark_id: -inverse
        @ (linear.bl[landmark_id] + linear.hpl[landmark_id].T @ pose_step)
        for landmark_id, inverse in inverses.items()
    }
    return pose_step, point_steps


def _graph_cost(
    graph: ObservationGraph,
    options: BundleOptions,
    scale_gauge: ScaleGauge,
) -> tuple[float, np.ndarray]:
    cost = 0.0
    errors = []
    angular_sigma = math.radians(options.angular_sigma_deg)
    angular_delta = options.angular_huber_delta_deg / options.angular_sigma_deg
    for item in graph.observations:
        residual = observation_residual_jacobians(
            graph.cameras[item.camera_id],
            graph.landmarks[item.landmark_id],
            item.bearing_camera,
        )[0]
        angular_error = float(np.linalg.norm(residual))
        errors.append(angular_error)
        cost += (
            item.confidence * _huber(angular_error / angular_sigma, angular_delta)[1]
        )
    for prior in graph.depth_priors:
        distance = float(
            np.linalg.norm(
                graph.landmarks[prior.landmark_id].position_world
                - graph.cameras[prior.camera_id].center_world
            )
        )
        residual = math.log(distance / prior.range_m) / prior.sigma_log_range
        cost += prior.confidence * _huber(residual, options.depth_huber_delta_log)[1]
    cost += 0.5 * _scale_residual(graph, scale_gauge) ** 2
    return cost, np.asarray(errors, dtype=np.float64)


def _add_scale_gauge(
    graph: ObservationGraph,
    pose_offsets: dict[str, int],
    gauge: ScaleGauge,
    hpp: np.ndarray,
    bp: np.ndarray,
) -> None:
    first = graph.cameras[gauge.reference_camera_id].center_world
    second = graph.cameras[gauge.anchor_camera_id].center_world
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        raise ValueError("scale anchor baseline is degenerate")
    residual = (distance - gauge.baseline_m) / gauge.sigma_m
    jacobians: list[tuple[int, np.ndarray]] = []
    for camera_id, sign in (
        (gauge.reference_camera_id, -1.0),
        (gauge.anchor_camera_id, 1.0),
    ):
        offset = pose_offsets.get(camera_id)
        if offset is not None:
            jacobian = np.zeros(6, dtype=np.float64)
            jacobian[3:] = sign * delta / (distance * gauge.sigma_m)
            jacobians.append((offset, jacobian))
            block = slice(offset, offset + 6)
            hpp[block, block] += np.outer(jacobian, jacobian)
            bp[block] += jacobian * residual
    if len(jacobians) == 2:
        (left, jl), (right, jr) = jacobians
        hpp[left : left + 6, right : right + 6] += np.outer(jl, jr)
        hpp[right : right + 6, left : left + 6] += np.outer(jr, jl)


def _scale_residual(graph: ObservationGraph, gauge: ScaleGauge) -> float:
    distance = float(
        np.linalg.norm(
            graph.cameras[gauge.anchor_camera_id].center_world
            - graph.cameras[gauge.reference_camera_id].center_world
        )
    )
    return (distance - gauge.baseline_m) / gauge.sigma_m


def _apply_step(
    graph: ObservationGraph,
    variable_ids: tuple[str, ...],
    pose_offsets: dict[str, int],
    pose_step: np.ndarray,
    point_steps: dict[str, np.ndarray],
) -> None:
    for camera_id in variable_ids:
        offset = pose_offsets[camera_id]
        camera = graph.cameras[camera_id]
        camera.rotation_world_to_camera = (
            rotation_exp(pose_step[offset : offset + 3])
            @ camera.rotation_world_to_camera
        )
        camera.center_world += pose_step[offset + 3 : offset + 6]
    for landmark_id, value in point_steps.items():
        graph.landmarks[landmark_id].position_world += value


def _snapshot_graph(
    graph: ObservationGraph,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    cameras = {
        key: (
            value.rotation_world_to_camera.copy(),
            value.center_world.copy(),
        )
        for key, value in graph.cameras.items()
    }
    landmarks = {
        key: value.position_world.copy() for key, value in graph.landmarks.items()
    }
    return cameras, landmarks


def _restore_graph(
    graph: ObservationGraph,
    snapshot: tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]],
) -> None:
    cameras, landmarks = snapshot
    for key, (rotation, center) in cameras.items():
        graph.cameras[key].rotation_world_to_camera = rotation
        graph.cameras[key].center_world = center
    for key, position in landmarks.items():
        graph.landmarks[key].position_world = position


def _perturbed_camera(camera: CameraState, delta: np.ndarray) -> CameraState:
    return CameraState(
        camera.camera_id,
        rotation_exp(delta[:3]) @ camera.rotation_world_to_camera,
        camera.center_world + delta[3:],
    )


def _huber(norm: float, delta: float) -> tuple[float, float]:
    absolute = abs(float(norm))
    if absolute <= delta:
        return 1.0, 0.5 * absolute * absolute
    return delta / max(absolute, 1e-15), delta * (absolute - 0.5 * delta)


def _validate_graph(graph: ObservationGraph, gauge: ScaleGauge) -> None:
    if gauge.reference_camera_id not in graph.cameras:
        raise ValueError("scale-gauge reference camera is unknown")
    if gauge.anchor_camera_id not in graph.cameras:
        raise ValueError("scale-gauge anchor camera is unknown")
    if gauge.baseline_m <= 0.0 or gauge.sigma_m <= 0.0:
        raise ValueError("scale gauge must use positive baseline and sigma")
    counts = {value: set() for value in graph.landmarks}
    for item in graph.observations:
        if (
            item.camera_id not in graph.cameras
            or item.landmark_id not in graph.landmarks
        ):
            raise ValueError("observation graph contains an unknown endpoint")
        counts[item.landmark_id].add(item.camera_id)
    if any(len(value) < 2 for value in counts.values()):
        raise ValueError("every optimized landmark requires at least two views")


def _unit(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("bearing must be finite and non-zero")
    return result / norm


__all__ = [
    "BearingObservation",
    "BundleOptions",
    "BundleReport",
    "CameraState",
    "LandmarkState",
    "MonocularRangePrior",
    "ObservationGraph",
    "ScaleGauge",
    "SchurBundleAdjuster",
    "finite_difference_observation_jacobians",
    "observation_residual_jacobians",
]
