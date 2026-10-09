"""Experimental metric bundle adjustment for sparse spherical landmarks.

The solver refines a sparse camera/landmark graph from panorama-frame unit
bearings.  Metric scale is supplied explicitly by a camera-baseline gauge;
optional monocular radial ranges are robust soft priors, never fixed 3D points.
It does not create or densify a depth map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from ._metric_landmark_solver import (
    BearingObservation as _BearingObservation,
    BundleOptions as _BundleOptions,
    CameraState as _CameraState,
    LandmarkState as _LandmarkState,
    MonocularRangePrior as _MonocularRangePrior,
    ObservationGraph as _ObservationGraph,
    ScaleGauge as _ScaleGauge,
    SchurBundleAdjuster as _SchurBundleAdjuster,
)

METRIC_SPHERICAL_LANDMARK_BA_INTERFACE = (
    "panorai-metric-spherical-landmark-ba/v1-experimental"
)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _readonly_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _unit_bearing(value: Any) -> np.ndarray:
    result = _readonly_vector(value, (3,), "bearing_camera")
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise ValueError("bearing_camera must be non-zero")
    normalized = np.array(result / norm, copy=True)
    normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True, slots=True)
class MetricSphericalCamera:
    """One metric camera pose using ``x_camera = R @ (X_world - C_world)``."""

    camera_id: str
    rotation_world_to_camera: np.ndarray
    center_world_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _identifier(self.camera_id, "camera_id"))
        rotation = _readonly_vector(
            self.rotation_world_to_camera, (3, 3), "rotation_world_to_camera"
        )
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation_world_to_camera must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_world_to_camera must have determinant +1")
        object.__setattr__(self, "rotation_world_to_camera", rotation)
        object.__setattr__(
            self,
            "center_world_m",
            _readonly_vector(self.center_world_m, (3,), "center_world_m"),
        )


@dataclass(frozen=True, slots=True)
class MetricSphericalLandmark:
    """Initial or refined metric landmark in the shared world frame."""

    landmark_id: str
    position_world_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "landmark_id", _identifier(self.landmark_id, "landmark_id")
        )
        object.__setattr__(
            self,
            "position_world_m",
            _readonly_vector(self.position_world_m, (3,), "position_world_m"),
        )


@dataclass(frozen=True, slots=True)
class MetricBearingObservation:
    """A unit bearing from one camera to one landmark."""

    camera_id: str
    landmark_id: str
    bearing_camera: np.ndarray
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _identifier(self.camera_id, "camera_id"))
        object.__setattr__(
            self, "landmark_id", _identifier(self.landmark_id, "landmark_id")
        )
        object.__setattr__(self, "bearing_camera", _unit_bearing(self.bearing_camera))
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must lie in (0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class MetricRadialRangePrior:
    """Soft monocular radial-range prior for one supported landmark."""

    camera_id: str
    landmark_id: str
    radial_range_m: float
    sigma_log_range: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _identifier(self.camera_id, "camera_id"))
        object.__setattr__(
            self, "landmark_id", _identifier(self.landmark_id, "landmark_id")
        )
        for name in ("radial_range_m", "sigma_log_range", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.confidence > 1.0:
            raise ValueError("confidence must not exceed 1")


@dataclass(frozen=True, slots=True)
class MetricScaleGauge:
    """Metric baseline that removes the reconstruction scale gauge."""

    reference_camera_id: str
    anchor_camera_id: str
    baseline_m: float
    sigma_m: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_camera_id",
            _identifier(self.reference_camera_id, "reference_camera_id"),
        )
        object.__setattr__(
            self,
            "anchor_camera_id",
            _identifier(self.anchor_camera_id, "anchor_camera_id"),
        )
        if self.reference_camera_id == self.anchor_camera_id:
            raise ValueError("the scale gauge requires two distinct cameras")
        for name in ("baseline_m", "sigma_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MetricLandmarkBAOptions:
    """Robust LM/Schur configuration for metric landmark refinement."""

    angular_sigma_deg: float = 0.25
    angular_huber_delta_deg: float = 1.0
    range_huber_delta_log: float = 0.10
    max_iterations: int = 20
    initial_damping: float = 1e-3
    minimum_damping: float = 1e-9
    maximum_damping: float = 1e9
    step_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        # The internal object owns the complete validation contract.
        _BundleOptions(
            angular_sigma_deg=self.angular_sigma_deg,
            angular_huber_delta_deg=self.angular_huber_delta_deg,
            depth_huber_delta_log=self.range_huber_delta_log,
            max_iterations=self.max_iterations,
            initial_damping=self.initial_damping,
            minimum_damping=self.minimum_damping,
            maximum_damping=self.maximum_damping,
            step_tolerance=self.step_tolerance,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricLandmarkBAReport:
    """Optimization diagnostics; accuracy is defined only on supplied landmarks."""

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricLandmarkBAResult:
    """Refined sparse metric reconstruction; this is not a dense depth map."""

    cameras: tuple[MetricSphericalCamera, ...]
    landmarks: tuple[MetricSphericalLandmark, ...]
    report: MetricLandmarkBAReport
    interface: str = METRIC_SPHERICAL_LANDMARK_BA_INTERFACE
    support: str = "supplied-landmarks-only"

    @property
    def camera_by_id(self) -> dict[str, MetricSphericalCamera]:
        return {item.camera_id: item for item in self.cameras}

    @property
    def landmark_by_id(self) -> dict[str, MetricSphericalLandmark]:
        return {item.landmark_id: item for item in self.landmarks}


def refine_metric_spherical_landmarks(
    cameras: Sequence[MetricSphericalCamera],
    landmarks: Sequence[MetricSphericalLandmark],
    observations: Sequence[MetricBearingObservation],
    *,
    radial_range_priors: Sequence[MetricRadialRangePrior] = (),
    fixed_camera_ids: Iterable[str],
    scale_gauge: MetricScaleGauge,
    options: MetricLandmarkBAOptions | None = None,
) -> MetricLandmarkBAResult:
    """Refine a sparse metric camera/landmark graph.

    Every landmark must have observations from at least two distinct cameras.
    ``fixed_camera_ids`` removes the pose gauge and must include the gauge's
    reference camera. The baseline fixes metric scale. Range values are radial
    distances from their named camera centre, not camera z-depth.

    Inputs are copied. The returned arrays are read-only, and no dense value is
    inferred for pixels that are not represented by an input landmark.
    """

    camera_items = tuple(cameras)
    landmark_items = tuple(landmarks)
    observation_items = tuple(observations)
    prior_items = tuple(radial_range_priors)
    if len(camera_items) < 2:
        raise ValueError("metric landmark BA requires at least two cameras")
    if not landmark_items:
        raise ValueError("metric landmark BA requires at least one landmark")
    if not observation_items:
        raise ValueError("metric landmark BA requires bearing observations")

    camera_ids = [item.camera_id for item in camera_items]
    landmark_ids = [item.landmark_id for item in landmark_items]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("camera_id values must be unique")
    if len(set(landmark_ids)) != len(landmark_ids):
        raise ValueError("landmark_id values must be unique")
    camera_set = set(camera_ids)
    landmark_set = set(landmark_ids)

    fixed = frozenset(
        _identifier(value, "fixed_camera_id") for value in fixed_camera_ids
    )
    if not fixed:
        raise ValueError("at least one camera must be fixed")
    if not fixed <= camera_set:
        raise ValueError("fixed_camera_ids contains an unknown camera")
    if scale_gauge.reference_camera_id not in fixed:
        raise ValueError("the scale-gauge reference camera must be fixed")
    if {
        scale_gauge.reference_camera_id,
        scale_gauge.anchor_camera_id,
    } - camera_set:
        raise ValueError("scale_gauge contains an unknown camera")

    grouped_observations: dict[str, list[MetricBearingObservation]] = {
        value: [] for value in landmark_ids
    }
    seen_observations: set[tuple[str, str]] = set()
    for item in observation_items:
        if item.camera_id not in camera_set or item.landmark_id not in landmark_set:
            raise ValueError("an observation refers to an unknown camera or landmark")
        key = (item.camera_id, item.landmark_id)
        if key in seen_observations:
            raise ValueError("duplicate camera-landmark observation")
        seen_observations.add(key)
        grouped_observations[item.landmark_id].append(item)

    priors_by_landmark: dict[str, MetricRadialRangePrior] = {}
    for item in prior_items:
        if item.camera_id not in camera_set or item.landmark_id not in landmark_set:
            raise ValueError("a radial-range prior refers to an unknown endpoint")
        if item.landmark_id in priors_by_landmark:
            raise ValueError("at most one radial-range prior is accepted per landmark")
        priors_by_landmark[item.landmark_id] = item

    graph = _ObservationGraph()
    for item in camera_items:
        graph.add_camera(
            _CameraState(
                item.camera_id,
                item.rotation_world_to_camera,
                item.center_world_m,
            )
        )
    landmarks_by_id = {item.landmark_id: item for item in landmark_items}
    for landmark_id in landmark_ids:
        source = landmarks_by_id[landmark_id]
        grouped = tuple(grouped_observations[landmark_id])
        internal_observations = tuple(
            _BearingObservation(
                item.camera_id,
                item.landmark_id,
                item.bearing_camera,
                item.confidence,
            )
            for item in grouped
        )
        prior = priors_by_landmark.get(landmark_id)
        internal_prior = (
            None
            if prior is None
            else _MonocularRangePrior(
                prior.camera_id,
                prior.landmark_id,
                prior.radial_range_m,
                prior.sigma_log_range,
                prior.confidence,
            )
        )
        graph.add_landmark(
            _LandmarkState(landmark_id, source.position_world_m),
            internal_observations,
            depth_prior=internal_prior,
        )

    settings = options or MetricLandmarkBAOptions()
    internal_report = _SchurBundleAdjuster(
        _BundleOptions(
            angular_sigma_deg=settings.angular_sigma_deg,
            angular_huber_delta_deg=settings.angular_huber_delta_deg,
            depth_huber_delta_log=settings.range_huber_delta_log,
            max_iterations=settings.max_iterations,
            initial_damping=settings.initial_damping,
            minimum_damping=settings.minimum_damping,
            maximum_damping=settings.maximum_damping,
            step_tolerance=settings.step_tolerance,
        )
    ).optimize(
        graph,
        fixed_camera_ids=fixed,
        scale_gauge=_ScaleGauge(
            scale_gauge.reference_camera_id,
            scale_gauge.anchor_camera_id,
            scale_gauge.baseline_m,
            scale_gauge.sigma_m,
        ),
    )

    result_cameras = tuple(
        MetricSphericalCamera(
            camera_id,
            graph.cameras[camera_id].rotation_world_to_camera,
            graph.cameras[camera_id].center_world,
        )
        for camera_id in camera_ids
    )
    result_landmarks = tuple(
        MetricSphericalLandmark(
            landmark_id,
            graph.landmarks[landmark_id].position_world,
        )
        for landmark_id in landmark_ids
    )
    report = MetricLandmarkBAReport(**asdict(internal_report))
    return MetricLandmarkBAResult(result_cameras, result_landmarks, report)


__all__ = [
    "METRIC_SPHERICAL_LANDMARK_BA_INTERFACE",
    "MetricBearingObservation",
    "MetricLandmarkBAOptions",
    "MetricLandmarkBAReport",
    "MetricLandmarkBAResult",
    "MetricRadialRangePrior",
    "MetricScaleGauge",
    "MetricSphericalCamera",
    "MetricSphericalLandmark",
    "refine_metric_spherical_landmarks",
]
