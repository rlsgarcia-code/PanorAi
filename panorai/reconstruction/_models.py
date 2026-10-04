"""Public result objects for Experimental spherical reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np


_INTERFACE = "panorai-spherical-reconstruction/v1"


def _readonly(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SphericalGlobalMapperOptions:
    """Numerical and admission policy for :class:`SphericalGlobalMapper`."""

    edge_admission: str = "accepted"
    min_panoramas: int = 3
    min_track_length: int = 2
    min_active_tracks_per_panorama: int = 3
    rotation_max_error_deg: float = 10.0
    rotation_robust_scale_deg: float = 5.0
    rotation_max_iterations: int = 100
    translation_max_error_deg: float = 20.0
    translation_consistency_rounds: int = 4
    translation_min_positive_depth_ratio: float = 0.55
    bearing_position_irls_steps: int = 3
    bearing_position_anchor_trials: int = 8
    require_multiview_corroboration: bool = True
    multiview_corroboration_min_track_length: int = 3
    multiview_corroboration_max_position_error_deg: float = 15.0
    max_reprojection_error_deg: float = 2.0
    min_triangulation_angle_deg: float = 0.5
    bundle_loss: str = "cauchy"
    bundle_loss_scale_deg: float = 1.0
    bundle_max_nfev: int = 100
    max_refinement_rounds: int = 3
    bundle_compute_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.edge_admission not in {"accepted", "successful"}:
            raise ValueError("edge_admission must be 'accepted' or 'successful'")
        for name in (
            "min_panoramas",
            "min_track_length",
            "min_active_tracks_per_panorama",
            "rotation_max_iterations",
            "translation_consistency_rounds",
            "bearing_position_irls_steps",
            "bearing_position_anchor_trials",
            "multiview_corroboration_min_track_length",
            "bundle_max_nfev",
            "max_refinement_rounds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_panoramas < 3:
            raise ValueError("min_panoramas must be at least 3")
        if self.min_track_length < 2:
            raise ValueError("min_track_length must be at least 2")
        if self.multiview_corroboration_min_track_length < 3:
            raise ValueError(
                "multiview_corroboration_min_track_length must be at least 3"
            )
        if not isinstance(self.require_multiview_corroboration, bool):
            raise TypeError("require_multiview_corroboration must be boolean")
        for name in (
            "rotation_max_error_deg",
            "rotation_robust_scale_deg",
            "translation_max_error_deg",
            "multiview_corroboration_max_position_error_deg",
            "max_reprojection_error_deg",
            "bundle_loss_scale_deg",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        value = self.min_triangulation_angle_deg
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 180.0
        ):
            raise ValueError("min_triangulation_angle_deg must be finite in [0, 180)")
        value = self.translation_min_positive_depth_ratio
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                "translation_min_positive_depth_ratio must be finite in [0, 1]"
            )
        if self.bundle_loss not in {"linear", "soft_l1", "huber", "cauchy"}:
            raise ValueError("bundle_loss must be linear, soft_l1, huber, or cauchy")
        if not isinstance(self.bundle_compute_backend, str):
            raise TypeError("bundle_compute_backend must be a string")
        if self.bundle_compute_backend not in {"auto", "numpy", "native"}:
            raise ValueError(
                "bundle_compute_backend must be 'auto', 'numpy', or 'native'"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalPairwisePoseEdge:
    """A match set and its panorama-B-from-panorama-A pose estimate."""

    matches: Any
    pose: Any

    def __post_init__(self) -> None:
        match_fields = (
            "panorama_id_a",
            "panorama_id_b",
            "feature_indices_a",
            "feature_indices_b",
            "bearings_a",
            "bearings_b",
            "descriptor_distances",
            "valid",
        )
        pose_fields = (
            "R",
            "t",
            "inlier_mask",
            "num_inliers",
            "quality_report",
        )
        if not all(hasattr(self.matches, name) for name in match_fields):
            raise TypeError("matches does not satisfy the spherical-match contract")
        if not all(hasattr(self.pose, name) for name in pose_fields):
            raise TypeError("pose does not satisfy the relative-pose contract")
        if self.panorama_id_a == self.panorama_id_b:
            raise ValueError("a pairwise edge must connect distinct panoramas")
        count = len(np.asarray(self.matches.feature_indices_a))
        if np.asarray(self.pose.inlier_mask).shape != (count,):
            raise ValueError("pose inlier_mask must align with the match rows")
        if int(self.pose.num_inliers) != int(np.asarray(self.pose.inlier_mask).sum()):
            raise ValueError("pose num_inliers must equal its inlier-mask count")

    @property
    def panorama_id_a(self) -> str:
        return str(self.matches.panorama_id_a)

    @property
    def panorama_id_b(self) -> str:
        return str(self.matches.panorama_id_b)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.panorama_id_a, self.panorama_id_b)

    @property
    def accepted(self) -> bool:
        return bool(self.pose.quality_report.accepted)


@dataclass(frozen=True, slots=True)
class SphericalCameraPose:
    panorama_id: str
    rotation_world_to_panorama: np.ndarray
    center_world: np.ndarray

    def __post_init__(self) -> None:
        rotation = _readonly(
            self.rotation_world_to_panorama, (3, 3), "rotation_world_to_panorama"
        )
        center = _readonly(self.center_world, (3,), "center_world")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation_world_to_panorama must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_world_to_panorama must have determinant +1")
        object.__setattr__(self, "rotation_world_to_panorama", rotation)
        object.__setattr__(self, "center_world", center)

    @property
    def R(self) -> np.ndarray:
        return self.rotation_world_to_panorama

    @property
    def center(self) -> np.ndarray:
        return self.center_world


@dataclass(frozen=True, slots=True)
class SphericalTrackObservation:
    panorama_id: str
    feature_index: int
    bearing_xyz: np.ndarray
    residual_rad: float
    active: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        bearing = _readonly(self.bearing_xyz, (3,), "bearing_xyz")
        norm = float(np.linalg.norm(bearing))
        if norm <= 0.0:
            raise ValueError("bearing_xyz must be non-zero")
        bearing = np.array(bearing / norm, copy=True)
        bearing.setflags(write=False)
        object.__setattr__(self, "bearing_xyz", bearing)
        if not math.isfinite(float(self.residual_rad)) and self.active:
            raise ValueError("active observation residual_rad must be finite")


@dataclass(frozen=True, slots=True)
class SphericalTrack:
    track_id: str
    point_xyz: np.ndarray
    observations: tuple[SphericalTrackObservation, ...]
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "point_xyz", _readonly(self.point_xyz, (3,), "point_xyz")
        )
        if len(self.observations) < 2:
            raise ValueError("a public track must retain at least two observations")
        ids = [item.panorama_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("a track may contain at most one feature per panorama")


@dataclass(frozen=True, slots=True)
class SphericalReconstructionDiagnostics:
    input_edge_count: int = 0
    successful_edge_count: int = 0
    admitted_edge_count: int = 0
    rejected_edge_pairs: tuple[tuple[str, str], ...] = ()
    rotation_filtered_pairs: tuple[tuple[str, str], ...] = ()
    translation_filtered_pairs: tuple[tuple[str, str], ...] = ()
    translation_flipped_pairs: tuple[tuple[str, str], ...] = ()
    translation_axis_errors_deg: tuple[tuple[tuple[str, str], float], ...] = ()
    translation_positive_depth_ratio: float | None = None
    bearing_position_anchors_tested: int = 0
    bearing_position_min_camera_positive_depth_ratio: float | None = None
    excluded_panoramas: tuple[str, ...] = ()
    candidate_match_count: int = 0
    track_count: int = 0
    track_conflict_count: int = 0
    filtered_observation_count: int = 0
    rotation_initial_cost: float | None = None
    rotation_final_cost: float | None = None
    position_initial_cost: float | None = None
    position_final_cost: float | None = None
    bundle_costs: tuple[tuple[str, float, float], ...] = ()
    reprojection_median_deg: float | None = None
    reprojection_p90_deg: float | None = None
    reprojection_max_deg: float | None = None
    camera_max_reprojection_deg: tuple[tuple[str, float], ...] = ()
    camera_active_track_counts: tuple[tuple[str, int], ...] = ()
    multiview_corroboration_passed: bool | None = None
    multiview_corroboration_track_count: int = 0
    multiview_corroboration_position_p90_deg: float | None = None
    multiview_corroboration_failure_reasons: tuple[str, ...] = ()
    scale_anchor: str | None = None
    ba_scale_anchor: str | None = None
    stage_messages: tuple[str, ...] = ()
    bundle_compute_backend: str = "numpy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalReconstructionResult:
    success: bool
    failure_reasons: Sequence[str]
    poses: Sequence[SphericalCameraPose]
    tracks: Sequence[SphericalTrack]
    pairwise_edges: Sequence[SphericalPairwisePoseEdge]
    admitted_edge_indices: Sequence[int]
    reference_panorama_id: str | None
    diagnostics: SphericalReconstructionDiagnostics
    options: SphericalGlobalMapperOptions
    interface: str = _INTERFACE
    scale: str = "arbitrary"
    _points_xyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64), repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        object.__setattr__(self, "poses", tuple(self.poses))
        object.__setattr__(self, "tracks", tuple(self.tracks))
        object.__setattr__(self, "pairwise_edges", tuple(self.pairwise_edges))
        object.__setattr__(
            self, "admitted_edge_indices", tuple(self.admitted_edge_indices)
        )
        points = np.array(self._points_xyz, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("points_xyz must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_xyz must be finite")
        if self.success and (not self.poses or not self.tracks):
            raise ValueError("a successful result must contain poses and tracks")
        if not self.success and (self.poses or self.tracks or len(points)):
            raise ValueError("an unsuccessful result cannot expose partial geometry")
        points.setflags(write=False)
        object.__setattr__(self, "_points_xyz", points)

    @property
    def points_xyz(self) -> np.ndarray:
        return self._points_xyz

    def pose(self, panorama_id: str) -> SphericalCameraPose:
        for item in self.poses:
            if item.panorama_id == panorama_id:
                return item
        raise KeyError(panorama_id)

    def require_success(self) -> "SphericalReconstructionResult":
        if not self.success:
            reasons = ", ".join(self.failure_reasons) or "unspecified failure"
            raise RuntimeError(f"spherical reconstruction failed: {reasons}")
        return self

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "success": self.success,
            "failure_reasons": self.failure_reasons,
            "panorama_count": len(self.poses),
            "point_count": int(len(self._points_xyz)),
            "track_count": len(self.tracks),
            "reference_panorama_id": self.reference_panorama_id,
            "pose_convention": "x_panorama=R_world_to_panorama@(X-C_world)",
            "scale": self.scale,
            "diagnostics": self.diagnostics.to_dict(),
            "options": self.options.to_dict(),
        }
