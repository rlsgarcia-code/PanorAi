"""Public models for the Experimental spherical visual SLAM facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from panorai.features import FeatureExtractorConfig, FeatureMatcherConfig
from panorai.reconstruction import (
    SphericalGlobalMapperOptions,
    SphericalReconstructionResult,
)


@dataclass(frozen=True, slots=True)
class SphericalSLAMOptions:
    """Front-end and global-back-end policy for a visual SLAM session."""

    temporal_window: int = 2
    min_features_per_frame: int = 64
    min_matches_per_pair: int = 16
    edge_margin_px: int = 12
    extractor: FeatureExtractorConfig = field(
        default_factory=lambda: FeatureExtractorConfig(
            method="sift", max_features=4096, edge_margin_px=0
        )
    )
    matcher: FeatureMatcherConfig = field(
        default_factory=lambda: FeatureMatcherConfig(
            method="flann", ratio_test=0.75, deduplicate_matches=False
        )
    )
    mapper: SphericalGlobalMapperOptions = field(
        default_factory=SphericalGlobalMapperOptions
    )

    def __post_init__(self) -> None:
        for name in (
            "temporal_window",
            "min_features_per_frame",
            "min_matches_per_pair",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.edge_margin_px, bool) or not isinstance(
            self.edge_margin_px, (int, np.integer)
        ):
            raise TypeError("edge_margin_px must be an integer")
        if self.edge_margin_px < 0:
            raise ValueError("edge_margin_px must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["extractor"] = self.extractor.to_dict()
        result["matcher"] = self.matcher.to_dict()
        result["mapper"] = self.mapper.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class SphericalImageFrame:
    """One timestamped calibrated-fisheye image supplied to SLAM."""

    frame_id: str
    timestamp_s: float
    image: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise TypeError("frame_id must be a non-empty string")
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        image = np.asarray(self.image)
        if image.ndim not in {2, 3} or (
            image.ndim == 3 and image.shape[2] not in {1, 3, 4}
        ):
            raise ValueError("image must use HW, HW1, HWC3, or HWC4 layout")
        if image.size == 0:
            raise ValueError("image cannot be empty")
        object.__setattr__(self, "frame_id", self.frame_id.strip())
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "image", image)


@dataclass(frozen=True, slots=True)
class SphericalSLAMFrameSummary:
    frame_id: str
    timestamp_s: float
    feature_count: int
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class SphericalSLAMPose:
    frame_id: str
    timestamp_s: float
    rotation_world_to_camera: np.ndarray
    center_world: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.array(self.rotation_world_to_camera, dtype=np.float64, copy=True)
        center = np.array(self.center_world, dtype=np.float64, copy=True)
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise ValueError("pose requires rotation (3,3) and center (3,)")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation_world_to_camera must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_world_to_camera must have determinant +1")
        rotation.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "rotation_world_to_camera", rotation)
        object.__setattr__(self, "center_world", center)

    @property
    def R(self) -> np.ndarray:
        return self.rotation_world_to_camera

    @property
    def center(self) -> np.ndarray:
        return self.center_world


@dataclass(frozen=True, slots=True)
class SphericalSLAMDiagnostics:
    input_frame_count: int
    keyframe_count: int
    dropped_frames: tuple[tuple[str, str], ...]
    candidate_pair_count: int
    retained_pair_count: int
    pair_match_counts: tuple[tuple[tuple[str, str], int], ...]
    backend_name: str
    backend_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalSLAMResult:
    """Immutable arbitrary-scale SLAM trajectory and sparse reconstruction."""

    success: bool
    failure_reasons: Sequence[str]
    poses: Sequence[SphericalSLAMPose]
    keyframes: Sequence[SphericalSLAMFrameSummary]
    reconstruction: SphericalReconstructionResult | None
    diagnostics: SphericalSLAMDiagnostics
    options: SphericalSLAMOptions
    interface: str = "panorai-spherical-slam/v1"
    scale: str = "arbitrary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        object.__setattr__(self, "poses", tuple(self.poses))
        object.__setattr__(self, "keyframes", tuple(self.keyframes))
        if self.success and (not self.poses or self.reconstruction is None):
            raise ValueError("successful SLAM result requires poses and reconstruction")
        if not self.success and self.poses:
            raise ValueError("unsuccessful SLAM result cannot expose poses")

    def pose(self, frame_id: str) -> SphericalSLAMPose:
        for pose in self.poses:
            if pose.frame_id == frame_id:
                return pose
        raise KeyError(frame_id)

    @property
    def centers_world(self) -> np.ndarray:
        if not self.poses:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack([pose.center_world for pose in self.poses])

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "success": self.success,
            "failure_reasons": self.failure_reasons,
            "scale": self.scale,
            "pose_count": len(self.poses),
            "keyframe_count": len(self.keyframes),
            "diagnostics": self.diagnostics.to_dict(),
            "options": self.options.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SphericalIncrementalSLAMOptions:
    """Policy for central-ERP incremental tracking and local mapping."""

    local_window_size: int = 6
    min_frame_features: int = 64
    min_pair_matches: int = 20
    min_map_correspondences: int = 8
    max_tracking_pose_candidates: int = 2
    keyframe_min_interval_s: float = 0.10
    keyframe_max_interval_s: float = 1.00
    keyframe_min_parallax_deg: float = 1.0
    keyframe_min_tracked_ratio: float = 0.65
    initial_baseline: float = 1.0
    min_triangulation_angle_deg: float = 0.5
    max_reprojection_error_deg: float = 2.0
    local_ba_loss: str = "cauchy"
    local_ba_loss_scale_deg: float = 1.0
    local_ba_max_nfev: int = 50
    local_ba_max_points: int = 512
    loop_min_keyframe_separation: int = 8
    loop_min_matches: int = 30
    max_loop_candidates: int = 5
    optimize_global_on_loop: bool = True
    global_refine_on_finish: bool = True
    edge_admission: str = "accepted"
    mapper: SphericalGlobalMapperOptions = field(
        default_factory=SphericalGlobalMapperOptions
    )

    def __post_init__(self) -> None:
        for name in (
            "local_window_size",
            "min_frame_features",
            "min_pair_matches",
            "min_map_correspondences",
            "max_tracking_pose_candidates",
            "local_ba_max_nfev",
            "local_ba_max_points",
            "loop_min_keyframe_separation",
            "loop_min_matches",
            "max_loop_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.local_window_size < 2:
            raise ValueError("local_window_size must be at least 2")
        for name in (
            "keyframe_min_interval_s",
            "keyframe_max_interval_s",
            "keyframe_min_parallax_deg",
            "initial_baseline",
            "min_triangulation_angle_deg",
            "max_reprojection_error_deg",
            "local_ba_loss_scale_deg",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.keyframe_max_interval_s < self.keyframe_min_interval_s:
            raise ValueError(
                "keyframe_max_interval_s must not be less than keyframe_min_interval_s"
            )
        if self.initial_baseline <= 0.0:
            raise ValueError("initial_baseline must be positive")
        if self.local_ba_loss_scale_deg <= 0.0:
            raise ValueError("local_ba_loss_scale_deg must be positive")
        ratio = self.keyframe_min_tracked_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float, np.number))
            or not math.isfinite(float(ratio))
            or not 0.0 <= float(ratio) <= 1.0
        ):
            raise ValueError("keyframe_min_tracked_ratio must be in [0, 1]")
        if self.local_ba_loss not in {"linear", "soft_l1", "huber", "cauchy"}:
            raise ValueError("local_ba_loss must be linear, soft_l1, huber, or cauchy")
        for name in ("optimize_global_on_loop", "global_refine_on_finish"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.edge_admission not in {"accepted", "successful"}:
            raise ValueError("edge_admission must be 'accepted' or 'successful'")
        if not isinstance(self.mapper, SphericalGlobalMapperOptions):
            raise TypeError("mapper must be SphericalGlobalMapperOptions")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mapper"] = self.mapper.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class SphericalMapObservation:
    """One keyframe bearing associated with a local-map point."""

    frame_id: str
    feature_index: int
    bearing_xyz: np.ndarray
    active: bool = True
    residual_deg: float = 0.0

    def __post_init__(self) -> None:
        bearing = np.array(self.bearing_xyz, dtype=np.float64, copy=True)
        if bearing.shape != (3,) or not np.isfinite(bearing).all():
            raise ValueError("bearing_xyz must be a finite vector with shape (3,)")
        norm = float(np.linalg.norm(bearing))
        if norm <= 0.0:
            raise ValueError("bearing_xyz must be non-zero")
        bearing /= norm
        bearing.setflags(write=False)
        object.__setattr__(self, "bearing_xyz", bearing)
        if isinstance(self.feature_index, bool) or int(self.feature_index) < 0:
            raise ValueError("feature_index must be a non-negative integer")
        if not math.isfinite(float(self.residual_deg)) or self.residual_deg < 0.0:
            raise ValueError("residual_deg must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SphericalMapPoint:
    """Immutable public snapshot of one arbitrary-scale local landmark."""

    point_id: str
    position_world: np.ndarray
    observations: tuple[SphericalMapObservation, ...]
    active: bool = True

    def __post_init__(self) -> None:
        point = np.array(self.position_world, dtype=np.float64, copy=True)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("position_world must be finite with shape (3,)")
        point.setflags(write=False)
        object.__setattr__(self, "position_world", point)
        object.__setattr__(self, "observations", tuple(self.observations))
        frame_ids = [item.frame_id for item in self.observations if item.active]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(
                "a map point may have at most one active observation per frame"
            )


@dataclass(frozen=True, slots=True)
class SphericalKeyframeSummary:
    frame_id: str
    timestamp_s: float
    pose: SphericalSLAMPose
    feature_count: int
    active_landmark_count: int


@dataclass(frozen=True, slots=True)
class SphericalLocalBAReport:
    keyframe_ids: tuple[str, ...]
    point_count: int
    observation_count: int
    variable_pose_count: int
    initial_cost: float
    final_cost: float
    removed_observation_count: int
    success: bool

    def __post_init__(self) -> None:
        if self.final_cost > self.initial_cost + 1e-10:
            raise ValueError("local BA final_cost must not exceed initial_cost")


@dataclass(frozen=True, slots=True)
class SphericalTrackingResult:
    """Online result returned immediately for one ERP input frame."""

    frame_id: str
    timestamp_s: float
    state: str
    pose: SphericalSLAMPose | None
    reference_frame_id: str | None
    is_keyframe: bool
    feature_count: int
    match_count: int
    inlier_count: int
    map_correspondence_count: int
    median_parallax_deg: float | None
    local_map_point_count: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        states = {"initializing", "tracking", "keyframe", "relocalized", "lost"}
        if self.state not in states:
            raise ValueError(f"state must be one of {sorted(states)}")
        if self.state in {"tracking", "keyframe", "relocalized"} and self.pose is None:
            raise ValueError("tracked states require a pose")
        if self.state == "lost" and self.pose is not None:
            raise ValueError("lost frames cannot expose a pose")
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class SphericalIncrementalSLAMDiagnostics:
    input_frame_count: int
    tracked_frame_count: int
    lost_frame_count: int
    keyframe_count: int
    map_point_count: int
    active_map_point_count: int
    relocalization_count: int
    loop_closure_count: int
    pairwise_edge_count: int
    local_ba_reports: tuple[SphericalLocalBAReport, ...]
    stage_messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalIncrementalSLAMResult:
    """Final snapshot of an incremental arbitrary-scale spherical map."""

    success: bool
    failure_reasons: tuple[str, ...]
    frames: tuple[SphericalTrackingResult, ...]
    keyframes: tuple[SphericalKeyframeSummary, ...]
    map_points: tuple[SphericalMapPoint, ...]
    diagnostics: SphericalIncrementalSLAMDiagnostics
    options: SphericalIncrementalSLAMOptions
    global_reconstruction: SphericalReconstructionResult | None = None
    interface: str = "panorai-spherical-incremental-slam/v1"
    scale: str = "arbitrary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "keyframes", tuple(self.keyframes))
        object.__setattr__(self, "map_points", tuple(self.map_points))
        if self.success and len(self.keyframes) < 2:
            raise ValueError("successful incremental SLAM needs at least two keyframes")

    @property
    def trajectory(self) -> tuple[SphericalSLAMPose, ...]:
        return tuple(item.pose for item in self.frames if item.pose is not None)

    def pose(self, frame_id: str) -> SphericalSLAMPose:
        for item in self.frames:
            if item.frame_id == frame_id and item.pose is not None:
                return item.pose
        raise KeyError(frame_id)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "success": self.success,
            "failure_reasons": self.failure_reasons,
            "scale": self.scale,
            "frame_count": len(self.frames),
            "trajectory_pose_count": len(self.trajectory),
            "keyframe_count": len(self.keyframes),
            "map_point_count": len(self.map_points),
            "global_refined": bool(
                self.global_reconstruction is not None
                and self.global_reconstruction.success
            ),
            "diagnostics": self.diagnostics.to_dict(),
            "options": self.options.to_dict(),
        }
