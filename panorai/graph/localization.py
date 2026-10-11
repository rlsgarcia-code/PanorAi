"""Pairwise spatial-semantic posterior estimation.

This module localizes exactly one accepted region pair.  It neither creates an
entity identity nor requires a graph to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._models import (
    PairFeatureEvidence,
    RegionCorrespondenceCandidate,
    RelativePoseEdge,
    SemanticRegionNode,
    SpatialSemanticPosterior,
)
from .observations import deterministic_id


@dataclass(frozen=True, slots=True)
class PairwiseLocalizationConfig:
    min_triangulated_points: int = 2
    min_triangulation_angle_deg: float = 0.5
    max_reprojection_error_deg: float = 2.0
    voxel_size: float | None = None
    preset: str = "conservative-v1"

    def __post_init__(self) -> None:
        if self.preset != "conservative-v1":
            raise ValueError(
                "the only implemented localization preset is conservative-v1"
            )
        if self.min_triangulated_points < 1:
            raise ValueError("min_triangulated_points must be positive")
        if self.min_triangulation_angle_deg < 0 or self.max_reprojection_error_deg <= 0:
            raise ValueError("triangulation thresholds are invalid")
        if self.voxel_size is not None and self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")


@dataclass(frozen=True, slots=True)
class PairwiseLocalizationInput:
    candidate: RegionCorrespondenceCandidate
    region_a: SemanticRegionNode
    region_b: SemanticRegionNode
    feature_evidence: PairFeatureEvidence
    pose: RelativePoseEdge
    bounds_xyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.candidate.state != "accepted":
            raise ValueError("localization requires an accepted correspondence")
        if (self.candidate.region_id_a, self.candidate.region_id_b) != (
            self.region_a.region_id,
            self.region_b.region_id,
        ):
            raise ValueError("candidate and region pair do not match")
        if self.pose.feature_evidence_id != self.feature_evidence.evidence_id:
            raise ValueError(
                "pose and localization must reuse the same feature evidence"
            )
        if self.bounds_xyz is not None:
            bounds = np.array(self.bounds_xyz, dtype=np.float64, copy=True)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError("bounds_xyz must have finite shape (2, 3)")
            if np.any(bounds[1] <= bounds[0]):
                raise ValueError("bounds_xyz maximum must exceed minimum")
            bounds.setflags(write=False)
            object.__setattr__(self, "bounds_xyz", bounds)


@dataclass(frozen=True, slots=True)
class PairwiseLocalizationResult:
    posterior: SpatialSemanticPosterior
    triangulated_points: np.ndarray
    supporting_match_indices: np.ndarray
    parallax_deg: np.ndarray
    reprojection_error_deg: np.ndarray

    def __post_init__(self) -> None:
        points = np.array(self.triangulated_points, dtype=np.float64, copy=True)
        support = np.array(self.supporting_match_indices, dtype=np.int64, copy=True)
        parallax = np.array(self.parallax_deg, dtype=np.float64, copy=True)
        errors = np.array(self.reprojection_error_deg, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("triangulated_points must have shape (N, 3)")
        if not (len(points) == len(support) == len(parallax) == len(errors)):
            raise ValueError("localization detail arrays must align")
        for value in (points, support, parallax, errors):
            value.setflags(write=False)
        object.__setattr__(self, "triangulated_points", points)
        object.__setattr__(self, "supporting_match_indices", support)
        object.__setattr__(self, "parallax_deg", parallax)
        object.__setattr__(self, "reprojection_error_deg", errors)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _triangulate(
    evidence: PairwiseLocalizationInput,
    policy: PairwiseLocalizationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = evidence.pose.rotation_b_from_a
    scale = (
        1.0
        if evidence.pose.translation_scale is None
        else evidence.pose.translation_scale
    )
    translation = evidence.pose.translation_direction_b_from_a * scale
    camera_b_in_a = -rotation.T @ translation
    points: list[np.ndarray] = []
    support: list[int] = []
    parallaxes: list[float] = []
    errors: list[float] = []
    for match_index in evidence.candidate.inlier_match_indices:
        index = int(match_index)
        ray_a = evidence.feature_evidence.bearings_a[index]
        ray_b = evidence.feature_evidence.bearings_b[index]
        ray_b_in_a = rotation.T @ ray_b
        parallax = _angle_deg(ray_a, ray_b_in_a)
        if parallax < policy.min_triangulation_angle_deg:
            continue
        coefficients, *_ = np.linalg.lstsq(
            np.column_stack((ray_a, -ray_b_in_a)), camera_b_in_a, rcond=None
        )
        depth_a, depth_b = coefficients
        if depth_a <= 0 or depth_b <= 0:
            continue
        point_a = depth_a * ray_a
        point_from_b = camera_b_in_a + depth_b * ray_b_in_a
        point = 0.5 * (point_a + point_from_b)
        point_b = rotation @ point + translation
        if np.linalg.norm(point) <= 1e-12 or np.linalg.norm(point_b) <= 1e-12:
            continue
        reprojection = max(
            _angle_deg(point / np.linalg.norm(point), ray_a),
            _angle_deg(point_b / np.linalg.norm(point_b), ray_b),
        )
        if reprojection > policy.max_reprojection_error_deg:
            continue
        if evidence.bounds_xyz is not None and (
            np.any(point < evidence.bounds_xyz[0])
            or np.any(point > evidence.bounds_xyz[1])
        ):
            continue
        points.append(point)
        support.append(index)
        parallaxes.append(parallax)
        errors.append(reprojection)
    return (
        np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(support, dtype=np.int64),
        np.asarray(parallaxes, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _angular_result(
    evidence: PairwiseLocalizationInput,
    points: np.ndarray,
    support: np.ndarray,
    parallaxes: np.ndarray,
    errors: np.ndarray,
) -> PairwiseLocalizationResult:
    bearing = evidence.region_a.centroid_bearing
    posterior = SpatialSemanticPosterior(
        posterior_id=deterministic_id(
            "posterior", evidence.candidate.candidate_id, "angular"
        ),
        frame=evidence.pose.view_id_a,
        units="direction",
        mode="unbounded_angular",
        support_kind="angular",
        log_probabilities=np.asarray([0.0]),
        support=np.asarray([bearing]),
        map_position_xyz=None,
        centroid_xyz=None,
        covariance_xyz=None,
        bounding_box_xyz=None,
        bearing_xyz=bearing,
        confidence=0.0,
        source_evidence_ids=(
            evidence.feature_evidence.evidence_id,
            evidence.pose.edge_id,
        ),
        diagnostics={
            "reason": "insufficient-finite-support",
            "triangulated_count": len(points),
        },
    )
    return PairwiseLocalizationResult(posterior, points, support, parallaxes, errors)


def localize_region_pair(
    evidence: PairwiseLocalizationInput,
    config: PairwiseLocalizationConfig | None = None,
) -> PairwiseLocalizationResult:
    """Estimate an honest angular, scale-free, or metric posterior."""

    policy = config or PairwiseLocalizationConfig()
    if not evidence.feature_evidence.source.allowed_as_method_evidence:
        raise ValueError("evaluation_only feature evidence cannot enter localization")
    if not evidence.pose.source.allowed_as_method_evidence:
        raise ValueError("evaluation_only pose evidence cannot enter localization")
    if not evidence.pose.accepted or evidence.pose.degenerate:
        raise ValueError("localization fails closed for rejected or degenerate pose")
    points, support, parallaxes, errors = _triangulate(evidence, policy)
    if len(points) < policy.min_triangulated_points:
        return _angular_result(evidence, points, support, parallaxes, errors)

    error_scale = max(policy.max_reprojection_error_deg, 1e-12)
    logp = -errors / error_scale
    logp -= float(np.max(logp))
    if evidence.pose.translation_scale is None:
        weights = np.exp(logp)
        bearing = np.sum(points * weights[:, None], axis=0)
        bearing /= np.linalg.norm(bearing)
        posterior = SpatialSemanticPosterior(
            posterior_id=deterministic_id(
                "posterior", evidence.candidate.candidate_id, "scale-free"
            ),
            frame=evidence.pose.view_id_a,
            units="normalized_baseline",
            mode="scale_free",
            support_kind="ray",
            log_probabilities=logp,
            support=points,
            map_position_xyz=None,
            centroid_xyz=None,
            covariance_xyz=None,
            bounding_box_xyz=None,
            bearing_xyz=bearing,
            confidence=float(
                np.clip(math.exp(-float(np.median(errors)) / error_scale), 0.0, 1.0)
            ),
            source_evidence_ids=(
                evidence.feature_evidence.evidence_id,
                evidence.pose.edge_id,
            ),
            diagnostics={"triangulated_count": len(points), "preset": policy.preset},
        )
        return PairwiseLocalizationResult(
            posterior, points, support, parallaxes, errors
        )

    voxel_size = policy.voxel_size
    if voxel_size is None:
        spread = np.ptp(points, axis=0)
        voxel_size = max(
            float(np.max(spread)) / 32.0, evidence.pose.translation_scale * 1e-4
        )
    keys = np.round(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxels = unique_keys.astype(np.float64) * voxel_size
    voxel_logp = np.full(len(voxels), -np.inf, dtype=np.float64)
    for index in range(len(voxels)):
        values = logp[inverse == index]
        maximum = float(np.max(values))
        voxel_logp[index] = maximum + math.log(float(np.exp(values - maximum).sum()))
    voxel_logp -= float(np.max(voxel_logp))
    probabilities = np.exp(voxel_logp)
    probabilities /= probabilities.sum()
    map_position = voxels[int(np.argmax(voxel_logp))]
    centroid = np.sum(voxels * probabilities[:, None], axis=0)
    centered = voxels - centroid
    covariance = (centered * probabilities[:, None]).T @ centered
    bbox = np.stack((voxels.min(axis=0), voxels.max(axis=0)))
    posterior = SpatialSemanticPosterior(
        posterior_id=deterministic_id(
            "posterior", evidence.candidate.candidate_id, "metric"
        ),
        frame=evidence.pose.view_id_a,
        units=evidence.pose.units or "unknown",
        mode="metric",
        support_kind="sparse_voxel",
        log_probabilities=voxel_logp,
        support=voxels,
        map_position_xyz=map_position,
        centroid_xyz=centroid,
        covariance_xyz=covariance,
        bounding_box_xyz=bbox,
        bearing_xyz=centroid,
        confidence=float(
            np.clip(math.exp(-float(np.median(errors)) / error_scale), 0.0, 1.0)
        ),
        source_evidence_ids=(
            evidence.feature_evidence.evidence_id,
            evidence.pose.edge_id,
        ),
        diagnostics={
            "triangulated_count": len(points),
            "voxel_count": len(voxels),
            "voxel_size": float(voxel_size),
            "preset": policy.preset,
        },
    )
    return PairwiseLocalizationResult(posterior, points, support, parallaxes, errors)


__all__ = [
    "PairwiseLocalizationConfig",
    "PairwiseLocalizationInput",
    "PairwiseLocalizationResult",
    "localize_region_pair",
]
