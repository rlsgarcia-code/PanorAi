"""Public data contracts for two-view semantic object localization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Literal

import numpy as np

from panorai.estimators import RelativePoseResult
from panorai.features import SphericalFeatureMatches, SphericalFeatureSet

OBJECT_LOCALIZATION_INTERFACE = "panorai-object-localization/v1"
OBJECT_LOCALIZATION_STABILITY = "experimental"

AssociationState = Literal["accepted", "ambiguous", "rejected"]
IdentityState = Literal["confirmed", "ambiguous"]
LocationState = Literal["localized", "view_only", "unobservable"]
LocationMode = Literal["metric_3d", "scale_free_3d", "bearing_only"]


def _readonly_vector(
    name: str, value: np.ndarray | None, width: int, *, finite: bool = True
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_matrix(
    name: str, value: np.ndarray | None, shape: tuple[int, int]
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_indices(name: str, value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.int64, copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SemanticQuery:
    """Text query resolved to explicit classes in a declared vocabulary.

    V1 does not guess how free text maps to classes.  A caller may use a
    classifier, synonym table, or UI selector, but must preserve both the
    original text and the resolved vocabulary IDs.
    """

    text: str
    class_ids: tuple[int, ...]
    vocabulary: str = "imagenet-1k"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be non-empty")
        if not isinstance(self.vocabulary, str) or not self.vocabulary.strip():
            raise ValueError("query vocabulary must be non-empty")
        if not self.class_ids:
            raise ValueError("query class_ids must contain at least one class")
        normalized: list[int] = []
        for value in self.class_ids:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("query class_ids must be non-negative integers")
            normalized.append(int(value))
        if len(set(normalized)) != len(normalized):
            raise ValueError("query class_ids must not contain duplicates")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "vocabulary", self.vocabulary.strip())
        object.__setattr__(self, "class_ids", tuple(normalized))


@dataclass(frozen=True, slots=True)
class SemanticRegionObservation:
    """One class-labelled region observed in one panorama.

    ``feature_indices`` index the corresponding
    :class:`panorai.features.SphericalFeatureSet`. The region keeps no
    descriptor copy, so matching provenance remains owned by the feature API.
    """

    region_id: str
    view_id: str
    class_id: int
    class_name: str
    semantic_score: float
    feature_indices: np.ndarray
    membership_weights: np.ndarray
    centroid_bearing: np.ndarray | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id or not self.view_id or not self.class_name:
            raise ValueError("region_id, view_id, and class_name must be non-empty")
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, Integral)
            or self.class_id < 0
        ):
            raise TypeError("class_id must be a non-negative integer")
        score = float(self.semantic_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("semantic_score must be finite and within [0, 1]")
        indices = _readonly_indices("feature_indices", self.feature_indices)
        weights = np.array(self.membership_weights, dtype=np.float64, copy=True)
        if weights.shape != indices.shape:
            raise ValueError("membership_weights must align with feature_indices")
        if not np.all(np.isfinite(weights)) or np.any((weights < 0) | (weights > 1)):
            raise ValueError("membership_weights must be finite and within [0, 1]")
        weights.setflags(write=False)
        centroid = _readonly_vector("centroid_bearing", self.centroid_bearing, 3)
        if centroid is not None:
            norm = float(np.linalg.norm(centroid))
            if norm <= 1e-12:
                raise ValueError("centroid_bearing must have non-zero norm")
            centroid = np.array(centroid / norm, copy=True)
            centroid.setflags(write=False)
        object.__setattr__(self, "semantic_score", score)
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "membership_weights", weights)
        object.__setattr__(self, "centroid_bearing", centroid)


@dataclass(frozen=True, slots=True)
class ObjectLocalizationConfig:
    """Conservative gates for the simple two-view v1 workflow."""

    min_region_matches: int = 3
    min_region_pose_inliers: int = 3
    max_median_pose_residual_deg: float = 1.5
    ambiguity_margin: float = 0.05
    require_accepted_pose: bool = True
    min_triangulated_points: int = 2
    min_triangulation_angle_deg: float = 0.5
    max_reprojection_error_deg: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "min_region_matches",
            "min_region_pose_inliers",
            "min_triangulated_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_median_pose_residual_deg",
            "ambiguity_margin",
            "min_triangulation_angle_deg",
            "max_reprojection_error_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PairObjectLocalizationInput:
    """All evidence needed by the two-view workflow.

    The pose convention is ``point_b = R @ point_a + t``.  The optional
    ``translation_scale`` multiplies the pose's unit translation direction.
    """

    view_id_a: str
    view_id_b: str
    query: SemanticQuery
    regions_a: tuple[SemanticRegionObservation, ...]
    regions_b: tuple[SemanticRegionObservation, ...]
    features_a: SphericalFeatureSet
    features_b: SphericalFeatureSet
    matches: SphericalFeatureMatches
    pose: RelativePoseResult
    translation_scale: float | None = None
    metric_units: str = "m"

    def __post_init__(self) -> None:
        if not self.view_id_a or not self.view_id_b or self.view_id_a == self.view_id_b:
            raise ValueError("view IDs must be non-empty and distinct")
        if not isinstance(self.query, SemanticQuery):
            raise TypeError("query must be a SemanticQuery")
        if self.features_a.panorama_id != self.view_id_a:
            raise ValueError("features_a.panorama_id must equal view_id_a")
        if self.features_b.panorama_id != self.view_id_b:
            raise ValueError("features_b.panorama_id must equal view_id_b")
        if self.matches.panorama_id_a != self.view_id_a:
            raise ValueError("matches.panorama_id_a must equal view_id_a")
        if self.matches.panorama_id_b != self.view_id_b:
            raise ValueError("matches.panorama_id_b must equal view_id_b")
        if len(self.matches) and (
            int(self.matches.feature_indices_a.max()) >= len(self.features_a)
            or int(self.matches.feature_indices_b.max()) >= len(self.features_b)
        ):
            raise ValueError("match feature index is outside its feature set")
        if len(self.pose.inlier_mask) != len(self.matches):
            raise ValueError("pose and match rows must align")
        if any(region.view_id != self.view_id_a for region in self.regions_a):
            raise ValueError("every regions_a observation must belong to view_id_a")
        if any(region.view_id != self.view_id_b for region in self.regions_b):
            raise ValueError("every regions_b observation must belong to view_id_b")
        if len({region.region_id for region in self.regions_a}) != len(self.regions_a):
            raise ValueError("region IDs must be unique within view A")
        if len({region.region_id for region in self.regions_b}) != len(self.regions_b):
            raise ValueError("region IDs must be unique within view B")
        for region, feature_count in (
            *((region, len(self.features_a)) for region in self.regions_a),
            *((region, len(self.features_b)) for region in self.regions_b),
        ):
            if (
                len(region.feature_indices)
                and int(region.feature_indices.max()) >= feature_count
            ):
                raise ValueError("region feature index is outside its feature set")
        if self.translation_scale is not None:
            scale = float(self.translation_scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("translation_scale must be finite and positive")
            object.__setattr__(self, "translation_scale", scale)
            if not self.metric_units:
                raise ValueError("metric_units must be non-empty")


@dataclass(frozen=True, slots=True)
class RegionAssociation:
    association_id: str
    region_id_a: str
    region_id_b: str
    class_id: int
    class_name: str
    match_indices: np.ndarray
    inlier_match_indices: np.ndarray
    match_count: int
    inlier_count: int
    semantic_score: float
    median_pose_residual_deg: float | None
    ranking_score: float
    state: AssociationState
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matches = _readonly_indices("match_indices", self.match_indices)
        inliers = _readonly_indices("inlier_match_indices", self.inlier_match_indices)
        if not set(inliers.tolist()).issubset(matches.tolist()):
            raise ValueError("inlier_match_indices must be a subset of match_indices")
        if self.match_count != len(matches) or self.inlier_count != len(inliers):
            raise ValueError("association counts must match their index arrays")
        if self.state not in {"accepted", "ambiguous", "rejected"}:
            raise ValueError("invalid association state")
        for name in ("semantic_score", "ranking_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.median_pose_residual_deg is not None and (
            not math.isfinite(self.median_pose_residual_deg)
            or self.median_pose_residual_deg < 0
        ):
            raise ValueError("median_pose_residual_deg must be finite and non-negative")
        object.__setattr__(self, "match_indices", matches)
        object.__setattr__(self, "inlier_match_indices", inliers)


@dataclass(frozen=True, slots=True)
class SpatialLocationHypothesis:
    state: LocationState
    mode: LocationMode
    frame: str
    units: str
    position_xyz: np.ndarray | None
    covariance_xyz: np.ndarray | None
    bearing_xyz: np.ndarray | None
    uncertainty_radius: float | None
    supporting_match_indices: np.ndarray
    triangulated_count: int
    median_parallax_deg: float | None
    median_reprojection_error_deg: float | None
    location_score: float

    def __post_init__(self) -> None:
        if self.state not in {"localized", "view_only", "unobservable"}:
            raise ValueError("invalid location state")
        if self.mode not in {"metric_3d", "scale_free_3d", "bearing_only"}:
            raise ValueError("invalid location mode")
        position = _readonly_vector("position_xyz", self.position_xyz, 3)
        covariance = _readonly_matrix("covariance_xyz", self.covariance_xyz, (3, 3))
        bearing = _readonly_vector("bearing_xyz", self.bearing_xyz, 3)
        support = _readonly_indices(
            "supporting_match_indices", self.supporting_match_indices
        )
        if bearing is not None:
            norm = float(np.linalg.norm(bearing))
            if norm <= 1e-12:
                raise ValueError("bearing_xyz must have non-zero norm")
            bearing = np.array(bearing / norm, copy=True)
            bearing.setflags(write=False)
        if self.state == "localized" and (position is None or covariance is None):
            raise ValueError("localized hypotheses require position and covariance")
        if self.state == "view_only" and bearing is None:
            raise ValueError("view_only hypotheses require a bearing")
        if self.triangulated_count != len(support):
            raise ValueError("triangulated_count must equal supporting match count")
        if not math.isfinite(self.location_score) or not 0 <= self.location_score <= 1:
            raise ValueError("location_score must be finite and within [0, 1]")
        if self.uncertainty_radius is not None and (
            not math.isfinite(self.uncertainty_radius) or self.uncertainty_radius < 0
        ):
            raise ValueError("uncertainty_radius must be finite and non-negative")
        object.__setattr__(self, "position_xyz", position)
        object.__setattr__(self, "covariance_xyz", covariance)
        object.__setattr__(self, "bearing_xyz", bearing)
        object.__setattr__(self, "supporting_match_indices", support)


@dataclass(frozen=True, slots=True)
class ObjectHypothesis:
    hypothesis_id: str
    class_id: int
    class_name: str
    identity_state: IdentityState
    view_ids: tuple[str, str]
    region_ids: tuple[str, str]
    association_id: str
    semantic_score: float
    identity_score: float
    location: SpatialLocationHypothesis

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.association_id or not self.class_name:
            raise ValueError(
                "hypothesis, association, and class identifiers are required"
            )
        if self.identity_state not in {"confirmed", "ambiguous"}:
            raise ValueError("invalid identity state")
        if len(self.view_ids) != 2 or len(self.region_ids) != 2:
            raise ValueError("object hypotheses require exactly two observations")
        for name in ("semantic_score", "identity_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class ObjectLocalizationResult:
    query: SemanticQuery
    hypotheses: tuple[ObjectHypothesis, ...]
    associations: tuple[RegionAssociation, ...]
    failure_reasons: tuple[str, ...] = ()
    interface: str = OBJECT_LOCALIZATION_INTERFACE
    stability: str = OBJECT_LOCALIZATION_STABILITY
    diagnostics: dict[str, int | float | str] = field(default_factory=dict)
