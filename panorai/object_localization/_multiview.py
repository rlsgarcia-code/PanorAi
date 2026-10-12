"""CAM-guided spatial hypotheses over registered spherical tracks.

This module deliberately treats dense semantic maps as directional evidence,
not as object masks.  Geometry and correspondence identity come from a
successful :class:`panorai.reconstruction.SphericalReconstructionResult`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from numbers import Integral, Real
from itertools import combinations
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    erp_pixels_to_rays,
    gnomonic_rotation,
    rays_to_erp_pixels,
    rays_to_gnomonic_pixels,
)
from panorai.reconstruction import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE = "panorai-multiview-object-hypotheses/v1"

MultiviewHypothesisState = Literal["proposed", "ambiguous", "confirmed"]
MultiviewLocationMode = Literal["metric_3d", "scale_free_3d", "bearing_only"]
BearingTriangulationState = Literal["accepted", "degenerate", "rejected"]


class RegisteredPoseInconsistencyError(RuntimeError):
    """Registered poses fail the systematic bearing-track consistency gate.

    Local low-parallax or ill-conditioned tracks are reported as degeneracies
    and do not cause this exception. It is raised only when enough
    non-degenerate tracks exist and the configured fraction fails cheirality
    or angular reprojection. Structured counts let callers abstain and report
    ``pose_inconsistent`` without parsing an error string.
    """

    def __init__(
        self,
        *,
        input_track_count: int,
        nondegenerate_track_count: int,
        rejected_track_count: int,
        rejected_fraction: float,
        reason_counts: Mapping[str, int],
    ) -> None:
        self.input_track_count = int(input_track_count)
        self.nondegenerate_track_count = int(nondegenerate_track_count)
        self.rejected_track_count = int(rejected_track_count)
        self.rejected_fraction = float(rejected_fraction)
        self.reason_counts = MappingProxyType(dict(sorted(reason_counts.items())))
        super().__init__(
            "registered poses are inconsistent with bearing tracks; "
            f"rejected {self.rejected_track_count}/"
            f"{self.nondegenerate_track_count} non-degenerate tracks "
            f"({dict(self.reason_counts)})"
        )


def _readonly(
    value: np.ndarray, *, dtype: np.dtype | type, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ValueError("floating arrays must be finite")
    result.setflags(write=False)
    return result


def _unit_vector(value: np.ndarray) -> np.ndarray:
    result = _readonly(value, dtype=np.float64, shape=(3,))
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("bearing must have non-zero norm")
    normalized = np.array(result / norm, copy=True)
    normalized.setflags(write=False)
    return normalized


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{kind}-{digest}"


@dataclass(frozen=True, slots=True)
class SemanticConcept:
    """Canonical concept supported by one or more named semantic channels.

    ``proxy_weights`` names channels in :class:`RegisteredSemanticEvidence`.
    Weights are normalized on construction, while absent channels contribute
    zero evidence rather than silently redistributing their mass.
    """

    concept_id: str
    display_name: str
    proxy_weights: tuple[tuple[str, float], ...]
    vocabulary: str = "imagenet-1k"

    def __post_init__(self) -> None:
        if not self.concept_id.strip() or not self.display_name.strip():
            raise ValueError("concept_id and display_name must be non-empty")
        if not self.vocabulary.strip():
            raise ValueError("vocabulary must be non-empty")
        if not self.proxy_weights:
            raise ValueError("proxy_weights must not be empty")
        names: list[str] = []
        weights: list[float] = []
        for proxy_id, weight in self.proxy_weights:
            if not isinstance(proxy_id, str) or not proxy_id.strip():
                raise ValueError("proxy identifiers must be non-empty strings")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError("proxy weights must be real numbers")
            numeric = float(weight)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError("proxy weights must be finite and positive")
            names.append(proxy_id.strip())
            weights.append(numeric)
        if len(names) != len(set(names)):
            raise ValueError("a concept must not repeat proxy identifiers")
        total = sum(weights)
        object.__setattr__(
            self,
            "proxy_weights",
            tuple((name, weight / total) for name, weight in zip(names, weights)),
        )
        object.__setattr__(self, "concept_id", self.concept_id.strip())
        object.__setattr__(self, "display_name", self.display_name.strip())
        object.__setattr__(self, "vocabulary", self.vocabulary.strip())


@dataclass(frozen=True, slots=True)
class RegisteredSemanticEvidence:
    """Dense semantic channels for one registered canonical ERP.

    The class is intentionally NumPy-only.  ``from_dense`` accepts the public
    ``DenseSemanticEvidence`` contract by attributes without importing the
    optional deep-learning or segmentation namespaces.
    """

    view_id: str
    channel_ids: tuple[str, ...]
    scores: np.ndarray
    support: np.ndarray
    vocabulary: str
    source_id: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.view_id.strip() or not self.source_id.strip():
            raise ValueError("view_id and source_id must be non-empty")
        if not self.vocabulary.strip():
            raise ValueError("vocabulary must be non-empty")
        channels = tuple(str(item).strip() for item in self.channel_ids)
        if not channels or any(not item for item in channels):
            raise ValueError("channel_ids must contain non-empty names")
        if len(channels) != len(set(channels)):
            raise ValueError("channel_ids must be unique")
        scores = _readonly(self.scores, dtype=np.float32)
        support = _readonly(self.support, dtype=bool)
        if scores.ndim != 3 or support.shape != scores.shape[1:]:
            raise ValueError("scores must be CHW and support must match HW")
        if scores.shape[0] != len(channels):
            raise ValueError("channel_ids must align with score channels")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("semantic scores must lie within [0,1]")
        if not np.any(support):
            raise ValueError("support must contain at least one valid sample")
        if np.any(scores[:, ~support] != 0.0):
            raise ValueError("scores outside support must be zero")
        object.__setattr__(self, "view_id", self.view_id.strip())
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "vocabulary", self.vocabulary.strip())
        object.__setattr__(self, "channel_ids", channels)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def from_dense(
        cls,
        view_id: str,
        evidence: object,
        *,
        source_id: str | None = None,
    ) -> RegisteredSemanticEvidence:
        """Adapt a ``DenseSemanticEvidence``-compatible object lazily."""

        required = ("concept_ids", "scores", "support", "vocabulary")
        if any(not hasattr(evidence, name) for name in required):
            raise TypeError("evidence does not satisfy the dense semantic contract")
        provenance = getattr(evidence, "provenance", {})
        resolved_source = source_id or str(
            provenance.get("source_id", f"dense-semantic:{view_id}")
        )
        return cls(
            view_id=view_id,
            channel_ids=tuple(getattr(evidence, "concept_ids")),
            scores=np.asarray(getattr(evidence, "scores")),
            support=np.asarray(getattr(evidence, "support")),
            vocabulary=str(getattr(evidence, "vocabulary")),
            source_id=resolved_source,
            provenance=dict(provenance),
        )

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.scores.shape[1]), int(self.scores.shape[2])


@dataclass(frozen=True, slots=True)
class SemanticTrackObservation:
    """One concept score sampled at one registered track observation."""

    view_id: str
    track_id: str
    concept_id: str
    feature_index: int
    supported: bool
    concept_score: float
    proxy_scores: tuple[tuple[str, float], ...]
    bearing_xyz: np.ndarray
    source_id: str
    geometric_residual_deg: float | None = None

    def __post_init__(self) -> None:
        for name in ("view_id", "track_id", "concept_id", "source_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.feature_index, bool) or not isinstance(
            self.feature_index, Integral
        ):
            raise TypeError("feature_index must be an integer")
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        score = float(self.concept_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("concept_score must lie within [0,1]")
        for proxy_id, proxy_score in self.proxy_scores:
            if not proxy_id or not math.isfinite(proxy_score):
                raise ValueError("proxy scores require valid IDs and values")
            if not 0.0 <= proxy_score <= 1.0:
                raise ValueError("proxy scores must lie within [0,1]")
        object.__setattr__(self, "concept_score", score)
        object.__setattr__(self, "bearing_xyz", _unit_vector(self.bearing_xyz))
        if self.geometric_residual_deg is not None:
            residual = float(self.geometric_residual_deg)
            if not math.isfinite(residual) or residual < 0.0 or residual > 180.0:
                raise ValueError("geometric_residual_deg must lie within [0,180]")
            object.__setattr__(self, "geometric_residual_deg", residual)


@dataclass(frozen=True, slots=True)
class RegisteredBearingObservation:
    """One feature bearing in a panorama with a known registered pose.

    The bearing uses the panorama frame.  It carries no range and therefore
    cannot imply a 3-D point on its own.
    """

    view_id: str
    feature_index: int
    bearing_xyz: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.view_id.strip():
            raise ValueError("view_id must be non-empty")
        if isinstance(self.feature_index, bool) or not isinstance(
            self.feature_index, Integral
        ):
            raise TypeError("feature_index must be an integer")
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "view_id", self.view_id.strip())
        object.__setattr__(self, "bearing_xyz", _unit_vector(self.bearing_xyz))
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class RegisteredBearingTrack:
    """Cross-view correspondence track without an input 3-D point."""

    track_id: str
    observations: tuple[RegisteredBearingObservation, ...]

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        observations = tuple(self.observations)
        if len(observations) < 2:
            raise ValueError("a bearing track requires at least two observations")
        view_ids = tuple(item.view_id for item in observations)
        if len(view_ids) != len(set(view_ids)):
            raise ValueError(
                "a bearing track may contain at most one observation per view"
            )
        object.__setattr__(self, "track_id", self.track_id.strip())
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(observations, key=lambda item: item.view_id)),
        )


@dataclass(frozen=True, slots=True)
class BearingTrackTriangulation:
    """Auditable estimate of one latent track point from registered rays.

    ``point_xyz`` is an estimate in the pose world frame, never ground truth.
    ``normal_condition`` and the residual fields are quality diagnostics, not
    a calibrated covariance model.
    """

    track_id: str
    state: BearingTriangulationState
    view_ids: tuple[str, ...]
    point_xyz: np.ndarray | None
    maximum_pairwise_parallax_deg: float
    normal_condition: float
    positive_depth_fraction: float
    maximum_angular_residual_deg: float | None
    rms_ray_distance: float | None
    angular_residuals_deg: tuple[float, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        if self.state not in {"accepted", "degenerate", "rejected"}:
            raise ValueError("invalid triangulation state")
        if tuple(sorted(set(self.view_ids))) != self.view_ids or len(self.view_ids) < 2:
            raise ValueError("view_ids must contain at least two sorted unique IDs")
        point = (
            None
            if self.point_xyz is None
            else _readonly(self.point_xyz, dtype=np.float64, shape=(3,))
        )
        if self.state == "accepted" and point is None:
            raise ValueError("accepted triangulation requires point_xyz")
        if self.state != "accepted" and point is not None:
            raise ValueError("failed triangulation cannot expose point_xyz")
        for name in (
            "maximum_pairwise_parallax_deg",
            "positive_depth_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        condition = float(self.normal_condition)
        if math.isnan(condition) or condition < 0.0:
            raise ValueError("normal_condition must be non-negative or infinity")
        if not 0.0 <= self.positive_depth_fraction <= 1.0:
            raise ValueError("positive_depth_fraction must lie within [0,1]")
        for name in ("maximum_angular_residual_deg", "rms_ray_distance"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        residuals = tuple(float(value) for value in self.angular_residuals_deg)
        if residuals and len(residuals) != len(self.view_ids):
            raise ValueError("angular residuals must align with view_ids")
        if any(not math.isfinite(value) or value < 0.0 for value in residuals):
            raise ValueError("angular residuals must be finite and non-negative")
        object.__setattr__(self, "point_xyz", point)
        object.__setattr__(self, "angular_residuals_deg", residuals)

    @property
    def accepted(self) -> bool:
        return self.state == "accepted"


@dataclass(frozen=True, slots=True)
class MultiviewSpatialRegion:
    """Spatial support for one semantic hypothesis.

    ``extent_covariance_xyz`` describes the spread of supporting reconstructed
    tracks.  It is deliberately not called estimator uncertainty.
    """

    mode: MultiviewLocationMode
    frame: str
    units: str
    support_kind: Literal["angular", "sparse_points"]
    center_xyz: np.ndarray | None
    extent_covariance_xyz: np.ndarray | None
    bounding_box_xyz: np.ndarray | None
    extent_radius: float | None
    support_points_xyz: np.ndarray
    support_weights: np.ndarray
    bearing_xyz: np.ndarray | None = None
    bearing_view_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"metric_3d", "scale_free_3d", "bearing_only"}:
            raise ValueError("invalid multiview location mode")
        if self.support_kind not in {"angular", "sparse_points"}:
            raise ValueError("invalid support_kind")
        if not self.frame or not self.units:
            raise ValueError("frame and units must be non-empty")
        points = _readonly(self.support_points_xyz, dtype=np.float64)
        weights = _readonly(self.support_weights, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("support_points_xyz must have shape (N,3)")
        if weights.shape != (len(points),):
            raise ValueError("support_weights must align with support points")
        if np.any(weights < 0.0):
            raise ValueError("support_weights must be non-negative")
        center = (
            None
            if self.center_xyz is None
            else _readonly(self.center_xyz, dtype=np.float64, shape=(3,))
        )
        covariance = (
            None
            if self.extent_covariance_xyz is None
            else _readonly(self.extent_covariance_xyz, dtype=np.float64, shape=(3, 3))
        )
        bounds = (
            None
            if self.bounding_box_xyz is None
            else _readonly(self.bounding_box_xyz, dtype=np.float64, shape=(2, 3))
        )
        bearing = None if self.bearing_xyz is None else _unit_vector(self.bearing_xyz)
        if self.mode == "bearing_only":
            if self.support_kind != "angular" or bearing is None:
                raise ValueError("bearing-only regions require angular support")
            if self.bearing_view_id is None:
                raise ValueError("bearing-only regions require bearing_view_id")
            if any(
                item is not None
                for item in (center, covariance, bounds, self.extent_radius)
            ):
                raise ValueError("bearing-only regions cannot expose finite extent")
        else:
            if self.support_kind != "sparse_points" or len(points) == 0:
                raise ValueError("3D regions require sparse point support")
            if center is None or covariance is None or bounds is None:
                raise ValueError("3D regions require center, covariance, and bounds")
            if self.extent_radius is None or self.extent_radius < 0.0:
                raise ValueError("3D regions require a non-negative extent radius")
        object.__setattr__(self, "support_points_xyz", points)
        object.__setattr__(self, "support_weights", weights)
        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "extent_covariance_xyz", covariance)
        object.__setattr__(self, "bounding_box_xyz", bounds)
        object.__setattr__(self, "bearing_xyz", bearing)


@dataclass(frozen=True, slots=True)
class HypothesisGnomonicRegion:
    """One auditable gnomonic window for a hypothesis in a registered view.

    The window is centred on the weighted mean direction of the projected
    sparse 3D support.  Its field of view contains that support plus the
    configured angular margin, subject to the configured maximum FOV.  This is
    a crop proposal, not a segmentation mask or an estimator covariance.
    """

    view_id: str
    spec: GnomonicSpec
    center_bearing_xyz: np.ndarray
    angular_radius_deg: float
    support_bearings_xyz: np.ndarray
    track_feature_indices: tuple[tuple[str, int], ...]
    sparse_support_coverage: float
    semantic_evidence_available: bool
    semantic_supporting: bool
    extent_source: str = "projected-sparse-tracks-plus-margin"

    def __post_init__(self) -> None:
        if not self.view_id.strip():
            raise ValueError("view_id must be non-empty")
        if not isinstance(self.spec, GnomonicSpec):
            raise TypeError("spec must be a GnomonicSpec")
        bearings = _readonly(self.support_bearings_xyz, dtype=np.float64)
        if bearings.ndim != 2 or bearings.shape[1:] != (3,):
            raise ValueError("support_bearings_xyz must have shape (N,3)")
        norms = np.linalg.norm(bearings, axis=1)
        if len(bearings) and not np.allclose(norms, 1.0, atol=1e-8):
            raise ValueError("support bearings must be unit vectors")
        pairs = tuple(
            (str(track_id), int(index))
            for track_id, index in self.track_feature_indices
        )
        if any(not track_id or index < 0 for track_id, index in pairs):
            raise ValueError(
                "track feature references must be non-empty and non-negative"
            )
        if tuple(sorted(set(pairs))) != pairs:
            raise ValueError("track_feature_indices must be sorted and unique")
        for name in ("angular_radius_deg", "sparse_support_coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.angular_radius_deg <= 180.0:
            raise ValueError("angular_radius_deg must lie within [0,180]")
        if not 0.0 <= self.sparse_support_coverage <= 1.0:
            raise ValueError("sparse_support_coverage must lie within [0,1]")
        if not self.extent_source.strip():
            raise ValueError("extent_source must be non-empty")
        object.__setattr__(
            self, "center_bearing_xyz", _unit_vector(self.center_bearing_xyz)
        )
        object.__setattr__(self, "support_bearings_xyz", bearings)
        object.__setattr__(self, "track_feature_indices", pairs)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(track_id for track_id, _ in self.track_feature_indices)


@dataclass(frozen=True, slots=True)
class HypothesisViewEmbedding:
    """Optional externally computed descriptor for one hypothesis face."""

    view_id: str
    vector: np.ndarray
    source_id: str
    preprocessing_id: str | None = None

    def __post_init__(self) -> None:
        if not self.view_id.strip() or not self.source_id.strip():
            raise ValueError("view_id and source_id must be non-empty")
        if self.preprocessing_id is not None and not self.preprocessing_id.strip():
            raise ValueError("preprocessing_id must be non-empty when provided")
        vector = _readonly(self.vector, dtype=np.float64)
        if vector.ndim != 1 or len(vector) == 0:
            raise ValueError("embedding vector must be a non-empty 1D array")
        if float(np.linalg.norm(vector)) <= 1e-12:
            raise ValueError("embedding vector must have non-zero norm")
        object.__setattr__(self, "vector", vector)


@dataclass(frozen=True, slots=True)
class HypothesisViewConsistency:
    """Independent correspondence and optional appearance evidence for a pair."""

    view_id_a: str
    view_id_b: str
    shared_track_count: int
    union_track_count: int
    track_jaccard: float
    smaller_view_track_coverage: float
    matched_feature_pairs: tuple[tuple[int, int], ...]
    median_angular_reprojection_residual_deg: float | None = None
    reprojection_agreement: float | None = None
    embedding_cosine_similarity: float | None = None
    embedding_source_ids: tuple[str, str] | None = None
    embedding_preprocessing_id: str | None = None

    def __post_init__(self) -> None:
        if not self.view_id_a or not self.view_id_b or self.view_id_a >= self.view_id_b:
            raise ValueError("view IDs must be non-empty and strictly ordered")
        if self.shared_track_count < 0 or self.union_track_count < 0:
            raise ValueError("track counts must be non-negative")
        if self.shared_track_count > self.union_track_count:
            raise ValueError("shared_track_count cannot exceed union_track_count")
        for name in ("track_jaccard", "smaller_view_track_coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0,1]")
        if len(self.matched_feature_pairs) != self.shared_track_count:
            raise ValueError("matched feature pairs must align with shared tracks")
        if self.median_angular_reprojection_residual_deg is None:
            if self.reprojection_agreement is not None:
                raise ValueError(
                    "reprojection agreement requires a residual measurement"
                )
        else:
            residual = float(self.median_angular_reprojection_residual_deg)
            if not math.isfinite(residual) or residual < 0.0:
                raise ValueError("median reprojection residual must be non-negative")
            if self.reprojection_agreement is None:
                raise ValueError("reprojection residual requires an agreement score")
            agreement = float(self.reprojection_agreement)
            if not math.isfinite(agreement) or not 0.0 <= agreement <= 1.0:
                raise ValueError("reprojection agreement must lie within [0,1]")
        if self.embedding_cosine_similarity is None:
            if (
                self.embedding_source_ids is not None
                or self.embedding_preprocessing_id is not None
            ):
                raise ValueError(
                    "embedding provenance requires an embedding similarity"
                )
        else:
            similarity = float(self.embedding_cosine_similarity)
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise ValueError("embedding cosine similarity must lie within [-1,1]")
            if self.embedding_source_ids is None:
                raise ValueError("embedding similarity requires source IDs")


@dataclass(frozen=True, slots=True)
class MultiviewObjectHypothesis:
    hypothesis_id: str
    concept_id: str
    concept_name: str
    vocabulary: str
    state: MultiviewHypothesisState
    view_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    proxy_ids: tuple[str, ...]
    semantic_score: float
    spatial: MultiviewSpatialRegion
    observations: tuple[SemanticTrackObservation, ...] = ()
    view_regions: tuple[HypothesisGnomonicRegion, ...] = ()
    source_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "concept_id", "concept_name", "vocabulary"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.state not in {"proposed", "ambiguous", "confirmed"}:
            raise ValueError("invalid hypothesis state")
        if tuple(sorted(set(self.view_ids))) != self.view_ids:
            raise ValueError("view_ids must be sorted and unique")
        if tuple(sorted(set(self.track_ids))) != self.track_ids:
            raise ValueError("track_ids must be sorted and unique")
        score = float(self.semantic_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("semantic_score must lie within [0,1]")
        if self.state == "confirmed" and len(self.view_ids) < 2:
            raise ValueError("confirmed hypotheses require at least two views")
        region_view_ids = tuple(item.view_id for item in self.view_regions)
        if tuple(sorted(set(region_view_ids))) != region_view_ids:
            raise ValueError("view_regions must be sorted by unique view ID")
        object.__setattr__(self, "semantic_score", score)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class SphericalObjectHypothesisConfig:
    minimum_observation_score: float = 0.25
    minimum_track_score: float = 0.25
    minimum_independent_views: int = 2
    minimum_tracks_per_hypothesis: int = 2
    ambiguity_margin: float = 0.05
    maximum_ambiguous_track_fraction: float = 0.50
    maximum_track_residual_deg: float = 2.0
    minimum_triangulation_angle_deg: float = 0.5
    maximum_triangulation_reprojection_error_deg: float = 2.0
    maximum_triangulation_condition_number: float = 1.0e8
    minimum_positive_depth_fraction: float = 1.0
    minimum_tracks_for_pose_consistency_gate: int = 4
    maximum_pose_inconsistent_track_fraction: float = 0.25
    spatial_cluster_radius: float | None = None
    auto_cluster_radius_baseline_fraction: float = 0.25
    emit_bearing_only_fallbacks: bool = True
    metric_units: str = "m"
    gnomonic_output_shape_hw: tuple[int, int] = (1024, 1024)
    minimum_gnomonic_fov_deg: float = 70.0
    maximum_gnomonic_fov_deg: float = 150.0
    gnomonic_margin_deg: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_independent_views",
            "minimum_tracks_per_hypothesis",
            "minimum_tracks_for_pose_consistency_gate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_independent_views < 2:
            raise ValueError("minimum_independent_views must be at least two")
        for name in (
            "minimum_observation_score",
            "minimum_track_score",
            "ambiguity_margin",
            "maximum_ambiguous_track_fraction",
            "maximum_pose_inconsistent_track_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0,1]")
        for name in (
            "maximum_track_residual_deg",
            "minimum_triangulation_angle_deg",
            "maximum_triangulation_reprojection_error_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_triangulation_condition_number)
            or self.maximum_triangulation_condition_number < 1.0
        ):
            raise ValueError(
                "maximum_triangulation_condition_number must be finite and at least one"
            )
        if (
            not math.isfinite(self.minimum_positive_depth_fraction)
            or not 0.0 <= self.minimum_positive_depth_fraction <= 1.0
        ):
            raise ValueError("minimum_positive_depth_fraction must lie within [0,1]")
        if self.spatial_cluster_radius is not None and (
            not math.isfinite(self.spatial_cluster_radius)
            or self.spatial_cluster_radius <= 0.0
        ):
            raise ValueError("spatial_cluster_radius must be finite and positive")
        if (
            not math.isfinite(self.auto_cluster_radius_baseline_fraction)
            or self.auto_cluster_radius_baseline_fraction <= 0.0
        ):
            raise ValueError("auto cluster fraction must be finite and positive")
        if not self.metric_units.strip():
            raise ValueError("metric_units must be non-empty")
        shape = tuple(self.gnomonic_output_shape_hw)
        if len(shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in shape
        ):
            raise ValueError(
                "gnomonic_output_shape_hw must contain two positive integers"
            )
        for name in (
            "minimum_gnomonic_fov_deg",
            "maximum_gnomonic_fov_deg",
            "gnomonic_margin_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.minimum_gnomonic_fov_deg <= self.maximum_gnomonic_fov_deg < 180.0:
            raise ValueError(
                "gnomonic FOV bounds must satisfy minimum <= maximum < 180"
            )
        object.__setattr__(
            self, "gnomonic_output_shape_hw", (int(shape[0]), int(shape[1]))
        )


@dataclass(frozen=True, slots=True)
class MultiviewObjectHypothesisResult:
    hypotheses: tuple[MultiviewObjectHypothesis, ...]
    track_observations: tuple[SemanticTrackObservation, ...]
    diagnostics: Mapping[str, int | float | str]
    bearing_track_triangulations: tuple[BearingTrackTriangulation, ...] = ()
    interface: str = MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE
    stability: str = "experimental"

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        triangulations = tuple(self.bearing_track_triangulations)
        if tuple(sorted(item.track_id for item in triangulations)) != tuple(
            item.track_id for item in triangulations
        ):
            raise ValueError("bearing track triangulations must be sorted by track_id")
        if len({item.track_id for item in triangulations}) != len(triangulations):
            raise ValueError("bearing track triangulation IDs must be unique")
        object.__setattr__(self, "bearing_track_triangulations", triangulations)

    @property
    def confirmed(self) -> tuple[MultiviewObjectHypothesis, ...]:
        return tuple(item for item in self.hypotheses if item.state == "confirmed")


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip(
            np.dot(first, second)
            / (float(np.linalg.norm(first)) * float(np.linalg.norm(second))),
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def triangulate_registered_bearing_track(
    track: RegisteredBearingTrack,
    poses: tuple[SphericalCameraPose, ...],
    config: SphericalObjectHypothesisConfig | None = None,
) -> BearingTrackTriangulation:
    """Estimate one latent world point from registered panorama bearings.

    Poses follow ``x_v = R_v @ (X - C_v)``.  A panorama bearing is transformed
    to the world ray ``d_v = R_v.T @ b_v``.  The estimate minimizes weighted
    squared distance to those rays.  It is returned only when parallax,
    conditioning, cheirality, and angular reprojection gates all pass.
    """

    policy = config or SphericalObjectHypothesisConfig()
    pose_by_view = {item.panorama_id: item for item in poses}
    if len(pose_by_view) != len(poses):
        raise ValueError("registered pose IDs must be unique")
    missing = sorted({item.view_id for item in track.observations} - set(pose_by_view))
    if missing:
        raise ValueError(f"bearing track references unregistered views: {missing}")

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    weights: list[float] = []
    for observation in track.observations:
        pose = pose_by_view[observation.view_id]
        direction = np.asarray(pose.R, dtype=np.float64).T @ np.asarray(
            observation.bearing_xyz, dtype=np.float64
        )
        direction /= np.linalg.norm(direction)
        origins.append(np.asarray(pose.center, dtype=np.float64))
        directions.append(direction)
        weights.append(observation.weight)

    origin_array = np.stack(origins)
    direction_array = np.stack(directions)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= weight_array.sum()
    parallaxes = [
        _angle_deg(direction_array[first], direction_array[second])
        for first in range(len(direction_array))
        for second in range(first + 1, len(direction_array))
    ]
    maximum_parallax = max(parallaxes, default=0.0)

    identity = np.eye(3, dtype=np.float64)
    projectors = identity[None, :, :] - (
        direction_array[:, :, None] * direction_array[:, None, :]
    )
    normal = np.sum(projectors * weight_array[:, None, None], axis=0)
    right_hand_side = np.sum(
        np.einsum("nij,nj->ni", projectors, origin_array) * weight_array[:, None],
        axis=0,
    )
    singular_values = np.linalg.svd(normal, compute_uv=False)
    normal_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 1e-15
        else math.inf
    )
    estimate, *_ = np.linalg.lstsq(normal, right_hand_side, rcond=None)

    rays_from_origins = estimate[None, :] - origin_array
    depths = np.einsum("ni,ni->n", rays_from_origins, direction_array)
    positive_depth_fraction = float(np.mean(depths > 0.0))
    angular_residuals = tuple(
        _angle_deg(ray, direction) if np.linalg.norm(ray) > 1e-12 else 180.0
        for ray, direction in zip(rays_from_origins, direction_array)
    )
    maximum_angular_residual = max(angular_residuals)
    orthogonal = np.einsum("nij,nj->ni", projectors, rays_from_origins)
    rms_ray_distance = float(
        math.sqrt(np.sum(weight_array * np.sum(orthogonal * orthogonal, axis=1)))
    )

    degenerate_reasons: list[str] = []
    if maximum_parallax < policy.minimum_triangulation_angle_deg:
        degenerate_reasons.append("insufficient-parallax")
    if normal_condition > policy.maximum_triangulation_condition_number:
        degenerate_reasons.append("ill-conditioned-ray-intersection")
    if degenerate_reasons:
        state: BearingTriangulationState = "degenerate"
        reasons = degenerate_reasons
    else:
        rejection_reasons: list[str] = []
        if positive_depth_fraction < policy.minimum_positive_depth_fraction:
            rejection_reasons.append("insufficient-positive-depth")
        if (
            maximum_angular_residual
            > policy.maximum_triangulation_reprojection_error_deg
        ):
            rejection_reasons.append("excessive-angular-reprojection-error")
        state = "rejected" if rejection_reasons else "accepted"
        reasons = rejection_reasons

    return BearingTrackTriangulation(
        track_id=track.track_id,
        state=state,
        view_ids=tuple(item.view_id for item in track.observations),
        point_xyz=estimate if state == "accepted" else None,
        maximum_pairwise_parallax_deg=maximum_parallax,
        normal_condition=normal_condition,
        positive_depth_fraction=positive_depth_fraction,
        maximum_angular_residual_deg=maximum_angular_residual,
        rms_ray_distance=rms_ray_distance,
        angular_residuals_deg=angular_residuals,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class _TrackCandidate:
    track_id: str
    point_xyz: np.ndarray
    concept_id: str
    score: float
    view_ids: tuple[str, ...]
    observations: tuple[SemanticTrackObservation, ...]
    ambiguous: bool


def _sample_channels(
    evidence: RegisteredSemanticEvidence,
    bearing: np.ndarray,
    concept: SemanticConcept,
) -> tuple[bool, float, tuple[tuple[str, float], ...]]:
    projected = rays_to_erp_pixels(np.asarray(bearing), evidence.shape_hw)
    x, y = np.asarray(projected.pixels_xy, dtype=np.float64)
    height, width = evidence.shape_hw
    column = int(math.floor(x + 0.5)) % width
    row = min(max(int(math.floor(y + 0.5)), 0), height - 1)
    supported = bool(evidence.support[row, column])
    channels = {name: index for index, name in enumerate(evidence.channel_ids)}
    proxy_scores = tuple(
        (
            proxy_id,
            float(evidence.scores[channels[proxy_id], row, column])
            if supported and proxy_id in channels
            else 0.0,
        )
        for proxy_id, _ in concept.proxy_weights
    )
    weights = dict(concept.proxy_weights)
    score = sum(weights[proxy_id] * value for proxy_id, value in proxy_scores)
    return supported, float(score), proxy_scores


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(
        math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))
    )


def _auto_cluster_radius(
    reconstruction: SphericalReconstructionResult,
    config: SphericalObjectHypothesisConfig,
) -> float:
    if config.spatial_cluster_radius is not None:
        return float(config.spatial_cluster_radius)
    centers = [
        np.asarray(item.center, dtype=np.float64) for item in reconstruction.poses
    ]
    distances = [
        float(np.linalg.norm(centers[first] - centers[second]))
        for first in range(len(centers))
        for second in range(first + 1, len(centers))
        if np.linalg.norm(centers[first] - centers[second]) > 1e-12
    ]
    if distances:
        scale = float(np.median(distances))
    else:
        points = np.asarray(reconstruction.points_xyz, dtype=np.float64)
        scale = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 1.0
    return max(scale * config.auto_cluster_radius_baseline_fraction, 1e-9)


def _complete_link_clusters(
    candidates: list[_TrackCandidate], radius: float
) -> list[list[_TrackCandidate]]:
    ordered = sorted(candidates, key=lambda item: (-item.score, item.track_id))
    clusters: list[list[_TrackCandidate]] = []
    for candidate in ordered:
        accepted: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            maximum = max(
                float(np.linalg.norm(candidate.point_xyz - item.point_xyz))
                for item in cluster
            )
            if maximum <= radius:
                accepted.append((maximum, index))
        if accepted:
            _, index = min(accepted)
            clusters[index].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _spatial_region(
    cluster: list[_TrackCandidate],
    reconstruction: SphericalReconstructionResult,
    config: SphericalObjectHypothesisConfig,
) -> MultiviewSpatialRegion:
    points = np.stack([item.point_xyz for item in cluster])
    weights = np.asarray([item.score for item in cluster], dtype=np.float64)
    weights /= weights.sum()
    center = np.sum(points * weights[:, None], axis=0)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    bounds = np.stack((points.min(axis=0), points.max(axis=0)))
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    metric = reconstruction.scale.startswith("metric")
    return MultiviewSpatialRegion(
        mode="metric_3d" if metric else "scale_free_3d",
        frame=f"reconstruction-world:{reconstruction.reference_panorama_id}",
        units=config.metric_units if metric else "arbitrary-reconstruction-scale",
        support_kind="sparse_points",
        center_xyz=center,
        extent_covariance_xyz=covariance,
        bounding_box_xyz=bounds,
        extent_radius=radius,
        support_points_xyz=points,
        support_weights=weights,
    )


def _gnomonic_spec_for_bearings(
    bearings: np.ndarray,
    weights: np.ndarray,
    config: SphericalObjectHypothesisConfig,
) -> tuple[GnomonicSpec, np.ndarray, float, float]:
    vector = np.sum(bearings * weights[:, None], axis=0)
    if float(np.linalg.norm(vector)) <= 1e-12:
        vector = bearings[int(np.argmax(weights))]
    center = vector / np.linalg.norm(vector)
    longitude = math.degrees(math.atan2(float(center[0]), float(center[2])))
    latitude = math.degrees(math.asin(float(np.clip(center[1], -1.0, 1.0))))
    probe = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=config.maximum_gnomonic_fov_deg,
        vfov_deg=config.maximum_gnomonic_fov_deg,
        output_shape_hw=config.gnomonic_output_shape_hw,
    )
    rotation = np.asarray(gnomonic_rotation(probe), dtype=np.float64)
    local = bearings @ rotation
    positive = local[:, 2] > 1e-12
    horizontal_offsets = np.full(len(bearings), 90.0, dtype=np.float64)
    vertical_offsets = np.full(len(bearings), 90.0, dtype=np.float64)
    horizontal_offsets[positive] = np.degrees(
        np.abs(np.arctan2(local[positive, 0], local[positive, 2]))
    )
    vertical_offsets[positive] = np.degrees(
        np.abs(np.arctan2(local[positive, 1], local[positive, 2]))
    )
    horizontal_fov = min(
        max(
            config.minimum_gnomonic_fov_deg,
            2.0 * (float(horizontal_offsets.max()) + config.gnomonic_margin_deg),
        ),
        config.maximum_gnomonic_fov_deg,
    )
    vertical_fov = min(
        max(
            config.minimum_gnomonic_fov_deg,
            2.0 * (float(vertical_offsets.max()) + config.gnomonic_margin_deg),
        ),
        config.maximum_gnomonic_fov_deg,
    )
    spec = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=horizontal_fov,
        vfov_deg=vertical_fov,
        output_shape_hw=config.gnomonic_output_shape_hw,
    )
    angular_radius = float(
        np.degrees(np.max(np.arccos(np.clip(bearings @ center, -1.0, 1.0))))
    )
    valid = np.asarray(rays_to_gnomonic_pixels(bearings, spec).valid, dtype=bool)
    coverage = float(np.average(valid, weights=weights))
    return spec, center, angular_radius, coverage


def _gnomonic_regions(
    cluster: list[_TrackCandidate],
    reconstruction: SphericalReconstructionResult,
    config: SphericalObjectHypothesisConfig,
    *,
    semantic_evidence_view_ids: set[str],
    semantic_supporting_view_ids: set[str],
) -> tuple[HypothesisGnomonicRegion, ...]:
    points = np.stack([item.point_xyz for item in cluster])
    weights = np.asarray([item.score for item in cluster], dtype=np.float64)
    weights /= weights.sum()
    tracks_by_id = {item.track_id: item for item in reconstruction.tracks}
    regions: list[HypothesisGnomonicRegion] = []
    for pose in sorted(reconstruction.poses, key=lambda item: item.panorama_id):
        directions = (points - np.asarray(pose.center_world)) @ np.asarray(
            pose.rotation_world_to_panorama
        ).T
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            continue
        bearings = directions / norms[:, None]
        spec, center, angular_radius, coverage = _gnomonic_spec_for_bearings(
            bearings, weights, config
        )
        track_features: list[tuple[str, int]] = []
        for candidate in cluster:
            track = tracks_by_id[candidate.track_id]
            matching = [
                item
                for item in track.observations
                if item.active and item.panorama_id == pose.panorama_id
            ]
            if matching:
                track_features.append(
                    (candidate.track_id, int(matching[0].feature_index))
                )
        regions.append(
            HypothesisGnomonicRegion(
                view_id=pose.panorama_id,
                spec=spec,
                center_bearing_xyz=center,
                angular_radius_deg=angular_radius,
                support_bearings_xyz=bearings,
                track_feature_indices=tuple(sorted(track_features)),
                sparse_support_coverage=coverage,
                semantic_evidence_available=(
                    pose.panorama_id in semantic_evidence_view_ids
                ),
                semantic_supporting=(pose.panorama_id in semantic_supporting_view_ids),
            )
        )
    return tuple(regions)


def _concept_map(
    evidence: RegisteredSemanticEvidence, concept: SemanticConcept
) -> np.ndarray:
    channels = {name: index for index, name in enumerate(evidence.channel_ids)}
    result = np.zeros(evidence.shape_hw, dtype=np.float64)
    for proxy_id, weight in concept.proxy_weights:
        if proxy_id in channels:
            result += weight * evidence.scores[channels[proxy_id]]
    result[~evidence.support] = 0.0
    return result


def _angular_hypothesis(
    evidence: RegisteredSemanticEvidence,
    concept: SemanticConcept,
    config: SphericalObjectHypothesisConfig,
) -> MultiviewObjectHypothesis | None:
    values = _concept_map(evidence, concept)
    selected = evidence.support & (values >= config.minimum_observation_score)
    if not np.any(selected):
        return None
    rows, columns = np.nonzero(selected)
    height, width = evidence.shape_hw
    pixels = np.column_stack((columns.astype(np.float64), rows.astype(np.float64)))
    rays = np.asarray(erp_pixels_to_rays(pixels, evidence.shape_hw), dtype=np.float64)
    latitudes = math.pi / 2.0 - (rows + 0.5) * math.pi / height
    weights = values[rows, columns] * np.cos(latitudes)
    vector = np.sum(rays * weights[:, None], axis=0)
    if np.linalg.norm(vector) <= 1e-12:
        peak = np.unravel_index(int(np.argmax(values)), values.shape)
        vector = np.asarray(
            erp_pixels_to_rays(np.asarray((peak[1], peak[0])), evidence.shape_hw)
        )
    bearing = vector / np.linalg.norm(vector)
    score = float(values[selected].max())
    proxy_ids = tuple(proxy_id for proxy_id, _ in concept.proxy_weights)
    hypothesis_id = _stable_id(
        "angular-hypothesis", concept.concept_id, evidence.view_id
    )
    spatial = MultiviewSpatialRegion(
        mode="bearing_only",
        frame=evidence.view_id,
        units="direction",
        support_kind="angular",
        center_xyz=None,
        extent_covariance_xyz=None,
        bounding_box_xyz=None,
        extent_radius=None,
        support_points_xyz=np.empty((0, 3), dtype=np.float64),
        support_weights=np.empty(0, dtype=np.float64),
        bearing_xyz=bearing,
        bearing_view_id=evidence.view_id,
    )
    longitude = math.degrees(math.atan2(float(bearing[0]), float(bearing[2])))
    latitude = math.degrees(math.asin(float(np.clip(bearing[1], -1.0, 1.0))))
    view_region = HypothesisGnomonicRegion(
        view_id=evidence.view_id,
        spec=GnomonicSpec(
            center_lat_deg=latitude,
            center_lon_deg=longitude,
            hfov_deg=config.minimum_gnomonic_fov_deg,
            vfov_deg=config.minimum_gnomonic_fov_deg,
            output_shape_hw=config.gnomonic_output_shape_hw,
        ),
        center_bearing_xyz=bearing,
        angular_radius_deg=0.0,
        support_bearings_xyz=np.asarray(bearing)[None, :],
        track_feature_indices=(),
        sparse_support_coverage=1.0,
        semantic_evidence_available=True,
        semantic_supporting=True,
        extent_source="semantic-centroid-minimum-window",
    )
    return MultiviewObjectHypothesis(
        hypothesis_id=hypothesis_id,
        concept_id=concept.concept_id,
        concept_name=concept.display_name,
        vocabulary=concept.vocabulary,
        state="proposed",
        view_ids=(evidence.view_id,),
        track_ids=(),
        proxy_ids=proxy_ids,
        semantic_score=score,
        spatial=spatial,
        view_regions=(view_region,),
        source_ids=(evidence.source_id,),
        reasons=("insufficient-track-supported-multiview-evidence",),
        provenance={
            "interface": MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE,
            "geometry_convention": "panorai-geometry-v1",
            "spatial_point_source": "none-bearing-only",
            "concept": {
                "concept_id": concept.concept_id,
                "vocabulary": concept.vocabulary,
                "proxy_weights": concept.proxy_weights,
            },
            "semantic_sources": (
                {
                    "view_id": evidence.view_id,
                    "source_id": evidence.source_id,
                    "vocabulary": evidence.vocabulary,
                    "provenance": dict(evidence.provenance),
                },
            ),
            "generator_config": asdict(config),
            "reduction_trace": {
                "state_reasons": ("insufficient-track-supported-multiview-evidence",),
            },
        },
    )


class SphericalObjectHypothesisGenerator:
    """Generate conservative semantic hypotheses from registered tracks."""

    def __init__(self, config: SphericalObjectHypothesisConfig | None = None) -> None:
        self.config = config or SphericalObjectHypothesisConfig()

    def generate(
        self,
        reconstruction: SphericalReconstructionResult,
        semantic_evidence: tuple[RegisteredSemanticEvidence, ...],
        concepts: tuple[SemanticConcept, ...],
    ) -> MultiviewObjectHypothesisResult:
        reconstruction.require_success()
        if len(reconstruction.poses) < 2:
            raise ValueError(
                "multiview hypotheses require at least two registered poses"
            )
        if len(semantic_evidence) < 2:
            raise ValueError("semantic evidence from at least two views is required")
        if not concepts:
            raise ValueError("at least one semantic concept is required")
        evidence_by_view = {item.view_id: item for item in semantic_evidence}
        if len(evidence_by_view) != len(semantic_evidence):
            raise ValueError("semantic evidence view IDs must be unique")
        pose_ids = {item.panorama_id for item in reconstruction.poses}
        missing = sorted(set(evidence_by_view) - pose_ids)
        if missing:
            raise ValueError(
                f"semantic evidence references unregistered views: {missing}"
            )
        concept_by_id = {item.concept_id: item for item in concepts}
        if len(concept_by_id) != len(concepts):
            raise ValueError("concept IDs must be unique")
        for concept in concepts:
            incompatible = [
                item.view_id
                for item in semantic_evidence
                if item.vocabulary != concept.vocabulary
            ]
            if incompatible:
                raise ValueError(
                    f"concept {concept.concept_id!r} has incompatible evidence vocabulary"
                )

        observations: list[SemanticTrackObservation] = []
        candidates_by_track: dict[str, dict[str, _TrackCandidate]] = {}
        for track in sorted(reconstruction.tracks, key=lambda item: item.track_id):
            if not track.valid:
                continue
            per_concept: dict[str, list[SemanticTrackObservation]] = {
                concept.concept_id: [] for concept in concepts
            }
            for item in track.observations:
                if not item.active or item.panorama_id not in evidence_by_view:
                    continue
                if (
                    math.degrees(float(item.residual_rad))
                    > self.config.maximum_track_residual_deg
                ):
                    continue
                evidence = evidence_by_view[item.panorama_id]
                for concept in concepts:
                    supported, score, proxy_scores = _sample_channels(
                        evidence, item.bearing_xyz, concept
                    )
                    observation = SemanticTrackObservation(
                        view_id=item.panorama_id,
                        track_id=track.track_id,
                        concept_id=concept.concept_id,
                        feature_index=item.feature_index,
                        supported=supported,
                        concept_score=score,
                        proxy_scores=proxy_scores,
                        bearing_xyz=item.bearing_xyz,
                        source_id=evidence.source_id,
                        geometric_residual_deg=math.degrees(float(item.residual_rad)),
                    )
                    observations.append(observation)
                    per_concept[concept.concept_id].append(observation)

            scores: dict[str, float] = {}
            positive_views: dict[str, tuple[str, ...]] = {}
            for concept in concepts:
                valid = [
                    item for item in per_concept[concept.concept_id] if item.supported
                ]
                positive = [
                    item
                    for item in valid
                    if item.concept_score >= self.config.minimum_observation_score
                ]
                score = _geometric_mean([item.concept_score for item in positive])
                scores[concept.concept_id] = score
                positive_views[concept.concept_id] = tuple(
                    sorted({item.view_id for item in positive})
                )
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            top_score = ranked[0][1] if ranked else 0.0
            second_score = ranked[1][1] if len(ranked) > 1 else -math.inf
            ambiguous = bool(
                len(ranked) > 1
                and top_score >= self.config.minimum_track_score
                and top_score - second_score < self.config.ambiguity_margin
            )
            track_candidates: dict[str, _TrackCandidate] = {}
            for concept in concepts:
                score = scores[concept.concept_id]
                if score < self.config.minimum_track_score:
                    continue
                if score + self.config.ambiguity_margin < top_score:
                    continue
                selected_observations = tuple(per_concept[concept.concept_id])
                track_candidates[concept.concept_id] = _TrackCandidate(
                    track_id=track.track_id,
                    point_xyz=np.asarray(track.point_xyz, dtype=np.float64),
                    concept_id=concept.concept_id,
                    score=score,
                    view_ids=positive_views[concept.concept_id],
                    observations=selected_observations,
                    ambiguous=ambiguous,
                )
            if track_candidates:
                candidates_by_track[track.track_id] = track_candidates

        radius = _auto_cluster_radius(reconstruction, self.config)
        hypotheses: list[MultiviewObjectHypothesis] = []
        concepts_with_spatial_support: set[str] = set()
        for concept in sorted(concepts, key=lambda item: item.concept_id):
            candidates = [
                item[concept.concept_id]
                for item in candidates_by_track.values()
                if concept.concept_id in item
            ]
            for cluster in _complete_link_clusters(candidates, radius):
                concepts_with_spatial_support.add(concept.concept_id)
                view_ids = tuple(
                    sorted({view for item in cluster for view in item.view_ids})
                )
                track_ids = tuple(sorted(item.track_id for item in cluster))
                ambiguous_fraction = sum(item.ambiguous for item in cluster) / len(
                    cluster
                )
                reasons: list[str] = []
                if len(track_ids) < self.config.minimum_tracks_per_hypothesis:
                    reasons.append("too-few-semantic-tracks")
                if len(view_ids) < self.config.minimum_independent_views:
                    reasons.append("too-few-independent-semantic-views")
                if ambiguous_fraction > self.config.maximum_ambiguous_track_fraction:
                    reasons.append("competing-concept-ambiguity")
                if "competing-concept-ambiguity" in reasons:
                    state: MultiviewHypothesisState = "ambiguous"
                elif reasons:
                    state = "proposed"
                else:
                    state = "confirmed"
                cluster_observations = tuple(
                    sorted(
                        (
                            observation
                            for item in cluster
                            for observation in item.observations
                        ),
                        key=lambda item: (item.view_id, item.track_id, item.concept_id),
                    )
                )
                source_ids = tuple(
                    sorted({item.source_id for item in cluster_observations})
                )
                semantic_score = float(
                    np.average(
                        [item.score for item in cluster],
                        weights=[max(len(item.view_ids), 1) for item in cluster],
                    )
                )
                hypotheses.append(
                    MultiviewObjectHypothesis(
                        hypothesis_id=_stable_id(
                            "object-hypothesis", concept.concept_id, track_ids
                        ),
                        concept_id=concept.concept_id,
                        concept_name=concept.display_name,
                        vocabulary=concept.vocabulary,
                        state=state,
                        view_ids=view_ids,
                        track_ids=track_ids,
                        proxy_ids=tuple(
                            proxy_id for proxy_id, _ in concept.proxy_weights
                        ),
                        semantic_score=semantic_score,
                        spatial=_spatial_region(cluster, reconstruction, self.config),
                        observations=cluster_observations,
                        view_regions=_gnomonic_regions(
                            cluster,
                            reconstruction,
                            self.config,
                            semantic_evidence_view_ids=set(evidence_by_view),
                            semantic_supporting_view_ids=set(view_ids),
                        ),
                        source_ids=source_ids,
                        reasons=tuple(reasons),
                        provenance={
                            "interface": MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE,
                            "geometry_convention": "panorai-geometry-v1",
                            "spatial_point_source": ("refined-sparse-reconstruction"),
                            "reconstruction_scale": reconstruction.scale,
                            "concept": {
                                "concept_id": concept.concept_id,
                                "vocabulary": concept.vocabulary,
                                "proxy_weights": concept.proxy_weights,
                            },
                            "semantic_sources": tuple(
                                {
                                    "view_id": view_id,
                                    "source_id": evidence_by_view[view_id].source_id,
                                    "vocabulary": evidence_by_view[view_id].vocabulary,
                                    "provenance": dict(
                                        evidence_by_view[view_id].provenance
                                    ),
                                }
                                for view_id in view_ids
                            ),
                            "generator_config": asdict(self.config),
                            "reduction_trace": {
                                "cluster_radius": radius,
                                "ambiguity_fraction": ambiguous_fraction,
                                "state_reasons": tuple(reasons),
                            },
                        },
                    )
                )

        if self.config.emit_bearing_only_fallbacks:
            for concept in sorted(concepts, key=lambda item: item.concept_id):
                if concept.concept_id in concepts_with_spatial_support:
                    continue
                for evidence in sorted(
                    semantic_evidence, key=lambda item: item.view_id
                ):
                    fallback = _angular_hypothesis(evidence, concept, self.config)
                    if fallback is not None:
                        hypotheses.append(fallback)

        hypotheses.sort(key=lambda item: (item.concept_id, item.hypothesis_id))
        observations.sort(
            key=lambda item: (item.concept_id, item.track_id, item.view_id)
        )
        return MultiviewObjectHypothesisResult(
            hypotheses=tuple(hypotheses),
            track_observations=tuple(observations),
            diagnostics={
                "registered_view_count": len(reconstruction.poses),
                "semantic_view_count": len(semantic_evidence),
                "concept_count": len(concepts),
                "valid_track_count": sum(item.valid for item in reconstruction.tracks),
                "semantic_candidate_track_count": len(candidates_by_track),
                "hypothesis_count": len(hypotheses),
                "confirmed_hypothesis_count": sum(
                    item.state == "confirmed" for item in hypotheses
                ),
                "spatial_cluster_radius": radius,
                "reconstruction_scale": reconstruction.scale,
                "spatial_point_source": "refined-sparse-reconstruction",
            },
        )

    def generate_from_bearings(
        self,
        *,
        poses: tuple[SphericalCameraPose, ...],
        tracks: tuple[RegisteredBearingTrack, ...],
        semantic_evidence: tuple[RegisteredSemanticEvidence, ...],
        concepts: tuple[SemanticConcept, ...],
        scale: str = "arbitrary",
        reference_view_id: str | None = None,
    ) -> MultiviewObjectHypothesisResult:
        """Generate hypotheses when registered poses and tracks have no points.

        Track points are estimated internally and retained only after geometric
        quality gates pass.  A systematic set of non-degenerate failures is a
        pose-consistency error, not a collection of harmless angular fallbacks.
        """

        if len(poses) < 2:
            raise ValueError("multiview hypotheses require at least two poses")
        pose_ids = tuple(item.panorama_id for item in poses)
        if len(set(pose_ids)) != len(pose_ids):
            raise ValueError("registered pose IDs must be unique")
        if len({item.track_id for item in tracks}) != len(tracks):
            raise ValueError("bearing track IDs must be unique")
        if len(semantic_evidence) < 2:
            raise ValueError("semantic evidence from at least two views is required")
        if not concepts:
            raise ValueError("at least one semantic concept is required")
        evidence_by_view = {item.view_id: item for item in semantic_evidence}
        if len(evidence_by_view) != len(semantic_evidence):
            raise ValueError("semantic evidence view IDs must be unique")
        missing_evidence_poses = sorted(set(evidence_by_view) - set(pose_ids))
        if missing_evidence_poses:
            raise ValueError(
                "semantic evidence references unregistered views: "
                f"{missing_evidence_poses}"
            )
        if len({item.concept_id for item in concepts}) != len(concepts):
            raise ValueError("concept IDs must be unique")
        for concept in concepts:
            if any(item.vocabulary != concept.vocabulary for item in semantic_evidence):
                raise ValueError(
                    f"concept {concept.concept_id!r} has incompatible evidence vocabulary"
                )
        if not scale.strip():
            raise ValueError("scale must be non-empty")
        resolved_reference = reference_view_id or sorted(pose_ids)[0]
        if resolved_reference not in set(pose_ids):
            raise ValueError("reference_view_id must identify a registered pose")

        triangulations = tuple(
            triangulate_registered_bearing_track(track, poses, self.config)
            for track in sorted(tracks, key=lambda item: item.track_id)
        )
        nondegenerate = tuple(
            item for item in triangulations if item.state != "degenerate"
        )
        inconsistent = tuple(item for item in nondegenerate if item.state == "rejected")
        inconsistent_fraction = (
            len(inconsistent) / len(nondegenerate) if nondegenerate else 0.0
        )
        if (
            len(nondegenerate) >= self.config.minimum_tracks_for_pose_consistency_gate
            and inconsistent_fraction
            > self.config.maximum_pose_inconsistent_track_fraction
        ):
            reason_counts: dict[str, int] = {}
            for item in inconsistent:
                for reason in item.reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            raise RegisteredPoseInconsistencyError(
                input_track_count=len(tracks),
                nondegenerate_track_count=len(nondegenerate),
                rejected_track_count=len(inconsistent),
                rejected_fraction=inconsistent_fraction,
                reason_counts=reason_counts,
            )

        input_tracks = {item.track_id: item for item in tracks}
        accepted_tracks: list[SphericalTrack] = []
        accepted_points: list[np.ndarray] = []
        for triangulation in triangulations:
            if not triangulation.accepted:
                continue
            source = input_tracks[triangulation.track_id]
            residual_by_view = dict(
                zip(triangulation.view_ids, triangulation.angular_residuals_deg)
            )
            observations = tuple(
                SphericalTrackObservation(
                    panorama_id=item.view_id,
                    feature_index=item.feature_index,
                    bearing_xyz=item.bearing_xyz,
                    residual_rad=math.radians(residual_by_view[item.view_id]),
                )
                for item in source.observations
            )
            point = np.asarray(triangulation.point_xyz, dtype=np.float64)
            accepted_tracks.append(
                SphericalTrack(triangulation.track_id, point, observations)
            )
            accepted_points.append(point)

        if accepted_tracks:
            reconstruction = SphericalReconstructionResult(
                success=True,
                failure_reasons=(),
                poses=tuple(poses),
                tracks=tuple(accepted_tracks),
                pairwise_edges=(),
                admitted_edge_indices=(),
                reference_panorama_id=resolved_reference,
                diagnostics=SphericalReconstructionDiagnostics(
                    stage_messages=(
                        "object-localization:triangulated-from-registered-bearings",
                    )
                ),
                options=SphericalGlobalMapperOptions(),
                scale=scale,
                _points_xyz=np.stack(accepted_points),
            )
            generated = self.generate(reconstruction, semantic_evidence, concepts)
            diagnostics = dict(generated.diagnostics)
            diagnostics.update(
                {
                    "spatial_point_source": "triangulated-registered-bearings",
                    "input_bearing_track_count": len(tracks),
                    "accepted_bearing_track_count": len(accepted_tracks),
                    "degenerate_bearing_track_count": sum(
                        item.state == "degenerate" for item in triangulations
                    ),
                    "pose_inconsistent_bearing_track_count": len(inconsistent),
                    "pose_inconsistent_track_fraction": inconsistent_fraction,
                }
            )
            triangulation_by_id = {item.track_id: item for item in triangulations}
            generated_hypotheses = tuple(
                replace(
                    hypothesis,
                    provenance={
                        **dict(hypothesis.provenance),
                        "spatial_point_source": ("triangulated-registered-bearings"),
                        "bearing_triangulation": tuple(
                            {
                                "track_id": track_id,
                                "state": triangulation_by_id[track_id].state,
                                "maximum_pairwise_parallax_deg": (
                                    triangulation_by_id[
                                        track_id
                                    ].maximum_pairwise_parallax_deg
                                ),
                                "normal_condition": triangulation_by_id[
                                    track_id
                                ].normal_condition,
                                "maximum_angular_residual_deg": (
                                    triangulation_by_id[
                                        track_id
                                    ].maximum_angular_residual_deg
                                ),
                            }
                            for track_id in hypothesis.track_ids
                        ),
                    },
                )
                for hypothesis in generated.hypotheses
            )
            return MultiviewObjectHypothesisResult(
                hypotheses=generated_hypotheses,
                track_observations=generated.track_observations,
                diagnostics=diagnostics,
                bearing_track_triangulations=triangulations,
            )

        hypotheses: list[MultiviewObjectHypothesis] = []
        if self.config.emit_bearing_only_fallbacks:
            for concept in sorted(concepts, key=lambda item: item.concept_id):
                for evidence in sorted(
                    semantic_evidence, key=lambda item: item.view_id
                ):
                    fallback = _angular_hypothesis(evidence, concept, self.config)
                    if fallback is not None:
                        hypotheses.append(fallback)
        hypotheses.sort(key=lambda item: (item.concept_id, item.hypothesis_id))
        return MultiviewObjectHypothesisResult(
            hypotheses=tuple(hypotheses),
            track_observations=(),
            diagnostics={
                "registered_view_count": len(poses),
                "semantic_view_count": len(semantic_evidence),
                "concept_count": len(concepts),
                "input_bearing_track_count": len(tracks),
                "accepted_bearing_track_count": 0,
                "degenerate_bearing_track_count": sum(
                    item.state == "degenerate" for item in triangulations
                ),
                "pose_inconsistent_bearing_track_count": len(inconsistent),
                "pose_inconsistent_track_fraction": inconsistent_fraction,
                "hypothesis_count": len(hypotheses),
                "confirmed_hypothesis_count": 0,
                "reconstruction_scale": scale,
                "spatial_point_source": "none-bearing-only",
            },
            bearing_track_triangulations=triangulations,
        )

    def describe(self) -> dict[str, str]:
        return {
            "interface": MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE,
            "stability": "experimental",
            "semantic_role": "directional-evidence-not-segmentation",
            "geometry": "refined-sparse-points-or-registered-bearing-triangulation",
            "scope": "two-or-more-registered-views",
        }


def hypothesis_view_consistency(
    hypothesis: MultiviewObjectHypothesis,
    embeddings: tuple[HypothesisViewEmbedding, ...] = (),
    *,
    reprojection_sigma_deg: float = 1.0,
) -> tuple[HypothesisViewConsistency, ...]:
    """Compare every view pair using registered tracks and optional embeddings.

    Embeddings are intentionally caller-owned: PanorAi neither selects nor
    loads an encoder here.  This keeps the core result useful with local
    descriptors, learned face embeddings, or no appearance backend at all.
    """

    if not math.isfinite(reprojection_sigma_deg) or reprojection_sigma_deg <= 0.0:
        raise ValueError("reprojection_sigma_deg must be finite and positive")
    embedding_by_view = {item.view_id: item for item in embeddings}
    if len(embedding_by_view) != len(embeddings):
        raise ValueError("embedding view IDs must be unique")
    region_by_view = {item.view_id: item for item in hypothesis.view_regions}
    unknown = sorted(set(embedding_by_view) - set(region_by_view))
    if unknown:
        raise ValueError(f"embeddings reference unknown hypothesis views: {unknown}")
    residual_by_track_view = {
        (item.track_id, item.view_id): item.geometric_residual_deg
        for item in hypothesis.observations
        if item.geometric_residual_deg is not None
    }
    result: list[HypothesisViewConsistency] = []
    for region_a, region_b in combinations(hypothesis.view_regions, 2):
        features_a = dict(region_a.track_feature_indices)
        features_b = dict(region_b.track_feature_indices)
        tracks_a = set(features_a)
        tracks_b = set(features_b)
        shared = sorted(tracks_a & tracks_b)
        union = tracks_a | tracks_b
        smaller = min(len(tracks_a), len(tracks_b))
        embedding_a = embedding_by_view.get(region_a.view_id)
        embedding_b = embedding_by_view.get(region_b.view_id)
        similarity: float | None = None
        sources: tuple[str, str] | None = None
        preprocessing_id: str | None = None
        if embedding_a is not None and embedding_b is not None:
            if embedding_a.vector.shape != embedding_b.vector.shape:
                raise ValueError("paired embeddings must have the same shape")
            if embedding_a.preprocessing_id != embedding_b.preprocessing_id:
                raise ValueError(
                    "paired embeddings must share one preprocessing contract"
                )
            similarity = float(
                np.dot(embedding_a.vector, embedding_b.vector)
                / (
                    np.linalg.norm(embedding_a.vector)
                    * np.linalg.norm(embedding_b.vector)
                )
            )
            similarity = float(np.clip(similarity, -1.0, 1.0))
            sources = (embedding_a.source_id, embedding_b.source_id)
            preprocessing_id = embedding_a.preprocessing_id
        pair_residuals = [
            0.5
            * (
                residual_by_track_view[(track_id, region_a.view_id)]
                + residual_by_track_view[(track_id, region_b.view_id)]
            )
            for track_id in shared
            if (track_id, region_a.view_id) in residual_by_track_view
            and (track_id, region_b.view_id) in residual_by_track_view
        ]
        median_residual = float(np.median(pair_residuals)) if pair_residuals else None
        reprojection_agreement = (
            math.exp(-median_residual / reprojection_sigma_deg)
            if median_residual is not None
            else None
        )
        result.append(
            HypothesisViewConsistency(
                view_id_a=region_a.view_id,
                view_id_b=region_b.view_id,
                shared_track_count=len(shared),
                union_track_count=len(union),
                track_jaccard=(len(shared) / len(union) if union else 0.0),
                smaller_view_track_coverage=(len(shared) / smaller if smaller else 0.0),
                matched_feature_pairs=tuple(
                    (features_a[track_id], features_b[track_id]) for track_id in shared
                ),
                median_angular_reprojection_residual_deg=median_residual,
                reprojection_agreement=reprojection_agreement,
                embedding_cosine_similarity=similarity,
                embedding_source_ids=sources,
                embedding_preprocessing_id=preprocessing_id,
            )
        )
    return tuple(result)


__all__ = [
    "MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE",
    "BearingTrackTriangulation",
    "HypothesisGnomonicRegion",
    "HypothesisViewConsistency",
    "HypothesisViewEmbedding",
    "MultiviewObjectHypothesis",
    "MultiviewObjectHypothesisResult",
    "MultiviewSpatialRegion",
    "RegisteredBearingObservation",
    "RegisteredBearingTrack",
    "RegisteredSemanticEvidence",
    "SemanticConcept",
    "SemanticTrackObservation",
    "SphericalObjectHypothesisConfig",
    "SphericalObjectHypothesisGenerator",
    "hypothesis_view_consistency",
    "triangulate_registered_bearing_track",
]
