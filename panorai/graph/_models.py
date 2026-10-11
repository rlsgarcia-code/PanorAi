"""Import-light contracts for the Experimental spatial-semantic graph.

The graph package owns evidence bookkeeping and graph state.  It deliberately
does not import or execute perception frontends.  Arrays are copied and made
read-only at contract boundaries so replay cannot be changed by caller-side
mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Literal, Mapping

import numpy as np

GRAPH_INTERFACE = "panorai-spatial-semantic-graph/v1"
GRAPH_STABILITY = "experimental"

AccessRole = Literal["method_input", "oracle_diagnostic", "evaluation_only"]
HypothesisState = Literal[
    "proposed", "ambiguous", "confirmed", "dormant", "rejected", "superseded"
]
PosteriorMode = Literal["unbounded_angular", "scale_free", "metric"]
AssociationState = Literal["accepted", "ambiguous", "rejected"]


class EvidenceMode(str, Enum):
    """How evidence entered the graph contract."""

    ESTIMATED = "estimated"
    ORACLE = "oracle"
    EXTERNAL = "external"
    DISABLED = "disabled"


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _array(
    name: str,
    value: Any,
    *,
    dtype: Any = np.float64,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    finite: bool = True,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _unit_rows(name: str, value: Any) -> np.ndarray:
    result = _array(name, value, ndim=2)
    if result.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(result):
        norms = np.linalg.norm(result, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError(f"{name} rows must have non-zero norm")
        result = np.array(result / norms[:, None], copy=True)
        result.setflags(write=False)
    return result


def _unit_vector(name: str, value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    result = _array(name, value, shape=(3,))
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    result = np.array(result / norm, copy=True)
    result.setflags(write=False)
    return result


def _probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Provenance and access policy for one evidence-producing modality."""

    source_id: str
    modality: str
    mode: EvidenceMode
    method_id: str
    interface_version: str
    access_role: AccessRole = "method_input"
    frame: str | None = None
    units: str | None = None
    split: str | None = None
    calibration_id: str | None = None
    correlation_group: str | None = None
    model_id: str | None = None
    config_hash: str | None = None
    generating_commit: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_id", "modality", "method_id", "interface_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        try:
            mode = EvidenceMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"unsupported evidence mode: {self.mode!r}") from exc
        if self.access_role not in {
            "method_input",
            "oracle_diagnostic",
            "evaluation_only",
        }:
            raise ValueError("invalid evidence access_role")
        if mode is EvidenceMode.ORACLE and self.access_role != "oracle_diagnostic":
            raise ValueError("oracle evidence must declare oracle_diagnostic access")
        if mode is EvidenceMode.DISABLED and self.access_role == "method_input":
            raise ValueError("disabled evidence cannot be a method input")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def allowed_as_method_evidence(self) -> bool:
        return (
            self.mode is not EvidenceMode.DISABLED
            and self.access_role != "evaluation_only"
        )


@dataclass(frozen=True, slots=True)
class SphericalViewNode:
    view_id: str
    frame: str
    split: str
    source_checksum: str
    access_role: AccessRole = "method_input"
    timestamp: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("view_id", "frame", "split", "source_checksum"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.access_role not in {
            "method_input",
            "oracle_diagnostic",
            "evaluation_only",
        }:
            raise ValueError("invalid view access_role")
        if self.timestamp is not None and not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PairFeatureEvidence:
    """Feature correspondences reused from the pose-supporting match set."""

    evidence_id: str
    view_id_a: str
    view_id_b: str
    feature_indices_a: np.ndarray
    feature_indices_b: np.ndarray
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    valid: np.ndarray
    pose_inlier_mask: np.ndarray
    pose_residuals_rad: np.ndarray
    descriptor_type: str
    descriptor_metric: str
    source: EvidenceSource

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "view_id_a",
            "view_id_b",
            "descriptor_type",
            "descriptor_metric",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.view_id_a == self.view_id_b:
            raise ValueError("pair feature evidence requires distinct views")
        indices_a = _array(
            "feature_indices_a", self.feature_indices_a, dtype=np.int64, ndim=1
        )
        indices_b = _array(
            "feature_indices_b", self.feature_indices_b, dtype=np.int64, ndim=1
        )
        if np.any(indices_a < 0) or np.any(indices_b < 0):
            raise ValueError("feature indices must be non-negative")
        bearings_a = _unit_rows("bearings_a", self.bearings_a)
        bearings_b = _unit_rows("bearings_b", self.bearings_b)
        valid = _array("valid", self.valid, dtype=bool, ndim=1, finite=False)
        inliers = _array(
            "pose_inlier_mask", self.pose_inlier_mask, dtype=bool, ndim=1, finite=False
        )
        residuals = _array("pose_residuals_rad", self.pose_residuals_rad, ndim=1)
        count = len(indices_a)
        if any(
            len(item) != count
            for item in (indices_b, bearings_a, bearings_b, valid, inliers, residuals)
        ):
            raise ValueError("pair feature arrays must have the same leading dimension")
        for name, value in (
            ("feature_indices_a", indices_a),
            ("feature_indices_b", indices_b),
            ("bearings_a", bearings_a),
            ("bearings_b", bearings_b),
            ("valid", valid),
            ("pose_inlier_mask", inliers),
            ("pose_residuals_rad", residuals),
        ):
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return int(len(self.feature_indices_a))


@dataclass(frozen=True, slots=True)
class RelativePoseEdge:
    edge_id: str
    view_id_a: str
    view_id_b: str
    rotation_b_from_a: np.ndarray
    translation_direction_b_from_a: np.ndarray
    accepted: bool
    degenerate: bool
    source: EvidenceSource
    feature_evidence_id: str
    translation_scale: float | None = None
    units: str | None = None
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("edge_id", "view_id_a", "view_id_b", "feature_evidence_id"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.view_id_a == self.view_id_b:
            raise ValueError("relative pose requires distinct views")
        rotation = _array("rotation_b_from_a", self.rotation_b_from_a, shape=(3, 3))
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-8
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_b_from_a must be a proper rotation")
        translation = _unit_vector(
            "translation_direction_b_from_a", self.translation_direction_b_from_a
        )
        if self.translation_scale is not None:
            scale = float(self.translation_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("translation_scale must be finite and positive")
            if not self.units:
                raise ValueError("metric pose requires units")
            object.__setattr__(self, "translation_scale", scale)
        object.__setattr__(self, "rotation_b_from_a", rotation)
        object.__setattr__(self, "translation_direction_b_from_a", translation)


@dataclass(frozen=True, slots=True)
class SemanticRegionNode:
    region_id: str
    view_id: str
    class_id: int
    class_name: str
    vocabulary: str
    semantic_score: float
    feature_indices: np.ndarray
    membership_weights: np.ndarray
    centroid_bearing: np.ndarray
    source: EvidenceSource
    embedding: np.ndarray | None = None
    encoder_id: str | None = None
    text_image_aligned: bool = False
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        for name in ("region_id", "view_id", "class_name", "vocabulary"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ValueError("class_id must be a non-negative integer")
        indices = _array(
            "feature_indices", self.feature_indices, dtype=np.int64, ndim=1
        )
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError("feature_indices must be unique and non-negative")
        weights = _array("membership_weights", self.membership_weights, ndim=1)
        if weights.shape != indices.shape or np.any((weights < 0.0) | (weights > 1.0)):
            raise ValueError("membership_weights must align and lie within [0, 1]")
        bearing = _unit_vector("centroid_bearing", self.centroid_bearing)
        embedding = None
        if self.embedding is not None:
            embedding = _array("embedding", self.embedding, ndim=1)
            if len(embedding) == 0 or float(np.linalg.norm(embedding)) <= 1e-12:
                raise ValueError("embedding must have non-zero norm")
            if not self.encoder_id:
                raise ValueError("embedded regions require encoder_id")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self, "semantic_score", _probability("semantic_score", self.semantic_score)
        )
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "membership_weights", weights)
        object.__setattr__(self, "centroid_bearing", bearing)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True, slots=True)
class RegionCorrespondenceCandidate:
    candidate_id: str
    region_id_a: str
    region_id_b: str
    view_id_a: str
    view_id_b: str
    class_id: int
    class_name: str
    match_indices: np.ndarray
    inlier_match_indices: np.ndarray
    semantic_score: float
    geometric_score: float
    ranking_score: float
    median_pose_residual_rad: float | None
    state: AssociationState
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "region_id_a",
            "region_id_b",
            "view_id_a",
            "view_id_b",
            "class_name",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        matches = _array("match_indices", self.match_indices, dtype=np.int64, ndim=1)
        inliers = _array(
            "inlier_match_indices", self.inlier_match_indices, dtype=np.int64, ndim=1
        )
        if np.any(matches < 0) or np.any(inliers < 0):
            raise ValueError("match indices must be non-negative")
        if not set(inliers.tolist()).issubset(matches.tolist()):
            raise ValueError("inlier match indices must be a subset of matches")
        if self.state not in {"accepted", "ambiguous", "rejected"}:
            raise ValueError("invalid correspondence state")
        object.__setattr__(self, "match_indices", matches)
        object.__setattr__(self, "inlier_match_indices", inliers)
        for name in ("semantic_score", "geometric_score", "ranking_score"):
            object.__setattr__(self, name, _probability(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RegionCorrespondenceEdge:
    edge_id: str
    candidate_id: str
    region_id_a: str
    region_id_b: str
    confidence: float
    source_evidence_ids: tuple[str, ...]
    correlation_groups: tuple[str, ...] = ()
    polarity: Literal["positive", "negative"] = "positive"

    def __post_init__(self) -> None:
        for name in ("edge_id", "candidate_id", "region_id_a", "region_id_b"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(
            self, "confidence", _probability("confidence", self.confidence)
        )
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("correspondence polarity must be positive or negative")


@dataclass(frozen=True, slots=True)
class SpatialSemanticPosterior:
    posterior_id: str
    frame: str
    units: str
    mode: PosteriorMode
    support_kind: Literal["angular", "ray", "sparse_voxel"]
    log_probabilities: np.ndarray
    support: np.ndarray
    map_position_xyz: np.ndarray | None
    centroid_xyz: np.ndarray | None
    covariance_xyz: np.ndarray | None
    bounding_box_xyz: np.ndarray | None
    bearing_xyz: np.ndarray | None
    confidence: float
    source_evidence_ids: tuple[str, ...]
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("posterior_id", "frame", "units"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.mode not in {"unbounded_angular", "scale_free", "metric"}:
            raise ValueError("invalid posterior mode")
        if self.support_kind not in {"angular", "ray", "sparse_voxel"}:
            raise ValueError("invalid support_kind")
        logp = _array("log_probabilities", self.log_probabilities, ndim=1)
        support = _array("support", self.support, ndim=2)
        if len(logp) != len(support) or support.shape[1:] != (3,):
            raise ValueError(
                "support must have shape (N, 3) and align with probabilities"
            )
        if len(logp) == 0:
            raise ValueError("posterior support cannot be empty")
        map_position = (
            None
            if self.map_position_xyz is None
            else _array("map_position_xyz", self.map_position_xyz, shape=(3,))
        )
        centroid = (
            None
            if self.centroid_xyz is None
            else _array("centroid_xyz", self.centroid_xyz, shape=(3,))
        )
        covariance = (
            None
            if self.covariance_xyz is None
            else _array("covariance_xyz", self.covariance_xyz, shape=(3, 3))
        )
        bbox = (
            None
            if self.bounding_box_xyz is None
            else _array("bounding_box_xyz", self.bounding_box_xyz, shape=(2, 3))
        )
        bearing = _unit_vector("bearing_xyz", self.bearing_xyz)
        if self.support_kind != "sparse_voxel" and any(
            item is not None for item in (map_position, centroid, covariance, bbox)
        ):
            raise ValueError("finite spatial summaries require sparse_voxel support")
        object.__setattr__(self, "log_probabilities", logp)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "map_position_xyz", map_position)
        object.__setattr__(self, "centroid_xyz", centroid)
        object.__setattr__(self, "covariance_xyz", covariance)
        object.__setattr__(self, "bounding_box_xyz", bbox)
        object.__setattr__(self, "bearing_xyz", bearing)
        object.__setattr__(
            self, "confidence", _probability("confidence", self.confidence)
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ObjectHypothesisNode:
    hypothesis_id: str
    class_id: int
    class_name: str
    state: HypothesisState
    region_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    correspondence_edge_ids: tuple[str, ...]
    posterior_ids: tuple[str, ...]
    confidence: float
    lineage_parent_ids: tuple[str, ...] = ()
    superseded_by_ids: tuple[str, ...] = ()
    independent_view_count: int = 0

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "class_name"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.state not in {
            "proposed",
            "ambiguous",
            "confirmed",
            "dormant",
            "rejected",
            "superseded",
        }:
            raise ValueError("invalid hypothesis state")
        if len(set(self.region_ids)) != len(self.region_ids) or len(
            set(self.view_ids)
        ) != len(self.view_ids):
            raise ValueError("hypothesis observations must be unique")
        if self.independent_view_count != len(set(self.view_ids)):
            raise ValueError("independent_view_count must equal unique view count")
        if self.state == "confirmed" and self.independent_view_count < 2:
            raise ValueError("a single view cannot confirm an entity")
        object.__setattr__(
            self, "confidence", _probability("confidence", self.confidence)
        )


@dataclass(frozen=True, slots=True)
class SpatialSemanticGraph:
    graph_id: str
    views: tuple[SphericalViewNode, ...]
    feature_evidence: tuple[PairFeatureEvidence, ...]
    pose_edges: tuple[RelativePoseEdge, ...]
    regions: tuple[SemanticRegionNode, ...]
    correspondence_candidates: tuple[RegionCorrespondenceCandidate, ...]
    correspondence_edges: tuple[RegionCorrespondenceEdge, ...]
    posteriors: tuple[SpatialSemanticPosterior, ...]
    hypotheses: tuple[ObjectHypothesisNode, ...]
    event_ids: tuple[str, ...]
    interface: str = GRAPH_INTERFACE
    stability: str = GRAPH_STABILITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _nonempty("graph_id", self.graph_id))

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "graph_id": self.graph_id,
            "views": len(self.views),
            "regions": len(self.regions),
            "correspondences": len(self.correspondence_edges),
            "hypotheses": len(self.hypotheses),
            "events": len(self.event_ids),
        }
