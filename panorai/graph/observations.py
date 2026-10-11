"""Explicit adapters from prepared PanorAi results to graph observations."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import numpy as np

from ._models import (
    EvidenceSource,
    PairFeatureEvidence,
    RelativePoseEdge,
    SemanticRegionNode,
)


def deterministic_id(kind: str, *parts: object) -> str:
    """Return a stable, content-derived identifier for graph records."""

    payload = json.dumps(
        [kind, *parts], sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"{kind}-{sha256(payload).hexdigest()[:24]}"


def feature_evidence_from_panorai(
    matches: Any,
    pose: Any,
    *,
    source: EvidenceSource,
    descriptor_type: str,
    descriptor_metric: str,
) -> PairFeatureEvidence:
    """Adapt public feature-match and relative-pose result objects.

    The adapter intentionally uses their public attributes and performs no
    matching or pose estimation.
    """

    view_a = str(matches.panorama_id_a)
    view_b = str(matches.panorama_id_b)
    evidence_id = deterministic_id(
        "feature-evidence",
        min(view_a, view_b),
        max(view_a, view_b),
        source.source_id,
        len(matches.feature_indices_a),
    )
    return PairFeatureEvidence(
        evidence_id=evidence_id,
        view_id_a=view_a,
        view_id_b=view_b,
        feature_indices_a=matches.feature_indices_a,
        feature_indices_b=matches.feature_indices_b,
        bearings_a=matches.bearings_a,
        bearings_b=matches.bearings_b,
        valid=matches.valid,
        pose_inlier_mask=pose.inlier_mask,
        pose_residuals_rad=pose.residuals_rad,
        descriptor_type=descriptor_type,
        descriptor_metric=descriptor_metric,
        source=source,
    )


def pose_edge_from_panorai(
    pose: Any,
    feature_evidence: PairFeatureEvidence,
    *,
    source: EvidenceSource,
    translation_scale: float | None = None,
    units: str | None = None,
) -> RelativePoseEdge:
    """Adapt a public relative-pose result without executing an estimator."""

    accepted = bool(getattr(getattr(pose, "quality_report", None), "accepted", False))
    edge_id = deterministic_id(
        "pose",
        min(feature_evidence.view_id_a, feature_evidence.view_id_b),
        max(feature_evidence.view_id_a, feature_evidence.view_id_b),
        source.source_id,
    )
    return RelativePoseEdge(
        edge_id=edge_id,
        view_id_a=feature_evidence.view_id_a,
        view_id_b=feature_evidence.view_id_b,
        rotation_b_from_a=pose.rotation,
        translation_direction_b_from_a=pose.translation_direction,
        accepted=accepted,
        degenerate=bool(pose.degenerate),
        source=source,
        feature_evidence_id=feature_evidence.evidence_id,
        translation_scale=translation_scale,
        units=units,
        failure_reasons=tuple(getattr(pose, "degeneracy_reasons", ())),
    )


def semantic_region_from_observation(
    observation: Any,
    *,
    vocabulary: str,
    source: EvidenceSource,
    embedding: np.ndarray | None = None,
    encoder_id: str | None = None,
    text_image_aligned: bool = False,
    correlation_group: str | None = None,
) -> SemanticRegionNode:
    """Adapt a feature-indexed semantic region observation."""

    bearing = observation.centroid_bearing
    if bearing is None:
        raise ValueError("graph regions require an explicit centroid_bearing")
    return SemanticRegionNode(
        region_id=observation.region_id,
        view_id=observation.view_id,
        class_id=observation.class_id,
        class_name=observation.class_name,
        vocabulary=vocabulary,
        semantic_score=observation.semantic_score,
        feature_indices=observation.feature_indices,
        membership_weights=observation.membership_weights,
        centroid_bearing=bearing,
        source=source,
        embedding=embedding,
        encoder_id=encoder_id,
        text_image_aligned=text_image_aligned,
        correlation_group=correlation_group,
    )


__all__ = [
    "deterministic_id",
    "feature_evidence_from_panorai",
    "pose_edge_from_panorai",
    "semantic_region_from_observation",
]
