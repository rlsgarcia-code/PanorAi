"""Two-view semantic-region association over reusable pose evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._models import (
    PairFeatureEvidence,
    RegionCorrespondenceCandidate,
    RegionCorrespondenceEdge,
    RelativePoseEdge,
    SemanticRegionNode,
)
from .observations import deterministic_id


@dataclass(frozen=True, slots=True)
class RegionAssociationConfig:
    """Versioned conservative policy for the two-view association baseline."""

    min_region_matches: int = 3
    min_region_pose_inliers: int = 3
    max_median_pose_residual_deg: float = 1.5
    ambiguity_margin: float = 0.05
    require_accepted_pose: bool = True
    preset: str = "conservative-v1"

    def __post_init__(self) -> None:
        if self.preset != "conservative-v1":
            raise ValueError(
                "the only implemented association preset is conservative-v1"
            )
        if self.min_region_matches < 1 or self.min_region_pose_inliers < 1:
            raise ValueError("match thresholds must be positive")
        if self.max_median_pose_residual_deg < 0 or self.ambiguity_margin < 0:
            raise ValueError("residual and ambiguity thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class RegionAssociationInput:
    regions_a: tuple[SemanticRegionNode, ...]
    regions_b: tuple[SemanticRegionNode, ...]
    feature_evidence: PairFeatureEvidence
    pose: RelativePoseEdge
    query_class_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        evidence = self.feature_evidence
        if self.pose.feature_evidence_id != evidence.evidence_id:
            raise ValueError(
                "pose and feature evidence do not share the same match set"
            )
        if (self.pose.view_id_a, self.pose.view_id_b) != (
            evidence.view_id_a,
            evidence.view_id_b,
        ):
            raise ValueError("pose and feature evidence view order must match")
        if any(item.view_id != evidence.view_id_a for item in self.regions_a):
            raise ValueError("regions_a must belong to feature-evidence view A")
        if any(item.view_id != evidence.view_id_b for item in self.regions_b):
            raise ValueError("regions_b must belong to feature-evidence view B")
        vocabularies = {item.vocabulary for item in (*self.regions_a, *self.regions_b)}
        if len(vocabularies) > 1:
            raise ValueError("region vocabularies are incompatible")


@dataclass(frozen=True, slots=True)
class RegionAssociationResult:
    candidates: tuple[RegionCorrespondenceCandidate, ...]
    edges: tuple[RegionCorrespondenceEdge, ...]
    diagnostics: dict[str, int | float | str]


def _pair_id(prefix: str, first: SemanticRegionNode, second: SemanticRegionNode) -> str:
    observations = sorted(
        ((first.view_id, first.region_id), (second.view_id, second.region_id))
    )
    return deterministic_id(prefix, observations)


def propose_region_correspondences(
    evidence: RegionAssociationInput,
    config: RegionAssociationConfig | None = None,
) -> tuple[RegionCorrespondenceCandidate, ...]:
    """Score class-compatible pairs without changing the pose result."""

    policy = config or RegionAssociationConfig()
    features = evidence.feature_evidence
    pose = evidence.pose
    if (
        not features.source.allowed_as_method_evidence
        or not pose.source.allowed_as_method_evidence
    ):
        raise ValueError(
            "evaluation_only or disabled evidence cannot enter association"
        )
    residuals_deg = np.degrees(features.pose_residuals_rad)
    pose_reasons: list[str] = []
    if policy.require_accepted_pose and not pose.accepted:
        pose_reasons.append("relative-pose-not-accepted")
    if pose.degenerate:
        pose_reasons.append("relative-pose-degenerate")

    candidates: list[RegionCorrespondenceCandidate] = []
    for region_a in evidence.regions_a:
        if (
            evidence.query_class_ids is not None
            and region_a.class_id not in evidence.query_class_ids
        ):
            continue
        mask_a = np.isin(features.feature_indices_a, region_a.feature_indices)
        for region_b in evidence.regions_b:
            if region_a.class_id != region_b.class_id:
                continue
            mask_b = np.isin(features.feature_indices_b, region_b.feature_indices)
            pair_mask = mask_a & mask_b & features.valid
            pair_indices = np.flatnonzero(pair_mask)
            inlier_indices = np.flatnonzero(pair_mask & features.pose_inlier_mask)
            median_residual = (
                float(np.median(residuals_deg[inlier_indices]))
                if len(inlier_indices)
                else None
            )
            reasons = list(pose_reasons)
            if len(pair_indices) < policy.min_region_matches:
                reasons.append("too-few-region-matches")
            if len(inlier_indices) < policy.min_region_pose_inliers:
                reasons.append("too-few-region-pose-inliers")
            if (
                median_residual is not None
                and median_residual > policy.max_median_pose_residual_deg
            ):
                reasons.append("high-region-pose-residual")
            semantic_score = math.sqrt(
                region_a.semantic_score * region_b.semantic_score
            )
            inlier_ratio = len(inlier_indices) / max(len(pair_indices), 1)
            residual_quality = (
                0.0
                if median_residual is None
                else math.exp(
                    -median_residual / max(policy.max_median_pose_residual_deg, 1e-12)
                )
            )
            ranking = float(
                np.clip(
                    0.35 * semantic_score
                    + 0.40 * inlier_ratio
                    + 0.25 * residual_quality,
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                RegionCorrespondenceCandidate(
                    candidate_id=_pair_id("region-candidate", region_a, region_b),
                    region_id_a=region_a.region_id,
                    region_id_b=region_b.region_id,
                    view_id_a=region_a.view_id,
                    view_id_b=region_b.view_id,
                    class_id=region_a.class_id,
                    class_name=region_a.class_name,
                    match_indices=pair_indices,
                    inlier_match_indices=inlier_indices,
                    semantic_score=semantic_score,
                    geometric_score=float(
                        np.clip(0.6 * inlier_ratio + 0.4 * residual_quality, 0.0, 1.0)
                    ),
                    ranking_score=ranking,
                    median_pose_residual_rad=(
                        None
                        if median_residual is None
                        else math.radians(median_residual)
                    ),
                    state="rejected" if reasons else "accepted",
                    reasons=tuple(reasons),
                )
            )
    return tuple(candidates)


def assign_region_correspondences(
    candidates: tuple[RegionCorrespondenceCandidate, ...],
    config: RegionAssociationConfig | None = None,
) -> tuple[RegionCorrespondenceCandidate, ...]:
    """Apply deterministic one-to-one assignment and expose ambiguity."""

    policy = config or RegionAssociationConfig()
    accepted_ids: set[str] = set()
    ambiguous_ids: set[str] = set()
    for class_id in sorted({item.class_id for item in candidates}):
        eligible = [
            item
            for item in candidates
            if item.class_id == class_id and item.state == "accepted"
        ]
        if not eligible:
            continue
        rows = sorted({item.region_id_a for item in eligible})
        columns = sorted({item.region_id_b for item in eligible})
        row_index = {value: index for index, value in enumerate(rows)}
        column_index = {value: index for index, value in enumerate(columns)}
        costs = np.full((len(rows), len(columns)), 1e6, dtype=np.float64)
        by_pair: dict[tuple[str, str], RegionCorrespondenceCandidate] = {}
        for item in eligible:
            costs[row_index[item.region_id_a], column_index[item.region_id_b]] = (
                -item.ranking_score
            )
            by_pair[(item.region_id_a, item.region_id_b)] = item
        selected_rows, selected_columns = linear_sum_assignment(costs)
        for row, column in zip(selected_rows, selected_columns, strict=True):
            if costs[row, column] >= 1e5:
                continue
            selected = by_pair[(rows[row], columns[column])]
            competing = [
                item.ranking_score
                for item in eligible
                if item.candidate_id != selected.candidate_id
                and (
                    item.region_id_a == selected.region_id_a
                    or item.region_id_b == selected.region_id_b
                )
            ]
            margin = (
                math.inf if not competing else selected.ranking_score - max(competing)
            )
            target = ambiguous_ids if margin < policy.ambiguity_margin else accepted_ids
            target.add(selected.candidate_id)

    result: list[RegionCorrespondenceCandidate] = []
    for item in candidates:
        if item.candidate_id in accepted_ids:
            result.append(item)
        elif item.candidate_id in ambiguous_ids:
            result.append(
                replace(item, state="ambiguous", reasons=("assignment-ambiguous",))
            )
        elif item.state == "accepted":
            result.append(
                replace(
                    item,
                    state="rejected",
                    reasons=("not-selected-by-one-to-one-assignment",),
                )
            )
        else:
            result.append(item)
    return tuple(result)


def associate_regions(
    evidence: RegionAssociationInput,
    config: RegionAssociationConfig | None = None,
) -> RegionAssociationResult:
    policy = config or RegionAssociationConfig()
    proposed = propose_region_correspondences(evidence, policy)
    assigned = assign_region_correspondences(proposed, policy)
    edges = tuple(
        RegionCorrespondenceEdge(
            edge_id=deterministic_id("region-edge", item.candidate_id),
            candidate_id=item.candidate_id,
            region_id_a=item.region_id_a,
            region_id_b=item.region_id_b,
            confidence=item.ranking_score,
            source_evidence_ids=(
                evidence.feature_evidence.evidence_id,
                evidence.pose.edge_id,
            ),
            correlation_groups=tuple(
                sorted(
                    {
                        value
                        for value in (
                            evidence.feature_evidence.source.correlation_group,
                            evidence.pose.source.correlation_group,
                        )
                        if value is not None
                    }
                )
            ),
        )
        for item in assigned
        if item.state == "accepted"
    )
    return RegionAssociationResult(
        candidates=assigned,
        edges=edges,
        diagnostics={
            "preset": policy.preset,
            "proposal_count": len(proposed),
            "accepted_count": len(edges),
            "ambiguous_count": sum(item.state == "ambiguous" for item in assigned),
        },
    )


__all__ = [
    "RegionAssociationConfig",
    "RegionAssociationInput",
    "RegionAssociationResult",
    "assign_region_correspondences",
    "associate_regions",
    "propose_region_correspondences",
]
