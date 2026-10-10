"""Region association from semantic and feature-match evidence."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._models import (
    OBJECT_LOCALIZATION_INTERFACE,
    ObjectLocalizationConfig,
    PairObjectLocalizationInput,
    RegionAssociation,
)

_NAMESPACE = uuid.UUID("80a53f08-c2ab-4d33-9881-aaf82faf1c91")


def _stable_id(prefix: str, *tokens: object) -> str:
    payload = json.dumps(
        [OBJECT_LOCALIZATION_INTERFACE, *tokens],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{prefix}_{uuid.uuid5(_NAMESPACE, payload).hex}"


def object_hypothesis_id(
    view_id_a: str, region_id_a: str, view_id_b: str, region_id_b: str
) -> str:
    """Return the direction-independent ID for one two-view object pair."""

    observations = sorted(((view_id_a, region_id_a), (view_id_b, region_id_b)))
    return _stable_id("object", observations)


def propose_region_associations(
    evidence: PairObjectLocalizationInput,
    config: ObjectLocalizationConfig | None = None,
) -> tuple[RegionAssociation, ...]:
    """Score every class-compatible region pair without assigning identities."""

    policy = config or ObjectLocalizationConfig()
    matches = evidence.matches
    pose = evidence.pose
    match_valid = np.asarray(matches.valid, dtype=bool)
    pose_inliers = np.asarray(pose.inlier_mask, dtype=bool)
    residuals_deg = np.degrees(np.asarray(pose.residuals_rad, dtype=np.float64))
    proposals: list[RegionAssociation] = []

    pose_reasons: tuple[str, ...] = ()
    if policy.require_accepted_pose and not pose.quality_report.accepted:
        pose_reasons = ("relative-pose-not-accepted",)
    if pose.degenerate:
        pose_reasons += ("relative-pose-degenerate",)

    for region_a in evidence.regions_a:
        if region_a.class_id not in evidence.query.class_ids:
            continue
        mask_a = np.isin(matches.feature_indices_a, region_a.feature_indices)
        for region_b in evidence.regions_b:
            if region_a.class_id != region_b.class_id:
                continue
            mask_b = np.isin(matches.feature_indices_b, region_b.feature_indices)
            pair_mask = mask_a & mask_b & match_valid
            pair_indices = np.flatnonzero(pair_mask)
            inlier_indices = np.flatnonzero(pair_mask & pose_inliers)
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
            ranking_score = float(
                np.clip(
                    0.35 * semantic_score
                    + 0.40 * inlier_ratio
                    + 0.25 * residual_quality,
                    0.0,
                    1.0,
                )
            )
            proposals.append(
                RegionAssociation(
                    association_id=_stable_id(
                        "association",
                        evidence.view_id_a,
                        region_a.region_id,
                        evidence.view_id_b,
                        region_b.region_id,
                    ),
                    region_id_a=region_a.region_id,
                    region_id_b=region_b.region_id,
                    class_id=int(region_a.class_id),
                    class_name=region_a.class_name,
                    match_indices=pair_indices,
                    inlier_match_indices=inlier_indices,
                    match_count=len(pair_indices),
                    inlier_count=len(inlier_indices),
                    semantic_score=semantic_score,
                    median_pose_residual_deg=median_residual,
                    ranking_score=ranking_score,
                    state="rejected" if reasons else "accepted",
                    reasons=tuple(reasons),
                )
            )
    return tuple(proposals)


def assign_region_associations(
    proposals: tuple[RegionAssociation, ...],
    config: ObjectLocalizationConfig | None = None,
) -> tuple[RegionAssociation, ...]:
    """Apply a global one-to-one assignment and expose ambiguous choices."""

    policy = config or ObjectLocalizationConfig()
    accepted_ids: set[str] = set()
    ambiguous_ids: set[str] = set()

    class_ids = sorted({item.class_id for item in proposals})
    for class_id in class_ids:
        candidates = [
            item
            for item in proposals
            if item.class_id == class_id and item.state == "accepted"
        ]
        if not candidates:
            continue
        rows = sorted({item.region_id_a for item in candidates})
        columns = sorted({item.region_id_b for item in candidates})
        row_index = {value: index for index, value in enumerate(rows)}
        column_index = {value: index for index, value in enumerate(columns)}
        costs = np.full((len(rows), len(columns)), 1e6, dtype=np.float64)
        by_pair: dict[tuple[str, str], RegionAssociation] = {}
        for item in candidates:
            costs[
                row_index[item.region_id_a], column_index[item.region_id_b]
            ] = -item.ranking_score
            by_pair[(item.region_id_a, item.region_id_b)] = item
        selected_rows, selected_columns = linear_sum_assignment(costs)
        for row, column in zip(selected_rows, selected_columns, strict=True):
            if costs[row, column] >= 1e5:
                continue
            selected = by_pair[(rows[row], columns[column])]
            competitors = [
                item.ranking_score
                for item in candidates
                if item.association_id != selected.association_id
                and (
                    item.region_id_a == selected.region_id_a
                    or item.region_id_b == selected.region_id_b
                )
            ]
            margin = (
                math.inf
                if not competitors
                else selected.ranking_score - max(competitors)
            )
            if margin < policy.ambiguity_margin:
                ambiguous_ids.add(selected.association_id)
            else:
                accepted_ids.add(selected.association_id)

    assigned: list[RegionAssociation] = []
    for item in proposals:
        if item.association_id in accepted_ids:
            assigned.append(item)
        elif item.association_id in ambiguous_ids:
            assigned.append(
                replace(item, state="ambiguous", reasons=("assignment-ambiguous",))
            )
        elif item.state == "accepted":
            assigned.append(
                replace(
                    item,
                    state="rejected",
                    reasons=("not-selected-by-one-to-one-assignment",),
                )
            )
        else:
            assigned.append(item)
    return tuple(assigned)
