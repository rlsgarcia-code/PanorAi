"""Detector-style reduction for multiview spherical object hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from ._multiview import MultiviewObjectHypothesis, MultiviewSpatialRegion


ReductionAction = Literal["selected", "fused", "soft_nms", "fused_soft_nms"]


def _unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie within [0,1]")
    return numeric


@dataclass(frozen=True, slots=True)
class HypothesisPairCues:
    """Caller-supplied optional evidence for one canonical hypothesis pair."""

    hypothesis_id_a: str
    hypothesis_id_b: str
    geometric_compatibility: float | None = None
    transfer_iou: float | None = None
    appearance_cosine_similarity: float | None = None
    soft_nms_overlap: float | None = None
    geometric_contradiction: bool = False
    depth_separated: bool = False
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.hypothesis_id_a.strip()
            or not self.hypothesis_id_b.strip()
            or self.hypothesis_id_a >= self.hypothesis_id_b
        ):
            raise ValueError("hypothesis IDs must be non-empty and strictly ordered")
        for name in (
            "geometric_compatibility",
            "transfer_iou",
            "soft_nms_overlap",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit_interval(value, name))
        if self.appearance_cosine_similarity is not None:
            similarity = float(self.appearance_cosine_similarity)
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise ValueError("appearance_cosine_similarity must lie within [-1,1]")
            object.__setattr__(self, "appearance_cosine_similarity", similarity)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class HypothesisAffinityConfig:
    """Frozen factor exponents and reduction thresholds."""

    semantic_exponent: float = 1.0
    geometric_exponent: float = 1.0
    track_exponent: float = 1.0
    transfer_exponent: float = 1.0
    appearance_exponent: float = 1.0
    affinity_threshold: float = 0.5
    geometric_scale_floor: float = 1.0e-6
    soft_nms_sigma: float = 0.5
    soft_nms_min_overlap: float = 0.0
    fuse_finite_components: bool = True

    def __post_init__(self) -> None:
        for name in (
            "semantic_exponent",
            "geometric_exponent",
            "track_exponent",
            "transfer_exponent",
            "appearance_exponent",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        _unit_interval(self.affinity_threshold, "affinity_threshold")
        _unit_interval(self.soft_nms_min_overlap, "soft_nms_min_overlap")
        if (
            not math.isfinite(self.geometric_scale_floor)
            or self.geometric_scale_floor <= 0.0
        ):
            raise ValueError("geometric_scale_floor must be finite and positive")
        if not math.isfinite(self.soft_nms_sigma) or self.soft_nms_sigma <= 0.0:
            raise ValueError("soft_nms_sigma must be finite and positive")


@dataclass(frozen=True, slots=True)
class HypothesisAffinity:
    """Auditable compatibility factors and their multiplicative affinity."""

    hypothesis_id_a: str
    hypothesis_id_b: str
    hard_compatible: bool
    hard_reasons: tuple[str, ...]
    factors: Mapping[str, float]
    affinity: float
    soft_nms_overlap: float

    def __post_init__(self) -> None:
        if (
            not self.hypothesis_id_a
            or not self.hypothesis_id_b
            or self.hypothesis_id_a >= self.hypothesis_id_b
        ):
            raise ValueError("hypothesis IDs must be strictly ordered")
        factors = {
            str(name): _unit_interval(value, str(name))
            for name, value in self.factors.items()
        }
        if not factors:
            raise ValueError("at least one affinity factor is required")
        affinity = _unit_interval(self.affinity, "affinity")
        overlap = _unit_interval(self.soft_nms_overlap, "soft_nms_overlap")
        if not self.hard_compatible and affinity != 0.0:
            raise ValueError("hard-incompatible pairs must have zero affinity")
        object.__setattr__(self, "factors", MappingProxyType(factors))
        object.__setattr__(self, "affinity", affinity)
        object.__setattr__(self, "soft_nms_overlap", overlap)


@dataclass(frozen=True, slots=True)
class HypothesisReductionCandidate:
    """One hypothesis and the independent factors used for fusion weight."""

    hypothesis: MultiviewObjectHypothesis
    geometric_quality: float = 1.0
    independent_view_factor: float = 1.0

    def __post_init__(self) -> None:
        _unit_interval(self.geometric_quality, "geometric_quality")
        if (
            not math.isfinite(self.independent_view_factor)
            or self.independent_view_factor <= 0.0
        ):
            raise ValueError(
                "independent_view_factor must be finite and strictly positive"
            )

    @property
    def fusion_weight(self) -> float:
        return (
            self.hypothesis.semantic_score
            * self.geometric_quality
            * self.independent_view_factor
        )


@dataclass(frozen=True, slots=True)
class ReducedHypothesisRecord:
    """A selected, fused, or softly decayed hypothesis record."""

    representative: MultiviewObjectHypothesis
    source_hypothesis_ids: tuple[str, ...]
    original_scores: tuple[tuple[str, float], ...]
    reduced_score: float
    action: ReductionAction
    fused_spatial: MultiviewSpatialRegion | None = None
    suppressed_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.source_hypothesis_ids))) != self.source_hypothesis_ids:
            raise ValueError("source_hypothesis_ids must be sorted and unique")
        score_ids = tuple(item[0] for item in self.original_scores)
        if score_ids != self.source_hypothesis_ids:
            raise ValueError("original scores must align with source IDs")
        for _, score in self.original_scores:
            _unit_interval(score, "original score")
        _unit_interval(self.reduced_score, "reduced_score")
        if self.action in {"fused", "fused_soft_nms"}:
            if len(self.source_hypothesis_ids) < 2 or self.fused_spatial is None:
                raise ValueError(
                    "fused records require multiple sources and spatial support"
                )
        elif self.fused_spatial is not None:
            raise ValueError("only fused records may expose fused_spatial")
        if self.action in {"soft_nms", "fused_soft_nms"} and not self.suppressed_by:
            raise ValueError("soft-NMS records require a suppressor")


@dataclass(frozen=True, slots=True)
class HypothesisReductionResult:
    records: tuple[ReducedHypothesisRecord, ...]
    pair_affinities: tuple[HypothesisAffinity, ...]
    diagnostics: Mapping[str, int | float]

    def __post_init__(self) -> None:
        ids = [
            source_id
            for record in self.records
            for source_id in record.source_hypothesis_ids
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("each input hypothesis must occur in exactly one record")
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def _pair_ids(
    first: MultiviewObjectHypothesis, second: MultiviewObjectHypothesis
) -> tuple[str, str]:
    if first.hypothesis_id == second.hypothesis_id:
        raise ValueError("pair hypotheses must have distinct IDs")
    return tuple(sorted((first.hypothesis_id, second.hypothesis_id)))


def _geometric_factor(
    first: MultiviewObjectHypothesis,
    second: MultiviewObjectHypothesis,
    config: HypothesisAffinityConfig,
) -> tuple[float | None, float | None]:
    one = first.spatial
    two = second.spatial
    if one.center_xyz is None or two.center_xyz is None:
        return None, None
    distance = float(np.linalg.norm(one.center_xyz - two.center_xyz))
    radius_sum = float(one.extent_radius or 0.0) + float(two.extent_radius or 0.0)
    scale = max(radius_sum, config.geometric_scale_floor)
    compatibility = math.exp(-0.5 * (distance / scale) ** 2)
    if radius_sum <= config.geometric_scale_floor:
        overlap = 1.0 if distance <= config.geometric_scale_floor else 0.0
    else:
        overlap = max(0.0, 1.0 - distance / radius_sum)
    return compatibility, overlap


def hypothesis_pair_affinity(
    first: MultiviewObjectHypothesis,
    second: MultiviewObjectHypothesis,
    *,
    cues: HypothesisPairCues | None = None,
    config: HypothesisAffinityConfig | None = None,
) -> HypothesisAffinity:
    """Compute hard gates and the available-factor product from Eq. 41."""

    policy = config or HypothesisAffinityConfig()
    id_a, id_b = _pair_ids(first, second)
    if cues is not None and (cues.hypothesis_id_a, cues.hypothesis_id_b) != (
        id_a,
        id_b,
    ):
        raise ValueError("pair cues do not identify the supplied hypotheses")
    by_id = {first.hypothesis_id: first, second.hypothesis_id: second}
    one, two = by_id[id_a], by_id[id_b]

    reasons: list[str] = []
    if one.vocabulary != two.vocabulary:
        reasons.append("incompatible-vocabulary")
    if one.concept_id != two.concept_id:
        reasons.append("incompatible-concept")
    finite_one = one.spatial.center_xyz is not None
    finite_two = two.spatial.center_xyz is not None
    if finite_one != finite_two:
        reasons.append("incompatible-spatial-mode")
    if finite_one and finite_two:
        if one.spatial.mode != two.spatial.mode:
            reasons.append("incompatible-spatial-mode")
        if one.spatial.frame != two.spatial.frame:
            reasons.append("incompatible-spatial-frame")
        if one.spatial.units != two.spatial.units:
            reasons.append("incompatible-spatial-units")
    if cues is not None and cues.geometric_contradiction:
        reasons.append("geometric-contradiction")
    if cues is not None and cues.depth_separated:
        reasons.append("depth-separated-support")
    reasons = sorted(set(reasons))

    factors: dict[str, float] = {"semantic": 1.0}
    geometric, geometric_overlap = _geometric_factor(one, two, policy)
    if cues is not None and cues.geometric_compatibility is not None:
        geometric = cues.geometric_compatibility
    if geometric is not None:
        factors["geometric"] = geometric
    track_union = set(one.track_ids) | set(two.track_ids)
    if track_union:
        factors["track"] = len(set(one.track_ids) & set(two.track_ids)) / len(
            track_union
        )
    if cues is not None and cues.transfer_iou is not None:
        factors["transfer"] = cues.transfer_iou
    if cues is not None and cues.appearance_cosine_similarity is not None:
        factors["appearance"] = 0.5 * (1.0 + cues.appearance_cosine_similarity)

    exponents = {
        "semantic": policy.semantic_exponent,
        "geometric": policy.geometric_exponent,
        "track": policy.track_exponent,
        "transfer": policy.transfer_exponent,
        "appearance": policy.appearance_exponent,
    }
    affinity = 1.0
    for name, value in factors.items():
        affinity *= value ** exponents[name]
    if reasons:
        affinity = 0.0

    if cues is not None and cues.soft_nms_overlap is not None:
        overlap = cues.soft_nms_overlap
    else:
        overlap_factors = []
        if geometric_overlap is not None:
            overlap_factors.append(geometric_overlap)
        if cues is not None and cues.transfer_iou is not None:
            overlap_factors.append(cues.transfer_iou)
        overlap = (
            float(np.prod(overlap_factors) ** (1.0 / len(overlap_factors)))
            if overlap_factors
            else 0.0
        )
    if reasons:
        overlap = 0.0
    return HypothesisAffinity(
        hypothesis_id_a=id_a,
        hypothesis_id_b=id_b,
        hard_compatible=not reasons,
        hard_reasons=tuple(reasons),
        factors=factors,
        affinity=float(affinity),
        soft_nms_overlap=float(overlap),
    )


def _fuse_spatial(
    candidates: list[HypothesisReductionCandidate],
) -> MultiviewSpatialRegion:
    spatial = [item.hypothesis.spatial for item in candidates]
    first = spatial[0]
    if any(
        item.center_xyz is None
        or item.mode != first.mode
        or item.frame != first.frame
        or item.units != first.units
        for item in spatial
    ):
        raise ValueError("finite fusion requires compatible 3-D spatial support")
    candidate_weights = np.asarray(
        [item.fusion_weight for item in candidates], dtype=np.float64
    )
    if float(candidate_weights.sum()) <= 0.0:
        candidate_weights = np.ones(len(candidates), dtype=np.float64)
    candidate_weights /= candidate_weights.sum()
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for candidate_weight, item in zip(candidate_weights, spatial):
        points.append(np.asarray(item.support_points_xyz, dtype=np.float64))
        weights.append(
            candidate_weight * np.asarray(item.support_weights, dtype=np.float64)
        )
    support_points = np.concatenate(points, axis=0)
    support_weights = np.concatenate(weights)
    support_weights /= support_weights.sum()
    center = np.sum(support_points * support_weights[:, None], axis=0)
    centered = support_points - center
    covariance = (centered * support_weights[:, None]).T @ centered
    bounds = np.stack((support_points.min(axis=0), support_points.max(axis=0)))
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    return MultiviewSpatialRegion(
        mode=first.mode,
        frame=first.frame,
        units=first.units,
        support_kind="sparse_points",
        center_xyz=center,
        extent_covariance_xyz=covariance,
        bounding_box_xyz=bounds,
        extent_radius=radius,
        support_points_xyz=support_points,
        support_weights=support_weights,
    )


def reduce_object_hypotheses(
    candidates: tuple[HypothesisReductionCandidate, ...],
    *,
    pair_cues: tuple[HypothesisPairCues, ...] = (),
    config: HypothesisAffinityConfig | None = None,
) -> HypothesisReductionResult:
    """Fuse affinity-graph components and retain Soft-NMS alternatives."""

    policy = config or HypothesisAffinityConfig()
    if len({item.hypothesis.hypothesis_id for item in candidates}) != len(candidates):
        raise ValueError("candidate hypothesis IDs must be unique")
    candidate_by_id = {item.hypothesis.hypothesis_id: item for item in candidates}
    cue_by_pair = {
        (item.hypothesis_id_a, item.hypothesis_id_b): item for item in pair_cues
    }
    if len(cue_by_pair) != len(pair_cues):
        raise ValueError("pair cues must identify unique pairs")
    unknown = sorted(
        {
            item_id
            for pair in cue_by_pair
            for item_id in pair
            if item_id not in candidate_by_id
        }
    )
    if unknown:
        raise ValueError(f"pair cues reference unknown hypotheses: {unknown}")

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.hypothesis.semantic_score,
            item.hypothesis.hypothesis_id,
        ),
    )
    affinity_by_pair: dict[tuple[str, str], HypothesisAffinity] = {}
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            pair = tuple(
                sorted(
                    (
                        first.hypothesis.hypothesis_id,
                        second.hypothesis.hypothesis_id,
                    )
                )
            )
            affinity_by_pair[pair] = hypothesis_pair_affinity(
                first.hypothesis,
                second.hypothesis,
                cues=cue_by_pair.get(pair),
                config=policy,
            )

    parent = {
        item.hypothesis.hypothesis_id: item.hypothesis.hypothesis_id for item in ordered
    }

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(first_id: str, second_id: str) -> None:
        first_root = find(first_id)
        second_root = find(second_id)
        if first_root == second_root:
            return
        lower, higher = sorted((first_root, second_root))
        parent[higher] = lower

    for pair, affinity in affinity_by_pair.items():
        if affinity.hard_compatible and affinity.affinity >= policy.affinity_threshold:
            union(*pair)
    groups_by_root: dict[str, list[HypothesisReductionCandidate]] = {}
    for candidate in ordered:
        root = find(candidate.hypothesis.hypothesis_id)
        groups_by_root.setdefault(root, []).append(candidate)
    groups = [
        groups_by_root[root]
        for root in sorted(
            groups_by_root,
            key=lambda item: (
                -max(
                    candidate.hypothesis.semantic_score
                    for candidate in groups_by_root[item]
                ),
                item,
            ),
        )
    ]

    records: list[ReducedHypothesisRecord] = []
    for group in groups:
        source_ids = tuple(sorted(item.hypothesis.hypothesis_id for item in group))
        representative = min(
            group,
            key=lambda item: (
                -item.hypothesis.semantic_score,
                item.hypothesis.hypothesis_id,
            ),
        ).hypothesis
        finite = all(item.hypothesis.spatial.center_xyz is not None for item in group)
        all_hard_compatible = all(
            affinity_by_pair[
                tuple(
                    sorted(
                        (
                            first.hypothesis.hypothesis_id,
                            second.hypothesis.hypothesis_id,
                        )
                    )
                )
            ].hard_compatible
            for first_index, first in enumerate(group)
            for second in group[first_index + 1 :]
        )
        if (
            len(group) > 1
            and finite
            and all_hard_compatible
            and policy.fuse_finite_components
        ):
            total_weight = sum(item.fusion_weight for item in group)
            reduced_score = (
                sum(
                    item.fusion_weight * item.hypothesis.semantic_score
                    for item in group
                )
                / total_weight
                if total_weight > 0.0
                else float(np.mean([item.hypothesis.semantic_score for item in group]))
            )
            records.append(
                ReducedHypothesisRecord(
                    representative=representative,
                    source_hypothesis_ids=source_ids,
                    original_scores=tuple(
                        (
                            item_id,
                            candidate_by_id[item_id].hypothesis.semantic_score,
                        )
                        for item_id in source_ids
                    ),
                    reduced_score=reduced_score,
                    action="fused",
                    fused_spatial=_fuse_spatial(group),
                )
            )
        else:
            for item in group:
                hypothesis = item.hypothesis
                records.append(
                    ReducedHypothesisRecord(
                        representative=hypothesis,
                        source_hypothesis_ids=(hypothesis.hypothesis_id,),
                        original_scores=(
                            (hypothesis.hypothesis_id, hypothesis.semantic_score),
                        ),
                        reduced_score=hypothesis.semantic_score,
                        action="selected",
                    )
                )

    records.sort(
        key=lambda item: (
            -item.reduced_score,
            item.source_hypothesis_ids,
        )
    )
    final_records: list[ReducedHypothesisRecord] = []
    for record in records:
        score = record.reduced_score
        suppressors: list[str] = []
        for selected in final_records:
            overlaps = []
            for source_a in record.source_hypothesis_ids:
                for source_b in selected.source_hypothesis_ids:
                    pair = tuple(sorted((source_a, source_b)))
                    affinity = affinity_by_pair.get(pair)
                    if affinity is not None and affinity.hard_compatible:
                        overlaps.append(affinity.soft_nms_overlap)
            overlap = max(overlaps, default=0.0)
            if overlap > policy.soft_nms_min_overlap:
                score *= math.exp(-(overlap**2) / policy.soft_nms_sigma)
                suppressors.extend(selected.source_hypothesis_ids)
        if suppressors and record.action == "selected":
            action: ReductionAction = "soft_nms"
        elif suppressors and record.action == "fused":
            action = "fused_soft_nms"
        else:
            action = record.action
        final_records.append(
            ReducedHypothesisRecord(
                representative=record.representative,
                source_hypothesis_ids=record.source_hypothesis_ids,
                original_scores=record.original_scores,
                reduced_score=score,
                action=action,
                fused_spatial=record.fused_spatial,
                suppressed_by=tuple(sorted(set(suppressors))),
            )
        )

    final_records.sort(
        key=lambda item: (
            -item.reduced_score,
            item.source_hypothesis_ids,
        )
    )
    return HypothesisReductionResult(
        records=tuple(final_records),
        pair_affinities=tuple(
            affinity_by_pair[pair] for pair in sorted(affinity_by_pair)
        ),
        diagnostics={
            "input_hypothesis_count": len(candidates),
            "output_record_count": len(final_records),
            "fused_record_count": sum(
                item.action in {"fused", "fused_soft_nms"} for item in final_records
            ),
            "soft_nms_record_count": sum(
                item.action in {"soft_nms", "fused_soft_nms"} for item in final_records
            ),
            "retained_source_hypothesis_count": sum(
                len(item.source_hypothesis_ids) for item in final_records
            ),
        },
    )
