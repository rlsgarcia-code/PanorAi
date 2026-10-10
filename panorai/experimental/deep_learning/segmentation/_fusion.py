"""Solid-angle-aware proposal filtering, fusion, and panoptic resolution."""

from __future__ import annotations

import math

import numpy as np

from ._models import (
    DenseSemanticEvidence,
    SphericalBinaryMask,
    SphericalMaskProposal,
    SphericalSegment,
    SphericalSegmentationConfig,
)
from ._semantic import concept_display_names, concept_proxy_names


def _resize_nearest(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values)
    if source.shape == shape_hw:
        return source.copy()
    source_height, source_width = source.shape
    target_height, target_width = shape_hw
    rows = np.clip(
        np.floor((np.arange(target_height) + 0.5) / target_height * source_height),
        0,
        source_height - 1,
    ).astype(np.int64)
    columns = np.mod(
        np.floor((np.arange(target_width) + 0.5) / target_width * source_width),
        source_width,
    ).astype(np.int64)
    return source[rows[:, None], columns[None, :]]


def solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if height < 1 or width < 1:
        raise ValueError("shape_hw must contain positive dimensions")
    latitude = math.pi / 2 - (np.arange(height) + 0.5) * math.pi / height
    return np.broadcast_to(np.cos(latitude)[:, None], shape_hw)


def weighted_intersection_over_union(
    first: np.ndarray,
    second: np.ndarray,
    *,
    support: np.ndarray | None = None,
) -> float:
    one = np.asarray(first, dtype=bool)
    two = np.asarray(second, dtype=bool)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("masks must have the same HW shape")
    if support is not None:
        valid = np.asarray(support, dtype=bool)
        if valid.shape != one.shape:
            raise ValueError("support must match mask shape")
        one &= valid
        two &= valid
    weights = solid_angle_weights(one.shape)
    intersection = float(weights[one & two].sum())
    union = float(weights[one | two].sum())
    return intersection / union if union > 0 else 1.0


def weighted_containment(first: np.ndarray, second: np.ndarray) -> float:
    """Return the fraction of the smaller weighted mask inside the other."""

    one = np.asarray(first, dtype=bool)
    two = np.asarray(second, dtype=bool)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("masks must have the same HW shape")
    weights = solid_angle_weights(one.shape)
    first_area = float(weights[one].sum())
    second_area = float(weights[two].sum())
    denominator = min(first_area, second_area)
    if denominator <= 0:
        return 1.0 if first_area == second_area else 0.0
    return float(weights[one & two].sum()) / denominator


def _weighted_overlap_metrics(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    one = np.asarray(first, dtype=bool)
    two = np.asarray(second, dtype=bool)
    weights = solid_angle_weights(one.shape)
    intersection = float(weights[one & two].sum())
    first_area = float(weights[one].sum())
    second_area = float(weights[two].sum())
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    return (
        intersection / union if union > 0 else 1.0,
        intersection / smaller if smaller > 0 else 1.0,
    )


def proposal_is_accepted(
    proposal: SphericalMaskProposal,
    config: SphericalSegmentationConfig,
) -> bool:
    pair_scores = sorted(proposal.predicted_iou, reverse=True)
    return bool(
        proposal.pairwise_iou >= config.minimum_pairwise_iou
        and 0.5 * (pair_scores[0] + pair_scores[1]) >= config.minimum_predicted_iou
        and proposal.stability_score >= config.minimum_stability
        and proposal.consensus.pixel_count >= config.minimum_pixels
    )


def _coarse_occupancy(
    mask: np.ndarray, maximum_shape: tuple[int, int] = (64, 128)
) -> np.ndarray:
    """Conservative block occupancy used only to reject impossible overlaps."""

    values = np.asarray(mask, dtype=bool)
    row_step = max(1, math.ceil(values.shape[0] / maximum_shape[0]))
    column_step = max(1, math.ceil(values.shape[1] / maximum_shape[1]))
    padded_height = math.ceil(values.shape[0] / row_step) * row_step
    padded_width = math.ceil(values.shape[1] / column_step) * column_step
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[: values.shape[0], : values.shape[1]] = values
    return padded.reshape(
        padded_height // row_step,
        row_step,
        padded_width // column_step,
        column_step,
    ).any(axis=(1, 3))


def semantic_support_for_mask(
    mask: SphericalBinaryMask,
    evidence: DenseSemanticEvidence,
    config: SphericalSegmentationConfig,
) -> tuple[str | None, float]:
    """Assign one proxy concept only when its evidence is enriched inside."""

    target = mask.to_array()
    if target.shape != evidence.support.shape:
        target = _resize_nearest(target, evidence.support.shape).astype(bool)
    target &= evidence.support
    if not target.any():
        return None, 0.0
    best_id: str | None = None
    best_score = 0.0
    for concept_id, channel in zip(evidence.concept_ids, evidence.scores, strict=True):
        inside = float(channel[target].mean())
        outside_mask = evidence.support & ~target
        outside = float(channel[outside_mask].mean()) if outside_mask.any() else 0.0
        contrast = inside - outside
        if (
            inside >= config.semantic_minimum_mean
            and contrast >= config.semantic_minimum_contrast
            and inside > best_score
        ):
            best_id = concept_id
            best_score = inside
    return best_id, best_score


def non_maximum_suppression(
    proposals: tuple[SphericalMaskProposal, ...],
    *,
    threshold: float,
) -> tuple[SphericalMaskProposal, ...]:
    """Suppress only duplicate proposals with compatible semantic identity."""

    retained: list[SphericalMaskProposal] = []
    retained_coarse: list[np.ndarray] = []
    retained_area: list[float] = []
    for proposal in sorted(
        proposals, key=lambda item: (-item.quality_score, item.proposal_id)
    ):
        current = proposal.consensus.to_array()
        current_coarse = _coarse_occupancy(current)
        weights = solid_angle_weights(current.shape)
        current_area = float(weights[current].sum())
        duplicate = False
        for previous, previous_coarse, previous_area in zip(
            retained, retained_coarse, retained_area, strict=True
        ):
            # Unknown is a real catalogue outcome, not a wildcard.  Keeping an
            # unknown and a semantic proposal as competitors preserves the
            # evidence needed to audit panoptic conflict resolution.
            compatible = proposal.concept_id == previous.concept_id
            if not compatible:
                continue
            maximum_iou = min(current_area, previous_area) / max(
                current_area, previous_area, np.finfo(np.float64).eps
            )
            if maximum_iou < threshold or not np.any(current_coarse & previous_coarse):
                continue
            if (
                weighted_intersection_over_union(current, previous.consensus.to_array())
                >= threshold
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(proposal)
            retained_coarse.append(current_coarse)
            retained_area.append(current_area)
    return tuple(retained)


def partition_proposals_by_origin(
    proposals: tuple[SphericalMaskProposal, ...],
) -> dict[str, tuple[SphericalMaskProposal, ...]]:
    """Partition proposals by their semantic-CAM or exhaustive root seed."""

    lookup = {proposal.proposal_id: proposal for proposal in proposals}

    def origin(proposal: SphericalMaskProposal) -> str:
        current = proposal
        visited: set[str] = set()
        while current.seed.source == "frontier":
            if current.proposal_id in visited:
                raise ValueError("proposal parent graph contains a cycle")
            visited.add(current.proposal_id)
            parent_id = current.parent_proposal_id
            if parent_id is None or parent_id not in lookup:
                return "unresolved"
            current = lookup[parent_id]
        return "cam" if current.seed.source == "semantic" else "coverage"

    groups: dict[str, list[SphericalMaskProposal]] = {
        "cam": [],
        "coverage": [],
        "unresolved": [],
    }
    for proposal in proposals:
        groups[origin(proposal)].append(proposal)
    return {name: tuple(items) for name, items in groups.items()}


def merge_proposals(
    proposals: tuple[SphericalMaskProposal, ...],
    config: SphericalSegmentationConfig,
    *,
    evidence: DenseSemanticEvidence | None = None,
    suppress_duplicates: bool = True,
) -> tuple[SphericalSegment, ...]:
    """Merge compatible overlapping/continued proposals into spherical instances."""

    accepted = tuple(item for item in proposals if proposal_is_accepted(item, config))
    pending = list(
        non_maximum_suppression(accepted, threshold=config.duplicate_iou)
        if suppress_duplicates
        else accepted
    )
    coarse = {
        item.proposal_id: _coarse_occupancy(item.consensus.to_array())
        for item in pending
    }
    groups: list[list[SphericalMaskProposal]] = []
    while pending:
        group = [pending.pop(0)]
        combined = group[0].consensus.to_array()
        combined_coarse = coarse[group[0].proposal_id].copy()
        changed = True
        while changed:
            changed = False
            for candidate in tuple(pending):
                compatible = all(
                    item.concept_id is None
                    or candidate.concept_id is None
                    or item.concept_id == candidate.concept_id
                    for item in group
                )
                linked = candidate.parent_proposal_id in {
                    item.proposal_id for item in group
                } or any(
                    item.parent_proposal_id == candidate.proposal_id for item in group
                )
                if not compatible:
                    continue
                if not linked and not np.any(
                    combined_coarse & coarse[candidate.proposal_id]
                ):
                    continue
                candidate_mask = candidate.consensus.to_array()
                overlap, containment = _weighted_overlap_metrics(
                    combined, candidate_mask
                )
                if (
                    linked
                    or overlap >= config.merge_iou
                    or containment >= config.merge_containment
                ):
                    group.append(candidate)
                    pending.remove(candidate)
                    combined |= candidate_mask
                    combined_coarse |= coarse[candidate.proposal_id]
                    changed = True
        groups.append(group)

    names = concept_display_names()
    segments: list[SphericalSegment] = []
    for segment_id, group in enumerate(groups, start=1):
        consensus = np.logical_or.reduce([item.consensus.to_array() for item in group])
        envelope = np.logical_or.reduce([item.envelope.to_array() for item in group])
        declared = max(group, key=lambda item: item.semantic_score).concept_id
        semantic_score = max(item.semantic_score for item in group)
        if evidence is not None:
            inferred, inferred_score = semantic_support_for_mask(
                SphericalBinaryMask.from_array(consensus), evidence, config
            )
            if inferred is not None:
                declared, semantic_score = inferred, inferred_score
        # The public explanation must describe the evidence for the final
        # assigned concept, not every provisional label that participated in
        # geometric fusion.  In particular, an unlabeled coverage proposal can
        # merge with a semantic proposal and the dense-evidence pass can then
        # choose a different, better-supported concept for the union.
        proxy_names = concept_proxy_names().get(declared, ())
        segments.append(
            SphericalSegment(
                segment_id=segment_id,
                concept_id=declared,
                concept_name=names.get(declared, "regiao desconhecida"),
                proxy_classes=proxy_names,
                consensus=SphericalBinaryMask.from_array(consensus),
                envelope=SphericalBinaryMask.from_array(envelope),
                quality_score=max(item.quality_score for item in group),
                semantic_score=semantic_score,
                proposal_ids=tuple(sorted(item.proposal_id for item in group)),
            )
        )
    return tuple(segments)


def resolve_panoptic_map(
    segments: tuple[SphericalSegment, ...], support: np.ndarray
) -> np.ndarray:
    """Assign every claimed pixel once using deterministic quality ordering."""

    valid = np.asarray(support, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("support must be HW")
    output = np.zeros(valid.shape, dtype=np.uint32)
    ordered = sorted(segments, key=lambda item: (-item.quality_score, item.segment_id))
    for segment in ordered:
        mask = segment.consensus.to_array()
        if mask.shape != valid.shape:
            mask = _resize_nearest(mask, valid.shape).astype(bool)
        claim = valid & mask & (output == 0)
        output[claim] = segment.segment_id
    return output
