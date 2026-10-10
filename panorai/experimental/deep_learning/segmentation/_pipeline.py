"""Hybrid direct-spherical semantic and streamed gnomonic segmentation."""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Callable

import numpy as np

from panorai.geometry import GnomonicSpec

from ._charts import (
    advance_chart,
    backproject_face_mask_sets,
    extract_mask_face,
    extract_rgb_face,
    face_point_direction,
)
from ._fusion import (
    merge_proposals,
    non_maximum_suppression,
    partition_proposals_by_origin,
    proposal_is_accepted,
    resolve_panoptic_map,
    semantic_support_for_mask,
    solid_angle_weights,
    weighted_containment,
)
from ._models import (
    DenseSemanticEvidence,
    SphericalBinaryMask,
    SphericalMaskProposal,
    SphericalSeed,
    SphericalSegmentationConfig,
    SphericalSegmentationResult,
)
from ._prompts import (
    SegmentPrompt,
    consensus_frontiers,
    majority_and_envelope,
    maximum_pairwise_iou,
    pairwise_mask_iou,
    prompt_from_mask,
    prompt_grid,
)
from ._semantic import (
    angular_distance_degrees,
    concept_proxy_names,
    fibonacci_coverage_seeds,
    select_semantic_seeds,
)

ProgressCallback = Callable[[dict[str, object]], None]


def _nearest_resize(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    values = np.asarray(mask)
    if values.shape == shape_hw:
        return values.copy()
    source_height, source_width = values.shape
    height, width = shape_hw
    rows = np.minimum(
        ((np.arange(height) + 0.5) / height * source_height).astype(np.int64),
        source_height - 1,
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) / width * source_width).astype(np.int64),
        source_width - 1,
    )
    return values[rows[:, None], columns[None, :]]


def _best_pair(masks: np.ndarray) -> tuple[int, int]:
    matrix = pairwise_mask_iou(masks)
    pairs = ((0, 1), (0, 2), (1, 2))
    return max(pairs, key=lambda pair: (matrix[pair], -pair[0], -pair[1]))


class SphericalSemanticSegmenter:
    """Segment a canonical ERP with semantic and exhaustive point prompts.

    ``mask_backend`` must provide ``encode(rgb_face)``, ``predict(embeddings,
    shape_hw, prompt)`` and ``release(embeddings)``.  The official
    :class:`Sam21HieraLargeSegmenter` implements this contract.
    """

    def __init__(
        self,
        mask_backend: Any,
        config: SphericalSegmentationConfig | None = None,
    ) -> None:
        self.mask_backend = mask_backend
        self.config = config or SphericalSegmentationConfig()

    def _proposal(
        self,
        output: Any,
        spec: GnomonicSpec,
        seed: SphericalSeed,
        output_shape_hw: tuple[int, int],
        support: np.ndarray,
        proposal_id: str,
        *,
        evidence: DenseSemanticEvidence | None,
        parent_proposal_id: str | None = None,
        erp_masks: np.ndarray | None = None,
    ) -> SphericalMaskProposal | None:
        masks = np.asarray(output.masks, dtype=bool)
        consensus_face, _ = majority_and_envelope(masks)
        first, second = _best_pair(masks)
        pairwise = maximum_pairwise_iou(masks)
        predicted = tuple(float(value) for value in output.predicted_iou)
        stability = 0.5 * (
            float(output.stability_scores[first])
            + float(output.stability_scores[second])
        )
        if (
            pairwise < self.config.minimum_pairwise_iou
            or 0.5 * (predicted[first] + predicted[second])
            < self.config.minimum_predicted_iou
            or stability < self.config.minimum_stability
            or np.count_nonzero(consensus_face) < self.config.minimum_pixels
        ):
            return None
        if erp_masks is None:
            erp_masks = backproject_face_mask_sets(
                masks[None], spec, output_shape_hw, support=support
            )[0]
        consensus, envelope = majority_and_envelope(erp_masks)
        alternatives = tuple(SphericalBinaryMask.from_array(item) for item in erp_masks)
        concept_id = seed.concept_id
        semantic_score = seed.score if concept_id is not None else 0.0
        proxy_lookup = concept_proxy_names()
        provisional = SphericalMaskProposal(
            proposal_id=proposal_id,
            seed=seed,
            alternatives=alternatives,  # type: ignore[arg-type]
            consensus=SphericalBinaryMask.from_array(consensus),
            envelope=SphericalBinaryMask.from_array(envelope),
            predicted_iou=predicted,  # type: ignore[arg-type]
            pairwise_iou=pairwise,
            stability_score=stability,
            semantic_score=semantic_score,
            concept_id=concept_id,
            proxy_classes=proxy_lookup.get(concept_id, ()),
            parent_proposal_id=parent_proposal_id,
        )
        if evidence is not None and concept_id is None:
            inferred, score = semantic_support_for_mask(
                provisional.consensus, evidence, self.config
            )
            if inferred is not None:
                provisional = SphericalMaskProposal(
                    proposal_id=provisional.proposal_id,
                    seed=provisional.seed,
                    alternatives=provisional.alternatives,
                    consensus=provisional.consensus,
                    envelope=provisional.envelope,
                    predicted_iou=provisional.predicted_iou,
                    pairwise_iou=provisional.pairwise_iou,
                    stability_score=provisional.stability_score,
                    semantic_score=score,
                    concept_id=inferred,
                    proxy_classes=proxy_lookup[inferred],
                    parent_proposal_id=parent_proposal_id,
                )
        return provisional

    def _face_output_is_accepted(self, output: Any) -> bool:
        masks = np.asarray(output.masks, dtype=bool)
        first, second = _best_pair(masks)
        predicted = tuple(float(value) for value in output.predicted_iou)
        stability = 0.5 * (
            float(output.stability_scores[first])
            + float(output.stability_scores[second])
        )
        consensus, _ = majority_and_envelope(masks)
        return bool(
            maximum_pairwise_iou(masks) >= self.config.minimum_pairwise_iou
            and 0.5 * (predicted[first] + predicted[second])
            >= self.config.minimum_predicted_iou
            and stability >= self.config.minimum_stability
            and np.count_nonzero(consensus) >= self.config.minimum_pixels
        )

    def _discover_at_seed(
        self,
        rgb: np.ndarray,
        source_support: np.ndarray,
        output_support: np.ndarray,
        seed: SphericalSeed,
        output_shape_hw: tuple[int, int],
        *,
        evidence: DenseSemanticEvidence | None,
        proposal_start: int,
    ) -> list[SphericalMaskProposal]:
        size = self.config.discovery_face_size
        spec = GnomonicSpec(
            center_lon_deg=seed.longitude_degrees,
            center_lat_deg=seed.latitude_degrees,
            hfov_deg=self.config.discovery_fov_degrees,
            vfov_deg=self.config.discovery_fov_degrees,
            output_shape_hw=(size, size),
        )
        face = extract_rgb_face(rgb, spec)
        support_face = extract_mask_face(source_support, spec)
        embeddings = self.mask_backend.encode(face)
        prompts = (
            (SegmentPrompt((((size - 1) / 2, (size - 1) / 2),), (1,)),)
            if seed.source == "semantic"
            else prompt_grid((size, size), self.config.prompt_grid_size)
        )
        valid_prompts: list[tuple[int, SegmentPrompt, SphericalSeed]] = []
        for prompt_index, prompt in enumerate(prompts):
            x, y = prompt.points_xy[0]
            if not support_face[int(round(y)), int(round(x))]:
                continue
            prompt_direction = face_point_direction(spec, (x, y))
            prompt_seed = SphericalSeed(
                seed_id=(
                    seed.seed_id
                    if seed.source == "semantic"
                    else f"{seed.seed_id}:p{prompt_index:02d}"
                ),
                longitude_degrees=prompt_direction[0],
                latitude_degrees=prompt_direction[1],
                source=seed.source,
                score=seed.score,
                concept_id=seed.concept_id,
            )
            valid_prompts.append((prompt_index, prompt, prompt_seed))

        proposals: list[SphericalMaskProposal] = []
        try:
            batch_size = self.config.prompt_batch_size
            for offset in range(0, len(valid_prompts), batch_size):
                batch = valid_prompts[offset : offset + batch_size]
                batch_prompts = tuple(item[1] for item in batch)
                if hasattr(self.mask_backend, "predict_batch"):
                    outputs = self.mask_backend.predict_batch(
                        embeddings, (size, size), batch_prompts
                    )
                else:
                    outputs = tuple(
                        self.mask_backend.predict(embeddings, (size, size), prompt)
                        for prompt in batch_prompts
                    )
                if len(outputs) != len(batch):
                    raise RuntimeError("mask backend returned the wrong batch length")
                accepted = [
                    (entry, output)
                    for entry, output in zip(batch, outputs, strict=True)
                    if self._face_output_is_accepted(output)
                ]
                if not accepted:
                    continue
                projected_sets = backproject_face_mask_sets(
                    np.stack([output.masks for _, output in accepted]),
                    spec,
                    output_shape_hw,
                    support=output_support,
                )
                for ((_, _, prompt_seed), output), projected in zip(
                    accepted, projected_sets, strict=True
                ):
                    proposal = self._proposal(
                        output,
                        spec,
                        prompt_seed,
                        output_shape_hw,
                        output_support,
                        f"proposal-{proposal_start + len(proposals):06d}",
                        evidence=evidence,
                        erp_masks=projected,
                    )
                    if proposal is not None:
                        proposals.append(proposal)
        finally:
            self.mask_backend.release(embeddings)
            del embeddings, face
        return proposals

    def _expand(
        self,
        root: SphericalMaskProposal,
        rgb: np.ndarray,
        support: np.ndarray,
        output_shape_hw: tuple[int, int],
        *,
        evidence: DenseSemanticEvidence | None,
        proposal_start: int,
    ) -> list[SphericalMaskProposal]:
        size = self.config.discovery_face_size
        initial_spec = GnomonicSpec(
            center_lon_deg=root.seed.longitude_degrees,
            center_lat_deg=root.seed.latitude_degrees,
            hfov_deg=self.config.expansion_fov_degrees,
            vfov_deg=self.config.expansion_fov_degrees,
            output_shape_hw=(size, size),
        )
        initial_face_mask = extract_mask_face(root.consensus.to_array(), initial_spec)
        initial_frontiers = consensus_frontiers(
            initial_face_mask,
            band_pixels=self.config.expansion_frontier_band_pixels,
            minimum_pixels=self.config.expansion_frontier_minimum_pixels,
        )
        queue = deque(
            (initial_spec, point, side, root, root.envelope.to_array())
            for side, point in initial_frontiers
        )
        visited = [(initial_spec.center_lon_deg, initial_spec.center_lat_deg)]
        children: list[SphericalMaskProposal] = []
        while queue and len(children) < self.config.maximum_expansion_faces:
            current_spec, frontier, side, parent, parent_envelope = queue.popleft()
            next_spec, anchor = advance_chart(
                current_spec,
                frontier,
                step_fraction=self.config.expansion_step_fraction,
            )
            direction = (next_spec.center_lon_deg, next_spec.center_lat_deg)
            if any(
                angular_distance_degrees(direction, old)
                < 0.30 * self.config.expansion_fov_degrees
                for old in visited
            ):
                continue
            prior_face = extract_mask_face(parent_envelope, next_spec)
            if not prior_face.any():
                continue
            prompt = prompt_from_mask(prior_face, anchor)
            center = ((size - 1) / 2, (size - 1) / 2)
            prompt = SegmentPrompt(
                prompt.points_xy + (center,),
                prompt.labels + (1,),
                prompt.box_xyxy,
                prompt.prior_mask,
            )
            face = extract_rgb_face(rgb, next_spec)
            embeddings = self.mask_backend.encode(face)
            try:
                output = self.mask_backend.predict(embeddings, (size, size), prompt)
            finally:
                self.mask_backend.release(embeddings)
                del embeddings, face
            consensus_face, _ = majority_and_envelope(output.masks)
            retained = np.count_nonzero(consensus_face & prior_face) / max(
                1, np.count_nonzero(prior_face)
            )
            if retained < self.config.expansion_prior_retention:
                continue
            child_seed = SphericalSeed(
                seed_id=f"frontier:{parent.proposal_id}:{side}",
                longitude_degrees=direction[0],
                latitude_degrees=direction[1],
                source="frontier",
                score=parent.semantic_score,
                concept_id=parent.concept_id,
            )
            child = self._proposal(
                output,
                next_spec,
                child_seed,
                output_shape_hw,
                support,
                f"proposal-{proposal_start + len(children):06d}",
                evidence=evidence,
                parent_proposal_id=parent.proposal_id,
            )
            if child is None:
                continue
            children.append(child)
            visited.append(direction)
            for next_side, next_frontier in consensus_frontiers(
                consensus_face,
                band_pixels=self.config.expansion_frontier_band_pixels,
                minimum_pixels=self.config.expansion_frontier_minimum_pixels,
            ):
                queue.append(
                    (
                        next_spec,
                        next_frontier,
                        next_side,
                        child,
                        child.envelope.to_array(),
                    )
                )
        return children

    def segment(
        self,
        rgb: np.ndarray,
        support: np.ndarray,
        *,
        evidence: DenseSemanticEvidence | None = None,
        output_shape_hw: tuple[int, int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> SphericalSegmentationResult:
        """Run semantic seeds, exhaustive coverage, expansion, and fusion."""

        panorama = np.asarray(rgb)
        valid = np.asarray(support, dtype=bool)
        if panorama.ndim != 3 or panorama.shape[2] != 3:
            raise ValueError("rgb must have shape (H,W,3)")
        if valid.shape != panorama.shape[:2] or not valid.any():
            raise ValueError("support must be a non-empty mask matching rgb")
        target_shape = (
            panorama.shape[:2] if output_shape_hw is None else output_shape_hw
        )
        target_support = _nearest_resize(valid, target_shape).astype(bool)
        semantic_seeds = (
            select_semantic_seeds(
                evidence,
                maximum_per_concept=self.config.semantic_peaks_per_concept,
                minimum_separation_degrees=self.config.semantic_peak_separation_degrees,
                relative_threshold=self.config.semantic_relative_threshold,
            )
            if evidence is not None
            else ()
        )
        coverage_seeds = fibonacci_coverage_seeds(
            valid, direction_count=self.config.coverage_direction_count
        )
        discovery_started = time.perf_counter()
        roots: list[SphericalMaskProposal] = []
        root_proposal_count_before_nms = 0
        all_seeds = (*semantic_seeds, *coverage_seeds)
        for seed_index, seed in enumerate(all_seeds):
            discovered = self._discover_at_seed(
                panorama,
                valid,
                target_support,
                seed,
                target_shape,
                evidence=evidence,
                proposal_start=root_proposal_count_before_nms,
            )
            roots.extend(discovered)
            root_proposal_count_before_nms += len(discovered)
            if seed_index % 4 == 3 or seed_index + 1 == len(all_seeds):
                roots = list(
                    non_maximum_suppression(
                        tuple(roots), threshold=self.config.duplicate_iou
                    )
                )
            if progress is not None:
                progress(
                    {
                        "phase": "discovery",
                        "seed_index": seed_index,
                        "seed_count": len(semantic_seeds) + len(coverage_seeds),
                        "proposal_count_before_nms": root_proposal_count_before_nms,
                        "proposal_count": len(roots),
                    }
                )
        roots_tuple = tuple(roots)
        discovery_seconds = time.perf_counter() - discovery_started
        expansion_started = time.perf_counter()
        provisional_segments = merge_proposals(
            roots_tuple,
            self.config,
            evidence=evidence,
            suppress_duplicates=False,
        )
        root_lookup = {item.proposal_id: item for item in roots_tuple}
        expansion_roots: list[SphericalMaskProposal] = []
        for segment in provisional_segments:
            members = [root_lookup[item] for item in segment.proposal_ids]
            representative = max(
                members, key=lambda item: (item.quality_score, item.proposal_id)
            )
            expansion_roots.append(
                SphericalMaskProposal(
                    proposal_id=representative.proposal_id,
                    seed=representative.seed,
                    alternatives=representative.alternatives,
                    consensus=segment.consensus,
                    envelope=segment.envelope,
                    predicted_iou=representative.predicted_iou,
                    pairwise_iou=representative.pairwise_iou,
                    stability_score=representative.stability_score,
                    semantic_score=segment.semantic_score,
                    concept_id=segment.concept_id,
                    proxy_classes=segment.proxy_classes,
                    parent_proposal_id=representative.parent_proposal_id,
                )
            )
        expanded: list[SphericalMaskProposal] = []
        for root_index, root in enumerate(expansion_roots):
            children = self._expand(
                root,
                panorama,
                target_support,
                target_shape,
                evidence=evidence,
                proposal_start=root_proposal_count_before_nms + len(expanded),
            )
            expanded.extend(children)
            if progress is not None:
                progress(
                    {
                        "phase": "expansion",
                        "root_index": root_index,
                        "root_count": len(expansion_roots),
                        "expanded_proposal_count": len(expanded),
                    }
                )
        expansion_seconds = time.perf_counter() - expansion_started
        fusion_started = time.perf_counter()
        proposals = tuple((*roots_tuple, *expanded))
        segments = merge_proposals(proposals, self.config, evidence=evidence)
        panoptic = resolve_panoptic_map(segments, target_support)
        weights = solid_angle_weights(target_shape)
        support_weight = float(weights[target_support].sum())
        covered_weight = float(weights[(panoptic > 0) & target_support].sum())
        by_origin = partition_proposals_by_origin(proposals)

        def source_metrics(source: str) -> tuple[int, float]:
            source_segments = merge_proposals(
                by_origin[source], self.config, evidence=evidence
            )
            source_map = resolve_panoptic_map(source_segments, target_support)
            coverage = float(weights[(source_map > 0) & target_support].sum())
            return len(source_segments), (
                coverage / support_weight if support_weight > 0 else 0.0
            )

        cam_segment_count, cam_coverage = source_metrics("cam")
        exhaustive_segment_count, exhaustive_coverage = source_metrics("coverage")
        proposal_lookup = {item.proposal_id: item for item in proposals}
        continuities = [
            weighted_containment(
                item.consensus.to_array(),
                proposal_lookup[item.parent_proposal_id].consensus.to_array(),
            )
            for item in expanded
            if item.parent_proposal_id in proposal_lookup
        ]
        fusion_seconds = time.perf_counter() - fusion_started
        diagnostics = {
            "semantic_seed_count": len(semantic_seeds),
            "coverage_seed_count": len(coverage_seeds),
            "root_proposal_count_before_nms": root_proposal_count_before_nms,
            "root_proposal_count_after_nms": len(roots_tuple),
            "pre_expansion_instance_count": len(expansion_roots),
            "duplicate_proposal_count": root_proposal_count_before_nms
            - len(roots_tuple),
            "expanded_proposal_count": len(expanded),
            "accepted_proposal_count": sum(
                proposal_is_accepted(item, self.config) for item in proposals
            ),
            "segment_count": len(segments),
            "unknown_segment_count": sum(item.concept_id is None for item in segments),
            "cam_only_segment_count": cam_segment_count,
            "exhaustive_only_segment_count": exhaustive_segment_count,
            "cam_only_solid_angle_coverage_fraction": cam_coverage,
            "exhaustive_only_solid_angle_coverage_fraction": exhaustive_coverage,
            "cam_coverage_enrichment_fraction": (
                (covered_weight / support_weight) - exhaustive_coverage
                if support_weight > 0
                else 0.0
            ),
            "mean_proposal_stability": (
                float(np.mean([item.stability_score for item in proposals]))
                if proposals
                else 0.0
            ),
            "mean_expansion_parent_containment": (
                float(np.mean(continuities)) if continuities else None
            ),
            "solid_angle_coverage_fraction": (
                covered_weight / support_weight if support_weight > 0 else 0.0
            ),
            "output_shape_hw": list(target_shape),
            "streaming": "one RGB face and one image embedding at a time",
            "discovery_seconds": discovery_seconds,
            "expansion_seconds": expansion_seconds,
            "fusion_seconds": fusion_seconds,
        }
        return SphericalSegmentationResult(
            segments=segments,
            proposals=proposals,
            panoptic_map=panoptic,
            support=target_support,
            diagnostics=diagnostics,
        )
