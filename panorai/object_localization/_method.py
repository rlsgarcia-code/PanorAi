"""One-call composition of the spherical object-hypothesis methodology."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np

from panorai.reconstruction import (
    SphericalCameraPose,
    SphericalReconstructionResult,
)

from ._anchor_fusion import (
    SphericalAnchorFusionConfig,
    SphericalAnchorFusionResult,
    SphericalAnchorPairCues,
    fuse_registered_spherical_anchors,
    generate_spherical_fov_anchors,
)
from ._hypothesis_reduction import (
    HypothesisAffinityConfig,
    HypothesisPairCues,
    HypothesisReductionCandidate,
    HypothesisReductionResult,
    reduce_object_hypotheses,
)
from ._multiview import (
    HypothesisGnomonicRegion,
    HypothesisViewConsistency,
    HypothesisViewEmbedding,
    MultiviewObjectHypothesis,
    MultiviewObjectHypothesisResult,
    RegisteredBearingTrack,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
    hypothesis_view_consistency,
)
from ._region_consistency import (
    RegisteredRadialDepth,
    SphericalRegionMetrics,
    evaluate_spherical_region,
    symmetric_spherical_region_transfer,
)


TransferRole = Literal["within-hypothesis", "reduction-cue"]


@dataclass(frozen=True, slots=True)
class HypothesisRegionEvaluation:
    hypothesis_id: str
    view_id: str
    metrics: SphericalRegionMetrics

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip() or not self.view_id.strip():
            raise ValueError("hypothesis_id and view_id must be non-empty")
        if self.metrics.view_id != self.view_id:
            raise ValueError("regional metrics must align with view_id")


@dataclass(frozen=True, slots=True)
class HypothesisViewPairEvaluation:
    hypothesis_id: str
    consistency: HypothesisViewConsistency

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")


@dataclass(frozen=True, slots=True)
class HypothesisTransferEvaluation:
    """Compact transfer score; transient ERP masks are intentionally omitted."""

    hypothesis_id_a: str
    hypothesis_id_b: str
    view_id_a: str
    view_id_b: str
    role: TransferRole
    forward_iou: float
    reverse_iou: float
    symmetric_iou: float

    def __post_init__(self) -> None:
        for name in (
            "hypothesis_id_a",
            "hypothesis_id_b",
            "view_id_a",
            "view_id_b",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.view_id_a == self.view_id_b:
            raise ValueError("transfer evaluation requires two distinct views")
        for name in ("forward_iou", "reverse_iou", "symmetric_iou"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0,1]")
        expected = 0.5 * (self.forward_iou + self.reverse_iou)
        if not np.isclose(self.symmetric_iou, expected, rtol=0.0, atol=1e-12):
            raise ValueError("symmetric_iou must be the directional mean")


@dataclass(frozen=True, slots=True)
class SphericalObjectHypothesisMethodConfig:
    generator: SphericalObjectHypothesisConfig = field(
        default_factory=SphericalObjectHypothesisConfig
    )
    affinity: HypothesisAffinityConfig = field(default_factory=HypothesisAffinityConfig)
    anchor_fusion: SphericalAnchorFusionConfig = field(
        default_factory=SphericalAnchorFusionConfig
    )
    anchor_scale_factors: tuple[float, ...] = (0.75, 1.0, 1.33)
    anchor_aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    reprojection_sigma_deg: float = 1.0
    enable_registered_transfer: bool = True
    enable_anchor_fusion: bool = True
    enable_reduction: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.reprojection_sigma_deg)
            or self.reprojection_sigma_deg <= 0.0
        ):
            raise ValueError("reprojection_sigma_deg must be finite and positive")
        for name in ("anchor_scale_factors", "anchor_aspect_ratios"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any(
                not np.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must contain finite positive values")
            object.__setattr__(self, name, tuple(sorted(set(values))))


@dataclass(frozen=True, slots=True)
class SphericalObjectHypothesisMethodResult:
    generation: MultiviewObjectHypothesisResult
    region_evaluations: tuple[HypothesisRegionEvaluation, ...]
    view_pair_evaluations: tuple[HypothesisViewPairEvaluation, ...]
    transfer_evaluations: tuple[HypothesisTransferEvaluation, ...]
    anchor_fusion: SphericalAnchorFusionResult | None
    reduction_pair_cues: tuple[HypothesisPairCues, ...]
    reduction: HypothesisReductionResult | None
    diagnostics: Mapping[str, int | float | str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


class SphericalObjectHypothesisMethod:
    """Compose the paper method without owning semantic or depth inference."""

    def __init__(
        self, config: SphericalObjectHypothesisMethodConfig | None = None
    ) -> None:
        self.config = config or SphericalObjectHypothesisMethodConfig()
        self.generator = SphericalObjectHypothesisGenerator(self.config.generator)

    @staticmethod
    def _compact_transfer(
        hypothesis_id_a: str,
        hypothesis_id_b: str,
        view_id_a: str,
        view_id_b: str,
        role: TransferRole,
        transfer: object,
    ) -> HypothesisTransferEvaluation:
        forward = getattr(transfer, "forward")
        reverse = getattr(transfer, "reverse")
        symmetric = float(getattr(transfer, "symmetric_solid_angle_iou"))
        return HypothesisTransferEvaluation(
            hypothesis_id_a=hypothesis_id_a,
            hypothesis_id_b=hypothesis_id_b,
            view_id_a=view_id_a,
            view_id_b=view_id_b,
            role=role,
            forward_iou=float(forward.solid_angle_iou),
            reverse_iou=float(reverse.solid_angle_iou),
            symmetric_iou=symmetric,
        )

    def run(
        self,
        *,
        semantic_evidence: tuple[RegisteredSemanticEvidence, ...],
        concepts: tuple[SemanticConcept, ...],
        reconstruction: SphericalReconstructionResult | None = None,
        poses: tuple[SphericalCameraPose, ...] | None = None,
        bearing_tracks: tuple[RegisteredBearingTrack, ...] | None = None,
        reconstruction_scale: str = "arbitrary",
        reference_view_id: str | None = None,
        radial_depth: tuple[RegisteredRadialDepth, ...] = (),
        independent_regions_by_hypothesis: Mapping[
            str, tuple[HypothesisGnomonicRegion, ...]
        ]
        | None = None,
        independent_region_provider: Callable[
            [MultiviewObjectHypothesis],
            tuple[HypothesisGnomonicRegion, ...],
        ]
        | None = None,
        embeddings_by_hypothesis: Mapping[str, tuple[HypothesisViewEmbedding, ...]]
        | None = None,
        pair_cues: tuple[HypothesisPairCues, ...] = (),
        anchor_pair_cues: tuple[SphericalAnchorPairCues, ...] = (),
        geometric_quality_by_hypothesis: Mapping[str, float] | None = None,
        independent_view_factor_by_hypothesis: Mapping[str, float] | None = None,
    ) -> SphericalObjectHypothesisMethodResult:
        """Run every available stage with exactly one declared geometry source."""

        has_reconstruction = reconstruction is not None
        has_bearing_path = poses is not None or bearing_tracks is not None
        if has_reconstruction == has_bearing_path:
            raise ValueError(
                "provide exactly one geometry source: reconstruction or poses plus bearing_tracks"
            )
        if has_reconstruction:
            assert reconstruction is not None
            generation = self.generator.generate(
                reconstruction, semantic_evidence, concepts
            )
            resolved_poses = tuple(reconstruction.poses)
            geometry_source = "refined-sparse-reconstruction"
        else:
            if poses is None or bearing_tracks is None:
                raise ValueError(
                    "the point-free path requires both poses and bearing_tracks"
                )
            generation = self.generator.generate_from_bearings(
                poses=poses,
                tracks=bearing_tracks,
                semantic_evidence=semantic_evidence,
                concepts=concepts,
                scale=reconstruction_scale,
                reference_view_id=reference_view_id,
            )
            resolved_poses = tuple(poses)
            geometry_source = "registered-poses-and-bearings"

        pose_by_view = {item.panorama_id: item for item in resolved_poses}
        if len(pose_by_view) != len(resolved_poses):
            raise ValueError("registered pose IDs must be unique")
        evidence_by_view = {item.view_id: item for item in semantic_evidence}
        concept_by_id = {item.concept_id: item for item in concepts}
        depth_by_view = {item.view_id: item for item in radial_depth}
        if len(depth_by_view) != len(radial_depth):
            raise ValueError("radial depth view IDs must be unique")
        unknown_depth = sorted(set(depth_by_view) - set(pose_by_view))
        if unknown_depth:
            raise ValueError(
                f"radial depth references unregistered views: {unknown_depth}"
            )

        embeddings = dict(embeddings_by_hypothesis or {})
        hypothesis_ids = {item.hypothesis_id for item in generation.hypotheses}
        unknown_embeddings = sorted(set(embeddings) - hypothesis_ids)
        if unknown_embeddings:
            raise ValueError(
                "embeddings reference unknown generated hypotheses: "
                f"{unknown_embeddings}"
            )
        if (
            independent_regions_by_hypothesis is not None
            and independent_region_provider is not None
        ):
            raise ValueError(
                "provide independent regions by mapping or provider, not both"
            )
        if independent_region_provider is not None:
            independent_regions = {
                item.hypothesis_id: tuple(independent_region_provider(item))
                for item in generation.hypotheses
            }
        else:
            independent_regions = dict(independent_regions_by_hypothesis or {})
        unknown_region_hypotheses = sorted(set(independent_regions) - hypothesis_ids)
        if unknown_region_hypotheses:
            raise ValueError(
                "independent regions reference unknown generated hypotheses: "
                f"{unknown_region_hypotheses}"
            )
        for hypothesis_id, regions in independent_regions.items():
            view_ids = tuple(item.view_id for item in regions)
            if len(set(view_ids)) != len(view_ids):
                raise ValueError(
                    f"independent regions repeat a view for {hypothesis_id!r}"
                )
            unknown_views = sorted(set(view_ids) - set(pose_by_view))
            if unknown_views:
                raise ValueError(
                    f"independent regions reference unregistered views: {unknown_views}"
                )

        anchor_fusion: SphericalAnchorFusionResult | None = None
        if self.config.enable_anchor_fusion and independent_regions:
            anchors = []
            for hypothesis in sorted(
                generation.hypotheses, key=lambda item: item.hypothesis_id
            ):
                for region in independent_regions.get(hypothesis.hypothesis_id, ()):
                    anchors.extend(
                        generate_spherical_fov_anchors(
                            region,
                            concept_id=hypothesis.concept_id,
                            vocabulary=hypothesis.vocabulary,
                            score=hypothesis.semantic_score,
                            anchor_prefix=hypothesis.hypothesis_id,
                            scale_factors=self.config.anchor_scale_factors,
                            aspect_ratios=self.config.anchor_aspect_ratios,
                            minimum_fov_deg=(
                                self.config.anchor_fusion.minimum_final_fov_deg
                            ),
                            maximum_fov_deg=(
                                self.config.anchor_fusion.maximum_final_fov_deg
                            ),
                            source_hypothesis_ids=(hypothesis.hypothesis_id,),
                        )
                    )
            anchor_fusion = fuse_registered_spherical_anchors(
                tuple(anchors),
                poses=resolved_poses,
                radial_depth=radial_depth,
                pair_cues=anchor_pair_cues,
                config=self.config.anchor_fusion,
            )

        region_evaluations: list[HypothesisRegionEvaluation] = []
        view_evaluations: list[HypothesisViewPairEvaluation] = []
        transfer_evaluations: list[HypothesisTransferEvaluation] = []
        for hypothesis in generation.hypotheses:
            concept = concept_by_id[hypothesis.concept_id]
            for region in hypothesis.view_regions:
                evidence = evidence_by_view.get(region.view_id)
                if evidence is None:
                    continue
                region_evaluations.append(
                    HypothesisRegionEvaluation(
                        hypothesis_id=hypothesis.hypothesis_id,
                        view_id=region.view_id,
                        metrics=evaluate_spherical_region(evidence, concept, region),
                    )
                )
            for consistency in hypothesis_view_consistency(
                hypothesis,
                embeddings.get(hypothesis.hypothesis_id, ()),
                reprojection_sigma_deg=self.config.reprojection_sigma_deg,
            ):
                view_evaluations.append(
                    HypothesisViewPairEvaluation(
                        hypothesis_id=hypothesis.hypothesis_id,
                        consistency=consistency,
                    )
                )
            if self.config.enable_registered_transfer:
                regions = {
                    item.view_id: item
                    for item in independent_regions.get(hypothesis.hypothesis_id, ())
                }
                transferable = sorted(set(regions) & set(depth_by_view))
                for view_a, view_b in combinations(transferable, 2):
                    transfer = symmetric_spherical_region_transfer(
                        regions[view_a],
                        regions[view_b],
                        depth_by_view[view_a],
                        depth_by_view[view_b],
                        pose_by_view[view_a],
                        pose_by_view[view_b],
                    )
                    transfer_evaluations.append(
                        self._compact_transfer(
                            hypothesis.hypothesis_id,
                            hypothesis.hypothesis_id,
                            view_a,
                            view_b,
                            "within-hypothesis",
                            transfer,
                        )
                    )

        cue_by_pair = {
            (item.hypothesis_id_a, item.hypothesis_id_b): item for item in pair_cues
        }
        if len(cue_by_pair) != len(pair_cues):
            raise ValueError("pair cues must identify unique pairs")
        if self.config.enable_registered_transfer and len(depth_by_view) >= 2:
            for first, second in combinations(
                sorted(
                    generation.hypotheses,
                    key=lambda item: item.hypothesis_id,
                ),
                2,
            ):
                if (
                    first.vocabulary != second.vocabulary
                    or first.concept_id != second.concept_id
                ):
                    continue
                regions_first = {
                    item.view_id: item
                    for item in independent_regions.get(first.hypothesis_id, ())
                }
                regions_second = {
                    item.view_id: item
                    for item in independent_regions.get(second.hypothesis_id, ())
                }
                transferable = sorted(
                    set(regions_first) & set(regions_second) & set(depth_by_view)
                )
                scores: list[float] = []
                for view_a, view_b in combinations(transferable, 2):
                    for region_a, region_b in (
                        (regions_first[view_a], regions_second[view_b]),
                        (regions_first[view_b], regions_second[view_a]),
                    ):
                        transfer = symmetric_spherical_region_transfer(
                            region_a,
                            region_b,
                            depth_by_view[region_a.view_id],
                            depth_by_view[region_b.view_id],
                            pose_by_view[region_a.view_id],
                            pose_by_view[region_b.view_id],
                        )
                        scores.append(transfer.symmetric_solid_angle_iou)
                        transfer_evaluations.append(
                            self._compact_transfer(
                                first.hypothesis_id,
                                second.hypothesis_id,
                                region_a.view_id,
                                region_b.view_id,
                                "reduction-cue",
                                transfer,
                            )
                        )
                if scores:
                    pair = (first.hypothesis_id, second.hypothesis_id)
                    score = float(np.median(scores))
                    existing = cue_by_pair.get(pair)
                    if existing is None:
                        cue_by_pair[pair] = HypothesisPairCues(
                            *pair,
                            transfer_iou=score,
                            soft_nms_overlap=score,
                            provenance={
                                "automatic_transfer": "median-registered-radial-depth"
                            },
                        )
                    else:
                        provenance = dict(existing.provenance)
                        provenance["automatic_transfer"] = (
                            "median-registered-radial-depth"
                        )
                        cue_by_pair[pair] = replace(
                            existing,
                            transfer_iou=(
                                existing.transfer_iou
                                if existing.transfer_iou is not None
                                else score
                            ),
                            soft_nms_overlap=(
                                existing.soft_nms_overlap
                                if existing.soft_nms_overlap is not None
                                else score
                            ),
                            provenance=provenance,
                        )

        reduction: HypothesisReductionResult | None = None
        if self.config.enable_reduction:
            geometric_quality = dict(geometric_quality_by_hypothesis or {})
            view_factor = dict(independent_view_factor_by_hypothesis or {})
            unknown_quality = sorted(
                (set(geometric_quality) | set(view_factor)) - hypothesis_ids
            )
            if unknown_quality:
                raise ValueError(
                    f"reduction weights reference unknown hypotheses: {unknown_quality}"
                )
            reduction = reduce_object_hypotheses(
                tuple(
                    HypothesisReductionCandidate(
                        item,
                        geometric_quality=geometric_quality.get(
                            item.hypothesis_id, 1.0
                        ),
                        independent_view_factor=view_factor.get(
                            item.hypothesis_id, 1.0
                        ),
                    )
                    for item in generation.hypotheses
                ),
                pair_cues=tuple(cue_by_pair[pair] for pair in sorted(cue_by_pair)),
                config=self.config.affinity,
            )

        region_evaluations.sort(key=lambda item: (item.hypothesis_id, item.view_id))
        view_evaluations.sort(
            key=lambda item: (
                item.hypothesis_id,
                item.consistency.view_id_a,
                item.consistency.view_id_b,
            )
        )
        transfer_evaluations.sort(
            key=lambda item: (
                item.role,
                item.hypothesis_id_a,
                item.hypothesis_id_b,
                item.view_id_a,
                item.view_id_b,
            )
        )
        return SphericalObjectHypothesisMethodResult(
            generation=generation,
            region_evaluations=tuple(region_evaluations),
            view_pair_evaluations=tuple(view_evaluations),
            transfer_evaluations=tuple(transfer_evaluations),
            anchor_fusion=anchor_fusion,
            reduction_pair_cues=tuple(
                cue_by_pair[pair] for pair in sorted(cue_by_pair)
            ),
            reduction=reduction,
            diagnostics={
                "geometry_source": geometry_source,
                "generated_hypothesis_count": len(generation.hypotheses),
                "region_evaluation_count": len(region_evaluations),
                "view_pair_evaluation_count": len(view_evaluations),
                "transfer_evaluation_count": len(transfer_evaluations),
                "reduction_pair_cue_count": len(cue_by_pair),
                "radial_depth_view_count": len(depth_by_view),
                "independent_region_hypothesis_count": len(independent_regions),
                "anchor_input_count": (
                    0
                    if anchor_fusion is None
                    else int(anchor_fusion.diagnostics["input_anchor_count"])
                ),
                "anchor_cluster_count": (
                    0
                    if anchor_fusion is None
                    else int(anchor_fusion.diagnostics["cluster_count"])
                ),
                "confirmed_anchor_cluster_count": (
                    0
                    if anchor_fusion is None
                    else int(anchor_fusion.diagnostics["confirmed_cluster_count"])
                ),
                "transient_transfer_masks_retained": 0,
            },
        )

    def describe(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "stability": "experimental",
                "geometry": "refined-sparse-or-registered-bearing-triangulation",
                "semantics": "precomputed-dense-directional-evidence",
                "depth": "optional-explicit-radial-range",
                "reduction": "affinity-fusion-and-soft-nms",
                "anchor_fusion": "complete-link-and-registered-consensus-refit",
                "segmentation": "optional-downstream-not-owned",
            }
        )
