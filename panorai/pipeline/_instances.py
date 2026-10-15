"""Composable CAM-to-region-to-mask stages for spherical instances.

The stages mirror the separation of responsibilities in region-based
detectors without claiming architectural equivalence to a trained Mask R-CNN:
CAM supplies proposal evidence, gnomonic charts define transient RoIs, a
promptable mask backend refines boundaries, and spherical fusion resolves
instances on the canonical ERP grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from ._fields import field_from_container
from ._results import StageResult


def _nearest_mask(mask: Any, shape_hw: tuple[int, int]) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape == shape_hw:
        return values.copy()
    if values.ndim != 2:
        raise ValueError("support and validity must be HW masks")
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


def _coerce_evidence(
    value: Any,
    *,
    concept_ids: tuple[str, ...],
    support: np.ndarray,
    vocabulary: str,
    provenance: Mapping[str, Any],
) -> Any:
    from panorai.experimental.deep_learning.segmentation import DenseSemanticEvidence

    if isinstance(value, DenseSemanticEvidence):
        scores = np.array(value.scores, copy=True)
        input_support = _nearest_mask(support, tuple(scores.shape[-2:]))
        resolved_support = input_support & np.asarray(value.support, dtype=bool)
        if not np.any(resolved_support):
            raise ValueError("CAM evidence support has no overlap with input support")
        scores[:, ~resolved_support] = 0
        return DenseSemanticEvidence(
            value.concept_ids,
            scores,
            resolved_support,
            vocabulary=value.vocabulary,
            provenance={**dict(provenance), **dict(value.provenance)},
        )
    if isinstance(value, Mapping):
        if "scores" not in value or "concept_ids" not in value:
            raise ValueError(
                "CAM mappings require 'concept_ids' and 'scores'; support is "
                "optional and is never inferred from score values"
            )
        scores = np.array(value["scores"], copy=True)
        if scores.ndim != 3:
            raise ValueError("CAM mapping scores must be CHW")
        input_support = _nearest_mask(support, tuple(scores.shape[-2:]))
        declared_support = np.asarray(value.get("support", input_support), dtype=bool)
        if declared_support.shape != scores.shape[-2:]:
            raise ValueError("CAM mapping support must match score HW")
        resolved_support = input_support & declared_support
        if not np.any(resolved_support):
            raise ValueError("CAM evidence support has no overlap with input support")
        scores[:, ~resolved_support] = 0
        return DenseSemanticEvidence(
            tuple(value["concept_ids"]),
            scores,
            resolved_support,
            vocabulary=str(value.get("vocabulary", vocabulary)),
            provenance={**dict(provenance), **dict(value.get("provenance", {}))},
        )
    if isinstance(value, tuple) and len(value) == 2:
        scores, explicit_support = value
    else:
        scores, explicit_support = value, None
    scores = np.array(scores, copy=True)
    if scores.ndim != 3:
        raise ValueError(
            "a CAM predictor must return DenseSemanticEvidence, a mapping, "
            "(scores, support), or a CHW score array"
        )
    if not concept_ids:
        raise ValueError(
            "concept_ids are required when the CAM predictor returns arrays"
        )
    input_support = _nearest_mask(support, tuple(scores.shape[-2:]))
    declared_support = (
        input_support
        if explicit_support is None
        else np.asarray(explicit_support, dtype=bool)
    )
    if declared_support.shape != scores.shape[-2:]:
        raise ValueError("CAM evidence support must match score HW")
    resolved_support = input_support & declared_support
    if not np.any(resolved_support):
        raise ValueError("CAM evidence support has no overlap with input support")
    scores[:, ~resolved_support] = 0
    return DenseSemanticEvidence(
        concept_ids,
        scores,
        resolved_support,
        vocabulary=vocabulary,
        provenance=dict(provenance),
    )


@dataclass(frozen=True, slots=True)
class CAMEvidenceFrame:
    """One ERP plus explicit dense semantic evidence and source masks."""

    panorama: Any
    evidence: Any
    source_field: str
    support: np.ndarray
    validity: np.ndarray

    interface = "panorai-cam-evidence-frame/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        support = np.array(self.support, dtype=bool, copy=True)
        validity = np.array(self.validity, dtype=bool, copy=True)
        if support.ndim != 2 or validity.shape != support.shape:
            raise ValueError(
                "CAM source support and validity must be matching HW masks"
            )
        support.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "validity", validity)

    @property
    def eligible(self) -> np.ndarray:
        """Return a writable mask valid for learned evidence and prompting."""

        return np.asarray(self.support & self.validity, dtype=bool).copy()


@dataclass(frozen=True, slots=True)
class CAMRegionProposal:
    """A CAM-derived spherical search region, never an object mask."""

    proposal_id: str
    seed: Any
    spec: Any
    score: float
    concept_id: str

    interface = "panorai-cam-region-proposal/v1-experimental"
    stability = "experimental"


@dataclass(frozen=True, slots=True)
class CAMRegionProposalSet:
    """CAM proposal families tied to their evidence-bearing panorama."""

    frame: CAMEvidenceFrame
    proposals: tuple[CAMRegionProposal, ...]
    parameters: Mapping[str, Any]

    interface = "panorai-cam-region-proposals/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if len({item.proposal_id for item in self.proposals}) != len(self.proposals):
            raise ValueError("CAM proposal identifiers must be unique")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def seeds(self) -> tuple[Any, ...]:
        return tuple(item.seed for item in self.proposals)


@dataclass(frozen=True, slots=True)
class SAMMaskProposalBatch:
    """Unfused SAM alternatives linked back to their CAM proposal set."""

    regions: CAMRegionProposalSet
    masks: Any

    interface = "panorai-sam-mask-proposal-batch/v1-experimental"
    stability = "experimental"


@dataclass(frozen=True, slots=True)
class SphericalCAM:
    """Run one direct-spherical CAM adapter and preserve its ERP context.

    ``predictor`` receives the selected source array. It may return a
    ``DenseSemanticEvidence`` object, a small mapping, ``(scores, support)``,
    or a CHW score array. Array returns require ``concept_ids``. Derived
    support comes only from the explicit input masks, never from score values.
    """

    predictor: Any
    concept_ids: tuple[str, ...] = ()
    input: str = "image"
    vocabulary: str = "custom-cam/v1"

    input_kind = "erp"
    output_kind = "cam_evidence"
    interface = "panorai-spherical-cam-stage/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if not callable(self.predictor):
            raise TypeError("SphericalCAM predictor must be callable")
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError("SphericalCAM input must be a non-empty field name")
        if not isinstance(self.vocabulary, str) or not self.vocabulary.strip():
            raise ValueError("SphericalCAM vocabulary must be non-empty")
        concepts = tuple(str(item).strip() for item in self.concept_ids)
        if any(not item for item in concepts) or len(set(concepts)) != len(concepts):
            raise ValueError("concept_ids must be unique non-empty strings")
        object.__setattr__(self, "concept_ids", concepts)

    def execute(self, value: Any) -> StageResult:
        field = field_from_container(value, self.input)
        if field.grid.domain != "erp":
            raise TypeError("SphericalCAM requires an ERP-aligned source field")
        if field.kind in {"labels", "mask", "depth"}:
            raise TypeError("SphericalCAM requires image-like continuous input")
        result = self.predictor(field.value)
        evidence = _coerce_evidence(
            result,
            concept_ids=self.concept_ids,
            support=np.asarray(field.support, dtype=bool),
            vocabulary=self.vocabulary,
            provenance={
                "pipeline_stage": "SphericalCAM",
                "predictor": type(self.predictor).__name__,
                "source_field": self.input,
            },
        )
        frame = CAMEvidenceFrame(
            panorama=value,
            evidence=evidence,
            source_field=self.input,
            support=np.asarray(field.support, dtype=bool),
            validity=np.asarray(field.validity, dtype=bool),
        )
        return StageResult(
            frame,
            {
                "domain": "sphere",
                "role": "shared-backbone-and-cam-head",
                "source_field": self.input,
                "concept_ids": list(evidence.concept_ids),
                "evidence_shape_hw": list(evidence.scores.shape[-2:]),
                "support_source": "explicit-raster-field",
            },
        )


@dataclass(frozen=True, slots=True)
class CAMRegionProposals:
    """Turn CAM maxima into geodesically separated spherical RoI families."""

    maximum_per_concept: int = 12
    minimum_separation_degrees: float = 12.0
    relative_threshold: float = 0.30
    minimum_score: float = 0.25
    fov_degrees: float = 70.0
    face_size: int = 1024

    input_kind = "cam_evidence"
    output_kind = "region_proposals"
    interface = "panorai-cam-region-proposal-stage/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_per_concept, bool)
            or not isinstance(self.maximum_per_concept, int)
            or self.maximum_per_concept < 1
        ):
            raise ValueError("maximum_per_concept must be a positive integer")
        if not 0.0 <= float(self.relative_threshold) <= 1.0:
            raise ValueError("relative_threshold must be within [0,1]")
        if not 0.0 <= float(self.minimum_score) <= 1.0:
            raise ValueError("minimum_score must be within [0,1]")
        if not 0.0 < float(self.minimum_separation_degrees) <= 180.0:
            raise ValueError("minimum_separation_degrees must be within (0,180]")
        if not 0.0 < float(self.fov_degrees) < 180.0:
            raise ValueError("fov_degrees must be within (0,180)")
        if (
            isinstance(self.face_size, bool)
            or not isinstance(self.face_size, int)
            or self.face_size < 1
        ):
            raise ValueError("face_size must be a positive integer")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, CAMEvidenceFrame):
            raise TypeError("CAMRegionProposals requires a CAMEvidenceFrame")
        from panorai.experimental.deep_learning.segmentation import (
            select_semantic_seeds,
        )
        from panorai.geometry import GnomonicSpec

        seeds = select_semantic_seeds(
            value.evidence,
            maximum_per_concept=self.maximum_per_concept,
            minimum_separation_degrees=self.minimum_separation_degrees,
            relative_threshold=self.relative_threshold,
        )
        seeds = tuple(seed for seed in seeds if seed.score >= self.minimum_score)
        proposals = tuple(
            CAMRegionProposal(
                proposal_id=f"cam-region-{index:06d}",
                seed=seed,
                spec=GnomonicSpec(
                    center_lon_deg=seed.longitude_degrees,
                    center_lat_deg=seed.latitude_degrees,
                    hfov_deg=self.fov_degrees,
                    vfov_deg=self.fov_degrees,
                    output_shape_hw=(self.face_size, self.face_size),
                ),
                score=seed.score,
                concept_id=str(seed.concept_id),
            )
            for index, seed in enumerate(seeds)
        )
        result = CAMRegionProposalSet(
            value,
            proposals,
            {
                "maximum_per_concept": self.maximum_per_concept,
                "minimum_separation_degrees": self.minimum_separation_degrees,
                "relative_threshold": self.relative_threshold,
                "minimum_score": self.minimum_score,
                "fov_degrees": self.fov_degrees,
                "face_size": self.face_size,
            },
        )
        return StageResult(
            result,
            {
                "domain": "sphere",
                "role": "cam-proposal-generator",
                "proposal_count": len(proposals),
                "proposal_semantics": "search-regions-not-masks",
                "component_prompting": "resolved-inside-each-transient-roi",
            },
        )


@dataclass(frozen=True, slots=True)
class SAMMaskRefinement:
    """Refine CAM regions through transient gnomonic charts and a SAM backend."""

    backend: Any
    config: Any = None
    output_shape_hw: tuple[int, int] | None = None
    expand: bool = True

    input_kind = "region_proposals"
    output_kind = "mask_proposals"
    interface = "panorai-sam-mask-refinement-stage/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if not all(
            callable(getattr(self.backend, name, None))
            for name in ("encode", "predict", "release")
        ):
            raise TypeError(
                "SAMMaskRefinement backend must provide encode, predict, and release"
            )
        if not isinstance(self.expand, bool):
            raise TypeError("expand must be boolean")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, CAMRegionProposalSet):
            raise TypeError("SAMMaskRefinement requires CAMRegionProposalSet")
        from panorai.experimental.deep_learning.segmentation import (
            SphericalSegmentationConfig,
            SphericalSemanticSegmenter,
        )

        field = field_from_container(value.frame.panorama, value.frame.source_field)
        rgb = np.asarray(field.value)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAMMaskRefinement requires HWC three-channel RGB")
        resolved_config = self.config or SphericalSegmentationConfig()
        if value.proposals:
            spec = value.proposals[0].spec
            if any(
                item.spec.hfov_deg != spec.hfov_deg
                or item.spec.vfov_deg != spec.vfov_deg
                or item.spec.output_shape_hw != spec.output_shape_hw
                for item in value.proposals
            ):
                raise ValueError(
                    "SAMMaskRefinement currently requires one shared RoI FOV and shape"
                )
            if (
                spec.hfov_deg != spec.vfov_deg
                or spec.output_shape_hw[0] != spec.output_shape_hw[1]
            ):
                raise ValueError(
                    "SAMMaskRefinement currently requires square equal-FOV RoIs"
                )
            resolved_config = replace(
                resolved_config,
                discovery_fov_degrees=float(spec.hfov_deg),
                discovery_face_size=int(spec.output_shape_hw[0]),
            )
        proposal_set = SphericalSemanticSegmenter(
            self.backend, resolved_config
        ).propose(
            rgb,
            value.frame.eligible,
            seeds=value.seeds,
            evidence=value.frame.evidence,
            output_shape_hw=self.output_shape_hw,
            expand=self.expand,
        )
        result = SAMMaskProposalBatch(value, proposal_set)
        return StageResult(
            result,
            {
                "domain": "hybrid",
                "role": "promptable-roi-mask-head",
                "roi_projection": "transient-gnomonic",
                "input_region_count": len(value.proposals),
                "mask_proposal_count": len(proposal_set.proposals),
                "alternatives_per_proposal": 3,
                "expanded": self.expand,
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalInstanceFusion:
    """Fuse SAM alternatives into supported spherical instance masks."""

    config: Any = None

    input_kind = "mask_proposals"
    output_kind = "result"
    interface = "panorai-spherical-instance-fusion-stage/v1-experimental"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SAMMaskProposalBatch):
            raise TypeError("SphericalInstanceFusion requires SAMMaskProposalBatch")
        from panorai.experimental.deep_learning.segmentation import (
            SphericalSemanticSegmenter,
        )

        result = SphericalSemanticSegmenter(None, self.config).fuse(
            value.masks,
            evidence=value.regions.frame.evidence,
        )
        return StageResult(
            result,
            {
                "domain": "sphere",
                "role": "instance-fusion-and-panoptic-resolution",
                "input_proposal_count": len(value.masks.proposals),
                "instance_count": len(result.segments),
                "support_semantics": "explicit-source-support-and-validity",
            },
        )


class CAMGuidedSphericalInstances:
    """Convenience constructor for the explicit CAM→RoI→SAM→fusion pipeline.

    The returned object delegates the complete :class:`Pipeline` interface.
    It is Mask-R-CNN-shaped in stage responsibility, but CAM replaces the RPN
    and an injected SAM backend replaces a trained per-class mask head.
    """

    interface = "panorai-cam-guided-spherical-instances/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        *,
        cam: Any,
        sam: Any,
        concept_ids: tuple[str, ...] = (),
        proposals: CAMRegionProposals | None = None,
        config: Any = None,
        output_shape_hw: tuple[int, int] | None = None,
        expand: bool = True,
        input: str = "image",
        vocabulary: str = "custom-cam/v1",
        memory: Any = None,
    ) -> None:
        from ._pipeline import Pipeline

        self.pipeline = Pipeline(
            (
                "cam",
                SphericalCAM(
                    cam,
                    concept_ids=concept_ids,
                    input=input,
                    vocabulary=vocabulary,
                ),
            ),
            ("proposals", proposals or CAMRegionProposals()),
            (
                "sam",
                SAMMaskRefinement(
                    sam,
                    config=config,
                    output_shape_hw=output_shape_hw,
                    expand=expand,
                ),
            ),
            ("instances", SphericalInstanceFusion(config=config)),
            memory=memory,
        )

    @property
    def named_steps(self) -> Mapping[str, Any]:
        return self.pipeline.named_steps

    @property
    def input_kind(self) -> str:
        return self.pipeline.input_kind

    @property
    def output_kind(self) -> str:
        return self.pipeline.output_kind

    def __call__(self, value: Any) -> Any:
        return self.pipeline(value)

    def run(self, value: Any, *, retain: Any = ()) -> Any:
        return self.pipeline.run(value, retain=retain)

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        return self.pipeline.get_params(deep=deep)

    def with_params(self, **params: Any) -> "CAMGuidedSphericalInstances":
        """Return a configured wrapper instead of leaking the inner Pipeline."""

        variant = object.__new__(type(self))
        variant.pipeline = self.pipeline.with_params(**params)
        return variant

    def describe(self) -> dict[str, Any]:
        description = self.pipeline.describe()
        return {
            **description,
            "interface": self.interface,
            "stability": self.stability,
            "architecture": "cam-proposals-transient-roi-sam-mask-fusion",
            "mask_rcnn_analogy": {
                "shared_features": "direct-spherical CAM predictor",
                "proposal_generator": "CAMRegionProposals",
                "roi_operation": "transient canonical gnomonic projection",
                "mask_head": "injected SAM backend",
                "reduction": "solid-angle spherical instance fusion",
            },
        }


__all__ = [
    "CAMEvidenceFrame",
    "CAMGuidedSphericalInstances",
    "CAMRegionProposal",
    "CAMRegionProposalSet",
    "CAMRegionProposals",
    "SAMMaskProposalBatch",
    "SAMMaskRefinement",
    "SphericalCAM",
    "SphericalInstanceFusion",
]
