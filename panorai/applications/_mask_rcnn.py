"""Executable spherical instance method organized by Mask R-CNN roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from panorai.data import EquirectangularImage
from panorai.experimental.deep_learning.mask_rcnn._architecture import (
    Backbone,
    BoxHead,
    MaskHead,
    MaskRCNNArchitecture,
    ProposalGenerator,
    RoIHeads,
)
from panorai.experimental.deep_learning.segmentation import (
    SphericalSegmentationConfig,
    SphericalSegmentationResult,
    SphericalSemanticSegmenter,
)
from panorai.pipeline import (
    CAMRegionProposals,
    Pipeline,
    SAMMaskRefinement,
    SphericalCAM,
    SphericalInstanceFusion,
    StageResult,
)
from panorai.pipeline._fields import field_from_container

from ._base import PipelineApplication, application_config


@dataclass(frozen=True, slots=True)
class InstanceHeuristicDecision:
    """One abstaining or accepted instance decision made by a heuristic."""

    class_id: int | None
    label: str | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.class_id is not None and (
            isinstance(self.class_id, bool) or self.class_id < 1
        ):
            raise ValueError("class_id must be positive or None for abstention")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be finite and within [0,1]")
        if self.class_id is not None and (self.label is None or not self.label.strip()):
            raise ValueError("accepted decisions require a non-empty label")


def _coerce_decision(value: Any) -> InstanceHeuristicDecision:
    if isinstance(value, InstanceHeuristicDecision):
        return value
    if value is None:
        return InstanceHeuristicDecision(None)
    if isinstance(value, Mapping):
        return InstanceHeuristicDecision(
            value.get("class_id"), value.get("label"), value.get("score", 0.0)
        )
    raise TypeError(
        "instance heuristic must return InstanceHeuristicDecision, a mapping, or None"
    )


@dataclass(frozen=True, slots=True)
class _InstanceHeuristicStage:
    heuristic: Any
    num_classes: int

    input_kind = "result"
    output_kind = "result"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SphericalSegmentationResult):
            raise TypeError("BoxHead heuristic requires SphericalSegmentationResult")
        decisions: dict[int, dict[str, Any]] = {}
        segments = []
        for segment in value.segments:
            decision = _coerce_decision(self.heuristic(segment))
            if decision.class_id is not None and decision.class_id >= self.num_classes:
                raise ValueError(
                    f"heuristic class_id {decision.class_id} is outside num_classes="
                    f"{self.num_classes} (class 0 is background)"
                )
            decisions[segment.segment_id] = {
                "class_id": decision.class_id,
                "label": decision.label,
                "score": float(decision.score),
                "abstained": decision.class_id is None,
            }
            segments.append(
                replace(
                    segment,
                    concept_id=(
                        None
                        if decision.class_id is None
                        else f"class:{decision.class_id}"
                    ),
                    concept_name=(
                        "unknown" if decision.class_id is None else str(decision.label)
                    ),
                    semantic_score=float(decision.score),
                )
            )
        diagnostics = dict(value.diagnostics)
        diagnostics["box_head"] = {
            "kind": "instance-heuristic",
            "num_classes_including_background": self.num_classes,
            "decisions": decisions,
            "accepted": sum(not item["abstained"] for item in decisions.values()),
            "abstained": sum(item["abstained"] for item in decisions.values()),
        }
        result = SphericalSegmentationResult(
            tuple(segments),
            value.proposals,
            value.panoptic_map,
            value.support,
            diagnostics,
        )
        return StageResult(
            result,
            {
                "domain": "sphere",
                "role": "box-head-instance-heuristic",
                "instance_count": len(segments),
            },
        )


@dataclass(frozen=True, slots=True)
class _SAMProposalStage:
    backend: Any
    config: SphericalSegmentationConfig
    output_shape_hw: tuple[int, int] | None

    input_kind = "erp"
    output_kind = "result"

    def execute(self, value: Any) -> StageResult:
        field = field_from_container(value, "image")
        rgb = np.asarray(field.value)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM proposal mode requires HWC three-channel RGB")
        eligible = np.asarray(field.support, dtype=bool) & np.asarray(
            field.validity, dtype=bool
        )
        result = SphericalSemanticSegmenter(self.backend, self.config).segment(
            rgb,
            eligible,
            output_shape_hw=self.output_shape_hw,
        )
        return StageResult(
            result,
            {
                "domain": "hybrid",
                "role": "sam-proposal-generator-and-mask-head",
                "proposal_count": len(result.proposals),
                "reuse_proposal_masks": True,
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalMaskRCNNWorkflow(PipelineApplication):
    """Executable Mask R-CNN workflow bound to CAM/SAM/heuristics.

    ``proposal_mode='cam'`` uses CAM only to choose spherical search regions;
    SAM refines the masks. ``proposal_mode='sam'`` lets SAM generate and reuse
    proposal masks. The BoxHead responsibility is fulfilled by an explicit
    abstaining instance heuristic, not advertised as a trained classifier.
    """

    proposal_mode: str
    proposal: Any
    heuristic: Any
    num_classes: int
    mask: Any = None
    concepts: tuple[str, ...] = ()
    reuse_proposal_masks: bool = False
    config: SphericalSegmentationConfig = SphericalSegmentationConfig()
    cam_proposals: CAMRegionProposals = CAMRegionProposals()
    output_shape_hw: tuple[int, int] | None = None
    expand: bool = True

    application_id = "panorai.applications.SphericalMaskRCNNWorkflow"
    legacy_application_ids = ("panorai.applications.SphericalMaskRCNNPaper",)

    def __post_init__(self) -> None:
        if self.proposal_mode not in {"cam", "sam"}:
            raise ValueError("proposal_mode must be 'cam' or 'sam'")
        if isinstance(self.num_classes, bool) or self.num_classes < 2:
            raise ValueError("num_classes must include background and be at least 2")
        if not callable(self.heuristic):
            raise TypeError("heuristic must be callable for instance BoxHead decisions")
        concepts = tuple(str(item).strip() for item in self.concepts)
        if any(not item for item in concepts) or len(set(concepts)) != len(concepts):
            raise ValueError("concepts must be unique non-empty strings")
        object.__setattr__(self, "concepts", concepts)
        if self.proposal_mode == "cam":
            if not callable(self.proposal):
                raise TypeError(
                    "CAM proposal mode requires a callable proposal adapter"
                )
            self._validate_sam(self.mask, parameter="mask")
            if self.reuse_proposal_masks:
                raise ValueError(
                    "reuse_proposal_masks applies only to SAM proposal mode"
                )
        else:
            self._validate_sam(self.proposal, parameter="proposal")
            if not self.reuse_proposal_masks:
                raise ValueError(
                    "SAM proposal mode currently requires reuse_proposal_masks=True; "
                    "a second mask-refinement adapter is not inferred silently"
                )
            if self.mask is not None:
                raise ValueError(
                    "mask must be omitted when SAM proposal masks are reused"
                )

    @staticmethod
    def _validate_sam(value: Any, *, parameter: str) -> None:
        missing = [
            name
            for name in ("encode", "predict", "release")
            if not callable(getattr(value, name, None))
        ]
        if missing:
            raise TypeError(
                f"{parameter} must provide the SAM adapter methods "
                f"encode/predict/release; missing {missing}"
            )

    @property
    def architecture(self) -> MaskRCNNArchitecture:
        mask_component = self.mask if self.proposal_mode == "cam" else self.proposal
        return MaskRCNNArchitecture(
            backbone=Backbone(
                self.proposal if self.proposal_mode == "cam" else None,
                trainable=False,
                domain="sphere" if self.proposal_mode == "cam" else "identity",
            ),
            proposal_generator=ProposalGenerator(
                self.cam_proposals if self.proposal_mode == "cam" else self.proposal,
                mode=self.proposal_mode,
                trainable=False,
            ),
            roi_heads=RoIHeads(
                BoxHead(self.heuristic, self.num_classes),
                MaskHead(
                    mask_component,
                    reuse_proposal_masks=self.reuse_proposal_masks,
                ),
            ),
            execution="mask-rcnn-workflow-cam-sam-heuristic",
        )

    def as_pipeline(self) -> Pipeline:
        heuristic = _InstanceHeuristicStage(self.heuristic, self.num_classes)
        if self.proposal_mode == "sam":
            return Pipeline(
                (
                    "proposals",
                    _SAMProposalStage(self.proposal, self.config, self.output_shape_hw),
                ),
                ("box_head", heuristic),
            )
        return Pipeline(
            (
                "backbone_cam",
                SphericalCAM(self.proposal, concept_ids=self.concepts),
            ),
            ("proposal_generator", self.cam_proposals),
            (
                "mask_head",
                SAMMaskRefinement(
                    self.mask,
                    config=self.config,
                    output_shape_hw=self.output_shape_hw,
                    expand=self.expand,
                ),
            ),
            ("fusion", SphericalInstanceFusion(config=self.config)),
            ("box_head", heuristic),
        )

    def run(
        self,
        panorama: Any,
        *,
        support_mask: np.ndarray | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalSegmentationResult:
        if isinstance(panorama, np.ndarray):
            panorama = EquirectangularImage(
                panorama, support_mask=support_mask, valid=validity_mask
            )
        elif support_mask is not None or validity_mask is not None:
            raise ValueError(
                "support_mask/validity_mask are accepted only with array input"
            )
        result = self.as_pipeline()(panorama)
        if not isinstance(result, SphericalSegmentationResult):
            raise RuntimeError(
                "spherical Mask R-CNN workflow returned an unexpected result"
            )
        return result

    def describe(self) -> dict[str, Any]:
        architecture = self.architecture
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "method": "mask-rcnn-workflow-cam-sam-instance-heuristic",
            "proposal_mode": self.proposal_mode,
            "num_classes_including_background": self.num_classes,
            "concepts": list(self.concepts),
            "roles": {
                "backbone": architecture.backbone.domain,
                "proposal_generator": architecture.proposal_generator.mode,
                "roi_align": architecture.roi_heads.roi_align.domain,
                "box_head": architecture.roi_heads.box_head.responsibility,
                "mask_head": (
                    "reuse-sam-proposals"
                    if self.reuse_proposal_masks
                    else "sam-region-refinement"
                ),
            },
            "external_requirements": [
                "user-supplied model adapters and weights",
                "their licenses must be accepted independently",
            ],
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(self.application_id, self.get_params(deep=False))


# Compatibility alias for the public Experimental name used before the workflow
# terminology became canonical.  Keep this as object identity, rather than a
# subclass, so isinstance checks and immutable ``with_params`` variants remain
# compatible.
SphericalMaskRCNNPaper = SphericalMaskRCNNWorkflow


__all__ = [
    "InstanceHeuristicDecision",
    "SphericalMaskRCNNPaper",
    "SphericalMaskRCNNWorkflow",
]
