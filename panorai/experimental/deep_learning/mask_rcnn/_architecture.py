"""Inspectable Mask R-CNN taxonomy for spherical execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Backbone:
    component: Any
    trainable: bool
    domain: str = "sphere"


@dataclass(frozen=True, slots=True)
class ProposalGenerator:
    component: Any
    mode: str
    trainable: bool


@dataclass(frozen=True, slots=True)
class RegionProposalNetwork(ProposalGenerator):
    """Canonical trainable RPN role; execution is intentionally unbound."""

    mode: str = "rpn"
    trainable: bool = True


@dataclass(frozen=True, slots=True)
class RoIAlign:
    domain: str = "transient-gnomonic"
    coordinate_contract: str = "panorai-geometry-v1"


@dataclass(frozen=True, slots=True)
class BoxHead:
    component: Any
    num_classes: int
    responsibility: str = "instance-classification-heuristic"
    trainable: bool = False


@dataclass(frozen=True, slots=True)
class MaskHead:
    component: Any
    reuse_proposal_masks: bool
    trainable: bool = False


@dataclass(frozen=True, slots=True)
class RoIHeads:
    box_head: BoxHead
    mask_head: MaskHead
    roi_align: RoIAlign = RoIAlign()


@dataclass(frozen=True, slots=True)
class MaskRCNNArchitecture:
    """Structural roles from Mask R-CNN, independent of their implementation."""

    backbone: Backbone
    proposal_generator: ProposalGenerator
    roi_heads: RoIHeads
    execution: str


__all__ = [
    "Backbone",
    "BoxHead",
    "MaskHead",
    "MaskRCNNArchitecture",
    "ProposalGenerator",
    "RegionProposalNetwork",
    "RoIAlign",
    "RoIHeads",
]
