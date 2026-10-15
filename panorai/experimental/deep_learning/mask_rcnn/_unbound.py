"""Non-executable placeholder for a future trainable spherical Mask R-CNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._architecture import (
    Backbone,
    BoxHead,
    MaskHead,
    MaskRCNNArchitecture,
    RegionProposalNetwork,
    RoIHeads,
)


class UnboundMaskRCNNError(RuntimeError):
    """The trainable architecture has roles but no implemented execution."""


@dataclass(frozen=True, slots=True)
class SphericalMaskRCNN:
    """Future trainable spherical Mask R-CNN, explicitly not executable.

    This object documents the conventional Backbone→RPN→RoIAlign→Box/Mask
    topology.  It is not a trainer, checkpoint loader, or inference model.
    """

    num_classes: int
    backbone: Any = None
    rpn: Any = None
    box_head: Any = None
    mask_head: Any = None

    interface = "panorai-spherical-mask-rcnn/v1-experimental-unbound"
    stability = "experimental"

    def __post_init__(self) -> None:
        if isinstance(self.num_classes, bool) or self.num_classes < 2:
            raise ValueError("num_classes must include background and be at least 2")

    @property
    def architecture(self) -> MaskRCNNArchitecture:
        return MaskRCNNArchitecture(
            backbone=Backbone(self.backbone, trainable=True),
            proposal_generator=RegionProposalNetwork(self.rpn),
            roi_heads=RoIHeads(
                BoxHead(
                    self.box_head,
                    self.num_classes,
                    responsibility="learned-box-classification-and-regression",
                    trainable=True,
                ),
                MaskHead(self.mask_head, reuse_proposal_masks=False, trainable=True),
            ),
            execution="unbound-trainable-architecture",
        )

    def run(self, *_: Any, **__: Any) -> Any:
        raise UnboundMaskRCNNError(
            "SphericalMaskRCNN defines the future trainable Mask R-CNN topology "
            "but PanorAi does not implement its training or inference path. Use "
            "mask_rcnn.paper(...) for the executable CAM/SAM paper method."
        )

    __call__ = run


__all__ = ["SphericalMaskRCNN", "UnboundMaskRCNNError"]
