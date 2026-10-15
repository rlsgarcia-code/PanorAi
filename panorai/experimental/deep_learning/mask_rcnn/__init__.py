"""Mask R-CNN taxonomy and the executable spherical paper-method factory."""

from __future__ import annotations

from typing import Any

from ._architecture import (
    Backbone,
    BoxHead,
    MaskHead,
    MaskRCNNArchitecture,
    ProposalGenerator,
    RegionProposalNetwork,
    RoIAlign,
    RoIHeads,
)
from ._unbound import SphericalMaskRCNN, UnboundMaskRCNNError


def paper(**kwargs: Any) -> Any:
    """Construct the executable CAM/SAM method described by the paper."""

    from panorai.applications._mask_rcnn import SphericalMaskRCNNPaper

    return SphericalMaskRCNNPaper(**kwargs)


def __getattr__(name: str) -> Any:
    if name == "SphericalMaskRCNNPaper":
        from panorai.applications._mask_rcnn import SphericalMaskRCNNPaper

        return SphericalMaskRCNNPaper
    raise AttributeError(name)


__all__ = [
    "Backbone",
    "BoxHead",
    "MaskHead",
    "MaskRCNNArchitecture",
    "ProposalGenerator",
    "RegionProposalNetwork",
    "RoIAlign",
    "RoIHeads",
    "SphericalMaskRCNN",
    "SphericalMaskRCNNPaper",
    "UnboundMaskRCNNError",
    "paper",
]
