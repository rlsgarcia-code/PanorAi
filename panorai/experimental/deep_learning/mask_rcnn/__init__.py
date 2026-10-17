"""Mask R-CNN taxonomy and executable spherical workflow factory."""

from __future__ import annotations

import warnings
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


def workflow(**kwargs: Any) -> Any:
    """Construct the executable spherical CAM/SAM Mask R-CNN workflow."""

    from panorai.applications._mask_rcnn import SphericalMaskRCNNWorkflow

    return SphericalMaskRCNNWorkflow(**kwargs)


def paper(**kwargs: Any) -> Any:
    """Compatibility factory for :func:`workflow`.

    ``paper`` was the original Experimental name.  It remains callable so
    existing code does not break, but new code should describe the executable
    object by its role rather than its research provenance.
    """

    warnings.warn(
        "mask_rcnn.paper(...) is deprecated; use mask_rcnn.workflow(...)",
        DeprecationWarning,
        stacklevel=2,
    )
    return workflow(**kwargs)


def __getattr__(name: str) -> Any:
    if name in {"SphericalMaskRCNNPaper", "SphericalMaskRCNNWorkflow"}:
        from panorai.applications._mask_rcnn import (
            SphericalMaskRCNNPaper,
            SphericalMaskRCNNWorkflow,
        )

        return {
            "SphericalMaskRCNNPaper": SphericalMaskRCNNPaper,
            "SphericalMaskRCNNWorkflow": SphericalMaskRCNNWorkflow,
        }[name]
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
    "SphericalMaskRCNNWorkflow",
    "UnboundMaskRCNNError",
    "paper",
    "workflow",
]
