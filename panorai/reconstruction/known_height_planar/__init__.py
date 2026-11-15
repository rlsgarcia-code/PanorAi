"""Experimental known-height planar mapping for one equirectangular image.

The application uses reviewed native-grid masks and confirmed plane contacts.
It never infers planarity, validity, or metric scale from pixel values.
"""

from ._application import KnownHeightPlanarMapping
from ._preview import preview_planar_relation
from ._models import (
    INTERFACE,
    KnownHeightPlanarMap,
    KnownHeightPlanarOptions,
    KnownHeightPlanarScene,
    PlanarConstraint,
    PlanarOrientationEvidence,
    PlanarPreview,
)
from ._stages import (
    AnchorFloorPlane,
    AssembleKnownHeightPlanarMap,
    RasterizeKnownHeightRanges,
    ResolveConnectedPlanes,
    ResolvePlanarPreviews,
    ResolvePlanarReferenceFrame,
    SegmentKnownHeightRegions,
    ValidateKnownHeightPlanarScene,
)

__all__ = [
    "INTERFACE",
    "AnchorFloorPlane",
    "AssembleKnownHeightPlanarMap",
    "KnownHeightPlanarMap",
    "KnownHeightPlanarMapping",
    "KnownHeightPlanarOptions",
    "KnownHeightPlanarScene",
    "PlanarConstraint",
    "PlanarOrientationEvidence",
    "PlanarPreview",
    "RasterizeKnownHeightRanges",
    "ResolveConnectedPlanes",
    "ResolvePlanarPreviews",
    "ResolvePlanarReferenceFrame",
    "SegmentKnownHeightRegions",
    "ValidateKnownHeightPlanarScene",
    "preview_planar_relation",
]
