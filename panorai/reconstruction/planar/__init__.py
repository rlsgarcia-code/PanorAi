"""Experimental human-guided reconstruction of planar spherical regions.

NumPy inputs, fixed metric camera poses, explicit masks and radial ranges.
SAM remains an optional injected segmentation backend.
"""

from ._models import (
    INTERFACE,
    RegisteredPlanarView,
    RegisteredPlanarScene,
    PlanarRegion,
    PlanarPointObservation,
    PlaneCorrespondence,
    PlanarReconstructionOptions,
    TriangulatedPlaneTrack,
    PlaneEstimate,
    PlanarRangeResult,
    PlanarWorkspace,
)
from ._stages import (
    ValidatePlanarScene,
    SegmentPlaneRegions,
    MatchPlaneRegions,
    TriangulatePlaneSupport,
    FitPlanes,
    PropagatePlaneRegions,
    RasterizePlaneRanges,
)
from ._application import HumanGuidedPlanarReconstruction
from ._discovery_application import SphericalPlanarDiscovery
from ._discovery_models import (
    PLANAR_DISCOVERY_INTERFACE,
    PlanarDiscoveryRequest,
    SphericalLineSeed,
    PlanarOrientationProposal,
    SphericalLineObservation,
    SphericalPlanarDiscoveryOptions,
    SphericalPlanarDiscoveryResult,
    VanishingBundle,
    VanishingLineSuggestion,
)
from ._discovery_stages import (
    AssemblePlanarDiscoveryResult,
    BuildOverlappingGnomonicAtlas,
    DetectAndFitGnomonicLines,
    EstimateVanishingBundles,
    MergeSphericalLineObservations,
    ProposePlanarOrientations,
    SuggestVanishingBundleLines,
    ValidatePlanarDiscoveryRequest,
)
from ._segmentation import segment_planar_region
from ._io import (
    SCENE_SCHEMA,
    encode_planar_mask,
    decode_planar_mask,
    planar_scene_from_document,
    region_document,
    planar_result_summary,
    export_planar_result,
)

__all__ = [
    "INTERFACE",
    "RegisteredPlanarView",
    "RegisteredPlanarScene",
    "PlanarRegion",
    "PlanarPointObservation",
    "PlaneCorrespondence",
    "PlanarReconstructionOptions",
    "TriangulatedPlaneTrack",
    "PlaneEstimate",
    "PlanarRangeResult",
    "PlanarWorkspace",
    "ValidatePlanarScene",
    "SegmentPlaneRegions",
    "MatchPlaneRegions",
    "TriangulatePlaneSupport",
    "FitPlanes",
    "PropagatePlaneRegions",
    "RasterizePlaneRanges",
    "HumanGuidedPlanarReconstruction",
    "PLANAR_DISCOVERY_INTERFACE",
    "SphericalPlanarDiscovery",
    "SphericalPlanarDiscoveryOptions",
    "PlanarDiscoveryRequest",
    "SphericalLineSeed",
    "SphericalPlanarDiscoveryResult",
    "SphericalLineObservation",
    "VanishingBundle",
    "VanishingLineSuggestion",
    "PlanarOrientationProposal",
    "ValidatePlanarDiscoveryRequest",
    "BuildOverlappingGnomonicAtlas",
    "DetectAndFitGnomonicLines",
    "MergeSphericalLineObservations",
    "EstimateVanishingBundles",
    "SuggestVanishingBundleLines",
    "ProposePlanarOrientations",
    "AssemblePlanarDiscoveryResult",
    "segment_planar_region",
    "SCENE_SCHEMA",
    "encode_planar_mask",
    "decode_planar_mask",
    "planar_scene_from_document",
    "region_document",
    "planar_result_summary",
    "export_planar_result",
]
