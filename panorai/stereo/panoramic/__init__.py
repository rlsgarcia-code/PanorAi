"""End-to-end dense stereo for registered or unregistered ERP pairs.

The classical route uses spherical geometry to create finite rectified views,
runs an interchangeable pinhole matcher on one pair at a time, converts
disparity to radial range, and fuses range back in the reference ERP.  The
alternative sphere-native route exposes the existing direct ERP plane sweep
through the same result contract.
"""

from ._planning import (
    RectifiedStereoViewPlanner,
    SphericalStereoViewPairer,
    spec_from_axes,
)
from ._full_sphere import (
    FullSphereMultiViewStereo,
    FullSphereStereoOptions,
    FullSphereStereoResult,
    OverlappingCubemapPlanner,
    RegisteredPanoramaSet,
    StereoSourceView,
)
from ._stages import (
    BackProjectStereoRange,
    DenseStereoPerPair,
    FuseStereoRange,
    PanoramicStereoInput,
    PlanRectifiedStereoViews,
    ProjectStereoViewPairs,
    ResolveStereoPose,
    ResolveStereoScale,
    SelectStereoViewPairs,
    SphereNativeStereo,
    ValidatePanoramicStereo,
)
from ._types import (
    PanoramicStereoPose,
    PanoramicStereoResult,
    RectifiedStereoViewSpec,
    StereoPairDiagnostic,
    StereoViewDescriptor,
    StereoViewPairScore,
    StereoViewSelectionDiagnostic,
    StereoViewPlan,
)

__all__ = [
    "BackProjectStereoRange",
    "DenseStereoPerPair",
    "FuseStereoRange",
    "FullSphereMultiViewStereo",
    "FullSphereStereoOptions",
    "FullSphereStereoResult",
    "OverlappingCubemapPlanner",
    "PanoramicStereoInput",
    "PanoramicStereoPose",
    "PanoramicStereoResult",
    "PlanRectifiedStereoViews",
    "ProjectStereoViewPairs",
    "RectifiedStereoViewPlanner",
    "RectifiedStereoViewSpec",
    "RegisteredPanoramaSet",
    "ResolveStereoPose",
    "ResolveStereoScale",
    "SelectStereoViewPairs",
    "SphereNativeStereo",
    "SphericalStereoViewPairer",
    "StereoPairDiagnostic",
    "StereoSourceView",
    "StereoViewDescriptor",
    "StereoViewPairScore",
    "StereoViewSelectionDiagnostic",
    "StereoViewPlan",
    "ValidatePanoramicStereo",
    "spec_from_axes",
]
