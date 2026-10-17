"""Experimental, non-learned stereo primitives and reference pipelines.

Inputs are already rectified pinhole pairs.  Spherical sampling,
rectification, and fusion stay in PanorAi's geometry/object workflow so the
same stereo stages can be applied independently to projected views.
"""

from ._operations import (
    ADCensusCost,
    AbsoluteDifferenceCost,
    BirchfieldTomasiCost,
    BoxCostAggregation,
    CensusCost,
    CrossBasedCostAggregation,
    ELASSupportPrior,
    MoreGlobalMatching,
    SemiGlobalMatching,
    disparity_labels,
)
from ._pipelines import (
    ADCensusConfig,
    CensusMGMConfig,
    ELASConfig,
    PatchMatchConfig,
    StereoBMConfig,
    StereoSGBMConfig,
    StereoValidationConfig,
    ad_census_pipeline,
    census_mgm_pipeline,
    elas_pipeline,
    patchmatch_pipeline,
    stereo_bm_pipeline,
    stereo_sgbm_pipeline,
)
from ._stages import (
    BuildStereoCost,
    OpenCVStereoMatcher,
    PatchMatchStereo,
    RefineStereoDisparity,
    SelectStereoDisparity,
    TransformStereoCost,
)
from ._types import (
    ClassicalStereoResult,
    RectifiedStereoGeometry,
    RectifiedStereoPair,
    StereoCostVolume,
    StereoDisparityCandidate,
)

__all__ = [
    "ADCensusConfig",
    "ADCensusCost",
    "AbsoluteDifferenceCost",
    "BirchfieldTomasiCost",
    "BoxCostAggregation",
    "BuildStereoCost",
    "CensusCost",
    "CensusMGMConfig",
    "ClassicalStereoResult",
    "CrossBasedCostAggregation",
    "ELASConfig",
    "ELASSupportPrior",
    "MoreGlobalMatching",
    "OpenCVStereoMatcher",
    "PatchMatchConfig",
    "PatchMatchStereo",
    "RectifiedStereoGeometry",
    "RectifiedStereoPair",
    "RefineStereoDisparity",
    "SelectStereoDisparity",
    "SemiGlobalMatching",
    "StereoBMConfig",
    "StereoCostVolume",
    "StereoDisparityCandidate",
    "StereoSGBMConfig",
    "StereoValidationConfig",
    "TransformStereoCost",
    "ad_census_pipeline",
    "census_mgm_pipeline",
    "disparity_labels",
    "elas_pipeline",
    "patchmatch_pipeline",
    "stereo_bm_pipeline",
    "stereo_sgbm_pipeline",
]
