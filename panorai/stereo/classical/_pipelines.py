"""Reference compositions for six non-learned rectified stereo baselines."""

from __future__ import annotations

from dataclasses import dataclass

from panorai.pipeline import Pipeline

from ._operations import (
    ADCensusCost,
    BoxCostAggregation,
    CensusCost,
    CrossBasedCostAggregation,
    ELASSupportPrior,
    MoreGlobalMatching,
    SemiGlobalMatching,
)
from ._stages import (
    BuildStereoCost,
    OpenCVStereoMatcher,
    PatchMatchStereo,
    RefineStereoDisparity,
    SelectStereoDisparity,
    TransformStereoCost,
)


@dataclass(frozen=True, slots=True)
class StereoValidationConfig:
    """Common explicit validation policy for classical stereo outputs."""

    uniqueness_margin: float = 0.02
    left_right_tolerance: float = 1.0
    speckle_window_size: int = 0
    speckle_range: float = 1.0
    median_kernel: int = 0

    def stage(self, method: str) -> RefineStereoDisparity:
        return RefineStereoDisparity(
            method=method,
            uniqueness_margin=self.uniqueness_margin,
            left_right_tolerance=self.left_right_tolerance,
            speckle_window_size=self.speckle_window_size,
            speckle_range=self.speckle_range,
            median_kernel=self.median_kernel,
        )


@dataclass(frozen=True, slots=True)
class StereoBMConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    block_size: int = 9
    uniqueness_ratio: int = 10
    texture_threshold: int = 10
    validation: StereoValidationConfig = StereoValidationConfig(uniqueness_margin=0.0)


@dataclass(frozen=True, slots=True)
class ADCensusConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    census_radius: int = 2
    aggregation_arm: int = 5
    aggregation_threshold: float = 0.08
    aggregation_passes: int = 2
    p1: float = 0.03
    p2: float = 0.18
    paths: int = 8
    validation: StereoValidationConfig = StereoValidationConfig()


@dataclass(frozen=True, slots=True)
class StereoSGBMConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    block_size: int = 5
    uniqueness_ratio: int = 10
    p1: int | None = None
    p2: int | None = None
    validation: StereoValidationConfig = StereoValidationConfig(uniqueness_margin=0.0)


@dataclass(frozen=True, slots=True)
class CensusMGMConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    census_radius: int = 2
    aggregation_window: int = 3
    variant: str = "tmgm"
    p1: float = 0.03
    p2: float = 0.18
    validation: StereoValidationConfig = StereoValidationConfig()


@dataclass(frozen=True, slots=True)
class ELASConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    census_radius: int = 2
    aggregation_window: int = 3
    support_step: int = 8
    support_uniqueness: float = 0.08
    support_radius: float = 4.0
    prior_weight: float = 0.20
    p1: float = 0.02
    p2: float = 0.12
    validation: StereoValidationConfig = StereoValidationConfig()


@dataclass(frozen=True, slots=True)
class PatchMatchConfig:
    min_disparity: int = 0
    max_disparity: int = 64
    iterations: int = 3
    patch_radius: int = 2
    random_seed: int = 0
    validation: StereoValidationConfig = StereoValidationConfig(
        uniqueness_margin=0.0,
        left_right_tolerance=1.5,
    )


def stereo_bm_pipeline(config: StereoBMConfig = StereoBMConfig()) -> Pipeline:
    return Pipeline(
        (
            "matching",
            OpenCVStereoMatcher(
                "bm",
                min_disparity=config.min_disparity,
                max_disparity=config.max_disparity,
                block_size=config.block_size,
                uniqueness_ratio=config.uniqueness_ratio,
                texture_threshold=config.texture_threshold,
            ),
        ),
        ("validation", config.validation.stage("stereo-bm")),
    )


def ad_census_pipeline(config: ADCensusConfig = ADCensusConfig()) -> Pipeline:
    return Pipeline(
        (
            "matching_cost",
            BuildStereoCost(
                ADCensusCost(census_radius=config.census_radius),
                config.min_disparity,
                config.max_disparity,
            ),
        ),
        (
            "cost_aggregation",
            TransformStereoCost(
                CrossBasedCostAggregation(
                    config.aggregation_arm,
                    config.aggregation_threshold,
                    config.aggregation_passes,
                ),
                "aggregation",
            ),
        ),
        (
            "optimization",
            TransformStereoCost(
                SemiGlobalMatching(config.p1, config.p2, config.paths),
                "optimization",
            ),
        ),
        ("selection", SelectStereoDisparity()),
        ("validation", config.validation.stage("ad-census")),
    )


def stereo_sgbm_pipeline(
    config: StereoSGBMConfig = StereoSGBMConfig(),
) -> Pipeline:
    return Pipeline(
        (
            "matching_and_optimization",
            OpenCVStereoMatcher(
                "sgbm",
                min_disparity=config.min_disparity,
                max_disparity=config.max_disparity,
                block_size=config.block_size,
                uniqueness_ratio=config.uniqueness_ratio,
                p1=config.p1,
                p2=config.p2,
            ),
        ),
        ("validation", config.validation.stage("stereo-sgbm")),
    )


def census_mgm_pipeline(config: CensusMGMConfig = CensusMGMConfig()) -> Pipeline:
    return Pipeline(
        (
            "matching_cost",
            BuildStereoCost(
                CensusCost(config.census_radius),
                config.min_disparity,
                config.max_disparity,
            ),
        ),
        (
            "cost_aggregation",
            TransformStereoCost(
                BoxCostAggregation(config.aggregation_window), "aggregation"
            ),
        ),
        (
            "optimization",
            TransformStereoCost(
                MoreGlobalMatching(config.variant, config.p1, config.p2),
                "optimization",
            ),
        ),
        ("selection", SelectStereoDisparity()),
        ("validation", config.validation.stage(f"census-{config.variant}")),
    )


def elas_pipeline(config: ELASConfig = ELASConfig()) -> Pipeline:
    """Build the documented PanorAi ELAS-style reference composition.

    This is an inspectable support-point/triangulation-prior pipeline.  It is
    not presented as binary-equivalent to the original libELAS implementation.
    """

    return Pipeline(
        (
            "matching_cost",
            BuildStereoCost(
                CensusCost(config.census_radius),
                config.min_disparity,
                config.max_disparity,
            ),
        ),
        (
            "cost_aggregation",
            TransformStereoCost(
                BoxCostAggregation(config.aggregation_window), "aggregation"
            ),
        ),
        (
            "support_prior",
            TransformStereoCost(
                ELASSupportPrior(
                    config.support_step,
                    config.support_uniqueness,
                    config.support_radius,
                    config.prior_weight,
                ),
                "prior",
            ),
        ),
        (
            "optimization",
            TransformStereoCost(
                SemiGlobalMatching(config.p1, config.p2, paths=4),
                "optimization",
            ),
        ),
        ("selection", SelectStereoDisparity()),
        ("validation", config.validation.stage("elas-style")),
    )


def patchmatch_pipeline(
    config: PatchMatchConfig = PatchMatchConfig(),
) -> Pipeline:
    return Pipeline(
        (
            "patchmatch",
            PatchMatchStereo(
                config.min_disparity,
                config.max_disparity,
                config.iterations,
                config.patch_radius,
                config.random_seed,
            ),
        ),
        ("validation", config.validation.stage("patchmatch-stereo")),
    )


__all__ = [
    "ADCensusConfig",
    "CensusMGMConfig",
    "ELASConfig",
    "PatchMatchConfig",
    "StereoBMConfig",
    "StereoSGBMConfig",
    "StereoValidationConfig",
    "ad_census_pipeline",
    "census_mgm_pipeline",
    "elas_pipeline",
    "patchmatch_pipeline",
    "stereo_bm_pipeline",
    "stereo_sgbm_pipeline",
]
