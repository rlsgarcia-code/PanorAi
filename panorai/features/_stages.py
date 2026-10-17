"""Composable stages for the spherical feature and matching frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from panorai.pipeline import StageResult

from ._config import SphericalFeaturePipelineConfig
from ._models import (
    SphericalFeatureMatches,
    SphericalFeaturePair,
    SphericalImagePair,
)
from ._pipeline import SphericalFeaturePipeline


def _frontend(
    config: SphericalFeaturePipelineConfig | None,
) -> SphericalFeaturePipeline:
    if config is None:
        return SphericalFeaturePipeline.for_relative_pose()
    if not isinstance(config, SphericalFeaturePipelineConfig):
        raise TypeError("frontend_config must be a SphericalFeaturePipelineConfig")
    return SphericalFeaturePipeline(config)


@dataclass(frozen=True, slots=True)
class ExtractSphericalFeaturePair:
    """Extract one PanorAi feature set for each image in a typed pair.

    ``frontend_config=None`` deliberately calls the frozen relative-pose
    reference profile.  A custom configuration is always explicit.
    """

    frontend_config: SphericalFeaturePipelineConfig | None = None

    input_kind = "spherical_image_pair"
    output_kind = "spherical_feature_pair"
    interface = "panorai-spherical-relative-pose-pipeline/v1-experimental"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SphericalImagePair):
            raise TypeError("ExtractSphericalFeaturePair requires SphericalImagePair")
        frontend = _frontend(self.frontend_config)
        reference = frontend.extract(
            value.reference,
            panorama_id=value.reference_id,
        )
        source = frontend.extract(
            value.source,
            panorama_id=value.source_id,
        )
        result = SphericalFeaturePair(reference, source)
        return StageResult(
            result,
            {
                "domain": "projected-view-features",
                "profile": frontend.config.preset_name,
                "profile_version": frontend.config.preset_version,
                "frontend_mode": (
                    "reference" if self.frontend_config is None else "custom"
                ),
                "reference_features": len(reference),
                "source_features": len(source),
            },
        )


@dataclass(frozen=True, slots=True)
class MatchSphericalFeatures:
    """Match an ordered feature pair with the selected frontend matcher."""

    frontend_config: SphericalFeaturePipelineConfig | None = None

    input_kind = "spherical_feature_pair"
    output_kind = "spherical_matches"
    interface = "panorai-spherical-relative-pose-pipeline/v1-experimental"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SphericalFeaturePair):
            raise TypeError("MatchSphericalFeatures requires SphericalFeaturePair")
        frontend = _frontend(self.frontend_config)
        result = frontend.match(value.reference, value.source)
        return StageResult(
            result,
            {
                "domain": "spherical-feature-matching",
                "profile": frontend.config.preset_name,
                "profile_version": frontend.config.preset_version,
                "frontend_mode": (
                    "reference" if self.frontend_config is None else "custom"
                ),
                "matches": len(result),
                "valid_matches": int(result.valid.sum()),
            },
        )


@dataclass(frozen=True, slots=True)
class ToBearingCorrespondences:
    """Convert feature matches to explicit weighted spherical bearings."""

    weighting: str = "uniform-valid"

    input_kind = "spherical_matches"
    output_kind = "bearing_correspondences"
    interface = "panorai-spherical-relative-pose-pipeline/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if self.weighting != "uniform-valid":
            raise ValueError("weighting must be 'uniform-valid'")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SphericalFeatureMatches):
            raise TypeError("ToBearingCorrespondences requires SphericalFeatureMatches")
        result = value.to_bearing_correspondences()
        return StageResult(
            result,
            {
                "domain": "bearing-geometry",
                "weighting": self.weighting,
                "correspondences": len(result.valid),
                "valid_correspondences": int(result.valid.sum()),
            },
        )


__all__ = [
    "ExtractSphericalFeaturePair",
    "MatchSphericalFeatures",
    "ToBearingCorrespondences",
]
