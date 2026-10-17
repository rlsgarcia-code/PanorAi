"""Pipeline stage adapters for PanorAi-owned geometric estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from panorai.pipeline import StageResult

from ._quality import RelativePoseAcceptancePolicy
from ._sampling import (
    FivePointSampler,
    SpatiallyWeightedFivePointSampler,
)
from .relative_pose import (
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
)


@dataclass(frozen=True, slots=True)
class EstimateRelativePose:
    """Estimate Essential, rotation, and unit translation direction.

    This is a sparse robust-estimation :class:`~panorai.pipeline.Stage`, not a
    Torch layer.  It delegates every numerical decision to
    :class:`SphericalRelativePoseEstimator`.
    """

    options: RelativePoseOptions = field(default_factory=RelativePoseOptions)
    sampler: FivePointSampler = field(default_factory=SpatiallyWeightedFivePointSampler)
    quality_policy: RelativePoseAcceptancePolicy = field(
        default_factory=RelativePoseAcceptancePolicy
    )

    input_kind = ("bearing_correspondences", "spherical_matches")
    output_kind = "relative_pose"
    interface = "panorai-spherical-relative-pose-pipeline/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if not isinstance(self.options, RelativePoseOptions):
            raise TypeError("options must be RelativePoseOptions")
        if not isinstance(self.quality_policy, RelativePoseAcceptancePolicy):
            raise TypeError("quality_policy must be RelativePoseAcceptancePolicy")
        if (
            not callable(getattr(self.sampler, "prepare", None))
            or not callable(getattr(self.sampler, "describe", None))
            or not isinstance(getattr(self.sampler, "name", None), str)
            or not self.sampler.name
        ):
            raise TypeError("sampler must implement the FivePointSampler protocol")

    def execute(self, value: Any) -> StageResult:
        required = ("bearings_a", "bearings_b", "valid")
        if not all(hasattr(value, name) for name in required):
            raise TypeError(
                "EstimateRelativePose requires spherical matches or bearing "
                "correspondences"
            )
        estimator = SphericalRelativePoseEstimator(
            self.options,
            sampler=self.sampler,
            quality_policy=self.quality_policy,
        )
        result = estimator.estimate(value)
        details: dict[str, Any] = {
            "domain": "spherical-bearing-geometry",
            "pose_convention": "source-from-reference",
            "translation": "unit-direction-only",
            "returned": result is not None,
            "sampler": self.sampler.name,
        }
        if result is not None:
            details.update(
                {
                    "num_inliers": result.num_inliers,
                    "accepted": result.quality_report.accepted,
                    "degenerate": result.degenerate,
                    "compute_backend": result.compute_backend,
                }
            )
        return StageResult(result, details)


__all__ = ["EstimateRelativePose"]
