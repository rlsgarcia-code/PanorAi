"""High-level spherical DoG and tangent-SIFT application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from panorai.data import EquirectangularImage
from panorai.features import (
    OpenCVTangentDescriptorV2Config,
    SphericalFeatureSet,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
)
from panorai.pipeline import Pipeline, StageResult

from ._base import PipelineApplication, application_config


@dataclass(frozen=True, slots=True)
class _DoGFeatureStage:
    config: SphericalDoGSIFTConfig
    panorama_id: str | None = None

    input_kind = "erp"
    output_kind = "result"

    def execute(self, value: Any) -> StageResult:
        result = SphericalDoGSIFTExtractor(self.config).extract(
            value, panorama_id=self.panorama_id
        )
        return StageResult(
            result,
            {
                "domain": "sphere",
                "algorithm": "spherical-dog-tangent-sift",
                "keypoint_count": len(result.features),
                "descriptor_shape": list(result.descriptors.shape),
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalDoGFeatures(PipelineApplication):
    """Detect spherical DoG keypoints and compute tangent-patch descriptors."""

    config: SphericalDoGSIFTConfig = SphericalDoGSIFTConfig()

    application_id = "panorai.applications.SphericalDoGFeatures"

    def as_pipeline(self) -> Pipeline:
        return Pipeline(_DoGFeatureStage(self.config))

    def run(
        self,
        panorama: np.ndarray | EquirectangularImage,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> SphericalFeatureSet:
        if isinstance(panorama, np.ndarray):
            panorama = EquirectangularImage(panorama, valid=validity_mask)
        elif validity_mask is not None:
            raise ValueError(
                "validity_mask is accepted only with an array input; public ERP "
                "containers already carry explicit validity"
            )
        result = Pipeline(_DoGFeatureStage(self.config, panorama_id))(panorama)
        if not isinstance(result, SphericalFeatureSet):
            raise RuntimeError("SphericalDoGFeatures returned an unexpected result")
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "canonical ERP HW/HWC with explicit support/validity",
            "output": "panorai.features.SphericalFeatureSet",
            "domain": "direct-spherical DoG plus tangent-patch descriptor",
            "dependencies": ["NumPy", "OpenCV SIFT"],
            "config": self.config.to_dict(),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(self.application_id, {"config": self.config})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SphericalDoGFeatures":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe SphericalDoGFeatures")
        parameters = config.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(
            parameters.get("config"), dict
        ):
            raise ValueError("SphericalDoGFeatures config is missing")
        values = dict(parameters["config"])
        descriptor = values.get("descriptor_config")
        if not isinstance(descriptor, dict):
            raise ValueError("descriptor_config must be a mapping")
        values["descriptor_config"] = OpenCVTangentDescriptorV2Config(**descriptor)
        if isinstance(values.get("selection_grid_shape"), list):
            values["selection_grid_shape"] = tuple(values["selection_grid_shape"])
        return cls(SphericalDoGSIFTConfig(**values))


__all__ = ["SphericalDoGFeatures"]
