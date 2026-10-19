"""High-level spherical DoG and tangent-SIFT application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from panorai.data import EquirectangularImage
from panorai.features import (
    OpenCVTangentDescriptorV2Config,
    SphericalDoGSIFTConfig,
    SphericalDoGSIFTExtractor,
)
from panorai.pipeline import Pipeline, StageResult

from ._base import PipelineApplication, application_config


DOG_REVIEWED_PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"
DOG_REVIEWED_INTERFACE = "panorai-spherical-dog-features/v1"
DOG_REVIEWED_CONFIG_FORMAT = "panorai-spherical-dog-features-config/v1"


def _reviewed_config(*, convolution_backend: str = "auto") -> SphericalDoGSIFTConfig:
    """Return the scientific parameters frozen by the VAL-093 oracle.

    ``convolution_backend`` is an execution choice rather than a scientific
    parameter.  The detector, selection, tangent-patch and RootSIFT decisions
    remain identical across the supported backends.
    """

    return SphericalDoGSIFTConfig(
        octaves=3,
        levels_per_octave=3,
        base_sigma_px=1.0,
        contrast_threshold=0.0005,
        edge_threshold=20.0,
        max_features=160,
        patch_size=32,
        descriptor_radius_sigmas=6.0,
        minimum_valid_fraction=0.99,
        minimum_valid_support_fraction=0.99,
        angular_dedup_threshold_deg=0.12,
        scale_dedup_log2=0.5,
        selection_policy="response",
        selection_grid_shape=(12, 24),
        descriptor_config=OpenCVTangentDescriptorV2Config(),
        convolution_backend=convolution_backend,
    )


def _has_reviewed_scientific_config(config: SphericalDoGSIFTConfig) -> bool:
    return replace(config, convolution_backend="auto") == _reviewed_config()


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
    config_format = DOG_REVIEWED_CONFIG_FORMAT
    legacy_config_formats = ("panorai-application-config/v1-experimental",)
    profile_id: ClassVar[str | None] = None

    @classmethod
    def reviewed_profile(
        cls, *, convolution_backend: str = "auto"
    ) -> "SphericalDoGFeatures":
        """Build the fixed candidate profile without promoting generic DoG.

        The returned object is deliberately still Experimental until the
        protocol-bound external holdout, installed-artifact matrix and
        independent review all pass.  A later tier-only change may promote
        this exact identity without making free-form configurations Stable.
        """

        return _ReviewedSphericalDoGFeatures(
            _reviewed_config(convolution_backend=convolution_backend)
        )

    def as_pipeline(self) -> Pipeline:
        return Pipeline(_DoGFeatureStage(self.config))

    def run(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: NDArray[Any] | None = None,
    ) -> Any:
        if isinstance(panorama, np.ndarray):
            panorama = EquirectangularImage(panorama, valid=validity_mask)
        elif validity_mask is not None:
            raise ValueError(
                "validity_mask is accepted only with an array input; public ERP "
                "containers already carry explicit validity"
            )
        return Pipeline(_DoGFeatureStage(self.config, panorama_id))(panorama)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "profile_id": self.profile_id,
            "input": "canonical ERP HW/HWC with explicit support/validity",
            "output": "panorai.features.SphericalFeatureSet",
            "domain": "direct-spherical DoG plus tangent-patch descriptor",
            "dependencies": ["NumPy", "OpenCV SIFT"],
            "config": self.config.to_dict(),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        if self.profile_id is not None:
            return application_config(
                self.application_id,
                {"profile_id": self.profile_id, "config": self.config},
                format_id=self.config_format,
            )
        return application_config(self.application_id, {"config": self.config})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SphericalDoGFeatures":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe SphericalDoGFeatures")
        format_id = config.get("format")
        if format_id not in {cls.config_format, *cls.legacy_config_formats}:
            raise ValueError("unknown SphericalDoGFeatures configuration format")
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
        restored = SphericalDoGSIFTConfig(**values)
        if format_id == cls.config_format:
            if parameters.get("profile_id") != DOG_REVIEWED_PROFILE_ID:
                raise ValueError("unknown SphericalDoGFeatures reviewed profile")
            if not _has_reviewed_scientific_config(restored):
                raise ValueError("reviewed DoG profile parameters have drifted")
            return cls.reviewed_profile(
                convolution_backend=restored.convolution_backend
            )
        if "profile_id" in parameters:
            raise ValueError("legacy DoG configuration cannot claim a profile")
        return cls(restored)


class _ReviewedSphericalDoGFeatures(SphericalDoGFeatures):
    """Runtime identity for the exact reviewed candidate profile."""

    interface = DOG_REVIEWED_INTERFACE
    stability = "experimental"
    profile_id = DOG_REVIEWED_PROFILE_ID

    def with_params(self, **params: Any) -> SphericalDoGFeatures:
        changed = SphericalDoGFeatures(self.config).with_params(**params)
        if not isinstance(changed, SphericalDoGFeatures):  # pragma: no cover
            raise RuntimeError("DoG parameter replacement returned the wrong type")
        if _has_reviewed_scientific_config(changed.config):
            return SphericalDoGFeatures.reviewed_profile(
                convolution_backend=changed.config.convolution_backend
            )
        return changed


__all__ = [
    "DOG_REVIEWED_CONFIG_FORMAT",
    "DOG_REVIEWED_INTERFACE",
    "DOG_REVIEWED_PROFILE_ID",
    "SphericalDoGFeatures",
]
