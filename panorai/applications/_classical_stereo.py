"""High-level wrappers for non-learned rectified stereo pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

from panorai.pipeline import Pipeline
from panorai.stereo.classical import (
    ADCensusConfig,
    CensusMGMConfig,
    ClassicalStereoResult,
    ELASConfig,
    PatchMatchConfig,
    RectifiedStereoGeometry,
    RectifiedStereoPair,
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

from ._base import PipelineApplication, application_config


ConfigT = TypeVar("ConfigT")


def _run(
    pipeline: Pipeline,
    reference: np.ndarray,
    source: np.ndarray,
    *,
    reference_support: np.ndarray | None,
    source_support: np.ndarray | None,
    geometry: RectifiedStereoGeometry | None,
) -> ClassicalStereoResult:
    result = pipeline(
        RectifiedStereoPair(
            reference,
            source,
            reference_support,
            source_support,
            geometry,
        )
    )
    if not isinstance(result, ClassicalStereoResult):
        raise RuntimeError("classical stereo pipeline returned an unexpected result")
    return result


def _description(
    application_id: str, method: str, config: Any, pipeline: Pipeline
) -> dict[str, Any]:
    return {
        "interface": "panorai-application/v1-experimental",
        "stability": "experimental",
        "application": application_id,
        "input": "one already-rectified pinhole pair with explicit support",
        "output": "ClassicalStereoResult with disparity and optional radial range",
        "method": method,
        "disparity": "x_reference - x_source in pixels",
        "metric_geometry": "optional fx/baseline converts z-depth to radial range",
        "configuration": application_config(application_id, {"config": config})[
            "parameters"
        ]["config"],
        "pipeline": pipeline.describe(),
    }


def _from_config(
    payload: dict[str, Any],
    *,
    application_id: str,
    config_type: Callable[..., ConfigT],
) -> ConfigT:
    if payload.get("application") != application_id:
        raise ValueError(f"configuration does not describe {application_id}")
    parameters = payload.get("parameters")
    raw = parameters.get("config") if isinstance(parameters, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("classical stereo configuration is missing")
    values = dict(raw)
    validation = values.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("classical stereo validation configuration is missing")
    values["validation"] = StereoValidationConfig(**validation)
    return config_type(**values)


class _RunMixin:
    config: Any

    def _pipeline(self) -> Pipeline:
        raise NotImplementedError

    def run(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        *,
        reference_support: np.ndarray | None = None,
        source_support: np.ndarray | None = None,
        geometry: RectifiedStereoGeometry | None = None,
    ) -> ClassicalStereoResult:
        return _run(
            self._pipeline(),
            reference,
            source,
            reference_support=reference_support,
            source_support=source_support,
            geometry=geometry,
        )

    def as_pipeline(self) -> Pipeline:
        return self._pipeline()

    def describe(self) -> dict[str, Any]:
        return _description(
            self.application_id, self.method_name, self.config, self._pipeline()
        )

    def to_config(self) -> dict[str, Any]:
        return application_config(self.application_id, {"config": self.config})


@dataclass(frozen=True, slots=True)
class StereoBMDepth(_RunMixin, PipelineApplication):
    """OpenCV StereoBM baseline with explicit PanorAi validation output."""

    config: StereoBMConfig = StereoBMConfig()
    application_id = "panorai.applications.StereoBMDepth"
    method_name = "OpenCV StereoBM"

    def _pipeline(self) -> Pipeline:
        return stereo_bm_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "StereoBMDepth":
        return cls(
            _from_config(
                payload, application_id=cls.application_id, config_type=StereoBMConfig
            )
        )


@dataclass(frozen=True, slots=True)
class ADCensusStereoDepth(_RunMixin, PipelineApplication):
    """AD-Census, cross aggregation and SGM reference composition."""

    config: ADCensusConfig = ADCensusConfig()
    application_id = "panorai.applications.ADCensusStereoDepth"
    method_name = "AD-Census + CBCA + SGM"

    def _pipeline(self) -> Pipeline:
        return ad_census_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ADCensusStereoDepth":
        return cls(
            _from_config(
                payload, application_id=cls.application_id, config_type=ADCensusConfig
            )
        )


@dataclass(frozen=True, slots=True)
class StereoSGBMDepth(_RunMixin, PipelineApplication):
    """OpenCV full-mode StereoSGBM baseline."""

    config: StereoSGBMConfig = StereoSGBMConfig()
    application_id = "panorai.applications.StereoSGBMDepth"
    method_name = "OpenCV StereoSGBM HH"

    def _pipeline(self) -> Pipeline:
        return stereo_sgbm_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "StereoSGBMDepth":
        return cls(
            _from_config(
                payload,
                application_id=cls.application_id,
                config_type=StereoSGBMConfig,
            )
        )


@dataclass(frozen=True, slots=True)
class CensusMGMStereoDepth(_RunMixin, PipelineApplication):
    """Census with MGM or tMGM two-dimensional regularization."""

    config: CensusMGMConfig = CensusMGMConfig()
    application_id = "panorai.applications.CensusMGMStereoDepth"
    method_name = "Census + MGM/tMGM"

    def _pipeline(self) -> Pipeline:
        return census_mgm_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "CensusMGMStereoDepth":
        return cls(
            _from_config(
                payload,
                application_id=cls.application_id,
                config_type=CensusMGMConfig,
            )
        )


@dataclass(frozen=True, slots=True)
class ELASStereoDepth(_RunMixin, PipelineApplication):
    """Inspectable ELAS-style support-point prior baseline."""

    config: ELASConfig = ELASConfig()
    application_id = "panorai.applications.ELASStereoDepth"
    method_name = "PanorAi ELAS-style support prior"

    def _pipeline(self) -> Pipeline:
        return elas_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ELASStereoDepth":
        return cls(
            _from_config(
                payload, application_id=cls.application_id, config_type=ELASConfig
            )
        )


@dataclass(frozen=True, slots=True)
class PatchMatchStereoDepth(_RunMixin, PipelineApplication):
    """Deterministic slanted-plane PatchMatch Stereo reference baseline."""

    config: PatchMatchConfig = PatchMatchConfig()
    application_id = "panorai.applications.PatchMatchStereoDepth"
    method_name = "PatchMatch Stereo"

    def _pipeline(self) -> Pipeline:
        return patchmatch_pipeline(self.config)

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "PatchMatchStereoDepth":
        return cls(
            _from_config(
                payload,
                application_id=cls.application_id,
                config_type=PatchMatchConfig,
            )
        )


__all__ = [
    "ADCensusStereoDepth",
    "CensusMGMStereoDepth",
    "ELASStereoDepth",
    "PatchMatchStereoDepth",
    "StereoBMDepth",
    "StereoSGBMDepth",
]
