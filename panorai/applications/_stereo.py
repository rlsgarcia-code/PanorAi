"""High-level metric-pose spherical stereo application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np

from panorai.pipeline import Pipeline, StageResult
from panorai.stereo import (
    SphericalDenseStereo,
    SphericalStereoOptions,
    SphericalStereoResult,
)

from ._base import PipelineApplication, application_config


@dataclass(frozen=True, slots=True)
class SphericalStereoInput:
    """Typed low-level input for :meth:`SphericalStereoDepth.as_pipeline`.

    The pose convention is ``X_source = R @ X_reference + t``.  Translation
    and returned radial range use the same physical unit.
    """

    reference: Annotated[np.ndarray, "panorai:erp-rgb"]
    source: Annotated[np.ndarray, "panorai:erp-rgb"]
    rotation_source_from_reference: Annotated[
        np.ndarray, "panorai:rotation-source-from-reference"
    ]
    translation_source_from_reference: Annotated[
        np.ndarray, "panorai:translation-source-from-reference"
    ]


@dataclass(frozen=True, slots=True)
class _StereoStage:
    options: SphericalStereoOptions

    input_kind = "stereo_pair"
    output_kind = "result"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, SphericalStereoInput):
            raise TypeError("spherical stereo stage requires SphericalStereoInput")
        result = SphericalDenseStereo(self.options).estimate(
            value.reference,
            value.source,
            value.rotation_source_from_reference,
            value.translation_source_from_reference,
        )
        return StageResult(
            result,
            {
                "domain": "direct-spherical",
                "quantity": "radial_range",
                "translation_units": "inherited-by-output-range",
                "valid_fraction": float(result.validity_mask.mean()),
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalStereoDepth(PipelineApplication):
    """Estimate radial range from two calibrated central ERPs."""

    options: SphericalStereoOptions = SphericalStereoOptions()

    application_id = "panorai.applications.SphericalStereoDepth"

    def as_pipeline(self) -> Pipeline:
        return Pipeline(_StereoStage(self.options))

    def run(
        self,
        reference: Annotated[np.ndarray, "panorai:erp-rgb"],
        source: Annotated[np.ndarray, "panorai:erp-rgb"],
        rotation_source_from_reference: Annotated[
            np.ndarray, "panorai:rotation-source-from-reference"
        ],
        translation_source_from_reference: Annotated[
            np.ndarray, "panorai:translation-source-from-reference"
        ],
    ) -> SphericalStereoResult:
        value = SphericalStereoInput(
            reference,
            source,
            rotation_source_from_reference,
            translation_source_from_reference,
        )
        result = self.as_pipeline()(value)
        if not isinstance(result, SphericalStereoResult):
            raise RuntimeError(
                "spherical stereo pipeline returned an unexpected result"
            )
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "two canonical ERPs plus metric relative pose",
            "output": "SphericalStereoResult with radial range/confidence/validity",
            "pose": "X_source = R @ X_reference + t",
            "units": "translation unit inherited by radial range",
            "domain": "direct-spherical inverse-range plane sweep",
            "options": self.options.to_dict(),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(self.application_id, {"options": self.options})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SphericalStereoDepth":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe SphericalStereoDepth")
        parameters = config.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(
            parameters.get("options"), dict
        ):
            raise ValueError("SphericalStereoDepth options are missing")
        return cls(SphericalStereoOptions(**parameters["options"]))


__all__ = ["SphericalStereoDepth", "SphericalStereoInput"]
