"""High-level central rotation-only panorama stitching application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np

from panorai.pipeline import Pipeline
from panorai.stitching import (
    BlendStitchingPanorama,
    CalibrateAndRegister,
    CompensateExposure,
    DetectStitchingFeatures,
    DiscoverOverlaps,
    PanoramaStitchingResult,
    StitchingInput,
    StitchingOptions,
    WarpRegisteredImages,
)

from ._base import PipelineApplication, application_config


ImageSequence = Annotated[
    StitchingInput | Sequence[np.ndarray], "panorai:image-sequence"
]


@dataclass(frozen=True, slots=True)
class PanoramaStitcher(PipelineApplication):
    """Register overlapping pinhole images and construct one partial ERP.

    The first profile is deliberately narrow: every input is assumed to come
    from a central pinhole camera rotating about one optical centre.  Missing
    horizontal field of view may be estimated only when the matched evidence
    makes it observable.  Translation, strong parallax, rolling shutter and
    disconnected overlap groups fail explicitly instead of being hidden by a
    visually plausible warp.
    """

    options: StitchingOptions = StitchingOptions()

    application_id = "panorai.applications.PanoramaStitcher"
    interface = "panorai-panorama-stitching/v1-experimental"
    stability = "experimental"
    config_format = "panorai-panorama-stitching-config/v1-experimental"
    parameter_names: ClassVar[frozenset[str]] = frozenset({"options"})

    def __post_init__(self) -> None:
        if not isinstance(self.options, StitchingOptions):
            raise TypeError("options must be StitchingOptions")

    def as_pipeline(self) -> Pipeline:
        return Pipeline(
            DetectStitchingFeatures(self.options),
            DiscoverOverlaps(self.options),
            CalibrateAndRegister(self.options),
            WarpRegisteredImages(self.options),
            CompensateExposure(self.options),
            BlendStitchingPanorama(self.options),
        )

    def run(
        self,
        images: ImageSequence,
        *,
        image_ids: Any = None,
        support_masks: Any = None,
        validity_masks: Any = None,
        horizontal_fov_deg: Any = None,
    ) -> PanoramaStitchingResult:
        if isinstance(images, StitchingInput):
            if any(
                item is not None
                for item in (
                    image_ids,
                    support_masks,
                    validity_masks,
                    horizontal_fov_deg,
                )
            ):
                raise ValueError(
                    "per-image arguments cannot accompany a StitchingInput"
                )
            stitching_input = images
        else:
            stitching_input = StitchingInput.from_arrays(
                images,
                image_ids=image_ids,
                support_masks=support_masks,
                validity_masks=validity_masks,
                horizontal_fov_deg=horizontal_fov_deg,
            )
        result = self.as_pipeline()(stitching_input)
        if not isinstance(result, PanoramaStitchingResult):
            raise RuntimeError("stitching pipeline returned an unexpected result")
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "N>=2 overlapping planar pinhole images",
            "output": "partial EquirectangularImage plus registration diagnostics",
            "model": "central rotation-only",
            "estimated": "overlap graph, missing horizontal FOV, SO(3) orientations, gains",
            "rejected": (
                "disconnected overlap, unobservable FOV, insufficient texture, "
                "incompatible central-rotation evidence"
            ),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(
            self.application_id,
            self.get_params(deep=False),
            format_id=self.config_format,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PanoramaStitcher":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe PanoramaStitcher")
        if config.get("format") != cls.config_format:
            raise ValueError("unknown PanoramaStitcher configuration format")
        parameters = config.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("PanoramaStitcher parameters must be a mapping")
        unknown = sorted(set(parameters) - cls.parameter_names)
        if unknown:
            raise ValueError(
                "unknown PanoramaStitcher parameters: " + ", ".join(unknown)
            )
        raw_options = parameters.get("options", {})
        if not isinstance(raw_options, dict):
            raise ValueError("PanoramaStitcher options must be a mapping")
        options = dict(raw_options)
        if "output_shape_hw" in options:
            value = options["output_shape_hw"]
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("output_shape_hw must contain height and width")
            options["output_shape_hw"] = tuple(value)
        try:
            return cls(StitchingOptions(**options))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid PanoramaStitcher parameters: {exc}") from exc


__all__ = ["PanoramaStitcher"]
