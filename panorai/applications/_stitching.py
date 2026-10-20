"""High-level central rotation-only panorama stitching application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

from panorai.pipeline import Pipeline
from panorai.stitching import (
    BlendStitchingPanorama,
    CalibrateAndRegister,
    CompensateExposure,
    DetectStitchingFeatures,
    DiscoverOverlaps,
    DiscoverRobustOverlaps,
    PanoramaStitchingResult,
    StitchingInput,
    StitchingOptions,
    StitchingRecoveryOptions,
    WarpRegisteredImages,
)

from ._base import PipelineApplication, application_config


STITCHING_REVIEWED_PROFILE_ID = "panorai-panorama-stitching/rotation-only-reviewed-v1"
STITCHING_REVIEWED_INTERFACE = "panorai-panorama-stitching/rotation-only-v1"
STITCHING_REVIEWED_CONFIG_FORMAT = "panorai-panorama-stitching-config/v1"
STITCHING_EXPERIMENTAL_CONFIG_FORMAT = (
    "panorai-panorama-stitching-config/v1-experimental"
)
STITCHING_ROBUST_PROFILE_ID = (
    "panorai-panorama-stitching/robust-rotation-v2-experimental"
)
STITCHING_ROBUST_INTERFACE = "panorai-panorama-stitching/rotation-only-v2-experimental"
STITCHING_ROBUST_CONFIG_FORMAT = "panorai-panorama-stitching-config/v2-experimental"


def _reviewed_options() -> StitchingOptions:
    """Return every scientific and rendering choice frozen by reviewed-v1."""

    return StitchingOptions(
        feature_method="sift",
        max_features=4000,
        ratio_test=0.75,
        min_pair_matches=20,
        min_pair_inliers=12,
        max_transfer_error_px=3.0,
        max_angular_error_deg=1.0,
        max_median_angular_error_deg=0.35,
        min_horizontal_fov_deg=30.0,
        max_horizontal_fov_deg=140.0,
        min_spatial_coverage=0.02,
        output_shape_hw=(512, 1024),
        interpolation="bilinear",
        exposure_compensation="gain",
        blend="feather",
        random_seed=0,
    )


def _robust_recovery_options() -> StitchingRecoveryOptions:
    """Return the selective recovery choices for robust rotation v2."""

    return StitchingRecoveryOptions(
        ratio_test=0.85,
        min_pair_matches=8,
        min_pair_inliers=8,
        min_feature_density=5.0e-4,
        feature_density_count_cap=128,
    )


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
    recovery: StitchingRecoveryOptions | None = None

    application_id = "panorai.applications.PanoramaStitcher"
    interface = "panorai-panorama-stitching/v1-experimental"
    stability = "experimental"
    config_format = STITCHING_REVIEWED_CONFIG_FORMAT
    legacy_config_formats = (
        STITCHING_EXPERIMENTAL_CONFIG_FORMAT,
        STITCHING_ROBUST_CONFIG_FORMAT,
    )
    profile_id: ClassVar[str | None] = None
    parameter_names: ClassVar[frozenset[str]] = frozenset({"options", "recovery"})

    def __post_init__(self) -> None:
        if not isinstance(self.options, StitchingOptions):
            raise TypeError("options must be StitchingOptions")
        if self.recovery is not None and not isinstance(
            self.recovery, StitchingRecoveryOptions
        ):
            raise TypeError("recovery must be StitchingRecoveryOptions or None")

    @classmethod
    def reviewed_profile(cls) -> "PanoramaStitcher":
        """Construct the frozen central rotation-only v1 candidate."""

        return _ReviewedPanoramaStitcher(_reviewed_options())

    @classmethod
    def robust_rotation_profile(cls) -> "PanoramaStitcher":
        """Construct the Experimental selective-recovery v2 profile."""

        return _RobustRotationPanoramaStitcher(
            _reviewed_options(), _robust_recovery_options()
        )

    def as_pipeline(self) -> Pipeline:
        overlap_stage = (
            DiscoverOverlaps(self.options)
            if self.recovery is None
            else DiscoverRobustOverlaps(self.options, self.recovery)
        )
        return Pipeline(
            DetectStitchingFeatures(self.options),
            overlap_stage,
            CalibrateAndRegister(self.options),
            WarpRegisteredImages(self.options),
            CompensateExposure(self.options),
            BlendStitchingPanorama(self.options),
        )

    def run(
        self,
        images: Any,
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
        if self.profile_id is not None:
            return replace(
                result,
                interface=self.interface,
                stability=self.stability,
                profile_id=self.profile_id,
            )
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "profile_id": self.profile_id,
            "selective_recovery": self.recovery is not None,
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
        if self.profile_id == STITCHING_REVIEWED_PROFILE_ID:
            return application_config(
                self.application_id,
                {"profile_id": self.profile_id, "options": self.options},
                format_id=self.config_format,
            )
        if self.profile_id == STITCHING_ROBUST_PROFILE_ID:
            return application_config(
                self.application_id,
                {
                    "profile_id": self.profile_id,
                    "options": self.options,
                    "recovery": self.recovery,
                },
                format_id=STITCHING_ROBUST_CONFIG_FORMAT,
            )
        parameters: dict[str, Any] = {"options": self.options}
        format_id = STITCHING_EXPERIMENTAL_CONFIG_FORMAT
        if self.recovery is not None:
            parameters["recovery"] = self.recovery
            format_id = STITCHING_ROBUST_CONFIG_FORMAT
        return application_config(
            self.application_id,
            parameters,
            format_id=format_id,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PanoramaStitcher":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe PanoramaStitcher")
        format_id = config.get("format")
        if format_id not in {cls.config_format, *cls.legacy_config_formats}:
            raise ValueError("unknown PanoramaStitcher configuration format")
        parameters = config.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("PanoramaStitcher parameters must be a mapping")
        allowed_parameters = set(cls.parameter_names)
        if format_id in {cls.config_format, STITCHING_ROBUST_CONFIG_FORMAT}:
            allowed_parameters.add("profile_id")
        unknown = sorted(set(parameters) - allowed_parameters)
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
            restored = StitchingOptions(**options)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid PanoramaStitcher parameters: {exc}") from exc

        raw_recovery = parameters.get("recovery")
        recovery = None
        if raw_recovery is not None:
            if not isinstance(raw_recovery, dict):
                raise ValueError("PanoramaStitcher recovery must be a mapping")
            try:
                recovery = StitchingRecoveryOptions(**raw_recovery)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid PanoramaStitcher recovery parameters: {exc}"
                ) from exc

        if format_id == cls.config_format:
            if parameters.get("profile_id") != STITCHING_REVIEWED_PROFILE_ID:
                raise ValueError("unknown PanoramaStitcher reviewed profile")
            if restored != _reviewed_options():
                raise ValueError("reviewed stitching profile parameters have drifted")
            if recovery is not None:
                raise ValueError("reviewed v1 profile cannot contain recovery options")
            return cls.reviewed_profile()
        if format_id == STITCHING_ROBUST_CONFIG_FORMAT:
            profile_id = parameters.get("profile_id")
            if profile_id is not None:
                if profile_id != STITCHING_ROBUST_PROFILE_ID:
                    raise ValueError("unknown PanoramaStitcher robust profile")
                if (
                    restored != _reviewed_options()
                    or recovery != _robust_recovery_options()
                ):
                    raise ValueError("robust stitching profile parameters have drifted")
                return cls.robust_rotation_profile()
            if recovery is None:
                raise ValueError("robust stitching configuration requires recovery")
            return cls(restored, recovery)
        if "profile_id" in parameters:
            raise ValueError(
                "experimental stitching configuration cannot claim a profile"
            )
        if recovery is not None:
            raise ValueError("v1 experimental config cannot contain recovery options")
        return cls(restored)


class _ReviewedPanoramaStitcher(PanoramaStitcher):
    """Runtime identity for the frozen central-rotation candidate profile."""

    interface = STITCHING_REVIEWED_INTERFACE
    stability = "experimental"
    profile_id = STITCHING_REVIEWED_PROFILE_ID

    def with_params(self, **params: Any) -> PanoramaStitcher:
        changed = PanoramaStitcher(self.options, self.recovery).with_params(**params)
        if not isinstance(changed, PanoramaStitcher):  # pragma: no cover
            raise RuntimeError("stitching parameter replacement returned wrong type")
        if changed.options == _reviewed_options():
            return PanoramaStitcher.reviewed_profile()
        return changed


class _RobustRotationPanoramaStitcher(PanoramaStitcher):
    """Runtime identity for the Experimental robust rotation v2 profile."""

    interface = STITCHING_ROBUST_INTERFACE
    stability = "experimental"
    profile_id = STITCHING_ROBUST_PROFILE_ID

    def with_params(self, **params: Any) -> PanoramaStitcher:
        changed = PanoramaStitcher(self.options, self.recovery).with_params(**params)
        if not isinstance(changed, PanoramaStitcher):  # pragma: no cover
            raise RuntimeError("stitching parameter replacement returned wrong type")
        if (
            changed.options == _reviewed_options()
            and changed.recovery == _robust_recovery_options()
        ):
            return PanoramaStitcher.robust_rotation_profile()
        return changed


__all__ = [
    "PanoramaStitcher",
    "STITCHING_REVIEWED_CONFIG_FORMAT",
    "STITCHING_REVIEWED_INTERFACE",
    "STITCHING_REVIEWED_PROFILE_ID",
    "STITCHING_ROBUST_CONFIG_FORMAT",
    "STITCHING_ROBUST_INTERFACE",
    "STITCHING_ROBUST_PROFILE_ID",
]
