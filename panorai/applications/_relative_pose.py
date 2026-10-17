"""High-level spherical Essential/R/t application."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from panorai.data import EquirectangularImage
from panorai.estimators import (
    EstimateRelativePose,
    FivePointSampler,
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    RelativePoseResult,
    SpatiallyWeightedFivePointSampler,
    UniformFivePointSampler,
)
from panorai.features import (
    ExtractSphericalFeaturePair,
    FaceSetSpec,
    FeatureExtractorConfig,
    FeatureMatcherConfig,
    MatchSphericalFeatures,
    SphericalBearingCorrespondences,
    SphericalFeatureMatches,
    SphericalFeaturePipeline,
    SphericalFeaturePipelineConfig,
    SphericalImagePair,
    ToBearingCorrespondences,
)
from panorai.pipeline import Pipeline

from ._base import (
    ApplicationConfigurationError,
    PipelineApplication,
    application_config,
)
from ._relative_pose_capture import (
    AssessRelativePoseCapture,
    RelativePoseCaptureAssessment,
    RelativePoseCaptureInput,
    RelativePoseCaptureMetadata,
    RelativePoseCapturePolicy,
)


_REFERENCE_PROFILE_ID = "panorai-relative-pose-reference-v1-experimental"
_REFERENCE_EVIDENCE_SCOPE = (
    "frozen spherical frontend validation and metadata-separated real-pair "
    "MSAC/refit selection; the complete application remains Experimental"
)

_REFERENCE_OPTIONS_V1: dict[str, Any] = {
    "max_angular_error_deg": 1.0,
    "confidence": 0.999,
    "min_inlier_ratio": 0.1,
    "min_num_trials": 32,
    "max_num_trials": 1000,
    "dynamic_trials_multiplier": 3.0,
    "min_inliers": 8,
    "local_optimization_steps": 3,
    "minimal_solver_starts": 16,
    "minimal_solver_max_nfev": 100,
    "refinement_max_nfev": 100,
    "random_seed": 0,
    "min_median_parallax_deg": 0.25,
    "minimal_solver": "polynomial",
    "scale_marginal_levels": 8,
    "scale_marginal_min_fraction": 0.2,
    "robust_refinement_steps": 2,
    "quality_cell_count": 20,
    "stability_trials": 6,
    "stability_fraction": 0.8,
    "stability_ransac_trials": 24,
    "model_competition_trials": 128,
    "model_competition_tie_margin": 0.01,
    "translation_orientation_method": "parallax-weighted",
    "translation_orientation_parallax_scale_deg": 1.0,
    "compute_backend": "auto",
    "hypothesis_ranking": "msac-first",
    "nonminimal_refit_max_steps": 100,
    "pose_refinement_method": "joint",
    "decoupled_rotation_trials": 512,
    "decoupled_refit_max_steps": 100,
    "decoupled_translation_min_score_margin": 0.15,
}

_REFERENCE_SAMPLER_V1: dict[str, Any] = {
    "min_angular_separation_deg": 8.0,
    "min_unique_cells": 4,
    "spherical_cell_count": 20,
    "density_radius_deg": 10.0,
    "density_power": 0.0,
    "diversity_power": 0.5,
    "uniform_trial_probability": 0.2,
    "max_design_condition_number": 1.0e6,
    "attempts_per_level": 24,
    "name": "spatially-weighted-five-point-v1",
    "supports_uniform_trial_bound": False,
}

_REFERENCE_ACCEPTANCE_V1: dict[str, Any] = {
    "min_inliers": 20,
    "min_inlier_ratio": 0.15,
    "min_occupied_cells": 4,
    "min_coverage_entropy": 0.25,
    "max_median_residual_deg": 0.5,
    "min_median_parallax_deg": 0.5,
    "min_cheirality_ratio": 0.6,
    "min_translation_orientation_margin": 0.05,
    "max_stability_rotation_p90_deg": 3.0,
    "max_stability_translation_p90_deg": 10.0,
    "min_essential_score_margin": 0.0,
    "require_essential_preferred": True,
    "require_stability": True,
    "interface": "panorai-relative-pose-acceptance/v1",
}


def _construct_frozen(
    value_type: type[Any], parameters: dict[str, Any], *, profile_part: str
) -> Any:
    """Fail closed when a versioned profile's dataclass schema changes."""

    declared = {item.name for item in fields(value_type) if item.init}
    frozen = set(parameters)
    if declared != frozen:
        missing = sorted(declared - frozen)
        stale = sorted(frozen - declared)
        raise RuntimeError(
            f"{_REFERENCE_PROFILE_ID} {profile_part} schema changed; "
            f"missing={missing}, stale={stale}. Define a reviewed profile "
            "revision instead of inheriting new defaults."
        )
    return value_type(**parameters)


def _as_erp(
    value: Any, validity: np.ndarray | None, *, name: str
) -> EquirectangularImage:
    if isinstance(value, EquirectangularImage):
        if validity is not None:
            raise ValueError(
                f"{name}_validity is accepted only with an array input; "
                "EquirectangularImage already carries explicit validity"
            )
        return value
    return EquirectangularImage(value, valid=validity)


def _feature_config_from_dict(value: Any) -> SphericalFeaturePipelineConfig:
    if not isinstance(value, dict):
        raise ApplicationConfigurationError("frontend_config must be a mapping")
    expected = {
        "interface",
        "stability",
        "preset_name",
        "preset_version",
        "minimum_opencv_version",
        "extractor",
        "matcher",
        "face_set",
    }
    if set(value) != expected:
        raise ApplicationConfigurationError(
            "frontend_config has unknown or missing parameters"
        )
    if value["stability"] != "stable":
        raise ApplicationConfigurationError(
            "frontend_config stability must be 'stable'"
        )
    extractor = value["extractor"]
    matcher = value["matcher"]
    face_set = value["face_set"]
    if not all(isinstance(item, dict) for item in (extractor, matcher, face_set)):
        raise ApplicationConfigurationError(
            "frontend extractor, matcher, and face_set must be mappings"
        )
    face_values = dict(face_set)
    effective_fov = face_values.pop("effective_fov_deg", None)
    try:
        result = SphericalFeaturePipelineConfig(
            extractor=FeatureExtractorConfig(**extractor),
            matcher=FeatureMatcherConfig(**matcher),
            face_set=FaceSetSpec(**face_values),
            interface=value["interface"],
            preset_name=value["preset_name"],
            preset_version=value["preset_version"],
            minimum_opencv_version=value["minimum_opencv_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigurationError(f"invalid frontend_config: {exc}") from exc
    try:
        declared_effective_fov = tuple(effective_fov)
    except TypeError as exc:
        raise ApplicationConfigurationError(
            "frontend face_set effective_fov_deg must contain two values"
        ) from exc
    if declared_effective_fov != result.face_set.effective_fov_deg:
        raise ApplicationConfigurationError(
            "frontend face_set effective_fov_deg does not match fov_deg + overlap_deg"
        )
    return result


def _sampler_to_config(value: Any) -> dict[str, Any]:
    if isinstance(value, SpatiallyWeightedFivePointSampler):
        kind = "spatially-weighted"
    elif isinstance(value, UniformFivePointSampler):
        kind = "uniform"
    else:
        raise ApplicationConfigurationError(
            "sampler is executable in memory but is not registered for safe "
            "application persistence"
        )
    parameters = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in {"name", "supports_uniform_trial_bound"}
    }
    return {"kind": kind, "parameters": parameters}


def _sampler_from_config(value: Any) -> Any:
    if not isinstance(value, dict) or set(value) != {"kind", "parameters"}:
        raise ApplicationConfigurationError("sampler configuration is invalid")
    if not isinstance(value["parameters"], dict):
        raise ApplicationConfigurationError("sampler parameters must be a mapping")
    sampler_type = {
        "spatially-weighted": SpatiallyWeightedFivePointSampler,
        "uniform": UniformFivePointSampler,
    }.get(value["kind"])
    if sampler_type is None:
        raise ApplicationConfigurationError(
            f"unregistered sampler kind {value['kind']!r}"
        )
    try:
        return sampler_type(**value["parameters"])
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigurationError(
            f"invalid sampler parameters: {exc}"
        ) from exc


def _capture_policy_from_config(value: Any) -> RelativePoseCapturePolicy | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ApplicationConfigurationError("capture_policy must be a mapping or null")
    parameters = dict(value)
    interval = parameters.get("metric_baseline_interval_m")
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ApplicationConfigurationError(
                "capture_policy metric_baseline_interval_m must contain two values"
            )
        parameters["metric_baseline_interval_m"] = tuple(interval)
    try:
        return RelativePoseCapturePolicy(**parameters)
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigurationError(f"invalid capture_policy: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RelativePoseProfile:
    """Complete, named configuration of the relative-pose computation.

    Capture eligibility is intentionally not part of this object: it is an
    independent pre-estimation policy with different evidence and inputs.
    """

    profile_id: str
    frontend_config: SphericalFeaturePipelineConfig | None
    options: RelativePoseOptions
    sampler: FivePointSampler
    quality_policy: RelativePoseAcceptancePolicy
    evidence_scope: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be a non-empty string")
        if self.frontend_config is not None and not isinstance(
            self.frontend_config, SphericalFeaturePipelineConfig
        ):
            raise TypeError(
                "frontend_config must be SphericalFeaturePipelineConfig or None"
            )
        if not isinstance(self.options, RelativePoseOptions):
            raise TypeError("options must be RelativePoseOptions")
        if not isinstance(self.quality_policy, RelativePoseAcceptancePolicy):
            raise TypeError("quality_policy must be RelativePoseAcceptancePolicy")
        if not callable(getattr(self.sampler, "prepare", None)) or not callable(
            getattr(self.sampler, "describe", None)
        ):
            raise TypeError("sampler must implement the FivePointSampler protocol")
        if not isinstance(self.evidence_scope, str) or not self.evidence_scope:
            raise ValueError("evidence_scope must be a non-empty string")
        if self.profile_id == _REFERENCE_PROFILE_ID:
            expected = (
                None,
                _construct_frozen(
                    RelativePoseOptions,
                    _REFERENCE_OPTIONS_V1,
                    profile_part="options",
                ),
                _construct_frozen(
                    SpatiallyWeightedFivePointSampler,
                    _REFERENCE_SAMPLER_V1,
                    profile_part="sampler",
                ),
                _construct_frozen(
                    RelativePoseAcceptancePolicy,
                    _REFERENCE_ACCEPTANCE_V1,
                    profile_part="acceptance policy",
                ),
                _REFERENCE_EVIDENCE_SCOPE,
            )
            observed = (
                self.frontend_config,
                self.options,
                self.sampler,
                self.quality_policy,
                self.evidence_scope,
            )
            if observed != expected:
                raise ValueError(
                    f"reserved profile_id {_REFERENCE_PROFILE_ID!r} requires "
                    "the canonical v1 payload"
                )

    @classmethod
    def evidence_backed_v1(cls) -> "RelativePoseProfile":
        """Return the frozen frontend plus selected MSAC/refit estimator."""

        return cls(
            profile_id=_REFERENCE_PROFILE_ID,
            frontend_config=None,
            options=_construct_frozen(
                RelativePoseOptions,
                _REFERENCE_OPTIONS_V1,
                profile_part="options",
            ),
            sampler=_construct_frozen(
                SpatiallyWeightedFivePointSampler,
                _REFERENCE_SAMPLER_V1,
                profile_part="sampler",
            ),
            quality_policy=_construct_frozen(
                RelativePoseAcceptancePolicy,
                _REFERENCE_ACCEPTANCE_V1,
                profile_part="acceptance policy",
            ),
            evidence_scope=_REFERENCE_EVIDENCE_SCOPE,
        )

    @property
    def resolved_frontend_config(self) -> SphericalFeaturePipelineConfig:
        if self.frontend_config is not None:
            return self.frontend_config
        return SphericalFeaturePipeline.for_relative_pose().config

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "frontend_mode": (
                "reference" if self.frontend_config is None else "custom"
            ),
            "frontend_config": self.resolved_frontend_config.to_dict(),
            "options": self.options.to_dict(),
            "sampler": self.sampler.describe(),
            "quality_policy": self.quality_policy.to_dict(),
            "evidence_scope": self.evidence_scope,
        }


@dataclass(frozen=True, slots=True)
class SphericalRelativePose(PipelineApplication):
    """Estimate spherical Essential, rotation, and translation direction.

    By default the image frontend calls
    ``SphericalFeaturePipeline.for_relative_pose()`` exactly. Passing a
    ``frontend_config`` explicitly selects a custom extraction/matching
    configuration.  The result never claims metric translation scale.
    """

    frontend_config: SphericalFeaturePipelineConfig | None = None
    options: RelativePoseOptions = field(default_factory=RelativePoseOptions)
    sampler: FivePointSampler = field(default_factory=SpatiallyWeightedFivePointSampler)
    quality_policy: RelativePoseAcceptancePolicy = field(
        default_factory=RelativePoseAcceptancePolicy
    )
    profile: RelativePoseProfile | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    capture_policy: RelativePoseCapturePolicy | None = None
    capture_mode: str = "enforce"

    application_id = "panorai.applications.SphericalRelativePose"
    interface = "panorai-spherical-relative-pose-application/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if self.frontend_config is not None and not isinstance(
            self.frontend_config, SphericalFeaturePipelineConfig
        ):
            raise TypeError(
                "frontend_config must be SphericalFeaturePipelineConfig or None"
            )
        if not isinstance(self.options, RelativePoseOptions):
            raise TypeError("options must be RelativePoseOptions")
        if (
            not callable(getattr(self.sampler, "prepare", None))
            or not callable(getattr(self.sampler, "describe", None))
            or not isinstance(getattr(self.sampler, "name", None), str)
            or not self.sampler.name
        ):
            raise TypeError("sampler must implement the FivePointSampler protocol")
        if not isinstance(self.quality_policy, RelativePoseAcceptancePolicy):
            raise TypeError("quality_policy must be RelativePoseAcceptancePolicy")
        if self.profile is not None:
            if not isinstance(self.profile, RelativePoseProfile):
                raise TypeError("profile must be RelativePoseProfile or None")
            if (
                self.frontend_config != self.profile.frontend_config
                or self.options != self.profile.options
                or self.sampler != self.profile.sampler
                or self.quality_policy != self.profile.quality_policy
            ):
                raise ValueError(
                    "profile identity does not match the application computation"
                )
        if self.capture_policy is not None and not isinstance(
            self.capture_policy, RelativePoseCapturePolicy
        ):
            raise TypeError("capture_policy must be RelativePoseCapturePolicy or None")
        if self.capture_mode not in {"enforce", "advisory"}:
            raise ValueError("capture_mode must be 'enforce' or 'advisory'")

    @classmethod
    def reference(
        cls,
        *,
        capture_policy: RelativePoseCapturePolicy | None = None,
        capture_mode: str = "enforce",
    ) -> "SphericalRelativePose":
        """Construct the named evidence-backed v1 computation profile."""

        return cls.from_profile(
            RelativePoseProfile.evidence_backed_v1(),
            capture_policy=capture_policy,
            capture_mode=capture_mode,
        )

    @classmethod
    def from_profile(
        cls,
        profile: RelativePoseProfile,
        *,
        capture_policy: RelativePoseCapturePolicy | None = None,
        capture_mode: str = "enforce",
    ) -> "SphericalRelativePose":
        """Construct an application from an inspectable computation profile."""

        if not isinstance(profile, RelativePoseProfile):
            raise TypeError("profile must be RelativePoseProfile")
        return cls(
            frontend_config=profile.frontend_config,
            options=profile.options,
            sampler=profile.sampler,
            quality_policy=profile.quality_policy,
            profile=profile,
            capture_policy=capture_policy,
            capture_mode=capture_mode,
        )

    @property
    def resolved_frontend_config(self) -> SphericalFeaturePipelineConfig:
        """Return the exact frontend configuration this instance will use."""

        if self.frontend_config is not None:
            return self.frontend_config
        return SphericalFeaturePipeline.for_relative_pose().config

    @property
    def resolved_profile(self) -> RelativePoseProfile | None:
        """Return the named profile when this instance matches it exactly."""

        if self.profile is not None:
            return self.profile
        reference = RelativePoseProfile.evidence_backed_v1()
        if (
            self.frontend_config == reference.frontend_config
            and self.options == reference.options
            and self.sampler == reference.sampler
            and self.quality_policy == reference.quality_policy
        ):
            return reference
        return None

    def with_params(self, **params: Any) -> "SphericalRelativePose":
        """Return an immutable variant and invalidate changed profile identity."""

        roots = {name.partition("__")[0] for name in params}
        if "profile" in roots:
            raise KeyError("profile identity is set only through from_profile()")
        if self.profile is None:
            updated = PipelineApplication.with_params(self, **params)
            assert isinstance(updated, SphericalRelativePose)
            return updated
        source = replace(self, profile=None)
        updated = PipelineApplication.with_params(source, **params)
        assert isinstance(updated, SphericalRelativePose)
        before = (
            self.frontend_config,
            self.options,
            self.sampler,
            self.quality_policy,
        )
        after = (
            updated.frontend_config,
            updated.options,
            updated.sampler,
            updated.quality_policy,
        )
        if after == before:
            return replace(updated, profile=self.profile)
        return updated

    def as_pipeline(self) -> Pipeline:
        steps: list[Any] = []
        if self.capture_policy is not None:
            steps.append(
                (
                    "assess_capture",
                    AssessRelativePoseCapture(self.capture_policy, self.capture_mode),
                )
            )
        steps.extend(
            (
                ("extract_features", ExtractSphericalFeaturePair(self.frontend_config)),
                ("match_features", MatchSphericalFeatures(self.frontend_config)),
                ("bearing_correspondences", ToBearingCorrespondences()),
                (
                    "estimate_pose",
                    EstimateRelativePose(
                        self.options,
                        self.sampler,
                        self.quality_policy,
                    ),
                ),
            )
        )
        return Pipeline(*steps)

    def run(
        self,
        reference: Any,
        source: Any,
        *,
        reference_id: str | None = None,
        source_id: str | None = None,
        reference_validity: np.ndarray | None = None,
        source_validity: np.ndarray | None = None,
        capture_metadata: RelativePoseCaptureMetadata | None = None,
    ) -> RelativePoseResult | None:
        pair = SphericalImagePair(
            _as_erp(reference, reference_validity, name="reference"),
            _as_erp(source, source_validity, name="source"),
            reference_id,
            source_id,
        )
        pipeline_input: Any = pair
        if self.capture_policy is not None:
            if capture_metadata is None:
                raise ValueError(
                    "capture_metadata is required when capture_policy is configured"
                )
            pipeline_input = RelativePoseCaptureInput(pair, capture_metadata)
        elif capture_metadata is not None:
            raise ValueError(
                "capture_metadata requires an explicit capture_policy; metadata "
                "must never be accepted and then ignored"
            )
        result = self.as_pipeline()(pipeline_input)
        if result is not None and not isinstance(result, RelativePoseResult):
            raise RuntimeError("relative-pose pipeline returned an unexpected result")
        return result

    def assess_capture(
        self,
        metadata: RelativePoseCaptureMetadata,
        *,
        policy: RelativePoseCapturePolicy | None = None,
    ) -> RelativePoseCaptureAssessment:
        """Assess capture facts without extracting features or estimating pose."""

        resolved = policy or self.capture_policy
        if resolved is None:
            raise ValueError("assess_capture requires an explicit capture policy")
        return resolved.assess(metadata)

    def estimate_matches(
        self, matches: SphericalFeatureMatches
    ) -> RelativePoseResult | None:
        """Estimate pose from already matched panorama-frame features."""

        pipeline = Pipeline(
            ToBearingCorrespondences(),
            EstimateRelativePose(self.options, self.sampler, self.quality_policy),
        )
        return pipeline(matches)

    def estimate_correspondences(
        self, correspondences: SphericalBearingCorrespondences
    ) -> RelativePoseResult | None:
        """Estimate pose from precomputed weighted bearing correspondences."""

        return Pipeline(
            EstimateRelativePose(self.options, self.sampler, self.quality_policy)
        )(correspondences)

    def describe(self) -> dict[str, Any]:
        resolved = self.resolved_frontend_config
        profile = self.resolved_profile
        profile_mode = "custom"
        if profile is not None:
            profile_mode = (
                "reference" if profile.profile_id == _REFERENCE_PROFILE_ID else "named"
            )
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "ordered reference/source EquirectangularImage pair",
            "output": "RelativePoseResult or None when no hypothesis survives",
            "pose_convention": "source-from-reference",
            "translation": "unit-direction-only",
            "profile_mode": profile_mode,
            "profile_id": None if profile is None else profile.profile_id,
            "reference_profile": RelativePoseProfile.evidence_backed_v1().to_dict(),
            "frontend_mode": "reference" if self.frontend_config is None else "custom",
            "reference_frontend": "SphericalFeaturePipeline.for_relative_pose",
            "frontend_config": resolved.to_dict(),
            "options": self.options.to_dict(),
            "sampler": self.sampler.describe(),
            "quality_policy": self.quality_policy.to_dict(),
            "capture_policy": (
                None if self.capture_policy is None else self.capture_policy.to_dict()
            ),
            "capture_mode": None if self.capture_policy is None else self.capture_mode,
            "capture_contract": (
                "not assessed" if self.capture_policy is None else "pre-estimation"
            ),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        profile = self.resolved_profile
        frontend = (
            {"mode": "reference", "config": None}
            if self.frontend_config is None
            else {"mode": "custom", "config": self.frontend_config.to_dict()}
        )
        return application_config(
            self.application_id,
            {
                "frontend": frontend,
                "options": self.options,
                "sampler": _sampler_to_config(self.sampler),
                "quality_policy": self.quality_policy,
                "profile": (
                    None
                    if profile is None
                    else {
                        "profile_id": profile.profile_id,
                        "evidence_scope": profile.evidence_scope,
                    }
                ),
                "capture_policy": self.capture_policy,
                "capture_mode": self.capture_mode,
            },
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SphericalRelativePose":
        if config.get("application") != cls.application_id:
            raise ApplicationConfigurationError(
                "configuration does not describe SphericalRelativePose"
            )
        parameters = config.get("parameters")
        expected = {
            "frontend",
            "options",
            "sampler",
            "quality_policy",
            "profile",
            "capture_policy",
            "capture_mode",
        }
        prior = {
            "frontend",
            "options",
            "sampler",
            "quality_policy",
            "capture_policy",
            "capture_mode",
        }
        legacy = {"frontend", "options", "sampler", "quality_policy"}
        if not isinstance(parameters, dict) or frozenset(parameters) not in {
            frozenset(expected),
            frozenset(prior),
            frozenset(legacy),
        }:
            raise ApplicationConfigurationError(
                "SphericalRelativePose parameters are missing or unknown"
            )
        frontend = parameters["frontend"]
        if not isinstance(frontend, dict) or set(frontend) != {"mode", "config"}:
            raise ApplicationConfigurationError("frontend configuration is invalid")
        mode = frontend["mode"]
        if mode in {"reference", "gold-standard"}:
            if frontend["config"] is not None:
                raise ApplicationConfigurationError(
                    "reference frontend must not embed a custom configuration"
                )
            frontend_config = None
        elif mode == "custom":
            frontend_config = _feature_config_from_dict(frontend["config"])
        else:
            raise ApplicationConfigurationError(f"unknown frontend mode {mode!r}")
        try:
            options = RelativePoseOptions(**parameters["options"])
            quality_policy = RelativePoseAcceptancePolicy(
                **parameters["quality_policy"]
            )
        except (TypeError, ValueError) as exc:
            raise ApplicationConfigurationError(
                f"invalid relative-pose configuration: {exc}"
            ) from exc
        sampler = _sampler_from_config(parameters["sampler"])
        profile_value = parameters.get("profile")
        profile = None
        if profile_value is not None:
            if not isinstance(profile_value, dict) or set(profile_value) != {
                "profile_id",
                "evidence_scope",
            }:
                raise ApplicationConfigurationError(
                    "profile metadata must contain profile_id and evidence_scope"
                )
            try:
                profile = RelativePoseProfile(
                    profile_id=profile_value["profile_id"],
                    frontend_config=frontend_config,
                    options=options,
                    sampler=sampler,
                    quality_policy=quality_policy,
                    evidence_scope=profile_value["evidence_scope"],
                )
            except (TypeError, ValueError) as exc:
                raise ApplicationConfigurationError(
                    f"invalid relative-pose profile metadata: {exc}"
                ) from exc
        return cls(
            frontend_config=frontend_config,
            options=options,
            sampler=sampler,
            quality_policy=quality_policy,
            profile=profile,
            capture_policy=_capture_policy_from_config(
                parameters.get("capture_policy")
            ),
            capture_mode=parameters.get("capture_mode", "enforce"),
        )


__all__ = ["RelativePoseProfile", "SphericalRelativePose"]
