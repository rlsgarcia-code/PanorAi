"""Typed capture eligibility for the spherical relative-pose application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Real
from typing import Any

from panorai.pipeline import StageResult


_SCANNER_CAPTURE_V1: dict[str, Any] = {
    "minimum_registered_overlap": 0.50,
    "preferred_registered_overlap": 0.70,
    "metric_baseline_interval_m": None,
    "minimum_capture_rate_hz": None,
    "target_max_step_translation_m": None,
    "target_max_step_rotation_deg": None,
    "maximum_step_translation_m": None,
    "maximum_step_rotation_deg": None,
    "profile_id": "panorai-scanner-assisted-capture-v1-experimental",
    "evidence_scope": (
        "retrospective public indoor and separately governed industrial "
        "eligibility evidence; not a universal precision guarantee"
    ),
}

_HILTI_CAPTURE_V1: dict[str, Any] = {
    "minimum_registered_overlap": None,
    "preferred_registered_overlap": None,
    "metric_baseline_interval_m": None,
    "minimum_capture_rate_hz": 5.0,
    "target_max_step_translation_m": 0.4,
    "target_max_step_rotation_deg": 15.0,
    "maximum_step_translation_m": 0.6,
    "maximum_step_rotation_deg": 30.0,
    "profile_id": "panorai-hilti-fisheye-temporal-development-v1",
    "evidence_scope": (
        "single Hilti fisheye sequence development envelope; not a "
        "cross-domain capture law"
    ),
}


def _optional_finite(
    value: float | None,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_closed: bool = True,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_closed else result <= minimum
        if invalid:
            relation = ">=" if minimum_closed else ">"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class RelativePoseCaptureMetadata:
    """Operator-visible facts about one ordered two-panorama capture.

    These values are not inferred from RGB pixels. Registered-cloud overlap
    requires independently registered depth support; metric baseline and scene
    distance require an external metric source. Missing information remains
    ``None`` rather than being replaced by a numeric sentinel.
    """

    registered_cloud_overlap_min: float | None = None
    baseline_m: float | None = None
    representative_scene_distance_m: float | None = None
    predicted_parallax_deg: float | None = None
    capture_rate_hz: float | None = None
    step_translation_m: float | None = None
    step_rotation_deg: float | None = None

    def __post_init__(self) -> None:
        for name, minimum, maximum, minimum_closed in (
            ("registered_cloud_overlap_min", 0.0, 1.0, True),
            ("baseline_m", 0.0, None, False),
            ("representative_scene_distance_m", 0.0, None, False),
            ("predicted_parallax_deg", 0.0, 180.0, True),
            ("capture_rate_hz", 0.0, None, False),
            ("step_translation_m", 0.0, None, True),
            ("step_rotation_deg", 0.0, 180.0, True),
        ):
            object.__setattr__(
                self,
                name,
                _optional_finite(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum,
                    minimum_closed=minimum_closed,
                ),
            )

    @property
    def baseline_depth_ratio(self) -> float | None:
        """Return baseline / representative distance when both are known."""

        if self.baseline_m is None or self.representative_scene_distance_m is None:
            return None
        return self.baseline_m / self.representative_scene_distance_m

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "baseline_depth_ratio": self.baseline_depth_ratio}


@dataclass(frozen=True, slots=True)
class RelativePoseCaptureAssessment:
    """Auditable capture decision made before feature extraction."""

    decision: str
    profile_id: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    observed: RelativePoseCaptureMetadata
    preferred: bool
    evidence_scope: str

    def __post_init__(self) -> None:
        if self.decision not in {
            "eligible",
            "ineligible",
            "insufficient-metadata",
        }:
            raise ValueError(
                "decision must be eligible, ineligible, or insufficient-metadata"
            )

    @property
    def eligible(self) -> bool:
        return self.decision == "eligible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "eligible": self.eligible,
            "profile_id": self.profile_id,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "observed": self.observed.to_dict(),
            "preferred": self.preferred,
            "evidence_scope": self.evidence_scope,
        }


@dataclass(frozen=True, slots=True)
class RelativePoseCapturePolicy:
    """Evidence-backed eligibility limits, separate from pose acceptance.

    The default profile represents the scanner-assisted multidataset evidence:
    50% minimum bidirectional registered-cloud overlap and 70% preferred
    overlap. It is an eligibility rule, not a calibrated probability of pose
    correctness. The policy deliberately carries no universal metric-baseline
    interval because the measured Matterport interval did not transfer as a
    physical law.
    """

    minimum_registered_overlap: float | None = 0.50
    preferred_registered_overlap: float | None = 0.70
    metric_baseline_interval_m: tuple[float, float] | None = None
    minimum_capture_rate_hz: float | None = None
    target_max_step_translation_m: float | None = None
    target_max_step_rotation_deg: float | None = None
    maximum_step_translation_m: float | None = None
    maximum_step_rotation_deg: float | None = None
    profile_id: str = "panorai-scanner-assisted-capture-v1-experimental"
    evidence_scope: str = (
        "retrospective public indoor and separately governed industrial "
        "eligibility evidence; not a universal precision guarantee"
    )

    def __post_init__(self) -> None:
        for name, maximum in (
            ("minimum_registered_overlap", 1.0),
            ("preferred_registered_overlap", 1.0),
            ("minimum_capture_rate_hz", None),
            ("target_max_step_translation_m", None),
            ("target_max_step_rotation_deg", 180.0),
            ("maximum_step_translation_m", None),
            ("maximum_step_rotation_deg", 180.0),
        ):
            minimum_closed = name not in {"minimum_capture_rate_hz"}
            object.__setattr__(
                self,
                name,
                _optional_finite(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    maximum=maximum,
                    minimum_closed=minimum_closed,
                ),
            )
        if (
            self.minimum_registered_overlap is not None
            and self.preferred_registered_overlap is not None
            and self.preferred_registered_overlap < self.minimum_registered_overlap
        ):
            raise ValueError(
                "preferred_registered_overlap must not be below the minimum"
            )
        interval = self.metric_baseline_interval_m
        if interval is not None:
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("metric_baseline_interval_m must be a two-value tuple")
            lower = _optional_finite(
                interval[0],
                "metric_baseline_interval_m[0]",
                minimum=0.0,
                minimum_closed=False,
            )
            upper = _optional_finite(
                interval[1],
                "metric_baseline_interval_m[1]",
                minimum=0.0,
                minimum_closed=False,
            )
            assert lower is not None and upper is not None
            if lower > upper:
                raise ValueError("metric baseline interval must be increasing")
            object.__setattr__(self, "metric_baseline_interval_m", (lower, upper))
        for target, maximum in (
            (self.target_max_step_translation_m, self.maximum_step_translation_m),
            (self.target_max_step_rotation_deg, self.maximum_step_rotation_deg),
        ):
            if target is not None and maximum is not None and target > maximum:
                raise ValueError("target step limit must not exceed the maximum")
        for name in ("profile_id", "evidence_scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        reserved = {
            _SCANNER_CAPTURE_V1["profile_id"]: _SCANNER_CAPTURE_V1,
            _HILTI_CAPTURE_V1["profile_id"]: _HILTI_CAPTURE_V1,
        }.get(self.profile_id)
        if reserved is not None and asdict(self) != reserved:
            raise ValueError(
                f"reserved profile_id {self.profile_id!r} requires its canonical "
                "payload"
            )

    @classmethod
    def scanner_assisted_reference(cls) -> "RelativePoseCapturePolicy":
        """Return the multidataset registered-overlap eligibility profile."""

        parameters = dict(_SCANNER_CAPTURE_V1)
        declared = {item.name for item in fields(cls) if item.init}
        frozen = set(parameters)
        if declared != frozen:
            missing = sorted(declared - frozen)
            stale = sorted(frozen - declared)
            raise RuntimeError(
                "scanner-assisted capture v1 schema changed; "
                f"missing={missing}, stale={stale}. Define a reviewed profile "
                "revision instead of inheriting new defaults."
            )
        return cls(**parameters)

    @classmethod
    def hilti_fisheye_development(cls) -> "RelativePoseCapturePolicy":
        """Return the explicitly development-only Hilti temporal envelope."""

        parameters = dict(_HILTI_CAPTURE_V1)
        declared = {item.name for item in fields(cls) if item.init}
        if declared != set(parameters):
            raise RuntimeError(
                "Hilti capture development v1 schema changed; define a reviewed "
                "profile revision instead of inheriting new defaults"
            )
        return cls(**parameters)

    def assess(
        self, metadata: RelativePoseCaptureMetadata
    ) -> RelativePoseCaptureAssessment:
        """Classify capture eligibility without running the pose estimator."""

        if not isinstance(metadata, RelativePoseCaptureMetadata):
            raise TypeError("metadata must be RelativePoseCaptureMetadata")
        missing: list[str] = []
        violations: list[str] = []
        warnings: list[str] = []

        overlap = metadata.registered_cloud_overlap_min
        if self.minimum_registered_overlap is not None:
            if overlap is None:
                missing.append("registered-cloud-overlap-unavailable")
            elif overlap < self.minimum_registered_overlap:
                violations.append("registered-cloud-overlap-below-minimum")
        if (
            overlap is not None
            and self.preferred_registered_overlap is not None
            and overlap < self.preferred_registered_overlap
            and not violations
        ):
            warnings.append("registered-cloud-overlap-below-preferred")

        if self.metric_baseline_interval_m is not None:
            if metadata.baseline_m is None:
                missing.append("metric-baseline-unavailable")
            elif not (
                self.metric_baseline_interval_m[0]
                <= metadata.baseline_m
                <= self.metric_baseline_interval_m[1]
            ):
                violations.append("metric-baseline-outside-supported-interval")

        if self.minimum_capture_rate_hz is not None:
            if metadata.capture_rate_hz is None:
                missing.append("capture-rate-unavailable")
            elif metadata.capture_rate_hz < self.minimum_capture_rate_hz:
                violations.append("capture-rate-below-minimum")

        self._assess_step(
            value=metadata.step_translation_m,
            label="translation",
            target=self.target_max_step_translation_m,
            maximum=self.maximum_step_translation_m,
            missing=missing,
            violations=violations,
            warnings=warnings,
        )
        self._assess_step(
            value=metadata.step_rotation_deg,
            label="rotation",
            target=self.target_max_step_rotation_deg,
            maximum=self.maximum_step_rotation_deg,
            missing=missing,
            violations=violations,
            warnings=warnings,
        )

        if violations:
            decision = "ineligible"
            reasons = tuple(violations)
        elif missing:
            decision = "insufficient-metadata"
            reasons = tuple(missing)
        else:
            decision = "eligible"
            reasons = ()
        return RelativePoseCaptureAssessment(
            decision=decision,
            profile_id=self.profile_id,
            reasons=reasons,
            warnings=tuple(warnings),
            observed=metadata,
            preferred=decision == "eligible" and not warnings,
            evidence_scope=self.evidence_scope,
        )

    @staticmethod
    def _assess_step(
        *,
        value: float | None,
        label: str,
        target: float | None,
        maximum: float | None,
        missing: list[str],
        violations: list[str],
        warnings: list[str],
    ) -> None:
        if target is None and maximum is None:
            return
        if value is None:
            missing.append(f"step-{label}-unavailable")
            return
        if maximum is not None and value > maximum:
            violations.append(f"step-{label}-above-maximum")
        elif target is not None and value > target:
            warnings.append(f"step-{label}-above-target")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelativePoseCaptureInput:
    """A spherical image pair plus independently supplied capture metadata."""

    pair: Any
    metadata: RelativePoseCaptureMetadata

    interface = "panorai-relative-pose-capture-input/v1-experimental"

    def __post_init__(self) -> None:
        from panorai.features import SphericalImagePair

        if not isinstance(self.pair, SphericalImagePair):
            raise TypeError("pair must be a SphericalImagePair")
        if not isinstance(self.metadata, RelativePoseCaptureMetadata):
            raise TypeError("metadata must be RelativePoseCaptureMetadata")


class RelativePoseCaptureError(ValueError):
    """Capture metadata failed an enforced eligibility policy."""

    def __init__(self, assessment: RelativePoseCaptureAssessment) -> None:
        self.assessment = assessment
        reasons = ", ".join(assessment.reasons) or assessment.decision
        super().__init__(f"relative-pose capture is not eligible: {reasons}")


@dataclass(frozen=True, slots=True)
class AssessRelativePoseCapture:
    """Pipeline Stage that checks capture facts before feature extraction."""

    policy: RelativePoseCapturePolicy
    mode: str = "enforce"

    input_kind = "relative_pose_capture_input"
    output_kind = "spherical_image_pair"
    interface = "panorai-relative-pose-capture-stage/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RelativePoseCapturePolicy):
            raise TypeError("policy must be RelativePoseCapturePolicy")
        if self.mode not in {"enforce", "advisory"}:
            raise ValueError("mode must be 'enforce' or 'advisory'")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, RelativePoseCaptureInput):
            raise TypeError(
                "AssessRelativePoseCapture requires RelativePoseCaptureInput"
            )
        assessment = self.policy.assess(value.metadata)
        if self.mode == "enforce" and not assessment.eligible:
            raise RelativePoseCaptureError(assessment)
        return StageResult(
            value.pair,
            {
                "domain": "capture-eligibility",
                "mode": self.mode,
                "assessment": assessment.to_dict(),
            },
        )


__all__ = [
    "AssessRelativePoseCapture",
    "RelativePoseCaptureAssessment",
    "RelativePoseCaptureError",
    "RelativePoseCaptureInput",
    "RelativePoseCaptureMetadata",
    "RelativePoseCapturePolicy",
]
