"""Fail-closed calibration contract for the probabilistic two-view API."""

from __future__ import annotations

from typing import Any

from panorai.estimators import (
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
)
from panorai.features import FeatureMatcherConfig, SphericalFeatureMatches

CALIBRATED_FRONTEND_ID = (
    "panorai-two-view-probability-frontend/3.5.0-native-dog-rootsift-v1"
)
CALIBRATED_FRONTEND_INTERFACE = "panorai-optimized-public-pair-features/v1"
CALIBRATED_POSE_ID = "panorai-two-view-probability-pose/3.5.0-v1"


class ProbabilityCalibrationContractError(ValueError):
    """Raised when inputs fall outside the frozen probability calibration."""


def calibrated_pose_estimator() -> SphericalRelativePoseEstimator:
    """Construct the exact R,t estimator used by the frozen post-pose model."""

    return SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=1.0,
            max_num_trials=1000,
            stability_trials=6,
            model_competition_trials=128,
            random_seed=7,
            hypothesis_ranking="msac-first",
            nonminimal_refit_max_steps=100,
        )
    )


def calibrated_matcher_configuration() -> dict[str, Any]:
    """Return the exact matcher configuration used by the overlap proxy."""

    return FeatureMatcherConfig(
        method="flann",
        ratio_test=0.72,
        cross_check=False,
        deduplicate_matches=True,
        angular_dedup_threshold_deg=0.15,
    ).to_dict()


def pose_contract(estimator: SphericalRelativePoseEstimator) -> dict[str, Any]:
    """Serialize every pose component that affects the calibrated post model."""

    return {
        "calibration_id": CALIBRATED_POSE_ID,
        "options": estimator.options.to_dict(),
        "sampler": estimator.sampler.describe(),
        "acceptance_policy": estimator.quality_policy.to_dict(),
    }


def require_calibrated_pose_estimator(
    estimator: SphericalRelativePoseEstimator,
) -> None:
    """Reject a pose estimator whose evidence distribution was not calibrated."""

    if type(estimator) is not SphericalRelativePoseEstimator:
        raise ProbabilityCalibrationContractError(
            "pose_estimator must be the exact SphericalRelativePoseEstimator "
            "implementation with the "
            "frozen PanorAi 3.5.0 probability calibration"
        )
    expected = pose_contract(calibrated_pose_estimator())
    actual = pose_contract(estimator)
    mismatches = tuple(
        name for name in ("options", "sampler", "acceptance_policy")
        if actual[name] != expected[name]
    )
    if mismatches:
        names = ", ".join(mismatches)
        raise ProbabilityCalibrationContractError(
            "pose_estimator is outside the frozen probability calibration; "
            f"incompatible component(s): {names}. Use the default estimator for "
            "calibrated probabilities or SphericalRelativePoseEstimator directly "
            "for custom geometry."
        )


def require_calibrated_matches(matches: SphericalFeatureMatches) -> None:
    """Reject matches that cannot prove the calibrated frontend configuration."""

    reasons: list[str] = []
    if matches.provenance.calibration_id != CALIBRATED_FRONTEND_ID:
        reasons.append("calibration_id")
    if matches.provenance.interface != CALIBRATED_FRONTEND_INTERFACE:
        reasons.append("provenance.interface")
    if matches.interface != CALIBRATED_FRONTEND_INTERFACE:
        reasons.append("matches.interface")
    if matches.matcher_name != "flann":
        reasons.append("matcher_name")
    if matches.matcher_config != calibrated_matcher_configuration():
        reasons.append("matcher_config")
    if matches.backend_name != "opencv":
        reasons.append("backend_name")
    if not matches.provenance.deduplicated:
        reasons.append("deduplicated")
    if reasons:
        details = ", ".join(reasons)
        raise ProbabilityCalibrationContractError(
            "matches are outside the frozen probability calibration; "
            f"incompatible or missing provenance: {details}. Obtain matches from "
            "OptimizedSphericalFrontend or use SphericalRelativePoseEstimator "
            "directly without calibrated probabilities."
        )
