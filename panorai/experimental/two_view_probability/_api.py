"""High-level two-EQR relative-pose estimation with calibrated evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any

import numpy as np

from panorai.estimators import RelativePoseResult, SphericalRelativePoseEstimator
from panorai.features import SphericalFeatureMatches

from ._frontend import FrontendResult, FrontendTimings, OptimizedSphericalFrontend
from ._contract import (
    CALIBRATED_FRONTEND_ID,
    ProbabilityCalibrationContractError,
    calibrated_pose_estimator,
    pose_contract,
    require_calibrated_matches,
    require_calibrated_pose_estimator,
)
from ._models import (
    BaselineEstimate,
    CaptureAdvisory,
    FrozenPoseProbabilityModels,
    MatchEvidence,
    OverlapPosterior,
    OverlapProxyModel,
    evidence_from_matches,
    post_features_from_pose,
)


@dataclass(frozen=True, slots=True)
class ProbabilisticTwoViewResult:
    """Pose plus pre/post evidence; ``accepted`` is the final selective decision."""

    pose: RelativePoseResult | None
    translation_m: np.ndarray | None
    baseline: BaselineEstimate
    match_evidence: MatchEvidence
    overlap: OverlapPosterior
    capture_advisory: CaptureAdvisory
    p_precise_post: float | None
    accepted: bool
    decision_reason: str
    frontend_timings: FrontendTimings | None
    pose_seconds: float
    total_seconds: float
    provenance: dict[str, Any]
    interface: str = "panorai-experimental-probabilistic-two-view/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "pose": (
                None
                if self.pose is None
                else {
                    "rotation": self.pose.rotation.tolist(),
                    "translation_direction": self.pose.translation_direction.tolist(),
                    "translation_m": (
                        None
                        if self.translation_m is None
                        else self.translation_m.tolist()
                    ),
                    "num_inliers": self.pose.num_inliers,
                    "num_trials": self.pose.num_trials,
                    "quality_report": self.pose.quality_report.to_dict(),
                }
            ),
            "baseline": asdict(self.baseline),
            "match_evidence": asdict(self.match_evidence),
            "overlap": asdict(self.overlap),
            "capture_advisory": asdict(self.capture_advisory),
            "p_precise_post": self.p_precise_post,
            "accepted": self.accepted,
            "decision_reason": self.decision_reason,
            "frontend_timings": (
                None if self.frontend_timings is None else asdict(self.frontend_timings)
            ),
            "pose_seconds": self.pose_seconds,
            "total_seconds": self.total_seconds,
            "provenance": dict(self.provenance),
        }


class ProbabilisticSphericalTwoViewEstimator:
    """Run optimized matching, R,t, and two calibrated evidence models.

    The overlap model is advisory and never changes the pose.  Final acceptance
    requires both the estimator's public quality policy and the frozen post-pose
    precision probability.  Metric translation magnitude comes from the supplied
    baseline because it is not observable from two calibrated central views.
    Frontend or pose-configuration drift fails closed before probability scoring.
    """

    def __init__(
        self,
        *,
        frontend: OptimizedSphericalFrontend | None = None,
        pose_estimator: SphericalRelativePoseEstimator | None = None,
        overlap_model: OverlapProxyModel | None = None,
        probability_models: FrozenPoseProbabilityModels | None = None,
    ) -> None:
        calibrated_frontend = OptimizedSphericalFrontend()
        self.frontend = frontend or calibrated_frontend
        if (
            type(self.frontend) is not OptimizedSphericalFrontend
            or self.frontend.configuration != calibrated_frontend.configuration
            or self.frontend.calibration_id != CALIBRATED_FRONTEND_ID
        ):
            raise ProbabilityCalibrationContractError(
                "frontend is outside the frozen probability calibration; use "
                "OptimizedSphericalFrontend with its default configuration"
            )
        self.pose_estimator = pose_estimator or calibrated_pose_estimator()
        require_calibrated_pose_estimator(self.pose_estimator)
        self.overlap_model = overlap_model or OverlapProxyModel.load_default()
        self.probability_models = (
            probability_models or FrozenPoseProbabilityModels.load_default()
        )

    def estimate(
        self,
        panorama_a: np.ndarray,
        panorama_b: np.ndarray,
        *,
        baseline: BaselineEstimate,
        validity_a: np.ndarray,
        validity_b: np.ndarray,
        panorama_ids: tuple[str, str] = ("a", "b"),
    ) -> ProbabilisticTwoViewResult:
        """Run the complete operation from two equal-resolution EQR images."""

        started = perf_counter()
        frontend = self.frontend.extract_and_match(
            panorama_a,
            panorama_b,
            validity_a=validity_a,
            validity_b=validity_b,
            panorama_ids=panorama_ids,
        )
        return self._estimate_frontend(frontend, baseline=baseline, started=started)

    def estimate_matches(
        self,
        matches: SphericalFeatureMatches,
        *,
        baseline: BaselineEstimate,
        keypoint_counts: tuple[int, int],
        valid_fractions: tuple[float, float] = (1.0, 1.0),
    ) -> ProbabilisticTwoViewResult:
        """Reuse matches carrying the exact calibrated frontend provenance."""

        started = perf_counter()
        require_calibrated_matches(matches)
        evidence = evidence_from_matches(
            matches,
            keypoint_counts=keypoint_counts,
            valid_fractions=valid_fractions,
        )
        return self._estimate_matches(
            matches,
            evidence=evidence,
            baseline=baseline,
            frontend_timings=None,
            started=started,
        )

    def _estimate_frontend(
        self,
        frontend: FrontendResult,
        *,
        baseline: BaselineEstimate,
        started: float,
    ) -> ProbabilisticTwoViewResult:
        if frontend.configuration != self.frontend.configuration:
            raise ProbabilityCalibrationContractError(
                "frontend result configuration does not match the frozen "
                "probability calibration"
            )
        require_calibrated_matches(frontend.matches)
        evidence = evidence_from_matches(
            frontend.matches,
            keypoint_counts=frontend.keypoint_counts,
            valid_fractions=frontend.valid_fractions,
        )
        return self._estimate_matches(
            frontend.matches,
            evidence=evidence,
            baseline=baseline,
            frontend_timings=frontend.timings,
            started=started,
        )

    def _estimate_matches(
        self,
        matches: SphericalFeatureMatches,
        *,
        evidence: MatchEvidence,
        baseline: BaselineEstimate,
        frontend_timings: FrontendTimings | None,
        started: float,
    ) -> ProbabilisticTwoViewResult:
        overlap = self.overlap_model.score(evidence)
        advisory = self.probability_models.capture_advisory(overlap, baseline)
        pose_started = perf_counter()
        pose = self.pose_estimator.estimate(matches.to_bearing_correspondences())
        pose_seconds = perf_counter() - pose_started

        p_post: float | None = None
        if pose is None:
            accepted, reason = False, "pose-not-returned"
        else:
            p_post = self.probability_models.score_post(
                post_features_from_pose(pose, match_count=evidence.match_count)
            )
            if not pose.quality_report.accepted:
                accepted, reason = False, "public-quality-rejected"
            elif p_post < self.probability_models.post_threshold:
                accepted, reason = False, "post-probability-below-threshold"
            else:
                accepted, reason = True, "accepted"

        try:
            runtime_version = version("panorai")
        except PackageNotFoundError:
            runtime_version = "source-tree"
        translation = (
            None
            if pose is None
            else np.asarray(pose.translation_direction) * baseline.mean_m
        )
        return ProbabilisticTwoViewResult(
            pose=pose,
            translation_m=translation,
            baseline=baseline,
            match_evidence=evidence,
            overlap=overlap,
            capture_advisory=advisory,
            p_precise_post=p_post,
            accepted=accepted,
            decision_reason=reason,
            frontend_timings=frontend_timings,
            pose_seconds=pose_seconds,
            total_seconds=perf_counter() - started,
            provenance={
                "runtime_panorai_version": runtime_version,
                "probability_model_panorai_version": "3.5.0",
                "probability_model_source_commit": "03c5b36b28225b24d3909286bf53250d7b532aa3",
                "probability_bundle_sha256": self.probability_models.sha256,
                "overlap_model_sha256": self.overlap_model.sha256,
                "frontend_route": "native-detect-batch-2+tangent-48-rootsift",
                "frontend_calibration_id": CALIBRATED_FRONTEND_ID,
                "frontend_configuration": self.frontend.configuration,
                "pose_calibration_contract": pose_contract(self.pose_estimator),
                "registered_clouds_required_at_runtime": False,
            },
        )
