"""Contract tests for the Experimental probabilistic two-view API."""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
    UniformFivePointSampler,
)
from panorai.experimental.two_view_probability import (
    BaselineEstimate,
    FrozenPoseProbabilityModels,
    MatchEvidence,
    OptimizedSphericalFrontend,
    OverlapProxyModel,
    ProbabilityCalibrationContractError,
    ProbabilisticSphericalTwoViewEstimator,
)
from panorai.features import MatchProvenance, SphericalFeatureMatches


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    x, y, z = vector
    skew = np.asarray(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    return (
        np.eye(3)
        + math.sin(angle) / angle * skew
        + (1 - math.cos(angle)) / angle**2 * (skew @ skew)
    )


def _matches(
    count: int = 48, *, calibrated_frontend: bool = False
) -> SphericalFeatureMatches:
    rng = np.random.default_rng(20261009)
    rotation = _rotation_exp(np.asarray((0.08, -0.13, 0.04)))
    translation = np.asarray((0.8, 0.1, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)
    points_a = rng.uniform((-3.0, -2.0, 3.0), (3.0, 2.0, 12.0), size=(count, 3))
    points_b = points_a @ rotation.T + translation
    bearings_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    indices = np.arange(count)
    frontend = OptimizedSphericalFrontend()
    interface = (
        "panorai-optimized-public-pair-features/v1"
        if calibrated_frontend
        else "panorai-spherical-feature-matches/v1"
    )
    return SphericalFeatureMatches(
        panorama_id_a="a",
        panorama_id_b="b",
        feature_indices_a=indices,
        feature_indices_b=indices,
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.linspace(0.1, 0.4, count),
        ratio_scores=np.linspace(0.2, 0.6, count),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool),
        matcher_name="flann" if calibrated_frontend else "bf",
        matcher_config=(
            frontend.configuration["matcher"] if calibrated_frontend else {}
        ),
        backend_name="opencv" if calibrated_frontend else "analytic-test",
        backend_version="1",
        provenance=MatchProvenance(
            interface=interface,
            source_checksums=("a", "b"),
            face_pairs=tuple(("a", "b") for _ in range(count)),
            face_pair_groups=tuple((("a", "b"),) for _ in range(count)),
            deduplicated=True,
            calibration_id=(
                frontend.calibration_id if calibrated_frontend else None
            ),
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "a", dtype=object),
        face_ids_b=np.full(count, "b", dtype=object),
        interface=interface,
    )


def test_runtime_contract_needs_two_images_baseline_and_masks_not_clouds() -> None:
    parameters = inspect.signature(
        ProbabilisticSphericalTwoViewEstimator.estimate
    ).parameters
    assert set(parameters) == {
        "self",
        "panorama_a",
        "panorama_b",
        "baseline",
        "validity_a",
        "validity_b",
        "panorama_ids",
    }
    assert (
        OptimizedSphericalFrontend().configuration["detector"]["convolution_backend"]
        == "native"
    )
    assert OptimizedSphericalFrontend().configuration["batch_size"] == 2
    assert OptimizedSphericalFrontend().configuration["patch_workers"] == 4


def test_overlap_proxy_is_normalized_and_reports_heldout_metrics() -> None:
    model = OverlapProxyModel.load_default()
    result = model.score(
        MatchEvidence(
            keypoint_count_min=900,
            match_count=180,
            descriptor_distance_median=0.25,
            descriptor_distance_p90=0.4,
            ratio_score_median=0.45,
            valid_fraction_min=1.0,
        )
    )
    assert math.isclose(sum(result.probabilities), 1.0, abs_tol=1e-12)
    assert 0.0 <= result.expected_overlap <= 1.0
    assert model.metrics["heldout_evaluation"]["count"] == 435


def test_packaged_overlap_provenance_does_not_publish_dataset_names() -> None:
    provenance = OverlapProxyModel.load_default().data_provenance
    assert provenance["dataset_count"] == 3
    assert "datasets" not in provenance


def test_explicit_overlap_capture_model_remains_available_offline() -> None:
    result = FrozenPoseProbabilityModels.load_default().score_capture_explicit(
        overlap=0.65,
        baseline_m=0.80,
    )
    assert result.inside_supported_envelope
    assert math.isclose(
        result.p_usable,
        result.p_accept * result.p_precise_given_accept,
        rel_tol=1e-12,
    )


def test_real_public_estimator_is_composed_without_mutating_pose() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    baseline = BaselineEstimate(mean_m=1.2, standard_deviation_m=0.05)
    result = estimator.estimate_matches(
        _matches(calibrated_frontend=True),
        baseline=baseline,
        keypoint_counts=(600, 550),
    )

    assert result.pose is not None
    assert math.isclose(
        np.linalg.norm(result.pose.translation_direction), 1.0, rel_tol=1e-7
    )
    assert math.isclose(np.linalg.norm(result.translation_m), 1.2, rel_tol=1e-7)
    assert 0.0 <= result.capture_advisory.p_usable <= 1.0
    assert result.p_precise_post is not None
    assert result.accepted == (
        result.pose.quality_report.accepted and result.p_precise_post >= 0.9
    )
    assert result.provenance["frontend_calibration_id"] == (
        OptimizedSphericalFrontend().calibration_id
    )
    assert result.provenance["pose_calibration_contract"]["options"] == (
        estimator.pose_estimator.options.to_dict()
    )


def test_uncalibrated_public_matches_fail_closed_before_probability_scoring() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    with pytest.raises(ProbabilityCalibrationContractError, match="calibration_id"):
        estimator.estimate_matches(
            _matches(),
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_tampered_calibrated_matcher_configuration_fails_closed() -> None:
    matches = _matches(calibrated_frontend=True)
    matches.matcher_config = {**matches.matcher_config, "ratio_test": 0.75}
    estimator = ProbabilisticSphericalTwoViewEstimator()
    with pytest.raises(ProbabilityCalibrationContractError, match="matcher_config"):
        estimator.estimate_matches(
            matches,
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_custom_pose_configuration_fails_closed_at_construction() -> None:
    pose_estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(max_angular_error_deg=0.15, random_seed=17)
    )
    with pytest.raises(ProbabilityCalibrationContractError, match="options"):
        ProbabilisticSphericalTwoViewEstimator(pose_estimator=pose_estimator)


def test_custom_pose_sampler_and_acceptance_policy_fail_closed() -> None:
    calibrated = ProbabilisticSphericalTwoViewEstimator().pose_estimator
    wrong_sampler = SphericalRelativePoseEstimator(
        calibrated.options,
        sampler=UniformFivePointSampler(),
        quality_policy=calibrated.quality_policy,
    )
    with pytest.raises(ProbabilityCalibrationContractError, match="sampler"):
        ProbabilisticSphericalTwoViewEstimator(pose_estimator=wrong_sampler)

    wrong_policy = SphericalRelativePoseEstimator(
        calibrated.options,
        sampler=calibrated.sampler,
        quality_policy=RelativePoseAcceptancePolicy(min_inliers=21),
    )
    with pytest.raises(
        ProbabilityCalibrationContractError, match="acceptance_policy"
    ):
        ProbabilisticSphericalTwoViewEstimator(pose_estimator=wrong_policy)


def test_pose_configuration_mutation_fails_before_scoring() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    estimator.pose_estimator.options = RelativePoseOptions(
        max_angular_error_deg=0.15,
        random_seed=17,
    )
    with pytest.raises(ProbabilityCalibrationContractError, match="options"):
        estimator.estimate_matches(
            _matches(calibrated_frontend=True),
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_custom_frontend_configuration_fails_closed() -> None:
    class DriftedFrontend(OptimizedSphericalFrontend):
        @property
        def configuration(self) -> dict[str, object]:
            configuration = super().configuration
            configuration["patch_workers"] = 1
            return configuration

    with pytest.raises(ProbabilityCalibrationContractError, match="frontend"):
        ProbabilisticSphericalTwoViewEstimator(frontend=DriftedFrontend())


def test_frontend_mutation_fails_before_scoring() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    estimator.frontend._patch_provider.max_workers = 1
    with pytest.raises(ProbabilityCalibrationContractError, match="frontend"):
        estimator.estimate_matches(
            _matches(calibrated_frontend=True),
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_mutated_frontend_cannot_stamp_calibrated_matches() -> None:
    frontend = OptimizedSphericalFrontend()
    frontend._patch_provider.max_workers = 1
    panorama = np.zeros((8, 16), dtype=np.uint8)
    validity = np.ones((8, 16), dtype=bool)
    with pytest.raises(ProbabilityCalibrationContractError, match="frontend"):
        frontend.extract_and_match(
            panorama,
            panorama,
            validity_a=validity,
            validity_b=validity,
        )


def test_custom_overlap_model_fails_closed_at_construction() -> None:
    calibrated = OverlapProxyModel.load_default()
    drifted_artifact = json.loads(json.dumps(calibrated._artifact))
    drifted_artifact["calibration_temperature"] *= 2.0
    drifted = OverlapProxyModel(drifted_artifact, sha256=calibrated.sha256)
    with pytest.raises(ProbabilityCalibrationContractError, match="overlap_model"):
        ProbabilisticSphericalTwoViewEstimator(overlap_model=drifted)


def test_overlap_model_mutation_fails_before_scoring() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    estimator.overlap_model._artifact["calibration_temperature"] *= 2.0
    with pytest.raises(ProbabilityCalibrationContractError, match="overlap_model"):
        estimator.estimate_matches(
            _matches(calibrated_frontend=True),
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_custom_probability_bundle_fails_closed_at_construction() -> None:
    calibrated = FrozenPoseProbabilityModels.load_default()
    drifted_bundle = json.loads(json.dumps(calibrated.bundle))
    drifted_bundle["operating_thresholds"]["p_precise_post_min"] = 0.0
    drifted = FrozenPoseProbabilityModels(
        drifted_bundle,
        sha256=calibrated.sha256,
    )
    with pytest.raises(
        ProbabilityCalibrationContractError, match="probability_models"
    ):
        ProbabilisticSphericalTwoViewEstimator(probability_models=drifted)


def test_separately_loaded_canonical_probability_bundle_is_supported() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator(
        probability_models=FrozenPoseProbabilityModels.load_default()
    )
    assert estimator.probability_models.sha256 == (
        FrozenPoseProbabilityModels.load_default().sha256
    )


def test_probability_bundle_mutation_fails_before_scoring() -> None:
    estimator = ProbabilisticSphericalTwoViewEstimator()
    estimator.probability_models.bundle["operating_thresholds"][
        "p_precise_post_min"
    ] = 0.0
    with pytest.raises(
        ProbabilityCalibrationContractError, match="probability_models"
    ):
        estimator.estimate_matches(
            _matches(calibrated_frontend=True),
            baseline=BaselineEstimate(0.8),
            keypoint_counts=(600, 550),
        )


def test_baseline_validation_rejects_nonphysical_values() -> None:
    for value in (0.0, -1.0, math.nan, math.inf):
        try:
            BaselineEstimate(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"baseline {value!r} should fail")
