"""Contract tests for the Experimental probabilistic two-view API."""

from __future__ import annotations

import inspect
import math

import numpy as np

from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator
from panorai.experimental.two_view_probability import (
    BaselineEstimate,
    FrozenPoseProbabilityModels,
    MatchEvidence,
    OptimizedSphericalFrontend,
    OverlapProxyModel,
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


def _matches(count: int = 48) -> SphericalFeatureMatches:
    rng = np.random.default_rng(20261009)
    rotation = _rotation_exp(np.asarray((0.08, -0.13, 0.04)))
    translation = np.asarray((0.8, 0.1, 0.25), dtype=np.float64)
    translation /= np.linalg.norm(translation)
    points_a = rng.uniform((-3.0, -2.0, 3.0), (3.0, 2.0, 12.0), size=(count, 3))
    points_b = points_a @ rotation.T + translation
    bearings_a = points_a / np.linalg.norm(points_a, axis=1, keepdims=True)
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    indices = np.arange(count)
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
        matcher_name="analytic-test",
        matcher_config={},
        backend_name="analytic-test",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-feature-matches/v1",
            source_checksums=("a", "b"),
            face_pairs=tuple(("a", "b") for _ in range(count)),
            face_pair_groups=tuple((("a", "b"),) for _ in range(count)),
            deduplicated=True,
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "a", dtype=object),
        face_ids_b=np.full(count, "b", dtype=object),
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
    pose_estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.15,
            min_num_trials=8,
            max_num_trials=80,
            min_inliers=10,
            local_optimization_steps=2,
            stability_trials=2,
            stability_ransac_trials=16,
            model_competition_trials=32,
            random_seed=17,
        )
    )
    estimator = ProbabilisticSphericalTwoViewEstimator(pose_estimator=pose_estimator)
    baseline = BaselineEstimate(mean_m=1.2, standard_deviation_m=0.05)
    result = estimator.estimate_matches(
        _matches(), baseline=baseline, keypoint_counts=(600, 550)
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


def test_baseline_validation_rejects_nonphysical_values() -> None:
    for value in (0.0, -1.0, math.nan, math.inf):
        try:
            BaselineEstimate(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"baseline {value!r} should fail")
