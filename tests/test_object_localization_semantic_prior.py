from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.estimators import RelativePoseOptions, estimate_relative_pose
from panorai.features import MatchProvenance, SphericalFeatureMatches
from panorai.object_localization import (
    SemanticMatchPriorConfig,
    SemanticQuery,
    SemanticRegionObservation,
    build_semantic_match_prior,
)


def _matches(
    bearings_a: np.ndarray | None = None,
    bearings_b: np.ndarray | None = None,
    *,
    valid: np.ndarray | None = None,
) -> SphericalFeatureMatches:
    if bearings_a is None:
        bearings_a = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            )
        )
    if bearings_b is None:
        bearings_b = np.array(bearings_a, copy=True)
    count = len(bearings_a)
    if valid is None:
        valid = np.ones(count, dtype=bool)
    return SphericalFeatureMatches(
        panorama_id_a="view-a",
        panorama_id_b="view-b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.zeros(count),
        ratio_scores=np.zeros(count),
        mutual=np.ones(count, dtype=bool),
        valid=valid,
        matcher_name="analytic",
        matcher_config={},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface="test-semantic-prior/v1",
            source_checksums=("view-a", "view-b"),
            face_pairs=(("synthetic", "synthetic"),),
            face_pair_groups=((("synthetic", "synthetic"),),),
            deduplicated=True,
        ),
        keypoint_responses=np.ones((count, 2)),
        face_ids_a=np.full(count, "synthetic", dtype=object),
        face_ids_b=np.full(count, "synthetic", dtype=object),
    )


def _region(
    view_id: str,
    region_id: str,
    class_id: int,
    indices: tuple[int, ...],
    memberships: tuple[float, ...],
    *,
    score: float = 1.0,
) -> SemanticRegionObservation:
    return SemanticRegionObservation(
        region_id=region_id,
        view_id=view_id,
        class_id=class_id,
        class_name=f"class-{class_id}",
        semantic_score=score,
        feature_indices=np.asarray(indices),
        membership_weights=np.asarray(memberships),
    )


def test_prior_is_query_conditioned_and_keeps_global_fallback() -> None:
    matches = _matches(valid=np.asarray((True, True, True, True, False, True)))
    regions_a = (
        _region("view-a", "a-7", 7, (0, 1, 2), (1.0, 0.5, 0.25)),
        _region("view-a", "a-11", 11, (3,), (1.0,), score=0.8),
    )
    regions_b = (
        _region("view-b", "b-7", 7, (0, 1, 2), (1.0, 0.25, 1.0), score=0.81),
        _region("view-b", "b-11", 11, (3,), (1.0,), score=0.8),
    )

    prior = build_semantic_match_prior(
        SemanticQuery(text="find class seven", class_ids=(7,)),
        regions_a,
        regions_b,
        matches,
    )

    np.testing.assert_allclose(
        prior.semantic_support,
        np.asarray((0.9, 0.9 * math.sqrt(0.125), 0.45, 0.0, 0.0, 0.0)),
    )
    assert prior.used_class_ids == (7,)
    assert np.array_equal(prior.supported_match_indices, np.asarray((0, 1, 2)))
    assert prior.sampling_weights[4] == 0.0
    assert np.all(prior.sampling_weights[matches.valid] > 0.0)
    assert prior.sampling_weights[5] > 0.0
    assert prior.sampling_weights[0] > prior.sampling_weights[5]
    assert prior.max_to_min_valid_weight_ratio <= 4.0 + 1e-12
    assert np.isclose(prior.sampling_weights[matches.valid].mean(), 1.0)
    assert prior.describe()["prefilters_correspondences"] is False


def test_absent_query_support_is_exact_uniform_fallback() -> None:
    matches = _matches()
    regions_a = (_region("view-a", "a-7", 7, (0,), (1.0,)),)
    regions_b = (_region("view-b", "b-7", 7, (0,), (1.0,)),)

    prior = build_semantic_match_prior(
        SemanticQuery(text="find another class", class_ids=(11,)),
        regions_a,
        regions_b,
        matches,
    )

    np.testing.assert_array_equal(prior.sampling_weights, np.ones(len(matches)))
    np.testing.assert_array_equal(prior.semantic_support, np.zeros(len(matches)))
    assert prior.used_class_ids == ()
    assert prior.fallback_reason == "no-shared-query-support"
    assert prior.effective_sample_size == len(matches)


def test_duplicate_regions_and_duplicate_query_classes_cannot_inflate_support() -> None:
    matches = _matches()
    region_a = _region("view-a", "a-7", 7, (0, 1), (1.0, 0.5))
    region_b = _region("view-b", "b-7", 7, (0, 1), (0.5, 1.0))
    baseline = build_semantic_match_prior(
        SemanticQuery(text="class seven", class_ids=(7,)),
        (region_a,),
        (region_b,),
        matches,
    )
    duplicated = build_semantic_match_prior(
        SemanticQuery(text="class seven", class_ids=(7,)),
        (region_a, _region("view-a", "a-7-copy", 7, (0, 1), (1.0, 0.5))),
        (region_b, _region("view-b", "b-7-copy", 7, (0, 1), (0.5, 1.0))),
        matches,
    )

    np.testing.assert_array_equal(
        duplicated.semantic_support, baseline.semantic_support
    )
    np.testing.assert_array_equal(
        duplicated.sampling_weights, baseline.sampling_weights
    )


def test_uniform_only_configuration_is_exactly_uniform() -> None:
    matches = _matches()
    prior = build_semantic_match_prior(
        SemanticQuery(text="class seven", class_ids=(7,)),
        (_region("view-a", "a", 7, (0,), (1.0,)),),
        (_region("view-b", "b", 7, (0,), (1.0,)),),
        matches,
        SemanticMatchPriorConfig(uniform_mix=1.0),
    )

    np.testing.assert_array_equal(prior.sampling_weights, np.ones(len(matches)))
    assert prior.fallback_reason == "uniform-only-configured"


def test_prior_rejects_cross_view_regions_and_invalid_configuration() -> None:
    matches = _matches()
    query = SemanticQuery(text="class seven", class_ids=(7,))
    wrong = _region("view-b", "wrong", 7, (0,), (1.0,))
    with pytest.raises(ValueError, match="view A"):
        build_semantic_match_prior(query, (wrong,), (), matches)
    with pytest.raises(ValueError, match="uniform_mix"):
        SemanticMatchPriorConfig(uniform_mix=0.0)
    with pytest.raises(ValueError, match="max_weight_ratio"):
        SemanticMatchPriorConfig(max_weight_ratio=0.5)


def _known_pose_bearings(count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261010)
    points = rng.normal(size=(count, 3))
    points[:, 2] = np.abs(points[:, 2]) + 3.0
    angle = 0.08
    rotation = np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    translation = np.asarray((0.8, 0.1, 0.2))
    points_b = points @ rotation.T + translation
    return (
        points / np.linalg.norm(points, axis=1, keepdims=True),
        points_b / np.linalg.norm(points_b, axis=1, keepdims=True),
    )


def test_prior_flows_into_pose_estimator_without_prefiltering_matches() -> None:
    bearings_a, bearings_b = _known_pose_bearings()
    matches = _matches(bearings_a, bearings_b)
    query = SemanticQuery(text="find object", class_ids=(7,))
    indices = tuple(range(10))
    region_a = _region("view-a", "a", 7, indices, (1.0,) * len(indices))
    region_b = _region("view-b", "b", 7, indices, (1.0,) * len(indices))
    prior = build_semantic_match_prior(query, (region_a,), (region_b,), matches)

    correspondences = prior.to_bearing_correspondences(matches)
    pose = estimate_relative_pose(
        correspondences,
        options=RelativePoseOptions(
            max_angular_error_deg=0.1,
            min_inlier_ratio=0.2,
            min_num_trials=8,
            max_num_trials=80,
            min_inliers=10,
            local_optimization_steps=2,
            minimal_solver_starts=16,
            random_seed=31,
        ),
    )

    assert pose is not None
    assert pose.num_inliers == len(matches)
    assert len(pose.inlier_mask) == len(matches)
    assert prior.describe()["prefilters_correspondences"] is False


def test_prior_cannot_be_attached_to_changed_match_validity() -> None:
    matches = _matches()
    prior = build_semantic_match_prior(
        SemanticQuery(text="class seven", class_ids=(7,)), (), (), matches
    )
    changed = _matches(valid=np.asarray((True, True, True, True, True, False)))

    with pytest.raises(ValueError, match="validity mask"):
        prior.to_bearing_correspondences(changed)
