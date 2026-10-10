from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from panorai.object_localization import (
    JOINT_MATCH_CLUSTERS_INTERFACE,
    JointMatchClusterConfig,
    ObjectLocalizationConfig,
    ObjectLocalizationPipeline,
    RegionAssociation,
    SemanticQuery,
    SemanticRegionObservation,
    cluster_region_association_matches,
)
from tests.test_object_localization import _region, _scene


def _broad_same_class_evidence():
    evidence, _ = _scene()
    all_indices = np.arange(len(evidence.features_a))
    broad_a = _region(
        evidence.view_id_a,
        "broad-chair-a",
        7,
        "chair",
        all_indices,
        evidence.features_a.bearings,
    )
    broad_b = _region(
        evidence.view_id_b,
        "broad-chair-b",
        7,
        "chair",
        all_indices,
        evidence.features_b.bearings,
    )
    return replace(
        evidence,
        query=SemanticQuery(text="find chairs", class_ids=(7,)),
        regions_a=(broad_a,),
        regions_b=(broad_b,),
    )


def _accepted_parent(evidence):
    result = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False)
    ).estimate(evidence)
    accepted = [item for item in result.associations if item.state == "accepted"]
    assert len(accepted) == 1
    return accepted[0]


def test_joint_match_clusters_split_a_broad_region_into_two_objects() -> None:
    evidence = _broad_same_class_evidence()
    parent = _accepted_parent(evidence)

    clustered = cluster_region_association_matches(
        parent,
        evidence,
        JointMatchClusterConfig(max_neighbor_angle_deg=5.0, min_matches=4),
    )

    assert clustered.interface == JOINT_MATCH_CLUSTERS_INTERFACE
    assert clustered.source_match_count == 24
    assert clustered.raw_cluster_count == 2
    assert clustered.retained_cluster_count == 2
    assert clustered.discarded_match_count == 0
    assert [len(item.match_indices) for item in clustered.clusters] == [12, 12]
    assert len({item.cluster_id for item in clustered.clusters}) == 2
    assert all(item.diameter_a_deg < 5.0 for item in clustered.clusters)
    assert all(item.diameter_b_deg < 5.0 for item in clustered.clusters)

    refined_evidence = replace(
        evidence,
        regions_a=clustered.regions_a,
        regions_b=clustered.regions_b,
    )
    first = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False)
    ).estimate(refined_evidence)
    second = ObjectLocalizationPipeline(
        ObjectLocalizationConfig(require_accepted_pose=False)
    ).estimate(refined_evidence)

    assert len(first.hypotheses) == 2
    assert [item.hypothesis_id for item in first.hypotheses] == [
        item.hypothesis_id for item in second.hypotheses
    ]
    assert all(item.location.state == "localized" for item in first.hypotheses)
    assert sorted(item.location.triangulated_count for item in first.hypotheses) == [
        12,
        12,
    ]


def test_cluster_ids_depend_on_feature_pairs_not_match_row_order() -> None:
    evidence = _broad_same_class_evidence()
    first = cluster_region_association_matches(
        _accepted_parent(evidence),
        evidence,
        JointMatchClusterConfig(max_neighbor_angle_deg=5.0),
    )
    order = np.arange(len(evidence.matches))[::-1]
    reordered_matches = replace(
        evidence.matches,
        feature_indices_a=evidence.matches.feature_indices_a[order],
        feature_indices_b=evidence.matches.feature_indices_b[order],
        bearings_a=evidence.matches.bearings_a[order],
        bearings_b=evidence.matches.bearings_b[order],
        descriptor_distances=evidence.matches.descriptor_distances[order],
        ratio_scores=evidence.matches.ratio_scores[order],
        mutual=evidence.matches.mutual[order],
        valid=evidence.matches.valid[order],
        keypoint_responses=evidence.matches.keypoint_responses[order],
        face_ids_a=evidence.matches.face_ids_a[order],
        face_ids_b=evidence.matches.face_ids_b[order],
    )
    reordered_pose = replace(
        evidence.pose,
        inlier_mask=evidence.pose.inlier_mask[order],
        residuals_rad=evidence.pose.residuals_rad[order],
    )
    reordered = replace(
        evidence,
        matches=reordered_matches,
        pose=reordered_pose,
    )
    second = cluster_region_association_matches(
        _accepted_parent(reordered),
        reordered,
        JointMatchClusterConfig(max_neighbor_angle_deg=5.0),
    )

    assert [item.cluster_id for item in first.clusters] == [
        item.cluster_id for item in second.clusters
    ]
    assert [item.region_a.region_id for item in first.clusters] == [
        item.region_a.region_id for item in second.clusters
    ]


def test_joint_bearing_distance_clusters_across_the_erp_seam() -> None:
    longitudes = np.radians(np.asarray((179.0, -179.0, 0.0)))
    bearings = np.column_stack((np.sin(longitudes), np.zeros(3), np.cos(longitudes)))
    parent_a = SemanticRegionObservation(
        region_id="seam-a",
        view_id="a",
        class_id=1,
        class_name="object",
        semantic_score=0.8,
        feature_indices=np.arange(3),
        membership_weights=np.ones(3),
        centroid_bearing=np.asarray((0.0, 0.0, -1.0)),
    )
    parent_b = replace(parent_a, region_id="seam-b", view_id="b")
    association = RegionAssociation(
        association_id="accepted-seam-pair",
        region_id_a="seam-a",
        region_id_b="seam-b",
        class_id=1,
        class_name="object",
        match_indices=np.arange(3),
        inlier_match_indices=np.arange(3),
        match_count=3,
        inlier_count=3,
        semantic_score=0.8,
        median_pose_residual_deg=0.0,
        ranking_score=0.9,
        state="accepted",
    )
    evidence = SimpleNamespace(
        regions_a=(parent_a,),
        regions_b=(parent_b,),
        view_id_a="a",
        view_id_b="b",
        features_a=SimpleNamespace(bearings=bearings),
        features_b=SimpleNamespace(bearings=bearings),
        matches=SimpleNamespace(
            bearings_a=bearings,
            bearings_b=bearings,
            feature_indices_a=np.arange(3),
            feature_indices_b=np.arange(3),
        ),
    )

    result = cluster_region_association_matches(
        association,
        evidence,
        JointMatchClusterConfig(max_neighbor_angle_deg=3.0, min_matches=2),
    )

    assert result.raw_cluster_count == 2
    assert result.retained_cluster_count == 1
    np.testing.assert_array_equal(result.clusters[0].match_indices, (0, 1))
    assert result.clusters[0].diameter_a_deg == pytest.approx(2.0)
    assert result.discarded_match_count == 1


def test_small_and_truncated_clusters_are_reported_without_promotion() -> None:
    evidence = _broad_same_class_evidence()
    parent = _accepted_parent(evidence)

    fragmented = cluster_region_association_matches(
        parent,
        evidence,
        JointMatchClusterConfig(max_neighbor_angle_deg=0.01, min_matches=2),
    )
    limited = cluster_region_association_matches(
        parent,
        evidence,
        JointMatchClusterConfig(
            max_neighbor_angle_deg=5.0,
            min_matches=2,
            max_clusters=1,
        ),
    )

    assert fragmented.retained_cluster_count == 0
    assert fragmented.discarded_small_cluster_count == 24
    assert fragmented.discarded_match_count == 24
    assert limited.retained_cluster_count == 1
    assert limited.truncated_cluster_count == 1
    assert limited.discarded_match_count == 12


def test_joint_match_clustering_fails_closed_on_invalid_use() -> None:
    with pytest.raises(ValueError, match="max_neighbor_angle_deg"):
        JointMatchClusterConfig(max_neighbor_angle_deg=0.0)
    with pytest.raises(ValueError, match="min_matches"):
        JointMatchClusterConfig(min_matches=0)
    with pytest.raises(ValueError, match="max_clusters"):
        JointMatchClusterConfig(max_clusters=0)

    evidence = _broad_same_class_evidence()
    rejected = replace(
        _accepted_parent(evidence),
        state="rejected",
        reasons=("test-rejection",),
    )
    with pytest.raises(ValueError, match="accepted"):
        cluster_region_association_matches(rejected, evidence)
