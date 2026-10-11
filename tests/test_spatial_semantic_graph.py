from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panorai.graph import (
    EvidenceMode,
    EvidenceSource,
    GraphArchiveError,
    GraphBuilderConfig,
    GraphQueryFilters,
    PairFeatureEvidence,
    PairwiseLocalizationInput,
    RegionAssociationInput,
    RegionCorrespondenceCandidate,
    RegionCorrespondenceEdge,
    RelativePoseEdge,
    SemanticRegionNode,
    SpatialSemanticGraphBuilder,
    SpatialSemanticGraphQuery,
    SphericalViewNode,
    associate_regions,
    load_graph_archive,
    localize_region_pair,
    save_graph_archive,
)


def _source(
    source_id: str,
    modality: str,
    *,
    frame: str | None = "world",
    role: str = "method_input",
) -> EvidenceSource:
    return EvidenceSource(
        source_id=source_id,
        modality=modality,
        mode=EvidenceMode.EXTERNAL,
        method_id="synthetic-v1",
        interface_version="test/v1",
        access_role=role,  # type: ignore[arg-type]
        frame=frame,
        units="m",
        split="test",
        correlation_group="pair-a-b",
    )


def _pair() -> tuple[
    tuple[SphericalViewNode, SphericalViewNode],
    PairFeatureEvidence,
    RelativePoseEdge,
    tuple[SemanticRegionNode, SemanticRegionNode],
]:
    points = np.asarray([[0.0, 0.0, 3.0], [0.0, 0.25, 3.2], [0.0, -0.2, 2.8]])
    bearings_a = points / np.linalg.norm(points, axis=1, keepdims=True)
    points_b = points + np.asarray([1.0, 0.0, 0.0])
    bearings_b = points_b / np.linalg.norm(points_b, axis=1, keepdims=True)
    feature_source = _source("features-a-b", "spherical-features")
    features = PairFeatureEvidence(
        evidence_id="features-a-b",
        view_id_a="a",
        view_id_b="b",
        feature_indices_a=np.arange(3),
        feature_indices_b=np.arange(3),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        valid=np.ones(3, dtype=bool),
        pose_inlier_mask=np.ones(3, dtype=bool),
        pose_residuals_rad=np.asarray([0.001, 0.0015, 0.001]),
        descriptor_type="float32",
        descriptor_metric="l2",
        source=feature_source,
    )
    pose = RelativePoseEdge(
        edge_id="pose-a-b",
        view_id_a="a",
        view_id_b="b",
        rotation_b_from_a=np.eye(3),
        translation_direction_b_from_a=np.asarray([1.0, 0.0, 0.0]),
        accepted=True,
        degenerate=False,
        source=_source("pose-source", "relative-pose"),
        feature_evidence_id=features.evidence_id,
        translation_scale=1.0,
        units="m",
    )
    embedding = np.asarray([0.8, 0.2, 0.0])
    region_a = SemanticRegionNode(
        region_id="chair-a",
        view_id="a",
        class_id=7,
        class_name="chair",
        vocabulary="test-classes",
        semantic_score=0.95,
        feature_indices=np.arange(3),
        membership_weights=np.ones(3),
        centroid_bearing=bearings_a.mean(axis=0),
        source=_source("semantic-a", "semantic-regions"),
        embedding=embedding,
        encoder_id="aligned-v1",
        text_image_aligned=True,
    )
    region_b = replace(
        region_a,
        region_id="chair-b",
        view_id="b",
        centroid_bearing=bearings_b.mean(axis=0),
        source=_source("semantic-b", "semantic-regions"),
    )
    views = (
        SphericalViewNode("a", "world", "test", "sha-a"),
        SphericalViewNode("b", "world", "test", "sha-b"),
    )
    return views, features, pose, (region_a, region_b)


def _complete_graph():
    views, features, pose, regions = _pair()
    association = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1],), features, pose)
    )
    assert len(association.edges) == 1
    localized = localize_region_pair(
        PairwiseLocalizationInput(
            association.candidates[0], regions[0], regions[1], features, pose
        )
    )
    builder = SpatialSemanticGraphBuilder(
        GraphBuilderConfig.conservative_v1("synthetic")
    )
    builder.add_many(
        (
            association.edges[0],
            regions[1],
            views[1],
            pose,
            association.candidates[0],
            features,
            views[0],
            regions[0],
            localized.posterior,
        )
    )
    return builder, builder.snapshot(), localized


def test_two_view_association_localization_and_builder_are_explicit() -> None:
    builder, graph, localized = _complete_graph()

    assert localized.posterior.mode == "metric"
    assert localized.posterior.support_kind == "sparse_voxel"
    assert graph.hypotheses[0].state == "confirmed"
    assert graph.hypotheses[0].independent_view_count == 2
    assert graph.describe()["interface"] == "panorai-spatial-semantic-graph/v1"
    assert builder.events


def test_pose_rejection_and_evaluation_only_evidence_fail_closed() -> None:
    views, features, pose, regions = _pair()
    rejected = replace(pose, accepted=False)
    result = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1],), features, rejected)
    )
    assert result.edges == ()
    assert result.candidates[0].reasons == ("relative-pose-not-accepted",)

    forbidden_source = replace(features.source, access_role="evaluation_only")
    forbidden = replace(features, source=forbidden_source)
    with pytest.raises(ValueError, match="evaluation_only"):
        associate_regions(
            RegionAssociationInput((regions[0],), (regions[1],), forbidden, pose)
        )


def test_association_is_symmetric_and_repeated_instances_remain_ambiguous() -> None:
    _, features, pose, regions = _pair()
    forward = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1],), features, pose)
    )
    reverse_features = replace(
        features,
        view_id_a="b",
        view_id_b="a",
        feature_indices_a=features.feature_indices_b,
        feature_indices_b=features.feature_indices_a,
        bearings_a=features.bearings_b,
        bearings_b=features.bearings_a,
    )
    reverse_pose = replace(
        pose,
        view_id_a="b",
        view_id_b="a",
        translation_direction_b_from_a=-pose.translation_direction_b_from_a,
    )
    reverse = associate_regions(
        RegionAssociationInput(
            (regions[1],), (regions[0],), reverse_features, reverse_pose
        )
    )
    assert reverse.candidates[0].candidate_id == forward.candidates[0].candidate_id
    assert reverse.candidates[0].ranking_score == forward.candidates[0].ranking_score

    duplicate = replace(regions[1], region_id="chair-b-duplicate")
    ambiguous = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1], duplicate), features, pose)
    )
    assert ambiguous.edges == ()
    assert sum(item.state == "ambiguous" for item in ambiguous.candidates) == 1


def test_missing_scale_and_missing_finite_support_are_not_metric() -> None:
    _, features, pose, regions = _pair()
    scale_free_pose = replace(pose, translation_scale=None, units=None)
    association = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1],), features, scale_free_pose)
    )
    scale_free = localize_region_pair(
        PairwiseLocalizationInput(
            association.candidates[0],
            regions[0],
            regions[1],
            features,
            scale_free_pose,
        )
    )
    assert scale_free.posterior.mode == "scale_free"
    assert scale_free.posterior.map_position_xyz is None

    sparse_candidate = replace(
        association.candidates[0],
        inlier_match_indices=np.asarray([0]),
    )
    angular = localize_region_pair(
        PairwiseLocalizationInput(
            sparse_candidate,
            regions[0],
            regions[1],
            features,
            scale_free_pose,
        )
    )
    assert angular.posterior.mode == "unbounded_angular"
    assert angular.posterior.bounding_box_xyz is None


def test_one_view_never_confirms_an_entity() -> None:
    views, _, _, regions = _pair()
    builder = SpatialSemanticGraphBuilder(
        GraphBuilderConfig.conservative_v1("one-view")
    )
    builder.add_view(views[0])
    builder.add_region(regions[0])
    hypothesis = builder.snapshot().hypotheses[0]
    assert hypothesis.state == "proposed"
    assert hypothesis.independent_view_count == 1


def test_vertical_slice_preserves_ambiguity_and_correlated_evidence() -> None:
    """Three views exercise repeated objects, missing depth, and contradiction."""

    views, features, pose, regions = _pair()
    association = associate_regions(
        RegionAssociationInput((regions[0],), (regions[1],), features, pose)
    )
    accepted = association.candidates[0]
    edge_ab = association.edges[0]

    view_c = SphericalViewNode("c", "world", "test", "sha-c")
    region_c = replace(
        regions[1],
        region_id="chair-c",
        view_id="c",
        source=_source("semantic-c", "semantic-regions"),
    )
    repeated_c = replace(
        region_c,
        region_id="chair-c-repeated",
        semantic_score=0.8,
        source=_source("semantic-c-repeat", "semantic-regions"),
    )
    candidate_bc = replace(
        accepted,
        candidate_id="candidate-b-c",
        region_id_a=regions[1].region_id,
        region_id_b=region_c.region_id,
        view_id_a="b",
        view_id_b="c",
    )
    edge_bc = replace(
        edge_ab,
        edge_id="edge-b-c",
        candidate_id=candidate_bc.candidate_id,
        region_id_a=regions[1].region_id,
        region_id_b=region_c.region_id,
        # Same group models correlated evidence and must not count twice.
        correlation_groups=("shared-three-view-track",),
    )
    edge_ab = replace(
        edge_ab,
        correlation_groups=("shared-three-view-track",),
    )
    contradiction = RegionCorrespondenceCandidate(
        candidate_id="candidate-a-c-negative",
        region_id_a=regions[0].region_id,
        region_id_b=region_c.region_id,
        view_id_a="a",
        view_id_b="c",
        class_id=regions[0].class_id,
        class_name=regions[0].class_name,
        match_indices=np.arange(3),
        inlier_match_indices=np.arange(3),
        semantic_score=0.95,
        geometric_score=0.95,
        ranking_score=0.95,
        median_pose_residual_rad=0.001,
        state="accepted",
    )
    negative_edge = RegionCorrespondenceEdge(
        edge_id="edge-a-c-negative",
        candidate_id=contradiction.candidate_id,
        region_id_a=regions[0].region_id,
        region_id_b=region_c.region_id,
        confidence=0.9,
        source_evidence_ids=(features.evidence_id, pose.edge_id),
        correlation_groups=("contradictory-track",),
        polarity="negative",
    )
    angular_candidate = replace(accepted, inlier_match_indices=np.asarray([0]))
    angular = localize_region_pair(
        PairwiseLocalizationInput(
            angular_candidate,
            regions[0],
            regions[1],
            features,
            replace(pose, translation_scale=None, units=None),
        )
    ).posterior

    builder = SpatialSemanticGraphBuilder(
        GraphBuilderConfig.conservative_v1("vertical-slice")
    )
    builder.add_many(
        (
            *views,
            view_c,
            *regions,
            region_c,
            repeated_c,
            accepted,
            candidate_bc,
            contradiction,
            edge_ab,
            edge_bc,
            negative_edge,
            angular,
        )
    )
    graph = builder.snapshot()
    three_view = next(item for item in graph.hypotheses if len(item.view_ids) == 3)
    repeated = next(
        item for item in graph.hypotheses if repeated_c.region_id in item.region_ids
    )

    assert three_view.state == "dormant"
    assert three_view.confidence <= edge_ab.confidence * (
        1.0 - negative_edge.confidence
    )
    assert repeated.state == "proposed"
    assert angular.mode == "unbounded_angular"


def test_merge_split_lineage_and_correlated_evidence_are_explicit() -> None:
    views, _, _, regions = _pair()
    builder = SpatialSemanticGraphBuilder(
        GraphBuilderConfig.conservative_v1("lifecycle")
    )
    builder.add_many((*views, *regions))
    initial = builder.snapshot()
    parent_ids = tuple(item.hypothesis_id for item in initial.hypotheses)
    merged_id = builder.merge_hypotheses(parent_ids)
    merged = builder.snapshot()
    merged_node = next(
        item for item in merged.hypotheses if item.hypothesis_id == merged_id
    )
    assert merged_node.lineage_parent_ids == tuple(sorted(parent_ids))
    assert all(
        next(item for item in merged.hypotheses if item.hypothesis_id == parent).state
        == "superseded"
        for parent in parent_ids
    )

    children = builder.split_hypothesis(
        merged_id, ((regions[0].region_id,), (regions[1].region_id,))
    )
    split = builder.snapshot()
    assert (
        next(item for item in split.hypotheses if item.hypothesis_id == merged_id).state
        == "superseded"
    )
    assert set(children).issubset({item.hypothesis_id for item in split.hypotheses})


def test_frame_and_split_mismatches_fail_at_snapshot() -> None:
    views, _, _, regions = _pair()
    builder = SpatialSemanticGraphBuilder(
        GraphBuilderConfig.conservative_v1("mismatch")
    )
    builder.add_view(views[0])
    builder.add_region(
        replace(
            regions[0],
            source=replace(regions[0].source, frame="other-frame"),
        )
    )
    with pytest.raises(ValueError, match="frame"):
        builder.snapshot()


def test_order_invariance_archive_round_trip_and_checksum(tmp_path: Path) -> None:
    builder, graph, _ = _complete_graph()
    archive = tmp_path / "archive"
    save_graph_archive(graph, archive, events=builder.events)
    loaded = load_graph_archive(archive)
    assert loaded.graph_id == graph.graph_id
    assert loaded.event_ids == graph.event_ids
    assert [item.hypothesis_id for item in loaded.hypotheses] == [
        item.hypothesis_id for item in graph.hypotheses
    ]

    sidecar = next((archive / "arrays").glob("*.npz"))
    sidecar.write_bytes(sidecar.read_bytes() + b"corrupt")
    with pytest.raises(GraphArchiveError, match="checksum"):
        load_graph_archive(archive)


def test_embedding_text_spatial_and_neighbor_queries() -> None:
    _, graph, _ = _complete_graph()
    query = SpatialSemanticGraphQuery(graph)
    result = query.query_embedding(np.asarray([1.0, 0.0, 0.0]), encoder_id="aligned-v1")
    assert result[0].score_components["embedding"] > 0.9
    assert result[0].evidence_ids

    class Encoder:
        encoder_id = "aligned-v1"
        text_image_aligned = True

        def encode_text(self, text: str) -> np.ndarray:
            return np.asarray([1.0, 0.0, 0.0])

    assert query.query_text("chair", encoder=Encoder())
    assert query.query_spatial(
        np.asarray([[-1.0, -1.0, 1.0], [1.0, 1.0, 5.0]]),
        frame="a",
        units="m",
        filters=GraphQueryFilters(hypothesis_states=("confirmed",)),
    )
    assert "chair-b" in query.neighbors("chair-a")

    class Unaligned(Encoder):
        text_image_aligned = False

    with pytest.raises(ValueError, match="text-image aligned"):
        query.query_text("chair", encoder=Unaligned())
