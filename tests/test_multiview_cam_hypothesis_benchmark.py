from __future__ import annotations

import math

import numpy as np
from PIL import Image
import pytest

from benchmarks.object_localization.run_multiview_cam_hypotheses import (
    STANFORD_VIEW_NAMES,
    _fit_component_region,
    _periodic_mask_components,
    _save_anchor_fusion_audit_panel,
    add_stanford_manual_region_metrics,
    align_reconstruction_to_reference_poses,
    build_independent_cam_region_provider,
    semantic_leave_one_view_out,
    spherical_score_percentile,
)
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.object_localization import (
    RegisteredBearingObservation,
    RegisteredBearingTrack,
    RegisteredPoseInconsistencyError,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
)
from panorai.reconstruction import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


def test_independent_cam_components_join_only_across_erp_seam():
    mask = np.zeros((5, 8), dtype=bool)
    mask[2, 0] = True
    mask[2, 7] = True
    mask[0, 3] = True
    mask[4, 3] = True

    components = _periodic_mask_components(mask)

    assert sorted(len(item) for item in components) == [1, 1, 2]
    seam = next(item for item in components if len(item) == 2)
    assert {tuple(value) for value in seam} == {(2, 0), (2, 7)}


def test_local_cam_component_is_fitted_as_compact_anchor_not_search_window():
    values = np.zeros((128, 256), dtype=np.float64)
    rows, columns = np.meshgrid(np.arange(58, 70), np.arange(118, 138), indexing="ij")
    cells = np.column_stack((rows.ravel(), columns.ravel()))
    values[cells[:, 0], cells[:, 1]] = 1.0

    region = _fit_component_region(
        view_id="analytic-view",
        cells_yx=cells,
        values=values,
        config=SphericalObjectHypothesisConfig(),
    )

    assert 8.0 <= region.spec.hfov_deg < 40.0
    assert 8.0 <= region.spec.vfov_deg < 30.0
    assert region.sparse_support_coverage == 1.0
    assert region.extent_source == ("independent-local-cam-component-threshold-0.60")


def test_local_cam_component_anchor_is_erp_seam_safe():
    values = np.zeros((128, 256), dtype=np.float64)
    left = np.column_stack((np.arange(60, 68), np.zeros(8, dtype=np.int64)))
    right = np.column_stack((np.arange(60, 68), np.full(8, 255, dtype=np.int64)))
    cells = np.concatenate((left, right), axis=0)
    values[cells[:, 0], cells[:, 1]] = 1.0

    region = _fit_component_region(
        view_id="analytic-seam-view",
        cells_yx=cells,
        values=values,
        config=SphericalObjectHypothesisConfig(),
    )

    assert abs(abs(region.spec.center_lon_deg) - 180.0) < 2.0
    assert region.spec.hfov_deg < 20.0
    assert region.sparse_support_coverage == 1.0


def test_independent_cam_provider_searches_registered_view_without_track_semantics():
    reconstruction, evidence, concepts = _loo_fixture()
    config = SphericalObjectHypothesisConfig(
        minimum_observation_score=0.25,
        minimum_track_score=0.25,
        minimum_tracks_per_hypothesis=2,
        spatial_cluster_radius=0.5,
    )
    generation = SphericalObjectHypothesisGenerator(config).generate(
        reconstruction,
        evidence[:2],
        concepts[:1],
    )
    hypothesis = generation.hypotheses[0]
    assert "c" not in hypothesis.view_ids

    provider, _ = build_independent_cam_region_provider(
        evidence,
        concepts[:1],
        config,
        threshold=0.75,
        minimum_component_cells=1,
    )

    regions = {item.view_id: item for item in provider(hypothesis)}
    assert "c" in regions
    assert regions["c"].track_feature_indices == ()
    assert regions["c"].sparse_support_coverage == 0.0
    assert regions["c"].semantic_supporting is False


def test_anchor_fusion_audit_panel_shows_all_four_stages(tmp_path):
    images = {
        "a": np.full((64, 128, 3), (70, 90, 110), dtype=np.uint8),
        "b": np.full((64, 128, 3), (95, 75, 60), dtype=np.uint8),
    }
    scores = np.zeros((1, 32, 64), dtype=np.float32)
    scores[0, 12:20, 26:38] = 0.9
    evidence = tuple(
        RegisteredSemanticEvidence(
            view_id,
            ("target",),
            scores,
            np.ones((32, 64), dtype=bool),
            "fixture/v1",
            f"cam:{view_id}",
        )
        for view_id in images
    )
    concepts = (SemanticConcept("target", "Target", (("target", 1.0),), "fixture/v1"),)

    def region(view_id: str, hfov: float, vfov: float):
        return {
            "view_id": view_id,
            "center_lat_deg": 0.0,
            "center_lon_deg": 0.0,
            "hfov_deg": hfov,
            "vfov_deg": vfov,
            "output_shape_hw": [128, 128],
            "sparse_support_coverage": 1.0,
            "extent_source": "fixture",
        }

    def anchor(anchor_id: str, view_id: str, hfov: float, action: str):
        return {
            "anchor": {
                "anchor_id": anchor_id,
                "view_id": view_id,
                "concept_id": "target",
                "vocabulary": "fixture/v1",
                "score": 0.9,
                "source_hypothesis_ids": ["hypothesis-1"],
                "proposal_family_ids": [f"family-{view_id}"],
                "provenance": {"scale_factor": 1.0, "aspect_ratio": 1.0},
                "region": region(view_id, hfov, 20.0),
            },
            "action": action,
            "representative_anchor_id": (anchor_id if action == "retained" else None),
            "solid_angle_iou": 1.0 if action == "retained" else None,
        }

    cluster = {
        "cluster_id": "cluster-1",
        "concept_id": "target",
        "vocabulary": "fixture/v1",
        "anchor_ids": ["a-base", "b-base"],
        "view_ids": ["a", "b"],
        "source_hypothesis_ids": ["hypothesis-1"],
        "proposal_family_ids": ["family-a", "family-b"],
        "score": 0.8,
        "state": "confirmed",
        "minimum_pair_affinity": 0.7,
        "consensus_regions": [region("a", 22.0, 21.0), region("b", 24.0, 20.0)],
    }
    evaluation = {
        "anchor_fusion": {
            "clusters": [cluster],
            "anchor_decisions": [
                anchor("a-base", "a", 20.0, "retained"),
                anchor("a-alt", "a", 30.0, "suppressed-capacity"),
                anchor("b-base", "b", 20.0, "retained"),
                anchor("b-alt", "b", 30.0, "suppressed-capacity"),
            ],
            "cluster_decisions": [
                {
                    "cluster": cluster,
                    "action": "retained",
                    "representative_cluster_id": "cluster-1",
                    "minimum_view_iou": 1.0,
                }
            ],
            "pair_affinities": [
                {
                    "anchor_id_a": "a-base",
                    "anchor_id_b": "b-base",
                    "factors": {"geometric": 1.0, "transfer": 0.7},
                    "affinity": 0.7,
                }
            ],
        }
    }
    output = tmp_path / "audit.jpg"

    _save_anchor_fusion_audit_panel(output, images, evidence, concepts, evaluation)

    with Image.open(output) as rendered:
        assert rendered.size == (2400, 708)


def _paint(scores: np.ndarray, channel: int, bearing: np.ndarray, value: float) -> None:
    shape = scores.shape[-2:]
    pixel = np.asarray(rays_to_erp_pixels(bearing, shape).pixels_xy)
    column = int(math.floor(float(pixel[0]) + 0.5)) % shape[1]
    row = min(max(int(math.floor(float(pixel[1]) + 0.5)), 0), shape[0] - 1)
    scores[channel, row, column] = value


def _loo_fixture():
    view_ids = ("a", "b", "c")
    centers = {
        "a": np.asarray((0.0, 0.0, 0.0)),
        "b": np.asarray((1.0, 0.0, 0.0)),
        "c": np.asarray((0.0, 1.0, 0.0)),
    }
    points = (
        np.asarray((0.0, 0.0, 5.0)),
        np.asarray((0.2, 0.0, 5.0)),
    )
    poses = tuple(SphericalCameraPose(key, np.eye(3), centers[key]) for key in view_ids)
    tracks = []
    for index, point in enumerate(points):
        observations = []
        for view_id in view_ids:
            bearing = point - centers[view_id]
            bearing /= np.linalg.norm(bearing)
            observations.append(SphericalTrackObservation(view_id, index, bearing, 0.0))
        tracks.append(SphericalTrack(f"track-{index}", point, tuple(observations)))
    reconstruction = SphericalReconstructionResult(
        success=True,
        failure_reasons=(),
        poses=poses,
        tracks=tuple(tracks),
        pairwise_edges=(),
        admitted_edge_indices=(),
        reference_panorama_id="a",
        diagnostics=SphericalReconstructionDiagnostics(),
        options=SphericalGlobalMapperOptions(),
        scale="metric-reference-registration",
        _points_xyz=np.stack(points),
    )
    evidence = []
    for view_id in view_ids:
        scores = np.zeros((2, 32, 64), dtype=np.float32)
        for track in tracks:
            observation = next(
                item for item in track.observations if item.panorama_id == view_id
            )
            _paint(scores, 0, observation.bearing_xyz, 0.8)
            _paint(scores, 1, observation.bearing_xyz, 0.2)
        evidence.append(
            RegisteredSemanticEvidence(
                view_id,
                ("target", "control"),
                scores,
                np.ones((32, 64), dtype=bool),
                "fixture/v1",
                f"cam:{view_id}",
            )
        )
    concepts = (
        SemanticConcept("target", "Target", (("target", 1.0),), "fixture/v1"),
        SemanticConcept("control", "Control", (("control", 1.0),), "fixture/v1"),
    )
    return reconstruction, tuple(evidence), concepts


def _point_free_tracks(reconstruction: SphericalReconstructionResult):
    return tuple(
        RegisteredBearingTrack(
            track.track_id,
            tuple(
                RegisteredBearingObservation(
                    observation.panorama_id,
                    observation.feature_index,
                    observation.bearing_xyz,
                )
                for observation in track.observations
            ),
        )
        for track in reconstruction.tracks
    )


def test_mapper_similarity_alignment_recovers_metric_pose_and_points():
    reference, _, _ = _loo_fixture()
    angle = math.radians(23.0)
    row_rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    scale = 2.4
    translation = np.asarray((3.0, -1.0, 0.7))

    def inverse(points: np.ndarray) -> np.ndarray:
        return ((np.asarray(points) - translation) @ row_rotation.T) / scale

    mapper_poses = tuple(
        SphericalCameraPose(
            pose.panorama_id,
            pose.R @ row_rotation.T,
            inverse(pose.center),
        )
        for pose in reference.poses
    )
    mapper_tracks = tuple(
        SphericalTrack(track.track_id, inverse(track.point_xyz), track.observations)
        for track in reference.tracks
    )
    mapper = SphericalReconstructionResult(
        success=True,
        failure_reasons=(),
        poses=mapper_poses,
        tracks=mapper_tracks,
        pairwise_edges=(),
        admitted_edge_indices=(),
        reference_panorama_id="a",
        diagnostics=reference.diagnostics,
        options=reference.options,
        scale="arbitrary",
        _points_xyz=np.stack([item.point_xyz for item in mapper_tracks]),
    )

    aligned, diagnostics = align_reconstruction_to_reference_poses(
        mapper, {item.panorama_id: item for item in reference.poses}
    )

    for pose in reference.poses:
        np.testing.assert_allclose(
            aligned.pose(pose.panorama_id).center, pose.center, atol=1e-12
        )
        np.testing.assert_allclose(aligned.pose(pose.panorama_id).R, pose.R, atol=1e-12)
    np.testing.assert_allclose(aligned.points_xyz, reference.points_xyz, atol=1e-12)
    assert diagnostics["camera_center_max_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["camera_rotation_max_error_deg"] == pytest.approx(0.0, abs=1e-6)


def test_leave_one_view_out_never_uses_held_out_semantic_evidence_for_generation():
    reconstruction, evidence, concepts = _loo_fixture()
    config = SphericalObjectHypothesisConfig(
        minimum_observation_score=0.25,
        minimum_track_score=0.25,
        minimum_tracks_per_hypothesis=2,
        spatial_cluster_radius=0.5,
    )

    result = semantic_leave_one_view_out(reconstruction, evidence, concepts, config)

    assert result["semantic_view_count"] == 3
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["held_out_view_id"] not in fold["generation_semantic_view_ids"]
        target = [item for item in fold["hypotheses"] if item["concept_id"] == "target"]
        assert len(target) == 1
        assert fold["held_out_view_id"] not in target[0]["training_view_ids"]
        assert target[0]["held_out_supported"] is True
        assert target[0]["held_out_score"] == pytest.approx(0.8)
        assert target[0]["held_out_concept_rank"] == 1
        assert target[0]["held_out_within_class_spherical_percentile"] == 1.0
    assert result["aggregate"]["held_out_top1_rate"] == 1.0
    assert result["aggregate"]["within_class_spherical_percentile"]["median"] == 1.0
    assert result["aggregate"]["repeat_with_all_view_within_cluster_radius_rate"] == 1.0


def test_leave_one_view_out_point_free_path_never_consumes_input_points():
    reconstruction, evidence, concepts = _loo_fixture()
    config = SphericalObjectHypothesisConfig(
        minimum_observation_score=0.25,
        minimum_track_score=0.25,
        minimum_tracks_per_hypothesis=2,
        spatial_cluster_radius=0.5,
    )

    result = semantic_leave_one_view_out(
        None,
        evidence,
        concepts,
        config,
        poses=tuple(reconstruction.poses),
        bearing_tracks=_point_free_tracks(reconstruction),
        reconstruction_scale="analytic",
        reference_view_id="a",
    )

    assert result["geometry_source"] == (
        "registered-poses-plus-point-free-bearing-tracks"
    )
    assert result["uses_input_point_xyz"] is False
    assert result["aggregate"]["held_out_top1_rate"] == 1.0


def test_leave_one_view_out_point_free_path_fails_closed_for_bad_poses():
    reconstruction, evidence, concepts = _loo_fixture()
    bad_tracks = []
    for track in _point_free_tracks(reconstruction):
        observations = list(track.observations)
        observations[1] = RegisteredBearingObservation(
            observations[1].view_id,
            observations[1].feature_index,
            np.asarray((1.0, 0.0, 0.0)),
        )
        bad_tracks.append(RegisteredBearingTrack(track.track_id, tuple(observations)))
    duplicated = tuple(
        RegisteredBearingTrack(f"{track.track_id}-{index}", track.observations)
        for index in range(2)
        for track in bad_tracks
    )

    with pytest.raises(RegisteredPoseInconsistencyError):
        semantic_leave_one_view_out(
            None,
            evidence,
            concepts,
            SphericalObjectHypothesisConfig(
                minimum_tracks_for_pose_consistency_gate=4,
                spatial_cluster_radius=0.5,
            ),
            poses=tuple(reconstruction.poses),
            bearing_tracks=duplicated,
            reference_view_id="a",
        )


def test_leave_one_view_out_requires_three_semantic_views():
    reconstruction, evidence, concepts = _loo_fixture()
    with pytest.raises(ValueError, match="at least three"):
        semantic_leave_one_view_out(
            reconstruction,
            evidence[:2],
            concepts,
            SphericalObjectHypothesisConfig(spatial_cluster_radius=0.5),
        )


def test_spherical_percentile_weights_rows_by_solid_angle():
    scores = np.zeros((1, 4, 8), dtype=np.float32)
    scores[0, 0] = 1.0
    evidence = RegisteredSemanticEvidence(
        "view",
        ("target",),
        scores,
        np.ones((4, 8), dtype=bool),
        "fixture/v1",
        "cam:view",
    )
    concept = SemanticConcept("target", "Target", (("target", 1.0),), "fixture/v1")

    percentile = spherical_score_percentile(evidence, concept, 0.0)

    polar_weight = math.cos(3.0 * math.pi / 8.0)
    equatorial_weight = math.cos(math.pi / 8.0)
    expected = (polar_weight + 2.0 * equatorial_weight) / (
        2.0 * polar_weight + 2.0 * equatorial_weight
    )
    assert percentile == pytest.approx(expected)


def test_stanford_manual_metric_scores_only_annotated_held_out_views():
    view_id = f"stanford2d3d::area_2::{STANFORD_VIEW_NAMES[0]}"
    inside = np.asarray(erp_pixels_to_rays(np.asarray((300.0, 700.0)), (1024, 2048)))
    outside = np.asarray(erp_pixels_to_rays(np.asarray((1500.0, 100.0)), (1024, 2048)))

    def region(bearing: np.ndarray) -> dict[str, object]:
        return {
            "center_lat_deg": math.degrees(math.asin(float(bearing[1]))),
            "center_lon_deg": math.degrees(
                math.atan2(float(bearing[0]), float(bearing[2]))
            ),
            "hfov_deg": 70.0,
            "vfov_deg": 70.0,
            "output_shape_hw": [1024, 1024],
        }

    evaluation = {
        "folds": [
            {
                "held_out_view_id": view_id,
                "hypotheses": [
                    {
                        "concept_id": "desk",
                        "state": "confirmed",
                        "projected_bearing_xyz": inside.tolist(),
                        "gnomonic_region": region(inside),
                    },
                    {
                        "concept_id": "desk",
                        "state": "proposed",
                        "projected_bearing_xyz": outside.tolist(),
                        "gnomonic_region": region(outside),
                    },
                ],
            },
            {
                "held_out_view_id": "unannotated",
                "hypotheses": [
                    {
                        "concept_id": "desk",
                        "state": "confirmed",
                        "projected_bearing_xyz": inside.tolist(),
                        "gnomonic_region": region(inside),
                    }
                ],
            },
        ]
    }

    add_stanford_manual_region_metrics(evaluation)

    metrics = evaluation["manual_region_evaluation"]
    assert metrics["projected_hypothesis_count"] == 2
    assert metrics["hit_rate"] == 0.5
    assert metrics["confirmed_hit_rate"] == 1.0
    assert 0.0 < metrics["window_annotation_recall_median"] <= 1.0
    assert 0.0 < metrics["window_spherical_iou_median"] < 1.0
    assert evaluation["folds"][1]["hypotheses"][0]["manual_region_hit"] is None
