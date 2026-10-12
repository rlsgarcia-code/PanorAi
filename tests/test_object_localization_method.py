from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.geometry import GnomonicSpec, erp_pixels_to_rays, rays_to_erp_pixels
from panorai.object_localization import (
    HypothesisGnomonicRegion,
    HypothesisPairCues,
    RegisteredBearingObservation,
    RegisteredBearingTrack,
    RegisteredRadialDepth,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
    SphericalObjectHypothesisMethod,
    SphericalObjectHypothesisMethodConfig,
)
from panorai.reconstruction import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


SHAPE_HW = (32, 64)


def _bearing(point: np.ndarray, center: np.ndarray) -> np.ndarray:
    value = point - center
    return value / np.linalg.norm(value)


def _fixture(
    points: tuple[np.ndarray, ...] | None = None,
):
    centers = {
        "a": np.asarray((0.0, 0.0, 0.0)),
        "b": np.asarray((1.0, 0.0, 0.0)),
        "c": np.asarray((0.0, 1.0, 0.0)),
    }
    if points is None:
        points = (
            np.asarray((0.0, 0.0, 5.0)),
            np.asarray((0.2, 0.0, 5.0)),
        )
    poses = tuple(
        SphericalCameraPose(view_id, np.eye(3), center)
        for view_id, center in centers.items()
    )
    tracks = []
    for point_index, point in enumerate(points):
        observations = tuple(
            SphericalTrackObservation(
                pose.panorama_id,
                point_index,
                _bearing(point, pose.center),
                math.radians(0.05),
            )
            for pose in poses
        )
        tracks.append(SphericalTrack(f"track-{point_index}", point, observations))
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
        scale="metric-fixture",
        _points_xyz=np.stack(points),
    )
    evidence = []
    for pose in poses:
        scores = np.zeros((1, *SHAPE_HW), dtype=np.float32)
        for track in tracks:
            observation = next(
                item
                for item in track.observations
                if item.panorama_id == pose.panorama_id
            )
            pixel = np.asarray(
                rays_to_erp_pixels(observation.bearing_xyz, SHAPE_HW).pixels_xy
            )
            column = int(math.floor(float(pixel[0]) + 0.5)) % SHAPE_HW[1]
            row = int(math.floor(float(pixel[1]) + 0.5))
            scores[0, row, column] = 0.8
        evidence.append(
            RegisteredSemanticEvidence(
                pose.panorama_id,
                ("target",),
                scores,
                np.ones(SHAPE_HW, dtype=bool),
                "fixture/v1",
                f"cam:{pose.panorama_id}",
            )
        )
    concepts = (
        SemanticConcept(
            "target",
            "Target",
            (("target", 1.0),),
            "fixture/v1",
        ),
    )
    return reconstruction, tuple(evidence), concepts


def _plane_depth(poses: tuple[SphericalCameraPose, ...]):
    height, width = SHAPE_HW
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    rays = np.asarray(
        erp_pixels_to_rays(np.column_stack((xx.ravel(), yy.ravel())), SHAPE_HW)
    ).reshape(height, width, 3)
    result = []
    for pose in poses:
        world_rays = rays @ pose.R
        support = world_rays[..., 2] > 1.0e-6
        values = np.full(SHAPE_HW, np.nan)
        values[support] = (6.0 - pose.center[2]) / world_rays[..., 2][support]
        result.append(
            RegisteredRadialDepth(
                pose.panorama_id,
                values,
                support,
                "m",
                f"plane-depth:{pose.panorama_id}",
            )
        )
    return tuple(result)


def _independent_region_provider(
    evidence: tuple[RegisteredSemanticEvidence, ...],
):
    evidence_by_view = {item.view_id: item for item in evidence}

    def provider(hypothesis):
        regions = []
        for view_id in hypothesis.view_ids:
            field = evidence_by_view[view_id].scores[0]
            rows, columns = np.nonzero(field > 0.0)
            rays = np.asarray(
                erp_pixels_to_rays(
                    np.column_stack(
                        (columns.astype(np.float64), rows.astype(np.float64))
                    ),
                    SHAPE_HW,
                )
            )
            vector = np.sum(rays * field[rows, columns][:, None], axis=0)
            center = vector / np.linalg.norm(vector)
            latitude = math.degrees(math.asin(float(center[1])))
            longitude = math.degrees(math.atan2(float(center[0]), float(center[2])))
            regions.append(
                HypothesisGnomonicRegion(
                    view_id=view_id,
                    spec=GnomonicSpec(
                        center_lat_deg=latitude,
                        center_lon_deg=longitude,
                        hfov_deg=70.0,
                        vfov_deg=70.0,
                        output_shape_hw=(128, 128),
                    ),
                    center_bearing_xyz=center,
                    angular_radius_deg=0.0,
                    support_bearings_xyz=center[None, :],
                    track_feature_indices=(),
                    sparse_support_coverage=1.0,
                    semantic_evidence_available=True,
                    semantic_supporting=True,
                    extent_source="independent-per-view-semantic-centroid",
                )
            )
        return tuple(regions)

    return provider


def _method() -> SphericalObjectHypothesisMethod:
    return SphericalObjectHypothesisMethod(
        SphericalObjectHypothesisMethodConfig(
            generator=SphericalObjectHypothesisConfig(
                minimum_observation_score=0.25,
                minimum_track_score=0.25,
                minimum_tracks_per_hypothesis=2,
                spatial_cluster_radius=0.5,
            ),
            anchor_scale_factors=(1.0,),
            anchor_aspect_ratios=(1.0,),
        )
    )


def test_method_config_exposes_frozen_detector_style_anchor_family():
    config = SphericalObjectHypothesisMethodConfig()

    assert config.anchor_scale_factors == (0.75, 1.0, 1.33)
    assert config.anchor_aspect_ratios == (0.5, 1.0, 2.0)
    with pytest.raises(ValueError, match="anchor_scale_factors"):
        SphericalObjectHypothesisMethodConfig(anchor_scale_factors=(0.0,))


def test_method_composes_generation_independent_transfer_and_reduction():
    reconstruction, evidence, concepts = _fixture()
    direct = SphericalObjectHypothesisGenerator(_method().config.generator).generate(
        reconstruction, evidence, concepts
    )

    result = _method().run(
        reconstruction=reconstruction,
        semantic_evidence=evidence,
        concepts=concepts,
        radial_depth=_plane_depth(tuple(reconstruction.poses)),
        independent_region_provider=_independent_region_provider(evidence),
    )

    assert tuple(item.hypothesis_id for item in result.generation.hypotheses) == tuple(
        item.hypothesis_id for item in direct.hypotheses
    )
    assert len(result.generation.confirmed) == 1
    assert len(result.region_evaluations) == 3
    assert all(
        item.metrics.semantic_mass_recall is not None
        for item in result.region_evaluations
    )
    assert len(result.view_pair_evaluations) == 3
    assert len(result.transfer_evaluations) == 3
    assert all(item.role == "within-hypothesis" for item in result.transfer_evaluations)
    assert result.anchor_fusion is not None
    assert len(result.anchor_fusion.clusters) == 1
    assert result.anchor_fusion.clusters[0].state == "confirmed"
    assert result.anchor_fusion.diagnostics["transient_masks_retained"] == 0
    assert result.reduction is not None
    assert len(result.reduction.records) == 1
    assert result.diagnostics["transient_transfer_masks_retained"] == 0
    assert result.diagnostics["geometry_source"] == "refined-sparse-reconstruction"
    assert result.diagnostics["confirmed_anchor_cluster_count"] == 1


def test_method_does_not_measure_tautological_generated_region_transfer():
    reconstruction, evidence, concepts = _fixture()

    result = _method().run(
        reconstruction=reconstruction,
        semantic_evidence=evidence,
        concepts=concepts,
        radial_depth=_plane_depth(tuple(reconstruction.poses)),
    )

    assert result.transfer_evaluations == ()
    assert result.anchor_fusion is None
    assert result.diagnostics["independent_region_hypothesis_count"] == 0


def test_method_can_disable_anchor_fusion_without_disabling_region_transfer():
    reconstruction, evidence, concepts = _fixture()
    method = SphericalObjectHypothesisMethod(
        SphericalObjectHypothesisMethodConfig(
            generator=_method().config.generator,
            enable_anchor_fusion=False,
        )
    )

    result = method.run(
        reconstruction=reconstruction,
        semantic_evidence=evidence,
        concepts=concepts,
        radial_depth=_plane_depth(tuple(reconstruction.poses)),
        independent_region_provider=_independent_region_provider(evidence),
    )

    assert result.anchor_fusion is None
    assert len(result.transfer_evaluations) == 3
    assert result.diagnostics["anchor_cluster_count"] == 0


def test_method_builds_automatic_transfer_cues_without_overwriting_caller_cues():
    reconstruction, evidence, concepts = _fixture(
        (
            np.asarray((0.0, 0.0, 5.0)),
            np.asarray((0.2, 0.0, 5.0)),
            np.asarray((3.0, 0.0, 5.0)),
            np.asarray((3.2, 0.0, 5.0)),
        )
    )
    generated = SphericalObjectHypothesisGenerator(_method().config.generator).generate(
        reconstruction, evidence, concepts
    )
    assert len(generated.confirmed) == 2
    hypothesis_ids = tuple(sorted(item.hypothesis_id for item in generated.confirmed))
    explicit = HypothesisPairCues(
        *hypothesis_ids,
        transfer_iou=0.123,
        geometric_contradiction=True,
        provenance={"caller": "frozen-fixture"},
    )

    result = _method().run(
        reconstruction=reconstruction,
        semantic_evidence=evidence,
        concepts=concepts,
        radial_depth=_plane_depth(tuple(reconstruction.poses)),
        independent_region_provider=_independent_region_provider(evidence),
        pair_cues=(explicit,),
    )

    assert len(result.reduction_pair_cues) == 1
    merged = result.reduction_pair_cues[0]
    assert merged.transfer_iou == pytest.approx(0.123)
    assert merged.soft_nms_overlap is not None
    assert merged.geometric_contradiction
    assert merged.provenance["caller"] == "frozen-fixture"
    assert "automatic_transfer" in merged.provenance
    reduction_transfers = [
        item for item in result.transfer_evaluations if item.role == "reduction-cue"
    ]
    assert len(reduction_transfers) == 6
    assert result.reduction is not None
    assert len(result.reduction.records) == 2


def test_automatic_transfer_skips_incompatible_concepts_before_projection():
    reconstruction, evidence, concepts = _fixture(
        (
            np.asarray((0.0, 0.0, 5.0)),
            np.asarray((0.2, 0.0, 5.0)),
            np.asarray((3.0, 0.0, 5.0)),
            np.asarray((3.2, 0.0, 5.0)),
        )
    )
    expanded_evidence = tuple(
        RegisteredSemanticEvidence(
            item.view_id,
            ("target", "other"),
            np.concatenate((item.scores, item.scores), axis=0),
            item.support,
            item.vocabulary,
            item.source_id,
        )
        for item in evidence
    )
    expanded_concepts = (
        *concepts,
        SemanticConcept("other", "Other", (("other", 1.0),), "fixture/v1"),
    )

    result = _method().run(
        reconstruction=reconstruction,
        semantic_evidence=expanded_evidence,
        concepts=expanded_concepts,
        radial_depth=_plane_depth(tuple(reconstruction.poses)),
        independent_region_provider=_independent_region_provider(expanded_evidence),
    )

    concept_by_id = {
        item.hypothesis_id: item.concept_id for item in result.generation.hypotheses
    }
    assert len(result.reduction_pair_cues) == 2
    assert all(
        concept_by_id[item.hypothesis_id_a] == concept_by_id[item.hypothesis_id_b]
        for item in result.reduction_pair_cues
    )


def test_method_point_free_path_never_receives_input_points():
    reconstruction, evidence, concepts = _fixture()
    bearing_tracks = tuple(
        RegisteredBearingTrack(
            track.track_id,
            tuple(
                RegisteredBearingObservation(
                    item.panorama_id,
                    item.feature_index,
                    item.bearing_xyz,
                )
                for item in track.observations
            ),
        )
        for track in reconstruction.tracks
    )

    result = _method().run(
        poses=tuple(reconstruction.poses),
        bearing_tracks=bearing_tracks,
        semantic_evidence=evidence,
        concepts=concepts,
        reconstruction_scale="metric-known-poses",
        reference_view_id="a",
    )

    assert result.diagnostics["geometry_source"] == "registered-poses-and-bearings"
    assert result.generation.diagnostics["accepted_bearing_track_count"] == 2
    assert all(item.accepted for item in result.generation.bearing_track_triangulations)


def test_method_requires_exactly_one_geometry_source():
    reconstruction, evidence, concepts = _fixture()

    with pytest.raises(ValueError, match="exactly one"):
        _method().run(semantic_evidence=evidence, concepts=concepts)
    with pytest.raises(ValueError, match="exactly one"):
        _method().run(
            reconstruction=reconstruction,
            poses=tuple(reconstruction.poses),
            bearing_tracks=(),
            semantic_evidence=evidence,
            concepts=concepts,
        )


def test_method_rejects_unknown_optional_evidence_ids():
    reconstruction, evidence, concepts = _fixture()
    invalid_depth = RegisteredRadialDepth(
        "unknown",
        np.ones(SHAPE_HW),
        np.ones(SHAPE_HW, dtype=bool),
        "m",
        "unknown-depth",
    )

    with pytest.raises(ValueError, match="unregistered views"):
        _method().run(
            reconstruction=reconstruction,
            semantic_evidence=evidence,
            concepts=concepts,
            radial_depth=(invalid_depth,),
        )
