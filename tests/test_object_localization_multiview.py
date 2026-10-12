from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from panorai.geometry import (
    erp_pixels_to_rays,
    rays_to_erp_pixels,
    rays_to_gnomonic_pixels,
)
from panorai.object_localization import (
    MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE,
    HypothesisViewEmbedding,
    HypothesisPairCues,
    HypothesisReductionCandidate,
    RegisteredRadialDepth,
    RegisteredBearingObservation,
    RegisteredBearingTrack,
    RegisteredPoseInconsistencyError,
    RegisteredSemanticEvidence,
    SemanticConcept,
    SphericalObjectHypothesisConfig,
    SphericalObjectHypothesisGenerator,
    evaluate_spherical_region,
    hypothesis_view_consistency,
    reduce_object_hypotheses,
    symmetric_spherical_region_transfer,
    triangulate_registered_bearing_track,
)
from panorai.reconstruction import (
    SphericalCameraPose,
    SphericalGlobalMapperOptions,
    SphericalReconstructionDiagnostics,
    SphericalReconstructionResult,
    SphericalTrack,
    SphericalTrackObservation,
)


SHAPE_HW = (64, 128)
VIEW_IDS = ("view-a", "view-b", "view-c")


def _bearing(point: np.ndarray, center: np.ndarray) -> np.ndarray:
    vector = point - center
    return vector / np.linalg.norm(vector)


def _fixture_reconstruction(*, scale: str = "metric-baseline-prior"):
    # Independent analytic fixture.  Cameras share the canonical +X right,
    # +Y up, +Z forward frame.  Points and centers use metres for the metric
    # case and the same coordinates in arbitrary units for the scale-free case.
    centers = {
        "view-a": np.asarray((0.0, 0.0, 0.0)),
        "view-b": np.asarray((1.0, 0.0, 0.0)),
        "view-c": np.asarray((0.0, 1.0, 0.0)),
    }
    points = {
        "pipe-0": np.asarray((-0.10, 0.00, 5.00)),
        "pipe-1": np.asarray((0.12, 0.02, 5.05)),
        "pipe-east-0": np.asarray((5.00, -0.04, 6.00)),
        "pipe-east-1": np.asarray((5.20, 0.03, 6.08)),
        "vessel-0": np.asarray((2.90, 0.00, 5.20)),
        "vessel-1": np.asarray((3.12, 0.05, 5.15)),
        "single-0": np.asarray((-3.00, 0.00, 5.00)),
        "single-1": np.asarray((-3.18, 0.04, 5.05)),
    }
    poses = tuple(
        SphericalCameraPose(view_id, np.eye(3), centers[view_id])
        for view_id in VIEW_IDS
    )
    tracks = []
    for track_id, point in points.items():
        observations = tuple(
            SphericalTrackObservation(
                panorama_id=view_id,
                feature_index=index,
                bearing_xyz=_bearing(point, centers[view_id]),
                residual_rad=math.radians(0.05),
            )
            for index, view_id in enumerate(VIEW_IDS)
        )
        tracks.append(SphericalTrack(track_id, point, observations))
    result = SphericalReconstructionResult(
        success=True,
        failure_reasons=(),
        poses=poses,
        tracks=tuple(tracks),
        pairwise_edges=(),
        admitted_edge_indices=(),
        reference_panorama_id="view-a",
        diagnostics=SphericalReconstructionDiagnostics(),
        options=SphericalGlobalMapperOptions(),
        scale=scale,
        _points_xyz=np.stack(tuple(points.values())),
    )
    return result, points


def _paint(scores: np.ndarray, channel: int, bearing: np.ndarray, value: float) -> None:
    pixel = np.asarray(rays_to_erp_pixels(bearing, SHAPE_HW).pixels_xy)
    column = int(math.floor(float(pixel[0]) + 0.5)) % SHAPE_HW[1]
    row = min(max(int(math.floor(float(pixel[1]) + 0.5)), 0), SHAPE_HW[0] - 1)
    scores[channel, row, column] = value


def _fixture_evidence(reconstruction: SphericalReconstructionResult):
    channels = ("coil", "reel", "barrel", "cabinet", "beaker")
    channel_index = {name: index for index, name in enumerate(channels)}
    tracks = {item.track_id: item for item in reconstruction.tracks}
    evidence = []
    for view_id in VIEW_IDS:
        scores = np.zeros((len(channels), *SHAPE_HW), dtype=np.float32)
        for track_id in ("pipe-0", "pipe-1", "pipe-east-0", "pipe-east-1"):
            observation = next(
                item
                for item in tracks[track_id].observations
                if item.panorama_id == view_id
            )
            _paint(scores, channel_index["coil"], observation.bearing_xyz, 0.8)
            _paint(scores, channel_index["reel"], observation.bearing_xyz, 0.6)
        for track_id in ("vessel-0", "vessel-1"):
            observation = next(
                item
                for item in tracks[track_id].observations
                if item.panorama_id == view_id
            )
            _paint(scores, channel_index["barrel"], observation.bearing_xyz, 0.9)
        if view_id == "view-a":
            for track_id in ("single-0", "single-1"):
                observation = next(
                    item
                    for item in tracks[track_id].observations
                    if item.panorama_id == view_id
                )
                _paint(scores, channel_index["cabinet"], observation.bearing_xyz, 0.85)
        # A semantic direction with no reconstructed track exercises the honest
        # bearing-only fallback.  Distinct columns make each view provenance
        # explicit without implying cross-view identity.
        scores[channel_index["beaker"], 8, 15 + VIEW_IDS.index(view_id)] = 0.75
        evidence.append(
            RegisteredSemanticEvidence(
                view_id=view_id,
                channel_ids=channels,
                scores=scores,
                support=np.ones(SHAPE_HW, dtype=bool),
                vocabulary="industrial-proxy/v1",
                source_id=f"cam:{view_id}",
                provenance={"fixture": "analytic-registered-tracks"},
            )
        )
    return tuple(evidence)


def _bearing_tracks_without_points(reconstruction: SphericalReconstructionResult):
    return tuple(
        RegisteredBearingTrack(
            track_id=track.track_id,
            observations=tuple(
                RegisteredBearingObservation(
                    view_id=item.panorama_id,
                    feature_index=item.feature_index,
                    bearing_xyz=item.bearing_xyz,
                )
                for item in track.observations
            ),
        )
        for track in reconstruction.tracks
    )


def _concepts():
    return (
        SemanticConcept(
            "piping",
            "Piping",
            (("coil", 1.0), ("reel", 1.0)),
            "industrial-proxy/v1",
        ),
        SemanticConcept(
            "vessel",
            "Vessel",
            (("barrel", 1.0),),
            "industrial-proxy/v1",
        ),
        SemanticConcept(
            "cabinet",
            "Cabinet",
            (("cabinet", 1.0),),
            "industrial-proxy/v1",
        ),
        SemanticConcept(
            "untracked-vessel",
            "Untracked vessel direction",
            (("beaker", 1.0),),
            "industrial-proxy/v1",
        ),
    )


def _generator():
    return SphericalObjectHypothesisGenerator(
        SphericalObjectHypothesisConfig(
            minimum_observation_score=0.25,
            minimum_track_score=0.25,
            minimum_tracks_per_hypothesis=2,
            spatial_cluster_radius=0.50,
        )
    )


def test_registered_tracks_produce_confirmed_multiview_spatial_hypotheses():
    reconstruction, points = _fixture_reconstruction()
    result = _generator().generate(
        reconstruction, _fixture_evidence(reconstruction), _concepts()
    )

    by_concept = {}
    for hypothesis in result.hypotheses:
        by_concept.setdefault(hypothesis.concept_id, []).append(hypothesis)

    assert len(by_concept["piping"]) == 2
    piping = next(
        item for item in by_concept["piping"] if item.track_ids == ("pipe-0", "pipe-1")
    )
    assert piping.state == "confirmed"
    assert piping.view_ids == VIEW_IDS
    assert piping.track_ids == ("pipe-0", "pipe-1")
    assert piping.proxy_ids == ("coil", "reel")
    assert piping.semantic_score == pytest.approx(0.7, abs=1e-6)
    assert piping.spatial.mode == "metric_3d"
    assert piping.spatial.units == "m"
    np.testing.assert_allclose(
        piping.spatial.center_xyz,
        0.5 * (points["pipe-0"] + points["pipe-1"]),
        atol=1e-6,
    )
    assert piping.spatial.extent_radius > 0.0
    assert piping.spatial.extent_covariance_xyz.shape == (3, 3)
    assert tuple(item.view_id for item in piping.view_regions) == VIEW_IDS
    for region in piping.view_regions:
        assert region.spec.hfov_deg >= 70.0
        assert region.spec.vfov_deg >= 70.0
        assert region.sparse_support_coverage == pytest.approx(1.0)
        projected = rays_to_gnomonic_pixels(region.support_bearings_xyz, region.spec)
        assert np.all(projected.valid)
        assert region.track_ids == ("pipe-0", "pipe-1")
    assert {item.source_id for item in piping.observations} == {
        "cam:view-a",
        "cam:view-b",
        "cam:view-c",
    }
    repeated = next(
        item
        for item in by_concept["piping"]
        if item.track_ids == ("pipe-east-0", "pipe-east-1")
    )
    assert repeated.state == "confirmed"
    assert repeated.hypothesis_id != piping.hypothesis_id

    vessel = by_concept["vessel"][0]
    assert vessel.state == "confirmed"
    assert vessel.track_ids == ("vessel-0", "vessel-1")
    assert vessel.semantic_score == pytest.approx(0.9)

    cabinet = by_concept["cabinet"][0]
    assert cabinet.state == "proposed"
    assert cabinet.view_ids == ("view-a",)
    assert cabinet.reasons == ("too-few-independent-semantic-views",)

    angular = by_concept["untracked-vessel"]
    assert len(angular) == 3
    assert all(item.state == "proposed" for item in angular)
    assert all(item.spatial.mode == "bearing_only" for item in angular)
    assert {item.spatial.bearing_view_id for item in angular} == set(VIEW_IDS)
    assert result.interface == MULTIVIEW_OBJECT_HYPOTHESES_INTERFACE
    assert result.diagnostics["confirmed_hypothesis_count"] == 3
    assert result.diagnostics["spatial_point_source"] == (
        "refined-sparse-reconstruction"
    )
    assert piping.provenance["geometry_convention"] == "panorai-geometry-v1"
    assert piping.provenance["spatial_point_source"] == (
        "refined-sparse-reconstruction"
    )
    assert piping.provenance["concept"]["proxy_weights"] == (
        ("coil", 0.5),
        ("reel", 0.5),
    )
    assert len(piping.provenance["semantic_sources"]) == 3
    assert piping.provenance["generator_config"]["minimum_track_score"] == 0.25
    assert len(result.confirmed) == 3


def test_pose_plus_bearings_estimates_latent_points_without_point_input():
    reconstruction, points = _fixture_reconstruction()
    result = _generator().generate_from_bearings(
        poses=tuple(reconstruction.poses),
        tracks=_bearing_tracks_without_points(reconstruction),
        semantic_evidence=_fixture_evidence(reconstruction),
        concepts=_concepts(),
        scale="metric-known-poses",
        reference_view_id="view-a",
    )

    piping = next(
        item for item in result.confirmed if item.track_ids == ("pipe-0", "pipe-1")
    )
    np.testing.assert_allclose(
        piping.spatial.center_xyz,
        0.5 * (points["pipe-0"] + points["pipe-1"]),
        atol=1e-8,
    )
    assert result.diagnostics["spatial_point_source"] == (
        "triangulated-registered-bearings"
    )
    assert result.diagnostics["accepted_bearing_track_count"] == len(
        reconstruction.tracks
    )
    assert all(item.accepted for item in result.bearing_track_triangulations)
    pipe = next(
        item
        for item in result.bearing_track_triangulations
        if item.track_id == "pipe-0"
    )
    np.testing.assert_allclose(pipe.point_xyz, points["pipe-0"], atol=1e-8)
    assert pipe.maximum_angular_residual_deg == pytest.approx(0.0, abs=1e-6)
    assert piping.provenance["spatial_point_source"] == (
        "triangulated-registered-bearings"
    )
    assert {
        item["track_id"] for item in piping.provenance["bearing_triangulation"]
    } == {"pipe-0", "pipe-1"}


def test_low_parallax_bearing_track_is_local_degeneracy_not_a_fake_point():
    poses = (
        SphericalCameraPose("a", np.eye(3), np.asarray((0.0, 0.0, 0.0))),
        SphericalCameraPose("b", np.eye(3), np.asarray((1e-5, 0.0, 0.0))),
    )
    point = np.asarray((0.0, 0.0, 10.0))
    track = RegisteredBearingTrack(
        "weak-parallax",
        tuple(
            RegisteredBearingObservation(
                pose.panorama_id,
                index,
                _bearing(point, pose.center),
            )
            for index, pose in enumerate(poses)
        ),
    )

    triangulation = triangulate_registered_bearing_track(
        track, poses, _generator().config
    )

    assert triangulation.state == "degenerate"
    assert triangulation.point_xyz is None
    assert "insufficient-parallax" in triangulation.reasons


def test_low_parallax_track_does_not_trigger_pose_failure():
    poses = (
        SphericalCameraPose("a", np.eye(3), np.asarray((0.0, 0.0, 0.0))),
        SphericalCameraPose("b", np.eye(3), np.asarray((1e-5, 0.0, 0.0))),
    )
    point = np.asarray((0.0, 0.0, 10.0))
    track = RegisteredBearingTrack(
        "weak-parallax",
        tuple(
            RegisteredBearingObservation(
                pose.panorama_id,
                index,
                _bearing(point, pose.center),
            )
            for index, pose in enumerate(poses)
        ),
    )
    evidence = tuple(
        RegisteredSemanticEvidence(
            view_id=pose.panorama_id,
            channel_ids=("coil",),
            scores=np.zeros((1, *SHAPE_HW), dtype=np.float32),
            support=np.ones(SHAPE_HW, dtype=bool),
            vocabulary="industrial-proxy/v1",
            source_id=f"cam:{pose.panorama_id}",
        )
        for pose in poses
    )

    result = _generator().generate_from_bearings(
        poses=poses,
        tracks=(track,),
        semantic_evidence=evidence,
        concepts=(
            SemanticConcept(
                "piping",
                "Piping",
                (("coil", 1.0),),
                "industrial-proxy/v1",
            ),
        ),
    )

    assert result.hypotheses == ()
    assert result.diagnostics["degenerate_bearing_track_count"] == 1
    assert result.diagnostics["pose_inconsistent_bearing_track_count"] == 0
    assert result.diagnostics["spatial_point_source"] == "none-bearing-only"


def test_bearing_triangulation_respects_world_to_panorama_rotations():
    angle_y = math.radians(27.0)
    angle_x = math.radians(-19.0)
    rotation_y = np.asarray(
        (
            (math.cos(angle_y), 0.0, math.sin(angle_y)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle_y), 0.0, math.cos(angle_y)),
        )
    )
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(angle_x), -math.sin(angle_x)),
            (0.0, math.sin(angle_x), math.cos(angle_x)),
        )
    )
    poses = (
        SphericalCameraPose("a", np.eye(3), np.asarray((0.0, 0.0, 0.0))),
        SphericalCameraPose("b", rotation_y, np.asarray((1.0, 0.0, 0.0))),
        SphericalCameraPose("c", rotation_x, np.asarray((0.0, 1.0, 0.0))),
    )
    expected = np.asarray((0.4, -0.2, 4.5))
    track = RegisteredBearingTrack(
        "rotated-cameras",
        tuple(
            RegisteredBearingObservation(
                pose.panorama_id,
                index,
                pose.R @ _bearing(expected, pose.center),
            )
            for index, pose in enumerate(poses)
        ),
    )

    triangulation = triangulate_registered_bearing_track(track, poses)

    assert triangulation.accepted
    np.testing.assert_allclose(triangulation.point_xyz, expected, atol=1e-9)
    assert triangulation.maximum_angular_residual_deg == pytest.approx(0.0, abs=1e-6)


def test_systematic_non_degenerate_track_failure_rejects_registered_poses():
    reconstruction, _ = _fixture_reconstruction()
    tracks = []
    for track in _bearing_tracks_without_points(reconstruction)[:4]:
        observations = list(track.observations)
        observations[1] = replace(
            observations[1],
            bearing_xyz=np.asarray((1.0, 0.0, 0.0)),
        )
        tracks.append(replace(track, observations=tuple(observations)))

    with pytest.raises(
        RegisteredPoseInconsistencyError, match="poses are inconsistent"
    ) as caught:
        _generator().generate_from_bearings(
            poses=tuple(reconstruction.poses),
            tracks=tuple(tracks),
            semantic_evidence=_fixture_evidence(reconstruction),
            concepts=_concepts(),
        )
    assert caught.value.input_track_count == 4
    assert caught.value.nondegenerate_track_count == 4
    assert caught.value.rejected_track_count == 4
    assert caught.value.rejected_fraction == 1.0
    assert sum(caught.value.reason_counts.values()) >= 4
    assert set(caught.value.reason_counts) <= {
        "excessive-angular-reprojection-error",
        "insufficient-positive-depth",
    }


def test_hypothesis_regions_expose_track_and_optional_embedding_consistency():
    reconstruction, _ = _fixture_reconstruction()
    result = _generator().generate(
        reconstruction, _fixture_evidence(reconstruction), _concepts()
    )
    piping = next(
        item for item in result.confirmed if item.track_ids == ("pipe-0", "pipe-1")
    )

    without_embeddings = hypothesis_view_consistency(piping)

    assert len(without_embeddings) == 3
    assert all(item.shared_track_count == 2 for item in without_embeddings)
    assert all(item.union_track_count == 2 for item in without_embeddings)
    assert all(item.track_jaccard == pytest.approx(1.0) for item in without_embeddings)
    assert all(
        item.median_angular_reprojection_residual_deg == pytest.approx(0.05)
        for item in without_embeddings
    )
    assert all(
        item.reprojection_agreement == pytest.approx(math.exp(-0.05))
        for item in without_embeddings
    )
    assert without_embeddings[0].matched_feature_pairs == ((0, 1), (0, 1))
    assert without_embeddings[0].embedding_cosine_similarity is None

    embeddings = (
        HypothesisViewEmbedding(
            "view-a", np.asarray((1.0, 0.0)), "fixture:a", "encoder/v1"
        ),
        HypothesisViewEmbedding(
            "view-b", np.asarray((0.8, 0.6)), "fixture:b", "encoder/v1"
        ),
        HypothesisViewEmbedding(
            "view-c", np.asarray((0.0, 1.0)), "fixture:c", "encoder/v1"
        ),
    )
    with_embeddings = hypothesis_view_consistency(piping, embeddings)

    assert with_embeddings[0].embedding_cosine_similarity == pytest.approx(0.8)
    assert with_embeddings[0].embedding_source_ids == ("fixture:a", "fixture:b")
    assert with_embeddings[0].embedding_preprocessing_id == "encoder/v1"


def test_generated_regions_feed_public_utility_and_transfer_apis():
    reconstruction, _ = _fixture_reconstruction()
    evidence = _fixture_evidence(reconstruction)
    concepts = _concepts()
    result = _generator().generate(reconstruction, evidence, concepts)
    piping = next(
        item for item in result.confirmed if item.track_ids == ("pipe-0", "pipe-1")
    )
    regions = {item.view_id: item for item in piping.view_regions}
    evidence_by_view = {item.view_id: item for item in evidence}
    poses = {item.panorama_id: item for item in reconstruction.poses}

    utility = evaluate_spherical_region(
        evidence_by_view["view-a"], concepts[0], regions["view-a"]
    )

    assert utility.semantic_mass_recall is not None
    assert utility.semantic_mass_recall > 0.0
    assert utility.observed_sphere_fraction is not None
    assert 0.0 < utility.observed_sphere_fraction < 1.0

    def plane_depth(view_id: str) -> RegisteredRadialDepth:
        height, width = SHAPE_HW
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        rays = np.asarray(
            erp_pixels_to_rays(np.column_stack((xx.ravel(), yy.ravel())), SHAPE_HW)
        ).reshape(height, width, 3)
        world_rays = rays @ poses[view_id].R
        support = world_rays[..., 2] > 1e-6
        values = np.full(SHAPE_HW, np.nan)
        values[support] = (10.0 - float(poses[view_id].center[2])) / world_rays[..., 2][
            support
        ]
        return RegisteredRadialDepth(
            view_id,
            values,
            support,
            "m",
            f"analytic-plane:{view_id}",
        )

    transfer = symmetric_spherical_region_transfer(
        regions["view-a"],
        regions["view-b"],
        plane_depth("view-a"),
        plane_depth("view-b"),
        poses["view-a"],
        poses["view-b"],
    )

    assert 0.0 <= transfer.symmetric_solid_angle_iou <= 1.0
    assert transfer.forward.source_supported_depth_count > 0
    assert transfer.reverse.source_supported_depth_count > 0


def test_generated_hypotheses_feed_detector_style_reduction():
    reconstruction, _ = _fixture_reconstruction()
    result = _generator().generate(
        reconstruction, _fixture_evidence(reconstruction), _concepts()
    )
    piping = sorted(
        (item for item in result.confirmed if item.concept_id == "piping"),
        key=lambda item: item.hypothesis_id,
    )
    assert len(piping) == 2
    duplicate = replace(
        piping[0],
        hypothesis_id="synthetic-duplicate",
        semantic_score=0.7,
    )

    def cues(first: str, second: str, **kwargs):
        one, two = sorted((first, second))
        return HypothesisPairCues(one, two, **kwargs)

    reduction = reduce_object_hypotheses(
        (
            HypothesisReductionCandidate(piping[0]),
            HypothesisReductionCandidate(duplicate),
            HypothesisReductionCandidate(piping[1]),
        ),
        pair_cues=(
            cues(
                piping[0].hypothesis_id,
                duplicate.hypothesis_id,
                geometric_compatibility=1.0,
                transfer_iou=1.0,
            ),
            cues(
                piping[0].hypothesis_id,
                piping[1].hypothesis_id,
                geometric_contradiction=True,
            ),
            cues(
                duplicate.hypothesis_id,
                piping[1].hypothesis_id,
                geometric_contradiction=True,
            ),
        ),
    )

    source_groups = {item.source_hypothesis_ids for item in reduction.records}
    assert (
        tuple(sorted((piping[0].hypothesis_id, duplicate.hypothesis_id)))
        in source_groups
    )
    assert (piping[1].hypothesis_id,) in source_groups
    assert reduction.diagnostics["fused_record_count"] == 1


def test_hypothesis_view_consistency_rejects_mixed_embedding_preprocessing():
    reconstruction, _ = _fixture_reconstruction()
    result = _generator().generate(
        reconstruction, _fixture_evidence(reconstruction), _concepts()
    )
    piping = next(
        item for item in result.confirmed if item.track_ids == ("pipe-0", "pipe-1")
    )

    with pytest.raises(ValueError, match="preprocessing contract"):
        hypothesis_view_consistency(
            piping,
            (
                HypothesisViewEmbedding(
                    "view-a", np.asarray((1.0, 0.0)), "fixture:a", "encoder/v1"
                ),
                HypothesisViewEmbedding(
                    "view-b", np.asarray((1.0, 0.0)), "fixture:b", "encoder/v2"
                ),
            ),
        )


def test_hypothesis_window_is_projected_into_registered_view_without_cam():
    reconstruction, _ = _fixture_reconstruction()
    evidence = _fixture_evidence(reconstruction)[:2]
    result = _generator().generate(reconstruction, evidence, _concepts()[:2])
    piping = next(
        item for item in result.confirmed if item.track_ids == ("pipe-0", "pipe-1")
    )

    region_c = next(item for item in piping.view_regions if item.view_id == "view-c")

    assert not region_c.semantic_evidence_available
    assert not region_c.semantic_supporting
    assert region_c.track_ids == ("pipe-0", "pipe-1")
    assert region_c.sparse_support_coverage == pytest.approx(1.0)


def test_hypothesis_ids_are_order_independent_and_scale_is_explicit():
    reconstruction, _ = _fixture_reconstruction(scale="arbitrary")
    evidence = _fixture_evidence(reconstruction)
    concepts = _concepts()

    forward = _generator().generate(reconstruction, evidence, concepts)
    reverse = _generator().generate(
        reconstruction, tuple(reversed(evidence)), tuple(reversed(concepts))
    )

    assert [item.hypothesis_id for item in forward.hypotheses] == [
        item.hypothesis_id for item in reverse.hypotheses
    ]
    spatial = [item for item in forward.hypotheses if item.track_ids]
    assert all(item.spatial.mode == "scale_free_3d" for item in spatial)
    assert all(
        item.spatial.units == "arbitrary-reconstruction-scale" for item in spatial
    )


def test_dense_evidence_adapter_is_model_agnostic_and_preserves_support():
    scores = np.zeros((2, 4, 8), dtype=np.float32)
    scores[0, 1, 2] = 0.5
    dense = SimpleNamespace(
        concept_ids=("coil", "reel"),
        scores=scores,
        support=np.ones((4, 8), dtype=bool),
        vocabulary="industrial-proxy/v1",
        provenance={"source_id": "dense:test", "model": "fixture"},
    )

    adapted = RegisteredSemanticEvidence.from_dense("view-a", dense)

    assert adapted.channel_ids == ("coil", "reel")
    assert adapted.source_id == "dense:test"
    assert adapted.provenance["model"] == "fixture"
    assert not adapted.scores.flags.writeable
    assert not adapted.support.flags.writeable


def test_actual_dense_semantic_evidence_integrates_without_conversion_code():
    from panorai.experimental.deep_learning.segmentation import DenseSemanticEvidence

    dense = DenseSemanticEvidence(
        concept_ids=("coil", "reel"),
        scores=np.zeros((2, 4, 8), dtype=np.float32),
        support=np.ones((4, 8), dtype=bool),
        vocabulary="industrial-proxy/v1",
        provenance={"source_id": "real-contract:test"},
    )

    adapted = RegisteredSemanticEvidence.from_dense("view-a", dense)

    assert adapted.channel_ids == dense.concept_ids
    assert np.array_equal(adapted.scores, dense.scores)
    assert adapted.source_id == "real-contract:test"


def test_unsupported_cam_cells_do_not_create_semantic_track_support():
    reconstruction, _ = _fixture_reconstruction()
    evidence = list(_fixture_evidence(reconstruction))
    first = evidence[0]
    support = first.support.copy()
    scores = first.scores.copy()
    pipe = next(item for item in reconstruction.tracks if item.track_id == "pipe-0")
    observation = next(
        item for item in pipe.observations if item.panorama_id == "view-a"
    )
    pixel = np.asarray(rays_to_erp_pixels(observation.bearing_xyz, SHAPE_HW).pixels_xy)
    column = int(math.floor(float(pixel[0]) + 0.5)) % SHAPE_HW[1]
    row = min(max(int(math.floor(float(pixel[1]) + 0.5)), 0), SHAPE_HW[0] - 1)
    support[row, column] = False
    scores[:, row, column] = 0.0
    evidence[0] = replace(first, scores=scores, support=support)

    result = _generator().generate(reconstruction, tuple(evidence), _concepts())

    sampled = [
        item
        for item in result.track_observations
        if item.track_id == "pipe-0"
        and item.view_id == "view-a"
        and item.concept_id == "piping"
    ]
    assert len(sampled) == 1
    assert sampled[0].supported is False
    assert sampled[0].concept_score == 0.0


def test_multiview_contract_fails_closed_for_invalid_registration_or_vocabulary():
    reconstruction, _ = _fixture_reconstruction()
    evidence = _fixture_evidence(reconstruction)
    generator = _generator()

    with pytest.raises(ValueError, match="at least two views"):
        generator.generate(reconstruction, evidence[:1], _concepts())
    with pytest.raises(RuntimeError, match="spherical reconstruction failed"):
        generator.generate(
            replace(
                reconstruction,
                success=False,
                failure_reasons=("fixture-failure",),
                poses=(),
                tracks=(),
                _points_xyz=np.empty((0, 3)),
            ),
            evidence,
            _concepts(),
        )
    incompatible = replace(_concepts()[0], vocabulary="imagenet-1k")
    with pytest.raises(ValueError, match="incompatible evidence vocabulary"):
        generator.generate(reconstruction, evidence, (incompatible,))


def test_configuration_and_concepts_reject_ambiguous_values():
    with pytest.raises(ValueError, match="at least two"):
        SphericalObjectHypothesisConfig(minimum_independent_views=1)
    with pytest.raises(ValueError, match="spatial_cluster_radius"):
        SphericalObjectHypothesisConfig(spatial_cluster_radius=0.0)
    with pytest.raises(ValueError, match="repeat proxy"):
        SemanticConcept("x", "X", (("coil", 1.0), ("coil", 2.0)))
    concept = SemanticConcept("x", "X", (("coil", 1.0), ("reel", 3.0)))
    assert concept.proxy_weights == (("coil", 0.25), ("reel", 0.75))
