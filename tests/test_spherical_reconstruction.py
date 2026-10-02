from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from panorai.estimators import (
    RelativePoseAcceptancePolicy,
    RelativePoseOptions,
    SphericalRelativePoseEstimator,
)
from panorai.features import MatchProvenance, SphericalFeatureMatches
from panorai.reconstruction import (
    SphericalGlobalMapper,
    SphericalGlobalMapperOptions,
    SphericalPairwisePoseEdge,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1.0 - math.cos(angle)) / angle**2 * (cross @ cross)
    )


def _scene(count: int = 36):
    rng = np.random.default_rng(20261002)
    ids = ("pano-a", "pano-b", "pano-c")
    rotations = {
        "pano-a": np.eye(3),
        "pano-b": _rotation_exp(np.asarray((0.03, -0.08, 0.02))),
        "pano-c": _rotation_exp(np.asarray((-0.04, 0.05, 0.07))),
    }
    centers = {
        "pano-a": np.zeros(3),
        "pano-b": np.asarray((1.0, 0.1, 0.05)),
        "pano-c": np.asarray((0.25, 0.85, 0.15)),
    }
    points = rng.uniform((-2.5, -1.8, 3.0), (2.5, 1.8, 8.0), size=(count, 3))
    bearings = {}
    for panorama_id in ids:
        vectors = (points - centers[panorama_id]) @ rotations[panorama_id].T
        bearings[panorama_id] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    matches = []
    for a, b in (("pano-a", "pano-b"), ("pano-b", "pano-c"), ("pano-a", "pano-c")):
        matches.append(
            SphericalFeatureMatches(
                panorama_id_a=a,
                panorama_id_b=b,
                feature_indices_a=np.arange(count),
                feature_indices_b=np.arange(count),
                bearings_a=bearings[a].copy(),
                bearings_b=bearings[b].copy(),
                descriptor_distances=np.linspace(0.01, 0.2, count),
                ratio_scores=np.full(count, 0.5),
                mutual=np.ones(count, dtype=bool),
                valid=np.ones(count, dtype=bool),
                matcher_name="synthetic-independent-oracle",
                matcher_config={},
                backend_name="test-oracle",
                backend_version="1",
                provenance=MatchProvenance(
                    interface="test-independent-multiview/v1",
                    source_checksums=(a, b),
                    face_pairs=tuple(("sphere", "sphere") for _ in range(count)),
                    face_pair_groups=tuple(
                        (("sphere", "sphere"),) for _ in range(count)
                    ),
                    deduplicated=True,
                ),
                keypoint_responses=np.ones((count, 2)),
                face_ids_a=np.full(count, "sphere", dtype=object),
                face_ids_b=np.full(count, "sphere", dtype=object),
            )
        )
    return matches, rotations, centers, points


def _matches_for_pair(a, b, bearings, count, *, extra_conflict=False):
    indices_a = np.arange(count)
    indices_b = np.arange(count)
    bearings_a = bearings[a].copy()
    bearings_b = bearings[b].copy()
    if extra_conflict:
        indices_a = np.concatenate((indices_a, np.asarray((1,))))
        indices_b = np.concatenate((indices_b, np.asarray((0,))))
        bearings_a = np.concatenate((bearings_a, bearings_a[1:2]))
        bearings_b = np.concatenate((bearings_b, bearings_b[0:1]))
    size = len(indices_a)
    return SphericalFeatureMatches(
        panorama_id_a=a,
        panorama_id_b=b,
        feature_indices_a=indices_a,
        feature_indices_b=indices_b,
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.linspace(0.01, 0.2, size),
        ratio_scores=np.full(size, 0.5),
        mutual=np.ones(size, dtype=bool),
        valid=np.ones(size, dtype=bool),
        matcher_name="synthetic-independent-oracle",
        matcher_config={},
        backend_name="test-oracle",
        backend_version="1",
        provenance=MatchProvenance(
            interface="test-independent-multiview/v1",
            source_checksums=(a, b),
            face_pairs=tuple(("sphere", "sphere") for _ in range(size)),
            face_pair_groups=tuple((("sphere", "sphere"),) for _ in range(size)),
            deduplicated=True,
        ),
        keypoint_responses=np.ones((size, 2)),
        face_ids_a=np.full(size, "sphere", dtype=object),
        face_ids_b=np.full(size, "sphere", dtype=object),
    )


def _oracle_graph(template_pose, camera_count, *, noise_deg=0.0, outlier_fraction=0.0):
    rng = np.random.default_rng(991 + camera_count)
    count = 24
    ids = tuple(f"view-{index:02d}" for index in range(camera_count))
    rotations = {ids[0]: np.eye(3)}
    centers = {ids[0]: np.zeros(3)}
    for index, panorama_id in enumerate(ids[1:], start=1):
        angle = 2.0 * math.pi * (index - 1) / max(2, camera_count - 1)
        centers[panorama_id] = np.asarray(
            (1.2 * math.cos(angle), 0.8 * math.sin(angle), 0.08 * index)
        )
        rotations[panorama_id] = _rotation_exp(
            np.asarray((0.01 * index, -0.015 * index, 0.008 * index))
        )
    points = rng.uniform((-2.0, -1.5, 3.5), (2.0, 1.5, 7.5), size=(count, 3))
    bearings = {}
    for panorama_id in ids:
        vectors = (points - centers[panorama_id]) @ rotations[panorama_id].T
        bearings[panorama_id] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        if noise_deg:
            bearings[panorama_id] += rng.normal(
                scale=math.radians(noise_deg), size=(count, 3)
            )
            bearings[panorama_id] /= np.linalg.norm(
                bearings[panorama_id], axis=1, keepdims=True
            )
    pair_ids = [(ids[index], ids[index + 1]) for index in range(camera_count - 1)]
    pair_ids.append((ids[-1], ids[0]))
    if camera_count >= 5:
        pair_ids.extend(
            (ids[index], ids[index + 2]) for index in range(camera_count - 2)
        )
    edges = []
    for pair_index, (a, b) in enumerate(pair_ids):
        matches = _matches_for_pair(a, b, bearings, count)
        inliers = np.ones(count, dtype=bool)
        if outlier_fraction:
            outlier_count = int(round(outlier_fraction * count))
            outliers = rng.choice(count, size=outlier_count, replace=False)
            inliers[outliers] = False
            matches.bearings_b[outliers] = rng.normal(size=(outlier_count, 3))
            matches.bearings_b[outliers] /= np.linalg.norm(
                matches.bearings_b[outliers], axis=1, keepdims=True
            )
        relative_rotation = rotations[b] @ rotations[a].T
        translation = rotations[b] @ (centers[a] - centers[b])
        translation /= np.linalg.norm(translation)
        essential = _skew(translation) @ relative_rotation
        essential /= np.linalg.norm(essential)
        quality = replace(
            template_pose.quality_report,
            num_correspondences=count,
            num_inliers=int(inliers.sum()),
            inlier_ratio=float(inliers.mean()),
            accepted=True,
            rejection_reasons=(),
        )
        pose = replace(
            template_pose,
            rotation=relative_rotation,
            translation_direction=translation,
            essential_matrix=essential,
            inlier_mask=inliers,
            residuals_rad=np.where(inliers, 0.0, np.inf),
            num_inliers=int(inliers.sum()),
            quality_report=quality,
        )
        edges.append(SphericalPairwisePoseEdge(matches, pose))
    return tuple(edges), rotations, centers


@pytest.fixture(scope="module")
def reconstruction_evidence():
    matches, rotations, centers, points = _scene()
    estimator = SphericalRelativePoseEstimator(
        RelativePoseOptions(
            max_angular_error_deg=0.1,
            min_num_trials=8,
            max_num_trials=40,
            min_inliers=8,
            local_optimization_steps=1,
            stability_trials=0,
            model_competition_trials=20,
            random_seed=9,
        ),
        quality_policy=RelativePoseAcceptancePolicy(
            min_inliers=5,
            min_inlier_ratio=0.0,
            min_occupied_cells=1,
            min_coverage_entropy=0.0,
            max_median_residual_deg=1.0,
            min_median_parallax_deg=0.0,
            min_cheirality_ratio=0.0,
            require_stability=False,
            require_essential_preferred=False,
        ),
    )
    mapper = SphericalGlobalMapper(
        estimator,
        SphericalGlobalMapperOptions(
            rotation_max_iterations=60,
            bundle_max_nfev=60,
            max_refinement_rounds=2,
        ),
    )
    edges = mapper.estimate_pairwise(matches)
    assert len(edges) == 3
    assert all(edge.accepted for edge in edges)
    return mapper, matches, edges, rotations, centers, points


def _similarity_align(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(singular.sum() / np.sum(source_zero**2))
    translation = target_mean - scale * source_mean @ rotation
    return scale * source @ rotation + translation


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(actual @ expected.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def test_global_mapper_recovers_exact_multiview_scene(reconstruction_evidence):
    mapper, _, edges, expected_rotations, expected_centers, _ = reconstruction_evidence

    result = mapper.reconstruct(edges=edges)

    assert result.success, result.failure_reasons
    assert result.reference_panorama_id == "pano-a"
    assert len(result.poses) == 3
    assert len(result.tracks) == 36
    assert result.points_xyz.shape == (36, 3)
    assert result.scale == "arbitrary"
    assert result.describe()["interface"] == "panorai-spherical-reconstruction/v1"
    for panorama_id, expected in expected_rotations.items():
        assert _rotation_error_deg(result.pose(panorama_id).R, expected) < 1e-3
    actual_centers = np.stack(
        [result.pose(name).center for name in sorted(expected_centers)]
    )
    target_centers = np.stack(
        [expected_centers[name] for name in sorted(expected_centers)]
    )
    aligned = _similarity_align(actual_centers, target_centers)
    assert np.max(np.linalg.norm(aligned - target_centers, axis=1)) < 1e-3
    assert not result.points_xyz.flags.writeable
    assert not result.pose("pano-a").R.flags.writeable
    assert all(
        after <= before + 1e-12 for _, before, after in result.diagnostics.bundle_costs
    )


def test_matches_and_precomputed_edges_are_equivalent(reconstruction_evidence):
    mapper, matches, edges, _, _, _ = reconstruction_evidence

    from_matches = mapper.reconstruct(matches=matches)
    from_edges = mapper.reconstruct(edges=edges)

    assert from_matches.success and from_edges.success
    assert np.allclose(from_matches.points_xyz, from_edges.points_xyz, atol=1e-9)
    assert from_matches.describe() == from_edges.describe()


def test_default_admission_excludes_rejected_edges(reconstruction_evidence):
    mapper, _, edges, _, _, _ = reconstruction_evidence
    rejected_pose = replace(
        edges[0].pose,
        quality_report=replace(
            edges[0].pose.quality_report,
            accepted=False,
            rejection_reasons=("test-rejected",),
        ),
    )
    rejected = SphericalPairwisePoseEdge(edges[0].matches, rejected_pose)

    result = mapper.reconstruct(edges=(rejected, edges[1], edges[2]))

    assert result.success
    assert result.diagnostics.rejected_edge_pairs == (rejected.pair,)
    assert result.diagnostics.admitted_edge_count == 2


def test_largest_component_and_input_order_are_deterministic(reconstruction_evidence):
    mapper, _, edges, _, _, _ = reconstruction_evidence
    forward = mapper.reconstruct(
        edges=edges,
        panorama_ids=("isolated-a", "isolated-b", "pano-a", "pano-b", "pano-c"),
    )
    reverse = mapper.reconstruct(
        edges=tuple(reversed(edges)),
        panorama_ids=("isolated-b", "pano-c", "pano-a", "isolated-a", "pano-b"),
    )

    assert forward.success and reverse.success
    assert forward.diagnostics.excluded_panoramas == ("isolated-a", "isolated-b")
    assert np.allclose(forward.points_xyz, reverse.points_xyz, atol=1e-8)
    assert [item.panorama_id for item in forward.poses] == [
        item.panorama_id for item in reverse.poses
    ]


def test_geometric_insufficiency_returns_explicit_empty_result(reconstruction_evidence):
    mapper, matches, edges, _, _, _ = reconstruction_evidence

    result = mapper.reconstruct(edges=edges[:1])

    assert not result.success
    assert result.failure_reasons == ("insufficient-connected-panoramas",)
    assert result.poses == () and result.tracks == ()
    assert result.points_xyz.shape == (0, 3)
    with pytest.raises(RuntimeError, match="insufficient-connected-panoramas"):
        result.require_success()
    with pytest.raises(ValueError, match="exactly one"):
        mapper.reconstruct(matches=matches, edges=edges)
    with pytest.raises(ValueError, match="exactly one"):
        mapper.reconstruct()


def test_duplicate_edges_and_inconsistent_bearings_fail_explicitly(
    reconstruction_evidence,
):
    mapper, _, edges, _, _, _ = reconstruction_evidence
    with pytest.raises(ValueError, match="duplicate unordered"):
        mapper.reconstruct(edges=(edges[0], edges[0], edges[1]))

    bad_matches = replace(
        edges[1].matches,
        bearings_a=edges[1].matches.bearings_a.copy(),
    )
    bad_matches.bearings_a[0] = np.asarray((1.0, 0.0, 0.0))
    bad_edge = SphericalPairwisePoseEdge(bad_matches, edges[1].pose)
    with pytest.raises(ValueError, match="inconsistent bearings"):
        mapper.reconstruct(edges=(edges[0], bad_edge, edges[2]))


def test_options_reject_ambiguous_or_unsafe_values():
    with pytest.raises(ValueError, match="edge_admission"):
        SphericalGlobalMapperOptions(edge_admission="maybe")
    with pytest.raises(ValueError, match="at least 3"):
        SphericalGlobalMapperOptions(min_panoramas=2)
    with pytest.raises(ValueError, match="bundle_loss"):
        SphericalGlobalMapperOptions(bundle_loss="unknown")


@pytest.mark.parametrize(
    ("camera_count", "noise_deg", "outlier_fraction", "center_tolerance"),
    [(5, 0.0, 0.0, 2e-3), (10, 0.02, 0.2, 3e-2)],
)
def test_global_mapper_scales_to_looped_oracle_graphs(
    reconstruction_evidence,
    camera_count,
    noise_deg,
    outlier_fraction,
    center_tolerance,
):
    _, _, template_edges, _, _, _ = reconstruction_evidence
    edges, expected_rotations, expected_centers = _oracle_graph(
        template_edges[0].pose,
        camera_count,
        noise_deg=noise_deg,
        outlier_fraction=outlier_fraction,
    )
    mapper = SphericalGlobalMapper(
        options=SphericalGlobalMapperOptions(
            bundle_max_nfev=40,
            max_refinement_rounds=2,
        )
    )

    result = mapper.reconstruct(edges=edges)

    assert result.success, result.failure_reasons
    assert len(result.poses) == camera_count
    for panorama_id, expected in expected_rotations.items():
        assert _rotation_error_deg(result.pose(panorama_id).R, expected) < 0.1
    names = sorted(expected_centers)
    actual = np.stack([result.pose(name).center for name in names])
    target = np.stack([expected_centers[name] for name in names])
    aligned = _similarity_align(actual, target)
    assert np.sqrt(np.mean((aligned - target) ** 2)) < center_tolerance


def test_bad_rotation_edge_is_filtered_without_losing_component(
    reconstruction_evidence,
):
    _, _, template_edges, _, _, _ = reconstruction_evidence
    edges, _, _ = _oracle_graph(template_edges[0].pose, 5)
    bad_rotation = _rotation_exp(np.asarray((0.0, 0.0, math.radians(35.0))))
    bad_pose = replace(
        edges[-1].pose,
        rotation=bad_rotation @ edges[-1].pose.R,
        essential_matrix=_skew(edges[-1].pose.t) @ bad_rotation @ edges[-1].pose.R,
    )
    bad_edge = SphericalPairwisePoseEdge(edges[-1].matches, bad_pose)
    mapper = SphericalGlobalMapper(
        options=SphericalGlobalMapperOptions(bundle_max_nfev=30)
    )

    result = mapper.reconstruct(edges=(*edges[:-1], bad_edge))

    assert result.success
    assert bad_edge.pair in result.diagnostics.rotation_filtered_pairs


def test_track_union_rejects_same_panorama_conflict(reconstruction_evidence):
    mapper, _, edges, _, _, _ = reconstruction_evidence
    conflicting_matches = _matches_for_pair(
        "pano-a",
        "pano-c",
        {
            "pano-a": edges[2].matches.bearings_a,
            "pano-c": edges[2].matches.bearings_b,
        },
        len(edges[2].matches),
        extra_conflict=True,
    )
    conflict_mask = np.concatenate((edges[2].pose.inlier_mask, np.asarray((True,))))
    conflict_pose = replace(
        edges[2].pose,
        inlier_mask=conflict_mask,
        residuals_rad=np.zeros(len(conflict_mask)),
        num_inliers=int(conflict_mask.sum()),
        quality_report=replace(
            edges[2].pose.quality_report,
            num_correspondences=len(conflict_mask),
            num_inliers=int(conflict_mask.sum()),
        ),
    )

    result = mapper.reconstruct(
        edges=(
            edges[0],
            edges[1],
            SphericalPairwisePoseEdge(conflicting_matches, conflict_pose),
        )
    )

    assert result.success
    assert result.diagnostics.track_conflict_count == 1
    assert all(
        len({observation.panorama_id for observation in track.observations})
        == len(track.observations)
        for track in result.tracks
    )


def test_pure_rotation_tracks_fail_without_fabricated_geometry(
    reconstruction_evidence,
):
    _, _, template_edges, _, _, _ = reconstruction_evidence
    count = 24
    rng = np.random.default_rng(77)
    world_rays = rng.normal(size=(count, 3))
    world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True)
    ids = ("rotation-a", "rotation-b", "rotation-c")
    rotations = {
        ids[0]: np.eye(3),
        ids[1]: _rotation_exp(np.asarray((0.1, -0.05, 0.03))),
        ids[2]: _rotation_exp(np.asarray((-0.04, 0.08, 0.02))),
    }
    bearings = {name: world_rays @ rotation.T for name, rotation in rotations.items()}
    edges = []
    for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[0], ids[2])):
        matches = _matches_for_pair(a, b, bearings, count)
        relative = rotations[b] @ rotations[a].T
        translation = np.asarray((1.0, 0.0, 0.0))
        pose = replace(
            template_edges[0].pose,
            rotation=relative,
            translation_direction=translation,
            essential_matrix=_skew(translation) @ relative,
            inlier_mask=np.ones(count, dtype=bool),
            residuals_rad=np.zeros(count),
            num_inliers=count,
            quality_report=replace(
                template_edges[0].pose.quality_report,
                num_correspondences=count,
                num_inliers=count,
                accepted=True,
                rejection_reasons=(),
            ),
        )
        edges.append(SphericalPairwisePoseEdge(matches, pose))
    mapper = SphericalGlobalMapper(
        options=SphericalGlobalMapperOptions(bundle_max_nfev=20)
    )

    result = mapper.reconstruct(edges=edges)

    assert not result.success
    assert "all-tracks-rejected-during-refinement" in result.failure_reasons
    assert result.points_xyz.shape == (0, 3)
