from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from benchmarks.spherical_multiview_depth.bidirectional import (
    BidirectionalCostVolumeOptions,
    BidirectionalSphericalCostVolume,
    bidirectional_cost_volume_batch,
)
from benchmarks.spherical_multiview_depth.continuous_residual import (
    ContinuousResidualOptions,
    interpolate_periodic_grid_to_native,
    solve_continuous_grid_residual,
)
from benchmarks.spherical_multiview_depth.grid_tangent import (
    GridTangentOptions,
    fuse_grid_tangent_proposals,
    infer_grid_tangent_proposal,
    propagate_fused_grid,
)
from benchmarks.spherical_multiview_depth.p74 import (
    P74_FROM_PANORAI,
    load_registered_pose,
    relative_pose_from_scene_transforms,
)
from benchmarks.spherical_multiview_depth.refinement import (
    DepthPrior,
    DifferentiableSphericalDepthRefiner,
    RefinementOptions,
    SourceView,
    project_points_to_grid,
    rays_for_pixels,
    warp_source,
)
from benchmarks.spherical_multiview_depth.tangent_seeds import (
    TangentDepthSeedOptions,
    propagate_tangent_depth_seeds,
    sample_erp_scalar,
    solve_tangent_depth_seeds,
)


torch = pytest.importorskip("torch")


def _ray_lattice(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    rows, columns = np.indices(shape_hw, dtype=np.float64)
    longitude = (columns + 0.5) / width * 2.0 * np.pi - np.pi
    latitude = np.pi / 2.0 - (rows + 0.5) / height * np.pi
    return np.stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        ),
        axis=-1,
    )


def _render_textured_sphere(
    camera_center: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    rays = _ray_lattice(shape_hw)
    center_dot_ray = np.sum(rays * camera_center, axis=-1)
    distance = -center_dot_ray + np.sqrt(
        center_dot_ray**2 + radius**2 - float(camera_center @ camera_center)
    )
    points = camera_center + distance[..., None] * rays
    rgb = np.stack(
        (
            0.5 + 0.22 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2]),
            0.5 + 0.25 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2]),
            0.5 + 0.25 * np.cos(6.0 * points[..., 2] + points[..., 0]),
        ),
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), distance.astype(np.float32)


def test_pixel_center_rays_and_projection_are_closed_form_inverses() -> None:
    shape = (9, 18)
    rows = torch.tensor([0, 4, 8], dtype=torch.long)
    columns = torch.tensor([0, 9, 17], dtype=torch.long)
    rays = rays_for_pixels(rows, columns, shape)
    grid = project_points_to_grid(rays * 3.0, shape)
    expected_x = 2.0 * (columns.to(torch.float32) + 0.5) / shape[1] - 1.0
    expected_y = 2.0 * (rows.to(torch.float32) + 0.5) / shape[0] - 1.0
    torch.testing.assert_close(grid[:, 0], expected_x, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(grid[:, 1], expected_y, atol=2e-6, rtol=0.0)


def test_identity_warp_is_exact_at_erp_seam_pixel_centres() -> None:
    shape = (8, 16)
    image = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(1, *shape)
    rows = torch.tensor([2, 2, 5, 5], dtype=torch.long)
    columns = torch.tensor([0, 15, 0, 15], dtype=torch.long)
    depth = torch.full((4,), 3.0)
    sampled = warp_source(
        image,
        rows,
        columns,
        depth,
        torch.eye(3),
        torch.zeros(3),
        shape,
    )
    torch.testing.assert_close(
        sampled[:, 0], image[0, rows, columns], atol=2e-5, rtol=0.0
    )


def test_warp_has_finite_nonzero_depth_gradient() -> None:
    shape = (8, 16)
    source = torch.linspace(0.0, 1.0, shape[0] * shape[1], dtype=torch.float64).reshape(
        1, *shape
    )
    rows = torch.tensor([3], dtype=torch.long)
    columns = torch.tensor([7], dtype=torch.long)
    depth = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    rotation = torch.eye(3, dtype=torch.float64)
    translation = torch.tensor([-0.3, 0.05, -0.1], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: warp_source(
            source, rows, columns, value, rotation, translation, shape
        ),
        (depth,),
        eps=1e-6,
        atol=2e-4,
        rtol=2e-3,
    )


def test_p74_registered_pose_conversion_composes_scene_points() -> None:
    angle = math.radians(20.0)
    target_rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source_rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    target_translation = np.asarray([2.0, -1.0, 0.5])
    source_translation = np.asarray([-0.4, 0.2, 1.3])
    pose = relative_pose_from_scene_transforms(
        target_rotation,
        target_translation,
        source_rotation,
        source_translation,
    )
    point_target_panorai = np.asarray([0.3, -0.7, 2.1])
    point_target_p74 = P74_FROM_PANORAI @ point_target_panorai
    point_scene = target_rotation @ point_target_p74 + target_translation
    expected_source_p74 = source_rotation.T @ (point_scene - source_translation)
    expected_source_panorai = P74_FROM_PANORAI.T @ expected_source_p74
    actual = (
        pose.rotation_source_from_target @ point_target_panorai
        + pose.translation_source_from_target_m
    )
    np.testing.assert_allclose(actual, expected_source_panorai, atol=1e-12)


@pytest.mark.parametrize("translation_key", ["translation", "translation_vector"])
def test_load_registered_pose_accepts_observed_and_legacy_translation_keys(
    tmp_path, translation_key: str
) -> None:
    target_path = tmp_path / "target.npz"
    source_path = tmp_path / "source.npz"
    target_members = {
        "rotation_matrix": np.eye(3),
        translation_key: np.asarray([1.0, 2.0, 3.0]),
    }
    source_members = {
        "rotation_matrix": np.eye(3),
        translation_key: np.asarray([2.0, 4.0, 6.0]),
    }
    np.savez(target_path, **target_members)
    np.savez(source_path, **source_members)
    pose = load_registered_pose(target_path, source_path)
    expected = P74_FROM_PANORAI.T @ np.asarray([-1.0, -2.0, -3.0])
    np.testing.assert_allclose(pose.translation_source_from_target_m, expected)


def test_multiview_refinement_reduces_analytic_depth_error() -> None:
    shape = (24, 48)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    center_a = np.asarray([0.35, 0.02, 0.10], dtype=np.float32)
    center_b = np.asarray([-0.28, -0.03, 0.16], dtype=np.float32)
    source_a, _ = _render_textured_sphere(center_a, shape)
    source_b, _ = _render_textured_sphere(center_b, shape)
    prior = np.clip(truth * 1.25, 0.3, 15.0)
    validity = np.ones(shape, dtype=bool)
    options = RefinementOptions(
        epochs=8,
        row_batch=6,
        learning_rate=0.03,
        prior_weight=0.10,
        pairwise_weight=0.05,
        min_texture_std=0.001,
        feature_window=3,
        seed=7,
    )
    result = DifferentiableSphericalDepthRefiner(options).refine(
        DepthPrior(prior, validity, provenance="analytic-25-percent-high"),
        target,
        (
            SourceView(source_a, np.eye(3), -center_a, view_id="a"),
            SourceView(source_b, np.eye(3), -center_b, view_id="b"),
        ),
    )
    refined = result.radial_range_m.cpu().numpy()
    prior_error = float(np.mean(np.abs(prior - truth) / truth))
    refined_error = float(np.mean(np.abs(refined - truth) / truth))
    assert refined_error < prior_error
    assert result.source_view_ids == ("a", "b")
    assert len(result.history) == options.epochs + 1
    assert np.isfinite(refined).all()


def test_bidirectional_cost_volume_reduces_analytic_depth_error() -> None:
    shape = (24, 48)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    sources = []
    for view_id, center in (
        ("a", np.asarray([0.35, 0.02, 0.10], dtype=np.float32)),
        ("b", np.asarray([-0.28, -0.03, 0.16], dtype=np.float32)),
    ):
        source, _ = _render_textured_sphere(center, shape)
        sources.append(SourceView(source, np.eye(3), -center, view_id=view_id))
    prior = truth * 1.25
    options = BidirectionalCostVolumeOptions(
        row_batch=6,
        feature_window=3,
        min_texture_std=0.001,
    )
    result = BidirectionalSphericalCostVolume(options).infer(
        DepthPrior(prior, np.ones(shape, dtype=bool), provenance="analytic-high"),
        target,
        sources,
    )
    prior_error = float(np.mean(np.abs(prior - truth) / truth))
    forward_error = float(
        np.mean(np.abs(result.forward_range_m.numpy() - truth) / truth)
    )
    reciprocal_error = float(
        np.mean(np.abs(result.reciprocal_range_m.numpy() - truth) / truth)
    )
    assert forward_error < prior_error
    assert reciprocal_error < prior_error
    assert result.source_view_ids == ("a", "b")
    assert all(row["reciprocal_pixels"] > 0 for row in result.source_summaries)


def test_bidirectional_batch_is_differentiable_with_respect_to_seed() -> None:
    shape = (12, 24)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    center = np.asarray([0.25, 0.01, 0.08], dtype=np.float32)
    source, _ = _render_textured_sphere(center, shape)
    target_tensor = torch.from_numpy(target).permute(2, 0, 1)
    source_tensor = torch.from_numpy(source).permute(2, 0, 1)
    from benchmarks.spherical_multiview_depth.refinement import photometric_features

    target_features, _ = photometric_features(target_tensor, window=3)
    source_features, _ = photometric_features(source_tensor, window=3)
    rows = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    columns = torch.tensor([8, 10, 12, 14], dtype=torch.long)
    seed = torch.from_numpy(truth[rows, columns] * 1.15).requires_grad_(True)
    result = bidirectional_cost_volume_batch(
        seed_range_m=seed,
        rows=rows,
        columns=columns,
        target_features_chw=target_features,
        source_features_chw=source_features,
        source_validity_hw=torch.ones(shape, dtype=torch.bool),
        target_validity_hw=torch.ones(shape, dtype=torch.bool),
        rotation_source_from_target=torch.eye(3),
        translation_source_from_target_m=torch.from_numpy(-center),
        shape_hw=shape,
        options=BidirectionalCostVolumeOptions(
            hypotheses=9,
            row_batch=4,
            feature_window=3,
            minimum_confidence=0.0,
        ),
    )
    gradient = torch.autograd.grad(result["forward_range_m"].sum(), seed)[0]
    assert torch.isfinite(gradient).all()
    assert torch.any(torch.abs(gradient) > 1e-6)


def test_tangent_matches_resolve_metric_depth_around_prior() -> None:
    shape = (48, 96)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    center = np.asarray([0.45, 0.04, 0.10], dtype=np.float64)
    rows = np.asarray([12, 16, 20, 24, 28, 32, 36])
    columns = np.asarray([8, 20, 34, 48, 62, 76, 88])
    target_bearings = _ray_lattice(shape)[rows, columns]
    points = target_bearings * truth[rows, columns, None]
    source_bearings = points - center
    source_bearings /= np.linalg.norm(source_bearings, axis=1, keepdims=True)
    prior = truth * 1.20
    options = TangentDepthSeedOptions(
        hypotheses=65,
        range_factor=1.5,
        maximum_reprojection_error_deg=0.05,
        maximum_ray_miss_m=0.01,
        minimum_parallax_deg=0.5,
    )
    result = solve_tangent_depth_seeds(
        prior,
        target_bearings,
        source_bearings,
        np.eye(3),
        -center,
        target_scale_deg=np.full(rows.size, 0.5),
        options=options,
    )
    assert result.accepted.all()
    prior_error = np.mean(
        np.abs(result.prior_range_m - truth[rows, columns]) / truth[rows, columns]
    )
    solved_error = np.mean(
        np.abs(result.solved_range_m - truth[rows, columns]) / truth[rows, columns]
    )
    assert solved_error < 1e-5
    assert solved_error < prior_error
    assert result.describe()["accepted_count"] == rows.size


def test_tangent_depth_seed_rejects_descriptor_outlier_and_range_boundary() -> None:
    shape = (24, 48)
    _rgb, truth = _render_textured_sphere(np.zeros(3), shape)
    rows = np.asarray([10, 12])
    columns = np.asarray([16, 28])
    target = _ray_lattice(shape)[rows, columns]
    center = np.asarray([0.5, 0.0, 0.0])
    points = target * truth[rows, columns, None]
    source = points - center
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    source[1] = np.asarray([0.0, 1.0, 0.0])
    prior = truth * 1.20
    result = solve_tangent_depth_seeds(
        prior,
        target,
        source,
        np.eye(3),
        -center,
        options=TangentDepthSeedOptions(
            maximum_reprojection_error_deg=0.1,
            maximum_ray_miss_m=0.01,
            minimum_parallax_deg=0.2,
        ),
    )
    np.testing.assert_array_equal(result.accepted, np.asarray([True, False]))


def test_tangent_seed_propagation_is_support_limited_and_improves_local_prior() -> None:
    shape = (48, 96)
    target_rgb, truth = _render_textured_sphere(np.zeros(3), shape)
    row = np.asarray([24])
    column = np.asarray([48])
    target = _ray_lattice(shape)[row, column]
    center = np.asarray([0.45, 0.02, 0.08])
    point = target * truth[row, column, None]
    source = point - center
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    prior = truth * 1.20
    options = TangentDepthSeedOptions(
        maximum_reprojection_error_deg=0.1,
        maximum_ray_miss_m=0.01,
        minimum_parallax_deg=0.2,
        minimum_propagation_radius_deg=4.0,
        maximum_propagation_radius_deg=4.0,
        minimum_propagation_weight=0.01,
    )
    sparse = solve_tangent_depth_seeds(
        prior,
        target,
        source,
        np.eye(3),
        -center,
        target_scale_deg=np.asarray([1.0]),
        options=options,
    )
    dense = propagate_tangent_depth_seeds(prior, target_rgb, (sparse,), options=options)
    assert dense.contributing_seed_count == 1
    assert 0 < dense.changed_mask.sum() < prior.size // 20
    assert np.array_equal(
        dense.radial_range_m[~dense.changed_mask], prior[~dense.changed_mask]
    )
    before = np.mean(
        np.abs(prior[dense.changed_mask] - truth[dense.changed_mask])
        / truth[dense.changed_mask]
    )
    after = np.mean(
        np.abs(dense.radial_range_m[dense.changed_mask] - truth[dense.changed_mask])
        / truth[dense.changed_mask]
    )
    assert after < before


def test_erp_scalar_sampling_wraps_longitude_without_pole_wrap() -> None:
    image = np.tile(np.arange(8, dtype=np.float64), (4, 1))
    rays = _ray_lattice((4, 8))[[1, 1], [0, 7]]
    values, pixels = sample_erp_scalar(image, rays)
    np.testing.assert_allclose(values, np.asarray([0.0, 7.0]), atol=1e-12)
    np.testing.assert_allclose(pixels[:, 1], np.asarray([1.0, 1.0]), atol=1e-12)


def test_grid_tangent_consensus_reduces_analytic_depth_error() -> None:
    shape = (32, 64)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    prior = truth * 1.20
    validity = np.ones(shape, dtype=bool)
    options = GridTangentOptions(
        stride_px=8,
        patch_samples=5,
        support_radius_deg=4.0,
        hypotheses=33,
        batch_size=32,
        minimum_patch_std=0.005,
        maximum_zncc_cost=0.55,
        minimum_cost_margin=0.00001,
        maximum_source_log_disagreement=0.08,
        minimum_propagation_weight=0.001,
    )
    proposals = []
    for view_id, center in (
        ("left", np.asarray([0.40, 0.02, 0.10], dtype=np.float32)),
        ("right", np.asarray([-0.35, -0.03, 0.12], dtype=np.float32)),
    ):
        source, _ = _render_textured_sphere(center, shape)
        proposals.append(
            infer_grid_tangent_proposal(
                prior,
                target,
                source,
                validity,
                validity,
                np.eye(3),
                -center,
                options=options,
                source_view_id=view_id,
            )
        )
    fused = fuse_grid_tangent_proposals(proposals)
    assert fused.consensus_accepted.sum() >= 4
    selected = fused.consensus_accepted
    gt = truth[fused.rows[selected], fused.columns[selected]]
    prior_error = np.mean(np.abs(fused.prior_range_m[selected] - gt) / gt)
    proposal_error = np.mean(np.abs(fused.consensus_range_m[selected] - gt) / gt)
    assert proposal_error < prior_error
    dense = propagate_fused_grid(prior, target, fused, mode="consensus")
    assert dense.changed_mask.any()
    before = np.mean(
        np.abs(prior[dense.changed_mask] - truth[dense.changed_mask])
        / truth[dense.changed_mask]
    )
    after = np.mean(
        np.abs(dense.radial_range_m[dense.changed_mask] - truth[dense.changed_mask])
        / truth[dense.changed_mask]
    )
    assert after < before


def test_grid_fusion_rejects_disagreeing_two_source_proposals() -> None:
    shape = (24, 48)
    target, truth = _render_textured_sphere(np.zeros(3), shape)
    validity = np.ones(shape, dtype=bool)
    center = np.asarray([0.4, 0.0, 0.1], dtype=np.float32)
    source, _ = _render_textured_sphere(center, shape)
    options = GridTangentOptions(
        stride_px=8,
        patch_samples=5,
        support_radius_deg=5.0,
        hypotheses=17,
        batch_size=32,
        minimum_patch_std=0.001,
        maximum_zncc_cost=0.8,
        minimum_cost_margin=0.0001,
        maximum_source_log_disagreement=0.02,
    )
    first = infer_grid_tangent_proposal(
        truth * 1.15,
        target,
        source,
        validity,
        validity,
        np.eye(3),
        -center,
        options=options,
        source_view_id="first",
    )
    accepted = np.flatnonzero(first.accepted)
    assert accepted.size
    conflicting_range = first.proposed_range_m.copy()
    conflicting_range[accepted[0]] *= 1.25
    second = replace(
        first,
        proposed_range_m=conflicting_range,
        source_view_id="second",
    )
    fused = fuse_grid_tangent_proposals((first, second))
    assert not fused.union_accepted[accepted[0]]
    assert not fused.consensus_accepted[accepted[0]]


def test_continuous_residual_improves_piecewise_depth_without_seam_artifact() -> None:
    shape = (48, 96)
    stride = 8
    rows, columns = np.meshgrid(
        np.arange(stride // 2, shape[0], stride),
        np.arange(stride // 2, shape[1], stride),
        indexing="ij",
    )
    rows = rows.ravel()
    columns = columns.ravel()
    prior = np.full(shape, 4.0, dtype=np.float32)
    truth = np.full(shape, 5.0, dtype=np.float32)
    truth[:, 32:64] = 3.2
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[..., 0] = 0.8
    rgb[:, 32:64, 0] = 0.1
    rgb[:, 32:64, 2] = 0.9
    accepted = np.zeros(rows.size, dtype=bool)
    accepted[(columns == 20) | (columns == 44) | (columns == 76)] = True
    proposed = np.full(rows.size, np.nan)
    proposed[accepted] = truth[rows[accepted], columns[accepted]]
    confidence = accepted.astype(np.float64)
    result = solve_continuous_grid_residual(
        prior,
        rgb,
        np.ones(shape, dtype=bool),
        rows,
        columns,
        proposed,
        accepted,
        confidence,
        options=ContinuousResidualOptions(
            seed_weight=96.0,
            anchor_weight=0.5,
            smoothness_weight=10.0,
            color_sigma=0.05,
        ),
    )
    before = np.mean(np.abs(prior - truth) / truth)
    after = np.mean(np.abs(result.radial_range_m - truth) / truth)
    assert after < before * 0.45
    assert result.diagnostics["cg_info"] == 0
    assert result.accepted_seed_count == int(accepted.sum())
    assert np.median(result.native_log_residual[:, :24]) > 0.1
    assert np.median(result.native_log_residual[:, 40:56]) < -0.1
    seam_difference = np.mean(
        np.abs(result.native_log_residual[:, 0] - result.native_log_residual[:, -1])
    )
    assert seam_difference < 0.01


def test_periodic_grid_interpolation_preserves_native_grid_samples() -> None:
    rows = np.asarray([2, 6, 10])
    columns = np.asarray([2, 6, 10, 14, 18, 22])
    grid = np.arange(rows.size * columns.size, dtype=np.float64).reshape(
        rows.size, columns.size
    )
    native = interpolate_periodic_grid_to_native(grid, rows, columns, (12, 24))
    np.testing.assert_allclose(native[np.ix_(rows, columns)], grid, atol=1e-6)
    expected_seam = 0.5 * grid[:, -1] + 0.5 * grid[:, 0]
    np.testing.assert_allclose(native[rows, 0], expected_seam, atol=1e-6)


def test_refinement_requires_declared_multiview_consensus() -> None:
    shape = (8, 16)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    prior = DepthPrior(np.full(shape, 3.0, dtype=np.float32), np.ones(shape, bool))
    source = SourceView(
        rgb,
        np.eye(3),
        np.asarray([0.1, 0.0, 0.0]),
        view_id="only-one",
    )
    with pytest.raises(ValueError, match="minimum_consistent_views"):
        DifferentiableSphericalDepthRefiner().refine(prior, rgb, (source,))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_range_m": 0.0}, "min_range_m"),
        ({"max_range_m": 0.2}, "max_range_m"),
        ({"epochs": 0}, "epochs"),
        ({"feature_window": 4}, "feature_window"),
        ({"photometric_clip": 0.0}, "photometric_clip"),
    ],
)
def test_options_reject_ambiguous_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RefinementOptions(**kwargs)
