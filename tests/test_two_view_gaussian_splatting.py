from __future__ import annotations

import math

import numpy as np
import torch

from benchmarks.two_view_gaussian_splatting.export_gaussian_point_cloud import (
    backproject_erp_centres,
    fuse_voxels,
    transform_source_points_to_target,
    write_binary_rgb_ply,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth_feedback import (
    GaussianDepthAnchors,
    GaussianDepthFeedbackOptions,
    rasterize_gaussian_depth_anchors,
    read_gaussian_centres_ply,
    refine_depth_from_gaussian_anchors,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth import (
    align_depth_scale_from_landmarks,
    erp_rays,
    project_erp,
    render_spherical_gaussians,
)
from benchmarks.two_view_gaussian_splatting.monocular_surface import (
    PriorConsistencyOptions,
    filter_prior_by_other_view,
    merge_prior_with_stereo,
)
from benchmarks.two_view_gaussian_splatting.run_stereo_surface_experiment import (
    _composite_primary_with_fallback,
)
from benchmarks.two_view_gaussian_splatting.surface_densification import (
    SurfaceDensificationOptions,
    densify_stereo_surface,
)


def test_erp_projection_round_trips_pixel_center_rays():
    rays = erp_rays((12, 24)).reshape(-1, 3)
    u, v, radial = project_erp(rays * 2.5, (12, 24))
    expected_v, expected_u = torch.meshgrid(
        torch.arange(12, dtype=torch.float32),
        torch.arange(24, dtype=torch.float32),
        indexing="ij",
    )
    torch.testing.assert_close(u, expected_u.reshape(-1), atol=2e-5, rtol=0.0)
    torch.testing.assert_close(v, expected_v.reshape(-1), atol=2e-5, rtol=0.0)
    torch.testing.assert_close(
        radial, torch.full_like(radial, 2.5), atol=2e-6, rtol=0.0
    )


def test_landmarks_recover_positive_global_depth_scale():
    height, width = 24, 48
    true_range = 4.0
    prior = np.full((height, width), true_range / 2.5, dtype=np.float32)
    rays = erp_rays((height, width)).numpy()
    sample_pixels = [(4, 3), (7, 14), (11, 27), (16, 39), (19, 45)]
    points = np.stack([rays[y, x] * true_range for y, x in sample_pixels])
    aligned, report = align_depth_scale_from_landmarks(prior, points)
    np.testing.assert_allclose(aligned, true_range, rtol=1e-6, atol=1e-6)
    assert math.isclose(float(report["scale"]), 2.5, rel_tol=1e-6)
    assert int(report["landmark_count"]) == len(sample_pixels)


def test_identity_gaussian_render_preserves_supported_color():
    height, width = 20, 40
    rows = torch.linspace(0.0, 1.0, height)[:, None].expand(height, width)
    columns = torch.linspace(0.0, 1.0, width)[None].expand(height, width)
    rgb = torch.stack((columns, rows, 0.25 * torch.ones_like(rows)), dim=2)
    depth = torch.full((height, width), 3.0)
    valid = torch.ones((height, width), dtype=torch.bool)
    rendered, coverage, rendered_depth = render_spherical_gaussians(
        rgb,
        depth,
        valid,
        torch.eye(3),
        torch.zeros(3),
        sigma_px=0.55,
        radius_px=1,
        opacity=0.98,
        occlusion_tau_m=0.05,
    )
    assert float(coverage.min()) > 0.7
    assert float(torch.mean(torch.abs(rendered - rgb))) < 0.025
    torch.testing.assert_close(rendered_depth, depth, atol=2e-5, rtol=0.0)


def test_surface_aligned_ewa_preserves_identity_surface():
    height, width = 20, 40
    rows = torch.linspace(0.0, 1.0, height)[:, None].expand(height, width)
    columns = torch.linspace(0.0, 1.0, width)[None].expand(height, width)
    rgb = torch.stack((columns, rows, 0.5 * torch.ones_like(rows)), dim=2)
    depth = torch.full((height, width), 4.0)
    valid = torch.ones((height, width), dtype=torch.bool)
    rendered, coverage, rendered_depth = render_spherical_gaussians(
        rgb,
        depth,
        valid,
        torch.eye(3),
        torch.zeros(3),
        sigma_px=0.6,
        radius_px=3,
        opacity=0.98,
        surface_aligned=True,
    )
    assert float(coverage.min()) > 0.7
    assert float(torch.mean(torch.abs(rendered - rgb))) < 0.03
    torch.testing.assert_close(rendered_depth, depth, atol=2e-5, rtol=0.0)


def test_densification_preserves_hard_seeds_and_color_boundary():
    height, width = 10, 20
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, : width // 2] = (20, 40, 60)
    rgb[:, width // 2 :] = (220, 210, 190)
    radial = np.full((height, width), np.nan, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    radial[5, 5] = 2.0
    radial[5, 15] = 6.0
    confidence[5, 5] = 0.9
    confidence[5, 15] = 0.9
    dense, valid, _, report = densify_stereo_surface(
        radial,
        confidence,
        rgb,
        np.ones((height, width), dtype=bool),
        options=SurfaceDensificationOptions(
            maximum_distance_px=4.0,
            iterations=4,
            color_sigma=0.02,
        ),
    )
    assert dense[5, 5] == radial[5, 5]
    assert dense[5, 15] == radial[5, 15]
    assert np.all(dense[valid & (np.indices(valid.shape)[1] < width // 2)] < 3.0)
    assert np.all(dense[valid & (np.indices(valid.shape)[1] >= width // 2)] > 5.0)
    assert int(report["seed_pixels"]) == 2
    assert int(report["dense_pixels"]) > 2


def test_densification_wraps_only_at_erp_longitude_seam():
    radial = np.full((7, 12), np.nan, dtype=np.float32)
    confidence = np.zeros_like(radial)
    radial[3, 0] = 3.0
    confidence[3, 0] = 1.0
    dense, valid, _, _ = densify_stereo_surface(
        radial,
        confidence,
        np.zeros((7, 12, 3), dtype=np.uint8),
        np.ones((7, 12), dtype=bool),
        options=SurfaceDensificationOptions(maximum_distance_px=2.0, iterations=2),
    )
    assert valid[3, -1]
    assert math.isclose(float(dense[3, -1]), 3.0, rel_tol=1e-6)
    assert not valid[0, 0]


def test_densification_uses_consistent_low_confidence_candidate():
    radial = np.full((5, 9), np.nan, dtype=np.float32)
    confidence = np.zeros_like(radial)
    radial[2, 4] = 2.0
    confidence[2, 4] = 0.9
    radial[2, 5] = 2.2
    confidence[2, 5] = 0.05
    dense, valid, _, report = densify_stereo_surface(
        radial,
        confidence,
        np.zeros((5, 9, 3), dtype=np.uint8),
        np.ones((5, 9), dtype=bool),
        options=SurfaceDensificationOptions(iterations=1, maximum_distance_px=1.5),
    )
    assert valid[2, 5]
    assert 2.0 < float(dense[2, 5]) < 2.2
    assert int(report["accepted_low_confidence_candidates"]) == 1


def test_identical_priors_pass_two_view_consistency_filter():
    height, width = 8, 16
    radial = np.full((height, width), 3.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    rgb = np.full((height, width, 3), 127, dtype=np.uint8)
    filtered, accepted, confidence, report = filter_prior_by_other_view(
        radial,
        valid,
        radial,
        valid,
        rgb,
        rgb,
        valid,
        valid,
        np.eye(3),
        np.zeros(3),
        options=PriorConsistencyOptions(
            maximum_log_depth_disagreement=0.01,
            maximum_rgb_l1=0.01,
        ),
    )
    assert np.all(accepted)
    np.testing.assert_array_equal(filtered, radial)
    assert np.all(confidence > 0.99)
    assert int(report["accepted_pixels"]) == height * width


def test_stereo_values_take_priority_when_merging_prior():
    stereo = np.full((3, 4), np.nan, dtype=np.float32)
    stereo[1, 2] = 2.0
    stereo_valid = np.isfinite(stereo)
    prior = np.full((3, 4), 5.0, dtype=np.float32)
    prior_valid = np.ones((3, 4), dtype=bool)
    merged, valid, confidence = merge_prior_with_stereo(
        stereo,
        stereo_valid,
        stereo_valid.astype(np.float32),
        prior,
        prior_valid,
        np.full((3, 4), 0.4, dtype=np.float32),
    )
    assert np.all(valid)
    assert merged[1, 2] == 2.0
    assert confidence[1, 2] == 1.0
    assert np.all(merged[~stereo_valid] == 5.0)


def test_primary_composite_uses_fallback_only_for_holes():
    shape = (2, 3)
    primary_color = np.zeros((*shape, 3), dtype=np.float32)
    primary_color[..., 0] = 1.0
    fallback_color = np.zeros((*shape, 3), dtype=np.float32)
    fallback_color[..., 1] = 1.0
    primary_coverage = np.ones(shape, dtype=np.float32)
    primary_coverage[0, 0] = 0.0
    fallback_coverage = np.ones(shape, dtype=np.float32)
    depth = np.ones(shape, dtype=np.float32)
    rendered, coverage = _composite_primary_with_fallback(
        (primary_color, primary_coverage, depth),
        (fallback_color, fallback_coverage, depth),
    )
    np.testing.assert_array_equal(rendered[0, 0], (0.0, 1.0, 0.0))
    np.testing.assert_array_equal(rendered[1, 1], (1.0, 0.0, 0.0))
    assert np.all(coverage == 1.0)


def test_point_cloud_backprojection_preserves_radial_range():
    radial = np.full((4, 8), 3.25, dtype=np.float32)
    valid = np.ones_like(radial, dtype=bool)
    points = backproject_erp_centres(radial, valid)
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1), 3.25, rtol=1e-6, atol=1e-6
    )


def test_source_to_target_point_transform_inverts_pose():
    angle = 0.3
    rotation = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    translation = np.array((0.4, -0.1, 0.2))
    target = np.array(((1.0, 2.0, 3.0), (-2.0, 0.5, 4.0)))
    source = target @ rotation.T + translation
    recovered = transform_source_points_to_target(source, rotation, translation)
    np.testing.assert_allclose(recovered, target, rtol=1e-6, atol=1e-6)


def test_voxel_fusion_averages_rgb_and_combines_view_bits():
    points = np.array(((0.001, 0.0, 0.0), (0.009, 0.0, 0.0), (0.02, 0.0, 0.0)))
    colors = np.array(((10, 20, 30), (30, 40, 50), (100, 110, 120)), dtype=np.uint8)
    fused, fused_colors, observations, view_bits = fuse_voxels(
        points,
        colors,
        np.array((1, 2, 1), dtype=np.uint8),
        voxel_size_m=0.01,
    )
    assert fused.shape == (2, 3)
    np.testing.assert_array_equal(fused_colors[0], (20, 30, 40))
    np.testing.assert_array_equal(observations, (2, 1))
    np.testing.assert_array_equal(view_bits, (3, 1))


def test_gaussian_cloud_ply_round_trip_preserves_feedback_fields(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.array(((1.0, 2.0, 3.0), (-1.0, 0.5, 4.0)), dtype=np.float32)
    colors = np.array(((10, 20, 30), (200, 210, 220)), dtype=np.uint8)
    observations = np.array((2, 19), dtype=np.uint16)
    view_mask = np.array((3, 2), dtype=np.uint8)
    write_binary_rgb_ply(path, points, colors, observations, view_mask)
    loaded = read_gaussian_centres_ply(path)
    np.testing.assert_array_equal(loaded[0], points)
    np.testing.assert_array_equal(loaded[1], colors)
    np.testing.assert_array_equal(loaded[2], observations)
    np.testing.assert_array_equal(loaded[3], view_mask)


def test_gaussian_depth_rasterizer_keeps_nearest_visible_surface():
    target_rgb = np.full((8, 16, 3), 128, dtype=np.uint8)
    points = np.array(((0.0, 0.0, 2.0), (0.0, 0.0, 5.0)), dtype=np.float32)
    colors = np.full((2, 3), 128, dtype=np.uint8)
    anchors = rasterize_gaussian_depth_anchors(
        points,
        colors,
        np.ones(2, dtype=np.uint16),
        np.full(2, 3, dtype=np.uint8),
        target_rgb,
        options=GaussianDepthFeedbackOptions(
            gaussian_sigma_px=0.6,
            gaussian_radius_px=1,
            occlusion_tolerance_m=0.01,
            iterations=1,
        ),
    )
    valid = np.isfinite(anchors.radial_m)
    assert np.any(valid)
    np.testing.assert_allclose(anchors.radial_m[valid], 2.0, atol=1e-6)
    assert np.all(anchors.view_mask[valid] == 3)


def test_two_view_gaussian_anchor_has_higher_confidence_than_target_only():
    target_rgb = np.full((8, 16, 3), 100, dtype=np.uint8)
    rays = erp_rays((8, 16)).numpy()
    points = np.stack((rays[3, 3] * 3.0, rays[3, 11] * 3.0))
    colors = np.full((2, 3), 100, dtype=np.uint8)
    anchors = rasterize_gaussian_depth_anchors(
        points,
        colors,
        np.full(2, 4, dtype=np.uint16),
        np.array((1, 3), dtype=np.uint8),
        target_rgb,
        options=GaussianDepthFeedbackOptions(
            gaussian_sigma_px=0.5,
            gaussian_radius_px=0,
            iterations=1,
        ),
    )
    target_confidence = anchors.confidence[anchors.view_mask == 1]
    both_confidence = anchors.confidence[anchors.view_mask == 3]
    assert target_confidence.size == 1
    assert both_confidence.size == 1
    assert float(both_confidence[0]) > 10.0 * float(target_confidence[0])


def test_gaussian_feedback_improves_constant_biased_prior():
    prior = np.full((16, 32), 2.0, dtype=np.float32)
    solve_shape = (8, 16)
    anchors = GaussianDepthAnchors(
        radial_m=np.full(solve_shape, 3.0, dtype=np.float32),
        confidence=np.ones(solve_shape, dtype=np.float32),
        view_mask=np.full(solve_shape, 3, dtype=np.uint8),
        point_count=np.ones(solve_shape, dtype=np.uint32),
    )
    result = refine_depth_from_gaussian_anchors(
        prior,
        np.full((*solve_shape, 3), 127, dtype=np.uint8),
        anchors,
        options=GaussianDepthFeedbackOptions(iterations=40),
    )
    assert float(np.mean(np.abs(result.refined_radial_m - 3.0))) < 0.03
    assert float(np.mean(result.refined_confidence)) > 0.9


def test_gaussian_feedback_does_not_propagate_across_strong_color_edge():
    prior = np.full((8, 16), 2.0, dtype=np.float32)
    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    rgb[:, 8:] = 255
    anchor_radial = np.full((8, 16), np.nan, dtype=np.float32)
    anchor_confidence = np.zeros((8, 16), dtype=np.float32)
    anchor_radial[:, 3] = 3.0
    anchor_confidence[:, 3] = 1.0
    anchors = GaussianDepthAnchors(
        radial_m=anchor_radial,
        confidence=anchor_confidence,
        view_mask=np.where(anchor_confidence > 0.0, 3, 0).astype(np.uint8),
        point_count=(anchor_confidence > 0.0).astype(np.uint32),
    )
    result = refine_depth_from_gaussian_anchors(
        prior,
        rgb,
        anchors,
        options=GaussianDepthFeedbackOptions(
            iterations=40,
            color_sigma=0.01,
            prior_residual_weight=0.05,
        ),
    )
    assert float(np.mean(result.refined_radial_m[:, 3])) > 2.8
    assert float(np.max(np.abs(result.refined_radial_m[:, 10:] - 2.0))) < 0.02
