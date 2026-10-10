from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.two_view_gaussian_splatting.export_gaussian_point_cloud import (  # noqa: E402
    backproject_erp_centres,
    fuse_voxels,
    transform_source_points_to_target,
    write_binary_rgb_ply,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth_feedback import (  # noqa: E402
    GaussianDepthAnchors,
    GaussianDepthFeedbackOptions,
    rasterize_gaussian_depth_anchors,
    read_gaussian_centres_ply,
    refine_depth_from_gaussian_anchors,
)
from benchmarks.two_view_gaussian_splatting.gaussian_depth import (  # noqa: E402
    GaussianDepthOptions,
    GaussianDepthStage,
    align_depth_scale_from_landmarks,
    erp_rays,
    optimize_gaussian_depth,
    optimize_hierarchical_gaussian_depth,
    project_erp,
    recommended_full_factor_gaussian_depth_stages,
    recommended_gaussian_depth_stages,
    render_spherical_gaussians,
    resize_periodic_field,
    transfer_log_range_correction,
    _compose_alpha_sorted,
)
from benchmarks.two_view_gaussian_splatting.p74 import (  # noqa: E402
    load_native_angular_rgb,
    load_native_registered_radial,
)
from benchmarks.two_view_gaussian_splatting.run_multiview_registered_surface_experiment import (  # noqa: E402
    _composite_ordered_fallback,
)
from benchmarks.two_view_gaussian_splatting.monocular_surface import (  # noqa: E402
    PriorConsistencyOptions,
    filter_prior_by_other_view,
    merge_prior_with_stereo,
)
from benchmarks.two_view_gaussian_splatting.run_stereo_surface_experiment import (  # noqa: E402
    _composite_primary_with_fallback,
    _composite_view_weighted,
)
from benchmarks.two_view_gaussian_splatting.rerender_two_surface_run import (  # noqa: E402
    _load_surface,
    _non_regression_checks,
)
from benchmarks.two_view_gaussian_splatting.surface_densification import (  # noqa: E402
    SurfaceDensificationOptions,
    densify_stereo_surface,
)
from benchmarks.two_view_gaussian_splatting.surface_quality import (  # noqa: E402
    SurfaceQualityOptions,
    assess_surface_quality,
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


def test_chunked_gaussian_render_matches_materialized_rasterizer():
    height, width = 18, 36
    rows = torch.linspace(0.0, 1.0, height)[:, None].expand(height, width)
    columns = torch.linspace(0.0, 1.0, width)[None].expand(height, width)
    rgb = torch.stack((columns, rows, columns * rows), dim=2)
    depth = 2.5 + 0.7 * columns
    valid = torch.ones((height, width), dtype=torch.bool)
    arguments = (
        rgb,
        depth,
        valid,
        torch.eye(3),
        torch.tensor((0.13, -0.02, 0.01)),
    )
    expected = render_spherical_gaussians(*arguments, sigma_px=0.75, radius_px=2)
    actual = render_spherical_gaussians(
        *arguments,
        sigma_px=0.75,
        radius_px=2,
        point_chunk_size=73,
    )
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6, equal_nan=True
        )


def test_front_to_back_alpha_compositing_orders_near_before_far():
    rendered, coverage, depth = _compose_alpha_sorted(
        torch.tensor((0, 0), dtype=torch.int64),
        torch.tensor((0.5, 0.5)),
        torch.tensor((3.0, 1.0)),
        torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))),
        shape_hw=(1, 1),
    )
    torch.testing.assert_close(
        rendered[0, 0], torch.tensor((2.0 / 3.0, 0.0, 1.0 / 3.0)), atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(coverage, torch.tensor(((0.75,),)))
    torch.testing.assert_close(depth, torch.tensor(((5.0 / 3.0,),)))


def test_chunked_alpha_renderer_is_bounded_and_respects_opacity_field():
    height, width = 10, 20
    rgb = torch.ones((height, width, 3))
    radial = torch.full((height, width), 3.0)
    valid = torch.ones((height, width), dtype=torch.bool)
    opacity = torch.ones((height, width))
    opacity[:, : width // 2] = 0.0
    rendered, coverage, depth = render_spherical_gaussians(
        rgb,
        radial,
        valid,
        torch.eye(3),
        torch.zeros(3),
        opacity_hw=opacity,
        sigma_px=0.45,
        radius_px=1,
        compositing_mode="alpha",
        alpha_depth_bins=4,
        point_chunk_size=31,
    )
    assert float(coverage[:, 2 : width // 2 - 2].max()) == 0.0
    assert float(coverage[:, width // 2 + 2 :].min()) > 0.5
    assert torch.all(rendered[:, 2 : width // 2 - 2] == 0.0)
    assert torch.all(torch.isnan(depth[:, 2 : width // 2 - 2]))


def test_surface_quality_removes_isolated_floater_without_filling_holes():
    radial = np.full((7, 12), 4.0, dtype=np.float32)
    valid = np.zeros_like(radial, dtype=bool)
    valid[3, 2:6] = True
    valid[1, 9] = True
    rgb = np.full((*radial.shape, 3), 0.5, dtype=np.float32)
    cleaned, opacity, report = assess_surface_quality(radial, valid, rgb)
    assert np.all(cleaned[3, 2:6])
    assert not cleaned[1, 9]
    assert not np.any(cleaned & ~valid)
    assert np.all(opacity[~cleaned] == 0.0)
    assert report["removed_floaters"] == 1


def test_surface_quality_treats_erp_longitude_as_periodic():
    radial = np.full((5, 8), 3.0, dtype=np.float32)
    valid = np.zeros_like(radial, dtype=bool)
    valid[2, 0] = True
    valid[2, -1] = True
    rgb = np.full((*radial.shape, 3), 0.5, dtype=np.float32)
    cleaned, opacity, _ = assess_surface_quality(radial, valid, rgb)
    assert cleaned[2, 0]
    assert cleaned[2, -1]
    assert opacity[2, 0] > 0.0
    assert opacity[2, -1] > 0.0


def test_surface_quality_downweights_textureless_and_clipped_evidence():
    height, width = 15, 24
    radial = np.full((height, width), 4.0, dtype=np.float32)
    valid = np.ones_like(radial, dtype=bool)
    rgb = np.full((height, width, 3), 0.5, dtype=np.float32)
    checker = (np.indices((height, width // 2)).sum(axis=0) % 2).astype(np.float32)
    rgb[:, width // 2 :, :] = 0.25 + 0.5 * checker[..., None]
    rgb[height // 2, width // 2 + 2] = 1.0
    _, opacity, _ = assess_surface_quality(
        radial,
        valid,
        rgb,
        options=SurfaceQualityOptions(texture_std_scale=0.1),
    )
    flat_mean = float(opacity[3:-3, 3 : width // 2 - 3].mean())
    textured_mean = float(opacity[3:-3, width // 2 + 3 : -3].mean())
    assert textured_mean > flat_mean
    assert opacity[height // 2, width // 2 + 2] < 1.0


def test_continuous_view_compositor_preserves_endpoints_and_fills_holes():
    shape = (2, 3)
    near_red = np.zeros((*shape, 3), dtype=np.float32)
    near_red[..., 0] = 1.0
    far_blue = np.zeros((*shape, 3), dtype=np.float32)
    far_blue[..., 2] = 1.0
    coverage_a = np.ones(shape, dtype=np.float32)
    coverage_b = np.ones(shape, dtype=np.float32)
    coverage_a[0, 0] = 0.0
    depth_a = np.full(shape, 2.0, dtype=np.float32)
    depth_b = np.full(shape, 3.0, dtype=np.float32)
    depth_a[0, 0] = np.nan
    layers = (
        (near_red, coverage_a, depth_a),
        (far_blue, coverage_b, depth_b),
    )
    at_a, coverage = _composite_view_weighted(layers, (1.0, 0.0), depth_tau_m=1e6)
    at_b, _ = _composite_view_weighted(layers, (0.0, 1.0), depth_tau_m=1e6)
    midpoint, _ = _composite_view_weighted(layers, (0.5, 0.5), depth_tau_m=1e6)
    np.testing.assert_allclose(at_a[1, 1], near_red[1, 1], atol=2e-6)
    np.testing.assert_allclose(at_b, far_blue, atol=2e-6)
    np.testing.assert_allclose(at_a[0, 0], far_blue[0, 0], atol=2e-6)
    assert 0.45 < float(midpoint[1, 1, 0]) < 0.55
    assert 0.45 < float(midpoint[1, 1, 2]) < 0.55
    assert coverage[0, 0] == 1.0


def test_continuous_view_compositor_depth_gate_suppresses_far_surface():
    shape = (1, 1)
    red = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    blue = np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    coverage = np.ones(shape, dtype=np.float32)
    rendered, _ = _composite_view_weighted(
        (
            (red, coverage, np.full(shape, 2.0, dtype=np.float32)),
            (blue, coverage, np.full(shape, 4.0, dtype=np.float32)),
        ),
        (0.5, 0.5),
        depth_tau_m=0.1,
    )
    assert float(rendered[0, 0, 0]) > 0.99
    assert float(rendered[0, 0, 2]) < 0.01


def test_renderer_non_regression_gate_rejects_training_view_degradation():
    checks = _non_regression_checks(
        {"rgb_l1": 0.0302, "covered_fraction": 0.8316},
        0.9512,
        {"rgb_l1": 0.0255, "covered_fraction": 0.8285},
        0.9486,
    )
    assert not checks["source_rgb_l1_not_worse_by_more_than_0_001"]
    assert checks["source_coverage_not_lower_by_more_than_0_005"]
    assert checks["midpoint_coverage_not_lower_by_more_than_0_005"]
    assert not all(checks.values())


def test_periodic_resize_and_log_range_transfer_preserve_constant_correction():
    aligned_small = np.linspace(2.0, 4.0, 32, dtype=np.float32).reshape(4, 8)
    optimized_small = aligned_small * 1.2
    aligned_large = np.linspace(1.5, 5.0, 128, dtype=np.float32).reshape(8, 16)
    transferred, correction = transfer_log_range_correction(
        aligned_small, optimized_small, aligned_large
    )
    np.testing.assert_allclose(correction, math.log(1.2), atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(transferred, aligned_large * 1.2, atol=2e-6, rtol=0.0)

    seam_field = np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    resized = resize_periodic_field(seam_field, (1, 16))
    assert float(resized[0, 0]) > 0.9
    assert float(resized[0, -1]) > 0.9
    assert abs(float(resized[0, 0] - resized[0, -1])) < 1e-6


def test_native_rgb_subpixel_integration_reduces_point_sampling_alias(tmp_path):
    source = np.zeros((24, 33, 3), dtype=np.uint8)
    source[:, ::2] = 255
    source[:, -1] = source[:, 0]
    path = tmp_path / "striped-native.png"
    assert cv2.imwrite(str(path), source)
    point_sampled, support = load_native_angular_rgb(path, (8, 16), antialias_samples=1)
    integrated, integrated_support = load_native_angular_rgb(
        path, (8, 16), antialias_samples=4
    )
    assert np.array_equal(support, integrated_support)
    point_mean = float(point_sampled[support, 0].mean())
    integrated_mean = float(integrated[support, 0].mean())
    assert abs(integrated_mean - 127.5) < abs(point_mean - 127.5)


def test_registered_xyz_regrid_returns_observed_radial_support(tmp_path):
    xyz = np.zeros((12, 25, 3), dtype=np.float64)
    xyz[..., 2] = 2.0
    path = tmp_path / "registered.npz"
    np.savez(path, xyz_image=xyz)
    radial, valid = load_native_registered_radial(path, (10, 20))
    assert radial.shape == (10, 20)
    assert valid.shape == radial.shape
    np.testing.assert_allclose(radial[valid], 2.0, atol=1e-6)
    assert np.all(np.isnan(radial[~valid]))
    assert np.all(valid[:8])
    assert not np.any(valid[8:])


def test_ordered_multiview_fallback_preserves_primary_and_fills_holes():
    shape = (2, 3)
    red = np.zeros((*shape, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros((*shape, 3), dtype=np.float32)
    green[..., 1] = 1.0
    blue = np.zeros((*shape, 3), dtype=np.float32)
    blue[..., 2] = 1.0
    primary_coverage = np.ones(shape, dtype=np.float32)
    primary_coverage[0, 0] = 0.0
    second_coverage = np.zeros(shape, dtype=np.float32)
    second_coverage[0, 0] = 1.0
    second_coverage[0, 1] = 0.1
    third_coverage = np.ones(shape, dtype=np.float32)
    depth = np.ones(shape, dtype=np.float32)
    rendered, coverage = _composite_ordered_fallback(
        (
            (red, primary_coverage, depth),
            (green, second_coverage, depth),
            (blue, third_coverage, depth),
        )
    )
    np.testing.assert_allclose(rendered[1, 1], red[1, 1])
    np.testing.assert_allclose(rendered[0, 0], green[0, 0])
    assert coverage.min() == 1.0


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
        projected_covariance_mode="jacobian",
    )
    assert float(coverage.min()) > 0.7
    assert float(torch.mean(torch.abs(rendered - rgb))) < 0.03
    torch.testing.assert_close(rendered_depth, depth, atol=2e-5, rtol=0.0)


def test_surface_aligned_covariance_scales_receive_gradients():
    height, width = 12, 24
    rows = torch.linspace(0.0, 1.0, height)[:, None].expand(height, width)
    columns = torch.linspace(0.0, 1.0, width)[None].expand(height, width)
    rgb = torch.stack((columns, rows, columns * rows), dim=2)
    depth = 3.0 + 0.4 * columns
    covariance = torch.zeros((height, width, 2), requires_grad=True)
    rendered, coverage, _ = render_spherical_gaussians(
        rgb,
        depth,
        torch.ones((height, width), dtype=torch.bool),
        torch.eye(3),
        torch.tensor((0.2, 0.0, 0.0)),
        sigma_px=0.7,
        radius_px=2,
        surface_aligned=True,
        covariance_log_scales_hw2=covariance,
    )
    loss = coverage.mean() + 0.1 * rendered[..., 0].mean()
    loss.backward()
    assert covariance.grad is not None
    assert torch.all(torch.isfinite(covariance.grad))
    assert int(torch.count_nonzero(covariance.grad)) > 0


def test_recommended_hierarchy_is_coarse_to_fine_and_covariance_last():
    stages = recommended_gaussian_depth_stages((1, 2, 3, 4))
    assert [stage.correction_shape_hw for stage in stages] == [
        (16, 32),
        (32, 64),
        (64, 128),
        (64, 128),
    ]
    assert [stage.iterations for stage in stages] == [1, 2, 3, 4]
    assert not stages[0].surface_aligned
    assert stages[-2].surface_aligned
    assert stages[-2].optimize_radial
    assert not stages[-2].optimize_covariance
    assert stages[-1].surface_aligned
    assert not stages[-1].optimize_radial
    assert stages[-1].optimize_covariance


def test_full_factor_hierarchy_unlocks_one_factor_at_a_time():
    stages = recommended_full_factor_gaussian_depth_stages((1, 2, 3, 4, 5, 6))
    assert [stage.iterations for stage in stages] == [1, 2, 3, 4, 5, 6]
    assert [stage.optimize_covariance for stage in stages] == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert [stage.optimize_opacity for stage in stages] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert [stage.optimize_color for stage in stages] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert all(not stage.optimize_radial for stage in stages[3:])


def test_hierarchical_optimizer_reports_each_unlocked_factor():
    height, width = 12, 24
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None]
    rgb[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    radial = np.full((height, width), 3.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    rays = erp_rays((height, width)).numpy()
    landmarks = np.stack((rays[2, 3] * 3.0, rays[6, 12] * 3.0, rays[9, 20] * 3.0))
    options = GaussianDepthOptions(
        gaussian_radius_px=2,
        stages=(
            GaussianDepthStage((2, 4), 1, 0.01),
            GaussianDepthStage((4, 8), 1, 0.01),
            GaussianDepthStage(
                (6, 12),
                1,
                0.01,
                surface_aligned=True,
            ),
            GaussianDepthStage(
                (6, 12),
                1,
                0.01,
                optimize_radial=False,
                surface_aligned=True,
                optimize_covariance=True,
                covariance_learning_rate=0.005,
            ),
        ),
    )
    result = optimize_hierarchical_gaussian_depth(
        rgb,
        radial,
        valid,
        rgb,
        valid,
        np.eye(3),
        np.zeros(3),
        landmarks,
        options=options,
    )
    assert result.radial_m.shape == radial.shape
    assert len(result.stage_radial_m) == 4
    assert all(
        stage_depth.shape == radial.shape for stage_depth in result.stage_radial_m
    )
    np.testing.assert_array_equal(result.stage_radial_m[-2], result.stage_radial_m[-1])
    np.testing.assert_array_equal(result.stage_radial_m[-1], result.radial_m)
    assert result.covariance_log_scales_hw2 is not None
    assert result.covariance_log_scales_hw2.shape == (height, width, 2)
    assert len(result.report["stages"]) == 4
    assert [entry["radial_parameter_count"] for entry in result.report["stages"]] == [
        8,
        32,
        72,
        0,
    ]
    assert result.report["stages"][-1]["covariance_parameter_count"] == 144
    assert result.report["optimized_factors"] == {
        "radial_means": True,
        "surface_aligned_covariance_scales": True,
        "tangential_means": False,
        "rgb_dc_residual": False,
        "higher_order_spherical_harmonics": False,
        "opacity": False,
        "pose": False,
        "split_or_prune": False,
    }
    assert np.all(np.isfinite(result.radial_m))


def test_full_factor_optimizer_returns_bounded_opacity_and_color():
    height, width = 8, 16
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(16, 224, width, dtype=np.uint8)[None]
    radial = np.full((height, width), 3.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    rays = erp_rays((height, width)).numpy()
    landmarks = np.stack((rays[1, 2] * 3.0, rays[4, 8] * 3.0, rays[6, 13] * 3.0))
    stages = (
        GaussianDepthStage((2, 4), 1, 0.01),
        GaussianDepthStage(
            (2, 4),
            1,
            0.01,
            optimize_radial=False,
            surface_aligned=True,
            optimize_covariance=True,
        ),
        GaussianDepthStage(
            (2, 4),
            1,
            0.01,
            optimize_radial=False,
            surface_aligned=True,
            optimize_opacity=True,
        ),
        GaussianDepthStage(
            (2, 4),
            1,
            0.01,
            optimize_radial=False,
            surface_aligned=True,
            optimize_color=True,
        ),
    )
    result = optimize_hierarchical_gaussian_depth(
        rgb,
        radial,
        valid,
        rgb,
        valid,
        np.eye(3),
        np.zeros(3),
        landmarks,
        options=GaussianDepthOptions(gaussian_radius_px=2, stages=stages),
    )
    assert result.opacity_hw is not None
    assert result.opacity_hw.shape == radial.shape
    assert np.all((result.opacity_hw > 0.0) & (result.opacity_hw < 1.0))
    assert result.color_hwc is not None
    assert result.color_hwc.shape == rgb.shape
    assert np.all((result.color_hwc >= 0.0) & (result.color_hwc <= 1.0))
    assert result.report["optimized_factors"]["opacity"]
    assert result.report["optimized_factors"]["rgb_dc_residual"]
    assert not result.report["optimized_factors"]["higher_order_spherical_harmonics"]


def test_legacy_single_grid_optimizer_contract_remains_available():
    height, width = 8, 16
    rgb = np.full((height, width, 3), 127, dtype=np.uint8)
    radial = np.full((height, width), 3.0, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    rays = erp_rays((height, width)).numpy()
    landmarks = np.stack((rays[1, 2] * 3.0, rays[4, 8] * 3.0, rays[6, 13] * 3.0))
    optimized, report = optimize_gaussian_depth(
        rgb,
        radial,
        valid,
        rgb,
        valid,
        np.eye(3),
        np.zeros(3),
        landmarks,
        options=GaussianDepthOptions(iterations=0),
    )
    np.testing.assert_array_equal(optimized, radial)
    assert len(report["stages"]) == 1
    assert report["stages"][0]["correction_shape_hw"] == [16, 32]
    assert report["optimized_factors"]["radial_means"]
    assert not report["optimized_factors"]["surface_aligned_covariance_scales"]


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


def test_frozen_two_surface_loader_preserves_view_specific_geometry(tmp_path):
    target = np.asarray([[2.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    source = np.asarray([[5.0, 6.0], [np.nan, 7.0]], dtype=np.float32)
    np.save(tmp_path / "stereo-surface-radial-m.npy", target)
    np.save(tmp_path / "stereo-surface-valid.npy", np.isfinite(target))
    np.save(tmp_path / "source-stereo-surface-radial-m.npy", source)
    np.save(tmp_path / "source-stereo-surface-valid.npy", np.isfinite(source))
    loaded_target = _load_surface(tmp_path, source=False)
    loaded_source = _load_surface(tmp_path, source=True)
    np.testing.assert_array_equal(loaded_target.valid, np.isfinite(target))
    np.testing.assert_array_equal(loaded_source.valid, np.isfinite(source))
    np.testing.assert_allclose(
        loaded_target.radial[np.isfinite(target)], target[np.isfinite(target)]
    )
    np.testing.assert_allclose(
        loaded_source.radial[np.isfinite(source)], source[np.isfinite(source)]
    )


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
