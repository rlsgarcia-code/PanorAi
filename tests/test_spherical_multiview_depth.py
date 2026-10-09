from __future__ import annotations

import math

import numpy as np
import pytest

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
