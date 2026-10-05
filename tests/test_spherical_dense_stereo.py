from __future__ import annotations

import numpy as np
import pytest

from panorai.geometry import erp_pixels_to_rays
from panorai.stereo import (
    SphericalDenseStereo,
    SphericalStereoOptions,
    colorize_spherical_range,
    estimate_spherical_range,
    render_spherical_stereo_result,
)


def _ray_lattice(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    y, x = np.indices(shape_hw, dtype=np.float32)
    return erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)


def _render_textured_sphere(
    camera_center: np.ndarray,
    shape_hw: tuple[int, int],
    radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    rays = _ray_lattice(shape_hw)
    center_dot_ray = np.sum(rays * camera_center, axis=-1)
    distance = -center_dot_ray + np.sqrt(
        center_dot_ray * center_dot_ray
        + radius * radius
        - float(camera_center @ camera_center)
    )
    points = camera_center + distance[..., None] * rays
    image = np.stack(
        (
            0.5
            + 0.2 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2])
            + 0.2 * np.cos(7.0 * points[..., 1]),
            0.5
            + 0.25 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2])
            + 0.15 * np.cos(8.0 * points[..., 0]),
            0.5
            + 0.25 * np.cos(6.0 * points[..., 2] + points[..., 0])
            + 0.15 * np.sin(9.0 * points[..., 1]),
        ),
        axis=-1,
    )
    return np.clip(image, 0.0, 1.0).astype(np.float32), distance.astype(np.float32)


def test_direct_spherical_stereo_recovers_analytic_radial_range() -> None:
    shape = (64, 128)
    center_b_in_a = np.asarray((0.35, 0.03, 0.12), dtype=np.float32)
    first, reference_range = _render_textured_sphere(np.zeros(3), shape)
    second, _ = _render_textured_sphere(center_b_in_a, shape)
    options = SphericalStereoOptions(
        min_range=2.5,
        max_range=5.5,
        num_hypotheses=96,
        window_size=5,
        pole_margin_fraction=0.05,
        min_texture_std=0.005,
        min_confidence=0.002,
        max_matching_cost=0.8,
    )

    result = SphericalDenseStereo(options).estimate(
        first, second, np.eye(3), -center_b_in_a
    )

    valid = result.validity_mask
    relative_error = (
        np.abs(result.range[valid] - reference_range[valid]) / reference_range[valid]
    )
    assert valid.mean() > 0.65
    assert float(np.mean(relative_error)) < 0.035
    assert float(np.median(relative_error)) < 0.02
    assert result.quantity == "radial_range"
    assert result.describe()["stability"] == "experimental"
    assert not result.range.flags.writeable

    colored = colorize_spherical_range(result.range, result.validity_mask)
    assert colored.shape == (*shape, 3)
    assert colored.dtype == np.uint8
    assert np.all(colored[~valid] == 0)

    panel = render_spherical_stereo_result(
        first,
        result,
        target_erp=second,
        reference_range=reference_range,
    )
    assert panel.dtype == np.uint8
    assert panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[0] > shape[0]
    assert panel.shape[1] > shape[1]


def test_seam_is_not_treated_as_an_invalid_image_border() -> None:
    shape = (32, 64)
    center_b_in_a = np.asarray((0.20, 0.0, 0.10), dtype=np.float32)
    first, reference_range = _render_textured_sphere(np.zeros(3), shape)
    second, _ = _render_textured_sphere(center_b_in_a, shape)
    result = estimate_spherical_range(
        first,
        second,
        np.eye(3),
        -center_b_in_a,
        options=SphericalStereoOptions(
            min_range=2.5,
            max_range=5.5,
            num_hypotheses=64,
            window_size=5,
            pole_margin_fraction=0.1,
            min_texture_std=0.002,
            min_confidence=0.0,
            max_matching_cost=0.9,
        ),
    )
    seam = result.validity_mask[:, (0, -1)]
    assert seam.mean() > 0.45
    error = (
        np.abs(result.range[:, (0, -1)][seam] - reference_range[:, (0, -1)][seam])
        / reference_range[:, (0, -1)][seam]
    )
    assert float(np.median(error)) < 0.07


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_range": 0.0}, "min_range"),
        ({"min_range": 2.0, "max_range": 1.0}, "max_range"),
        ({"num_hypotheses": 2}, "num_hypotheses"),
        ({"window_size": 4}, "window_size"),
        ({"intensity_weight": 0.8, "gradient_weight": 0.3}, "sum to 1"),
    ],
)
def test_options_reject_ambiguous_or_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SphericalStereoOptions(**kwargs)


def test_inputs_and_pose_are_validated() -> None:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    options = SphericalStereoOptions(num_hypotheses=3)
    with pytest.raises(ValueError, match="same HW shape"):
        estimate_spherical_range(
            image, image[:8], np.eye(3), np.ones(3), options=options
        )
    with pytest.raises(ValueError, match="orthonormal"):
        estimate_spherical_range(
            image, image, np.ones((3, 3)), np.ones(3), options=options
        )
    with pytest.raises(ValueError, match="non-zero metric scale"):
        estimate_spherical_range(image, image, np.eye(3), np.zeros(3), options=options)


def test_range_visualization_rejects_ambiguous_inputs() -> None:
    range_map = np.ones((8, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="HW"):
        colorize_spherical_range(range_map[..., None])
    with pytest.raises(TypeError, match="boolean"):
        colorize_spherical_range(range_map, np.ones_like(range_map))
    with pytest.raises(ValueError, match="strictly increasing"):
        colorize_spherical_range(range_map, value_range=(2.0, 1.0))
    with pytest.raises(ValueError, match="colormap"):
        colorize_spherical_range(range_map, colormap="not-a-map")
