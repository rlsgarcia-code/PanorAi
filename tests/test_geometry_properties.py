"""Independent properties and boundary cases for geometry-v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panorai.geometry import (
    CUBE_FACE_BASES,
    CUBE_FACE_ORDER,
    GnomonicSpec,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    erp_pixels_to_rays,
    gnomonic_to_equirectangular,
    rays_to_erp_pixels,
)


FIXTURE = Path(__file__).parent / "fixtures/geometry/v1/properties.json"
EXTERNAL_FIXTURE = Path(__file__).parent / "fixtures/geometry/v1/external.json"


def _cases() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _external_cases() -> dict:
    return json.loads(EXTERNAL_FIXTURE.read_text(encoding="utf-8"))


def _reference_erp_rays(pixels_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Direct spherical equations, independent from PanorAi helpers."""

    height, width = shape_hw
    longitude = 2.0 * np.pi * (pixels_xy[..., 0] + 0.5) / width - np.pi
    latitude = np.pi / 2.0 - np.pi * (pixels_xy[..., 1] + 0.5) / height
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.sin(latitude),
            np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    )


def _reference_erp_grid(shape_hw: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape_hw, dtype=np.float64)
    return _reference_erp_rays(np.stack((x, y), axis=-1), shape_hw)


def _reference_basis(spec: GnomonicSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = np.deg2rad(spec.center_lat_deg)
    longitude = np.deg2rad(spec.center_lon_deg)
    forward = np.array(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        )
    )
    right = np.array((np.cos(longitude), 0.0, -np.sin(longitude)))
    up = np.array(
        (
            -np.sin(longitude) * np.sin(latitude),
            np.cos(latitude),
            -np.cos(longitude) * np.sin(latitude),
        )
    )
    return forward, right, up


def _reference_support(spec: GnomonicSpec, shape_hw: tuple[int, int]) -> np.ndarray:
    rays = _reference_erp_grid(shape_hw)
    forward, right, up = _reference_basis(spec)
    denominator = rays @ forward
    unrolled_x = (rays @ right) / np.where(denominator != 0.0, denominator, 1.0)
    unrolled_y = -(rays @ up) / np.where(denominator != 0.0, denominator, 1.0)
    roll = np.deg2rad(spec.roll_deg)
    plane_x = np.cos(roll) * unrolled_x + np.sin(roll) * unrolled_y
    plane_y = -np.sin(roll) * unrolled_x + np.cos(roll) * unrolled_y
    x_limit = np.tan(np.deg2rad(spec.hfov_deg) / 2.0)
    y_limit = np.tan(np.deg2rad(spec.vfov_deg) / 2.0)
    return (
        (denominator > 0.0)
        & (np.abs(plane_x) <= x_limit + 1e-12)
        & (np.abs(plane_y) <= y_limit + 1e-12)
    )


def _cube_edge_ray(basis: list[list[int]], side: str, parameter: float) -> np.ndarray:
    if side == "top":
        x, y = parameter, -1.0
    elif side == "right":
        x, y = 1.0, parameter
    elif side == "bottom":
        x, y = parameter, 1.0
    elif side == "left":
        x, y = -1.0, parameter
    else:  # pragma: no cover - fixture schema guard
        raise AssertionError(f"unknown side: {side}")
    forward, right, up = (np.asarray(value, dtype=np.float64) for value in basis)
    ray = forward + x * right - y * up
    return ray / np.linalg.norm(ray)


def test_random_and_adversarial_pixel_ray_round_trips() -> None:
    cases = _cases()
    policy = cases["round_trip"]
    shape = tuple(policy["erp_shape_hw"])
    height, width = shape
    rng = np.random.default_rng(cases["random_seed"])
    random_pixels = np.column_stack(
        (
            rng.uniform(-3.0 * width, 4.0 * width, policy["random_samples"]),
            rng.uniform(-0.5 + 1e-9, height - 0.5 - 1e-9, policy["random_samples"]),
        )
    )
    adversarial = np.array(
        [
            [-0.5, (height - 1) / 2],
            [width - 0.5, (height - 1) / 2],
            [0.5, -0.5 + 1e-12],
            [width - 1.5, height - 0.5 - 1e-12],
        ],
        dtype=np.float64,
    )
    pixels = np.concatenate((random_pixels, adversarial))

    expected_rays = _reference_erp_rays(pixels, shape)
    actual_rays = erp_pixels_to_rays(pixels, shape)
    # Equivalent operation order can differ by a few float64 ulps after the
    # deliberately large longitude wraps used above.
    np.testing.assert_allclose(actual_rays, expected_rays, atol=1e-14, rtol=1e-14)

    projected = rays_to_erp_pixels(actual_rays, shape)
    reconstructed = _reference_erp_rays(projected.pixels_xy, shape)
    dots = np.sum(actual_rays * reconstructed, axis=-1)
    angular_error = np.arccos(np.clip(dots, -1.0, 1.0))
    assert projected.valid.all()
    assert float(np.max(angular_error)) <= policy["maximum_angular_error_rad"]

    expected_x = np.mod(pixels[:, 0], width)
    x_delta = np.abs(projected.pixels_xy[:, 0] - expected_x)
    circular_x_error = np.minimum(x_delta, width - x_delta)
    assert float(np.max(circular_x_error)) <= policy["maximum_interior_pixel_error"]
    assert float(np.max(np.abs(projected.pixels_xy[:, 1] - pixels[:, 1]))) <= policy[
        "maximum_interior_pixel_error"
    ]


@pytest.mark.parametrize("case", _cases()["gnomonic_support_cases"])
def test_gnomonic_backprojection_support_matches_closed_form(case: dict) -> None:
    values = dict(case)
    erp_shape = tuple(values.pop("erp_shape_hw"))
    values["output_shape_hw"] = tuple(values["output_shape_hw"])
    spec = GnomonicSpec(**values)
    face = np.ones(spec.output_shape_hw, dtype=np.float64)

    actual = gnomonic_to_equirectangular(
        face, spec, erp_shape, interpolation="nearest", fill_value=-7.0
    )
    expected_support = _reference_support(spec, erp_shape)

    np.testing.assert_array_equal(actual.support_mask, expected_support)
    assert np.all(actual.data[~expected_support] == -7.0)
    assert np.all(actual.data[expected_support] == 1.0)


def test_all_cubemap_edges_and_vertices_match_hand_derived_bases() -> None:
    cases = _cases()
    expected_bases = cases["cube_face_bases"]
    assert tuple(CUBE_FACE_ORDER) == tuple(cases["cube_face_order"])
    for face in CUBE_FACE_ORDER:
        np.testing.assert_array_equal(CUBE_FACE_BASES[face], expected_bases[face])

    for face, side, neighbour, neighbour_side, reverse in cases["directed_edges"]:
        for parameter in (-1.0, -0.37, 0.0, 0.41, 1.0):
            left = _cube_edge_ray(expected_bases[face], side, parameter)
            neighbour_parameter = -parameter if reverse else parameter
            right = _cube_edge_ray(
                expected_bases[neighbour], neighbour_side, neighbour_parameter
            )
            np.testing.assert_allclose(left, right, atol=1e-15, rtol=0.0)

    for vertex in cases["vertices_xyz"]:
        direction = np.asarray(vertex, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        incident = []
        for face in CUBE_FACE_ORDER:
            forward, right, up = (
                np.asarray(value, dtype=np.float64) for value in expected_bases[face]
            )
            denominator = direction @ forward
            if np.isclose(denominator, 1.0 / np.sqrt(3.0)):
                incident.append(face)
                local_x = (direction @ right) / denominator
                local_y = -(direction @ up) / denominator
                assert local_x in {-1.0, 1.0}
                assert local_y in {-1.0, 1.0}
                reconstructed = forward + local_x * right - local_y * up
                reconstructed /= np.linalg.norm(reconstructed)
                np.testing.assert_allclose(reconstructed, direction, atol=1e-15, rtol=0.0)
        assert len(incident) == 3
        scores = [
            direction @ np.asarray(expected_bases[face][0], dtype=np.float64)
            for face in CUBE_FACE_ORDER
        ]
        expected_winner = CUBE_FACE_ORDER[int(np.argmax(scores))]
        assert expected_winner == incident[0]


def test_exact_equatorial_cubemap_edges_follow_canonical_tie_order() -> None:
    cases = _cases()["exact_equatorial_edge_winners"]
    faces = {
        face: np.full((3, 3), index, dtype=np.uint8)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }
    result = cubemap_to_equirectangular(
        faces, tuple(cases["erp_shape_hw"]), interpolation="nearest"
    )
    np.testing.assert_array_equal(result.data[1], cases["middle_row_face_indices"])


def test_smooth_cubemap_round_trip_has_measured_error_bounds() -> None:
    policy = _cases()["smooth_cube_round_trip"]
    erp_shape = tuple(policy["erp_shape_hw"])
    signal = _reference_erp_grid(erp_shape)
    cubemap = equirectangular_to_cubemap(
        signal, tuple(policy["face_shape_hw"]), interpolation="bilinear"
    )
    reconstructed = cubemap_to_equirectangular(
        {face: result.data for face, result in cubemap.items()},
        erp_shape,
        interpolation="bilinear",
    ).data
    point_error = np.linalg.norm(reconstructed - signal, axis=-1)
    rmse = float(np.sqrt(np.mean(np.square(point_error))))
    maximum = float(np.max(point_error))

    assert rmse <= policy["maximum_rmse"], (rmse, maximum)
    assert maximum <= policy["maximum_point_error"], (rmse, maximum)


def test_shared_samples_match_pinned_py360convert_reference() -> None:
    """Compare rays shared despite the libraries' different output lattices."""

    reference = _external_cases()
    source_shape = tuple(reference["source"]["shape_hw"])
    source = _reference_erp_grid(source_shape)
    signal = source[..., 0] + 2.0 * source[..., 1] + 3.0 * source[..., 2]
    tolerance = reference["absolute_tolerance"]

    from panorai.geometry import equirectangular_to_gnomonic

    for case in reference["gnomonic_center_cases"]:
        spec = GnomonicSpec(
            center_lat_deg=case["center_lat_deg"],
            center_lon_deg=case["center_lon_deg"],
            hfov_deg=case["hfov_deg"],
            vfov_deg=case["vfov_deg"],
            output_shape_hw=tuple(case["output_shape_hw"]),
        )
        result = equirectangular_to_gnomonic(signal, spec).data
        center = tuple(value // 2 for value in spec.output_shape_hw)
        assert abs(float(result[center]) - case["py360convert_value"]) <= tolerance

    cubemap_case = reference["cubemap_horizontal_face_centers"]
    cubemap = equirectangular_to_cubemap(signal, cubemap_case["face_width"])
    for face in ("front", "right", "back", "left"):
        assert abs(float(cubemap[face].data[1, 1]) - cubemap_case[face]) <= tolerance


def test_torch_bilinear_input_gradients_pass_gradcheck() -> None:
    torch = pytest.importorskip("torch")
    source = torch.linspace(0.1, 0.9, 5 * 9, dtype=torch.float64).reshape(5, 9)
    source.requires_grad_(True)
    spec = GnomonicSpec(
        center_lat_deg=11.0,
        center_lon_deg=-23.0,
        hfov_deg=71.0,
        vfov_deg=49.0,
        roll_deg=17.0,
        output_shape_hw=(3, 5),
    )

    def project(image):
        from panorai.geometry import equirectangular_to_gnomonic

        return equirectangular_to_gnomonic(image, spec).data

    assert torch.autograd.gradcheck(project, (source,), eps=1e-6, atol=1e-5, rtol=1e-4)

    face = torch.linspace(0.2, 1.0, 3 * 5, dtype=torch.float64).reshape(3, 5)
    face.requires_grad_(True)

    def back_project(image):
        return gnomonic_to_equirectangular(
            image, spec, (5, 9), fill_value=0.0
        ).data

    assert torch.autograd.gradcheck(
        back_project, (face,), eps=1e-6, atol=1e-5, rtol=1e-4
    )
