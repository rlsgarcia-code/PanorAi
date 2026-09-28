#!/usr/bin/env python3
"""Run independent geometry-v1 properties against source or an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_ROOT = REPOSITORY_ROOT / "tests/fixtures/geometry/v1"


def _load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _reference_erp_rays(pixels_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
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


def _reference_grid(shape_hw: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape_hw, dtype=np.float64)
    return _reference_erp_rays(np.stack((x, y), axis=-1), shape_hw)


def _reference_support(spec, shape_hw: tuple[int, int]) -> np.ndarray:
    rays = _reference_grid(shape_hw)
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
    coordinates = {
        "top": (parameter, -1.0),
        "right": (1.0, parameter),
        "bottom": (parameter, 1.0),
        "left": (-1.0, parameter),
    }
    x, y = coordinates[side]
    forward, right, up = (np.asarray(value, dtype=np.float64) for value in basis)
    ray = forward + x * right - y * up
    return ray / np.linalg.norm(ray)


def _assert_installed_origin(
    package_file: str,
    *,
    source_root: Path,
    package_version: str,
    expected_version: str | None,
) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise AssertionError(f"panorai resolved inside source checkout: {origin}")
    distribution_version = version("panorai")
    assert package_version == distribution_version
    if expected_version is not None:
        assert distribution_version == expected_version
    return origin


def _verify_checksums(root: Path, manifest: dict) -> None:
    for name, metadata in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == metadata["sha256"], f"checksum mismatch: {name}"


def _verify_external_reference(root: Path, external: dict) -> None:
    import py360convert

    expected_version = external["reference"]["version"]
    assert py360convert.__version__ == expected_version
    assert version("py360convert") == expected_version
    source_shape = tuple(external["source"]["shape_hw"])
    rays = _reference_grid(source_shape)
    signal = rays[..., 0] + 2.0 * rays[..., 1] + 3.0 * rays[..., 2]
    tolerance = external["absolute_tolerance"]
    for case in external["gnomonic_center_cases"]:
        output = py360convert.e2p(
            signal,
            (case["hfov_deg"], case["vfov_deg"]),
            case["center_lon_deg"],
            case["center_lat_deg"],
            tuple(case["output_shape_hw"]),
            mode="bilinear",
        )
        center = tuple(value // 2 for value in case["output_shape_hw"])
        assert abs(float(output[center]) - case["py360convert_value"]) <= tolerance
    cube_case = external["cubemap_horizontal_face_centers"]
    cube = py360convert.e2c(
        signal,
        face_w=cube_case["face_width"],
        mode="bilinear",
        cube_format="dict",
    )
    for face, key in {"front": "F", "right": "R", "back": "B", "left": "L"}.items():
        assert abs(float(cube[key][1, 1]) - cube_case[face]) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-root", type=Path, default=DEFAULT_FIXTURES_ROOT)
    parser.add_argument("--source-checkout", action="store_true")
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--expected-version")
    parser.add_argument("--require-torch-gradcheck", action="store_true")
    parser.add_argument("--verify-py360convert", action="store_true")
    args = parser.parse_args()
    if args.source_checkout and args.require_installed:
        parser.error("--source-checkout and --require-installed are mutually exclusive")
    if args.source_checkout and str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    import panorai
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

    if args.require_installed:
        origin = _assert_installed_origin(
            panorai.__file__,
            source_root=args.source_root,
            package_version=panorai.__version__,
            expected_version=args.expected_version,
        )
    else:
        origin = Path(panorai.__file__).resolve()

    root = args.fixtures_root.resolve()
    manifest = _load(root, "manifest.json")
    _verify_checksums(root, manifest)
    properties = _load(root, "properties.json")
    external = _load(root, "external.json")

    round_trip = properties["round_trip"]
    shape = tuple(round_trip["erp_shape_hw"])
    height, width = shape
    rng = np.random.default_rng(properties["random_seed"])
    pixels = np.column_stack(
        (
            rng.uniform(-3.0 * width, 4.0 * width, round_trip["random_samples"]),
            rng.uniform(-0.5 + 1e-9, height - 0.5 - 1e-9, round_trip["random_samples"]),
        )
    )
    expected_rays = _reference_erp_rays(pixels, shape)
    actual_rays = erp_pixels_to_rays(pixels, shape)
    np.testing.assert_allclose(actual_rays, expected_rays, atol=1e-14, rtol=1e-14)
    projected = rays_to_erp_pixels(actual_rays, shape)
    reconstructed = _reference_erp_rays(projected.pixels_xy, shape)
    angular_error = np.arccos(
        np.clip(np.sum(actual_rays * reconstructed, axis=-1), -1.0, 1.0)
    )
    max_angular_error = float(np.max(angular_error))
    assert max_angular_error <= round_trip["maximum_angular_error_rad"]

    for case in properties["gnomonic_support_cases"]:
        values = dict(case)
        output_shape = tuple(values.pop("erp_shape_hw"))
        values["output_shape_hw"] = tuple(values["output_shape_hw"])
        spec = GnomonicSpec(**values)
        result = gnomonic_to_equirectangular(
            np.ones(spec.output_shape_hw, dtype=np.float64),
            spec,
            output_shape,
            interpolation="nearest",
            fill_value=-7.0,
        )
        np.testing.assert_array_equal(
            result.support_mask, _reference_support(spec, output_shape)
        )

    expected_bases = properties["cube_face_bases"]
    assert tuple(CUBE_FACE_ORDER) == tuple(properties["cube_face_order"])
    for face in CUBE_FACE_ORDER:
        np.testing.assert_array_equal(CUBE_FACE_BASES[face], expected_bases[face])
    for face, side, neighbour, neighbour_side, reverse in properties["directed_edges"]:
        for parameter in (-1.0, -0.37, 0.0, 0.41, 1.0):
            first = _cube_edge_ray(expected_bases[face], side, parameter)
            neighbour_parameter = -parameter if reverse else parameter
            second = _cube_edge_ray(
                expected_bases[neighbour], neighbour_side, neighbour_parameter
            )
            np.testing.assert_allclose(first, second, atol=1e-15, rtol=0.0)
    for vertex in properties["vertices_xyz"]:
        direction = np.asarray(vertex, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        incident = 0
        for face in CUBE_FACE_ORDER:
            forward, right, up = (
                np.asarray(value, dtype=np.float64) for value in expected_bases[face]
            )
            denominator = direction @ forward
            if np.isclose(denominator, 1.0 / np.sqrt(3.0)):
                incident += 1
                local_x = (direction @ right) / denominator
                local_y = -(direction @ up) / denominator
                reconstructed = forward + local_x * right - local_y * up
                reconstructed /= np.linalg.norm(reconstructed)
                np.testing.assert_allclose(
                    reconstructed, direction, atol=1e-15, rtol=0.0
                )
        assert incident == 3

    tie_case = properties["exact_equatorial_edge_winners"]
    faces = {
        face: np.full((3, 3), index, dtype=np.uint8)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }
    tied = cubemap_to_equirectangular(
        faces, tuple(tie_case["erp_shape_hw"]), interpolation="nearest"
    ).data
    np.testing.assert_array_equal(tied[1], tie_case["middle_row_face_indices"])

    smooth = properties["smooth_cube_round_trip"]
    smooth_shape = tuple(smooth["erp_shape_hw"])
    signal = _reference_grid(smooth_shape)
    cube = equirectangular_to_cubemap(
        signal, tuple(smooth["face_shape_hw"]), interpolation="bilinear"
    )
    restored = cubemap_to_equirectangular(
        {face: result.data for face, result in cube.items()},
        smooth_shape,
        interpolation="bilinear",
    ).data
    point_error = np.linalg.norm(restored - signal, axis=-1)
    rmse = float(np.sqrt(np.mean(np.square(point_error))))
    maximum_error = float(np.max(point_error))
    assert rmse <= smooth["maximum_rmse"]
    assert maximum_error <= smooth["maximum_point_error"]

    external_shape = tuple(external["source"]["shape_hw"])
    external_rays = _reference_grid(external_shape)
    external_signal = (
        external_rays[..., 0]
        + 2.0 * external_rays[..., 1]
        + 3.0 * external_rays[..., 2]
    )
    external_tolerance = external["absolute_tolerance"]
    from panorai.geometry import equirectangular_to_gnomonic

    for case in external["gnomonic_center_cases"]:
        spec = GnomonicSpec(
            center_lat_deg=case["center_lat_deg"],
            center_lon_deg=case["center_lon_deg"],
            hfov_deg=case["hfov_deg"],
            vfov_deg=case["vfov_deg"],
            output_shape_hw=tuple(case["output_shape_hw"]),
        )
        data = equirectangular_to_gnomonic(external_signal, spec).data
        center = tuple(value // 2 for value in spec.output_shape_hw)
        assert abs(float(data[center]) - case["py360convert_value"]) <= external_tolerance

    # Independent closed-form normalized bilinear check. The centre ray maps
    # to x=1.5,y=0.5; two valid neighbours contribute 0.25 each.
    normalized_source = np.array(
        ((2.0, 2.0, 4.0, 4.0), (2.0, 2.0, 4.0, 4.0)), dtype=np.float64
    )
    normalized_validity = np.array(
        ((False, True, True, False), (False, False, False, False)), dtype=bool
    )
    normalized = equirectangular_to_gnomonic(
        normalized_source,
        GnomonicSpec(output_shape_hw=(1, 1)),
        invalid_policy="renormalize",
        validity_mask=normalized_validity,
        min_valid_weight=0.5,
    )
    np.testing.assert_allclose(normalized.data, [[3.0]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(normalized.valid_weight, [[0.5]], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(normalized.validity_mask, [[True]])

    if args.require_torch_gradcheck:
        import torch

        source = torch.linspace(0.1, 0.9, 5 * 9, dtype=torch.float64).reshape(5, 9)
        source.requires_grad_(True)
        spec = GnomonicSpec(11.0, -23.0, 71.0, 49.0, 17.0, (3, 5))

        def operation(image):
            from panorai.geometry import equirectangular_to_gnomonic

            return equirectangular_to_gnomonic(image, spec).data

        assert torch.autograd.gradcheck(
            operation, (source,), eps=1e-6, atol=1e-5, rtol=1e-4
        )

        normalized_source_torch = torch.tensor(
            normalized_source, dtype=torch.float64, requires_grad=True
        )
        normalized_validity_torch = torch.tensor(
            normalized_validity, dtype=torch.bool
        )

        def normalized_operation(image):
            from panorai.geometry import equirectangular_to_gnomonic

            return equirectangular_to_gnomonic(
                image,
                GnomonicSpec(output_shape_hw=(1, 1)),
                invalid_policy="renormalize",
                validity_mask=normalized_validity_torch,
                min_valid_weight=0.5,
            ).data

        assert torch.autograd.gradcheck(
            normalized_operation,
            (normalized_source_torch,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        normalized_operation(normalized_source_torch).sum().backward()
        expected_gradient = np.zeros_like(normalized_source)
        expected_gradient[0, 1:3] = 0.5
        np.testing.assert_allclose(
            normalized_source_torch.grad.detach().numpy(),
            expected_gradient,
            atol=1e-12,
            rtol=0,
        )

    if args.verify_py360convert:
        _verify_external_reference(root, external)

    report = {
        "contract": manifest["contract"],
        "version": panorai.__version__,
        "origin": str(origin),
        "random_samples": round_trip["random_samples"],
        "maximum_angular_error_rad": max_angular_error,
        "smooth_round_trip_rmse": rmse,
        "smooth_round_trip_maximum_error": maximum_error,
        "torch_gradcheck": args.require_torch_gradcheck,
        "py360convert_verified": args.verify_py360convert,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
