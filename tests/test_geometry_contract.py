"""Executable checks for the documented geometry-v1 boundary contract.

Expected values here are derived from the equations and canonical axes in
``docs/geometry-v1.md``. They do not call private geometry helpers.
"""

from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai.geometry import (
    CUBE_FACE_BASES,
    CUBE_FACE_ORDER,
    GnomonicSpec,
    cubemap_to_equirectangular,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_to_equirectangular,
    rays_to_erp_pixels,
)


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _contract() -> dict:
    resource = files("panorai.geometry").joinpath("geometry-v1.yaml")
    # Some legacy tests replace ``sys.modules['yaml']`` at collection time.
    # Parse in an isolated interpreter so this test proves the real packaged
    # resource is valid PyYAML rather than accidentally exercising a stub.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,sys,yaml; print(json.dumps(yaml.safe_load(sys.stdin.read())))",
        ],
        input=resource.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def _reference_erp_rays(shape_hw: tuple[int, int]) -> np.ndarray:
    """Closed-form ERP oracle, independent from the package implementation."""

    height, width = shape_hw
    y, x = np.indices(shape_hw, dtype=np.float64)
    longitude = 2.0 * np.pi * (x + 0.5) / width - np.pi
    latitude = np.pi / 2.0 - np.pi * (y + 0.5) / height
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.sin(latitude),
            np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    )


def _reference_gnomonic_rays(spec: GnomonicSpec) -> np.ndarray:
    """Closed-form tangent-plane oracle for the canonical viewing basis."""

    height, width = spec.output_shape_hw
    x_limit = np.tan(np.deg2rad(spec.hfov_deg) / 2.0)
    y_limit = np.tan(np.deg2rad(spec.vfov_deg) / 2.0)
    x = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * x_limit
    y = (2.0 * (np.arange(height) + 0.5) / height - 1.0) * y_limit
    x, y = np.meshgrid(x, y)
    roll = np.deg2rad(spec.roll_deg)
    plane_x = np.cos(roll) * x - np.sin(roll) * y
    plane_y = np.sin(roll) * x + np.cos(roll) * y

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
    rays = forward + plane_x[..., None] * right - plane_y[..., None] * up
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def test_packaged_contract_and_public_document_agree_on_normative_rules() -> None:
    contract = _contract()
    document = (ROOT / "docs/geometry-v1.md").read_text(encoding="utf-8")

    assert contract["contract"] == "geometry-v1"
    assert contract["frame"] == {"x": "right", "y": "up", "z": "forward"}
    assert contract["erp"]["longitude_interval"] == "[-pi, pi)"
    assert contract["depth"] == "radial-range"
    assert contract["gnomonic"]["positive_roll"].startswith("clockwise")
    assert contract["sampling"]["nearest"]["index"] == "floor(coordinate+0.5)"
    bilinear = contract["sampling"]["bilinear"]
    assert bilinear["default_invalid_policy"] == "propagate"
    assert bilinear["invalid_policies"] == ["propagate", "renormalize"]
    assert bilinear["renormalize"]["outputs"] == [
        "validity_mask",
        "valid_weight",
    ]
    for phrase in (
        "Positive roll is a clockwise camera rotation",
        "the first face in",
        "support_mask` describes only geometric coverage",
        '`invalid_policy="renormalize"`',
    ):
        assert phrase in document


def test_exact_poles_and_seam_use_the_documented_representatives() -> None:
    # Provenance: direct evaluation of atan2/asin inverse ERP equations for the
    # signed canonical axes; float64 tolerance follows geometry-v1.
    shape = (6, 8)
    rays = np.array(
        (
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    projected = rays_to_erp_pixels(rays, shape)
    expected = np.array(((3.5, -0.5), (3.5, 5.5), (7.5, 2.5), (3.5, 2.5)))

    np.testing.assert_allclose(projected.pixels_xy, expected, atol=1e-12, rtol=0)
    assert projected.valid.tolist() == [True, True, True, True]
    np.testing.assert_allclose(projected.ranges, 1.0, atol=0, rtol=0)

    pole_coordinates = np.array(((-0.5, -0.5), (2.25, -0.5), (7.5, -0.5)))
    north = erp_pixels_to_rays(pole_coordinates, shape)
    np.testing.assert_allclose(
        north, np.tile((0.0, 1.0, 0.0), (len(pole_coordinates), 1)), atol=1e-12
    )


def test_rectangular_fov_and_positive_roll_match_closed_form_rays() -> None:
    # Provenance: the source panorama is the closed-form ray field above. A
    # high-resolution field makes bilinear resampling error much smaller than
    # the semantic roll/FOV signal while remaining independent of PanorAi.
    panorama = _reference_erp_rays((360, 720))
    spec = GnomonicSpec(
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        hfov_deg=90.0,
        vfov_deg=60.0,
        roll_deg=90.0,
        output_shape_hw=(3, 5),
    )
    actual = equirectangular_to_gnomonic(panorama, spec).data
    expected = _reference_gnomonic_rays(spec)

    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    # At +90 degrees, top-centre moves toward +right and right-centre toward
    # -up, which fixes the sign without relying on a verbal convention alone.
    assert actual[0, 2, 0] > 0.0
    assert abs(actual[0, 2, 1]) < 3e-5
    assert actual[1, -1, 1] < 0.0


def test_nearest_half_pixel_ties_and_erp_seam_wrap_are_exact() -> None:
    values = np.arange(4, dtype=np.int16)[None, :]
    front = equirectangular_to_gnomonic(
        values,
        GnomonicSpec(center_lon_deg=0.0, output_shape_hw=(1, 1)),
        interpolation="nearest",
    )
    seam = equirectangular_to_gnomonic(
        values,
        GnomonicSpec(center_lon_deg=-180.0, output_shape_hw=(1, 1)),
        interpolation="nearest",
    )

    # lon=0 maps to x=1.5: floor(1.5+0.5)=2. The exact seam maps to
    # x=3.5: floor(3.5+0.5)=4, followed by modulo 4 -> column zero.
    assert front.data.item() == 2
    assert seam.data.item() == 0
    assert front.data.dtype == values.dtype


def test_bilinear_wraps_erp_and_replicates_planar_borders() -> None:
    panorama = np.array(((0.0, 10.0, 20.0, 30.0),), dtype=np.float64)
    seam = equirectangular_to_gnomonic(
        panorama,
        GnomonicSpec(center_lon_deg=-180.0, output_shape_hw=(1, 1)),
    )
    # The seam stencil is the last/first pair at equal weight.
    assert seam.data.item() == pytest.approx(15.0, abs=1e-12)

    face = np.array(((2.0, 8.0),), dtype=np.float64)
    backprojected = gnomonic_to_equirectangular(
        face,
        GnomonicSpec(hfov_deg=90.0, vfov_deg=90.0, output_shape_hw=face.shape),
        (1, 4),
    )
    # ERP longitudes are -135, -45, +45, +135 degrees. The closed support
    # includes +/-45, which map to the face boundaries and replicate 2/8.
    np.testing.assert_array_equal(
        backprojected.support_mask, ((False, True, True, False),)
    )
    np.testing.assert_allclose(backprojected.data[0, 1:3], (2.0, 8.0), atol=1e-12)
    assert np.isnan(backprojected.data[0, (0, 3)]).all()


def test_strict_nan_propagation_is_current_safety_baseline() -> None:
    face = np.array(((1.0, np.nan),), dtype=np.float32)
    result = gnomonic_to_equirectangular(
        face,
        GnomonicSpec(output_shape_hw=face.shape),
        (1, 1),
    )

    assert result.support_mask.item()
    assert np.isnan(result.data.item())
    assert _contract()["sampling"]["bilinear"]["default_invalid_policy"] == (
        "propagate"
    )


def test_cubemap_bases_adjacency_and_edge_tie_order_are_explicit() -> None:
    contract = _contract()["cubemap"]
    expected_bases = {
        "front": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        "right": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        "back": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        "up": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        "down": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    }
    expected_adjacency = {
        "front": {
            "left": "left.right:same",
            "right": "right.left:same",
            "top": "up.bottom:same",
            "bottom": "down.top:same",
        },
        "right": {
            "left": "front.right:same",
            "right": "back.left:same",
            "top": "up.right:reversed",
            "bottom": "down.right:same",
        },
        "back": {
            "left": "right.right:same",
            "right": "left.left:same",
            "top": "up.top:reversed",
            "bottom": "down.bottom:reversed",
        },
        "left": {
            "left": "back.right:same",
            "right": "front.left:same",
            "top": "up.left:same",
            "bottom": "down.left:reversed",
        },
        "up": {
            "left": "left.top:same",
            "right": "right.top:reversed",
            "top": "back.top:reversed",
            "bottom": "front.top:same",
        },
        "down": {
            "left": "left.bottom:reversed",
            "right": "right.bottom:same",
            "top": "front.bottom:same",
            "bottom": "back.bottom:reversed",
        },
    }

    assert tuple(contract["order"]) == CUBE_FACE_ORDER
    assert {
        face: tuple(tuple(vector) for vector in CUBE_FACE_BASES[face])
        for face in CUBE_FACE_ORDER
    } == expected_bases
    assert contract["edge_adjacency"] == expected_adjacency

    faces = {
        face: np.full((2, 2), index, dtype=np.int8)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }
    result = cubemap_to_equirectangular(faces, (1, 4), interpolation="nearest")
    # Equatorial longitudes are -135, -45, +45, +135 degrees. Each is an
    # exact two-face tie, resolved by front,right,back,left,up,down order.
    np.testing.assert_array_equal(result.data, ((2, 0, 0, 1),))


def test_support_is_independent_from_validity_value_and_fill() -> None:
    face = np.full((1, 2), np.nan, dtype=np.float32)
    result = gnomonic_to_equirectangular(
        face,
        GnomonicSpec(output_shape_hw=face.shape),
        (1, 4),
        interpolation="nearest",
        fill_value=7.0,
    )

    np.testing.assert_array_equal(result.support_mask, ((False, True, True, False),))
    assert np.isnan(result.data[result.support_mask]).all()
    np.testing.assert_array_equal(result.data[~result.support_mask], (7.0, 7.0))

    black = np.zeros((4, 8, 3), dtype=np.float32)
    projected = equirectangular_to_gnomonic(
        black, GnomonicSpec(output_shape_hw=(3, 5))
    )
    assert projected.support_mask.all()
    assert not projected.data.any()


@pytest.mark.parametrize("dtype", (np.float16, np.float32, np.float64))
def test_numpy_bilinear_preserves_documented_floating_dtype(dtype) -> None:
    image = np.linspace(0, 1, 8 * 16, dtype=dtype).reshape(8, 16)
    result = equirectangular_to_gnomonic(
        image, GnomonicSpec(output_shape_hw=(5, 7))
    )
    assert result.data.dtype == image.dtype


def test_torch_cpu_bilinear_precision_policy_matches_packaged_tolerances() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(73)
    spec = GnomonicSpec(11.0, -67.0, 101.0, 63.0, 17.0, (19, 27))
    tolerances = _contract()["tolerances"]

    for numpy_dtype, torch_dtype, key in (
        (np.float64, torch.float64, "float64_bilinear"),
        (np.float32, torch.float32, "float32_bilinear"),
        (np.float16, torch.float16, "float16_bilinear"),
    ):
        numpy_image = rng.normal(size=(24, 48, 3)).astype(numpy_dtype)
        numpy_result = equirectangular_to_gnomonic(numpy_image, spec).data
        torch_image = torch.from_numpy(numpy_image).permute(2, 0, 1).to(torch_dtype)
        torch_result = (
            equirectangular_to_gnomonic(torch_image, spec)
            .data.permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        tolerance = tolerances[key]
        np.testing.assert_allclose(
            torch_result,
            numpy_result,
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        )
        assert torch_result.dtype == numpy_result.dtype

    bfloat_input = torch.linspace(0, 1, 3 * 24 * 48, dtype=torch.bfloat16).reshape(
        3, 24, 48
    )
    bfloat_result = equirectangular_to_gnomonic(bfloat_input, spec).data
    float_result = equirectangular_to_gnomonic(bfloat_input.float(), spec).data
    tolerance = tolerances["bfloat16_bilinear"]
    torch.testing.assert_close(
        bfloat_result.float(),
        float_result,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )
    assert bfloat_result.dtype == torch.bfloat16
