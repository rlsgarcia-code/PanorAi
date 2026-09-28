from __future__ import annotations

import numpy as np
import pytest

from panorai.geometry import (
    CUBE_FACE_ORDER,
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_to_equirectangular,
)


def test_gnomonic_project_result_composes_directly_with_back_project() -> None:
    y, x = np.indices((18, 36), dtype=np.float64)
    panorama = np.stack((x / 36.0, y / 18.0), axis=-1)
    projector = GnomonicProjector(
        GnomonicSpec(12.0, -33.0, 92.0, 61.0, 17.0, (11, 15)),
        interpolation="bilinear",
        fill_value=-3.0,
    )

    projected = projector.project(panorama)
    composed = projector.back_project(projected, panorama.shape[:2])
    explicit = projector.back_project(projected.data, panorama.shape[:2])

    np.testing.assert_allclose(composed.data, explicit.data, equal_nan=True)
    np.testing.assert_array_equal(composed.support_mask, explicit.support_mask)


def test_cubemap_project_results_compose_directly_with_back_project() -> None:
    panorama = np.arange(20 * 40, dtype=np.float32).reshape(20, 40)
    projector = CubemapProjector(CubemapSpec((13, 17)), interpolation="nearest")

    projected = projector.project(panorama)
    composed = projector.back_project(projected, panorama.shape)
    explicit = projector.back_project(
        {name: value.data for name, value in projected.items()}, panorama.shape
    )

    assert tuple(projected) == CUBE_FACE_ORDER
    np.testing.assert_array_equal(composed.data, explicit.data)
    np.testing.assert_array_equal(composed.support_mask, explicit.support_mask)


def test_torch_project_results_compose_without_backend_conversion() -> None:
    torch = pytest.importorskip("torch")
    panorama = torch.linspace(0.0, 1.0, 2 * 3 * 12 * 24, dtype=torch.float64)
    panorama = panorama.reshape(2, 3, 12, 24).requires_grad_(True)
    projector = GnomonicProjector(
        GnomonicSpec(7.0, 391.0, 83.0, 57.0, -343.0, (7, 9))
    )

    projected = projector.project(panorama)
    restored = projector.back_project(projected, panorama.shape[-2:])
    restored.data.nan_to_num().sum().backward()

    assert restored.data.shape == panorama.shape
    assert restored.data.dtype == panorama.dtype
    assert restored.data.device == panorama.device
    assert panorama.grad is not None
    assert torch.isfinite(panorama.grad).all()


def test_spec_angles_are_finite_floats_normalized_to_canonical_interval() -> None:
    spec = GnomonicSpec(
        center_lat_deg=np.float32(12.5),
        center_lon_deg=540,
        hfov_deg=np.float64(90),
        vfov_deg=60,
        roll_deg=-540,
        output_shape_hw=(np.int64(7), np.int32(9)),
    )

    assert spec.center_lat_deg == 12.5
    assert spec.center_lon_deg == -180.0
    assert spec.hfov_deg == 90.0
    assert spec.vfov_deg == 60.0
    assert spec.roll_deg == -180.0
    assert spec.output_shape_hw == (7, 9)
    assert all(
        isinstance(value, float)
        for value in (
            spec.center_lat_deg,
            spec.center_lon_deg,
            spec.hfov_deg,
            spec.vfov_deg,
            spec.roll_deg,
        )
    )


def test_periodically_equivalent_specs_produce_identical_results() -> None:
    panorama = np.arange(16 * 32, dtype=np.float64).reshape(16, 32)
    canonical = GnomonicProjector(GnomonicSpec(8.0, 31.0, 91.0, 53.0, 17.0, (7, 9)))
    wrapped = GnomonicProjector(
        GnomonicSpec(8.0, 31.0 + 720.0, 91.0, 53.0, 17.0 - 1080.0, (7, 9))
    )

    assert canonical.spec == wrapped.spec
    np.testing.assert_array_equal(
        canonical.project(panorama).data, wrapped.project(panorama).data
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("center_lat_deg", "0", TypeError),
        ("center_lon_deg", True, TypeError),
        ("hfov_deg", np.nan, ValueError),
        ("vfov_deg", np.inf, ValueError),
        ("roll_deg", object(), TypeError),
    ],
)
def test_spec_rejects_non_numeric_or_non_finite_angles(field, value, error) -> None:
    values = {field: value}
    with pytest.raises(error, match=field):
        GnomonicSpec(**values)


@pytest.mark.parametrize("shape", [(3.5, 4), (True, 4), (0, 4), (3,), "3x4"])
def test_shapes_reject_truncation_boolean_nonpositive_and_malformed_values(shape) -> None:
    with pytest.raises((TypeError, ValueError), match="output_shape_hw"):
        GnomonicSpec(output_shape_hw=shape)

    face = np.ones((3, 4), dtype=np.float32)
    with pytest.raises((TypeError, ValueError), match="output_shape_hw"):
        gnomonic_to_equirectangular(face, GnomonicSpec((0)), shape)


def test_invalid_projector_interpolation_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        GnomonicProjector(GnomonicSpec(), interpolation="cubic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interpolation"):
        CubemapProjector(CubemapSpec(), interpolation="cubic")  # type: ignore[arg-type]


def test_canonical_functions_reject_non_array_inputs_explicitly() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray or torch.Tensor"):
        erp_pixels_to_rays([[0.0, 0.0]], (2, 4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="numpy.ndarray or torch.Tensor"):
        equirectangular_to_gnomonic(  # type: ignore[arg-type]
            [[0.0, 1.0], [2.0, 3.0]], GnomonicSpec(output_shape_hw=(1, 1))
        )


def test_numpy_image_layout_is_rejected_before_geometry_work() -> None:
    image = np.zeros((1, 2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="NumPy images must use HW or HWC"):
        equirectangular_to_gnomonic(image, GnomonicSpec(output_shape_hw=(1, 1)))
