"""Contract tests for the authorized invalid-aware bilinear modes.

Expected values are closed-form four-neighbour weighted sums from O-003. The
tests exercise only public geometry/container interfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from panorai.data import EquirectangularImage
from panorai.geometry import (
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    ProjectionResult,
    equirectangular_to_gnomonic,
    gnomonic_to_equirectangular,
)


def _center_spec() -> GnomonicSpec:
    # A 1x1 view at lon=lat=0 samples x=1.5,y=0.5 in a 2x4 ERP, so the
    # independent bilinear oracle is the arithmetic mean of four neighbours.
    return GnomonicSpec(output_shape_hw=(1, 1))


def test_projection_result_two_field_construction_remains_compatible() -> None:
    data = np.ones((1, 1), dtype=np.float32)
    support = np.ones((1, 1), dtype=bool)

    result = ProjectionResult(data, support)

    assert result.data is data
    assert result.support_mask is support
    assert result.validity_mask is None
    assert result.valid_weight is None


def test_propagate_remains_default_and_keeps_zero_weight_nan_behavior() -> None:
    source = np.full((3, 3), np.nan, dtype=np.float32)
    source[1, 1] = 8.0
    spec = GnomonicSpec(output_shape_hw=(1, 1))

    implicit = equirectangular_to_gnomonic(source, spec)
    explicit = equirectangular_to_gnomonic(
        source, spec, invalid_policy="propagate"
    )

    assert np.isnan(implicit.data.item())
    np.testing.assert_array_equal(implicit.data, explicit.data)
    assert implicit.validity_mask is None
    assert implicit.valid_weight is None


def test_renormalize_matches_closed_form_and_exposes_weight_and_validity() -> None:
    source = np.array(((2.0, 2.0, 4.0, 4.0), (2.0, 2.0, 4.0, 4.0)))
    validity = np.array(
        ((False, True, True, False), (False, False, False, False)), dtype=bool
    )

    result = equirectangular_to_gnomonic(
        source,
        _center_spec(),
        invalid_policy="renormalize",
        validity_mask=validity,
        min_valid_weight=0.5,
    )

    # Values 2 and 4 each carry weight 0.25: (0.5 + 1.0) / 0.5 = 3.
    assert result.data.item() == pytest.approx(3.0, abs=1e-12)
    assert result.valid_weight.item() == pytest.approx(0.5, abs=1e-12)
    assert result.validity_mask.item()
    assert result.support_mask.item()


def test_renormalize_exact_pixel_ignores_zero_weight_nan_neighbors() -> None:
    source = np.full((3, 3), np.nan, dtype=np.float64)
    source[1, 1] = 8.0
    validity = np.zeros((3, 3), dtype=bool)
    validity[1, 1] = True

    result = equirectangular_to_gnomonic(
        source,
        GnomonicSpec(output_shape_hw=(1, 1)),
        invalid_policy="renormalize",
        validity_mask=validity,
        min_valid_weight=1.0,
    )

    assert result.data.item() == pytest.approx(8.0, abs=0.0)
    assert result.valid_weight.item() == pytest.approx(1.0, abs=0.0)
    assert result.validity_mask.item()


def test_renormalize_threshold_and_all_invalid_return_nan_without_losing_support() -> None:
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    one_valid = np.zeros_like(source, dtype=bool)
    one_valid[0, 1] = True
    none_valid = np.zeros_like(source, dtype=bool)

    weak = equirectangular_to_gnomonic(
        source,
        _center_spec(),
        invalid_policy="renormalize",
        validity_mask=one_valid,
        min_valid_weight=0.3,
    )
    empty = equirectangular_to_gnomonic(
        source,
        _center_spec(),
        invalid_policy="renormalize",
        validity_mask=none_valid,
        min_valid_weight=0.1,
    )

    assert weak.valid_weight.item() == pytest.approx(0.25, abs=1e-7)
    assert not weak.validity_mask.item()
    assert weak.support_mask.item()
    assert np.isnan(weak.data.item())
    assert empty.valid_weight.item() == pytest.approx(0.0, abs=0.0)
    assert not empty.validity_mask.item()
    assert empty.support_mask.item()
    assert np.isnan(empty.data.item())


def test_nonfinite_channel_invalidates_shared_spatial_sample() -> None:
    source = np.ones((2, 4, 2), dtype=np.float32)
    source[0, 1] = (2.0, np.nan)
    source[0, 2] = (4.0, 40.0)
    validity = np.zeros((2, 4), dtype=bool)
    validity[0, 1:3] = True

    result = equirectangular_to_gnomonic(
        source,
        _center_spec(),
        invalid_policy="renormalize",
        validity_mask=validity,
        min_valid_weight=0.25,
    )

    # The NaN in one channel invalidates that spatial neighbour for both.
    np.testing.assert_allclose(result.data[0, 0], (4.0, 40.0))
    assert result.valid_weight.item() == pytest.approx(0.25, abs=1e-7)


def test_renormalize_uses_wrapped_seam_and_replicated_planar_borders() -> None:
    panorama = np.array(((2.0, 10.0, 20.0, 8.0),), dtype=np.float64)
    panorama_valid = np.array(((True, False, False, True),), dtype=bool)
    seam = equirectangular_to_gnomonic(
        panorama,
        GnomonicSpec(center_lon_deg=-180.0, output_shape_hw=(1, 1)),
        invalid_policy="renormalize",
        validity_mask=panorama_valid,
        min_valid_weight=1.0,
    )
    assert seam.data.item() == pytest.approx(5.0, abs=1e-12)
    assert seam.valid_weight.item() == pytest.approx(1.0, abs=1e-12)

    face = np.array(((2.0, 8.0),), dtype=np.float64)
    face_valid = np.array(((True, False),), dtype=bool)
    backprojected = gnomonic_to_equirectangular(
        face,
        GnomonicSpec(output_shape_hw=face.shape),
        (1, 4),
        invalid_policy="renormalize",
        validity_mask=face_valid,
        min_valid_weight=1.0,
    )
    np.testing.assert_array_equal(
        backprojected.support_mask, ((False, True, True, False),)
    )
    np.testing.assert_array_equal(
        backprojected.validity_mask, ((False, True, False, False),)
    )
    np.testing.assert_allclose(backprojected.valid_weight, ((0.0, 1.0, 0.0, 0.0),))
    assert backprojected.data[0, 1] == pytest.approx(2.0)
    assert np.isnan(backprojected.data[0, 2])


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"invalid_policy": "unknown"}, ValueError, "invalid_policy"),
        ({"invalid_policy": "renormalize"}, ValueError, "validity_mask"),
        (
            {"invalid_policy": "renormalize", "validity_mask": np.ones((2, 4), bool)},
            ValueError,
            "min_valid_weight",
        ),
        (
            {
                "invalid_policy": "renormalize",
                "validity_mask": np.ones((2, 4), bool),
                "min_valid_weight": 0.0,
            },
            ValueError,
            "min_valid_weight",
        ),
        (
            {"validity_mask": np.ones((2, 4), bool)},
            ValueError,
            "only valid with invalid_policy='renormalize'",
        ),
    ],
)
def test_invalid_policy_options_fail_explicitly(kwargs, error, message) -> None:
    source = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(error, match=message):
        equirectangular_to_gnomonic(source, _center_spec(), **kwargs)


def test_validity_mask_requires_boolean_matching_backend_and_shape() -> None:
    source = np.ones((2, 4), dtype=np.float32)
    base = {
        "invalid_policy": "renormalize",
        "min_valid_weight": 0.5,
    }
    with pytest.raises(TypeError, match="boolean"):
        equirectangular_to_gnomonic(
            source, _center_spec(), validity_mask=np.ones((2, 4)), **base
        )
    with pytest.raises(ValueError, match="shape"):
        equirectangular_to_gnomonic(
            source, _center_spec(), validity_mask=np.ones((1, 4), bool), **base
        )


def test_renormalize_rejects_nearest_and_integer_or_boolean_data() -> None:
    validity = np.ones((2, 4), dtype=bool)
    for source in (
        np.ones((2, 4), dtype=np.int16),
        np.ones((2, 4), dtype=bool),
    ):
        with pytest.raises(TypeError, match="floating"):
            equirectangular_to_gnomonic(
                source,
                _center_spec(),
                invalid_policy="renormalize",
                validity_mask=validity,
                min_valid_weight=0.5,
            )
    with pytest.raises(ValueError, match="bilinear"):
        equirectangular_to_gnomonic(
            np.ones((2, 4), dtype=np.float32),
            _center_spec(),
            interpolation="nearest",
            invalid_policy="renormalize",
            validity_mask=validity,
            min_valid_weight=0.5,
        )


def test_real_projectors_and_container_mask_compose_in_normalized_mode() -> None:
    depth = np.arange(8 * 16, dtype=np.float32).reshape(8, 16) + 1.0
    depth[3:5, 7:9] = np.nan
    valid = np.isfinite(depth)
    container = EquirectangularImage(depth, support_mask=valid)
    gnomonic = GnomonicProjector(
        GnomonicSpec(output_shape_hw=(5, 7)),
        invalid_policy="renormalize",
        min_valid_weight=0.25,
    )

    face = gnomonic.project(
        container.data, validity_mask=container.support_mask
    )
    restored = gnomonic.back_project(face, depth.shape)

    assert face.validity_mask is not None
    assert face.valid_weight is not None
    assert restored.validity_mask is not None
    assert restored.valid_weight is not None
    assert np.isnan(restored.data[restored.support_mask & ~restored.validity_mask]).all()

    cubemap = CubemapProjector(
        CubemapSpec((5, 5)),
        invalid_policy="renormalize",
        min_valid_weight=0.25,
    )
    faces = cubemap.project(container.data, validity_mask=container.support_mask)
    cube_restored = cubemap.back_project(faces, depth.shape)
    assert all(face.validity_mask is not None for face in faces.values())
    assert cube_restored.validity_mask.shape == depth.shape
    assert cube_restored.valid_weight.shape == depth.shape


def test_torch_nchw_parity_validity_weight_and_closed_form_gradient() -> None:
    torch = pytest.importorskip("torch")
    numpy_source = np.array(
        [
            [[[2.0, 2.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0]]],
            [[[6.0, 6.0, 10.0, 10.0], [6.0, 6.0, 10.0, 10.0]]],
        ],
        dtype=np.float64,
    )
    numpy_valid = np.array(
        [
            [[False, True, True, False], [False, False, False, False]],
            [[False, True, True, False], [False, False, False, False]],
        ],
        dtype=bool,
    )
    source = torch.tensor(numpy_source, dtype=torch.float64, requires_grad=True)
    validity = torch.tensor(numpy_valid, dtype=torch.bool)

    result = equirectangular_to_gnomonic(
        source,
        _center_spec(),
        invalid_policy="renormalize",
        validity_mask=validity,
        min_valid_weight=0.5,
    )
    result.data.sum().backward()

    np.testing.assert_allclose(result.data.detach().numpy()[:, 0, 0, 0], (3.0, 8.0))
    np.testing.assert_allclose(result.valid_weight.detach().numpy()[:, 0, 0], 0.5)
    assert result.validity_mask.shape == (2, 1, 1)
    assert result.validity_mask.all()
    expected_gradient = np.zeros_like(numpy_source)
    expected_gradient[:, 0, 0, 1:3] = 0.5
    np.testing.assert_allclose(source.grad.numpy(), expected_gradient, atol=1e-12)


def test_torch_mask_must_share_backend_device_and_nchw_batch_shape() -> None:
    torch = pytest.importorskip("torch")
    source = torch.ones((2, 3, 2, 4), dtype=torch.float32)
    options = {
        "invalid_policy": "renormalize",
        "min_valid_weight": 0.5,
    }
    with pytest.raises(TypeError, match="same backend"):
        equirectangular_to_gnomonic(
            source,
            _center_spec(),
            validity_mask=np.ones((2, 2, 4), dtype=bool),
            **options,
        )
    with pytest.raises(ValueError, match="shape"):
        equirectangular_to_gnomonic(
            source,
            _center_spec(),
            validity_mask=torch.ones((2, 4), dtype=torch.bool),
            **options,
        )
