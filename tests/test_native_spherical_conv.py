from __future__ import annotations

import numpy as np
import pytest

from panorai.core import SphericalExecutionSpec
from panorai.core.numpy import SphericalConv2d
from panorai.core.numpy import _native


native = pytest.importorskip("panorai._native._geometry")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("shape", "weight_shape", "kwargs"),
    [
        ((2, 3, 11, 22), (4, 3, 3, 3), {"padding": 1}),
        (
            (1, 4, 13, 26),
            (6, 2, 3, 2),
            {"stride": (2, 3), "padding": (2, 1), "dilation": (2, 1), "groups": 2},
        ),
        ((1, 4, 9, 18), (4, 1, 3, 3), {"padding": 1, "groups": 4}),
        ((1, 2, 8, 16), (3, 2, 1, 1), {}),
    ],
)
def test_native_convolution_matches_numpy_reference(
    dtype: type[np.floating],
    shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    kwargs: dict[str, object],
) -> None:
    rng = np.random.default_rng(1307)
    values = rng.normal(size=shape).astype(dtype)
    weights = rng.normal(size=weight_shape).astype(dtype)
    bias = rng.normal(size=(weight_shape[0],)).astype(dtype)
    numpy_result = SphericalConv2d(
        weights,
        bias,
        angular_step_deg=6.25,
        execution=SphericalExecutionSpec(backend="numpy", max_sampled_elements=4096),
        **kwargs,
    )(values)
    native_result = SphericalConv2d(
        weights,
        bias,
        angular_step_deg=6.25,
        execution=SphericalExecutionSpec(backend="native"),
        **kwargs,
    )(values)
    tolerance = 2e-5 if dtype is np.float32 else 2e-12
    np.testing.assert_allclose(
        native_result, numpy_result, atol=tolerance, rtol=tolerance
    )


def test_native_convolution_matches_default_erp_steps_at_seam_and_poles() -> None:
    height, width = 9, 18
    values = np.zeros((1, 2, height, width), dtype=np.float64)
    values[:, 0, :, 0] = 1.0
    values[:, 0, 0, :] += np.linspace(0.0, 1.0, width)
    values[:, 1, -1, :] = np.linspace(1.0, 0.0, width)
    weights = np.arange(36, dtype=np.float64).reshape(2, 2, 3, 3) / 35.0
    expected = SphericalConv2d(
        weights,
        padding=1,
        execution=SphericalExecutionSpec(backend="numpy"),
    )(values)
    actual = SphericalConv2d(
        weights,
        padding=1,
        execution=SphericalExecutionSpec(backend="native"),
    )(values)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_native_convolution_preserves_constant_analytic_oracle() -> None:
    values = np.ones((2, 3, 12, 24), dtype=np.float32)
    weights = np.ones((4, 3, 3, 3), dtype=np.float32) / 27.0
    actual = SphericalConv2d(
        weights,
        padding=1,
        execution=SphericalExecutionSpec(backend="native"),
    )(values)
    np.testing.assert_allclose(actual, 1.0, atol=2e-6, rtol=0.0)


def test_native_convolution_matches_mixed_dtype_noncontiguous_chw_input() -> None:
    rng = np.random.default_rng(1308)
    values = rng.normal(size=(2, 10, 20)).astype(np.float32)[:, :, ::-1]
    weights = rng.normal(size=(3, 2, 2, 2)).astype(np.float64)
    bias = rng.normal(size=3).astype(np.float32)
    expected = SphericalConv2d(
        weights,
        bias,
        padding=1,
        execution=SphericalExecutionSpec(backend="numpy"),
    )(values)
    actual = SphericalConv2d(
        weights,
        bias,
        padding=1,
        execution=SphericalExecutionSpec(backend="native"),
    )(values)
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_native_convolution_preserves_empty_batch_contract() -> None:
    values = np.empty((0, 1, 4, 8), dtype=np.float32)
    weights = np.ones((2, 1, 3, 3), dtype=np.float32)
    expected = SphericalConv2d(
        weights,
        padding=1,
        execution=SphericalExecutionSpec(backend="numpy"),
    )(values)
    actual = SphericalConv2d(
        weights,
        padding=1,
        execution=SphericalExecutionSpec(backend="native"),
    )(values)
    assert actual.shape == expected.shape == (0, 2, 4, 8)


def test_auto_dispatches_to_native_when_extension_supports_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False
    implementation = _native.native_spherical_conv2d

    def record(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal called
        called = True
        return implementation(*args, **kwargs)

    monkeypatch.setattr(_native, "native_spherical_conv2d", record)
    SphericalConv2d(np.ones((1, 1, 1, 1), dtype=np.float32))(
        np.ones((1, 1, 4, 8), dtype=np.float32)
    )
    assert called


def test_explicit_numpy_never_calls_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("native backend was called")

    monkeypatch.setattr(_native, "native_spherical_conv2d", forbidden)
    result = SphericalConv2d(
        np.ones((1, 1, 1, 1), dtype=np.float32),
        execution=SphericalExecutionSpec(backend="numpy"),
    )(np.ones((1, 1, 4, 8), dtype=np.float32))
    np.testing.assert_array_equal(result, 1.0)


def test_native_entry_point_rejects_invalid_dtype_and_groups() -> None:
    with pytest.raises(TypeError, match="same native float32 or float64"):
        native.spherical_conv2d(
            np.zeros((1, 1, 4, 8), dtype=np.uint8),
            np.ones((1, 1, 3, 3), dtype=np.float32),
            None,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            np.pi / 4,
            np.pi / 4,
        )
    with pytest.raises(ValueError, match="groups and weight channels"):
        native.spherical_conv2d(
            np.zeros((1, 2, 4, 8), dtype=np.float32),
            np.ones((2, 1, 3, 3), dtype=np.float32),
            None,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            np.pi / 4,
            np.pi / 4,
        )
