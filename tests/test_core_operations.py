from __future__ import annotations

from dataclasses import FrozenInstanceError
import subprocess
import sys

import numpy as np
import pytest

from panorai.core import SphericalConv2dSpec
from panorai.core.geometry import (
    GnomonicSpec as CoreGnomonicSpec,
    tangent_offsets_to_rays,
)
from panorai.core.numpy import (
    Sequential,
    SphericalAreaMean,
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalGaussianBlur2d,
    SphericalResize,
    SphericalScaleExtrema,
    spherical_filter2d,
    spherical_solid_angle_weights,
)
from panorai.geometry import GnomonicSpec
from panorai.image_processing import spherical_filter2d as legacy_filter2d


def test_core_import_does_not_load_torch() -> None:
    script = "import sys, panorai.core; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_geometry_facade_preserves_object_identity() -> None:
    assert CoreGnomonicSpec is GnomonicSpec


def test_tangent_offsets_support_per_bearing_sampling_lattices() -> None:
    bearings = np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    east = np.zeros((2, 3, 4), dtype=np.float64)
    north = np.zeros_like(east)
    rays = tangent_offsets_to_rays(bearings, east, north)
    assert rays.shape == (2, 3, 4, 3)
    np.testing.assert_allclose(
        rays[0], np.broadcast_to(bearings[0], rays[0].shape), atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        rays[1], np.broadcast_to(bearings[1], rays[1].shape), atol=1e-15, rtol=0.0
    )


def test_specs_are_immutable_and_compute_planar_compatible_shape() -> None:
    spec = SphericalConv2dSpec(3, 8, 3, stride=2, padding=1)
    assert spec.output_shape((17, 34)) == (9, 17)
    with pytest.raises(FrozenInstanceError):
        spec.groups = 2  # type: ignore[misc]


def test_legacy_filter_is_the_core_function() -> None:
    assert legacy_filter2d is spherical_filter2d


def test_numpy_conv_preserves_constant_and_channel_contract() -> None:
    values = np.ones((2, 3, 12, 24), dtype=np.float64)
    weights = np.ones((4, 3, 3, 3), dtype=np.float64) / 27.0
    result = SphericalConv2d(weights, padding=1)(values)
    assert result.shape == (2, 4, 12, 24)
    np.testing.assert_allclose(result, 1.0, atol=1e-12, rtol=0.0)


def test_numpy_modules_compose_without_application_specific_class() -> None:
    values = np.zeros((1, 1, 16, 32), dtype=np.float32)
    values[..., 8, 16] = 1.0
    model = Sequential(
        SphericalGaussianBlur2d(1.0, kernel_size=3),
        SphericalResize((8, 16)),
    )
    result = model(values)
    assert result.shape == (1, 1, 8, 16)
    assert np.isfinite(result).all()
    assert float(result.max()) > 0.0


def test_scale_extrema_are_strict_in_space_and_scale() -> None:
    values = np.zeros((1, 1, 5, 9, 18), dtype=np.float64)
    values[:, :, 1:4, 4, 9] = (1.0, 3.0, 1.0)
    extrema = SphericalScaleExtrema()(values)
    assert extrema[0, 0, 2, 4, 9]
    values[0, 0, 2, 4, 10] = 3.0
    assert not SphericalScaleExtrema()(values)[0, 0, 2, 4, 9]


def test_general_numpy_convolution_explicit_native_request_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from panorai.core import SphericalExecutionSpec
    from panorai.core.numpy import _native

    layer = SphericalConv2d(
        np.ones((1, 1, 3, 3), dtype=np.float32),
        padding=1,
        execution=SphericalExecutionSpec(backend="native"),
    )
    monkeypatch.setattr(_native, "_native_geometry", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        layer(np.ones((1, 1, 8, 16), dtype=np.float32))


def test_exact_solid_angle_weights_integrate_to_four_pi() -> None:
    weights = spherical_solid_angle_weights((181, 362))
    assert not weights.flags.writeable
    assert float(weights.sum()) == pytest.approx(4.0 * np.pi, abs=1e-12)
    mean = SphericalAreaMean()(np.ones((1, 2, 181, 362), dtype=np.float64))
    np.testing.assert_allclose(mean, 1.0, atol=1e-12, rtol=0.0)


def test_numpy_and_torch_convolution_share_the_same_mathematics() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalConv2d as TorchSphericalConv2d

    rng = np.random.default_rng(812)
    values = rng.normal(size=(1, 2, 9, 18)).astype(np.float64)
    weights = rng.normal(size=(3, 2, 3, 3)).astype(np.float64)
    bias = rng.normal(size=(3,)).astype(np.float64)
    expected = SphericalConv2d(
        weights, bias, padding=1, angular_step_deg=7.5
    )(values)
    source = torch.nn.Conv2d(2, 3, 3, padding=1, dtype=torch.float64)
    with torch.no_grad():
        source.weight.copy_(torch.from_numpy(weights))
        source.bias.copy_(torch.from_numpy(bias))
    actual = TorchSphericalConv2d.from_torch(
        source, angular_step_deg=7.5
    )(torch.from_numpy(values))
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-9, rtol=1e-9)


def test_numpy_and_torch_transposed_convolution_share_the_same_mathematics() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import (
        SphericalConvTranspose2d as TorchSphericalConvTranspose2d,
    )

    rng = np.random.default_rng(813)
    values = rng.normal(size=(1, 2, 5, 10)).astype(np.float64)
    weights = rng.normal(size=(2, 3, 3, 3)).astype(np.float64)
    bias = rng.normal(size=(3,)).astype(np.float64)
    expected = SphericalConvTranspose2d(
        weights, bias, stride=2, padding=1, output_padding=1
    )(values)
    source = torch.nn.ConvTranspose2d(
        2, 3, 3, stride=2, padding=1, output_padding=1, dtype=torch.float64
    )
    with torch.no_grad():
        source.weight.copy_(torch.from_numpy(weights))
        source.bias.copy_(torch.from_numpy(bias))
    actual = TorchSphericalConvTranspose2d(source)(torch.from_numpy(values))
    assert actual.shape == expected.shape == (1, 3, 10, 20)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-9, rtol=1e-9)


def test_torch_direct_constructor_and_autograd() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalConv2d as TorchSphericalConv2d

    layer = TorchSphericalConv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=1,
        dtype=torch.float64,
    )
    values = torch.randn(1, 2, 7, 14, dtype=torch.float64, requires_grad=True)
    layer(values).square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
