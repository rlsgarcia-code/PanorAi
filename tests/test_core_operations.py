from __future__ import annotations

from dataclasses import FrozenInstanceError
import subprocess
import sys

import numpy as np
import pytest

from panorai.core import SphericalConv2dSpec, SphericalExecutionSpec
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


def test_torch_solid_angle_weights_and_moments_match_independent_numpy_oracle() -> (
    None
):
    torch = pytest.importorskip("torch")
    from panorai.core.torch import (
        spherical_solid_angle_weights as torch_solid_angle_weights,
        spherical_weighted_moments,
    )

    rng = np.random.default_rng(816)
    values = rng.normal(size=(2, 3, 7, 14)).astype(np.float64)
    support = rng.random((2, 1, 7, 14)) > 0.2
    area = spherical_solid_angle_weights((7, 14))
    effective = area[None, None] * support
    denominator = effective.sum()
    expected_mean = (values * effective).sum(axis=(0, 2, 3)) / denominator
    expected_variance = (
        (values - expected_mean[None, :, None, None]) ** 2 * effective
    ).sum(axis=(0, 2, 3)) / denominator

    actual_area = torch_solid_angle_weights((7, 14), dtype=torch.float64)
    actual_mean, actual_variance = spherical_weighted_moments(
        torch.from_numpy(values), support=torch.from_numpy(support)
    )

    np.testing.assert_allclose(actual_area.numpy(), area, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(
        actual_mean.numpy(), expected_mean, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        actual_variance.numpy(), expected_variance, atol=1e-14, rtol=0.0
    )


def test_spherical_batch_norm_matches_batch_norm_for_uniform_area_lattice() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalBatchNorm2d

    torch.manual_seed(817)
    reference = torch.nn.BatchNorm2d(3, momentum=0.2, dtype=torch.float64)
    source = torch.nn.BatchNorm2d(3, momentum=0.2, dtype=torch.float64)
    source.load_state_dict(reference.state_dict())
    spherical = SphericalBatchNorm2d.from_torch(source)
    values = torch.randn(2, 3, 1, 9, dtype=torch.float64)

    expected = reference(values)
    actual = spherical(values)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        spherical.running_mean, reference.running_mean, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        spherical.running_var, reference.running_var, atol=1e-12, rtol=1e-12
    )
    assert spherical.num_batches_tracked.item() == 1


def test_spherical_moments_do_not_overcount_polar_rows() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import spherical_weighted_moments

    values = torch.zeros(1, 1, 4, 8, dtype=torch.float64)
    values[..., (0, -1), :] = 1.0
    mean, _ = spherical_weighted_moments(values)

    assert values.mean().item() == pytest.approx(0.5)
    assert mean.item() == pytest.approx(1.0 - 2**-0.5, abs=1e-15)


def test_spherical_batch_norm_preserves_pretrained_eval_and_object_identity() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalBatchNorm2d

    source = torch.nn.BatchNorm2d(2, dtype=torch.float64).eval()
    with torch.no_grad():
        source.weight.copy_(torch.tensor((1.5, 0.75), dtype=torch.float64))
        source.bias.copy_(torch.tensor((-0.2, 0.4), dtype=torch.float64))
        source.running_mean.copy_(torch.tensor((0.3, -0.6), dtype=torch.float64))
        source.running_var.copy_(torch.tensor((1.2, 2.4), dtype=torch.float64))
    values = torch.randn(2, 2, 7, 14, dtype=torch.float64)
    expected = source(values)
    converted = SphericalBatchNorm2d.from_torch(source)

    assert converted.weight is source.weight
    assert converted.bias is source.bias
    assert converted.running_mean is source.running_mean
    assert converted.running_var is source.running_var
    assert converted.num_batches_tracked is source.num_batches_tracked
    torch.testing.assert_close(converted(values), expected, atol=0.0, rtol=0.0)


def test_spherical_batch_norm_support_and_autograd_are_explicit() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalBatchNorm2d

    values = torch.randn(
        2, 2, 5, 10, dtype=torch.float64, requires_grad=True
    )
    support = torch.ones(2, 1, 5, 10, dtype=torch.bool)
    support[..., 0, :] = False
    values_with_unsupported_nan = values.detach().clone()
    values_with_unsupported_nan[..., 0, :] = torch.nan
    layer = SphericalBatchNorm2d(
        2, track_running_stats=False, dtype=torch.float64
    )
    result = layer(values, support=support)
    masked_result = layer(values_with_unsupported_nan, support=support)
    result.square().mean().backward()

    assert torch.isfinite(masked_result[..., 1:, :]).all()
    assert torch.isnan(masked_result[..., 0, :]).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    with pytest.raises(TypeError, match="boolean"):
        layer(values.detach(), support=support.float())
    with pytest.raises(ValueError, match="at least one"):
        layer(values.detach(), support=torch.zeros_like(support))


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


def test_spherical_transposed_convolution_is_the_forward_adjoint() -> None:
    """The whole ERP exercises periodic seam and polar border accumulation."""

    rng = np.random.default_rng(814)
    values = rng.normal(size=(1, 4, 8, 16)).astype(np.float64)
    weights = rng.normal(size=(6, 2, 3, 3)).astype(np.float64)
    forward = SphericalConv2d(weights, stride=2, padding=1, groups=2)
    probes = rng.normal(size=forward(values).shape).astype(np.float64)
    transpose = SphericalConvTranspose2d(
        weights,
        stride=2,
        padding=1,
        output_padding=1,
        groups=2,
    )

    lhs = np.vdot(forward(values), probes)
    rhs = np.vdot(values, transpose(probes))

    np.testing.assert_allclose(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_numpy_transposed_convolution_chunking_preserves_the_adjoint() -> None:
    rng = np.random.default_rng(815)
    values = rng.normal(size=(1, 2, 5, 10)).astype(np.float64)
    weights = rng.normal(size=(2, 3, 3, 3)).astype(np.float64)
    full = SphericalConvTranspose2d(
        weights, stride=2, padding=1, output_padding=1
    )(values)
    chunked = SphericalConvTranspose2d(
        weights,
        stride=2,
        padding=1,
        output_padding=1,
        execution=SphericalExecutionSpec(max_sampled_elements=32),
    )(values)

    np.testing.assert_allclose(chunked, full, atol=1e-12, rtol=1e-12)


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


def test_torch_conv_and_resize_avoid_mps_unsupported_sampler_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the original MPS failure paths with a backend-independent oracle."""

    torch = pytest.importorskip("torch")
    import torch.nn.functional as torch_functional

    from panorai.core.torch import (
        SphericalConv2d as TorchSphericalConv2d,
        SphericalResize as TorchSphericalResize,
    )

    original_grid_sample = torch_functional.grid_sample

    def reject_border_padding(*args: object, **kwargs: object):
        if kwargs.get("padding_mode", "zeros") == "border":
            raise NotImplementedError("MPS does not implement border padding")
        return original_grid_sample(*args, **kwargs)

    def reject_sinc(*args: object, **kwargs: object):
        raise NotImplementedError("MPS does not implement torch.sinc")

    monkeypatch.setattr(torch, "sinc", reject_sinc)
    monkeypatch.setattr(torch_functional, "grid_sample", reject_border_padding)

    values = torch.randn(1, 2, 7, 14, dtype=torch.float64, requires_grad=True)
    layer = TorchSphericalConv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=1,
        dtype=torch.float64,
    )
    resized = TorchSphericalResize((5, 10))(layer(values))
    resized.square().mean().backward()

    assert resized.shape == (1, 3, 5, 10)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
