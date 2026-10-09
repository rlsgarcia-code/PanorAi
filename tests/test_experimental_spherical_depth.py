from __future__ import annotations

import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from panorai.experimental.deep_learning import (  # noqa: E402
    METRIC3D_CONVNEXT_TINY_V1,
    METRIC3D_SOURCE,
    SPHERICAL_METRIC_DEPTH_INTERFACE,
    SphericalConv2d,
    SphericalConvTranspose2d,
    SphericalMetric3D,
    UpstreamTermsNotAcceptedError,
    acquire_metric3d_convnext_tiny_v1,
    port_module_with_report,
)
from panorai.experimental.deep_learning import depth as depth_module  # noqa: E402


def test_metric3d_identity_and_license_boundary_are_explicit() -> None:
    assert SPHERICAL_METRIC_DEPTH_INTERFACE.endswith("/v1-experimental")
    assert METRIC3D_SOURCE.sha256 == (
        "929432fd1f7d1f2c45f51407e4ef30ca5758eb207b15e51c9b480048176ee365"
    )
    assert METRIC3D_CONVNEXT_TINY_V1.sha256 == (
        "bc41f5f919bb0388bbc88fe1d9e60b49b826c620b4acc1b6c10f473f6d4741a5"
    )
    assert "BSD-2-Clause" in METRIC3D_SOURCE.license_statement
    assert "No separate" in METRIC3D_CONVNEXT_TINY_V1.license_statement
    assert "never redistributes" in METRIC3D_CONVNEXT_TINY_V1.redistribution


def test_acquisition_requires_opt_in_before_touching_cache(tmp_path: Path) -> None:
    cache = tmp_path / "absent"
    with pytest.raises(UpstreamTermsNotAcceptedError, match="unresolved"):
        acquire_metric3d_convnext_tiny_v1(
            accept_upstream_terms=False,
            cache_dir=cache,
        )
    assert not cache.exists()


def test_download_verifies_full_sha_and_reuses_external_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"external checkpoint bytes"
    spec = depth_module.ExternalArtifactSpec(
        kind="test",
        url="https://example.invalid/model.pth",
        filename="model.pth",
        sha256=depth_module.hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_statement="test-only",
        redistribution="test-only",
    )

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    calls = 0

    def open_url(request, timeout):
        nonlocal calls
        calls += 1
        assert request.full_url == spec.url
        assert timeout == 3.0
        return Response(payload)

    monkeypatch.setattr(depth_module, "urlopen", open_url)
    destination = tmp_path / spec.filename
    assert depth_module._download(spec, destination, timeout_seconds=3.0) is False
    assert depth_module._download(spec, destination, timeout_seconds=3.0) is True
    assert destination.read_bytes() == payload
    assert calls == 1


def test_spherical_transpose_preserves_parameters_shape_and_chunking() -> None:
    torch.manual_seed(11)
    source = nn.ConvTranspose2d(
        4,
        6,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        groups=2,
    ).double()
    full = SphericalConvTranspose2d(source)
    chunked = SphericalConvTranspose2d(source, max_sampled_elements=16)
    values = torch.randn(1, 4, 5, 10, dtype=torch.float64)

    expected_shape = source(values).shape
    full_result = full(values)
    chunked_result = chunked(values)

    assert full.weight is source.weight and full.bias is source.bias
    assert full_result.shape == expected_shape
    torch.testing.assert_close(chunked_result, full_result, rtol=1e-12, atol=1e-12)


def test_spherical_convolution_chunking_matches_full_lattice() -> None:
    torch.manual_seed(13)
    source = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2).double()
    full = SphericalConv2d(source)
    chunked = SphericalConv2d(source, max_sampled_elements=20)
    values = torch.randn(1, 4, 6, 12, dtype=torch.float64)

    torch.testing.assert_close(chunked(values), full(values), rtol=1e-12, atol=1e-12)


def test_fractional_angular_support_has_closed_form_equatorial_offset() -> None:
    source = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), bias=False).double()
    with torch.no_grad():
        source.weight.zero_()
        source.weight[0, 0, 0, 2] = 1.0
    spherical = SphericalConv2d(source, angular_step_scale=1.5)
    values = torch.arange(18, dtype=torch.float64).repeat(9, 1)[None, None]

    result = spherical(values)

    # Row four is the equator.  The east tap is exactly 1.5 native longitude
    # cells away there, so bilinear sampling of the x ramp has value x + 1.5.
    assert result.shape == values.shape
    assert result[0, 0, 4, 7].item() == pytest.approx(8.5, abs=1e-12)


def test_fractional_angular_support_preserves_chunking_and_autograd() -> None:
    torch.manual_seed(17)
    source = nn.Conv2d(2, 3, kernel_size=3, padding=1).double()
    full = SphericalConv2d(source, angular_step_scale=(1.75, 2.25))
    chunked = SphericalConv2d(
        source,
        angular_step_scale=(1.75, 2.25),
        max_sampled_elements=18,
    )
    values = torch.randn(1, 2, 5, 10, dtype=torch.float64, requires_grad=True)

    torch.testing.assert_close(chunked(values), full(values), rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(full, (values,), eps=1e-6, atol=2e-4)


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, (1.0, 0.0), (1.0, float("inf")), (1.0,)]
)
def test_fractional_angular_support_rejects_invalid_scale(scale) -> None:
    with pytest.raises(ValueError, match="angular_step_scale"):
        SphericalConv2d(nn.Conv2d(1, 1, 3), angular_step_scale=scale)


def test_spherical_one_by_one_transpose_is_planar_exact_and_differentiable() -> None:
    source = nn.ConvTranspose2d(2, 3, kernel_size=1, bias=True).double()
    spherical = SphericalConvTranspose2d(source)
    values = torch.randn(1, 2, 3, 6, dtype=torch.float64, requires_grad=True)

    expected = source(values)
    actual = spherical(values)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)
    actual.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert source.weight.grad is not None and torch.isfinite(source.weight.grad).all()


def test_port_collapses_reflection_and_ports_all_learned_spatial_layers() -> None:
    model = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(2, 3, 3),
        nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1),
    )
    parameters = tuple(model.parameters())

    report = port_module_with_report(model, max_sampled_elements=256)
    result = model(torch.randn(1, 2, 8, 16))

    assert result.shape == (1, 2, 16, 32)
    assert report.collapsed_reflection_pads == ("0",)
    assert [layer.source_type for layer in report.layers] == [
        "Conv2d",
        "ConvTranspose2d",
    ]
    assert all(layer.parameter_identity_preserved for layer in report.layers)
    assert all(layer.angular_step_scale == (1.0, 1.0) for layer in report.layers)
    assert tuple(model.parameters()) == parameters
    assert report.remaining_planar_spatial_layers == ()


def test_resize_free_wrapper_returns_radial_metres_without_shape_change() -> None:
    class DummyMetric3D:
        def inference(self, payload):
            values = payload["input"]
            prediction = values.new_full(
                (values.shape[0], 1, *values.shape[-2:]), 100.0
            )
            return prediction, None, {}

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

    wrapper = SphericalMetric3D(DummyMetric3D())
    values = torch.full((2, 3, 64, 128), 127.0)

    radial = wrapper(values)

    assert radial.shape == (2, 1, 64, 128)
    expected = 100.0 * 128.0 / (2.0 * 3.141592653589793) / 1000.0
    assert radial.unique().item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("shape", [(1, 3, 64, 64), (1, 3, 65, 130)])
def test_resize_free_wrapper_rejects_noncanonical_lattices(shape) -> None:
    class Unused:
        def inference(self, payload):
            pytest.fail("invalid shape reached upstream inference")

    with pytest.raises(ValueError):
        SphericalMetric3D(Unused())(torch.ones(shape))


def test_resize_free_wrapper_rejects_implicit_dtype_conversion() -> None:
    class Unused:
        def inference(self, payload):
            pytest.fail("invalid dtype reached upstream inference")

    with pytest.raises(TypeError, match="torch.float32"):
        SphericalMetric3D(Unused())(torch.ones(1, 3, 64, 128, dtype=torch.float64))
