from __future__ import annotations

import copy
import hashlib

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from panorai.experimental.deep_learning import (  # noqa: E402
    SPHERICAL_TORCH_CONVOLUTION_INTERFACE,
    ImageNetFCN,
    SUPPORTED_IMAGENET_MODELS,
    SphericalConv2d,
    class_activation_map,
    port_module_with_report,
    prefetch_imagenet_weights,
    spherical_area_average,
    sphericalize,
)


def test_deep_learning_namespace_exposes_versioned_spherical_contract() -> None:
    assert SPHERICAL_TORCH_CONVOLUTION_INTERFACE == (
        "panorai-spherical-torch-convolution/v1"
    )
    assert ImageNetFCN.SUPPORTED == ("alexnet", "vgg16", "resnet18")
    assert SUPPORTED_IMAGENET_MODELS == ImageNetFCN.SUPPORTED


def test_prefetch_downloads_to_external_cache_and_records_full_hash(
    tmp_path, monkeypatch
) -> None:
    from panorai.experimental.deep_learning import pretrained

    payload = b"synthetic checkpoint for acquisition contract"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / f"resnet18-{digest[:8]}.pth"

    class FakeWeights:
        url = f"https://download.example/{checkpoint.name}"

        def __str__(self) -> str:
            return "FakeResNet18.DEFAULT"

        def get_state_dict(self, *, progress: bool, check_hash: bool):
            assert progress is False
            assert check_hash is True
            checkpoint.write_bytes(payload)
            return {"layer": object()}

    fake = FakeWeights()
    monkeypatch.setattr(pretrained, "_resolve_model", lambda name: (object(), fake))
    monkeypatch.setattr(pretrained, "_checkpoint_path", lambda weights: checkpoint)

    record = prefetch_imagenet_weights(("resnet18",), progress=False)[0]

    assert record.model_name == "resnet18"
    assert record.cache_path == str(checkpoint)
    assert record.sha256 == digest
    assert record.size_bytes == len(payload)
    assert record.previously_cached is False


def test_prefetch_rejects_a_corrupt_cached_checkpoint_before_loading(
    tmp_path, monkeypatch
) -> None:
    from panorai.experimental.deep_learning import pretrained

    expected = hashlib.sha256(b"expected").hexdigest()
    checkpoint = tmp_path / f"alexnet-{expected[:8]}.pth"
    checkpoint.write_bytes(b"corrupt")

    class FakeWeights:
        url = f"https://download.example/{checkpoint.name}"

        def get_state_dict(self, **kwargs):
            pytest.fail("corrupt checkpoint reached Torchvision deserialization")

    fake = FakeWeights()
    monkeypatch.setattr(pretrained, "_resolve_model", lambda name: (object(), fake))
    monkeypatch.setattr(pretrained, "_checkpoint_path", lambda weights: checkpoint)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        prefetch_imagenet_weights(("alexnet",), progress=False)


def test_spherical_one_by_one_reuses_parameters_and_is_exact() -> None:
    source = nn.Conv2d(3, 4, kernel_size=1, bias=True).double()
    adapter = SphericalConv2d(source)
    values = torch.randn(2, 3, 5, 10, dtype=torch.float64)

    expected = source(values)
    actual = adapter(values)

    assert adapter.weight is source.weight
    assert adapter.bias is source.bias
    assert adapter.interface == SPHERICAL_TORCH_CONVOLUTION_INTERFACE
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_spherical_convolution_is_longitude_roll_equivariant_and_finite_at_poles() -> (
    None
):
    source = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True).double()
    adapter = SphericalConv2d(source)
    values = torch.randn(1, 2, 8, 16, dtype=torch.float64)

    reference = adapter(values)
    shifted = adapter(torch.roll(values, shifts=4, dims=-1))

    torch.testing.assert_close(
        shifted,
        torch.roll(reference, shifts=4, dims=-1),
        rtol=1e-11,
        atol=1e-11,
    )
    assert torch.isfinite(reference[..., (0, -1), :]).all()


def test_spherical_convolution_gradcheck_covers_input_weight_and_bias() -> None:
    source = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True).double()
    adapter = SphericalConv2d(source)
    values = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda image, weight, bias: torch.func.functional_call(
            adapter, {"weight": weight, "bias": bias}, (image,)
        ),
        (values, adapter.weight, adapter.bias),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_spherical_area_average_respects_solid_angle() -> None:
    values = torch.zeros(1, 1, 4, 8, dtype=torch.float64)
    values[:, :, (0, -1)] = 1.0
    polar = spherical_area_average(values)
    values.zero_()
    values[:, :, (1, 2)] = 1.0
    equatorial = spherical_area_average(values)

    assert equatorial.item() > polar.item()
    constant = torch.full((2, 3, 7, 14), 2.5, dtype=torch.float64)
    torch.testing.assert_close(
        spherical_area_average(constant),
        torch.full((2, 3), 2.5, dtype=torch.float64),
    )


def test_class_activation_map_selects_upsamples_and_normalizes() -> None:
    logits = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]], [[-2.0, -1.0], [0.0, 4.0]]]],
        dtype=torch.float64,
    )

    result = class_activation_map(logits, 1, output_shape=(4, 8))

    assert result.shape == (1, 4, 8)
    assert result.min().item() == pytest.approx(0.0)
    assert result.max().item() == pytest.approx(1.0)


def test_resnet18_fcn_conversion_matches_original_planar_logits() -> None:
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    original = copy.deepcopy(model)
    values = torch.randn(1, 3, 224, 224)
    fcn = ImageNetFCN(model, "resnet18").eval()

    with torch.inference_mode():
        expected = original(values)
        dense = fcn.forward_dense(values)
        actual = fcn(values, spherical_average=False)

    assert dense.features.shape == (1, 512, 7, 7)
    assert dense.logits.shape == (1, 1000, 7, 7)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sphericalize_preserves_weights_and_replaces_nested_ops() -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(3, 4, kernel_size=1)),
    )
    weights = [parameter for parameter in model.parameters()]

    sphericalize(model)

    assert type(model[0]).__name__ == "SphericalConv2d"
    assert type(model[1][0]).__name__ == "SphericalMaxPool2d"
    assert type(model[1][1]).__name__ == "SphericalConv2d"
    assert list(model.parameters()) == weights


def test_core_port_report_traces_every_spatial_replacement() -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(3, 4, kernel_size=1)),
    )

    report = port_module_with_report(model)

    assert [layer.path for layer in report.layers] == ["0", "2.0", "2.1"]
    assert [layer.source_type for layer in report.layers] == [
        "Conv2d",
        "MaxPool2d",
        "Conv2d",
    ]
    assert report.layers[0].parameter_identity_preserved is True
    assert report.layers[1].parameter_identity_preserved is None
    assert report.remaining_planar_spatial_layers == ()
    assert not any(
        isinstance(layer, (nn.Conv2d, nn.MaxPool2d)) for layer in model.modules()
    )
