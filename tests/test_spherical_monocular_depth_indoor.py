from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "spherical_monocular_depth"
sys.path.insert(0, str(BENCHMARK))
SPEC = importlib.util.spec_from_file_location(
    "indoor_cnn", BENCHMARK / "indoor_cnn.py"
)
assert SPEC is not None and SPEC.loader is not None
indoor_cnn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(indoor_cnn)


def test_tensor_input_matches_imagenet_contract() -> None:
    torch = pytest.importorskip("torch")
    rgb = np.asarray([[[0, 127.5, 255]]], dtype=np.float32)
    result = indoor_cnn.tensor_input(rgb)
    expected = (
        torch.tensor([0.0, 0.5, 1.0])
        - torch.tensor(indoor_cnn.RGB_MEAN)
    ) / torch.tensor(indoor_cnn.RGB_STD)
    torch.testing.assert_close(result[0, :, 0, 0], expected)


def test_post_op_conv_port_preserves_parameter_norm_and_activation() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class ConvWithPost(nn.Conv2d):
        def __init__(self) -> None:
            super().__init__(2, 2, 1, bias=True)
            self.norm = nn.BatchNorm2d(2)
            self.activation = torch.relu

        def forward(self, values):
            return self.activation(self.norm(super().forward(values)))

    source = ConvWithPost().eval()
    model = nn.Sequential(source)
    values = torch.randn(1, 2, 4, 8)
    expected = model(values)
    weight = source.weight
    norm = source.norm
    _, records = indoor_cnn._port_post_op_convolutions(
        model, max_sampled_elements=64, angular_step_scale=(1.5, 2.0)
    )
    actual = model(values)
    torch.testing.assert_close(actual, expected)
    assert model[0].convolution.weight is weight
    assert model[0].norm is norm
    assert model[0].convolution.angular_step_scale == (1.5, 2.0)
    assert records[0]["angular_step_scale"] == [1.5, 2.0]


def test_native_erp_angular_scale_matches_reference_focal_support() -> None:
    scale = indoor_cnn.angular_step_scale_for_erp((4128, 8256))

    assert scale[0] == pytest.approx(4128 / (np.pi * 519.0))
    assert scale[1] == pytest.approx(8256 / (2.0 * np.pi * 519.0))
    assert scale[0] == pytest.approx(scale[1])


@pytest.mark.parametrize("shape", [(0, 0), (32, 65)])
def test_native_erp_angular_scale_rejects_invalid_lattice(shape) -> None:
    with pytest.raises(ValueError):
        indoor_cnn.angular_step_scale_for_erp(shape)


def test_spherical_inference_applies_published_focal_scale_without_resize() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class ConstantModel(nn.Module):
        def forward(self, values):
            output = torch.ones(
                values.shape[0], 1, values.shape[-2], values.shape[-1]
            )
            return output, {}, {}

    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    result = indoor_cnn.infer_spherical(ConstantModel(), rgb)
    expected = max(
        indoor_cnn.MODEL_DEPTH_RANGE_M[0],
        64.0 / (2.0 * np.pi) / indoor_cnn.CANONICAL_FOCAL_PX,
    )
    assert result.shape == (32, 64)
    np.testing.assert_allclose(result, expected)


def test_spherical_inference_rejects_non_erp_shape() -> None:
    with pytest.raises(ValueError, match="2:1 ERP"):
        indoor_cnn.infer_spherical(None, np.zeros((32, 65, 3), dtype=np.uint8))
