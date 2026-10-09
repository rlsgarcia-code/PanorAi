from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


BENCHMARK = (
    Path(__file__).parents[1] / "benchmarks" / "spherical_monocular_depth"
)
sys.path.insert(0, str(BENCHMARK))
SPEC = importlib.util.spec_from_file_location(
    "panoramic_cnn", BENCHMARK / "panoramic_cnn.py"
)
assert SPEC is not None and SPEC.loader is not None
panoramic_cnn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panoramic_cnn)


def test_tensor_input_uses_imagenet_normalization() -> None:
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    values = panoramic_cnn.tensor_input(rgb).numpy()
    expected = -panoramic_cnn.RGB_MEAN / panoramic_cnn.RGB_STD
    assert values.shape == (1, 3, 2, 4)
    np.testing.assert_allclose(values[0, :, 0, 0], expected)


def test_cubemap_strip_preserves_native_angular_density() -> None:
    rgb = np.zeros((64, 128, 3), dtype=np.uint8)
    strip = panoramic_cnn.cubemap_strip(rgb)
    assert strip.shape == (32, 32 * 6, 3)


def test_cubemap_strip_rejects_non_two_to_one_input() -> None:
    with pytest.raises(ValueError, match="2:1"):
        panoramic_cnn.cubemap_strip(np.zeros((64, 96, 3), dtype=np.uint8))


def test_inference_rejects_shape_different_from_constructed_grid() -> None:
    class Model:
        equi_h = 64
        equi_w = 128

    with pytest.raises(ValueError, match="constructed"):
        panoramic_cnn.infer_unifuse(
            Model(), np.zeros((32, 64, 3), dtype=np.uint8)
        )
