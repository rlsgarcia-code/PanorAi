from importlib.resources import files

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from panorai.geometry import (  # noqa: E402
    GnomonicSpec,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    equirectangular_to_gnomonic,
    gnomonic_to_equirectangular,
)


def test_numpy_torch_grid_sampling_parity() -> None:
    rng = np.random.default_rng(7)
    numpy_image = rng.normal(size=(24, 48, 3)).astype(np.float32)
    torch_image = torch.from_numpy(numpy_image).permute(2, 0, 1)
    spec = GnomonicSpec(11.0, -67.0, 101.0, 63.0, 17.0, (19, 27))

    numpy_nearest = equirectangular_to_gnomonic(
        numpy_image, spec, interpolation="nearest"
    ).data
    torch_nearest = equirectangular_to_gnomonic(
        torch_image, spec, interpolation="nearest"
    ).data.permute(1, 2, 0).numpy()
    numpy_bilinear = equirectangular_to_gnomonic(numpy_image, spec).data
    torch_bilinear = (
        equirectangular_to_gnomonic(torch_image, spec).data.permute(1, 2, 0).numpy()
    )

    tolerance = yaml.safe_load(
        files("panorai.geometry")
        .joinpath("geometry-v1.yaml")
        .read_text(encoding="utf-8")
    )["tolerances"]["float32_bilinear"]
    assert np.array_equal(numpy_nearest, torch_nearest)
    np.testing.assert_allclose(
        numpy_bilinear,
        torch_bilinear,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )


def test_numpy_torch_backprojection_mask_and_bilinear_parity() -> None:
    rng = np.random.default_rng(9)
    numpy_face = rng.normal(size=(17, 29, 2)).astype(np.float32)
    torch_face = torch.from_numpy(numpy_face).permute(2, 0, 1)
    spec = GnomonicSpec(12.0, -17.0, 103.0, 57.0, 13.0, (17, 29))

    numpy_result = gnomonic_to_equirectangular(numpy_face, spec, (24, 48))
    torch_result = gnomonic_to_equirectangular(torch_face, spec, (24, 48))

    assert np.array_equal(numpy_result.support_mask, torch_result.support_mask.numpy())
    assert np.allclose(
        numpy_result.data,
        torch_result.data.permute(1, 2, 0).numpy(),
        atol=5e-6,
        rtol=5e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_preserves_batch_channels_device_dtype_and_gradient(dtype) -> None:
    image = torch.linspace(0.0, 1.0, 2 * 3 * 20 * 40, dtype=dtype).reshape(2, 3, 20, 40)
    image.requires_grad_(True)
    spec = GnomonicSpec(output_shape_hw=(13, 21), roll_deg=8.0)

    result = equirectangular_to_gnomonic(image, spec)
    result.data.square().mean().backward()

    assert result.data.shape == (2, 3, 13, 21)
    assert result.data.dtype == dtype
    assert result.data.device == image.device
    assert result.support_mask.device == image.device
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0


def test_torch_integer_nearest_cubemap_is_exact() -> None:
    labels = (torch.arange(24 * 48).reshape(24, 48) % 13).to(torch.int64)

    cube = equirectangular_to_cubemap(labels, (17, 17), interpolation="nearest")
    result = cubemap_to_equirectangular(
        {name: projected.data for name, projected in cube.items()},
        labels.shape,
        interpolation="nearest",
    )

    assert result.data.dtype == labels.dtype
    assert result.data.device == labels.device
    assert set(result.data.unique().tolist()).issubset(set(labels.unique().tolist()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_cuda_smoke() -> None:
    image = torch.ones((1, 3, 12, 24), device="cuda")
    result = equirectangular_to_gnomonic(
        image, GnomonicSpec(output_shape_hw=(8, 8))
    )
    assert result.data.device.type == "cuda"


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_torch_mps_smoke() -> None:
    image = torch.ones((1, 3, 12, 24), device="mps")
    result = equirectangular_to_gnomonic(
        image, GnomonicSpec(output_shape_hw=(8, 8))
    )
    assert result.data.device.type == "mps"
