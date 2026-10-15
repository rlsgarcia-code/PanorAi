from __future__ import annotations

import math

import numpy as np
import pytest

import panorai as pa
from panorai.ops import spherical as S


def test_functional_surface_delegates_to_historical_operations() -> None:
    from panorai import image_processing as historical

    assert S.filter2d is historical.spherical_filter2d
    assert S.box_blur is historical.spherical_box_blur
    assert S.gaussian_blur is historical.spherical_gaussian_blur
    assert S.median_blur is historical.spherical_median_blur
    assert S.bilateral_filter is historical.spherical_bilateral_filter
    assert S.sobel is historical.spherical_sobel
    assert S.gradient is historical.spherical_gradient
    assert S.laplacian is historical.spherical_laplacian
    assert S.canny is historical.spherical_canny
    assert S.resize is historical.spherical_resize
    assert S.rotate is historical.spherical_rotate


def test_area_statistics_use_exact_erp_solid_angle() -> None:
    values = np.ones((9, 18), dtype=np.float64)
    histogram, edges = S.area_histogram(values, bins=(0.5, 1.5))

    assert S.area_average(values) == pytest.approx(1.0, abs=1e-15)
    assert S.AreaAverage()(values) == pytest.approx(1.0, abs=1e-15)
    assert histogram.shape == (1,)
    assert histogram[0] == pytest.approx(4.0 * math.pi, abs=2e-14)
    np.testing.assert_array_equal(edges, (0.5, 1.5))
    configured_histogram, _ = S.AreaHistogram(bins=np.asarray((0.5, 1.5)))(values)
    np.testing.assert_array_equal(configured_histogram, histogram)

    rgb = np.stack((values, values * 2.0, values * 3.0), axis=-1)
    np.testing.assert_allclose(S.area_average(rgb), (1.0, 2.0, 3.0), atol=1e-15)


def test_resample_matches_canonical_pixel_centre_rays() -> None:
    from panorai.geometry import erp_pixels_to_rays

    values = np.arange(8 * 16, dtype=np.float64).reshape(8, 16)
    pixels = np.asarray(((0.0, 0.0), (15.0, 7.0), (8.0, 4.0)))
    rays = erp_pixels_to_rays(pixels, values.shape)

    np.testing.assert_array_equal(
        S.resample(values, rays, interpolation="nearest"),
        values[pixels[:, 1].astype(int), pixels[:, 0].astype(int)],
    )
    np.testing.assert_array_equal(
        S.Resample(rays, interpolation="nearest")(values),
        values[pixels[:, 1].astype(int), pixels[:, 0].astype(int)],
    )


def test_configured_filter_objects_match_functional_implementations() -> None:
    rng = np.random.default_rng(91)
    image = rng.normal(size=(8, 16)).astype(np.float64)
    kernel = np.asarray(((0, 1, 0), (1, -4, 1), (0, 1, 0)), dtype=np.float64)
    configured = (
        (
            S.Filter2D(kernel, backend="numpy"),
            S.filter2d(image, kernel, backend="numpy"),
        ),
        (S.BoxBlur(3, backend="numpy"), S.box_blur(image, 3, backend="numpy")),
        (S.MedianBlur(3), S.median_blur(image, 3)),
        (
            S.BilateralFilter(3, 2.0, 1.0),
            S.bilateral_filter(image, 3, 2.0, 1.0),
        ),
        (
            S.Sobel(direction="east", backend="numpy"),
            S.sobel(image, direction="east", backend="numpy"),
        ),
        (
            S.Gradient(component="magnitude", backend="numpy"),
            S.gradient(image, backend="numpy").magnitude,
        ),
        (S.Laplacian(backend="numpy"), S.laplacian(image, backend="numpy")),
    )
    for operation, expected in configured:
        np.testing.assert_array_equal(operation(image), expected)


def test_slic_and_pyramid_adapters_delegate_without_algorithm_copies() -> None:
    from panorai.image_processing import reconstruct_laplacian_pyramid

    rows, columns = np.indices((16, 32))
    image = np.stack(
        (
            columns / 31.0,
            rows / 15.0,
            (rows + columns) / 46.0,
        ),
        axis=-1,
    ).astype(np.float64)
    labels = pa.Pipeline(
        pa.OnSphere(
            pa.Output(S.SLIC(n_segments=8, max_num_iter=2), kind="labels"),
            output="labels",
        )
    )(pa.EquirectangularImage(image))

    assert isinstance(labels, pa.RasterField)
    assert labels.values.dtype == np.int32
    assert labels.grid.is_canonical_erp
    assert labels.grid.transforms[-1].operation == "slic"

    gaussian = S.GaussianPyramid(3, backend="numpy")(image)
    laplacian = S.LaplacianPyramid(3, backend="numpy")(image)
    assert tuple(level.shape[:2] for level in gaussian) == ((16, 32), (8, 16), (4, 8))
    np.testing.assert_allclose(
        reconstruct_laplacian_pyramid(laplacian), image, atol=1e-12, rtol=0.0
    )


def test_dog_adapter_returns_the_existing_public_result_type() -> None:
    from panorai.features import (
        SphericalDoGDetectorConfig,
        SphericalKeypointSet,
    )

    image = np.zeros((32, 64), dtype=np.float32)
    image[16, 32] = 1.0
    detector = S.DoGDetector(
        SphericalDoGDetectorConfig(
            octaves=1,
            levels_per_octave=2,
            max_keypoints=10,
            minimum_valid_support_fraction=0.5,
        )
    )
    result = detector(pa.EquirectangularImage(image), panorama_id="ops-adapter")

    assert isinstance(result, SphericalKeypointSet)
    assert result.panorama_id == "ops-adapter"
    assert detector.describe()["implementation"] == (
        "panorai.features.SphericalDoGDetector"
    )


def test_torch_pool_resize_global_mean_and_grid_validation() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from panorai.ops.spherical import nn as SNN

    values = torch.arange(8 * 16, dtype=torch.float64).reshape(1, 1, 8, 16)
    validity = torch.ones((1, 8, 16), dtype=torch.bool)
    validity[:, 3, 7] = False
    average = SNN.AvgPool2d(
        3,
        stride=2,
        padding=1,
        invalid_policy="renormalize",
        min_valid_weight=0.5,
    )
    pooled, support, valid = average.forward_with_masks(
        values, torch.ones_like(validity), validity
    )
    assert tuple(pooled.shape) == (1, 1, 4, 8)
    assert tuple(support.shape) == (1, 4, 8)
    assert tuple(valid.shape) == (1, 4, 8)
    assert torch.isfinite(pooled[valid[:, None]]).all()

    resize = SNN.Resize((6, 12))
    resized = resize(values)
    assert tuple(resized.shape) == (1, 1, 6, 12)
    torch.testing.assert_close(
        SNN.Interpolate(size=(6, 12))(values), resized, rtol=0.0, atol=0.0
    )

    constant = torch.full((2, 3, 8, 16), 7.0, dtype=torch.float64)
    torch.testing.assert_close(
        SNN.GlobalAveragePool()(constant),
        torch.full((2, 3), 7.0, dtype=torch.float64),
    )

    model = nn.Sequential(
        SNN.Conv2d(1, 2, 3, stride=2, padding=1, dtype=torch.float64),
        nn.ReLU(),
        SNN.ConvTranspose2d(
            2,
            1,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            dtype=torch.float64,
        ),
    )
    report = SNN.validate(model, input_shape=(8, 16))
    assert report.input_grid["shape_hw"] == (8, 16)
    assert report.output_grid["shape_hw"] == (8, 16)
    assert report.layers[0].resolution_changed
    assert report.layers[1].input_grid["shape_hw"] == (4, 8)
    assert report.layers[2].output_grid["alignment"] == "erp-pixel-centres"
    assert report.dtype == "float64"
    assert report.device == "cpu"
