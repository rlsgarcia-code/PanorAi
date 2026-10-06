from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.image_processing import (
    reconstruct_laplacian_pyramid,
    spherical_bilateral_filter,
    spherical_box_blur,
    spherical_canny,
    spherical_equalize_histogram,
    spherical_filter2d,
    spherical_gaussian_blur,
    spherical_gaussian_pyramid,
    spherical_gradient,
    spherical_laplacian,
    spherical_laplacian_pyramid,
    spherical_median_blur,
    spherical_resize,
    spherical_rotate,
)


def _ray_component_image(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    height, width = shape
    y, x = np.indices(shape, dtype=np.float64)
    longitude = ((x + 0.5) / width) * 2.0 * np.pi - np.pi
    latitude = np.pi / 2.0 - ((y + 0.5) / height) * np.pi
    value = np.sin(longitude) * np.cos(latitude)
    east = np.cos(longitude)
    north = -np.sin(longitude) * np.sin(latitude)
    return value, east, north


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 3])
def test_normalized_smoothing_preserves_constants_at_seam_and_poles(
    dtype, channels
) -> None:
    shape = (31, 62) if channels is None else (31, 62, channels)
    image = np.full(shape, 17, dtype=dtype)

    for result in (
        spherical_box_blur(image, 5, backend="numpy"),
        spherical_gaussian_blur(image, 5, 1.2, backend="numpy"),
        spherical_median_blur(image, 3),
        spherical_bilateral_filter(image, 3, sigma_color=10, sigma_space=1),
    ):
        np.testing.assert_allclose(result, 17.0, atol=1e-6, rtol=0.0)
        assert result.shape == image.shape
        assert result.dtype == (np.float32 if dtype == np.float32 else np.float64)


def test_custom_kernel_uses_open_cv_correlation_orientation() -> None:
    image = np.zeros((21, 42), dtype=np.float64)
    image[:, 22:] = 1.0
    east_difference = spherical_filter2d(
        image, np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    )

    assert east_difference[10, 21] > 0.9
    assert east_difference[10, 22] > 0.9
    assert east_difference[10, 0] < -0.9


def test_spherical_gradient_matches_closed_form_tangent_derivatives() -> None:
    image, expected_east, expected_north = _ray_component_image((91, 182))
    gradient = spherical_gradient(image, normalize_by_angle=True, backend="numpy")

    np.testing.assert_allclose(gradient.east, expected_east, atol=1.5e-3, rtol=0.0)
    np.testing.assert_allclose(gradient.north, expected_north, atol=1.5e-3, rtol=0.0)
    np.testing.assert_allclose(
        gradient.magnitude,
        np.hypot(expected_east, expected_north),
        atol=1.5e-3,
        rtol=0.0,
    )
    assert np.isfinite(gradient.magnitude[[0, -1]]).all()


def test_laplacian_annihilates_constant_image() -> None:
    result = spherical_laplacian(
        np.ones((15, 30, 2), dtype=np.float32), backend="numpy"
    )
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_median_removes_isolated_salt_and_bilateral_preserves_step() -> None:
    noisy = np.zeros((21, 42), dtype=np.float64)
    noisy[10, 20] = 255.0
    assert spherical_median_blur(noisy, 3)[10, 20] == 0.0

    step = np.zeros((21, 42), dtype=np.float64)
    step[:, 21:] = 255.0
    gaussian = spherical_gaussian_blur(step, 5, 1.2, backend="numpy")
    bilateral = spherical_bilateral_filter(step, 5, sigma_color=10.0, sigma_space=1.2)
    assert bilateral[10, 20] < gaussian[10, 20]
    assert bilateral[10, 21] > gaussian[10, 21]


def test_rotation_by_one_longitude_pixel_is_exact_with_nearest_sampling() -> None:
    height, width = 10, 20
    image = np.broadcast_to(np.arange(width), (height, width)).astype(np.float64)
    yaw = 2.0 * np.pi / width
    rotation = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ]
    )

    rotated = spherical_rotate(image, rotation, interpolation="nearest")

    np.testing.assert_array_equal(rotated, np.roll(image, 1, axis=1))


def test_resize_preserves_constant_hwc_and_validates_rotation() -> None:
    image = np.full((9, 18, 3), 0.25, dtype=np.float32)
    resized = spherical_resize(image, (17, 34))
    assert resized.shape == (17, 34, 3)
    assert resized.dtype == np.float32
    np.testing.assert_allclose(resized, 0.25, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="orthonormal"):
        spherical_rotate(image, np.diag([1.0, 1.0, 2.0]))


def test_unweighted_histogram_equalization_matches_opencv() -> None:
    cv2 = pytest.importorskip("cv2")
    image = np.random.default_rng(4).integers(0, 64, size=(37, 74), dtype=np.uint8)

    actual = spherical_equalize_histogram(image, area_weighted=False)

    np.testing.assert_array_equal(actual, cv2.equalizeHist(image))


def test_equalized_image_composes_with_real_spherical_feature_extractor() -> None:
    pytest.importorskip("cv2")
    from panorai.features import FaceSetSpec, FeatureExtractor, FeatureExtractorConfig

    y, x = np.indices((96, 192))
    image = np.where(((x // 8) + (y // 8)) % 2, 180, 40).astype(np.uint8)
    equalized = spherical_equalize_histogram(image)
    result = FeatureExtractor(
        FeatureExtractorConfig(
            method="orb",
            max_features=120,
            edge_margin_px=2,
            parameters=(("edgeThreshold", 5), ("fastThreshold", 5), ("patchSize", 15)),
        )
    ).extract(
        equalized,
        face_set_spec=FaceSetSpec(
            sampler="cube", shape_hw=(48, 48), fov_deg=(90.0, 90.0)
        ),
    )

    assert len(result.features) > 0
    assert result.panorama_checksum


def test_spherical_histogram_weights_rows_by_solid_angle_and_honors_mask() -> None:
    image = np.array(
        [
            [0, 0, 0, 0],
            [64, 64, 64, 64],
            [128, 128, 128, 128],
            [192, 192, 192, 192],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    weighted = spherical_equalize_histogram(image)
    planar = spherical_equalize_histogram(image, area_weighted=False)

    assert weighted[1, 0] > planar[1, 0]
    assert weighted[-1, 0] == 255

    mask = np.ones_like(image, dtype=bool)
    mask[2] = False
    masked = spherical_equalize_histogram(image, mask=mask)
    assert not np.array_equal(masked, weighted)
    assert masked[-1, 0] == 255


def test_canny_connects_a_seam_crossing_band() -> None:
    image = np.zeros((48, 96), dtype=np.float64)
    image[:, :5] = 255.0
    image[:, -5:] = 255.0

    edges = spherical_canny(
        image, 10.0, 20.0, gaussian_ksize=3, gaussian_sigma=0.8, backend="numpy"
    )

    assert edges.dtype == np.uint8
    assert set(np.unique(edges)) <= {0, 255}
    assert np.count_nonzero(edges[:, 4:8]) > image.shape[0] // 2
    assert np.count_nonzero(edges[:, -8:-4]) > image.shape[0] // 2
    assert np.array_equal(edges[:, 0], edges[:, -1])


def test_canny_zero_thresholds_do_not_turn_flat_pixels_into_edges() -> None:
    edges = spherical_canny(
        np.zeros((24, 48), dtype=np.float32),
        0.0,
        0.0,
        gaussian_ksize=3,
        backend="numpy",
    )

    assert not edges.any()


def test_gaussian_and_laplacian_pyramids_have_expected_shapes_and_reconstruct() -> None:
    image = np.random.default_rng(5).normal(size=(33, 66, 2)).astype(np.float32)
    gaussian = spherical_gaussian_pyramid(image, 4, backend="numpy")
    laplacian = spherical_laplacian_pyramid(image, 4, backend="numpy")

    assert [level.shape[:2] for level in gaussian] == [
        (33, 66),
        (17, 33),
        (9, 17),
        (5, 9),
    ]
    assert [level.shape for level in laplacian] == [level.shape for level in gaussian]
    reconstructed = reconstruct_laplacian_pyramid(laplacian)
    np.testing.assert_allclose(reconstructed, image, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: spherical_filter2d(np.zeros((4, 8)), np.ones((2, 2))), ValueError),
        (lambda: spherical_box_blur(np.zeros((4, 8)), 2), ValueError),
        (
            lambda: spherical_equalize_histogram(np.zeros((4, 8), np.float32)),
            TypeError,
        ),
        (lambda: spherical_canny(np.zeros((4, 8, 3)), 1, 2), ValueError),
        (lambda: spherical_canny(np.zeros((4, 8)), 3, 2), ValueError),
        (lambda: spherical_resize(np.zeros((4, 8)), (1, 8)), ValueError),
    ],
)
def test_public_validation_fails_before_processing(call, error) -> None:
    with pytest.raises(error):
        call()
