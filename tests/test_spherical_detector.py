from __future__ import annotations

import numpy as np

from panorai.features import (
    SphericalCoarseDoGDetector,
    SphericalCoarseDoGDetectorConfig,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
)
from panorai.geometry import erp_pixels_to_rays


def _config(**changes: object) -> SphericalDoGDetectorConfig:
    values: dict[str, object] = {
        "octaves": 2,
        "max_keypoints": 80,
        "contrast_threshold": 0.003,
        "minimum_valid_support_fraction": 0.95,
        "selection_grid_shape": (6, 12),
        "convolution_backend": "numpy",
    }
    values.update(changes)
    return SphericalDoGDetectorConfig(**values)  # type: ignore[arg-type]


def _textured_panorama(shape: tuple[int, int] = (64, 128)) -> np.ndarray:
    yy, xx = np.indices(shape)
    values = np.sin(xx * 0.41) + np.sin(yy * 0.57) + np.sin((xx + yy) * 0.19)
    return np.clip(values * 35.0 + 128.0, 0.0, 255.0).astype(np.uint8)


def test_detector_returns_descriptor_free_public_keypoints() -> None:
    result = SphericalDoGDetector(_config()).detect(
        _textured_panorama(), panorama_id="synthetic"
    )

    assert 20 <= len(result) <= 80
    assert result.interface == "panorai-spherical-dog-detector/v1"
    assert not hasattr(result, "descriptors")
    assert result.bearings.shape == (len(result), 3)
    assert result.source_erp_xy.shape == (len(result), 2)
    assert np.allclose(np.linalg.norm(result.bearings, axis=1), 1.0)
    assert np.all(result.scales_deg > 0.0)
    assert np.all(result.valid_support_fractions >= 0.95)
    diagnostics = result.diagnostics
    assert diagnostics.raw_extrema_count >= diagnostics.valid_support_count
    assert diagnostics.valid_support_count >= diagnostics.unique_after_deduplication
    assert diagnostics.unique_after_deduplication >= diagnostics.output_after_budget
    assert diagnostics.output_after_budget == len(result)
    assert (
        sum(item.valid_support_count for item in diagnostics.octaves)
        == diagnostics.valid_support_count
    )


def test_detector_is_deterministic_and_longitude_roll_equivariant() -> None:
    image = _textured_panorama()
    detector = SphericalDoGDetector(_config(max_keypoints=100))
    first = detector.detect(image)
    repeated = detector.detect(image)
    shifted = detector.detect(np.roll(image, 9, axis=1))

    assert np.array_equal(first.source_erp_xy, repeated.source_erp_xy)
    shifted_x = (first.source_erp_xy[:, 0] + 9.0) % image.shape[1]
    expected_pixels = np.stack((shifted_x, first.source_erp_xy[:, 1]), axis=1)
    expected = erp_pixels_to_rays(expected_pixels, image.shape)
    similarity = expected @ shifted.bearings.T
    nearest_deg = np.degrees(np.arccos(np.clip(similarity.max(axis=1), -1.0, 1.0)))
    assert np.mean(nearest_deg < 1.5) >= 0.8


def test_scale_aware_validity_rejects_shadow_boundary() -> None:
    image = _textured_panorama()
    validity = np.ones(image.shape, dtype=bool)
    validity[48:, :] = False
    detector = SphericalDoGDetector(
        _config(
            max_keypoints=100,
            minimum_valid_support_fraction=0.995,
        )
    )
    result = detector.detect(image, validity_mask=validity)

    assert len(result) > 0
    pixels = np.rint(result.source_erp_xy).astype(np.int64)
    assert validity[pixels[:, 1], pixels[:, 0]].all()
    assert np.all(result.valid_support_fractions >= 0.995)


def test_equal_area_selection_uses_more_cells_than_response_only() -> None:
    image = _textured_panorama((96, 192))
    balanced = SphericalDoGDetector(
        _config(max_keypoints=40, selection_grid_shape=(6, 12))
    ).detect(image)
    response = SphericalDoGDetector(
        _config(max_keypoints=40, selection_policy="response")
    ).detect(image)

    def occupied(result) -> int:
        bearings = result.bearings
        rows = np.clip(((bearings[:, 1] + 1.0) * 3.0).astype(int), 0, 5)
        lon = np.arctan2(bearings[:, 0], bearings[:, 2])
        columns = np.clip(((lon + np.pi) / (2.0 * np.pi) * 12).astype(int), 0, 11)
        return len(set(zip(rows.tolist(), columns.tolist())))

    assert occupied(balanced) >= occupied(response)


def test_coarse_detector_promotes_geometry_and_provenance_to_source_erp() -> None:
    image = _textured_panorama((128, 256))
    detector = SphericalCoarseDoGDetector(
        SphericalCoarseDoGDetectorConfig(
            proposal_height=64,
            octaves=2,
            max_keypoints=80,
            contrast_threshold=0.003,
            convolution_backend="numpy",
        )
    )

    result = detector.detect(image, panorama_id="coarse-synthetic")

    assert 20 <= len(result) <= 80
    assert result.interface == "panorai-spherical-coarse-dog-detector/v1"
    assert result.source_shape_hw == image.shape
    assert result.config.to_dict()["proposal_height"] == 64
    assert result.config.to_dict()["fine_verification"] is None
    assert np.all((result.source_erp_xy[:, 0] >= -0.5))
    assert np.all((result.source_erp_xy[:, 0] < image.shape[1] - 0.5))
    assert np.all((result.source_erp_xy[:, 1] >= -0.5))
    assert np.all((result.source_erp_xy[:, 1] < image.shape[0] - 0.5))
    expected = erp_pixels_to_rays(result.source_erp_xy, image.shape)
    assert np.allclose(expected, result.bearings, atol=1e-12)


def test_coarse_detector_can_vectorize_native_resolution_fine_verification() -> None:
    image = _textured_panorama((256, 512))
    config = SphericalCoarseDoGDetectorConfig(
        proposal_height=64,
        octaves=2,
        max_keypoints=80,
        contrast_threshold=0.003,
        convolution_backend="numpy",
        fine_verification="tangent-dog",
        fine_candidate_multiplier=2.0,
        fine_patch_size=15,
    )

    result = SphericalCoarseDoGDetector(config).detect(image)

    assert len(result) == 80
    assert result.config.to_dict()["fine_verification"] == "tangent-dog"
    assert result.diagnostics.unique_after_deduplication >= len(result)
    assert np.all(result.responses > 0.0)
    assert np.all(result.valid_support_fractions >= 0.97)
    expected = erp_pixels_to_rays(result.source_erp_xy, image.shape)
    assert np.allclose(expected, result.bearings, atol=1e-12)


def test_coarse_area_resampling_suppresses_nyquist_checkerboard() -> None:
    yy, xx = np.indices((128, 256))
    checkerboard = (((xx + yy) % 2) * 255).astype(np.uint8)
    from panorai.features._spherical_detector import _solid_angle_area_downsample

    reduced, valid = _solid_angle_area_downsample(
        checkerboard, np.ones(checkerboard.shape, dtype=bool), (64, 128)
    )

    assert valid.all()
    assert float(np.std(reduced.astype(np.float64))) <= 0.5
    assert 127.0 <= float(np.mean(reduced)) <= 128.0


def test_coarse_gaussian_resampling_suppresses_checkerboard_and_normalizes_support() -> (
    None
):
    yy, xx = np.indices((128, 256))
    checkerboard = (((xx + yy) % 2) * 255).astype(np.uint8)
    validity = np.ones(checkerboard.shape, dtype=bool)
    validity[48:80, 96:160] = False
    from panorai.features._spherical_detector import (
        _gaussian_antialiased_downsample,
    )

    reduced, valid = _gaussian_antialiased_downsample(
        checkerboard,
        validity,
        (32, 64),
        sigma_px=2.0,
    )

    assert not valid.all()
    assert np.isfinite(reduced).all()
    assert float(np.std(reduced[valid].astype(np.float64))) <= 1.0
    assert 126.0 <= float(np.mean(reduced[valid])) <= 129.0


def test_coarse_spherical_gaussian_resampling_is_roll_equivariant() -> None:
    image = _textured_panorama((128, 256)).astype(np.float32)
    validity = np.ones(image.shape, dtype=bool)
    from panorai.features._spherical_detector import (
        _spherical_gaussian_antialiased_downsample,
    )

    reduced, valid = _spherical_gaussian_antialiased_downsample(
        image,
        validity,
        (32, 64),
        sigma_px=2.0,
        intermediate_height=64,
        backend="numpy",
    )
    shifted, shifted_valid = _spherical_gaussian_antialiased_downsample(
        np.roll(image, 8, axis=1),
        validity,
        (32, 64),
        sigma_px=2.0,
        intermediate_height=64,
        backend="numpy",
    )

    assert valid.all() and shifted_valid.all()
    assert np.allclose(np.roll(reduced, 2, axis=1), shifted, atol=1e-5)
