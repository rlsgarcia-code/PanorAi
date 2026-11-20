from __future__ import annotations

import numpy as np
import pytest

from panorai.features import (
    SphericalHarrisConfig,
    SphericalHarrisDetector,
    SphericalHarrisKeypointSet,
    spherical_harris_response,
)


def _config(**changes: object) -> SphericalHarrisConfig:
    values: dict[str, object] = {
        "block_size": 5,
        "block_sigma_px": 1.2,
        "threshold_rel": 0.03,
        "nms_radius_px": 3,
        "max_keypoints": 64,
        "minimum_valid_support_fraction": 0.999,
        "selection_policy": "response",
        "backend": "numpy",
    }
    values.update(changes)
    return SphericalHarrisConfig(**values)  # type: ignore[arg-type]


def _rectangle(shape: tuple[int, int] = (65, 130)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    image[24:48, 45:87] = 255
    return image


def _assert_near_each(
    actual_xy: np.ndarray, expected_xy: tuple[tuple[float, float], ...], radius: float
) -> None:
    assert len(actual_xy) >= len(expected_xy)
    for expected in expected_xy:
        distance = np.linalg.norm(actual_xy - np.asarray(expected), axis=1)
        assert float(distance.min()) <= radius


def test_equatorial_rectangle_recovers_analytic_corners_not_edges() -> None:
    image = _rectangle()
    result = SphericalHarrisDetector(_config()).detect(
        image, panorama_id="equatorial-rectangle"
    )

    assert isinstance(result, SphericalHarrisKeypointSet)
    assert result.interface == "panorai-spherical-harris-detector/v1"
    assert result.panorama_id == "equatorial-rectangle"
    assert not hasattr(result, "descriptors")
    assert np.allclose(np.linalg.norm(result.bearings, axis=1), 1.0)
    _assert_near_each(
        result.source_erp_xy,
        ((45.0, 24.0), (86.0, 24.0), (45.0, 47.0), (86.0, 47.0)),
        radius=3.0,
    )

    edge = np.zeros_like(image)
    edge[:, 65:] = 255
    # A binary longitude split has two meridian boundaries that meet at both
    # poles. Exclude those endpoints so the oracle contains only isolated
    # one-dimensional edges, not their true spherical intersections.
    edge_validity = np.ones(edge.shape, dtype=bool)
    edge_validity[:8] = False
    edge_validity[-8:] = False
    edge_result = SphericalHarrisDetector(_config()).detect(
        edge, validity_mask=edge_validity
    )
    assert len(edge_result) == 0


def test_equatorial_localization_agrees_with_planar_opencv_oracle() -> None:
    import cv2

    image = _rectangle()
    spherical = spherical_harris_response(image, _config()).response
    planar = cv2.cornerHarris(
        image.astype(np.float32) / 255.0, blockSize=5, ksize=3, k=0.04
    )

    for x, y in ((45, 24), (86, 24), (45, 47), (86, 47)):
        region = np.s_[y - 4 : y + 5, x - 4 : x + 5]
        spherical_peak = np.unravel_index(
            np.nanargmax(spherical[region]), spherical[region].shape
        )
        planar_peak = np.unravel_index(np.argmax(planar[region]), planar[region].shape)
        assert np.linalg.norm(np.subtract(spherical_peak, planar_peak)) <= 2.0


def test_response_keeps_numeric_value_support_and_validity_separate() -> None:
    image = _rectangle()
    validity = np.ones(image.shape, dtype=bool)
    validity[18:34, 38:56] = False

    result = spherical_harris_response(image, _config(), validity_mask=validity)

    assert result.response.shape == image.shape
    assert result.support_fraction.shape == image.shape
    assert result.validity_mask.shape == image.shape
    assert result.validity_mask.dtype == np.bool_
    assert result.support_fraction.dtype == np.float32
    assert np.isnan(result.response[~result.validity_mask]).all()
    assert np.isfinite(result.response[result.validity_mask]).all()
    assert np.all((0.0 <= result.support_fraction) & (result.support_fraction <= 1.0))
    assert not result.validity_mask[24, 45]

    detected = SphericalHarrisDetector(_config()).detect(
        image, validity_mask=validity
    )
    assert np.all(detected.valid_support_fractions >= 0.999)
    assert np.linalg.norm(detected.source_erp_xy - (45.0, 24.0), axis=1).min() > 4.0


def test_invalid_boundary_on_constant_image_does_not_create_false_corners() -> None:
    image = np.full((65, 130), 127, dtype=np.uint8)
    validity = np.ones(image.shape, dtype=bool)
    validity[:, :65] = False

    result = SphericalHarrisDetector(_config()).detect(
        image, validity_mask=validity
    )

    assert len(result) == 0


def test_longitude_seam_corners_are_detected_on_both_sides() -> None:
    image = np.zeros((65, 130), dtype=np.uint8)
    image[23:49, :12] = 255
    image[23:49, 118:] = 255

    result = SphericalHarrisDetector(_config()).detect(image)

    _assert_near_each(
        result.source_erp_xy,
        ((11.0, 23.0), (118.0, 23.0), (11.0, 48.0), (118.0, 48.0)),
        radius=3.5,
    )


def test_response_and_detections_are_longitude_roll_equivariant() -> None:
    image = _rectangle()
    shift = 19
    detector = SphericalHarrisDetector(_config())

    original_response = detector.response(image)
    rolled_response = detector.response(np.roll(image, shift, axis=1))
    np.testing.assert_allclose(
        rolled_response.response,
        np.roll(original_response.response, shift, axis=1),
        rtol=1e-5,
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        rolled_response.validity_mask,
        np.roll(original_response.validity_mask, shift, axis=1),
    )

    original = detector.detect(image)
    rolled = detector.detect(np.roll(image, shift, axis=1))
    expected = original.source_erp_xy.copy()
    expected[:, 0] = np.mod(expected[:, 0] + shift, image.shape[1])
    assert len(original) == len(rolled)
    _assert_near_each(rolled.source_erp_xy, tuple(map(tuple, expected)), radius=0.1)
    np.testing.assert_allclose(
        np.sort(original.responses), np.sort(rolled.responses), rtol=1e-5, atol=1e-7
    )


def test_polar_response_is_finite_on_valid_support() -> None:
    rng = np.random.default_rng(27)
    image = rng.integers(0, 256, size=(48, 96), dtype=np.uint8)

    result = spherical_harris_response(image, _config(block_size=3))

    assert result.validity_mask[0].all()
    assert result.validity_mask[-1].all()
    assert np.isfinite(result.response[result.validity_mask]).all()


def test_selection_is_deterministic_and_bounded() -> None:
    yy, xx = np.indices((65, 130))
    image = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)
    detector = SphericalHarrisDetector(_config(max_keypoints=17))

    first = detector.detect(image, panorama_id="checker")
    second = detector.detect(image, panorama_id="checker")

    assert len(first) == 17
    np.testing.assert_array_equal(first.source_erp_xy, second.source_erp_xy)
    np.testing.assert_array_equal(first.responses, second.responses)
    assert first.describe() == second.describe()
    assert first.diagnostics.output_after_budget == 17


@pytest.mark.parametrize(
    "changes",
    (
        {"block_size": 4},
        {"block_sigma_px": 0.0},
        {"harris_k": 0.0},
        {"threshold_rel": 1.1},
        {"nms_radius_px": 0},
        {"max_keypoints": 0},
        {"minimum_valid_support_fraction": 1.1},
        {"selection_policy": "unknown"},
        {"backend": "unknown"},
    ),
)
def test_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**changes)
