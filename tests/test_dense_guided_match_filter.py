from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.features import MatchProvenance, SphericalFeatureMatches
from panorai.geometry import erp_pixels_to_rays
from panorai.stereo import (
    DenseMatchFilterOptions,
    SphericalStereoOptions,
    SphericalStereoResult,
    filter_matches_by_dense_range,
)


def _matches(
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> SphericalFeatureMatches:
    count = bearings_a.shape[0]
    return SphericalFeatureMatches(
        panorama_id_a="a",
        panorama_id_b="b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.linspace(0.1, 0.2, count, dtype=np.float32),
        ratio_scores=np.full(count, 0.5, dtype=np.float32),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool) if valid is None else valid,
        matcher_name="analytic",
        matcher_config={},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-features/v1",
            source_checksums=("a", "b"),
            face_pairs=tuple(("erp", "erp") for _ in range(count)),
            face_pair_groups=tuple((("erp", "erp"),) for _ in range(count)),
            deduplicated=False,
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "erp", dtype=object),
        face_ids_b=np.full(count, "erp", dtype=object),
    )


def _dense_result(
    range_map: np.ndarray,
    validity: np.ndarray,
    confidence: np.ndarray,
) -> SphericalStereoResult:
    shape = range_map.shape
    return SphericalStereoResult(
        range=range_map,
        validity_mask=validity,
        confidence=confidence,
        matching_cost=np.full(shape, 0.1, dtype=np.float32),
        hypothesis_index=np.ones(shape, dtype=np.int32),
        inverse_range_hypotheses=np.asarray((0.1, 0.2, 0.3), dtype=np.float32),
        options=SphericalStereoOptions(
            min_range=1.0,
            max_range=10.0,
            num_hypotheses=3,
            bidirectional_consistency=False,
        ),
    )


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def _transform_bearings(
    bearings: np.ndarray,
    ranges: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    points = bearings * ranges[:, None]
    target = points @ rotation.T + translation
    return target / np.linalg.norm(target, axis=1, keepdims=True)


def test_dense_filter_classifies_accept_reject_and_unsupported_without_mutation() -> (
    None
):
    shape = (8, 16)
    source_pixels = np.asarray(((0, 3), (15, 3), (4, 3), (8, 3), (10, 3)))
    bearings_a = erp_pixels_to_rays(source_pixels.astype(np.float64), shape)
    rotation = _rotation_y(7.0)
    translation = np.asarray((0.4, -0.1, 0.2))
    ranges = np.full(len(source_pixels), 4.0)
    bearings_b = _transform_bearings(bearings_a, ranges, rotation, translation)
    bearings_b[1] = _rotation_y(8.0) @ bearings_b[1]
    matches = _matches(
        bearings_a,
        bearings_b,
        valid=np.asarray((True, True, True, True, False)),
    )
    original_a = matches.bearings_a.copy()
    original_b = matches.bearings_b.copy()

    range_map = np.full(shape, 4.0, dtype=np.float32)
    validity = np.ones(shape, dtype=bool)
    validity[3, 4] = False
    confidence = np.ones(shape, dtype=np.float32)
    confidence[3, 8] = 0.2
    dense = _dense_result(range_map, validity, confidence)

    result = filter_matches_by_dense_range(
        matches,
        dense,
        rotation,
        translation,
        options=DenseMatchFilterOptions(
            max_angular_error_deg=1.0,
            min_dense_confidence=0.5,
            min_valid_weight=0.99,
        ),
    )

    assert result.accepted_mask.tolist() == [True, False, False, False, False]
    assert result.rejected_mask.tolist() == [False, True, False, False, False]
    assert result.unsupported_mask.tolist() == [False, False, True, True, True]
    assert result.input_valid_mask.tolist() == [True, True, True, True, False]
    assert math.degrees(result.angular_residual_rad[0]) < 1e-5
    assert math.degrees(result.angular_residual_rad[1]) > 7.0
    np.testing.assert_array_equal(matches.bearings_a, original_a)
    np.testing.assert_array_equal(matches.bearings_b, original_b)
    assert not result.accepted_mask.flags.writeable
    assert result.describe()["unsupported_count"] == 3


def test_dense_filter_wraps_bilinear_sampling_across_erp_seam() -> None:
    shape = (8, 16)
    source_pixels = np.asarray(((-0.25, 3.0),), dtype=np.float64)
    bearings_a = erp_pixels_to_rays(source_pixels, shape)
    rotation = np.eye(3)
    translation = np.asarray((0.5, 0.0, 0.0))
    bearings_b = _transform_bearings(
        bearings_a, np.asarray((4.0,)), rotation, translation
    )
    matches = _matches(bearings_a, bearings_b)
    range_map = np.full(shape, np.nan, dtype=np.float32)
    validity = np.zeros(shape, dtype=bool)
    confidence = np.zeros(shape, dtype=np.float32)
    range_map[3, 15] = 4.0
    validity[3, 15] = True
    confidence[3, 15] = 0.8
    dense = _dense_result(range_map, validity, confidence)

    permissive = filter_matches_by_dense_range(
        matches,
        dense,
        rotation,
        translation,
        options=DenseMatchFilterOptions(min_valid_weight=0.2),
    )
    strict = filter_matches_by_dense_range(
        matches,
        dense,
        rotation,
        translation,
        options=DenseMatchFilterOptions(min_valid_weight=0.5),
    )

    assert permissive.sampled_valid_weight[0] == pytest.approx(0.25, abs=1e-12)
    assert permissive.sampled_range[0] == pytest.approx(4.0)
    assert permissive.accepted_mask.tolist() == [True]
    assert strict.unsupported_mask.tolist() == [True]


def test_filter_result_integrates_with_bearing_correspondence_contract() -> None:
    shape = (8, 16)
    pixels = np.asarray(((2, 3), (7, 4)), dtype=np.float64)
    bearings_a = erp_pixels_to_rays(pixels, shape)
    rotation = np.eye(3)
    translation = np.asarray((0.4, 0.0, 0.0))
    bearings_b = _transform_bearings(bearings_a, np.full(2, 3.0), rotation, translation)
    matches = _matches(bearings_a, bearings_b)
    dense = _dense_result(
        np.full(shape, 3.0, dtype=np.float32),
        np.ones(shape, dtype=bool),
        np.ones(shape, dtype=np.float32),
    )

    result = filter_matches_by_dense_range(matches, dense, rotation, translation)
    correspondences = result.to_bearing_correspondences()

    assert correspondences.valid.tolist() == [True, True]
    assert correspondences.weights.tolist() == [1.0, 1.0]
    np.testing.assert_array_equal(correspondences.bearings_a, matches.bearings_a)
    np.testing.assert_array_equal(correspondences.bearings_b, matches.bearings_b)
    correspondences.bearings_a[0, 0] = 99.0
    assert matches.bearings_a[0, 0] != 99.0


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"max_angular_error_deg": 0.0}, "max_angular_error_deg"),
        ({"max_angular_error_deg": 181.0}, "max_angular_error_deg"),
        ({"min_dense_confidence": -0.1}, "min_dense_confidence"),
        ({"min_valid_weight": 0.0}, "min_valid_weight"),
        ({"min_valid_weight": 1.1}, "min_valid_weight"),
    ),
)
def test_dense_filter_options_reject_invalid_thresholds(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DenseMatchFilterOptions(**kwargs)


def test_dense_filter_rejects_zero_baseline() -> None:
    shape = (8, 16)
    bearings = erp_pixels_to_rays(np.asarray(((2.0, 3.0),)), shape)
    matches = _matches(bearings, bearings)
    dense = _dense_result(
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=bool),
        np.ones(shape, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="non-zero scale"):
        filter_matches_by_dense_range(matches, dense, np.eye(3), np.zeros(3))
