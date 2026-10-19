from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from panorai.evaluation import (
    EvaluationProtocol,
    EvaluationRecord,
    evaluate_binary_regions,
    evaluate_features,
    evaluate_pose,
    evaluate_radial_range,
    solid_angle_weights,
)


def _independent_pixel_area(shape_hw: tuple[int, int], row: int) -> float:
    height, width = shape_hw
    north = math.pi / 2.0 - row * math.pi / height
    south = math.pi / 2.0 - (row + 1) * math.pi / height
    return (2.0 * math.pi / width) * (math.sin(north) - math.sin(south))


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def _equatorial_bearing(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray((math.sin(angle), 0.0, math.cos(angle)))


def test_exact_solid_angle_weights_integrate_to_four_pi() -> None:
    shape = (37, 74)
    weights = solid_angle_weights(shape)

    assert weights.shape == shape
    assert not weights.flags.writeable
    assert solid_angle_weights((np.int64(4), np.int64(8))).shape == (4, 8)
    assert float(weights.sum()) == pytest.approx(4.0 * math.pi, abs=1e-12)
    for row in (0, shape[0] // 2, shape[0] - 1):
        assert weights[row, 0] == pytest.approx(
            _independent_pixel_area(shape, row), abs=1e-15
        )


def test_region_metrics_are_area_weighted_and_validity_is_explicit() -> None:
    shape = (4, 8)
    reference = np.zeros(shape, dtype=bool)
    prediction = np.zeros(shape, dtype=bool)
    reference[0, 0] = True
    reference[2, 0] = True
    prediction[0, 0] = True
    prediction[2, 1] = True
    support = np.ones(shape, dtype=bool)
    prediction_validity = np.ones(shape, dtype=bool)
    prediction_validity[1, :] = False

    result = evaluate_binary_regions(
        prediction,
        reference,
        support=support,
        prediction_validity=prediction_validity,
        boundary_tolerance_deg=0.0,
    )

    polar = _independent_pixel_area(shape, 0)
    equatorial = _independent_pixel_area(shape, 2)
    expected_iou = polar / (polar + 2.0 * equatorial)
    assert result.iou == pytest.approx(expected_iou)
    assert result.intersection_solid_angle_sr == pytest.approx(polar)
    assert result.union_solid_angle_sr == pytest.approx(polar + 2.0 * equatorial)
    assert result.coverage < 1.0
    assert result.supported_solid_angle_sr == pytest.approx(4.0 * math.pi)
    assert result.comparable_solid_angle_sr == pytest.approx(
        4.0 * math.pi - 8.0 * _independent_pixel_area(shape, 1)
    )


def test_region_boundary_metric_is_seam_periodic_and_rotation_identity_is_exact() -> (
    None
):
    mask = np.zeros((18, 36), dtype=bool)
    mask[7:12, -2:] = True
    mask[7:12, :3] = True

    result = evaluate_binary_regions(mask, mask, boundary_tolerance_deg=0.0)

    assert result.iou == 1.0
    assert result.boundary_precision == 1.0
    assert result.boundary_recall == 1.0
    assert result.boundary_f1 == 1.0


def test_region_evaluation_rejects_non_boolean_masks_and_shape_mismatch() -> None:
    with pytest.raises(TypeError, match="boolean dtype"):
        evaluate_binary_regions(
            np.zeros((4, 8), dtype=np.uint8),
            np.zeros((4, 8), dtype=bool),
        )
    with pytest.raises(ValueError, match="shapes must match"):
        evaluate_binary_regions(
            np.zeros((4, 8), dtype=bool),
            np.zeros((5, 10), dtype=bool),
        )


def test_feature_evaluation_uses_one_to_one_angular_matching_and_scale_levels() -> None:
    reference = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    predicted = np.asarray(
        (
            (math.cos(math.radians(1)), math.sin(math.radians(1)), 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )

    result = evaluate_features(
        predicted,
        reference,
        threshold_deg=2.0,
        predicted_scales=np.asarray((2.0, 4.0, 8.0)),
        reference_scales=np.asarray((1.0, 4.0, 16.0)),
    )

    assert result.matched_count == 2
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.repeatability == pytest.approx(2 / 3)
    assert result.median_angular_error_deg == pytest.approx(0.5, abs=1e-10)
    assert result.maximum_angular_error_deg == pytest.approx(1.0, abs=1e-10)
    assert result.median_scale_error_levels == pytest.approx(0.5)


def test_feature_matching_maximizes_valid_cardinality_before_error() -> None:
    predicted = np.stack((_equatorial_bearing(0.0), _equatorial_bearing(2.0)))
    reference = np.stack((_equatorial_bearing(0.0), _equatorial_bearing(-4.0)))

    result = evaluate_features(predicted, reference, threshold_deg=5.0)

    assert result.matched_count == 2
    assert result.repeatability == 1.0
    assert result.matched_indices == ((0, 1), (1, 0))


def test_pose_evaluation_has_closed_form_so3_and_direction_oracles() -> None:
    result = evaluate_pose(
        _rotation_y(30.0),
        np.eye(3),
        estimated_translation_direction=np.asarray((1.0, 0.0, 0.0)),
        reference_translation_direction=np.asarray((0.0, 0.0, 1.0)),
    )

    assert result.rotation_error_deg == pytest.approx(30.0, abs=1e-12)
    assert result.translation_direction_error_deg == pytest.approx(90.0, abs=1e-12)

    sign_invariant = evaluate_pose(
        np.eye(3),
        np.eye(3),
        estimated_translation_direction=np.asarray((1.0, 0.0, 0.0)),
        reference_translation_direction=np.asarray((-1.0, 0.0, 0.0)),
        translation_sign_invariant=True,
    )
    assert sign_invariant.translation_direction_error_deg == 0.0


def test_depth_evaluation_preserves_valid_zero_and_weights_solid_angle() -> None:
    shape = (4, 8)
    reference = np.ones(shape, dtype=np.float64)
    prediction = reference.copy()
    reference[0, 0] = 0.0
    prediction[0, 0] = 0.0
    prediction[2, 0] = 2.0
    confidence = np.ones(shape, dtype=np.float64)
    confidence[2, 0] = 0.0

    result = evaluate_radial_range(
        prediction,
        reference,
        units="m",
        confidence=confidence,
    )

    equatorial = _independent_pixel_area(shape, 2)
    assert result.coverage == 1.0
    assert result.comparable_count == 32
    assert result.relative_count == 31
    assert result.relative_coverage < 1.0
    assert result.mae == pytest.approx(equatorial / (4.0 * math.pi))
    assert result.abs_rel == pytest.approx(
        equatorial / (4.0 * math.pi - _independent_pixel_area(shape, 0))
    )
    assert result.top_confidence_quartile_mae == 0.0
    assert result.bottom_confidence_quartile_mae is not None
    assert result.confidence_error_spearman is not None
    assert result.confidence_error_spearman < 0.0


def test_depth_evaluation_uses_masks_not_nan_or_numeric_zero_as_validity() -> None:
    prediction = np.zeros((4, 8), dtype=np.float64)
    reference = np.zeros((4, 8), dtype=np.float64)
    validity = np.ones((4, 8), dtype=bool)
    prediction[1, 2] = np.nan
    validity[1, 2] = False

    result = evaluate_radial_range(
        prediction,
        reference,
        units="m",
        prediction_validity=validity,
    )

    assert result.coverage < 1.0
    assert result.mae == 0.0
    assert result.abs_rel is None
    assert result.relative_count == 0
    with pytest.raises(ValueError, match="finite where prediction_validity"):
        evaluate_radial_range(prediction, reference, units="m")


def test_protocol_and_record_round_trip_canonical_json_without_imports() -> None:
    protocol = EvaluationProtocol(
        protocol_id="region-held-out-v1",
        version=1,
        evaluator="evaluate_binary_regions",
        metric_names=("iou", "boundary_f1"),
        parameters={"boundary_tolerance_deg": 1.0, "weighting": "solid-angle"},
    )
    record = EvaluationRecord(
        protocol=protocol,
        subject_id="scene-001",
        input_id="prediction-sha256:abc",
        reference_id="reference-sha256:def",
        split="held-out",
        algorithm_id="example-segmenter/v1",
        algorithm_config={"threshold": 0.5},
        package_version="3.8.0.dev0",
        backend="numpy",
        metrics={"iou": 0.75, "boundary_f1": 0.5},
        environment={"python": "3.12", "numpy": "2.0"},
    )

    payload = record.to_json()
    restored = EvaluationRecord.from_json(payload)

    assert payload == restored.to_json()
    assert restored.protocol.fingerprint == protocol.fingerprint
    assert restored.metrics == record.metrics
    assert "__class__" not in payload
    assert json.loads(payload)["geometry_contract"] == "geometry-v1"


def test_records_fail_closed_on_protocol_or_nonfinite_metric_changes() -> None:
    protocol = EvaluationProtocol(
        protocol_id="depth-v1",
        version=1,
        evaluator="evaluate_radial_range",
        metric_names=("mae",),
    )
    with pytest.raises(ValueError, match="exactly match"):
        EvaluationRecord(
            protocol=protocol,
            subject_id="subject",
            input_id="input",
            reference_id="reference",
            split="development",
            algorithm_id="algorithm",
            algorithm_config={},
            package_version="3.8.0.dev0",
            backend="numpy",
            metrics={"rmse": 1.0},
        )
    with pytest.raises(ValueError, match="finite or None"):
        EvaluationRecord(
            protocol=protocol,
            subject_id="subject",
            input_id="input",
            reference_id="reference",
            split="development",
            algorithm_id="algorithm",
            algorithm_config={},
            package_version="3.8.0.dev0",
            backend="numpy",
            metrics={"mae": float("nan")},
        )


def test_evaluation_import_keeps_torch_optional() -> None:
    before = set(sys.modules)
    __import__("panorai.evaluation")
    loaded = set(sys.modules) - before
    assert not any(name == "torch" or name.startswith("torch.") for name in loaded)
