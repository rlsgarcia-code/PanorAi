"""Geometry-aware spherical evaluation functions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from panorai.geometry import erp_pixels_to_rays

from ._results import (
    BinaryRegionEvaluation,
    DepthEvaluation,
    FeatureEvaluation,
    PoseEvaluation,
)
from ._validation import (
    array,
    binary_mask,
    boolean_mask,
    finite_nonnegative,
    normalize_bearings,
    rotation_matrix,
    solid_angle_weights,
    unit_vector,
    weighted_mean,
)


def _ratio(numerator: float, denominator: float, *, empty: float) -> float:
    return numerator / denominator if denominator > 0.0 else empty


def _erp_rays(shape_hw: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape_hw)
    pixels = np.stack((columns, rows), axis=-1)
    return np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float64)


def _boundary(mask: np.ndarray, domain: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask)

    right_mask = np.roll(mask, -1, axis=1)
    right_domain = domain & np.roll(domain, -1, axis=1)
    horizontal = right_domain & (mask != right_mask)
    result |= horizontal
    result |= np.roll(horizontal, 1, axis=1)

    vertical_domain = domain[:-1] & domain[1:]
    vertical = vertical_domain & (mask[:-1] != mask[1:])
    result[:-1] |= vertical
    result[1:] |= vertical
    return result & domain


def _boundary_match(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rays: np.ndarray,
    tolerance_rad: float,
    weights: np.ndarray,
) -> float:
    source_count = int(np.count_nonzero(source))
    target_count = int(np.count_nonzero(target))
    if source_count == 0:
        return 1.0 if target_count == 0 else 0.0
    if target_count == 0:
        return 0.0
    tree = cKDTree(rays[target])
    chord, _ = tree.query(rays[source], k=1)
    angular = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    source_weights = weights[source]
    return float(
        np.sum(source_weights[angular <= tolerance_rad]) / source_weights.sum()
    )


def evaluate_binary_regions(
    prediction: Any,
    reference: Any,
    *,
    support: Any | None = None,
    prediction_validity: Any | None = None,
    reference_validity: Any | None = None,
    boundary_tolerance_deg: float = 1.0,
) -> BinaryRegionEvaluation:
    """Evaluate two binary canonical-ERP regions with exact area weights.

    ``support`` is geometric coverage.  The two validity masks are data
    validity and are intersected only for comparison; neither is inferred from
    the binary values.  Boundary matching uses great-circle distance between
    ERP pixel centres and longitude-periodic four-neighbour boundaries.
    """

    predicted = binary_mask(prediction, name="prediction")
    expected = binary_mask(reference, name="reference")
    if predicted.shape != expected.shape:
        raise ValueError("prediction and reference shapes must match")
    shape = tuple(int(item) for item in predicted.shape)
    geometric_support = boolean_mask(support, name="support", shape_hw=shape)
    predicted_valid = boolean_mask(
        prediction_validity,
        name="prediction_validity",
        shape_hw=shape,
    )
    reference_valid = boolean_mask(
        reference_validity,
        name="reference_validity",
        shape_hw=shape,
    )
    predicted_valid &= geometric_support
    reference_valid &= geometric_support
    comparable = predicted_valid & reference_valid
    weights = solid_angle_weights(shape)
    supported_area = float(np.sum(weights[geometric_support]))
    comparable_area = float(np.sum(weights[comparable]))

    selected_prediction = predicted & comparable
    selected_reference = expected & comparable
    intersection = selected_prediction & selected_reference
    union = selected_prediction | selected_reference
    prediction_area = float(np.sum(weights[selected_prediction]))
    reference_area = float(np.sum(weights[selected_reference]))
    intersection_area = float(np.sum(weights[intersection]))
    union_area = float(np.sum(weights[union]))

    tolerance = finite_nonnegative(
        boundary_tolerance_deg, name="boundary_tolerance_deg"
    )
    if tolerance > 180.0:
        raise ValueError("boundary_tolerance_deg must not exceed 180")
    predicted_boundary = _boundary(predicted, comparable)
    reference_boundary = _boundary(expected, comparable)
    rays = _erp_rays(shape)
    boundary_precision = _boundary_match(
        predicted_boundary,
        reference_boundary,
        rays=rays,
        tolerance_rad=math.radians(tolerance),
        weights=weights,
    )
    boundary_recall = _boundary_match(
        reference_boundary,
        predicted_boundary,
        rays=rays,
        tolerance_rad=math.radians(tolerance),
        weights=weights,
    )
    boundary_f1 = _ratio(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
        empty=1.0,
    )
    return BinaryRegionEvaluation(
        iou=_ratio(intersection_area, union_area, empty=1.0),
        dice=_ratio(
            2.0 * intersection_area,
            prediction_area + reference_area,
            empty=1.0,
        ),
        precision=_ratio(intersection_area, prediction_area, empty=1.0),
        recall=_ratio(intersection_area, reference_area, empty=1.0),
        coverage=_ratio(comparable_area, supported_area, empty=0.0),
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        supported_solid_angle_sr=supported_area,
        comparable_solid_angle_sr=comparable_area,
        prediction_solid_angle_sr=prediction_area,
        reference_solid_angle_sr=reference_area,
        intersection_solid_angle_sr=intersection_area,
        union_solid_angle_sr=union_area,
        boundary_tolerance_deg=tolerance,
    )


def _validity_vector(value: Any | None, *, name: str, count: int) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=bool)
    result = array(value, name=name, ndim=1)
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    return result


def _scale_vector(value: Any | None, *, name: str, count: int) -> np.ndarray | None:
    if value is None:
        return None
    result = array(value, name=name, dtype=np.float64, ndim=1)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def evaluate_features(
    predicted_bearings: Any,
    reference_bearings: Any,
    *,
    threshold_deg: float,
    prediction_validity: Any | None = None,
    reference_validity: Any | None = None,
    predicted_scales: Any | None = None,
    reference_scales: Any | None = None,
) -> FeatureEvaluation:
    """Compute deterministic one-to-one angular feature repeatability."""

    predicted_all = normalize_bearings(predicted_bearings, name="predicted_bearings")
    reference_all = normalize_bearings(reference_bearings, name="reference_bearings")
    predicted_valid = _validity_vector(
        prediction_validity,
        name="prediction_validity",
        count=len(predicted_all),
    )
    reference_valid = _validity_vector(
        reference_validity,
        name="reference_validity",
        count=len(reference_all),
    )
    predicted_scale_all = _scale_vector(
        predicted_scales, name="predicted_scales", count=len(predicted_all)
    )
    reference_scale_all = _scale_vector(
        reference_scales, name="reference_scales", count=len(reference_all)
    )
    if (predicted_scale_all is None) != (reference_scale_all is None):
        raise ValueError(
            "predicted_scales and reference_scales must be supplied together"
        )
    threshold = finite_nonnegative(threshold_deg, name="threshold_deg")
    if threshold > 180.0:
        raise ValueError("threshold_deg must not exceed 180")

    predicted_indices = np.flatnonzero(predicted_valid)
    reference_indices = np.flatnonzero(reference_valid)
    predicted = predicted_all[predicted_indices]
    reference = reference_all[reference_indices]
    matches: list[tuple[int, int]] = []
    errors: list[float] = []
    scale_errors: list[float] = []
    if len(predicted) and len(reference):
        angles = np.rad2deg(np.arccos(np.clip(predicted @ reference.T, -1.0, 1.0)))
        # Repeatability is cardinality under a frozen angular threshold;
        # angular error is only the secondary objective. A raw minimum-sum
        # assignment can sacrifice a valid pair for one exceptionally accurate
        # match plus one out-of-threshold match.
        invalid_penalty = (min(len(predicted), len(reference)) + 1) * 181.0
        assignment_cost = np.where(
            angles <= threshold,
            angles,
            invalid_penalty + angles,
        )
        rows, columns = linear_sum_assignment(assignment_cost)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            error = float(angles[row, column])
            if error > threshold:
                continue
            predicted_index = int(predicted_indices[row])
            reference_index = int(reference_indices[column])
            matches.append((predicted_index, reference_index))
            errors.append(error)
            if predicted_scale_all is not None and reference_scale_all is not None:
                scale_errors.append(
                    abs(
                        math.log2(
                            predicted_scale_all[predicted_index]
                            / reference_scale_all[reference_index]
                        )
                    )
                )
    count = len(matches)
    predicted_count = len(predicted)
    reference_count = len(reference)
    return FeatureEvaluation(
        predicted_count=predicted_count,
        reference_count=reference_count,
        matched_count=count,
        precision=_ratio(count, predicted_count, empty=1.0),
        recall=_ratio(count, reference_count, empty=1.0),
        repeatability=_ratio(count, min(predicted_count, reference_count), empty=1.0),
        median_angular_error_deg=(float(np.median(errors)) if errors else None),
        maximum_angular_error_deg=(max(errors) if errors else None),
        median_scale_error_levels=(
            float(np.median(scale_errors)) if scale_errors else None
        ),
        maximum_scale_error_levels=(max(scale_errors) if scale_errors else None),
        threshold_deg=threshold,
        matched_indices=tuple(matches),
    )


def evaluate_pose(
    estimated_rotation: Any,
    reference_rotation: Any,
    *,
    estimated_translation_direction: Any | None = None,
    reference_translation_direction: Any | None = None,
    translation_sign_invariant: bool = False,
) -> PoseEvaluation:
    """Evaluate source-from-reference rotation and optional translation direction."""

    estimated = rotation_matrix(estimated_rotation, name="estimated_rotation")
    reference = rotation_matrix(reference_rotation, name="reference_rotation")
    relative = estimated @ reference.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = math.degrees(math.acos(float(cosine)))
    if (estimated_translation_direction is None) != (
        reference_translation_direction is None
    ):
        raise ValueError(
            "estimated and reference translation directions must be supplied together"
        )
    translation_error = None
    if estimated_translation_direction is not None:
        estimated_t = unit_vector(
            estimated_translation_direction,
            name="estimated_translation_direction",
        )
        reference_t = unit_vector(
            reference_translation_direction,
            name="reference_translation_direction",
        )
        dot = float(estimated_t @ reference_t)
        if translation_sign_invariant:
            dot = abs(dot)
        translation_error = math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))
    return PoseEvaluation(
        rotation_error_deg=rotation_error,
        translation_direction_error_deg=translation_error,
        translation_sign_invariant=bool(translation_sign_invariant),
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = quantile * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(sorted_values[index])


def _weighted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    total = float(np.sum(weights))
    if len(first) < 2 or total <= 0.0:
        return None
    first_mean = float(np.sum(first * weights) / total)
    second_mean = float(np.sum(second * weights) / total)
    first_delta = first - first_mean
    second_delta = second - second_mean
    covariance = float(np.sum(weights * first_delta * second_delta) / total)
    first_variance = float(np.sum(weights * first_delta**2) / total)
    second_variance = float(np.sum(weights * second_delta**2) / total)
    denominator = math.sqrt(first_variance * second_variance)
    return covariance / denominator if denominator > 0.0 else None


def evaluate_radial_range(
    prediction: Any,
    reference: Any,
    *,
    units: str,
    support: Any | None = None,
    prediction_validity: Any | None = None,
    reference_validity: Any | None = None,
    confidence: Any | None = None,
) -> DepthEvaluation:
    """Evaluate radial range without inferring validity from numeric values."""

    predicted = array(prediction, name="prediction", dtype=np.float64, ndim=2)
    expected = array(reference, name="reference", dtype=np.float64, ndim=2)
    if predicted.shape != expected.shape or min(predicted.shape) < 2:
        raise ValueError("prediction and reference must share an HW shape")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must be a non-empty string")
    shape = tuple(int(item) for item in predicted.shape)
    geometric_support = boolean_mask(support, name="support", shape_hw=shape)
    predicted_valid = boolean_mask(
        prediction_validity,
        name="prediction_validity",
        shape_hw=shape,
    )
    reference_valid = boolean_mask(
        reference_validity,
        name="reference_validity",
        shape_hw=shape,
    )
    predicted_valid &= geometric_support
    reference_valid &= geometric_support
    if np.any(~np.isfinite(predicted[predicted_valid])):
        raise ValueError("prediction must be finite where prediction_validity is true")
    if np.any(~np.isfinite(expected[reference_valid])):
        raise ValueError("reference must be finite where reference_validity is true")
    if np.any(predicted[predicted_valid] < 0.0):
        raise ValueError("valid radial-range prediction values cannot be negative")
    if np.any(expected[reference_valid] < 0.0):
        raise ValueError("valid radial-range reference values cannot be negative")

    weights = solid_angle_weights(shape)
    evaluation_support = geometric_support & reference_valid
    comparable = evaluation_support & predicted_valid
    supported_area = float(np.sum(weights[evaluation_support]))
    comparable_area = float(np.sum(weights[comparable]))
    error = np.abs(predicted - expected)
    squared_error = error**2
    mae = weighted_mean(error, weights, comparable)
    mse = weighted_mean(squared_error, weights, comparable)

    relative = comparable & (expected > 0.0)
    relative_area = float(np.sum(weights[relative]))
    relative_error = np.zeros_like(expected)
    relative_error[relative] = error[relative] / expected[relative]
    abs_rel = weighted_mean(relative_error, weights, relative)
    ratio = np.ones_like(expected)
    positive_prediction = relative & (predicted > 0.0)
    ratio[positive_prediction] = np.maximum(
        predicted[positive_prediction] / expected[positive_prediction],
        expected[positive_prediction] / predicted[positive_prediction],
    )
    ratio[relative & ~positive_prediction] = np.inf

    confidence_spearman = None
    top_mae = None
    bottom_mae = None
    if confidence is not None:
        confidence_values = array(
            confidence, name="confidence", dtype=np.float64, ndim=2
        )
        if confidence_values.shape != shape:
            raise ValueError("confidence must match the depth shape")
        if np.any(~np.isfinite(confidence_values[comparable])):
            raise ValueError("confidence must be finite where depth is comparable")
        selected_confidence = confidence_values[comparable]
        selected_error = error[comparable]
        selected_weights = weights[comparable]
        if len(selected_confidence):
            low = _weighted_quantile(selected_confidence, selected_weights, 0.25)
            high = _weighted_quantile(selected_confidence, selected_weights, 0.75)
            bottom = comparable & (confidence_values <= low)
            top = comparable & (confidence_values >= high)
            bottom_mae = weighted_mean(error, weights, bottom)
            top_mae = weighted_mean(error, weights, top)
            confidence_spearman = _weighted_correlation(
                rankdata(selected_confidence, method="average"),
                rankdata(selected_error, method="average"),
                selected_weights,
            )

    def delta(threshold: float) -> float | None:
        values = np.zeros_like(expected)
        values[relative] = ratio[relative] < threshold
        return weighted_mean(values, weights, relative)

    return DepthEvaluation(
        coverage=_ratio(comparable_area, supported_area, empty=0.0),
        mae=mae,
        rmse=(math.sqrt(mse) if mse is not None else None),
        abs_rel=abs_rel,
        delta1=delta(1.25),
        delta2=delta(1.25**2),
        delta3=delta(1.25**3),
        relative_coverage=_ratio(relative_area, supported_area, empty=0.0),
        confidence_error_spearman=confidence_spearman,
        top_confidence_quartile_mae=top_mae,
        bottom_confidence_quartile_mae=bottom_mae,
        comparable_count=int(np.count_nonzero(comparable)),
        supported_count=int(np.count_nonzero(evaluation_support)),
        relative_count=int(np.count_nonzero(relative)),
        comparable_solid_angle_sr=comparable_area,
        supported_solid_angle_sr=supported_area,
        units=units.strip(),
    )
