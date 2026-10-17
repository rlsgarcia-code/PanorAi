"""Benchmark-only promotion protocol for dense radial-range estimates.

The protocol deliberately keeps semantic exclusion, geometric observability,
prediction validity, and numerical values separate.  Invalid predictions stay
in the accuracy denominator instead of disappearing from the reported metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels


RANGE_BANDS_M = (
    ("near_0_3_to_2_m", 0.3, 2.0, False),
    ("mid_2_to_5_m", 2.0, 5.0, False),
    ("far_5_to_12_m", 5.0, 12.0, True),
)
CONFIDENCE_RETENTION_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def _boolean_mask(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != shape:
        raise ValueError(f"{name} must be a boolean mask with shape {shape}")
    return result


@dataclass(frozen=True, slots=True)
class EvaluationPopulation:
    """Fixed non-sky, scanner-valid and geometrically observable population."""

    mask: np.ndarray
    audit: dict[str, int | float | str]


def build_evaluation_population(
    truth_range: np.ndarray,
    *,
    explicit_non_sky: np.ndarray,
    geometric_support: np.ndarray,
    observable: np.ndarray,
    min_range_m: float = 0.3,
    max_range_m: float = 12.0,
    non_sky_policy: str,
) -> EvaluationPopulation:
    """Build one prediction-independent evaluation population.

    ``explicit_non_sky`` must come from dataset semantics or an independently
    declared sensor-return proxy.  It is never inferred from RGB or from the
    method under evaluation.
    """

    truth = np.asarray(truth_range, dtype=np.float64)
    if truth.ndim != 2:
        raise ValueError("truth_range must have shape HW")
    if not 0.0 < min_range_m < max_range_m:
        raise ValueError("range bounds must satisfy 0 < min < max")
    if not isinstance(non_sky_policy, str) or not non_sky_policy.strip():
        raise ValueError("non_sky_policy must be a non-empty string")
    shape = truth.shape
    non_sky = _boolean_mask(explicit_non_sky, shape, "explicit_non_sky")
    support = _boolean_mask(geometric_support, shape, "geometric_support")
    visible = _boolean_mask(observable, shape, "observable")
    finite_range = np.isfinite(truth) & (truth >= min_range_m) & (truth <= max_range_m)
    population = non_sky & support & finite_range & visible
    total = truth.size
    audit: dict[str, int | float | str] = {
        "non_sky_policy": non_sky_policy,
        "total_pixels": total,
        "explicit_non_sky_pixels": int(non_sky.sum()),
        "geometric_support_pixels": int((non_sky & support).sum()),
        "finite_in_range_pixels": int((non_sky & support & finite_range).sum()),
        "observable_pixels": int(population.sum()),
        "excluded_as_sky_pixels": int((~non_sky).sum()),
        "evaluation_fraction_of_raster": float(population.sum() / total),
    }
    result = np.array(population, dtype=bool, copy=True)
    result.setflags(write=False)
    return EvaluationPopulation(result, audit)


def _summarize_mask(
    truth: np.ndarray,
    prediction: np.ndarray,
    population: np.ndarray,
    prediction_validity: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int | None]:
    denominator = int(population.sum())
    valid = (
        population & prediction_validity & np.isfinite(prediction) & (prediction > 0.0)
    )
    valid_count = int(valid.sum())
    population_weight = float(weights[population].sum())
    valid_weight = float(weights[valid].sum())
    if denominator == 0:
        return {
            "population": 0,
            "predicted_valid": 0,
            "population_weight": 0.0,
            "predicted_valid_weight": 0.0,
            "coverage": None,
            "delta_1_25": None,
            "median_abs_error_m_valid": None,
            "mean_abs_error_m_valid": None,
        }
    if population_weight <= 0.0:
        raise ValueError("population must have positive sample weight")
    absolute = np.abs(prediction[valid] - truth[valid])
    ratio = np.maximum(
        prediction[valid] / truth[valid], truth[valid] / prediction[valid]
    )
    selected_weights = weights[valid]
    return {
        "population": denominator,
        "predicted_valid": valid_count,
        "population_weight": population_weight,
        "predicted_valid_weight": valid_weight,
        "coverage": float(valid_weight / population_weight),
        # Missing predictions are failures by construction: the numerator is
        # valid-and-correct while the denominator is the fixed population.
        "delta_1_25": float(selected_weights[ratio < 1.25].sum() / population_weight),
        "median_abs_error_m_valid": (
            _weighted_quantile(absolute, selected_weights, 0.5) if valid_count else None
        ),
        "mean_abs_error_m_valid": (
            float(np.average(absolute, weights=selected_weights))
            if valid_count
            else None
        ),
    }


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = quantile * cumulative[-1]
    return float(ordered[np.searchsorted(cumulative, threshold, side="left")])


def summarize_range_prediction(
    truth_range: np.ndarray,
    predicted_range: np.ndarray,
    *,
    population: np.ndarray,
    prediction_validity: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report fixed-denominator overall and range-stratified accuracy."""

    truth = np.asarray(truth_range, dtype=np.float64)
    prediction = np.asarray(predicted_range, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth_range and predicted_range must share shape HW")
    mask = _boolean_mask(population, truth.shape, "population")
    valid = _boolean_mask(prediction_validity, truth.shape, "prediction_validity")
    if sample_weights is None:
        weights = np.ones(truth.shape, dtype=np.float64)
        weighting = "uniform"
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != truth.shape:
            raise ValueError("sample_weights must share shape HW")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be finite and non-negative")
        weighting = "provided"
    bands: dict[str, dict[str, float | int | None]] = {}
    for name, lower, upper, upper_inclusive in RANGE_BANDS_M:
        upper_mask = truth <= upper if upper_inclusive else truth < upper
        band = mask & (truth >= lower) & upper_mask
        bands[name] = _summarize_mask(truth, prediction, band, valid, weights)
    return {
        "weighting": weighting,
        "overall": _summarize_mask(truth, prediction, mask, valid, weights),
        "range_bands": bands,
    }


def summarize_confidence_selectivity(
    truth_range: np.ndarray,
    predicted_range: np.ndarray,
    confidence: np.ndarray,
    *,
    population: np.ndarray,
    prediction_validity: np.ndarray,
    sample_weights: np.ndarray | None = None,
    retention_fractions: tuple[float, ...] = CONFIDENCE_RETENTION_FRACTIONS,
) -> dict[str, Any]:
    """Measure confidence ranking without interpreting it as probability.

    Fractions retain a deterministic prefix of the valid predictions ordered
    by decreasing confidence. Fixed-population coverage and accuracy are
    reported beside conditional accuracy so abstention cannot hide missing
    predictions.
    """

    truth = np.asarray(truth_range, dtype=np.float64)
    prediction = np.asarray(predicted_range, dtype=np.float64)
    score = np.asarray(confidence, dtype=np.float64)
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("truth_range and predicted_range must share shape HW")
    if score.shape != truth.shape:
        raise ValueError("confidence must share shape HW")
    mask = _boolean_mask(population, truth.shape, "population")
    valid = _boolean_mask(prediction_validity, truth.shape, "prediction_validity")
    fractions = tuple(float(value) for value in retention_fractions)
    if (
        not fractions
        or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions
        )
        or any(right <= left for left, right in zip(fractions, fractions[1:]))
    ):
        raise ValueError(
            "retention_fractions must be finite, strictly increasing, and in (0, 1]"
        )
    if sample_weights is None:
        weights = np.ones(truth.shape, dtype=np.float64)
        weighting = "uniform"
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != truth.shape:
            raise ValueError("sample_weights must share shape HW")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be finite and non-negative")
        weighting = "provided"
    population_weight = float(weights[mask].sum())
    if population_weight <= 0.0:
        raise ValueError("population must have positive sample weight")
    eligible = (
        mask & valid & np.isfinite(prediction) & (prediction > 0.0) & np.isfinite(score)
    )
    flat_indices = np.flatnonzero(eligible)
    if flat_indices.size == 0:
        return {
            "weighting": weighting,
            "population_weight": population_weight,
            "prediction_weight": 0.0,
            "retention": {},
            "monotonic_conditional_delta_1_25": None,
            "top_quartile_conditional_delta_gain": None,
        }
    flat_score = score.ravel()[flat_indices]
    order = np.lexsort((flat_indices, -flat_score))
    ordered_indices = flat_indices[order]
    ordered_weights = weights.ravel()[ordered_indices]
    cumulative_weight = np.cumsum(ordered_weights)
    prediction_weight = float(cumulative_weight[-1])
    ordered_truth = truth.ravel()[ordered_indices]
    ordered_prediction = prediction.ravel()[ordered_indices]
    ratio = np.maximum(
        ordered_prediction / ordered_truth,
        ordered_truth / ordered_prediction,
    )
    correct = ratio < 1.25
    absolute_error = np.abs(ordered_prediction - ordered_truth)
    retention: dict[str, dict[str, float | int]] = {}
    conditional_values = []
    for fraction in fractions:
        target = fraction * prediction_weight
        end = int(np.searchsorted(cumulative_weight, target, side="left")) + 1
        end = min(end, ordered_indices.size)
        selected_weights = ordered_weights[:end]
        selected_weight = float(selected_weights.sum())
        correct_weight = float(selected_weights[correct[:end]].sum())
        conditional_delta = correct_weight / selected_weight
        conditional_values.append(conditional_delta)
        key = f"{fraction:.2f}"
        retention[key] = {
            "target_prediction_fraction": fraction,
            "selected_pixels": end,
            "selected_weight": selected_weight,
            "selected_correct_weight": correct_weight,
            "actual_prediction_weight_fraction": selected_weight / prediction_weight,
            "fixed_population_coverage": selected_weight / population_weight,
            "fixed_population_delta_1_25": correct_weight / population_weight,
            "conditional_delta_1_25": conditional_delta,
            "median_abs_error_m": _weighted_quantile(
                absolute_error[:end], selected_weights, 0.5
            ),
            "minimum_selected_confidence": float(flat_score[order[end - 1]]),
        }
    monotonic = all(
        left + 1e-12 >= right
        for left, right in zip(conditional_values, conditional_values[1:])
    )
    return {
        "weighting": weighting,
        "population_weight": population_weight,
        "prediction_weight": prediction_weight,
        "retention": retention,
        "monotonic_conditional_delta_1_25": monotonic,
        "top_quartile_conditional_delta_gain": (
            conditional_values[0] - conditional_values[-1]
        ),
    }


def same_surface_observability(
    reference_range: np.ndarray,
    source_range: np.ndarray,
    *,
    source_support: np.ndarray,
    rotation_source_from_reference: np.ndarray,
    translation_source_from_reference: np.ndarray,
    absolute_tolerance_m: float = 0.10,
    relative_tolerance: float = 0.05,
) -> np.ndarray:
    """Return reference pixels whose true surface is visible in the source.

    The oracle uses dataset depth and provided pose only. It is intentionally a
    benchmark operation and is never available to the stereo matcher.
    """

    truth = np.asarray(reference_range, dtype=np.float64)
    source = np.asarray(source_range, dtype=np.float64)
    if truth.ndim != 2 or source.ndim != 2:
        raise ValueError("reference_range and source_range must have shape HW")
    source_mask = _boolean_mask(source_support, source.shape, "source_support")
    rotation = np.asarray(rotation_source_from_reference, dtype=np.float64)
    translation = np.asarray(translation_source_from_reference, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("pose must contain a 3x3 rotation and 3-vector")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("pose must be finite")
    if not np.isfinite(absolute_tolerance_m) or absolute_tolerance_m < 0.0:
        raise ValueError("absolute_tolerance_m must be finite and non-negative")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")

    height, width = truth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((xx, yy), axis=-1)
    rays = np.asarray(erp_pixels_to_rays(pixels, truth.shape), dtype=np.float64)
    reference_valid = np.isfinite(truth) & (truth > 0.05)
    safe_truth = np.where(reference_valid, truth, 1.0)
    points_source = (rays * safe_truth[..., None]) @ rotation.T + translation
    projected = rays_to_erp_pixels(points_source, source.shape)
    projected_pixels = np.asarray(projected.pixels_xy, dtype=np.float64)
    finite_projected = np.isfinite(projected_pixels).all(axis=-1)
    safe_projected = np.where(finite_projected[..., None], projected_pixels, 0.0)
    source_height, source_width = source.shape
    source_x = np.mod(
        np.floor(safe_projected[..., 0] + 0.5).astype(np.int64),
        source_width,
    )
    source_y = np.floor(safe_projected[..., 1] + 0.5).astype(np.int64)
    geometric = (
        np.asarray(projected.valid, dtype=bool)
        & finite_projected
        & (source_y >= 0)
        & (source_y < source_height)
    )
    safe_y = np.clip(source_y, 0, source_height - 1)
    sampled_range = source[safe_y, source_x]
    sampled_support = source_mask[safe_y, source_x]
    expected_range = np.linalg.norm(points_source, axis=-1)
    tolerance = np.maximum(absolute_tolerance_m, relative_tolerance * expected_range)
    observable = (
        reference_valid
        & geometric
        & sampled_support
        & np.isfinite(sampled_range)
        & (sampled_range > 0.05)
        & (np.abs(sampled_range - expected_range) <= tolerance)
    )
    result = np.array(observable, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "CONFIDENCE_RETENTION_FRACTIONS",
    "EvaluationPopulation",
    "RANGE_BANDS_M",
    "build_evaluation_population",
    "same_surface_observability",
    "summarize_confidence_selectivity",
    "summarize_range_prediction",
]
