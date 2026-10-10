"""Conservative native-ERP densification around immutable spherical anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AnchorDensificationOptions:
    angular_radius_deg: float = 1.05
    maximum_abs_log_correction: float = math.log(8.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.angular_radius_deg < 30.0:
            raise ValueError("angular_radius_deg must lie in (0, 30)")
        if (
            not math.isfinite(self.maximum_abs_log_correction)
            or self.maximum_abs_log_correction <= 0.0
        ):
            raise ValueError("maximum_abs_log_correction must be positive")


@dataclass(frozen=True, slots=True)
class AnchorDensificationResult:
    radial_range_m: np.ndarray
    support: np.ndarray
    log_correction: np.ndarray
    anchor_rows: np.ndarray
    anchor_columns: np.ndarray
    anchor_ranges_m: np.ndarray
    input_anchor_count: int
    unique_anchor_count: int


@dataclass(frozen=True, slots=True)
class HarmonicDensificationResult:
    radial_range_m: np.ndarray
    log_correction: np.ndarray
    degree: int
    coefficients: np.ndarray
    cross_validation_mae_log_by_degree: dict[int, float]
    anchor_mae_log: float


def _unit(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("anchor_bearings must have shape (N, 3)")
    norms = np.linalg.norm(array, axis=1)
    if not np.all(np.isfinite(array)) or np.any(norms <= 0.0):
        raise ValueError("anchor_bearings must be finite and non-zero")
    return array / norms[:, None]


def _anchor_cells(
    bearings: np.ndarray,
    ranges_m: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from panorai.geometry import rays_to_erp_pixels

    pixels = rays_to_erp_pixels(bearings, shape_hw).pixels_xy
    height, width = shape_hw
    columns = np.mod(np.rint(pixels[:, 0]).astype(np.int64), width)
    rows = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, height - 1)
    linear = rows * width + columns
    unique, inverse = np.unique(linear, return_inverse=True)
    result = np.empty(unique.size, dtype=np.float64)
    result_bearings = np.empty((unique.size, 3), dtype=np.float64)
    for index in range(unique.size):
        result[index] = float(np.exp(np.median(np.log(ranges_m[inverse == index]))))
        mean_bearing = bearings[inverse == index].mean(axis=0)
        result_bearings[index] = mean_bearing / np.linalg.norm(mean_bearing)
    return unique // width, unique % width, result, result_bearings


def densify_log_range_anchors(
    prior_radial_m: Any,
    anchor_bearings: Any,
    anchor_ranges_m: Any,
    *,
    options: AnchorDensificationOptions | None = None,
) -> AnchorDensificationResult:
    """Spread bounded anchor residuals locally while preserving hard anchors.

    The output shape is identical to the prior. Longitude wraps, latitude does
    not. Pixels outside the union of declared angular caps remain bit-identical
    to the input. Unique anchor cells are overwritten exactly after blending.
    """

    settings = options or AnchorDensificationOptions()
    prior = np.asarray(prior_radial_m)
    if prior.ndim != 2 or prior.dtype.kind != "f":
        raise ValueError("prior_radial_m must be a floating HW array")
    bearings = _unit(anchor_bearings)
    ranges = np.asarray(anchor_ranges_m, dtype=np.float64)
    if ranges.shape != (bearings.shape[0],):
        raise ValueError("anchor_ranges_m must have shape (N,)")
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("anchor ranges must be positive and finite")

    height, width = prior.shape
    anchor_rows, anchor_columns, cell_ranges, cell_bearings = _anchor_cells(
        bearings, ranges, prior.shape
    )
    prior_at_cells = prior[anchor_rows, anchor_columns].astype(np.float64)
    if np.any(~np.isfinite(prior_at_cells)) or np.any(prior_at_cells <= 0.0):
        raise ValueError("every anchor cell must have a valid positive prior")
    residuals = np.clip(
        np.log(cell_ranges / prior_at_cells),
        -settings.maximum_abs_log_correction,
        settings.maximum_abs_log_correction,
    )

    radius = math.radians(settings.angular_radius_deg)
    anchor_angles = np.arccos(np.clip(cell_bearings @ cell_bearings.T, -1.0, 1.0))
    anchor_q = np.clip(anchor_angles / radius, 0.0, 1.0)
    anchor_kernel = (1.0 - anchor_q) ** 4 * (4.0 * anchor_q + 1.0)
    anchor_kernel[anchor_angles >= radius] = 0.0
    try:
        coefficients = np.linalg.solve(anchor_kernel, residuals)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(anchor_kernel, residuals, rcond=1e-10)[0]

    correction = np.zeros(prior.shape, dtype=np.float32)
    support = np.zeros(prior.shape, dtype=bool)
    row_radius = int(math.ceil(radius / math.pi * height)) + 1
    for row, column, coefficient, ray in zip(
        anchor_rows, anchor_columns, coefficients, cell_bearings
    ):
        anchor_latitude = math.asin(float(np.clip(ray[1], -1.0, 1.0)))
        anchor_longitude = math.atan2(float(ray[0]), float(ray[2]))
        cosine_latitude = max(abs(math.cos(anchor_latitude)), 1e-3)
        column_radius = min(
            width // 2,
            int(math.ceil(radius / (2.0 * math.pi) * width / cosine_latitude)) + 1,
        )
        rows = np.arange(max(0, row - row_radius), min(height, row + row_radius + 1))
        columns_unwrapped = np.arange(
            column - column_radius, column + column_radius + 1
        )
        columns = np.mod(columns_unwrapped, width)
        latitudes = math.pi / 2.0 - (rows.astype(np.float64) + 0.5) / height * math.pi
        longitudes = (
            columns_unwrapped.astype(np.float64) + 0.5
        ) / width * 2.0 * math.pi - math.pi
        cosine = np.sin(latitudes)[:, None] * math.sin(anchor_latitude) + np.cos(
            latitudes
        )[:, None] * math.cos(anchor_latitude) * np.cos(
            longitudes[None, :] - anchor_longitude
        )
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        local = angles < radius
        q = np.clip(angles / radius, 0.0, 1.0)
        kernel = (1.0 - q) ** 4 * (4.0 * q + 1.0) * local
        for local_row, output_row in enumerate(rows):
            np.add.at(
                correction[output_row],
                columns,
                (kernel[local_row] * coefficient).astype(np.float32),
            )
            support[output_row, columns[kernel[local_row] > 0.0]] = True

    correction[support] = np.clip(
        correction[support],
        -settings.maximum_abs_log_correction,
        settings.maximum_abs_log_correction,
    )
    result = np.array(prior, copy=True)
    valid_support = support & np.isfinite(prior) & (prior > 0.0)
    result[valid_support] = (
        prior[valid_support] * np.exp(correction[valid_support])
    ).astype(prior.dtype)
    result[anchor_rows, anchor_columns] = cell_ranges.astype(prior.dtype)
    support[anchor_rows, anchor_columns] = True
    correction[anchor_rows, anchor_columns] = np.log(
        cell_ranges / prior_at_cells
    ).astype(np.float32)
    return AnchorDensificationResult(
        radial_range_m=result,
        support=support,
        log_correction=correction,
        anchor_rows=anchor_rows,
        anchor_columns=anchor_columns,
        anchor_ranges_m=cell_ranges,
        input_anchor_count=int(bearings.shape[0]),
        unique_anchor_count=int(cell_ranges.size),
    )


def _harmonic_basis(bearings: np.ndarray, degree: int) -> np.ndarray:
    if degree not in (0, 1, 2):
        raise ValueError("harmonic degree must be 0, 1, or 2")
    x, y, z = bearings.T
    columns = [np.ones(x.shape, dtype=np.float64)]
    if degree >= 1:
        columns.extend((x, y, z))
    if degree >= 2:
        columns.extend((x * y, x * z, y * z, x * x - y * y, 3.0 * z * z - 1.0))
    return np.column_stack(columns)


def _ridge_fit(
    basis: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    effective = np.ones(target.shape, dtype=np.float64) if weights is None else weights
    weighted_basis = basis * np.sqrt(effective)[:, None]
    weighted_target = target * np.sqrt(effective)
    regularizer = np.eye(basis.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        weighted_basis.T @ weighted_basis + regularizer,
        weighted_basis.T @ weighted_target,
    )


def _robust_harmonic_fit(
    basis: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    huber_delta_log: float,
    iterations: int = 5,
) -> np.ndarray:
    coefficients = _ridge_fit(basis, target, ridge=ridge)
    for _ in range(iterations):
        residual = basis @ coefficients - target
        magnitude = np.abs(residual)
        weights = np.ones_like(magnitude)
        selected = magnitude > huber_delta_log
        weights[selected] = huber_delta_log / magnitude[selected]
        coefficients = _ridge_fit(basis, target, ridge=ridge, weights=weights)
    return coefficients


def densify_harmonic_log_range(
    prior_radial_m: Any,
    anchor_bearings: Any,
    anchor_ranges_m: Any,
    *,
    maximum_degree: int = 2,
    ridge: float = 0.10,
    huber_delta_log: float = 0.20,
    maximum_abs_log_correction: float = math.log(8.0),
    row_chunk: int = 64,
) -> HarmonicDensificationResult:
    """Fit and apply a low-frequency spherical log-range correction.

    Degree is selected only from leave-one-anchor-out error. Ground truth and
    dense image evidence are not inputs. Degree zero is always a candidate.
    """

    prior = np.asarray(prior_radial_m)
    if prior.ndim != 2 or prior.dtype.kind != "f":
        raise ValueError("prior_radial_m must be a floating HW array")
    if maximum_degree not in (0, 1, 2):
        raise ValueError("maximum_degree must be 0, 1, or 2")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be positive and finite")
    if not math.isfinite(huber_delta_log) or huber_delta_log <= 0.0:
        raise ValueError("huber_delta_log must be positive and finite")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    bearings = _unit(anchor_bearings)
    ranges = np.asarray(anchor_ranges_m, dtype=np.float64)
    if ranges.shape != (bearings.shape[0],):
        raise ValueError("anchor_ranges_m must have shape (N,)")
    from panorai.geometry import rays_to_erp_pixels

    pixels = rays_to_erp_pixels(bearings, prior.shape).pixels_xy
    columns = np.mod(np.rint(pixels[:, 0]).astype(np.int64), prior.shape[1])
    rows = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, prior.shape[0] - 1)
    prior_samples = prior[rows, columns].astype(np.float64)
    valid = (
        np.isfinite(prior_samples)
        & (prior_samples > 0.0)
        & np.isfinite(ranges)
        & (ranges > 0.0)
    )
    bearings = bearings[valid]
    target = np.log(ranges[valid] / prior_samples[valid])
    if target.size < 6:
        raise ValueError("at least six valid anchors are required")

    cv: dict[int, float] = {}
    for degree in range(maximum_degree + 1):
        basis = _harmonic_basis(bearings, degree)
        errors = np.empty(target.size, dtype=np.float64)
        for heldout in range(target.size):
            keep = np.arange(target.size) != heldout
            coefficients = _ridge_fit(basis[keep], target[keep], ridge=ridge)
            errors[heldout] = abs(
                float(basis[heldout] @ coefficients - target[heldout])
            )
        cv[degree] = float(np.median(errors))
    best = min(cv, key=lambda degree: (cv[degree], degree))
    basis = _harmonic_basis(bearings, best)
    coefficients = _robust_harmonic_fit(
        basis,
        target,
        ridge=ridge,
        huber_delta_log=huber_delta_log,
    )

    height, width = prior.shape
    longitude = (
        np.arange(width, dtype=np.float64) + 0.5
    ) / width * 2.0 * math.pi - math.pi
    correction = np.empty(prior.shape, dtype=np.float32)
    result = np.array(prior, copy=True)
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        latitude = (
            math.pi / 2.0
            - (np.arange(start, stop, dtype=np.float64) + 0.5) / height * math.pi
        )
        cosine_latitude = np.cos(latitude)[:, None]
        rays = np.stack(
            np.broadcast_arrays(
                cosine_latitude * np.sin(longitude)[None, :],
                np.sin(latitude)[:, None],
                cosine_latitude * np.cos(longitude)[None, :],
            ),
            axis=2,
        ).reshape(-1, 3)
        local = np.clip(
            _harmonic_basis(rays, best) @ coefficients,
            -maximum_abs_log_correction,
            maximum_abs_log_correction,
        ).reshape(stop - start, width)
        correction[start:stop] = local.astype(np.float32)
        source = np.asarray(prior[start:stop])
        selected = np.isfinite(source) & (source > 0.0)
        result_chunk = result[start:stop]
        result_chunk[selected] = (source[selected] * np.exp(local[selected])).astype(
            prior.dtype
        )
    anchor_residual = basis @ coefficients - target
    return HarmonicDensificationResult(
        radial_range_m=result,
        log_correction=correction,
        degree=best,
        coefficients=coefficients,
        cross_validation_mae_log_by_degree=cv,
        anchor_mae_log=float(np.mean(np.abs(anchor_residual))),
    )


__all__ = [
    "AnchorDensificationOptions",
    "AnchorDensificationResult",
    "HarmonicDensificationResult",
    "densify_log_range_anchors",
    "densify_harmonic_log_range",
]
