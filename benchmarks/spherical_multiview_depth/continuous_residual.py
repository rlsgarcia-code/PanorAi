"""Continuous edge-aware log-depth residuals on a periodic spherical grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


INTERFACE = "panorai-experimental-continuous-grid-residual/v1"


@dataclass(frozen=True, slots=True)
class ContinuousResidualOptions:
    """Weights for a deterministic quadratic residual-field solve."""

    seed_weight: float = 64.0
    anchor_weight: float = 1.0
    smoothness_weight: float = 12.0
    color_sigma: float = 0.12
    log_depth_sigma: float = 0.25
    minimum_edge_weight: float = 0.02
    latitude_cosine_floor: float = 0.10
    max_abs_log_residual: float = math.log(1.5)
    min_range_m: float = 0.3
    max_range_m: float = 15.0
    cg_rtol: float = 1e-8
    cg_maxiter: int = 2_000
    native_row_chunk: int = 64

    def __post_init__(self) -> None:
        for name in (
            "seed_weight",
            "anchor_weight",
            "smoothness_weight",
            "color_sigma",
            "log_depth_sigma",
            "latitude_cosine_floor",
            "max_abs_log_residual",
            "min_range_m",
            "max_range_m",
            "cg_rtol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")
        if not 0.0 <= self.minimum_edge_weight <= 1.0:
            raise ValueError("minimum_edge_weight must lie in [0, 1]")
        if self.latitude_cosine_floor > 1.0:
            raise ValueError("latitude_cosine_floor must not exceed 1")
        for name in ("cg_maxiter", "native_row_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContinuousResidualResult:
    """Native prediction and its solved coarse spherical correction field."""

    radial_range_m: np.ndarray
    native_log_residual: np.ndarray
    grid_log_residual: np.ndarray
    grid_rows: np.ndarray
    grid_columns: np.ndarray
    accepted_seed_count: int
    diagnostics: dict[str, Any]
    options: ContinuousResidualOptions
    interface: str = INTERFACE


@dataclass(frozen=True, slots=True)
class _GridLayout:
    rows: np.ndarray
    columns: np.ndarray
    row_index: np.ndarray
    column_index: np.ndarray
    stride: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.rows.size), int(self.columns.size))


def solve_continuous_grid_residual(
    prior_range_m: Any,
    target_rgb: Any,
    target_validity: Any,
    rows: Any,
    columns: Any,
    proposed_range_m: Any,
    accepted: Any,
    confidence: Any,
    *,
    options: ContinuousResidualOptions | None = None,
) -> ContinuousResidualResult:
    """Fit a periodic, edge-aware log-depth correction and evaluate it natively.

    The input RGB and radial-range arrays remain on their original ERP lattice.
    Only the correction field is parameterized on the supplied regular grid.
    """

    settings = options or ContinuousResidualOptions()
    prior = np.asarray(prior_range_m)
    rgb = np.asarray(target_rgb)
    validity = np.asarray(target_validity, dtype=bool)
    if prior.ndim != 2:
        raise ValueError("prior_range_m must have shape (H, W)")
    if rgb.shape != (*prior.shape, 3):
        raise ValueError("target_rgb must have shape (H, W, 3)")
    if validity.shape != prior.shape:
        raise ValueError("target_validity must match prior_range_m")
    if not np.issubdtype(prior.dtype, np.floating):
        raise TypeError("prior_range_m must use a floating dtype")
    if not np.isfinite(prior).all():
        raise ValueError("prior_range_m must be finite")
    if (prior <= 0.0).any():
        raise ValueError("prior_range_m must be positive")

    row_array = np.asarray(rows, dtype=np.int64)
    column_array = np.asarray(columns, dtype=np.int64)
    proposed = np.asarray(proposed_range_m, dtype=np.float64)
    seed_mask = np.asarray(accepted, dtype=bool)
    seed_confidence = np.asarray(confidence, dtype=np.float64)
    count = row_array.size
    for name, value in (
        ("columns", column_array),
        ("proposed_range_m", proposed),
        ("accepted", seed_mask),
        ("confidence", seed_confidence),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
    if row_array.shape != (count,):
        raise ValueError("rows must have shape (N,)")
    if not seed_mask.any():
        raise ValueError("at least one accepted proposal is required")
    if not np.isfinite(proposed[seed_mask]).all() or (proposed[seed_mask] <= 0.0).any():
        raise ValueError("accepted proposed ranges must be positive and finite")
    if (
        not np.isfinite(seed_confidence).all()
        or (seed_confidence < 0.0).any()
        or (seed_confidence > 1.0).any()
    ):
        raise ValueError("confidence must be finite and lie in [0, 1]")

    layout = _regular_grid_layout(row_array, column_array, prior.shape)
    grid_shape = layout.shape
    flat_index = layout.row_index * grid_shape[1] + layout.column_index
    grid_prior = prior[row_array, column_array].astype(np.float64)
    grid_validity = validity[row_array, column_array]
    grid_rgb = _normalize_rgb(rgb[row_array, column_array])

    prior_field = _scatter_grid(grid_prior, flat_index, grid_shape)
    validity_field = _scatter_grid(grid_validity, flat_index, grid_shape).astype(bool)
    rgb_field = np.empty((*grid_shape, 3), dtype=np.float64)
    for channel in range(3):
        rgb_field[..., channel] = _scatter_grid(
            grid_rgb[:, channel], flat_index, grid_shape
        )
    accepted_field = _scatter_grid(seed_mask, flat_index, grid_shape).astype(bool)
    confidence_field = _scatter_grid(seed_confidence, flat_index, grid_shape)
    proposal_field = _scatter_grid(proposed, flat_index, grid_shape)
    accepted_field &= validity_field

    seed_delta = np.zeros(grid_shape, dtype=np.float64)
    seed_delta[accepted_field] = np.log(
        np.clip(
            proposal_field[accepted_field],
            settings.min_range_m,
            settings.max_range_m,
        )
        / prior_field[accepted_field]
    )
    seed_delta = np.clip(
        seed_delta,
        -settings.max_abs_log_residual,
        settings.max_abs_log_residual,
    )
    data_weight = (
        settings.seed_weight * confidence_field * accepted_field.astype(np.float64)
    )

    edge_i, edge_j, edge_weight = _spherical_edges(
        prior_field,
        rgb_field,
        validity_field,
        layout.rows,
        prior.shape,
        settings,
    )
    matrix, right_hand_side = _normal_equations(
        grid_shape,
        edge_i,
        edge_j,
        edge_weight,
        data_weight,
        seed_delta,
        settings,
    )
    iteration_count = 0

    def count_iteration(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    inverse_diagonal = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda value: value / matrix.diagonal(),
        dtype=np.float64,
    )
    solved, solver_info = sparse_linalg.cg(
        matrix,
        right_hand_side,
        rtol=settings.cg_rtol,
        atol=0.0,
        maxiter=settings.cg_maxiter,
        M=inverse_diagonal,
        callback=count_iteration,
    )
    if solver_info != 0:
        raise RuntimeError(
            f"continuous residual conjugate gradient failed with info={solver_info}"
        )
    solved = np.clip(
        solved.reshape(grid_shape),
        -settings.max_abs_log_residual,
        settings.max_abs_log_residual,
    )
    native_residual = interpolate_periodic_grid_to_native(
        solved,
        layout.rows,
        layout.columns,
        prior.shape,
        row_chunk=settings.native_row_chunk,
    )
    prediction = np.clip(
        prior.astype(np.float32, copy=False) * np.exp(native_residual),
        settings.min_range_m,
        settings.max_range_m,
    ).astype(np.float32, copy=False)
    data_error = solved[accepted_field] - seed_delta[accepted_field]
    flattened = solved.ravel()
    pair_difference = flattened[edge_i] - flattened[edge_j]
    linear_residual = matrix @ solved.ravel() - right_hand_side
    diagnostics = {
        "grid_shape": list(grid_shape),
        "grid_stride_px": layout.stride,
        "accepted_seed_count": int(accepted_field.sum()),
        "edge_count": int(edge_weight.size),
        "cg_iterations": iteration_count,
        "cg_info": int(solver_info),
        "linear_relative_residual": float(
            np.linalg.norm(linear_residual)
            / max(np.linalg.norm(right_hand_side), np.finfo(np.float64).eps)
        ),
        "weighted_seed_rmse_log": float(
            np.sqrt(
                np.average(
                    data_error * data_error,
                    weights=np.maximum(
                        confidence_field[accepted_field],
                        np.finfo(np.float64).eps,
                    ),
                )
            )
        ),
        "maximum_abs_grid_log_residual": float(np.max(np.abs(solved))),
        "median_abs_grid_log_residual": float(np.median(np.abs(solved))),
        "data_energy": float(np.sum(data_weight * (solved - seed_delta) ** 2)),
        "anchor_energy": float(settings.anchor_weight * np.sum(solved * solved)),
        "smoothness_energy": float(np.sum(edge_weight * pair_difference**2)),
    }
    return ContinuousResidualResult(
        radial_range_m=prediction,
        native_log_residual=native_residual,
        grid_log_residual=solved.astype(np.float32),
        grid_rows=layout.rows.copy(),
        grid_columns=layout.columns.copy(),
        accepted_seed_count=int(accepted_field.sum()),
        diagnostics=diagnostics,
        options=settings,
    )


def interpolate_periodic_grid_to_native(
    grid_values: Any,
    grid_rows: Any,
    grid_columns: Any,
    output_shape_hw: tuple[int, int],
    *,
    row_chunk: int = 64,
) -> np.ndarray:
    """Evaluate a regular correction lattice on native ERP pixel centres."""

    values = np.asarray(grid_values, dtype=np.float64)
    rows = np.asarray(grid_rows, dtype=np.int64)
    columns = np.asarray(grid_columns, dtype=np.int64)
    if values.shape != (rows.size, columns.size):
        raise ValueError("grid_values must match grid_rows x grid_columns")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    height, width = output_shape_hw
    layout = _regular_grid_layout_from_axes(rows, columns, output_shape_hw)
    stride = layout.stride
    column_coordinate = np.mod(
        (np.arange(width, dtype=np.float64) - columns[0]) / stride,
        columns.size,
    )
    column0 = np.floor(column_coordinate).astype(np.int64)
    column1 = (column0 + 1) % columns.size
    column_weight = (column_coordinate - column0).astype(np.float64)
    output = np.empty((height, width), dtype=np.float32)
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        row_coordinate = np.clip(
            (np.arange(start, stop, dtype=np.float64) - rows[0]) / stride,
            0.0,
            rows.size - 1.0,
        )
        row0 = np.floor(row_coordinate).astype(np.int64)
        row1 = np.minimum(row0 + 1, rows.size - 1)
        row_weight = row_coordinate - row0
        top = (
            values[row0[:, None], column0[None, :]] * (1.0 - column_weight[None, :])
            + values[row0[:, None], column1[None, :]] * column_weight[None, :]
        )
        bottom = (
            values[row1[:, None], column0[None, :]] * (1.0 - column_weight[None, :])
            + values[row1[:, None], column1[None, :]] * column_weight[None, :]
        )
        output[start:stop] = (
            top * (1.0 - row_weight[:, None]) + bottom * row_weight[:, None]
        ).astype(np.float32)
    return output


def _regular_grid_layout(
    rows: np.ndarray,
    columns: np.ndarray,
    shape_hw: tuple[int, int],
) -> _GridLayout:
    if rows.ndim != 1 or columns.ndim != 1 or rows.shape != columns.shape:
        raise ValueError("rows and columns must have matching shape (N,)")
    if not rows.size:
        raise ValueError("the correction grid cannot be empty")
    height, width = shape_hw
    if (
        (rows < 0).any()
        or (rows >= height).any()
        or (columns < 0).any()
        or (columns >= width).any()
    ):
        raise ValueError("grid coordinates must lie inside the ERP")
    row_axis = np.unique(rows)
    column_axis = np.unique(columns)
    layout = _regular_grid_layout_from_axes(row_axis, column_axis, shape_hw)
    if row_axis.size * column_axis.size != rows.size:
        raise ValueError("rows and columns must describe a complete Cartesian grid")
    row_index = np.searchsorted(row_axis, rows)
    column_index = np.searchsorted(column_axis, columns)
    flat = row_index * column_axis.size + column_index
    if np.unique(flat).size != rows.size:
        raise ValueError("grid coordinates must be unique")
    return _GridLayout(
        rows=row_axis,
        columns=column_axis,
        row_index=row_index,
        column_index=column_index,
        stride=layout.stride,
    )


def _regular_grid_layout_from_axes(
    rows: np.ndarray,
    columns: np.ndarray,
    shape_hw: tuple[int, int],
) -> _GridLayout:
    if rows.size < 2 or columns.size < 3:
        raise ValueError("the grid needs at least 2 rows and 3 columns")
    row_step = np.diff(rows)
    column_step = np.diff(columns)
    stride = int(row_step[0])
    if (
        stride < 1
        or not np.all(row_step == stride)
        or not np.all(column_step == stride)
    ):
        raise ValueError("grid rows and columns must use one regular stride")
    height, width = shape_hw
    if columns.size * stride != width:
        raise ValueError("grid columns must cover one complete longitude period")
    if rows[0] != stride // 2 or columns[0] != stride // 2:
        raise ValueError("grid axes must use the regular pixel-centre offset")
    if rows[-1] >= height or columns[-1] >= width:
        raise ValueError("grid axes must lie inside output_shape_hw")
    return _GridLayout(
        rows=rows,
        columns=columns,
        row_index=np.empty(0, dtype=np.int64),
        column_index=np.empty(0, dtype=np.int64),
        stride=stride,
    )


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    if np.issubdtype(rgb.dtype, np.integer):
        maximum = np.iinfo(rgb.dtype).max
        return rgb.astype(np.float64) / float(maximum)
    value = rgb.astype(np.float64)
    finite = value[np.isfinite(value)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError("floating target_rgb must lie in [0, 1]")
    if not np.isfinite(value).all():
        raise ValueError("target_rgb must be finite")
    return value


def _scatter_grid(
    values: np.ndarray,
    flat_index: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    output = np.empty(shape[0] * shape[1], dtype=values.dtype)
    output[flat_index] = values
    return output.reshape(shape)


def _spherical_edges(
    prior: np.ndarray,
    rgb: np.ndarray,
    validity: np.ndarray,
    grid_rows: np.ndarray,
    native_shape_hw: tuple[int, int],
    options: ContinuousResidualOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, _ = prior.shape
    index = np.arange(prior.size, dtype=np.int64).reshape(prior.shape)
    east_i = index.ravel()
    east_j = np.roll(index, -1, axis=1).ravel()
    if height > 1:
        south_i = index[:-1].ravel()
        south_j = index[1:].ravel()
    else:
        south_i = np.empty(0, dtype=np.int64)
        south_j = np.empty(0, dtype=np.int64)
    edge_i = np.concatenate((east_i, south_i))
    edge_j = np.concatenate((east_j, south_j))
    flat_rgb = rgb.reshape(-1, 3)
    flat_log_prior = np.log(prior.ravel())
    color_difference = np.sqrt(
        np.mean((flat_rgb[edge_i] - flat_rgb[edge_j]) ** 2, axis=1)
    )
    depth_difference = np.abs(flat_log_prior[edge_i] - flat_log_prior[edge_j])
    affinity = np.exp(
        -0.5 * (color_difference / options.color_sigma) ** 2
        - 0.5 * (depth_difference / options.log_depth_sigma) ** 2
    )
    affinity = (
        options.minimum_edge_weight + (1.0 - options.minimum_edge_weight) * affinity
    )
    latitude = (
        math.pi / 2.0
        - (grid_rows.astype(np.float64) + 0.5) / native_shape_hw[0] * math.pi
    )
    cosine = np.maximum(np.cos(latitude), options.latitude_cosine_floor)
    east_metric = np.repeat(1.0 / cosine, prior.shape[1])
    south_metric = np.repeat(
        0.5 * (cosine[:-1] + cosine[1:]),
        prior.shape[1],
    )
    metric = np.concatenate((east_metric, south_metric))
    valid_flat = validity.ravel()
    supported = valid_flat[edge_i] & valid_flat[edge_j]
    weight = options.smoothness_weight * affinity * metric * supported
    keep = weight > 0.0
    return edge_i[keep], edge_j[keep], weight[keep]


def _normal_equations(
    shape: tuple[int, int],
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
    data_weight: np.ndarray,
    seed_delta: np.ndarray,
    options: ContinuousResidualOptions,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    count = shape[0] * shape[1]
    diagonal = np.full(count, options.anchor_weight, dtype=np.float64)
    np.add.at(diagonal, edge_i, edge_weight)
    np.add.at(diagonal, edge_j, edge_weight)
    diagonal += data_weight.ravel()
    rows = np.concatenate((np.arange(count), edge_i, edge_j))
    columns = np.concatenate((np.arange(count), edge_j, edge_i))
    values = np.concatenate((diagonal, -edge_weight, -edge_weight))
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr()
    right_hand_side = (data_weight * seed_delta).ravel()
    return matrix, right_hand_side


__all__ = [
    "ContinuousResidualOptions",
    "ContinuousResidualResult",
    "INTERFACE",
    "interpolate_periodic_grid_to_native",
    "solve_continuous_grid_residual",
]
