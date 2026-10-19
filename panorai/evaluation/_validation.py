"""Validation and spherical weighting shared by public evaluators."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np

from panorai.core.numpy import spherical_solid_angle_weights


def reject_torch(value: object, *, name: str) -> None:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        raise TypeError(
            f"{name} is NumPy-canonical; convert Torch explicitly with "
            f"{name}.detach().cpu().numpy()"
        )


def array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    reject_torch(value, name=name)
    result = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return result


def boolean_mask(
    value: Any | None,
    *,
    name: str,
    shape_hw: tuple[int, int],
    default: bool = True,
) -> np.ndarray:
    if value is None:
        return np.full(shape_hw, default, dtype=bool)
    result = array(value, name=name, ndim=2)
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if result.shape != shape_hw:
        raise ValueError(f"{name} must have shape {shape_hw}")
    return result


def binary_mask(value: Any, *, name: str) -> np.ndarray:
    result = array(value, name=name, ndim=2)
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if min(result.shape) < 2:
        raise ValueError(f"{name} must have dimensions of at least 2")
    return result


def solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return exact canonical-ERP pixel solid angles in steradians."""

    if (
        len(shape_hw) != 2
        or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
            for item in shape_hw
        )
        or any(item < 1 for item in shape_hw)
    ):
        raise ValueError("shape_hw must contain two positive integers")
    return spherical_solid_angle_weights(tuple(int(item) for item in shape_hw))


def weighted_sum(
    values: np.ndarray, weights: np.ndarray, selected: np.ndarray
) -> float:
    return float(np.sum(values[selected] * weights[selected], dtype=np.float64))


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
) -> float | None:
    denominator = float(np.sum(weights[selected], dtype=np.float64))
    if denominator <= 0.0:
        return None
    return float(
        np.sum(values[selected] * weights[selected], dtype=np.float64) / denominator
    )


def normalize_bearings(value: Any, *, name: str) -> np.ndarray:
    result = array(value, name=name, dtype=np.float64, ndim=2)
    if result.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N,3)")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{name} cannot contain zero vectors")
    result /= norms[:, None]
    result.setflags(write=False)
    return result


def rotation_matrix(value: Any, *, name: str) -> np.ndarray:
    result = array(value, name=name, dtype=np.float64)
    if result.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3)")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(result)), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} must have determinant +1")
    result.setflags(write=False)
    return result


def unit_vector(value: Any, *, name: str) -> np.ndarray:
    result = array(value, name=name, dtype=np.float64)
    if result.shape != (3,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be one finite vector of shape (3,)")
    norm = float(np.linalg.norm(result))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} cannot be zero")
    result /= norm
    result.setflags(write=False)
    return result


def finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
