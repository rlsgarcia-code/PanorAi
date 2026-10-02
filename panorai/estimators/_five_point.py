"""PanorAi-owned polynomial five-correspondence essential solver.

The implementation builds the calibrated essential constraints directly in
the four-dimensional epipolar nullspace.  For each projective coefficient
chart it eliminates the ten cubic monomials and obtains the remaining roots
from a 10 by 10 action matrix.  No OpenCV or COLMAP solver is called.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


# Cubic monomials first, followed by the quotient basis used by the z action.
_MONOMIALS = (
    (3, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
    (2, 0, 1),
    (1, 1, 1),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (0, 0, 3),
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 2),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
)
_MONOMIAL_INDEX = {value: index for index, value in enumerate(_MONOMIALS)}
_BASIS = _MONOMIALS[10:]
_REAL_TOLERANCE = 1e-7


def solve_five_point_essential(
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    *,
    max_epipolar_error: float = 5e-6,
    max_constraint_error: float = 5e-5,
) -> list[np.ndarray]:
    """Return real essential matrices satisfying five bearing correspondences.

    The result is sign-deduplicated and normalized to Frobenius norm one.  All
    four projective coefficient charts are attempted so solutions with a zero
    coordinate in one chart are not silently lost.
    """

    b1 = _five_bearings(bearings_a, "bearings_a")
    b2 = _five_bearings(bearings_b, "bearings_b")
    equations = np.einsum("ni,nj->nij", b2, b1).reshape(5, 9)
    if np.linalg.matrix_rank(equations, tol=1e-11) < 5:
        return []
    _, _, vh = np.linalg.svd(equations, full_matrices=True)
    nullspace = vh[-4:].T

    candidates: list[np.ndarray] = []
    for constant_index in range(4):
        variable_indices = tuple(index for index in range(4) if index != constant_index)
        coefficient_vectors = _solve_chart(nullspace, variable_indices, constant_index)
        for coefficients in coefficient_vectors:
            norm = np.linalg.norm(coefficients)
            if not math.isfinite(float(norm)) or norm <= np.finfo(np.float64).eps:
                continue
            essential = (nullspace @ (coefficients / norm)).reshape(3, 3)
            essential /= np.linalg.norm(essential)
            epipolar = np.abs(np.einsum("ni,ij,nj->n", b2, essential, b1))
            eet = essential @ essential.T
            cubic = 2.0 * eet @ essential - np.trace(eet) * essential
            if (
                epipolar.max(initial=0.0) > max_epipolar_error
                or np.linalg.norm(cubic) > max_constraint_error
                or abs(np.linalg.det(essential)) > max_constraint_error
            ):
                continue
            if _same_projective_matrix(essential, candidates):
                continue
            candidates.append(essential)
    return candidates


def _solve_chart(
    nullspace: np.ndarray,
    variable_indices: tuple[int, int, int],
    constant_index: int,
) -> Iterable[np.ndarray]:
    entries = np.empty((3, 3), dtype=object)
    for row in range(3):
        for column in range(3):
            flat = 3 * row + column
            polynomial = _constant(nullspace[flat, constant_index])
            for axis, coefficient_index in enumerate(variable_indices):
                polynomial = _add(
                    polynomial,
                    _variable(axis, nullspace[flat, coefficient_index]),
                )
            entries[row, column] = polynomial

    eet = _poly_matrix_multiply(entries, entries.T)
    cubic = _poly_matrix_multiply(eet, entries)
    trace = _add(_add(eet[0, 0], eet[1, 1]), eet[2, 2])
    constraints = []
    for row in range(3):
        for column in range(3):
            constraints.append(
                _subtract(
                    _scale(cubic[row, column], 2.0),
                    _multiply(trace, entries[row, column]),
                )
            )
    constraints.append(_determinant(entries))
    coefficient_matrix = np.stack(constraints)
    leading = coefficient_matrix[:, :10]
    if np.linalg.matrix_rank(leading, tol=1e-10) < 10:
        return ()
    condition = np.linalg.cond(leading)
    if not math.isfinite(float(condition)) or condition > 1e12:
        return ()
    reduction = -np.linalg.solve(leading, coefficient_matrix[:, 10:])

    action = np.zeros((10, 10), dtype=np.float64)
    for column, monomial in enumerate(_BASIS):
        product = (monomial[0], monomial[1], monomial[2] + 1)
        index = _MONOMIAL_INDEX[product]
        if index < 10:
            action[:, column] = reduction[index]
        else:
            action[index - 10, column] = 1.0

    eigenvalues, eigenvectors = np.linalg.eig(action.T)
    results = []
    for eigenvalue, vector in zip(eigenvalues, eigenvectors.T, strict=True):
        if abs(float(np.imag(eigenvalue))) > _REAL_TOLERANCE:
            continue
        if np.max(np.abs(np.imag(vector))) > 1e-5:
            continue
        real = np.real(vector)
        scale = real[-1]
        if abs(float(scale)) <= 1e-10:
            continue
        xyz = real[6:9] / scale
        if not np.all(np.isfinite(xyz)):
            continue
        coefficients = np.empty(4, dtype=np.float64)
        coefficients[constant_index] = 1.0
        coefficients[list(variable_indices)] = xyz
        results.append(coefficients)
    return results


def _five_bearings(value: np.ndarray, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (5, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape (5, 3)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 64 * np.finfo(np.float64).eps):
        raise ValueError(f"{name} rows must be non-zero")
    return array / norms[:, None]


def _constant(value: float) -> np.ndarray:
    result = np.zeros(20, dtype=np.float64)
    result[-1] = value
    return result


def _variable(axis: int, value: float) -> np.ndarray:
    result = np.zeros(20, dtype=np.float64)
    exponent = [0, 0, 0]
    exponent[axis] = 1
    result[_MONOMIAL_INDEX[tuple(exponent)]] = value
    return result


def _add(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first + second


def _subtract(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first - second


def _scale(polynomial: np.ndarray, value: float) -> np.ndarray:
    return value * polynomial


def _multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.zeros(20, dtype=np.float64)
    for first_index, second_index in itertools.product(
        np.flatnonzero(first), np.flatnonzero(second)
    ):
        a = _MONOMIALS[int(first_index)]
        b = _MONOMIALS[int(second_index)]
        exponent = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        if sum(exponent) <= 3:
            result[_MONOMIAL_INDEX[exponent]] += (
                first[first_index] * second[second_index]
            )
    return result


def _poly_matrix_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.empty((first.shape[0], second.shape[1]), dtype=object)
    for row in range(first.shape[0]):
        for column in range(second.shape[1]):
            value = np.zeros(20, dtype=np.float64)
            for inner in range(first.shape[1]):
                value += _multiply(first[row, inner], second[inner, column])
            result[row, column] = value
    return result


def _determinant(matrix: np.ndarray) -> np.ndarray:
    positive = _add(
        _add(
            _multiply(_multiply(matrix[0, 0], matrix[1, 1]), matrix[2, 2]),
            _multiply(_multiply(matrix[0, 1], matrix[1, 2]), matrix[2, 0]),
        ),
        _multiply(_multiply(matrix[0, 2], matrix[1, 0]), matrix[2, 1]),
    )
    negative = _add(
        _add(
            _multiply(_multiply(matrix[0, 2], matrix[1, 1]), matrix[2, 0]),
            _multiply(_multiply(matrix[0, 1], matrix[1, 0]), matrix[2, 2]),
        ),
        _multiply(_multiply(matrix[0, 0], matrix[1, 2]), matrix[2, 1]),
    )
    return _subtract(positive, negative)


def _same_projective_matrix(
    candidate: np.ndarray, previous: list[np.ndarray], tolerance: float = 1e-5
) -> bool:
    return any(
        min(
            np.linalg.norm(candidate - item),
            np.linalg.norm(candidate + item),
        )
        < tolerance
        for item in previous
    )
