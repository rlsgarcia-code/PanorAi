"""Minimal-set samplers for spherical five-point RANSAC.

Samplers in this module only propose five correspondence indices.  They do
not remove correspondences from robust scoring, inlier classification, or
local refinement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Protocol

import numpy as np


_SAMPLER_INTERFACE = "panorai-five-point-sampler/v1"
_SAMPLE_SIZE = 5


@dataclass(frozen=True, slots=True)
class FivePointSample:
    """One minimal-set proposal and its geometric diagnostics."""

    indices: np.ndarray
    strategy: str
    relaxation_level: int
    min_separation_a_deg: float
    min_separation_b_deg: float
    unique_cells_a: int
    unique_cells_b: int
    design_condition_number: float

    def __post_init__(self) -> None:
        indices = np.array(self.indices, dtype=np.int64, copy=True)
        if indices.shape != (_SAMPLE_SIZE,):
            raise ValueError("five-point sample indices must have shape (5,)")
        if np.any(indices < 0):
            raise ValueError("five-point sample indices must be non-negative")
        if len(np.unique(indices)) != _SAMPLE_SIZE:
            raise ValueError("five-point sample indices must be unique")
        if not self.strategy:
            raise ValueError("five-point sample strategy must not be empty")
        if (
            isinstance(self.relaxation_level, bool)
            or not isinstance(self.relaxation_level, Integral)
            or self.relaxation_level < 0
        ):
            raise ValueError("relaxation_level must be a non-negative integer")
        for name, value in (
            ("min_separation_a_deg", self.min_separation_a_deg),
            ("min_separation_b_deg", self.min_separation_b_deg),
        ):
            if not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 180.0:
                raise ValueError(f"{name} must be in [0, 180]")
        for name, value in (
            ("unique_cells_a", self.unique_cells_a),
            ("unique_cells_b", self.unique_cells_b),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if not 1 <= value <= _SAMPLE_SIZE:
                raise ValueError(f"{name} must be in [1, 5]")
        if (
            not isinstance(self.design_condition_number, Real)
            or self.design_condition_number <= 0.0
            or math.isnan(float(self.design_condition_number))
        ):
            raise ValueError("design_condition_number must be positive")
        indices.setflags(write=False)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True, slots=True)
class FivePointSamplingDiagnostics:
    """Aggregate provenance for the minimal sets proposed during RANSAC."""

    sampler_name: str
    sampler_configuration: tuple[tuple[str, Any], ...]
    samples_drawn: int
    strict_spatial_samples: int
    relaxed_spatial_samples: int
    uniform_samples: int
    failed_draws: int
    median_min_separation_a_deg: float
    median_min_separation_b_deg: float
    median_design_condition_number: float
    adaptive_uniform_trial_bound_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sampler_configuration"] = dict(self.sampler_configuration)
        return result


class _PreparedSampler(Protocol):
    name: str
    supports_uniform_trial_bound: bool

    def draw(self, rng: np.random.Generator) -> FivePointSample | None: ...


class FivePointSampler(Protocol):
    """Protocol for injectable five-point proposal components."""

    name: str
    supports_uniform_trial_bound: bool

    def prepare(
        self,
        bearings_a: np.ndarray,
        bearings_b: np.ndarray,
        active_indices: np.ndarray,
        sampling_weights: np.ndarray | None = None,
    ) -> _PreparedSampler: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class UniformFivePointSampler:
    """Classic uniform sampling without correspondence prefiltering."""

    name: str = "uniform-five-point-v1"
    supports_uniform_trial_bound: bool = True

    def prepare(
        self,
        bearings_a: np.ndarray,
        bearings_b: np.ndarray,
        active_indices: np.ndarray,
        sampling_weights: np.ndarray | None = None,
    ) -> _PreparedSampler:
        b1, b2, active = _validate_preparation_inputs(
            bearings_a, bearings_b, active_indices
        )
        if sampling_weights is not None:
            _validate_sampling_weights(sampling_weights, len(b1), active)
        return _PreparedUniformSampler(b1, b2, active, self.name)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": _SAMPLER_INTERFACE,
            "name": self.name,
            "proposal": "uniform-without-replacement",
            "prefilters_correspondences": False,
            "supports_uniform_trial_bound": True,
        }


@dataclass(frozen=True, slots=True)
class SpatiallyWeightedFivePointSampler:
    """Propose diverse, well-conditioned five-point minimal sets.

    Input weights define the initial proposal mass. Optional inverse-density
    weighting can be enabled explicitly.
    Each subsequent draw is additionally weighted by angular distance from the
    points already selected in *both* panoramas.  Strict separation and cell
    diversity are relaxed progressively; uniform sampling is used only as the
    final fallback when no spatial proposal satisfies the conditioning gate.

    Every explicitly valid correspondence retains positive proposal mass.
    Robust scoring and refinement remain the responsibility of RANSAC and use
    the complete valid correspondence set.
    """

    min_angular_separation_deg: float = 8.0
    min_unique_cells: int = 4
    spherical_cell_count: int = 20
    density_radius_deg: float = 10.0
    density_power: float = 0.0
    diversity_power: float = 0.5
    uniform_trial_probability: float = 0.2
    max_design_condition_number: float = 1.0e6
    attempts_per_level: int = 24
    name: str = "spatially-weighted-five-point-v1"
    supports_uniform_trial_bound: bool = False

    def __post_init__(self) -> None:
        _finite_between(
            "min_angular_separation_deg",
            self.min_angular_separation_deg,
            0.0,
            180.0,
            lower_closed=True,
        )
        _positive_int("min_unique_cells", self.min_unique_cells)
        if self.min_unique_cells > _SAMPLE_SIZE:
            raise ValueError("min_unique_cells must not exceed 5")
        _positive_int("spherical_cell_count", self.spherical_cell_count)
        if self.spherical_cell_count < self.min_unique_cells:
            raise ValueError("spherical_cell_count must be at least min_unique_cells")
        _finite_between(
            "density_radius_deg",
            self.density_radius_deg,
            0.0,
            180.0,
        )
        if (
            isinstance(self.density_power, bool)
            or not isinstance(self.density_power, Real)
            or not math.isfinite(float(self.density_power))
            or self.density_power < 0.0
        ):
            raise ValueError("density_power must be finite and non-negative")
        if (
            isinstance(self.diversity_power, bool)
            or not isinstance(self.diversity_power, Real)
            or not math.isfinite(float(self.diversity_power))
            or self.diversity_power < 0.0
        ):
            raise ValueError("diversity_power must be finite and non-negative")
        if (
            isinstance(self.uniform_trial_probability, bool)
            or not isinstance(self.uniform_trial_probability, Real)
            or not math.isfinite(float(self.uniform_trial_probability))
            or not 0.0 <= self.uniform_trial_probability < 1.0
        ):
            raise ValueError("uniform_trial_probability must be in the interval [0, 1)")
        if (
            isinstance(self.max_design_condition_number, bool)
            or not isinstance(self.max_design_condition_number, Real)
            or not math.isfinite(float(self.max_design_condition_number))
            or self.max_design_condition_number <= 1.0
        ):
            raise ValueError(
                "max_design_condition_number must be finite and greater than 1"
            )
        _positive_int("attempts_per_level", self.attempts_per_level)

    def prepare(
        self,
        bearings_a: np.ndarray,
        bearings_b: np.ndarray,
        active_indices: np.ndarray,
        sampling_weights: np.ndarray | None = None,
    ) -> _PreparedSampler:
        b1, b2, active = _validate_preparation_inputs(
            bearings_a, bearings_b, active_indices
        )
        prior = _validate_sampling_weights(sampling_weights, len(b1), active)
        centers = _fibonacci_sphere(self.spherical_cell_count)
        active_a = b1[active]
        active_b = b2[active]
        cells_a = np.argmax(active_a @ centers.T, axis=1)
        cells_b = np.argmax(active_b @ centers.T, axis=1)
        base_weights = prior
        if self.density_power > 0.0:
            density_cosine = math.cos(math.radians(self.density_radius_deg))
            density_a = _local_density_counts(active_a, density_cosine)
            density_b = _local_density_counts(active_b, density_cosine)
            joint_density = np.sqrt(density_a * density_b)
            base_weights = base_weights * joint_density ** (-float(self.density_power))
        positive_floor = max(float(base_weights.max(initial=0.0)) * 1e-9, 1e-15)
        base_weights = np.maximum(base_weights, positive_floor)
        return _PreparedSpatialSampler(
            bearings_a=b1,
            bearings_b=b2,
            active_indices=active,
            base_weights=base_weights,
            cells_a=cells_a,
            cells_b=cells_b,
            config=self,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": _SAMPLER_INTERFACE,
            "name": self.name,
            "proposal": "quality-density-angular-diversity",
            "min_angular_separation_deg": self.min_angular_separation_deg,
            "min_unique_cells": self.min_unique_cells,
            "spherical_cell_count": self.spherical_cell_count,
            "density_radius_deg": self.density_radius_deg,
            "density_power": self.density_power,
            "diversity_power": self.diversity_power,
            "uniform_trial_probability": self.uniform_trial_probability,
            "max_design_condition_number": self.max_design_condition_number,
            "attempts_per_level": self.attempts_per_level,
            "relaxation": "uniform-mixture-strict-half-separation-uniform-fallback",
            "prefilters_correspondences": False,
            "supports_uniform_trial_bound": False,
        }


@dataclass(slots=True)
class _PreparedUniformSampler:
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    active_indices: np.ndarray
    name: str
    supports_uniform_trial_bound: bool = True

    def draw(self, rng: np.random.Generator) -> FivePointSample | None:
        if len(self.active_indices) < _SAMPLE_SIZE:
            return None
        indices = rng.choice(self.active_indices, size=_SAMPLE_SIZE, replace=False)
        return _sample_diagnostics(
            indices,
            self.bearings_a,
            self.bearings_b,
            strategy=self.name,
            relaxation_level=0,
            cell_count=20,
        )


@dataclass(slots=True)
class _PreparedSpatialSampler:
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    active_indices: np.ndarray
    base_weights: np.ndarray
    cells_a: np.ndarray
    cells_b: np.ndarray
    config: SpatiallyWeightedFivePointSampler
    supports_uniform_trial_bound: bool = False

    @property
    def name(self) -> str:
        return self.config.name

    def draw(self, rng: np.random.Generator) -> FivePointSample | None:
        if len(self.active_indices) < _SAMPLE_SIZE:
            return None
        if rng.random() < self.config.uniform_trial_probability:
            uniform = self._draw_uniform(rng, "uniform-mixture-five-point-v1", 2)
            if uniform is not None:
                return uniform
        levels = (
            (
                0,
                float(self.config.min_angular_separation_deg),
                self.config.min_unique_cells,
            ),
            (
                1,
                0.5 * float(self.config.min_angular_separation_deg),
                max(2, self.config.min_unique_cells - 1),
            ),
        )
        for level, separation_deg, minimum_cells in levels:
            for _ in range(self.config.attempts_per_level):
                local = self._draw_weighted_indices(rng, separation_deg)
                if local is None:
                    continue
                if len(np.unique(self.cells_a[local])) < minimum_cells:
                    continue
                if len(np.unique(self.cells_b[local])) < minimum_cells:
                    continue
                indices = self.active_indices[local]
                sample = _sample_diagnostics(
                    indices,
                    self.bearings_a,
                    self.bearings_b,
                    strategy=self.name,
                    relaxation_level=level,
                    cell_count=self.config.spherical_cell_count,
                )
                if (
                    sample.design_condition_number
                    <= self.config.max_design_condition_number
                ):
                    return sample

        return self._draw_uniform(rng, "uniform-fallback-five-point-v1", 2)

    def _draw_uniform(
        self,
        rng: np.random.Generator,
        strategy: str,
        relaxation_level: int,
    ) -> FivePointSample | None:
        indices = rng.choice(self.active_indices, size=_SAMPLE_SIZE, replace=False)
        return _sample_diagnostics(
            indices,
            self.bearings_a,
            self.bearings_b,
            strategy=strategy,
            relaxation_level=relaxation_level,
            cell_count=self.config.spherical_cell_count,
        )

    def _draw_weighted_indices(
        self, rng: np.random.Generator, separation_deg: float
    ) -> np.ndarray | None:
        count = len(self.active_indices)
        chosen: list[int] = []
        cosine_limit = math.cos(math.radians(separation_deg))
        for _ in range(_SAMPLE_SIZE):
            eligible = np.ones(count, dtype=bool)
            if chosen:
                eligible[np.asarray(chosen, dtype=np.int64)] = False
                selected_a = self.bearings_a[self.active_indices[chosen]]
                selected_b = self.bearings_b[self.active_indices[chosen]]
                dot_a = self.bearings_a[self.active_indices] @ selected_a.T
                dot_b = self.bearings_b[self.active_indices] @ selected_b.T
                if separation_deg > 0.0:
                    eligible &= np.all(dot_a <= cosine_limit + 1e-15, axis=1)
                    eligible &= np.all(dot_b <= cosine_limit + 1e-15, axis=1)
                min_angle_a = np.min(np.arccos(np.clip(dot_a, -1.0, 1.0)), axis=1)
                min_angle_b = np.min(np.arccos(np.clip(dot_b, -1.0, 1.0)), axis=1)
                diversity = (
                    (0.05 + min_angle_a / math.pi) * (0.05 + min_angle_b / math.pi)
                ) ** float(self.config.diversity_power)
            else:
                diversity = np.ones(count, dtype=np.float64)
            proposal = np.where(eligible, self.base_weights * diversity, 0.0)
            total = float(proposal.sum())
            if not math.isfinite(total) or total <= 0.0:
                return None
            proposal /= total
            chosen.append(int(rng.choice(count, p=proposal)))
        return np.asarray(chosen, dtype=np.int64)


def _sample_diagnostics(
    indices: np.ndarray,
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    *,
    strategy: str,
    relaxation_level: int,
    cell_count: int,
) -> FivePointSample:
    sample_a = bearings_a[indices]
    sample_b = bearings_b[indices]
    centers = _fibonacci_sphere(cell_count)
    cells_a = np.argmax(sample_a @ centers.T, axis=1)
    cells_b = np.argmax(sample_b @ centers.T, axis=1)
    equations = np.einsum("ni,nj->nij", sample_b, sample_a).reshape(_SAMPLE_SIZE, 9)
    singular = np.linalg.svd(equations, compute_uv=False)
    condition = (
        float(singular[0] / singular[-1])
        if singular[-1] > 64 * np.finfo(np.float64).eps
        else math.inf
    )
    return FivePointSample(
        indices=indices,
        strategy=strategy,
        relaxation_level=relaxation_level,
        min_separation_a_deg=_minimum_pairwise_angle_deg(sample_a),
        min_separation_b_deg=_minimum_pairwise_angle_deg(sample_b),
        unique_cells_a=int(len(np.unique(cells_a))),
        unique_cells_b=int(len(np.unique(cells_b))),
        design_condition_number=condition,
    )


def _minimum_pairwise_angle_deg(bearings: np.ndarray) -> float:
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    upper = dots[np.triu_indices(len(bearings), k=1)]
    return float(np.degrees(np.min(np.arccos(upper))))


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    longitude = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.stack((radius * np.cos(longitude), y, radius * np.sin(longitude)), axis=1)


def _local_density_counts(
    bearings: np.ndarray, cosine_limit: float, *, chunk_size: int = 512
) -> np.ndarray:
    """Count angular neighbours without allocating a full NxN matrix."""

    counts = np.empty(len(bearings), dtype=np.int64)
    for start in range(0, len(bearings), chunk_size):
        stop = min(len(bearings), start + chunk_size)
        counts[start:stop] = (bearings[start:stop] @ bearings.T >= cosine_limit).sum(
            axis=1
        )
    return np.maximum(counts, 1)


def _validate_preparation_inputs(
    bearings_a: np.ndarray,
    bearings_b: np.ndarray,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1 = np.array(bearings_a, dtype=np.float64, copy=True)
    b2 = np.array(bearings_b, dtype=np.float64, copy=True)
    active_raw = np.asarray(active_indices)
    if not np.issubdtype(active_raw.dtype, np.integer):
        raise TypeError("active_indices must have integer dtype")
    active = np.array(active_raw, dtype=np.int64, copy=True)
    if b1.ndim != 2 or b1.shape[1:] != (3,) or b2.shape != b1.shape:
        raise ValueError("sampler bearings must have matching shape (N, 3)")
    if active.ndim != 1 or np.any(active < 0) or np.any(active >= len(b1)):
        raise ValueError("active_indices must be a valid one-dimensional index array")
    if len(np.unique(active)) != len(active):
        raise ValueError("active_indices must be unique")
    for name, bearings in (("bearings_a", b1), ("bearings_b", b2)):
        selected = bearings[active]
        norms = np.linalg.norm(selected, axis=1)
        if not np.all(np.isfinite(selected)) or np.any(
            norms <= 64 * np.finfo(np.float64).eps
        ):
            raise ValueError(f"active {name} rows must be finite and non-zero")
        bearings[active] = selected / norms[:, None]
    return b1, b2, active


def _validate_sampling_weights(
    sampling_weights: np.ndarray | None,
    count: int,
    active: np.ndarray,
) -> np.ndarray:
    if sampling_weights is None:
        return np.ones(len(active), dtype=np.float64)
    weights = np.asarray(sampling_weights)
    if weights.shape != (count,):
        raise ValueError("sampling_weights must have shape (N,)")
    if not np.issubdtype(weights.dtype, np.number):
        raise TypeError("sampling_weights must have numeric dtype")
    selected = np.asarray(weights[active], dtype=np.float64)
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError("valid sampling_weights must be finite and non-negative")
    maximum = float(selected.max(initial=0.0))
    if maximum <= 0.0:
        return np.ones(len(active), dtype=np.float64)
    floor = max(maximum * 1e-9, 1e-15)
    return np.maximum(selected, floor)


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_between(
    name: str,
    value: Any,
    lower: float,
    upper: float,
    *,
    lower_closed: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    lower_ok = number >= lower if lower_closed else number > lower
    if not math.isfinite(number) or not lower_ok or number >= upper:
        left = "[" if lower_closed else "("
        raise ValueError(f"{name} must be in {left}{lower}, {upper})")
