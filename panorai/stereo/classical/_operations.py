"""Reusable classical stereo costs, aggregators, and optimizers.

These primitives operate on already rectified pinhole pairs.  Projection and
rectification remain responsibilities of :mod:`panorai.geometry` and the
object workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from ._types import RectifiedStereoPair, StereoCostVolume

_INVALID_COST = np.float32(1_000.0)


def disparity_labels(min_disparity: int, max_disparity: int) -> np.ndarray:
    if isinstance(min_disparity, bool) or not isinstance(min_disparity, int):
        raise TypeError("min_disparity must be an integer")
    if isinstance(max_disparity, bool) or not isinstance(max_disparity, int):
        raise TypeError("max_disparity must be an integer")
    if min_disparity < 0 or max_disparity <= min_disparity:
        raise ValueError("disparity bounds require 0 <= min < max")
    return np.arange(min_disparity, max_disparity + 1, dtype=np.float32)


def _shift_for_disparity(
    source: np.ndarray, source_support: np.ndarray, disparity: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = source.shape
    # Preserve integer Census descriptors exactly.  Routing uint64 descriptors
    # through float32 silently destroys their high bits and changes Hamming
    # distances for windows larger than 3x3.
    shifted = np.zeros((height, width), dtype=source.dtype)
    support = np.zeros((height, width), dtype=bool)
    if disparity == 0:
        shifted[:] = source
        support[:] = source_support
    elif disparity < width:
        shifted[:, disparity:] = source[:, : width - disparity]
        support[:, disparity:] = source_support[:, : width - disparity]
    return shifted, support


def _census(
    image: np.ndarray,
    radius: int,
    *,
    tap_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1 or radius > 3:
        raise ValueError("census_radius must be in [1, 3]")
    if (
        isinstance(tap_stride, bool)
        or not isinstance(tap_stride, (int, np.integer))
        or tap_stride < 1
    ):
        raise ValueError("tap_stride must be a positive integer")
    stride = int(tap_stride)
    height, width = image.shape
    descriptor = np.zeros((height, width), dtype=np.uint64)
    bit = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(
                np.roll(image, -dy * stride, axis=0),
                -dx * stride,
                axis=1,
            )
            descriptor |= (shifted >= image).astype(np.uint64) << np.uint64(bit)
            bit += 1
    border = radius * stride
    valid = np.ones((height, width), dtype=bool)
    valid[:border] = False
    valid[-border:] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    return descriptor, valid


_BIT_COUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _hamming(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(left, right)
    bytes_view = xor.view(np.uint8).reshape(*xor.shape, 8)
    return _BIT_COUNT[bytes_view].sum(axis=-1, dtype=np.uint16).astype(np.float32)


def _bt_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_prev = np.concatenate((left[:, :1], left[:, :-1]), axis=1)
    left_next = np.concatenate((left[:, 1:], left[:, -1:]), axis=1)
    right_prev = np.concatenate((right[:, :1], right[:, :-1]), axis=1)
    right_next = np.concatenate((right[:, 1:], right[:, -1:]), axis=1)
    left_low = np.minimum(left, 0.5 * (left + np.minimum(left_prev, left_next)))
    left_high = np.maximum(left, 0.5 * (left + np.maximum(left_prev, left_next)))
    right_low = np.minimum(right, 0.5 * (right + np.minimum(right_prev, right_next)))
    right_high = np.maximum(right, 0.5 * (right + np.maximum(right_prev, right_next)))
    first = np.maximum.reduce(
        (np.zeros_like(left), left - right_high, right_low - left)
    )
    second = np.maximum.reduce(
        (np.zeros_like(left), right - left_high, left_low - right)
    )
    return np.minimum(first, second).astype(np.float32)


class _CostBase:
    def _one_direction(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        reference_support: np.ndarray,
        source_support: np.ndarray,
        disparities: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def build(
        self, pair: RectifiedStereoPair, disparities: np.ndarray
    ) -> StereoCostVolume:
        left = self._one_direction(
            pair.reference,
            pair.source,
            pair.reference_support,
            pair.source_support,
            disparities,
        )
        right_flipped = self._one_direction(
            np.fliplr(pair.source),
            np.fliplr(pair.reference),
            np.fliplr(pair.source_support),
            np.fliplr(pair.reference_support),
            disparities,
        )
        return StereoCostVolume(
            pair,
            disparities,
            left,
            np.flip(right_flipped, axis=1),
            {"cost": self.describe()},
        )


@dataclass(frozen=True, slots=True)
class AbsoluteDifferenceCost(_CostBase):
    """Truncated absolute intensity-difference cost."""

    truncation: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < float(self.truncation) <= 1.0:
            raise ValueError("truncation must be in (0, 1]")

    def _one_direction(
        self, reference, source, reference_support, source_support, disparities
    ):
        volume = np.full(
            (*reference.shape, disparities.size), _INVALID_COST, np.float32
        )
        for index, label in enumerate(disparities.astype(np.int32)):
            shifted, support = _shift_for_disparity(source, source_support, int(label))
            valid = reference_support & support
            cost = np.minimum(np.abs(reference - shifted), self.truncation)
            volume[..., index] = np.where(valid, cost / self.truncation, _INVALID_COST)
        return volume

    def describe(self) -> dict[str, Any]:
        return {"name": "absolute-difference", "truncation": self.truncation}


@dataclass(frozen=True, slots=True)
class BirchfieldTomasiCost(_CostBase):
    """Sampling-insensitive Birchfield--Tomasi intensity cost."""

    truncation: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < float(self.truncation) <= 1.0:
            raise ValueError("truncation must be in (0, 1]")

    def _one_direction(
        self, reference, source, reference_support, source_support, disparities
    ):
        volume = np.full(
            (*reference.shape, disparities.size), _INVALID_COST, np.float32
        )
        for index, label in enumerate(disparities.astype(np.int32)):
            shifted, support = _shift_for_disparity(source, source_support, int(label))
            valid = reference_support & support
            cost = np.minimum(_bt_cost(reference, shifted), self.truncation)
            volume[..., index] = np.where(valid, cost / self.truncation, _INVALID_COST)
        return volume

    def describe(self) -> dict[str, Any]:
        return {"name": "birchfield-tomasi", "truncation": self.truncation}


@dataclass(frozen=True, slots=True)
class CensusCost(_CostBase):
    """Normalized Hamming distance between Census descriptors."""

    radius: int = 2

    def _one_direction(
        self, reference, source, reference_support, source_support, disparities
    ):
        reference_census, reference_census_valid = _census(reference, self.radius)
        source_census, source_census_valid = _census(source, self.radius)
        bits = (2 * self.radius + 1) ** 2 - 1
        volume = np.full(
            (*reference.shape, disparities.size), _INVALID_COST, np.float32
        )
        for index, label in enumerate(disparities.astype(np.int32)):
            shifted, support = _shift_for_disparity(
                source_census, source_support & source_census_valid, int(label)
            )
            valid = reference_support & reference_census_valid & support
            cost = _hamming(reference_census, shifted.astype(np.uint64)) / float(bits)
            volume[..., index] = np.where(valid, cost, _INVALID_COST)
        return volume

    def describe(self) -> dict[str, Any]:
        return {"name": "census", "radius": self.radius}


@dataclass(frozen=True, slots=True)
class ADCensusCost(_CostBase):
    """AD-Census unary cost used by the complete AD-Census pipeline."""

    census_radius: int = 2
    ad_weight: float = 0.25
    census_weight: float = 0.75
    ad_truncation: float = 0.20

    def __post_init__(self) -> None:
        if not math.isclose(self.ad_weight + self.census_weight, 1.0, abs_tol=1e-12):
            raise ValueError("ad_weight and census_weight must sum to 1")

    def _one_direction(
        self, reference, source, reference_support, source_support, disparities
    ):
        ad = AbsoluteDifferenceCost(self.ad_truncation)._one_direction(
            reference, source, reference_support, source_support, disparities
        )
        census = CensusCost(self.census_radius)._one_direction(
            reference, source, reference_support, source_support, disparities
        )
        valid = (ad < _INVALID_COST) & (census < _INVALID_COST)
        combined = self.ad_weight * ad + self.census_weight * census
        return np.where(valid, combined, _INVALID_COST).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "name": "ad-census",
            "census_radius": self.census_radius,
            "ad_weight": self.ad_weight,
            "census_weight": self.census_weight,
            "ad_truncation": self.ad_truncation,
        }


@dataclass(frozen=True, slots=True)
class BoxCostAggregation:
    """Fixed-window valid-normalized cost aggregation."""

    window_size: int = 5

    def __post_init__(self) -> None:
        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")

    def _apply(self, volume: np.ndarray) -> np.ndarray:
        valid = volume < _INVALID_COST
        sums = cv2.boxFilter(
            np.where(valid, volume, 0.0),
            -1,
            (self.window_size, self.window_size),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        counts = cv2.boxFilter(
            valid.astype(np.float32),
            -1,
            (self.window_size, self.window_size),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        return np.where(
            counts > 0, sums / np.maximum(counts, 1.0), _INVALID_COST
        ).astype(np.float32)

    def __call__(self, volume: StereoCostVolume) -> StereoCostVolume:
        return volume.replaced(
            left=self._apply(volume.left),
            right=self._apply(volume.right),
            metadata={"aggregation": self.describe()},
        )

    def describe(self) -> dict[str, Any]:
        return {"name": "box", "window_size": self.window_size}


def _cross_arms(image: np.ndarray, maximum: int, threshold: float) -> np.ndarray:
    height, width = image.shape
    arms = np.zeros((height, width, 4), dtype=np.int16)  # left, right, up, down
    for y in range(height):
        for x in range(width):
            center = float(image[y, x])
            for slot, (dy, dx) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
                length = 0
                for step in range(1, maximum + 1):
                    yy, xx = y + dy * step, x + dx * step
                    if not (0 <= yy < height and 0 <= xx < width):
                        break
                    if abs(float(image[yy, xx]) - center) > threshold:
                        break
                    length = step
                arms[y, x, slot] = length
    return arms


def _cross_aggregate_one(
    costs: np.ndarray, guidance: np.ndarray, maximum: int, threshold: float
) -> np.ndarray:
    height, width, labels = costs.shape
    arms = _cross_arms(guidance, maximum, threshold)
    valid = costs < _INVALID_COST
    values = np.where(valid, costs, 0.0)
    row_values = np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0), (0, 0)))
    row_counts = np.pad(
        np.cumsum(valid.astype(np.float32), axis=1), ((0, 0), (1, 0), (0, 0))
    )
    result = np.full_like(costs, _INVALID_COST)
    for y in range(height):
        for x in range(width):
            total = np.zeros(labels, dtype=np.float32)
            count = np.zeros(labels, dtype=np.float32)
            top = max(0, y - int(arms[y, x, 2]))
            bottom = min(height - 1, y + int(arms[y, x, 3]))
            for yy in range(top, bottom + 1):
                left = max(0, x - min(int(arms[y, x, 0]), int(arms[yy, x, 0])))
                right = min(
                    width - 1,
                    x + min(int(arms[y, x, 1]), int(arms[yy, x, 1])),
                )
                total += row_values[yy, right + 1] - row_values[yy, left]
                count += row_counts[yy, right + 1] - row_counts[yy, left]
            result[y, x] = np.where(
                count > 0, total / np.maximum(count, 1.0), _INVALID_COST
            )
    return result


@dataclass(frozen=True, slots=True)
class CrossBasedCostAggregation:
    """Color-adaptive cross support aggregation (CBCA)."""

    maximum_arm: int = 5
    intensity_threshold: float = 0.08
    passes: int = 2

    def __post_init__(self) -> None:
        if self.maximum_arm < 1:
            raise ValueError("maximum_arm must be positive")
        if not 0.0 < self.intensity_threshold <= 1.0:
            raise ValueError("intensity_threshold must be in (0, 1]")
        if self.passes < 1:
            raise ValueError("passes must be positive")

    def _apply(self, volume: np.ndarray, guidance: np.ndarray) -> np.ndarray:
        result = volume
        for _ in range(self.passes):
            result = _cross_aggregate_one(
                result, guidance, self.maximum_arm, self.intensity_threshold
            )
        return result

    def __call__(self, volume: StereoCostVolume) -> StereoCostVolume:
        return volume.replaced(
            left=self._apply(volume.left, volume.pair.reference),
            right=self._apply(volume.right, volume.pair.source),
            metadata={"aggregation": self.describe()},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "cross-based",
            "maximum_arm": self.maximum_arm,
            "intensity_threshold": self.intensity_threshold,
            "passes": self.passes,
        }


def _transition(previous: np.ndarray, p1: float, p2: float) -> np.ndarray:
    minimum = float(np.min(previous))
    same = previous
    lower = np.concatenate(([np.inf], previous[:-1] + p1))
    upper = np.concatenate((previous[1:] + p1, [np.inf]))
    return (
        np.minimum.reduce((same, lower, upper, np.full_like(previous, minimum + p2)))
        - minimum
    )


def _sgm_one(
    costs: np.ndarray,
    guidance: np.ndarray,
    directions: tuple[tuple[int, int], ...],
    p1: float,
    p2: float,
    edge_threshold: float,
) -> np.ndarray:
    height, width, _ = costs.shape
    total = np.zeros_like(costs)
    for dy, dx in directions:
        message = np.zeros_like(costs)
        ys = range(height) if dy >= 0 else range(height - 1, -1, -1)
        xs = range(width) if dx >= 0 else range(width - 1, -1, -1)
        for y in ys:
            for x in xs:
                py, px = y - dy, x - dx
                if not (0 <= py < height and 0 <= px < width):
                    message[y, x] = costs[y, x]
                    continue
                local_p2 = (
                    p2
                    if abs(float(guidance[y, x] - guidance[py, px])) < edge_threshold
                    else max(p1, 0.25 * p2)
                )
                message[y, x] = costs[y, x] + _transition(message[py, px], p1, local_p2)
        total += message
    return total / float(len(directions))


@dataclass(frozen=True, slots=True)
class SemiGlobalMatching:
    """Four- or eight-path Hirschmüller-style cost regularization."""

    p1: float = 0.03
    p2: float = 0.18
    paths: int = 8
    edge_threshold: float = 0.08

    def __post_init__(self) -> None:
        if self.paths not in {4, 8}:
            raise ValueError("paths must be 4 or 8")
        if self.p1 < 0 or self.p2 < self.p1:
            raise ValueError("SGM requires 0 <= p1 <= p2")

    def _directions(self) -> tuple[tuple[int, int], ...]:
        base = ((0, 1), (0, -1), (1, 0), (-1, 0))
        if self.paths == 4:
            return base
        return base + ((1, 1), (1, -1), (-1, 1), (-1, -1))

    def __call__(self, volume: StereoCostVolume) -> StereoCostVolume:
        directions = self._directions()
        return volume.replaced(
            left=_sgm_one(
                volume.left,
                volume.pair.reference,
                directions,
                self.p1,
                self.p2,
                self.edge_threshold,
            ),
            right=_sgm_one(
                volume.right,
                volume.pair.source,
                directions,
                self.p1,
                self.p2,
                self.edge_threshold,
            ),
            metadata={"optimization": self.describe()},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "sgm",
            "p1": self.p1,
            "p2": self.p2,
            "paths": self.paths,
            "edge_threshold": self.edge_threshold,
        }


def _mgm_one(
    costs: np.ndarray,
    guidance: np.ndarray,
    p1: float,
    p2: float,
    edge_threshold: float,
    variant: str,
) -> np.ndarray:
    height, width, _ = costs.shape
    total = np.zeros_like(costs)
    sweeps = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for sy, sx in sweeps:
        message = np.zeros_like(costs)
        ys = range(height) if sy > 0 else range(height - 1, -1, -1)
        xs = range(width) if sx > 0 else range(width - 1, -1, -1)
        for y in ys:
            for x in xs:
                predecessors = ((y, x - sx), (y - sy, x))
                if variant == "tmgm":
                    predecessors += ((y - sy, x - sx),)
                transitions = []
                for py, px in predecessors:
                    if 0 <= py < height and 0 <= px < width:
                        local_p2 = (
                            p2
                            if abs(float(guidance[y, x] - guidance[py, px]))
                            < edge_threshold
                            else max(p1, 0.25 * p2)
                        )
                        transitions.append(_transition(message[py, px], p1, local_p2))
                message[y, x] = (
                    costs[y, x]
                    if not transitions
                    else costs[y, x] + np.mean(transitions, axis=0)
                )
        total += message
    return total / float(len(sweeps))


@dataclass(frozen=True, slots=True)
class MoreGlobalMatching:
    """Two-dimensional MGM/tMGM min-sum aggregation."""

    variant: str = "mgm"
    p1: float = 0.03
    p2: float = 0.18
    edge_threshold: float = 0.08

    def __post_init__(self) -> None:
        if self.variant not in {"mgm", "tmgm"}:
            raise ValueError("variant must be 'mgm' or 'tmgm'")
        if self.p1 < 0 or self.p2 < self.p1:
            raise ValueError("MGM requires 0 <= p1 <= p2")

    def __call__(self, volume: StereoCostVolume) -> StereoCostVolume:
        return volume.replaced(
            left=_mgm_one(
                volume.left,
                volume.pair.reference,
                self.p1,
                self.p2,
                self.edge_threshold,
                self.variant,
            ),
            right=_mgm_one(
                volume.right,
                volume.pair.source,
                self.p1,
                self.p2,
                self.edge_threshold,
                self.variant,
            ),
            metadata={"optimization": self.describe()},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.variant,
            "p1": self.p1,
            "p2": self.p2,
            "edge_threshold": self.edge_threshold,
        }


def _support_prior_one(
    costs: np.ndarray,
    disparities: np.ndarray,
    step: int,
    uniqueness: float,
    radius: float,
    weight: float,
) -> tuple[np.ndarray, int]:
    order = np.partition(costs, 1, axis=2)
    best = np.argmin(costs, axis=2)
    margin = (order[..., 1] - order[..., 0]) / np.maximum(order[..., 1], 1e-6)
    yy, xx = np.indices(best.shape)
    selected = (
        (yy % step == 0)
        & (xx % step == 0)
        & (margin >= uniqueness)
        & (order[..., 0] < _INVALID_COST)
    )
    points = np.column_stack((xx[selected], yy[selected]))
    values = disparities[best[selected]]
    if values.size < 3:
        return costs.copy(), int(values.size)
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError

    query = np.column_stack((xx.ravel(), yy.ravel()))
    nearest = NearestNDInterpolator(points, values)(query)
    try:
        linear = LinearNDInterpolator(points, values, fill_value=np.nan)(query)
    except QhullError:
        # A scanline or another degenerate support set cannot define a 2-D
        # triangulation.  Nearest support is a deterministic conservative
        # fallback and keeps the prior explicitly approximate.
        linear = np.full(query.shape[0], np.nan, dtype=np.float64)
    prior = np.where(np.isfinite(linear), linear, nearest).reshape(best.shape)
    distance = np.abs(disparities[None, None, :] - prior[..., None])
    penalty = weight * np.minimum(distance / max(radius, 1e-6), 1.0)
    constrained = costs + penalty.astype(np.float32)
    constrained[distance > radius] += np.float32(weight)
    return constrained.astype(np.float32), int(values.size)


@dataclass(frozen=True, slots=True)
class ELASSupportPrior:
    """ELAS-style triangulated support-point prior over disparity labels."""

    support_step: int = 8
    uniqueness: float = 0.08
    radius: float = 4.0
    prior_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.support_step < 1 or self.radius <= 0 or self.prior_weight < 0:
            raise ValueError("invalid ELAS support-prior configuration")

    def __call__(self, volume: StereoCostVolume) -> StereoCostVolume:
        left, left_count = _support_prior_one(
            volume.left,
            volume.disparities,
            self.support_step,
            self.uniqueness,
            self.radius,
            self.prior_weight,
        )
        right, right_count = _support_prior_one(
            volume.right,
            volume.disparities,
            self.support_step,
            self.uniqueness,
            self.radius,
            self.prior_weight,
        )
        return volume.replaced(
            left=left,
            right=right,
            metadata={
                "prior": {
                    **self.describe(),
                    "left_support_points": left_count,
                    "right_support_points": right_count,
                }
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "elas-support-prior",
            "support_step": self.support_step,
            "uniqueness": self.uniqueness,
            "radius": self.radius,
            "prior_weight": self.prior_weight,
        }


def _bilinear_row(row: np.ndarray, x: float) -> float | None:
    if x < 0.0 or x > row.size - 1:
        return None
    x0 = int(math.floor(x))
    x1 = min(x0 + 1, row.size - 1)
    alpha = x - x0
    return float((1.0 - alpha) * row[x0] + alpha * row[x1])


def _plane_patch_cost(
    reference: np.ndarray,
    source: np.ndarray,
    reference_support: np.ndarray,
    source_support: np.ndarray,
    y: int,
    x: int,
    plane: tuple[float, float, float],
    radius: int,
) -> float:
    slope_x, slope_y, center_disparity = plane
    total = 0.0
    weight_sum = 0.0
    center = float(reference[y, x])
    height, width = reference.shape
    for dy in range(-radius, radius + 1):
        yy = y + dy
        if not 0 <= yy < height:
            continue
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if not 0 <= xx < width or not reference_support[yy, xx]:
                continue
            disparity = center_disparity + slope_x * dx + slope_y * dy
            target_x = xx - disparity
            target_index = int(round(target_x))
            if not 0 <= target_index < width or not source_support[yy, target_index]:
                continue
            sampled = _bilinear_row(source[yy], target_x)
            if sampled is None:
                continue
            color_weight = math.exp(-abs(float(reference[yy, xx]) - center) / 0.08)
            spatial_weight = math.exp(
                -(dx * dx + dy * dy) / max(2.0 * radius * radius, 1.0)
            )
            weight = color_weight * spatial_weight
            total += weight * min(abs(float(reference[yy, xx]) - sampled), 0.25) / 0.25
            weight_sum += weight
    return total / weight_sum if weight_sum > 0 else float(_INVALID_COST)


def patchmatch_disparity(
    pair: RectifiedStereoPair,
    disparities: np.ndarray,
    *,
    iterations: int,
    patch_radius: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic slanted-plane PatchMatch reference implementation."""

    initial_volume = AbsoluteDifferenceCost().build(pair, disparities).left
    initial = disparities[np.argmin(initial_volume, axis=2)].astype(np.float32)
    height, width = pair.shape_hw
    slopes_x = np.zeros((height, width), dtype=np.float32)
    slopes_y = np.zeros((height, width), dtype=np.float32)
    current = initial.copy()
    costs = np.full((height, width), _INVALID_COST, dtype=np.float32)
    rng = np.random.default_rng(random_seed)
    for y in range(height):
        for x in range(width):
            costs[y, x] = _plane_patch_cost(
                pair.reference,
                pair.source,
                pair.reference_support,
                pair.source_support,
                y,
                x,
                (0.0, 0.0, float(current[y, x])),
                patch_radius,
            )
    minimum, maximum = float(disparities[0]), float(disparities[-1])
    for iteration in range(iterations):
        forward = iteration % 2 == 0
        ys = range(height) if forward else range(height - 1, -1, -1)
        xs = range(width) if forward else range(width - 1, -1, -1)
        neighbours = ((0, -1), (-1, 0)) if forward else ((0, 1), (1, 0))
        for y in ys:
            for x in xs:
                candidates: list[tuple[float, float, float]] = []
                for dy, dx in neighbours:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        sx = float(slopes_x[yy, xx])
                        sy = float(slopes_y[yy, xx])
                        d = float(current[yy, xx] + sx * (x - xx) + sy * (y - yy))
                        candidates.append((sx, sy, d))
                radius_d = 0.5 * (maximum - minimum)
                radius_s = 0.5
                while radius_d >= 0.5:
                    candidates.append(
                        (
                            float(slopes_x[y, x] + rng.uniform(-radius_s, radius_s)),
                            float(slopes_y[y, x] + rng.uniform(-radius_s, radius_s)),
                            float(current[y, x] + rng.uniform(-radius_d, radius_d)),
                        )
                    )
                    radius_d *= 0.5
                    radius_s *= 0.5
                for sx, sy, disparity in candidates:
                    disparity = float(np.clip(disparity, minimum, maximum))
                    candidate_cost = _plane_patch_cost(
                        pair.reference,
                        pair.source,
                        pair.reference_support,
                        pair.source_support,
                        y,
                        x,
                        (sx, sy, disparity),
                        patch_radius,
                    )
                    if candidate_cost < float(costs[y, x]):
                        slopes_x[y, x] = sx
                        slopes_y[y, x] = sy
                        current[y, x] = disparity
                        costs[y, x] = candidate_cost
    return current, costs, np.isfinite(costs) & (costs < _INVALID_COST)


__all__ = [
    "ADCensusCost",
    "AbsoluteDifferenceCost",
    "BirchfieldTomasiCost",
    "BoxCostAggregation",
    "CensusCost",
    "CrossBasedCostAggregation",
    "ELASSupportPrior",
    "MoreGlobalMatching",
    "SemiGlobalMatching",
    "disparity_labels",
    "patchmatch_disparity",
]
