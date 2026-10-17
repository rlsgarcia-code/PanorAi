"""Inspectable pipeline stages shared by classical stereo applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from panorai.pipeline import StageResult

from ._operations import disparity_labels, patchmatch_disparity
from ._types import (
    ClassicalStereoResult,
    RectifiedStereoPair,
    StereoCostVolume,
    StereoDisparityCandidate,
)


def _require_pair(value: Any) -> RectifiedStereoPair:
    if not isinstance(value, RectifiedStereoPair):
        raise TypeError("stage requires a RectifiedStereoPair")
    return value


def _require_volume(value: Any) -> StereoCostVolume:
    if not isinstance(value, StereoCostVolume):
        raise TypeError("stage requires a StereoCostVolume")
    return value


@dataclass(frozen=True, slots=True)
class BuildStereoCost:
    """Build a bidirectional cost volume with a configured cost operation."""

    operation: Any
    min_disparity: int = 0
    max_disparity: int = 64

    input_kind = "rectified_stereo_pair"
    output_kind = "stereo_cost_volume"
    stage_name = "matching_cost"

    def execute(self, value: Any) -> StageResult:
        pair = _require_pair(value)
        labels = disparity_labels(self.min_disparity, self.max_disparity)
        result = self.operation.build(pair, labels)
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "operation": self.operation.describe(),
                "disparity_labels": int(labels.size),
                "disparity_interval": [self.min_disparity, self.max_disparity],
            },
        )


@dataclass(frozen=True, slots=True)
class TransformStereoCost:
    """Apply a reusable aggregation, prior, or optimization operation."""

    operation: Any
    role: str

    input_kind = "stereo_cost_volume"
    output_kind = "stereo_cost_volume"

    def __post_init__(self) -> None:
        if self.role not in {"aggregation", "prior", "optimization"}:
            raise ValueError("role must be aggregation, prior, or optimization")

    @property
    def stage_name(self) -> str:
        return self.role

    def execute(self, value: Any) -> StageResult:
        volume = _require_volume(value)
        result = self.operation(volume)
        if not isinstance(result, StereoCostVolume):
            raise TypeError("stereo cost operation must return StereoCostVolume")
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "role": self.role,
                "operation": self.operation.describe(),
            },
        )


def _select_one(
    costs: np.ndarray, disparities: np.ndarray, *, subpixel: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    best_index = np.argmin(costs, axis=2)
    ordered = np.partition(costs, 1, axis=2)
    best_cost = np.take_along_axis(costs, best_index[..., None], axis=2)[..., 0]
    second_cost = ordered[..., 1]
    disparity = disparities[best_index].astype(np.float32)
    valid = best_cost < 999.0
    if subpixel and disparities.size >= 3:
        interior = valid & (best_index > 0) & (best_index < disparities.size - 1)
        lower_index = np.maximum(best_index - 1, 0)
        upper_index = np.minimum(best_index + 1, disparities.size - 1)
        lower = np.take_along_axis(costs, lower_index[..., None], axis=2)[..., 0]
        upper = np.take_along_axis(costs, upper_index[..., None], axis=2)[..., 0]
        denominator = lower - 2.0 * best_cost + upper
        stable = interior & (np.abs(denominator) > 1e-8)
        offset = np.zeros_like(disparity)
        offset[stable] = np.clip(
            0.5 * (lower[stable] - upper[stable]) / denominator[stable],
            -0.5,
            0.5,
        )
        disparity += offset
    disparity = np.where(valid, disparity, np.nan).astype(np.float32)
    return disparity, best_cost, second_cost, valid


@dataclass(frozen=True, slots=True)
class SelectStereoDisparity:
    """Select bidirectional disparities from an optimized cost volume."""

    subpixel: bool = True

    input_kind = "stereo_cost_volume"
    output_kind = "stereo_disparity_candidate"
    stage_name = "disparity_selection"

    def execute(self, value: Any) -> StageResult:
        volume = _require_volume(value)
        left, left_cost, second, left_valid = _select_one(
            volume.left, volume.disparities, subpixel=self.subpixel
        )
        right, _, _, right_valid = _select_one(
            volume.right, volume.disparities, subpixel=self.subpixel
        )
        result = StereoDisparityCandidate(
            volume.pair,
            left,
            right,
            left_cost,
            second,
            left_valid & np.isfinite(left),
            {**dict(volume.metadata), "right_validity": right_valid},
        )
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "subpixel": self.subpixel,
                "backend_valid_fraction": float(result.backend_validity.mean()),
            },
        )


def _uint8_image(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _opencv_matcher(
    method: str,
    *,
    min_disparity: int,
    max_disparity: int,
    block_size: int,
    uniqueness_ratio: int,
    texture_threshold: int,
    p1: int | None,
    p2: int | None,
) -> Any:
    labels = max_disparity - min_disparity + 1
    number = max(16, ((labels + 15) // 16) * 16)
    if method == "bm":
        matcher = cv2.StereoBM_create(numDisparities=number, blockSize=block_size)
        matcher.setMinDisparity(min_disparity)
        matcher.setTextureThreshold(texture_threshold)
        matcher.setUniquenessRatio(uniqueness_ratio)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(-1)
        return matcher
    channels = 1
    resolved_p1 = p1 if p1 is not None else 8 * channels * block_size**2
    resolved_p2 = p2 if p2 is not None else 32 * channels * block_size**2
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=number,
        blockSize=block_size,
        P1=resolved_p1,
        P2=resolved_p2,
        disp12MaxDiff=-1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=0,
        speckleRange=0,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )


def _selected_ad_cost(pair: RectifiedStereoPair, disparity: np.ndarray) -> np.ndarray:
    height, width = pair.shape_hw
    yy, xx = np.indices((height, width))
    safe_disparity = np.where(np.isfinite(disparity), disparity, 0.0)
    target = np.rint(xx - safe_disparity).astype(np.int32)
    inside = np.isfinite(disparity) & (target >= 0) & (target < width)
    clipped = np.clip(target, 0, width - 1)
    sampled = pair.source[yy, clipped]
    return np.where(inside, np.abs(pair.reference - sampled), np.inf).astype(np.float32)


@dataclass(frozen=True, slots=True)
class OpenCVStereoMatcher:
    """Run OpenCV's complete StereoBM or StereoSGBM implementation."""

    method: str
    min_disparity: int = 0
    max_disparity: int = 64
    block_size: int = 9
    uniqueness_ratio: int = 10
    texture_threshold: int = 10
    p1: int | None = None
    p2: int | None = None

    input_kind = "rectified_stereo_pair"
    output_kind = "stereo_disparity_candidate"
    stage_name = "opencv_matcher"

    def __post_init__(self) -> None:
        if self.method not in {"bm", "sgbm"}:
            raise ValueError("method must be 'bm' or 'sgbm'")
        disparity_labels(self.min_disparity, self.max_disparity)
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer >= 3")

    def _one(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        matcher = _opencv_matcher(
            self.method,
            min_disparity=self.min_disparity,
            max_disparity=self.max_disparity,
            block_size=self.block_size,
            uniqueness_ratio=self.uniqueness_ratio,
            texture_threshold=self.texture_threshold,
            p1=self.p1,
            p2=self.p2,
        )
        return (
            matcher.compute(_uint8_image(left), _uint8_image(right)).astype(np.float32)
            / 16.0
        )

    def execute(self, value: Any) -> StageResult:
        pair = _require_pair(value)
        left = self._one(pair.reference, pair.source)
        right = np.fliplr(self._one(np.fliplr(pair.source), np.fliplr(pair.reference)))
        valid = (
            np.isfinite(left)
            & (left >= self.min_disparity)
            & (left <= self.max_disparity)
        )
        cost = _selected_ad_cost(pair, left)
        result = StereoDisparityCandidate(
            pair,
            left,
            right,
            cost,
            cost + np.float32(1.0),
            valid,
            {
                "backend": f"opencv-{self.method}",
                "backend_uniqueness_ratio": self.uniqueness_ratio,
            },
        )
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "backend": f"opencv-{self.method}",
                "disparity_interval": [self.min_disparity, self.max_disparity],
                "backend_valid_fraction": float(valid.mean()),
            },
        )


@dataclass(frozen=True, slots=True)
class PatchMatchStereo:
    """Run the deterministic slanted-plane PatchMatch reference matcher."""

    min_disparity: int = 0
    max_disparity: int = 64
    iterations: int = 3
    patch_radius: int = 2
    random_seed: int = 0

    input_kind = "rectified_stereo_pair"
    output_kind = "stereo_disparity_candidate"
    stage_name = "patchmatch"

    def __post_init__(self) -> None:
        disparity_labels(self.min_disparity, self.max_disparity)
        if self.iterations < 1 or self.patch_radius < 1:
            raise ValueError("iterations and patch_radius must be positive")

    def execute(self, value: Any) -> StageResult:
        pair = _require_pair(value)
        labels = disparity_labels(self.min_disparity, self.max_disparity)
        left, cost, valid = patchmatch_disparity(
            pair,
            labels,
            iterations=self.iterations,
            patch_radius=self.patch_radius,
            random_seed=self.random_seed,
        )
        flipped = RectifiedStereoPair(
            np.fliplr(pair.source),
            np.fliplr(pair.reference),
            np.fliplr(pair.source_support),
            np.fliplr(pair.reference_support),
        )
        right_flipped, _, _ = patchmatch_disparity(
            flipped,
            labels,
            iterations=self.iterations,
            patch_radius=self.patch_radius,
            random_seed=self.random_seed,
        )
        result = StereoDisparityCandidate(
            pair,
            left,
            np.fliplr(right_flipped),
            cost,
            cost + np.float32(1.0),
            valid,
            {"backend": "panorai-patchmatch", "random_seed": self.random_seed},
        )
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "backend": "panorai-patchmatch",
                "iterations": self.iterations,
                "random_seed": self.random_seed,
            },
        )


@dataclass(frozen=True, slots=True)
class RefineStereoDisparity:
    """Apply uniqueness, support, LR consistency, speckle and range output."""

    method: str
    uniqueness_margin: float = 0.02
    left_right_tolerance: float = 1.0
    speckle_window_size: int = 0
    speckle_range: float = 1.0
    median_kernel: int = 0

    input_kind = "stereo_disparity_candidate"
    output_kind = "classical_stereo_result"
    stage_name = "validation"

    def __post_init__(self) -> None:
        if not 0.0 <= self.uniqueness_margin <= 1.0:
            raise ValueError("uniqueness_margin must be in [0, 1]")
        if self.left_right_tolerance < 0 or self.speckle_window_size < 0:
            raise ValueError("consistency and speckle parameters must be non-negative")
        if self.median_kernel not in {0, 3, 5, 7}:
            raise ValueError("median_kernel must be 0, 3, 5, or 7")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, StereoDisparityCandidate):
            raise TypeError("stage requires a StereoDisparityCandidate")
        pair = value.pair
        height, width = pair.shape_hw
        yy, xx = np.indices((height, width))
        safe_left = np.where(np.isfinite(value.left), value.left, 0.0)
        target = np.rint(xx - safe_left).astype(np.int32)
        inside = np.isfinite(value.left) & (target >= 0) & (target < width)
        clipped = np.clip(target, 0, width - 1)
        right = value.right[yy, clipped]
        source_supported = pair.source_support[yy, clipped]
        consistency = (
            inside
            & np.isfinite(right)
            & (np.abs(value.left - right) <= self.left_right_tolerance)
        )
        denominator = np.maximum(np.abs(value.left_second_cost), 1e-6)
        finite_costs = np.isfinite(value.left_cost) & np.isfinite(
            value.left_second_cost
        )
        margin = np.zeros(pair.shape_hw, dtype=np.float32)
        margin[finite_costs] = np.clip(
            (value.left_second_cost[finite_costs] - value.left_cost[finite_costs])
            / denominator[finite_costs],
            0.0,
            1.0,
        )
        unique = margin >= self.uniqueness_margin
        supported = pair.reference_support & inside & source_supported
        valid = value.backend_validity & supported & unique & consistency
        # Rectified metric geometry uses z = f * baseline / disparity and the
        # public convention is d = x_left - x_right.  A non-positive
        # disparity therefore cannot carry finite positive radial range.  Do
        # not leave such samples marked valid while radial_range stores NaN.
        if pair.geometry is not None:
            valid &= np.isfinite(value.left) & (value.left > 0.0)
        occlusion = value.backend_validity & (~inside | ~source_supported)
        mismatch = value.backend_validity & supported & (~unique | ~consistency)

        disparity = value.left.copy()
        if self.speckle_window_size:
            fixed = np.where(valid, np.rint(disparity * 16.0), -32768).astype(np.int16)
            cv2.filterSpeckles(
                fixed,
                -32768,
                self.speckle_window_size,
                int(round(self.speckle_range * 16.0)),
            )
            valid &= fixed != -32768
        if self.median_kernel:
            filled = np.where(valid, disparity, 0.0).astype(np.float32)
            filtered = cv2.medianBlur(filled, self.median_kernel)
            kernel = np.ones((self.median_kernel, self.median_kernel), np.uint8)
            complete = cv2.erode(valid.astype(np.uint8), kernel).astype(bool)
            disparity = np.where(complete, filtered, disparity)

        disparity = np.where(valid, disparity, np.nan).astype(np.float32)
        confidence = np.where(valid, margin * np.exp(-value.left_cost), 0.0).astype(
            np.float32
        )
        matching_cost = np.where(valid, value.left_cost, np.nan).astype(np.float32)
        radial = (
            None
            if pair.geometry is None
            else pair.geometry.radial_range(disparity, valid)
        )
        result = ClassicalStereoResult(
            disparity,
            valid,
            confidence,
            matching_cost,
            occlusion,
            mismatch,
            radial,
            self.method,
            {
                "candidate": dict(value.metadata),
                "left_right_tolerance_px": self.left_right_tolerance,
                "uniqueness_margin": self.uniqueness_margin,
                "supported_fraction": float(supported.mean()),
            },
        )
        return StageResult(
            result,
            {
                "domain": "rectified-pinhole",
                "method": self.method,
                "quantity": "disparity_px",
                "optional_quantity": "radial_range",
                "valid_fraction": float(valid.mean()),
                "occlusion_fraction": float(occlusion.mean()),
                "mismatch_fraction": float(mismatch.mean()),
            },
        )


__all__ = [
    "BuildStereoCost",
    "OpenCVStereoMatcher",
    "PatchMatchStereo",
    "RefineStereoDisparity",
    "SelectStereoDisparity",
    "TransformStereoCost",
]
