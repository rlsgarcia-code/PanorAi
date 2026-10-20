"""Immutable typed values for registration and panorama stitching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from panorai.calibration import PinholeIntrinsics, RotationCalibrationResult
from panorai.data import EquirectangularImage


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _mask(value: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return _readonly(array, dtype=bool)


@dataclass(frozen=True, slots=True)
class StitchingOptions:
    """Configuration for the central rotation-only stitching profile."""

    feature_method: str = "sift"
    max_features: int = 4000
    ratio_test: float = 0.75
    min_pair_matches: int = 20
    min_pair_inliers: int = 12
    max_transfer_error_px: float = 3.0
    max_angular_error_deg: float = 1.0
    max_median_angular_error_deg: float = 0.35
    min_horizontal_fov_deg: float = 30.0
    max_horizontal_fov_deg: float = 140.0
    min_spatial_coverage: float = 0.02
    output_shape_hw: tuple[int, int] = (512, 1024)
    interpolation: str = "bilinear"
    exposure_compensation: str = "gain"
    blend: str = "feather"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.feature_method not in {"sift", "orb"}:
            raise ValueError("feature_method must be 'sift' or 'orb'")
        for name, minimum in (
            ("max_features", 32),
            ("min_pair_matches", 4),
            # Every accepted pair first passes the four-point homography
            # family, even though the subsequent rotation fit needs only
            # three bearings.
            ("min_pair_inliers", 4),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.min_pair_inliers > self.min_pair_matches:
            raise ValueError("min_pair_inliers must not exceed min_pair_matches")
        if not 0.0 < float(self.ratio_test) < 1.0:
            raise ValueError("ratio_test must be in (0, 1)")
        if not 0.0 < float(self.max_transfer_error_px):
            raise ValueError("max_transfer_error_px must be positive")
        if not 0.0 < float(self.max_angular_error_deg) < 45.0:
            raise ValueError("max_angular_error_deg must be in (0, 45)")
        if not 0.0 < float(self.max_median_angular_error_deg) < 45.0:
            raise ValueError("max_median_angular_error_deg must be in (0, 45)")
        if not 1.0 < self.min_horizontal_fov_deg < self.max_horizontal_fov_deg < 179.0:
            raise ValueError("invalid horizontal FOV search interval")
        if not 0.0 <= float(self.min_spatial_coverage) <= 1.0:
            raise ValueError("min_spatial_coverage must be in [0, 1]")
        if (
            not isinstance(self.output_shape_hw, tuple)
            or len(self.output_shape_hw) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, np.integer))
                for item in self.output_shape_hw
            )
            or min(self.output_shape_hw) < 2
        ):
            raise ValueError("output_shape_hw must be an integer (height, width) pair")
        if self.interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        if self.exposure_compensation not in {"none", "gain"}:
            raise ValueError("exposure_compensation must be 'none' or 'gain'")
        if self.blend not in {"average", "feather"}:
            raise ValueError("blend must be 'average' or 'feather'")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, (int, np.integer)
        ):
            raise TypeError("random_seed must be an integer")


@dataclass(frozen=True, slots=True)
class StitchingImage:
    """One planar source image with explicit geometric support and validity."""

    image_id: str
    values: np.ndarray
    support: np.ndarray | None = None
    validity: np.ndarray | None = None
    horizontal_fov_deg: float | None = None
    interface: str = "panorai-stitching-image/v1-experimental"
    stability: str = "experimental"

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a non-empty string")
        values = np.asarray(self.values)
        if values.ndim not in {2, 3} or min(values.shape[:2]) < 2:
            raise ValueError("values must have shape (H, W) or (H, W, C)")
        if values.ndim == 3 and values.shape[2] not in {1, 3, 4}:
            raise ValueError("image channels must be 1, 3 or 4")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("values must have a numeric dtype")
        finite = np.isfinite(values)
        if values.ndim == 3:
            finite = np.all(finite, axis=-1)
        shape = values.shape[:2]
        support = np.ones(shape, dtype=bool) if self.support is None else self.support
        validity = (
            np.ones(shape, dtype=bool) if self.validity is None else self.validity
        )
        support = _mask(support, shape, name="support")
        validity = _mask(validity, shape, name="validity")
        if np.any(validity & ~support):
            raise ValueError("validity cannot extend outside geometric support")
        if np.any(validity & ~finite):
            raise ValueError("non-finite source values cannot be marked valid")
        if self.horizontal_fov_deg is not None and not (
            math.isfinite(float(self.horizontal_fov_deg))
            and 1.0 < float(self.horizontal_fov_deg) < 179.0
        ):
            raise ValueError("horizontal_fov_deg must be in (1, 179) or None")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "validity", validity)

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.values.shape[:2]


@dataclass(frozen=True, slots=True)
class StitchingInput:
    images: tuple[StitchingImage, ...]
    interface: str = "panorai-stitching-input/v1-experimental"
    stability: str = "experimental"

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if len(images) < 2 or any(
            not isinstance(item, StitchingImage) for item in images
        ):
            raise ValueError("stitching requires at least two StitchingImage values")
        identifiers = [item.image_id for item in images]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("stitching image identifiers must be unique")
        channels = {
            1 if item.values.ndim == 2 else item.values.shape[2] for item in images
        }
        if len(channels) != 1:
            raise ValueError("all stitching images must have the same channel count")
        object.__setattr__(self, "images", images)

    @classmethod
    def from_arrays(
        cls,
        images: Any,
        *,
        image_ids: Any = None,
        support_masks: Any = None,
        validity_masks: Any = None,
        horizontal_fov_deg: Any = None,
    ) -> "StitchingInput":
        values = tuple(np.asarray(item) for item in images)
        count = len(values)
        ids = (
            tuple(f"image-{index:03d}" for index in range(count))
            if image_ids is None
            else tuple(image_ids)
        )
        support = (None,) * count if support_masks is None else tuple(support_masks)
        validity = (None,) * count if validity_masks is None else tuple(validity_masks)
        if horizontal_fov_deg is None or isinstance(horizontal_fov_deg, (int, float)):
            fovs = (horizontal_fov_deg,) * count
        else:
            fovs = tuple(horizontal_fov_deg)
        if not all(len(item) == count for item in (ids, support, validity, fovs)):
            raise ValueError("all per-image inputs must match the image count")
        return cls(
            tuple(
                StitchingImage(identifier, image, mask, valid, fov)
                for identifier, image, mask, valid, fov in zip(
                    ids, values, support, validity, fovs, strict=True
                )
            )
        )


@dataclass(frozen=True, slots=True)
class StitchingFeatureSet:
    image_id: str
    pixels_xy: np.ndarray
    descriptors: np.ndarray
    responses: np.ndarray
    method: str

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels_xy, dtype=np.float64)
        descriptors = np.asarray(self.descriptors)
        responses = np.asarray(self.responses, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
            raise ValueError("feature pixels must have finite shape (N, 2)")
        if descriptors.ndim != 2 or descriptors.shape[0] != len(pixels):
            raise ValueError("descriptors must align with feature pixels")
        if responses.shape != (len(pixels),) or not np.all(np.isfinite(responses)):
            raise ValueError("responses must align with feature pixels")
        object.__setattr__(self, "pixels_xy", _readonly(pixels))
        object.__setattr__(self, "descriptors", _readonly(descriptors))
        object.__setattr__(self, "responses", _readonly(responses))


@dataclass(frozen=True, slots=True)
class StitchingFeatures:
    source: StitchingInput
    features: tuple[StitchingFeatureSet, ...]


@dataclass(frozen=True, slots=True)
class ImagePairMatches:
    index_a: int
    index_b: int
    feature_indices_a: np.ndarray
    feature_indices_b: np.ndarray
    descriptor_distances: np.ndarray

    def __post_init__(self) -> None:
        left = np.asarray(self.feature_indices_a, dtype=np.int64)
        right = np.asarray(self.feature_indices_b, dtype=np.int64)
        distance = np.asarray(self.descriptor_distances, dtype=np.float64)
        if left.ndim != 1 or right.shape != left.shape or distance.shape != left.shape:
            raise ValueError("pair match arrays must align with shape (N,)")
        if np.any(left < 0) or np.any(right < 0) or not np.all(np.isfinite(distance)):
            raise ValueError("match indices and distances are invalid")
        object.__setattr__(self, "feature_indices_a", _readonly(left))
        object.__setattr__(self, "feature_indices_b", _readonly(right))
        object.__setattr__(self, "descriptor_distances", _readonly(distance))

    @property
    def count(self) -> int:
        return len(self.feature_indices_a)


@dataclass(frozen=True, slots=True)
class StitchingMatches:
    features: StitchingFeatures
    pairs: tuple[ImagePairMatches, ...]


@dataclass(frozen=True, slots=True)
class ImagePairRegistration:
    index_a: int
    index_b: int
    rotation_b_from_a: np.ndarray
    calibration: RotationCalibrationResult
    match_indices: np.ndarray
    homography: np.ndarray
    homography_inlier_count: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_b_from_a, dtype=np.float64)
        homography = np.asarray(self.homography, dtype=np.float64)
        indices = np.asarray(self.match_indices, dtype=np.int64)
        if rotation.shape != (3, 3) or homography.shape != (3, 3):
            raise ValueError("pair registration matrices must have shape (3, 3)")
        object.__setattr__(self, "rotation_b_from_a", _readonly(rotation))
        object.__setattr__(self, "homography", _readonly(homography))
        object.__setattr__(self, "match_indices", _readonly(indices))

    @property
    def num_inliers(self) -> int:
        return self.calibration.num_inliers


@dataclass(frozen=True, slots=True)
class RegisteredImageSet:
    source: StitchingInput
    intrinsics: tuple[PinholeIntrinsics, ...]
    camera_to_world: tuple[np.ndarray, ...]
    pairwise: tuple[ImagePairRegistration, ...]
    feature_counts: tuple[int, ...]
    reference_index: int
    rotation_initial_cost: float
    rotation_final_cost: float
    rejected_pairs: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.source.images)
        if (
            len(self.intrinsics) != count
            or len(self.camera_to_world) != count
            or len(self.feature_counts) != count
        ):
            raise ValueError(
                "registered intrinsics and rotations must cover all images"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, np.integer))
            or int(item) < 0
            for item in self.feature_counts
        ):
            raise ValueError(
                "feature_counts must contain one non-negative integer per image"
            )
        if not 0 <= self.reference_index < count:
            raise ValueError("reference_index is outside the source-image range")
        rotations = []
        for rotation in self.camera_to_world:
            array = np.asarray(rotation, dtype=np.float64)
            if array.shape != (3, 3) or not np.allclose(
                array.T @ array, np.eye(3), atol=1e-7
            ):
                raise ValueError("camera_to_world entries must be SO(3) matrices")
            rotations.append(_readonly(array))
        object.__setattr__(self, "camera_to_world", tuple(rotations))
        object.__setattr__(
            self, "feature_counts", tuple(int(item) for item in self.feature_counts)
        )


@dataclass(frozen=True, slots=True)
class WarpedImageSet:
    registered: RegisteredImageSet
    values: tuple[np.ndarray, ...]
    support: tuple[np.ndarray, ...]
    validity: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    gains: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.registered.source.images)
        if not all(
            len(item) == count
            for item in (
                self.values,
                self.support,
                self.validity,
                self.weights,
                self.gains,
            )
        ):
            raise ValueError("warped-image fields must cover every source image")
        shape = self.values[0].shape[:2]
        values = []
        support = []
        validity = []
        weights = []
        for index in range(count):
            image = np.asarray(self.values[index])
            if image.shape[:2] != shape:
                raise ValueError("warped images must share one ERP lattice")
            values.append(_readonly(image))
            current_support = _mask(self.support[index], shape, name="support")
            current_validity = _mask(self.validity[index], shape, name="validity")
            if np.any(current_validity & ~current_support):
                raise ValueError("warped validity cannot extend outside support")
            support.append(current_support)
            validity.append(current_validity)
            weight = np.asarray(self.weights[index], dtype=np.float64)
            if (
                weight.shape != shape
                or np.any(weight < 0.0)
                or not np.all(np.isfinite(weight))
            ):
                raise ValueError(
                    "warp weights must be finite, non-negative and spatial"
                )
            if np.any(weight[~current_validity] != 0.0):
                raise ValueError("warp weights must be zero outside valid samples")
            weights.append(_readonly(weight))
        gains = tuple(float(item) for item in self.gains)
        if any(not math.isfinite(item) or item <= 0.0 for item in gains):
            raise ValueError("gains must be finite and positive")
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "support", tuple(support))
        object.__setattr__(self, "validity", tuple(validity))
        object.__setattr__(self, "weights", tuple(weights))
        object.__setattr__(self, "gains", gains)


@dataclass(frozen=True, slots=True)
class StitchingDiagnostics:
    input_image_count: int
    feature_counts: tuple[int, ...]
    candidate_pair_count: int
    accepted_pair_count: int
    rejected_pairs: tuple[str, ...]
    reference_image_id: str
    estimated_fov_count: int
    rotation_initial_cost: float
    rotation_final_cost: float
    coverage: float
    overlap_fraction: float
    gains: tuple[float, ...]
    model: str = "central-rotation"


@dataclass(frozen=True, slots=True)
class PanoramaStitchingResult:
    panorama: EquirectangularImage
    registered: RegisteredImageSet
    contribution_map: np.ndarray
    support_count: np.ndarray
    diagnostics: StitchingDiagnostics
    interface: str = "panorai-panorama-stitching/v1-experimental"
    stability: str = "experimental"
    profile_id: str | None = None

    def __post_init__(self) -> None:
        shape = self.panorama.shape[:2]
        contribution = np.asarray(self.contribution_map, dtype=np.int32)
        count = np.asarray(self.support_count, dtype=np.int32)
        if contribution.shape != shape or count.shape != shape:
            raise ValueError("contribution maps must align with the panorama lattice")
        object.__setattr__(self, "contribution_map", _readonly(contribution))
        object.__setattr__(self, "support_count", _readonly(count))


__all__ = [
    "ImagePairMatches",
    "ImagePairRegistration",
    "PanoramaStitchingResult",
    "RegisteredImageSet",
    "StitchingDiagnostics",
    "StitchingFeatureSet",
    "StitchingFeatures",
    "StitchingImage",
    "StitchingInput",
    "StitchingMatches",
    "StitchingOptions",
    "WarpedImageSet",
]
