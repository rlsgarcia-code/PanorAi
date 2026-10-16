"""Registered multi-stage algorithms operating on spherical results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

from ._operations import Canny


@dataclass(frozen=True, slots=True)
class SLIC:
    """Configured adapter for the historical intrinsic spherical SLIC."""

    n_segments: int = 100
    compactness: float = 10.0
    max_num_iter: int = 10
    convergence_tol_deg: float = 1e-3
    enforce_connectivity: bool = True
    min_size_factor: float = 0.5
    start_label: int = 1
    mask: Any | None = None

    spherical_operation = True
    spherical_algorithm = True
    canonical_erp_output = True
    preserves_shadow_state = True
    interpolation = "nearest"
    interface = "panorai-spherical-slic/v1"
    stability = "stable"

    def __post_init__(self) -> None:
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.dtype != np.bool_ or mask.ndim != 2:
                raise TypeError("SLIC mask must be a boolean HW array or None")
            mask = mask.copy()
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from panorai.image_processing import spherical_slic

        return spherical_slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            max_num_iter=self.max_num_iter,
            convergence_tol_deg=self.convergence_tol_deg,
            enforce_connectivity=self.enforce_connectivity,
            min_size_factor=self.min_size_factor,
            start_label=self.start_label,
            mask=self.mask,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("SLIC supports only invalid_policy='propagate'")
        usable = np.asarray(support, dtype=bool) & np.asarray(validity, dtype=bool)
        if self.mask is not None:
            if self.mask.shape != usable.shape:
                raise ValueError("SLIC mask must match the input ERP shape")
            usable &= self.mask
        from panorai.image_processing import spherical_slic

        labels = spherical_slic(
            np.asarray(values),
            n_segments=self.n_segments,
            compactness=self.compactness,
            max_num_iter=self.max_num_iter,
            convergence_tol_deg=self.convergence_tol_deg,
            enforce_connectivity=self.enforce_connectivity,
            min_size_factor=self.min_size_factor,
            start_label=self.start_label,
            mask=usable,
        )
        return labels, np.asarray(support, dtype=bool).copy(), usable

    def transform_mask(self, mask: Any) -> np.ndarray:
        return np.asarray(mask, dtype=bool).copy()

    def transform_grid(self, input_grid: Any) -> Any:
        return input_grid.with_transform(
            operation="slic",
            output_shape_hw=input_grid.shape_hw,
            domain=input_grid.domain,
            alignment=input_grid.alignment,
            first_center_px_yx=input_grid.first_center_px_yx,
            pixel_stride_yx=input_grid.pixel_stride_yx,
            parameters={
                "n_segments": self.n_segments,
                "compactness": self.compactness,
                "max_num_iter": self.max_num_iter,
                "enforce_connectivity": self.enforce_connectivity,
                "start_label": self.start_label,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "interface": self.interface,
            "stability": self.stability,
            "domain": "sphere",
            "n_segments": self.n_segments,
            "compactness": self.compactness,
            "max_num_iter": self.max_num_iter,
            "enforce_connectivity": self.enforce_connectivity,
            "start_label": self.start_label,
            "has_mask": self.mask is not None,
        }


@dataclass(frozen=True, slots=True)
class DoGDetector:
    """Thin immutable adapter for :class:`features.SphericalDoGDetector`."""

    config: Any | None = None

    spherical_algorithm = True
    interface = "panorai-spherical-dog-detector/v2"
    stability = "experimental"

    def __post_init__(self) -> None:
        if self.config is not None:
            from panorai.features import SphericalDoGDetectorConfig

            if not isinstance(self.config, SphericalDoGDetectorConfig):
                raise TypeError("config must be SphericalDoGDetectorConfig or None")

    def __call__(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: np.ndarray | None = None,
    ) -> Any:
        from panorai.features import SphericalDoGDetector

        return SphericalDoGDetector(self.config).detect(
            panorama,
            panorama_id=panorama_id,
            validity_mask=validity_mask,
        )

    detect = __call__

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "interface": self.interface,
            "stability": self.stability,
            "domain": "sphere",
            "implementation": "panorai.features.SphericalDoGDetector",
            "config": None if self.config is None else self.config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GaussianPyramid:
    """Configured adapter for the historical spherical Gaussian pyramid."""

    levels: int
    sigma: float = 1.0
    ksize: int = 5
    backend: str = "auto"

    spherical_algorithm = True

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        from panorai.image_processing import spherical_gaussian_pyramid

        return spherical_gaussian_pyramid(
            image,
            self.levels,
            sigma=self.sigma,
            ksize=self.ksize,
            backend=self.backend,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "levels": self.levels,
            "sigma": self.sigma,
            "ksize": self.ksize,
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class LaplacianPyramid(GaussianPyramid):
    """Configured adapter for the reconstructible spherical Laplacian pyramid."""

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        from panorai.image_processing import spherical_laplacian_pyramid

        return spherical_laplacian_pyramid(
            image,
            self.levels,
            sigma=self.sigma,
            ksize=self.ksize,
            backend=self.backend,
        )


@dataclass(frozen=True, slots=True)
class SphericalNMS:
    """Deterministic angular NMS for :class:`SphericalObservations`.

    The implementation delegates grouping and tie-breaking to the existing
    spherical-feature deduplication utility. Invalid observations and scores
    below ``min_score`` are removed before grouping. Labels are isolated by
    default so detections from different classes do not suppress one another.
    """

    angle_deg: float = 1.5
    class_aware: bool = True
    min_score: float | None = None

    spherical_algorithm = True

    def __post_init__(self) -> None:
        if isinstance(self.angle_deg, bool) or not isinstance(self.angle_deg, Real):
            raise TypeError("angle_deg must be a real number")
        if not math.isfinite(float(self.angle_deg)) or not 0 <= self.angle_deg <= 180:
            raise ValueError("angle_deg must be finite and in [0, 180]")
        if not isinstance(self.class_aware, bool):
            raise TypeError("class_aware must be boolean")
        if self.min_score is not None:
            if isinstance(self.min_score, bool) or not isinstance(self.min_score, Real):
                raise TypeError("min_score must be a real number or None")
            if not math.isfinite(float(self.min_score)):
                raise ValueError("min_score must be finite")

    def __call__(self, observations: Any) -> Any:
        from panorai.features import deduplicate_spherical_keypoints
        from panorai.pipeline import SphericalObservations

        if not isinstance(observations, SphericalObservations):
            raise TypeError("SphericalNMS requires SphericalObservations")
        eligible = observations.valid.copy()
        if self.min_score is not None:
            eligible &= observations.scores >= float(self.min_score)
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            return observations.take(np.empty(0, dtype=np.int64))

        if self.class_aware and observations.labels is not None:
            groups = [
                eligible_indices[observations.labels[eligible_indices] == label]
                for label in np.unique(observations.labels[eligible_indices])
            ]
        else:
            groups = [eligible_indices]
        selected = []
        threshold = math.radians(float(self.angle_deg))
        for group in groups:
            deduplication = deduplicate_spherical_keypoints(
                observations.bearings_xyz[group],
                observations.scores[group],
                angular_threshold_rad=threshold,
            )
            selected.extend(group[deduplication.selected_indices].tolist())
        return observations.take(np.asarray(sorted(selected), dtype=np.int64))

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "domain": "sphere",
            "angle_deg": float(self.angle_deg),
            "class_aware": self.class_aware,
            "min_score": self.min_score,
            "selection": "highest-score-then-source-order",
        }


__all__ = [
    "Canny",
    "DoGDetector",
    "GaussianPyramid",
    "LaplacianPyramid",
    "SLIC",
    "SphericalNMS",
]
