"""Pinhole bearings and rotation-only focal calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from panorai.model_fitting import RobustFitOptions, RotationModel, fit_rotation


def _positive_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
        or any(not isinstance(item, (int, np.integer)) for item in value)
    ):
        raise TypeError("image_size_hw must be an integer (height, width) pair")
    height, width = (int(item) for item in value)
    if height < 2 or width < 2:
        raise ValueError("image dimensions must both be at least 2")
    return height, width


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Pixel-centre pinhole calibration in PanorAi's camera frame.

    Camera rays use ``+X`` right, ``+Y`` up and ``+Z`` forward.  Image rows
    increase downward, so the vertical projection contains an explicit sign
    reversal.  Distortion is intentionally absent from this first contract.
    """

    image_size_hw: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    interface: str = "panorai-pinhole-intrinsics/v1-experimental"
    stability: str = "experimental"

    def __post_init__(self) -> None:
        height, width = _image_size(self.image_size_hw)
        fx = _positive_finite("fx", self.fx)
        fy = _positive_finite("fy", self.fy)
        cx = float(self.cx)
        cy = float(self.cy)
        if not math.isfinite(cx) or not math.isfinite(cy):
            raise ValueError("principal point must be finite")
        if not -0.5 <= cx <= width - 0.5 or not -0.5 <= cy <= height - 0.5:
            raise ValueError("principal point must lie on the pixel-boundary domain")
        object.__setattr__(self, "image_size_hw", (height, width))
        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)

    @classmethod
    def from_horizontal_fov(
        cls,
        image_size_hw: tuple[int, int],
        horizontal_fov_deg: float,
        *,
        principal_point_xy: tuple[float, float] | None = None,
    ) -> "PinholeIntrinsics":
        height, width = _image_size(image_size_hw)
        fov = float(horizontal_fov_deg)
        if not math.isfinite(fov) or not 1.0 < fov < 179.0:
            raise ValueError("horizontal_fov_deg must be finite and in (1, 179)")
        focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        principal = (
            ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
            if principal_point_xy is None
            else principal_point_xy
        )
        return cls((height, width), focal, focal, *principal)

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.image_size_hw[1] / (2.0 * self.fx)))

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.image_size_hw[0] / (2.0 * self.fy)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "image_size_hw": list(self.image_size_hw),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "distortion": "none",
        }


def pinhole_pixels_to_rays(
    pixels_xy: np.ndarray, intrinsics: PinholeIntrinsics
) -> np.ndarray:
    """Convert pixel-centre coordinates to unit camera-frame bearings."""

    pixels = np.asarray(pixels_xy, dtype=np.float64)
    if pixels.shape[-1:] != (2,) or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels_xy must have finite shape (..., 2)")
    rays = np.empty(pixels.shape[:-1] + (3,), dtype=np.float64)
    rays[..., 0] = (pixels[..., 0] - intrinsics.cx) / intrinsics.fx
    rays[..., 1] = -(pixels[..., 1] - intrinsics.cy) / intrinsics.fy
    rays[..., 2] = 1.0
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def pinhole_rays_to_pixels(
    rays_xyz: np.ndarray, intrinsics: PinholeIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame rays and return pixels plus positive-depth mask."""

    rays = np.asarray(rays_xyz, dtype=np.float64)
    if rays.shape[-1:] != (3,) or not np.all(np.isfinite(rays)):
        raise ValueError("rays_xyz must have finite shape (..., 3)")
    positive = rays[..., 2] > np.finfo(np.float64).eps
    denominator = np.where(positive, rays[..., 2], 1.0)
    pixels = np.empty(rays.shape[:-1] + (2,), dtype=np.float64)
    pixels[..., 0] = intrinsics.fx * rays[..., 0] / denominator + intrinsics.cx
    pixels[..., 1] = intrinsics.cy - intrinsics.fy * rays[..., 1] / denominator
    pixels[~positive] = np.nan
    return pixels, positive


@dataclass(frozen=True, slots=True)
class RotationCalibrationOptions:
    """Assumptions and search limits for rotation-only pair calibration."""

    min_horizontal_fov_deg: float = 30.0
    max_horizontal_fov_deg: float = 140.0
    max_angular_error_deg: float = 1.0
    min_inliers: int = 12
    robust_trials: int = 512
    local_optimization_steps: int = 3
    random_seed: int = 0
    min_spatial_coverage: float = 0.02

    def __post_init__(self) -> None:
        minimum = float(self.min_horizontal_fov_deg)
        maximum = float(self.max_horizontal_fov_deg)
        if not (1.0 < minimum < maximum < 179.0):
            raise ValueError("horizontal FOV bounds must satisfy 1 < min < max < 179")
        if not 0.0 < float(self.max_angular_error_deg) < 45.0:
            raise ValueError("max_angular_error_deg must be in (0, 45)")
        if self.min_inliers < 3 or self.robust_trials < 1:
            raise ValueError("min_inliers >= 3 and robust_trials >= 1 are required")
        if self.local_optimization_steps < 0:
            raise ValueError("local_optimization_steps must be non-negative")
        if not 0.0 <= float(self.min_spatial_coverage) <= 1.0:
            raise ValueError("min_spatial_coverage must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RotationCalibrationResult:
    """Rotation-only calibration with explicit observability diagnostics."""

    intrinsics_a: PinholeIntrinsics
    intrinsics_b: PinholeIntrinsics
    rotation_b_from_a: np.ndarray
    inlier_mask: np.ndarray
    angular_residuals_rad: np.ndarray
    estimated_fov: bool
    observable: bool
    degeneracy_reasons: tuple[str, ...]
    spatial_coverage: float
    objective_at_solution: float
    objective_at_bounds: tuple[float, float]
    interface: str = "panorai-rotation-pair-calibration/v1-experimental"
    stability: str = "experimental"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_b_from_a, dtype=np.float64)
        inliers = np.asarray(self.inlier_mask, dtype=bool)
        residuals = np.asarray(self.angular_residuals_rad, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_b_from_a must have finite shape (3, 3)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation_b_from_a must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation_b_from_a must have determinant +1")
        if inliers.ndim != 1 or residuals.shape != inliers.shape:
            raise ValueError("calibration masks and residuals must align")
        if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise ValueError("angular residuals must be finite and non-negative")
        rotation = rotation.copy()
        inliers = inliers.copy()
        residuals = residuals.copy()
        for value in (rotation, inliers, residuals):
            value.setflags(write=False)
        object.__setattr__(self, "rotation_b_from_a", rotation)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "angular_residuals_rad", residuals)

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def median_angular_error_deg(self) -> float:
        values = self.angular_residuals_rad[self.inlier_mask]
        return float(np.degrees(np.median(values))) if len(values) else math.inf


def _coverage(pixels: np.ndarray, image_size_hw: tuple[int, int]) -> float:
    height, width = image_size_hw
    extent = np.ptp(np.asarray(pixels, dtype=np.float64), axis=0)
    return float((extent[0] / width) * (extent[1] / height))


def calibrate_rotation_pair(
    pixels_a_xy: np.ndarray,
    pixels_b_xy: np.ndarray,
    image_size_a_hw: tuple[int, int],
    image_size_b_hw: tuple[int, int],
    *,
    horizontal_fov_a_deg: float | None = None,
    horizontal_fov_b_deg: float | None = None,
    options: RotationCalibrationOptions | None = None,
) -> RotationCalibrationResult:
    """Calibrate and register a rotating central-camera image pair.

    When neither FOV is supplied the estimator uses the explicit assumption
    that both images share one horizontal FOV.  When one FOV is supplied, it
    remains fixed while the missing FOV is estimated.  Two supplied FOV values
    are used independently.  The result reports whether calibration was
    estimated and whether the evidence made that estimate observable.
    """

    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

    config = options or RotationCalibrationOptions()
    size_a = _image_size(image_size_a_hw)
    size_b = _image_size(image_size_b_hw)
    pixels_a = np.asarray(pixels_a_xy, dtype=np.float64)
    pixels_b = np.asarray(pixels_b_xy, dtype=np.float64)
    if (
        pixels_a.ndim != 2
        or pixels_a.shape[1] != 2
        or pixels_b.shape != pixels_a.shape
        or len(pixels_a) < 3
        or not np.all(np.isfinite(pixels_a))
        or not np.all(np.isfinite(pixels_b))
    ):
        raise ValueError("pixel correspondences must be aligned finite arrays (N, 2)")

    coverage = min(_coverage(pixels_a, size_a), _coverage(pixels_b, size_b))
    estimated = horizontal_fov_a_deg is None or horizontal_fov_b_deg is None
    known_a = None if horizontal_fov_a_deg is None else float(horizontal_fov_a_deg)
    known_b = None if horizontal_fov_b_deg is None else float(horizontal_fov_b_deg)
    family = RotationModel()

    def intrinsics_for(candidate: float) -> tuple[PinholeIntrinsics, PinholeIntrinsics]:
        fov_a = candidate if known_a is None else known_a
        fov_b = candidate if known_b is None else known_b
        return (
            PinholeIntrinsics.from_horizontal_fov(size_a, fov_a),
            PinholeIntrinsics.from_horizontal_fov(size_b, fov_b),
        )

    def objective(candidate: float) -> float:
        intrinsics_a, intrinsics_b = intrinsics_for(candidate)
        bearings_a = pinhole_pixels_to_rays(pixels_a, intrinsics_a)
        bearings_b = pinhole_pixels_to_rays(pixels_b, intrinsics_b)
        model = family.fit(bearings_a, bearings_b)
        if model is None:
            return 1e6
        residuals = family.residuals(model, bearings_a, bearings_b)
        threshold = math.radians(float(config.max_angular_error_deg))
        # Calibration must retain curvature outside the final inlier band.  A
        # clipped MSAC objective becomes flat for a poor focal guess and can
        # drive a bounded optimizer to an arbitrary endpoint.  Correspondence
        # outliers are handled by the caller's model-fit stage and by the
        # final robust rotation fit below.
        return float(np.mean((residuals / threshold) ** 2))

    if known_a is not None and known_b is not None:
        selected = 0.5 * (known_a + known_b)
        bounds_objective = (objective(selected), objective(selected))
    else:
        lower = float(config.min_horizontal_fov_deg)
        upper = float(config.max_horizontal_fov_deg)
        solved = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 96},
        )
        selected = float(solved.x)
        bounds_objective = (objective(lower), objective(upper))

    intrinsics_a, intrinsics_b = intrinsics_for(selected)
    bearings_a = pinhole_pixels_to_rays(pixels_a, intrinsics_a)
    bearings_b = pinhole_pixels_to_rays(pixels_b, intrinsics_b)
    fit = fit_rotation(
        bearings_a,
        bearings_b,
        options=RobustFitOptions(
            residual_threshold=math.radians(float(config.max_angular_error_deg)),
            method="msac",
            min_trials=min(32, config.robust_trials),
            max_trials=config.robust_trials,
            local_optimization_steps=config.local_optimization_steps,
            min_inliers=config.min_inliers,
            random_seed=config.random_seed,
        ),
    )
    solution_objective = objective(selected)
    reasons: list[str] = []
    if coverage < config.min_spatial_coverage:
        reasons.append("insufficient-spatial-coverage")
    if estimated:
        span = config.max_horizontal_fov_deg - config.min_horizontal_fov_deg
        if (
            min(
                selected - config.min_horizontal_fov_deg,
                config.max_horizontal_fov_deg - selected,
            )
            < 0.01 * span
        ):
            reasons.append("fov-at-search-bound")
        bound_improvement = min(bounds_objective) - solution_objective
        if bound_improvement <= max(1e-8, 0.01 * min(bounds_objective)):
            reasons.append("fov-weakly-observable")
    if fit.num_inliers < config.min_inliers:
        reasons.append("insufficient-inliers")
    return RotationCalibrationResult(
        intrinsics_a=intrinsics_a,
        intrinsics_b=intrinsics_b,
        rotation_b_from_a=np.asarray(fit.model),
        inlier_mask=fit.inlier_mask,
        angular_residuals_rad=fit.residuals,
        estimated_fov=estimated,
        observable=not reasons,
        degeneracy_reasons=tuple(reasons),
        spatial_coverage=coverage,
        objective_at_solution=solution_objective,
        objective_at_bounds=bounds_objective,
    )


__all__ = [
    "PinholeIntrinsics",
    "RotationCalibrationOptions",
    "RotationCalibrationResult",
    "calibrate_rotation_pair",
    "pinhole_pixels_to_rays",
    "pinhole_rays_to_pixels",
]
