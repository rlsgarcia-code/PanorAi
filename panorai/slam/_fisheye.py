"""Calibrated central equidistant-fisheye geometry for visual SLAM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class FisheyeRayProjection:
    """Camera rays and validity produced from fisheye raster coordinates."""

    rays_xyz: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        rays = np.array(self.rays_xyz, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        if rays.shape[:-1] != valid.shape or rays.shape[-1:] != (3,):
            raise ValueError("rays_xyz must have shape valid.shape + (3,)")
        rays.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "rays_xyz", rays)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class EquidistantFisheyeCamera:
    """OpenCV/Kalibr pinhole-equidistant camera with PanorAi output axes.

    Raster coordinates use a top-left origin and ``(x, y)`` order. OpenCV's
    local camera coordinates are converted to PanorAi's ``+X`` right, ``+Y``
    up, ``+Z`` forward convention before rays leave this object.
    """

    image_shape_hw: tuple[int, int]
    focal_xy: tuple[float, float]
    principal_xy: tuple[float, float]
    distortion: tuple[float, float, float, float]
    camera_id: str = "cam0"
    max_theta_deg: float = 100.0

    def __post_init__(self) -> None:
        shape = _pair(self.image_shape_hw, "image_shape_hw", integer=True)
        focal = _pair(self.focal_xy, "focal_xy")
        principal = _pair(self.principal_xy, "principal_xy")
        try:
            distortion = tuple(float(item) for item in self.distortion)
        except (TypeError, ValueError) as exc:
            raise TypeError("distortion must contain four real values") from exc
        if len(distortion) != 4 or not np.isfinite(distortion).all():
            raise ValueError("distortion must contain four finite values")
        if focal[0] <= 0.0 or focal[1] <= 0.0:
            raise ValueError("focal lengths must be positive")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise TypeError("camera_id must be a non-empty string")
        theta = float(self.max_theta_deg)
        if not math.isfinite(theta) or not 0.0 < theta < 180.0:
            raise ValueError("max_theta_deg must be finite in (0, 180)")
        object.__setattr__(self, "image_shape_hw", (int(shape[0]), int(shape[1])))
        object.__setattr__(self, "focal_xy", focal)
        object.__setattr__(self, "principal_xy", principal)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "camera_id", self.camera_id.strip())
        object.__setattr__(self, "max_theta_deg", theta)

    @classmethod
    def from_kalibr_yaml(
        cls,
        path: str | Path,
        *,
        camera_id: str = "cam0",
        max_theta_deg: float = 100.0,
    ) -> EquidistantFisheyeCamera:
        """Load one ``pinhole``/``equidistant`` camera from Kalibr YAML."""

        text = Path(path).read_text(encoding="utf-8")
        # OpenCV's ``%YAML:1.0`` marker is not YAML 1.2 syntax.
        if text.startswith("%YAML:"):
            text = "\n".join(text.splitlines()[1:])
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict) or camera_id not in payload:
            raise ValueError(f"camera {camera_id!r} is absent from calibration")
        data = payload[camera_id]
        if data.get("camera_model") != "pinhole":
            raise ValueError("only Kalibr pinhole camera_model is supported")
        if data.get("distortion_model") != "equidistant":
            raise ValueError("only Kalibr equidistant distortion is supported")
        intrinsics = tuple(data.get("intrinsics", ()))
        resolution = tuple(data.get("resolution", ()))
        distortion = tuple(data.get("distortion_coeffs", ()))
        if len(intrinsics) != 4 or len(resolution) != 2:
            raise ValueError("calibration intrinsics/resolution are malformed")
        return cls(
            image_shape_hw=(int(resolution[1]), int(resolution[0])),
            focal_xy=(float(intrinsics[0]), float(intrinsics[1])),
            principal_xy=(float(intrinsics[2]), float(intrinsics[3])),
            distortion=distortion,
            camera_id=camera_id,
            max_theta_deg=max_theta_deg,
        )

    def pixels_to_rays(self, pixels_xy: Any) -> FisheyeRayProjection:
        """Invert the equidistant model for arbitrary pixel coordinates."""

        pixels = np.asarray(pixels_xy)
        if pixels.ndim == 0 or pixels.shape[-1:] != (2,):
            raise ValueError("pixels_xy must have a final dimension of length 2")
        if np.issubdtype(pixels.dtype, np.complexfloating):
            raise TypeError("pixels_xy must be real-valued")
        pixels = pixels.astype(np.float64, copy=False)
        height, width = self.image_shape_hw
        x = (pixels[..., 0] - self.principal_xy[0]) / self.focal_xy[0]
        y_down = (pixels[..., 1] - self.principal_xy[1]) / self.focal_xy[1]
        theta_d = np.hypot(x, y_down)
        finite = np.isfinite(pixels).all(axis=-1) & np.isfinite(theta_d)
        inside = (
            (pixels[..., 0] >= -0.5)
            & (pixels[..., 0] <= width - 0.5)
            & (pixels[..., 1] >= -0.5)
            & (pixels[..., 1] <= height - 0.5)
        )
        theta = _invert_equidistant(theta_d, self.distortion)
        limit = math.radians(self.max_theta_deg)
        valid = finite & inside & np.isfinite(theta) & (theta >= 0.0) & (theta <= limit)
        radial = np.zeros_like(theta)
        nonzero = theta_d > 32.0 * np.finfo(np.float64).eps
        radial[nonzero] = np.sin(theta[nonzero]) / theta_d[nonzero]
        ray_x = x * radial
        # Convert raster-down OpenCV camera coordinates to PanorAi +Y up.
        ray_y = -y_down * radial
        ray_z = np.cos(theta)
        ray_x = np.where(nonzero, ray_x, 0.0)
        ray_y = np.where(nonzero, ray_y, 0.0)
        ray_z = np.where(nonzero, ray_z, 1.0)
        rays = np.stack((ray_x, ray_y, ray_z), axis=-1)
        norm = np.linalg.norm(rays, axis=-1)
        valid &= np.isfinite(rays).all(axis=-1) & (norm > 0.0)
        rays = rays / np.where(valid, norm, 1.0)[..., None]
        rays = np.where(valid[..., None], rays, np.nan)
        return FisheyeRayProjection(rays, valid)

    def valid_pixel_mask(self, *, edge_margin_px: int = 0) -> np.ndarray:
        """Return geometric support, optionally eroded by a raster margin."""

        if isinstance(edge_margin_px, bool) or not isinstance(
            edge_margin_px, (int, np.integer)
        ):
            raise TypeError("edge_margin_px must be an integer")
        if edge_margin_px < 0:
            raise ValueError("edge_margin_px must be non-negative")
        height, width = self.image_shape_hw
        y, x = np.indices((height, width), dtype=np.float64)
        mask = self.pixels_to_rays(np.stack((x, y), axis=-1)).valid.copy()
        margin = int(edge_margin_px)
        if margin:
            mask[:margin] = False
            mask[-margin:] = False
            mask[:, :margin] = False
            mask[:, -margin:] = False
        return mask

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair(value: Any, name: str, *, integer: bool = False) -> tuple[Any, Any]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two values") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must contain two values")
    if integer:
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer))
            for item in items
        ):
            raise TypeError(f"{name} values must be integers")
        if any(int(item) <= 0 for item in items):
            raise ValueError(f"{name} values must be positive")
        return int(items[0]), int(items[1])
    try:
        result = float(items[0]), float(items[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} values must be real") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} values must be finite")
    return result


def _invert_equidistant(
    theta_d: np.ndarray, distortion: tuple[float, float, float, float]
) -> np.ndarray:
    theta = np.array(theta_d, dtype=np.float64, copy=True)
    k1, k2, k3, k4 = distortion
    for _ in range(12):
        t2 = theta * theta
        polynomial = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
        derivative = 1.0 + t2 * (
            3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4))
        )
        safe = np.where(np.abs(derivative) > 1e-12, derivative, np.nan)
        step = (theta * polynomial - theta_d) / safe
        theta -= np.where(np.isfinite(step), step, 0.0)
    return theta
