"""Camera calibration primitives used by registration applications."""

from ._pinhole import (
    PinholeIntrinsics,
    RotationCalibrationOptions,
    RotationCalibrationResult,
    calibrate_rotation_pair,
    pinhole_pixels_to_rays,
    pinhole_rays_to_pixels,
)

__all__ = [
    "PinholeIntrinsics",
    "RotationCalibrationOptions",
    "RotationCalibrationResult",
    "calibrate_rotation_pair",
    "pinhole_pixels_to_rays",
    "pinhole_rays_to_pixels",
]
