from typing import Tuple, Any
import numpy as np
import logging
import math
from ...utils.exceptions import TransformationError, ConfigurationError
from ..base.transform import BaseCoordinateTransformer

logger = logging.getLogger('spherical_projections.gnomonic_projection.gnomonic.transform')

class GnomonicTransformer(BaseCoordinateTransformer):
    """
    Transformer for Gnomonic Projection.
    Converts geographic (lat, lon) to image coordinates on the Gnomonic plane.
    """

    def __init__(self, config: Any):
        logger.debug("Initializing GnomonicTransformer.")
        required_attrs = ["lon_min", "lon_max", "lat_min", "lat_max", "fov_deg", "R", "x_points", "y_points"]
        missing = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing:
            raise ConfigurationError(f"Missing required config attributes: {', '.join(missing)}")
        self.config = config
        logger.info("GnomonicTransformer initialized.")

    def _compute_image_coords(self, values: np.ndarray, min_val: float, max_val: float, size: int) -> np.ndarray:
        normalized = (values - min_val) / (max_val - min_val) * (size - 1)
        logger.debug("Computed normalized image coordinates.")
        return normalized

    def spherical_to_image_coords(self, lat: np.ndarray, lon: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Mapping spherical coordinates to image coordinates.")
        H, W = shape
        # Clamp extreme values if needed
        lon[lon > 180] = -360 + lon[lon > 180]
        lon[lon < -180] = 180 + (lon[lon < -180] + 180)
        lat[lat > 90] = -180 + lat[lat > 90]
        map_x = self._compute_image_coords(lon, self.config.lon_min, self.config.lon_max, W)
        map_y = self._compute_image_coords(lat, self.config.lat_max, self.config.lat_min, H)
        return map_x, map_y

    def projection_to_image_coords(self, x: np.ndarray, y: np.ndarray, config: Any) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Mapping projection coordinates to image coordinates.")
        half_fov_rad = (config.fov_deg / 2) * math.pi / 180.0
        x_max = math.tan(half_fov_rad) * config.R
        y_max = math.tan(half_fov_rad) * config.R
        x_min, y_min = -x_max, -y_max
        map_x = self._compute_image_coords(x, x_min, x_max, config.x_points)
        map_y = self._compute_image_coords(y, y_max, y_min, config.y_points)
        return map_x, map_y
