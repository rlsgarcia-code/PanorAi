from typing import Any, Tuple
import numpy as np
import logging
import math
from ..base.grid import BaseGridGeneration

logger = logging.getLogger('spherical_projections.gnomonic_projection.gnomonic.grid')

class GnomonicGridGeneration(BaseGridGeneration):
    """Grid generation for the Gnomonic projection.

    This implementation relies directly on :func:`numpy.linspace` and
    :func:`numpy.meshgrid` when constructing both the projection and
    spherical grids.
    """

    def projection_grid(self, delta_lat=0, delta_lon=0) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Generating Gnomonic projection grid.")
        half_fov_rad = (self.config.fov_deg / 2) * math.pi / 180.0
        x_max = math.tan(half_fov_rad) * self.config.R
        y_max = math.tan(half_fov_rad) * self.config.R
        x_vals = np.linspace(-x_max, x_max, self.config.x_points)
        y_vals = np.linspace(-y_max, y_max, self.config.y_points)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        return grid_x, grid_y

    def spherical_grid(self, delta_lat=0, delta_lon=0) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Generating Gnomonic spherical grid.")
        lon_vals = np.linspace(self.config.lon_min, self.config.lon_max, self.config.lon_points) + delta_lon
        lat_vals = np.linspace(self.config.lat_min, self.config.lat_max, self.config.lat_points) + delta_lat
        grid_lon, grid_lat = np.meshgrid(lon_vals, lat_vals)
        return grid_lon, grid_lat
