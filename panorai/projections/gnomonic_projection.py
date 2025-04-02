# panorai/projections/gnomonic_projection.py

import numpy as np
from typing import Tuple, Optional
from .registry import ProjectionRegistry
from ..utils.exceptions import ProcessingError, ConfigurationError
from .gnomonic.config import GnomonicConfig
from .gnomonic.grid import GnomonicGridGeneration
from .gnomonic.strategy import GnomonicProjectionStrategy
from .gnomonic.transform import GnomonicTransformer
from .base.interpolation import BaseInterpolation

@ProjectionRegistry.register('gnomonic')
class GnomonicProjection:
    """
    Concrete class for Gnomonic projection (NumPy-based).
    """

    def __init__(self, config: Optional[GnomonicConfig] = None,
                phi1_deg: float = 0.0, lam0_deg: float = 0.0, fov_deg: float = 90.0, **kwargs):
        try:
            # Use provided shared configuration if available
            if config is None:
                config = GnomonicConfig(phi1_deg=phi1_deg, lam0_deg=lam0_deg, fov_deg=fov_deg, **kwargs)
            self.config = config
            self.grid_generator = GnomonicGridGeneration(self.config)
            self.strategy = GnomonicProjectionStrategy(self.config)
            self.transformer = GnomonicTransformer(self.config)
            self.interpolation = BaseInterpolation(self.config)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize GnomonicProjection: {e}") from e


    def project(self, eq_img: np.ndarray) -> np.ndarray:
        """
        Project an equirectangular (NumPy) image onto the Gnomonic plane.
        """
        try:
            grid_x, grid_y = self.grid_generator.projection_grid()
            lat, lon = self.strategy.from_projection_to_spherical(grid_x, grid_y)
            map_x, map_y = self.transformer.spherical_to_image_coords(lat, lon, eq_img.shape[:2])
            projected_img = self.interpolation.interpolate(eq_img, map_x, map_y)
            return projected_img
        except Exception as e:
            raise ProcessingError(f"Error during forward projection: {e}") from e

    def back_project(self, face_img: np.ndarray, eq_shape: Tuple[int, int]) -> np.ndarray:
        """
        Back-project a Gnomonic face onto the equirectangular image plane.
        """
        if eq_shape:
            self.config.update(lat_points=eq_shape[0], lon_points=eq_shape[1])
        try:
            lon_grid, lat_grid = self.grid_generator.spherical_grid()
            x, y, mask = self.strategy.from_spherical_to_projection(lat_grid, lon_grid)
            map_x, map_y = self.transformer.projection_to_image_coords(x, y, self.config)
            eq_img = self.interpolation.interpolate(face_img, map_x, map_y, mask)
            return np.flip(eq_img, axis=0)
        except Exception as e:
            raise ProcessingError(f"Error during backward projection: {e}") from e

    # def create_face(self, eq_image: EquirectangularImage) -> GnomonicFace:
    #     """
    #     Create a GnomonicFace from an EquirectangularImage.
    #     """
    #     try:
    #         projected_channels = self.project(eq_image.data)
    #         gnomonic_face = GnomonicFace(
    #             data=projected_channels,
    #             lat=self.config["phi1_deg"],
    #             lon=self.config["lam0_deg"],
    #             fov=self.config["fov_deg"]
    #         )
    #         return gnomonic_face
    #     except Exception as e:
    #         raise ProcessingError(f"Error while creating GnomonicFace: {e}") from e