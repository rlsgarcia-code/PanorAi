"""
gnomonic_image.py
=================

Implements the GnomonicFace class, which represents a single gnomonic-projected
face taken from an equirectangular image.
"""

import numpy as np
from typing import Union, Tuple, Optional

from .spherical_data import SphericalData


class GnomonicFace(SphericalData):
    """
    Represents a gnomonic face extracted from an equirectangular image.

    - Supports multi-channel data (via SphericalData).
    - Allows dynamic projection attachment for forward or backward transformations.
    - Can be converted back to EquirectangularImage for reconstruction.
    """

    def __init__(
        self,
        data: Union[np.ndarray, dict],
        lat: float,
        lon: float,
        fov: float,
        **projection_kwargs
    ):
        """
        Initialize a GnomonicFace.

        Args:
            data (np.ndarray | dict): Either a single-channel array or a dict of channels.
            lat (float): Latitude (degrees) of the tangent point.
            lon (float): Longitude (degrees) of the tangent point.
            fov (float): Field of view in degrees.
            **projection_kwargs: Additional keyword arguments for the attached projection.
        """
        super().__init__(data, lat, lon)
        self.fov = fov

        # Determine shape
        if isinstance(data, dict):
            first_key = next(iter(data.keys()))
            H, W = data[first_key].shape[:2]
        else:
            H, W = data.shape[:2]

        # Attach a default gnomonic projection for this face
        self.projection = None
        self.attach_projection("gnomonic", lat, lon, fov, x_points=W, y_points=H, **projection_kwargs)

    def attach_projection(self, name: str, lat: float, lon: float, fov: float, **kwargs):
        """
        Attach a named projection to this gnomonic face.

        Args:
            name (str): The projection name (e.g., 'gnomonic').
            lat (float): Latitude of the tangent point.
            lon (float): Longitude of the tangent point.
            fov (float): Field of view in degrees.
            **kwargs: Additional projection configuration.
        """
        from panorai.factory.panorai_factory import PanoraiFactory
        self.projection = PanoraiFactory.get_projection(name, lat=lat, lon=lon, fov=fov, **kwargs)

    def to_equirectangular(
        self,
        eq_shape: Tuple[int, int],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        fov: Optional[float] = None
    ) -> "EquirectangularImage":
        """
        Converts a gnomonic face back into an equirectangular image.

        Args:
            eq_shape (Tuple[int, int]): The shape (height, width) for the target equirectangular image.
            lat (float, optional): Override the latitude if needed.
            lon (float, optional): Override the longitude if needed.
            fov (float, optional): Override the field of view if needed.

        Returns:
            EquirectangularImage
        """
        from .equirectangular_image import EquirectangularImage
        # Possibly update the attached projection or use current one
        projection, (lat_used, lon_used, fov_used) = self.dynamic_projection(lat, lon, fov)
        # Back-projection to equirectangular
        new_data = self.apply_projection(lambda d: projection.back_project(d, eq_shape))
        return EquirectangularImage(new_data, lat=0.0, lon=0.0)

    def to_pcd(
        self,
        model=None,
        grad_threshold: float = 0.1,
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        inter_mask: np.ndarray = None
    ):
        """
        Convert this GnomonicFace into a Point Cloud (PCD).

        Args:
            model_name (str): Name or identifier of the model used to interpret the face data.
            grad_threshold (float): Gradient threshold for valid depth estimation.
            min_radius (float): Minimum allowable radius in the PCD.
            max_radius (float): Maximum allowable radius in the PCD.
            inter_mask (np.ndarray): Mask valid pixels

        Returns:
            Some form of PCD object from the PCDHandler.
        """
        if not model:
            raise ValueError('You need to pass a monocular depth estimation model as "model".')
        else:
            from ..pcd.handler import PCDHandler  # Adjust according to real location
            return PCDHandler.gnomonic_face_to_pcd(
                self,
                model=model,
                grad_threshold=grad_threshold,
                min_radius=min_radius,
                max_radius=max_radius,
                inter_mask=inter_mask
            )

    def clone(self) -> "GnomonicFace":
        """
        Creates a deep copy of this GnomonicFace object.

        Returns:
            GnomonicFace
        """
        new_face = GnomonicFace(
            data=self.data_clone(),
            lat=self.lat,
            lon=self.lon,
            fov=self.fov
        )
        new_face.projection = self.projection
        return new_face

    def __repr__(self):
        return f"GnomonicFace(lat={self.lat}, lon={self.lon}, fov={self.fov})"