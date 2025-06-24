"""
gnomonic_image.py
=================

Implements the GnomonicFace class, which represents a single gnomonic-projected
face taken from an equirectangular image.
"""

import numpy as np
from typing import Union, Tuple, Optional
from PIL import Image  # only if needed for internal usage

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
        """Initialize a :class:`GnomonicFace`.

        Args:
            data: A single-channel array or a dictionary of channel arrays.
            lat: Latitude of the tangent point in degrees.
            lon: Longitude of the tangent point in degrees.
            fov: Field of view in degrees.
            **projection_kwargs: Additional projection arguments.

        Examples:
            >>> face = GnomonicFace(np.zeros((10, 10, 3)), 0.0, 0.0, 90)
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
            name: Projection name such as ``"gnomonic"``.
            lat: Latitude of the tangent point in degrees.
            lon: Longitude of the tangent point in degrees.
            fov: Field of view in degrees.
            **kwargs: Additional projection configuration.

        Examples:
            >>> face.attach_projection("gnomonic", 0.0, 0.0, 90)
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except Exception:
            # Optional dependency missing during lightweight unit tests or
            # additional import errors when running the module in isolation.
            self.projection = None
            return
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
            eq_shape: ``(height, width)`` of the resulting panorama.
            lat: Optional latitude override in degrees.
            lon: Optional longitude override in degrees.
            fov: Optional field of view override in degrees.

        Returns:
            EquirectangularImage

        Examples:
            >>> eq = face.to_equirectangular((512, 1024))
        """
        from .equirectangular_image import EquirectangularImage
        # Possibly update the attached projection or use current one
        projection, (lat_used, lon_used, fov_used) = self.dynamic_projection(lat, lon, fov)
        # Back-projection to equirectangular without mutating this face
        from .multi_handler import MultiChannelHandler
        handler = MultiChannelHandler(self.data_clone())
        new_data = handler.apply_projection(lambda d: projection.back_project(d, eq_shape))
        handler.data = new_data
        handler.squeeze_singleton_channels()
        return EquirectangularImage(handler.data, lat=0.0, lon=0.0)

    def to_pcd(
        self,
        model=None,
        depth: np.ndarray = None,
        grad_threshold: float = 0.1,
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        inter_mask: np.ndarray = None
    ):
        """
        Convert this GnomonicFace into a Point Cloud (PCD).

        Args:
            model: Depth estimation model to use.
            depth: Optional depth array used instead of ``model``.
            grad_threshold: Gradient threshold for valid depth estimation.
            min_radius: Minimum allowable radius in the PCD.
            max_radius: Maximum allowable radius in the PCD.
            inter_mask: Mask indicating valid pixels.

        Returns:
            Some form of PCD object from the PCDHandler.

        Examples:
            >>> pcd = face.to_pcd(model=my_model)
        """
        if (not model) & ( not isinstance(depth, np.ndarray)):
            raise ValueError('You need to pass either a monocular depth estimation model as "model" or a numpy array as depth.')
        else:
            from ..pcd.handler import PCDHandler  # Adjust according to real location
            return PCDHandler.gnomonic_face_to_pcd(
                self,
                model=model,
                depth=depth,
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

        Examples:
            >>> cloned = face.clone()
        """
        new_face = GnomonicFace(
            data=self.data_clone(),
            lat=self.lat,
            lon=self.lon,
            fov=self.fov
        )
        new_face.projection = self.projection
        return new_face

    def show(self) -> None:
        """Display the face using :mod:`PIL.Image` for a quick preview.

        Examples:
            >>> face.show()
        """
        arr = np.asarray(self.get_data())
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        Image.fromarray(arr).show()

    def __repr__(self):
        return f"GnomonicFace(lat={self.lat}, lon={self.lon}, fov={self.fov})"