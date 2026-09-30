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
        self._workflow_metadata = None
        self._workflow_support = None
        self.spec = None

        # Determine shape
        if isinstance(data, dict):
            first_key = next(iter(data.keys()))
            first = data[first_key]
        else:
            first = data
        is_torch = type(first).__module__ == "torch" or type(first).__module__.startswith("torch.")
        H, W = first.shape[-2:] if is_torch else first.shape[:2]

        # Attach a default gnomonic projection for this face
        self.projection = None
        self.attach_projection("gnomonic", lat, lon, fov, x_points=W, y_points=H, **projection_kwargs)

    def _workflow_data(self):
        if self._workflow_metadata is None:
            raise TypeError("This face was not created by the experimental views() workflow")
        if isinstance(self.data, dict):
            return self.data
        return {next(iter(self._workflow_metadata)): self.data}

    @property
    def image(self):
        return self._workflow_data().get("image") if self._workflow_metadata else None

    @property
    def depth(self):
        return self._workflow_data().get("depth") if self._workflow_metadata else None

    @property
    def labels(self):
        return self._workflow_data().get("labels") if self._workflow_metadata else None

    def validity(self, modality: str):
        from ._workflow import clone_array

        if self._workflow_metadata is None or modality not in self._workflow_metadata:
            raise KeyError(f"unknown modality {modality!r}")
        return clone_array(self._workflow_metadata[modality]["validity"])

    @property
    def hfov_deg(self):
        return self.spec.hfov_deg if self.spec is not None else self.fov

    @property
    def vfov_deg(self):
        return self.spec.vfov_deg if self.spec is not None else self.fov

    @property
    def roll_deg(self):
        return self.spec.roll_deg if self.spec is not None else 0.0

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
        fov: Optional[float] = None,
        return_mask: bool = False,
    ) -> Union["EquirectangularImage", Tuple["EquirectangularImage", np.ndarray]]:
        """
        Converts a gnomonic face back into an equirectangular image.

        Args:
            eq_shape: ``(height, width)`` of the resulting panorama.
            lat: Optional latitude override in degrees.
            lon: Optional longitude override in degrees.
            fov: Optional field of view override in degrees.
            return_mask: Return ``(image, support_mask)`` when true.

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
        support_mask = None

        def back_project(data):
            nonlocal support_mask
            if not return_mask:
                return projection.back_project(data, eq_shape)
            projected, mask = projection.back_project(
                data, eq_shape, return_mask=True
            )
            mask = np.asarray(mask, dtype=bool)
            support_mask = mask if support_mask is None else support_mask & mask
            return projected

        new_data = handler.apply_projection(back_project)
        handler.data = new_data
        handler.squeeze_singleton_channels()
        image = EquirectangularImage(
            handler.data,
            lat=0.0,
            lon=0.0,
            support_mask=support_mask,
        )
        return (image, support_mask) if return_mask else image

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
        new_face.spec = self.spec
        if self._workflow_metadata is not None:
            from ._workflow import clone_array, copy_metadata

            new_face._workflow_metadata = copy_metadata(self._workflow_metadata)
            new_face._workflow_support = {
                key: clone_array(value)
                for key, value in self._workflow_support.items()
            }
            new_face.support_mask = clone_array(self.support_mask)
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
