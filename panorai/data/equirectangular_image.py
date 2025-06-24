"""
equirectangular_image.py
========================

Implements the EquirectangularImage class, which represents a panoramic
equirectangular image (potentially multi-channel).
"""

import numpy as np
from typing import Tuple, List, Union
from PIL import Image  # only if needed for internal usage

from .spherical_data import SphericalData

from panorai.preprocessing.preprocessor import Preprocessor  # assumed import in original code


class EquirectangularImage(SphericalData):
    """
    Represents an equirectangular image with optional multi-channel support.

    - Inherits from SphericalData, which extends multi-channel capabilities.
    - Allows attaching samplers and projection transforms dynamically.
    - Can be converted into one or multiple GnomonicFaces.
    """

    def __init__(
        self,
        data: Union[np.ndarray, dict],
        shadow_angle: float = 0.0,
        lat: float = 0.0,
        lon: float = 0.0,
    ) -> None:
        """Initialize an :class:`EquirectangularImage`.

        Args:
            data: Input array or dictionary of channel arrays.
            shadow_angle: Angle used for shadow correction.
            lat: Latitude of the image centre in degrees.
            lon: Longitude of the image centre in degrees.

        Examples:
            >>> img = EquirectangularImage(np.zeros((512, 1024, 3)))
        """
        super().__init__(data, lat, lon)
        self.shadow_angle = shadow_angle

        # Attach default sampler and projection
        self.sampler = None
        self.projection = None
        self.attach_sampler('cube')
        self.attach_projection("gnomonic")

    def attach_sampler(self, name: str, **kwargs):
        """
        Attach a named sampler for tangent points or other sampling strategies.

        Args:
            name: The sampler name (for example ``"cube"``).
            **kwargs: Additional sampler configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.attach_sampler("cube")
        """
        try:
            import panorai.samplers  # ensure default samplers registered
            from panorai.factory.panorai_factory import PanoraiFactory
            from panorai.samplers.default_samplers import CubeSampler
        except Exception:
            # Optional dependency missing during lightweight unit tests
            # or additional import errors when the full package is not
            # available (e.g. during isolated unit tests).
            self.sampler = None
            return
        try:
            self.sampler = PanoraiFactory.get_sampler(name, **kwargs)
        except Exception:
            # If the factory cannot provide the sampler (e.g., registries
            # haven't been populated), fall back to ``CubeSampler`` when
            # requesting the default 'cube' sampler.
            if name == "cube":
                self.sampler = CubeSampler(**kwargs)
            else:
                self.sampler = None

    def attach_projection(self, name: str, lat: float = 0.0, lon: float = 0.0, fov: float = 90.0, **kwargs):
        """
        Attach a projection method used for converting equirectangular data
        to gnomonic or other coordinate systems.

        Args:
            name: Projection name such as ``"gnomonic"``.
            lat: Latitude of the projection centre.
            lon: Longitude of the projection centre.
            fov: Field of view in degrees.
            **kwargs: Additional projection configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.attach_projection("gnomonic", lat=0.0, lon=0.0, fov=90)
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except Exception:
            # Optional dependency missing during lightweight unit tests or
            # additional import errors when the full package is unavailable.
            self.projection = None
            return
        self.projection = PanoraiFactory.get_projection(name, lat=lat, lon=lon, fov=fov, **kwargs)

    def preprocess(
        self,
        delta_lat: float = 0.0,
        delta_lon: float = 0.0,
        shadow_angle: float = 0.0,
        resize_factor: Union[float, None] = None,
        preprocessing_config: dict = None
    ):
        """
        Applies preprocessing transformations to the equirectangular image.

        - Adjust lat/lon
        - Update shadow angle
        - Resize, if requested
        - Additional custom preprocessing steps (from config)

        Args:
            delta_lat: Shift in latitude in degrees.
            delta_lon: Shift in longitude in degrees.
            shadow_angle: Shadow correction angle.
            resize_factor: Factor by which to resize the image.
            preprocessing_config: Additional :class:`Preprocessor` configuration.

        Examples:
            >>> img = EquirectangularImage(np.zeros((2, 4, 3)))
            >>> img.preprocess(delta_lat=1.0, delta_lon=1.0)
        """
        def _preprocess_func(x):
            return Preprocessor.preprocess_eq(
                x,
                delta_lat=delta_lat,
                delta_lon=delta_lon,
                shadow_angle=shadow_angle,
                resize_factor=resize_factor,
                config=preprocessing_config
            )

        # print('preprocessing...')
        self.data = _preprocess_func(self.data)
        self.lat += delta_lat
        self.lon += delta_lon
        self.shadow_angle = shadow_angle

    def to_gnomonic(self, lat: float, lon: float, fov: float, **kwargs) -> "GnomonicFace":
        """
        Projects the equirectangular image to a single gnomonic face.

        Args:
            lat: Latitude of the tangent point in degrees.
            lon: Longitude of the tangent point in degrees.
            fov: Field of view in degrees.
            **kwargs: Additional projection parameters.

        Returns:
            GnomonicFace: The resulting gnomonic face object.

        Examples:
            >>> face = img.to_gnomonic(lat=0.0, lon=0.0, fov=90)
        """
        from .gnomonic_image import GnomonicFace
        # 1) Possibly update or use attached projection
        projection, (lat, lon, fov) = self.dynamic_projection(lat, lon, fov, **kwargs)
        # 2) Apply projection on a clone to avoid mutating this object's data
        from .multi_handler import MultiChannelHandler
        handler = MultiChannelHandler(self.data_clone())
        projected_data = handler.apply_projection(lambda d: projection.project(d))
        return GnomonicFace(projected_data, lat, lon, fov)

    def to_gnomonic_face_set(
        self,
        fov: float = 90.0,
        sampling_method: Union[str, None] = None,
        rotations: List[Tuple[float, float]] = []
    ) -> "GnomonicFaceSet":
        """
        Samples multiple gnomonic faces from the equirectangular image.

        - Attaches a sampler if none is set or if `sampling_method` is specified.
        - Applies a list of (lat, lon) rotations for additional sampling.

        Args:
            fov: Field of view for each face in degrees.
            sampling_method: Sampler name, such as ``"cube"``.
            rotations: Additional ``(lat, lon)`` rotations to sample.

        Returns:
            GnomonicFaceSet: A collection (set) of gnomonic faces.

        Examples:
            >>> faces = img.to_gnomonic_face_set(fov=90)
        """
        from .gnomonic_imageset import GnomonicFaceSet
        if self.sampler is None:
            self.attach_sampler(sampling_method or "cube")

        if self.sampler is None:
            raise AttributeError("Sampler is not attached")

        tangent_points = self.sampler.get_tangent_points()
        if rotations:
            tangent_points = self.augment_with_rotations(tangent_points, rotations)

        faces = [
            self.to_gnomonic(lat=tp[0], lon=tp[1], fov=fov)
            for tp in tangent_points
        ]
        return GnomonicFaceSet(faces)

    def augment_with_rotations(
        self,
        tangent_points: List[Tuple[float, float]],
        rotations: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Augments each existing tangent point with a list of additional rotations.

        Args:
            tangent_points: Original ``(lat, lon)`` pairs.
            rotations: Each entry is ``(delta_lat, delta_lon)``.

        Returns:
            List[Tuple[float, float]]: Combined original and rotated tangent points.

        Examples:
            >>> img.augment_with_rotations([(0.0, 0.0)], [(10.0, 0.0)])
        """
        augmented = []
        for point in tangent_points:
            for (dlat, dlon) in rotations:
                augmented.append((point[0] + dlat, point[1] + dlon))
        return tangent_points + augmented

    def to_pcd(
        self,
        grad_threshold: float = 1.0,
        min_radius: float = 0.5,
        max_radius: float = 30.0
    ):
        """
        Convert this EquirectangularImage into a Point Cloud (PCD).

        Uses the PCDHandler (assumed external code) for the conversion.

        Args:
            grad_threshold: Gradient threshold for depth estimation.
            min_radius: Minimum valid radius.
            max_radius: Maximum valid radius.

        Returns:
            Some form of PCD object from the PCDHandler.

        Examples:
            >>> pcd = img.to_pcd()
        """
        from ..pcd.handler import PCDHandler  # Keep consistent with your project
        return PCDHandler.equirectangular_image_to_pcd(
            self,
            grad_threshold=grad_threshold,
            min_radius=min_radius,
            max_radius=max_radius
        )

    def clone(self) -> "EquirectangularImage":
        """
        Creates a deep copy of this object, preserving data
        and core attributes (lat, lon, shadow_angle, etc.).

        Returns:
            EquirectangularImage

        Examples:
            >>> img_copy = img.clone()
        """
        new_obj = EquirectangularImage(
            data=self.data_clone(),
            shadow_angle=self.shadow_angle,
            lat=self.lat,
            lon=self.lon
        )
        new_obj.sampler = self.sampler
        new_obj.projection = self.projection
        return new_obj

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data.

        Examples:
            >>> img.shape
        """
        return self.get_shape()

    def show(self) -> None:
        """Display the image using :mod:`PIL.Image` for a quick preview.

        Examples:
            >>> img.show()
        """
        arr = np.asarray(self.get_data())
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        Image.fromarray(arr).show()

    def __repr__(self):
        return (
            f"EquirectangularImage("
            f"lat={self.lat}, lon={self.lon}, shadow_angle={self.shadow_angle})"
        )