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
        lon: float = 0.0
    ) -> None:
        """
        Initializes an EquirectangularImage instance.

        Args:
            data (np.ndarray | Dict[str, np.ndarray]): Input data, either a single NumPy array
                or a dictionary of channels -> arrays.
            shadow_angle (float): Angle for shadow correction.
            lat (float): Latitude (degrees) of the image center.
            lon (float): Longitude (degrees) of the image center.
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
            name (str): The sampler name (e.g., 'cube').
            **kwargs: Additional config for the sampler.
        """
        from panorai.factory.panorai_factory import PanoraiFactory
        self.sampler = PanoraiFactory.get_sampler(name, **kwargs)

    def attach_projection(self, name: str, lat: float = 0.0, lon: float = 0.0, fov: float = 90.0, **kwargs):
        """
        Attach a projection method used for converting equirectangular data 
        to gnomonic or other coordinate systems.

        Args:
            name (str): The projection name (e.g., 'gnomonic').
            lat (float): Latitude for the projection center.
            lon (float): Longitude for the projection center.
            fov (float): Field of view in degrees.
            **kwargs: Additional config for the projection.
        """
        from panorai.factory.panorai_factory import PanoraiFactory
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
            delta_lat (float): Shift in latitude (degrees).
            delta_lon (float): Shift in longitude (degrees).
            shadow_angle (float): Shadow correction angle.
            resize_factor (float | None): Factor by which to resize the image.
            preprocessing_config (dict, optional): Additional config for Preprocessor.
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
            lat (float): Latitude of the tangent point (degrees).
            lon (float): Longitude of the tangent point (degrees).
            fov (float): Field of view in degrees.
            **kwargs: Additional parameters to pass into the projection.

        Returns:
            GnomonicFace: The resulting gnomonic face object.
        """
        from .gnomonic_image import GnomonicFace
        # 1) Possibly update or use attached projection
        projection, (lat, lon, fov) = self.dynamic_projection(lat, lon, fov, **kwargs)
        # 2) Apply projection
        projected_data = self.apply_projection(lambda d: projection.project(d))
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
            fov (float): Field of view in degrees for each face.
            sampling_method (str, optional): Sampler to use (e.g., 'cube').
            rotations (List[Tuple[float, float]]): Additional lat/lon shifts to apply.

        Returns:
            GnomonicFaceSet: A collection (set) of gnomonic faces.
        """
        from .gnomonic_imageset import GnomonicFaceSet
        if sampling_method:
            self.attach_sampler(sampling_method)

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
            tangent_points (List[Tuple[float, float]]): Original lat/lon pairs.
            rotations (List[Tuple[float, float]]): Each entry is (delta_lat, delta_lon).

        Returns:
            List[Tuple[float, float]]: Combined original and rotated tangent points.
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
            grad_threshold (float): Gradient threshold for depth estimation.
            min_radius (float): Minimum valid radius.
            max_radius (float): Maximum valid radius.

        Returns:
            Some form of PCD object from the PCDHandler.
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
        """Returns the shape of the underlying array or multi-channel data."""
        return self.get_shape()

    def __repr__(self):
        return (
            f"EquirectangularImage("
            f"lat={self.lat}, lon={self.lon}, shadow_angle={self.shadow_angle})"
        )