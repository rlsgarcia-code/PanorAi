"""
factory.py
==========

Defines the factory class (`DataFactory`) responsible for creating various
spherical data objects from different input sources.
"""

import os
import numpy as np
from typing import Union, Dict, List, Any, Literal
from PIL import Image

from .equirectangular_image import EquirectangularImage
from .gnomonic_image import GnomonicFace
from .gnomonic_imageset import GnomonicFaceSet


class DataFactory:
    """
    Factory class for creating spherical data objects:
      - ``EquirectangularImage``
      - ``GnomonicFace``
      - ``GnomonicFaceSet``

    Provides convenient methods for constructing objects from:
      - NumPy arrays
      - Dicts of NumPy arrays (multi-channel)
      - Lists of gnomonic faces
      - PIL images
      - Image files on disk
    """

    def __init__(self, data: Union[EquirectangularImage, GnomonicFace, GnomonicFaceSet]) -> None:
        """
        Initializes the DataFactory with a pre-existing spherical data object.

        Args:
            data (EquirectangularImage | GnomonicFace | GnomonicFaceSet): 
                The spherical data object to manage or serialize.
        """
        self.data = data

    @classmethod
    def from_array(cls, data: np.ndarray, data_type: str) -> Union[EquirectangularImage, GnomonicFace]:
        """
        Creates a spherical data object from a NumPy array.

        Args:
            data (np.ndarray): Input array representing the spherical data.
            data_type (str): Must be either 'equirectangular' or 'gnomonic_face'.

        Returns:
            EquirectangularImage or GnomonicFace
        """
        if data_type.lower() == "equirectangular":
            return EquirectangularImage(data)
        elif data_type.lower() == "gnomonic_face":
            return GnomonicFace(data, lat=0.0, lon=0.0, fov=90.0)
        else:
            raise ValueError("data_type must be either 'equirectangular' or 'gnomonic_face'")

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray], data_type: str) -> Union[EquirectangularImage, GnomonicFace]:
        """
        Creates a multi-channel spherical data object from a dictionary of NumPy arrays.

        Args:
            data (Dict[str, np.ndarray]): Mapping of channel name -> NumPy array.
            data_type (str): 'equirectangular' or 'gnomonic_face'.

        Returns:
            EquirectangularImage or GnomonicFace
        """
        if data_type.lower() == "equirectangular":
            return EquirectangularImage(data)
        elif data_type.lower() == "gnomonic_face":
            return GnomonicFace(data, lat=0.0, lon=0.0, fov=90.0)
        else:
            raise ValueError("data_type must be either 'equirectangular' or 'gnomonic_face'")

    @classmethod
    def from_list(cls, faces: List[GnomonicFace], channel_name: str = "default") -> GnomonicFaceSet:
        """
        Creates a GnomonicFaceSet from a list of GnomonicFace objects.

        Args:
            faces (List[GnomonicFace]): A list of gnomonic faces.
            channel_name (str): Label for the data channel.

        Returns:
            GnomonicFaceSet
        """
        face_set = GnomonicFaceSet(faces, channel_name)

        # Attach a default blender if multiple faces are present
        if len(faces) > 1:
            face_set.attach_blender("average")

        return face_set

    @classmethod
    def from_pil(cls, img: Image.Image, data_type: str) -> Union[EquirectangularImage, GnomonicFace]:
        """
        Creates a spherical data object from a PIL image.

        Args:
            img (PIL.Image.Image): Input image loaded via PIL.
            data_type (str): Must be either 'equirectangular' or 'gnomonic_face'.

        Returns:
            EquirectangularImage or GnomonicFace
        """
        data = np.asarray(img).copy()
        return cls.from_array(data, data_type)

    @classmethod
    def from_file(cls, file_path: str, data_type: Literal["equirectangular", "gnomonic_face"]) -> Union[EquirectangularImage, GnomonicFace]:
        """
        Creates a spherical data object from an image file on disk.

        Args:
            file_path (str): Path to the image file.
            data_type (Literal["equirectangular", "gnomonic_face"]): 
                Type of data to construct.

        Returns:
            EquirectangularImage or GnomonicFace

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If loading the image fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {e}")

        return cls.from_pil(img, data_type)