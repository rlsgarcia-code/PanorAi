"""
panorai_data_rewritten
======================

A reorganized version of the `panorai_data` module, containing:
 - Factory methods for creating spherical data objects
 - Equirectangular image class
 - Gnomonic face class
 - Gnomonic face set class
 - Shared spherical data abstraction and multi-channel handling
 - Utility functions and exceptions
"""

from .factory import DataFactory
from .equirectangular_image import EquirectangularImage
from .gnomonic_image import GnomonicFace
from .gnomonic_imageset import GnomonicFaceSet
from .xyz_image_dataset import XYZImageDataset

__all__ = [
    "DataFactory",
    "EquirectangularImage",
    "GnomonicFace",
    "GnomonicFaceSet",
    "XYZImageDataset",
]
