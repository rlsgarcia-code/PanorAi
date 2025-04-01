from typing import Any, Tuple
import numpy as np
import logging
from ...utils.exceptions import TransformationError

logger = logging.getLogger('spherical_projections.base.transform')

class BaseCoordinateTransformer:
    """
    Base class for coordinate transformations.
    """

    def __init__(self, config: Any) -> None:
        logger.debug("Initializing BaseCoordinateTransformer.")
        self.config = config

    @classmethod
    def spherical_to_image_coords(
        cls, lat: np.ndarray, lon: np.ndarray, config: Any, shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement spherical_to_image_coords.")

    @staticmethod
    def projection_to_image_coords(
        x: np.ndarray, y: np.ndarray, config: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement projection_to_image_coords.")