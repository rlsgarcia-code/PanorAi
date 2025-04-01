from typing import Any, Tuple
import numpy as np
import logging
from ...utils.exceptions import ProcessingError

logger = logging.getLogger('spherical_projections.base.strategy')

class BaseProjectionStrategy:
    """
    Base class for projection strategies.
    """

    @classmethod
    def from_spherical_to_projection(cls, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Starting forward projection in BaseProjectionStrategy.")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ProcessingError("x and y must be NumPy ndarrays.")
        raise NotImplementedError("Subclasses must implement from_spherical_to_projection.")

    @classmethod
    def from_projection_to_spherical(cls, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.debug("Starting backward projection in BaseProjectionStrategy.")
        if not isinstance(lat, np.ndarray) or not isinstance(lon, np.ndarray):
            raise ProcessingError("lat and lon must be NumPy ndarrays.")
        raise NotImplementedError("Subclasses must implement from_projection_to_spherical.")