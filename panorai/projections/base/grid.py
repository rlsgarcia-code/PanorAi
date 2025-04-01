from typing import Any, Tuple
import numpy as np
import logging
from ...utils.exceptions import GridGenerationError, ProcessingError

logger = logging.getLogger('spherical_projections.base.grid')

class BaseGridGeneration:
    """
    Base class for grid generation in projections.
    """

    def __init__(self, config: Any):
        """
        Initialize with a configuration object supporting attribute access.
        """
        logger.debug("Initializing BaseGridGeneration.")
        self.config = config

    def projection_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the forward projection grid.
        Must be implemented by subclasses.
        """
        logger.debug("projection_grid method called (Base class).")
        raise NotImplementedError("Subclasses must implement projection_grid.")