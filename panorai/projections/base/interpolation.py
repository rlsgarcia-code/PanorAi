from typing import Any, Optional
import cv2
import numpy as np
import logging
from ...utils.exceptions import InterpolationError

logger = logging.getLogger('spherical_projections.base.interpolation')

class BaseInterpolation:
    """
    Base class for image interpolation.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize with a configuration object.
        """
        logger.debug("Initializing BaseInterpolation.")
        # Ensure the config provides the necessary attributes.
        for attr in ("interpolation", "borderMode", "borderValue"):
            if not hasattr(config, attr):
                error_msg = f"Config must have '{attr}' attribute."
                logger.error(error_msg)
                raise TypeError(error_msg)
        self.config = config
        logger.info("BaseInterpolation initialized successfully.")

    def interpolate(
        self,
        input_img: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform interpolation using OpenCV.
        """
        logger.debug("Starting image interpolation.")
        if not isinstance(input_img, np.ndarray):
            raise InterpolationError("input_img must be a NumPy ndarray.")
        if not isinstance(map_x, np.ndarray) or not isinstance(map_y, np.ndarray):
            raise InterpolationError("map_x and map_y must be NumPy ndarrays.")

        try:
            map_x_32 = map_x.astype(np.float32)
            map_y_32 = map_y.astype(np.float32)
            logger.debug("Converted map_x and map_y to float32.")
        except Exception as e:
            raise InterpolationError(f"Conversion error: {e}") from e

        try:
            # Use the underlying integer values from enum fields
            result = cv2.remap(
                input_img,
                map_x_32,
                map_y_32,
                interpolation=self.config.interpolation.value,
                borderMode=self.config.borderMode.value,
                borderValue=self.config.borderValue
            )
            logger.debug("OpenCV remap executed successfully.")
        except cv2.error as e:
            raise InterpolationError(f"OpenCV remap failed: {e}") from e

        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise InterpolationError("mask must be a NumPy ndarray.")
            if mask.shape != result.shape[:2]:
                raise InterpolationError(
                    "mask shape must match the first two dimensions of the result.")
            if result.ndim == 2:
                result *= mask
            else:
                result *= mask[:, :, None]
            logger.debug("Mask applied successfully.")

        logger.info("Image interpolation completed successfully.")
        return result