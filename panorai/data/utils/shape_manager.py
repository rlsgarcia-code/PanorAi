### panorai_data_rewritten/utils/shape_manager.py ###
import numpy as np
from typing import Union

TensorOrArray = Union[np.ndarray]

class ShapeManager:
    """
    Utility class for managing and validating array shapes.

    - Converts inputs to NumPy arrays.
    - Used during loading, preprocessing, and conversion steps.
    """

    @staticmethod
    def to_numpy(data: TensorOrArray, dtype=np.float32) -> np.ndarray:
        """
        Converts arbitrary input to a NumPy array with optional dtype.

        Args:
            data (TensorOrArray): Input data.
            dtype (type, optional): Data type to convert to. Defaults to np.float32.

        Returns:
            np.ndarray: Standardized NumPy array.
        """
        return np.array(data, dtype=dtype)