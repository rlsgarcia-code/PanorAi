# panorai/utils/validation.py

import numpy as np
from .exceptions import InvalidDataError, ChannelMismatchError

def validate_image_data(data):
    if isinstance(data, np.ndarray):
        if data.ndim not in {2, 3}:
            raise InvalidDataError(f"Invalid ndarray shape {data.shape}. Expected 2D or 3D (H,W[,C]).")
    elif isinstance(data, dict):
        if not data:
            raise InvalidDataError("Provided image dictionary is empty.")
        first_shape = next(iter(data.values())).shape[:2]
        for key, arr in data.items():
            if arr.shape[:2] != first_shape:
                raise ChannelMismatchError(
                    f"Inconsistent shape in channel '{key}'. Expected {first_shape}, got {arr.shape[:2]}."
                )
    else:
        raise InvalidDataError(f"Data type {type(data)} is not supported. Must be np.ndarray or Dict[str,np.ndarray].")

def validate_gnomonic_data(data):
    if isinstance(data, np.ndarray):
        if data.ndim not in {2, 3}:
            raise InvalidDataError(f"Invalid GnomonicFace array shape {data.shape}. Expected (H, W[, C]).")
    elif isinstance(data, dict):
        if not data:
            raise InvalidDataError("GnomonicFace: 'data' dictionary cannot be empty.")
        ref_shape = next(iter(data.values())).shape[:2]
        for key, arr in data.items():
            if arr.shape[:2] != ref_shape:
                raise ChannelMismatchError(
                    f"Channel '{key}' has inconsistent shape. Expected {ref_shape}, got {arr.shape[:2]}."
                )
    else:
        raise InvalidDataError("GnomonicFace data must be a NumPy array or dictionary of NumPy arrays.")