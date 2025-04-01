### panorai_data_rewritten/utils/misc.py ###
import numpy as np

def ensure_type_consistency(input1, input2):
    """
    Ensures that `input2` matches the type or dtype of `input1`.
    Primarily used to normalize data types across operations.

    Args:
        input1: Reference input.
        input2: Value to cast or convert.

    Returns:
        Converted `input2` matching the type of `input1`.

    Raises:
        TypeError: If the input1 type is not supported.
    """
    if isinstance(input1, np.ndarray):
        dtype = input1.dtype
        if isinstance(input2, np.ndarray):
            return input2.astype(dtype)
        else:
            return np.array(input2, dtype=dtype)

    if isinstance(input1, int):
        return int(input2)

    if isinstance(input1, float):
        return float(input2)

    raise TypeError(f"Unsupported input1 type: {type(input1)}")

