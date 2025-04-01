"""
array_data.py
=============

Defines a lightweight wrapper class (`ArrayLikeData`) providing NumPy-like
interfaces for indexing, universal functions, etc., on top of the raw data.
"""

import numpy as np
from typing import Union, Tuple


class ArrayLikeData:
    """
    Provides NumPy-like behavior for image data objects.

    This class is intended to be inherited by classes that store their data
    either as a single NumPy array or as multi-channel data.
    """

    def __array__(self, dtype=None, copy=True) -> np.ndarray:
        """
        Converts the object into a NumPy array.

        Args:
            dtype: If provided, cast the returned array to this dtype.
            copy (bool): Whether to return a copy or allow a shared buffer.

        Returns:
            np.ndarray
        """
        arr = self.get_data()
        if dtype:
            arr = arr.astype(dtype, copy=copy)
        return arr.copy() if copy else arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Ensures that NumPy universal functions (ufuncs) can operate on this class.
        """
        # Convert any `ArrayLikeData` instances in inputs to actual np.ndarray
        real_inputs = tuple(np.asarray(i) if isinstance(i, ArrayLikeData) else i
                            for i in inputs)
        result = getattr(ufunc, method)(*real_inputs, **kwargs)

        # If ufunc returns multiple arrays, wrap each in the same class if it’s an array
        if isinstance(result, tuple):
            return tuple(self.__class__(r) if isinstance(r, np.ndarray) else r
                         for r in result)
        elif isinstance(result, np.ndarray):
            return self.__class__(result)
        return result

    def __getitem__(self, index) -> Union[np.ndarray, "ArrayLikeData"]:
        """
        Allows array-like indexing of the underlying data.

        Args:
            index: Standard Python/NumPy indexing.
        """
        if self.is_multi_channel():
            # If multi-channel, return a dict of channel slices
            return {
                ch: self.data[ch][index]
                for ch in self.get_channels()
            }
        return self.data[index]

    def __setitem__(self, index, value) -> None:
        """
        Allows setting values like an array.

        Args:
            index: Python/NumPy indexing.
            value: The value(s) to set.
        """
        if self.is_multi_channel():
            for ch in self.get_channels():
                self.data[ch][index] = value
        else:
            self.data[index] = value

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the underlying data.

        For multi-channel data, returns the shape of the first channel.
        """
        if self.is_multi_channel():
            first_channel = next(iter(self.data.values()))
            return first_channel.shape
        return self.data.shape