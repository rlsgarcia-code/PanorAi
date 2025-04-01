"""
multi_data.py
=============

Defines MultiChannelData, a base class that uses MultiChannelHandler to
support both single-channel and multi-channel data seamlessly.
"""

from typing import Union, Dict
import numpy as np

from .multi_handler import MultiChannelHandler


class MultiChannelData:
    """
    Base class for handling single-channel or multi-channel data 
    via a `MultiChannelHandler`.

    Inheritors must define how `get_data()` is obtained or overridden.
    """

    def __init__(self, data: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Initializes the MultiChannelHandler with the provided data.

        Args:
            data (np.ndarray | Dict[str, np.ndarray]): Single or multi-channel data.
        """
        self.multi_channel_handler = MultiChannelHandler(data)

    def is_multi_channel(self) -> bool:
        """Returns True if data is stored as multiple channels."""
        return self.multi_channel_handler.is_multi_channel()

    def get_data(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns the underlying data in the raw format (dict or array).
        Subclasses often override or rely on other hooking methods.
        """
        return self.multi_channel_handler.get_data()

    def get_channels(self):
        """List of channel names if multi-channel, or ['single'] otherwise."""
        return self.multi_channel_handler.get_channels()

    def stack(self):
        """Stacks multi-channel data into a single array (H,W,C)."""
        return self.multi_channel_handler.stack()

    def unstack(self, stacked_data, keys_order, channel_counts):
        """Unstacks a multi-channel array into separate channels."""
        self.multi_channel_handler.unstack(stacked_data, keys_order, channel_counts)

    def apply_on_stacked(self, func):
        """
        Convenience method: stack channels, apply a function, unstack results.
        """
        return self.multi_channel_handler.apply_on_stacked(func)

    def apply_projection(self, projection):
        """
        Apply a projection function to the data (stacked if multi-channel).
        """
        return self.multi_channel_handler.apply_projection(projection)

    def preprocess(self, preprocess_func):
        """
        Apply a generic preprocessing function to each channel or entire array.
        """
        return self.multi_channel_handler.preprocess(preprocess_func)

    def get_shape(self):
        """Returns shape of the underlying single or multi-channel data."""
        return self.multi_channel_handler.get_shape()

    def data_clone(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns a deep copy of the current data (multi-channel or single).

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]
        """
        return self.multi_channel_handler.data_clone()

    def set_type(self) -> None:
        """
        Internal utility to track the type of the underlying data.
        """
        self.multi_channel_handler.set_type()