### panorai_data_rewritten/handlers/multi_channel_handler.py ###
"""
multi_channel_handler.py
=========================

Defines `MultiChannelHandler`, a class responsible for managing single- and multi-
channel image data. Provides functionality for stacking, unstacking, applying
projections and preprocessing transformations.
"""

import numpy as np
import copy
from typing import Dict, Union, List, Callable, Tuple

from ..utils.exceptions import InvalidDataError, ChannelMismatchError
from .utils.shape_manager import ShapeManager

TensorOrArray = Union[np.ndarray]
MultiChannelDict = Dict[str, TensorOrArray]


class MultiChannelHandler:
    """
    Handles multi-channel image operations such as:
      - Stacking/unstacking
      - Transformations
      - Projection applications
      - Preprocessing
      - Cloning
    """

    def __init__(self, data: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Initializes the handler with data.

        Args:
            data (np.ndarray | Dict[str, np.ndarray]): Single-channel array or multi-channel dictionary.
        """
        if isinstance(data, np.ndarray):
            self._is_multi_channel = False
            self.data: Union[np.ndarray, Dict[str, np.ndarray]] = data.copy()
        elif isinstance(data, dict):
            self._is_multi_channel = True
            self.data = {key: ShapeManager.to_numpy(value).copy() for key, value in data.items()}
        else:
            try:
                array_data = ShapeManager.to_numpy(data)
            except Exception as exc:
                raise ValueError(
                    "Data must be a NumPy array or a dictionary of NumPy arrays."
                ) from exc

            self._is_multi_channel = False
            self.data = array_data.copy()

    def is_multi_channel(self) -> bool:
        """Returns True if the data is multi-channel."""
        return self._is_multi_channel

    def get_channels(self) -> List[str]:
        """Returns list of channel names."""
        return list(self.data.keys()) if self._is_multi_channel else ["single"]

    def stack(self) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Stacks multi-channel data into a single array.

        Returns:
            stacked (np.ndarray): Combined array of all channels.
            keys_order (List[str]): Order of keys stacked.
            channel_counts (List[int]): Channels per original key.
        """
        if not self._is_multi_channel:
            raise InvalidDataError("Stacking is only applicable to multi-channel data.")

        keys_order = sorted(self.data.keys())
        channel_counts = []
        arrays = []

        for key in keys_order:
            arr = self.data[key]
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            channel_counts.append(arr.shape[-1])
            arrays.append(arr)

        stacked = np.concatenate(arrays, axis=-1)
        return stacked, keys_order, channel_counts

    def unstack(self, stacked_data: np.ndarray, keys_order: List[str], channel_counts: List[int]) -> None:
        """
        Unstacks a stacked array into separate channels.

        Args:
            stacked_data (np.ndarray): The stacked data array.
            keys_order (List[str]): List of channel keys.
            channel_counts (List[int]): Number of channels for each key.
        """
        if not self._is_multi_channel:
            raise InvalidDataError("Unstacking is only applicable to multi-channel data.")

        total_channels = sum(channel_counts)
        if stacked_data.shape[-1] != total_channels:
            raise ChannelMismatchError(
                f"Processed output has {stacked_data.shape[-1]} channels but expected {total_channels}."
            )

        spatial_size = stacked_data.shape[:2]
        indices = np.cumsum(channel_counts)[:-1]
        splits = np.split(stacked_data, indices, axis=-1)

        self.data = {}
        for key, split, count in zip(keys_order, splits, channel_counts):
            arr = split.reshape(spatial_size + (split.shape[-1],))
            self.data[key] = arr

    def apply_on_stacked(self, func: Callable[[np.ndarray], np.ndarray]) -> "MultiChannelHandler":
        """
        Stack channels, apply transformation, unstack again.

        Args:
            func (Callable): Function applied to stacked data.

        Returns:
            MultiChannelHandler: Self with updated data.
        """
        if not self._is_multi_channel:
            raise InvalidDataError("apply_on_stacked is only applicable to multi-channel data.")

        stacked, keys_order, channel_counts = self.stack()
        transformed = func(stacked)

        if transformed.shape[-1] != sum(channel_counts):
            raise ChannelMismatchError(
                f"Processed output has {transformed.shape[-1]} channels but expected {sum(channel_counts)}."
            )

        self.unstack(transformed, keys_order, channel_counts)
        return self

    def apply_projection(self, projection: Callable[[np.ndarray], np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Applies a projection transformation.

        Args:
            projection (Callable): Projection function.

        Returns:
            np.ndarray | Dict[str, np.ndarray]: Transformed output.
        """
        if self._is_multi_channel:
            return self.apply_on_stacked(projection).data.copy()
        else:
            return projection(self.data).copy()

    def preprocess(self, preprocess_func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Applies a preprocessing function to each channel.

        Args:
            preprocess_func (Callable): Preprocessing function.
        """
        if self._is_multi_channel:
            self.apply_on_stacked(preprocess_func)
        else:
            self.data = preprocess_func(self.data)

    def get_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the underlying data."""
        if self._is_multi_channel:
            first_channel = next(iter(self.data.values()))
            return first_channel.shape
        return self.data.shape

    def data_clone(self) -> Union[np.ndarray, MultiChannelDict]:
        """Returns a deep copy of the current data."""
        if self._is_multi_channel:
            return {key: copy.deepcopy(value) for key, value in self.data.items()}
        return copy.deepcopy(self.data)

    def set_type(self) -> None:
        """Tracks and stores the data type internally."""
        if self._is_multi_channel:
            first_value = next(iter(self.data.values()))
            self._type = type(first_value)
        else:
            self._type = type(self.data)

    def squeeze_singleton_channels(self) -> None:
        """Remove trailing singleton channel dimensions from each stored array."""
        if not self._is_multi_channel:
            if isinstance(self.data, np.ndarray) and self.data.ndim == 3 and self.data.shape[-1] == 1:
                self.data = self.data[..., 0]
            return

        for key, arr in self.data.items():
            if arr.ndim == 3 and arr.shape[-1] == 1:
                self.data[key] = arr[..., 0]
