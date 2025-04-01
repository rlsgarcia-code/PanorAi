"""
spherical_data.py
=================

Defines the SphericalData abstract base class, which combines array-like 
and multi-channel capabilities along with spherical coordinate attributes 
(lat, lon).
"""

from abc import ABC
import numpy as np
from typing import Any, Tuple, Union, Dict

from .array_data import ArrayLikeData
from .multi_data import MultiChannelData


class SphericalData(ArrayLikeData, MultiChannelData, ABC):
    """
    Base class for spherical image data.

    - Inherits from ArrayLikeData for NumPy-like slicing and indexing.
    - Inherits from MultiChannelData for multi-channel support.
    - Tracks spherical positioning with lat/lon.
    """

    def __init__(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        lat: float = 0.0,
        lon: float = 0.0
    ):
        """
        Initialize a SphericalData object.

        Args:
            data (np.ndarray | Dict[str, np.ndarray]): The underlying image data 
                (single- or multi-channel).
            lat (float): The latitude (degrees) associated with this data.
            lon (float): The longitude (degrees) associated with this data.
        """
        # MultiChannelData constructor
        MultiChannelData.__init__(self, data)
        self.lat = lat
        self.lon = lon
        self.projection = None  # Will be set in subclasses if needed
        self.set_type()

    def update_attributes(self, lat, lon, fov):
        """
        Convenience method to unify lat, lon, and fov updates from the 
        existing projection config or user overrides.

        Returns:
            A tuple of new (lat, lon, fov) values.
        """
        return {
            'lat': lat if lat is not None else self.projection.config.phi1_deg,
            'lon': lon if lon is not None else self.projection.config.lam0_deg,
            'fov': fov if fov is not None else self.projection.config.fov_deg
        }.values()

    def dynamic_projection(self, lat=None, lon=None, fov=None, **kwargs):
        """
        If lat/lon/fov overrides are provided, update the existing projection,
        otherwise use the current projection parameters.

        Returns:
            (projection, (lat, lon, fov)) 
            Updated projection and the resulting lat/lon/fov values.
        """
        if not self.projection:
            raise AttributeError("No projection attached to this data.")

        call_update = any(v is not None for v in [lat, lon, fov])
        if call_update:
            # Update from user overrides
            lat, lon, fov = self.update_attributes(lat, lon, fov)
            self.projection.config.update(
                phi1_deg=lat,
                lam0_deg=lon,
                fov_deg=fov
            )
        return self.projection, (lat, lon, fov)

    def get_data(self) -> np.ndarray:
        """
        Returns the underlying array data.

        - For multi-channel, returns the channels stacked along the last dimension.
        - For single-channel, returns the single array as is.

        Returns:
            np.ndarray
        """
        if self.is_multi_channel():
            stacked, _, _ = self.stack()
            return stacked
        return self.data

    @property
    def data(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Direct access to the stored (multi- or single-channel) data."""
        return self.multi_channel_handler.data

    @data.setter
    def data(self, new_data: Union[np.ndarray, Dict[str, np.ndarray]]):
        """Allows updating the underlying data with a new array or dict of arrays."""
        self.multi_channel_handler.data = new_data

    def attach_projection(self, name: str, **kwargs):
        """
        To be implemented in subclasses that define specialized projection usage.
        """
        raise NotImplementedError("attach_projection must be implemented by the subclass.")

    def attach_sampler(self, name: str, **kwargs):
        """
        To be implemented in subclasses that define specialized sampling usage.
        """
        raise NotImplementedError("attach_sampler must be implemented by the subclass.")

    def to_numpy(self, dtype=None, copy=True) -> np.ndarray:
        """
        Convenience method to get the data as a NumPy array with optional dtype/copy.
        """
        return self.__array__(dtype=dtype, copy=copy)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"lat={self.lat}, lon={self.lon})"
        )