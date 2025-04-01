"""
base_samplers.py
================

Defines the abstract Sampler class, which outlines the interface
for generating tangent (lat, lon) points on a sphere.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from .config import SamplerConfig

class Sampler(ABC):
    """
    Abstract base class for sphere samplers (NumPy-based).
    Each sampler must define how `get_tangent_points()` works.
    """

    def __init__(self, config: Optional[SamplerConfig] = None, **kwargs: Any) -> None:
        """
        If no config is provided, build one from kwargs.

        Args:
            config (SamplerConfig, optional): Pre-built configuration object.
            **kwargs: Additional parameters that get passed to SamplerConfig.
        """
        if config is not None:
            self.config = config
        else:
            self.config = SamplerConfig(**kwargs)

    @abstractmethod
    def get_tangent_points(self) -> List[Tuple[float, float]]:
        """
        Returns a list of (lat, lon) pairs in degrees that define
        the tangent points on the sphere.
        """
        pass

    def update(self, **kwargs: Any) -> None:
        """
        Update config parameters.
        """
        self.config.update(**kwargs)

    def _rotate_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Apply any configured rotations to the original set of points.
        This is invoked at the end of `get_tangent_points()` for convenience.
        """
        rotations = self.config.rotations if hasattr(self.config, "rotations") else []
        augmented_points = points[:]
        for lat_rot, lon_rot in rotations:
            for lat, lon in points:
                new_lat = lat + lat_rot
                new_lon = lon + lon_rot

                if new_lat > 90:
                    new_lat = 180 - new_lat
                    new_lon += 180
                elif new_lat < -90:
                    new_lat = -180 - new_lat
                    new_lon += 180

                new_lon = (new_lon + 180) % 360 - 180
                augmented_points.append((new_lat, new_lon))

        return augmented_points