"""
Base module for projection components.

This module includes base classes for projection strategy,
grid generation, interpolation, and coordinate transformations.
"""

from .strategy import BaseProjectionStrategy
from .grid import BaseGridGeneration
from .interpolation import BaseInterpolation
from .transform import BaseCoordinateTransformer
from ...utils.exceptions import (
    ProjectionError,
    ConfigurationError,
    RegistrationError,
    ProcessingError,
    GridGenerationError,
    TransformationError,
    InterpolationError,
)

__all__ = [
    "BaseProjectionStrategy",
    "BaseGridGeneration",
    "BaseInterpolation",
    "BaseCoordinateTransformer",
    "ProjectionError",
    "ConfigurationError",
    "RegistrationError",
    "ProcessingError",
    "GridGenerationError",
    "TransformationError",
    "InterpolationError",
]