### panorai_data_rewritten/utils/__init__.py ###
"""
Utilities for PanorAI Data Module
=================================

Includes custom exceptions, shape utilities, and type consistency tools.
"""

# Expose key utilities
from ...utils.exceptions import (
    PanoraiError,
    InvalidDataError,
    ChannelMismatchError,
    MetadataValidationError,
    DataConversionError,
    MissingChannelError,
)
from .misc import ensure_type_consistency
from .shape_manager import ShapeManager

__all__ = [
    "PanoraiError",
    "InvalidDataError",
    "ChannelMismatchError",
    "MetadataValidationError",
    "DataConversionError",
    "MissingChannelError",
    "ensure_type_consistency",
    "ShapeManager",
]