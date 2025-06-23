# panorai/utils/exceptions.py

class PanoraiError(Exception):
    """Base exception for all Panorai-related errors."""
    pass

class InvalidDataError(PanoraiError):
    """Raised when provided data is invalid or incorrectly formatted."""
    pass

class ChannelMismatchError(PanoraiError):
    """Raised when image channels do not have matching shapes."""
    pass

class MetadataValidationError(PanoraiError):
    """Raised when lat/lon/fov metadata is inconsistent or malformed."""
    pass

class DataConversionError(PanoraiError):
    """Raised when NumPy/Torch conversion fails."""
    pass

class MissingChannelError(PanoraiError):
    """Raised when a requested channel is not found in the data."""
    pass

class FaceSetError(PanoraiError):
    """Raised for errors in GnomonicFaceSet operations."""
    pass

class ImageProcessingError(PanoraiError):
    """Raised for errors during preprocessing or transformation."""
    pass


"""
Custom exception classes for the Gnomonic Projection module.
"""

class ProjectionError(Exception):
    pass

class ConfigurationError(ProjectionError):
    pass

class RegistrationError(ProjectionError):
    pass

class ProcessingError(ProjectionError):
    pass

class GridGenerationError(ProjectionError):
    pass

class TransformationError(ProjectionError):
    pass

class InterpolationError(ProjectionError):
    pass