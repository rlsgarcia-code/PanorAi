### data/utils/data_exceptions.py ###
class PanoraiError(Exception):
    """Base class for exceptions in the PanorAi framework."""
    pass


class InvalidDataError(PanoraiError):
    """Raised when provided data is invalid or corrupted."""
    pass


class ChannelMismatchError(PanoraiError):
    """Raised when channel counts or shapes are inconsistent."""
    pass


class MetadataValidationError(PanoraiError):
    """Raised when lat/lon/fov metadata is inconsistent or malformed."""
    pass


class DataConversionError(PanoraiError):
    """Raised when NumPy/Torch conversion fails (Torch support deprecated)."""
    pass


class MissingChannelError(PanoraiError):
    """Raised when a requested channel is not found in the data."""
    pass