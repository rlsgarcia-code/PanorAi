class PanoraiFactoryError(Exception):
    """Base class for all PanoraiFactory errors."""
    pass

class SamplerNotFoundError(PanoraiFactoryError):
    """Raised when a requested sampler is not registered."""
    def __init__(self, name, available):
        message = f"❌ Sampler '{name}' not found. Available samplers: {', '.join(available)}"
        super().__init__(message)

class BlenderNotFoundError(PanoraiFactoryError):
    """Raised when a requested blender is not registered."""
    def __init__(self, name, available):
        message = f"❌ Blender '{name}' not found. Available blenders: {', '.join(available)}"
        super().__init__(message)

class ProjectionNotFoundError(PanoraiFactoryError):
    """Raised when a requested projection is not registered."""
    def __init__(self, name, available):
        message = f"❌ Projection '{name}' not found. Available projections: {', '.join(available)}"
        super().__init__(message)

class ConfigNotFoundError(PanoraiFactoryError):
    """Raised when a requested configuration is not registered."""
    def __init__(self, name, available):
        message = f"❌ Configuration '{name}' not found. Available configurations: {', '.join(available)}"
        super().__init__(message)