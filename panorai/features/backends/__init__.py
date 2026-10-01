"""Advanced feature backends.

Normal users should construct :class:`panorai.features.SphericalFeaturePipeline`.
"""

from .opencv import OpenCVFeatureBackend

__all__ = ["OpenCVFeatureBackend"]
