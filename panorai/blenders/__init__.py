# panorai/pipelines/blender/__init__.py

from .registry import BlenderRegistry
from .base_blenders import BaseBlender

# Import and register all available (NumPy-based) blenders
from .average import AverageBlender
from .feathering import FeatheringBlender
from .gaussian import GaussianBlender
from .closest import ClosestBlender
from .overlap_counter import OverlapCounterBlender
from .overlap_std_blender import OverlapStdBlender
from .std_feathering import OverlapStdFeatheredBlender
from .huber import HuberBlender
from .huber_no_confidence import HuberNoConfidenceBlender
from .huber_spatial_blender import HuberSpatialBlender
from .bundle_adjustment import BundleAdjustmentBlender

__all__ = [
    "BlenderRegistry",
    "BaseBlender",
    "AverageBlender",
    "FeatheringBlender",
    "GaussianBlender",
    "ClosestBlender",
    "OverlapCounterBlender",
    "OverlapStdBlender",
    "OverlapStdFeatheredBlender",
    "HuberBlender",
    "HuberNoConfidenceBlender",
    "HuberSpatialBlender",
    "BundleAdjustmentBlender"
]