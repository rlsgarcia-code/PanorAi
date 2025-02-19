# panorai/pipeline/blender/default_blenders.py

from .registry import BlenderRegistry
from .feathering import FeatheringBlender
from .gaussian import GaussianBlender
from .average import AverageBlender
from .closest import ClosestBlender
import logging

# Initialize logger for this module
logger = logging.getLogger('blender.default_blenders')

# Default blenders
DEFAULT_BLENDERS = {
    "FeatheringBlender": FeatheringBlender,
    "GaussianBlender": GaussianBlender,
    "AverageBlender": AverageBlender,
    "ClosestBlender": ClosestBlender
}


def register_default_blenders():
    logger.debug("Registering default blenders.")
    for name, blender_cls in DEFAULT_BLENDERS.items():
        BlenderRegistry.register(name, blender_cls)
    
    logger.debug("All default blenders registered.")