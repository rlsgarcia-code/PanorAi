# panorai/pipelines/blender/registry.py

from typing import Any, Dict, Type
import logging
from functools import wraps

logger = logging.getLogger("blender.registry")

class BlenderRegistry:
    """
    Registry for managing blending strategies.
    """
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(blender_class):
            cls._registry[name] = blender_class
            logger.info(f"Blender '{name}' registered successfully.")
            return blender_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any):
        logger.debug(f"Retrieving blender '{name}' with override parameters: {kwargs}")
        if name not in cls._registry:
            error_msg = f"Blender '{name}' not found in the registry."
            logger.error(error_msg)
            raise ValueError(error_msg)

        blender = cls._registry[name](**kwargs)
        return blender

    @classmethod
    def available_blenders(cls) -> list:
        logger.debug("Listing all registered blenders.")
        return list(cls._registry.keys())