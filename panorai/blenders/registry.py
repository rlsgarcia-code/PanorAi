# panorai/pipelines/blender/registry.py

from importlib import import_module
from typing import Any, Dict, Type
import logging

logger = logging.getLogger("blender.registry")


class BlenderRegistry:
    """
    Registry for managing blending strategies.
    """

    _registry: Dict[str, Type[Any]] = {}

    _builtins = {
        "average": ("panorai.blenders.average", "AverageBlender"),
        "bundle_adjustment": (
            "panorai.blenders.bundle_adjustment",
            "BundleAdjustmentBlender",
        ),
        "closest": ("panorai.blenders.closest", "ClosestBlender"),
        "counter": ("panorai.blenders.overlap_counter", "OverlapCounterBlender"),
        "feathering": ("panorai.blenders.feathering", "FeatheringBlender"),
        "gaussian": ("panorai.blenders.gaussian", "GaussianBlender"),
        "huber": ("panorai.blenders.huber", "HuberBlender"),
        "huber_no_confidence": (
            "panorai.blenders.huber_no_confidence",
            "HuberNoConfidenceBlender",
        ),
        "huber_spatial": (
            "panorai.blenders.huber_spatial_blender",
            "HuberSpatialBlender",
        ),
        "std": ("panorai.blenders.overlap_std_blender", "OverlapStdBlender"),
        "std_feathered": (
            "panorai.blenders.std_feathering",
            "OverlapStdFeatheredBlender",
        ),
    }

    @classmethod
    def _ensure_builtin_blenders(cls) -> None:
        for name, (module_name, class_name) in cls._builtins.items():
            if name in cls._registry:
                continue
            module = import_module(module_name)
            cls._registry.setdefault(name, getattr(module, class_name))

    @classmethod
    def register(cls, name: str):
        def decorator(blender_class):
            cls._registry[name] = blender_class
            logger.info(f"Blender '{name}' registered successfully.")
            return blender_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any):
        cls._ensure_builtin_blenders()
        logger.debug(f"Retrieving blender '{name}' with override parameters: {kwargs}")
        if name not in cls._registry:
            error_msg = f"Blender '{name}' not found in the registry."
            logger.error(error_msg)
            raise ValueError(error_msg)

        blender = cls._registry[name](**kwargs)
        return blender

    @classmethod
    def available_blenders(cls) -> list:
        cls._ensure_builtin_blenders()
        logger.debug("Listing all registered blenders.")
        return list(cls._registry.keys())
