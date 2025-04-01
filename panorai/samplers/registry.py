"""
registry.py
===========

Provides a `SamplerRegistry` for managing multiple sphere sampler classes,
allowing them to be referenced by name and instantiated with common params.
"""

import logging
from typing import Type, Dict, Any
from .base_samplers import Sampler

logger = logging.getLogger("sampler.registry")

class SamplerRegistryError(Exception):
    """Base class for registry-related errors."""
    pass

class SamplerNotFoundError(SamplerRegistryError):
    """Raised when a requested sampler is not found."""
    pass

class SamplerRegistry:
    """
    Registry for managing different sphere samplers.
    Allows registering custom Sampler classes under a string name
    and creating them later via `create()`.
    """
    
    _registry: Dict[str, Type[Sampler]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a Sampler subclass under a given name.

        Args:
            name (str): Unique identifier for the sampler.
        """
        def decorator(sampler_cls: Type[Sampler]) -> Type[Sampler]:
            if name in cls._registry:
                raise SamplerRegistryError(f"Sampler '{name}' is already registered.")
            cls._registry[name] = sampler_cls
            logger.info(f"Sampler '{name}' registered successfully.")
            return sampler_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Sampler:
        """
        Instantiates a sampler from the registry by name.

        Args:
            name (str): Sampler identifier.
            **kwargs: Parameters to pass into the sampler's constructor.

        Returns:
            Sampler: Instantiated sampler object.

        Raises:
            SamplerNotFoundError: If the name is not in the registry.
        """
        logger.debug(f"Retrieving sampler '{name}' with params: {kwargs}")
        if name not in cls._registry:
            raise SamplerNotFoundError(f"Sampler '{name}' not found in the registry.")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_samplers(cls) -> list:
        """
        List all registered sampler names.
        """
        logger.debug("Listing all registered samplers.")
        return [key for key in cls._registry]