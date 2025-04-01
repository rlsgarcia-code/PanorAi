"""
config.py
=========

Provides the `SamplerConfig` class for storing sampler
parameters (number of points, rotations, etc.).
"""

from typing import Any, Optional, List
from pydantic import BaseModel, Field
import logging
from ..config.registry import ConfigRegistry
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger("panorai.pipelines.sampler.config")

class SamplerConfigModel(BaseModel):
    """
    Pydantic-based model containing sampler parameters.
    """
    rotations: Optional[List[tuple]] = Field(default_factory=list)
    n_points: Optional[int] = Field(default=100)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

@ConfigRegistry.register("sampler_config")
class SamplerConfig:
    """
    Wraps a pydantic SamplerConfigModel, providing typed fields 
    for sampler usage. 
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize from kwargs, validating via SamplerConfigModel.

        Raises:
            ConfigurationError: If validation fails.
        """
        logger.info("Initializing SamplerConfig with parameters: %s", kwargs)
        try:
            self._config = SamplerConfigModel(**kwargs)
            logger.info("SamplerConfig initialized successfully.")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize SamplerConfig: {e}") from e

    def update(self, **kwargs: Any) -> None:
        """
        Update config fields with new parameters.
        """
        logger.debug("Updating SamplerConfig with: %s", kwargs)
        self._config = self._config.model_copy(update=kwargs)

    def __getattr__(self, item: str) -> Any:
        """
        Allows direct attribute-like access to the underlying model fields.
        """
        try:
            return getattr(self._config, item)
        except AttributeError:
            raise AttributeError(f"'SamplerConfig' object has no attribute '{item}'") from None

    def __getitem__(self, key: str) -> Any:
        """
        Dict-like access.
        """
        if hasattr(self._config, key):
            return getattr(self._config, key)
        raise KeyError(f"'{key}' not found in SamplerConfig.")

    def __iter__(self):
        """
        Allows iteration over the config's dictionary representation.
        """
        return iter(self._config.dict())

    def __repr__(self) -> str:
        """
        String representation for debugging/logging.
        """
        return f"SamplerConfig({self._config.dict()})"