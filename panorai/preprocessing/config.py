from typing import Any, Optional
from pydantic import BaseModel, Field, validator
import logging
from ..config.registry import ConfigRegistry
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger("panorai.preprocessing.config")

class PreprocessorConfigModel(BaseModel):
    shadow_angle: float = Field(0.0)
    delta_lat: float = Field(0.0)
    delta_lon: float = Field(0.0)
    resize_factor: float = Field(1.0)
    resize_method: str = Field("skimage")

    @validator("resize_factor")
    def validate_resize_factor(cls, v):
        if v <= 0:
            raise ValueError("resize_factor must be positive.")
        return v

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

@ConfigRegistry.register("preprocessor_config")
class PreprocessorConfig:
    def __init__(self, **kwargs: Any) -> None:
        logger.info("Initializing PreprocessorConfig with parameters: %s", kwargs)
        try:
            self._config = PreprocessorConfigModel(**kwargs)
            logger.info("PreprocessorConfig initialized successfully.")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize PreprocessorConfig: {e}") from e

    def update(self, **kwargs: Any) -> None:
        logger.debug("Updating PreprocessorConfig with: %s", kwargs)
        try:
            self._config = self._config.model_copy(update=kwargs)
            logger.info("PreprocessorConfig updated successfully.")
        except Exception as e:
            raise ConfigurationError(f"Failed to update PreprocessorConfig: {e}") from e

    def __getattr__(self, item: str) -> Any:
        try:
            return getattr(self._config, item)
        except AttributeError:
            raise AttributeError(f"'PreprocessorConfig' object has no attribute '{item}'") from None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self._config, key):
            return getattr(self._config, key)
        raise KeyError(f"'{key}' not found in PreprocessorConfig.")

    def __iter__(self):
        return iter(self._config.dict())

    def __repr__(self) -> str:
        return f"PreprocessorConfig({self._config.dict()})"