from typing import Any, Optional
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
except ImportError:  # pragma: no cover - Pydantic < 2 compatibility
    from pydantic import BaseModel, Field, validator
    from functools import partial
    field_validator = partial(validator, allow_reuse=True)  # type: ignore
    ConfigDict = dict  # type: ignore
import cv2
import logging
from enum import Enum
from ...config.registry import ConfigRegistry  # Centralized registry
from ...utils.exceptions import ConfigurationError

logger = logging.getLogger("gnomonic.config")

class OpenCVInterpolation(Enum):
    """Valid OpenCV interpolation methods."""
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4

class OpenCVBorderMode(Enum):
    """Valid OpenCV border modes."""
    BORDER_CONSTANT = cv2.BORDER_CONSTANT
    BORDER_REPLICATE = cv2.BORDER_REPLICATE
    BORDER_REFLECT = cv2.BORDER_REFLECT
    BORDER_WRAP = cv2.BORDER_WRAP
    BORDER_REFLECT_101 = cv2.BORDER_REFLECT_101
    BORDER_TRANSPARENT = cv2.BORDER_TRANSPARENT

class GnomonicConfigModel(BaseModel):
    """Pydantic model for Gnomonic projection configuration.
    
    This model holds:
      - Projection parameters: R, phi1_deg, lam0_deg, fov_deg.
      - Grid resolutions: x_points, y_points, lon_points, lat_points.
      - Geographic bounds: lon_min, lon_max, lat_min, lat_max.
      - Interpolation settings: interpolation, borderMode, borderValue.
    """
    R: float = Field(1.0, description="Radius of the sphere (e.g., Earth) in consistent units.")
    phi1_deg: float = Field(0.0, description="Latitude of the projection center in degrees.")
    lam0_deg: float = Field(0.0, description="Longitude of the projection center in degrees.")
    fov_deg: float = Field(90.0, description="Field of view in degrees.")

    x_points: int = Field(1024, description="Number of grid points in the x-direction.")
    y_points: int = Field(1024, description="Number of grid points in the y-direction.")
    lon_points: int = Field(1024 * 2, description="Number of longitude points for inverse grid mapping.")
    lat_points: int = Field(512 * 2, description="Number of latitude points for inverse grid mapping.")

    lon_min: float = Field(-180.0, description="Minimum longitude in the grid (degrees).")
    lon_max: float = Field(180.0, description="Maximum longitude in the grid (degrees).")
    lat_min: float = Field(-90.0, description="Minimum latitude in the grid (degrees).")
    lat_max: float = Field(90.0, description="Maximum latitude in the grid (degrees).")

    interpolation: OpenCVInterpolation = Field(OpenCVInterpolation.INTER_NEAREST, description="Interpolation method for OpenCV remap.")
    borderMode: OpenCVBorderMode = Field(OpenCVBorderMode.BORDER_CONSTANT, description="Border mode for OpenCV remap.")
    borderValue: Optional[Any] = Field(default=0, description="Border value for OpenCV remap.")

    @field_validator("fov_deg")
    def validate_fov(cls, v):
        if not (0 < v < 180):
            raise ValueError("Field of view (fov_deg) must be between 0 and 180 degrees.")
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

@ConfigRegistry.register("gnomonic_config")
class GnomonicConfig:
    """Configuration class for Gnomonic projections using Pydantic for validation.
    
    Provides hybrid attribute and dictionary access.
    """

    def __init__(self, **kwargs: Any) -> None:
        logger.info("Initializing GnomonicConfig with parameters: %s", kwargs)
        try:
            self._config = GnomonicConfigModel(**kwargs)
            logger.info("GnomonicConfig initialized successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize GnomonicConfig: {e}"
            logger.exception(error_msg)
            raise ConfigurationError(error_msg) from e

    def update(self, **kwargs: Any) -> None:
        logger.debug("Updating GnomonicConfig with parameters: %s", kwargs)
        try:
            if hasattr(self._config, "model_copy"):
                self._config = self._config.model_copy(update=kwargs)
            else:
                self._config = self._config.copy(update=kwargs)
            logger.info("GnomonicConfig updated successfully.")
        except Exception as e:
            error_msg = f"Failed to update GnomonicConfig: {e}"
            logger.exception(error_msg)
            raise ConfigurationError(error_msg) from e

    def __getattr__(self, item: str) -> Any:
        logger.debug("Accessing GnomonicConfig attribute '%s'.", item)
        try:
            return getattr(self._config, item)
        except AttributeError:
            error_msg = f"'GnomonicConfig' object has no attribute '{item}'"
            logger.error(error_msg)
            raise AttributeError(error_msg) from None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self._config, key):
            return getattr(self._config, key)
        raise KeyError(f"'{key}' not found in GnomonicConfig.")

    def _to_dict(self) -> dict:
        if hasattr(self._config, "model_dump"):
            return self._config.model_dump()
        return self._config.dict()

    def __iter__(self):
        return iter(self._to_dict())

    def __repr__(self) -> str:
        return f"GnomonicConfig({self._to_dict()})"
