import numpy as np
from typing import Optional, Any
from .transformations import PreprocessEquirectangularImage

class Preprocessor:
    """
    Handles preprocessing (NumPy-based).
    """

    @staticmethod
    def preprocess_eq(
        eq_image: np.ndarray,
        shadow_angle: Optional[float] = None,
        delta_lat: Optional[float] = None,
        delta_lon: Optional[float] = None,
        resize_factor: Optional[float] = None,
        resize_method: Optional[str] = None,
        config: Optional[Any] = None
    ) -> np.ndarray:
        """
        Applies transformations to an equirectangular image using NumPy-based logic.
        """

        if config is not None:
            if hasattr(config._config, "model_dump"):
                defaults = config._config.model_dump()
            else:
                defaults = config._config.dict()
            #print('shadow_angle')
            if shadow_angle is None:
                shadow_angle = defaults.get("shadow_angle", 0.0)
            if delta_lat is None:
                delta_lat = defaults.get("delta_lat", 0.0)
            if delta_lon is None:
                delta_lon = defaults.get("delta_lon", 0.0)
            if resize_factor is None:
                resize_factor = defaults.get("resize_factor", 1.0)
            if resize_method is None:
                resize_method = defaults.get("resize_method", "skimage")
        else:
            shadow_angle = shadow_angle if shadow_angle is not None else 0.0
            delta_lat = delta_lat if delta_lat is not None else 0.0
            delta_lon = delta_lon if delta_lon is not None else 0.0
            resize_factor = resize_factor if resize_factor is not None else 1.0
            resize_method = resize_method if resize_method is not None else "skimage"

        dtype = eq_image.dtype
        processed_eq_image = PreprocessEquirectangularImage.preprocess(
            eq_image,
            shadow_angle=shadow_angle,
            delta_lat=delta_lat,
            delta_lon=delta_lon,
            resize_factor=resize_factor,
            resize_method=resize_method
        )
        
        return processed_eq_image.astype(dtype)