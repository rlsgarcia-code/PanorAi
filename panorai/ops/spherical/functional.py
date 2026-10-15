"""Short aliases for the existing public sphere-native NumPy functions."""

from panorai.image_processing import (
    spherical_gaussian_blur as gaussian_blur,
    spherical_resize as resize,
    spherical_rotate as rotate,
)

__all__ = ["gaussian_blur", "resize", "rotate"]
