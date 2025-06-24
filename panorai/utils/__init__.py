from .resizer import ResizerConfig, ImageResizer
from .logging_config import setup_logging

# PreprocessEquirectangularImage relies on optional heavy dependencies
# (numpy, cv2, skimage). Import it lazily so that modules which only
# require the lightweight utilities can still be imported when those
# dependencies are not installed.  Tests that need the real
# implementation will ensure the dependencies are available.
try:  # pragma: no cover - simple import guard
    from ..preprocessing.transformations import PreprocessEquirectangularImage
except Exception:  # noqa: BLE001 - broad to cover any optional deps missing
    PreprocessEquirectangularImage = None

__all__ = [
    "ResizerConfig",
    "ImageResizer",
    "PreprocessEquirectangularImage",
    "setup_logging",
]