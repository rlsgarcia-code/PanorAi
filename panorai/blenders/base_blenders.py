# panorai/pipelines/blender/base_blenders.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseBlender(ABC):
    def __init__(self, **kwargs: Any) -> None:
        """
        Base Blender initialization.
        """
        self.params: Dict[str, Any] = kwargs
        
    @abstractmethod
    def blend(self, images, masks, *, return_mask=False):
        """
        Blend equal-shaped images using explicit spatial validity masks.

        Implementations must not infer validity from pixel values. By default
        they return an array with the documented shape. With
        ``return_mask=True`` they return ``(array, support_mask)``.
        """
        pass

    def update(self, **kwargs):
        """Update the blending strategy with new parameters."""
        self.params.update(kwargs)
