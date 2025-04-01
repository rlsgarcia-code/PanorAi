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
    def blend(self, images, masks):
        """
        Perform blending on a set of images.
        """
        pass

    def update(self, **kwargs):
        """Update the blending strategy with new parameters."""
        self.params.update(kwargs)