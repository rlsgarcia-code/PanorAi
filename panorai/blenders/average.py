import numpy as np
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

@BlenderRegistry.register("average")
class AverageBlender(BaseBlender):
    def blend(self, images, masks, **kwargs):
        """
        Blends images using simple averaging (NumPy-based).
        """
        
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)
  
        for img, mask in zip(images, masks):
            if img.ndim == 3:
                equirect_mask = np.any(img != 0, axis=-1)
            else:
                equirect_mask = img != 0
            combined += img
            weight_map += equirect_mask

        valid_weights = weight_map > 0
        if combined.ndim == 3:
            combined[valid_weights] /= weight_map[valid_weights, None]
        else:
            combined[valid_weights] /= weight_map[valid_weights]
        combined[~valid_weights] = 0
        
        return combined
