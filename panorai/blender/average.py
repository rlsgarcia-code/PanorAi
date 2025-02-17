# panorai/pipeline/blender/feathering.py

import numpy as np
from .base_blenders import BaseBlender
from typing import Any

import numpy as np




class AverageBlender(BaseBlender):


    def blend(self, images, masks, **kwargs):
        """
        Blends images using feathering.

        :param images: List of image arrays.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)
  
        for img, mask in zip(images, masks):
            
            equirect_mask = (np.mean(img,axis=-1) > 0 )
            # Apply blending
            combined += img 
            weight_map += equirect_mask

        # Normalize the blended image
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Ensure zero weights remain zero
        combined[~valid_weights] = 0
        return combined