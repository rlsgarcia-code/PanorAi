# panorai/pipeline/blender/feathering.py

import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_blenders import BaseBlender
from typing import Any

class FeatheringBlender(BaseBlender):


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
            valid_mask = np.max(img > 0, axis=-1).astype(np.float32)

            # Feather the mask using Euclidean distance transform
            distance = distance_transform_edt(valid_mask)
            feathered_mask = distance / distance.max()  # Normalize to [0, 1]

            # Apply blending
            combined += img * feathered_mask[..., None]
            weight_map += feathered_mask

        # Normalize the blended image
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Ensure zero weights remain zero
        combined[~valid_weights] = 0
        print(combined.dtype, combined.max)
        return combined