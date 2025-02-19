import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_blenders import BaseBlender
from typing import Any


class ClosestBlender(BaseBlender):

    def blend(self, images, masks, **kwargs):
        """
        Blends images by selecting the value closest to the center of the valid mask.

        :param images: List of image arrays.
        :param masks: List of corresponding masks.
        :return: Blended image.
        """
        if not images or len(images) == 0:
            raise ValueError("Images must be a non-empty list.")

        img_shape = images[0].shape
        blended = np.zeros(img_shape, dtype=np.float32)
        mask_sums = np.zeros(img_shape[:2], dtype=np.float32)

        # Compute distance transforms for valid regions
        distances = []
        valid_masks = []

        for img in images:
            valid_mask = np.max(img > 0, axis=-1).astype(bool)  # Extract mask from image
            valid_masks.append(valid_mask)
            distance = distance_transform_edt(valid_mask.astype(np.float32))
            distances.append(distance)

        # Stack distances and ignore invalid regions before selecting closest index
        distance_stack = np.stack(distances, axis=-1)
        distance_stack = np.where(distance_stack == 0, np.inf, distance_stack)        

        # Get the closest image index per pixel
        closest_indices = np.argmin(distance_stack, axis=-1)

        # Blend images by selecting the closest valid pixels
        for i, img in enumerate(images):
            selected = (closest_indices == i) & valid_masks[i]  # Ensure only valid pixels are used


            if selected.any():  # Skip empty selections
                blended[selected] += img[selected]
                mask_sums[selected] += 1
 

        # Normalize blended pixels
        valid_pixels = mask_sums > 0
        blended[valid_pixels] /= mask_sums[valid_pixels, None]  # Avoid overwriting with zeros

        return blended