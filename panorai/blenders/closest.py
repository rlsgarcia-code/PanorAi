import numpy as np
from scipy.ndimage import distance_transform_edt
from .registry import BlenderRegistry
from .base_blenders import BaseBlender

@BlenderRegistry.register("closest")
class ClosestBlender(BaseBlender):
    def blend(self, images, masks, **kwargs):
        """
        Blends images by selecting the value from whichever image is 
        closest to the center of a valid region.
        """
        if not images or len(images) == 0:
            raise ValueError("Images must be a non-empty list.")

        img_shape = images[0].shape
        blended = np.zeros(img_shape, dtype=np.float32)
        mask_sums = np.zeros(img_shape[:2], dtype=np.float32)

        distances = []
        valid_masks = []

        for img in images:
            valid_mask = np.max(img > 0, axis=-1).astype(bool)
            valid_masks.append(valid_mask)
            distance = distance_transform_edt(valid_mask.astype(np.float32))
            distances.append(distance)

        distance_stack = np.stack(distances, axis=-1)
        distance_stack = np.where(distance_stack == 0, np.inf, distance_stack)
        closest_indices = np.argmin(distance_stack, axis=-1)

        for i, img in enumerate(images):
            selected = (closest_indices == i) & valid_masks[i]
            if selected.any():
                blended[selected] += img[selected]
                mask_sums[selected] += 1

        valid_pixels = mask_sums > 0
        blended[valid_pixels] /= mask_sums[valid_pixels, None]
        return blended