# panorai/pipelines/blender/overlap_counter.py
import numpy as np
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

@BlenderRegistry.register("counter")
class OverlapCounterBlender(BaseBlender):
    """
    Blender that counts the number of overlapping valid pixels.

    For each pixel location, it determines how many images
    contribute a valid (non-zero) value.
    
    The output is a single-channel image (with an extra dimension added)
    that represents the count per pixel.
    """

    def blend(self, images, masks, **kwargs):
        # Validate inputs
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        # Assume all images have the same spatial shape and number of channels.
        img_shape = images[0].shape  # e.g. (H, W, C)
        count_map = np.zeros(img_shape[:2], dtype=np.float32)

        # For each image, we assume a pixel is valid if any channel is > 0.
        for img in images:
            valid_mask = (np.max(img, axis=-1) > 0).astype(np.float32)
            count_map += valid_mask

        # Expand dims to create a 3D array (H, W, 1)
        return count_map[..., None]