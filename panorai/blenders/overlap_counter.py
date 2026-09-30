# panorai/pipelines/blender/overlap_counter.py
import numpy as np
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
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

    def blend(self, images, masks, return_mask=False, **kwargs):
        images, masks = prepare_blend_inputs(images, masks)

        # Assume all images have the same spatial shape and number of channels.
        img_shape = images[0].shape  # e.g. (H, W, C)
        count_map = np.zeros(img_shape[:2], dtype=np.float32)

        for mask in masks:
            count_map += mask

        # Expand dims to create a 3D array (H, W, 1)
        return finish_blend(count_map[..., None], masks, return_mask)
