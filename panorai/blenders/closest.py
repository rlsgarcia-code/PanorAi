import numpy as np
from scipy.ndimage import distance_transform_edt
from .registry import BlenderRegistry
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs

@BlenderRegistry.register("closest")
class ClosestBlender(BaseBlender):
    def blend(self, images, masks, return_mask=False, **kwargs):
        """
        Blends images by selecting the value from whichever image is 
        closest to the center of a valid region.
        """
        images, masks = prepare_blend_inputs(images, masks)

        img_shape = images[0].shape
        blended = np.zeros(img_shape, dtype=np.float32)

        distances = []

        for mask in masks:
            distances.append(distance_transform_edt(mask))

        distance_stack = np.stack(distances, axis=-1)
        valid_stack = np.stack(masks, axis=-1)
        distance_stack = np.where(valid_stack, distance_stack, -np.inf)
        closest_indices = np.argmax(distance_stack, axis=-1)
        support_mask = np.any(valid_stack, axis=-1)

        for i, img in enumerate(images):
            selected = (closest_indices == i) & masks[i] & support_mask
            if selected.any():
                blended[selected] = img[selected]

        return finish_blend(blended, masks, return_mask)
