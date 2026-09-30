import numpy as np
from .base_blenders import BaseBlender
from ._inputs import expand_mask, finish_blend, masked_values, prepare_blend_inputs
from .registry import BlenderRegistry

@BlenderRegistry.register("average")
class AverageBlender(BaseBlender):
    def blend(self, images, masks, return_mask=False, **kwargs):
        """
        Blend images using the explicit validity masks.

        A zero-valued or black pixel contributes whenever its mask is true.
        Set ``return_mask=True`` to also receive the union support mask.
        """

        images, masks = prepare_blend_inputs(images, masks)

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        for img, mask in zip(images, masks):
            combined += masked_values(img, mask)
            weight_map += mask

        valid_weights = weight_map > 0
        denominator = expand_mask(weight_map, combined)
        np.divide(combined, denominator, out=combined, where=denominator > 0)
        combined[~valid_weights] = 0

        return finish_blend(combined, masks, return_mask)
