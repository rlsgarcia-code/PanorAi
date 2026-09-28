import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_blenders import BaseBlender
from ._inputs import expand_mask, finish_blend, masked_values, prepare_blend_inputs
from .registry import BlenderRegistry

@BlenderRegistry.register("feathering")
class FeatheringBlender(BaseBlender):
    """Blends images using a simple feathering approach."""

    def blend(self, images, masks, return_mask=False, **kwargs):
        images, masks = prepare_blend_inputs(images, masks)

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        for img, mask in zip(images, masks):
            distance = distance_transform_edt(mask)
            if distance.max() > 0:
                feathered_mask = distance / distance.max()
            else:
                feathered_mask = distance
            combined += masked_values(img, mask) * expand_mask(feathered_mask, img)
            weight_map += feathered_mask

        valid_weights = weight_map > 0
        denominator = expand_mask(weight_map, combined)
        np.divide(combined, denominator, out=combined, where=denominator > 0)
        combined[~valid_weights] = 0
        return finish_blend(combined, masks, return_mask)
