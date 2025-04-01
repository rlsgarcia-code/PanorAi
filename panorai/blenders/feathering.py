import numpy as np
from scipy.ndimage import distance_transform_edt
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

@BlenderRegistry.register("feathering")
class FeatheringBlender(BaseBlender):
    """Blends images using a simple feathering approach."""

    def blend(self, images, masks, **kwargs):
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        for img, mask in zip(images, masks):
            valid_mask = np.max(img > 0, axis=-1).astype(np.float32)
            distance = distance_transform_edt(valid_mask)
            if distance.max() > 0:
                feathered_mask = distance / distance.max()
            else:
                feathered_mask = distance
            combined += img * feathered_mask[..., None]
            weight_map += feathered_mask

        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]
        combined[~valid_weights] = 0
        return combined