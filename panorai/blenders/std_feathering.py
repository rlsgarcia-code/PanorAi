import numpy as np
import scipy.ndimage as ndi
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
from .registry import BlenderRegistry

@BlenderRegistry.register("std_feathered")
class OverlapStdFeatheredBlender(BaseBlender):
    """
    Blender que calcula o desvio padrão dos pixels sobrepostos,
    utilizando uma estratégia de feathering para suavizar transições.

    - Usa `ndi.distance_transform_edt()` para calcular distâncias dentro da máscara.
    - Garante que `masks` tenha o mesmo shape de `images` antes da aplicação.
    - Aplica pesos suaves para evitar transições abruptas entre imagens sobrepostas.
    """

    def blend(self, images, masks, return_mask=False, **kwargs):
        images, masks = prepare_blend_inputs(images, masks)
        stacked = np.stack(images, axis=0).astype(np.float64, copy=False)
        stacked_masks = np.stack(masks, axis=0)
        if stacked.ndim == 3:
            stacked = stacked[..., None]
        B, H, W, C = stacked.shape

        # Compute distance transform (feathering weights)
        feathering_weights = np.zeros_like(stacked, dtype=np.float32)
        for i in range(B):
            distance = ndi.distance_transform_edt(stacked_masks[i])
            feathering_weights[i] = distance[..., None]

        # Normalize weights per image
        feathering_weights += 1e-6  # Avoid division by zero
        feathering_weights /= np.max(feathering_weights, axis=(1, 2), keepdims=True)  # Normalize to [0,1]

        valid_mask = np.broadcast_to(stacked_masks[..., None], stacked.shape)

        # Apply feathering weights; invalid pixels become NaN
        weighted_images = np.where(valid_mask, stacked * feathering_weights, np.nan)
        support_mask = np.any(stacked_masks, axis=0)
        weighted_images[:, ~support_mask, :] = 0.0

        # Compute standard deviation across batch (B-axis)
        std_map = np.nanstd(weighted_images, axis=0)  # (H, W, C)

        # Reduce multi-channel std to a single-channel output
        if std_map.ndim == 3:
            std_map = np.mean(std_map, axis=-1, keepdims=True)  # (H, W, 1)

        std_map[~support_mask] = 0
        return finish_blend(std_map, masks, return_mask)
