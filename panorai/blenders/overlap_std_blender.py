import numpy as np
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
from .registry import BlenderRegistry

@BlenderRegistry.register("std")
class OverlapStdBlender(BaseBlender):
    """
    Blender que calcula o desvio padrão dos pixels sobrepostos.

    - Evita expansão incorreta dos canais.
    - Mantém `masks` no formato correto (B, H, W, C).
    - Usa `np.where()` corretamente sem gerar dimensões extras.
    """

    def blend(self, images, masks, return_mask=False, **kwargs):
        images, masks = prepare_blend_inputs(images, masks)
        stacked = np.stack(images, axis=0).astype(np.float64, copy=False)
        stacked_masks = np.stack(masks, axis=0)
        valid = (
            np.broadcast_to(stacked_masks[..., None], stacked.shape)
            if stacked.ndim == 4
            else stacked_masks
        )
        masked_images = np.where(valid, stacked, np.nan)
        support_mask = np.any(stacked_masks, axis=0)
        if stacked.ndim == 4:
            masked_images[:, ~support_mask, :] = 0.0
        else:
            masked_images[:, ~support_mask] = 0.0

        # Calculando desvio padrão ao longo do batch (B)
        std_map = np.nanstd(masked_images, axis=0)  # (H, W, C)

        # Se for multi-canal, reduz para um único canal
        if std_map.ndim == 3:
            std_map = np.mean(std_map, axis=-1, keepdims=True)  # (H, W, 1)
        else:
            std_map = std_map[..., None]

        std_map[~support_mask] = 0
        return finish_blend(std_map, masks, return_mask)
