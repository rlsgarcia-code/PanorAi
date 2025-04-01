import numpy as np
import scipy.ndimage as ndi
from .base_blenders import BaseBlender
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

    def blend(self, images, masks, **kwargs):
        # Stack as NumPy arrays
        images = np.stack(images, axis=0)  # Ensure (B, H, W, C)
        masks = np.stack(masks, axis=0).astype(bool)  # Ensure (B, H, W, C)

        B, H, W, C = images.shape  # Get batch, height, width, channels

        # Ensure masks match images shape
        assert masks.shape == images.shape, "Erro: Masks e Images precisam ter o mesmo shape!"

        # Compute distance transform (feathering weights)
        feathering_weights = np.zeros_like(images, dtype=np.float32)  # (B, H, W, C)
        for i in range(B):
            for ch in range(C):
                feathering_weights[i, ..., ch] = ndi.distance_transform_edt(masks[i, ..., ch])

        # Normalize weights per image
        feathering_weights += 1e-6  # Avoid division by zero
        feathering_weights /= np.max(feathering_weights, axis=(1, 2), keepdims=True)  # Normalize to [0,1]

        # Compute valid pixels
        valid_pixels = np.any(images > 0, axis=-1, keepdims=True)  # (B, H, W, 1)
        valid_mask = masks & np.broadcast_to(valid_pixels, images.shape)  # Ensure (B, H, W, C)

        # Apply feathering weights; invalid pixels become NaN
        weighted_images = np.where(valid_mask, images * feathering_weights, np.nan)

        # Compute standard deviation across batch (B-axis)
        std_map = np.nanstd(weighted_images, axis=0)  # (H, W, C)

        # Reduce multi-channel std to a single-channel output
        if std_map.ndim == 3:
            std_map = np.mean(std_map, axis=-1, keepdims=True)  # (H, W, 1)

        return std_map  # Output shape: (H, W, 1)