import numpy as np
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

@BlenderRegistry.register("std")
class OverlapStdBlender(BaseBlender):
    """
    Blender que calcula o desvio padrão dos pixels sobrepostos.

    - Evita expansão incorreta dos canais.
    - Mantém `masks` no formato correto (B, H, W, C).
    - Usa `np.where()` corretamente sem gerar dimensões extras.
    """

    def blend(self, images, masks, **kwargs):
        # Verificando os shapes antes de modificar
        images = np.stack(images, axis=0)  # Garante que seja (B, H, W, C)
        masks = np.stack(masks, axis=0).astype(bool)  # Garante que seja (B, H, W, C)

        B, H, W, C = images.shape  # Pegando dimensões corretas

        # Garante que as máscaras **NÃO** sejam modificadas incorretamente
        assert masks.shape == images.shape, "Erro: Masks e Images precisam ter o mesmo shape!"

        # Criando a máscara de pixels válidos corretamente
        valid_pixels = np.any(images > 0, axis=-1, keepdims=True)  # (B, H, W, 1)
        valid_mask = masks & np.broadcast_to(valid_pixels, images.shape)  # Agora garantimos que ambos são (B, H, W, C)

        # Aplicando a máscara corretamente
        masked_images = np.where(valid_mask, images, np.nan)

        # Calculando desvio padrão ao longo do batch (B)
        std_map = np.nanstd(masked_images, axis=0)  # (H, W, C)

        # Se for multi-canal, reduz para um único canal
        if std_map.ndim == 3:
            std_map = np.mean(std_map, axis=-1, keepdims=True)  # (H, W, 1)

        return std_map  # Retorna shape correto (H, W, 1)