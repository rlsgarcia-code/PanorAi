import logging
import numpy as np
from scipy.optimize import minimize
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("huber")
class HuberBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, return_mask=False, **kwargs):
        """
        Blends a stack of radius images using robust estimation with Huber loss.

        This method optimizes for the best radius at each pixel using a **vectorized approach**
        instead of looping over individual pixels.

        Parameters:
        - images: List of equal-shaped (H, W) or (H, W, C) arrays.
        - masks: List of (H, W) masks indicating valid pixels in each image.
        - delta: Huber threshold. For residuals <= delta, behaves like L2 loss; otherwise like L1.

        Returns:
        - combined: Array with the same shape as one input image.
        """
        logger.info('Starting Huber blending...')

        if delta <= 0:
            raise ValueError("delta must be positive.")
        images, masks = prepare_blend_inputs(images, masks)

        # Stack images and masks
        stacked = np.stack(images)
        stacked_masks = np.stack(masks)
        B = stacked.shape[0]
        output_shape = stacked.shape[1:]

        # Flatten spatial/channel values while broadcasting spatial validity.
        if stacked.ndim == 4:
            expanded_masks = np.broadcast_to(stacked_masks[..., None], stacked.shape)
        else:
            expanded_masks = stacked_masks
        stacked_flat = stacked.reshape(B, -1)
        masks_flat = expanded_masks.reshape(B, -1)

        # Only optimize valid pixels (where at least one mask is nonzero)
        valid_pixels = np.any(masks_flat, axis=0)  # Shape: (H*W,)
        valid_indices = np.where(valid_pixels)[0]  # Indices of valid pixels

        def huber_loss(r, points, delta):
            """Huber loss function for vectorized radius optimization."""
            residuals = r - points  # Shape: (N,)
            abs_residuals = np.abs(residuals)
            quadratic = np.minimum(abs_residuals, delta)
            linear = abs_residuals - quadratic
            return np.sum(0.5 * quadratic**2 + delta * linear)

        optimized = np.zeros(stacked_flat.shape[1], dtype=np.float32)
        for idx in valid_indices:
            observations = stacked_flat[:, idx][masks_flat[:, idx]]
            initial = float(np.median(observations))
            res = minimize(
                huber_loss,
                np.array([initial]),
                args=(observations, delta),
                method='L-BFGS-B',
            )
            optimized[idx] = float(res.x[0]) if res.success else initial

        combined = optimized.reshape(output_shape)
        return finish_blend(combined, masks, return_mask)
