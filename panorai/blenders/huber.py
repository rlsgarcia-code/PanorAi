import logging
import numpy as np
from scipy.optimize import minimize
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("huber")
class HuberBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, **kwargs):
        """
        Blends a stack of radius images using robust estimation with Huber loss.

        This method optimizes for the best radius at each pixel using a **vectorized approach**
        instead of looping over individual pixels.

        Parameters:
        - images: List of (H, W) arrays representing backprojected radius values (per view).
        - masks: List of (H, W) masks indicating valid pixels in each image.
        - delta: Huber threshold. For residuals <= delta, behaves like L2 loss; otherwise like L1.

        Returns:
        - combined_radius: (H, W) array representing fused radius map using robust estimation.
        """
        logger.info('Starting Huber blending...')

        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        # Stack images and masks
        stacked = np.stack(images)   # (B, H, W)
        masks = np.stack(masks)      # (B, H, W)
        B, H, W, _ = stacked.shape

        # Flatten to (H*W, B) for batch processing
        stacked_flat = stacked.reshape(B, -1)  # (B, H*W)
        masks_flat = masks.reshape(B, -1)      # (B, H*W)

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

        # Prepare initial estimates (use median for robustness)
        initial_guesses = np.median(stacked_flat[:, valid_indices], axis=0)  # Shape: (N_valid,)

        # Optimize using vectorized scipy.minimize
        optimized_radii = np.zeros(H * W, dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            observations = stacked_flat[:, idx][masks_flat[:, idx] > 0]  # Get valid radius values
            res = minimize(huber_loss, initial_guesses[i], args=(observations, delta), method='L-BFGS-B')
            optimized_radii[idx] = res.x if res.success else initial_guesses[i]

        # Reshape back to (H, W)
        combined_radius = optimized_radii.reshape(H, W)

        # Return as a 3-channel image for consistency
        return np.repeat(combined_radius[..., None], 3, axis=-1)  # Shape: (H, W, 3)
