import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("huber_spatial")
class HuberSpatialBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, sigma=1.0, return_mask=False, **kwargs):
        """
        Huber blending with **spatial consistency** enforced via **Gaussian smoothing**.

        Instead of processing pixels independently, this method **regularizes** the result
        by applying a **Gaussian filter** to ensure smooth transitions between nearby pixels.

        Parameters:
        - images: List of (H, W) or (H, W, 3) arrays representing backprojected radius values.
        - masks: List of (H, W) binary masks indicating valid pixels per image.
        - delta: Huber threshold. Residuals <= delta behave like L2 loss; larger residuals like L1.
        - sigma: Gaussian kernel standard deviation (higher = more smoothing).

        Returns:
        - combined_radius: (H, W, 3) array representing the fused radius map with smoothness.
        """
        logger.info('Starting spatially consistent Huber blending...')

        if delta <= 0 or sigma < 0:
            raise ValueError("delta must be positive and sigma must be non-negative.")
        images, masks = prepare_blend_inputs(images, masks)

        stacked = np.stack(images).astype(np.float64, copy=False)
        stacked_masks = np.stack(masks)

        # Detect input shape
        B, H, W = stacked.shape[:3]  # Always extract first three dims
        is_multi_channel = stacked.ndim == 4  # True if input is (B, H, W, 3)

        if not is_multi_channel:
            stacked = stacked[..., None]  # Convert to (B, H, W, 1)
            logger.debug("Detected single-channel input, converting to 4D for processing.")

        # Mask invalid values (convert to NaN)
        valid = np.broadcast_to(stacked_masks[..., None], stacked.shape)
        stacked = np.where(valid, stacked, np.nan)
        support_mask = np.any(stacked_masks, axis=0)
        stacked[:, ~support_mask, :] = 0.0

        # **Fast Median Approximation (Percentile)**
        median_radii = np.nanpercentile(stacked, 50, axis=0)  # (H, W, C)

        # **Compute Residuals & Huber Weights In-Place**
        residuals = np.abs(stacked - median_radii[None, :, :, :])  # (B, H, W, C)

        # Compute weights in-place
        huber_weights = np.ones_like(residuals)
        large_residuals = residuals > delta
        huber_weights[large_residuals] = delta / (residuals[large_residuals] + 1e-6)

        # **Weighted Sum (Avoiding Huge Temporary Arrays)**
        weighted_sum = np.nansum(stacked * huber_weights, axis=0)  # (H, W, C)
        huber_weights = np.where(valid, huber_weights, 0.0)
        weight_total = np.sum(huber_weights, axis=0)

        # Compute final fused radius map
        combined_radius = weighted_sum / (weight_total + 1e-6)  # Avoid div by zero

        # Apply normalized smoothing without letting unsupported zero values
        # contaminate supported pixels.
        for c in range(combined_radius.shape[-1]):
            numerator = gaussian_filter(
                np.where(support_mask, combined_radius[..., c], 0.0), sigma=sigma
            )
            denominator = gaussian_filter(support_mask.astype(np.float64), sigma=sigma)
            np.divide(
                numerator,
                denominator,
                out=combined_radius[..., c],
                where=denominator > 1e-12,
            )
            combined_radius[..., c][~support_mask] = 0

        if not is_multi_channel:
            combined_radius = combined_radius[..., 0]

        return finish_blend(combined_radius, masks, return_mask)
