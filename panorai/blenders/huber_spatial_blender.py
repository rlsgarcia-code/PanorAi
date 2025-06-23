import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("huber_spatial")
class HuberSpatialBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, sigma=1.0, **kwargs):
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

        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        stacked = np.stack(images)  # (B, H, W) or (B, H, W, 3)
        masks = np.stack(masks)     # (B, H, W)

        # Detect input shape
        B, H, W = stacked.shape[:3]  # Always extract first three dims
        is_multi_channel = stacked.ndim == 4  # True if input is (B, H, W, 3)

        if not is_multi_channel:
            stacked = stacked[..., None]  # Convert to (B, H, W, 1)
            logger.debug("Detected single-channel input, converting to 4D for processing.")

        # Mask invalid values (convert to NaN)
        stacked[~masks.astype(bool)] = np.nan

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
        weight_total = np.nansum(huber_weights, axis=0)  # (H, W, C)

        # Compute final fused radius map
        combined_radius = weighted_sum / (weight_total + 1e-6)  # Avoid div by zero

        # **Apply Spatial Smoothing for Consistency**
        for c in range(combined_radius.shape[-1]):
            combined_radius[..., c] = gaussian_filter(combined_radius[..., c], sigma=sigma)

        # Ensure output is always 3-channel (H, W, 3)
        if combined_radius.shape[-1] == 1:
            combined_radius = np.repeat(combined_radius, 3, axis=-1)  # Convert (H, W, 1) → (H, W, 3)

        return combined_radius  # (H, W, 3)
