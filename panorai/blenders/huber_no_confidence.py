import logging
import numpy as np
from .base_blenders import BaseBlender
from ._inputs import finish_blend, prepare_blend_inputs
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("huber_no_confidence")
class HuberNoConfidenceBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, return_mask=False, **kwargs):
        """
        Blends a stack of radius images using robust estimation with Huber loss.
        This version **does not require confidence maps** and instead estimates confidence
        implicitly using local variance and Huber weighting.

        It automatically detects whether input has **1 or 3 channels** and processes accordingly.

        Parameters:
        - images: List of (H, W) or (H, W, 3) arrays representing backprojected radius values.
        - masks: List of (H, W) binary masks indicating valid pixels per image.
        - delta: Huber threshold. Residuals <= delta behave like L2 loss; larger residuals like L1.

        Returns:
        - combined_radius: (H, W, 3) array representing the fused radius map using robust estimation.
        """
        logger.info('Starting Huber blending (no explicit confidence maps)...')

        if delta <= 0:
            raise ValueError("delta must be positive.")
        images, masks = prepare_blend_inputs(images, masks)

        # Stack inputs into (B, H, W) or (B, H, W, 3)
        stacked = np.stack(images).astype(np.float64, copy=False)
        stacked_masks = np.stack(masks)

        # Detect input shape
        B, H, W = stacked.shape[:3]  # Always extract first three dims
        is_multi_channel = stacked.ndim == 4  # True if input is (B, H, W, 3)

        # Handle single-channel case by expanding dimensions
        if not is_multi_channel:
            stacked = stacked[..., None]  # Convert to (B, H, W, 1)
            logger.debug("Detected single-channel input, converting to 4D for processing.")

        # Mask invalid values
        valid = np.broadcast_to(stacked_masks[..., None], stacked.shape)
        stacked = np.where(valid, stacked, np.nan)
        support_mask = np.any(stacked_masks, axis=0)
        stacked[:, ~support_mask, :] = 0.0

        # Compute per-pixel median and mean
        median_radii = np.nanmedian(stacked, axis=0)  # (H, W, C)
        mean_radii = np.nanmean(stacked, axis=0)      # (H, W, C)

        # Compute per-pixel variance
        var_radii = np.nanvar(stacked, axis=0)  # (H, W, C)

        # Compute residuals
        residuals_median = np.abs(stacked - median_radii[None, :, :, :])  # (B, H, W, C)
        residuals_mean = np.abs(stacked - mean_radii[None, :, :, :])      # (B, H, W, C)

        # Apply Huber weighting
        is_small_median = residuals_median <= delta
        is_small_mean = residuals_mean <= delta

        huber_weights_median = np.where(is_small_median, 1, delta / (residuals_median + 1e-6))
        huber_weights_mean = np.where(is_small_mean, 1, delta / (residuals_mean + 1e-6))

        # Estimate implicit confidence → Lower variance = Higher reliability
        implicit_confidence = np.exp(-var_radii)  # (H, W, C)

        # Combine Huber & implicit confidence weighting
        final_weights_median = np.where(
            valid, huber_weights_median * implicit_confidence[None, :, :, :], 0.0
        )
        final_weights_mean = np.where(
            valid, huber_weights_mean * implicit_confidence[None, :, :, :], 0.0
        )

        # Compute weighted sums
        weighted_sum_median = np.nansum(stacked * final_weights_median, axis=0)
        weighted_sum_mean = np.nansum(stacked * final_weights_mean, axis=0)

        weight_total_median = np.nansum(final_weights_median, axis=0)
        weight_total_mean = np.nansum(final_weights_mean, axis=0)

        # Compute final fused radius map (blend of median & mean)
        combined_radius = (
            (weighted_sum_median / (weight_total_median + 1e-6)) * 0.5 +
            (weighted_sum_mean / (weight_total_mean + 1e-6)) * 0.5
        )

        combined_radius[~support_mask] = 0
        if not is_multi_channel:
            combined_radius = combined_radius[..., 0]

        return finish_blend(combined_radius, masks, return_mask)
