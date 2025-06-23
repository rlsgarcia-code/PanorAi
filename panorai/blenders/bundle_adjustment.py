import logging
import numpy as np
from scipy.optimize import least_squares
from .base_blenders import BaseBlender
from .registry import BlenderRegistry

logger = logging.getLogger(__name__)

@BlenderRegistry.register("bundle_adjustment")
class BundleAdjustmentBlender(BaseBlender):
    def blend(self, images, masks, delta=1.0, **kwargs):
        """
        Performs bundle adjustment on overlapping equirectangular depth images.

        - Computes per-pixel spherical directions automatically from latitude/longitude.
        - Optimizes per-image **scaling factors** to correct depth inconsistencies.
        - Optimizes per-pixel **local adjustments** to refine depth.
        - Enforces **overlapping region consistency** (same lat/lon pixels should have the same depth).

        Parameters:
        - images: List of (H, W) radius maps for each image.
        - masks: List of (H, W) binary masks indicating valid pixels in each image.
        - delta: Huber threshold for residual weighting.

        Returns:
        - refined_radius: (H, W, 3) fused depth map with optimized consistency.
        """
        logger.info('Starting Bundle Adjustment...')

        B = len(images)
        H, W , _= images[0].shape

        # Create latitude/longitude meshgrid (equirectangular projection)
        u, v = np.meshgrid(np.linspace(-180, 180, W), np.linspace(90, -90, H))

        def to_xyz(lat, lon, R=1):
            """Convert lat/lon to Cartesian coordinates on a sphere with radius R."""
            lat = np.radians(lat)
            lon = np.radians(lon)
            x = R * np.cos(lat) * np.cos(lon)
            y = R * np.cos(lat) * np.sin(lon)
            z = R * np.sin(lat)
            return x, y, z

        # Stack inputs
        stacked_radii = np.stack(images)  # Shape: (B, H, W)
        stacked_masks = np.stack(masks)   # Shape: (B, H, W)

        # Convert radius maps to (X, Y, Z) coordinates
        X, Y, Z = to_xyz(lat=v, lon=u, R=stacked_radii)  # Shape: (B, H, W)

        # Create a per-image scaling factor (initially set to 1.0)
        scale_factors = np.ones(B)

        # Flatten valid pixels for optimization
        valid_pixels = stacked_masks.astype(bool)
        valid_indices = np.argwhere(valid_pixels)

        # Create a per-pixel adjustment term (one for each valid pixel)
        per_pixel_adjustments = np.zeros_like(stacked_radii)

        # **Define the optimization function**
        def loss_function(params):
            """Computes the consistency loss across overlapping regions."""
            s_b = params[:B]  # Extract per-image scale factors
            r_p = params[B:].reshape(stacked_radii.shape)  # Extract per-pixel adjustments

            residuals = []

            # Enforce consistency in overlapping regions
            for b1 in range(B):
                for b2 in range(b1 + 1, B):  # Only check unique pairs
                    overlap_mask = valid_pixels[b1] & valid_pixels[b2]  # Find overlap
                    if np.any(overlap_mask):
                        # Compute projected depth consistency error
                        R1 = s_b[b1] * stacked_radii[b1] + r_p[b1]
                        R2 = s_b[b2] * stacked_radii[b2] + r_p[b2]

                        # Compute residuals in overlap regions
                        residuals.append((R1 - R2)[overlap_mask])
            
            return np.concatenate(residuals) if residuals else np.zeros(B)

        # **Optimize using Trust Region Reflective least squares**
        initial_params = np.concatenate([scale_factors, per_pixel_adjustments.flatten()])
        result = least_squares(loss_function, initial_params, method='trf')

        # Extract optimized parameters
        optimized_scales = result.x[:B]
        optimized_adjustments = result.x[B:].reshape(stacked_radii.shape)

        # Compute final refined depth
        refined_radius = optimized_scales[:, None, None] * stacked_radii + optimized_adjustments

        # Ensure output is 3-channel
        return np.repeat(np.mean(refined_radius, axis=0, keepdims=True), 3, axis=-1)
