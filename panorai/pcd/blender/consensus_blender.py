"""
consensus_blender.py
====================

Implements an `EquirectangularConsensusBlender` that merges 
multiple gnomonic faces in equirectangular space using 
consensus (weighted averaging) plus optional smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from ..data import PCD
from ..handler import PCDHandler
from panorai.data.gnomonic_image import GnomonicFace
from .base_blender import BaseBlender

class EquirectangularConsensusBlender(BaseBlender):
    """
    2nd-level strategy that:
    - Projects each face's radius & color into a shared equirectangular grid.
    - Applies Gaussian filtering for smoothness.
    - Combines them with a weighted average (consensus).
    """

    def __init__(self, eq_shape=(512, 1024), min_radius=0.0, max_radius=20.0, smooth_sigma=1.0):
        super().__init__(min_radius, max_radius)
        self.eq_shape = eq_shape
        self.smooth_sigma = smooth_sigma

    def process_faceset(self, faceset, model, grad_threshold, feather_exp):
        radius_stack, color_stack, weight_stack = [], [], []

        for face in faceset:
            face_img = np.array(face)
            depth = model(face_img).astype(np.float32)
            Hf, Wf = depth.shape

            # local radius
            u, v = np.meshgrid(np.linspace(-1, 1, Wf),
                               np.linspace(-1, 1, Hf),
                               indexing='xy')
            R = np.sqrt((depth*u)**2 + (depth*v)**2 + depth**2)

            # gradient-based weighting
            grad_mask = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold).astype(np.float32)
            wmask = np.clip((1.0 - np.sqrt(u**2 + v**2)) ** feather_exp, 0, 1) * grad_mask

            radial_face = GnomonicFace(np.repeat(R[..., None], 3, axis=-1),
                                       face.lat, face.lon, face.fov)
            weight_face = GnomonicFace(np.repeat(wmask[..., None], 3, axis=-1),
                                       face.lat, face.lon, face.fov)
            color_face  = GnomonicFace(face_img.astype(np.float32), 
                                       face.lat, face.lon, face.fov)

            eq_R = np.array(radial_face.to_equirectangular(self.eq_shape))[..., 0]
            eq_W = np.array(weight_face.to_equirectangular(self.eq_shape))[..., 0]
            eq_C = np.array(color_face.to_equirectangular(self.eq_shape)) / 255.

            radius_stack.append(eq_R)
            color_stack.append(eq_C)
            weight_stack.append(eq_W)

        # smooth
        radius_stack = gaussian_filter(np.stack(radius_stack), sigma=(0, self.smooth_sigma, self.smooth_sigma))
        weight_stack = gaussian_filter(np.stack(weight_stack), sigma=(0, self.smooth_sigma, self.smooth_sigma))
        color_stack  = gaussian_filter(np.stack(color_stack), sigma=(0, self.smooth_sigma, self.smooth_sigma, 0))

        weighted_sum_radius = np.sum(radius_stack * weight_stack, axis=0)
        total_weights = np.sum(weight_stack, axis=0)
        blended_radius = np.where(total_weights > 1e-6, weighted_sum_radius / total_weights, 0)

        weighted_sum_color = np.sum(color_stack * weight_stack[..., None], axis=0)
        blended_color = np.where(
            total_weights[..., None] > 1e-6,
            weighted_sum_color / (total_weights[..., None] + 1e-6),
            0
        )

        return self._to_pcd(blended_radius, blended_color)