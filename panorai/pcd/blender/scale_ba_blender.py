"""
scale_ba_blender.py
===================

Implements ScaleBundleAdjustmentBlender, the 5th-level approach 
that attempts to refine per-face scaling for maximum consistency.
"""

import numpy as np
import logging
from scipy.optimize import minimize
from ..data import PCD
from ..handler import PCDHandler
from panorai.data.gnomonic_image import GnomonicFace
from .base_blender import BaseBlender

logger = logging.getLogger(__name__)

def huber_loss(x, delta=1.0):
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5*delta))

def estimate_initial_scales_from_overlap(radius_stack, valid_stack, ref_idx=None):
    """
    Quick routine that measures pairwise overlaps across faces, 
    computing a rough scale guess from their median ratio.
    """
    F, H, W = radius_stack.shape
    N = H*W
    R = radius_stack.reshape(F, N)
    V = valid_stack.reshape(F, N)
    scale_matrix = np.ones((F, F))
    overlap_count = np.zeros((F, F))

    for i in range(F):
        for j in range(i+1, F):
            mask = V[i] & V[j]
            if np.sum(mask) < 10:
                continue
            ratios = R[i, mask] / (R[j, mask] + 1e-6)
            median_ij = np.median(ratios)
            scale_matrix[i, j] = median_ij
            scale_matrix[j, i] = 1.0 / median_ij
            overlap_count[i, j] = overlap_count[j, i] = np.sum(mask)

    mean_errors = np.mean(np.abs(np.log(scale_matrix + 1e-6)), axis=1)
    if ref_idx is None:
        ref_idx = np.argmin(mean_errors)

    initial_scales = scale_matrix[ref_idx]
    initial_scales[ref_idx] = 1.0
    initial_scales = np.clip(initial_scales, 0.5, 2.0)

    return initial_scales, overlap_count, ref_idx

class ScaleBundleAdjustmentBlender(BaseBlender):
    """
    5th-level strategy:
    - Projects each face to equirect
    - Minimizes an overlapping cost function to find per-face scale factors
    - Produces a globally consistent radius map
    """

    def __init__(self, eq_shape=(512, 1024), min_radius=0.0, max_radius=20.0,
                 match_threshold=0.01, max_iter=90, huber_delta=1.0, ref_idx=None):
        super().__init__(min_radius, max_radius)
        if len(eq_shape) < 2:
            raise ValueError("eq_shape must have at least two dimensions")
        self.eq_shape = eq_shape[:2]
        self.max_iter = max_iter
        self.huber_delta = huber_delta
        self.ref_idx = ref_idx
        logger.info(f"Initialized ScaleBundleAdjustmentBlender with eq_shape={self.eq_shape}, "
                    f"min_radius={min_radius}, max_radius={max_radius}, max_iter={max_iter}, "
                    f"ref_idx={ref_idx}")

    def process_faceset(self, faceset, model, grad_threshold, feather_exp):
        eq_shape = self.eq_shape
        H_eq, W_eq = eq_shape
        radius_maps, grad_masks, chunks = [], [], []

        # Gather each face's equirect radius
        for face in faceset:
            print(face.shape)
            face_img = np.array(face)
            depth = model(face_img).astype(np.float32)
            print(depth.shape)
            Hf, Wf = depth.shape

            # Local radius
            u, v = np.meshgrid(np.linspace(-1, 1, Wf), np.linspace(-1, 1, Hf), indexing='xy')
            R_local = np.sqrt((depth*u)**2 + (depth*v)**2 + depth**2)

            grad_mask_local = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold).astype(np.float32)
            grad_face = GnomonicFace(np.repeat(grad_mask_local[...,None], 3, axis=-1),
                                     face.lat, face.lon, face.fov)
            eq_grad = np.array(grad_face.to_equirectangular(self.eq_shape))[..., 0]

            grad_masks.append(eq_grad > 0.5)

            radial_face = GnomonicFace(np.repeat(R_local[...,None], 3, axis=-1),
                                       face.lat, face.lon, face.fov)
            eq_R = np.array(radial_face.to_equirectangular(eq_shape))[..., 0]
            radius_maps.append(eq_R)

            # stash raw 3D points
            X = depth*u
            Y = depth*v
            Z = depth
            xyz_ccs = np.stack([Y.ravel(), X.ravel(), Z.ravel()], axis=1)
            xyz_wcs = PCDHandler.rotate_ccs_to_wcs(xyz_ccs, face.lon, 90 - face.lat)
            valid = depth.ravel() > 0

            pts = xyz_wcs[valid]
            print(pts.shape, face_img.reshape(-1,3).shape)
            cols = (face_img.reshape(-1, 3)/255.)[valid]
            chunks.append({"points": pts, "colors": cols})

        R_stack = np.stack(radius_maps, axis=0)
        G_stack = np.stack(grad_masks, axis=0)
        V_stack = ((G_stack) & (R_stack>0) & (R_stack>=self.min_radius) & (R_stack<=self.max_radius)).astype(np.float32)

        # estimate initial scales
        initial_scales, overlap_weights, ref_idx = estimate_initial_scales_from_overlap(
            R_stack, V_stack.astype(bool), self.ref_idx
        )
        logger.info(f"Using face {ref_idx} as reference (scale=1.0)")

        # define cost
        F = R_stack.shape[0]
        N = H_eq * W_eq
        R_flat = R_stack.reshape(F, N)
        V_flat = V_stack.reshape(F, N)

        def cost_function(rel_scales):
            full_scales = initial_scales.copy()
            full_scales[np.arange(F) != ref_idx] = rel_scales
            scaled_R = full_scales[:, None]*R_flat

            loss = 0.0
            count = 0.0
            for i in range(F):
                for j in range(i+1, F):
                    mask = (V_flat[i]*V_flat[j])>0.5
                    if np.sum(mask) < 10:
                        continue
                    log_diff = np.log(scaled_R[i, mask]+1e-6) - np.log(scaled_R[j, mask]+1e-6)
                    pixel_loss = huber_loss(log_diff, delta=self.huber_delta)
                    weight = overlap_weights[i, j]
                    loss += (pixel_loss.sum()) * weight
                    count += np.sum(mask)*weight

            reg_term = np.sum((full_scales-1.0)**2)
            final_loss = (loss/(count+1e-6)) + 0.01*reg_term
            return final_loss

        x0 = initial_scales[np.arange(F)!=ref_idx]
        result = minimize(
            cost_function,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(0,3.0)]*len(x0),
            options={"maxiter": self.max_iter}
        )

        optimized_scales = initial_scales.copy()
        optimized_scales[np.arange(F)!=ref_idx] = result.x
        logger.info(f"Optimized scales: {optimized_scales}")

        # Final blended radius
        scaled_R = optimized_scales[:, None, None]*R_stack
        weighted_sum = np.sum(scaled_R*V_stack, axis=0)
        total_w = np.sum(V_stack, axis=0)
        blended_radius = np.where(total_w>0, weighted_sum/total_w, 0)

        # Also produce merged 3D chunk
        adjusted_points = [chunks[i]["points"]*optimized_scales[i] for i in range(F)]
        merged_points = np.concatenate(adjusted_points, axis=0)
        merged_colors = np.concatenate([chunks[i]["colors"] for i in range(F)], axis=0)

        # filter final again for min/max
        final_radii = np.linalg.norm(merged_points, axis=1)
        final_mask = (final_radii>=self.min_radius) & (final_radii<=self.max_radius)

        return PCD(merged_points[final_mask], merged_colors[final_mask], radius_image=blended_radius)