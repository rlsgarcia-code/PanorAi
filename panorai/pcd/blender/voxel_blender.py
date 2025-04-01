"""
voxel_blender.py
================

Implements a VoxelBlender that aggregates points/colors in 
3D voxel cells, returning a weighted centroid per voxel.
"""

import numpy as np
from panorai.pcd.data import PCD
from panorai.pcd.handler import PCDHandler
from panorai.data.gnomonic_image import GnomonicFace
from .base_blender import BaseBlender

class VoxelBlender(BaseBlender):
    """
    4th-level strategy: 
    - Converts each face to 3D points
    - Buckets them into voxel cells
    - Computes a weighted average for each cell
    """

    def __init__(self, voxel_size=0.005, min_radius=0.0, max_radius=5.0, **kwargs):
        super().__init__(min_radius, max_radius)
        self.voxel_size = voxel_size

    def process_faceset(self, faceset, model, grad_threshold, feather_exp):
        """
        Gathers all face points, merges them into a single Nx7 array, 
        then does voxel-based blending.

        Returns:
            PCD
        """
        merged_chunks = []

        for face in faceset:
            face_img = np.array(face)
            depth = model(face_img).astype(np.float32)
            Hf, Wf = depth.shape

            # gradient + local radius
            grad_mask = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold)
            u, v = np.meshgrid(np.linspace(-1, 1, Wf),
                               np.linspace(-1, 1, Hf),
                               indexing='xy')
            X = depth * u
            Y = depth * v
            Z = depth

            xyz_ccs = np.stack([Y.ravel(), X.ravel(), Z.ravel()], axis=1)
            xyz_wcs = PCDHandler.rotate_ccs_to_wcs(xyz_ccs, face.lon, 90 - face.lat)

            color_flat = face_img.reshape(-1, 3) / 255.0
            wmask = self._radial_weight_mask(Hf, Wf, feather_exp).ravel()
            valid_mask = (depth.ravel() > 0) & grad_mask.ravel()

            pts = xyz_wcs[valid_mask]
            cols = color_flat[valid_mask]
            wts = wmask[valid_mask]

            chunk = np.concatenate([pts, wts[:, None], cols], axis=1)  # (N, 7)
            merged_chunks.append(chunk)

        if not merged_chunks:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        merged_array = np.concatenate(merged_chunks, axis=0)
        return self._blend(merged_array)

    def _blend(self, merged_array):
        # filter out-of-bounds
        pts = merged_array[:, :3]
        w = merged_array[:, 3]
        cols = merged_array[:, 4:]

        radii = np.linalg.norm(pts, axis=1)
        in_bounds = (radii >= self.min_radius) & (radii <= self.max_radius)
        pts = pts[in_bounds]
        w = w[in_bounds]
        cols = cols[in_bounds]

        if pts.shape[0] == 0:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        # bucket by voxel
        vox_idx = np.floor(pts / self.voxel_size).astype(np.int32)
        voxel_map = {}

        for i, idx3 in enumerate(vox_idx):
            key = tuple(idx3)
            if w[i] < 1e-6:
                continue
            if key not in voxel_map:
                voxel_map[key] = {
                    "sum_w": 0.0,
                    "sum_xyz": np.zeros(3),
                    "sum_rgb": np.zeros(3),
                }
            voxel_map[key]["sum_w"] += w[i]
            voxel_map[key]["sum_xyz"] += w[i] * pts[i]
            voxel_map[key]["sum_rgb"] += w[i] * cols[i]

        blended_pts, blended_cols = [], []
        for vox in voxel_map.values():
            if vox["sum_w"] < 1e-6:
                continue
            blended_pts.append(vox["sum_xyz"] / vox["sum_w"])
            blended_cols.append(vox["sum_rgb"] / vox["sum_w"])

        if not blended_pts:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        return PCD(np.array(blended_pts), np.array(blended_cols), radius_image=None)

    def _radial_weight_mask(self, H, W, exp):
        yy, xx = np.indices((H, W))
        cy, cx = (H - 1)/2, (W - 1)/2
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_r = np.sqrt(cy**2 + cx**2)
        weight = 1.0 - (dist / max_r)**exp
        weight[weight < 0] = 0
        return weight.astype(np.float32)