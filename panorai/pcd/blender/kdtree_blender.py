"""
kdtree_blender.py
=================

Defines a KDTreeBlender that merges 3D points within a search radius,
averaging their positions and colors.
"""

import numpy as np
import open3d as o3d
from ..data import PCD
from ..handler import PCDHandler
from ...data.gnomonic_image import GnomonicFace
from .base_blender import BaseBlender

class KDTreeBlender(BaseBlender):
    """
    3rd-level strategy: merges partial point clouds using a KD-tree 
    to find neighbors within a certain radius, then do a weighted 
    average of their positions and colors.
    """

    def __init__(self, merge_radius=0.05, min_radius=0.0, max_radius=20.0, **kwargs):
        super().__init__(min_radius, max_radius)
        self.merge_radius = merge_radius

    def process_faceset(self, faceset, model, grad_threshold=0.1, feather_exp=1.0):
        """
        1) For each GnomonicFace, project to 3D using model-based depth.
        2) Combine all points into one array.
        3) Merge them with KD-tree radius search.

        Returns:
            PCD: merged point cloud.
        """
        merged_chunks = []

        def radial_weight_mask(H, W, exp):
            yy, xx = np.indices((H, W))
            cy, cx = (H - 1) / 2, (W - 1) / 2
            norm = np.sqrt((yy - cy)**2 + (xx - cx)**2) / np.sqrt(cy**2 + cx**2)
            return np.clip(1.0 - norm**exp, 0.0, 1.0)

        for face in faceset:
            face_img = np.array(face)
            depth = model(face_img).astype(np.float32)
            Hf, Wf = depth.shape

            grad_mask = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold)
            wmask = radial_weight_mask(Hf, Wf, feather_exp)

            # local CCS -> WCS
            u, v = np.meshgrid(np.linspace(-1, 1, Wf), np.linspace(-1, 1, Hf), indexing='xy')
            X = depth * u
            Y = depth * v
            Z = depth

            xyz_ccs = np.stack([Y.ravel(), X.ravel(), Z.ravel()], axis=1)
            xyz_wcs = PCDHandler.rotate_ccs_to_wcs(xyz_ccs, face.lon, 90 - face.lat)

            valid = (depth.ravel() > 0) & grad_mask.ravel()
            color_flat = (face_img.reshape(-1, 3) / 255.0)[valid]
            weights_flat = wmask.ravel()[valid]
            points_valid = xyz_wcs[valid]

            # (x, y, z, w, R, G, B)
            chunk_array = np.concatenate([
                points_valid,
                weights_flat[:, None],
                color_flat
            ], axis=1)
            merged_chunks.append(chunk_array)

        if not merged_chunks:
            return PCD(np.zeros((0, 3)), np.zeros((0, 3)))

        merged_array = np.concatenate(merged_chunks, axis=0)
        return self._blend(merged_array)

    def _blend(self, merged_array: np.ndarray) -> PCD:
        """
        Build a KD-tree of points, do a radius search, 
        then weighted-average each neighborhood.
        """
        if merged_array.shape[0] == 0:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        points = merged_array[:, :3]
        weights = merged_array[:, 3]
        colors = merged_array[:, 4:]

        # Remove out-of-bounds distances
        radii = np.linalg.norm(points, axis=1)
        in_bounds = (radii >= self.min_radius) & (radii <= self.max_radius)
        points = points[in_bounds]
        weights = weights[in_bounds]
        colors = colors[in_bounds]

        if points.shape[0] == 0:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        kd_tree = o3d.geometry.KDTreeFlann(pc)

        N = points.shape[0]
        visited = np.zeros(N, dtype=bool)

        final_positions = []
        final_colors = []

        for i in range(N):
            if visited[i]:
                continue
            [_, idxs, _] = kd_tree.search_radius_vector_3d(pc.points[i], self.merge_radius)

            sum_w = 0.0
            sum_w_xyz = np.zeros(3)
            sum_w_rgb = np.zeros(3)

            for j in idxs:
                wj = weights[j]
                sum_w += wj
                sum_w_xyz += wj * points[j]
                sum_w_rgb += wj * colors[j]
                visited[j] = True

            if sum_w < 1e-12:
                continue

            final_positions.append(sum_w_xyz / sum_w)
            final_colors.append(sum_w_rgb / sum_w)

        if not final_positions:
            return PCD(np.zeros((0,3)), np.zeros((0,3)))

        return PCD(np.vstack(final_positions), np.vstack(final_colors), radius_image=None)