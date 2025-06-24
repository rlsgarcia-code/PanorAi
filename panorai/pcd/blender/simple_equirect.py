"""
simple_equirect.py
==================

Provides the simplest equirectangular blending approach: 
accumulate (radius, color) in a single 2D grid with weighting.
"""

import numpy as np
from panorai.data.gnomonic_image import GnomonicFace
from panorai.pcd.data import PCD
from panorai.pcd.handler import PCDHandler
from .base_blender import BaseBlender

class SimpleEquirectBlender(BaseBlender):
    """
    A simple approach that:
      1) Projects each face onto an equirectangular grid.
      2) Accumulates radius and color.
      3) Takes a weighted average per pixel.
    """

    def __init__(self, eq_shape=(512, 1024), min_radius=0.0, max_radius=20.0):
        super().__init__(min_radius, max_radius)
        if len(eq_shape) < 2:
            raise ValueError("eq_shape must have at least two dimensions")
        self.H, self.W = eq_shape[:2]
        self._reset_accumulators()

    def _reset_accumulators(self):
        self.sum_w = np.zeros((self.H, self.W), dtype=np.float32)
        self.sum_r = np.zeros((self.H, self.W), dtype=np.float32)
        self.sum_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)

    def process_faceset(self, faceset, model, grad_threshold=0.1, feather_exp=1.0):
        """
        Projects each face to equirect, accumulates radius & color, 
        and finalizes into a single PCD.

        Args:
            faceset: GnomonicFaceSet
            model: Depth model for each face
            grad_threshold: gradient cutoff
            feather_exp: radial feather exponent
        """
        def radial_weight_mask(H, W, exp):
            # simple radial attenuation
            yy, xx = np.indices((H, W))
            cy, cx = (H - 1) / 2, (W - 1) / 2
            norm = np.sqrt((yy - cy)**2 + (xx - cx)**2) / np.sqrt(cy**2 + cx**2)
            return np.clip(1.0 - norm**exp, 0.0, 1.0)

        for face in faceset:
            face_img = np.array(face)                # (Hf, Wf, 3)
            depth = model(face_img).astype(np.float32)   # (Hf, Wf)
            Hf, Wf = depth.shape

            # compute radius
            u, v = np.meshgrid(np.linspace(-1, 1, Wf),
                               np.linspace(-1, 1, Hf),
                               indexing='xy')
            R = self.compute_radius(depth, u, v)

            # gradient masking
            grad_mask = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold).astype(np.float32)
            wmask = radial_weight_mask(Hf, Wf, feather_exp) * grad_mask

            # reproject to equirect
            def reproject(array):
                face_obj = GnomonicFace(np.repeat(array[..., None], 3, axis=-1),
                                        face.lat, face.lon, face.fov)
                return np.array(face_obj.to_equirectangular((self.H, self.W)))[..., 0]

            eq_r = reproject(R)
            eq_w = reproject(wmask)
            eq_c = np.array(face.to_equirectangular((self.H, self.W))).astype(np.float32) / 255.

            valid = (eq_r >= self.min_radius) & (eq_r <= self.max_radius)
            self.accumulate(eq_r, eq_c, eq_w, valid)

        return self._finalize()

    def accumulate(self, radius_map, color_map, weight_map, valid_mask):
        w = weight_map * valid_mask
        self.sum_w += w
        self.sum_r += radius_map * w
        self.sum_rgb += color_map * w[..., None]

    def _finalize(self):
        valid = (self.sum_w > 1e-6)
        radius = np.zeros_like(self.sum_r)
        colors = np.zeros_like(self.sum_rgb)

        radius[valid] = self.sum_r[valid] / self.sum_w[valid]
        colors[valid] = self.sum_rgb[valid] / self.sum_w[valid, None]

        # Convert to 3D points
        lat_vals = np.linspace(90, -90, self.H)
        lon_vals = np.linspace(-180, 180, self.W)
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        lat_rad = np.radians(lat_grid)
        lon_rad = np.radians(lon_grid)

        X = radius * np.cos(lat_rad) * np.cos(lon_rad)
        Y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        Z = radius * np.sin(lat_rad)

        xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        rgb = colors.reshape(-1, 3)
        mask = valid.reshape(-1)

        return PCD(xyz[mask], rgb[mask], radius_image=radius)