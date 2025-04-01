"""
base_blender.py
===============

Defines a `BaseBlender` abstract class to unify how different
PCD blending strategies handle multiple partial point clouds.
"""

from abc import ABC, abstractmethod
import numpy as np
from ..data import PCD

class BaseBlender(ABC):
    """
    Base class for all PCD blending implementations. 
    Implements common parameters such as min/max radius 
    and an abstract `process_faceset()` method.
    """

    def __init__(self, min_radius=0.0, max_radius=20.0):
        """
        Args:
            min_radius (float): Minimum distance threshold for points.
            max_radius (float): Maximum distance threshold for points.
        """
        self.min_radius = min_radius
        self.max_radius = max_radius

    @abstractmethod
    def process_faceset(self, faceset, model, grad_threshold=0.1, feather_exp=2.0):
        """
        Must be overridden to handle how multiple GnomonicFaces
        get merged into a single PCD.

        Args:
            faceset: A GnomonicFaceSet to blend.
            model: Depth model used to compute radius or depth.
            grad_threshold (float): Edge gradient threshold.
            feather_exp (float): Exponent for radial weighting or blending.

        Returns:
            PCD: The resulting merged point cloud.
        """
        pass

    @staticmethod
    def compute_radius(depth, u, v):
        """
        Given a local coordinate system (u,v) ∈ [-1..1], 
        compute a radial distance from a center based on depth.

        Args:
            depth (np.ndarray): Depth map, shape (H, W).
            u, v (np.ndarray): Meshgrids, each shape (H, W).

        Returns:
            np.ndarray: radius map, shape (H, W).
        """
        return np.sqrt((depth * u)**2 + (depth * v)**2 + depth**2)

    def _to_pcd(self, radius, color_map):
        """
        Common helper for equirectangular data. 
        Converts (H, W) radius + color arrays into a PCD, 
        filtering out-of-bound points.

        Args:
            radius (np.ndarray): (H, W) radius map.
            color_map (np.ndarray): (H, W, 3) color map.

        Returns:
            PCD: final point cloud
        """
        H, W = radius.shape

        # Convert lat-lon grid to cartesian
        lat_vals = np.linspace(90, -90, H)
        lon_vals = np.linspace(-180, 180, W)
        lon, lat = np.meshgrid(lon_vals, lat_vals)
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)

        X = radius * np.cos(lat_rad) * np.cos(lon_rad)
        Y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        Z = radius * np.sin(lat_rad)

        valid = (radius > self.min_radius) & (radius < self.max_radius)
        xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        rgb = color_map.reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        return PCD(xyz[valid_flat], rgb[valid_flat], radius_image=radius)