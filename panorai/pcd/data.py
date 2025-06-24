
"""
data.py
=======

Defines the core ``PCD`` class which encapsulates both NumPy based point
cloud data and an optional ``open3d.PointCloud`` object for richer
manipulation or visualisation.
"""

from __future__ import annotations

import open3d as o3d
import numpy as np

class PCD:
    """
    Represents a point cloud data container that allows intuitive access
    to both NumPy arrays (points, colors, etc.) and the corresponding
    Open3D PointCloud object.

    - Stores (N,3) XYZ points, (N,3) RGB colors
    - Optionally holds a reference to a 'radius_image' object 
      for spherical context or additional metadata.
    """

    def __init__(self, points: np.ndarray, 
                 colors: np.ndarray, 
                 radius_image: np.ndarray = None,
                 indexes: np.ndarray = None,
                 shape: tuple = None):
        """
        Initializes the PCD with 3D point coordinates, colors, and 
        an optional radius image.

        Args:
            points (np.ndarray): Shape (N, 3), each row is [x, y, z].
            colors (np.ndarray): Shape (N, 3), each row is [r, g, b].
            radius_image (np.ndarray): Optional image or GnomonicFace object storing radius data.
        """
        self._points = points
        self._colors = colors
        self._radius_image = radius_image
        self._shape = shape
        self._indexes = indexes

        # Build an open3d.PointCloud when the geometry and utility modules
        # are available.  If they are missing (e.g. Open3D is not installed),
        # fall back to a very small stand-in object that simply stores the
        # points and colors arrays.  ``self._o3d`` always references an object
        # with ``points`` and ``colors`` attributes so callers do not have to
        # check for Open3D availability themselves.
        if hasattr(o3d, "geometry") and hasattr(o3d, "utility"):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(colors)
        else:
            class _FallbackPointCloud:
                pass

            pc = _FallbackPointCloud()
            pc.points = points
            pc.colors = colors

        self._o3d = pc

    def hook_bias_adjustment(self, func):
        """
        Demonstrates a specialized bias adjustment that modifies
        point distances (radii) and reprojects them.

        - Takes a function `func` that maps radius -> new radius
        - Constructs a new PCD with the adjusted distances

        Args:
            func (Callable[[np.ndarray], np.ndarray]): A function that
                accepts the old radii array and returns a new radii array.
        
        Returns:
            PCD: New point cloud with updated coordinates and radius image.
        """
        from ..data import GnomonicFace  # Delayed import to avoid circles
        R = np.linalg.norm(self.points, axis=1)
        R_safe = np.where(R == 0, 1e-8, R)
        unit = self.points / R_safe[:, None]
        points = unit * func(R)[:, None]  # apply the function to adjust distances
        colors = self.colors
        radius_image = func(np.array(self.radius_image))

        return PCD(
            points,
            colors,
            GnomonicFace(
                radius_image,
                lat=self.radius_image.lat,
                lon=self.radius_image.lon,
                fov=self.radius_image.fov
            )
        )

    @property
    def points(self) -> np.ndarray:
        """(N,3) NumPy array of XYZ points in the cloud."""
        return self._points

    @property
    def indexes(self) -> np.ndarray:
        """(N,3) NumPy array of points indexes in the cloud."""
        return self._indexes

    @property
    def shape(self) -> np.ndarray:
        """(H,W) original face shape."""
        return self._shape
    
    @property
    def radius_image(self) -> np.ndarray:
        """
        Reference to a radius image or GnomonicFace that provides
        spherical metadata, or None if not available.
        """
        return self._radius_image

    @property
    def colors(self) -> np.ndarray:
        """(N,3) NumPy array of RGB colors in the cloud."""
        return self._colors

    @property
    def o3d(self) -> o3d.geometry.PointCloud:
        """
        The internal Open3D PointCloud object. 
        Useful if you want to call methods like `.estimate_normals()`.
        """
        return self._o3d