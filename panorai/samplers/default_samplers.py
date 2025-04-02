"""
default_samplers.py
===================

Implements several built-in Sampler classes:
 - CubeSampler
 - IcosahedronSampler
 - FibonacciSampler
 - SpiralSampler
 - BlueNoiseSampler
 - HEALPixSampler

Each sampler produces a list of (lat, lon) tangent points
based on a specific geometric or random strategy.
"""

import numpy as np
from typing import List, Tuple, Any
from .registry import SamplerRegistry
from .base_samplers import Sampler
# import healpy as hp #=== on hold

@SamplerRegistry.register("cube")
class CubeSampler(Sampler):
    """
    A simple sampler that returns 6 orthogonal directions 
    (front, right, back, left, up, down).
    """

    def get_tangent_points(self) -> List[Tuple[float, float]]:
        points = [
            (0, 0),
            (0, 90),
            (0, 180),
            (0, -90),
            (90, 0),
            (-90, 0)
        ]
        return self._rotate_points(points)

@SamplerRegistry.register("icosahedron")
class IcosahedronSampler(Sampler):
    """
    Generates points based on an icosahedron's vertices.
    Can be subdivided for higher density.
    """

    def __init__(self, subdivisions: int = 0, **kwargs: Any) -> None:
        super().__init__(n_points=kwargs.get("n_points", 10), subdivisions=subdivisions, **kwargs)

    def _generate_icosahedron(self) -> np.ndarray:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1,  phi,  0], [1,  phi,  0], [-1, -phi,  0], [1, -phi,  0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [phi,  0, -1], [phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ])
        verts = np.array([v/np.linalg.norm(v) for v in verts])
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [5, 4, 9], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        subdivisions = self.config.extra.get("subdivisions", 0) if hasattr(self.config, "extra") else 0
        return self._subdivide(verts, faces, subdivisions)

    def _subdivide(self, verts: np.ndarray, faces: np.ndarray, subdivisions: int) -> np.ndarray:
        for _ in range(subdivisions):
            mid_cache = {}
            new_faces = []

            def get_midpoint(v1_idx, v2_idx, verts):
                key = tuple(sorted((v1_idx, v2_idx)))
                if key not in mid_cache:
                    mid = (verts[v1_idx] + verts[v2_idx]) / 2
                    mid = mid / np.linalg.norm(mid)
                    verts = np.vstack([verts, mid])
                    mid_cache[key] = len(verts) - 1
                return mid_cache[key], verts

            for f in faces:
                a, b, c = f
                ab, verts = get_midpoint(a, b, verts)
                bc, verts = get_midpoint(b, c, verts)
                ca, verts = get_midpoint(c, a, verts)
                new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
            faces = np.array(new_faces)
        return verts

    def _cartesian_to_lat_lon(self, v):
        x, y, z = v
        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

    def get_tangent_points(self) -> List[Tuple[float, float]]:
        vertices = self._generate_icosahedron()
        points = [self._cartesian_to_lat_lon(v) for v in vertices]
        return self._rotate_points(points)

@SamplerRegistry.register("fibonacci")
class FibonacciSampler(Sampler):
    """
    Distributes points on a sphere using the Fibonacci spiral approach.
    """

    def __init__(self, n_points: int = 10, **kwargs: Any) -> None:
        super().__init__(n_points=n_points, **kwargs)

    def get_tangent_points(self) -> List[Tuple[float, float]]:
        n_points = self.config.n_points
        indices = np.arange(0, n_points) + 0.5
        phi = (1 + np.sqrt(5)) / 2
        theta = np.arccos(1 - 2 * indices / n_points)
        angle = 2 * np.pi * indices / phi
        x = np.sin(theta) * np.cos(angle)
        y = np.sin(theta) * np.sin(angle)
        z = np.cos(theta)
        points = [self._cartesian_to_lat_lon((x[i], y[i], z[i])) for i in range(len(x))]
        return self._rotate_points(points)

    def _cartesian_to_lat_lon(self, cartesian):
        x, y, z = cartesian
        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

@SamplerRegistry.register("spiral")
class SpiralSampler(Sampler):
    """
    Generates points via a simple spiral pattern across the sphere.
    """

    def __init__(self, n_points: int = 100, **kwargs):
        super().__init__(n_points=n_points, **kwargs)

    def get_tangent_points(self) -> List[Tuple[float, float]]:
        n = self.config.n_points
        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, 2 * np.pi * n / np.sqrt(n), n)
        latitudes = np.degrees(np.arccos(1 - 2 * theta / np.pi)) - 90
        longitudes = np.degrees(phi) % 360 - 180
        return list(zip(latitudes, longitudes))

@SamplerRegistry.register("blue_noise")
class BlueNoiseSampler(Sampler):
    """
    Randomly samples points on a sphere, ensuring that no two points
    are within a small distance of each other (approx 'blue noise').
    """

    def __init__(self, n_points: int = 100, **kwargs):
        super().__init__(n_points=n_points, **kwargs)

    def get_tangent_points(self) -> List[Tuple[float, float]]:
        n = self.config.n_points
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        sampled_points = []
        for i in range(n):
            p = (x[i], y[i], z[i])
            # crude, unoptimized approach just for demonstration
            if all(np.linalg.norm(p - np.array(q)) > 0.1 for q in sampled_points):
                sampled_points.append(p)
        points = [self._cartesian_to_lat_lon(p) for p in sampled_points]
        return points

    def _cartesian_to_lat_lon(self, cartesian):
        x, y, z = cartesian
        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

# @SamplerRegistry.register("healpix")
# class HEALPixSampler(Sampler):
#     """
#     Uses HEALPix (Hierarchical Equal Area isoLatitude Pixelation) 
#     to generate a set of lat/lon points.
#     """

#     def __init__(self, nside: int = 4, **kwargs):
#         super().__init__(nside=nside, **kwargs)

#     def get_tangent_points(self) -> List[Tuple[float, float]]:
#         npix = hp.nside2npix(self.config.nside if hasattr(self.config, "nside") else 4)
#         theta, phi = hp.pix2ang(self.config.nside if hasattr(self.config, "nside") else 4, np.arange(npix))
#         latitudes = 90 - np.degrees(theta)
#         longitudes = np.degrees(phi) - 180
#         return list(zip(latitudes, longitudes))