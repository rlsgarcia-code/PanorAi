"""Stable spherical-geometry API for PanorAi 3.x.

The canonical frame is +X right, +Y up and +Z forward. ERP coordinates refer
to pixel centres, start at the top-left, wrap horizontally, and store latitude
from +pi/2 at the top to -pi/2 at the bottom. Depth values are radial ranges.
"""

from ._contracts import (
    CubemapSpec,
    ERPPointProjection,
    GnomonicFaceGeometry,
    GnomonicPointProjection,
    GnomonicRayProjection,
    GnomonicSpec,
    ProjectionResult,
)
from ._engine import (
    CUBE_FACE_BASES,
    CUBE_FACE_ORDER,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_face_geometry,
    gnomonic_intrinsics,
    gnomonic_pixel_map,
    gnomonic_pixels_to_rays,
    gnomonic_rotation,
    gnomonic_to_equirectangular,
    rays_to_gnomonic_pixels,
    rays_to_erp_pixels,
)
from ._projectors import CubemapProjector, GnomonicProjector
from ._typing import (
    ArrayLike,
    Interpolation,
    InvalidPolicy,
    NumpyArray,
    ShapeHW,
    TensorLike,
)

__all__ = [
    "CUBE_FACE_BASES",
    "CUBE_FACE_ORDER",
    "ArrayLike",
    "CubemapProjector",
    "CubemapSpec",
    "ERPPointProjection",
    "GnomonicFaceGeometry",
    "GnomonicPointProjection",
    "GnomonicProjector",
    "GnomonicRayProjection",
    "GnomonicSpec",
    "Interpolation",
    "InvalidPolicy",
    "NumpyArray",
    "ProjectionResult",
    "ShapeHW",
    "TensorLike",
    "cubemap_to_equirectangular",
    "equirectangular_to_cubemap",
    "equirectangular_to_gnomonic",
    "erp_pixels_to_rays",
    "gnomonic_face_geometry",
    "gnomonic_intrinsics",
    "gnomonic_pixel_map",
    "gnomonic_pixels_to_rays",
    "gnomonic_rotation",
    "gnomonic_to_equirectangular",
    "rays_to_gnomonic_pixels",
    "rays_to_erp_pixels",
]
