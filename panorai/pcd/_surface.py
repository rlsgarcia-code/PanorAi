"""Optional Open3D adapters for canonical reconstruction results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from panorai.reconstruction import SurfaceMesh


def surface_mesh_to_open3d(surface: SurfaceMesh) -> Any:
    """Convert a :class:`SurfaceMesh` without making Open3D mandatory.

    Geometry remains in the canonical XYZ world frame and units recorded by
    ``surface``.  Invalid vertex colors are not attached as invented black;
    colors are exported only when every vertex has an explicitly valid color.
    """

    from panorai.reconstruction import SurfaceMesh

    if not isinstance(surface, SurfaceMesh):
        raise TypeError("surface must be a SurfaceMesh")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "surface_mesh_to_open3d requires the optional 'pcd' extra"
        ) from exc
    result = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(surface.vertices_xyz)),
        triangles=o3d.utility.Vector3iVector(np.asarray(surface.faces)),
    )
    result.vertex_normals = o3d.utility.Vector3dVector(np.asarray(surface.normals_xyz))
    if (
        surface.colors_rgb is not None
        and surface.color_validity is not None
        and bool(np.all(surface.color_validity))
    ):
        result.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(surface.colors_rgb)
        )
    return result


__all__ = ["surface_mesh_to_open3d"]
