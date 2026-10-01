from __future__ import annotations

import numpy as np
import numpy.typing as npt

from panorai.geometry import (
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    ProjectionResult,
    GnomonicPointProjection,
    GnomonicRayProjection,
    gnomonic_pixels_to_rays,
    rays_to_gnomonic_pixels,
)


FloatImage = npt.NDArray[np.float64]
FloatPoints = npt.NDArray[np.float64]


def project_view(image: FloatImage) -> ProjectionResult[FloatImage]:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    return projector.project(image)


def round_trip_view(image: FloatImage) -> FloatImage:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    output_shape = (image.shape[0], image.shape[1])
    return projector.back_project(projector.project(image), output_shape).data


def round_trip_cube(image: FloatImage) -> FloatImage:
    projector = CubemapProjector(CubemapSpec((5, 7)))
    output_shape = (image.shape[0], image.shape[1])
    return projector.back_project(projector.project(image), output_shape).data


def pixels_to_rays(points: FloatPoints) -> GnomonicRayProjection[FloatPoints]:
    return gnomonic_pixels_to_rays(points, GnomonicSpec(output_shape_hw=(5, 7)))


def rays_to_pixels(points: FloatPoints) -> GnomonicPointProjection[FloatPoints]:
    return rays_to_gnomonic_pixels(points, GnomonicSpec(output_shape_hw=(5, 7)))


source = np.ones((9, 18), dtype=np.float64)
view = project_view(source)
assert view.data.shape == (5, 7)
assert round_trip_view(source).shape == source.shape
assert round_trip_cube(source).shape == source.shape
assert pixels_to_rays(np.asarray([[3.0, 2.0]])).rays_xyz.shape == (1, 3)
assert rays_to_pixels(np.asarray([[0.0, 0.0, 1.0]])).pixels_xy.shape == (1, 2)
