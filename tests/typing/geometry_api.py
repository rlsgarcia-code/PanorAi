from __future__ import annotations

import numpy as np
import numpy.typing as npt

from panorai.geometry import (
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    ProjectionResult,
)


FloatImage = npt.NDArray[np.float64]


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


source = np.ones((9, 18), dtype=np.float64)
view = project_view(source)
assert view.data.shape == (5, 7)
assert round_trip_view(source).shape == source.shape
assert round_trip_cube(source).shape == source.shape
