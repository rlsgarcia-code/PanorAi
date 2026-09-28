from __future__ import annotations

import torch

from panorai.geometry import GnomonicProjector, GnomonicSpec, ProjectionResult


def project_view(image: torch.Tensor) -> ProjectionResult[torch.Tensor]:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    return projector.project(image)


def round_trip_view(image: torch.Tensor) -> torch.Tensor:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    output_shape = (image.shape[-2], image.shape[-1])
    return projector.back_project(projector.project(image), output_shape).data


source = torch.ones((2, 3, 9, 18), dtype=torch.float64)
view = project_view(source)
assert view.data.shape == (2, 3, 5, 7)
assert round_trip_view(source).shape == source.shape
