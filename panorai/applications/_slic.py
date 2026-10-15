"""High-level spherical SLIC application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from panorai.data import EquirectangularImage
from panorai.ops.spherical import SLIC
from panorai.pipeline import OnSphere, Output, Pipeline, RasterField

from ._base import PipelineApplication, application_config


@dataclass(frozen=True, slots=True)
class SphericalSLIC(PipelineApplication):
    """Segment an ERP into intrinsic, seam-periodic spherical superpixels."""

    n_segments: int = 100
    compactness: float = 10.0
    max_num_iter: int = 10
    convergence_tol_deg: float = 1e-3
    enforce_connectivity: bool = True
    min_size_factor: float = 0.5
    start_label: int = 1

    application_id = "panorai.applications.SphericalSLIC"

    def _operation(self) -> SLIC:
        return SLIC(
            n_segments=self.n_segments,
            compactness=self.compactness,
            max_num_iter=self.max_num_iter,
            convergence_tol_deg=self.convergence_tol_deg,
            enforce_connectivity=self.enforce_connectivity,
            min_size_factor=self.min_size_factor,
            start_label=self.start_label,
        )

    def as_pipeline(self) -> Pipeline:
        return Pipeline(
            OnSphere(
                Output(self._operation(), kind="labels"),
                output="labels",
            )
        )

    def run(
        self,
        panorama: Any,
        *,
        validity_mask: np.ndarray | None = None,
    ) -> RasterField:
        if isinstance(panorama, np.ndarray):
            panorama = EquirectangularImage(panorama, valid=validity_mask)
        elif validity_mask is not None:
            raise ValueError(
                "validity_mask is accepted only with an array input; public ERP "
                "containers already carry explicit validity"
            )
        result = self.as_pipeline()(panorama)
        if not isinstance(result, RasterField):
            raise RuntimeError("SphericalSLIC pipeline returned an unexpected result")
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "canonical ERP RGB with explicit support/validity",
            "output": "RasterField(kind='labels')",
            "domain": "direct-spherical",
            "seam": "periodic-longitude",
            "algorithm": self._operation().describe(),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(self.application_id, self.get_params(deep=False))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SphericalSLIC":
        if config.get("application") != cls.application_id:
            raise ValueError("configuration does not describe SphericalSLIC")
        parameters = config.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("SphericalSLIC parameters must be a mapping")
        return cls(**parameters)


__all__ = ["SphericalSLIC"]
