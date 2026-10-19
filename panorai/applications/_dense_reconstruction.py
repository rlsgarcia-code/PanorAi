"""Registered spherical MVS, TSDF fusion, and surface reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from panorai.pipeline import Pipeline
from panorai.reconstruction import (
    BuildRegisteredReferenceSets,
    ColorizeSurface,
    EstimateMultiViewRanges,
    ExtractSurface,
    FuseTSDF,
    RegisteredSphericalScene,
    SphericalDenseReconstructionResult,
    TSDFSpec,
    ValidateDenseReconstruction,
)
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.panoramic import (
    FullSphereStereoOptions,
    OverlappingCubemapPlanner,
)

from ._base import PipelineApplication, application_config


def _options_from_dict(raw: dict[str, Any]) -> FullSphereStereoOptions:
    planner = raw.get("planner")
    stereo = raw.get("stereo")
    if not isinstance(planner, dict) or not isinstance(stereo, dict):
        raise ValueError("MVS planner and stereo options are required")
    values = dict(raw)
    values["planner"] = OverlappingCubemapPlanner(**planner)
    values["stereo"] = SphericalStereoOptions(**stereo)
    return FullSphereStereoOptions(**values)


def _reviewed_mvs_options(filter_backend: str) -> FullSphereStereoOptions:
    return replace(
        FullSphereStereoOptions.reviewed_profile(filter_backend=filter_backend),
        min_source_evidence=2,
        min_range_consensus_sources=2,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SphericalDenseReconstruction(PipelineApplication):
    """Reconstruct a world-frame surface from registered metric ERPs.

    The application is Experimental.  ``reviewed_profile`` freezes the
    intended future Stable subset, but does not itself claim Stable maturity.
    """

    voxel_size: float
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    mvs_options: FullSphereStereoOptions = FullSphereStereoOptions()
    truncation_distance: float | None = None
    max_voxels: int = 8_000_000
    chunk_voxels: int = 262_144
    min_surface_views: int = 2
    min_valid_weight: float = 1.0
    weighting: str = "uniform"
    colorize: bool = True
    retain_intermediates: str = "ranges"
    profile_id: str | None = None

    application_id = "panorai.applications.SphericalDenseReconstruction"
    interface = "panorai-spherical-dense-reconstruction/v1-experimental"
    stability = "experimental"

    def __post_init__(self) -> None:
        self._tsdf_spec()
        if self.retain_intermediates not in {"none", "ranges", "all"}:
            raise ValueError("retain_intermediates must be 'none', 'ranges', or 'all'")
        if (
            self.profile_id is not None
            and self.profile_id != self.reviewed_profile_id()
        ):
            raise ValueError("unknown dense-reconstruction profile_id")
        if self.profile_id is not None and not self._matches_reviewed_profile():
            raise ValueError(
                "reviewed profile_id cannot be attached to modified profile parameters"
            )

    @staticmethod
    def reviewed_profile_id() -> str:
        return "panorai-spherical-dense-reconstruction/reviewed-2026-10-19"

    @classmethod
    def reviewed_profile(
        cls,
        *,
        voxel_size: float,
        bounds: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None,
        filter_backend: str = "auto",
    ) -> "SphericalDenseReconstruction":
        """Build the frozen candidate profile; external promotion gates remain open."""

        options = _reviewed_mvs_options(filter_backend)
        return cls(
            voxel_size=voxel_size,
            bounds=bounds,
            mvs_options=options,
            truncation_distance=4.0 * voxel_size,
            max_voxels=8_000_000,
            min_surface_views=2,
            min_valid_weight=1.0,
            weighting="uniform",
            colorize=True,
            retain_intermediates="ranges",
            profile_id=cls.reviewed_profile_id(),
        )

    def _matches_reviewed_profile(self) -> bool:
        expected_options = _reviewed_mvs_options(self.mvs_options.stereo.filter_backend)
        return (
            self.mvs_options == expected_options
            and self.truncation_distance == 4.0 * self.voxel_size
            and self.max_voxels == 8_000_000
            and self.chunk_voxels == 262_144
            and self.min_surface_views == 2
            and self.min_valid_weight == 1.0
            and self.weighting == "uniform"
            and self.colorize
            and self.retain_intermediates == "ranges"
        )

    def with_params(self, **params: Any) -> "SphericalDenseReconstruction":
        updates = dict(params)
        if self.profile_id is not None and set(updates).difference({"bounds"}):
            updates["profile_id"] = None
        updated = PipelineApplication.with_params(self, **updates)
        if not isinstance(updated, SphericalDenseReconstruction):
            raise RuntimeError("application parameter update changed concrete type")
        if updated.profile_id is not None and not updated._matches_reviewed_profile():
            return replace(updated, profile_id=None)
        return updated

    def _tsdf_spec(self) -> TSDFSpec:
        return TSDFSpec(
            voxel_size=self.voxel_size,
            bounds=self.bounds,
            truncation_distance=self.truncation_distance,
            max_voxels=self.max_voxels,
            chunk_voxels=self.chunk_voxels,
            min_surface_views=self.min_surface_views,
            min_valid_weight=self.min_valid_weight,
            weighting=self.weighting,
        )

    def as_pipeline(self) -> Pipeline:
        return Pipeline(
            ("build_registered_reference_sets", BuildRegisteredReferenceSets()),
            ("estimate_multi_view_ranges", EstimateMultiViewRanges(self.mvs_options)),
            ("fuse_tsdf", FuseTSDF(self._tsdf_spec())),
            ("extract_surface", ExtractSurface()),
            ("colorize_surface", ColorizeSurface(self.colorize)),
            (
                "validate_dense_reconstruction",
                ValidateDenseReconstruction(self.retain_intermediates),
            ),
        )

    def run(
        self, scene: RegisteredSphericalScene
    ) -> SphericalDenseReconstructionResult:
        if not isinstance(scene, RegisteredSphericalScene):
            raise TypeError("scene must be a RegisteredSphericalScene")
        if self.profile_id == self.reviewed_profile_id() and len(scene.views) < 3:
            raise ValueError("the reviewed profile requires at least three views")
        pipeline_run = self.as_pipeline().run(scene)
        result = pipeline_run.output
        if not isinstance(result, SphericalDenseReconstructionResult):
            raise RuntimeError(
                "dense reconstruction pipeline returned an unexpected result"
            )
        diagnostics = dict(result.diagnostics)
        diagnostics["pipeline_stages"] = tuple(
            record.name for record in pipeline_run.records
        )
        diagnostics["stage_diagnostics"] = {
            record.name: dict(record.details) for record in pipeline_run.records
        }
        diagnostics["profile_id"] = self.profile_id
        return replace(result, diagnostics=diagnostics)

    def describe(self) -> dict[str, Any]:
        spec = self._tsdf_spec()
        return {
            "interface": self.interface,
            "stability": self.stability,
            "application": self.application_id,
            "input": "RegisteredSphericalScene with central ERPs and metric poses",
            "output": "SphericalDenseReconstructionResult with optional SurfaceMesh",
            "route": "multireference-full-sphere-mvs -> dense-tsdf -> marching-cubes",
            "profile_id": self.profile_id,
            "profile_status": "candidate-not-promoted" if self.profile_id else "custom",
            "confidence_semantics": "not compared across references in uniform mode",
            "unknown_semantics": "TSDF weight zero; never inferred free space",
            "tsdf": asdict(spec),
            "pipeline": self.as_pipeline().describe(),
        }

    def to_config(self) -> dict[str, Any]:
        return application_config(
            self.application_id,
            {
                "voxel_size": self.voxel_size,
                "bounds": self.bounds,
                "mvs_options": asdict(self.mvs_options),
                "truncation_distance": self.truncation_distance,
                "max_voxels": self.max_voxels,
                "chunk_voxels": self.chunk_voxels,
                "min_surface_views": self.min_surface_views,
                "min_valid_weight": self.min_valid_weight,
                "weighting": self.weighting,
                "colorize": self.colorize,
                "retain_intermediates": self.retain_intermediates,
                "profile_id": self.profile_id,
            },
        )

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "SphericalDenseReconstruction":
        if payload.get("application") != cls.application_id:
            raise ValueError(
                "configuration does not describe SphericalDenseReconstruction"
            )
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("dense reconstruction parameters are missing")
        raw_options = parameters.get("mvs_options")
        if not isinstance(raw_options, dict):
            raise ValueError("dense reconstruction MVS options are missing")
        values = dict(parameters)
        values["mvs_options"] = _options_from_dict(raw_options)
        raw_bounds = values.get("bounds")
        if raw_bounds is not None:
            values["bounds"] = tuple(tuple(float(x) for x in row) for row in raw_bounds)
        return cls(**values)


__all__ = ["SphericalDenseReconstruction"]
