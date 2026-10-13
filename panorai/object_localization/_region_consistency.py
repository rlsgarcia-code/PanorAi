"""Regional utility and registered transfer for spherical hypotheses.

The operations in this module implement the region-level equations from the
multiview CAM hypothesis methodology. They deliberately keep semantic support,
radial-depth validity, and numeric values as separate quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from panorai.core.numpy import spherical_solid_angle_weights as _core_solid_angle_weights

from panorai.geometry import (
    GnomonicSpec,
    erp_pixels_to_rays,
    rays_to_erp_pixels,
    rays_to_gnomonic_pixels,
)
from panorai.reconstruction import SphericalCameraPose

from ._multiview import (
    HypothesisGnomonicRegion,
    RegisteredSemanticEvidence,
    SemanticConcept,
)


def _readonly(value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def spherical_solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return exact ERP pixel-cell area weights from the spherical core."""

    return _core_solid_angle_weights(shape_hw)


def spherical_region_mask(
    spec: GnomonicSpec,
    shape_hw: tuple[int, int],
    *,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Rasterize a gnomonic footprint on a canonical ERP by pixel membership."""

    height, width = shape_hw
    if height < 1 or width < 1:
        raise ValueError("shape_hw must contain positive dimensions")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunk_rows must be a positive integer")
    result = np.zeros((height, width), dtype=bool)
    columns = np.arange(width, dtype=np.float64)
    for row_start in range(0, height, chunk_rows):
        row_stop = min(row_start + chunk_rows, height)
        rows = np.arange(row_start, row_stop, dtype=np.float64)
        xx, yy = np.meshgrid(columns, rows)
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
        rays = erp_pixels_to_rays(pixels, shape_hw)
        projected = rays_to_gnomonic_pixels(rays, spec)
        result[row_start:row_stop] = np.asarray(projected.valid, dtype=bool).reshape(
            row_stop - row_start, width
        )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SphericalRegionMetrics:
    """Solid-angle-aware utility measurements for one evidence-bearing region."""

    view_id: str
    concept_id: str
    semantic_mass_recall: float | None
    observed_sphere_fraction: float | None
    mean_score_enrichment: float | None
    supported_evidence_mass: float
    region_evidence_mass: float
    supported_solid_angle_weight: float
    region_solid_angle_weight: float
    region_pixel_count: int
    supported_region_pixel_count: int

    def __post_init__(self) -> None:
        if not self.view_id.strip() or not self.concept_id.strip():
            raise ValueError("view_id and concept_id must be non-empty")
        for name in (
            "semantic_mass_recall",
            "observed_sphere_fraction",
            "mean_score_enrichment",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "supported_evidence_mass",
            "region_evidence_mass",
            "supported_solid_angle_weight",
            "region_solid_angle_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.region_pixel_count < 0 or self.supported_region_pixel_count < 0:
            raise ValueError("pixel counts must be non-negative")


def _concept_field(
    evidence: RegisteredSemanticEvidence, concept: SemanticConcept
) -> np.ndarray:
    if evidence.vocabulary != concept.vocabulary:
        raise ValueError("concept and evidence vocabularies are incompatible")
    channels = {name: index for index, name in enumerate(evidence.channel_ids)}
    values = np.zeros(evidence.shape_hw, dtype=np.float64)
    for proxy_id, weight in concept.proxy_weights:
        if proxy_id in channels:
            values += weight * evidence.scores[channels[proxy_id]]
    return values


def evaluate_spherical_region(
    evidence: RegisteredSemanticEvidence,
    concept: SemanticConcept,
    region: HypothesisGnomonicRegion,
) -> SphericalRegionMetrics:
    """Evaluate semantic mass, angular cost, and enrichment over support."""

    if evidence.view_id != region.view_id:
        raise ValueError("region and semantic evidence must identify the same view")
    values = _concept_field(evidence, concept)
    region_mask = spherical_region_mask(region.spec, evidence.shape_hw)
    support = np.asarray(evidence.support, dtype=bool)
    inside = support & region_mask
    weights = spherical_solid_angle_weights(evidence.shape_hw)

    supported_area = float(weights[support].sum())
    region_area = float(weights[inside].sum())
    supported_mass = float((weights[support] * values[support]).sum())
    region_mass = float((weights[inside] * values[inside]).sum())

    mass_recall = region_mass / supported_mass if supported_mass > 0.0 else None
    sphere_fraction = region_area / supported_area if supported_area > 0.0 else None
    if supported_mass > 0.0 and supported_area > 0.0 and region_area > 0.0:
        global_mean = supported_mass / supported_area
        local_mean = region_mass / region_area
        enrichment = local_mean / global_mean
    else:
        enrichment = None
    return SphericalRegionMetrics(
        view_id=evidence.view_id,
        concept_id=concept.concept_id,
        semantic_mass_recall=mass_recall,
        observed_sphere_fraction=sphere_fraction,
        mean_score_enrichment=enrichment,
        supported_evidence_mass=supported_mass,
        region_evidence_mass=region_mass,
        supported_solid_angle_weight=supported_area,
        region_solid_angle_weight=region_area,
        region_pixel_count=int(region_mask.sum()),
        supported_region_pixel_count=int(inside.sum()),
    )


@dataclass(frozen=True, slots=True)
class RegisteredRadialDepth:
    """Radial range attached to one registered panorama.

    Support is authoritative. Supported samples must be finite and strictly
    positive; unsupported numeric values are ignored and never become zero
    range.
    """

    view_id: str
    values: np.ndarray
    support: np.ndarray
    units: str
    source_id: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.view_id.strip()
            or not self.units.strip()
            or not self.source_id.strip()
        ):
            raise ValueError("view_id, units, and source_id must be non-empty")
        values = _readonly(self.values, dtype=np.float64)
        support = _readonly(self.support, dtype=bool)
        if values.ndim != 2 or support.shape != values.shape:
            raise ValueError("radial depth and support must have the same HW shape")
        invalid = support & (~np.isfinite(values) | (values <= 0.0))
        if np.any(invalid):
            raise ValueError(
                "supported radial depth samples must be finite and strictly positive"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.values.shape[0]), int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class DirectionalRegionTransfer:
    """One registered region transfer and its target-domain overlap."""

    source_view_id: str
    target_view_id: str
    transferred_mask: np.ndarray
    target_region_mask: np.ndarray
    target_support: np.ndarray
    solid_angle_iou: float
    source_region_pixel_count: int
    source_supported_depth_count: int
    transferred_target_pixel_count: int

    def __post_init__(self) -> None:
        if not self.source_view_id.strip() or not self.target_view_id.strip():
            raise ValueError("source and target view IDs must be non-empty")
        transferred = _readonly(self.transferred_mask, dtype=bool)
        target = _readonly(self.target_region_mask, dtype=bool)
        support = _readonly(self.target_support, dtype=bool)
        if transferred.ndim != 2 or target.shape != transferred.shape:
            raise ValueError("transferred and target masks must share one HW shape")
        if support.shape != transferred.shape:
            raise ValueError("target support must match target masks")
        if (
            not math.isfinite(self.solid_angle_iou)
            or not 0.0 <= self.solid_angle_iou <= 1.0
        ):
            raise ValueError("solid_angle_iou must lie within [0,1]")
        for name in (
            "source_region_pixel_count",
            "source_supported_depth_count",
            "transferred_target_pixel_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "transferred_mask", transferred)
        object.__setattr__(self, "target_region_mask", target)
        object.__setattr__(self, "target_support", support)


@dataclass(frozen=True, slots=True)
class SymmetricRegionTransfer:
    """Two independently directed transfers and their symmetric score."""

    forward: DirectionalRegionTransfer
    reverse: DirectionalRegionTransfer
    symmetric_solid_angle_iou: float

    def __post_init__(self) -> None:
        if self.forward.source_view_id != self.reverse.target_view_id or (
            self.forward.target_view_id != self.reverse.source_view_id
        ):
            raise ValueError("forward and reverse transfers must exchange view IDs")
        expected = 0.5 * (self.forward.solid_angle_iou + self.reverse.solid_angle_iou)
        if not math.isclose(
            self.symmetric_solid_angle_iou, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("symmetric score must be the mean of both directions")


def _weighted_iou(first: np.ndarray, second: np.ndarray, support: np.ndarray) -> float:
    one = np.asarray(first, dtype=bool) & support
    two = np.asarray(second, dtype=bool) & support
    weights = spherical_solid_angle_weights(one.shape)
    intersection = float(weights[one & two].sum())
    union = float(weights[one | two].sum())
    return intersection / union if union > 0.0 else 1.0


def transfer_spherical_region(
    source_region: HypothesisGnomonicRegion,
    target_region: HypothesisGnomonicRegion,
    source_depth: RegisteredRadialDepth,
    source_pose: SphericalCameraPose,
    target_pose: SphericalCameraPose,
    *,
    target_support: np.ndarray | None = None,
) -> DirectionalRegionTransfer:
    """Transfer one region through radial depth into another registered view."""

    if source_region.view_id != source_depth.view_id:
        raise ValueError("source region and radial depth view IDs must match")
    if source_pose.panorama_id != source_region.view_id:
        raise ValueError("source pose and region view IDs must match")
    if target_pose.panorama_id != target_region.view_id:
        raise ValueError("target pose and region view IDs must match")
    support = (
        np.ones(source_depth.shape_hw, dtype=bool)
        if target_support is None
        else np.asarray(target_support, dtype=bool)
    )
    if support.ndim != 2:
        raise ValueError("target_support must be an HW mask")
    target_shape = (int(support.shape[0]), int(support.shape[1]))

    source_mask = spherical_region_mask(source_region.spec, source_depth.shape_hw)
    selected = source_mask & source_depth.support
    transferred = np.zeros(target_shape, dtype=bool)
    if np.any(selected):
        rows, columns = np.nonzero(selected)
        pixels = np.column_stack((columns.astype(np.float64), rows.astype(np.float64)))
        camera_rays = np.asarray(
            erp_pixels_to_rays(pixels, source_depth.shape_hw), dtype=np.float64
        )
        world_rays = camera_rays @ np.asarray(source_pose.R, dtype=np.float64)
        ranges = source_depth.values[rows, columns]
        points = (
            np.asarray(source_pose.center, dtype=np.float64)[None, :]
            + ranges[:, None] * world_rays
        )
        target_vectors = (
            points - np.asarray(target_pose.center, dtype=np.float64)[None, :]
        ) @ np.asarray(target_pose.R, dtype=np.float64).T
        projected = rays_to_erp_pixels(target_vectors, target_shape)
        valid = np.asarray(projected.valid, dtype=bool)
        target_pixels = np.asarray(projected.pixels_xy, dtype=np.float64)
        if np.any(valid):
            target_columns = (
                np.floor(target_pixels[valid, 0] + 0.5).astype(np.int64)
                % target_shape[1]
            )
            target_rows = np.floor(target_pixels[valid, 1] + 0.5).astype(np.int64)
            inside = (target_rows >= 0) & (target_rows < target_shape[0])
            transferred[target_rows[inside], target_columns[inside]] = True
    transferred &= support
    target_mask = spherical_region_mask(target_region.spec, target_shape) & support
    return DirectionalRegionTransfer(
        source_view_id=source_region.view_id,
        target_view_id=target_region.view_id,
        transferred_mask=transferred,
        target_region_mask=target_mask,
        target_support=support,
        solid_angle_iou=_weighted_iou(transferred, target_mask, support),
        source_region_pixel_count=int(source_mask.sum()),
        source_supported_depth_count=int(selected.sum()),
        transferred_target_pixel_count=int(transferred.sum()),
    )


def symmetric_spherical_region_transfer(
    first_region: HypothesisGnomonicRegion,
    second_region: HypothesisGnomonicRegion,
    first_depth: RegisteredRadialDepth,
    second_depth: RegisteredRadialDepth,
    first_pose: SphericalCameraPose,
    second_pose: SphericalCameraPose,
) -> SymmetricRegionTransfer:
    """Evaluate symmetric overlap using two independently inferred regions."""

    if first_depth.units != second_depth.units:
        raise ValueError("radial depth units must match for symmetric transfer")
    forward = transfer_spherical_region(
        first_region,
        second_region,
        first_depth,
        first_pose,
        second_pose,
        target_support=second_depth.support,
    )
    reverse = transfer_spherical_region(
        second_region,
        first_region,
        second_depth,
        second_pose,
        first_pose,
        target_support=first_depth.support,
    )
    return SymmetricRegionTransfer(
        forward=forward,
        reverse=reverse,
        symmetric_solid_angle_iou=0.5
        * (forward.solid_angle_iou + reverse.solid_angle_iou),
    )
