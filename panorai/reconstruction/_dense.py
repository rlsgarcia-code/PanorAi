"""MVS orchestration, dense TSDF fusion, and surface extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.geometry import (
    camera_rays_to_world_points,
    erp_pixels_to_rays,
    rays_to_erp_pixels,
)
from panorai.pipeline import StageResult
from panorai.stereo.panoramic import (
    FullSphereMultiViewStereo,
    FullSphereStereoOptions,
    RegisteredPanoramaSet,
)

from ._dense_models import (
    MultiViewRangeMap,
    MultiViewRangeSet,
    RegisteredSphericalScene,
    SphericalDenseReconstructionResult,
    SurfaceMesh,
    TSDFSpec,
    TSDFVolume,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredReferenceSets:
    """Deterministic MVS inputs, one for every selected reference camera."""

    scene: RegisteredSphericalScene
    sets: tuple[RegisteredPanoramaSet, ...]

    def __post_init__(self) -> None:
        items = tuple(self.sets)
        if not items:
            raise ValueError("at least one registered reference set is required")
        object.__setattr__(self, "sets", items)


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseReconstructionWorkspace:
    """Typed state passed between public dense-reconstruction stages."""

    scene: RegisteredSphericalScene
    reference_sets: RegisteredReferenceSets | None = None
    ranges: MultiViewRangeSet | None = None
    volume: TSDFVolume | None = None
    surface: SurfaceMesh | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))


def _pixel_grid(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.indices((height, width), dtype=np.float64)
    return np.stack((xx, yy), axis=-1)


def _bilinear_sample(
    values: np.ndarray,
    pixels_xy: np.ndarray,
    *,
    validity: np.ndarray,
    min_valid_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an ERP with periodic longitude and explicit valid-weight gating."""

    source = np.asarray(values)
    valid_source = np.asarray(validity, dtype=bool)
    if source.ndim not in {2, 3}:
        raise ValueError("bilinear ERP samples require HW or HWC values")
    if valid_source.shape != source.shape[:2]:
        raise ValueError("validity must match the ERP spatial shape")
    points = np.asarray(pixels_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixels_xy must have shape (N, 2)")
    height, width = source.shape[:2]
    finite = np.all(np.isfinite(points), axis=1)
    x = np.mod(np.where(finite, points[:, 0], 0.0), width)
    y_raw = np.where(finite, points[:, 1], 0.0)
    in_vertical_domain = (y_raw >= -0.5) & (y_raw <= height - 0.5)
    y = np.clip(y_raw, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    weights = np.stack(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ),
        axis=1,
    )
    neighbors = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    neighbor_validity = np.stack(
        [valid_source[row, column] for row, column in neighbors], axis=1
    )
    finite_values = np.stack(
        [
            (
                np.isfinite(source[row, column])
                if source.ndim == 2
                else np.all(np.isfinite(source[row, column]), axis=-1)
            )
            for row, column in neighbors
        ],
        axis=1,
    )
    usable = neighbor_validity & finite_values
    valid_weight = np.sum(weights * usable, axis=1)
    accepted = (
        finite & in_vertical_domain & (valid_weight >= float(min_valid_weight) - 1e-12)
    )
    gathered = np.stack([source[row, column] for row, column in neighbors], axis=1)
    safe = np.where(usable, weights, 0.0)
    if source.ndim == 3:
        gathered = np.where(usable[..., None], gathered, 0.0)
        sampled = np.sum(gathered * safe[..., None], axis=1)
        sampled /= np.maximum(valid_weight, 1e-12)[:, None]
        sampled[~accepted] = np.nan
    else:
        gathered = np.where(usable, gathered, 0.0)
        sampled = np.sum(gathered * safe, axis=1)
        sampled /= np.maximum(valid_weight, 1e-12)
        sampled[~accepted] = np.nan
    return sampled, accepted


def _range_world_bounds(range_set: MultiViewRangeSet, margin: float) -> np.ndarray:
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    observed = 0
    for range_map in range_set.maps:
        mask = range_map.validity & range_map.sampling_support
        if not np.any(mask):
            continue
        shape = range_map.radial_range.shape
        rays = erp_pixels_to_rays(_pixel_grid(shape), shape)[mask]
        points = camera_rays_to_world_points(
            rays,
            range_map.radial_range[mask],
            range_map.pose.R,
            range_map.pose.center,
        )
        finite = np.all(np.isfinite(points), axis=1)
        if not np.any(finite):
            continue
        points = points[finite]
        lower = np.minimum(lower, np.min(points, axis=0))
        upper = np.maximum(upper, np.max(points, axis=0))
        observed += len(points)
    if observed == 0:
        raise ValueError("cannot derive TSDF bounds without valid radial ranges")
    return np.stack((lower - margin, upper + margin))


def fuse_tsdf(range_set: MultiViewRangeSet, spec: TSDFSpec) -> TSDFVolume:
    """Fuse accepted radial ranges into a bounded dense world-frame TSDF."""

    if not isinstance(range_set, MultiViewRangeSet):
        raise TypeError("range_set must be a MultiViewRangeSet")
    if not isinstance(spec, TSDFSpec):
        raise TypeError("spec must be a TSDFSpec")
    if spec.truncation_distance is None:  # normalized by TSDFSpec.__post_init__
        raise RuntimeError("TSDFSpec did not resolve truncation_distance")
    truncation = float(spec.truncation_distance)
    bounds = (
        np.asarray(spec.bounds, dtype=np.float64)
        if spec.bounds is not None
        else _range_world_bounds(range_set, truncation)
    )
    shape_xyz = np.ceil((bounds[1] - bounds[0]) / spec.voxel_size).astype(np.int64)
    shape_xyz = np.maximum(shape_xyz, 2)
    voxel_count = int(np.prod(shape_xyz, dtype=np.int64))
    if voxel_count > spec.max_voxels:
        suggested = float(
            spec.voxel_size * (voxel_count / spec.max_voxels) ** (1.0 / 3.0)
        )
        raise ValueError(
            f"TSDF volume requires {voxel_count:,} voxels, exceeding "
            f"max_voxels={spec.max_voxels:,}; increase voxel_size to at least "
            f"approximately {suggested:.6g} {range_set.scene.units} or provide bounds"
        )
    shape_zyx = tuple(int(item) for item in shape_xyz[::-1])
    weighted_sum = np.zeros(voxel_count, dtype=np.float64)
    total_weight = np.zeros(voxel_count, dtype=np.float64)
    view_count = np.zeros(voxel_count, dtype=np.uint16)
    for start in range(0, voxel_count, spec.chunk_voxels):
        stop = min(start + spec.chunk_voxels, voxel_count)
        flat = np.arange(start, stop, dtype=np.int64)
        zz, yy, xx = np.unravel_index(flat, shape_zyx)
        points_world = (
            bounds[0]
            + (np.stack((xx, yy, zz), axis=1).astype(np.float64) + 0.5)
            * spec.voxel_size
        )
        chunk_sum = np.zeros(stop - start, dtype=np.float64)
        chunk_weight = np.zeros(stop - start, dtype=np.float64)
        chunk_views = np.zeros(stop - start, dtype=np.uint16)
        for range_map in range_set.maps:
            camera_points = (points_world - range_map.pose.center) @ range_map.pose.R.T
            projection = rays_to_erp_pixels(camera_points, range_map.radial_range.shape)
            source_validity = range_map.validity & range_map.sampling_support
            sampled_range, accepted = _bilinear_sample(
                range_map.radial_range,
                projection.pixels_xy,
                validity=source_validity,
                min_valid_weight=spec.min_valid_weight,
            )
            accepted &= projection.valid
            sdf = sampled_range - projection.ranges
            accepted &= np.isfinite(sdf)
            # A depth sample observes free space before the surface and only a
            # truncation band behind it. Farther voxels are occluded unknown,
            # not measured occupied space.
            accepted &= sdf >= -truncation
            if spec.weighting == "confidence":
                sampled_confidence, confidence_valid = _bilinear_sample(
                    range_map.confidence,
                    projection.pixels_xy,
                    validity=source_validity,
                    min_valid_weight=spec.min_valid_weight,
                )
                accepted &= confidence_valid & (sampled_confidence > 0.0)
                observation_weight = np.where(accepted, sampled_confidence, 0.0)
            else:
                observation_weight = accepted.astype(np.float64)
            normalized = np.clip(sdf / truncation, -1.0, 1.0)
            chunk_sum += np.where(accepted, normalized * observation_weight, 0.0)
            chunk_weight += observation_weight
            chunk_views += accepted.astype(np.uint16)
        weighted_sum[start:stop] = chunk_sum
        total_weight[start:stop] = chunk_weight
        view_count[start:stop] = chunk_views
    tsdf = np.ones(voxel_count, dtype=np.float32)
    observed = total_weight > 0.0
    tsdf[observed] = (weighted_sum[observed] / total_weight[observed]).astype(
        np.float32
    )
    return TSDFVolume(
        tsdf_zyx=tsdf.reshape(shape_zyx),
        weight_zyx=total_weight.astype(np.float32).reshape(shape_zyx),
        view_count_zyx=view_count.reshape(shape_zyx),
        origin_xyz=bounds[0],
        voxel_size=spec.voxel_size,
        truncation_distance=truncation,
        units=range_set.scene.units,
        frame=range_set.scene.frame,
        min_surface_views=spec.min_surface_views,
        diagnostics={
            "bounds_xyz": bounds.tolist(),
            "shape_zyx": list(shape_zyx),
            "voxel_count": voxel_count,
            "observed_voxel_count": int(np.count_nonzero(observed)),
            "weighting": spec.weighting,
            "unknown_semantics": "weight-zero",
        },
    )


def _sample_grid_nearest(grid_zyx: np.ndarray, vertices_zyx: np.ndarray) -> np.ndarray:
    indices = np.rint(vertices_zyx).astype(np.int64)
    maximum = np.asarray(grid_zyx.shape[:3], dtype=np.int64) - 1
    indices = np.clip(indices, 0, maximum)
    return grid_zyx[indices[:, 0], indices[:, 1], indices[:, 2]]


def extract_surface(volume: TSDFVolume) -> SurfaceMesh:
    """Extract a zero-level surface without inventing unknown-space geometry."""

    if not isinstance(volume, TSDFVolume):
        raise TypeError("volume must be a TSDFVolume")
    from scipy.ndimage import maximum_filter

    # A zero crossing lies between voxel centres. Require multiview evidence
    # in its immediate cell neighbourhood rather than demanding that every
    # free-space and behind-surface corner was seen by the same cameras.
    evidence_radius = max(
        1, int(np.ceil(volume.truncation_distance / volume.voxel_size))
    )
    local_view_count = maximum_filter(
        volume.view_count_zyx,
        size=2 * evidence_radius + 1,
        mode="constant",
        cval=0,
    )
    eligible = (volume.weight_zyx > 0.0) & (
        local_view_count >= volume.min_surface_views
    )
    if not np.any(eligible):
        raise ValueError("TSDF has no voxels observed by the minimum view count")
    # scikit-image requests writable C-contiguous buffers even though it does
    # not mutate them.  Public result arrays stay read-only; only this private
    # extraction buffer is copied.
    values = np.array(volume.tsdf_zyx, dtype=np.float32, copy=True, order="C")
    eligible_values = values[eligible]
    if np.min(eligible_values) > 0.0 or np.max(eligible_values) < 0.0:
        raise ValueError("TSDF has no zero crossing in sufficiently observed voxels")
    from skimage.measure import marching_cubes

    # NumPy 2.5 deprecates assigning ``array.shape``. Released scikit-image
    # versions still do that once while decoding the internal Marching Cubes
    # lookup tables. Keep ``-W error`` effective for every other warning while
    # containing this exact third-party compatibility warning to the external
    # call that emits it.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Setting the shape on a NumPy array has been deprecated in "
                r"NumPy 2\.5\."
            ),
            category=DeprecationWarning,
        )
        vertices_zyx, faces, marching_normals_zyx, _ = marching_cubes(
            values,
            level=0.0,
            spacing=(volume.voxel_size,) * 3,
            gradient_direction="ascent",
            mask=np.array(eligible, dtype=bool, copy=True, order="C"),
            allow_degenerate=False,
        )
    grid_vertices_zyx = vertices_zyx / volume.voxel_size
    vertex_weight = _sample_grid_nearest(volume.weight_zyx, grid_vertices_zyx)
    vertex_views = _sample_grid_nearest(local_view_count, grid_vertices_zyx)
    keep_vertex = (vertex_weight > 0.0) & (vertex_views >= volume.min_surface_views)
    keep_face = np.all(keep_vertex[faces], axis=1)
    face_centres = np.mean(grid_vertices_zyx[faces], axis=1)
    cell_origin = np.floor(face_centres).astype(np.int64)
    cell_origin = np.clip(
        cell_origin,
        0,
        np.asarray(volume.tsdf_zyx.shape, dtype=np.int64) - 2,
    )
    cell_observed = np.ones(len(faces), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                cell_observed &= (
                    volume.weight_zyx[
                        cell_origin[:, 0] + dz,
                        cell_origin[:, 1] + dy,
                        cell_origin[:, 2] + dx,
                    ]
                    > 0.0
                )
    keep_face &= cell_observed
    faces = faces[keep_face]
    if not len(faces):
        raise ValueError("zero crossing touches only unknown or weakly observed voxels")
    used = np.unique(faces)
    remap = np.full(len(vertices_zyx), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[faces]
    vertices_zyx = vertices_zyx[used]
    marching_normals_zyx = marching_normals_zyx[used]
    vertex_weight = vertex_weight[used]
    vertex_views = vertex_views[used]
    vertices_xyz = vertices_zyx[:, ::-1] + volume.origin_xyz + 0.5 * volume.voxel_size
    # Marching Cubes' returned normal convention is not the TSDF sign
    # contract.  Orient from the sampled scalar gradient explicitly so normals
    # always point toward positive TSDF (observed free space).
    gradients_zyx = np.stack(
        np.gradient(values, volume.voxel_size, edge_order=1), axis=-1
    )
    gradient_zyx = _sample_grid_nearest(gradients_zyx, grid_vertices_zyx[used])
    gradient_lengths = np.linalg.norm(gradient_zyx, axis=1)
    zero_gradient = gradient_lengths <= 1e-12
    gradient_zyx[zero_gradient] = marching_normals_zyx[zero_gradient]
    normals_xyz = gradient_zyx[:, ::-1]
    lengths = np.linalg.norm(normals_xyz, axis=1)
    normals_xyz = normals_xyz / np.maximum(lengths[:, None], 1e-12)
    return SurfaceMesh(
        vertices_xyz=vertices_xyz,
        faces=faces,
        normals_xyz=normals_xyz,
        vertex_weight=vertex_weight,
        vertex_view_count=vertex_views,
        units=volume.units,
        frame=volume.frame,
        provenance={
            "method": "dense-tsdf-marching-cubes",
            "voxel_size": volume.voxel_size,
            "truncation_distance": volume.truncation_distance,
            "normal_orientation": "toward-positive-tsdf",
        },
    )


def _rgb_image(image: Any) -> np.ndarray:
    pixels = np.asarray(image.image)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=-1)
    elif pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    elif pixels.shape[-1] >= 3:
        pixels = pixels[..., :3]
    else:
        raise ValueError("colorization requires grayscale or RGB-like ERP images")
    if np.issubdtype(pixels.dtype, np.integer):
        pixels = pixels.astype(np.float64) / np.iinfo(pixels.dtype).max
    else:
        pixels = pixels.astype(np.float64)
    return pixels


def colorize_surface(
    surface: SurfaceMesh,
    range_set: MultiViewRangeSet,
    *,
    truncation_distance: float,
) -> SurfaceMesh:
    """Median-fuse range-consistent colors while preserving valid black."""

    count = len(surface.vertices_xyz)
    contributions: list[list[np.ndarray]] = [[] for _ in range(count)]
    for range_map in range_set.maps:
        view = range_set.scene.view(range_map.reference_id)
        camera_points = (
            surface.vertices_xyz - range_map.pose.center
        ) @ range_map.pose.R.T
        projection = rays_to_erp_pixels(camera_points, range_map.radial_range.shape)
        range_validity = range_map.validity & range_map.sampling_support
        sampled_range, accepted = _bilinear_sample(
            range_map.radial_range,
            projection.pixels_xy,
            validity=range_validity,
        )
        accepted &= projection.valid
        accepted &= np.abs(sampled_range - projection.ranges) <= truncation_distance
        image_validity = view.effective_validity()
        colors, color_valid = _bilinear_sample(
            _rgb_image(view.image),
            projection.pixels_xy,
            validity=image_validity,
        )
        accepted &= color_valid
        for index in np.flatnonzero(accepted):
            contributions[int(index)].append(colors[index])
    colors = np.zeros((count, 3), dtype=np.float32)
    validity = np.zeros(count, dtype=bool)
    contribution_count = np.zeros(count, dtype=np.uint16)
    for index, items in enumerate(contributions):
        if not items:
            continue
        colors[index] = np.median(np.stack(items), axis=0)
        validity[index] = True
        contribution_count[index] = len(items)
    return replace(
        surface,
        colors_rgb=colors,
        color_validity=validity,
        color_contribution_count=contribution_count,
        provenance={
            **dict(surface.provenance),
            "color_fusion": "per-channel-median",
            "color_range_tolerance": truncation_distance,
        },
    )


@dataclass(frozen=True, slots=True)
class BuildRegisteredReferenceSets:
    """Create one deterministic registered MVS set per reference view."""

    reference_ids: tuple[str, ...] | None = None
    input_kind = "registered-spherical-scene"
    output_kind = "dense-reconstruction-workspace"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, RegisteredSphericalScene):
            raise TypeError(
                "BuildRegisteredReferenceSets requires a RegisteredSphericalScene"
            )
        ids = (
            tuple(sorted(value.panorama_ids))
            if self.reference_ids is None
            else tuple(self.reference_ids)
        )
        if len(ids) != len(set(ids)):
            raise ValueError("reference_ids must be unique")
        sets = tuple(value.registered_set(reference_id) for reference_id in ids)
        registered = RegisteredReferenceSets(scene=value, sets=sets)
        workspace = DenseReconstructionWorkspace(scene=value, reference_sets=registered)
        return StageResult(
            workspace,
            {
                "domain": "registered-spherical-scene",
                "reference_ids": list(ids),
                "source_policy": "all-other-views-sorted-by-id",
            },
        )


@dataclass(frozen=True, slots=True)
class EstimateMultiViewRanges:
    """Run the existing full-sphere MVS engine for every reference."""

    options: FullSphereStereoOptions = field(default_factory=FullSphereStereoOptions)
    input_kind = "dense-reconstruction-workspace"
    output_kind = "dense-reconstruction-workspace"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, DenseReconstructionWorkspace):
            raise TypeError(
                "EstimateMultiViewRanges requires a reconstruction workspace"
            )
        if value.reference_sets is None:
            raise ValueError("registered reference sets have not been built")
        engine = FullSphereMultiViewStereo(self.options)
        maps = tuple(
            MultiViewRangeMap.from_stereo_result(
                reference_id=item.reference_id,
                pose=value.scene.view(item.reference_id).pose,
                result=engine.estimate(item),
            )
            for item in value.reference_sets.sets
        )
        ranges = MultiViewRangeSet(
            scene=value.scene,
            maps=maps,
            diagnostics={
                "mvs": engine.describe(),
                "reference_count": len(maps),
            },
        )
        return StageResult(
            replace(value, ranges=ranges),
            {
                "domain": "multireference-mvs",
                "reference_count": len(maps),
                "confidence_semantics": "within-reference-ranking-only",
            },
        )


@dataclass(frozen=True, slots=True)
class FuseTSDF:
    """Fuse a range set (or workspace range set) into a dense TSDF."""

    spec: TSDFSpec
    input_kind = ("dense-reconstruction-workspace", "multi-view-range-set")
    output_kind = ("dense-reconstruction-workspace", "tsdf-volume")
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        ranges = (
            value.ranges if isinstance(value, DenseReconstructionWorkspace) else value
        )
        if not isinstance(ranges, MultiViewRangeSet):
            raise TypeError(
                "FuseTSDF requires a MultiViewRangeSet or populated workspace"
            )
        try:
            volume = fuse_tsdf(ranges, self.spec)
        except ValueError as exc:
            if not isinstance(value, DenseReconstructionWorkspace) or not str(
                exc
            ).startswith("cannot derive TSDF bounds"):
                raise
            rejected = replace(
                value,
                rejection_reasons=(
                    *value.rejection_reasons,
                    "no-valid-range-observations",
                ),
            )
            return StageResult(
                rejected,
                {
                    "domain": "world-tsdf",
                    "status": "rejected",
                    "reason": "no-valid-range-observations",
                },
            )
        output = (
            replace(value, volume=volume)
            if isinstance(value, DenseReconstructionWorkspace)
            else volume
        )
        return StageResult(
            output,
            {
                "domain": "world-tsdf",
                "voxel_size": self.spec.voxel_size,
                "truncation_distance": self.spec.truncation_distance,
                "weighting": self.spec.weighting,
            },
        )


@dataclass(frozen=True, slots=True)
class ExtractSurface:
    """Extract a world-frame mesh from a sufficiently observed TSDF."""

    input_kind = ("dense-reconstruction-workspace", "tsdf-volume")
    output_kind = ("dense-reconstruction-workspace", "surface-mesh")
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        volume = (
            value.volume if isinstance(value, DenseReconstructionWorkspace) else value
        )
        if isinstance(value, DenseReconstructionWorkspace) and volume is None:
            return StageResult(
                value,
                {
                    "domain": "world-surface",
                    "status": "skipped",
                    "reason": "volume-unavailable",
                },
            )
        if not isinstance(volume, TSDFVolume):
            raise TypeError(
                "ExtractSurface requires a TSDFVolume or populated workspace"
            )
        try:
            surface = extract_surface(volume)
        except ValueError as exc:
            if not isinstance(value, DenseReconstructionWorkspace):
                raise
            rejected = replace(
                value,
                rejection_reasons=(*value.rejection_reasons, str(exc)),
            )
            return StageResult(
                rejected,
                {
                    "domain": "world-surface",
                    "status": "rejected",
                    "reason": str(exc),
                },
            )
        output = (
            replace(value, surface=surface)
            if isinstance(value, DenseReconstructionWorkspace)
            else surface
        )
        return StageResult(
            output,
            {
                "domain": "world-surface",
                "vertex_count": len(surface.vertices_xyz),
                "face_count": len(surface.faces),
            },
        )


@dataclass(frozen=True, slots=True)
class ColorizeSurface:
    """Fuse range-consistent ERP colors into surface vertices."""

    enabled: bool = True
    input_kind = "dense-reconstruction-workspace"
    output_kind = "dense-reconstruction-workspace"
    stability = "experimental"

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, DenseReconstructionWorkspace):
            raise TypeError("ColorizeSurface requires a reconstruction workspace")
        if value.surface is None and value.rejection_reasons:
            return StageResult(
                value,
                {
                    "domain": "world-surface-color",
                    "enabled": self.enabled,
                    "status": "skipped-after-rejection",
                },
            )
        if value.surface is None or value.ranges is None or value.volume is None:
            raise ValueError(
                "surface, ranges, and volume are required before colorization"
            )
        surface = (
            colorize_surface(
                value.surface,
                value.ranges,
                truncation_distance=value.volume.truncation_distance,
            )
            if self.enabled
            else value.surface
        )
        return StageResult(
            replace(value, surface=surface),
            {
                "domain": "world-surface-color",
                "enabled": self.enabled,
                "colored_vertex_count": (
                    0
                    if surface.color_validity is None
                    else int(np.count_nonzero(surface.color_validity))
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class ValidateDenseReconstruction:
    """Turn scientific acceptance or rejection into an auditable result."""

    retain_intermediates: str = "ranges"
    input_kind = "dense-reconstruction-workspace"
    output_kind = "spherical-dense-reconstruction-result"
    stability = "experimental"

    def __post_init__(self) -> None:
        if self.retain_intermediates not in {"none", "ranges", "all"}:
            raise ValueError("retain_intermediates must be 'none', 'ranges', or 'all'")

    def execute(self, value: Any) -> StageResult:
        if not isinstance(value, DenseReconstructionWorkspace):
            raise TypeError("ValidateDenseReconstruction requires a workspace")
        reasons = list(value.rejection_reasons)
        if value.surface is None:
            reasons.append("surface-unavailable")
        elif len(value.surface.faces) == 0:
            reasons.append("surface-has-no-faces")
        status = "rejected" if reasons else "accepted"
        ranges = (
            value.ranges if self.retain_intermediates in {"ranges", "all"} else None
        )
        volume = value.volume if self.retain_intermediates == "all" else None
        result = SphericalDenseReconstructionResult(
            status=status,
            reasons=tuple(reasons),
            diagnostics={
                **dict(value.diagnostics),
                "retention": self.retain_intermediates,
                "profile_scope": "registered-metric-poses",
            },
            surface=value.surface,
            ranges=ranges,
            volume=volume,
        )
        return StageResult(
            result,
            {
                "domain": "dense-reconstruction-validation",
                "status": status,
                "reasons": list(reasons),
            },
        )


__all__ = [
    "BuildRegisteredReferenceSets",
    "ColorizeSurface",
    "DenseReconstructionWorkspace",
    "EstimateMultiViewRanges",
    "ExtractSurface",
    "FuseTSDF",
    "RegisteredReferenceSets",
    "ValidateDenseReconstruction",
    "colorize_surface",
    "extract_surface",
    "fuse_tsdf",
]
