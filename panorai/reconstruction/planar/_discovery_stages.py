"""Public stages for deterministic spherical planar-orientation discovery."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
)

from ._discovery_models import (
    PlanarDiscoveryRequest,
    PlanarOrientationProposal,
    SphericalLineObservation,
    SphericalPlanarDiscoveryOptions,
    SphericalPlanarDiscoveryResult,
    VanishingBundle,
    VanishingLineSuggestion,
)
from ._models import PlaneEstimate


def _out(value: Any, **details: Any):
    from panorai.pipeline import StageResult

    return StageResult(value, details)


class _PlanarDiscoveryStage:
    """Bridge discovery stages to the public ``Stage.execute`` contract."""

    def execute(self, value: Any):
        return self._run(value)


def _unit(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("cannot normalize a degenerate vector")
    return np.asarray(value, dtype=np.float64) / length


def _canonical(value: np.ndarray) -> np.ndarray:
    result = _unit(value)
    index = int(np.argmax(np.abs(result)))
    return -result if result[index] < 0 else result


def _angle_deg(
    first: np.ndarray, second: np.ndarray, *, antipodal: bool = False
) -> float:
    dot = float(np.dot(first, second))
    if antipodal:
        dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _stable_id(prefix: str, *values: Any) -> str:
    digest = sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(np.round(value.astype(np.float64), 6).tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return f"{prefix}-{digest.hexdigest()[:12]}"


@dataclass(frozen=True)
class _AtlasChart:
    chart_id: str
    spec: GnomonicSpec
    rgb: np.ndarray
    domain: np.ndarray


@dataclass(frozen=True)
class _RawLine:
    chart_id: str
    erp_endpoints_xy: np.ndarray
    rays: np.ndarray
    normal: np.ndarray
    midpoint: np.ndarray
    angular_length_deg: float
    support_pixels: int
    residual_px: float
    edge_support: float
    score: float


@dataclass(frozen=True)
class _DiscoveryWorkspace:
    request: PlanarDiscoveryRequest
    options: SphericalPlanarDiscoveryOptions
    charts: tuple[_AtlasChart, ...] = ()
    raw_lines: tuple[_RawLine, ...] = ()
    lines: tuple[SphericalLineObservation, ...] = ()
    bundles: tuple[VanishingBundle, ...] = ()
    proposals: tuple[PlanarOrientationProposal, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidatePlanarDiscoveryRequest(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = "planar-discovery-request"
    output_kind = "planar-discovery-request"

    def _run(self, value: Any):
        if not isinstance(value, PlanarDiscoveryRequest):
            raise TypeError("expected PlanarDiscoveryRequest")
        image = np.asarray(value.panorama.image)
        return _out(
            value,
            shape_hw=list(image.shape[:2]),
            revision=value.revision,
            reviewed_lines=len(value.line_decisions),
            reviewed_bundles=len(value.bundle_memberships),
            reviewed_planes=len(value.plane_bundle_groups),
        )


def _atlas_centres() -> tuple[tuple[float, float], ...]:
    equator = tuple((0.0, float(lon)) for lon in range(-180, 180, 60))
    upper = tuple((50.0, float(lon)) for lon in (-135, -45, 45, 135))
    lower = tuple((-50.0, float(lon)) for lon in (-135, -45, 45, 135))
    return equator + upper + lower


@dataclass(frozen=True)
class BuildOverlappingGnomonicAtlas(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = "planar-discovery-request"
    output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, PlanarDiscoveryRequest):
            raise TypeError("expected PlanarDiscoveryRequest")
        image = np.asarray(value.panorama.image)
        support = (
            np.ones(image.shape[:2], dtype=bool)
            if value.panorama.support_mask is None
            else np.asarray(value.panorama.support_mask, dtype=bool)
        )
        domain = support & np.asarray(value.panorama.validity("image"), dtype=bool)
        charts: list[_AtlasChart] = []
        size = self.options.atlas_face_size
        for index, (latitude, longitude) in enumerate(_atlas_centres()):
            spec = GnomonicSpec(
                center_lat_deg=latitude,
                center_lon_deg=longitude,
                hfov_deg=self.options.atlas_hfov_deg,
                vfov_deg=self.options.atlas_hfov_deg,
                output_shape_hw=(size, size),
            )
            projected = equirectangular_to_gnomonic(
                image.astype(np.float32),
                spec,
                interpolation="bilinear",
                invalid_policy="renormalize",
                validity_mask=domain,
                min_valid_weight=0.999,
            )
            projected_domain = equirectangular_to_gnomonic(
                domain,
                spec,
                interpolation="nearest",
            ).data.astype(bool)
            if projected.validity_mask is not None:
                projected_domain &= np.asarray(projected.validity_mask, dtype=bool)
            rgb = np.asarray(projected.data).clip(0, 255).astype(np.uint8)
            rgb[~projected_domain] = 0
            charts.append(
                _AtlasChart(
                    f"chart-{index:02d}",
                    spec,
                    rgb,
                    projected_domain,
                )
            )
        workspace = _DiscoveryWorkspace(value, self.options, charts=tuple(charts))
        return _out(
            workspace,
            charts=len(charts),
            face_size=size,
            hfov_deg=self.options.atlas_hfov_deg,
            projection="panorai.geometry.equirectangular_to_gnomonic",
        )


def _line_candidates(chart: _AtlasChart, options: SphericalPlanarDiscoveryOptions):
    import cv2

    gray = cv2.cvtColor(chart.rgb, cv2.COLOR_RGB2GRAY)
    if options.local_contrast_clip_limit > 0:
        gray = cv2.createCLAHE(
            clipLimit=options.local_contrast_clip_limit,
            tileGridSize=(8, 8),
        ).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(
        gray,
        options.canny_low,
        options.canny_high,
        L2gradient=True,
    )
    edges[~chart.domain] = 0
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(gray)[0]
    edge_points_yx = np.argwhere(edges > 0)
    if found is None:
        return (), edge_points_yx
    results: list[tuple[np.ndarray, int, float, float]] = []
    for raw in np.asarray(found).reshape(-1, 4):
        line = raw.reshape(2, 2).astype(np.float64)
        direction = line[1] - line[0]
        length = float(np.linalg.norm(direction))
        if length < options.minimum_segment_px:
            continue
        count = max(8, int(np.ceil(length)))
        samples = np.linspace(line[0], line[1], count)
        xy = np.rint(samples).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, chart.rgb.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, chart.rgb.shape[0] - 1)
        if float(chart.domain[xy[:, 1], xy[:, 0]].mean()) < 0.9:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        low = np.floor(line.min(axis=0) - 5).astype(int)
        high = np.ceil(line.max(axis=0) + 5).astype(int)
        inside = edge_points_yx[
            (edge_points_yx[:, 1] >= low[0])
            & (edge_points_yx[:, 1] <= high[0])
            & (edge_points_yx[:, 0] >= low[1])
            & (edge_points_yx[:, 0] <= high[1])
        ]
        if not len(inside):
            continue
        points = inside[:, ::-1].astype(np.float64)
        relative = points - line[0]
        along = relative @ direction
        across = np.abs(relative @ normal)
        points = points[(along >= -3) & (along <= length + 3) & (across <= 3)]
        edge_support = min(1.0, float(len(points) / max(length, 1.0)))
        if len(points) < 8 or edge_support < options.minimum_edge_support:
            continue
        centre = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
        fitted_direction = vh[0]
        if fitted_direction @ direction < 0:
            fitted_direction = -fitted_direction
        fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
        fitted = np.stack(
            [
                point - fitted_normal * ((point - centre) @ fitted_normal)
                for point in line
            ]
        )
        residual = float(np.median(np.abs((points - centre) @ fitted_normal)))
        results.append((fitted, int(len(points)), residual, edge_support))
    results.sort(key=lambda item: (-np.linalg.norm(item[0][1] - item[0][0]), item[2]))
    return tuple(results[:160]), edge_points_yx


@dataclass(frozen=True)
class DetectAndFitGnomonicLines(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        raw_lines: list[_RawLine] = []
        edge_pixels = 0
        edge_preview: list[np.ndarray] = []
        shape = np.asarray(value.request.panorama.image).shape[:2]
        for chart in value.charts:
            candidates, chart_edges_yx = _line_candidates(chart, self.options)
            edge_pixels += len(chart_edges_yx)
            if len(chart_edges_yx):
                stride = max(1, int(np.ceil(len(chart_edges_yx) / 1500)))
                chart_edges_xy = chart_edges_yx[::stride, ::-1].astype(np.float64)
                lifted = gnomonic_pixels_to_rays(chart_edges_xy, chart.spec)
                valid = np.asarray(lifted.valid, dtype=bool)
                if valid.any():
                    edge_preview.append(
                        np.asarray(
                            rays_to_erp_pixels(
                                np.asarray(lifted.rays_xyz)[valid], shape
                            ).pixels_xy,
                            dtype=np.float64,
                        )
                    )
            for chart_line, support_pixels, residual, edge_support in candidates:
                projection = gnomonic_pixels_to_rays(chart_line, chart.spec)
                if not np.asarray(projection.valid).all():
                    continue
                rays = np.asarray(projection.rays_xyz, dtype=np.float64)
                normal = np.cross(rays[0], rays[1])
                if np.linalg.norm(normal) < 1e-8:
                    continue
                normal = _canonical(normal)
                angular = _angle_deg(rays[0], rays[1])
                if angular < self.options.minimum_line_length_deg:
                    continue
                midpoint = _unit(rays[0] + rays[1])
                erp = rays_to_erp_pixels(rays, shape)
                if not np.asarray(erp.valid).all():
                    continue
                score = angular * edge_support / (1.0 + residual)
                raw_lines.append(
                    _RawLine(
                        chart.chart_id,
                        np.asarray(erp.pixels_xy, dtype=np.float64),
                        rays,
                        normal,
                        midpoint,
                        angular,
                        support_pixels,
                        residual,
                        edge_support,
                        score,
                    )
                )
        if edge_preview:
            preview = np.unique(
                np.rint(np.concatenate(edge_preview, axis=0) * 2.0) / 2.0,
                axis=0,
            )
            if len(preview) > 20000:
                preview = preview[:: int(np.ceil(len(preview) / 20000))][:20000]
        else:
            preview = np.empty((0, 2), dtype=np.float64)
        result = _DiscoveryWorkspace(
            value.request,
            value.options,
            value.charts,
            tuple(raw_lines),
            diagnostics={
                **value.diagnostics,
                "edge_pixels_in_atlas": edge_pixels,
                "local_contrast_clip_limit": self.options.local_contrast_clip_limit,
                "_edge_preview_erp_xy": preview,
            },
        )
        return _out(
            result,
            edge_pixels=edge_pixels,
            edge_preview_points=len(preview),
            chart_line_observations=len(raw_lines),
            fit_domain="gnomonic",
            detector="CLAHE + Canny + OpenCV LSD + local TLS"
            if self.options.local_contrast_clip_limit > 0
            else "Canny + OpenCV LSD + local TLS",
            local_contrast_clip_limit=self.options.local_contrast_clip_limit,
        )


@dataclass(frozen=True)
class MergeSphericalLineObservations(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        groups: list[list[_RawLine]] = []
        for candidate in sorted(value.raw_lines, key=lambda item: -item.score):
            matched = None
            for index, group in enumerate(groups):
                representative = group[0]
                if (
                    _angle_deg(candidate.normal, representative.normal, antipodal=True)
                    <= self.options.merge_normal_deg
                    and _angle_deg(candidate.midpoint, representative.midpoint)
                    <= self.options.merge_midpoint_deg
                ):
                    matched = index
                    break
            if matched is None:
                groups.append([candidate])
            else:
                groups[matched].append(candidate)
        lines: list[SphericalLineObservation] = []
        for group in groups:
            representative = max(group, key=lambda item: item.score)
            line_id = _stable_id(
                "line",
                _canonical(representative.normal),
                representative.midpoint,
            )
            state = value.request.line_decisions.get(line_id, "detected")
            lines.append(
                SphericalLineObservation(
                    line_id,
                    representative.erp_endpoints_xy,
                    representative.normal,
                    representative.angular_length_deg,
                    sum(item.support_pixels for item in group),
                    min(item.residual_px for item in group),
                    max(item.edge_support for item in group),
                    max(item.score for item in group),
                    tuple(sorted({item.chart_id for item in group})),
                    state,
                )
            )
        lines.sort(key=lambda item: (-item.score, item.line_id))
        automatic_total = len(lines)
        # The automatic budget must never evict explicit human evidence.
        # Manual seeds are appended after this bounded automatic selection.
        lines = lines[: self.options.maximum_lines]
        shape = np.asarray(value.request.panorama.image).shape[:2]
        known_ids = {line.line_id for line in lines}
        for seed in value.request.manual_line_seeds:
            if seed.line_id in known_ids:
                raise ValueError(
                    f"manual line seed {seed.line_id!r} duplicates a detected line"
                )
            rays = np.asarray(
                erp_pixels_to_rays(seed.erp_endpoints_xy, shape), dtype=np.float64
            )
            normal = np.cross(rays[0], rays[1])
            if np.linalg.norm(normal) < 1e-8:
                raise ValueError(f"manual line seed {seed.line_id!r} is degenerate")
            angular = _angle_deg(rays[0], rays[1])
            if angular < self.options.minimum_line_length_deg:
                raise ValueError(f"manual line seed {seed.line_id!r} is too short")
            lines.append(
                SphericalLineObservation(
                    seed.line_id,
                    seed.erp_endpoints_xy,
                    _canonical(normal),
                    angular,
                    2,
                    0.0,
                    1.0,
                    angular,
                    ("manual",),
                    seed.state,
                    "human",
                )
            )
        support = (
            np.ones(shape, dtype=bool)
            if value.request.panorama.support_mask is None
            else np.asarray(value.request.panorama.support_mask, dtype=bool)
        )
        support &= np.asarray(value.request.panorama.validity("image"), dtype=bool)
        associated: list[SphericalLineObservation] = []
        for line in lines:
            rays = np.asarray(
                erp_pixels_to_rays(line.erp_endpoints_xy, shape), dtype=np.float64
            )
            count = max(8, int(np.ceil(line.angular_length_deg * 2.0)))
            weights = np.linspace(0.0, 1.0, count)[:, None]
            samples = (1.0 - weights) * rays[0] + weights * rays[1]
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
            pixels = np.asarray(rays_to_erp_pixels(samples, shape).pixels_xy)
            x = np.mod(np.rint(pixels[:, 0]).astype(int), shape[1])
            y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, shape[0] - 1)
            valid = support[y, x]
            region_support = {
                region.plane_id: float(
                    np.mean(np.asarray(region.mask, dtype=bool)[y[valid], x[valid]])
                )
                if valid.any()
                else 0.0
                for region in value.request.regions
            }
            associated.append(replace(line, region_support=region_support))
        lines = associated
        total = automatic_total + len(value.request.manual_line_seeds)
        result = _DiscoveryWorkspace(
            value.request,
            value.options,
            value.charts,
            value.raw_lines,
            tuple(lines),
            diagnostics={
                **value.diagnostics,
                "deduplicated_lines_total": total,
                "deduplicated_lines_retained": len(lines),
            },
        )
        return _out(
            result,
            lines_total=total,
            lines_retained=len(lines),
            rejected=sum(line.state == "rejected" for line in lines),
            coordinate_grid="original-erp-pixel-centres",
        )


def _bundle_from_lines(
    bundle_id: str,
    lines: list[SphericalLineObservation],
    *,
    source: str,
    options: SphericalPlanarDiscoveryOptions,
) -> VanishingBundle | None:
    if len(lines) < 2:
        return None
    normals = np.stack([line.great_circle_normal_camera for line in lines])
    _, _, vh = np.linalg.svd(normals, full_matrices=True)
    direction = _canonical(vh[-1])
    residuals = np.degrees(np.arcsin(np.clip(np.abs(normals @ direction), 0.0, 1.0)))
    threshold = options.vanishing_residual_deg
    keep = residuals <= threshold
    minimum = 2 if source == "human" else options.minimum_bundle_lines
    if int(keep.sum()) < minimum:
        return None
    kept = [line for line, accepted in zip(lines, keep, strict=True) if accepted]
    residual = float(np.median(residuals[keep]))
    support = float(sum(line.angular_length_deg for line in kept))
    if source == "automatic" and support < options.minimum_bundle_support_deg:
        return None
    confidence = float(
        np.clip(
            (1.0 - residual / max(threshold, 1e-9))
            * min(1.0, support / (options.minimum_bundle_support_deg * 2.0)),
            0.0,
            1.0,
        )
    )
    return VanishingBundle(
        bundle_id,
        direction,
        tuple(sorted(line.line_id for line in kept)),
        residual,
        support,
        confidence,
        "accepted" if source == "human" else "draft",
        source,
    )


@dataclass(frozen=True)
class EstimateVanishingBundles(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        line_by_id = {line.line_id: line for line in value.lines}
        active = [line for line in value.lines if line.state != "rejected"]
        bundles: list[VanishingBundle] = []
        human_line_ids: set[str] = set()
        for bundle_id, line_ids in sorted(value.request.bundle_memberships.items()):
            unknown = set(line_ids) - set(line_by_id)
            if unknown:
                raise ValueError(
                    f"bundle {bundle_id!r} references unknown lines {sorted(unknown)}"
                )
            selected = [
                line_by_id[line_id]
                for line_id in line_ids
                if line_by_id[line_id].state != "rejected"
            ]
            bundle = _bundle_from_lines(
                bundle_id,
                selected,
                source="human",
                options=self.options,
            )
            if bundle is None:
                raise ValueError(f"bundle {bundle_id!r} is geometrically degenerate")
            bundles.append(bundle)
            human_line_ids.update(bundle.line_ids)
        automatic_lines = [
            line for line in active if line.line_id not in human_line_ids
        ]
        automatic_lines = automatic_lines[: min(len(automatic_lines), 220)]
        candidates: dict[tuple[int, int, int], tuple[float, np.ndarray]] = {}
        bin_size = max(np.sin(np.deg2rad(self.options.vanishing_residual_deg)), 1e-3)
        intersections = 0
        for left_index, left in enumerate(automatic_lines):
            for right in automatic_lines[left_index + 1 :]:
                direction = np.cross(
                    left.great_circle_normal_camera,
                    right.great_circle_normal_camera,
                )
                if np.linalg.norm(direction) < 1e-5:
                    continue
                direction = _canonical(direction)
                key = tuple(np.rint(direction / bin_size).astype(int).tolist())
                weight = min(left.angular_length_deg, right.angular_length_deg)
                previous = candidates.get(key)
                candidates[key] = (
                    weight + (0.0 if previous is None else previous[0]),
                    direction,
                )
                intersections += 1
                if intersections >= self.options.maximum_intersections:
                    break
            if intersections >= self.options.maximum_intersections:
                break
        assigned: set[str] = set()
        for _, candidate in sorted(candidates.values(), key=lambda item: -item[0]):
            if len(bundles) >= self.options.maximum_bundles:
                break
            if any(
                _angle_deg(candidate, bundle.direction_camera, antipodal=True)
                <= self.options.vanishing_residual_deg * 1.5
                for bundle in bundles
            ):
                continue
            supporting = []
            for line in automatic_lines:
                if line.line_id in assigned:
                    continue
                residual = np.degrees(
                    np.arcsin(
                        np.clip(
                            abs(float(line.great_circle_normal_camera @ candidate)),
                            0.0,
                            1.0,
                        )
                    )
                )
                if residual <= self.options.vanishing_residual_deg:
                    supporting.append(line)
            bundle = _bundle_from_lines(
                "pending",
                supporting,
                source="automatic",
                options=self.options,
            )
            if bundle is None:
                continue
            bundle_id = _stable_id("bundle", bundle.direction_camera)
            bundle = VanishingBundle(
                bundle_id,
                bundle.direction_camera,
                bundle.line_ids,
                bundle.residual_deg,
                bundle.support_deg,
                bundle.confidence,
                bundle.state,
                bundle.source,
            )
            bundles.append(bundle)
            assigned.update(bundle.line_ids)
        result = _DiscoveryWorkspace(
            value.request,
            value.options,
            value.charts,
            value.raw_lines,
            value.lines,
            tuple(bundles),
            diagnostics={
                **value.diagnostics,
                "vanishing_intersections_evaluated": intersections,
                "unbundled_lines": len(active)
                - len({line_id for bundle in bundles for line_id in bundle.line_ids}),
            },
        )
        return _out(
            result,
            bundles=len(bundles),
            human_bundles=sum(bundle.source == "human" for bundle in bundles),
            intersections_evaluated=intersections,
            projective_antipodal=True,
        )


@dataclass(frozen=True)
class SuggestVanishingBundleLines(_PlanarDiscoveryStage):
    """Rank non-member lines against each reviewed projective direction."""

    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        active = [line for line in value.lines if line.state != "rejected"]
        assigned = {
            line_id: bundle.bundle_id
            for bundle in value.bundles
            for line_id in bundle.line_ids
        }
        bundles: list[VanishingBundle] = []
        total = 0
        threshold = self.options.suggestion_residual_deg
        for bundle in value.bundles:
            decisions = value.request.bundle_line_decisions.get(bundle.bundle_id, {})
            candidates: list[VanishingLineSuggestion] = []
            members = set(bundle.line_ids)
            for line in active:
                if line.line_id in members:
                    continue
                residual = float(
                    np.degrees(
                        np.arcsin(
                            np.clip(
                                abs(
                                    float(
                                        line.great_circle_normal_camera
                                        @ bundle.direction_camera
                                    )
                                ),
                                0.0,
                                1.0,
                            )
                        )
                    )
                )
                if residual > threshold:
                    continue
                confidence = float(
                    np.clip(
                        (1.0 - residual / max(threshold, 1e-9))
                        * np.sqrt(line.edge_support),
                        0.0,
                        1.0,
                    )
                )
                candidates.append(
                    VanishingLineSuggestion(
                        line.line_id,
                        residual,
                        confidence,
                        decisions.get(line.line_id, "suggested"),
                        assigned.get(line.line_id),
                    )
                )
            candidates.sort(
                key=lambda item: (
                    item.state == "rejected",
                    item.residual_deg,
                    -item.confidence,
                    item.line_id,
                )
            )
            reviewed = [item for item in candidates if item.state == "rejected"]
            pending = [item for item in candidates if item.state == "suggested"]
            candidates = [
                *pending[: self.options.maximum_suggestions_per_bundle],
                *reviewed,
            ]
            total += len(candidates)
            bundles.append(replace(bundle, suggestions=tuple(candidates)))
        result = replace(
            value,
            bundles=tuple(bundles),
            diagnostics={
                **value.diagnostics,
                "vanishing_line_suggestions": total,
                "suggestion_residual_deg": threshold,
            },
        )
        return _out(
            result,
            suggestions=total,
            residual_threshold_deg=threshold,
            automatic_acceptance=False,
            projective_antipodal=True,
        )


def _plane_proposal(
    proposal_id: str,
    bundles: list[VanishingBundle],
    line_by_id: dict[str, SphericalLineObservation],
    *,
    source: str,
    plane_id: str | None = None,
) -> PlanarOrientationProposal | None:
    directions = np.stack([bundle.direction_camera for bundle in bundles])
    _, singular, vh = np.linalg.svd(directions, full_matrices=True)
    if len(singular) < 2 or singular[1] < 0.12:
        return None
    normal = _canonical(vh[-1])
    residuals = np.abs(directions @ normal)
    line_ids = tuple(
        sorted({line_id for bundle in bundles for line_id in bundle.line_ids})
    )
    points = np.concatenate(
        [line_by_id[line_id].erp_endpoints_xy for line_id in line_ids]
    )
    separation = min(
        _angle_deg(left.direction_camera, right.direction_camera, antipodal=True)
        for index, left in enumerate(bundles)
        for right in bundles[index + 1 :]
    )
    confidence = float(
        np.clip(
            np.mean([bundle.confidence for bundle in bundles])
            * min(1.0, np.sin(np.deg2rad(separation)) / 0.5)
            * (1.0 - min(1.0, float(np.mean(residuals)) * 5.0)),
            0.0,
            1.0,
        )
    )
    state = "accepted" if source == "human" else "draft"
    estimate = PlaneEstimate(
        plane_id=proposal_id,
        normal_world=None,
        offset_m=None,
        covariance=None,
        tangent_basis=None,
        state=state,
        diagnostics=MappingProxyType(
            {
                "kind": "orientation-only",
                "normal_frame": "camera",
                "metric_offset": "unavailable",
                "bundle_residual": float(np.mean(residuals)),
                "minimum_direction_separation_deg": separation,
            }
        ),
    )
    return PlanarOrientationProposal(
        proposal_id,
        estimate,
        normal,
        tuple(bundle.bundle_id for bundle in bundles),
        line_ids,
        points,
        confidence,
        state,
        None,
        source,
        plane_id,
    )


@dataclass(frozen=True)
class ProposePlanarOrientations(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = output_kind = "planar-discovery-workspace"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        line_by_id = {line.line_id: line for line in value.lines}
        bundle_by_id = {bundle.bundle_id: bundle for bundle in value.bundles}
        proposals: list[PlanarOrientationProposal] = []
        used_groups: set[tuple[str, ...]] = set()
        for proposal_id, bundle_ids in sorted(
            value.request.plane_bundle_groups.items()
        ):
            unknown = set(bundle_ids) - set(bundle_by_id)
            if unknown:
                raise ValueError(
                    f"plane {proposal_id!r} references unknown bundles {sorted(unknown)}"
                )
            group = tuple(sorted(bundle_ids))
            proposal = _plane_proposal(
                proposal_id,
                [bundle_by_id[bundle_id] for bundle_id in bundle_ids],
                line_by_id,
                source="human",
            )
            if proposal is None:
                raise ValueError(
                    f"plane {proposal_id!r} uses degenerate direction bundles"
                )
            proposals.append(proposal)
            used_groups.add(group)
        for plane_id, selected_ids in sorted(value.request.plane_line_groups.items()):
            unknown = set(selected_ids) - set(line_by_id)
            if unknown:
                raise ValueError(
                    f"plane {plane_id!r} references unknown lines {sorted(unknown)}"
                )
            selected = set(selected_ids)
            trimmed: list[VanishingBundle] = []
            for bundle in value.bundles:
                members = tuple(
                    line_id for line_id in bundle.line_ids if line_id in selected
                )
                if len(members) >= 2:
                    trimmed.append(
                        replace(
                            bundle, line_ids=members, source="human", state="accepted"
                        )
                    )
            if len(trimmed) < 2:
                continue
            proposal = _plane_proposal(
                f"orientation-{plane_id}",
                trimmed,
                line_by_id,
                source="human",
                plane_id=plane_id,
            )
            if proposal is not None:
                proposals.append(proposal)
                used_groups.add(tuple(sorted(bundle.bundle_id for bundle in trimmed)))
        for region in value.request.regions:
            if region.plane_id in value.request.plane_line_groups:
                continue
            regional: list[VanishingBundle] = []
            for bundle in value.bundles:
                members = tuple(
                    line_id
                    for line_id in bundle.line_ids
                    if line_by_id[line_id].region_support.get(region.plane_id, 0.0)
                    >= 0.6
                    and line_by_id[line_id].angular_length_deg
                    * line_by_id[line_id].region_support.get(region.plane_id, 0.0)
                    >= 2.0
                )
                if len(members) >= 2:
                    regional.append(replace(bundle, line_ids=members))
            for index, left in enumerate(regional):
                for right in regional[index + 1 :]:
                    group = tuple(sorted((left.bundle_id, right.bundle_id)))
                    if group in used_groups:
                        continue
                    proposal = _plane_proposal(
                        _stable_id("region-plane", region.plane_id, *group),
                        [left, right],
                        line_by_id,
                        source="automatic",
                        plane_id=region.plane_id,
                    )
                    if proposal is not None:
                        proposals.append(proposal)
        automatic = [bundle for bundle in value.bundles if bundle.state != "rejected"]
        for index, left in enumerate(automatic):
            for right in automatic[index + 1 :]:
                group = tuple(sorted((left.bundle_id, right.bundle_id)))
                if group in used_groups:
                    continue
                left_charts = {
                    chart
                    for line_id in left.line_ids
                    for chart in line_by_id[line_id].chart_ids
                }
                right_charts = {
                    chart
                    for line_id in right.line_ids
                    for chart in line_by_id[line_id].chart_ids
                }
                if not left_charts & right_charts:
                    continue
                proposal_id = _stable_id("plane", *group)
                proposal = _plane_proposal(
                    proposal_id,
                    [left, right],
                    line_by_id,
                    source="automatic",
                )
                if proposal is not None:
                    proposals.append(proposal)
        proposals.sort(
            key=lambda item: (
                0 if item.source == "human" else 1,
                -item.confidence,
                item.proposal_id,
            )
        )
        total = len(proposals)
        proposals = proposals[: self.options.maximum_proposals]
        result = _DiscoveryWorkspace(
            value.request,
            value.options,
            value.charts,
            value.raw_lines,
            value.lines,
            value.bundles,
            tuple(proposals),
            diagnostics={
                **value.diagnostics,
                "planar_proposals_total": total,
                "planar_proposals_retained": len(proposals),
            },
        )
        return _out(
            result,
            proposals_total=total,
            proposals_retained=len(proposals),
            metric_offsets_published=0,
            review_required=True,
        )


@dataclass(frozen=True)
class AssemblePlanarDiscoveryResult(_PlanarDiscoveryStage):
    options: SphericalPlanarDiscoveryOptions = SphericalPlanarDiscoveryOptions()
    input_kind = "planar-discovery-workspace"
    output_kind = "spherical-planar-discovery-result"

    def _run(self, value: Any):
        if not isinstance(value, _DiscoveryWorkspace):
            raise TypeError("expected planar discovery workspace")
        reviewed = bool(
            value.request.bundle_memberships
            or value.request.plane_bundle_groups
            or value.request.plane_line_groups
            or value.request.manual_line_seeds
        )
        diagnostics = dict(value.diagnostics)
        edge_preview = np.asarray(
            diagnostics.pop("_edge_preview_erp_xy", np.empty((0, 2))),
            dtype=np.float64,
        )
        state = (
            "empty" if not value.lines else "reviewed" if reviewed else "needs-review"
        )
        result = SphericalPlanarDiscoveryResult(
            value.request,
            value.lines,
            value.bundles,
            value.proposals,
            edge_preview,
            diagnostics={
                **diagnostics,
                "atlas_charts": len(value.charts),
                "scientific_owner": "panorai.reconstruction.planar.SphericalPlanarDiscovery",
                "metric_geometry_published": False,
            },
            state=state,
        )
        return _out(
            result,
            state=state,
            lines=len(result.lines),
            bundles=len(result.bundles),
            proposals=len(result.proposals),
            metric_geometry_published=False,
        )


PLANAR_DISCOVERY_STAGE_TYPES = (
    ValidatePlanarDiscoveryRequest,
    BuildOverlappingGnomonicAtlas,
    DetectAndFitGnomonicLines,
    MergeSphericalLineObservations,
    EstimateVanishingBundles,
    SuggestVanishingBundleLines,
    ProposePlanarOrientations,
    AssemblePlanarDiscoveryResult,
)
