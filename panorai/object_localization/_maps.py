"""Adapters from semantic score maps to feature-indexed observations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from panorai.geometry import erp_pixels_to_rays

from ._models import SemanticRegionObservation


SEMANTIC_REGION_PROPOSALS_INTERFACE = "panorai-semantic-region-proposals/v1"


@dataclass(frozen=True, slots=True)
class SemanticRegionProposalConfig:
    """Topology and support gates for splitting one CAM into candidates."""

    threshold: float = 0.5
    connectivity: int = 8
    min_component_cells: int = 1
    min_feature_count: int = 1
    max_regions: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, Real):
            raise TypeError("threshold must be a real number")
        if not np.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in the interval (0, 1]")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        for name in ("min_component_cells", "min_feature_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_regions is not None and (
            isinstance(self.max_regions, bool)
            or not isinstance(self.max_regions, Integral)
            or self.max_regions < 1
        ):
            raise ValueError("max_regions must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class SemanticRegionProposalResult:
    """Retained feature-indexed regions plus component rejection counts."""

    regions: tuple[SemanticRegionObservation, ...]
    map_shape_hw: tuple[int, int]
    threshold: float
    connectivity: int
    raw_component_count: int
    retained_component_count: int
    rejected_small_component_count: int
    rejected_no_feature_count: int
    truncated_component_count: int
    seam_crossing_component_count: int
    interface: str = SEMANTIC_REGION_PROPOSALS_INTERFACE

    def __post_init__(self) -> None:
        if self.retained_component_count != len(self.regions):
            raise ValueError("retained_component_count must match regions")
        counts = (
            self.raw_component_count,
            self.retained_component_count,
            self.rejected_small_component_count,
            self.rejected_no_feature_count,
            self.truncated_component_count,
            self.seam_crossing_component_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in counts
        ):
            raise ValueError("proposal counts must be non-negative integers")
        if self.seam_crossing_component_count > self.raw_component_count:
            raise ValueError("seam-crossing count cannot exceed raw components")

    def describe(self) -> dict[str, object]:
        return {
            "interface": self.interface,
            "map_shape_hw": self.map_shape_hw,
            "threshold": self.threshold,
            "connectivity": self.connectivity,
            "raw_component_count": self.raw_component_count,
            "retained_component_count": self.retained_component_count,
            "rejected_small_component_count": self.rejected_small_component_count,
            "rejected_no_feature_count": self.rejected_no_feature_count,
            "truncated_component_count": self.truncated_component_count,
            "seam_crossing_component_count": self.seam_crossing_component_count,
        }


@dataclass(frozen=True, slots=True)
class _MapComponent:
    cells_yx: np.ndarray
    peak: float
    mass: float
    anchor: int
    crosses_seam: bool


def _validated_score_map(score_map: np.ndarray) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("score_map must be a non-empty (H, W) array")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("score_map values must be finite and within [0, 1]")
    return values


def _periodic_components(
    mask: np.ndarray, values: np.ndarray, connectivity: int
) -> list[_MapComponent]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    if connectivity == 4:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        offsets = tuple(
            (dy, dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        )
    components: list[_MapComponent] = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        cells: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in offsets:
                neighbour_y = y + dy
                if neighbour_y < 0 or neighbour_y >= height:
                    continue
                neighbour_x = (x + dx) % width
                if (
                    mask[neighbour_y, neighbour_x]
                    and not visited[neighbour_y, neighbour_x]
                ):
                    visited[neighbour_y, neighbour_x] = True
                    stack.append((neighbour_y, neighbour_x))
        cells_yx = np.asarray(sorted(cells), dtype=np.int64)
        component_values = values[cells_yx[:, 0], cells_yx[:, 1]]
        x_values = cells_yx[:, 1]
        components.append(
            _MapComponent(
                cells_yx=cells_yx,
                peak=float(component_values.max()),
                mass=float(component_values.sum()),
                anchor=int(np.min(cells_yx[:, 0] * width + x_values)),
                crosses_seam=bool(
                    np.any(x_values == 0) and np.any(x_values == width - 1)
                ),
            )
        )
    return components


def semantic_region_from_map(
    score_map: np.ndarray,
    *,
    erp_shape_hw: tuple[int, int],
    features,
    region_id: str,
    class_id: int,
    class_name: str,
    threshold: float = 0.5,
    semantic_score: float | None = None,
    source_id: str | None = None,
) -> SemanticRegionObservation:
    """Index panorama features inside one CAM/semantic region score map.

    The caller supplies one map per candidate region.  V1 deliberately does
    not decide how a class map is split into instances.  Pixel-center nearest
    sampling maps ERP feature coordinates to maps of any positive resolution.
    """

    values = _validated_score_map(score_map)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    erp_h, erp_w = erp_shape_hw
    if erp_h < 1 or erp_w < 1:
        raise ValueError("erp_shape_hw must contain positive dimensions")
    if features.panorama_id == "":
        raise ValueError("features.panorama_id must be non-empty")

    pixels = np.asarray(features.source_erp_xy, dtype=np.float64)
    if len(pixels) and (
        not np.all(np.isfinite(pixels))
        or np.any(pixels[:, 1] < -0.5)
        or np.any(pixels[:, 1] >= erp_h - 0.5)
    ):
        raise ValueError(
            "feature ERP rows must lie inside the image and all coordinates "
            "must be finite"
        )

    map_h, map_w = values.shape
    wrapped_x = np.mod(pixels[:, 0] + 0.5, erp_w) - 0.5
    map_x = np.floor((wrapped_x + 0.5) * map_w / erp_w).astype(np.int64)
    map_y = np.floor((pixels[:, 1] + 0.5) * map_h / erp_h).astype(np.int64)
    map_x = np.clip(map_x, 0, map_w - 1)
    map_y = np.clip(map_y, 0, map_h - 1)
    sampled = values[map_y, map_x] if len(pixels) else np.empty(0)
    selected = np.flatnonzero(sampled >= threshold)
    weights = sampled[selected]

    if len(selected):
        weighted = np.sum(features.bearings[selected] * weights[:, None], axis=0)
        if np.linalg.norm(weighted) <= 1e-12:
            weighted = np.sum(features.bearings[selected], axis=0)
        centroid = weighted / np.linalg.norm(weighted)
    else:
        peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
        erp_pixel = np.asarray(
            [
                (peak_x + 0.5) * erp_w / map_w - 0.5,
                (peak_y + 0.5) * erp_h / map_h - 0.5,
            ],
            dtype=np.float64,
        )
        centroid = np.asarray(erp_pixels_to_rays(erp_pixel, erp_shape_hw))

    region_score = float(values.max() if semantic_score is None else semantic_score)
    return SemanticRegionObservation(
        region_id=region_id,
        view_id=features.panorama_id,
        class_id=class_id,
        class_name=class_name,
        semantic_score=region_score,
        feature_indices=selected,
        membership_weights=weights,
        centroid_bearing=centroid,
        source_id=source_id,
    )


def semantic_regions_from_map(
    score_map: np.ndarray,
    *,
    erp_shape_hw: tuple[int, int],
    features,
    region_id_prefix: str,
    class_id: int,
    class_name: str,
    config: SemanticRegionProposalConfig | None = None,
    semantic_score: float | None = None,
    source_id: str | None = None,
) -> SemanticRegionProposalResult:
    """Split one CAM into deterministic ERP-seam-aware region candidates.

    Horizontal neighbours wrap periodically; vertical neighbours do not.
    Components are ordered by peak, activation mass, cell count, then raster
    anchor. Only components with enough CAM cells and indexed features are
    returned.
    """

    if not isinstance(region_id_prefix, str) or not region_id_prefix.strip():
        raise ValueError("region_id_prefix must be non-empty")
    policy = config or SemanticRegionProposalConfig()
    values = _validated_score_map(score_map)
    components = _periodic_components(
        values >= float(policy.threshold), values, policy.connectivity
    )
    components.sort(
        key=lambda item: (
            -item.peak,
            -item.mass,
            -len(item.cells_yx),
            item.anchor,
        )
    )
    rejected_small = 0
    rejected_no_feature = 0
    truncated = 0
    regions: list[SemanticRegionObservation] = []
    for component in components:
        if len(component.cells_yx) < policy.min_component_cells:
            rejected_small += 1
            continue
        component_map = np.zeros_like(values)
        component_map[component.cells_yx[:, 0], component.cells_yx[:, 1]] = values[
            component.cells_yx[:, 0], component.cells_yx[:, 1]
        ]
        candidate = semantic_region_from_map(
            component_map,
            erp_shape_hw=erp_shape_hw,
            features=features,
            region_id=(
                f"{region_id_prefix.strip()}:{int(class_id)}:{len(regions):03d}"
            ),
            class_id=class_id,
            class_name=class_name,
            threshold=float(policy.threshold),
            semantic_score=(
                component.peak if semantic_score is None else semantic_score
            ),
            source_id=(
                f"{source_id}#component-{len(regions):03d}"
                if source_id is not None
                else None
            ),
        )
        if len(candidate.feature_indices) < policy.min_feature_count:
            rejected_no_feature += 1
            continue
        if policy.max_regions is not None and len(regions) >= policy.max_regions:
            truncated += 1
            continue
        regions.append(candidate)
    return SemanticRegionProposalResult(
        regions=tuple(regions),
        map_shape_hw=tuple(int(value) for value in values.shape),
        threshold=float(policy.threshold),
        connectivity=policy.connectivity,
        raw_component_count=len(components),
        retained_component_count=len(regions),
        rejected_small_component_count=rejected_small,
        rejected_no_feature_count=rejected_no_feature,
        truncated_component_count=truncated,
        seam_crossing_component_count=sum(item.crosses_seam for item in components),
    )
