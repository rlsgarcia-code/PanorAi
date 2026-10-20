"""Sphere-native topology and measurement on canonical ERP rasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from panorai.core.numpy import spherical_solid_angle_weights
from panorai.data.regions import (
    EnclosingSphericalCap,
    RegionAdjacencyEdge,
    RegionAdjacencyGraph,
    RegionMeasurement,
    RegionMeasurements,
    SphericalBoundary,
    SphericalBoundarySet,
    SphericalContour,
    SphericalContourSet,
)
from panorai.geometry import erp_pixels_to_rays

Connectivity = Literal[4, 8]
DistanceUnits = Literal["degrees", "radians"]
INTERFACE = "panorai-spherical-region-analysis/v1"
MaskArray = Annotated[np.ndarray, "panorai:mask"]
LabelsArray = Annotated[np.ndarray, "panorai:labels"]
ScalarRaster = Annotated[np.ndarray, "panorai:scalar-raster"]
CategoricalRaster = Annotated[np.ndarray, "panorai:categorical-raster"]


def _reject_torch(value: object, *, name: str) -> None:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        raise TypeError(f"{name} is NumPy-canonical; convert Torch explicitly")


def _connectivity(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("connectivity must be 4 or 8")
    result = int(value)
    if result not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    return result


def _mask(value: Any, *, name: str) -> np.ndarray:
    _reject_torch(value, name=name)
    result = np.asarray(value)
    if result.ndim != 2 or min(result.shape) < 2:
        raise ValueError(f"{name} must use an HW shape with dimensions at least 2")
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return result


def _labels(value: Any, *, name: str = "labels") -> np.ndarray:
    _reject_torch(value, name=name)
    result = np.asarray(value)
    if result.ndim != 2 or min(result.shape) < 2:
        raise ValueError(f"{name} must use an HW shape with dimensions at least 2")
    if result.dtype == np.bool_ or not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype distinct from bool")
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative labels")
    return result


def _domain(
    shape_hw: tuple[int, int],
    *,
    support: Any | None,
    validity: Any | None,
) -> np.ndarray:
    result = np.ones(shape_hw, dtype=bool)
    for value, name in ((support, "support"), (validity, "validity")):
        if value is None:
            continue
        selected = _mask(value, name=name)
        if selected.shape != shape_hw:
            raise ValueError(f"{name} must have shape {shape_hw}")
        result &= selected
    return result


def _rays(shape_hw: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape_hw, dtype=np.float64)
    pixels = np.stack((columns, rows), axis=-1)
    return np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float64)


def _neighbours(
    row: int,
    column: int,
    shape_hw: tuple[int, int],
    connectivity: int,
):
    height, width = shape_hw
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    for delta_row, delta_column in offsets:
        neighbour_row = row + delta_row
        if 0 <= neighbour_row < height:
            yield neighbour_row, (column + delta_column) % width


def connected_components(
    mask: MaskArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    connectivity: Connectivity = 8,
) -> LabelsArray:
    """Label a boolean ERP using longitude-periodic 4- or 8-connectivity.

    Zero is background. Positive components are ordered by their first ERP
    raster index. Unsupported or invalid pixels remain zero and never connect
    otherwise separate regions.
    """

    selected = _mask(mask, name="mask")
    neighbours = _connectivity(connectivity)
    domain = _domain(selected.shape, support=support, validity=validity)
    foreground = selected & domain
    structure = ndimage.generate_binary_structure(2, 1 if neighbours == 4 else 2)
    provisional, count = ndimage.label(foreground, structure=structure)
    if count == 0:
        return np.zeros(selected.shape, dtype=np.int32)

    parent = np.arange(count + 1, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(one: int, two: int) -> None:
        if not one or not two:
            return
        root_one = find(one)
        root_two = find(two)
        if root_one != root_two:
            parent[max(root_one, root_two)] = min(root_one, root_two)

    height = selected.shape[0]
    row_offsets = (0,) if neighbours == 4 else (-1, 0, 1)
    for row in range(height):
        for delta in row_offsets:
            other = row + delta
            if 0 <= other < height:
                union(int(provisional[row, 0]), int(provisional[other, -1]))

    flat = provisional.ravel()
    roots = np.zeros_like(flat, dtype=np.int64)
    positive = flat > 0
    roots[positive] = np.fromiter(
        (find(int(item)) for item in flat[positive]),
        dtype=np.int64,
        count=int(np.count_nonzero(positive)),
    )
    first_indices: dict[int, int] = {}
    for index, root in enumerate(roots):
        if root:
            first_indices.setdefault(int(root), index)
    ordered = sorted(first_indices, key=first_indices.__getitem__)
    mapping = np.zeros(count + 1, dtype=np.int32)
    for label, root in enumerate(ordered, start=1):
        mapping[root] = label
    return mapping[roots].reshape(selected.shape)


def geodesic_distance_transform(
    mask: MaskArray,
    *,
    target: bool = True,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    units: DistanceUnits = "degrees",
) -> ScalarRaster:
    """Distance from every valid ERP centre to the nearest selected centre."""

    selected = _mask(mask, name="mask")
    if not isinstance(target, (bool, np.bool_)):
        raise TypeError("target must be boolean")
    if units not in {"degrees", "radians"}:
        raise ValueError("units must be 'degrees' or 'radians'")
    domain = _domain(selected.shape, support=support, validity=validity)
    target_mask = (selected == bool(target)) & domain
    if not np.any(target_mask):
        raise ValueError("the valid domain contains no target pixels")
    rays = _rays(selected.shape)
    tree = cKDTree(rays[target_mask])
    chord, _ = tree.query(rays[domain], k=1)
    distance = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    distance[np.isclose(chord, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    distance[np.isclose(chord, 2.0, atol=1e-12, rtol=0.0)] = math.pi
    if units == "degrees":
        distance = np.rad2deg(distance)
    result = np.full(selected.shape, np.nan, dtype=np.float64)
    result[domain] = distance
    return result


def region_boundaries(
    values: CategoricalRaster,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    connectivity: Connectivity = 4,
    include_support_boundary: bool = False,
) -> MaskArray:
    """Return inner boundary pixels for a boolean mask or positive labels."""

    _reject_torch(values, name="values")
    source = np.asarray(values)
    if source.dtype == np.bool_:
        active = source
        comparable = source.astype(np.int64)
    else:
        comparable = _labels(source, name="values")
        active = comparable > 0
    neighbours = _connectivity(connectivity)
    domain = _domain(source.shape, support=support, validity=validity)
    result = np.zeros(source.shape, dtype=bool)
    for row, column in np.argwhere(active & domain):
        for other_row, other_column in _neighbours(
            int(row), int(column), source.shape, neighbours
        ):
            if not domain[other_row, other_column]:
                if include_support_boundary:
                    result[row, column] = True
                    break
            elif comparable[other_row, other_column] != comparable[row, column]:
                result[row, column] = True
                break
    return result


def fill_holes(
    mask: MaskArray,
    *,
    max_hole_solid_angle_sr: float,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    connectivity: Connectivity = 8,
) -> MaskArray:
    """Fill small background components that do not touch missing data.

    A sphere has no distinguished exterior. Therefore the maximum solid angle
    is mandatory and defines which closed background components count as
    holes. Components adjacent to unsupported or invalid pixels are preserved.
    """

    selected = _mask(mask, name="mask")
    if isinstance(max_hole_solid_angle_sr, bool) or not isinstance(
        max_hole_solid_angle_sr, Real
    ):
        raise TypeError("max_hole_solid_angle_sr must be a finite real number")
    threshold = float(max_hole_solid_angle_sr)
    if not math.isfinite(threshold):
        raise ValueError("max_hole_solid_angle_sr must be finite")
    if not 0.0 < threshold < 4.0 * math.pi:
        raise ValueError("max_hole_solid_angle_sr must lie within (0, 4*pi)")
    neighbours = _connectivity(connectivity)
    domain = _domain(selected.shape, support=support, validity=validity)
    background = connected_components(
        ~selected,
        support=domain,
        connectivity=neighbours,
    )
    weights = spherical_solid_angle_weights(selected.shape)
    result = selected.copy()
    for label in range(1, int(background.max()) + 1):
        component = background == label
        if float(np.sum(weights[component])) > threshold:
            continue
        touches_missing = False
        for row, column in np.argwhere(component):
            if any(
                not domain[other_row, other_column]
                for other_row, other_column in _neighbours(
                    int(row), int(column), selected.shape, neighbours
                )
            ):
                touches_missing = True
                break
        if not touches_missing:
            result[component] = True
    return result


def watershed(
    gradient: ScalarRaster,
    markers: LabelsArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    connectivity: Connectivity = 8,
) -> LabelsArray:
    """Deterministic marker-controlled minimax watershed on the ERP graph.

    Flooding minimizes the maximum gradient encountered along a path from a
    marker, then path length. Remaining equal-cost conflicts select the
    smaller positive marker label. Marker pixels remain fixed.
    """

    _reject_torch(gradient, name="gradient")
    values = np.asarray(gradient)
    seeds = _labels(markers, name="markers")
    if values.shape != seeds.shape:
        raise ValueError("gradient and markers must share one HW shape")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("gradient must use a floating dtype")
    neighbours = _connectivity(connectivity)
    domain = _domain(values.shape, support=support, validity=validity)
    if np.any(~np.isfinite(values[domain])):
        raise ValueError("gradient must be finite inside the valid domain")
    if np.any((seeds > 0) & ~domain):
        raise ValueError("positive markers must lie inside support and validity")
    if not np.any((seeds > 0) & domain):
        raise ValueError("at least one positive marker is required")

    labels = np.zeros(seeds.shape, dtype=np.int64)
    costs = np.full(seeds.shape, np.inf, dtype=np.float64)
    steps = np.full(seeds.shape, np.iinfo(np.int64).max, dtype=np.int64)
    queue: list[tuple[float, int, int, int, int]] = []
    for row, column in np.argwhere((seeds > 0) & domain):
        label = int(seeds[row, column])
        cost = float(values[row, column])
        labels[row, column] = label
        costs[row, column] = cost
        steps[row, column] = 0
        heapq.heappush(queue, (cost, 0, label, int(row), int(column)))

    tolerance = 64.0 * np.finfo(np.float64).eps
    while queue:
        cost, step, label, row, column = heapq.heappop(queue)
        if (
            cost > costs[row, column] + tolerance
            or step != steps[row, column]
            or label != labels[row, column]
        ):
            continue
        for other_row, other_column in _neighbours(
            row, column, values.shape, neighbours
        ):
            if not domain[other_row, other_column]:
                continue
            marker_label = int(seeds[other_row, other_column])
            if marker_label > 0 and marker_label != label:
                continue
            candidate = max(cost, float(values[other_row, other_column]))
            candidate_step = step + 1
            current = float(costs[other_row, other_column])
            current_step = int(steps[other_row, other_column])
            current_label = int(labels[other_row, other_column])
            if candidate < current - tolerance or (
                math.isclose(candidate, current, abs_tol=tolerance, rel_tol=0.0)
                and (
                    candidate_step < current_step
                    or (
                        candidate_step == current_step
                        and (current_label == 0 or label < current_label)
                    )
                )
            ):
                costs[other_row, other_column] = candidate
                steps[other_row, other_column] = candidate_step
                labels[other_row, other_column] = label
                heapq.heappush(
                    queue,
                    (candidate, candidate_step, label, other_row, other_column),
                )
    return labels


def _edge_lengths(shape_hw: tuple[int, int]) -> tuple[float, np.ndarray]:
    height, width = shape_hw
    vertical = math.pi / height
    latitude_edges = math.pi / 2.0 - np.arange(height + 1) * math.pi / height
    horizontal = (2.0 * math.pi / width) * np.abs(np.cos(latitude_edges))
    return vertical, horizontal


def _perimeters(
    labels: np.ndarray,
    domain: np.ndarray,
) -> tuple[dict[int, float], dict[int, float], dict[tuple[int, int], float]]:
    height, width = labels.shape
    vertical_length, horizontal_lengths = _edge_lengths(labels.shape)
    perimeter: dict[int, float] = {}
    unsupported: dict[int, float] = {}
    adjacency: dict[tuple[int, int], float] = {}

    def add(target: dict[Any, float], key: Any, length: float) -> None:
        target[key] = target.get(key, 0.0) + length

    for row in range(height):
        for column in range(width):
            label = int(labels[row, column])
            right_column = (column + 1) % width
            current_valid = bool(domain[row, column])
            right_valid = bool(domain[row, right_column])
            if current_valid and right_valid:
                other = int(labels[row, right_column])
                if label != other:
                    if label > 0:
                        add(perimeter, label, vertical_length)
                    if other > 0:
                        add(perimeter, other, vertical_length)
                    if label > 0 and other > 0:
                        add(adjacency, tuple(sorted((label, other))), vertical_length)
            elif current_valid != right_valid:
                valid_label = label if current_valid else int(labels[row, right_column])
                if valid_label > 0:
                    add(unsupported, valid_label, vertical_length)

            if row + 1 >= height:
                continue
            length = float(horizontal_lengths[row + 1])
            lower_valid = bool(domain[row + 1, column])
            if current_valid and lower_valid:
                other = int(labels[row + 1, column])
                if label != other:
                    if label > 0:
                        add(perimeter, label, length)
                    if other > 0:
                        add(perimeter, other, length)
                    if label > 0 and other > 0:
                        add(adjacency, tuple(sorted((label, other))), length)
            elif current_valid != lower_valid:
                valid_label = label if current_valid else int(labels[row + 1, column])
                if valid_label > 0:
                    add(unsupported, valid_label, length)
    return perimeter, unsupported, adjacency


def measure_regions(
    labels: LabelsArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
) -> RegionMeasurements:
    """Measure positive ERP labels using solid angle and angular geometry."""

    source = _labels(labels)
    domain = _domain(source.shape, support=support, validity=validity)
    weights = spherical_solid_angle_weights(source.shape)
    support_area = float(np.sum(weights[domain]))
    if support_area <= 0.0:
        raise ValueError("support and validity must select at least one pixel")
    rays = _rays(source.shape)
    perimeter, unsupported, _ = _perimeters(source, domain)
    records: list[RegionMeasurement] = []
    for label in sorted(int(item) for item in np.unique(source[domain]) if item > 0):
        selected = (source == label) & domain
        area = float(np.sum(weights[selected]))
        weighted = np.sum(rays[selected] * weights[selected, None], axis=0)
        norm = float(np.linalg.norm(weighted))
        centroid = (
            None if norm <= 4096.0 * np.finfo(np.float64).eps else weighted / norm
        )
        cap_center = rays[selected][0] if centroid is None else centroid
        radius = math.degrees(
            math.acos(float(np.clip(np.min(rays[selected] @ cap_center), -1.0, 1.0)))
        )
        records.append(
            RegionMeasurement(
                label=label,
                pixel_count=int(np.count_nonzero(selected)),
                solid_angle_sr=area,
                support_fraction=area / support_area,
                angular_perimeter_rad=perimeter.get(label, 0.0),
                unsupported_boundary_rad=unsupported.get(label, 0.0),
                centroid_bearing_xyz=centroid,
                enclosing_cap=EnclosingSphericalCap(cap_center, radius),
            )
        )
    return RegionMeasurements(source.shape, support_area, tuple(records))


def region_adjacency_graph(
    labels: LabelsArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
) -> RegionAdjacencyGraph:
    """Return positive-label contacts and shared angular boundary lengths."""

    source = _labels(labels)
    domain = _domain(source.shape, support=support, validity=validity)
    _, _, contacts = _perimeters(source, domain)
    present = tuple(sorted(int(item) for item in np.unique(source[domain]) if item > 0))
    edges = tuple(
        RegionAdjacencyEdge(pair, length) for pair, length in sorted(contacts.items())
    )
    return RegionAdjacencyGraph(present, edges)


def extract_boundaries(
    labels: LabelsArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
) -> SphericalBoundarySet:
    """Lift each label's inner boundary pixel centres to unit bearings."""

    source = _labels(labels)
    domain = _domain(source.shape, support=support, validity=validity)
    rays = _rays(source.shape)
    perimeter, _, _ = _perimeters(source, domain)
    records = []
    for label in sorted(int(item) for item in np.unique(source[domain]) if item > 0):
        boundary = region_boundaries(
            source == label,
            support=domain,
            connectivity=4,
        )
        records.append(
            SphericalBoundary(label, rays[boundary], perimeter.get(label, 0.0))
        )
    return SphericalBoundarySet(source.shape, tuple(records))


def _contour_vertex(
    row_edge: int,
    column_edge: int,
    shape_hw: tuple[int, int],
) -> tuple[int, int]:
    """Canonicalize a cell-corner vertex on the ERP sphere."""

    height, width = shape_hw
    if row_edge in {0, height}:
        # Every longitude at a pole is the same spherical vertex.
        return row_edge, 0
    return row_edge, column_edge % width


def _contour_rays(
    vertices: list[tuple[int, int]],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    pixels = np.asarray(
        [(column - 0.5, row - 0.5) for row, column in vertices],
        dtype=np.float64,
    )
    return np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float64)


def _contour_length(rays: np.ndarray, *, closed: bool) -> float:
    if len(rays) < 2:
        return 0.0
    first = rays
    second = np.roll(rays, -1, axis=0) if closed else rays[1:]
    if not closed:
        first = rays[:-1]
    dots = np.sum(first * second, axis=1)
    return float(np.sum(np.arccos(np.clip(dots, -1.0, 1.0))))


def trace_contours(
    labels: LabelsArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    include_support_boundary: bool = False,
) -> SphericalContourSet:
    """Trace ordered sub-pixel contours along canonical ERP cell interfaces.

    Longitude is periodic and all cell corners on either polar edge are one
    spherical vertex.  The returned vertices therefore remain ordered across
    the displayed ERP seam without inventing a latitude wrap.  Boundaries
    adjacent to missing support or invalid data are omitted by default, which
    yields honest open contours; callers may include those interfaces
    explicitly with ``include_support_boundary=True``.
    """

    source = _labels(labels)
    if not isinstance(include_support_boundary, bool):
        raise TypeError("include_support_boundary must be boolean")
    domain = _domain(source.shape, support=support, validity=validity)
    height, width = source.shape
    contours: list[SphericalContour] = []

    for label in sorted(int(item) for item in np.unique(source[domain]) if item > 0):
        edges: list[tuple[tuple[int, int], tuple[int, int], int]] = []

        def add_edge(
            start: tuple[int, int],
            end: tuple[int, int],
            direction: int,
        ) -> None:
            edges.append(
                (
                    _contour_vertex(*start, source.shape),
                    _contour_vertex(*end, source.shape),
                    direction,
                )
            )

        for row, column in np.argwhere((source == label) & domain):
            row = int(row)
            column = int(column)
            left = (column - 1) % width
            right = (column + 1) % width

            if row > 0:
                neighbour_valid = bool(domain[row - 1, column])
                if (neighbour_valid and source[row - 1, column] != label) or (
                    not neighbour_valid and include_support_boundary
                ):
                    add_edge((row, column), (row, column + 1), 0)
            if bool(domain[row, right]):
                if source[row, right] != label:
                    add_edge((row, column + 1), (row + 1, column + 1), 1)
            elif include_support_boundary:
                add_edge((row, column + 1), (row + 1, column + 1), 1)
            if row + 1 < height:
                neighbour_valid = bool(domain[row + 1, column])
                if (neighbour_valid and source[row + 1, column] != label) or (
                    not neighbour_valid and include_support_boundary
                ):
                    add_edge((row + 1, column + 1), (row + 1, column), 2)
            if bool(domain[row, left]):
                if source[row, left] != label:
                    add_edge((row + 1, column), (row, column), 3)
            elif include_support_boundary:
                add_edge((row + 1, column), (row, column), 3)

        by_start: dict[tuple[int, int], list[int]] = {}
        for index, edge in enumerate(edges):
            by_start.setdefault(edge[0], []).append(index)
        for indices in by_start.values():
            indices.sort(key=lambda index: (edges[index][2], edges[index][1]))

        unused = set(range(len(edges)))
        while unused:
            incoming_vertices = {edges[index][1] for index in unused}
            open_starts = [
                index for index in unused if edges[index][0] not in incoming_vertices
            ]
            edge_index = min(
                open_starts or unused,
                key=lambda index: (edges[index][0], edges[index][2], edges[index][1]),
            )
            start, end, direction = edges[edge_index]
            vertices = [start]
            unused.remove(edge_index)
            current = end
            incoming = direction
            closed = current == start
            while not closed:
                candidates = [
                    index for index in by_start.get(current, ()) if index in unused
                ]
                if not candidates:
                    vertices.append(current)
                    break
                edge_index = min(
                    candidates,
                    key=lambda index: (
                        {1: 0, 0: 1, 3: 2, 2: 3}[(edges[index][2] - incoming) % 4],
                        edges[index][2],
                        edges[index][1],
                    ),
                )
                vertices.append(current)
                unused.remove(edge_index)
                _, current, incoming = edges[edge_index]
                closed = current == start
            if len(vertices) < (3 if closed else 2):
                continue
            rays = _contour_rays(vertices, source.shape)
            contours.append(
                SphericalContour(
                    label=label,
                    bearings_xyz=rays,
                    closed=closed,
                    angular_length_rad=_contour_length(rays, closed=closed),
                )
            )

    contours.sort(
        key=lambda item: (
            item.label,
            not item.closed,
            -item.angular_length_rad,
            tuple(item.bearings_xyz[0]),
        )
    )
    return SphericalContourSet(source.shape, tuple(contours))


def skeletonize(
    mask: MaskArray,
    *,
    support: MaskArray | None = None,
    validity: MaskArray | None = None,
    minimum_radius_deg: float = 0.0,
    connectivity: Connectivity = 8,
) -> MaskArray:
    """Return deterministic ridges of the intrinsic distance-to-background.

    The result is a sampled spherical medial axis. A centre is retained when
    it is a strict local ridge across at least one opposite neighbour pair.
    Distances are exact great-circle angles between ERP pixel centres; this is
    not a planar row/column thinning pass. Missing support and invalid data are
    absent domain vertices and are never crossed. A region filling its entire
    supported domain has no observable background and is rejected.
    """

    selected = _mask(mask, name="mask")
    domain = _domain(selected.shape, support=support, validity=validity)
    current = selected & domain
    if not np.any(current):
        return np.zeros(selected.shape, dtype=bool)
    if bool(np.all(current[domain])):
        raise ValueError(
            "a region filling its supported domain has no distinguished skeleton"
        )
    if isinstance(minimum_radius_deg, bool) or not isinstance(minimum_radius_deg, Real):
        raise TypeError("minimum_radius_deg must be a real number")
    minimum_radius = float(minimum_radius_deg)
    if not math.isfinite(minimum_radius) or not 0.0 <= minimum_radius <= 180.0:
        raise ValueError("minimum_radius_deg must lie within [0, 180]")
    neighbours = _connectivity(connectivity)

    distance = geodesic_distance_transform(
        selected,
        target=False,
        support=domain,
        units="degrees",
    )
    negative = np.full(selected.shape, -np.inf, dtype=np.float64)

    def shifted(delta_row: int, delta_column: int) -> np.ndarray:
        shifted_values = np.roll(distance, delta_column, axis=1)
        shifted_domain = np.roll(current, delta_column, axis=1)
        if delta_row > 0:
            shifted_values = np.vstack(
                (negative[:delta_row], shifted_values[:-delta_row])
            )
            shifted_domain = np.vstack(
                (np.zeros_like(current[:delta_row]), shifted_domain[:-delta_row])
            )
        elif delta_row < 0:
            amount = -delta_row
            shifted_values = np.vstack((shifted_values[amount:], negative[:amount]))
            shifted_domain = np.vstack(
                (shifted_domain[amount:], np.zeros_like(current[:amount]))
            )
        return np.where(shifted_domain, shifted_values, -np.inf)

    tolerance = 64.0 * np.finfo(np.float64).eps * 180.0
    pairs = [
        (shifted(0, -1), shifted(0, 1)),
        (shifted(-1, 0), shifted(1, 0)),
    ]
    if neighbours == 8:
        pairs.extend(
            (
                (shifted(-1, -1), shifted(1, 1)),
                (shifted(-1, 1), shifted(1, -1)),
            )
        )
    ridge = np.zeros(selected.shape, dtype=bool)
    for first, second in pairs:
        ridge |= (
            (distance >= first - tolerance)
            & (distance >= second - tolerance)
            & ((distance > first + tolerance) | (distance > second + tolerance))
        )
    return current & ridge & (distance >= minimum_radius)


@dataclass(frozen=True, slots=True)
class _RegionRasterOperation:
    operation_name: ClassVar[str]
    interface: ClassVar[str] = INTERFACE
    stability: ClassVar[str] = "stable"
    spherical_operation: ClassVar[bool] = True
    canonical_erp_output: ClassVar[bool] = True
    preserves_shadow_state: ClassVar[bool] = True
    interpolation: ClassVar[str] = "nearest"

    def transform_mask(self, mask: Any) -> np.ndarray:
        return _mask(mask, name="mask").copy()

    def transform_grid(self, input_grid: Any) -> Any:
        return input_grid.with_transform(
            operation=self.operation_name,
            output_shape_hw=input_grid.shape_hw,
            domain=input_grid.domain,
            alignment=input_grid.alignment,
            first_center_px_yx=input_grid.first_center_px_yx,
            pixel_stride_yx=input_grid.pixel_stride_yx,
            parameters=asdict(self),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "interface": self.interface,
            "stability": self.stability,
            "parameters": asdict(self),
        }


@dataclass(frozen=True, slots=True)
class ConnectedComponents(_RegionRasterOperation):
    """Configured longitude-periodic connected-component operation."""

    connectivity: Connectivity = 8
    operation_name: ClassVar[str] = "connected_components"

    def __post_init__(self) -> None:
        object.__setattr__(self, "connectivity", _connectivity(self.connectivity))

    def __call__(self, values: MaskArray) -> LabelsArray:
        return connected_components(values, connectivity=self.connectivity)

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("region topology uses strict validity propagation")
        return (
            connected_components(
                values,
                support=support,
                validity=validity,
                connectivity=self.connectivity,
            ),
            self.transform_mask(support),
            self.transform_mask(validity),
        )


@dataclass(frozen=True, slots=True)
class RegionBoundaries(_RegionRasterOperation):
    """Configured inner-boundary extraction operation."""

    connectivity: Connectivity = 4
    include_support_boundary: bool = False
    operation_name: ClassVar[str] = "region_boundaries"

    def __post_init__(self) -> None:
        object.__setattr__(self, "connectivity", _connectivity(self.connectivity))
        if not isinstance(self.include_support_boundary, bool):
            raise TypeError("include_support_boundary must be boolean")

    def __call__(self, values: CategoricalRaster) -> MaskArray:
        return region_boundaries(
            values,
            connectivity=self.connectivity,
            include_support_boundary=self.include_support_boundary,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("region topology uses strict validity propagation")
        return (
            region_boundaries(
                values,
                support=support,
                validity=validity,
                connectivity=self.connectivity,
                include_support_boundary=self.include_support_boundary,
            ),
            self.transform_mask(support),
            self.transform_mask(validity),
        )


@dataclass(frozen=True, slots=True)
class GeodesicDistanceTransform(_RegionRasterOperation):
    """Configured exact pixel-centre great-circle distance transform."""

    target: bool = True
    units: DistanceUnits = "degrees"
    operation_name: ClassVar[str] = "geodesic_distance_transform"

    def __post_init__(self) -> None:
        if not isinstance(self.target, bool):
            raise TypeError("target must be boolean")
        if self.units not in {"degrees", "radians"}:
            raise ValueError("units must be 'degrees' or 'radians'")

    def __call__(self, values: MaskArray) -> ScalarRaster:
        return geodesic_distance_transform(values, target=self.target, units=self.units)

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("region distance uses strict validity propagation")
        return (
            geodesic_distance_transform(
                values,
                target=self.target,
                support=support,
                validity=validity,
                units=self.units,
            ),
            self.transform_mask(support),
            self.transform_mask(validity),
        )


@dataclass(frozen=True, slots=True)
class FillHoles(_RegionRasterOperation):
    """Configured solid-angle-bounded hole filling."""

    max_hole_solid_angle_sr: float
    connectivity: Connectivity = 8
    operation_name: ClassVar[str] = "fill_holes"

    def __post_init__(self) -> None:
        threshold = float(self.max_hole_solid_angle_sr)
        if isinstance(self.max_hole_solid_angle_sr, bool) or not isinstance(
            self.max_hole_solid_angle_sr, Real
        ):
            raise TypeError("max_hole_solid_angle_sr must be a real number")
        if not math.isfinite(threshold) or not 0.0 < threshold < 4.0 * math.pi:
            raise ValueError("max_hole_solid_angle_sr must lie within (0, 4*pi)")
        object.__setattr__(self, "max_hole_solid_angle_sr", threshold)
        object.__setattr__(self, "connectivity", _connectivity(self.connectivity))

    def __call__(self, values: MaskArray) -> MaskArray:
        return fill_holes(
            values,
            max_hole_solid_angle_sr=self.max_hole_solid_angle_sr,
            connectivity=self.connectivity,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("hole filling uses strict validity propagation")
        return (
            fill_holes(
                values,
                max_hole_solid_angle_sr=self.max_hole_solid_angle_sr,
                support=support,
                validity=validity,
                connectivity=self.connectivity,
            ),
            self.transform_mask(support),
            self.transform_mask(validity),
        )


@dataclass(frozen=True, slots=True)
class Skeletonize(_RegionRasterOperation):
    """Configured intrinsic distance-ridge skeleton operation."""

    minimum_radius_deg: float = 0.0
    connectivity: Connectivity = 8
    operation_name: ClassVar[str] = "skeletonize"

    def __post_init__(self) -> None:
        if isinstance(self.minimum_radius_deg, bool) or not isinstance(
            self.minimum_radius_deg, Real
        ):
            raise TypeError("minimum_radius_deg must be a real number")
        radius = float(self.minimum_radius_deg)
        if not math.isfinite(radius) or not 0.0 <= radius <= 180.0:
            raise ValueError("minimum_radius_deg must lie within [0, 180]")
        object.__setattr__(self, "minimum_radius_deg", radius)
        object.__setattr__(self, "connectivity", _connectivity(self.connectivity))

    def __call__(self, values: MaskArray) -> MaskArray:
        return skeletonize(
            values,
            minimum_radius_deg=self.minimum_radius_deg,
            connectivity=self.connectivity,
        )

    def apply_with_masks(
        self,
        values: Any,
        support: Any,
        validity: Any,
        *,
        invalid_policy: str,
        min_valid_weight: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if invalid_policy != "propagate" or min_valid_weight is not None:
            raise ValueError("skeletonization uses strict validity propagation")
        return (
            skeletonize(
                values,
                support=support,
                validity=validity,
                minimum_radius_deg=self.minimum_radius_deg,
                connectivity=self.connectivity,
            ),
            self.transform_mask(support),
            self.transform_mask(validity),
        )


__all__ = [
    "ConnectedComponents",
    "FillHoles",
    "GeodesicDistanceTransform",
    "RegionBoundaries",
    "Skeletonize",
    "connected_components",
    "extract_boundaries",
    "fill_holes",
    "geodesic_distance_transform",
    "measure_regions",
    "region_adjacency_graph",
    "region_boundaries",
    "skeletonize",
    "trace_contours",
    "watershed",
]
