"""SLIC superpixels defined on the unit sphere."""

from __future__ import annotations

from collections import defaultdict, deque
import math

import numpy as np


_LOCAL_DOMAIN_MAX_AREA_SR = 1.0
_LOCAL_DOMAIN_MIN_RESULTANT = 0.75


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must use HWC layout with exactly three RGB channels")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("image height and width must both be at least 2")
    if np.issubdtype(image.dtype, np.bool_):
        raise TypeError("boolean RGB images are not supported")
    if np.issubdtype(image.dtype, np.unsignedinteger):
        maximum = np.iinfo(image.dtype).max
        return image.astype(np.float64) / maximum
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("image must use an unsigned integer or floating dtype")
    values = image.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("image must contain only finite RGB values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("floating RGB images must be normalized to [0, 1]")
    return values


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite non-negative real number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _erp_rays(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    longitude = ((x + 0.5) / width) * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - ((y + 0.5) / height) * np.pi
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            np.sin(longitude) * cos_latitude,
            np.broadcast_to(np.sin(latitude), (height, width)),
            np.cos(longitude) * cos_latitude,
        ),
        axis=-1,
    )


def _solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return the exact solid angle represented by every ERP pixel."""

    height, width = shape_hw
    north_edges = np.pi / 2.0 - np.arange(height, dtype=np.float64) * np.pi / height
    south_edges = north_edges - np.pi / height
    row_area = (2.0 * np.pi / width) * (np.sin(north_edges) - np.sin(south_edges))
    return np.broadcast_to(row_area[:, None], (height, width))


def _fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    longitude = index * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.stack((radius * np.sin(longitude), y, radius * np.cos(longitude)), axis=1)


def _initial_centers(
    colors: np.ndarray,
    rays: np.ndarray,
    weights: np.ndarray,
    count: int,
    *,
    compactness: float,
    angular_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose rotation-covariant textured seeds with a uniform fallback."""

    valid_area = angular_scale**2 * count
    centroid = np.sum(weights[:, None] * rays, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    directional_resultant = centroid_norm / valid_area
    if (
        valid_area < _LOCAL_DOMAIN_MAX_AREA_SR
        and centroid_norm > np.finfo(np.float64).eps
        and directional_resultant >= _LOCAL_DOMAIN_MIN_RESULTANT
    ):
        centroid /= centroid_norm
        indices = np.empty(count, dtype=np.int64)
        indices[0] = int(np.argmin(rays @ centroid))
        selected_rays = [rays[indices[0]]]
        nearest_similarity = rays @ selected_rays[0]
        for cluster in range(1, count):
            indices[cluster] = int(np.argmin(nearest_similarity))
            selected_rays.append(rays[indices[cluster]])
            nearest_similarity = np.maximum(
                nearest_similarity, rays @ selected_rays[-1]
            )
        center_rays = np.asarray(selected_rays)
        for _ in range(12):
            labels = np.argmax(rays @ center_rays.T, axis=1)
            updated = center_rays.copy()
            for cluster in range(count):
                members = labels == cluster
                if not np.any(members):
                    continue
                direction = np.sum(weights[members, None] * rays[members], axis=0)
                norm = np.linalg.norm(direction)
                if norm > np.finfo(np.float64).eps:
                    updated[cluster] = direction / norm
            if np.allclose(updated, center_rays, rtol=0.0, atol=1e-12):
                break
            center_rays = updated
        used = np.zeros(colors.shape[0], dtype=bool)
        for cluster, direction in enumerate(center_rays):
            similarity = rays @ direction
            similarity[used] = -np.inf
            indices[cluster] = int(np.argmax(similarity))
            used[indices[cluster]] = True
        return colors[indices].copy(), rays[indices].copy()

    mean_color = np.average(colors, axis=0, weights=weights)
    saliency = np.einsum("ni,ni->n", colors - mean_color, colors - mean_color)
    scale = max(1.0, float(np.max(saliency, initial=0.0)))
    if float(np.ptp(saliency)) > np.finfo(np.float64).eps * scale * 64.0:
        indices = np.empty(count, dtype=np.int64)
        selected = np.zeros(colors.shape[0], dtype=bool)
        indices[0] = int(np.argmax(saliency))
        selected[indices[0]] = True
        minimum_distance = np.full(colors.shape[0], np.inf, dtype=np.float64)
        spatial_weight = (compactness / angular_scale) ** 2
        for cluster in range(1, count):
            previous = indices[cluster - 1]
            color_delta = colors - colors[previous]
            color_distance = np.einsum("ni,ni->n", color_delta, color_delta)
            angular_distance = np.arccos(np.clip(rays @ rays[previous], -1.0, 1.0))
            distance = color_distance + spatial_weight * angular_distance**2
            minimum_distance = np.minimum(minimum_distance, distance)
            minimum_distance[selected] = -np.inf
            indices[cluster] = int(np.argmax(minimum_distance))
            selected[indices[cluster]] = True
        return colors[indices].copy(), rays[indices].copy()

    targets = _fibonacci_directions(count)
    used = np.zeros(rays.shape[0], dtype=bool)
    indices = np.empty(count, dtype=np.int64)
    for cluster, target in enumerate(targets):
        similarity = rays @ target
        similarity[used] = -np.inf
        selected = int(np.argmax(similarity))
        indices[cluster] = selected
        used[selected] = True
    return colors[indices].copy(), rays[indices].copy()


def _assign_labels(
    colors: np.ndarray,
    rays: np.ndarray,
    center_colors: np.ndarray,
    center_rays: np.ndarray,
    *,
    compactness: float,
    angular_scale: float,
    chunk_size: int | None = None,
) -> np.ndarray:
    labels = np.empty(colors.shape[0], dtype=np.int32)
    spatial_weight = (compactness / angular_scale) ** 2
    if chunk_size is None:
        # Bound the largest temporary (N_chunk, K, 3) colour array to roughly
        # 24 MiB in float64 for ordinary K. This keeps high segment counts from
        # turning the vectorized assignment into an unbounded allocation.
        chunk_size = max(1, min(32_768, 1_048_576 // center_colors.shape[0]))
    # A planar SLIC centre visits a 2S-by-2S window, whose corner is sqrt(2)S
    # away. The same normalized support radius is used as a spherical cap.
    search_radius = min(np.pi, math.sqrt(2.0) * angular_scale)
    for start in range(0, colors.shape[0], chunk_size):
        stop = min(colors.shape[0], start + chunk_size)
        color_difference = colors[start:stop, None, :] - center_colors[None, :, :]
        color_distance_squared = np.einsum(
            "nki,nki->nk", color_difference, color_difference
        )
        cosine = np.clip(rays[start:stop] @ center_rays.T, -1.0, 1.0)
        angular_distance = np.arccos(cosine)
        distance = color_distance_squared + spatial_weight * angular_distance**2
        local = angular_distance <= search_radius
        distance[~local] = np.inf
        uncovered = ~np.any(local, axis=1)
        if np.any(uncovered):
            # A strongly masked domain can leave a hole in the initial
            # Fibonacci support. Assign such samples to their nearest centre;
            # later updates normally restore local coverage.
            nearest = np.argmin(angular_distance[uncovered], axis=1)
            distance[uncovered] = np.inf
            distance[np.flatnonzero(uncovered), nearest] = 0.0
        labels[start:stop] = np.argmin(distance, axis=1)
    return labels


def _update_centers(
    colors: np.ndarray,
    rays: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    center_colors: np.ndarray,
    center_rays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = center_colors.shape[0]
    totals = np.bincount(labels, weights=weights, minlength=count)
    occupied = totals > 0.0
    updated_colors = center_colors.copy()
    updated_rays = center_rays.copy()

    for channel in range(3):
        sums = np.bincount(
            labels, weights=weights * colors[:, channel], minlength=count
        )
        updated_colors[occupied, channel] = sums[occupied] / totals[occupied]

    ray_sums = np.column_stack(
        [
            np.bincount(labels, weights=weights * rays[:, axis], minlength=count)
            for axis in range(3)
        ]
    )
    norms = np.linalg.norm(ray_sums, axis=1)
    valid_direction = occupied & (norms > np.finfo(np.float64).eps)
    updated_rays[valid_direction] = (
        ray_sums[valid_direction] / norms[valid_direction, None]
    )
    return updated_colors, updated_rays


def _component_map(
    labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], dict[tuple[int, int], int]]:
    """Split raw clusters into seam-periodic four-connected components."""

    height, width = labels.shape
    components = np.full((height, width), -1, dtype=np.int32)
    members: list[np.ndarray] = []
    edges: dict[tuple[int, int], int] = defaultdict(int)

    for start_y, start_x in np.argwhere(valid):
        if components[start_y, start_x] >= 0:
            continue
        component = len(members)
        raw_label = labels[start_y, start_x]
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        components[start_y, start_x] = component
        pixels: list[int] = []
        while queue:
            y, x = queue.popleft()
            pixels.append(y * width + x)
            neighbours = ((y, (x - 1) % width), (y, (x + 1) % width))
            if y > 0:
                neighbours += ((y - 1, x),)
            if y + 1 < height:
                neighbours += ((y + 1, x),)
            for ny, nx in neighbours:
                if not valid[ny, nx] or labels[ny, nx] != raw_label:
                    continue
                if components[ny, nx] < 0:
                    components[ny, nx] = component
                    queue.append((ny, nx))
        members.append(np.asarray(pixels, dtype=np.int64))

    # Count each undirected component boundary once. Horizontal pairs wrap.
    for y, x in np.argwhere(valid):
        here = int(components[y, x])
        for ny, nx in ((int(y), (int(x) + 1) % width), (int(y) + 1, int(x))):
            if ny >= height or not valid[ny, nx]:
                continue
            there = int(components[ny, nx])
            if here != there:
                edge = (min(here, there), max(here, there))
                edges[edge] += 1
    return components, members, edges


def _enforce_connectivity(
    labels: np.ndarray,
    valid: np.ndarray,
    solid_angles: np.ndarray,
    *,
    expected_area: float,
    min_size_factor: float,
    start_label: int,
) -> np.ndarray:
    components, members, edges = _component_map(labels, valid)
    count = len(members)
    parent = np.arange(count, dtype=np.int32)
    areas = np.asarray(
        [solid_angles.ravel()[indices].sum() for indices in members],
        dtype=np.float64,
    )
    neighbours: list[dict[int, int]] = [defaultdict(int) for _ in range(count)]
    for (left, right), boundary in edges.items():
        neighbours[left][right] += boundary
        neighbours[right][left] += boundary

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    minimum_area = expected_area * min_size_factor
    for component in np.argsort(areas):
        root = find(int(component))
        if root != component or areas[root] >= minimum_area:
            continue
        candidates: dict[int, int] = defaultdict(int)
        for neighbour, boundary in neighbours[root].items():
            neighbour_root = find(neighbour)
            if neighbour_root != root:
                candidates[neighbour_root] += boundary
        if not candidates:
            continue
        target = max(
            candidates,
            key=lambda item: (candidates[item], areas[find(item)], -find(item)),
        )
        target = find(target)
        parent[root] = target
        areas[target] += areas[root]
        for neighbour, boundary in neighbours[root].items():
            neighbour_root = find(neighbour)
            if neighbour_root != target:
                neighbours[target][neighbour_root] += boundary
                neighbours[neighbour_root][target] += boundary

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int32)
    unique_roots = np.unique(roots)
    compact = {
        int(root): start_label + index for index, root in enumerate(unique_roots)
    }
    component_labels = np.asarray(
        [compact[int(root)] for root in roots], dtype=np.int32
    )
    result = np.full(labels.shape, -1, dtype=np.int32)
    result[valid] = component_labels[components[valid]]
    return result


def spherical_slic(
    image: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    *,
    max_num_iter: int = 10,
    convergence_tol_deg: float = 1e-3,
    enforce_connectivity: bool = True,
    min_size_factor: float = 0.5,
    start_label: int = 1,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Segment an RGB equirectangular panorama with spherical SLIC.

    Unlike planar or per-view SLIC, the spatial term is the intrinsic
    great-circle distance ``acos(ray_pixel @ ray_center)`` between canonical
    ERP pixel-centre rays. No gnomonic image or projected pixel coordinate
    participates in clustering. The result is an ``HW int32`` label array.
    Geodesically concentrated local valid domains below one steradian use
    farthest-point coverage followed by solid-angle spherical Lloyd
    refinement, which tends to the regular planar seed layout as curvature
    shrinks. Larger or disconnected textured inputs use deterministic
    farthest-point seeds under the same
    colour-plus-intrinsic-distance metric as assignment, so a common 3D
    rotation does not pin the initialization to the ERP frame. Uniform inputs
    fall back to an equal-area Fibonacci distribution. Centroid updates are
    solid-angle weighted and normalized on the unit sphere, and connectivity
    wraps across the longitude seam.

    Parameters
    ----------
    image:
        ``HWC`` RGB panorama. Unsigned integers use their full dtype range;
        floating input must be finite and normalized to ``[0, 1]``.
    n_segments:
        Requested number of initial superpixels. Connectivity enforcement can
        change the final count.
    compactness:
        Weight of normalized angular distance relative to CIE Lab distance.
        Larger values produce more spatially regular superpixels.
    max_num_iter:
        Maximum assignment/update iterations.
    convergence_tol_deg:
        Stop when every occupied spherical centroid moves by no more than this
        angle, or earlier when assignments stop changing.
    enforce_connectivity:
        Split disconnected assignments and merge components smaller than
        ``min_size_factor`` times the requested mean solid angle.
    min_size_factor:
        Non-negative small-component threshold used by connectivity cleanup.
    start_label:
        Non-negative first label. Pixels outside ``mask`` are always ``-1``.
    mask:
        Optional boolean ``HW`` validity mask. Only valid pixels participate in
        initialization, clustering, area statistics, and connectivity.

    """

    # Keep ``import panorai.image_processing`` NumPy-only. scikit-image is a
    # core runtime dependency, but its colour stack (and SciPy) is needed only
    # when this operation is actually invoked.
    from skimage.color import rgb2lab

    rgb = _validate_rgb(image)
    segments = _validate_positive_integer(n_segments, "n_segments")
    iterations = _validate_positive_integer(max_num_iter, "max_num_iter")
    compactness_value = _validate_finite_nonnegative(compactness, "compactness")
    tolerance = _validate_finite_nonnegative(convergence_tol_deg, "convergence_tol_deg")
    size_factor = _validate_finite_nonnegative(min_size_factor, "min_size_factor")
    if isinstance(start_label, bool) or not isinstance(start_label, (int, np.integer)):
        raise TypeError("start_label must be a non-negative integer")
    first_label = int(start_label)
    if first_label < 0:
        raise ValueError("start_label must be non-negative")
    if not isinstance(enforce_connectivity, (bool, np.bool_)):
        raise TypeError("enforce_connectivity must be boolean")

    shape = rgb.shape[:2]
    if mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean numpy.ndarray")
        if mask.shape != shape:
            raise ValueError("mask must have the same HW shape as image")
        valid = mask
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("mask must select at least one pixel")
    if segments > valid_count:
        raise ValueError("n_segments cannot exceed the number of valid pixels")
    if first_label > np.iinfo(np.int32).max - valid_count:
        raise ValueError("start_label is too large for an int32 label result")

    lab = rgb2lab(rgb)
    all_rays = _erp_rays(shape)
    all_solid_angles = _solid_angle_weights(shape)
    colors = lab[valid]
    rays = all_rays[valid]
    weights = all_solid_angles[valid]
    valid_area = float(weights.sum())
    expected_area = valid_area / segments
    # sqrt(area) is the spherical counterpart of S=sqrt(N/K) in planar SLIC.
    angular_scale = math.sqrt(expected_area)
    center_colors, center_rays = _initial_centers(
        colors,
        rays,
        weights,
        segments,
        compactness=compactness_value,
        angular_scale=angular_scale,
    )
    previous_labels: np.ndarray | None = None
    tolerance_rad = math.radians(tolerance)

    for _ in range(iterations):
        labels = _assign_labels(
            colors,
            rays,
            center_colors,
            center_rays,
            compactness=compactness_value,
            angular_scale=angular_scale,
        )
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        updated_colors, updated_rays = _update_centers(
            colors, rays, weights, labels, center_colors, center_rays
        )
        color_motion = np.linalg.norm(updated_colors - center_colors, axis=1)
        angular_motion = np.arccos(
            np.clip(np.einsum("ij,ij->i", center_rays, updated_rays), -1.0, 1.0)
        )
        center_colors, center_rays = updated_colors, updated_rays
        previous_labels = labels
        if (
            angular_motion.max(initial=0.0) <= tolerance_rad
            and color_motion.max(initial=0.0) <= 1e-6
        ):
            break

    # Reassign once to the final updated centroids.
    labels = _assign_labels(
        colors,
        rays,
        center_colors,
        center_rays,
        compactness=compactness_value,
        angular_scale=angular_scale,
    )
    raw = np.full(shape, -1, dtype=np.int32)
    raw[valid] = labels
    if enforce_connectivity:
        return _enforce_connectivity(
            raw,
            valid,
            all_solid_angles,
            expected_area=expected_area,
            min_size_factor=size_factor,
            start_label=first_label,
        )
    raw[valid] += first_label
    return raw


__all__ = ["spherical_slic"]
