#!/usr/bin/env python3
"""Generate real-image integration figures for spherical region analysis.

The source is the checksum-pinned CC0 Nature Reserve Forest ERP already
tracked by the documentation. The photometric sky candidate is a deterministic
heuristic, not semantic ground truth. Every region result is computed with the
public PanorAi API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "panorai-region-analysis-matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from panorai import EquirectangularImage  # noqa: E402
from panorai.ops import spherical as S  # noqa: E402


SOURCE = REPOSITORY_ROOT / "docs/_static/tutorials/nature-reserve-forest-erp.jpg"
SOURCE_SHA256 = "5333c2cecc55468fcbc64a252f4bc097fab064cae70bead4f76d6c2e6101c524"
WORKING_SIZE_WH = (360, 180)
EXTENT = (-180.0, 180.0, -90.0, 90.0)
MATLAB_BLUE = "#0072BD"
MATLAB_ORANGE = "#D95319"
MATLAB_YELLOW = "#EDB120"
MATLAB_PURPLE = "#7E2F8E"
MATLAB_GREEN = "#77AC30"
MATLAB_CYAN = "#4DBEEE"
MATLAB_RED = "#A2142F"
PALETTE = (
    "#FFFFFF",
    MATLAB_BLUE,
    MATLAB_ORANGE,
    MATLAB_YELLOW,
    MATLAB_PURPLE,
    MATLAB_GREEN,
    MATLAB_CYAN,
    MATLAB_RED,
)
OUTPUTS = (
    "region-analysis-real-connected-components.jpg",
    "region-analysis-real-geodesic-distance.jpg",
    "region-analysis-real-watershed.jpg",
    "region-analysis-real-boundaries.jpg",
    "region-analysis-real-fill-holes.jpg",
    "region-analysis-real-measurements.jpg",
)


@dataclass(frozen=True)
class Evidence:
    rgb: np.ndarray
    luminance: np.ndarray
    sky_candidate: np.ndarray
    component_labels: np.ndarray
    retained_components: np.ndarray
    target_component: np.ndarray
    distance_degrees: np.ndarray
    filled_candidate: np.ndarray
    gradient: np.ndarray
    markers: np.ndarray
    watershed_labels: np.ndarray
    boundaries: np.ndarray
    measurements: Any
    adjacency: Any
    threshold_luminance: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#4D4D4D",
            "axes.facecolor": "#FFFFFF",
            "axes.grid": False,
            "axes.labelcolor": "#333333",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.facecolor": "#FFFFFF",
            "text.color": "#262626",
            "xtick.color": "#4D4D4D",
            "ytick.color": "#4D4D4D",
        }
    )


def _erp_axis(axis: Any, title: str) -> None:
    axis.set_title(title, pad=8)
    axis.set_xlabel("longitude (degrees)")
    axis.set_ylabel("latitude (degrees)")
    axis.set_xticks((-180, -90, 0, 90, 180))
    axis.set_yticks((-90, -45, 0, 45, 90))
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.axvline(-180, color=MATLAB_RED, linestyle="--", linewidth=1.0)
    axis.axvline(180, color=MATLAB_RED, linestyle="--", linewidth=1.0)


def _show_rgb(axis: Any, rgb: np.ndarray, title: str) -> None:
    axis.imshow(rgb, origin="upper", extent=EXTENT, interpolation="nearest")
    _erp_axis(axis, title)


def _show_mask(axis: Any, mask: np.ndarray, title: str) -> None:
    axis.imshow(
        mask,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap(("#FFFFFF", MATLAB_BLUE)),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _erp_axis(axis, title)


def _show_labels(
    axis: Any, labels: np.ndarray, title: str, *, alpha: float = 1.0
) -> None:
    maximum = max(int(np.max(labels)), 1)
    colors = tuple(PALETTE[index % len(PALETTE)] for index in range(maximum + 1))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    axis.imshow(
        labels,
        origin="upper",
        extent=EXTENT,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        interpolation="nearest",
    )
    _erp_axis(axis, title)


def _save_jpeg(figure: Any, path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGB")
        image.save(
            path,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=True,
            exif=b"",
        )
        return image.size


def _compact_largest_components(labels: np.ndarray, count: int = 6) -> np.ndarray:
    measurements = S.measure_regions(labels)
    selected = sorted(
        measurements.regions,
        key=lambda region: (-region.solid_angle_sr, region.label),
    )[:count]
    result = np.zeros(labels.shape, dtype=np.int32)
    for output_label, region in enumerate(selected, start=1):
        result[labels == region.label] = output_label
    return result


def build_evidence(source: Path) -> Evidence:
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(f"source checksum mismatch: {source}")
    panorama = EquirectangularImage.open(source).resize(WORKING_SIZE_WH)
    rgb = np.asarray(panorama.data, dtype=np.uint8)
    normalized = rgb.astype(np.float64) / 255.0
    luminance = (
        0.2126 * normalized[..., 0]
        + 0.7152 * normalized[..., 1]
        + 0.0722 * normalized[..., 2]
    )
    blue_excess = normalized[..., 2] - 0.5 * (normalized[..., 0] + normalized[..., 1])
    threshold_luminance = float(np.quantile(luminance, 0.70))
    sky_candidate = (luminance >= threshold_luminance) & (blue_excess > 0.015)
    component_labels = S.connected_components(sky_candidate, connectivity=8)
    retained_components = _compact_largest_components(component_labels)
    component_evidence = S.measure_regions(component_labels)
    largest = max(
        component_evidence.regions,
        key=lambda region: (region.solid_angle_sr, -region.label),
    )
    target_component = component_labels == largest.label
    distance_degrees = S.geodesic_distance_transform(
        target_component,
        units="degrees",
    )
    filled_candidate = S.fill_holes(
        sky_candidate,
        max_hole_solid_angle_sr=0.002,
        connectivity=8,
    )

    smoothed = S.gaussian_blur(luminance, ksize=7, sigma=1.5)
    gradient = np.asarray(S.gradient(smoothed, operator="scharr").magnitude)
    markers = np.zeros(luminance.shape, dtype=np.int32)
    marker_windows = (
        (20, 90, 0, 180),
        (20, 90, 180, 360),
        (90, 160, 0, 180),
        (90, 160, 180, 360),
    )
    for label, (row_start, row_stop, column_start, column_stop) in enumerate(
        marker_windows,
        start=1,
    ):
        window = gradient[row_start:row_stop, column_start:column_stop]
        local = np.unravel_index(int(np.argmin(window)), window.shape)
        markers[row_start + local[0], column_start + local[1]] = label
    watershed_labels = S.watershed(gradient, markers, connectivity=8)
    boundaries = S.region_boundaries(watershed_labels, connectivity=4)
    measurements = S.measure_regions(watershed_labels)
    adjacency = S.region_adjacency_graph(watershed_labels)
    S.extract_boundaries(watershed_labels)
    return Evidence(
        rgb=rgb,
        luminance=luminance,
        sky_candidate=sky_candidate,
        component_labels=component_labels,
        retained_components=retained_components,
        target_component=target_component,
        distance_degrees=distance_degrees,
        filled_candidate=filled_candidate,
        gradient=gradient,
        markers=markers,
        watershed_labels=watershed_labels,
        boundaries=boundaries,
        measurements=measurements,
        adjacency=adjacency,
        threshold_luminance=threshold_luminance,
    )


def _components_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    _show_rgb(axes[0], evidence.rgb, "Real CC0 panorama")
    _show_mask(axes[1], evidence.sky_candidate, "Photometric sky candidate")
    _show_rgb(axes[2], evidence.rgb, "Six largest spherical components")
    masked = np.ma.masked_where(
        evidence.retained_components == 0,
        evidence.retained_components,
    )
    maximum = int(evidence.retained_components.max())
    axes[2].imshow(
        masked,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap(PALETTE[1 : maximum + 1]),
        alpha=0.68,
        interpolation="nearest",
    )
    figure.suptitle(
        "Connected components on a real spherical image",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "The mask is a reproducible photometric heuristic, not semantic ground truth.",
        ha="center",
        color="#555555",
    )
    return _save_jpeg(figure, path)


def _distance_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    _show_rgb(axes[0], evidence.rgb, "Real CC0 panorama")
    _show_rgb(axes[1], evidence.rgb, "Largest connected sky candidate")
    target = np.ma.masked_where(~evidence.target_component, evidence.target_component)
    axes[1].imshow(
        target,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap((MATLAB_BLUE,)),
        alpha=0.62,
        interpolation="nearest",
    )
    image = axes[2].imshow(
        evidence.distance_degrees,
        origin="upper",
        extent=EXTENT,
        cmap="viridis",
        vmin=0,
        vmax=180,
        interpolation="nearest",
    )
    _erp_axis(axes[2], "Great-circle distance to that region")
    figure.colorbar(image, ax=axes[2], pad=0.02, shrink=0.78).set_label("degrees")
    figure.suptitle(
        "Geodesic distance from a region observed in a real panorama",
        fontsize=15,
        fontweight="bold",
    )
    return _save_jpeg(figure, path)


def _watershed_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    _show_rgb(axes[0], evidence.rgb, "Real panorama and fixed markers")
    rows, columns = np.nonzero(evidence.markers)
    longitude = (columns + 0.5) * 360.0 / evidence.markers.shape[1] - 180.0
    latitude = 90.0 - (rows + 0.5) * 180.0 / evidence.markers.shape[0]
    axes[0].scatter(
        longitude,
        latitude,
        c=PALETTE[1:5],
        s=72,
        edgecolors="white",
        linewidths=1.0,
    )
    scale = max(float(np.percentile(evidence.gradient, 99.5)), 1e-12)
    image = axes[1].imshow(
        np.clip(evidence.gradient / scale, 0.0, 1.0),
        origin="upper",
        extent=EXTENT,
        cmap="gray_r",
        interpolation="nearest",
    )
    _erp_axis(axes[1], "Sphere-aware Scharr flooding cost")
    figure.colorbar(image, ax=axes[1], pad=0.02, shrink=0.78).set_label(
        "normalized cost"
    )
    _show_rgb(axes[2], evidence.rgb, "Marker-controlled watershed")
    _show_labels(
        axes[2],
        evidence.watershed_labels,
        "Marker-controlled watershed",
        alpha=0.48,
    )
    figure.suptitle(
        "Watershed follows real spherical image gradients",
        fontsize=15,
        fontweight="bold",
    )
    return _save_jpeg(figure, path)


def _boundaries_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    _show_rgb(axes[0], evidence.rgb, "Real CC0 panorama")
    _show_labels(axes[1], evidence.watershed_labels, "Typed watershed labels")
    _show_rgb(axes[2], evidence.rgb, "Inner boundaries over the source")
    boundary = np.ma.masked_where(~evidence.boundaries, evidence.boundaries)
    axes[2].imshow(
        boundary,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap((MATLAB_YELLOW,)),
        interpolation="nearest",
    )
    figure.suptitle(
        "Region boundaries remain continuous across the ERP seam",
        fontsize=15,
        fontweight="bold",
    )
    return _save_jpeg(figure, path)


def _fill_holes_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    changed = evidence.filled_candidate & ~evidence.sky_candidate
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    _show_rgb(axes[0], evidence.rgb, "Real CC0 panorama")
    _show_mask(axes[1], evidence.sky_candidate, "Sky candidate before filling")
    _show_rgb(axes[2], evidence.rgb, "Filled candidate and changed samples")
    filled = np.ma.masked_where(~evidence.filled_candidate, evidence.filled_candidate)
    delta = np.ma.masked_where(~changed, changed)
    axes[2].imshow(
        filled,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap((MATLAB_BLUE,)),
        alpha=0.48,
        interpolation="nearest",
    )
    axes[2].imshow(
        delta,
        origin="upper",
        extent=EXTENT,
        cmap=ListedColormap((MATLAB_YELLOW,)),
        interpolation="nearest",
    )
    figure.suptitle(
        "Solid-angle hole filling on a real photometric mask",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Blue = retained candidate; yellow = holes no larger than 0.002 sr that were filled.",
        ha="center",
        color="#555555",
    )
    return _save_jpeg(figure, path)


def _measurements_figure(evidence: Evidence, path: Path) -> tuple[int, int]:
    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.45, 0.75, 1.2))
    map_axis = figure.add_subplot(grid[0, 0])
    graph_axis = figure.add_subplot(grid[0, 1])
    table_axis = figure.add_subplot(grid[0, 2])
    _show_rgb(map_axis, evidence.rgb, "Regions and spherical centroids")
    _show_labels(
        map_axis,
        evidence.watershed_labels,
        "Regions and spherical centroids",
        alpha=0.42,
    )
    positions: dict[int, tuple[float, float]] = {}
    for region in evidence.measurements.regions:
        if region.centroid_bearing_xyz is None:
            continue
        bearing = region.centroid_bearing_xyz
        longitude = math.degrees(math.atan2(float(bearing[0]), float(bearing[2])))
        latitude = math.degrees(math.asin(float(np.clip(bearing[1], -1.0, 1.0))))
        positions[region.label] = (longitude, latitude)
        map_axis.scatter(
            (longitude,),
            (latitude,),
            marker="+",
            s=95,
            linewidths=2,
            color="black",
        )
        map_axis.text(longitude + 3, latitude + 3, f"C{region.label}")

    graph_axis.set_title("Region adjacency graph", pad=8)
    graph_positions = {
        label: (
            math.cos(2.0 * math.pi * index / len(positions)),
            math.sin(2.0 * math.pi * index / len(positions)),
        )
        for index, label in enumerate(sorted(positions))
    }
    for edge in evidence.adjacency.edges:
        one, two = edge.labels
        start = graph_positions[one]
        end = graph_positions[two]
        graph_axis.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            color="#666666",
            linewidth=1.0 + edge.shared_boundary_rad,
            zorder=1,
        )
    for label, (x_value, y_value) in graph_positions.items():
        graph_axis.scatter(
            (x_value,),
            (y_value,),
            s=500,
            color=PALETTE[label],
            edgecolor="#333333",
            zorder=2,
        )
        graph_axis.text(
            x_value,
            y_value,
            str(label),
            ha="center",
            va="center",
            fontweight="bold",
            zorder=3,
        )
    graph_axis.set_xlim(-1.35, 1.35)
    graph_axis.set_ylim(-1.35, 1.35)
    graph_axis.axis("off")

    table_axis.axis("off")
    table_axis.set_title("Geometry-aware measurements", pad=8)
    rows = [
        (
            str(region.label),
            f"{region.solid_angle_sr:.3f}",
            f"{100.0 * region.support_fraction:.1f}",
            f"{region.angular_perimeter_rad:.3f}",
        )
        for region in evidence.measurements.regions
    ]
    table = table_axis.table(
        cellText=rows,
        colLabels=("label", "area (sr)", "support (%)", "perimeter (rad)"),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(fontweight="bold")
    figure.suptitle(
        "Physical measurements and adjacency from real-image regions",
        fontsize=15,
        fontweight="bold",
    )
    return _save_jpeg(figure, path)


def _marker_directions(markers: np.ndarray) -> list[list[float]]:
    result = []
    for label in range(1, int(markers.max()) + 1):
        row, column = np.argwhere(markers == label)[0]
        result.append(
            [
                float((column + 0.5) * 360.0 / markers.shape[1] - 180.0),
                float(90.0 - (row + 0.5) * 180.0 / markers.shape[0]),
            ]
        )
    return result


def _metadata(
    evidence: Evidence,
    output_dir: Path,
    dimensions: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    component_count = int(evidence.component_labels.max())
    changed_count = int(
        np.count_nonzero(evidence.filled_candidate & ~evidence.sky_candidate)
    )
    return {
        "schema": "panorai-region-analysis-real-evidence/v1",
        "source": {
            "filename": SOURCE.name,
            "sha256": SOURCE_SHA256,
            "license": "CC0-1.0",
            "asset": "Nature Reserve Forest",
        },
        "working_size_wh": list(WORKING_SIZE_WH),
        "photometric_candidate": {
            "interpretation": "deterministic sky-like candidate; not semantic ground truth",
            "luminance_quantile": 0.70,
            "luminance_threshold": evidence.threshold_luminance,
            "blue_excess_threshold": 0.015,
            "foreground_fraction": float(evidence.sky_candidate.mean()),
        },
        "region_analysis": {
            "component_count": component_count,
            "displayed_component_count": int(evidence.retained_components.max()),
            "distance_units": "degrees",
            "watershed_marker_strategy": (
                "minimum sphere-aware Scharr cost in each of four fixed ERP windows"
            ),
            "watershed_marker_longitude_latitude_deg": _marker_directions(
                evidence.markers
            ),
            "watershed_region_count": len(evidence.measurements.regions),
            "adjacency_edge_count": len(evidence.adjacency.edges),
            "fill_holes_max_solid_angle_sr": 0.002,
            "filled_pixel_count": changed_count,
        },
        "operations": [
            "EquirectangularImage.open",
            "EquirectangularImage.resize",
            "connected_components",
            "geodesic_distance_transform",
            "watershed",
            "region_boundaries",
            "fill_holes",
            "measure_regions",
            "region_adjacency_graph",
            "extract_boundaries",
        ],
        "outputs": {
            filename: {
                "dimensions_wh": list(dimensions[filename]),
                "sha256": _sha256(output_dir / filename),
            }
            for filename in OUTPUTS
        },
    }


def main() -> None:
    args = _arguments()
    _configure_style()
    evidence = build_evidence(args.source)
    builders = (
        _components_figure,
        _distance_figure,
        _watershed_figure,
        _boundaries_figure,
        _fill_holes_figure,
        _measurements_figure,
    )
    dimensions: dict[str, tuple[int, int]] = {}
    for filename, builder in zip(OUTPUTS, builders, strict=True):
        path = args.output_dir / filename
        dimensions[filename] = builder(evidence, path)
        print(path)
    metadata_path = args.output_dir / "region-analysis-real-metadata.json"
    metadata = _metadata(evidence, args.output_dir, dimensions)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
