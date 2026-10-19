#!/usr/bin/env python3
"""Generate reproducible scientific SVGs for spherical region analysis.

Every output is computed with the public PanorAi API over analytic ERP
fixtures.  The figures contain no external media and deliberately expose the
longitude seam, support and validity instead of presenting planar mock-ups.
"""

from __future__ import annotations

import argparse
from html import escape
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np

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

from panorai.core.numpy import spherical_solid_angle_weights  # noqa: E402
from panorai.geometry import erp_pixels_to_rays  # noqa: E402
from panorai.ops import spherical as S  # noqa: E402


SHAPE_HW = (36, 72)
EXTENT = (-180.0, 180.0, -90.0, 90.0)
MATLAB_BLUE = "#0072BD"
MATLAB_ORANGE = "#D95319"
MATLAB_YELLOW = "#EDB120"
MATLAB_PURPLE = "#7E2F8E"
MATLAB_GREEN = "#77AC30"
MATLAB_CYAN = "#4DBEEE"
MATLAB_RED = "#A2142F"
INVALID_GRAY = "#D9D9D9"
LABEL_COLORS = (
    "#FFFFFF",
    MATLAB_BLUE,
    MATLAB_ORANGE,
    MATLAB_YELLOW,
    MATLAB_PURPLE,
    MATLAB_GREEN,
    MATLAB_CYAN,
)

OUTPUTS = (
    "region-analysis-connected-components.svg",
    "region-analysis-geodesic-distance.svg",
    "region-analysis-watershed.svg",
    "region-analysis-boundaries.svg",
    "region-analysis-fill-holes.svg",
    "region-analysis-measurements.svg",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/_static/tutorials",
    )
    return parser.parse_args()


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
            "svg.fonttype": "none",
            "svg.hashsalt": "panorai-region-analysis-v1",
            "text.color": "#262626",
            "xtick.color": "#4D4D4D",
            "ytick.color": "#4D4D4D",
        }
    )


def _save_svg(
    figure: Any,
    path: Path,
    *,
    title: str,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "PanorAi documentation generator"},
    )
    plt.close(figure)
    content = path.read_text(encoding="utf-8")
    svg_start = content.index("<svg")
    content = content[svg_start:]
    generated_ids = dict.fromkeys(re.findall(r'id="([mp][0-9a-f]{10})"', content))
    for index, generated_id in enumerate(generated_ids):
        content = content.replace(generated_id, f"panorai-svg-{index}")
    close = content.index(">") + 1
    accessible = (
        f"\n <title>{escape(title)}</title>\n <desc>{escape(description)}</desc>"
    )
    serialized = content[:close] + accessible + content[close:]
    serialized = "\n".join(line.rstrip() for line in serialized.splitlines()) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _erp_axis(axis: Any, title: str, *, seam: bool = True) -> None:
    axis.set_title(title, pad=8)
    axis.set_xlabel("longitude (degrees)")
    axis.set_ylabel("latitude (degrees)")
    axis.set_xticks((-180, -90, 0, 90, 180))
    axis.set_yticks((-90, -45, 0, 45, 90))
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_aspect("auto")
    if seam:
        axis.axvline(-180, color=MATLAB_RED, linestyle="--", linewidth=1.4)
        axis.axvline(180, color=MATLAB_RED, linestyle="--", linewidth=1.4)


def _show_mask(
    axis: Any,
    mask: np.ndarray,
    title: str,
    *,
    validity: np.ndarray | None = None,
) -> None:
    display = np.ma.array(mask.astype(float), mask=False)
    if validity is not None:
        display.mask = ~validity
    cmap = ListedColormap(("#FFFFFF", MATLAB_BLUE))
    cmap.set_bad(INVALID_GRAY)
    axis.imshow(
        display,
        origin="upper",
        extent=EXTENT,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _erp_axis(axis, title)


def _show_labels(
    axis: Any,
    labels: np.ndarray,
    title: str,
    *,
    validity: np.ndarray | None = None,
) -> None:
    display = np.ma.array(labels, mask=False)
    if validity is not None:
        display.mask = ~validity
    maximum = max(int(np.max(labels)), 1)
    colors = LABEL_COLORS[: maximum + 1]
    cmap = ListedColormap(colors)
    cmap.set_bad(INVALID_GRAY)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    axis.imshow(
        display,
        origin="upper",
        extent=EXTENT,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    _erp_axis(axis, title)


def _rays(shape_hw: tuple[int, int] = SHAPE_HW) -> np.ndarray:
    rows, columns = np.indices(shape_hw)
    pixels = np.stack((columns, rows), axis=-1)
    return np.asarray(erp_pixels_to_rays(pixels, shape_hw), dtype=np.float64)


def _longitude_latitude(bearings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.arctan2(bearings[..., 0], bearings[..., 2])
    latitude = np.arcsin(np.clip(bearings[..., 1], -1.0, 1.0))
    return longitude, latitude


def _connected_components_figure(path: Path) -> None:
    mask = np.zeros(SHAPE_HW, dtype=bool)
    mask[12:25, :8] = True
    mask[12:25, -8:] = True
    mask[8:18, 28:45] = True
    validity = np.ones(SHAPE_HW, dtype=bool)
    validity[7:20, 36] = False
    labels = S.connected_components(mask, validity=validity, connectivity=4)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), constrained_layout=True)
    _show_mask(axes[0], mask, "Input mask and invalid barrier", validity=validity)
    axes[0].text(
        0,
        78,
        "gray = invalid (absent from topology)",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    _show_labels(
        axes[1],
        labels,
        f"Periodic components: {int(labels.max())} labels",
        validity=validity,
    )
    axes[1].annotate(
        "one component crosses the seam",
        xy=(177, 8),
        xytext=(42, 66),
        arrowprops={"arrowstyle": "->", "color": MATLAB_RED},
        color=MATLAB_RED,
    )
    figure.suptitle(
        "Connected components use spherical ERP topology",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Spherical connected components",
        description=(
            "An analytic ERP mask is labeled with periodic longitude. A region "
            "touching both image edges remains one component, while an invalid "
            "gray band disconnects a central region."
        ),
    )


def _geodesic_distance_figure(path: Path) -> None:
    target = np.zeros(SHAPE_HW, dtype=bool)
    target[SHAPE_HW[0] // 2, 1] = True
    distance = S.geodesic_distance_transform(target, units="degrees")

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), constrained_layout=True)
    _show_mask(axes[0], target, "Selected target pixel centre")
    axes[0].scatter((-172.5,), (-2.5,), s=80, color=MATLAB_RED, marker="x", zorder=4)
    image = axes[1].imshow(
        distance,
        origin="upper",
        extent=EXTENT,
        cmap="viridis",
        vmin=0,
        vmax=180,
        interpolation="nearest",
    )
    longitude = np.linspace(-177.5, 177.5, SHAPE_HW[1])
    latitude = np.linspace(87.5, -87.5, SHAPE_HW[0])
    axes[1].contour(
        longitude,
        latitude,
        distance,
        levels=(30, 60, 90, 120, 150),
        colors="white",
        linewidths=0.7,
        alpha=0.8,
    )
    _erp_axis(axes[1], "Nearest-centre great-circle distance")
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.02, shrink=0.86)
    colorbar.set_label("angular distance (degrees)")
    figure.suptitle(
        "Geodesic distance follows the sphere, not the ERP rectangle",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Geodesic distance transform",
        description=(
            "The distance to one ERP pixel centre is measured as a great-circle "
            "angle. The map is continuous across the longitude seam and reaches "
            "180 degrees at the antipodal direction."
        ),
    )


def _watershed_figure(path: Path) -> None:
    rows, columns = np.indices(SHAPE_HW)
    longitude = (columns + 0.5) * 360.0 / SHAPE_HW[1] - 180.0
    latitude = 90.0 - (rows + 0.5) * 180.0 / SHAPE_HW[0]
    ridge = np.exp(-((longitude / 24.0) ** 2))
    polar = 0.22 * (np.abs(latitude) / 90.0) ** 2
    gradient = (0.12 + 0.78 * ridge + polar).astype(np.float64)
    markers = np.zeros(SHAPE_HW, dtype=np.int32)
    markers[SHAPE_HW[0] // 2, 16] = 1
    markers[SHAPE_HW[0] // 2, 55] = 2
    partition = S.watershed(gradient, markers, connectivity=4)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), constrained_layout=True)
    image = axes[0].imshow(
        gradient,
        origin="upper",
        extent=EXTENT,
        cmap="gray_r",
        interpolation="bilinear",
    )
    marker_rows, marker_columns = np.nonzero(markers)
    marker_lon = (marker_columns + 0.5) * 360.0 / SHAPE_HW[1] - 180.0
    marker_lat = 90.0 - (marker_rows + 0.5) * 180.0 / SHAPE_HW[0]
    axes[0].scatter(
        marker_lon,
        marker_lat,
        c=(MATLAB_BLUE, MATLAB_ORANGE),
        edgecolors="black",
        linewidths=0.7,
        s=85,
        zorder=4,
    )
    _erp_axis(axes[0], "Gradient and fixed markers")
    figure.colorbar(image, ax=axes[0], pad=0.02, shrink=0.86).set_label("flooding cost")
    _show_labels(axes[1], partition, "Deterministic minimax partition")
    axes[1].axvline(0, color="black", linewidth=1.0, alpha=0.7)
    figure.suptitle(
        "Marker-controlled watershed floods the periodic ERP graph",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Spherical marker-controlled watershed",
        description=(
            "Two fixed markers flood an analytic ERP gradient. The partition is "
            "computed on a longitude-periodic graph with deterministic minimax, "
            "path-length and marker-label tie breaking."
        ),
    )


def _boundaries_figure(path: Path) -> None:
    labels = np.zeros(SHAPE_HW, dtype=np.int32)
    labels[11:26, :9] = 1
    labels[11:26, -9:] = 1
    labels[8:18, 28:43] = 2
    boundaries = S.region_boundaries(labels, connectivity=4)
    lifted = S.extract_boundaries(labels)

    figure = plt.figure(figsize=(14.2, 4.3), constrained_layout=True)
    grid = figure.add_gridspec(1, 3)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2], projection="mollweide"),
    )
    _show_labels(axes[0], labels, "Typed label field")
    _show_mask(axes[1], boundaries, "Inner boundary pixels")
    for boundary, color in zip(
        lifted.boundaries,
        (MATLAB_BLUE, MATLAB_ORANGE),
        strict=True,
    ):
        longitude, latitude = _longitude_latitude(boundary.bearings_xyz)
        axes[2].scatter(
            longitude,
            latitude,
            s=13,
            color=color,
            label=f"label {boundary.label}",
        )
    axes[2].set_title("Boundary centres lifted to bearings", pad=18)
    axes[2].grid(True, color="#D0D0D0", linewidth=0.5)
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.21), ncol=2)
    figure.suptitle(
        "Boundaries retain spherical position and angular length",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Spherical region boundaries",
        description=(
            "Positive ERP labels produce inner boundary pixels without a false "
            "edge at the longitude seam. The same samples are lifted to unit "
            "bearings and shown on a Mollweide sphere."
        ),
    )


def _fill_holes_figure(path: Path) -> None:
    mask = np.zeros(SHAPE_HW, dtype=bool)
    mask[7:30, 9:63] = True
    mask[16:19, 30:33] = False
    mask[11:15, 48:52] = False
    validity = np.ones(SHAPE_HW, dtype=bool)
    validity[10:12, 47:49] = False
    weights = spherical_solid_angle_weights(SHAPE_HW)
    threshold = float(weights[16:19, 30:33].sum()) * 1.05
    filled = S.fill_holes(
        mask,
        max_hole_solid_angle_sr=threshold,
        validity=validity,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.1), constrained_layout=True)
    _show_mask(axes[0], mask, "Input mask", validity=validity)
    _show_mask(axes[1], ~validity, "Missing-data boundary")
    _show_mask(axes[2], filled, "After solid-angle hole filling", validity=validity)
    axes[2].annotate(
        "small closed hole filled",
        xy=(-22, 2),
        xytext=(-155, 65),
        arrowprops={"arrowstyle": "->", "color": MATLAB_GREEN},
        color="#4F772D",
    )
    axes[2].annotate(
        "hole touching invalid data preserved",
        xy=(68, 27),
        xytext=(5, -73),
        arrowprops={"arrowstyle": "->", "color": MATLAB_RED},
        color=MATLAB_RED,
    )
    figure.suptitle(
        f"Hole filling requires a physical threshold ({threshold:.4f} sr)",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Solid-angle constrained spherical hole filling",
        description=(
            "A small closed background component is filled because its solid "
            "angle is below the explicit threshold. A second component touching "
            "invalid gray data is preserved because a sphere has no privileged "
            "exterior."
        ),
    )


def _measurement_labels() -> np.ndarray:
    labels = np.zeros(SHAPE_HW, dtype=np.int32)
    labels[7:18, 7:26] = 1
    labels[7:18, 26:43] = 2
    labels[18:30, 15:34] = 3
    labels[18:30, 34:55] = 4
    return labels


def _measurements_figure(path: Path) -> None:
    labels = _measurement_labels()
    measurements = S.measure_regions(labels)
    graph = S.region_adjacency_graph(labels)

    figure = plt.figure(figsize=(15.4, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.55, 0.8, 1.15))
    map_axis = figure.add_subplot(grid[0, 0])
    graph_axis = figure.add_subplot(grid[0, 1])
    table_axis = figure.add_subplot(grid[0, 2])
    _show_labels(map_axis, labels, "Region labels and spherical centroids")

    for region in measurements.regions:
        assert region.centroid_bearing_xyz is not None
        longitude, latitude = _longitude_latitude(region.centroid_bearing_xyz)
        longitude_deg = math.degrees(float(longitude))
        latitude_deg = math.degrees(float(latitude))
        map_axis.scatter(
            (longitude_deg,),
            (latitude_deg,),
            marker="+",
            s=90,
            linewidths=2,
            color="black",
            zorder=4,
        )
        map_axis.text(
            longitude_deg + 3,
            latitude_deg + 3,
            f"C{region.label}",
            fontsize=8,
            fontweight="bold",
        )

    graph_axis.set_title("Region adjacency graph", pad=8)
    graph_positions = {
        1: (-0.75, 0.62),
        2: (0.75, 0.62),
        3: (-0.55, -0.62),
        4: (0.65, -0.62),
    }
    for edge in graph.edges:
        one, two = edge.labels
        x_values = (graph_positions[one][0], graph_positions[two][0])
        y_values = (graph_positions[one][1], graph_positions[two][1])
        graph_axis.plot(
            x_values,
            y_values,
            color="#666666",
            linewidth=1.0 + edge.shared_boundary_rad * 2.0,
            zorder=1,
        )
        midpoint = (np.mean(x_values), np.mean(y_values))
        graph_axis.text(
            midpoint[0],
            midpoint[1],
            f"{edge.shared_boundary_rad:.2f} rad",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    for label, (x_value, y_value) in graph_positions.items():
        graph_axis.scatter(
            (x_value,),
            (y_value,),
            s=520,
            color=LABEL_COLORS[label],
            edgecolor="#333333",
            zorder=2,
        )
        graph_axis.text(
            x_value,
            y_value,
            str(label),
            ha="center",
            va="center",
            color="white" if label in {1, 2, 4} else "#222222",
            fontweight="bold",
            zorder=3,
        )
    graph_axis.set_xlim(-1.25, 1.25)
    graph_axis.set_ylim(-1.05, 1.05)
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
        for region in measurements.regions
    ]
    table = table_axis.table(
        cellText=rows,
        colLabels=("label", "area (sr)", "support (%)", "perimeter (rad)"),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.15, 0.27, 0.28, 0.30),
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
        "Region measurements use solid angle and angular boundaries",
        fontsize=15,
        fontweight="bold",
    )
    _save_svg(
        figure,
        path,
        title="Spherical region measurements and adjacency",
        description=(
            "Four analytic ERP regions are shown with their solid-angle-weighted "
            "centroids, adjacency graph, shared angular boundary lengths, solid "
            "angles, support fractions and angular perimeters."
        ),
    )


def main() -> None:
    args = _arguments()
    _configure_style()
    builders = (
        _connected_components_figure,
        _geodesic_distance_figure,
        _watershed_figure,
        _boundaries_figure,
        _fill_holes_figure,
        _measurements_figure,
    )
    for filename, builder in zip(OUTPUTS, builders, strict=True):
        builder(args.output_dir / filename)
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
