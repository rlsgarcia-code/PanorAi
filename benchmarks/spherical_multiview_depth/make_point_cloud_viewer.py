#!/usr/bin/env python3
"""Build a self-contained WebGL viewer for benchmark PLY point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one or more PanorAi binary PLY clouds to an HTML viewer."
    )
    parser.add_argument(
        "cloud",
        nargs="+",
        help="LABEL=/path/to/cloud.ply; each cloud becomes a selectable trace",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument(
        "--static-preview",
        type=Path,
        help="optional PNG fallback for clients that block local interactive HTML",
    )
    return parser.parse_args()


def read_panorai_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"truncated PLY header: {path}")
            header.extend(byte)
        text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise ValueError("only PanorAi binary-little-endian PLY is supported")
        vertex_line = next(
            line for line in text.splitlines() if line.startswith("element vertex ")
        )
        count = int(vertex_line.split()[-1])
        vertices = np.fromfile(stream, dtype=VERTEX_DTYPE, count=count)
    if vertices.size != count:
        raise ValueError(f"truncated PLY body: {path}")
    return vertices


def _parse_cloud(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("cloud must use LABEL=/path/to/cloud.ply")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError("cloud label and path must be non-empty")
    return label.strip(), Path(raw_path).expanduser().resolve()


def build_viewer(
    clouds: list[tuple[str, Path]], output: Path, *, max_points: int
) -> dict[str, Any]:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    import plotly.graph_objects as go

    traces = []
    rows = []
    for index, (label, path) in enumerate(clouds):
        vertices = read_panorai_ply(path)
        stride = max(1, int(np.ceil(vertices.size / max_points)))
        sampled = vertices[::stride][:max_points]
        colors = np.column_stack((sampled["red"], sampled["green"], sampled["blue"]))
        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]
        traces.append(
            go.Scatter3d(
                x=sampled["x"],
                y=sampled["y"],
                z=sampled["z"],
                mode="markers",
                marker={"size": 1.2, "color": color_strings, "opacity": 0.9},
                name=label,
                visible=index == 0,
                hoverinfo="skip",
            )
        )
        rows.append(
            {
                "label": label,
                "source": str(path),
                "dense_vertices": int(vertices.size),
                "viewer_vertices": int(sampled.size),
                "stride": stride,
            }
        )
    buttons = []
    for index, (label, _) in enumerate(clouds):
        visible = [item == index for item in range(len(clouds))]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}, {"title": f"P74 W121 — {label}"}],
            }
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=f"P74 W121 — {clouds[0][0]}",
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "+X right (m)",
            "yaxis_title": "+Y up (m)",
            "zaxis_title": "+Z forward (m)",
            "camera": {"eye": {"x": 0.0, "y": 0.3, "z": -1.7}},
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01}],
        annotations=[
            {
                "text": (
                    "Drag to orbit · wheel to zoom · dropdown switches clouds · "
                    "viewer is deterministically decimated; PLY remains metric"
                ),
                "x": 0.5,
                "y": 0.01,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            }
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    return {"output": str(output), "clouds": rows}


def build_static_preview(
    clouds: list[tuple[str, Path]], output: Path, *, max_points: int = 25_000
) -> None:
    """Render comparable fixed-camera views without requiring a browser."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cameras = ((18, -90, "front"), (75, -90, "top"), (25, -35, "oblique"))
    sampled_clouds = []
    for label, path in clouds:
        vertices = read_panorai_ply(path)
        stride = max(1, int(np.ceil(vertices.size / max_points)))
        sampled_clouds.append((label, vertices[::stride][:max_points]))
    coordinates = np.concatenate(
        [
            np.column_stack((vertices["x"], vertices["z"], vertices["y"]))
            for _, vertices in sampled_clouds
        ],
        axis=0,
    )
    low, high = np.quantile(coordinates, (0.005, 0.995), axis=0)
    padding = 0.05 * np.maximum(high - low, 1e-6)
    low -= padding
    high += padding
    figure = plt.figure(figsize=(16, 5.2 * len(clouds)), constrained_layout=True)
    for row, (label, sampled) in enumerate(sampled_clouds):
        colors = (
            np.column_stack((sampled["red"], sampled["green"], sampled["blue"])) / 255.0
        )
        for column, (elevation, azimuth, view_name) in enumerate(cameras):
            axis = figure.add_subplot(
                len(clouds),
                len(cameras),
                row * len(cameras) + column + 1,
                projection="3d",
            )
            axis.scatter(
                sampled["x"],
                sampled["z"],
                sampled["y"],
                c=colors,
                s=0.16,
                linewidths=0,
                depthshade=False,
            )
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(low[1], high[1])
            axis.set_zlim(low[2], high[2])
            axis.set_box_aspect(high - low)
            axis.set_title(f"{label} — {view_name}")
            axis.set_axis_off()
    figure.suptitle(
        "P74 W121 point clouds · same metric axes · central 99% display crop"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    clouds = [_parse_cloud(value) for value in args.cloud]
    result = build_viewer(clouds, args.output, max_points=args.max_points)
    if args.static_preview is not None:
        build_static_preview(clouds, args.static_preview)
    for row in result["clouds"]:
        print(
            f"{row['label']}: {row['viewer_vertices']} viewer points from "
            f"{row['dense_vertices']} PLY vertices"
        )
    print(result["output"])
    if args.static_preview is not None:
        print(args.static_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
