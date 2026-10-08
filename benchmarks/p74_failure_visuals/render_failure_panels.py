#!/usr/bin/env python3
"""Render visual diagnostics for failed frozen P74 coarse-pose cells."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.relative_pose_frontend import run_benchmark as frontend  # noqa: E402
from panorai.estimators import spherical_tangent_sampson_error  # noqa: E402
from panorai.geometry import rays_to_erp_pixels  # noqa: E402

DEFAULT_METHODS = Path(
    "/private/tmp/panorai-val010-p74-canonical/pairs-method-inputs.jsonl"
)
DEFAULT_REFERENCES = Path(
    "/private/tmp/panorai-val010-p74-canonical/pairs-evaluation.jsonl"
)
DEFAULT_PREDICTIONS = Path("/private/tmp/panorai-val010-p74-confirm/predictions.jsonl")
DEFAULT_EVALUATIONS = Path(
    "/private/tmp/panorai-val010-p74-confirm-eval/evaluated-cells.jsonl"
)
DEFAULT_OUTPUT = Path("/private/tmp/panorai-val012-p74-failure-visuals")
VARIANT = "coarse-dog-gaussian"
DISPLAY_SHAPE_HW = (512, 1024)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _index(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _short_view(value: str) -> str:
    return value.rsplit("+", 1)[-1]


def _safe_pair_name(method: dict[str, Any], index: int) -> str:
    left = _short_view(method["from_view_id"])
    right = _short_view(method["to_view_id"])
    group = method["spatial_group_id"].split("::")[-1].replace("_concluido", "")
    return f"{index:02d}_{group}_{left}_to_{right}".replace("+", "-")


def _ground_truth_errors(
    prediction: dict[str, Any], reference: dict[str, Any]
) -> np.ndarray:
    bearings_a = np.asarray(
        prediction.get("match_bearings_a", []), dtype=np.float64
    ).reshape(-1, 3)
    bearings_b = np.asarray(
        prediction.get("match_bearings_b", []), dtype=np.float64
    ).reshape(-1, 3)
    if not len(bearings_a):
        return np.empty(0, dtype=np.float64)
    adapter = frontend._PANORAI_FROM_DATASET[reference["dataset_id"]]
    rotation = (
        adapter.T
        @ np.asarray(reference["reference"]["R_to_from"], dtype=np.float64)
        @ adapter
    )
    translation = adapter.T @ np.asarray(
        reference["reference"]["t_to_from_m"], dtype=np.float64
    )
    tx = np.asarray(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.degrees(
        spherical_tangent_sampson_error(bearings_a, bearings_b, tx @ rotation)
    )


def _display_image(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    height, width = DISPLAY_SHAPE_HW
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(
        support.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    display = small.astype(np.float32)
    tint = np.asarray((78.0, 40.0, 92.0), dtype=np.float32)
    display[~valid] = 0.35 * display[~valid] + 0.65 * tint
    return np.clip(display, 0.0, 255.0).astype(np.uint8)


def _load_images(
    method: dict[str, Any], variant: frontend.Variant, cache: dict[str, tuple]
) -> tuple[np.ndarray, np.ndarray]:
    displays = []
    for role in ("from", "to"):
        path = method[f"{role}_rgb_path"]
        if path not in cache:
            image, support = frontend._read_pair_side(method, role, variant)
            if support is None:
                support = np.ones(image.shape[:2], dtype=bool)
            cache[path] = (image, support)
        displays.append(_display_image(*cache[path]))
    return displays[0], displays[1]


def _match_pixels(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bearings_a = np.asarray(
        prediction.get("match_bearings_a", []), dtype=np.float64
    ).reshape(-1, 3)
    bearings_b = np.asarray(
        prediction.get("match_bearings_b", []), dtype=np.float64
    ).reshape(-1, 3)
    return (
        rays_to_erp_pixels(bearings_a, DISPLAY_SHAPE_HW).pixels_xy,
        rays_to_erp_pixels(bearings_b, DISPLAY_SHAPE_HW).pixels_xy,
    )


def _colors(errors: np.ndarray) -> np.ndarray:
    result = np.empty((len(errors), 4), dtype=np.float64)
    result[errors <= 1.0] = matplotlib.colors.to_rgba("#20b26b", 0.92)
    result[(errors > 1.0) & (errors <= 2.0)] = matplotlib.colors.to_rgba(
        "#f1b82d", 0.92
    )
    result[errors > 2.0] = matplotlib.colors.to_rgba("#ef4444", 0.82)
    return result


def _show_panorama(
    axis: plt.Axes,
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    title: str,
) -> None:
    axis.imshow(image)
    if len(pixels):
        axis.scatter(
            pixels[:, 0],
            pixels[:, 1],
            c=colors,
            s=22,
            linewidths=0.45,
            edgecolors="white",
        )
    axis.axhline(
        DISPLAY_SHAPE_HW[0] * (150.0 / 180.0),
        color="#d8b4fe",
        linewidth=1.0,
        linestyle="--",
    )
    axis.set_title(title, fontsize=11, loc="left")
    axis.set_xlim(0, DISPLAY_SHAPE_HW[1])
    axis.set_ylim(DISPLAY_SHAPE_HW[0], 0)
    axis.axis("off")


def _draw_connectors(
    axis: plt.Axes,
    image_a: np.ndarray,
    image_b: np.ndarray,
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    colors: np.ndarray,
) -> None:
    width = image_a.shape[1]
    gap = 24
    canvas = np.full((image_a.shape[0], width * 2 + gap, 3), 24, dtype=np.uint8)
    canvas[:, :width] = image_a
    canvas[:, width + gap :] = image_b
    axis.imshow(canvas)
    for point_a, point_b, color in zip(pixels_a, pixels_b, colors):
        axis.plot(
            (point_a[0], point_b[0] + width + gap),
            (point_a[1], point_b[1]),
            color=color,
            linewidth=0.65,
            alpha=0.62,
        )
    if len(pixels_a):
        axis.scatter(pixels_a[:, 0], pixels_a[:, 1], c=colors, s=13, edgecolors="none")
        axis.scatter(
            pixels_b[:, 0] + width + gap,
            pixels_b[:, 1],
            c=colors,
            s=13,
            edgecolors="none",
        )
    axis.axvspan(width, width + gap, color="#111827")
    axis.text(width / 2, 24, "FROM", color="white", ha="center", fontsize=9)
    axis.text(
        width + gap + width / 2,
        24,
        "TO",
        color="white",
        ha="center",
        fontsize=9,
    )
    axis.set_title(
        "Correspondences (line = one match; color = GT epipolar residual)",
        fontsize=11,
        loc="left",
    )
    axis.axis("off")


def _spatial_occupancy(pixels: np.ndarray) -> float:
    """Return occupied fraction of a 12x5 grid over observed P74 support."""

    if not len(pixels):
        return 0.0
    columns = np.clip(
        np.floor(pixels[:, 0] / DISPLAY_SHAPE_HW[1] * 12.0).astype(int), 0, 11
    )
    observed_height = DISPLAY_SHAPE_HW[0] * (150.0 / 180.0)
    rows = np.clip(np.floor(pixels[:, 1] / observed_height * 5.0).astype(int), 0, 4)
    return len(set(zip(rows.tolist(), columns.tolist()))) / 60.0


def _metric_text(
    evaluation: dict[str, Any],
    errors: np.ndarray,
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
) -> str:
    rotation = evaluation.get("rotation_error_deg")
    translation = evaluation.get("translation_direction_error_deg")
    pose = "returned" if evaluation["pose_returned"] else "NOT returned"
    rotation_text = "n/a" if rotation is None else f"{rotation:.2f} deg"
    translation_text = "n/a" if translation is None else f"{translation:.2f} deg"
    return "\n".join(
        (
            f"Pose: {pose} | evaluator stage: {evaluation['failure_stage']}",
            f"Matches: {evaluation['match_count']} | estimator inliers: "
            f"{evaluation['inlier_count']}",
            f"GT epipolar <=1 deg: {np.mean(errors <= 1.0):.1%} | "
            f"<=2 deg: {np.mean(errors <= 2.0):.1%}",
            f"GT epipolar median: {np.median(errors):.2f} deg",
            f"Rotation error: {rotation_text} | translation error: {translation_text}",
            f"Occupied 12x5 bins: FROM {_spatial_occupancy(pixels_a):.1%} | "
            f"TO {_spatial_occupancy(pixels_b):.1%}",
        )
    )


def _render_detail(
    output_path: Path,
    method: dict[str, Any],
    prediction: dict[str, Any],
    evaluation: dict[str, Any],
    reference: dict[str, Any],
    images: tuple[np.ndarray, np.ndarray],
) -> None:
    errors = _ground_truth_errors(prediction, reference)
    pixels_a, pixels_b = _match_pixels(prediction)
    if not (len(errors) == len(pixels_a) == len(pixels_b)):
        raise RuntimeError(
            "match bearings and epipolar residuals have different lengths"
        )
    colors = _colors(errors)
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.08, 0.62))
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])
    links = fig.add_subplot(grid[1, :])
    histogram = fig.add_subplot(grid[2, 0])
    notes = fig.add_subplot(grid[2, 1])

    _show_panorama(
        left,
        images[0],
        pixels_a,
        colors,
        f"FROM: {method['from_view_id']}",
    )
    _show_panorama(
        right,
        images[1],
        pixels_b,
        colors,
        f"TO: {method['to_view_id']}",
    )
    _draw_connectors(links, *images, pixels_a, pixels_b, colors)

    bins = np.asarray((0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0))
    clipped = np.minimum(errors, np.nextafter(90.0, 0.0))
    histogram.hist(clipped, bins=bins, color="#64748b", edgecolor="white")
    histogram.axvspan(0.0, 1.0, color="#20b26b", alpha=0.2)
    histogram.axvspan(1.0, 2.0, color="#f1b82d", alpha=0.2)
    histogram.axvspan(2.0, 90.0, color="#ef4444", alpha=0.08)
    histogram.set_xscale("symlog", linthresh=1.0)
    histogram.set_xticks(bins[1:])
    histogram.set_xticklabels(("1", "2", "5", "10", "20", "45", ">=90"))
    histogram.set_xlabel("Ground-truth epipolar residual (degrees)")
    histogram.set_ylabel("Match count")
    histogram.grid(axis="y", alpha=0.2)

    notes.axis("off")
    notes.text(
        0.0,
        0.94,
        _metric_text(evaluation, errors, pixels_a, pixels_b),
        va="top",
        family="monospace",
        fontsize=11,
        linespacing=1.5,
    )
    legend = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#20b26b",
            label="<=1 deg",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#f1b82d",
            label="1-2 deg",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ef4444",
            label=">2 deg",
        ),
        Line2D(
            [0],
            [0],
            color="#d8b4fe",
            linestyle="--",
            label="30 deg unsupported south cap",
        ),
    )
    notes.legend(handles=legend, loc="lower left", frameon=False, ncol=2)
    fig.suptitle(
        f"P74 strict-validation failure | {method['stratum']} | {VARIANT}",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=140, facecolor="white")
    plt.close(fig)


def _render_contact_sheet(
    output_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(15, 3.0 * len(cases)), constrained_layout=True
    )
    for row_index, case in enumerate(cases):
        method = case["method"]
        evaluation = case["evaluation"]
        prediction = case["prediction"]
        errors = case["errors"]
        pixels_a, pixels_b = _match_pixels(prediction)
        colors = _colors(errors)
        metric = (
            f"matches={evaluation['match_count']}  inliers={evaluation['inlier_count']}  "
            f"GT<=2deg={np.mean(errors <= 2.0):.0%}  "
            f"median={np.median(errors):.1f}deg  stage={evaluation['failure_stage']}"
        )
        _show_panorama(
            axes[row_index, 0],
            case["images"][0],
            pixels_a,
            colors,
            f"{row_index + 1}. {_short_view(method['from_view_id'])} | {metric}",
        )
        _show_panorama(
            axes[row_index, 1],
            case["images"][1],
            pixels_b,
            colors,
            f"{_short_view(method['to_view_id'])} | pose="
            f"{'returned' if evaluation['pose_returned'] else 'none'}",
        )
    fig.suptitle(
        "P74 coarse-dog-gaussian: all seven strict-validation failures\n"
        "green <=1 deg, amber 1-2 deg, red >2 deg against ground truth",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)


def _validate_frozen_metrics(evaluation: dict[str, Any], errors: np.ndarray) -> None:
    if len(errors) != evaluation["match_count"]:
        raise RuntimeError("rendered match count differs from frozen evaluation")
    expected = evaluation["epipolar_match_fraction_2deg"]
    actual = float(np.mean(errors <= 2.0))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"epipolar oracle mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    methods = _index(_read_jsonl(args.methods), "pair_id")
    references = _index(_read_jsonl(args.references), "pair_id")
    predictions = _index(_read_jsonl(args.predictions), "pair_id", "variant")
    failed = [
        row
        for row in _read_jsonl(args.evaluations)
        if row["variant"] == VARIANT and not row["strict_success"]
    ]
    if len(failed) != 7:
        raise RuntimeError(
            f"expected seven frozen strict failures, found {len(failed)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cases: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, evaluation in enumerate(failed, start=1):
        pair_id = evaluation["pair_id"]
        method = methods[(pair_id,)]
        reference = references[(pair_id,)]
        prediction = predictions[(pair_id, VARIANT)]
        variant = frontend.Variant(**prediction["configuration"])
        images = _load_images(method, variant, image_cache)
        errors = _ground_truth_errors(prediction, reference)
        _validate_frozen_metrics(evaluation, errors)
        filename = _safe_pair_name(method, index) + ".png"
        output_path = args.output_dir / filename
        _render_detail(
            output_path,
            method,
            prediction,
            evaluation,
            reference,
            images,
        )
        cases.append(
            {
                "method": method,
                "prediction": prediction,
                "evaluation": evaluation,
                "errors": errors,
                "images": images,
            }
        )
        manifest.append(
            {
                "pair_id": pair_id,
                "panel": str(output_path),
                "match_count": evaluation["match_count"],
                "inlier_count": evaluation["inlier_count"],
                "pose_returned": evaluation["pose_returned"],
                "failure_stage": evaluation["failure_stage"],
                "epipolar_match_fraction_2deg": float(np.mean(errors <= 2.0)),
                "epipolar_match_median_error_deg": float(np.median(errors)),
                "rotation_error_deg": evaluation.get("rotation_error_deg"),
                "translation_direction_error_deg": evaluation.get(
                    "translation_direction_error_deg"
                ),
            }
        )

    contact_sheet = args.output_dir / "contact-sheet.png"
    _render_contact_sheet(contact_sheet, cases)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"rendered {len(cases)} failed pairs")
    print(contact_sheet)
    for row in manifest:
        print(row["panel"])


if __name__ == "__main__":
    main()
