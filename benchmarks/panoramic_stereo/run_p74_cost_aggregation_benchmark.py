#!/usr/bin/env python3
"""Factor matching cost and spatial aggregation on scanner-verified P74 pixels.

This is a private-data development benchmark. Scanner depth is used only to
define the evaluation population and correct inverse-range label; it is never
made available to a matching cost or aggregation operation.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_real_domain_validation import (
    _p74_archive,
    relative_pose,
    sha256,
)
from benchmarks.panoramic_stereo.stability_protocol import (
    build_evaluation_population,
    summarize_range_prediction,
)
from panorai.geometry import (
    GnomonicProjector,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
)
from panorai.stereo.classical._operations import (
    _INVALID_COST,
    _census,
    _hamming,
    _mgm_one,
    _sgm_one,
)
from panorai.stereo.panoramic import OverlappingCubemapPlanner


SCHEMA = "panorai-p74-cost-aggregation-benchmark/v1"
COSTS = ("ad", "current", "zncc", "census", "ad-census")
AGGREGATORS = (
    "raw",
    "box",
    "bilateral",
    "sgm4",
    "tmgm",
    "box-sgm4",
    "box-tmgm",
    "bilateral-sgm4",
    "bilateral-tmgm",
)
EQUATORIAL_FACES = ("front", "right", "back", "left")


def _gray(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        result = value.astype(np.float32)
    else:
        result = cv2.cvtColor(value[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    if result.max(initial=0.0) > 1.0:
        result /= 255.0
    return result


def _remap_wrapped(
    value: np.ndarray, pixels_xy: np.ndarray, *, nearest: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    height, width = value.shape[:2]
    pixels = np.asarray(pixels_xy, dtype=np.float32)
    x = np.mod(pixels[..., 0], width).astype(np.float32)
    y = pixels[..., 1].astype(np.float32)
    geometric = np.isfinite(x) & np.isfinite(y) & (y >= -0.5) & (y <= height - 0.5)
    safe_x = np.where(geometric, x, 0.0).astype(np.float32)
    safe_y = np.where(geometric, y, -1.0).astype(np.float32)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    sampled = cv2.remap(
        value,
        safe_x,
        safe_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled, geometric


def _valid_box_mean(
    values: np.ndarray, valid: np.ndarray, window_size: int
) -> np.ndarray:
    weighted = np.where(valid, values, 0.0).astype(np.float32)
    sums = cv2.boxFilter(
        weighted,
        -1,
        (window_size, window_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    counts = cv2.boxFilter(
        valid.astype(np.float32),
        -1,
        (window_size, window_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    return np.where(
        counts > 0.5 * window_size**2,
        sums / np.maximum(counts, 1.0),
        _INVALID_COST,
    ).astype(np.float32)


def _valid_joint_bilateral_mean(
    values: np.ndarray,
    valid: np.ndarray,
    guidance: np.ndarray,
    *,
    radius: int = 3,
    sigma_color: float = 0.08,
    sigma_space: float = 3.0,
) -> np.ndarray:
    """Aggregate valid costs without smoothing across strong image edges."""

    height, width = values.shape
    pad = radius
    padded_values = np.pad(values, pad, mode="reflect")
    padded_valid = np.pad(valid, pad, mode="reflect")
    padded_guidance = np.pad(guidance, pad, mode="reflect")
    weighted_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    spatial_sum = 0.0
    center = np.asarray(guidance, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space))
            spatial_sum += spatial
            ys = slice(pad + dy, pad + dy + height)
            xs = slice(pad + dx, pad + dx + width)
            neighbour = padded_values[ys, xs]
            neighbour_valid = padded_valid[ys, xs]
            guide_delta = padded_guidance[ys, xs] - center
            color = np.exp(
                -(guide_delta * guide_delta) / (2.0 * sigma_color * sigma_color)
            )
            weight = spatial * color * neighbour_valid
            weighted_sum += weight * neighbour
            weight_sum += weight
    usable = weight_sum > 0.10 * spatial_sum
    return np.where(
        usable,
        weighted_sum / np.maximum(weight_sum, 1e-12),
        _INVALID_COST,
    ).astype(np.float32)


def _zncc_cost(
    reference: np.ndarray,
    warped: np.ndarray,
    valid: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    mask = valid.astype(np.float32)
    kernel = (window_size, window_size)

    def summed(value: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            np.where(valid, value, 0.0).astype(np.float32),
            -1,
            kernel,
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )

    count = cv2.boxFilter(
        mask,
        -1,
        kernel,
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    ref_sum = summed(reference)
    src_sum = summed(warped)
    ref2_sum = summed(reference * reference)
    src2_sum = summed(warped * warped)
    product_sum = summed(reference * warped)
    denominator = np.maximum(count, 1.0)
    covariance = product_sum - ref_sum * src_sum / denominator
    variance_ref = np.maximum(ref2_sum - ref_sum * ref_sum / denominator, 0.0)
    variance_src = np.maximum(src2_sum - src_sum * src_sum / denominator, 0.0)
    scale = np.sqrt(variance_ref * variance_src)
    usable = (count > 0.7 * window_size**2) & (scale > 1e-6)
    correlation = np.zeros_like(reference, dtype=np.float32)
    correlation[usable] = covariance[usable] / scale[usable]
    cost = 0.5 * (1.0 - np.clip(correlation, -1.0, 1.0))
    return np.where(usable, cost, _INVALID_COST).astype(np.float32)


def build_cost_volumes(
    reference: np.ndarray,
    warped_stack: np.ndarray,
    validity_stack: np.ndarray,
    *,
    census_radius: int = 2,
) -> dict[str, np.ndarray]:
    """Return predeclared raw HWD cost volumes for one plane-sweep face."""

    reference = np.asarray(reference, dtype=np.float32)
    warped_stack = np.asarray(warped_stack, dtype=np.float32)
    validity_stack = np.asarray(validity_stack, dtype=bool)
    if warped_stack.ndim != 3 or validity_stack.shape != warped_stack.shape:
        raise ValueError("warped_stack and validity_stack must have shape DHW")
    if warped_stack.shape[1:] != reference.shape:
        raise ValueError("warped stack must match reference shape")

    depth_count, height, width = warped_stack.shape
    result = {
        name: np.full((height, width, depth_count), _INVALID_COST, np.float32)
        for name in COSTS
    }
    reference_east = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    reference_north = -cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    reference_census, reference_census_valid = _census(reference, census_radius)
    bits = (2 * census_radius + 1) ** 2 - 1
    support_kernel = np.ones((2 * census_radius + 1, 2 * census_radius + 1), np.uint8)

    for index in range(depth_count):
        warped = warped_stack[index]
        valid = validity_stack[index]
        difference = np.abs(reference - warped)
        ad = np.minimum(difference, 0.20) / 0.20
        east = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        north = -cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gradient = np.minimum(
            0.5 * (np.abs(reference_east - east) + np.abs(reference_north - north)),
            1.0,
        )
        current = 0.75 * np.minimum(difference, 1.0) + 0.25 * gradient
        source_census, source_census_valid = _census(warped, census_radius)
        neighbourhood_valid = cv2.erode(
            valid.astype(np.uint8), support_kernel, iterations=1
        ).astype(bool)
        census_valid = (
            valid & neighbourhood_valid & reference_census_valid & source_census_valid
        )
        census = _hamming(reference_census, source_census) / float(bits)
        zncc = _zncc_cost(reference, warped, valid)
        values = {
            "ad": ad,
            "current": current,
            "zncc": zncc,
            "census": census,
            "ad-census": 0.25 * ad + 0.75 * census,
        }
        for name, value in values.items():
            local_valid = census_valid if "census" in name else valid
            if name == "zncc":
                local_valid = zncc < _INVALID_COST
            result[name][..., index] = np.where(
                local_valid, value, _INVALID_COST
            ).astype(np.float32)
    return result


def aggregate_cost_volume(
    volume_hwd: np.ndarray,
    guidance: np.ndarray,
    method: str,
    *,
    box_size: int = 7,
) -> np.ndarray:
    """Apply one controlled spatial aggregation to an HWD volume."""

    volume = np.asarray(volume_hwd, dtype=np.float32)
    if method == "raw":
        return volume.copy()
    if method in {"box", "box-sgm4", "box-tmgm"}:
        output = np.empty_like(volume)
        for index in range(volume.shape[2]):
            valid = volume[..., index] < _INVALID_COST
            output[..., index] = _valid_box_mean(volume[..., index], valid, box_size)
        if method == "box":
            return output
        volume = output
    if method in {"bilateral", "bilateral-sgm4", "bilateral-tmgm"}:
        output = np.empty_like(volume)
        for index in range(volume.shape[2]):
            valid = volume[..., index] < _INVALID_COST
            output[..., index] = _valid_joint_bilateral_mean(
                volume[..., index], valid, guidance
            )
        if method == "bilateral":
            return output
        volume = output
    if method in {"sgm4", "box-sgm4", "bilateral-sgm4"}:
        return _sgm_one(
            volume,
            guidance,
            ((0, 1), (0, -1), (1, 0), (-1, 0)),
            0.03,
            0.18,
            0.08,
        ).astype(np.float32)
    if method in {"tmgm", "box-tmgm", "bilateral-tmgm"}:
        return _mgm_one(volume, guidance, 0.03, 0.18, 0.08, "tmgm").astype(np.float32)
    raise ValueError(f"unsupported aggregation method: {method}")


def evaluate_cost_volume(
    volume_hwd: np.ndarray,
    inverse_ranges: np.ndarray,
    truth_range: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate winner and correct-label rank without using truth for selection."""

    costs = np.asarray(volume_hwd, dtype=np.float32)
    mask = np.asarray(evaluation_mask, dtype=bool).copy()
    true_inverse = np.reciprocal(np.maximum(truth_range, 1e-12))
    truth_index = np.argmin(
        np.abs(true_inverse[..., None] - inverse_ranges[None, None, :]), axis=2
    )
    valid_cost = np.min(costs, axis=2) < _INVALID_COST
    mask &= np.isfinite(truth_range)
    selected = np.argmin(costs, axis=2)
    predicted_range = np.reciprocal(inverse_ranges[selected])
    winner_error = np.abs(selected - truth_index)

    lower = np.maximum(truth_index - 1, 0)
    upper = np.minimum(truth_index + 1, inverse_ranges.size - 1)
    yy, xx = np.indices(truth_index.shape)
    truth_band_cost = np.minimum.reduce(
        (
            costs[yy, xx, lower],
            costs[yy, xx, truth_index],
            costs[yy, xx, upper],
        )
    )
    rank = np.sum(costs < truth_band_cost[..., None], axis=2)
    ordered = np.partition(costs, 1, axis=2)
    best = ordered[..., 0]
    second = ordered[..., 1]
    margin = (second - best) / np.maximum(np.abs(second), 1e-6)
    shifted = costs - np.min(costs, axis=2, keepdims=True)
    probability = np.exp(-np.minimum(shifted, 50.0) / 0.10)
    probability /= np.maximum(probability.sum(axis=2, keepdims=True), 1e-12)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=2
    ) / math.log(costs.shape[2])

    prediction_valid = mask & valid_cost & np.isfinite(predicted_range)
    count = int(mask.sum())
    if count == 0:
        return {"evaluated": 0}
    ratio = np.maximum(
        predicted_range[prediction_valid] / truth_range[prediction_valid],
        truth_range[prediction_valid] / predicted_range[prediction_valid],
    )
    absolute = np.abs(predicted_range[prediction_valid] - truth_range[prediction_valid])
    return {
        "evaluated": count,
        "predicted_valid": int(prediction_valid.sum()),
        "coverage": float(prediction_valid.sum() / count),
        "winner_within_1_label": float(
            np.count_nonzero(prediction_valid & (winner_error <= 1)) / count
        ),
        "truth_band_top_1": float(
            np.count_nonzero(prediction_valid & (rank < 1)) / count
        ),
        "truth_band_top_3": float(
            np.count_nonzero(prediction_valid & (rank < 3)) / count
        ),
        "truth_band_top_5": float(
            np.count_nonzero(prediction_valid & (rank < 5)) / count
        ),
        "delta_1_25": float(np.count_nonzero(ratio < 1.25) / count),
        "median_abs_error_m": (
            float(np.median(absolute)) if absolute.size else math.nan
        ),
        "mean_abs_error_m": (float(np.mean(absolute)) if absolute.size else math.nan),
        "median_margin": (
            float(np.median(margin[prediction_valid]))
            if np.any(prediction_valid)
            else math.nan
        ),
        "median_normalized_entropy": (
            float(np.median(entropy[prediction_valid]))
            if np.any(prediction_valid)
            else math.nan
        ),
    }


def _face_stack(
    spec,
    reference_rgb: np.ndarray,
    reference_support: np.ndarray,
    reference_range: np.ndarray,
    source_rgb: np.ndarray,
    source_support: np.ndarray,
    source_range: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    inverse_ranges: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | float | str],
]:
    linear = GnomonicProjector(
        spec,
        interpolation="bilinear",
        invalid_policy="renormalize",
        min_valid_weight=0.5,
    )
    nearest = GnomonicProjector(spec, interpolation="nearest")
    reference_face = linear.project(
        _gray(reference_rgb), validity_mask=reference_support
    )
    truth_face = nearest.project(np.nan_to_num(reference_range, nan=0.0))
    scanner_return_face = nearest.project(
        (
            reference_support & np.isfinite(reference_range) & (reference_range > 0.05)
        ).astype(np.uint8)
    )
    support = np.asarray(reference_face.support_mask, dtype=bool)
    if reference_face.validity_mask is not None:
        support &= np.asarray(reference_face.validity_mask, dtype=bool)

    height, width = spec.output_shape_hw
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((xx, yy), axis=-1)
    rays = np.asarray(gnomonic_pixels_to_rays(pixels, spec).rays_xyz)
    source_gray = _gray(source_rgb)
    warped_stack = np.empty((inverse_ranges.size, height, width), np.float32)
    validity_stack = np.zeros_like(warped_stack, dtype=bool)
    for index, inverse_range in enumerate(inverse_ranges):
        points_source = (rays / float(inverse_range)) @ rotation.T + translation
        projected = rays_to_erp_pixels(points_source, source_gray.shape)
        warped, geometric = _remap_wrapped(source_gray, projected.pixels_xy)
        sampled_support, _ = _remap_wrapped(
            source_support.astype(np.uint8), projected.pixels_xy, nearest=True
        )
        warped_stack[index] = warped
        validity_stack[index] = (
            geometric
            & np.asarray(projected.valid, dtype=bool)
            & (sampled_support >= 1)
            & support
        )

    truth = np.asarray(truth_face.data, dtype=np.float32)
    true_points_source = (rays * truth[..., None]) @ rotation.T + translation
    true_projected = rays_to_erp_pixels(true_points_source, source_range.shape)
    sampled_source_range, geometric = _remap_wrapped(
        np.nan_to_num(source_range, nan=0.0),
        true_projected.pixels_xy,
        nearest=True,
    )
    sampled_source_support, _ = _remap_wrapped(
        source_support.astype(np.uint8), true_projected.pixels_xy, nearest=True
    )
    predicted_source_range = np.linalg.norm(true_points_source, axis=-1)
    visibility_tolerance = np.maximum(0.10, 0.05 * predicted_source_range)
    observable = (
        geometric
        & np.asarray(true_projected.valid, dtype=bool)
        & (sampled_source_support >= 1)
        & (sampled_source_range > 0.05)
        & (
            np.abs(sampled_source_range - predicted_source_range)
            <= visibility_tolerance
        )
    )
    # The 100-degree faces overlap by ten degrees. Evaluate only the central
    # ninety-degree core, while leaving the overlap available to aggregation.
    border = max(1, int(round(0.05 * min(height, width))))
    core = np.zeros((height, width), dtype=bool)
    core[border:-border, border:-border] = True
    # P74 has no semantic sky label. A positive scanner return is the explicit,
    # sensor-derived non-sky proxy; the 0.0001 m no-return sentinel is excluded.
    # This decision is independent of RGB and of the stereo prediction.
    explicit_non_sky = np.asarray(scanner_return_face.data, dtype=bool)
    population = build_evaluation_population(
        truth,
        explicit_non_sky=explicit_non_sky,
        geometric_support=support & core,
        observable=observable,
        min_range_m=0.30,
        max_range_m=12.0,
        non_sky_policy="scanner-return proxy; no RGB-derived sky decision",
    )
    return (
        np.asarray(reference_face.data, dtype=np.float32),
        truth,
        population.mask,
        warped_stack,
        validity_stack,
        explicit_non_sky,
        observable,
        population.audit,
    )


def _combine_metrics(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    total = sum(int(record.get("evaluated", 0)) for record in records)
    if total == 0:
        return {"evaluated": 0}
    predicted_valid = sum(int(record.get("predicted_valid", 0)) for record in records)
    keys = tuple(
        key
        for key in records[0]
        if key not in {"evaluated", "predicted_valid", "coverage"}
    )
    return {
        "evaluated": total,
        "predicted_valid": predicted_valid,
        "coverage": float(predicted_valid / total),
        **{
            key: float(
                sum(float(record[key]) * int(record["evaluated"]) for record in records)
                / total
            )
            for key in keys
        },
    }


def _render_summary(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("winner_within_1_label", "truth_band_top_5", "delta_1_25")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    values = summary["combined"]
    for axis, metric in zip(axes, metrics, strict=True):
        matrix = np.asarray(
            [
                [values[cost][aggregation][metric] for aggregation in AGGREGATORS]
                for cost in COSTS
            ]
        )
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(AGGREGATORS)), AGGREGATORS, rotation=30)
        axis.set_yticks(range(len(COSTS)), COSTS)
        axis.set_title(metric.replace("_", " "))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle("P74 G100→G099 · scanner-visible matching cost × aggregation")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _render_population_audit(
    path: Path,
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(EQUATORIAL_FACES), 3, figsize=(9, 10), constrained_layout=True
    )
    titles = ("reference face", "scanner non-sky proxy", "evaluated")
    for row, face_id in enumerate(EQUATORIAL_FACES):
        values = panels[face_id]
        for column, (title, value) in enumerate(zip(titles, values, strict=True)):
            axis = axes[row, column]
            axis.imshow(value, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"{face_id} · {title}")
            axis.axis("off")
    figure.suptitle("P74 G100→G099 evaluation population · sky/no-return excluded")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p74-root",
        type=Path,
        default=Path("data/private/p74"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--hypotheses", type=int, default=48)
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("ERP width must be twice its height")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.height, args.width)
    prefix = "P-74+MD-05_concluido_326+"
    reference_path = args.p74_root / "npzs" / f"{prefix}G100.npz"
    source_path = args.p74_root / "npzs" / f"{prefix}G099.npz"
    reference = _p74_archive(reference_path, shape)
    source = _p74_archive(source_path, shape)
    rotation, translation = relative_pose(
        reference[3], reference[4], source[3], source[4]
    )
    inverse_ranges = np.linspace(
        1.0 / 12.0, 1.0 / 0.30, args.hypotheses, dtype=np.float32
    )
    planner = OverlappingCubemapPlanner(face_size=args.face_size, face_fov_deg=100.0)
    face_records: dict[str, dict[str, dict[str, Any]]] = {}
    combined_parts: dict[tuple[str, str], list[dict[str, float | int]]] = defaultdict(
        list
    )
    timings: dict[str, dict[str, float]] = defaultdict(dict)
    peaks: dict[str, dict[str, int]] = defaultdict(dict)
    population_audits: dict[str, dict[str, int | float | str]] = {}
    population_panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    promotion_truth: list[np.ndarray] = []
    promotion_prediction: list[np.ndarray] = []
    promotion_validity: list[np.ndarray] = []
    for face_id, face_spec in planner.plan():
        if face_id not in EQUATORIAL_FACES:
            continue
        print(f"[{face_id}] building plane sweep", flush=True)
        (
            ref_face,
            truth,
            evaluation,
            warped,
            valid,
            explicit_non_sky,
            _observable,
            population_audit,
        ) = _face_stack(
            face_spec,
            reference[0],
            reference[1],
            reference[2],
            source[0],
            source[1],
            source[2],
            rotation,
            translation,
            inverse_ranges,
        )
        population_audits[face_id] = population_audit
        population_panels[face_id] = (
            ref_face,
            explicit_non_sky.astype(np.float32),
            evaluation.astype(np.float32),
        )
        raw_volumes = build_cost_volumes(ref_face, warped, valid)
        face_records[face_id] = {}
        for cost_name in COSTS:
            face_records[face_id][cost_name] = {}
            for aggregation in AGGREGATORS:
                tracemalloc.start()
                started = time.perf_counter()
                aggregated = aggregate_cost_volume(
                    raw_volumes[cost_name], ref_face, aggregation
                )
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                metrics = evaluate_cost_volume(
                    aggregated, inverse_ranges, truth, evaluation
                )
                face_records[face_id][cost_name][aggregation] = metrics
                combined_parts[(cost_name, aggregation)].append(metrics)
                if cost_name == "ad-census" and aggregation == "box-tmgm":
                    selected = np.argmin(aggregated, axis=2)
                    predicted = np.reciprocal(inverse_ranges[selected])
                    predicted_valid = np.min(aggregated, axis=2) < _INVALID_COST
                    promotion_truth.append(truth[evaluation])
                    promotion_prediction.append(predicted[evaluation])
                    promotion_validity.append(predicted_valid[evaluation])
                timings[cost_name][aggregation] = (
                    timings[cost_name].get(aggregation, 0.0) + elapsed
                )
                peaks[cost_name][aggregation] = max(
                    peaks[cost_name].get(aggregation, 0), peak
                )
                print(
                    f"  {cost_name:10s} {aggregation:5s} "
                    f"n={metrics.get('evaluated', 0)} "
                    f"top1±1={metrics.get('winner_within_1_label', math.nan):.3f} "
                    f"delta={metrics.get('delta_1_25', math.nan):.3f}",
                    flush=True,
                )

    combined = {
        cost: {
            aggregation: _combine_metrics(combined_parts[(cost, aggregation)])
            for aggregation in AGGREGATORS
        }
        for cost in COSTS
    }
    concatenated_truth = np.concatenate(promotion_truth)[None, :]
    concatenated_prediction = np.concatenate(promotion_prediction)[None, :]
    concatenated_validity = np.concatenate(promotion_validity)[None, :]
    promotion_population = np.ones_like(concatenated_validity, dtype=bool)
    promotion_metrics = summarize_range_prediction(
        concatenated_truth,
        concatenated_prediction,
        population=promotion_population,
        prediction_validity=concatenated_validity,
    )
    audit_integer_keys = (
        "total_pixels",
        "explicit_non_sky_pixels",
        "geometric_support_pixels",
        "finite_in_range_pixels",
        "observable_pixels",
        "excluded_as_sky_pixels",
    )
    population_totals = {
        key: sum(int(audit[key]) for audit in population_audits.values())
        for key in audit_integer_keys
    }
    population_totals["evaluation_fraction_of_raster"] = float(
        population_totals["observable_pixels"] / population_totals["total_pixels"]
    )
    summary = {
        "schema": SCHEMA,
        "evidence_target": "dirty-source-checkout-private-p74",
        "pair": "G100-to-G099",
        "pose": "NPZ metric ground truth",
        "evaluation_oracle": ("scanner same-surface visibility within max(0.10m, 5%)"),
        "evaluation_population": {
            "non_sky_policy": (
                "explicit scanner-return proxy; no RGB/value-derived sky mask"
            ),
            "fixed_denominator": True,
            "missing_prediction_counts_as_failure": True,
            "per_face": population_audits,
            "totals": population_totals,
        },
        "configuration": {
            "erp_shape_hw": list(shape),
            "face_size": args.face_size,
            "face_fov_deg": 100.0,
            "evaluated_faces": list(EQUATORIAL_FACES),
            "overlap_excluded_from_metrics": True,
            "range_m": [0.30, 12.0],
            "hypotheses": args.hypotheses,
            "costs": list(COSTS),
            "aggregators": list(AGGREGATORS),
        },
        "assets": {
            "G100": {"path": str(reference_path), "sha256": sha256(reference_path)},
            "G099": {"path": str(source_path), "sha256": sha256(source_path)},
        },
        "face_records": face_records,
        "combined": combined,
        "combined_metric_note": (
            "Ratios use the exact fixed population. Matrix medians are "
            "population-weighted face medians; promotion_candidate contains "
            "the exact concatenated-pixel median."
        ),
        "promotion_candidate": {
            "cost": "ad-census",
            "aggregation": "box-tmgm",
            "metrics": promotion_metrics,
        },
        "aggregation_elapsed_seconds": timings,
        "aggregation_peak_traced_bytes": peaks,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _render_summary(args.output_dir / "cost-aggregation-summary.png", summary)
    _render_population_audit(
        args.output_dir / "evaluation-population.png", population_panels
    )
    print(json.dumps(combined, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
