#!/usr/bin/env python3
"""Build an auditable visual report for six classical stereo pipelines.

The report uses one frozen ERP pair from Stanford2D3D and Matterport360. It
selects one rectified face and one textured, locally range-continuous probe per
dataset *before* comparing methods. Public pipeline intermediates are then
retained through :class:`panorai.pipeline.Pipeline`.

OpenCV StereoBM/StereoSGBM do not expose their internal cost volumes.  For
those two backends the figure therefore shows an explicitly labelled
diagnostic cost (block AD or block Birchfield--Tomasi), never a fabricated
backend intermediate.  PatchMatch exposes its final proposal/cost but not its
iteration history; its explanatory curve is a zero-slope weighted patch scan.
"""

# ruff: noqa: E402 -- benchmark runners load the source checkout explicitly

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_real_domain_validation import (
    METHODS,
    PairData,
    json_safe,
    load_public_pair,
    make_matcher,
)
from panorai.geometry import (
    GnomonicProjector,
    gnomonic_pixels_to_rays,
    rays_to_gnomonic_pixels,
)
from panorai.stereo.classical import (
    AbsoluteDifferenceCost,
    BirchfieldTomasiCost,
    BoxCostAggregation,
    RectifiedStereoGeometry,
    RectifiedStereoPair,
    StereoCostVolume,
    StereoDisparityCandidate,
    disparity_labels,
)
from panorai.stereo.panoramic import PanoramicStereoPose, RectifiedStereoViewPlanner


SCHEMA = "panorai-six-method-stage-report/v1"
DATASETS = ("stanford2d3d", "matterport360")
DISPLAY_NAMES = {
    "stereo-bm": "StereoBM",
    "ad-census": "AD-Census + CBCA + SGM",
    "stereo-sgbm": "StereoSGBM HH",
    "census-tmgm": "Census + box + tMGM",
    "elas-style": "PanorAi ELAS-style",
    "patchmatch": "PatchMatch Stereo",
}
METHOD_NOTES = {
    "stereo-bm": (
        "OpenCV StereoBM baseline. The backend returns disparity, not its cost "
        "volume; the plotted curve is a disclosed block-AD diagnostic."
    ),
    "ad-census": (
        "PanorAi AD-Census unary evidence, cross-based cost aggregation, "
        "eight-path SGM, subpixel selection and explicit validation."
    ),
    "stereo-sgbm": (
        "OpenCV full-mode StereoSGBM (HH). The backend cost volume is opaque; "
        "the plotted block Birchfield--Tomasi curve is diagnostic only."
    ),
    "census-tmgm": (
        "PanorAi Census unary evidence, box aggregation and two-dimensional "
        "tMGM regularization before subpixel selection."
    ),
    "elas-style": (
        "Inspectable PanorAi support-point/triangulation-prior composition. "
        "It is ELAS-style and is not binary-equivalent to libELAS."
    ),
    "patchmatch": (
        "Deterministic slanted-plane PatchMatch reference. The curve is a "
        "zero-slope slice; propagation and random search also test slopes."
    ),
}


@dataclass(frozen=True, slots=True)
class RectifiedSample:
    dataset_id: str
    pair_id: str
    face_id: str
    reference_rgb: np.ndarray
    source_rgb: np.ndarray
    pair: RectifiedStereoPair
    reference_range: np.ndarray
    reference_range_valid: np.ndarray
    reference_disparity: np.ndarray
    probe_yx: tuple[int, int]
    probe_texture_std: float
    probe_range_cv: float
    joint_support_fraction: float
    face_texture_std: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection_valid(result: Any) -> np.ndarray:
    valid = np.asarray(result.support_mask, dtype=bool).copy()
    if result.validity_mask is not None:
        valid &= np.asarray(result.validity_mask, dtype=bool)
    data = np.asarray(result.data)
    finite = np.isfinite(data) if data.ndim == 2 else np.all(np.isfinite(data), axis=2)
    return valid & finite


def _project_rgb(image: np.ndarray, support: np.ndarray, spec: Any) -> tuple[np.ndarray, np.ndarray]:
    value = image.astype(np.float32) / 255.0
    result = GnomonicProjector(
        spec,
        interpolation="bilinear",
        invalid_policy="renormalize",
        min_valid_weight=0.5,
    ).project(value, validity_mask=support)
    valid = _projection_valid(result)
    data = np.asarray(result.data, dtype=np.float32)
    data = np.where(valid[..., None], data, 0.0)
    return data, valid


def _project_range(value: np.ndarray, support: np.ndarray, spec: Any) -> tuple[np.ndarray, np.ndarray]:
    valid_input = support & np.isfinite(value) & (value > 0.05)
    storage = np.where(valid_input, value, 0.0).astype(np.float32)
    projector = GnomonicProjector(
        spec,
        interpolation="nearest",
        invalid_policy="propagate",
    )
    result = projector.project(storage)
    projected_validity = projector.project(valid_input.astype(np.uint8))
    valid = (
        _projection_valid(result)
        & _projection_valid(projected_validity)
        & np.asarray(projected_validity.data, dtype=bool)
        & (np.asarray(result.data) > 0.05)
    )
    data = np.where(valid, np.asarray(result.data, dtype=np.float32), np.nan)
    return data, valid


def _gray(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb[..., :3], dtype=np.float32) @ np.asarray(
        (0.2126, 0.7152, 0.0722), dtype=np.float32
    )


def _local_mean(value: np.ndarray, valid: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    total = cv2.boxFilter(
        np.where(valid, value, 0.0).astype(np.float32),
        -1,
        (window, window),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    count = cv2.boxFilter(
        valid.astype(np.float32),
        -1,
        (window, window),
        normalize=False,
        borderType=cv2.BORDER_REFLECT101,
    )
    return total / np.maximum(count, 1.0), count


def _texture_std(gray: np.ndarray, valid: np.ndarray, window: int = 9) -> np.ndarray:
    mean, count = _local_mean(gray, valid, window)
    second, _ = _local_mean(gray * gray, valid, window)
    variance = np.maximum(second - mean * mean, 0.0)
    return np.where(count >= 0.8 * window * window, np.sqrt(variance), 0.0)


def _range_cv(value: np.ndarray, valid: np.ndarray, window: int = 7) -> np.ndarray:
    safe = np.where(valid, value, 0.0).astype(np.float32)
    mean, count = _local_mean(safe, valid, window)
    second, _ = _local_mean(safe * safe, valid, window)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return np.where(
        (count >= 0.8 * window * window) & (mean > 0.05),
        std / np.maximum(mean, 1e-6),
        np.inf,
    )


def _geometry(spec: Any, baseline: float) -> RectifiedStereoGeometry:
    height, width = spec.output_shape_hw
    fx = 0.5 * (width - 1) / math.tan(math.radians(spec.hfov_deg) / 2.0)
    fy = 0.5 * (height - 1) / math.tan(math.radians(spec.vfov_deg) / 2.0)
    return RectifiedStereoGeometry(
        focal_length_x=fx,
        focal_length_y=fy,
        baseline=baseline,
        principal_point_x=0.5 * (width - 1),
        principal_point_y=0.5 * (height - 1),
    )


def _covisible_disparity(
    radial_range: np.ndarray,
    reference_valid: np.ndarray,
    source_range: np.ndarray,
    source_valid: np.ndarray,
    reference_spec: Any,
    source_spec: Any,
    rotation_source_from_reference: np.ndarray,
    translation_source_from_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = radial_range.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    rays = np.asarray(
        gnomonic_pixels_to_rays(pixels, reference_spec).rays_xyz,
        dtype=np.float64,
    )
    safe_range = np.where(reference_valid, radial_range, 0.0).ravel()
    points_reference = rays * safe_range[:, None]
    points_source = (
        rotation_source_from_reference @ points_reference.T
    ).T + translation_source_from_reference
    expected_source_range = np.linalg.norm(points_source, axis=1)
    safe_norm = np.maximum(expected_source_range, 1e-12)
    source_rays = points_source / safe_norm[:, None]
    projected = rays_to_gnomonic_pixels(source_rays, source_spec)
    target_xy = np.asarray(projected.pixels_xy, dtype=np.float64)
    target_x = target_xy[:, 0]
    target_y = target_xy[:, 1]
    target_inside = (
        np.asarray(projected.valid, dtype=bool)
        & (target_x >= 0.0)
        & (target_x <= width - 1.0)
        & (target_y >= 0.0)
        & (target_y <= height - 1.0)
    )
    sample_x = np.clip(
        np.rint(np.where(np.isfinite(target_x), target_x, 0.0)).astype(np.int64),
        0,
        width - 1,
    )
    sample_y = np.clip(
        np.rint(np.where(np.isfinite(target_y), target_y, 0.0)).astype(np.int64),
        0,
        height - 1,
    )
    observed = source_range[sample_y, sample_x]
    observed_valid = source_valid[sample_y, sample_x]
    tolerance = np.maximum(0.15, 0.08 * expected_source_range)
    visible = (
        reference_valid.ravel()
        & target_inside
        & observed_valid
        & np.isfinite(observed)
        & (np.abs(observed - expected_source_range) <= tolerance)
        & (np.abs(target_y - pixels[:, 1]) <= 1.25)
    )
    disparity = pixels[:, 0] - target_x
    disparity = np.where(visible, disparity, np.nan).reshape(height, width)
    return disparity.astype(np.float32), visible.reshape(height, width)


def _choose_probe(
    pair: RectifiedStereoPair,
    radial_range: np.ndarray,
    range_valid: np.ndarray,
    disparity: np.ndarray,
    max_disparity: int,
) -> tuple[tuple[int, int], float, float]:
    height, width = pair.shape_hw
    texture = _texture_std(pair.reference, pair.reference_support, 9)
    continuity = _range_cv(radial_range, range_valid, 7)
    yy, xx = np.indices((height, width))
    target_x = np.rint(xx - np.where(np.isfinite(disparity), disparity, 0.0)).astype(
        np.int32
    )
    target_inside = (target_x >= 0) & (target_x < width)
    source_at_target = pair.source_support[
        yy,
        np.clip(target_x, 0, width - 1),
    ]
    interior = (
        (yy >= 8)
        & (yy < height - 8)
        & (xx >= max_disparity + 8)
        & (xx < width - 8)
    )
    feasible = (
        range_valid
        & pair.reference_support
        & target_inside
        & source_at_target
        & np.isfinite(disparity)
        & (disparity >= 2.0)
        & (disparity <= max_disparity - 2.0)
        & (xx - disparity >= 8.0)
        & interior
    )
    strict = feasible & (continuity <= 0.08)
    candidates = strict if np.any(strict) else feasible & (continuity <= 0.20)
    if not np.any(candidates):
        candidates = feasible
    if not np.any(candidates):
        raise RuntimeError("no outcome-blind reference-valid probe is available")
    score = np.where(candidates, texture / (1.0 + 4.0 * continuity), -np.inf)
    flat = int(np.argmax(score))
    y, x = np.unravel_index(flat, score.shape)
    return (int(y), int(x)), float(texture[y, x]), float(continuity[y, x])


def build_rectified_sample(
    pair: PairData,
    planner: RectifiedStereoViewPlanner,
    max_disparity: int,
) -> RectifiedSample:
    baseline = float(np.linalg.norm(pair.translation_source_from_reference))
    pose = PanoramicStereoPose(
        pair.rotation_source_from_reference,
        pair.translation_source_from_reference,
        baseline,
        "m",
        "metric",
        "supplied-registered-pose",
    )
    plan = planner.plan(pose)
    choices: list[tuple[float, Any, Any]] = []
    for item in plan.pairs:
        ref_rgb, ref_support = _project_rgb(
            pair.reference_rgb, pair.reference_support, item.reference_spec
        )
        src_rgb, src_support = _project_rgb(
            pair.source_rgb, pair.source_support, item.source_spec
        )
        joint = ref_support & src_support
        texture = min(
            float(np.std(_gray(ref_rgb)[joint])) if np.any(joint) else 0.0,
            float(np.std(_gray(src_rgb)[joint])) if np.any(joint) else 0.0,
        )
        # Face selection uses only input support and texture, never method output.
        choices.append((float(joint.mean()) * texture, item, (ref_rgb, src_rgb, ref_support, src_support, texture)))
    _, selected, payload = max(choices, key=lambda item: (item[0], -item[1].longitude_index))
    ref_rgb, src_rgb, ref_support, src_support, face_texture = payload
    reference_range, range_valid = _project_range(
        pair.reference_range,
        pair.reference_support,
        selected.reference_spec,
    )
    source_range, source_range_valid = _project_range(
        pair.source_range,
        pair.source_support,
        selected.source_spec,
    )
    geometry = _geometry(selected.reference_spec, baseline)
    rectified = RectifiedStereoPair(
        ref_rgb,
        src_rgb,
        ref_support,
        src_support,
        geometry,
    )
    gt_disparity, covisible = _covisible_disparity(
        reference_range,
        range_valid,
        source_range,
        source_range_valid,
        selected.reference_spec,
        selected.source_spec,
        pair.rotation_source_from_reference,
        pair.translation_source_from_reference,
    )
    probe, probe_texture, probe_cv = _choose_probe(
        rectified,
        reference_range,
        range_valid,
        gt_disparity,
        max_disparity,
    )
    return RectifiedSample(
        dataset_id=pair.dataset_id,
        pair_id=pair.pair_id,
        face_id=selected.pair_id,
        reference_rgb=ref_rgb,
        source_rgb=src_rgb,
        pair=rectified,
        reference_range=reference_range,
        reference_range_valid=range_valid & covisible,
        reference_disparity=gt_disparity,
        probe_yx=probe,
        probe_texture_std=probe_texture,
        probe_range_cv=probe_cv,
        joint_support_fraction=float((ref_support & src_support).mean()),
        face_texture_std=float(face_texture),
    )


def _finite_curve(volume: StereoCostVolume, probe: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    curve = np.asarray(volume.left[probe], dtype=np.float64)
    curve[curve >= 999.0] = np.nan
    return np.asarray(volume.disparities, dtype=np.float64), curve


def _patch_curve(sample: RectifiedSample, radius: int, max_disparity: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = sample.probe_yx
    labels = np.arange(0, max_disparity + 1, dtype=np.float64)
    costs = np.full(labels.shape, np.nan, dtype=np.float64)
    ref = sample.pair.reference
    src = sample.pair.source
    ref_support = sample.pair.reference_support
    src_support = sample.pair.source_support
    center = float(ref[y, x])
    for index, disparity in enumerate(labels):
        total = 0.0
        weight_sum = 0.0
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if not 0 <= yy < ref.shape[0]:
                continue
            for dx in range(-radius, radius + 1):
                xx = x + dx
                target = int(round(xx - disparity))
                if (
                    not 0 <= xx < ref.shape[1]
                    or not 0 <= target < src.shape[1]
                    or not ref_support[yy, xx]
                    or not src_support[yy, target]
                ):
                    continue
                color_weight = math.exp(-abs(float(ref[yy, xx]) - center) / 0.08)
                spatial_weight = math.exp(
                    -(dx * dx + dy * dy) / max(2.0 * radius * radius, 1.0)
                )
                weight = color_weight * spatial_weight
                total += weight * min(abs(float(ref[yy, xx]) - float(src[yy, target])), 0.25) / 0.25
                weight_sum += weight
        if weight_sum > 0:
            costs[index] = total / weight_sum
    return labels, costs


def _diagnostic_volumes(
    method: str,
    sample: RectifiedSample,
    args: argparse.Namespace,
) -> list[tuple[str, StereoCostVolume]]:
    labels = disparity_labels(0, args.max_disparity)
    if method == "stereo-bm":
        raw = AbsoluteDifferenceCost(truncation=1.0).build(sample.pair, labels)
        block = BoxCostAggregation(max(5, args.block_size))(raw)
        return [("diagnostic pixel AD", raw), ("diagnostic block AD", block)]
    if method == "stereo-sgbm":
        raw = BirchfieldTomasiCost(truncation=0.25).build(sample.pair, labels)
        block = BoxCostAggregation(args.block_size)(raw)
        return [("diagnostic BT", raw), ("diagnostic block BT", block)]
    return []


def _candidate_minima(
    disparities: np.ndarray,
    costs: np.ndarray,
    *,
    count: int = 6,
    separation: float = 3.0,
) -> list[tuple[float, float]]:
    finite = np.isfinite(costs)
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        return []
    local = [
        int(index)
        for index in indices
        if (index == 0 or not np.isfinite(costs[index - 1]) or costs[index] <= costs[index - 1])
        and (index == costs.size - 1 or not np.isfinite(costs[index + 1]) or costs[index] <= costs[index + 1])
    ]
    if not local:
        local = [int(indices[np.argmin(costs[indices])])]
    selected: list[int] = []
    for index in sorted(local, key=lambda item: (costs[item], disparities[item])):
        if all(abs(float(disparities[index] - disparities[other])) >= separation for other in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return [(float(disparities[index]), float(costs[index])) for index in selected]


def _normalized_curve(costs: np.ndarray) -> np.ndarray:
    result = np.full(costs.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(costs)
    if not np.any(finite):
        return result
    values = costs[finite]
    low, high = np.percentile(values, (2.0, 98.0))
    if high <= low + 1e-12:
        result[finite] = 0.0
    else:
        result[finite] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return result


def _disparity_error(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid & np.isfinite(reference), np.abs(prediction - reference), np.nan)


def _method_run(
    method: str,
    sample: RectifiedSample,
    args: argparse.Namespace,
) -> tuple[Any, Mapping[str, Any], list[tuple[str, np.ndarray, np.ndarray]], dict[str, Any]]:
    matcher = make_matcher(args, method)
    pipeline = matcher.as_pipeline()
    retain = tuple(pipeline.named_steps)
    run = pipeline.run(sample.pair, retain=retain)
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    stage_maps: list[tuple[str, np.ndarray]] = []
    retained = dict(run.items())
    for name, value in retained.items():
        if isinstance(value, StereoCostVolume):
            disparities, curve = _finite_curve(value, sample.probe_yx)
            curves.append((name.replace("_", " "), disparities, curve))
            argmin = np.argmin(value.left, axis=2)
            chosen = value.disparities[argmin].astype(np.float32)
            chosen[np.min(value.left, axis=2) >= 999.0] = np.nan
            stage_maps.append((name.replace("_", " "), chosen))
        elif isinstance(value, StereoDisparityCandidate):
            stage_maps.append((name.replace("_", " "), value.left))
    if method in {"stereo-bm", "stereo-sgbm"}:
        for name, volume in _diagnostic_volumes(method, sample, args):
            disparities, curve = _finite_curve(volume, sample.probe_yx)
            curves.append((name, disparities, curve))
    elif method == "patchmatch":
        disparities, curve = _patch_curve(sample, args.patch_radius, args.max_disparity)
        curves.append(("zero-slope patch scan", disparities, curve))
    if not curves:
        raise RuntimeError(f"{method} did not expose or define a diagnostic curve")
    primary_name, primary_disparities, primary_costs = curves[0]
    candidates = _candidate_minima(primary_disparities, primary_costs)
    y, x = sample.probe_yx
    result = run.output
    predicted = float(result.disparity[y, x]) if result.validity[y, x] else math.nan
    truth = float(sample.reference_disparity[y, x])
    metadata = {
        "pipeline": pipeline.describe(),
        "records": [
            {"name": record.name, "stage": record.stage, "details": dict(record.details)}
            for record in run.records
        ],
        "stage_maps": stage_maps,
        "primary_curve": primary_name,
        "candidates": candidates,
        "probe_reference_disparity_px": truth,
        "probe_predicted_disparity_px": predicted,
        "probe_prediction_valid": bool(result.validity[y, x]),
    }
    return result, retained, curves, metadata


def _draw_input(axis: Any, image: np.ndarray, title: str, probe: tuple[int, int], *, source_x: float | None = None) -> None:
    axis.imshow(np.clip(image, 0.0, 1.0))
    y, x = probe
    axis.axhline(y, color="#00d4ff", linewidth=1.0)
    marker_x = x if source_x is None else source_x
    axis.plot(marker_x, y, marker="+", color="#ff2d95", markersize=11, markeredgewidth=2)
    axis.set_title(title)
    axis.axis("off")


def _draw_candidate_strip(
    axis: Any,
    sample: RectifiedSample,
    candidates: list[tuple[float, float]],
) -> None:
    y, x = sample.probe_yx
    radius = 6
    labels = [("A", 0.0)] + [(f"d={value:.0f}", value) for value, _ in candidates[:5]]
    patches = []
    for index, (_, disparity) in enumerate(labels):
        image = sample.reference_rgb if index == 0 else sample.source_rgb
        center_x = x if index == 0 else int(round(x - disparity))
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="constant")
        cy, cx = y + radius, center_x + radius
        patches.append(padded[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1])
    montage = np.concatenate(patches, axis=1)
    axis.imshow(np.clip(montage, 0.0, 1.0))
    width = patches[0].shape[1]
    for index, (label, _) in enumerate(labels):
        axis.text(
            index * width + width / 2,
            2.0,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
    axis.set_title("Reference patch and raw-cost candidates")
    axis.axis("off")


def _render_stage_panel(
    path: Path,
    method: str,
    sample: RectifiedSample,
    result: Any,
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    metadata: Mapping[str, Any],
) -> None:
    y, x = sample.probe_yx
    truth = float(sample.reference_disparity[y, x])
    source_x = x - truth
    candidates = list(metadata["candidates"])
    stage_maps = list(metadata["stage_maps"])
    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.05, 1.0))
    _draw_input(
        figure.add_subplot(grid[0, 0]),
        sample.reference_rgb,
        f"A · {sample.face_id} · probe ({x}, {y})",
        sample.probe_yx,
    )
    _draw_input(
        figure.add_subplot(grid[0, 1]),
        sample.source_rgb,
        f"B · GT target x={source_x:.1f}",
        sample.probe_yx,
        source_x=source_x,
    )
    _draw_candidate_strip(figure.add_subplot(grid[0, 2:]), sample, candidates)

    curve_axis = figure.add_subplot(grid[1, :2])
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, max(len(curves), 1)))
    for color, (name, disparities, costs) in zip(colors, curves, strict=True):
        curve_axis.plot(
            disparities,
            _normalized_curve(costs),
            label=name,
            color=color,
            linewidth=1.4,
        )
    primary = curves[0]
    for rank, (disparity, cost) in enumerate(candidates, start=1):
        normalized = np.interp(
            disparity,
            primary[1][np.isfinite(primary[2])],
            _normalized_curve(primary[2])[np.isfinite(primary[2])],
        )
        curve_axis.scatter(disparity, normalized, s=28, color="#ff2d95", zorder=5)
        curve_axis.text(disparity, min(1.05, normalized + 0.05), str(rank), fontsize=8, ha="center")
    curve_axis.axvline(truth, color="#16a34a", linestyle="--", linewidth=1.5, label="reference disparity")
    predicted = metadata["probe_predicted_disparity_px"]
    if predicted is not None and math.isfinite(float(predicted)):
        curve_axis.axvline(float(predicted), color="#7c3aed", linestyle=":", linewidth=1.5, label="method output")
    curve_axis.set(
        title="Matching cost along the rectified epipolar (lower is better)",
        xlabel="disparity d [px] (B column = xA − d)",
        ylabel="per-curve robust normalization",
        ylim=(-0.03, 1.10),
    )
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(fontsize=8, ncol=2)

    primary_disparities, primary_costs = primary[1], primary[2]
    # Local row-vs-disparity slice around the same probe.
    heat_axis = figure.add_subplot(grid[1, 2])
    # Reconstruct a compact explanatory slice from the primary curve by
    # evaluating neighbouring probes when an actual volume is unavailable.
    rows = np.arange(max(0, y - 14), min(sample.pair.shape_hw[0], y + 15))
    if metadata.get("primary_volume") is not None:
        volume = metadata["primary_volume"]
        slice_data = np.asarray(volume.left[rows, x, :], dtype=np.float64)
        slice_data[slice_data >= 999.0] = np.nan
        heat_title = "Unary cost slice"
    else:
        slice_data = np.tile(primary_costs[None, :], (rows.size, 1))
        heat_title = "Diagnostic curve repeated (backend volume unavailable)"
    finite = np.isfinite(slice_data)
    lo, hi = (0.0, 1.0)
    if np.any(finite):
        lo, hi = np.percentile(slice_data[finite], (2.0, 98.0))
        if hi <= lo:
            hi = lo + 1.0
    heat_axis.imshow(
        np.ma.masked_invalid(slice_data),
        aspect="auto",
        cmap="magma_r",
        vmin=lo,
        vmax=hi,
        extent=(primary_disparities[0], primary_disparities[-1], rows[-1], rows[0]),
    )
    heat_axis.axvline(truth, color="#16a34a", linestyle="--", linewidth=1.0)
    heat_axis.axhline(y, color="#00d4ff", linewidth=1.0)
    heat_axis.set(title=heat_title, xlabel="disparity [px]", ylabel="face row")

    stage_axis = figure.add_subplot(grid[1, 3])
    if stage_maps:
        labels = [name for name, _ in stage_maps]
        values = [
            float(value[y, x]) if np.isfinite(value[y, x]) else math.nan
            for _, value in stage_maps
        ]
        xs = np.arange(len(labels))
        stage_axis.plot(xs, values, marker="o", color="#2563eb")
        stage_axis.axhline(truth, color="#16a34a", linestyle="--", linewidth=1.0)
        stage_axis.set_xticks(xs, [label.replace("cost ", "") for label in labels], rotation=25, ha="right", fontsize=8)
        stage_axis.set_ylabel("selected disparity [px]")
        stage_axis.set_title("Probe decision after each exposed stage")
        stage_axis.grid(axis="y", alpha=0.2)
    else:
        stage_axis.axis("off")
        stage_axis.text(0.5, 0.5, "No intermediate disparity stage exposed", ha="center", va="center", wrap=True)

    valid_reference = sample.reference_range_valid & np.isfinite(sample.reference_disparity)
    output_valid = np.asarray(result.validity, dtype=bool)
    vmax = float(np.nanpercentile(sample.reference_disparity[valid_reference], 98))
    panels = (
        (result.disparity, output_valid, "Final disparity [px]", "turbo", 0.0, vmax),
        (sample.reference_disparity, valid_reference, "Reference disparity [px]", "turbo", 0.0, vmax),
        (_disparity_error(result.disparity, sample.reference_disparity, output_valid), output_valid & valid_reference, "Absolute disparity error [px]", "magma", 0.0, min(16.0, vmax)),
        (result.confidence, output_valid, "Final confidence / validity", "viridis", 0.0, 1.0),
    )
    for column, (data, valid, title, cmap, vmin, panel_vmax) in enumerate(panels):
        axis = figure.add_subplot(grid[2, column])
        axis.imshow(np.ma.masked_where(~valid, data), cmap=cmap, vmin=vmin, vmax=panel_vmax)
        axis.plot(x, y, marker="+", color="white", markersize=10, markeredgewidth=1.5)
        axis.set_title(title)
        axis.axis("off")

    valid_eval = output_valid & valid_reference
    delta = np.nan
    median_error = np.nan
    if np.any(valid_eval):
        error = np.abs(result.disparity[valid_eval] - sample.reference_disparity[valid_eval])
        delta = float(np.mean(error < 1.0))
        median_error = float(np.median(error))
    predicted_text = (
        f"{float(predicted):.2f}px" if math.isfinite(float(predicted)) else "invalid"
    )
    figure.suptitle(
        f"{DISPLAY_NAMES[method]} · {sample.dataset_id} · {sample.pair_id}\n"
        f"face={sample.face_id}, support={sample.joint_support_fraction:.3f}, "
        f"probe GT={truth:.2f}px, output={predicted_text}, "
        f"face valid={output_valid.mean():.3f}, median |Δd|={median_error:.2f}px, |Δd|<1={delta:.3f}",
        fontsize=13,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _load_full_records(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload.get("records", []):
            if (
                record.get("dataset_id") not in DATASETS
                or record.get("status") != "completed"
                or record.get("pose_mode") != "provided"
            ):
                continue
            records[(record["dataset_id"], record["method"])] = record
    return records


def _render_panorama_montage(
    path: Path,
    method: str,
    pairs: Mapping[str, PairData],
    records: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        len(DATASETS),
        4,
        figsize=(18, 3.4 * len(DATASETS)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, dataset_id in enumerate(DATASETS):
        pair = pairs[dataset_id]
        record = records[(dataset_id, method)]
        with np.load(record["arrays"], allow_pickle=False) as archive:
            estimate = np.asarray(archive["estimated_radial_range_m"], dtype=np.float32)
            validity = np.asarray(archive["validity"], dtype=bool)
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
            reference = np.asarray(archive["reference_radial_range_m"], dtype=np.float32)
            support = np.asarray(archive["reference_support"], dtype=bool)
        reference_valid = support & np.isfinite(reference) & (reference > 0.05)
        evaluated = reference_valid & validity & np.isfinite(estimate) & (estimate > 0.0)
        low, high = np.percentile(reference[reference_valid], (2.0, 98.0))
        relative_error = np.full(reference.shape, np.nan, dtype=np.float32)
        relative_error[evaluated] = np.abs(estimate[evaluated] - reference[evaluated]) / reference[evaluated]
        panels = (
            (pair.reference_rgb, None, f"{dataset_id} · ERP A", None, None),
            (estimate, validity, "Fused radial range [m]", "turbo", (low, high)),
            (relative_error, evaluated, "Absolute relative error", "magma", (0.0, 1.0)),
            (confidence, validity, "Confidence / validity", "viridis", (0.0, 1.0)),
        )
        for column, (data, valid, title, cmap, limits) in enumerate(panels):
            axis = axes[row, column]
            if cmap is None:
                axis.imshow(data)
            else:
                axis.imshow(
                    np.ma.masked_where(~valid, data),
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                )
            if column == 1:
                depth = record["depth"]
                title += f"\ncoverage={depth['solid_angle_coverage']:.3f}, δ<1.25={depth['delta_1_25']:.3f}"
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.suptitle(f"{DISPLAY_NAMES[method]} · complete panoramic route", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _method_metrics_table(
    method: str,
    records: Mapping[tuple[str, str], dict[str, Any]],
) -> list[str]:
    lines = [
        "| Dataset | Coverage | Correct coverage | AbsRel | Median AE [m] | δ<1.25 | Time [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_id in DATASETS:
        record = records[(dataset_id, method)]
        depth = record["depth"]
        correct = depth["solid_angle_coverage"] * depth["delta_1_25"]
        lines.append(
            f"| {dataset_id} | {depth['solid_angle_coverage']:.3f} | {correct:.3f} | "
            f"{depth['abs_rel']:.3f} | {depth['median_abs_error_m']:.3f} | "
            f"{depth['delta_1_25']:.3f} | {record['elapsed_seconds']:.2f} |"
        )
    return lines


def _write_report(
    path: Path,
    samples: Mapping[str, RectifiedSample],
    stage_records: Mapping[tuple[str, str], dict[str, Any]],
    full_records: Mapping[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Six classical stereo methods — stage-by-stage visual report",
        "",
        "> Status: Experimental test evidence. This report does not promote any method to Stable.",
        "",
        "## Scope and common protocol",
        "",
        "The cohort contains one frozen registered pair from each of Stanford2D3D and Matterport360. The same ERP resolution, supplied metric pose, rectified view planner, disparity interval, support rules, and evaluation definitions are used for all six methods.",
        "",
        "Sky/no-return is excluded from every metric by the dataset oracle: a reference sample must be inside geometric support, finite, and have positive radial range (>0.05 m). Missing method output is not silently counted as correct; coverage and conditional accuracy are reported separately.",
        "",
        "The illustrated face is selected once per dataset by joint support × input texture. The probe is selected before any method is run: it must have valid range in both views, agree with the supplied pose under a source-depth visibility check, have feasible disparity, strong local texture, and locally continuous reference range. This avoids choosing a different favorable or occluded pixel for each method.",
        "",
        f"- ERP: `{args.height}×{args.width}`",
        f"- rectified face: `{args.face_size}×{args.face_size}`; FOV `{args.face_hfov:.0f}°×{args.face_vfov:.0f}°`; overlap `{args.face_overlap:.0f}°`",
        f"- disparity labels: `0…{args.max_disparity}` px",
        "- green dashed marker: scanner/dataset reference disparity; magenta numbered markers: separated minima of the first unary/diagnostic curve; purple dotted marker: method output",
        "- cost curves are independently normalized only for display; minima and candidate coordinates remain unchanged",
        "",
        "### Frozen image/probe cohort",
        "",
        "| Dataset | Pair | Face | Joint support | Face texture σ | Probe (x,y) | Probe texture σ | Local range CV | Reference d [px] |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for dataset_id in DATASETS:
        sample = samples[dataset_id]
        y, x = sample.probe_yx
        lines.append(
            f"| {dataset_id} | `{sample.pair_id}` | `{sample.face_id}` | "
            f"{sample.joint_support_fraction:.3f} | {sample.face_texture_std:.3f} | "
            f"({x},{y}) | {sample.probe_texture_std:.3f} | {sample.probe_range_cv:.3f} | "
            f"{sample.reference_disparity[y, x]:.2f} |"
        )
    lines.extend(
        (
            "",
            "## How to read the figures",
            "",
            "The top row shows the same rectified A/B pair, the pose-defined horizontal epipolar and patches at the best separated candidates. The middle row shows the cost curve, a local row/disparity slice, and how the selected disparity changes after each exposed stage. The bottom row compares final disparity, reference disparity, absolute error, and confidence/validity. A method may preserve the correct candidate without selecting it; that distinction is visible rather than collapsed into one score.",
        )
    )
    for index, method in enumerate(METHODS, start=1):
        lines.extend(
            (
                "",
                f"## {index}. {DISPLAY_NAMES[method]}",
                "",
                METHOD_NOTES[method],
                "",
                "### Pipeline stages",
                "",
            )
        )
        topology = stage_records[(DATASETS[0], method)]["pipeline"]["steps"]
        lines.extend(
            (
                "| Stage | Implementation | Output |",
                "| --- | --- | --- |",
            )
        )
        for step in topology:
            lines.append(f"| `{step['name']}` | `{step['stage']}` | `{step['output_kind']}` |")
        lines.extend(("", "### Complete panoramic output", "", f"![{DISPLAY_NAMES[method]} panoramic output](figures/{method}/panorama-montage.png)", "", "### Quantitative result", ""))
        lines.extend(_method_metrics_table(method, full_records))
        for dataset_id in DATASETS:
            record = stage_records[(dataset_id, method)]
            lines.extend(
                (
                    "",
                    f"### {dataset_id}",
                    "",
                    f"![{DISPLAY_NAMES[method]} on {dataset_id}](figures/{method}/{dataset_id}-stages.png)",
                    "",
                    f"Probe result: reference `{record['probe_reference_disparity_px']:.2f}px`; "
                    + (
                        f"selected `{record['probe_predicted_disparity_px']:.2f}px`."
                        if record["probe_prediction_valid"]
                        else "method output invalid after validation."
                    ),
                )
            )
            candidates = ", ".join(f"{value:.1f}px" for value, _ in record["candidates"])
            lines.append(f"Raw/diagnostic separated candidates: {candidates or 'none'}.")
    lines.extend(
        (
            "",
            "## Cross-method interpretation",
            "",
            "This small cohort is explanatory, not a replacement for the frozen holdout. It reveals *where* a method changes the decision: unary evidence, local aggregation, global regularization, proposal search, or validation. Aggregate ranking must still come from the larger holdout protocol.",
            "",
            "- StereoBM and StereoSGBM are compiled baselines with opaque internal cost volumes. Their diagnostic curves must not be interpreted as exact backend state.",
            "- AD-Census, Census+tMGM, and ELAS-style expose their real cost volumes at every named stage through `Pipeline.run(..., retain=...)`.",
            "- PatchMatch exposes its selected proposal and final cost, but not every random/propagated proposal. The zero-slope scan is a controlled explanation of the local objective.",
            "- Conditional error can look good for a very sparse method. `Correct coverage = coverage × δ<1.25` prevents that from being hidden.",
            "",
            "## Reproduction and provenance",
            "",
            "See `manifest.json` beside this report for source commit, package versions, input hashes, exact configurations, figure hashes, per-stage records, and probe/candidate values. The figures and report are external benchmark artifacts and are not package contents.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, action="append", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--face-hfov", type=float, default=90.0)
    parser.add_argument("--face-vfov", type=float, default=110.0)
    parser.add_argument("--face-overlap", type=float, default=30.0)
    parser.add_argument("--max-disparity", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=5)
    parser.add_argument("--left-right-tolerance", type=float, default=1.5)
    parser.add_argument("--speckle-window-size", type=int, default=0)
    parser.add_argument("--patch-radius", type=int, default=1)
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("canonical ERP evaluation requires width == 2 * height")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = (args.height, args.width)
    pairs = {
        "stanford2d3d": load_public_pair("stanford2d3d", args.stanford_root, shape_hw),
        "matterport360": load_public_pair("matterport360", args.matterport_root, shape_hw),
    }
    full_records = _load_full_records(args.full_summary)
    expected = {(dataset_id, method) for dataset_id in DATASETS for method in METHODS}
    missing = sorted(expected - set(full_records))
    if missing:
        raise ValueError(f"full-summary records are missing: {missing}")
    planner = RectifiedStereoViewPlanner(
        hfov_deg=args.face_hfov,
        vfov_deg=args.face_vfov,
        overlap_deg=args.face_overlap,
        output_shape_hw=(args.face_size, args.face_size),
    )
    samples = {
        dataset_id: build_rectified_sample(pair, planner, args.max_disparity)
        for dataset_id, pair in pairs.items()
    }
    stage_records: dict[tuple[str, str], dict[str, Any]] = {}
    for method in METHODS:
        method_dir = args.output_dir / "figures" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for dataset_id in DATASETS:
            print(f"[{method}] {dataset_id}", flush=True)
            sample = samples[dataset_id]
            result, retained, curves, metadata = _method_run(method, sample, args)
            primary_volume = next(
                (value for value in retained.values() if isinstance(value, StereoCostVolume)),
                None,
            )
            render_metadata = dict(metadata)
            render_metadata["primary_volume"] = primary_volume
            panel = method_dir / f"{dataset_id}-stages.png"
            _render_stage_panel(panel, method, sample, result, curves, render_metadata)
            serializable = dict(metadata)
            serializable.pop("stage_maps", None)
            stage_records[(dataset_id, method)] = serializable
        montage = method_dir / "panorama-montage.png"
        _render_panorama_montage(montage, method, pairs, full_records)
    _write_report(
        args.output_dir / "REPORT.md",
        samples,
        stage_records,
        full_records,
        args,
    )
    figure_hashes = {
        str(path.relative_to(args.output_dir)): _sha256(path)
        for path in sorted((args.output_dir / "figures").rglob("*.png"))
    }
    commit = "unknown"
    try:
        import subprocess

        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        pass
    manifest = {
        "schema": SCHEMA,
        "source_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"stanford_root", "matterport_root", "output_dir"}
        },
        "samples": {
            key: {
                "dataset_id": value.dataset_id,
                "pair_id": value.pair_id,
                "face_id": value.face_id,
                "probe_yx": list(value.probe_yx),
                "probe_texture_std": value.probe_texture_std,
                "probe_range_cv": value.probe_range_cv,
                "joint_support_fraction": value.joint_support_fraction,
                "face_texture_std": value.face_texture_std,
                "reference_disparity_px": float(value.reference_disparity[value.probe_yx]),
                "provenance": pairs[key].provenance,
            }
            for key, value in samples.items()
        },
        "stage_records": {
            f"{dataset_id}/{method}": record
            for (dataset_id, method), record in stage_records.items()
        },
        "full_records": {
            f"{dataset_id}/{method}": record
            for (dataset_id, method), record in full_records.items()
        },
        "figure_sha256": figure_hashes,
        "report_sha256_before_manifest": _sha256(args.output_dir / "REPORT.md"),
        "artifact_policy": {
            "source_tree": "generator only; no input or output raster",
            "wheel": "none",
            "sdist": "none",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            json_safe(manifest),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
