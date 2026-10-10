#!/usr/bin/env python3
"""Densify frozen G/M sparse geometry with immutable local spherical anchors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metric_landmark_ba.anchor_densification import (  # noqa: E402
    AnchorDensificationOptions,
    densify_harmonic_log_range,
    densify_log_range_anchors,
)
from benchmarks.metric_landmark_ba.p74_protocol import (  # noqa: E402
    sample_erp_nearest,
    sha256,
)

SCHEMA = "panorai-anchor-preserving-spherical-densification/v2"
TARGETS = {
    "G": "P-74+MD-05_concluido_326+G046",
    "M": "P-74+MD-08_missing_files+M-014",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val048-root", required=True, type=Path)
    parser.add_argument("--val020-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(TARGETS),
        default=tuple(TARGETS),
    )
    parser.add_argument("--angular-radius-deg", type=float, default=1.05)
    parser.add_argument("--normal-stride", type=int, default=16)
    parser.add_argument("--ply-stride", type=int, default=16)
    return parser.parse_args()


def _dense_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    *,
    region: np.ndarray | None = None,
    row_chunk: int = 64,
) -> dict[str, float | int]:
    count = 0
    abs_rel = 0.0
    squared = 0.0
    delta_1 = 0.0
    log_sum = 0.0
    log_squared = 0.0
    height = prediction.shape[0]
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        pred = np.asarray(prediction[start:stop], dtype=np.float64)
        target = np.asarray(truth[start:stop], dtype=np.float64)
        # ``validity`` may already be a writable bool array.  Always copy it:
        # the region intersection below must not narrow later evaluations.
        selected = np.array(validity[start:stop], dtype=bool, copy=True)
        if region is not None:
            selected &= region[start:stop]
        selected &= (
            np.isfinite(pred) & np.isfinite(target) & (pred > 0.0) & (target > 0.0)
        )
        if not selected.any():
            continue
        pred = pred[selected]
        target = target[selected]
        error = pred - target
        log_error = np.log(pred) - np.log(target)
        count += int(pred.size)
        abs_rel += float(np.sum(np.abs(error) / target))
        squared += float(np.sum(error**2))
        delta_1 += float(np.sum(np.maximum(pred / target, target / pred) < 1.25))
        log_sum += float(np.sum(log_error))
        log_squared += float(np.sum(log_error**2))
    if count == 0:
        return {"count": 0}
    mean_log = log_sum / count
    return {
        "count": count,
        "abs_rel": abs_rel / count,
        "rmse_m": math.sqrt(squared / count),
        "delta_1": delta_1 / count,
        "log_rmse": math.sqrt(log_squared / count),
        "si_log_rmse": math.sqrt(max(0.0, log_squared / count - mean_log**2)),
    }


def _normal_error_deg(
    prediction: np.ndarray,
    truth: np.ndarray,
    validity: np.ndarray,
    *,
    stride: int,
) -> dict[str, float | int]:
    from panorai.geometry import erp_pixels_to_rays

    if stride < 2:
        raise ValueError("normal stride must be at least two")
    rows = np.arange(1, prediction.shape[0] - 1, stride, dtype=np.int64)
    columns = np.arange(0, prediction.shape[1], stride, dtype=np.int64)
    xx, yy = np.meshgrid(columns, rows)
    center_rows = yy.ravel()
    center_columns = xx.ravel()
    neighbours = {
        "west": (center_rows, np.mod(center_columns - 1, prediction.shape[1])),
        "east": (center_rows, np.mod(center_columns + 1, prediction.shape[1])),
        "north": (center_rows - 1, center_columns),
        "south": (center_rows + 1, center_columns),
    }

    def points(array: np.ndarray, row: np.ndarray, column: np.ndarray) -> np.ndarray:
        pixels = np.column_stack((column, row))
        rays = erp_pixels_to_rays(pixels, prediction.shape)
        ranges = np.asarray(array[row, column], dtype=np.float64)
        return rays * ranges[:, None]

    pred = {name: points(prediction, *index) for name, index in neighbours.items()}
    target = {name: points(truth, *index) for name, index in neighbours.items()}

    def normals(values: dict[str, np.ndarray]) -> np.ndarray:
        result = np.cross(
            values["east"] - values["west"],
            values["south"] - values["north"],
        )
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norm, 1e-12)

    pred_normals = normals(pred)
    true_normals = normals(target)
    normal_valid = np.asarray(validity[center_rows, center_columns], dtype=bool)
    for neighbour_rows, neighbour_columns in neighbours.values():
        normal_valid &= np.asarray(
            validity[neighbour_rows, neighbour_columns], dtype=bool
        )
    normal_valid &= np.isfinite(pred_normals).all(axis=1)
    normal_valid &= np.isfinite(true_normals).all(axis=1)
    cosine = np.abs(np.sum(pred_normals * true_normals, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosine[normal_valid], 0.0, 1.0)))
    return {
        "count": int(angles.size),
        "mean_deg": float(np.mean(angles)),
        "median_deg": float(np.median(angles)),
        "p90_deg": float(np.quantile(angles, 0.9)),
        "stride": stride,
    }


def _write_ply(
    path: Path,
    radial: np.ndarray,
    validity: np.ndarray,
    *,
    stride: int,
) -> dict[str, Any]:
    from panorai.geometry import erp_pixels_to_rays

    rows = np.arange(0, radial.shape[0], stride, dtype=np.int64)
    columns = np.arange(0, radial.shape[1], stride, dtype=np.int64)
    xx, yy = np.meshgrid(columns, rows)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    ranges = np.asarray(radial[np.ix_(rows, columns)], dtype=np.float64).ravel()
    selected = np.asarray(validity[np.ix_(rows, columns)], dtype=bool).ravel()
    selected &= np.isfinite(ranges) & (ranges > 0.0)
    points = erp_pixels_to_rays(pixels[selected], radial.shape) * ranges[selected, None]
    points = points.astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment anchor-preserving spherical densification\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "written_points": int(points.shape[0]),
        "stride": stride,
    }


def _run_family(args: argparse.Namespace, family: str) -> dict[str, Any]:
    target = TARGETS[family]
    prediction_path = args.val048_root / family / "predictions-before-ground-truth.npz"
    prior_path = args.val020_root / f"{target}-spherical-radial.npy"
    gt_path = args.val020_root / f"{target}-gt-radial.npy"
    validity_path = args.val020_root / f"{target}-evaluation-validity.npy"
    with np.load(prediction_path) as sparse:
        bearings = np.asarray(sparse["target_bearings"], dtype=np.float64)
        anchors = np.asarray(sparse["fixed_pose_ranges_m"], dtype=np.float64)
    prior = np.load(prior_path, mmap_mode="r")
    prior_at_anchors = sample_erp_nearest(prior, bearings).astype(np.float64)
    scale_inputs = (
        np.isfinite(prior_at_anchors)
        & (prior_at_anchors > 0.0)
        & np.isfinite(anchors)
        & (anchors > 0.0)
    )
    if not scale_inputs.any():
        raise RuntimeError("no valid anchors for robust global scale")
    global_scale = float(
        np.exp(
            np.median(np.log(anchors[scale_inputs] / prior_at_anchors[scale_inputs]))
        )
    )
    global_scaled = (np.asarray(prior) * global_scale).astype(prior.dtype)
    options = AnchorDensificationOptions(
        angular_radius_deg=args.angular_radius_deg,
    )
    dense = densify_log_range_anchors(prior, bearings, anchors, options=options)
    harmonic = densify_harmonic_log_range(prior, bearings, anchors)
    anchored_harmonic = densify_log_range_anchors(
        harmonic.radial_range_m,
        bearings,
        anchors,
        options=options,
    )
    family_output = args.output / family
    family_output.mkdir(parents=True, exist_ok=True)
    dense_path = family_output / "local-rbf-radial-m.npy"
    global_path = family_output / "global-scale-radial-m.npy"
    support_path = family_output / "densification-support.npy"
    correction_path = family_output / "log-correction.npy"
    harmonic_path = family_output / "harmonic-radial-m.npy"
    harmonic_correction_path = family_output / "harmonic-log-correction.npy"
    anchored_harmonic_path = family_output / "harmonic-hard-anchor-radial-m.npy"
    anchored_harmonic_correction_path = (
        family_output / "harmonic-hard-anchor-local-log-correction.npy"
    )
    np.save(dense_path, dense.radial_range_m)
    np.save(global_path, global_scaled)
    np.save(support_path, dense.support)
    np.save(correction_path, dense.log_correction)
    np.save(harmonic_path, harmonic.radial_range_m)
    np.save(harmonic_correction_path, harmonic.log_correction)
    np.save(anchored_harmonic_path, anchored_harmonic.radial_range_m)
    np.save(
        anchored_harmonic_correction_path,
        anchored_harmonic.log_correction,
    )
    freeze = {
        "local_rbf": {"path": str(dense_path), "sha256": sha256(dense_path)},
        "global_scale": {"path": str(global_path), "sha256": sha256(global_path)},
        "support": {"path": str(support_path), "sha256": sha256(support_path)},
        "log_correction": {
            "path": str(correction_path),
            "sha256": sha256(correction_path),
        },
        "harmonic": {"path": str(harmonic_path), "sha256": sha256(harmonic_path)},
        "harmonic_log_correction": {
            "path": str(harmonic_correction_path),
            "sha256": sha256(harmonic_correction_path),
        },
        "harmonic_hard_anchor": {
            "path": str(anchored_harmonic_path),
            "sha256": sha256(anchored_harmonic_path),
        },
        "harmonic_hard_anchor_local_log_correction": {
            "path": str(anchored_harmonic_correction_path),
            "sha256": sha256(anchored_harmonic_correction_path),
        },
        "ground_truth_opened_before_freeze": False,
    }
    freeze_path = family_output / "prediction-freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    exact_anchor_error = np.max(
        np.abs(
            dense.radial_range_m[dense.anchor_rows, dense.anchor_columns]
            - dense.anchor_ranges_m
        )
    )
    outside_identity = np.array_equal(
        dense.radial_range_m[~dense.support], np.asarray(prior)[~dense.support]
    )
    if exact_anchor_error > 1e-5 or not outside_identity:
        raise RuntimeError("densification preservation contract failed")
    anchored_harmonic_error = np.max(
        np.abs(
            anchored_harmonic.radial_range_m[
                anchored_harmonic.anchor_rows,
                anchored_harmonic.anchor_columns,
            ]
            - anchored_harmonic.anchor_ranges_m
        )
    )
    anchored_harmonic_outside_identity = np.array_equal(
        anchored_harmonic.radial_range_m[~anchored_harmonic.support],
        harmonic.radial_range_m[~anchored_harmonic.support],
    )
    if anchored_harmonic_error > 1e-5 or not anchored_harmonic_outside_identity:
        raise RuntimeError("anchored harmonic preservation contract failed")

    # Evaluation-only boundary starts here.
    truth = np.load(gt_path, mmap_mode="r")
    validity = np.load(validity_path, mmap_mode="r").astype(bool)
    anchor_truth = sample_erp_nearest(truth, bearings)
    anchor_valid = sample_erp_nearest(validity, bearings).astype(bool)
    anchor_prior = sample_erp_nearest(prior, bearings)
    anchor_global = sample_erp_nearest(global_scaled, bearings)
    anchor_dense = sample_erp_nearest(dense.radial_range_m, bearings)
    anchor_harmonic = sample_erp_nearest(harmonic.radial_range_m, bearings)
    anchor_anchored_harmonic = sample_erp_nearest(
        anchored_harmonic.radial_range_m, bearings
    )
    from benchmarks.metric_landmark_ba.p74_protocol import radial_metrics

    metrics = {
        "full": {
            "prior": _dense_metrics(prior, truth, validity),
            "global_scale": _dense_metrics(global_scaled, truth, validity),
            "local_rbf": _dense_metrics(dense.radial_range_m, truth, validity),
            "harmonic": _dense_metrics(harmonic.radial_range_m, truth, validity),
            "harmonic_hard_anchor": _dense_metrics(
                anchored_harmonic.radial_range_m, truth, validity
            ),
        },
        "support": {
            "prior": _dense_metrics(prior, truth, validity, region=dense.support),
            "local_rbf": _dense_metrics(
                dense.radial_range_m, truth, validity, region=dense.support
            ),
        },
        "anchors": {
            "prior": radial_metrics(anchor_prior, anchor_truth, anchor_valid),
            "global_scale": radial_metrics(anchor_global, anchor_truth, anchor_valid),
            "local_rbf": radial_metrics(anchor_dense, anchor_truth, anchor_valid),
            "harmonic": radial_metrics(anchor_harmonic, anchor_truth, anchor_valid),
            "harmonic_hard_anchor": radial_metrics(
                anchor_anchored_harmonic, anchor_truth, anchor_valid
            ),
        },
        "normals": {
            "prior": _normal_error_deg(
                prior, truth, validity, stride=args.normal_stride
            ),
            "global_scale": _normal_error_deg(
                global_scaled,
                truth,
                validity,
                stride=args.normal_stride,
            ),
            "local_rbf": _normal_error_deg(
                dense.radial_range_m,
                truth,
                validity,
                stride=args.normal_stride,
            ),
            "harmonic": _normal_error_deg(
                harmonic.radial_range_m,
                truth,
                validity,
                stride=args.normal_stride,
            ),
            "harmonic_hard_anchor": _normal_error_deg(
                anchored_harmonic.radial_range_m,
                truth,
                validity,
                stride=args.normal_stride,
            ),
        },
    }
    local_ply = _write_ply(
        family_output / "local-rbf-surface.ply",
        dense.radial_range_m,
        validity,
        stride=args.ply_stride,
    )
    global_ply = _write_ply(
        family_output / "global-scale-surface.ply",
        global_scaled,
        validity,
        stride=args.ply_stride,
    )
    harmonic_ply = _write_ply(
        family_output / "harmonic-surface.ply",
        harmonic.radial_range_m,
        validity,
        stride=args.ply_stride,
    )
    anchored_harmonic_ply = _write_ply(
        family_output / "harmonic-hard-anchor-surface.ply",
        anchored_harmonic.radial_range_m,
        validity,
        stride=args.ply_stride,
    )
    result = {
        "family": family,
        "target": target,
        "shape_hw": list(prior.shape),
        "options": {
            "angular_radius_deg": options.angular_radius_deg,
            "kernel": "compact-wendland-c2",
            "maximum_abs_log_correction": options.maximum_abs_log_correction,
        },
        "global_scale": global_scale,
        "global_scale_anchor_count": int(scale_inputs.sum()),
        "harmonic": {
            "degree": harmonic.degree,
            "coefficients": harmonic.coefficients.tolist(),
            "cross_validation_mae_log_by_degree": {
                str(degree): value
                for degree, value in harmonic.cross_validation_mae_log_by_degree.items()
            },
            "anchor_mae_log": harmonic.anchor_mae_log,
        },
        "input_anchor_count": dense.input_anchor_count,
        "unique_anchor_count": dense.unique_anchor_count,
        "support_pixel_count": int(dense.support.sum()),
        "support_fraction": float(dense.support.mean()),
        "maximum_anchor_absolute_error_m": float(exact_anchor_error),
        "outside_support_byte_identical": outside_identity,
        "harmonic_hard_anchor": {
            "maximum_anchor_absolute_error_m": float(anchored_harmonic_error),
            "local_support_pixel_count": int(anchored_harmonic.support.sum()),
            "local_support_fraction": float(anchored_harmonic.support.mean()),
            "outside_local_support_byte_identical_to_harmonic": (
                anchored_harmonic_outside_identity
            ),
        },
        "freeze": freeze,
        "metrics": metrics,
        "ply": {
            "global_scale": global_ply,
            "local_rbf": local_ply,
            "harmonic": harmonic_ply,
            "harmonic_hard_anchor": anchored_harmonic_ply,
        },
    }
    result_path = family_output / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [_run_family(args, family) for family in args.families]
    aggregate = {
        "schema": SCHEMA,
        "prediction_blind_fixed_options": True,
        "results": results,
    }
    path = args.output / "results.json"
    path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                row["family"]: {
                    "support_fraction": row["support_fraction"],
                    "full_prior": row["metrics"]["full"]["prior"],
                    "full_global_scale": row["metrics"]["full"]["global_scale"],
                    "full_local_rbf": row["metrics"]["full"]["local_rbf"],
                    "full_harmonic": row["metrics"]["full"]["harmonic"],
                    "full_harmonic_hard_anchor": row["metrics"]["full"][
                        "harmonic_hard_anchor"
                    ],
                    "normal_prior": row["metrics"]["normals"]["prior"],
                    "normal_global_scale": row["metrics"]["normals"]["global_scale"],
                    "normal_local_rbf": row["metrics"]["normals"]["local_rbf"],
                    "normal_harmonic": row["metrics"]["normals"]["harmonic"],
                    "normal_harmonic_hard_anchor": row["metrics"]["normals"][
                        "harmonic_hard_anchor"
                    ],
                    "harmonic_model": row["harmonic"],
                }
                for row in results
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
