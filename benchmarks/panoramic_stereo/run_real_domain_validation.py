#!/usr/bin/env python3
"""Run the panoramic dense-stereo application on two public external domains.

Stanford2D3D and Matterport3D remain governed by their dataset agreements.
This runner writes only to an ignored output directory and never copies source
media into the repository or distribution.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.applications import (
    ADCensusStereoDepth,
    CensusMGMStereoDepth,
    ELASStereoDepth,
    PanoramicDenseStereo,
    PatchMatchStereoDepth,
    StereoBMDepth,
    StereoSGBMDepth,
)
from panorai.stereo.classical import (
    ADCensusConfig,
    CensusMGMConfig,
    ELASConfig,
    PatchMatchConfig,
    StereoBMConfig,
    StereoSGBMConfig,
    StereoValidationConfig,
)
from panorai.stereo.panoramic import RectifiedStereoViewPlanner


SCHEMA = "panorai-panoramic-stereo-real-domains/v2"
METHODS = (
    "stereo-bm",
    "ad-census",
    "stereo-sgbm",
    "census-tmgm",
    "elas-style",
    "patchmatch",
)
SELECTION = {
    "stanford2d3d": {
        "pair_id": "essential-stat-2024",
        "group": "area_2",
        "reference": "camera_a81c748601ef4e5db4b74c4494ee61bc_office_3_frame_equirectangular_domain",
        "source": "camera_aa113cffd2b046bb8a224f9b1bdc2e38_office_3_frame_equirectangular_domain",
        "registered_overlap": 0.9798083575801613,
        "selection": "pre-existing registered-pair audit",
    },
    "matterport360": {
        "pair_id": "essential-stat-1109",
        "group": "qoiz87jewz2",
        "reference": "bec7a60657d6483996aa2f0d03b9d691",
        "source": "bf9b94aa9b6b43e68e27ae3bfea235c5",
        "registered_overlap": 0.9649612447741418,
        "selection": "pre-existing registered-pair audit",
    },
}


@dataclass(frozen=True, slots=True)
class PairData:
    dataset_id: str
    pair_id: str
    reference_rgb: np.ndarray
    source_rgb: np.ndarray
    reference_support: np.ndarray
    source_support: np.ndarray
    reference_range: np.ndarray
    source_range: np.ndarray
    rotation_source_from_reference: np.ndarray
    translation_source_from_reference: np.ndarray
    provenance: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_source_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def nearest_rotation(value: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(value, dtype=np.float64))
    sign = np.linalg.det(left @ right)
    return left @ np.diag((1.0, 1.0, sign)) @ right


def relative_pose(
    rotation_world_to_reference: np.ndarray,
    center_reference_world: np.ndarray,
    rotation_world_to_source: np.ndarray,
    center_source_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X_source = R @ X_reference + t``."""

    rotation = rotation_world_to_source @ rotation_world_to_reference.T
    translation = rotation_world_to_source @ (
        center_reference_world - center_source_world
    )
    return nearest_rotation(rotation), np.asarray(translation, dtype=np.float64)


def _public_pose(dataset_id: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if dataset_id == "stanford2d3d":
        matrix = np.loadtxt(path).reshape(3, 4)
        rotation_dataset = matrix[:, :3]
        center_dataset = -rotation_dataset.T @ matrix[:, 3]
        basis = np.diag((1.0, -1.0, 1.0))
    elif dataset_id == "matterport360":
        values = np.loadtxt(path).reshape(7)
        center_dataset = values[:3]
        rotation_dataset = Rotation.from_quat(values[3:]).as_matrix().T
        basis = np.diag((1.0, 1.0, -1.0))
    else:
        raise ValueError(f"unsupported public dataset: {dataset_id}")
    return (
        nearest_rotation(basis @ rotation_dataset @ basis),
        basis @ center_dataset,
    )


def _resize_rgb(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode RGB: {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, shape_hw[::-1], interpolation=cv2.INTER_AREA)


def _resize_range(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"radial range must be HW: {path}")
    return cv2.resize(value, shape_hw[::-1], interpolation=cv2.INTER_NEAREST_EXACT)


def load_public_pair(
    dataset_id: str, root: Path, shape_hw: tuple[int, int]
) -> PairData:
    return load_public_pair_selection(
        dataset_id,
        root,
        shape_hw,
        SELECTION[dataset_id],
    )


def load_public_pair_selection(
    dataset_id: str,
    root: Path,
    shape_hw: tuple[int, int],
    selected: Mapping[str, Any],
) -> PairData:
    """Load one explicit licensed public pair without selecting by outcome."""

    if dataset_id not in {"stanford2d3d", "matterport360"}:
        raise ValueError(f"unsupported public dataset: {dataset_id}")
    required = ("pair_id", "group", "reference", "source")
    missing_fields = [name for name in required if not selected.get(name)]
    if missing_fields:
        raise ValueError(
            "public pair selection is missing fields: " + ", ".join(missing_fields)
        )
    group = str(selected["group"])
    names = (str(selected["reference"]), str(selected["source"]))
    rgb_paths = [root / "rgb" / group / f"{name}.png" for name in names]
    depth_paths = [root / "depth" / group / f"{name}.npy" for name in names]
    pose_paths = [root / "poses" / group / f"{name}.txt" for name in names]
    paths = (*rgb_paths, *depth_paths, *pose_paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing dataset assets: " + ", ".join(missing))
    pose_a = _public_pose(dataset_id, pose_paths[0])
    pose_b = _public_pose(dataset_id, pose_paths[1])
    rotation, translation = relative_pose(*pose_a, *pose_b)
    reference = _resize_rgb(rgb_paths[0], shape_hw)
    source = _resize_rgb(rgb_paths[1], shape_hw)
    reference_range = _resize_range(depth_paths[0], shape_hw)
    source_range = _resize_range(depth_paths[1], shape_hw)
    support = np.ones(shape_hw, dtype=bool)
    return PairData(
        dataset_id,
        str(selected["pair_id"]),
        reference,
        source,
        support,
        support.copy(),
        reference_range,
        source_range,
        rotation,
        translation,
        {
            "license": (
                "Stanford 2D-3D-S agreement; local research copy"
                if dataset_id == "stanford2d3d"
                else "Matterport3D terms; local licensed research copy"
            ),
            "registered_overlap": selected.get(
                "registered_overlap",
                selected.get("minimum_directional_cloud_overlap"),
            ),
            "selection": selected.get(
                "selection", "explicit frozen benchmark manifest"
            ),
            "assets": {
                path.name: {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                }
                for path in paths
            },
        },
    )


def solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    latitude = (
        math.pi / 2.0 - math.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    )
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[np.searchsorted(cumulative, q * cumulative[-1])])


def depth_metrics(
    estimate: np.ndarray,
    validity: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> dict[str, float | int]:
    weights = solid_angle_weights(reference.shape)
    reference_valid = (
        np.asarray(support, dtype=bool) & np.isfinite(reference) & (reference > 0.05)
    )
    evaluated = (
        reference_valid
        & np.asarray(validity, dtype=bool)
        & np.isfinite(estimate)
        & (estimate > 0.0)
    )
    support_weight = float(weights[reference_valid].sum())
    evaluated_weight = float(weights[evaluated].sum())
    result: dict[str, float | int] = {
        "reference_pixels": int(reference_valid.sum()),
        "evaluated_pixels": int(evaluated.sum()),
        "solid_angle_coverage": evaluated_weight / max(support_weight, 1e-12),
    }
    if not np.any(evaluated):
        return {
            **result,
            "abs_rel": math.nan,
            "rmse_m": math.nan,
            "median_abs_error_m": math.nan,
            "delta_1_25": math.nan,
            "correct_solid_angle_coverage_delta_1_25": 0.0,
        }
    truth = reference[evaluated].astype(np.float64)
    prediction = estimate[evaluated].astype(np.float64)
    selected_weights = weights[evaluated]
    selected_weights /= selected_weights.sum()
    absolute = np.abs(prediction - truth)
    ratio = np.maximum(prediction / truth, truth / prediction)
    delta_1_25 = float(np.sum(selected_weights * (ratio < 1.25)))
    return {
        **result,
        "abs_rel": float(np.sum(selected_weights * absolute / truth)),
        "rmse_m": float(np.sqrt(np.sum(selected_weights * (prediction - truth) ** 2))),
        "median_abs_error_m": _weighted_quantile(absolute, selected_weights, 0.5),
        "delta_1_25": delta_1_25,
        # Unlike conditional delta, this answers how much of the complete
        # supported sphere is both predicted and within the error threshold.
        "correct_solid_angle_coverage_delta_1_25": (
            result["solid_angle_coverage"] * delta_1_25
        ),
    }


def rotation_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    delta = estimate @ reference.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def direction_error_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    first = estimate / np.linalg.norm(estimate)
    second = reference / np.linalg.norm(reference)
    return math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0))))


def make_matcher(args: argparse.Namespace, method: str) -> Any:
    validation = StereoValidationConfig(
        uniqueness_margin=0.0,
        left_right_tolerance=args.left_right_tolerance,
        speckle_window_size=args.speckle_window_size,
        speckle_range=2.0,
        median_kernel=0,
    )
    common = {
        "min_disparity": 0,
        "max_disparity": args.max_disparity,
        "validation": validation,
    }
    if method == "stereo-bm":
        return StereoBMDepth(
            StereoBMConfig(
                **common,
                block_size=max(5, args.block_size),
                uniqueness_ratio=args.uniqueness_ratio,
                texture_threshold=10,
            )
        )
    if method == "ad-census":
        return ADCensusStereoDepth(ADCensusConfig(**common))
    if method == "stereo-sgbm":
        return StereoSGBMDepth(
            StereoSGBMConfig(
                **common,
                block_size=args.block_size,
                uniqueness_ratio=args.uniqueness_ratio,
            )
        )
    if method == "census-tmgm":
        return CensusMGMStereoDepth(CensusMGMConfig(**common, variant="tmgm"))
    if method == "elas-style":
        return ELASStereoDepth(ELASConfig(**common))
    if method == "patchmatch":
        # The pure-Python reference is intentionally inspectable rather than
        # optimized.  Keep the real-domain comparison bounded while retaining
        # propagation, random search and slanted-plane patch evaluation.
        return PatchMatchStereoDepth(
            PatchMatchConfig(**common, iterations=1, patch_radius=1)
        )
    raise ValueError(f"unsupported method: {method}")


def make_application(
    args: argparse.Namespace, pose_mode: str, method: str = "stereo-sgbm"
) -> PanoramicDenseStereo:
    matcher = make_matcher(args, method)
    planner = RectifiedStereoViewPlanner(
        hfov_deg=args.face_hfov,
        vfov_deg=args.face_vfov,
        overlap_deg=args.face_overlap,
        output_shape_hw=(args.face_size, args.face_size),
    )
    return PanoramicDenseStereo(
        matcher=matcher,
        pose_mode=pose_mode,
        planner=planner,
        min_joint_support=args.min_joint_support,
        min_texture_std=args.min_texture_std,
    )


def _render_panel(
    path: Path,
    pair: PairData,
    result: Any,
    metrics: dict[str, Any],
    mode: str,
    method: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_valid = (
        pair.reference_support
        & np.isfinite(pair.reference_range)
        & (pair.reference_range > 0.05)
    )
    lower, upper = np.percentile(pair.reference_range[reference_valid], (2.0, 98.0))
    relative_error = np.full(pair.reference_range.shape, np.nan, dtype=np.float32)
    evaluated = reference_valid & result.validity
    relative_error[evaluated] = (
        np.abs(result.radial_range[evaluated] - pair.reference_range[evaluated])
        / pair.reference_range[evaluated]
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].imshow(pair.reference_rgb)
    axes[0, 0].set_title("ERP A")
    axes[0, 1].imshow(pair.source_rgb)
    axes[0, 1].set_title("ERP B")
    axes[0, 2].imshow(
        np.ma.masked_where(~reference_valid, pair.reference_range),
        cmap="turbo",
        vmin=lower,
        vmax=upper,
    )
    axes[0, 2].set_title("Reference radial range [m]")
    axes[1, 0].imshow(
        np.ma.masked_invalid(result.radial_range),
        cmap="turbo",
        vmin=lower,
        vmax=upper,
    )
    axes[1, 0].set_title("PanorAi fused radial range [m]")
    axes[1, 1].imshow(
        np.ma.masked_invalid(relative_error), cmap="magma", vmin=0, vmax=1
    )
    axes[1, 1].set_title("Absolute relative error")
    axes[1, 2].imshow(result.confidence, cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("Fusion confidence")
    axes[2, 0].imshow(result.contribution_count, cmap="cividis")
    axes[2, 0].set_title("Contributing rectified views")
    axes[2, 1].imshow(
        np.ma.masked_invalid(result.disagreement), cmap="inferno", vmin=0, vmax=1
    )
    axes[2, 1].set_title("Cross-view disagreement [m]")
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        1.0,
        "\n".join(
            (
                f"dataset: {pair.dataset_id}",
                f"pair: {pair.pair_id}",
                f"pose: {mode}",
                f"method: {method}",
                f"coverage: {metrics['solid_angle_coverage']:.3f}",
                f"AbsRel: {metrics['abs_rel']:.3f}",
                f"RMSE: {metrics['rmse_m']:.3f} m",
                f"delta<1.25: {metrics['delta_1_25']:.3f}",
                f"accepted pairs: {len(result.pair_diagnostics)}",
                f"rejected pairs: {sum(not x.accepted for x in result.view_selection)}",
            )
        ),
        va="top",
        family="monospace",
    )
    for axis in axes.flat[:8]:
        axis.axis("off")
    figure.suptitle(f"{pair.dataset_id} · {method} · {mode} pose · {pair.pair_id}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_pair(
    pair: PairData,
    args: argparse.Namespace,
    output_dir: Path,
    pose_mode: str,
    method: str = "stereo-sgbm",
) -> dict[str, Any]:
    app = make_application(args, pose_mode, method)
    baseline = float(np.linalg.norm(pair.translation_source_from_reference))
    started = time.perf_counter()
    try:
        if pose_mode == "provided":
            result = app.run(
                pair.reference_rgb,
                pair.source_rgb,
                rotation_source_from_reference=pair.rotation_source_from_reference,
                translation_source_from_reference=pair.translation_source_from_reference,
                units="m",
                reference_validity=pair.reference_support,
                source_validity=pair.source_support,
            )
        else:
            result = app.run(
                pair.reference_rgb,
                pair.source_rgb,
                baseline=baseline,
                units="m",
                reference_validity=pair.reference_support,
                source_validity=pair.source_support,
            )
    except Exception as error:
        return {
            "dataset_id": pair.dataset_id,
            "pair_id": pair.pair_id,
            "pose_mode": pose_mode,
            "method": method,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }
    elapsed = time.perf_counter() - started
    metrics = depth_metrics(
        result.radial_range,
        result.validity,
        pair.reference_range,
        pair.reference_support,
    )
    pose_metrics = {
        "rotation_error_deg": rotation_error_deg(
            result.pose.rotation_source_from_reference,
            pair.rotation_source_from_reference,
        ),
        "translation_direction_error_deg": direction_error_deg(
            result.pose.translation_source_from_reference,
            pair.translation_source_from_reference,
        ),
    }
    stem = f"{pair.dataset_id}-{method}-{pose_mode}"
    arrays_path = output_dir / f"{stem}.npz"
    panel_path = output_dir / f"{stem}.jpg"
    np.savez_compressed(
        arrays_path,
        estimated_radial_range_m=result.radial_range,
        validity=result.validity,
        confidence=result.confidence,
        disagreement_m=result.disagreement,
        contribution_count=result.contribution_count,
        reference_radial_range_m=pair.reference_range,
        reference_support=pair.reference_support,
    )
    _render_panel(panel_path, pair, result, metrics, pose_mode, method)
    return {
        "dataset_id": pair.dataset_id,
        "pair_id": pair.pair_id,
        "pose_mode": pose_mode,
        "method": method,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "baseline_m": baseline,
        "pose": pose_metrics,
        "depth": metrics,
        "accepted_view_pairs": len(result.pair_diagnostics),
        "rejected_view_pairs": sum(not item.accepted for item in result.view_selection),
        "view_selection": [
            {
                "pair_id": item.pair_id,
                "accepted": item.accepted,
                "joint_support_fraction": item.joint_support_fraction,
                "texture_std": item.texture_std,
                "pose_inliers": item.pose_inliers,
                "median_epipolar_row_error_px": item.median_epipolar_row_error_px,
                "rejection_reasons": list(item.rejection_reasons),
            }
            for item in result.view_selection
        ],
        "pair_diagnostics": [
            {
                "pair_id": item.pair_id,
                "valid_fraction": item.valid_fraction,
                "median_confidence": item.median_confidence,
                "median_range": item.median_range,
                "method": item.method,
            }
            for item in result.pair_diagnostics
        ],
        "panel": str(panel_path.resolve()),
        "arrays": str(arrays_path.resolve()),
    }


def _write_report(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Panoramic dense stereo on two public real domains",
        "",
        "This is development evidence from external local data, not a promotion gate.",
        "Dataset imagery and generated panels remain external and non-distributable.",
        "",
        "| Dataset | Method | Pose | Status | Coverage | AbsRel | RMSE m | δ<1.25 | R err | t err | Time s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        if record["status"] != "completed":
            lines.append(
                f"| {record['dataset_id']} | {record['method']} | {record['pose_mode']} | "
                f"{record['status']}: {record['error']} | — | — | — | — | — | — | "
                f"{record['elapsed_seconds']:.2f} |"
            )
            continue
        depth = record["depth"]
        pose = record["pose"]
        lines.append(
            f"| {record['dataset_id']} | {record['method']} | {record['pose_mode']} | completed | "
            f"{depth['solid_angle_coverage']:.3f} | {depth['abs_rel']:.3f} | "
            f"{depth['rmse_m']:.3f} | {depth['delta_1_25']:.3f} | "
            f"{pose['rotation_error_deg']:.2f}° | "
            f"{pose['translation_direction_error_deg']:.2f}° | "
            f"{record['elapsed_seconds']:.2f} |"
        )
    lines.extend(
        (
            "",
            "Coverage is solid-angle weighted over valid reference range. Metrics are",
            "computed only where the method returns valid radial range. The provided-pose",
            "run isolates dense matching; the estimated-pose run exercises the full A-to-Z",
            "application while borrowing only the known baseline magnitude for metric scale.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stanford-root",
        type=Path,
        default=Path("data/stanford2d3d"),
    )
    parser.add_argument(
        "--matterport-root",
        type=Path,
        default=Path("data/matterport360"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--face-size", type=int, default=768)
    parser.add_argument("--face-hfov", type=float, default=90.0)
    parser.add_argument("--face-vfov", type=float, default=110.0)
    parser.add_argument("--face-overlap", type=float, default=30.0)
    parser.add_argument("--max-disparity", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=5)
    parser.add_argument("--left-right-tolerance", type=float, default=1.5)
    parser.add_argument("--speckle-window-size", type=int, default=0)
    parser.add_argument("--min-joint-support", type=float, default=0.10)
    parser.add_argument("--min-texture-std", type=float, default=0.005)
    parser.add_argument(
        "--pose-modes",
        nargs="+",
        choices=("provided", "estimate"),
        default=("provided", "estimate"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=("stereo-sgbm",),
    )
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("canonical ERP evaluation requires width == 2 * height")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_hw = (args.height, args.width)
    pairs = (
        load_public_pair("stanford2d3d", args.stanford_root, shape_hw),
        load_public_pair("matterport360", args.matterport_root, shape_hw),
    )
    records: list[dict[str, Any]] = []
    for pair in pairs:
        for method in args.methods:
            for pose_mode in args.pose_modes:
                print(
                    f"[{pair.dataset_id}] method={method} pose={pose_mode}",
                    flush=True,
                )
                record = run_pair(pair, args, args.output_dir, pose_mode, method)
                records.append(record)
                print(
                    json.dumps(json_safe(record), indent=2, default=str),
                    flush=True,
                )
    source_commit, source_tree_dirty = git_source_state()
    manifest = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "source_tree_dirty": source_tree_dirty,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "source-checkout external-local-data",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in {"stanford_root", "matterport_root", "output_dir"}
        },
        "selections": SELECTION,
        "provenance": {pair.dataset_id: pair.provenance for pair in pairs},
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    _write_report(args.output_dir / "REPORT.md", records)


if __name__ == "__main__":
    main()
