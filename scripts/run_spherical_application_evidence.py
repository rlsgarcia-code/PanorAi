#!/usr/bin/env python3
"""Generate prospective evidence for spherical application promotion.

The command deliberately separates outcome-blind pair selection from metric
execution.  ``select`` may inspect only frozen split/covariate/overlap records;
``evaluate-core`` consumes the immutable selection and evaluates SLIC, DoG and
stereo.  Model checkpoints and datasets remain external to the repository.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.applications import (  # noqa: E402
    SphericalDoGFeatures,
    SphericalSLIC,
    SphericalStereoDepth,
)
from panorai.features import FeatureMatcher, SphericalDoGSIFTConfig  # noqa: E402
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels  # noqa: E402
from panorai.image_processing._slic import (  # noqa: E402
    _assign_labels,
    _erp_rays,
    _solid_angle_weights,
)
from panorai.stereo import SphericalStereoOptions  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "contracts/spherical-application-promotion-v1.json"
SELECTION_CONTRACT = "panorai-prospective-application-selection/v1"
EVIDENCE_CONTRACT = "panorai-spherical-application-promotion/v1"
CORPUS_NAMES = {
    "matterport360": "Matterport3D",
    "stanford2d3d": "Stanford2D3D",
}
FRAME_FROM_PANORAI = {
    "matterport360": np.diag((1.0, 1.0, -1.0)),
    "stanford2d3d": np.diag((1.0, -1.0, 1.0)),
}
SELECTION_POLICY = {
    "partition": "evaluation",
    "datasets": sorted(CORPUS_NAMES),
    "minimum_directional_cloud_overlap": 0.50,
    "minimum_baseline_m": 0.15,
    "maximum_baseline_m": 2.50,
    "pairs_per_corpus": 3,
    "ordering": (
        "descending overlap, then ascending baseline, then pair_id; "
        "prefer unused spatial groups and views before deterministic fill"
    ),
    "method_outputs_used": False,
}


def sha256_file(path: Path) -> str:
    """Return a content hash without interpreting the file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable_json(path: Path, value: Any) -> None:
    """Create an evidence file, refusing to replace different bytes."""

    payload = canonical_json(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _view_parts(view_id: str) -> tuple[str, str, str]:
    parts = view_id.split("::")
    if len(parts) != 3 or parts[0] not in CORPUS_NAMES:
        raise ValueError(f"unsupported canonical view id: {view_id!r}")
    return parts[0], parts[1], parts[2]


def _asset_paths(dataset_root: Path, view_id: str) -> dict[str, Path]:
    _, group, stem = _view_parts(view_id)
    return {
        "rgb": dataset_root / "rgb" / group / f"{stem}.png",
        "depth": dataset_root / "depth" / group / f"{stem}.npy",
    }


def _asset_record(dataset_root: Path, view_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"view_id": view_id}
    for kind, path in _asset_paths(dataset_root, view_id).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[kind] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def select_pairs(
    split_records: Iterable[dict[str, Any]],
    overlap_records: Iterable[dict[str, Any]],
    dataset_roots: dict[str, Path],
) -> list[dict[str, Any]]:
    """Select outcome-blind held-out pairs using the frozen policy."""

    overlaps = {item["pair_id"]: item for item in overlap_records}
    selected: list[dict[str, Any]] = []
    for dataset_id in sorted(CORPUS_NAMES):
        candidates = []
        for item in split_records:
            if item.get("dataset_id") != dataset_id:
                continue
            if item.get("partition") != SELECTION_POLICY["partition"]:
                continue
            overlap = overlaps.get(item.get("pair_id"))
            if overlap is None:
                continue
            minimum_overlap = float(overlap["minimum_directional_cloud_overlap"])
            baseline = float(item["covariates"]["baseline_m"])
            if minimum_overlap < SELECTION_POLICY["minimum_directional_cloud_overlap"]:
                continue
            if not (
                SELECTION_POLICY["minimum_baseline_m"]
                <= baseline
                <= SELECTION_POLICY["maximum_baseline_m"]
            ):
                continue
            candidates.append((item, overlap, minimum_overlap, baseline))
        candidates.sort(key=lambda row: (-row[2], row[3], row[0]["pair_id"]))

        chosen: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
        used_groups: set[str] = set()
        used_views: set[str] = set()
        for prefer_new_group in (True, False):
            for candidate in candidates:
                item = candidate[0]
                if candidate in chosen:
                    continue
                views = {item["from_view_id"], item["to_view_id"]}
                if views & used_views:
                    continue
                if prefer_new_group and item["spatial_group_id"] in used_groups:
                    continue
                chosen.append(candidate)
                used_groups.add(item["spatial_group_id"])
                used_views.update(views)
                if len(chosen) == SELECTION_POLICY["pairs_per_corpus"]:
                    break
            if len(chosen) == SELECTION_POLICY["pairs_per_corpus"]:
                break
        if len(chosen) != SELECTION_POLICY["pairs_per_corpus"]:
            raise RuntimeError(
                f"{dataset_id}: selected {len(chosen)} independent pairs; "
                f"requires {SELECTION_POLICY['pairs_per_corpus']}"
            )

        for item, overlap, minimum_overlap, baseline in chosen:
            selected.append(
                {
                    "pair_id": item["pair_id"],
                    "dataset_id": dataset_id,
                    "corpus": CORPUS_NAMES[dataset_id],
                    "split": "heldout",
                    "source_partition": item["partition"],
                    "source_dataset_split": item["dataset_split"],
                    "spatial_group_id": item["spatial_group_id"],
                    "independence_component_id": item["independence_component_id"],
                    "minimum_directional_cloud_overlap": minimum_overlap,
                    "baseline_m": baseline,
                    "from": _asset_record(
                        dataset_roots[dataset_id], item["from_view_id"]
                    ),
                    "to": _asset_record(dataset_roots[dataset_id], item["to_view_id"]),
                    "reference": item["reference"],
                    "selection_provenance": {
                        "split_record_pair_id": item["pair_id"],
                        "overlap_record_pair_id": overlap["pair_id"],
                        "outcome_blind": True,
                    },
                }
            )
    return selected


def build_selection_manifest(
    split_path: Path,
    overlap_path: Path,
    dataset_roots: dict[str, Path],
) -> dict[str, Any]:
    return {
        "contract": SELECTION_CONTRACT,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "selection_policy": SELECTION_POLICY,
        "sources": {
            "split_jsonl": {
                "path": str(split_path.resolve()),
                "sha256": sha256_file(split_path),
            },
            "overlap_jsonl": {
                "path": str(overlap_path.resolve()),
                "sha256": sha256_file(overlap_path),
            },
            "dataset_roots": {
                name: str(path.resolve()) for name, path in dataset_roots.items()
            },
        },
        "licenses": {
            "Matterport3D": "external local copy; governed by Matterport3D terms",
            "Stanford2D3D": "external local copy; governed by Stanford2D3D terms",
        },
        "records": select_pairs(
            read_jsonl(split_path), read_jsonl(overlap_path), dataset_roots
        ),
    }


def _periodically_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if not len(points):
        return False
    height, width = mask.shape
    start = tuple(int(value) for value in points[0])
    seen = {start}
    queue = deque((start,))
    while queue:
        y, x = queue.popleft()
        neighbours = ((y, (x - 1) % width), (y, (x + 1) % width))
        if y:
            neighbours += ((y - 1, x),)
        if y + 1 < height:
            neighbours += ((y + 1, x),)
        for neighbour in neighbours:
            if mask[neighbour] and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == int(mask.sum())


def evaluate_slic_analytic() -> dict[str, Any]:
    shape = (48, 96)
    y, x = np.indices(shape)
    textured = np.stack(
        (
            0.5 + 0.25 * np.sin(2.0 * np.pi * x / shape[1]),
            0.5 + 0.25 * np.cos(2.0 * np.pi * y / shape[0]),
            0.5 + 0.2 * np.sin(4.0 * np.pi * x / shape[1]),
        ),
        axis=-1,
    ).astype(np.float32)
    image = np.full((*shape, 3), 0.5, dtype=np.float32)
    application = SphericalSLIC(n_segments=12, max_num_iter=15, min_size_factor=0.25)
    first = application(image).values
    second = application(image).values
    labels = np.unique(first)
    weights = _solid_angle_weights(shape)

    seam_image = np.zeros((*shape, 3), dtype=np.uint8)
    seam_image[..., 2] = 96
    seam_image[:, :8] = (255, 0, 0)
    seam_image[:, -8:] = (255, 0, 0)
    seam = SphericalSLIC(
        n_segments=8,
        compactness=2.0,
        max_num_iter=15,
        min_size_factor=0.25,
        start_label=0,
    )(seam_image).values

    mask = np.ones(shape, dtype=bool)
    mask[:4] = False
    mask[-3:] = False
    mask[20:24, 40:48] = False
    masked = SphericalSLIC(
        n_segments=9, compactness=6.0, max_num_iter=12, start_label=7
    )(textured, validity_mask=mask).values

    rng = np.random.default_rng(8)
    rays = rng.normal(size=(80, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    centres = rays[[2, 17, 31, 48, 67]]
    colours = rng.normal(size=(80, 3))
    centre_colours = colours[[2, 17, 31, 48, 67]]
    angle = np.deg2rad(73.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    assigned = _assign_labels(
        colours, rays, centre_colours, centres, compactness=7.0, angular_scale=0.8
    )
    rotated = _assign_labels(
        colours,
        rays @ rotation.T,
        centre_colours,
        centres @ rotation.T,
        compactness=7.0,
        angular_scale=0.8,
    )

    erp_rays = _erp_rays((360, 720))
    row, column = 179, 360
    intrinsic = math.acos(
        float(np.clip(erp_rays[row, column] @ erp_rays[row + 1, column + 1], -1, 1))
    )
    planar = math.hypot(math.pi / 360.0, math.pi / 360.0)
    planar_relative_error = abs(intrinsic - planar) / planar

    return {
        "application": "slic",
        "evaluation": "analytic",
        "metrics": {
            "solid_angle_4pi": math.isclose(
                float(weights.sum()), 4.0 * math.pi, abs_tol=1e-12
            ),
            "connected_labels": all(
                _periodically_connected(first == label) for label in labels
            ),
            "deterministic": bool(np.array_equal(first, second)),
            "periodic_seam": bool(
                np.array_equal(seam[:, 0], seam[:, -1])
                and _periodically_connected(seam == seam[shape[0] // 2, 0])
            ),
            "analytic_cardinalities_and_masks": bool(
                len(labels) == 12
                and np.all(masked[~mask] == -1)
                and masked[mask].min() == 7
            ),
            "controlled_rotation_stability": bool(np.array_equal(assigned, rotated)),
            "planar_low_distortion_limit": planar_relative_error <= 0.01,
        },
        "details": {
            "shape_hw": list(shape),
            "observed_label_count": int(len(labels)),
            "solid_angle_sum": float(weights.sum()),
            "equatorial_one_pixel_diagonal_relative_error": planar_relative_error,
        },
    }


def _angular_distance_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.rad2deg(np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)))


def _blob_image(
    shape_hw: tuple[int, int], centres: np.ndarray, sigma_deg: float
) -> np.ndarray:
    height, width = shape_hw
    y, x = np.indices(shape_hw, dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    image = np.full(shape_hw, 25.0, dtype=np.float64)
    for centre in centres:
        distance = np.arccos(np.clip(rays @ centre, -1.0, 1.0))
        image += 220.0 * np.exp(-0.5 * (distance / np.deg2rad(sigma_deg)) ** 2)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _bearing(longitude_deg: float, latitude_deg: float) -> np.ndarray:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    return np.asarray(
        (
            math.sin(longitude) * math.cos(latitude),
            math.sin(latitude),
            math.cos(longitude) * math.cos(latitude),
        )
    )


def evaluate_dog_analytic() -> dict[str, Any]:
    shape = (128, 256)
    sigma_deg = 5.0
    centres = np.stack(
        (
            _bearing(179.0, 74.0),
            _bearing(-70.0, 5.0),
            _bearing(45.0, -48.0),
        )
    )
    config = SphericalDoGSIFTConfig(
        octaves=3,
        levels_per_octave=3,
        base_sigma_px=1.0,
        contrast_threshold=0.0005,
        edge_threshold=20.0,
        max_features=80,
        patch_size=32,
        convolution_backend="numpy",
    )
    result = SphericalDoGFeatures(config)(_blob_image(shape, centres, sigma_deg))
    errors = []
    scale_errors = []
    expected_scale_level = (
        math.log2(sigma_deg / (180.0 / shape[0] * config.base_sigma_px))
        * config.levels_per_octave
    )
    for centre in centres:
        angular = _angular_distance_deg(result.bearings, centre)
        closest = int(np.argmin(angular)) if len(angular) else -1
        errors.append(float(angular[closest]) if closest >= 0 else 180.0)
        if closest >= 0:
            observed = (
                math.log2(
                    result.features[closest].scale
                    / (180.0 / shape[0] * config.base_sigma_px)
                )
                * config.levels_per_octave
            )
            scale_errors.append(abs(observed - expected_scale_level))
        else:
            scale_errors.append(float(config.levels_per_octave * config.octaves))
    half_step = 0.5 * 180.0 / shape[0]
    return {
        "application": "dog",
        "evaluation": "analytic",
        "metrics": {
            "isolated_blob_detection_rate": float(
                np.mean(np.asarray(errors) <= half_step)
            ),
            "maximum_angular_error_deg": max(errors),
            "lattice_half_step_deg": half_step,
            "maximum_scale_level_error": max(scale_errors),
            "seam_and_high_latitude_corpus": True,
        },
        "details": {
            "shape_hw": list(shape),
            "blob_sigma_deg": sigma_deg,
            "angular_errors_deg": errors,
            "scale_level_errors": scale_errors,
            "feature_count": len(result),
        },
    }


def _resize_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, shape_hw[::-1], interpolation=cv2.INTER_AREA)


def _resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(depth, shape_hw[::-1], interpolation=cv2.INTER_NEAREST)


def _load_rgb(asset: dict[str, Any], shape_hw: tuple[int, int]) -> np.ndarray:
    path = Path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"RGB checksum changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(_resize_image(image, shape_hw), cv2.COLOR_BGR2RGB)


def _load_depth(asset: dict[str, Any], shape_hw: tuple[int, int]) -> np.ndarray:
    path = Path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"depth checksum changed: {path}")
    depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HW: {path}")
    return _resize_depth(depth, shape_hw)


def _panorai_pose(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    dataset_id = record["dataset_id"]
    adapter = FRAME_FROM_PANORAI[dataset_id]
    reference = record["reference"]
    raw_rotation = np.asarray(reference["R_to_from"], dtype=np.float64)
    raw_translation = np.asarray(reference["t_to_from_m"], dtype=np.float64)
    return adapter.T @ raw_rotation @ adapter, adapter.T @ raw_translation


def _sample_nearest(values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    finite = np.isfinite(xy).all(axis=1)
    safe = np.where(finite[:, None], xy, 0.0)
    x = np.mod(np.rint(safe[:, 0]).astype(np.int64), width)
    y = np.clip(np.rint(safe[:, 1]).astype(np.int64), 0, height - 1)
    return values[y, x]


def _visible_feature_oracle(
    bearings: np.ndarray,
    source_xy: np.ndarray,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = _sample_nearest(depth_a, source_xy)
    points_b = (rotation @ (bearings * depth[:, None]).T).T + translation
    range_b = np.linalg.norm(points_b, axis=1)
    expected_bearings = np.divide(
        points_b,
        range_b[:, None],
        out=np.zeros_like(points_b),
        where=range_b[:, None] > 0,
    )
    projection = rays_to_erp_pixels(expected_bearings, depth_b.shape)
    expected_xy = np.asarray(projection.pixels_xy)
    target_depth = _sample_nearest(depth_b, expected_xy)
    tolerance = np.maximum(0.15, 0.05 * range_b)
    visible = (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(target_depth)
        & (target_depth > 0)
        & np.isfinite(range_b)
        & (range_b > 0)
        & np.asarray(projection.valid, dtype=bool)
        & (np.abs(target_depth - range_b) <= tolerance)
    )
    return expected_bearings, visible


def _minimum_angular_errors(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if len(expected) == 0:
        return np.empty(0, dtype=np.float64)
    if len(observed) == 0:
        return np.full(len(expected), 180.0)
    cosine = np.clip(expected @ observed.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(np.max(cosine, axis=1)))


def evaluate_dog_external_pair(
    record: dict[str, Any], shape_hw: tuple[int, int] = (192, 384)
) -> dict[str, Any]:
    image_a = _load_rgb(record["from"]["rgb"], shape_hw)
    image_b = _load_rgb(record["to"]["rgb"], shape_hw)
    depth_a = _load_depth(record["from"]["depth"], shape_hw)
    depth_b = _load_depth(record["to"]["depth"], shape_hw)
    rotation, translation = _panorai_pose(record)
    config = SphericalDoGSIFTConfig(
        octaves=3,
        levels_per_octave=3,
        max_features=600,
        patch_size=32,
        contrast_threshold=0.003,
        convolution_backend="numpy",
    )
    application = SphericalDoGFeatures(config)
    features_a = application(image_a, panorama_id=record["from"]["view_id"])
    features_b = application(image_b, panorama_id=record["to"]["view_id"])
    expected, visible = _visible_feature_oracle(
        features_a.bearings,
        features_a.source_erp_xy,
        depth_a,
        depth_b,
        rotation,
        translation,
    )
    errors = _minimum_angular_errors(expected[visible], features_b.bearings)
    repeatable = errors <= 1.0
    matches = FeatureMatcher().match(features_a, features_b)
    match_expected, match_visible = _visible_feature_oracle(
        features_a.bearings[matches.feature_indices_a],
        features_a.source_erp_xy[matches.feature_indices_a],
        depth_a,
        depth_b,
        rotation,
        translation,
    )
    match_errors = _angular_distance_deg(match_expected, matches.bearings_b)
    verified = match_visible & (match_errors <= 1.0) & matches.valid
    return {
        "pair_id": record["pair_id"],
        "visible_feature_count": int(visible.sum()),
        "repeatable_feature_count": int(repeatable.sum()),
        "repeatability": float(repeatable.mean()) if len(repeatable) else 0.0,
        "repeatable_location_errors_deg": errors[repeatable].tolist(),
        "match_count": int(len(matches)),
        "verified_match_count": int(verified.sum()),
        "verified_inlier_precision": (float(verified.mean()) if len(verified) else 0.0),
    }


def aggregate_dog_external(
    records: list[dict[str, Any]], corpus: str
) -> dict[str, Any]:
    pairs = [evaluate_dog_external_pair(item) for item in records]
    visible = sum(item["visible_feature_count"] for item in pairs)
    repeatable = sum(item["repeatable_feature_count"] for item in pairs)
    matches = sum(item["match_count"] for item in pairs)
    verified = sum(item["verified_match_count"] for item in pairs)
    location_errors = [
        error for item in pairs for error in item["repeatable_location_errors_deg"]
    ]
    return {
        "application": "dog",
        "evaluation": "external",
        "split": "heldout",
        "corpus": corpus,
        "metrics": {
            "repeatability": repeatable / visible if visible else 0.0,
            "median_location_error_deg": (
                float(np.median(location_errors)) if location_errors else 180.0
            ),
            "verified_inlier_precision": verified / matches if matches else 0.0,
        },
        "details": {"pairs": pairs},
    }


def _render_textured_sphere(
    camera_center: np.ndarray,
    shape_hw: tuple[int, int],
    radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape_hw, dtype=np.float32)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)
    centre_dot_ray = np.sum(rays * camera_center, axis=-1)
    distance = -centre_dot_ray + np.sqrt(
        centre_dot_ray**2 + radius**2 - float(camera_center @ camera_center)
    )
    points = camera_center + distance[..., None] * rays
    image = np.stack(
        (
            0.5
            + 0.2 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2])
            + 0.2 * np.cos(7.0 * points[..., 1]),
            0.5
            + 0.25 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2])
            + 0.15 * np.cos(8.0 * points[..., 0]),
            0.5
            + 0.25 * np.cos(6.0 * points[..., 2] + points[..., 0])
            + 0.15 * np.sin(9.0 * points[..., 1]),
        ),
        axis=-1,
    )
    return np.clip(image, 0, 1).astype(np.float32), distance.astype(np.float32)


def _solid_angle_metric_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    rows = np.arange(height, dtype=np.float64)
    latitude = math.pi / 2.0 - (rows + 0.5) * math.pi / height
    weights = np.cos(latitude)[:, None] * np.ones((1, width), dtype=np.float64)
    return weights / weights.sum()


def stereo_metrics(
    estimate: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    oracle = np.isfinite(reference) & (reference > 0)
    accepted = oracle & valid & np.isfinite(estimate) & (estimate > 0)
    weights = _solid_angle_metric_weights(reference.shape)
    coverage = float(weights[accepted].sum() / weights[oracle].sum())
    if not np.any(accepted):
        return {
            "abs_rel": float("inf"),
            "delta1": 0.0,
            "coverage": coverage,
            "errors": [],
            "confidences": [],
        }
    relative = np.abs(estimate[accepted] - reference[accepted]) / reference[accepted]
    ratio = np.maximum(
        estimate[accepted] / reference[accepted],
        reference[accepted] / estimate[accepted],
    )
    accepted_weights = weights[accepted]
    accepted_weights /= accepted_weights.sum()
    return {
        "abs_rel": float(np.sum(accepted_weights * relative)),
        "delta1": float(np.sum(accepted_weights * (ratio < 1.25))),
        "coverage": coverage,
        "errors": relative.tolist(),
        "confidences": confidence[accepted].tolist(),
    }


def _stereo_options(*, analytic: bool) -> SphericalStereoOptions:
    return SphericalStereoOptions(
        min_range=2.5 if analytic else 0.30,
        max_range=5.5 if analytic else 12.0,
        num_hypotheses=128 if analytic else 96,
        window_size=5,
        pole_margin_fraction=0.0,
        min_texture_std=0.002,
        min_confidence=0.0,
        max_matching_cost=0.9,
        bidirectional_consistency=False,
        filter_backend="numpy",
        pyramid_levels=1,
    )


def evaluate_stereo_analytic() -> dict[str, Any]:
    shape = (64, 128)
    centre_b_in_a = np.asarray((0.35, 0.03, 0.12), dtype=np.float32)
    first, reference = _render_textured_sphere(np.zeros(3), shape)
    second, _ = _render_textured_sphere(centre_b_in_a, shape)
    result = SphericalStereoDepth(_stereo_options(analytic=True))(
        first, second, np.eye(3), -centre_b_in_a
    )
    metrics = stereo_metrics(
        result.range, result.validity_mask, result.confidence, reference
    )
    return {
        "application": "stereo",
        "evaluation": "analytic",
        "corpus": "textured-sphere",
        "metrics": {key: metrics[key] for key in ("abs_rel", "delta1", "coverage")},
        "details": {
            "shape_hw": list(shape),
            "options": asdict(_stereo_options(analytic=True)),
        },
    }


def evaluate_stereo_external_pair(
    record: dict[str, Any], shape_hw: tuple[int, int] = (96, 192)
) -> dict[str, Any]:
    image_a = _load_rgb(record["from"]["rgb"], shape_hw)
    image_b = _load_rgb(record["to"]["rgb"], shape_hw)
    depth_a = _load_depth(record["from"]["depth"], shape_hw)
    rotation, translation = _panorai_pose(record)
    application = SphericalStereoDepth(_stereo_options(analytic=False))
    result = application(image_a, image_b, rotation, translation)
    metrics = stereo_metrics(
        result.range, result.validity_mask, result.confidence, depth_a
    )
    return {
        "pair_id": record["pair_id"],
        "abs_rel": metrics["abs_rel"],
        "delta1": metrics["delta1"],
        "coverage": metrics["coverage"],
        "errors": metrics["errors"],
        "confidences": metrics["confidences"],
    }


def _confidence_metrics(pairs: list[dict[str, Any]]) -> tuple[bool, float, list[float]]:
    errors = np.asarray(
        [error for item in pairs for error in item["errors"]], dtype=np.float64
    )
    confidence = np.asarray(
        [value for item in pairs for value in item["confidences"]], dtype=np.float64
    )
    if len(errors) < 4 or np.all(confidence == confidence[0]):
        return False, 0.0, []
    edges = np.quantile(confidence, (0.0, 0.25, 0.5, 0.75, 1.0))
    bins = []
    for index in range(4):
        include = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == 3
            else confidence < edges[index + 1]
        )
        bins.append(float(np.mean(errors[include])) if np.any(include) else math.inf)
    monotonic = all(left >= right for left, right in zip(bins, bins[1:]))
    overall = float(np.mean(errors))
    reduction = (overall - bins[-1]) / overall if overall > 0 else 0.0
    return monotonic, reduction, bins


def aggregate_stereo_external(
    records: list[dict[str, Any]], corpus: str
) -> dict[str, Any]:
    pairs = [evaluate_stereo_external_pair(item) for item in records]
    monotonic, reduction, confidence_bins = _confidence_metrics(pairs)
    first = records[0]
    shape = (48, 96)
    blank = np.zeros((*shape, 3), dtype=np.uint8)
    rotation, translation = _panorai_pose(first)
    degeneration = SphericalStereoDepth(_stereo_options(analytic=False))(
        blank, blank, rotation, translation
    )
    abstained = float(degeneration.validity_mask.mean()) <= 0.05
    return {
        "application": "stereo",
        "evaluation": "external",
        "split": "heldout",
        "corpus": corpus,
        "metrics": {
            "median_abs_rel": float(np.median([item["abs_rel"] for item in pairs])),
            "median_delta1": float(np.median([item["delta1"] for item in pairs])),
            "accepted_coverage": float(np.median([item["coverage"] for item in pairs])),
            "confidence_error_monotonic": monotonic,
            "top_confidence_quartile_error_reduction": reduction,
            "degenerations_diagnosed_or_abstained": abstained,
        },
        "details": {
            "pairs": pairs,
            "confidence_quartile_mean_relative_errors": confidence_bins,
            "textureless_valid_fraction": float(degeneration.validity_mask.mean()),
            "options": asdict(_stereo_options(analytic=False)),
        },
    }


def evaluate_core(selection: dict[str, Any]) -> dict[str, Any]:
    if selection.get("contract") != SELECTION_CONTRACT:
        raise ValueError("selection manifest has the wrong contract")
    if selection.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        raise ValueError("selection manifest targets a different protocol")
    records = selection.get("records")
    if not isinstance(records, list):
        raise ValueError("selection records must be a list")
    evidence = [
        evaluate_slic_analytic(),
        evaluate_dog_analytic(),
        evaluate_stereo_analytic(),
    ]
    for corpus in sorted(CORPUS_NAMES.values()):
        subset = [item for item in records if item.get("corpus") == corpus]
        if not subset:
            raise ValueError(f"selection has no records for {corpus}")
        evidence.append(aggregate_dog_external(subset, corpus))
        evidence.append(aggregate_stereo_external(subset, corpus))
    return {
        "contract": EVIDENCE_CONTRACT,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "selection_sha256": hashlib.sha256(canonical_json(selection)).hexdigest(),
        "records": evidence,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--split-jsonl", type=Path, required=True)
    select.add_argument("--overlap-jsonl", type=Path, required=True)
    select.add_argument("--matterport-root", type=Path, required=True)
    select.add_argument("--stanford-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    core = subparsers.add_parser("evaluate-core")
    core.add_argument("--selection", type=Path, required=True)
    core.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        manifest = build_selection_manifest(
            args.split_jsonl.resolve(),
            args.overlap_jsonl.resolve(),
            {
                "matterport360": args.matterport_root.resolve(),
                "stanford2d3d": args.stanford_root.resolve(),
            },
        )
        write_immutable_json(args.output.resolve(), manifest)
    else:
        evidence = evaluate_core(_load_json_object(args.selection.resolve()))
        write_immutable_json(args.output.resolve(), evidence)


if __name__ == "__main__":
    main()
