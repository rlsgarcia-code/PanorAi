#!/usr/bin/env python3
"""Build a P74 pose-evaluation queue from bidirectional co-visibility.

This is an evaluation-data tool, not a public PanorAi API.  It consumes the
organized XYZ arrays and frozen scan poses only to decide whether a pair is a
meaningful pose-recovery trial.  RGB is deliberately not read during pair
selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable
import zipfile

import numpy as np
from scipy.spatial import cKDTree


SCHEMA_VERSION = "panorai-p74-pair-eligibility/v1"
PAIR_SCHEMA_VERSION = "panorai-p74-pair-observability/v1"
QUEUE_SCHEMA_VERSION = "panorai-p74-pose-queue/v1"
NATIVE_ADAPTER = "eq-native-polar-0-150-endpoint-inclusive/v1"
P74_POLAR_LIMIT_RAD = math.radians(150.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", required=True, type=Path)
    parser.add_argument("--rgb-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-points", type=int, default=6000)
    parser.add_argument("--candidate-radius-m", type=float, default=15.0)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.25)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--cloud-distance-tolerance-m", type=float, default=0.25)
    parser.add_argument("--minimum-cloud-overlap", type=float, default=0.50)
    parser.add_argument("--primary-cloud-overlap", type=float, default=0.70)
    parser.add_argument("--minimum-covisible-overlap", type=float, default=0.03)
    parser.add_argument("--minimum-directional-overlap", type=float, default=0.03)
    parser.add_argument("--minimum-coverage", type=float, default=0.08)
    parser.add_argument("--primary-coverage", type=float, default=0.15)
    parser.add_argument("--minimum-parallax-samples", type=int, default=64)
    parser.add_argument("--minimum-parallax-deg", type=float, default=0.5)
    parser.add_argument("--queue-pairs-per-family", type=int, default=3)
    parser.add_argument("--old-pairs", type=Path)
    return parser.parse_args()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mmap_npy_member(path: Path, member: str) -> np.memmap:
    """Memory-map an uncompressed NPY member without materializing the NPZ."""

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{member} is compressed; direct mmap is unsafe")
        header_offset = info.header_offset
    with path.open("rb") as stream:
        stream.seek(header_offset)
        header = stream.read(30)
        name_length = struct.unpack_from("<H", header, 26)[0]
        extra_length = struct.unpack_from("<H", header, 28)[0]
        stream.seek(header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            reader = np.lib.format.read_array_header_1_0
        elif version in {(2, 0), (3, 0)}:
            reader = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"unsupported NPY version {version} in {path}:{member}")
        shape, fortran_order, dtype = reader(stream)
        offset = stream.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def family_id(panorama_id: str) -> str:
    parts = panorama_id.split("+")
    if len(parts) < 3 or parts[0] != "P-74":
        raise ValueError(f"unexpected P74 panorama ID: {panorama_id}")
    return "+".join(parts[1:-1])


def native_pixels_from_local(
    points_local: np.ndarray, *, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project P74 scan-local points into its audited native polar raster."""

    points = np.asarray(points_local, dtype=np.float64)
    radial = np.linalg.norm(points, axis=1)
    finite = np.all(np.isfinite(points), axis=1) & (radial > 1e-8)
    safe_radial = np.where(finite, radial, 1.0)
    unit = points / safe_radial[:, None]
    longitude = np.arctan2(-unit[:, 1], unit[:, 0])
    polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    period = width - 1
    x_float = np.mod(longitude / (2.0 * math.pi) * period, period)
    y_float = polar / P74_POLAR_LIMIT_RAD * (height - 1)
    x = np.mod(np.rint(x_float).astype(np.int64), period)
    y = np.clip(np.rint(y_float).astype(np.int64), 0, height - 1)
    supported = finite & (polar <= P74_POLAR_LIMIT_RAD + 1e-12)
    return x, y, radial, supported


def grid_coordinates(
    height: int, width: int, target: int
) -> tuple[np.ndarray, np.ndarray]:
    proposed = max(target * 4, target)
    rows = max(2, int(round(math.sqrt(proposed * height / width))))
    columns = max(2, int(math.ceil(proposed / rows)))
    row_ids = np.linspace(0, height - 1, rows, dtype=np.int64)
    column_ids = np.linspace(0, width - 1, columns, dtype=np.int64)
    rr, cc = np.meshgrid(row_ids, column_ids, indexing="ij")
    return rr.ravel(), cc.ravel()


@dataclass
class Sample:
    panorama_id: str
    family: str
    npz_path: Path
    rgb_path: Path
    xyz: np.memmap
    rotation: np.ndarray
    translation: np.ndarray
    sampled_local: np.ndarray
    sampled_scene: np.ndarray
    scene_tree: cKDTree
    proxy_range: np.ndarray
    proxy_valid: np.ndarray
    sampled_rows: np.ndarray
    sampled_columns: np.ndarray
    proposed_samples: int
    valid_grid_samples: int

    @property
    def height(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def width(self) -> int:
        return int(self.xyz.shape[1])


def load_sample(path: Path, rgb_root: Path, target: int) -> Sample:
    panorama_id = path.stem
    xyz = mmap_npy_member(path, "xyz_image.npy")
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"unexpected XYZ shape in {path}: {xyz.shape}")
    rotation = np.asarray(
        mmap_npy_member(path, "rotation_matrix.npy"), dtype=np.float64
    ).copy()
    translation = (
        np.asarray(mmap_npy_member(path, "translation.npy"), dtype=np.float64)
        .reshape(3)
        .copy()
    )
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-9
    ):
        raise ValueError(f"invalid rotation in {path}")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError(f"improper rotation in {path}")
    if not np.all(np.isfinite(translation)):
        raise ValueError(f"non-finite translation in {path}")

    rows, columns = grid_coordinates(xyz.shape[0], xyz.shape[1], target)
    local = np.asarray(xyz[rows, columns], dtype=np.float64)
    radial = np.linalg.norm(local, axis=1)
    valid = np.all(np.isfinite(local), axis=1) & (radial > 1e-3)
    grid_row_count = len(np.unique(rows))
    grid_column_count = len(np.unique(columns))
    proxy_range = radial.reshape(grid_row_count, grid_column_count)
    proxy_valid = valid.reshape(grid_row_count, grid_column_count)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > target:
        keep = np.linspace(0, len(valid_indices) - 1, target, dtype=np.int64)
        valid_indices = valid_indices[keep]
    rgb_path = rgb_root / f"{panorama_id}_rgb.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    sampled_local = local[valid_indices]
    sampled_scene = sampled_local @ rotation.T + translation
    return Sample(
        panorama_id=panorama_id,
        family=family_id(panorama_id),
        npz_path=path,
        rgb_path=rgb_path,
        xyz=xyz,
        rotation=rotation,
        translation=translation,
        sampled_local=sampled_local,
        sampled_scene=sampled_scene,
        scene_tree=cKDTree(sampled_scene),
        proxy_range=proxy_range,
        proxy_valid=proxy_valid,
        sampled_rows=rows[valid_indices],
        sampled_columns=columns[valid_indices],
        proposed_samples=int(len(rows)),
        valid_grid_samples=int(np.count_nonzero(valid)),
    )


def coverage_fraction(
    rows: np.ndarray, columns: np.ndarray, *, height: int, width: int
) -> float:
    if not len(rows):
        return 0.0
    row_bins = np.minimum((rows.astype(np.float64) / height * 5).astype(int), 4)
    period = width - 1
    column_bins = np.minimum(
        (np.mod(columns, period).astype(np.float64) / period * 12).astype(int), 11
    )
    return float(len(np.unique(row_bins * 12 + column_bins)) / 60.0)


def directed_metrics(
    source: Sample,
    target: Sample,
    *,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> tuple[dict, np.ndarray]:
    local = source.sampled_local
    source_count = int(len(local))
    scene = local @ source.rotation.T + source.translation
    target_local = (scene - target.translation) @ target.rotation
    x, y, predicted_range, in_native_support = native_pixels_from_local(
        target_local, height=target.height, width=target.width
    )
    proxy_y = np.rint(
        y / (target.height - 1) * (target.proxy_range.shape[0] - 1)
    ).astype(np.int64)
    proxy_x = np.rint(
        x / (target.width - 1) * (target.proxy_range.shape[1] - 1)
    ).astype(np.int64)
    observed_range = target.proxy_range[proxy_y, proxy_x]
    supported = in_native_support & target.proxy_valid[proxy_y, proxy_x]
    tolerance = np.maximum(
        absolute_tolerance_m,
        relative_tolerance * np.where(supported, observed_range, 0.0),
    )
    residual = np.abs(predicted_range - observed_range)
    consistent = supported & (residual <= tolerance)
    comparable = supported & (predicted_range <= observed_range + tolerance)

    consistent_scene = scene[consistent]
    rays_source = consistent_scene - source.translation
    rays_target = consistent_scene - target.translation
    denominator = np.linalg.norm(rays_source, axis=1) * np.linalg.norm(
        rays_target, axis=1
    )
    valid_angle = denominator > 1e-12
    cosine = (
        np.sum(rays_source[valid_angle] * rays_target[valid_angle], axis=1)
        / denominator[valid_angle]
    )
    parallax = np.arccos(np.clip(cosine, -1.0, 1.0))

    consistent_count = int(np.count_nonzero(consistent))
    comparable_count = int(np.count_nonzero(comparable))
    return {
        "source_points": source_count,
        "native_supported_points": int(np.count_nonzero(in_native_support)),
        "target_valid_points": int(np.count_nonzero(supported)),
        "comparable_points": comparable_count,
        "consistent_points": consistent_count,
        "target_support_fraction": (
            float(np.count_nonzero(supported) / source_count) if source_count else 0.0
        ),
        "consistent_source_fraction": (
            float(consistent_count / source_count) if source_count else 0.0
        ),
        "visible_consistency": (
            float(consistent_count / comparable_count) if comparable_count else 0.0
        ),
        "source_coverage_fraction": coverage_fraction(
            source.sampled_rows[consistent],
            source.sampled_columns[consistent],
            height=source.height,
            width=source.width,
        ),
        "target_coverage_fraction": coverage_fraction(
            y[consistent], x[consistent], height=target.height, width=target.width
        ),
    }, parallax


def scene_cloud_intersection_metrics(
    left: Sample, right: Sample, *, distance_tolerance_m: float
) -> dict:
    """Measure the sampled cloud intersection in the shared scene frame.

    The directional fractions answer how much of each scan has a surface in the
    other scan within the metric tolerance.  Their geometric mean is the single
    continuous difficulty reference used by the queue.
    """

    left_scene = left.sampled_scene
    right_scene = right.sampled_scene
    if not len(left_scene) or not len(right_scene):
        return {
            "distance_tolerance_m": distance_tolerance_m,
            "left_points": int(len(left_scene)),
            "right_points": int(len(right_scene)),
            "left_in_right_count": 0,
            "right_in_left_count": 0,
            "left_in_right_fraction": 0.0,
            "right_in_left_fraction": 0.0,
            "symmetric_fraction": 0.0,
            "matched_distance_m": quantiles(np.empty(0)),
        }
    left_distances, _ = right.scene_tree.query(
        left_scene, k=1, distance_upper_bound=distance_tolerance_m
    )
    right_distances, _ = left.scene_tree.query(
        right_scene, k=1, distance_upper_bound=distance_tolerance_m
    )
    left_matched = np.isfinite(left_distances)
    right_matched = np.isfinite(right_distances)
    left_fraction = float(np.count_nonzero(left_matched) / len(left_scene))
    right_fraction = float(np.count_nonzero(right_matched) / len(right_scene))
    matched_distances = np.concatenate(
        (left_distances[left_matched], right_distances[right_matched])
    )
    return {
        "distance_tolerance_m": distance_tolerance_m,
        "left_points": int(len(left_scene)),
        "right_points": int(len(right_scene)),
        "left_in_right_count": int(np.count_nonzero(left_matched)),
        "right_in_left_count": int(np.count_nonzero(right_matched)),
        "left_in_right_fraction": left_fraction,
        "right_in_left_fraction": right_fraction,
        "symmetric_fraction": math.sqrt(left_fraction * right_fraction),
        "matched_distance_m": quantiles(matched_distances),
    }


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def classify_pair(
    *,
    scene_cloud_overlap: float,
    minimum_directional_cloud_overlap: float,
    covisible_overlap: float,
    minimum_directional: float,
    coverage: float,
    parallax_count: int,
    parallax_median_rad: float | None,
    minimum_cloud_overlap: float,
    primary_cloud_overlap: float,
    minimum_covisible_overlap: float,
    minimum_directional_overlap: float,
    minimum_coverage: float,
    primary_coverage: float,
    minimum_parallax_samples: int,
    minimum_parallax_deg: float,
) -> tuple[str, list[str]]:
    if (
        scene_cloud_overlap < 0.01
        or covisible_overlap < 0.01
        or minimum_directional < 0.005
    ):
        return "no_overlap_negative", ["cloud_or_covisibility_below_negative_gate"]

    failures = []
    if minimum_directional_cloud_overlap < minimum_cloud_overlap:
        failures.append("one_direction_scene_cloud_overlap_below_pose_minimum")
    if covisible_overlap < minimum_covisible_overlap:
        failures.append("covisible_overlap_below_pose_minimum")
    if minimum_directional < minimum_directional_overlap:
        failures.append("one_direction_below_pose_minimum")
    if coverage < minimum_coverage:
        failures.append("spatial_coverage_below_pose_minimum")
    if parallax_count < minimum_parallax_samples:
        failures.append("parallax_samples_below_pose_minimum")
    if parallax_median_rad is None:
        failures.append("parallax_unavailable")
    elif parallax_median_rad < math.radians(minimum_parallax_deg):
        failures.append("median_parallax_below_pose_minimum")
    if failures:
        return "ambiguous", failures

    stress = []
    if scene_cloud_overlap < primary_cloud_overlap:
        stress.append("scene_cloud_overlap_below_primary")
    if coverage < primary_coverage:
        stress.append("coverage_below_primary")
    if stress:
        return "pose_stress", stress
    return "pose_primary", ["primary_observability_thresholds_met"]


def pair_record(left: Sample, right: Sample, args: argparse.Namespace) -> dict:
    if left.family != right.family:
        raise ValueError("cross-family P74 pair prohibited")
    left_to_right, parallax_lr = directed_metrics(
        left,
        right,
        absolute_tolerance_m=args.absolute_tolerance_m,
        relative_tolerance=args.relative_tolerance,
    )
    right_to_left, parallax_rl = directed_metrics(
        right,
        left,
        absolute_tolerance_m=args.absolute_tolerance_m,
        relative_tolerance=args.relative_tolerance,
    )
    cloud_intersection = scene_cloud_intersection_metrics(
        left,
        right,
        distance_tolerance_m=args.cloud_distance_tolerance_m,
    )
    directional = (
        left_to_right["consistent_source_fraction"],
        right_to_left["consistent_source_fraction"],
    )
    covisible_overlap = math.sqrt(directional[0] * directional[1])
    coverage_terms = (
        left_to_right["source_coverage_fraction"],
        left_to_right["target_coverage_fraction"],
        right_to_left["source_coverage_fraction"],
        right_to_left["target_coverage_fraction"],
    )
    coverage = min(coverage_terms)
    parallax = np.concatenate((parallax_lr, parallax_rl))
    parallax_stats = quantiles(parallax)
    minimum_directional_cloud_overlap = min(
        cloud_intersection["left_in_right_fraction"],
        cloud_intersection["right_in_left_fraction"],
    )
    status, reasons = classify_pair(
        scene_cloud_overlap=cloud_intersection["symmetric_fraction"],
        minimum_directional_cloud_overlap=minimum_directional_cloud_overlap,
        covisible_overlap=covisible_overlap,
        minimum_directional=min(directional),
        coverage=coverage,
        parallax_count=len(parallax),
        parallax_median_rad=parallax_stats["median"],
        minimum_cloud_overlap=args.minimum_cloud_overlap,
        primary_cloud_overlap=args.primary_cloud_overlap,
        minimum_covisible_overlap=args.minimum_covisible_overlap,
        minimum_directional_overlap=args.minimum_directional_overlap,
        minimum_coverage=args.minimum_coverage,
        primary_coverage=args.primary_coverage,
        minimum_parallax_samples=args.minimum_parallax_samples,
        minimum_parallax_deg=args.minimum_parallax_deg,
    )
    return {
        "schema": PAIR_SCHEMA_VERSION,
        "pair_id": f"p74-pair::{left.family}::{left.panorama_id}::{right.panorama_id}",
        "family_id": left.family,
        "from_view_id": left.panorama_id,
        "to_view_id": right.panorama_id,
        "from_rgb_path": str(left.rgb_path),
        "to_rgb_path": str(right.rgb_path),
        "baseline_m": float(np.linalg.norm(left.translation - right.translation)),
        "difficulty_reference": {
            "quantity": ("minimum_directional_scene_cloud_intersection_fraction"),
            "higher_means": "more_shared_scene_and_easier_expected_pair",
            "value": minimum_directional_cloud_overlap,
        },
        "scene_cloud_intersection": cloud_intersection,
        "minimum_directional_scene_cloud_overlap": (minimum_directional_cloud_overlap),
        "symmetric_covisible_fraction": covisible_overlap,
        "minimum_directional_consistent_fraction": min(directional),
        "minimum_spatial_coverage_fraction": coverage,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "parallax_proxy": {
            "sample_count": int(len(parallax)),
            "units": "rad",
            **parallax_stats,
        },
        "selection": {"status": status, "reasons": reasons},
    }


class UnionFind:
    def __init__(self, nodes: Iterable[str]):
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        self.parent[max(root_left, root_right)] = min(root_left, root_right)
        return True


def maximum_spanning_forest(nodes: list[str], pairs: list[dict]) -> list[str]:
    """Return a deterministic co-visible forest; closure edges are never required."""

    union_find = UnionFind(nodes)
    selected = []
    eligible = [
        row
        for row in pairs
        if row["selection"]["status"] in {"pose_primary", "pose_stress"}
    ]
    eligible.sort(
        key=lambda row: (
            -row["difficulty_reference"]["value"],
            row["pair_id"],
        )
    )
    for row in eligible:
        if union_find.union(row["from_view_id"], row["to_view_id"]):
            selected.append(row["pair_id"])
    return selected


def stratified_queue(pairs: list[dict], pairs_per_family: int) -> list[dict]:
    """Select overlap-spread eligible pairs without imposing a three-view clique."""

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        if row["selection"]["status"] in {"pose_primary", "pose_stress"}:
            by_family[row["family_id"]].append(row)
    queue = []
    for family in sorted(by_family):
        candidates = sorted(
            by_family[family],
            key=lambda row: (
                row["difficulty_reference"]["value"],
                row["pair_id"],
            ),
        )
        count = min(pairs_per_family, len(candidates))
        if not count:
            continue
        indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        for rank, index in enumerate(indices):
            source = candidates[int(index)]
            queue.append(
                {
                    "schema": QUEUE_SCHEMA_VERSION,
                    "queue_id": f"p74-pose::{family}::{rank + 1:02d}",
                    "family_id": family,
                    "pair_id": source["pair_id"],
                    "from_view_id": source["from_view_id"],
                    "to_view_id": source["to_view_id"],
                    "from_rgb_path": source["from_rgb_path"],
                    "to_rgb_path": source["to_rgb_path"],
                    "eligibility_status": source["selection"]["status"],
                    "scene_cloud_overlap": source["difficulty_reference"]["value"],
                    "symmetric_scene_cloud_overlap": source["scene_cloud_intersection"][
                        "symmetric_fraction"
                    ],
                    "symmetric_covisible_fraction": source[
                        "symmetric_covisible_fraction"
                    ],
                    "selection_basis": ("eligible_scene_cloud_overlap_order_statistic"),
                }
            )
    return queue


def old_pair_keys(path: Path | None) -> set[frozenset[str]]:
    if path is None:
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {frozenset((row["from_view_id"], row["to_view_id"])) for row in rows}


def pair_key(row: dict) -> frozenset[str]:
    return frozenset((row["from_view_id"], row["to_view_id"]))


def main() -> None:
    args = parse_args()
    if args.sample_points < 64:
        raise ValueError("sample-points must be >= 64")
    paths = sorted(args.npz_root.glob("P-74+*.npz"))
    if not paths:
        raise FileNotFoundError(f"no P74 NPZs under {args.npz_root}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[family_id(path.stem)].append(path)

    all_pairs = []
    nodes = []
    graph_rows = []
    for family in sorted(groups):
        samples = [
            load_sample(path, args.rgb_root, args.sample_points)
            for path in groups[family]
        ]
        samples.sort(key=lambda item: item.panorama_id)
        for sample in samples:
            nodes.append(
                {
                    "panorama_id": sample.panorama_id,
                    "family_id": family,
                    "npz_path": str(sample.npz_path),
                    "rgb_path": str(sample.rgb_path),
                    "shape": list(sample.xyz.shape),
                    "translation_m": sample.translation.tolist(),
                    "proposed_grid_samples": sample.proposed_samples,
                    "valid_grid_samples": sample.valid_grid_samples,
                    "retained_samples": int(len(sample.sampled_local)),
                }
            )
        family_pairs = []
        candidate_count = 0
        for index, left in enumerate(samples):
            for right in samples[index + 1 :]:
                baseline = float(np.linalg.norm(left.translation - right.translation))
                if baseline > args.candidate_radius_m:
                    continue
                candidate_count += 1
                family_pairs.append(pair_record(left, right, args))
        family_pairs.sort(key=lambda row: row["pair_id"])
        all_pairs.extend(family_pairs)
        forest = maximum_spanning_forest(
            [sample.panorama_id for sample in samples], family_pairs
        )
        status_counts = Counter(row["selection"]["status"] for row in family_pairs)
        graph_rows.append(
            {
                "family_id": family,
                "node_count": len(samples),
                "candidate_pair_count": candidate_count,
                "status_counts": dict(sorted(status_counts.items())),
                "maximum_spanning_forest_pair_ids": forest,
                "forest_edge_count": len(forest),
                "clique_required": False,
            }
        )
        print(
            canonical_json(
                {
                    "family": family,
                    "nodes": len(samples),
                    "candidates": candidate_count,
                    "statuses": dict(sorted(status_counts.items())),
                }
            ),
            flush=True,
        )
        for sample in samples:
            sample.xyz._mmap.close()

    all_pairs.sort(key=lambda row: row["pair_id"])
    queue = stratified_queue(all_pairs, args.queue_pairs_per_family)
    old_keys = old_pair_keys(args.old_pairs)
    old_rows = [row for row in all_pairs if pair_key(row) in old_keys]
    old_found = {pair_key(row) for row in old_rows}
    if old_keys and old_found != old_keys:
        missing = sorted("::".join(sorted(key)) for key in old_keys - old_found)
        raise RuntimeError(f"old queue pairs missing from candidates: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / "nodes.jsonl"
    pairs_path = args.output_dir / "all-candidate-pairs.jsonl"
    eligible_path = args.output_dir / "pose-eligible-pairs.jsonl"
    negatives_path = args.output_dir / "no-overlap-negatives.jsonl"
    queue_path = args.output_dir / "pose-evaluation-queue.jsonl"
    graph_path = args.output_dir / "family-graphs.jsonl"
    old_path = args.output_dir / "old-nine-audit.jsonl"
    write_jsonl(nodes_path, sorted(nodes, key=lambda row: row["panorama_id"]))
    write_jsonl(pairs_path, all_pairs)
    write_jsonl(
        eligible_path,
        [
            row
            for row in all_pairs
            if row["selection"]["status"] in {"pose_primary", "pose_stress"}
        ],
    )
    write_jsonl(
        negatives_path,
        [
            row
            for row in all_pairs
            if row["selection"]["status"] == "no_overlap_negative"
        ],
    )
    write_jsonl(queue_path, queue)
    write_jsonl(graph_path, graph_rows)
    write_jsonl(old_path, old_rows)

    counts = Counter(row["selection"]["status"] for row in all_pairs)
    old_counts = Counter(row["selection"]["status"] for row in old_rows)
    output_paths = (
        nodes_path,
        pairs_path,
        eligible_path,
        negatives_path,
        queue_path,
        graph_path,
        old_path,
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "dataset": "P-74",
        "input": {
            "npz_root": str(args.npz_root),
            "rgb_root": str(args.rgb_root),
            "panorama_count": len(paths),
            "family_count": len(groups),
            "old_pairs": str(args.old_pairs) if args.old_pairs else None,
        },
        "geometry": {
            "native_adapter": NATIVE_ADAPTER,
            "scene_from_scan_local": "x_scene = R @ x_local + t",
            "validity": "finite(xyz) and norm(xyz) > 1e-3 m",
            "horizontal_longitude": "atan2(-Y, X)",
            "horizontal_period_px": "width - 1",
            "polar_support_deg": [0.0, 150.0],
            "sampling": "nearest",
            "occlusion_proxy": (
                "nearest sample in the deterministic per-view polar range grid"
            ),
            "absolute_tolerance_m": args.absolute_tolerance_m,
            "relative_tolerance": args.relative_tolerance,
            "cloud_distance_tolerance_m": args.cloud_distance_tolerance_m,
            "sample_points_per_view_max": args.sample_points,
        },
        "selection_thresholds": {
            "candidate_radius_m": args.candidate_radius_m,
            "minimum_cloud_overlap": args.minimum_cloud_overlap,
            "primary_cloud_overlap": args.primary_cloud_overlap,
            "minimum_covisible_overlap": args.minimum_covisible_overlap,
            "minimum_directional_overlap": args.minimum_directional_overlap,
            "minimum_coverage": args.minimum_coverage,
            "primary_coverage": args.primary_coverage,
            "minimum_parallax_samples": args.minimum_parallax_samples,
            "minimum_parallax_deg": args.minimum_parallax_deg,
            "negative_overlap": 0.01,
            "negative_minimum_directional_overlap": 0.005,
        },
        "counts": {
            "candidate_pairs": len(all_pairs),
            "pairs_by_status": dict(sorted(counts.items())),
            "queue_pairs": len(queue),
            "old_pairs_found": len(old_rows),
            "old_pairs_by_status": dict(sorted(old_counts.items())),
        },
        "outputs": {
            path.name: {
                "path": str(path),
                "records": sum(1 for line in path.read_text().splitlines() if line),
                "sha256": sha256(path),
            }
            for path in output_paths
        },
        "independence_warning": (
            "XYZ, poses and co-visibility derive from the same registered scans; "
            "they define trial eligibility, not an independent pose-accuracy oracle."
        ),
        "rgb_used_for_selection": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
