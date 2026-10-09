#!/usr/bin/env python3
"""Measure registered-cloud overlap for Matterport360 and Stanford2D3D pairs."""

# ruff: noqa: E402 -- benchmark scripts run directly from a source checkout

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.geometry import erp_pixels_to_rays

SCHEMA = "panorai-registered-cloud-pair-overlap/v1"
MANIFEST_SCHEMA = "panorai-registered-cloud-overlap-manifest/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(item)
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def equal_area_pixel_grid(shape_hw: tuple[int, int], target: int) -> np.ndarray:
    """Return deterministic approximately equal-area ERP pixel centres."""

    if target < 4:
        raise ValueError("target must be at least four")
    height, width = shape_hw
    rows = max(2, int(round(math.sqrt(target / 2.0))))
    columns = max(2, int(math.ceil(target / rows)))
    z = 1.0 - 2.0 * (np.arange(rows, dtype=np.float64) + 0.5) / rows
    latitude = np.arcsin(np.clip(z, -1.0, 1.0))
    y = (0.5 - latitude / math.pi) * height - 0.5
    x = (np.arange(columns, dtype=np.float64) + 0.5) / columns * width - 0.5
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    rounded = np.rint(pixels).astype(np.int64)
    rounded[:, 0] %= width
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    return np.unique(rounded, axis=0)


def sampled_world_cloud(view: dict[str, Any], target: int) -> np.ndarray:
    depth_path = Path(view["depth"]["path"])
    depth = np.load(depth_path, mmap_mode="r")
    if depth.ndim != 2:
        raise ValueError(f"expected 2D radial range at {depth_path}, got {depth.shape}")
    pixels = equal_area_pixel_grid(tuple(depth.shape), target)
    x = pixels[:, 0]
    y = pixels[:, 1]
    ranges = np.asarray(depth[y, x], dtype=np.float64)
    rays = np.asarray(
        erp_pixels_to_rays(pixels.astype(np.float64), tuple(depth.shape)),
        dtype=np.float64,
    )
    valid = np.all(np.isfinite(rays), axis=1) & np.isfinite(ranges) & (ranges > 0.0)
    local = rays[valid] * ranges[valid, None]
    transform = np.asarray(view["pose"]["matrix"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"invalid pose shape for {view['view_id']}: {transform.shape}")
    world = local @ transform[:3, :3].T + transform[:3, 3]
    return world[np.all(np.isfinite(world), axis=1)]


def cloud_overlap(
    first: np.ndarray, second: np.ndarray, *, distance_tolerance_m: float
) -> dict[str, Any]:
    if distance_tolerance_m <= 0.0:
        raise ValueError("distance tolerance must be positive")
    if not len(first) or not len(second):
        first_fraction = second_fraction = 0.0
        first_count = second_count = 0
    else:
        first_distances, _ = cKDTree(second).query(
            first, k=1, distance_upper_bound=distance_tolerance_m
        )
        second_distances, _ = cKDTree(first).query(
            second, k=1, distance_upper_bound=distance_tolerance_m
        )
        first_count = int(np.count_nonzero(np.isfinite(first_distances)))
        second_count = int(np.count_nonzero(np.isfinite(second_distances)))
        first_fraction = first_count / len(first)
        second_fraction = second_count / len(second)
    return {
        "from_points": int(len(first)),
        "to_points": int(len(second)),
        "from_in_to_count": first_count,
        "to_in_from_count": second_count,
        "from_in_to_fraction": float(first_fraction),
        "to_in_from_fraction": float(second_fraction),
        "minimum_directional_cloud_overlap": float(
            min(first_fraction, second_fraction)
        ),
        "symmetric_cloud_overlap": float(math.sqrt(first_fraction * second_fraction)),
        "distance_tolerance_m": float(distance_tolerance_m),
    }


def _relative_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.inv(second) @ first


def _group_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    views = {row["view_id"]: row for row in job["views"]}
    clouds = {
        view_id: sampled_world_cloud(view, job["sample_points"])
        for view_id, view in views.items()
    }
    records = []
    for pair in job["pairs"]:
        first_id = pair["from_view_id"]
        second_id = pair["to_view_id"]
        first_pose = np.asarray(views[first_id]["pose"]["matrix"], dtype=np.float64)
        second_pose = np.asarray(views[second_id]["pose"]["matrix"], dtype=np.float64)
        relative = _relative_transform(first_pose, second_pose)
        reference = np.asarray(pair["reference"]["T_to_from_from"], dtype=np.float64)
        transform_max_abs_error = float(np.max(np.abs(relative - reference)))
        if transform_max_abs_error > 1e-6:
            raise ValueError(
                f"pose/reference mismatch for {pair['pair_id']}: "
                f"{transform_max_abs_error}"
            )
        metrics = cloud_overlap(
            clouds[first_id],
            clouds[second_id],
            distance_tolerance_m=job["distance_tolerance_m"],
        )
        records.append(
            {
                "schema": SCHEMA,
                "pair_id": pair["pair_id"],
                "dataset_id": pair["dataset_id"],
                "spatial_group_id": pair["spatial_group_id"],
                "from_view_id": first_id,
                "to_view_id": second_id,
                "sampling": {
                    "policy": "equal-area-erp-nearest-depth",
                    "requested_points_per_view": job["sample_points"],
                    "depth_quantity": "radial_range_m",
                },
                "reference_transform_max_abs_error": transform_max_abs_error,
                **metrics,
            }
        )
    return records


def measure(
    pairs: list[dict[str, Any]],
    views: list[dict[str, Any]],
    *,
    sample_points: int,
    distance_tolerance_m: float,
    workers: int,
) -> list[dict[str, Any]]:
    view_index = {str(row["view_id"]): row for row in views}
    if len(view_index) != len(views):
        raise ValueError("duplicate view_id")
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate pair_id")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if (
            pair["from_view_id"] not in view_index
            or pair["to_view_id"] not in view_index
        ):
            raise ValueError(f"missing view metadata for {pair['pair_id']}")
        by_group[str(pair["spatial_group_id"])].append(pair)
    jobs = []
    for group, selected_pairs in sorted(by_group.items()):
        view_ids = {
            view_id
            for pair in selected_pairs
            for view_id in (pair["from_view_id"], pair["to_view_id"])
        }
        jobs.append(
            {
                "group": group,
                "pairs": selected_pairs,
                "views": [view_index[view_id] for view_id in sorted(view_ids)],
                "sample_points": sample_points,
                "distance_tolerance_m": distance_tolerance_m,
            }
        )
    if workers == 1:
        batches = [_group_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = list(executor.map(_group_job, jobs))
    records = [record for batch in batches for record in batch]
    records.sort(key=lambda row: (row["dataset_id"], row["pair_id"]))
    if len(records) != len(pairs):
        raise RuntimeError("not every pair produced an overlap record")
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    pairs_path = args.pairs_evaluation.resolve()
    views_path = args.views_evaluation.resolve()
    pairs = _read_jsonl(pairs_path)
    views = _read_jsonl(views_path)
    records = measure(
        pairs,
        views,
        sample_points=args.sample_points,
        distance_tolerance_m=args.distance_tolerance_m,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "registered-cloud-overlap.jsonl"
    _atomic_text(
        rows_path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records
        ),
    )
    datasets = {}
    for dataset in sorted({row["dataset_id"] for row in records}):
        values = np.asarray(
            [
                row["minimum_directional_cloud_overlap"]
                for row in records
                if row["dataset_id"] == dataset
            ],
            dtype=np.float64,
        )
        datasets[dataset] = {
            "pairs": int(len(values)),
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(np.max(values)),
        }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "sources": {
            "pairs_evaluation": {
                "path": str(pairs_path),
                "sha256": _sha256(pairs_path),
            },
            "views_evaluation": {
                "path": str(views_path),
                "sha256": _sha256(views_path),
            },
        },
        "configuration": {
            "sample_points": args.sample_points,
            "distance_tolerance_m": args.distance_tolerance_m,
            "workers": args.workers,
            "sampling_policy": "equal-area-erp-nearest-depth",
            "depth_quantity": "radial_range_m",
        },
        "pair_count": len(records),
        "datasets": datasets,
        "output": {"path": str(rows_path.resolve()), "sha256": _sha256(rows_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-evaluation", type=Path, required=True)
    parser.add_argument("--views-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--distance-tolerance-m", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
