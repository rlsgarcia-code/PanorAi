#!/usr/bin/env python3
"""Summarize the exact-v3.5.0 replay of representative taxonomy pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(outcome: dict[str, Any]) -> str:
    if not outcome.get("returned"):
        return "no-pose"
    if outcome.get("catastrophic_accepted"):
        return "catastrophic-accepted"
    if outcome.get("accepted") or outcome.get("quality_accepted"):
        return "precise-accepted" if outcome.get("precise") else "imprecise-accepted"
    return "returned-rejected"


def run(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    historical = {
        (row["dataset_id"], row["pair_id"]): row for row in taxonomy["representatives"]
    }
    raw_paths = sorted(args.results_dir.glob("*.json"))
    if len(raw_paths) != len(historical):
        raise ValueError(
            f"expected {len(historical)} replay files, found {len(raw_paths)}"
        )
    comparisons = []
    raw_hashes = {}
    for path in raw_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["dataset_id"], row["pair_id"])
        if key not in historical:
            raise KeyError(f"unexpected replay key: {key}")
        old = historical[key]
        pose = row["pose"]
        if row["native"] != {
            "convolution_backend": "native",
            "native_filter_available": True,
            "native_pose_kernels_available": True,
            "numpy_fallback_permitted": False,
        }:
            raise RuntimeError(f"native-route contract failed for {key}")
        if row["route"]["route"]["detector_method"] != "detect_batch":
            raise RuntimeError(f"batch detector contract failed for {key}")
        if row["route"]["route"]["patch_provider_max_workers"] != 4:
            raise RuntimeError(f"patch-worker contract failed for {key}")
        comparisons.append(
            {
                "taxonomy_category": old["category"],
                "dataset_id": key[0],
                "pair_id": key[1],
                "historical_state": _state(old["outcomes"]),
                "optimized_main_state": _state(pose),
                "matches": row["counts"]["matches"],
                "inliers": pose.get("inlier_count"),
                "rotation_error_deg": pose.get("rotation_error_deg"),
                "translation_direction_error_deg": pose.get(
                    "translation_direction_error_deg"
                ),
                "raw_quality_score": pose.get("raw_quality_score"),
                "detection_pair_seconds": row["timings_seconds"]["detection_pair"],
                "pair_total_seconds": row["timings_seconds"]["pair_total"],
                "peak_rss_mib": row["peak_rss_mib"],
            }
        )
        raw_hashes[path.name] = _sha256(path)
    comparisons.sort(key=lambda row: row["taxonomy_category"])
    package_versions = {
        json.loads(path.read_text(encoding="utf-8"))["package"]["version"]
        for path in raw_paths
    }
    import_paths = {
        json.loads(path.read_text(encoding="utf-8"))["package"]["import_path"]
        for path in raw_paths
    }
    if package_versions != {"3.5.0"} or len(import_paths) != 1:
        raise RuntimeError("replay did not use one isolated PanorAi 3.5.0 install")
    result = {
        "schema": "panorai-unified-taxonomy-replay-summary/v1",
        "status": "five-pair mechanism check; not population-level validation",
        "package": {
            "version": "3.5.0",
            "import_path": next(iter(import_paths)),
            "wheel": str(args.wheel.resolve()),
            "wheel_sha256": _sha256(args.wheel),
            "origin_main_commit": args.origin_main_commit,
            "origin_main_tree": args.origin_main_tree,
            "wheel_source_commit": args.wheel_source_commit,
            "wheel_source_tree": args.wheel_source_tree,
            "tree_identity_proves_same_source": (
                args.origin_main_tree == args.wheel_source_tree
            ),
        },
        "route": {
            "resolution_hw": [1024, 2048],
            "detector": "SphericalDoGDetector.detect_batch(batch=2)",
            "convolution_backend": "native",
            "max_keypoints": 4096,
            "patch_workers": 4,
            "descriptor": "48x48 upright tangent RootSIFT, one scale",
            "explicit_validity_masks": True,
        },
        "comparisons": comparisons,
        "aggregate": {
            "pairs": len(comparisons),
            "median_detection_pair_seconds": float(
                np.median([row["detection_pair_seconds"] for row in comparisons])
            ),
            "median_pair_total_seconds": float(
                np.median([row["pair_total_seconds"] for row in comparisons])
            ),
            "maximum_peak_rss_mib": float(
                max(row["peak_rss_mib"] for row in comparisons)
            ),
            "catastrophic_accepted": sum(
                row["optimized_main_state"] == "catastrophic-accepted"
                for row in comparisons
            ),
            "precise_accepted": sum(
                row["optimized_main_state"] == "precise-accepted" for row in comparisons
            ),
        },
        "raw_result_sha256": raw_hashes,
        "source_taxonomy": {
            "path": str(args.taxonomy.resolve()),
            "sha256": _sha256(args.taxonomy),
        },
    }
    if not result["package"]["tree_identity_proves_same_source"]:
        raise RuntimeError("wheel source tree differs from frozen v3.5.0 source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--origin-main-commit", required=True)
    parser.add_argument("--origin-main-tree", required=True)
    parser.add_argument("--wheel-source-commit", required=True)
    parser.add_argument("--wheel-source-tree", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
