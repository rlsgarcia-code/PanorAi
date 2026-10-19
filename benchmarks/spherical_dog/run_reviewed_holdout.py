#!/usr/bin/env python3
"""Evaluate the frozen DoG reviewed-v1 cohort with an independent oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.applications import SphericalDoGFeatures  # noqa: E402
from panorai.features import FeatureMatcher, FeatureMatcherConfig  # noqa: E402


PROTOCOL = ROOT / "contracts" / "dog-reviewed-profile-promotion-v1.json"
EXCLUSIONS = ROOT / "contracts" / "dog-development-exclusions-v1.json"
PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"
SHAPE_HW = (512, 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _git_tracked_status(root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _validate_checkout(root: Path, source_commit: str) -> None:
    if _git_head(root) != source_commit:
        raise ValueError("source_commit does not match the evaluator checkout HEAD")
    if _git_tracked_status(root):
        raise ValueError("evaluator checkout has tracked modifications")


def _safe_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"asset escapes its corpus root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_rgb(path: Path) -> NDArray[Any]:
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"could not decode RGB image {path}")
        image = (
            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB) if decoded.ndim == 3 else decoded
        )
    image = np.asarray(image)
    if (
        image.ndim not in {2, 3}
        or image.shape[0] <= 0
        or image.shape[1] != 2 * image.shape[0]
    ):
        raise ValueError(f"{path}: reviewed holdout requires a 2:1 HW/HWC ERP")
    if image.shape[:2] != SHAPE_HW:
        image = cv2.resize(image, SHAPE_HW[::-1], interpolation=cv2.INTER_AREA)
    return cast(NDArray[Any], image)


def _load_range(path: Path) -> NDArray[np.float64]:
    if path.suffix.lower() != ".npy":
        raise ValueError("reviewed holdout range assets must use NumPy .npy")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 2 * value.shape[0]:
        raise ValueError(f"{path}: reviewed holdout requires a 2:1 HW radial range")
    if value.shape != SHAPE_HW:
        value = cast(
            NDArray[np.float64],
            cv2.resize(value, SHAPE_HW[::-1], interpolation=cv2.INTER_NEAREST),
        )
    return value


def _nearest_range(
    depth: NDArray[np.float64], xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    x = np.floor(xy[:, 0] + 0.5).astype(np.int64) % depth.shape[1]
    y = np.floor(xy[:, 1] + 0.5).astype(np.int64)
    valid_y = (y >= 0) & (y < depth.shape[0])
    y = np.clip(y, 0, depth.shape[0] - 1)
    result = depth[y, x]
    return np.where(valid_y, result, np.nan)


def _bearings_to_erp(
    bearings: NDArray[np.float64], shape_hw: tuple[int, int]
) -> NDArray[np.float64]:
    """Independent pixel-centre projection under geometry-v1 conventions."""

    height, width = shape_hw
    normalized = bearings / np.linalg.norm(bearings, axis=1, keepdims=True)
    longitude = np.arctan2(normalized[:, 0], normalized[:, 2])
    latitude = np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0))
    x = (longitude + math.pi) * width / (2.0 * math.pi) - 0.5
    y = (math.pi / 2.0 - latitude) * height / math.pi - 0.5
    return np.stack((x, y), axis=1)


def _expected_in_source(
    features: Any,
    reference_range: NDArray[np.float64],
    source_range: NDArray[np.float64],
    reference: dict[str, Any],
    source: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    radial = _nearest_range(reference_range, features.source_erp_xy)
    r_ref = np.asarray(reference["R_world_to_camera"], dtype=np.float64)
    c_ref = np.asarray(reference["center_world"], dtype=np.float64)
    r_src = np.asarray(source["R_world_to_camera"], dtype=np.float64)
    c_src = np.asarray(source["center_world"], dtype=np.float64)
    camera_points = features.bearings * radial[:, None]
    world_points = c_ref + (r_ref.T @ camera_points.T).T
    source_points = (r_src @ (world_points - c_src).T).T
    projected_range = np.linalg.norm(source_points, axis=1)
    expected = np.divide(
        source_points,
        projected_range[:, None],
        out=np.zeros_like(source_points),
        where=projected_range[:, None] > 0,
    )
    source_xy = _bearings_to_erp(
        expected, (source_range.shape[0], source_range.shape[1])
    )
    observed = _nearest_range(source_range, source_xy)
    tolerance = np.maximum(0.05, 0.02 * projected_range)
    visible = (
        np.isfinite(radial)
        & (radial > 0)
        & np.isfinite(projected_range)
        & (projected_range > 0)
        & np.isfinite(observed)
        & (observed > 0)
        & (np.abs(observed - projected_range) <= tolerance)
    )
    return expected, visible


def _angular_errors(
    expected: NDArray[np.float64], observed: NDArray[np.float64]
) -> NDArray[np.float64]:
    if not len(expected):
        return np.empty(0, dtype=np.float64)
    if not len(observed):
        return np.full(len(expected), 180.0, dtype=np.float64)
    cosine = np.clip(expected @ observed.T, -1.0, 1.0)
    return np.asarray(np.rad2deg(np.arccos(np.max(cosine, axis=1))))


def _paired_angular_errors(
    expected: NDArray[np.float64], observed: NDArray[np.float64]
) -> NDArray[np.float64]:
    cosine = np.clip(np.sum(expected * observed, axis=1), -1.0, 1.0)
    return np.asarray(np.rad2deg(np.arccos(cosine)))


def _evaluate_pair(row: dict[str, Any], root: Path) -> dict[str, Any]:
    reference = row["reference"]
    source = row["source"]
    image_a = _load_rgb(_safe_asset(root, reference["rgb"]))
    image_b = _load_rgb(_safe_asset(root, source["rgb"]))
    range_a = _load_range(_safe_asset(root, reference["range"]))
    range_b = _load_range(_safe_asset(root, source["range"]))
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="auto")

    started = time.perf_counter()
    features_a = method(image_a, panorama_id=reference["view_id"])
    features_b = method(image_b, panorama_id=source["view_id"])
    elapsed = time.perf_counter() - started
    expected, visible = _expected_in_source(
        features_a, range_a, range_b, reference, source
    )
    location_errors = _angular_errors(expected[visible], features_b.bearings)
    repeatable = location_errors <= 1.0

    matcher = FeatureMatcher(FeatureMatcherConfig())
    matches = matcher.match(features_a, features_b)
    match_expected, match_visible = _expected_in_source(
        type(
            "MatchedReferenceFeatures",
            (),
            {
                "source_erp_xy": features_a.source_erp_xy[matches.feature_indices_a],
                "bearings": features_a.bearings[matches.feature_indices_a],
            },
        )(),
        range_a,
        range_b,
        reference,
        source,
    )
    match_errors = _paired_angular_errors(match_expected, matches.bearings_b)
    verified = match_visible & matches.valid & (match_errors <= 1.0)
    return {
        "pair_id": row["pair_id"],
        "scene_id": row["scene_id"],
        "reference_feature_count": len(features_a),
        "source_feature_count": len(features_b),
        "visible_feature_count": int(visible.sum()),
        "repeatable_feature_count": int(repeatable.sum()),
        "location_errors_deg": location_errors.tolist(),
        "match_count": len(matches),
        "verified_match_count": int(verified.sum()),
        "elapsed_seconds": elapsed,
    }


def _validate_manifest(
    manifest_path: Path, protocol_path: Path, exclusions_path: Path
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "panorai-spherical-dog-reviewed-holdout/v1":
        raise ValueError("unexpected DoG holdout manifest schema")
    if manifest.get("profile_id") != PROFILE_ID:
        raise ValueError("holdout manifest uses the wrong profile")
    if manifest["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("holdout manifest protocol SHA-256 has drifted")
    if manifest["development_exclusions"]["sha256"] != _sha256(exclusions_path):
        raise ValueError("holdout manifest exclusion ledger SHA-256 has drifted")
    canonical = dict(manifest)
    expected = canonical.pop("canonical_payload_sha256", None)
    if expected != _canonical_sha(canonical):
        raise ValueError("holdout manifest canonical payload SHA-256 is invalid")
    if manifest.get("frozen_before_inference") is not True:
        raise ValueError("manifest was not frozen before inference")
    protocol = _load_json(protocol_path)
    manifest_frozen_at = _parse_timestamp(
        manifest.get("frozen_at"), label="manifest frozen_at"
    )
    protocol_frozen_at = _parse_timestamp(
        protocol.get("frozen_at"), label="protocol frozen_at"
    )
    if manifest_frozen_at <= protocol_frozen_at:
        raise ValueError("holdout manifest must be frozen after the protocol")
    selection = protocol.get("policy", {}).get("cohort_selection")
    if not isinstance(selection, dict):
        raise ValueError("promotion protocol has no cohort selection policy")
    minimum_overlap = float(selection["minimum_directional_cloud_overlap"])
    minimum_baseline = float(selection["minimum_metric_baseline_m"])
    maximum_baseline = float(selection["maximum_metric_baseline_m"])
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise ValueError("reviewed holdout must contain exactly twelve pairs")
    scenes = [(item["corpus"], item["scene_id"]) for item in pairs]
    views = [
        (item["corpus"], item[key]["view_id"])
        for item in pairs
        for key in ("reference", "source")
    ]
    if len(scenes) != len(set(scenes)) or len(views) != len(set(views)):
        raise ValueError("reviewed holdout must be scene- and view-disjoint")
    exclusions = _load_json(exclusions_path)
    denied = exclusions.get("excluded_groups")
    if not isinstance(denied, dict):
        raise ValueError("development exclusion ledger is invalid")
    for item in pairs:
        exclusion_group_id = item.get("exclusion_group_id")
        if not isinstance(exclusion_group_id, str) or not exclusion_group_id:
            raise ValueError("reviewed holdout has no exclusion group identity")
        if exclusion_group_id in set(denied.get(item["corpus"], ())):
            raise ValueError("reviewed holdout reuses a development scene")
        overlap = item.get("directional_cloud_overlap")
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not math.isfinite(float(overlap))
            or float(overlap) < minimum_overlap
        ):
            raise ValueError("reviewed holdout contains an ineligible overlap")
        try:
            reference_center = [
                float(value) for value in item["reference"]["center_world"]
            ]
            source_center = [float(value) for value in item["source"]["center_world"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "reviewed holdout contains an invalid camera centre"
            ) from exc
        if (
            len(reference_center) != 3
            or len(source_center) != 3
            or not all(
                math.isfinite(value) for value in reference_center + source_center
            )
        ):
            raise ValueError("reviewed holdout contains an invalid camera centre")
        baseline = math.dist(reference_center, source_center)
        recorded_baseline = item.get("baseline_m")
        if (
            isinstance(recorded_baseline, bool)
            or not isinstance(recorded_baseline, (int, float))
            or not math.isclose(
                float(recorded_baseline), baseline, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("reviewed holdout baseline does not match camera centres")
        if not minimum_baseline <= baseline <= maximum_baseline:
            raise ValueError("reviewed holdout contains an ineligible baseline")
    return manifest


def run(
    *,
    manifest_path: Path,
    roots: dict[str, Path],
    source_commit: str,
    protocol_path: Path = PROTOCOL,
    exclusions_path: Path = EXCLUSIONS,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    _validate_checkout(ROOT, source_commit)
    manifest = _validate_manifest(manifest_path, protocol_path, exclusions_path)
    records: list[dict[str, Any]] = []
    for corpus in ("Matterport3D", "Stanford2D3D"):
        rows = [item for item in manifest["pairs"] if item["corpus"] == corpus]
        if len(rows) != 6:
            raise ValueError(f"{corpus} must contain exactly six pairs")
        details = [_evaluate_pair(item, roots[corpus]) for item in rows]
        visible = sum(item["visible_feature_count"] for item in details)
        repeatable = sum(item["repeatable_feature_count"] for item in details)
        matches = sum(item["match_count"] for item in details)
        verified = sum(item["verified_match_count"] for item in details)
        errors = [error for item in details for error in item["location_errors_deg"]]
        detail_sha = _canonical_sha(details)
        records.append(
            {
                "application": "spherical-dog-reviewed-profile",
                "evaluation": "external",
                "split": "post-protocol-heldout",
                "profile_id": PROFILE_ID,
                "corpus": corpus,
                "source_commit": source_commit,
                "cohort_manifest_sha256": _sha256(manifest_path),
                "evidence_sha256": detail_sha,
                "metrics": {
                    "pair_count": len(rows),
                    "scene_count": len({item["scene_id"] for item in rows}),
                    "repeatability": repeatable / visible if visible else 0.0,
                    "median_location_error_deg": (
                        float(np.median(errors)) if errors else 180.0
                    ),
                    "verified_inlier_precision": verified / matches if matches else 0.0,
                    "scene_and_view_disjoint": True,
                    "development_groups_excluded": True,
                    "manifest_frozen_before_inference": True,
                    "pose_range_occlusion_oracle": True,
                },
                "details": details,
            }
        )
    return {
        "contract": "panorai-spherical-dog-reviewed-profile-promotion/v1",
        "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": _sha256(manifest_path),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    evidence = run(
        manifest_path=args.manifest,
        roots={
            "Matterport3D": args.matterport_root,
            "Stanford2D3D": args.stanford_root,
        },
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"evidence={args.output} sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
