#!/usr/bin/env python3
"""Evaluate frozen ImageNet/Stanford semantic pairs with native label maps."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCHEMA = "panorai-imagenet-stanford-semantic-evaluation/v1"
PREFLIGHT_SCHEMA = "panorai-imagenet-stanford-semantic-preflight/v1"
DEFAULT_MAPPING = Path(__file__).with_name("semantic_pairs.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(
    path: Path, *, include_related: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != "panorai-imagenet-stanford-semantic-pairs/v1":
        raise ValueError("semantic-pair mapping has an unexpected schema")
    allowed_tiers = {"primary", "related"} if include_related else {"primary"}
    pairs = [pair for pair in mapping["pairs"] if pair["tier"] in allowed_tiers]
    indices = [int(pair["imagenet_index"]) for pair in pairs]
    if len(indices) != len(set(indices)):
        raise ValueError("semantic-pair mapping contains duplicate ImageNet indices")
    return mapping, sorted(pairs, key=lambda pair: int(pair["imagenet_index"]))


def load_stanford_labels(path: Path, expected_sha256: str | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError(
            "semantic_labels.json SHA-256 does not match the frozen mapping"
        )
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(
        isinstance(item, str) for item in labels
    ):
        raise ValueError("semantic_labels.json must contain a JSON list of strings")
    return labels


def decode_stanford_semantic(path: Path) -> np.ndarray:
    """Decode the official RGB big-endian base-256 semantic index image."""

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def stanford_support(indices: np.ndarray, labels: list[str]) -> np.ndarray:
    return (indices > 0) & (indices < len(labels))


def stanford_class_mask(
    indices: np.ndarray, labels: list[str], class_name: str
) -> np.ndarray:
    selected = np.fromiter(
        (
            index
            for index, label in enumerate(labels)
            if label.split("_", 1)[0] == class_name
        ),
        dtype=np.uint32,
    )
    if selected.size == 0:
        raise ValueError(f"Stanford class {class_name!r} is absent from label metadata")
    valid = indices < len(labels)
    mask = np.zeros(indices.shape, dtype=bool)
    mask[valid] = np.isin(indices[valid], selected)
    return mask


def spherical_row_weights(height: int) -> np.ndarray:
    """Exact per-column solid angle of each ERP row, without the constant dlon."""

    edges = np.pi / 2.0 - np.arange(height + 1, dtype=np.float64) * np.pi / height
    return np.sin(edges[:-1]) - np.sin(edges[1:])


def _weighted_area(mask: np.ndarray, row_weights: np.ndarray) -> float:
    return float((mask * row_weights[:, None]).sum(dtype=np.float64))


def _top_area_mask(
    cam: np.ndarray,
    support: np.ndarray,
    target_area: float,
    row_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Threshold a quantized CAM to approximately match a target solid angle."""

    encoded = np.round(np.clip(cam, 0.0, 1.0) * 65535.0).astype(np.uint16)
    histogram = np.zeros(65536, dtype=np.float64)
    for row, weight in enumerate(row_weights):
        values = encoded[row, support[row]]
        if values.size:
            levels, counts = np.unique(values, return_counts=True)
            histogram[levels] += counts * weight
    descending_area = np.cumsum(histogram[::-1])
    offset = int(np.searchsorted(descending_area, target_area, side="left"))
    threshold = 65535 - min(offset, 65535)
    predicted = support & (encoded >= threshold)
    return predicted, float(threshold / 65535.0)


def _peak_to_mask_degrees(peak_row: int, peak_col: int, mask: np.ndarray) -> float:
    height, width = mask.shape
    peak_lat = np.pi / 2.0 - (peak_row + 0.5) * np.pi / height
    peak_lon = -np.pi + (peak_col + 0.5) * 2.0 * np.pi / width
    best_dot = -1.0
    for row in np.flatnonzero(mask.any(axis=1)):
        columns = np.flatnonzero(mask[row])
        latitude = np.pi / 2.0 - (row + 0.5) * np.pi / height
        longitudes = -np.pi + (columns + 0.5) * 2.0 * np.pi / width
        dots = math.sin(peak_lat) * math.sin(latitude) + math.cos(peak_lat) * math.cos(
            latitude
        ) * np.cos(longitudes - peak_lon)
        best_dot = max(best_dot, float(dots.max()))
    return float(np.degrees(np.arccos(np.clip(best_dot, -1.0, 1.0))))


def localization_metrics(
    cam: np.ndarray, target: np.ndarray, support: np.ndarray
) -> dict[str, Any]:
    if cam.ndim != 2 or cam.shape != target.shape or target.shape != support.shape:
        raise ValueError("cam, target, and support must be same-shaped HW arrays")
    if not np.isfinite(cam).all():
        raise ValueError("cam must be finite")
    if np.any(target & ~support):
        raise ValueError("target must be a subset of valid semantic support")
    row_weights = spherical_row_weights(cam.shape[0])
    full_area = float(cam.shape[1] * row_weights.sum())
    support_area = _weighted_area(support, row_weights)
    target_area = _weighted_area(target, row_weights)
    if support_area <= 0.0:
        raise ValueError("semantic image has no valid support")
    if target_area <= 0.0:
        raise ValueError("target class is absent")

    weighted_cam = cam.astype(np.float64) * row_weights[:, None]
    full_mass = float(weighted_cam.sum())
    support_mass = float(weighted_cam[support].sum())
    target_mass = float(weighted_cam[target].sum())

    target_fraction_support = target_area / support_area
    baseline = {
        "valid_support_spherical_fraction": support_area / full_area,
        "target_spherical_fraction_full": target_area / full_area,
        "target_spherical_fraction_within_support": target_fraction_support,
        "random_peak_hit_probability": target_fraction_support,
        "random_expected_mass_inside": target_fraction_support,
    }
    if full_mass <= 0.0:
        return {
            **baseline,
            "cam_has_positive_mass": False,
            "cam_mass_inside_target_full": None,
            "cam_mass_inside_target_within_support": None,
            "cam_mass_lift_over_random": None,
            "peak_defined": False,
            "peak_row": None,
            "peak_col": None,
            "peak_in_valid_support": None,
            "peak_hits_target": None,
            "peak_to_target_degrees": None,
            "top_area_threshold": None,
            "top_area_predicted_fraction_within_support": None,
            "top_area_iou": None,
            "random_expected_top_area_iou": None,
        }

    peak_flat = int(np.argmax(cam))
    peak_row, peak_col = np.unravel_index(peak_flat, cam.shape)
    peak_metrics = {
        "peak_defined": True,
        "peak_row": int(peak_row),
        "peak_col": int(peak_col),
        "peak_in_valid_support": bool(support[peak_row, peak_col]),
        "peak_hits_target": bool(target[peak_row, peak_col]),
        "peak_to_target_degrees": _peak_to_mask_degrees(
            int(peak_row), int(peak_col), target
        ),
    }
    if support_mass <= 0.0:
        return {
            **baseline,
            "cam_has_positive_mass": True,
            "cam_mass_inside_target_full": target_mass / full_mass,
            "cam_mass_inside_target_within_support": None,
            "cam_mass_lift_over_random": None,
            **peak_metrics,
            "top_area_threshold": None,
            "top_area_predicted_fraction_within_support": None,
            "top_area_iou": None,
            "random_expected_top_area_iou": None,
        }
    predicted, threshold = _top_area_mask(cam, support, target_area, row_weights)
    predicted_area = _weighted_area(predicted, row_weights)
    intersection = _weighted_area(predicted & target, row_weights)
    union = _weighted_area(predicted | target, row_weights)
    predicted_fraction_support = predicted_area / support_area
    random_intersection = target_fraction_support * predicted_fraction_support
    random_union = (
        target_fraction_support + predicted_fraction_support - random_intersection
    )

    return {
        **baseline,
        "cam_has_positive_mass": True,
        "cam_mass_inside_target_full": target_mass / full_mass,
        "cam_mass_inside_target_within_support": target_mass / support_mass,
        "cam_mass_lift_over_random": (target_mass / support_mass)
        / target_fraction_support,
        **peak_metrics,
        "top_area_threshold": threshold,
        "top_area_predicted_fraction_within_support": predicted_fraction_support,
        "top_area_iou": intersection / union,
        "random_expected_top_area_iou": random_intersection / random_union,
    }


def _load_result_paths(summary_path: Path) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = summary.get("results")
    if not isinstance(paths, list):
        raise ValueError("dataset summary must contain a results list")
    return [Path(path) for path in paths]


def _semantic_candidates(result: dict[str, Any], root: Path) -> list[Path]:
    sample = result["dataset_sample"]
    view_parts = str(sample["view_id"]).split("::")
    if len(view_parts) != 3:
        raise ValueError("Stanford view_id must contain dataset, area, and frame")
    area, frame = view_parts[1], view_parts[2]
    filename = frame if frame.endswith(".png") else f"{frame}.png"
    candidates = [root / area / "pano_semantic" / filename]
    if root.name != "raw":
        candidates.append(root / "raw" / area / "pano_semantic" / filename)
    return candidates


def _resolve_semantic(
    result: dict[str, Any], root: Path
) -> tuple[Path | None, list[Path]]:
    candidates = _semantic_candidates(result, root)
    return next((path for path in candidates if path.is_file()), None), candidates


def preflight(
    summary_path: Path,
    semantic_root: Path,
    labels_path: Path,
    mapping_path: Path = DEFAULT_MAPPING,
    *,
    include_related: bool = False,
) -> dict[str, Any]:
    mapping, pairs = load_pairs(mapping_path, include_related=include_related)
    required_indices = {int(pair["imagenet_index"]) for pair in pairs}
    expected_names = {
        int(pair["imagenet_index"]): pair["imagenet_name"] for pair in pairs
    }
    result_paths = _load_result_paths(summary_path)
    stanford_results: list[tuple[Path, dict[str, Any]]] = []
    missing_results: list[str] = []
    for result_path in result_paths:
        if not result_path.is_file():
            missing_results.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("dataset_sample", {}).get("dataset_id") == "stanford2d3d":
            stanford_results.append((result_path, result))

    missing_semantic: dict[str, list[str]] = {}
    found_semantic: dict[str, str] = {}
    missing_cams: list[dict[str, Any]] = []
    mismatched_class_names: list[dict[str, Any]] = []
    for result_path, result in stanford_results:
        view_id = result["dataset_sample"]["view_id"]
        semantic, candidates = _resolve_semantic(result, semantic_root)
        if semantic is None:
            missing_semantic.setdefault(view_id, [str(path) for path in candidates])
        else:
            found_semantic[view_id] = str(semantic)
        available: set[int] = set()
        for prediction in result.get("predictions", []):
            class_index = int(prediction["class_index"])
            if (result_path.parent / prediction.get("heatmap", "")).is_file():
                available.add(class_index)
            if (
                class_index in expected_names
                and prediction.get("class_name") != expected_names[class_index]
            ):
                mismatched_class_names.append(
                    {
                        "result": str(result_path),
                        "class_index": class_index,
                        "expected": expected_names[class_index],
                        "actual": prediction.get("class_name"),
                    }
                )
        missing = sorted(required_indices - available)
        if missing:
            missing_cams.append(
                {
                    "result": str(result_path),
                    "missing_class_indices": missing,
                }
            )

    labels_status: dict[str, Any] = {
        "path": str(labels_path),
        "exists": labels_path.is_file(),
    }
    if labels_path.is_file():
        actual_hash = _sha256(labels_path)
        expected_hash = mapping["stanford2d3d"]["semantic_labels_sha256"]
        labels_status.update(
            {
                "sha256": actual_hash,
                "expected_sha256": expected_hash,
                "hash_matches": actual_hash == expected_hash,
            }
        )
    ready = bool(stanford_results) and not (
        missing_results
        or missing_semantic
        or missing_cams
        or mismatched_class_names
        or not labels_status["exists"]
        or labels_status.get("hash_matches") is False
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "ready": ready,
        "summary": str(summary_path),
        "mapping": str(mapping_path),
        "mapping_sha256": _sha256(mapping_path),
        "include_related": include_related,
        "required_class_indices": sorted(required_indices),
        "result_count": len(result_paths),
        "stanford_result_count": len(stanford_results),
        "unique_stanford_views": len(
            {result["dataset_sample"]["view_id"] for _, result in stanford_results}
        ),
        "found_semantic": found_semantic,
        "missing_semantic": missing_semantic,
        "missing_result_files": missing_results,
        "results_missing_requested_cams": missing_cams,
        "mismatched_class_names": mismatched_class_names,
        "labels": labels_status,
    }


def _load_heatmap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        encoded = np.asarray(image, dtype=np.float32)
    return encoded / 65535.0


def _binary_ranking(scores: Iterable[tuple[float, bool]]) -> dict[str, float | None]:
    values = list(scores)
    positives = [score for score, present in values if present]
    negatives = [score for score, present in values if not present]
    auroc: float | None = None
    if positives and negatives:
        auroc = float(
            np.mean(
                [
                    1.0 if positive > negative else 0.5 if positive == negative else 0.0
                    for positive in positives
                    for negative in negatives
                ]
            )
        )
    average_precision: float | None = None
    if positives:
        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        seen_positive = 0
        precisions: list[float] = []
        for rank, (_, present) in enumerate(ordered, start=1):
            if present:
                seen_positive += 1
                precisions.append(seen_positive / rank)
        average_precision = float(np.mean(precisions))
    return {"auroc": auroc, "average_precision": average_precision}


def _mean(items: list[float]) -> float | None:
    return float(np.mean(items)) if items else None


def _defined_localization_values(
    records: list[dict[str, Any]], key: str
) -> list[float]:
    return [
        float(record["localization"][key])
        for record in records
        if record["localization"][key] is not None
    ]


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["imagenet_index"])].append(record)
    output: list[dict[str, Any]] = []
    for (model, index), values in sorted(grouped.items()):
        positives = [value for value in values if value["target_present"]]
        output.append(
            {
                "model": model,
                "imagenet_index": index,
                "imagenet_name": values[0]["imagenet_name"],
                "stanford_class": values[0]["stanford_class"],
                "tier": values[0]["tier"],
                "sample_count": len(values),
                "positive_count": len(positives),
                "negative_count": len(values) - len(positives),
                "positive_defined_cam_count": sum(
                    bool(value["localization"]["cam_has_positive_mass"])
                    for value in positives
                ),
                **_binary_ranking(
                    (float(value["probability"]), bool(value["target_present"]))
                    for value in values
                ),
                "positive_peak_hit_rate": _mean(
                    _defined_localization_values(positives, "peak_hits_target")
                ),
                "positive_mean_peak_to_target_degrees": _mean(
                    _defined_localization_values(positives, "peak_to_target_degrees")
                ),
                "positive_mean_cam_mass_inside_support": _mean(
                    _defined_localization_values(
                        positives, "cam_mass_inside_target_within_support"
                    )
                ),
                "positive_mean_mass_lift_over_random": _mean(
                    _defined_localization_values(positives, "cam_mass_lift_over_random")
                ),
                "positive_mean_top_area_iou": _mean(
                    _defined_localization_values(positives, "top_area_iou")
                ),
            }
        )
    return output


def evaluate(
    summary_path: Path,
    semantic_root: Path,
    labels_path: Path,
    mapping_path: Path = DEFAULT_MAPPING,
    *,
    include_related: bool = False,
) -> dict[str, Any]:
    readiness = preflight(
        summary_path,
        semantic_root,
        labels_path,
        mapping_path,
        include_related=include_related,
    )
    if not readiness["ready"]:
        raise RuntimeError("semantic-pair preflight is not ready")
    mapping, pairs = load_pairs(mapping_path, include_related=include_related)
    labels = load_stanford_labels(
        labels_path, mapping["stanford2d3d"]["semantic_labels_sha256"]
    )
    pair_by_index = {int(pair["imagenet_index"]): pair for pair in pairs}
    records: list[dict[str, Any]] = []
    semantic_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for result_path in _load_result_paths(summary_path):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("dataset_sample", {}).get("dataset_id") != "stanford2d3d":
            continue
        semantic_path, _ = _resolve_semantic(result, semantic_root)
        assert semantic_path is not None
        if semantic_path not in semantic_cache:
            indices = decode_stanford_semantic(semantic_path)
            semantic_cache[semantic_path] = (indices, stanford_support(indices, labels))
        indices, support = semantic_cache[semantic_path]
        masks: dict[str, np.ndarray] = {}
        for prediction in result["predictions"]:
            class_index = int(prediction["class_index"])
            pair = pair_by_index.get(class_index)
            if pair is None:
                continue
            stanford_class = pair["stanford_class"]
            if stanford_class not in masks:
                masks[stanford_class] = stanford_class_mask(
                    indices, labels, stanford_class
                )
            target = masks[stanford_class]
            heatmap_path = result_path.parent / prediction["heatmap"]
            cam = _load_heatmap(heatmap_path)
            if cam.shape != indices.shape:
                raise ValueError(
                    f"heatmap/semantic shape mismatch: {cam.shape} != {indices.shape}"
                )
            present = bool(target.any())
            records.append(
                {
                    "result": str(result_path),
                    "view_id": result["dataset_sample"]["view_id"],
                    "model": result["model"],
                    "imagenet_index": class_index,
                    "imagenet_name": pair["imagenet_name"],
                    "stanford_class": stanford_class,
                    "relation_kind": pair["relation_kind"],
                    "tier": pair["tier"],
                    "probability": float(prediction["probability"]),
                    "score": float(prediction["score"]),
                    "rank": int(prediction["rank"]),
                    "target_present": present,
                    "semantic_sha256": _sha256(semantic_path),
                    "heatmap_sha256": _sha256(heatmap_path),
                    "localization": (
                        localization_metrics(cam, target, support) if present else None
                    ),
                }
            )
    return {
        "schema": SCHEMA,
        "summary": str(summary_path),
        "mapping": str(mapping_path),
        "mapping_sha256": _sha256(mapping_path),
        "semantic_labels": str(labels_path),
        "semantic_labels_sha256": _sha256(labels_path),
        "include_related": include_related,
        "record_count": len(records),
        "records": records,
        "aggregates": _aggregate(records),
        "metric_contract": {
            "area": "exact ERP pixel solid angle; invalid Stanford labels excluded from support-conditioned metrics",
            "cam": "uint16 normalized weak-localization map; not a segmentation probability",
            "top_area_iou": "CAM threshold chosen so predicted valid-support area approximately equals target area",
            "random_baseline": "uniform random direction/mass over valid labeled solid angle",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-related", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    readiness = preflight(
        args.summary,
        args.semantic_root,
        args.semantic_labels,
        args.mapping,
        include_related=args.include_related,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only or not readiness["ready"]:
        args.output.write_text(
            json.dumps(readiness, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(readiness, indent=2, sort_keys=True))
        if not readiness["ready"] and not args.preflight_only:
            raise SystemExit(
                "semantic-pair inputs are incomplete; wrote a preflight report instead of metrics"
            )
        return
    output = evaluate(
        args.summary,
        args.semantic_root,
        args.semantic_labels,
        args.mapping,
        include_related=args.include_related,
    )
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["aggregates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
