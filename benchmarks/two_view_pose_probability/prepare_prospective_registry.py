#!/usr/bin/env python3
"""Audit or freeze an outcome-blind E8 image and pair registry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

try:
    from run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        REGISTRY_SCHEMA,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_prospective_confirmation import (
        AUTHORIZATION,
        CANDIDATE_SCHEMA,
        REGISTRY_SCHEMA,
    )


IMAGE_SCHEMA = "panorai-two-view-prospective-image/v1"
PAIR_SCHEMA = "panorai-two-view-prospective-pair-candidate/v1"
AUDIT_SCHEMA = "panorai-two-view-prospective-registry-audit/v1"
ACQUISITION_SCHEMA = "panorai-two-view-prospective-acquisition-plan/v1"
AUTHORIZED_PLAN_STATUS = "AUTHORIZED_FOR_COLLECTION"
FORBIDDEN_FIELDS = {
    "accepted",
    "catastrophic_accepted",
    "ground_truth",
    "inlier_count",
    "matches",
    "outcomes",
    "p_accept_capture",
    "p_precise_capture_given_accept",
    "p_precise_post",
    "p_usable_capture",
    "pose",
    "post",
    "precise",
    "primary",
    "reference_rotation",
    "reference_translation",
    "returned",
    "rotation",
    "rotation_error_deg",
    "selected",
    "translation",
    "translation_direction_error_deg",
}
REQUIRED_CAPTURE_FIELDS = (
    "registered_cloud_overlap_min",
    "baseline_m",
    "baseline_depth_ratio",
    "representative_scene_distance_m",
    "predicted_parallax_deg",
    "rgb_similarity",
)
REQUIRED_IMAGE_FRACTIONS = (
    "valid_fraction",
    "clipped_fraction",
    "texture_occupancy",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_forbidden(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_FIELDS:
                return f"{path}.{key}"
            violation = _contains_forbidden(child, f"{path}.{key}")
            if violation is not None:
                return violation
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violation = _contains_forbidden(child, f"{path}[{index}]")
            if violation is not None:
                return violation
    return None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _finite(row: Mapping[str, Any], field: str) -> float:
    value = float(row.get(field, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_images(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if row.get("schema") != IMAGE_SCHEMA or not image_id:
            raise ValueError("invalid prospective image identity or schema")
        if image_id in indexed:
            raise ValueError(f"duplicate image_id: {image_id}")
        violation = _contains_forbidden(row)
        if violation is not None:
            raise ValueError(f"image record contains forbidden field: {violation}")
        for field in ("group_id", "domain_id", "sensor_id", "capture_timestamp"):
            if not str(row.get(field, "")):
                raise ValueError(f"image {image_id} has no {field}")
        for field in ("panorama_sha256", "validity_mask_sha256"):
            if not _is_sha256(row.get(field)):
                raise ValueError(f"image {image_id} has invalid {field}")
        if int(row.get("width", 0)) <= 0 or int(row.get("height", 0)) <= 0:
            raise ValueError(f"image {image_id} has invalid dimensions")
        validity_source = str(row.get("validity_source", ""))
        if not validity_source or "black" in validity_source.lower():
            raise ValueError(f"image {image_id} lacks an explicit validity source")
        for field in REQUIRED_IMAGE_FRACTIONS:
            value = _finite(row, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"image {image_id} has invalid {field}")
        if _finite(row, "blur_score") < 0.0:
            raise ValueError(f"image {image_id} has invalid blur_score")
        indexed[image_id] = row
    return indexed


def _validate_pairs(
    rows: list[dict[str, Any]],
    images: Mapping[str, dict[str, Any]],
    *,
    overlap_minimum: float,
    baseline_interval: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    indexed = {}
    unordered_pairs = set()
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if row.get("schema") != PAIR_SCHEMA or not pair_id:
            raise ValueError("invalid prospective pair identity or schema")
        if pair_id in indexed:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        violation = _contains_forbidden(row)
        if violation is not None:
            raise ValueError(f"pair record contains forbidden field: {violation}")
        image_ids = row.get("image_ids")
        if not isinstance(image_ids, list) or len(image_ids) != 2:
            raise ValueError(f"pair {pair_id} must contain exactly two image IDs")
        first, second = str(image_ids[0]), str(image_ids[1])
        if first == second or first not in images or second not in images:
            raise ValueError(f"pair {pair_id} contains invalid image IDs")
        unordered = tuple(sorted((first, second)))
        if unordered in unordered_pairs:
            raise ValueError(f"duplicate or reversed pair: {pair_id}")
        unordered_pairs.add(unordered)
        group = str(row.get("group_id", ""))
        domain = str(row.get("domain_id", ""))
        for image_id in unordered:
            image = images[image_id]
            if image["group_id"] != group or image["domain_id"] != domain:
                raise ValueError(f"pair {pair_id} crosses image group or domain")
        capture = row.get("capture")
        if not isinstance(capture, dict):
            raise ValueError(f"pair {pair_id} has no capture record")
        for field in REQUIRED_CAPTURE_FIELDS:
            if field not in capture:
                raise ValueError(f"pair {pair_id} has no capture.{field}")
        overlap = _finite(capture, "registered_cloud_overlap_min")
        baseline = _finite(capture, "baseline_m")
        if not overlap_minimum <= overlap <= 1.0:
            raise ValueError(f"pair {pair_id} is outside the overlap envelope")
        if not baseline_interval[0] <= baseline <= baseline_interval[1]:
            raise ValueError(f"pair {pair_id} is outside the baseline envelope")
        if _finite(capture, "baseline_depth_ratio") < 0.0:
            raise ValueError(f"pair {pair_id} has invalid baseline_depth_ratio")
        if _finite(capture, "representative_scene_distance_m") <= 0.0:
            raise ValueError(f"pair {pair_id} has invalid scene distance")
        parallax = _finite(capture, "predicted_parallax_deg")
        similarity = _finite(capture, "rgb_similarity")
        if not 0.0 <= parallax <= 180.0 or not 0.0 <= similarity <= 1.0:
            raise ValueError(f"pair {pair_id} has invalid visible capture fields")
        indexed[pair_id] = row
    return indexed


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("prospective registry output directory must not exist")
    candidate = _read_json(args.candidate)
    plan = _read_json(args.acquisition_plan)
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate schema is invalid")
    if plan.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError("acquisition plan schema is invalid")
    candidate_hash = _sha256(args.candidate)
    if plan.get("candidate_readiness", {}).get("candidate_sha256") != candidate_hash:
        raise ValueError("candidate does not match the acquisition plan")

    recommendation = plan["recommended_plan"]
    overlap_minimum = float(args.overlap_minimum)
    baseline_interval = tuple(float(value) for value in args.baseline_interval)
    if len(baseline_interval) != 2 or baseline_interval[0] > baseline_interval[1]:
        raise ValueError("baseline interval is invalid")
    images = _validate_images(_read_jsonl(args.images))
    pairs = _validate_pairs(
        _read_jsonl(args.pairs),
        images,
        overlap_minimum=overlap_minimum,
        baseline_interval=(baseline_interval[0], baseline_interval[1]),
    )

    images_by_group = Counter(str(row["group_id"]) for row in images.values())
    pairs_by_group = Counter(str(row["group_id"]) for row in pairs.values())
    group_domains = {}
    for row in images.values():
        group = str(row["group_id"])
        domain = str(row["domain_id"])
        previous = group_domains.setdefault(group, domain)
        if previous != domain:
            raise ValueError(f"group {group} maps to multiple domains")
    if any(
        group_domains[str(row["group_id"])] != str(row["domain_id"])
        for row in pairs.values()
    ):
        raise ValueError("a group maps to multiple domains")
    groups_by_domain = Counter(group_domains.values())
    expected_domains = set(plan["balanced_domain_design"]["labels"])
    unexpected_domains = set(groups_by_domain) - expected_domains
    if unexpected_domains:
        raise ValueError(f"unexpected prospective domains: {sorted(unexpected_domains)}")

    planned_groups = int(recommendation["groups_per_domain"])
    planned_images = int(recommendation["proposed_unique_images_per_group"])
    planned_pairs = int(recommendation["candidates_per_group"])
    deficits = {
        "groups_by_domain": {
            domain: max(0, planned_groups - groups_by_domain[domain])
            for domain in sorted(expected_domains)
        },
        "images_by_group": {
            group: max(0, planned_images - images_by_group[group])
            for group in sorted(group_domains)
        },
        "pairs_by_group": {
            group: max(0, planned_pairs - pairs_by_group[group])
            for group in sorted(group_domains)
        },
    }
    complete = not any(deficits["groups_by_domain"].values()) and not any(
        deficits[section][group]
        for section in ("images_by_group", "pairs_by_group")
        for group in deficits[section]
    )
    authorized = (
        candidate.get("authorization") == AUTHORIZATION
        and plan.get("status") == AUTHORIZED_PLAN_STATUS
    )
    report = {
        "schema": AUDIT_SCHEMA,
        "mode": args.mode,
        "status": (
            "READY_TO_FREEZE"
            if complete and authorized
            else "COMPLETE_BUT_NOT_AUTHORIZED"
            if complete
            else "INCOMPLETE"
        ),
        "candidate": {
            "sha256": candidate_hash,
            "authorization": candidate.get("authorization"),
        },
        "acquisition_plan": {
            "sha256": _sha256(args.acquisition_plan),
            "status": plan.get("status"),
        },
        "inputs": {
            "images_sha256": _sha256(args.images),
            "pairs_sha256": _sha256(args.pairs),
        },
        "counts": {
            "unique_images": len(images),
            "candidate_pairs": len(pairs),
            "groups": len(group_domains),
            "groups_by_domain": dict(sorted(groups_by_domain.items())),
            "images_by_group": dict(sorted(images_by_group.items())),
            "pairs_by_group": dict(sorted(pairs_by_group.items())),
        },
        "capture_envelope": {
            "registered_cloud_overlap_min": overlap_minimum,
            "baseline_m_closed_interval": list(baseline_interval),
        },
        "deficits": deficits,
        "complete": complete,
        "authorized": authorized,
        "reference_fields_opened": False,
    }
    if args.mode == "freeze":
        if not complete:
            raise ValueError("prospective registry does not meet the frozen plan")
        if not authorized:
            raise PermissionError("candidate and acquisition plan are not authorized")
    args.output_dir.mkdir(parents=True)
    if args.mode == "freeze":
        registry_rows = []
        for pair_id, pair in sorted(pairs.items()):
            registry_rows.append(
                {
                    "schema": REGISTRY_SCHEMA,
                    "pair_id": pair_id,
                    "group_id": pair["group_id"],
                    "domain_id": pair["domain_id"],
                    "image_ids": pair["image_ids"],
                    "image_evidence": [
                        {
                            "image_id": image_id,
                            "panorama_sha256": images[image_id]["panorama_sha256"],
                            "validity_mask_sha256": images[image_id][
                                "validity_mask_sha256"
                            ],
                        }
                        for image_id in pair["image_ids"]
                    ],
                    "capture": pair["capture"],
                    "candidate_sha256": candidate_hash,
                    "acquisition_plan_sha256": _sha256(args.acquisition_plan),
                }
            )
        registry_path = args.output_dir / "prospective-registry.jsonl"
        _atomic_jsonl(registry_path, registry_rows)
        report["registry_sha256"] = _sha256(registry_path)
        report["status"] = "FROZEN_OUTCOME_BLIND_REGISTRY"
    _atomic_json(args.output_dir / "registry-audit.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "freeze"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--overlap-minimum", type=float, default=0.50)
    parser.add_argument(
        "--baseline-interval",
        type=float,
        nargs=2,
        default=(0.16196559975867944, 2.1546692039871624),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = prepare(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
