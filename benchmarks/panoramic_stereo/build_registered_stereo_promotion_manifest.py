#!/usr/bin/env python3
"""Build the post-protocol registered-stereo promotion cohort.

This command performs selection only. It must run before any dense prediction
for the selected views is opened. Selection uses registered geometry, asset
presence and the frozen balance rules; RGB, range values and predictions are
never inspected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from benchmarks.panoramic_stereo.build_heldout_confidence_manifest import (
        build_manifest as build_pair_manifest,
    )
else:
    from build_heldout_confidence_manifest import build_manifest as build_pair_manifest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "contracts" / "registered-full-sphere-stereo-promotion-v1.json"
PRIOR_MANIFESTS = tuple(
    ROOT / "benchmarks" / "panoramic_stereo" / name
    for name in (
        "public_angular_neighborhood_manifest.json",
        "public_confidence_validity_manifest.json",
        "public_cost_default_manifest.json",
        "public_multipair_manifest.json",
        "public_multiscale_cost_manifest.json",
        "public_multiview_manifest.json",
        "public_three_source_manifest.json",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("frozen_at must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _load_protocol(protocol_path: Path) -> dict[str, Any]:
    payload = json.loads(protocol_path.read_text())
    if payload.get("contract") != (
        "panorai-registered-full-sphere-stereo-promotion/v1"
    ):
        raise ValueError("unexpected registered-stereo promotion protocol")
    return payload


def _verify_prior_manifests(
    protocol: dict[str, Any], prior_manifests: tuple[Path, ...]
) -> None:
    missing = [str(path) for path in prior_manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required prior manifests are missing: {missing}")
    expected = set(protocol["policy"]["forbidden_cohort_manifest_sha256"])
    observed = {_sha256(path) for path in prior_manifests}
    if observed != expected:
        raise ValueError(
            "prior manifest hashes differ from the frozen forbidden cohort set"
        )


def build_promotion_manifest(
    *,
    overlap_path: Path,
    analysis_path: Path,
    roots: dict[str, Path],
    frozen_at: str,
    protocol_path: Path = PROTOCOL,
    prior_manifests: tuple[Path, ...] = PRIOR_MANIFESTS,
) -> dict[str, Any]:
    """Select a new 6+6 cohort without opening matching or range outcomes."""

    protocol = _load_protocol(protocol_path)
    _verify_prior_manifests(protocol, prior_manifests)

    frozen_timestamp = _parse_timestamp(frozen_at)
    protocol_date = datetime.fromisoformat(str(protocol["frozen_at"])).replace(
        tzinfo=timezone.utc
    )
    if frozen_timestamp < protocol_date:
        raise ValueError("cohort must be frozen after the promotion protocol")

    payload = build_pair_manifest(
        overlap_path=overlap_path,
        analysis_path=analysis_path,
        exclude_manifests=prior_manifests,
        roots=roots,
    )
    payload.update(
        {
            "schema": "panorai-registered-stereo-promotion-selection/v1",
            "protocol_id": "STAB-136-registered-full-sphere-stereo-promotion",
            "promotion_protocol": {
                "path": protocol_path.name,
                "sha256": _sha256(protocol_path),
            },
            "frozen_at": frozen_timestamp.isoformat().replace("+00:00", "Z"),
            "frozen_before_dense_evaluation": True,
        }
    )
    payload["selection_policy"].update(
        {
            "post_protocol_holdout": True,
            "all_pre_protocol_public_stereo_manifests_excluded": True,
            "prediction_independent_non_sky_required_during_evaluation": True,
            "fixed_denominator_required_during_evaluation": True,
        }
    )

    manifest_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forbidden = set(protocol["policy"]["forbidden_cohort_manifest_sha256"])
    if manifest_sha in forbidden:
        raise ValueError("selected cohort matches a forbidden prior cohort")
    payload["canonical_payload_sha256"] = manifest_sha
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite an already frozen manifest: {args.output}"
        )
    payload = build_promotion_manifest(
        overlap_path=args.overlap,
        analysis_path=args.analysis,
        roots={
            "stanford2d3d": args.stanford_root,
            "matterport360": args.matterport_root,
        },
        frozen_at=args.frozen_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"frozen={args.output} sha256={_sha256(args.output)} pairs=12")


if __name__ == "__main__":
    main()
