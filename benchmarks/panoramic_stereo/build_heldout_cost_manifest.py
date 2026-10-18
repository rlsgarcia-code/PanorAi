#!/usr/bin/env python3
"""Build the outcome-blind VAL-117 matching-cost selection manifest."""

# ruff: noqa: E402 -- the source-checkout root precedes benchmark imports

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.build_heldout_confidence_manifest import (
    build_manifest as build_pair_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    args = parser.parse_args()
    payload = build_pair_manifest(
        overlap_path=args.overlap,
        analysis_path=args.analysis,
        exclude_manifests=tuple(args.exclude_manifest),
        roots={
            "stanford2d3d": args.stanford_root,
            "matterport360": args.matterport_root,
        },
    )
    payload["schema"] = "panorai-public-matching-cost-selection/v1"
    payload["protocol_id"] = "VAL-117-heldout-matching-cost-default"
    payload["selection_policy"]["comparison"] = (
        "intensity-gradient versus AD-Census with all downstream stages fixed"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
