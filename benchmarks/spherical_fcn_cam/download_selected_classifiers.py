#!/usr/bin/env python3
"""Acquire the selected Places365 and OpenCLIP checkpoints externally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.experimental.deep_learning import (  # noqa: E402
    acquire_openclip_rn50,
    acquire_places365_resnet18,
)


MODELS = ("places365-resnet18", "openclip-rn50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODELS,
        help="model to fetch; repeat or omit for both selected models",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--accept-upstream-terms",
        action="store_true",
        help="confirm review of each upstream model card, license and attribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_upstream_terms:
        raise ValueError("--accept-upstream-terms is required")
    names = tuple(args.model) if args.model else MODELS
    records = []
    for name in names:
        if name == "places365-resnet18":
            assets = acquire_places365_resnet18(
                accept_upstream_terms=True,
                cache_dir=args.cache_dir,
            )
        else:
            assets = acquire_openclip_rn50(
                accept_upstream_terms=True,
                cache_dir=args.cache_dir,
            )
        records.append(assets.to_dict())
    print(
        json.dumps(
            {
                "schema": "panorai-selected-classifier-cache/v1",
                "models": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
