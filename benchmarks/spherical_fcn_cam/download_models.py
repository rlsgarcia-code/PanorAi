#!/usr/bin/env python3
"""Prefetch and verify the three spherical FCN/CAM ImageNet checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorai.experimental.deep_learning import (  # noqa: E402
    SUPPORTED_IMAGENET_MODELS,
    prefetch_imagenet_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=SUPPORTED_IMAGENET_MODELS,
        help="checkpoint to fetch; repeat the option or omit it for all three",
    )
    parser.add_argument("--quiet", action="store_true", help="hide download progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = tuple(args.model) if args.model else SUPPORTED_IMAGENET_MODELS
    records = prefetch_imagenet_weights(names, progress=not args.quiet)
    print(
        json.dumps(
            {
                "schema": "panorai-pretrained-imagenet-cache/v1",
                "checkpoints": [record.to_dict() for record in records],
                "distribution_boundary": (
                    "Checkpoints remain in the user-controlled Torch cache and are "
                    "not part of PanorAi source, wheels, or sdists."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
