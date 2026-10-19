#!/usr/bin/env python3
"""Installed-wheel consumer for Stable spherical research evaluation."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.evaluation import (
        EvaluationProtocol,
        EvaluationRecord,
        evaluate_binary_regions,
        evaluate_features,
        evaluate_pose,
        evaluate_radial_range,
    )

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    mask = np.zeros((8, 16), dtype=bool)
    mask[2:6, :3] = True
    mask[2:6, -2:] = True
    regions = evaluate_binary_regions(mask, mask, boundary_tolerance_deg=1.0)
    bearings = np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    features = evaluate_features(bearings, bearings, threshold_deg=0.1)
    pose = evaluate_pose(
        np.eye(3),
        np.eye(3),
        predicted_translation=np.asarray((1.0, 0.0, 0.0)),
        reference_translation=np.asarray((1.0, 0.0, 0.0)),
    )
    depth = evaluate_radial_range(
        np.ones((8, 16), dtype=np.float64),
        np.ones((8, 16), dtype=np.float64),
        units="m",
    )
    assert regions.iou == 1.0
    assert features.repeatability == 1.0
    assert pose.rotation_error_deg == 0.0
    assert depth.mae == 0.0 and depth.coverage == 1.0

    protocol = EvaluationProtocol(
        protocol_id="installed-wheel-region-v1",
        version=1,
        evaluator="panorai.evaluation.evaluate_binary_regions",
        metric_names=("iou", "boundary_f1"),
        parameters={"boundary_tolerance_deg": 1.0},
    )
    record = EvaluationRecord(
        protocol=protocol,
        subject_id="analytic-seam-region",
        input_id="analytic:prediction",
        reference_id="analytic:reference",
        split="analytic",
        algorithm_id="identity/v1",
        algorithm_config={},
        package_version=version("panorai"),
        backend="numpy",
        metrics={"iou": regions.iou, "boundary_f1": regions.boundary_f1},
    )
    restored = EvaluationRecord.from_json(record.to_json())
    assert restored.protocol.fingerprint == protocol.fingerprint
    assert restored.metrics == record.metrics

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-evaluation",
                "version": version("panorai"),
                "origin": str(origin),
                "region_iou": regions.iou,
                "feature_repeatability": features.repeatability,
                "rotation_error_deg": pose.rotation_error_deg,
                "depth_mae": depth.mae,
                "protocol_fingerprint": protocol.fingerprint,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
