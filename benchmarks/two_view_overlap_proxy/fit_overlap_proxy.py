#!/usr/bin/env python3
"""Fit the RGB-match overlap proxy used by the Experimental two-view API.

The registered point-cloud overlap is a training label only.  Runtime features
come from the two-image frontend before relative-pose estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

SCHEMA = "panorai-rgb-overlap-proxy/v1"
BINS = np.asarray((0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0))
MIDPOINTS = np.asarray((0.05, 0.2, 0.4, 0.6, 0.8, 0.95))
FEATURE_NAMES = (
    "log1p_keypoint_count_min",
    "log1p_match_count",
    "logit_match_rate",
    "descriptor_distance_median",
    "descriptor_distance_p90",
    "ratio_score_median",
    "logit_valid_fraction_min",
    "descriptor_distance_median_missing",
    "descriptor_distance_p90_missing",
    "ratio_score_median_missing",
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def _raw_features(row: dict[str, Any]) -> list[float]:
    post = row["post"]
    keypoints = max(float(post["keypoint_count_min"]), 0.0)
    matches = max(float(post["match_count"]), 0.0)
    match_rate = matches / max(keypoints, 1.0)
    descriptors = (
        post.get("descriptor_distance_median"),
        post.get("descriptor_distance_p90"),
        post.get("ratio_score_median"),
    )
    values = [
        math.log1p(keypoints),
        math.log1p(matches),
        _logit(match_rate),
        *[float(value) if value is not None else math.nan for value in descriptors],
        _logit(float(post["valid_fraction_min"])),
        *[1.0 if value is None else 0.0 for value in descriptors],
    ]
    return values


def _thresholds(raw: np.ndarray) -> np.ndarray:
    first = raw[0]
    increments = np.logaddexp(0.0, raw[1:]) + 1e-4
    return np.concatenate(([first], first + np.cumsum(increments)))


def _probabilities(
    x: np.ndarray, beta: np.ndarray, cuts: np.ndarray, temperature: float
) -> np.ndarray:
    eta = x @ beta
    cumulative = _sigmoid((cuts[None, :] - eta[:, None]) / temperature)
    return np.concatenate(
        (cumulative[:, :1], np.diff(cumulative, axis=1), 1.0 - cumulative[:, -1:]),
        axis=1,
    )


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = len(BINS) - 1
    initial_cuts = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    raw = np.concatenate(
        (
            [initial_cuts[0]],
            np.log(np.expm1(np.diff(initial_cuts))),
            np.zeros(x.shape[1]),
        )
    )

    def objective(parameters: np.ndarray) -> float:
        cuts = _thresholds(parameters[: classes - 1])
        beta = parameters[classes - 1 :]
        probabilities = np.clip(_probabilities(x, beta, cuts, 1.0), 1e-12, 1.0)
        return float(
            -np.log(probabilities[np.arange(y.size), y]).sum()
            + 0.5 * np.sum(beta * beta)
        )

    fitted = minimize(objective, raw, method="L-BFGS-B", options={"maxiter": 2000})
    if not fitted.success:
        raise RuntimeError(f"ordinal fit failed: {fitted.message}")
    return fitted.x[classes - 1 :], _thresholds(fitted.x[: classes - 1])


def _metrics(probabilities: np.ndarray, overlap: np.ndarray) -> dict[str, float | int]:
    expected = probabilities @ MIDPOINTS
    target = overlap >= 0.5
    predicted = probabilities[:, 3:].sum(axis=1) >= 0.5
    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    fn = int(np.sum(~predicted & target))
    return {
        "count": int(overlap.size),
        "overlap_mae": float(np.mean(np.abs(expected - overlap))),
        "overlap_ge_0_5_brier": float(
            np.mean((probabilities[:, 3:].sum(axis=1) - target) ** 2)
        ),
        "overlap_ge_0_5_precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
        "overlap_ge_0_5_recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-bundle", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.features.read_text().splitlines()
        if line.strip()
    ]
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["independence_component_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise RuntimeError("an independence component crosses data splits")
    raw = np.asarray([_raw_features(row) for row in rows], dtype=np.float64)
    overlap = np.asarray(
        [row["capture"]["registered_cloud_overlap_min"] for row in rows]
    )
    split = np.asarray([row["split"] for row in rows])
    development = split == "development"
    calibration = split == "calibration"
    evaluation = split == "evaluation"

    medians = np.nanmedian(raw[development], axis=0)
    medians[7:] = 0.0
    filled = np.where(np.isnan(raw), medians, raw)
    mean = filled[development].mean(axis=0)
    scale = filled[development].std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = (filled - mean) / scale
    y = np.clip(np.digitize(overlap, BINS) - 1, 0, len(BINS) - 2)
    beta, cuts = _fit(x[development], y[development])

    def calibration_objective(log_temperature: float) -> float:
        p = np.clip(
            _probabilities(x[calibration], beta, cuts, math.exp(log_temperature)),
            1e-12,
            1.0,
        )
        return float(-np.log(p[np.arange(p.shape[0]), y[calibration]]).mean())

    calibrated = minimize_scalar(
        calibration_objective, bounds=(-3.0, 3.0), method="bounded"
    )
    temperature = float(math.exp(calibrated.x))
    output = {
        "schema": SCHEMA,
        "model_id": "rgb-match-ordinal-overlap-v1",
        "status": "retrospective experimental model; not release reliability",
        "training_target": "registered_cloud_overlap_min from aligned registered depth clouds",
        "runtime_inputs": "two EQR images, explicit validity masks, and optimized frontend matches; no depth/cloud input",
        "overlap_bin_edges": BINS.tolist(),
        "overlap_bin_representatives": MIDPOINTS.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "imputation_medians": medians.tolist(),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "coefficients": beta.tolist(),
        "cumulative_thresholds": cuts.tolist(),
        "calibration_temperature": temperature,
        "supported_runtime_ranges": {
            name: [
                float(np.min(filled[development, index])),
                float(np.max(filled[development, index])),
            ]
            for index, name in enumerate(FEATURE_NAMES[:7])
        },
        "data_provenance": {
            "features_sha256": _sha256(args.features),
            "pair_count": len(rows),
            "image_observation_count": 2 * len(rows),
            "unique_panorama_count": None,
            "unique_panorama_count_note": (
                "not recoverable from the aligned feature table because it stores "
                "pair IDs but not both source panorama IDs"
            ),
            "development_count": int(development.sum()),
            "calibration_count": int(calibration.sum()),
            "evaluation_count": int(evaluation.sum()),
            "dataset_count": len({row["dataset_id"] for row in rows}),
            "independence_component_count": len(
                {row["independence_component_id"] for row in rows}
            ),
            "group_disjoint_splits": True,
        },
        "metrics": {
            "development": _metrics(
                _probabilities(x[development], beta, cuts, temperature),
                overlap[development],
            ),
            "calibration": _metrics(
                _probabilities(x[calibration], beta, cuts, temperature),
                overlap[calibration],
            ),
            "heldout_evaluation": _metrics(
                _probabilities(x[evaluation], beta, cuts, temperature),
                overlap[evaluation],
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    if args.frozen_bundle or args.bundle_output:
        if not (args.frozen_bundle and args.bundle_output):
            parser.error(
                "--frozen-bundle and --bundle-output must be supplied together"
            )
        expected = "4c481d6b84e9507c8656b1b57fb24359b3db30ff639839df5c510abcae9a7f62"
        if _sha256(args.frozen_bundle) != expected:
            raise RuntimeError("frozen PanorAi 3.5.0 bundle checksum mismatch")
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_output.write_bytes(args.frozen_bundle.read_bytes())
    print(json.dumps(output["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
