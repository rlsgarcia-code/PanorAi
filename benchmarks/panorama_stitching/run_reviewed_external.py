#!/usr/bin/env python3
"""Replay the frozen external panorama-stitching evidence manifest.

The source images remain outside the repository and distribution.  This runner
verifies every byte against the pre-inference manifest before decoding it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_case(
    root: Path, case: dict[str, Any], shape_hw: tuple[int, int]
) -> tuple[np.ndarray, ...]:
    height, width = shape_hw
    images = []
    for relative, expected in zip(case["paths"], case["sha256"], strict=True):
        path = root / relative
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"{case['id']}: SHA-256 mismatch for {relative}: {observed}"
            )
        with Image.open(path) as source:
            images.append(
                np.asarray(
                    source.convert("RGB").resize(
                        (width, height), resample=Image.Resampling.LANCZOS
                    )
                )
            )
    return tuple(images)


def _seam_energy(images: tuple[np.ndarray, ...], fov: float) -> float:
    from panorai.stitching import (
        CalibrateAndRegister,
        CompensateExposure,
        DetectStitchingFeatures,
        DiscoverOverlaps,
        StitchingInput,
        WarpRegisteredImages,
    )
    from panorai.applications import PanoramaStitcher

    method = PanoramaStitcher.reviewed_profile()
    value: Any = StitchingInput.from_arrays(images, horizontal_fov_deg=fov)
    for stage in (
        DetectStitchingFeatures(method.options),
        DiscoverOverlaps(method.options),
        CalibrateAndRegister(method.options),
        WarpRegisteredImages(method.options),
        CompensateExposure(method.options),
    ):
        value = stage.execute(value).value
    differences = []
    for left in range(len(value.values)):
        for right in range(left + 1, len(value.values)):
            overlap = value.validity[left] & value.validity[right]
            if np.any(overlap):
                delta = np.abs(
                    value.values[left][overlap] - value.values[right][overlap]
                )
                differences.append(np.asarray(delta, dtype=np.float64).reshape(-1))
    if not differences:
        return float("nan")
    return float(np.median(np.concatenate(differences)) / 255.0)


def _run_positive(
    images: tuple[np.ndarray, ...], *, fov: float, case_id: str
) -> dict[str, Any]:
    from panorai.applications import PanoramaStitcher

    method = PanoramaStitcher.reviewed_profile()
    identifiers = tuple(f"{case_id}-{index}" for index in range(len(images)))
    outcomes = []
    started = time.perf_counter()
    for _ in range(2):
        try:
            outcomes.append(
                method(
                    images,
                    image_ids=identifiers,
                    horizontal_fov_deg=fov,
                )
            )
        except Exception as exc:  # an explicit rejection is valid evidence
            outcomes.append((type(exc).__name__, str(exc)))
    elapsed = time.perf_counter() - started
    if not all(hasattr(item, "panorama") for item in outcomes):
        failures = [item for item in outcomes if isinstance(item, tuple)]
        return {
            "accepted": False,
            "failure_type": failures[0][0] if failures else "mixed-outcome",
            "failure_message": failures[0][1] if failures else "mixed outcome",
            "deterministic_replay": len(failures) == 2 and failures[0] == failures[1],
            "elapsed_seconds": elapsed,
        }
    first, second = outcomes
    deterministic = bool(
        np.array_equal(first.panorama.data, second.panorama.data)
        and np.array_equal(first.panorama.support_mask, second.panorama.support_mask)
        and first.diagnostics == second.diagnostics
    )
    residuals = tuple(
        edge.calibration.median_angular_error_deg for edge in first.registered.pairwise
    )
    return {
        "accepted": True,
        "accepted_pair_count": first.diagnostics.accepted_pair_count,
        "connected": first.diagnostics.accepted_pair_count >= len(images) - 1,
        "coverage": first.diagnostics.coverage,
        "overlap_fraction": first.diagnostics.overlap_fraction,
        "median_angular_residual_deg": (
            float(np.median(residuals)) if residuals else float("nan")
        ),
        "normalized_overlap_seam_error": _seam_energy(images, fov),
        "deterministic_replay": deterministic,
        "rejected_pairs": list(first.diagnostics.rejected_pairs),
        "elapsed_seconds": elapsed,
    }


def _run_negative(
    images: tuple[np.ndarray, ...], *, fov: float, case_id: str
) -> dict[str, Any]:
    from panorai.applications import PanoramaStitcher

    method = PanoramaStitcher.reviewed_profile()
    failures = []
    for _ in range(2):
        try:
            method(
                images,
                image_ids=tuple(f"{case_id}-{index}" for index in range(len(images))),
                horizontal_fov_deg=fov,
            )
        except Exception as exc:  # failure class is the evidence under test
            failures.append((type(exc).__name__, str(exc)))
        else:
            failures.append(("ACCEPTED", ""))
    return {
        "accepted": failures[0][0] == "ACCEPTED",
        "failure_type": failures[0][0],
        "failure_message": failures[0][1],
        "deterministic_failure": failures[0] == failures[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("contracts/panorama-stitching-huginsample-v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    preprocessing = manifest["preprocessing"]
    shape_hw = tuple(int(item) for item in preprocessing["shape_hw"])
    fov = float(preprocessing["horizontal_fov_deg"])
    records = []
    for case in manifest["cases"]:
        images = _load_case(args.dataset_root, case, shape_hw)
        if case["classification"].endswith("positive"):
            record = _run_positive(images, fov=fov, case_id=case["id"])
        else:
            record = _run_negative(images, fov=fov, case_id=case["id"])
        record.update(
            {
                "case_id": case["id"],
                "classification": case["classification"],
            }
        )
        records.append(record)

    positive = [item for item in records if item["classification"].endswith("positive")]
    negative = [
        item for item in records if item["classification"] == "prospective-negative"
    ]
    passed = bool(
        all(
            item["accepted"]
            and item["connected"]
            and item["coverage"] > 0.0
            and np.isfinite(item["median_angular_residual_deg"])
            and np.isfinite(item["normalized_overlap_seam_error"])
            and item["deterministic_replay"]
            for item in positive
        )
        and all(
            not item["accepted"] and item["deterministic_failure"] for item in negative
        )
    )
    report = {
        "contract": "panorai-panorama-stitching-external-evidence/v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_commit": manifest["source"]["commit"],
        "dataset_packaged": False,
        "profile_id": "panorai-panorama-stitching/rotation-only-reviewed-v1",
        "python": sys.version.split()[0],
        "passed": passed,
        "records": records,
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
