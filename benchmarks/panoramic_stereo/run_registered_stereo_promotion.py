#!/usr/bin/env python3
"""Run the frozen registered-stereo analytic oracle against an installed wheel."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

import panorai
from panorai.applications import FullSphereDenseStereo
from panorai.geometry import erp_pixels_to_rays
from panorai.stereo.panoramic import (
    PanoramicStereoPose,
    RegisteredPanoramaSet,
    StereoSourceView,
)


PROFILE_ID = "panorai-registered-full-sphere-stereo/reviewed-2026-10-17"
SHAPE_HW = (512, 1024)
SPHERE_RADIUS_M = 4.0
SOURCE_CENTRES_XYZ_M = (
    (0.35, 0.0, 0.0),
    (0.0, 0.35, 0.0),
)
ROOT = Path(__file__).resolve().parents[2]


def _textured_sphere_view(
    shape_hw: tuple[int, int],
    camera_centre_xyz_m: tuple[float, float, float],
    sphere_radius_m: float,
) -> np.ndarray:
    yy, xx = np.indices(shape_hw, dtype=np.float64)
    rays = np.asarray(
        erp_pixels_to_rays(np.stack((xx, yy), axis=-1), shape_hw),
        dtype=np.float64,
    )
    centre = np.asarray(camera_centre_xyz_m, dtype=np.float64)
    projection = np.einsum("...i,i->...", rays, centre)
    distance = -projection + np.sqrt(
        projection * projection
        + sphere_radius_m * sphere_radius_m
        - float(centre @ centre)
    )
    points = centre + distance[..., None] * rays
    unit = points / np.linalg.norm(points, axis=-1, keepdims=True)
    longitude = np.arctan2(unit[..., 0], unit[..., 2])
    latitude = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
    return np.stack(
        (
            0.5 + 0.24 * np.sin(13.0 * longitude + 3.0 * latitude),
            0.5 + 0.24 * np.sin(9.0 * longitude - 11.0 * latitude),
            0.5 + 0.24 * np.cos(17.0 * longitude + 7.0 * latitude),
        ),
        axis=-1,
    ).astype(np.float32)


def _source(
    source_id: str,
    centre_xyz_m: tuple[float, float, float],
) -> StereoSourceView:
    centre = np.asarray(centre_xyz_m, dtype=np.float64)
    return StereoSourceView(
        source_id=source_id,
        panorama=_textured_sphere_view(
            SHAPE_HW,
            centre_xyz_m,
            SPHERE_RADIUS_M,
        ),
        pose=PanoramicStereoPose(
            rotation_source_from_reference=np.eye(3, dtype=np.float64),
            translation_source_from_reference=-centre,
            baseline=float(np.linalg.norm(centre)),
            units="m",
            scale_status="metric",
            provenance="frozen-installed-analytic-sphere",
        ),
    )


def _registered_fixture() -> RegisteredPanoramaSet:
    return RegisteredPanoramaSet(
        reference_id="analytic-reference",
        reference=_textured_sphere_view(
            SHAPE_HW,
            (0.0, 0.0, 0.0),
            SPHERE_RADIUS_M,
        ),
        sources=tuple(
            _source(f"analytic-source-{index}", centre)
            for index, centre in enumerate(SOURCE_CENTRES_XYZ_M)
        ),
    )


def _solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    latitude = (
        math.pi / 2.0 - math.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    )
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width)).copy()


def _result_hash(result) -> str:
    digest = hashlib.sha256()
    for name in (
        "radial_range",
        "validity",
        "confidence",
        "matching_confidence",
        "consistency_confidence",
        "disagreement",
        "sampling_support",
        "observable",
        "max_parallax_deg",
        "observable_source_count",
        "source_count",
        "source_consensus_count",
        "contribution_count",
    ):
        value = np.ascontiguousarray(getattr(result, name))
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _metrics(result, repeated, alternative) -> dict[str, float | bool]:
    validity = np.asarray(result.validity, dtype=bool)
    finite = np.isfinite(result.radial_range)
    weights = _solid_angle_weights(SHAPE_HW)
    population_weight = float(weights.sum())
    valid_weight = float(weights[validity].sum())
    relative = np.abs(result.radial_range[validity] - SPHERE_RADIUS_M) / SPHERE_RADIUS_M
    ratio = np.maximum(
        result.radial_range[validity] / SPHERE_RADIUS_M,
        SPHERE_RADIUS_M / result.radial_range[validity],
    )
    correct_weight = float(weights[validity][ratio < 1.25].sum())
    boundary = np.zeros(SHAPE_HW, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return {
        "abs_rel": float(np.average(relative, weights=weights[validity])),
        "fixed_delta_1_25": correct_weight / population_weight,
        "coverage": valid_weight / population_weight,
        "finite_exactly_equals_validity": bool(np.array_equal(finite, validity)),
        "cost_independent_observability": bool(
            np.array_equal(result.observable, alternative.observable)
            and np.array_equal(
                result.observable_source_count,
                alternative.observable_source_count,
            )
            and np.array_equal(
                result.max_parallax_deg,
                alternative.max_parallax_deg,
            )
        ),
        "seam_and_poles_covered": bool(
            np.all(result.sampling_support[boundary])
            and np.all(result.observable[boundary])
        ),
        "deterministic_replay": _result_hash(result) == _result_hash(repeated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "numpy"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        parser.error("--source-commit must be a lowercase 40-character Git SHA")
    origin = Path(panorai.__file__).resolve()
    if origin == ROOT or ROOT in origin.parents:
        raise RuntimeError("promotion oracle must import an installed wheel")

    registered = _registered_fixture()
    application = FullSphereDenseStereo.reviewed_profile(filter_backend=args.backend)
    if application.options.profile_id != PROFILE_ID:
        raise RuntimeError("runtime reviewed profile does not match the protocol")

    results = []
    elapsed = []
    for _ in range(2):
        started = time.perf_counter()
        results.append(application(registered))
        elapsed.append(time.perf_counter() - started)
    alternative = FullSphereDenseStereo(
        replace(application.options, matching_cost="multiscale-ad-census")
    )(registered)

    record = {
        "application": "registered-full-sphere-stereo",
        "evaluation": "analytic",
        "corpus": "textured-sphere",
        "profile_id": PROFILE_ID,
        "backend": args.backend,
        "source_commit": args.source_commit,
        "artifact_version": importlib.metadata.version("panorai"),
        "installed_origin": str(origin),
        "metrics": _metrics(results[0], results[1], alternative),
        "diagnostics": {
            "result_sha256": _result_hash(results[0]),
            "elapsed_seconds": elapsed,
            "median_elapsed_seconds": statistics.median(elapsed),
            "shape_hw": list(SHAPE_HW),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
