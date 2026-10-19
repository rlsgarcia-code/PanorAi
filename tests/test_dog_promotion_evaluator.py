from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.spherical_dog.run_reviewed_analytic import build_record
from benchmarks.spherical_dog.run_reviewed_holdout import (
    _expected_in_source,
    _load_range,
    _load_rgb,
)
from scripts.evaluate_spherical_application_promotion import (
    _verify_bound_resources,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "contracts" / "dog-reviewed-profile-promotion-v1.json"
PROFILE = "panorai-spherical-dog-features/reviewed-v1"


def _record(evaluation: str, metrics: dict, **extra) -> dict:
    return {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": evaluation,
        "profile_id": PROFILE,
        "source_commit": "a" * 40,
        "metrics": metrics,
        **extra,
    }


def _passing_evidence(protocol_sha: str) -> dict:
    analytic_metrics = {
        "isolated_blob_detection_rate": 1.0,
        "maximum_angular_error_deg": 0.1,
        "maximum_scale_level_error": 0.7,
        "seam_and_high_latitude_corpus": True,
        "reviewed_factory_matches_frozen_profile": True,
        "deterministic_replay": True,
    }
    external_metrics = {
        "pair_count": 6,
        "scene_count": 6,
        "repeatability": 0.60,
        "median_location_error_deg": 1.0,
        "verified_inlier_precision": 0.80,
        "scene_and_view_disjoint": True,
        "development_groups_excluded": True,
        "manifest_frozen_before_inference": True,
        "pose_range_occlusion_oracle": True,
    }
    performance_metrics = {
        "installed_wheel_isolated": True,
        "warmup_recorded": True,
        "median_recorded": True,
        "p95_recorded": True,
        "peak_rss_recorded": True,
        "raw_samples_recorded": True,
    }
    installed_metrics = {
        "exact_commit_and_wheel_sha_bound": True,
        "source_checkout_isolated": True,
        "profile_identity_roundtrip": True,
        "real_application_execution": True,
        "generic_constructor_remains_experimental": True,
    }
    records = [
        _record("analytic", analytic_metrics, backend=backend, evidence_sha256="e" * 64)
        for backend in ("native", "numpy")
    ]
    records.extend(
        _record(
            "external",
            external_metrics,
            split="post-protocol-heldout",
            corpus=corpus,
            cohort_manifest_sha256="c" * 64,
            evidence_sha256="e" * 64,
        )
        for corpus in ("Matterport3D", "Stanford2D3D")
    )
    records.extend(
        _record(
            "performance",
            performance_metrics,
            resolution=resolution,
            wheel_sha256="w" * 64,
            hardware_id="test-hardware",
            raw_samples_sha256="r" * 64,
        )
        for resolution in ("512x1024", "2048x4096", "native")
    )
    records.extend(
        _record(
            "installed-wheel",
            installed_metrics,
            wheel_sha256="w" * 64,
            artifact_version="3.8.1.dev0",
            platform=platform,
            python=python,
        )
        for platform, python in (
            ("macos-arm64", "3.11"),
            ("macos-arm64", "3.14"),
            ("linux-aarch64", "3.11"),
            ("linux-aarch64", "3.14"),
        )
    )
    records.append(
        _record(
            "independent-review",
            {
                "reviewer_is_not_implementer": True,
                "source_evidence_reproduced": True,
                "artifact_evidence_reproduced": True,
                "heldout_evidence_reproduced": True,
                "generic_and_coarse_routes_not_promoted": True,
            },
            reviewer_id="reviewer-1",
        )
    )
    return {
        "contract": "panorai-spherical-dog-reviewed-profile-promotion/v1",
        "protocol_sha256": protocol_sha,
        "records": records,
    }


def test_frozen_protocol_passes_only_with_every_gate_record() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    result = evaluate(protocol, _passing_evidence(digest), protocol_sha256=digest)

    assert result["passed"] is True
    assert all(item["passed"] for item in result["requirements"])


def test_external_threshold_failure_keeps_profile_unpromoted() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    evidence = _passing_evidence(digest)
    matterport = next(
        item for item in evidence["records"] if item.get("corpus") == "Matterport3D"
    )
    matterport["metrics"] = {**matterport["metrics"], "repeatability": 0.599999}

    result = evaluate(protocol, evidence, protocol_sha256=digest)

    assert result["passed"] is False
    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-reviewed-scene-disjoint-heldout"
    )
    assert heldout["passed"] is False
    assert any("repeatability" in failure for failure in heldout["failures"])


def test_protocol_bound_development_ledger_is_verified(tmp_path) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    _verify_bound_resources(PROTOCOL, protocol)

    copied_protocol = tmp_path / PROTOCOL.name
    copied_ledger = tmp_path / "dog-development-exclusions-v1.json"
    copied_protocol.write_bytes(PROTOCOL.read_bytes())
    source_ledger = ROOT / "contracts" / copied_ledger.name
    copied_ledger.write_bytes(source_ledger.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        _verify_bound_resources(copied_protocol, protocol)


def test_pose_range_occlusion_oracle_uses_metric_world_geometry() -> None:
    height, width = 8, 16
    source_xy = np.asarray(((width / 2.0 - 0.5, height / 2.0 - 0.5),))
    features = SimpleNamespace(
        source_erp_xy=source_xy,
        bearings=np.asarray(((0.0, 0.0, 1.0),)),
    )
    radial = np.full((height, width), 4.0, dtype=np.float64)
    pose = {
        "R_world_to_camera": np.eye(3).tolist(),
        "center_world": [0.0, 0.0, 0.0],
    }

    expected, visible = _expected_in_source(features, radial, radial, pose, pose)

    np.testing.assert_allclose(expected, ((0.0, 0.0, 1.0),), atol=1e-12)
    assert visible.tolist() == [True]

    occluded = radial.copy()
    occluded[:, :] = 2.0
    _, visible_occluded = _expected_in_source(features, radial, occluded, pose, pose)
    assert visible_occluded.tolist() == [False]


def test_holdout_loader_resamples_native_erps_with_frozen_policies(tmp_path) -> None:
    import cv2

    y, x = np.indices((1024, 2048), dtype=np.float32)
    rgb = np.stack((x % 255, y % 255, np.zeros_like(x)), axis=-1).astype(np.uint8)
    radial = (1.0 + x / 2048.0 + y / 1024.0).astype(np.float64)
    radial[100, 200] = np.nan
    rgb_path = tmp_path / "native.png"
    range_path = tmp_path / "native.npy"
    assert cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(range_path, radial, allow_pickle=False)

    loaded_rgb = _load_rgb(rgb_path)
    loaded_range = _load_range(range_path)

    assert loaded_rgb.shape == (512, 1024, 3)
    assert loaded_range.shape == (512, 1024)
    np.testing.assert_array_equal(
        loaded_range,
        cv2.resize(radial, (1024, 512), interpolation=cv2.INTER_NEAREST),
    )


def test_reviewed_analytic_adapter_requires_deterministic_exact_replay() -> None:
    calls = 0

    def evaluator(*, convolution_backend: str) -> dict:
        nonlocal calls
        calls += 1
        return {
            "metrics": {
                "isolated_blob_detection_rate": 1.0,
                "maximum_angular_error_deg": 0.1,
                "maximum_scale_level_error": 0.5,
                "seam_and_high_latitude_corpus": True,
                "reviewed_factory_matches_frozen_profile": True,
            },
            "details": {"backend": convolution_backend},
        }

    record = build_record(
        backend="numpy",
        source_commit="a" * 40,
        evaluator=evaluator,
    )

    assert record["metrics"]["deterministic_replay"] is True
    assert record["profile_id"] == PROFILE
    assert calls == 2
