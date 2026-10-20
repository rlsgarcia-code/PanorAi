from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from benchmarks.spherical_dog.run_installed_profile import (
    _validated_platform_id,
    _validate_artifact_version,
    _validate_direct_url,
    _validate_imported_distribution,
)
from benchmarks.spherical_dog.run_reviewed_analytic import (
    _validate_execution_binding,
    build_record,
)
from benchmarks.spherical_dog.run_reviewed_holdout import (
    REVIEWED_MATCHER_PARAMETERS,
    _bind_source_checkout_import,
    _expected_in_source,
    _load_range,
    _load_rgb,
    _reviewed_matcher_config,
    _validate_checkout,
)
from scripts.evaluate_spherical_application_promotion import (
    _verify_bound_resources,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "contracts" / "dog-reviewed-profile-promotion-v1.json"
PROFILE = "panorai-spherical-dog-features/reviewed-v1"
SOURCE_COMMIT = "a" * 40


def _record(evaluation: str, metrics: dict, **extra) -> dict:
    return {
        "application": "spherical-dog-reviewed-profile",
        "evaluation": evaluation,
        "profile_id": PROFILE,
        "source_commit": SOURCE_COMMIT,
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
    result = evaluate(
        protocol,
        _passing_evidence(digest),
        protocol_sha256=digest,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert result["passed"] is True
    assert all(item["passed"] for item in result["requirements"])


def test_frozen_protocol_requires_an_explicit_candidate_commit() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="requires expected_source_commit"):
        evaluate(protocol, _passing_evidence(digest), protocol_sha256=digest)


def test_external_threshold_failure_keeps_profile_unpromoted() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    evidence = _passing_evidence(digest)
    matterport = next(
        item for item in evidence["records"] if item.get("corpus") == "Matterport3D"
    )
    matterport["metrics"] = {**matterport["metrics"], "repeatability": 0.599999}

    result = evaluate(
        protocol,
        evidence,
        protocol_sha256=digest,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert result["passed"] is False
    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-reviewed-scene-disjoint-heldout"
    )
    assert heldout["passed"] is False
    assert any("repeatability" in failure for failure in heldout["failures"])


def test_evaluator_rejects_evidence_mixed_across_source_commits() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    evidence = _passing_evidence(digest)
    evidence["records"][0]["source_commit"] = "b" * 40

    result = evaluate(
        protocol,
        evidence,
        protocol_sha256=digest,
        expected_source_commit=SOURCE_COMMIT,
    )

    analytic = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-reviewed-analytic-oracle"
    )
    assert analytic["passed"] is False
    assert any("expected candidate commit" in item for item in analytic["failures"])


def test_evaluator_requires_one_joint_heldout_manifest() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    evidence = _passing_evidence(digest)
    stanford = next(
        item for item in evidence["records"] if item.get("corpus") == "Stanford2D3D"
    )
    stanford["cohort_manifest_sha256"] = "d" * 64

    result = evaluate(
        protocol,
        evidence,
        protocol_sha256=digest,
        expected_source_commit=SOURCE_COMMIT,
    )

    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-reviewed-scene-disjoint-heldout"
    )
    assert heldout["passed"] is False
    assert any(
        "share one cohort_manifest_sha256" in item for item in heldout["failures"]
    )


def test_evaluator_requires_both_python_cells_on_two_platforms() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    evidence = _passing_evidence(digest)
    macos_314 = next(
        item
        for item in evidence["records"]
        if item.get("platform") == "macos-arm64" and item.get("python") == "3.14"
    )
    macos_314["python"] = "3.11"

    result = evaluate(
        protocol,
        evidence,
        protocol_sha256=digest,
        expected_source_commit=SOURCE_COMMIT,
    )

    installed = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "dog-reviewed-installed-artifacts"
    )
    assert installed["passed"] is False
    assert any("matrix groups" in item for item in installed["failures"])


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


def test_holdout_matcher_freezes_every_configuration_field() -> None:
    assert asdict(_reviewed_matcher_config()) == dict(REVIEWED_MATCHER_PARAMETERS)


def test_installed_record_rejects_an_unrelated_wheel_hash(tmp_path) -> None:
    installed = tmp_path / "installed.whl"
    claimed = tmp_path / "claimed.whl"
    installed.write_bytes(b"installed")
    claimed.write_bytes(b"different")
    installed_sha = hashlib.sha256(installed.read_bytes()).hexdigest()
    payload = {
        "url": installed.as_uri(),
        "archive_info": {"hashes": {"sha256": installed_sha}},
    }

    with pytest.raises(AssertionError, match="different wheel"):
        _validate_direct_url(payload, claimed)


def test_installed_platform_identity_is_derived_from_the_host(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    assert _validated_platform_id(None) == "macos-arm64"
    assert _validated_platform_id("macos-arm64") == "macos-arm64"
    with pytest.raises(AssertionError, match="does not match"):
        _validated_platform_id("linux-aarch64")


def _fake_installed_wheel(tmp_path):
    site = tmp_path / "site"
    package = site / "panorai"
    package.mkdir(parents=True)
    members = {
        "panorai/__init__.py": b"VALUE = 1\n",
        "panorai/module.py": b"VALUE = 2\n",
    }
    for member, content in members.items():
        path = site / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    wheel = tmp_path / "panorai.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    installed_distribution = SimpleNamespace(
        locate_file=lambda member: site / str(member)
    )
    return package, wheel, installed_distribution


def test_installed_binding_rejects_a_shadow_package(tmp_path) -> None:
    package, wheel, installed_distribution = _fake_installed_wheel(tmp_path)
    shadow = tmp_path / "shadow" / "panorai" / "__init__.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("VALUE = 3\n")

    with pytest.raises(AssertionError, match="validated distribution"):
        _validate_imported_distribution(
            package=SimpleNamespace(__file__=str(shadow)),
            installed_distribution=installed_distribution,
            wheel=wheel,
        )

    assert package.is_dir()


def test_installed_binding_rejects_modified_or_extra_importable_files(tmp_path) -> None:
    package, wheel, installed_distribution = _fake_installed_wheel(tmp_path)
    imported = SimpleNamespace(__file__=str(package / "__init__.py"))
    (package / "module.py").write_text("VALUE = 99\n")
    with pytest.raises(AssertionError, match="differs from validated wheel"):
        _validate_imported_distribution(
            package=imported,
            installed_distribution=installed_distribution,
            wheel=wheel,
        )
    (package / "module.py").write_text("VALUE = 2\n")
    (package / "injected.py").write_text("VALUE = 3\n")
    with pytest.raises(AssertionError, match="absent from validated wheel"):
        _validate_imported_distribution(
            package=imported,
            installed_distribution=installed_distribution,
            wheel=wheel,
        )


@pytest.mark.parametrize("abbreviation", ("a" * 7, "a" * 9, "a" * 40))
def test_artifact_version_accepts_the_scm_commit_abbreviation(
    abbreviation: str,
) -> None:
    _validate_artifact_version(
        source_commit="a" * 40,
        artifact_version=f"3.8.1.dev1+g{abbreviation}",
    )


@pytest.mark.parametrize(
    "artifact_version",
    ("3.8.1.dev1+gaaaaaa", "3.8.1.dev1+gbbbbbbb", "3.8.1.dev1"),
)
def test_artifact_version_rejects_an_unbound_commit_marker(
    artifact_version: str,
) -> None:
    with pytest.raises(AssertionError, match="not bound"):
        _validate_artifact_version(
            source_commit="a" * 40,
            artifact_version=artifact_version,
        )


def test_installed_analytic_record_requires_an_exact_wheel() -> None:
    with pytest.raises(ValueError, match="requires --wheel"):
        _validate_execution_binding(
            source_root=None,
            wheel=None,
            source_commit="a" * 40,
        )


def test_source_analytic_record_rejects_a_wheel_at_the_same_time(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_execution_binding(
            source_root=tmp_path,
            wheel=tmp_path / "unrelated.whl",
            source_commit="a" * 40,
        )


def test_holdout_runner_rejects_a_dirty_tracked_checkout(monkeypatch, tmp_path) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        "benchmarks.spherical_dog.run_installed_profile._git_output",
        lambda root, *args: (
            commit if args[:2] == ("rev-parse", "HEAD") else " M dirty.py"
        ),
    )

    with pytest.raises(ValueError, match="tracked modifications"):
        _validate_checkout(tmp_path, commit)


def test_holdout_runner_rejects_panorai_from_outside_validated_checkout(
    monkeypatch, tmp_path
) -> None:
    fake_package = SimpleNamespace(
        __file__=str(tmp_path.parent / "installed/panorai/__init__.py")
    )
    monkeypatch.setitem(__import__("sys").modules, "panorai", fake_package)

    with pytest.raises(ValueError, match="outside the validated checkout"):
        _bind_source_checkout_import(tmp_path)


def test_source_evidence_rejects_untracked_native_extension(
    monkeypatch, tmp_path
) -> None:
    commit = "a" * 40
    package = tmp_path / "panorai" / "_native"
    package.mkdir(parents=True)
    (package / "_geometry.cpython-312-darwin.so").write_bytes(b"not-a-real-binary")

    def git_output(root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return commit
        return ""

    monkeypatch.setattr(
        "benchmarks.spherical_dog.run_installed_profile._git_output",
        git_output,
    )

    with pytest.raises(ValueError, match="untracked importable artifacts"):
        _validate_checkout(tmp_path, commit)
