from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from panorai.applications import FullSphereDenseStereo, PanoramicDenseStereo
from panorai.stereo.panoramic import (
    FullSphereMultiViewStereo,
    FullSphereStereoOptions,
)
from scripts.evaluate_spherical_application_promotion import evaluate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "contracts" / "registered-full-sphere-stereo-promotion-v1.json"
MANIFEST_PATH = ROOT / "contracts" / "public-api-surfaces-v2.json"
PROFILE_ID = "panorai-registered-full-sphere-stereo/reviewed-2026-10-17"


def _protocol() -> tuple[dict, str]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    return protocol, digest


def _metrics(**updates):
    value = {
        "abs_rel": 0.01,
        "fixed_delta_1_25": 0.99,
        "coverage": 0.99,
        "conditional_gross_gt_2x": 0.10,
        "pair_count": 6,
        "finite_exactly_equals_validity": True,
        "cost_independent_observability": True,
        "seam_and_poles_covered": True,
        "deterministic_replay": True,
        "prediction_independent_non_sky": True,
        "fixed_denominator": True,
        "provided_metric_pose": True,
        "invalid_values_are_nan": True,
        "exact_commit_and_wheel_sha_bound": True,
        "source_checkout_isolated": True,
        "native_extension_loaded": True,
        "analytic_oracle_passed": True,
        "streaming_memory_bound_verified": True,
        "public_facade_config_roundtrip": True,
        "reviewer_is_not_implementer": True,
        "source_evidence_reproduced": True,
        "artifact_evidence_reproduced": True,
        "heldout_evidence_reproduced": True,
        "automatic_pose_not_promoted": True,
        "confidence_not_claimed_as_probability": True,
    }
    value.update(updates)
    return value


def _passing_evidence(protocol: dict, digest: str) -> dict:
    common = {
        "application": "registered-full-sphere-stereo",
        "profile_id": PROFILE_ID,
        "source_commit": "a" * 40,
    }
    records = [
        {
            **common,
            "evaluation": "analytic",
            "corpus": "textured-sphere",
            "backend": backend,
            "metrics": _metrics(),
        }
        for backend in ("native", "numpy")
    ]
    records.extend(
        {
            **common,
            "evaluation": "external",
            "split": "post-protocol-heldout",
            "corpus": corpus,
            "cohort_manifest_sha256": f"new-{corpus}",
            "evidence_sha256": f"evidence-{corpus}",
            "metrics": _metrics(
                fixed_delta_1_25=0.30,
                coverage=0.50,
                conditional_gross_gt_2x=0.30,
            ),
        }
        for corpus in ("Matterport3D", "Stanford2D3D")
    )
    records.extend(
        {
            **common,
            "evaluation": "installed-wheel",
            "platform": platform,
            "python": python,
            "wheel_sha256": f"wheel-{platform}-{python}",
            "artifact_version": "3.7.1.dev0",
            "metrics": _metrics(),
        }
        for platform, python in (
            ("linux-x86_64", "3.11"),
            ("linux-x86_64", "3.14"),
            ("macos-arm64", "3.11"),
            ("macos-arm64", "3.14"),
        )
    )
    records.append(
        {
            **common,
            "evaluation": "independent-review",
            "reviewer_id": "independent-reviewer-1",
            "metrics": _metrics(),
        }
    )
    return {
        "contract": protocol["contract"],
        "protocol_sha256": digest,
        "records": records,
    }


def test_registered_pose_contract_is_granular_and_not_promoted() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in manifest["surfaces"]}

    core = by_id["registered-full-sphere-stereo-core"]
    facade = by_id["registered-full-sphere-stereo-application"]
    automatic = by_id["panoramic-dense-stereo-workflow"]

    assert core["tier"] == facade["tier"] == "experimental"
    assert core["contract"] == facade["contract"]
    assert "ResolveStereoPose" not in core["classified_exports"]
    assert "ResolveStereoPose" in automatic["classified_exports"]
    assert "FullSphereMultiViewStereo" not in automatic["classified_exports"]


def test_reviewed_profile_has_explicit_runtime_provenance() -> None:
    options = FullSphereStereoOptions.reviewed_profile(filter_backend="numpy")
    engine = FullSphereMultiViewStereo(options)
    application = FullSphereDenseStereo.reviewed_profile(filter_backend="numpy")

    assert options.profile_id == PROFILE_ID
    assert options.experimental_extensions == ()
    assert engine.describe() == {
        "interface": "panorai-registered-full-sphere-stereo/v1-experimental",
        "stability": "experimental",
        "input": "RegisteredPanoramaSet",
        "pose_policy": "provided-only",
        "quantity": "radial_range",
        "profile_id": PROFILE_ID,
        "experimental_extensions": [],
        "confidence_semantics": "within-result ranking score; not a calibrated probability",
    }
    description = application.describe()
    assert description["interface"] == engine.interface
    assert description["pose_policy"] == "provided-only"
    assert description["profile_id"] == PROFILE_ID
    assert description["experimental_extensions"] == []
    restored = FullSphereDenseStereo.from_config(application.to_config())
    assert restored == application
    assert restored.options.profile_id == PROFILE_ID
    assert PanoramicDenseStereo().interface != engine.interface


def test_unreviewed_options_do_not_inherit_candidate_profile() -> None:
    reviewed = FullSphereStereoOptions.reviewed_profile(filter_backend="numpy")

    multiscale = replace(reviewed, matching_cost="multiscale-ad-census")
    custom = replace(reviewed, min_parallax_deg=0.5)

    assert multiscale.profile_id is None
    assert multiscale.experimental_extensions == ("multiscale-ad-census",)
    assert custom.profile_id is None
    assert custom.experimental_extensions == ("custom-unreviewed-profile",)


def test_prospective_registered_stereo_protocol_can_pass_complete_evidence() -> None:
    protocol, digest = _protocol()

    result = evaluate(
        protocol,
        _passing_evidence(protocol, digest),
        protocol_sha256=digest,
    )

    assert result["passed"] is True


def test_prospective_protocol_rejects_a_previously_observed_cohort() -> None:
    protocol, digest = _protocol()
    evidence = _passing_evidence(protocol, digest)
    external = next(
        record for record in evidence["records"] if record["evaluation"] == "external"
    )
    external["cohort_manifest_sha256"] = protocol["policy"][
        "forbidden_cohort_manifest_sha256"
    ][0]

    result = evaluate(protocol, evidence, protocol_sha256=digest)

    assert result["passed"] is False
    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "registered-stereo-new-public-heldout"
    )
    assert any("previously observed evidence" in item for item in heldout["failures"])


def test_prospective_protocol_rejects_missing_provenance() -> None:
    protocol, digest = _protocol()
    evidence = _passing_evidence(protocol, digest)
    external = next(
        record for record in evidence["records"] if record["evaluation"] == "external"
    )
    external.pop("cohort_manifest_sha256")

    result = evaluate(protocol, evidence, protocol_sha256=digest)

    assert result["passed"] is False
    heldout = next(
        item
        for item in result["requirements"]
        if item["requirement"] == "registered-stereo-new-public-heldout"
    )
    assert any(
        "cohort_manifest_sha256: missing" in item for item in heldout["failures"]
    )
