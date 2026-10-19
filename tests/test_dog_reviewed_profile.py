from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from panorai.applications import Application, SphericalDoGFeatures
from panorai.catalog import public_catalog
from panorai.features import SphericalFeatureSet


PROFILE_ID = "panorai-spherical-dog-features/reviewed-v1"


def _texture() -> np.ndarray:
    y, x = np.indices((96, 192), dtype=np.float64)
    value = (
        100.0
        + 60.0 * np.sin(x / 7.0)
        + 45.0 * np.cos(y / 5.0)
        + 35.0 * np.sin((x + 2.0 * y) / 11.0)
    )
    image = np.clip(value, 0.0, 255.0).astype(np.uint8)
    return np.stack((image, np.roll(image, 9, axis=1), image), axis=-1)


def test_reviewed_factory_freezes_complete_scientific_profile() -> None:
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")

    assert method.profile_id == PROFILE_ID
    assert method.interface == "panorai-spherical-dog-features/v1"
    assert method.stability == "experimental"
    assert method.config.octaves == 3
    assert method.config.levels_per_octave == 3
    assert method.config.base_sigma_px == 1.0
    assert method.config.contrast_threshold == 0.0005
    assert method.config.edge_threshold == 20.0
    assert method.config.max_features == 160
    assert method.config.patch_size == 32
    assert method.config.selection_policy == "response"
    assert method.config.descriptor_config.root_sift is True
    assert method.config.descriptor_config.orientation_policy == "fixed-zero"
    assert method.describe()["profile_id"] == PROFILE_ID


def test_reviewed_factory_has_a_separate_experimental_catalog_surface() -> None:
    surface = public_catalog().surface("spherical-applications-dog-reviewed-profile")

    assert surface.contract == "panorai-spherical-dog-features/v1"
    assert surface.tier == "experimental"
    assert surface.public_symbols == ("SphericalDoGFeatures.reviewed_profile",)


def test_scientific_change_drops_identity_but_backend_change_preserves_it() -> None:
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")

    changed = method.with_params(config__contrast_threshold=0.001)
    backend = method.with_params(config__convolution_backend="auto")

    assert type(changed) is SphericalDoGFeatures
    assert changed.profile_id is None
    assert changed.interface == "panorai-application/v1-experimental"
    assert changed.stability == "experimental"
    assert backend.profile_id == PROFILE_ID
    assert backend.config.convolution_backend == "auto"


def test_reviewed_profile_round_trips_and_rejects_identity_drift(tmp_path) -> None:
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    payload = method.to_config()
    restored = SphericalDoGFeatures.from_config(payload)
    path = method.save(tmp_path / "dog.json")
    loaded = Application.load(path)

    assert restored == method
    assert loaded == method
    assert payload["format"] == "panorai-spherical-dog-features-config/v1"
    assert payload["parameters"]["profile_id"] == PROFILE_ID

    drifted = copy.deepcopy(payload)
    drifted["parameters"]["config"]["max_features"] = 161
    with pytest.raises(ValueError, match="parameters have drifted"):
        SphericalDoGFeatures.from_config(drifted)

    unknown = copy.deepcopy(payload)
    unknown["parameters"]["profile_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown .* reviewed profile"):
        SphericalDoGFeatures.from_config(unknown)


def test_generic_configuration_cannot_claim_reviewed_identity() -> None:
    payload = SphericalDoGFeatures().to_config()
    payload["parameters"]["profile_id"] = PROFILE_ID

    with pytest.raises(ValueError, match="legacy .* cannot claim"):
        SphericalDoGFeatures.from_config(payload)


def test_reviewed_profile_executes_real_detector_patch_and_rootsift() -> None:
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    result = method(_texture(), panorama_id="reviewed-real-execution")

    assert isinstance(result, SphericalFeatureSet)
    assert result.panorama_id == "reviewed-real-execution"
    assert result.descriptor_type == "root-sift-float32"
    assert result.descriptors.shape == (len(result), 128)
    assert len(result) > 0
    np.testing.assert_allclose(np.linalg.norm(result.bearings, axis=1), 1.0, atol=1e-6)


def test_reviewed_config_is_canonical_json() -> None:
    method = SphericalDoGFeatures.reviewed_profile(convolution_backend="numpy")
    encoded = json.dumps(method.to_config(), sort_keys=True, allow_nan=False)

    assert PROFILE_ID in encoded
    assert "pickle" not in encoded
