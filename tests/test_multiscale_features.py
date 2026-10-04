from __future__ import annotations

import copy

import numpy as np
import pytest

from panorai.features import (
    MultiscaleEmbeddingConfig,
    MultiscaleFeatureSet,
    MultiscaleSphericalFeaturePipeline,
    OpenCVContextEmbedding,
)


def _textured_panorama(*, shift: int = 0) -> np.ndarray:
    height, width = 128, 256
    y, x = np.indices((height, width))
    checker = (((x // 8) + (y // 8)) % 2) * 127
    image = np.stack(
        (
            (x * 5 + checker + shift) % 256,
            (y * 7 + checker // 2 + shift) % 256,
            ((x + y) * 3 + checker + shift) % 256,
        ),
        axis=-1,
    )
    return image.astype(np.uint8)


def _pipeline(**config_changes) -> MultiscaleSphericalFeaturePipeline:
    config = MultiscaleEmbeddingConfig(
        local_fov_deg=(45.0, 45.0),
        local_grid_size=1,
        max_local_views_per_root=1,
        context_similarity_threshold=config_changes.pop(
            "context_similarity_threshold", 1.0
        ),
        scale_similarity_threshold=config_changes.pop(
            "scale_similarity_threshold", 1.0
        ),
        node_match_similarity_threshold=config_changes.pop(
            "node_match_similarity_threshold", 0.0
        ),
        node_match_top_k=1,
        cross_scale_nms_threshold_deg=0.2,
        embedding_shape_hw=(32, 32),
        **config_changes,
    )
    return MultiscaleSphericalFeaturePipeline.from_preset(
        "sift-bf",
        multiscale_config=config,
        face_sampler="cube",
        face_fov_deg=90.0,
        face_shape_hw=(96, 96),
        edge_margin_px=4,
        ratio_test=0.8,
        max_features=512,
    )


def test_multiscale_config_is_serializable_and_validated():
    config = MultiscaleEmbeddingConfig()
    assert config.to_dict()["interface"] == "panorai-multiscale-visual-features/v1"
    with pytest.raises(ValueError, match="local_grid_size"):
        MultiscaleEmbeddingConfig(local_grid_size=0)
    with pytest.raises(ValueError, match="context_similarity_threshold"):
        MultiscaleEmbeddingConfig(context_similarity_threshold=1.1)
    with pytest.raises(ValueError, match="local_fov_deg"):
        MultiscaleEmbeddingConfig(local_fov_deg=(0.0, 40.0))


def test_reference_embedding_is_deterministic_finite_and_mask_aware():
    image = _textured_panorama()[:64, :64]
    mask = np.ones((64, 64), dtype=bool)
    mask[:8] = False
    provider = OpenCVContextEmbedding(shape_hw=(32, 32))
    first = provider.embed([image, image], masks=[mask, mask])
    second = provider.embed([image], masks=[mask])
    assert first.shape == (2, 112)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first[0], second[0])
    assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-6)


def test_extract_is_deterministic_and_records_cross_scale_evidence():
    image = _textured_panorama()
    original = image.copy()
    pipeline = _pipeline()
    first = pipeline.extract(image, panorama_id="pano")
    second = pipeline.extract(image, panorama_id="pano")

    assert isinstance(first, MultiscaleFeatureSet)
    assert len(first.nodes) == 12  # six cube roots plus one local candidate each
    assert sum(node.selected for node in first.nodes) == 12
    assert all(node.solid_angle_sr > 0.0 for node in first.nodes)
    assert all(
        node.same_support_similarity is not None
        for node in first.nodes
        if node.level == 1
    )
    assert first.feature_guidance.shape == (len(first),)
    assert first.feature_scale_states.shape == (len(first),)
    assert set(first.feature_scale_states) <= {
        "persistent-multiscale",
        "local-split-support",
        "local-birth",
        "wide-only",
    }
    assert np.isfinite(first.feature_guidance).all()
    assert np.all(first.feature_guidance > 0.0)
    np.testing.assert_array_equal(first.base.descriptors, second.base.descriptors)
    np.testing.assert_allclose(first.base.bearings, second.base.bearings, atol=0.0)
    np.testing.assert_array_equal(image, original)
    description = first.describe()
    assert description["interface"] == "panorai-multiscale-visual-features/v1"
    assert description["embedding_provider"] == {
        "name": "opencv-context",
        "version": "1",
    }


def test_matching_routes_through_opencv_and_returns_proposal_only_weights():
    pipeline = _pipeline()
    first = pipeline.extract(_textured_panorama(), panorama_id="a")
    second = pipeline.extract(_textured_panorama(shift=3), panorama_id="b")
    matches = pipeline.match(first, second)

    assert len(matches) > 0
    assert matches.base.matcher_name.endswith("+embedding-routing")
    assert matches.sampling_weights.shape == (len(matches),)
    assert np.isfinite(matches.sampling_weights).all()
    assert np.all(matches.sampling_weights > 0.0)
    assert matches.diagnostics["global_fallback_retained"] is True
    assert matches.diagnostics["all_matches_scored_by_estimator"] is True
    correspondences = matches.to_bearing_correspondences()
    np.testing.assert_array_equal(correspondences.bearings_a, matches.bearings_a)
    np.testing.assert_array_equal(correspondences.valid, matches.valid)
    np.testing.assert_array_equal(
        correspondences.weights,
        matches.sampling_weights * matches.valid.astype(np.float32),
    )


def test_no_compatible_nodes_is_exact_global_fallback():
    pipeline = _pipeline(node_match_similarity_threshold=0.99)
    first = pipeline.extract(_textured_panorama(), panorama_id="a")
    second = pipeline.extract(_textured_panorama(shift=7), panorama_id="b")
    # Force a deterministic graph with no compatible node pair.  This alters
    # only test-owned result objects and leaves descriptor matching unchanged.
    second = copy.deepcopy(second)
    for node in second.nodes:
        node.embedding = -first.nodes[0].embedding.copy()

    expected = pipeline.base_pipeline.match(first.base, second.base)
    observed = pipeline.match(first, second)
    np.testing.assert_array_equal(
        observed.base.feature_indices_a, expected.feature_indices_a
    )
    np.testing.assert_array_equal(
        observed.base.feature_indices_b, expected.feature_indices_b
    )
    np.testing.assert_allclose(
        observed.base.descriptor_distances, expected.descriptor_distances
    )
    assert observed.diagnostics["candidate_node_pairs"] == 0
    assert set(observed.match_sources) <= {"global-fallback"}


class _BadEmbeddingProvider:
    name = "bad"
    version = "1"

    def embed(self, images, *, masks=None):
        return np.full((len(images), 3), np.nan, dtype=np.float32)


def test_embedding_contract_rejects_non_finite_provider_output():
    pipeline = MultiscaleSphericalFeaturePipeline(
        _pipeline().base_pipeline,
        MultiscaleEmbeddingConfig(
            local_grid_size=1,
            max_local_views_per_root=1,
            embedding_shape_hw=(16, 16),
        ),
        embedding_provider=_BadEmbeddingProvider(),
    )
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.extract(_textured_panorama())


class _RecordingEmbeddingProvider:
    name = "recording"
    version = "1"

    def __init__(self):
        self.valid_fractions = []

    def embed(self, images, *, masks=None):
        self.valid_fractions.extend(float(np.mean(mask)) for mask in masks)
        return np.asarray(
            [[float(np.mean(image)), float(np.std(image)) + 1.0] for image in images],
            dtype=np.float32,
        )


def test_erp_validity_is_projected_into_embedding_masks():
    provider = _RecordingEmbeddingProvider()
    config = MultiscaleEmbeddingConfig(
        local_grid_size=1,
        max_local_views_per_root=1,
        context_similarity_threshold=1.0,
        scale_similarity_threshold=1.0,
        embedding_shape_hw=(16, 16),
    )
    pipeline = MultiscaleSphericalFeaturePipeline(
        _pipeline().base_pipeline,
        config,
        embedding_provider=provider,
    )
    image = _textured_panorama()
    validity = np.ones(image.shape[:2], dtype=bool)
    validity[:, : image.shape[1] // 2] = False
    pipeline.extract(image, validity_mask=validity)
    assert provider.valid_fractions
    assert any(fraction < 0.75 for fraction in provider.valid_fractions)
    assert all(0.0 <= fraction <= 1.0 for fraction in provider.valid_fractions)


def test_pipeline_description_states_non_filtering_invariants():
    description = _pipeline().describe()
    assert description["status"] == "Experimental"
    assert description["invariants"] == {
        "opencv_owns_local_matching": True,
        "global_fallback_retained": True,
        "embedding_guidance_is_proposal_only": True,
        "semantic_labels_required": False,
    }
