from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_multiscale_ablation import (
    VARIANTS,
    _paired,
    proposal_variants,
)


class _Matches:
    def __init__(self) -> None:
        self.feature_indices_a = np.asarray([0, 1], dtype=np.int64)
        self.feature_indices_b = np.asarray([0, 1], dtype=np.int64)

    def __len__(self) -> int:
        return 2


class _Config:
    node_match_similarity_threshold = 0.55
    node_match_top_k = 1
    fallback_weight = 0.35
    regional_weight_gain = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "node_match_similarity_threshold": self.node_match_similarity_threshold,
            "node_match_top_k": self.node_match_top_k,
            "fallback_weight": self.fallback_weight,
            "regional_weight_gain": self.regional_weight_gain,
        }


def _feature_set(embedding: np.ndarray) -> SimpleNamespace:
    node = SimpleNamespace(
        selected=True,
        embedding=np.asarray(embedding, dtype=np.float32),
        feature_indices=np.asarray([0], dtype=np.int64),
    )
    return SimpleNamespace(
        nodes=(node,),
        config=_Config(),
        feature_scale_states=np.asarray(
            ["persistent-multiscale", "wide-only"], dtype=object
        ),
    )


def test_proposal_ablation_preserves_matches_and_isolates_interventions() -> None:
    first = _feature_set(np.asarray([1.0, 0.0]))
    second = _feature_set(np.asarray([1.0, 0.0]))

    weights, diagnostics = proposal_variants(first, second, _Matches())

    assert tuple(weights) == VARIANTS
    assert all(value.shape == (2,) for value in weights.values())
    assert all(np.mean(value) == pytest.approx(1.0) for value in weights.values())
    np.testing.assert_array_equal(weights["representation-uniform"], [1.0, 1.0])
    assert weights["persistence-proposal"][0] > weights["persistence-proposal"][1]
    assert weights["embedding-proposal"][0] > weights["embedding-proposal"][1]
    assert (
        weights["persistence-embedding-proposal"][0]
        > weights["embedding-proposal"][0]
    )
    assert diagnostics["candidate_node_pairs"] == 1
    assert diagnostics["matches_with_embedding_evidence"] == 1


def test_embedding_proposal_never_adds_or_filters_correspondences() -> None:
    first = _feature_set(np.asarray([1.0, 0.0]))
    second = _feature_set(np.asarray([-1.0, 0.0]))

    weights, diagnostics = proposal_variants(first, second, _Matches())

    assert diagnostics["candidate_node_pairs"] == 0
    assert diagnostics["matches_with_embedding_evidence"] == 0
    np.testing.assert_array_equal(weights["embedding-proposal"], [1.0, 1.0])
    assert all(len(value) == len(_Matches()) for value in weights.values())


def test_paired_comparison_counts_direction_and_rejects_id_mismatch() -> None:
    baseline = [
        {"pair_id": "a", "primary_success": False},
        {"pair_id": "b", "primary_success": True},
        {"pair_id": "c", "primary_success": True},
        {"pair_id": "d", "primary_success": False},
    ]
    candidate = [
        {"pair_id": "a", "primary_success": True},
        {"pair_id": "b", "primary_success": False},
        {"pair_id": "c", "primary_success": True},
        {"pair_id": "d", "primary_success": False},
    ]

    result = _paired(candidate, baseline)

    assert result == {
        "gains": 1,
        "losses": 1,
        "net_change": 0,
        "ties_success": 1,
        "ties_failure": 1,
        "mcnemar_exact_p": 1.0,
    }
    with pytest.raises(RuntimeError, match="IDs differ"):
        _paired(candidate[:-1], baseline)
