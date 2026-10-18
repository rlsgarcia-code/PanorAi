from __future__ import annotations

import numpy as np
import pytest

from benchmarks.panoramic_stereo.run_soft_source_consensus_validation import (
    soft_source_consensus_score,
)


def test_soft_source_consensus_score_has_fixed_bounded_factors() -> None:
    confidence = np.full((1, 3), 0.8, dtype=np.float32)
    source_count = np.full((1, 3), 2, dtype=np.uint16)
    consensus = np.array([[0, 1, 2]], dtype=np.uint16)
    validity = np.ones((1, 3), dtype=bool)

    score = soft_source_consensus_score(confidence, consensus, source_count, validity)

    assert np.allclose(score, [[0.4, 0.6, 0.8]])
    assert score.dtype == np.float32
    assert not score.flags.writeable


def test_soft_source_consensus_score_preserves_invalid_zero() -> None:
    score = soft_source_consensus_score(
        np.array([[0.9, 0.0]], dtype=np.float32),
        np.array([[2, 0]], dtype=np.uint16),
        np.array([[2, 0]], dtype=np.uint16),
        np.array([[True, False]]),
    )

    assert np.allclose(score, [[0.9, 0.0]])


@pytest.mark.parametrize(
    ("confidence", "consensus", "count", "validity", "message"),
    [
        (
            [[0.5]],
            [[2]],
            [[1]],
            [[True]],
            "bounded by source count",
        ),
        (
            [[0.5]],
            [[0]],
            [[0]],
            [[True]],
            "must have source evidence",
        ),
        (
            [[1.1]],
            [[1]],
            [[1]],
            [[True]],
            "finite and in",
        ),
    ],
)
def test_soft_source_consensus_score_rejects_invalid_contracts(
    confidence,
    consensus,
    count,
    validity,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        soft_source_consensus_score(
            np.asarray(confidence, dtype=np.float32),
            np.asarray(consensus, dtype=np.int16),
            np.asarray(count, dtype=np.int16),
            np.asarray(validity, dtype=bool),
        )
