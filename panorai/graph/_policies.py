"""Versioned graph lifecycle policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    policy_id: str
    min_independent_views: int
    min_confidence: float
    dormant_below_confidence: float

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if self.min_independent_views < 2:
            raise ValueError("promotion requires at least two independent views")
        if not 0.0 <= self.dormant_below_confidence <= self.min_confidence <= 1.0:
            raise ValueError("promotion confidence thresholds are inconsistent")

    @classmethod
    def conservative_v1(cls) -> "PromotionPolicy":
        return cls(
            policy_id="panorai-graph-promotion/conservative-v1",
            min_independent_views=2,
            min_confidence=0.55,
            dormant_below_confidence=0.20,
        )


__all__ = ["PromotionPolicy"]
