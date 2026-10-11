"""Simple composition API for association plus localization."""

from __future__ import annotations

from ._graph_compat import estimate_via_graph
from ._models import (
    ObjectLocalizationConfig,
    ObjectLocalizationResult,
    PairObjectLocalizationInput,
)


class ObjectLocalizationPipeline:
    """Associate and localize semantic regions in exactly two views."""

    def __init__(self, config: ObjectLocalizationConfig | None = None) -> None:
        self.config = config or ObjectLocalizationConfig()

    def estimate(
        self, evidence: PairObjectLocalizationInput
    ) -> ObjectLocalizationResult:
        return estimate_via_graph(evidence, self.config)

    def describe(self) -> dict[str, str]:
        return {
            "interface": "panorai-object-localization/v1",
            "stability": "experimental",
            "scope": "two-view-association-and-localization",
            "identity": "deterministic-two-view-region-pair",
            "graph": "panorai-spatial-semantic-graph/v1",
        }
