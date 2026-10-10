"""Simple composition API for association plus localization."""

from __future__ import annotations

from ._association import (
    assign_region_associations,
    object_hypothesis_id,
    propose_region_associations,
)
from ._localization import localize_region_association
from ._models import (
    ObjectHypothesis,
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
        proposals = propose_region_associations(evidence, self.config)
        associations = assign_region_associations(proposals, self.config)
        hypotheses: list[ObjectHypothesis] = []
        for association in associations:
            if association.state != "accepted":
                continue
            location = localize_region_association(
                association, evidence, config=self.config
            )
            hypotheses.append(
                ObjectHypothesis(
                    hypothesis_id=object_hypothesis_id(
                        evidence.view_id_a,
                        association.region_id_a,
                        evidence.view_id_b,
                        association.region_id_b,
                    ),
                    class_id=association.class_id,
                    class_name=association.class_name,
                    identity_state="confirmed",
                    view_ids=(evidence.view_id_a, evidence.view_id_b),
                    region_ids=(association.region_id_a, association.region_id_b),
                    association_id=association.association_id,
                    semantic_score=association.semantic_score,
                    identity_score=association.ranking_score,
                    location=location,
                )
            )
        failure_reasons = (
            ("relative-pose-not-accepted",)
            if self.config.require_accepted_pose
            and not evidence.pose.quality_report.accepted
            else ()
        )
        return ObjectLocalizationResult(
            query=evidence.query,
            hypotheses=tuple(hypotheses),
            associations=associations,
            failure_reasons=failure_reasons,
            diagnostics={
                "region_count_a": len(evidence.regions_a),
                "region_count_b": len(evidence.regions_b),
                "proposal_count": len(proposals),
                "accepted_association_count": sum(
                    item.state == "accepted" for item in associations
                ),
                "hypothesis_count": len(hypotheses),
                "query_vocabulary": evidence.query.vocabulary,
            },
        )

    def describe(self) -> dict[str, str]:
        return {
            "interface": "panorai-object-localization/v1",
            "stability": "experimental",
            "scope": "two-view-association-and-localization",
            "identity": "deterministic-two-view-region-pair",
            "graph": "not-built",
        }
