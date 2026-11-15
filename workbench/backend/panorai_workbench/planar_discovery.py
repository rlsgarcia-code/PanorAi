"""JSON translation for the public spherical planar discovery application."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from panorai import EquirectangularImage
from panorai.reconstruction.planar import (
    PlanarDiscoveryRequest,
    PlanarRegion,
    SphericalLineSeed,
    SphericalPlanarDiscovery,
    SphericalPlanarDiscoveryOptions,
    decode_planar_mask,
)


def discover_planar_orientations(
    image: np.ndarray,
    document: dict[str, Any],
) -> dict[str, Any]:
    """Run public PanorAi discovery and return browser-safe reviewed evidence."""

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("image must be HWC RGB uint8")
    support = np.ones(source.shape[:2], dtype=bool)
    regions = tuple(
        PlanarRegion(
            plane_id=item["plane_id"],
            view_id="single-view",
            label=item.get("label", item["plane_id"]),
            state=item.get("state", "draft"),
            mask=decode_planar_mask(item["mask"], source.shape[:2]),
            interpretation=item.get("interpretation", "final"),
            source=item.get("source", "workbench-human"),
        )
        for item in document.get("regions", ())
    )
    request = PlanarDiscoveryRequest(
        EquirectangularImage(source, support_mask=support, valid=support),
        line_decisions=document.get("line_decisions", {}),
        bundle_memberships={
            str(key): tuple(value)
            for key, value in document.get("bundle_memberships", {}).items()
        },
        bundle_line_decisions={
            str(bundle_id): {
                str(line_id): str(decision)
                for line_id, decision in decisions.items()
            }
            for bundle_id, decisions in document.get("bundle_line_decisions", {}).items()
        },
        plane_bundle_groups={
            str(key): tuple(value)
            for key, value in document.get("plane_bundle_groups", {}).items()
        },
        manual_line_seeds=tuple(
            SphericalLineSeed(
                str(item["line_id"]),
                np.asarray(item["erp_endpoints_xy"], dtype=np.float64),
                str(item.get("state", "accepted")),
            )
            for item in document.get("manual_line_seeds", ())
        ),
        regions=regions,
        plane_line_groups={
            str(key): tuple(value)
            for key, value in document.get("plane_line_groups", {}).items()
        },
        revision=str(document.get("revision", "workbench-discovery-1")),
        provenance={
            "source": "panorai-workbench",
            "review": "automatic evidence is never accepted silently",
        },
    )
    option_values = document.get("options", {})
    if not isinstance(option_values, dict):
        raise TypeError("options must be an object")
    options = SphericalPlanarDiscoveryOptions(**option_values)
    run = SphericalPlanarDiscovery(options).run(request)
    result = run.output
    return {
        "schema": "panorai-workbench-planar-discovery/v1",
        "interface": result.interface,
        "state": result.state,
        "revision": request.revision,
        "image_shape_hw": list(source.shape[:2]),
        "effective_options": asdict(options),
        "edge_preview_erp_xy": result.edge_preview_erp_xy.tolist(),
        "lines": [
            {
                "line_id": line.line_id,
                "erp_endpoints_xy": line.erp_endpoints_xy.tolist(),
                "great_circle_normal_camera": line.great_circle_normal_camera.tolist(),
                "angular_length_deg": line.angular_length_deg,
                "support_pixels": line.support_pixels,
                "residual_px": line.residual_px,
                "edge_support": line.edge_support,
                "score": line.score,
                "chart_ids": list(line.chart_ids),
                "state": line.state,
                "source": line.source,
                "region_support": dict(line.region_support),
            }
            for line in result.lines
        ],
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "direction_camera": bundle.direction_camera.tolist(),
                "line_ids": list(bundle.line_ids),
                "residual_deg": bundle.residual_deg,
                "support_deg": bundle.support_deg,
                "confidence": bundle.confidence,
                "state": bundle.state,
                "source": bundle.source,
                "suggestions": [
                    {
                        "line_id": suggestion.line_id,
                        "residual_deg": suggestion.residual_deg,
                        "confidence": suggestion.confidence,
                        "state": suggestion.state,
                        "assigned_bundle_id": suggestion.assigned_bundle_id,
                    }
                    for suggestion in bundle.suggestions
                ],
            }
            for bundle in result.bundles
        ],
        "proposals": [
            {
                "proposal_id": proposal.proposal_id,
                "normal_camera": proposal.normal_camera.tolist(),
                "offset_m": proposal.estimate.offset_m,
                "bundle_ids": list(proposal.bundle_ids),
                "line_ids": list(proposal.line_ids),
                "roi_erp_xy": proposal.roi_erp_xy.tolist(),
                "confidence": proposal.confidence,
                "state": proposal.state,
                "reason": proposal.reason,
                "source": proposal.source,
                "plane_id": proposal.plane_id,
                "diagnostics": dict(proposal.estimate.diagnostics),
            }
            for proposal in result.proposals
        ],
        "diagnostics": dict(result.diagnostics),
        "pipeline": run.manifest(),
    }
