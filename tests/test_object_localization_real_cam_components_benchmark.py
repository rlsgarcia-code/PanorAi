from __future__ import annotations

import numpy as np

from benchmarks.object_localization.run_real_cam_object_components import (
    _entity_name,
    _map_component_to_manual,
)


def test_component_mapping_exposes_merged_manual_instances() -> None:
    masks = {
        "desk-left-a": np.asarray((True, True, False, False, False, False)),
        "desk-right-a": np.asarray((False, False, True, True, False, False)),
        "bookcase-a": np.asarray((False, False, False, False, True, True)),
    }
    manual = [
        {"region_id": "desk-left-a", "class_id": 526},
        {"region_id": "desk-right-a", "class_id": 526},
        {"region_id": "bookcase-a", "class_id": 453},
    ]

    result = _map_component_to_manual(np.asarray((0, 1, 2)), 526, manual, masks)

    assert result["best_manual_region_id"] == "desk-left-a"
    assert result["best_component_purity"] == 2 / 3
    assert result["best_manual_coverage"] == 1.0
    assert result["merges_multiple_manual_instances"] is True
    assert [
        item["manual_region_id"] for item in result["overlapping_manual_regions"]
    ] == ["desk-left-a", "desk-right-a"]


def test_component_mapping_is_class_conditioned_and_can_be_unmapped() -> None:
    masks = {
        "desk-a": np.asarray((True, False, False)),
        "bookcase-a": np.asarray((False, True, False)),
    }
    manual = [
        {"region_id": "desk-a", "class_id": 526},
        {"region_id": "bookcase-a", "class_id": 453},
    ]

    mapped = _map_component_to_manual(np.asarray((1,)), 453, manual, masks)
    unmapped = _map_component_to_manual(np.asarray((2,)), 453, manual, masks)

    assert mapped["best_manual_region_id"] == "bookcase-a"
    assert unmapped["best_manual_region_id"] is None
    assert unmapped["overlapping_manual_regions"] == []


def test_entity_name_removes_only_view_suffix() -> None:
    assert _entity_name("conference-a") == "conference"
    assert _entity_name("workdesk-b") == "workdesk"
    assert _entity_name("object-12") == "object-12"
    assert _entity_name(None) is None
