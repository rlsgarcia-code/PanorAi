from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from panorai.geometry import erp_pixels_to_rays
from panorai.object_localization import (
    SEMANTIC_REGION_PROPOSALS_INTERFACE,
    SemanticRegionProposalConfig,
    semantic_regions_from_map,
)


ERP_SHAPE = (6, 12)
MAP_SHAPE = (3, 6)


def _pixel_for_cell(y: int, x: int) -> np.ndarray:
    return np.asarray(
        (
            (x + 0.5) * ERP_SHAPE[1] / MAP_SHAPE[1] - 0.5,
            (y + 0.5) * ERP_SHAPE[0] / MAP_SHAPE[0] - 0.5,
        )
    )


def _features(cells: tuple[tuple[int, int], ...]):
    pixels = np.stack([_pixel_for_cell(y, x) for y, x in cells])
    return SimpleNamespace(
        panorama_id="view-a",
        source_erp_xy=pixels,
        bearings=np.asarray(erp_pixels_to_rays(pixels, ERP_SHAPE)),
    )


def test_components_merge_across_horizontal_erp_seam() -> None:
    score_map = np.zeros(MAP_SHAPE, dtype=np.float32)
    score_map[1, 0] = 0.9
    score_map[1, 5] = 0.8
    score_map[1, 2] = 0.7
    features = _features(((1, 0), (1, 5), (1, 2)))

    result = semantic_regions_from_map(
        score_map,
        erp_shape_hw=ERP_SHAPE,
        features=features,
        region_id_prefix="desk",
        class_id=526,
        class_name="desk",
        config=SemanticRegionProposalConfig(threshold=0.5, connectivity=4),
        source_id="cam:view-a:desk",
    )

    assert result.interface == SEMANTIC_REGION_PROPOSALS_INTERFACE
    assert result.raw_component_count == 2
    assert result.retained_component_count == 2
    assert result.seam_crossing_component_count == 1
    assert [region.region_id for region in result.regions] == [
        "desk:526:000",
        "desk:526:001",
    ]
    np.testing.assert_array_equal(result.regions[0].feature_indices, (0, 1))
    np.testing.assert_allclose(result.regions[0].membership_weights, (0.9, 0.8))
    np.testing.assert_array_equal(result.regions[1].feature_indices, (2,))
    assert result.regions[0].source_id == "cam:view-a:desk#component-000"


def test_connectivity_four_separates_diagonals_and_eight_merges_them() -> None:
    score_map = np.zeros(MAP_SHAPE, dtype=np.float32)
    score_map[0, 1] = 0.9
    score_map[1, 2] = 0.8
    features = _features(((0, 1), (1, 2)))

    four = semantic_regions_from_map(
        score_map,
        erp_shape_hw=ERP_SHAPE,
        features=features,
        region_id_prefix="object",
        class_id=1,
        class_name="object",
        config=SemanticRegionProposalConfig(threshold=0.5, connectivity=4),
    )
    eight = semantic_regions_from_map(
        score_map,
        erp_shape_hw=ERP_SHAPE,
        features=features,
        region_id_prefix="object",
        class_id=1,
        class_name="object",
        config=SemanticRegionProposalConfig(threshold=0.5, connectivity=8),
    )

    assert four.raw_component_count == 2
    assert len(four.regions) == 2
    assert eight.raw_component_count == 1
    assert len(eight.regions) == 1
    np.testing.assert_array_equal(eight.regions[0].feature_indices, (0, 1))


def test_component_filters_report_small_no_feature_and_truncated_candidates() -> None:
    score_map = np.zeros(MAP_SHAPE, dtype=np.float32)
    score_map[0, 0] = 1.0  # no feature support
    score_map[1, 2:4] = (0.9, 0.8)  # retained
    score_map[2, 4:6] = (0.7, 0.7)  # truncated after feature validation
    score_map[0, 4] = 0.6  # too small
    features = _features(((1, 2), (1, 3), (2, 4), (2, 5)))

    result = semantic_regions_from_map(
        score_map,
        erp_shape_hw=ERP_SHAPE,
        features=features,
        region_id_prefix="candidate",
        class_id=4,
        class_name="candidate",
        config=SemanticRegionProposalConfig(
            threshold=0.5,
            connectivity=4,
            min_component_cells=2,
            min_feature_count=2,
            max_regions=1,
        ),
    )

    assert result.raw_component_count == 4
    assert result.rejected_small_component_count == 2
    assert result.rejected_no_feature_count == 0
    assert result.truncated_component_count == 1
    assert len(result.regions) == 1
    np.testing.assert_array_equal(result.regions[0].feature_indices, (0, 1))


def test_component_without_features_is_rejected_before_deterministic_ranking() -> None:
    score_map = np.zeros(MAP_SHAPE, dtype=np.float32)
    score_map[0, 0] = 1.0
    score_map[2, 3] = 0.8
    features = _features(((2, 3),))

    result = semantic_regions_from_map(
        score_map,
        erp_shape_hw=ERP_SHAPE,
        features=features,
        region_id_prefix="candidate",
        class_id=4,
        class_name="candidate",
        config=SemanticRegionProposalConfig(threshold=0.5),
    )

    assert result.raw_component_count == 2
    assert result.rejected_no_feature_count == 1
    assert result.regions[0].region_id == "candidate:4:000"
    np.testing.assert_array_equal(result.regions[0].feature_indices, (0,))


def test_empty_thresholded_map_returns_explicit_empty_diagnostics() -> None:
    result = semantic_regions_from_map(
        np.full(MAP_SHAPE, 0.2, dtype=np.float32),
        erp_shape_hw=ERP_SHAPE,
        features=_features(((1, 1),)),
        region_id_prefix="empty",
        class_id=9,
        class_name="empty",
        config=SemanticRegionProposalConfig(threshold=0.5),
    )

    assert result.regions == ()
    assert result.raw_component_count == 0
    assert result.describe()["retained_component_count"] == 0


def test_component_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="interval"):
        SemanticRegionProposalConfig(threshold=0.0)
    with pytest.raises(ValueError, match="connectivity"):
        SemanticRegionProposalConfig(connectivity=6)
    with pytest.raises(ValueError, match="min_component_cells"):
        SemanticRegionProposalConfig(min_component_cells=0)
    with pytest.raises(ValueError, match="max_regions"):
        SemanticRegionProposalConfig(max_regions=0)
