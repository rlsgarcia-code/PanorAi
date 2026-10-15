from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mask_rcnn_paper_checkpoint_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("mask_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_house_taxonomy_and_oracle_keep_instances_separate(tmp_path: Path) -> None:
    module = _module()
    house = tmp_path / "scene.house"
    house.write_text(
        "C 0 1 television 1 television 0 0 0 0 0\n"
        "C 1 2 pillow 2 cushion 0 0 0 0 0\n"
        "O 4 0 0 0 0 0\n"
        "O 7 0 0 0 0 0\n"
        "O 8 0 1 0 0 0\n",
        encoding="utf-8",
    )
    categories, objects = module.parse_house_taxonomy(house)
    ids = np.asarray(((4, 4, 0, 7), (4, 0, 8, 7)))
    support = ids != 0

    instances = module.oracle_instances(
        ids,
        support,
        categories,
        objects,
        ("television", "pillow"),
        minimum_pixels=1,
    )

    assert categories[0] == ("television",)
    assert len(instances["television"]) == 2
    assert len(instances["pillow"]) == 1
    assert sorted(int(mask.sum()) for mask in instances["television"]) == [2, 3]


def test_mask_metrics_use_spherical_iou_and_fail_closed_for_missing_concepts() -> None:
    module = _module()
    target = np.zeros((8, 16), dtype=bool)
    target[3:5, 6:9] = True
    wrong = np.roll(target, 5, axis=1)

    metrics = module.evaluate_masks(
        [("television", target)],
        [("pillow", wrong)],
        {"television": [target]},
    )

    assert metrics["proposal_recall_spherical_iou_0_5"] == pytest.approx(1.0)
    assert metrics["median_fused_mask_spherical_iou"] == pytest.approx(0.0)
    assert metrics["non_abstained_instance_precision"] == pytest.approx(0.0)
    assert metrics["non_abstained_instance_coverage"] == pytest.approx(0.0)


def test_development_view_requires_exact_fixed_index() -> None:
    module = _module()
    report = {"views": [{"panorama_index": 1}]}

    assert module._development_view(report, 1)["panorama_index"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        module._development_view(report, 2)
