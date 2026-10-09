from __future__ import annotations

from benchmarks.spherical_fcn_cam.run_experiment import _cam_statistics, _load_erp
from benchmarks.spherical_fcn_cam.run_public_datasets import (
    DATASET_LICENSES,
    FCN_ADAPTER,
    METHOD_SCHEMA,
    RUNNER,
    SPHERICAL_CORE,
    _sha256,
    _run_one,
    select_group_distinct_samples,
)
import numpy as np
from PIL import Image
import pytest


def _record(dataset: str, group: str, view: str, partition: str = "development"):
    return {
        "schema": METHOD_SCHEMA,
        "dataset_id": dataset,
        "partition": partition,
        "spatial_group_id": group,
        "from_view_id": f"{dataset}::{group}::{view}",
        "pair_id": f"pair-{dataset}-{group}-{view}",
    }


def test_selection_is_deterministic_group_distinct_and_excludes_heldout() -> None:
    records = [
        _record("matterport360", "building-b", "view-z"),
        _record("matterport360", "building-a", "view-z"),
        _record("matterport360", "building-a", "view-a"),
        _record("matterport360", "building-c", "view-a", "heldout"),
        _record("stanford2d3d", "area-2", "view-b"),
        _record("stanford2d3d", "area-2", "view-a"),
    ]

    selected = select_group_distinct_samples(
        reversed(records),
        datasets=("matterport360", "stanford2d3d"),
        partition="development",
        groups_per_dataset=2,
    )

    assert [(item["dataset_id"], item["spatial_group_id"]) for item in selected] == [
        ("matterport360", "building-a"),
        ("matterport360", "building-b"),
        ("stanford2d3d", "area-2"),
    ]
    assert selected[0]["from_view_id"].endswith("view-a")


def test_cam_statistics_use_spherical_area_and_validate_input() -> None:
    cam = np.zeros((4, 8), dtype=np.float32)
    cam[0] = 1.0
    polar = _cam_statistics(cam)
    cam[:] = 0.0
    cam[1] = 1.0
    equatorial = _cam_statistics(cam)

    assert polar["mean"] == pytest.approx(equatorial["mean"])
    assert polar["spherical_mean"] < equatorial["spherical_mean"]
    assert polar["spherical_fraction_ge_0_5"] < equatorial["spherical_fraction_ge_0_5"]
    with pytest.raises(ValueError, match="finite HW"):
        _cam_statistics(np.array([np.nan]))


def test_completed_matching_result_is_resumed_without_subprocess(
    tmp_path, monkeypatch
) -> None:
    record = _record("matterport360", "building-a", "view-a")
    source = tmp_path / "input.png"
    source.write_bytes(b"not decoded on resume")
    record["from_rgb_path"] = str(source)
    result_path = (
        tmp_path / "output" / "matterport360" / "view-a" / "resnet18" / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    expected = {
        "dataset_sample": {"view_id": record["from_view_id"]},
        "input": {"license": DATASET_LICENSES["matterport360"]},
        "input_resolution_mode": "resized",
        "implementation": {
            "runner_sha256": _sha256(RUNNER),
            "spherical_core_sha256": _sha256(SPHERICAL_CORE),
            "fcn_adapter_sha256": _sha256(FCN_ADAPTER),
        },
    }
    result_path.write_text(__import__("json").dumps(expected))
    monkeypatch.setattr(
        "benchmarks.spherical_fcn_cam.run_public_datasets.subprocess.run",
        lambda *args, **kwargs: pytest.fail("resume unexpectedly launched subprocess"),
    )

    actual, actual_path = _run_one(
        record,
        "resnet18",
        output_dir=tmp_path / "output",
        erp_height=224,
        preserve_input_resolution=False,
        top_k=3,
        threads=4,
    )

    assert actual == expected
    assert actual_path == result_path


def test_erp_loader_can_preserve_or_explicitly_resize_source_lattice(tmp_path) -> None:
    source = tmp_path / "erp.png"
    Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(source)

    native_rgb, native_tensor = _load_erp(source, None)
    resized_rgb, resized_tensor = _load_erp(source, 4)

    assert native_rgb.shape == (8, 16, 3)
    assert native_tensor.shape == (1, 3, 8, 16)
    assert resized_rgb.shape == (4, 8, 3)
    assert resized_tensor.shape == (1, 3, 4, 8)
