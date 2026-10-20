from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from panorai_workbench import runs as runs_module
from panorai_workbench.runs import RunStore, _input_support, _store_root


def _erp_bytes() -> bytes:
    y, x = np.mgrid[:16, :32]
    image = np.stack(
        ((x * 8) % 256, (y * 16) % 256, ((x + y) * 5) % 256), axis=-1
    ).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def test_automatic_support_removes_only_black_voids_connected_to_both_poles() -> None:
    image = np.full((12, 24, 3), 127, dtype=np.uint8)
    image[:2] = 0
    image[-3:] = 0
    image[6, 8:12] = 0  # A valid black object inside the observed footprint.

    support, diagnostic = _input_support(image, "auto-polar-voids")

    assert not support[:2].any()
    assert not support[-3:].any()
    assert support[6, 8:12].all()
    assert diagnostic["detected_polar_voids"] is True


def test_automatic_support_keeps_full_erp_and_full_sphere_is_explicit() -> None:
    image = np.zeros((12, 24, 3), dtype=np.uint8)
    image[0, 0] = 64
    image[-1, -1] = 64

    automatic, diagnostic = _input_support(image, "auto-polar-voids")
    explicit, _ = _input_support(image, "full-sphere")

    assert automatic.all()
    assert explicit.all()
    assert diagnostic["detected_polar_voids"] is False


def test_default_store_is_anchored_to_the_workbench_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("PANORAI_WORKBENCH_HOME", raising=False)
    monkeypatch.chdir(tmp_path)

    assert (
        _store_root()
        == (Path(__file__).resolve().parents[2] / "outputs" / "workbench").resolve()
    )


def test_installed_backend_finds_workbench_from_launch_checkout(
    monkeypatch, tmp_path
) -> None:
    workbench_root = Path(__file__).resolve().parents[2]
    fake_module = tmp_path / "site-packages" / "panorai_workbench" / "runs.py"
    monkeypatch.setattr(runs_module, "__file__", str(fake_module))
    monkeypatch.chdir(workbench_root.parent)

    assert runs_module._application_root() == workbench_root


def test_run_store_deduplicates_inputs_and_persists_a_public_geometry_run(
    tmp_path,
) -> None:
    store = RunStore(tmp_path)
    payload = _erp_bytes()
    first = store.ingest(payload, "first.png")
    second = store.ingest(payload, "same-content.png")
    assert first["sha256"] == second["sha256"]
    assert first["path"] == second["path"]

    created = store.create(
        experiment_id="gnomonic-projection",
        workspace="single",
        parameters={"output_size": 64, "hfov_deg": 75.0},
        inputs=[first],
    )
    completed = store.execute(created["id"])
    reloaded = store.load(created["id"])
    assert completed["status"] == reloaded["status"] == "completed"
    assert reloaded["inputs"][0]["sha256"] == first["sha256"]
    assert reloaded["parameters"]["output_size"] == 64
    assert {item["type"] for item in reloaded["artifacts"]} == {"raster", "mask"}
    assert store.input_path(created["id"], 0).read_bytes() == payload
    with pytest.raises(KeyError):
        store.input_path(created["id"], 1)


def test_run_contract_rejects_unknown_or_out_of_range_parameters(tmp_path) -> None:
    store = RunStore(tmp_path)
    input_record = store.ingest(_erp_bytes(), "erp.png")
    with pytest.raises(ValueError, match="unknown parameter"):
        store.create(
            experiment_id="gnomonic-projection",
            workspace="single",
            parameters={"private_import": "panorai._internal"},
            inputs=[input_record],
        )
    with pytest.raises(ValueError, match="below its minimum"):
        store.create(
            experiment_id="gnomonic-projection",
            workspace="single",
            parameters={"output_size": 1},
            inputs=[input_record],
        )


def test_discrete_raster_fields_are_nearest_neighbour_by_contract() -> None:
    from panorai.pipeline import RasterField, RasterGrid

    labels = np.zeros((4, 8), dtype=np.int32)
    support = np.ones_like(labels, dtype=bool)
    field = RasterField(
        "labels",
        labels,
        "labels",
        support,
        support,
        RasterGrid("erp", labels.shape, labels.shape, "pixel-centres"),
    )
    assert field.interpolation == "nearest"


def test_cancelled_run_is_terminal_and_never_publishes_artifacts(tmp_path) -> None:
    store = RunStore(tmp_path)
    input_record = store.ingest(_erp_bytes(), "erp.png")
    created = store.create(
        experiment_id="gnomonic-projection",
        workspace="single",
        parameters={"output_size": 64},
        inputs=[input_record],
    )
    cancelled = store.cancel(created["id"])
    assert cancelled["status"] == "cancelled"
    assert store.execute(created["id"])["status"] == "cancelled"
    assert store.load(created["id"])["artifacts"] == []


def test_backend_restart_marks_orphaned_runs_as_failed(tmp_path) -> None:
    store = RunStore(tmp_path)
    input_record = store.ingest(_erp_bytes(), "erp.png")
    created = store.create(
        experiment_id="gnomonic-projection",
        workspace="single",
        parameters={},
        inputs=[input_record],
    )
    run = store.load(created["id"])
    run["status"] = "running"
    run["worker_pid"] = 2_147_483_647
    store.save(run)

    recovered = RunStore(tmp_path).load(created["id"])
    assert recovered["status"] == "failed"
    assert recovered["error"].startswith("RunInterrupted:")
    assert recovered["events"][-1]["state"] == "failed"
