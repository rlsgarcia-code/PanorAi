"""Run the required Workbench scientific smoke checks on synthetic ERPs."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from panorai_workbench.runs import (
    RunStore,
    _input_support,
    _operation_result,
)


def _erp(height: int, width: int, *, polar_voids: bool) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    checker = ((x // max(2, width // 32) + y // max(2, height // 16)) % 2) * 96
    image = np.stack(
        (
            (x * 255 // max(width - 1, 1) + checker) % 256,
            (y * 255 // max(height - 1, 1) + checker) % 256,
            ((x + 2 * y) * 7 + checker) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    image[height // 2 - 2 : height // 2 + 2, width // 3 : width // 3 + 8] = 0
    if polar_voids:
        cap = max(2, height // 12)
        image[:cap] = 0
        image[-cap:] = 0
    return image


def _png(image: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog-source-width", type=int, default=1024)
    parser.add_argument("--dog-working-width", type=int, default=512)
    args = parser.parse_args()
    if args.dog_source_width <= args.dog_working_width:
        raise SystemExit("dog source width must exceed working width")

    full = _erp(48, 96, polar_voids=False)
    full_support, full_info = _input_support(full, "auto-polar-voids")
    if not full_support.all() or full_info["detected_polar_voids"]:
        raise SystemExit("complete ERP was incorrectly assigned polar voids")

    partial = _erp(48, 96, polar_voids=True)
    partial_support, partial_info = _input_support(partial, "auto-polar-voids")
    if not partial_info["detected_polar_voids"]:
        raise SystemExit("polar voids were not detected")
    if partial_support[0].any() or partial_support[-1].any():
        raise SystemExit("polar voids remained supported")
    if not partial_support[24, 32:40].all():
        raise SystemExit("interior black pixels were incorrectly invalidated")

    slic_labels, slic_provenance = _operation_result(
        "spherical-slic",
        partial,
        {
            "n_segments": 24,
            "compactness": 8.0,
            "working_width": "48",
            "input_support": "auto-polar-voids",
        },
    )
    slic_labels = np.asarray(slic_labels)
    if slic_labels.shape != partial.shape[:2] or slic_labels.dtype.kind not in "iu":
        raise SystemExit("SLIC did not restore integer labels on the original ERP")
    if not np.all(slic_labels[~partial_support] == -1):
        raise SystemExit("SLIC did not preserve explicit unsupported labels")
    if not np.all(slic_labels[24, 32:40] >= 0):
        raise SystemExit("SLIC rejected valid interior black pixels")
    if "nearest" not in str(slic_provenance["workbench_resolution"]["restored_with"]):
        raise SystemExit("SLIC labels were not restored with nearest-neighbour")

    dog_height = args.dog_source_width // 2
    dog = _erp(dog_height, args.dog_source_width, polar_voids=True)
    with tempfile.TemporaryDirectory(prefix="panorai-workbench-smoke-") as temporary:
        store = RunStore(Path(temporary))
        item = store.ingest(_png(dog), "scientific-smoke.png")

        gnomonic = store.create(
            experiment_id="gnomonic-projection",
            workspace="single",
            parameters={"output_size": 96, "hfov_deg": 80.0},
            inputs=[item],
        )
        gnomonic = store.execute(gnomonic["id"])
        if gnomonic["status"] != "completed":
            raise SystemExit(f"gnomonic smoke failed: {gnomonic.get('error')}")

        dog_run = store.create(
            experiment_id="spherical-dog-features",
            workspace="single",
            parameters={
                "max_features": 128,
                "contrast_threshold": 0.004,
                "working_width": str(args.dog_working_width),
                "input_support": "auto-polar-voids",
            },
            inputs=[item],
        )
        dog_run = store.execute(dog_run["id"])
        if dog_run["status"] != "completed":
            raise SystemExit(f"DoG smoke failed: {dog_run.get('error')}")
        keypoint_artifact = next(
            item for item in dog_run["artifacts"] if item["id"] == "keypoints"
        )
        table = json.loads(
            store.artifact_path(dog_run["id"], keypoint_artifact["id"]).read_text(
                encoding="utf-8"
            )
        )
        for feature in table["features"]:
            x, y = feature["erp_xy"]
            if not (0 <= x < args.dog_source_width and 0 <= y < dog_height):
                raise SystemExit(
                    "DoG reported working-grid coordinates as source ERP pixels"
                )
        expected_working = [
            args.dog_working_width // 2,
            args.dog_working_width,
        ]
        if dog_run["metrics"]["working_shape_hw"] != expected_working:
            raise SystemExit("DoG did not use the requested reduced working resolution")

    print(
        json.dumps(
            {
                "gnomonic": "completed",
                "slic_labels": int(np.unique(slic_labels[slic_labels >= 0]).size),
                "full_erp_support_fraction": 1.0,
                "polar_support_fraction": partial_info["valid_fraction"],
                "dog_working_shape_hw": expected_working,
                "dog_keypoints": len(table["features"]),
                "dog_coordinates": "original-erp",
                "discrete_restore": "nearest",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
