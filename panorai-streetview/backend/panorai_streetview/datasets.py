"""Explicit external-dataset importers for local validation scenes.

Dataset paths are accepted only by this local command-line adapter.  They are
never exposed as browser-controlled API parameters and are not persisted in
project manifests.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .models import ImagePairInput, ProjectRecord, is_valid_registered_pose
from .store import ProjectStore


STANFORD_TO_PANORAI = np.diag((1.0, -1.0, 1.0))
SUPPORTED_SELECTION_SCHEMA = "panorai-public-three-source-stereo-selection/v1"


def _nearest_rotation(value: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(value, dtype=np.float64))
    sign = np.linalg.det(left @ right)
    return left @ np.diag((1.0, 1.0, sign)) @ right


def _stanford_pose(path: Path) -> dict[str, list[Any]]:
    try:
        matrix = np.loadtxt(path, dtype=np.float64).reshape(3, 4)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid Stanford pose file: {path.name}") from exc
    rotation_dataset = matrix[:, :3]
    center_dataset = -rotation_dataset.T @ matrix[:, 3]
    rotation = _nearest_rotation(
        STANFORD_TO_PANORAI @ rotation_dataset @ STANFORD_TO_PANORAI
    )
    center = STANFORD_TO_PANORAI @ center_dataset
    pose = {
        "rotation_world_to_panorama": rotation.tolist(),
        "center_world": center.tolist(),
    }
    if not is_valid_registered_pose(pose):
        raise ValueError(f"Stanford pose is not a finite rigid transform: {path.name}")
    return pose


def import_stanford_selection(
    *,
    store: ProjectStore,
    dataset_root: Path,
    selection_manifest: Path,
    set_id: str,
    project_name: str | None = None,
    input_contract: Literal["no-poses", "registered-ground-truth"] = "no-poses",
) -> ProjectRecord:
    """Import one frozen Stanford set under an explicit input contract.

    The default application contract reads RGB pixels only.  Dataset poses and
    derived pair metrics are available solely through the explicit
    ``registered-ground-truth`` validation mode.
    """

    dataset_root = dataset_root.resolve()
    selection_manifest = selection_manifest.resolve()
    try:
        manifest_bytes = selection_manifest.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("selection manifest is not readable JSON") from exc
    if payload.get("schema") != SUPPORTED_SELECTION_SCHEMA:
        raise ValueError("unsupported Stanford selection manifest schema")
    selected = next(
        (item for item in payload.get("sets", []) if item.get("set_id") == set_id),
        None,
    )
    if not isinstance(selected, dict) or selected.get("dataset_id") != "stanford2d3d":
        raise ValueError(f"unknown Stanford selection set: {set_id}")
    group = str(selected.get("group", ""))
    reference = str(selected.get("reference", ""))
    sources = selected.get("sources")
    if not group or not reference or not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("a Street View validation set requires at least two pairs")

    view_ids = [reference, *(str(item.get("source", "")) for item in sources)]
    if any(not view_id for view_id in view_ids) or len(set(view_ids)) != len(view_ids):
        raise ValueError("selection contains empty or duplicate view identifiers")
    selection_sha = sha256(manifest_bytes).hexdigest()
    uploads: list[tuple[str, str, bytes]] = []
    metadata: dict[str, Any] = {}
    registered_ground_truth = input_contract == "registered-ground-truth"
    for index, view_id in enumerate(view_ids):
        filename = f"{view_id}.png"
        image_path = dataset_root / "rgb" / group / filename
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"missing Stanford RGB view: {filename}") from exc
        uploads.append((filename, "image/png", image_bytes))
        metadata[filename] = {
            "dataset": {
                "id": "stanford2d3d",
                "group": group,
                "view_id": view_id,
                "role": (
                    "reference"
                    if registered_ground_truth and index == 0
                    else "source"
                    if registered_ground_truth
                    else "input"
                ),
                "selection_set_id": set_id,
                "selection_manifest_sha256": selection_sha,
                "input_contract": input_contract,
            },
        }
        if registered_ground_truth:
            pose_path = dataset_root / "poses" / group / f"{view_id}.txt"
            metadata[filename]["pose"] = _stanford_pose(pose_path)

    if registered_ground_truth:
        pair_inputs = [
            ImagePairInput(
                id=str(item["pair_id"]),
                reference_filename=f"{reference}.png",
                source_filename=f"{item['source']}.png",
                baseline_m=float(item["baseline_m"]),
                minimum_directional_overlap=float(
                    item["minimum_directional_cloud_overlap"]
                ),
                provenance={
                    "dataset": "stanford2d3d",
                    "selection_set_id": set_id,
                    "selection_manifest_sha256": selection_sha,
                    "input_contract": input_contract,
                },
            )
            for item in sources
        ]
    else:
        pair_inputs = [
            ImagePairInput(
                id=f"candidate-{index:02d}",
                reference_filename=f"{left}.png",
                source_filename=f"{right}.png",
                provenance={
                    "dataset": "stanford2d3d",
                    "selection_set_id": set_id,
                    "selection_manifest_sha256": selection_sha,
                    "input_contract": input_contract,
                    "candidate_policy": "complete-four-view-graph",
                },
            )
            for index, (left, right) in enumerate(combinations(view_ids, 2), start=1)
        ]
    return store.create(
        name=project_name or f"Stanford · {group} · {set_id}",
        workflow=(
            "spherical_registered"
            if registered_ground_truth
            else "spherical_unregistered"
        ),
        uploads=uploads,
        metadata=metadata,
        pair_inputs=pair_inputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a frozen external Stanford2D3D multi-pair scene."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument(
        "--input-contract",
        choices=("no-poses", "registered-ground-truth"),
        default="no-poses",
        help="default: import pixels and candidate pairs without reading pose truth",
    )
    args = parser.parse_args()
    project = import_stanford_selection(
        store=ProjectStore(args.data_root),
        dataset_root=args.dataset_root,
        selection_manifest=args.selection_manifest,
        set_id=args.set_id,
        project_name=args.name,
        input_contract=args.input_contract,
    )
    print(project.model_dump_json(indent=2, by_alias=True))


if __name__ == "__main__":
    main()
