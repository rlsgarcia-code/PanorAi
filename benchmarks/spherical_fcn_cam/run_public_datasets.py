#!/usr/bin/env python3
"""Evaluate spherical ImageNet FCNs on licensed local public-dataset ERPs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_experiment.py")
SPHERICAL_CORE = ROOT / "panorai/image_processing/torch.py"
FCN_ADAPTER = ROOT / "panorai/experimental/deep_learning/fcn.py"
SCHEMA = "panorai-spherical-fcn-public-datasets/v1"
METHOD_SCHEMA = "panorai-relative-pose-frontend-method-input/v1"
DEFAULT_MODELS = ("alexnet", "vgg16", "resnet18")
DEFAULT_DATASETS = ("matterport360", "stanford2d3d")
DATASET_LICENSES = {
    "matterport360": (
        "Matterport3D Terms of Use; local licensed research copy; "
        "https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data"
    ),
    "stanford2d3d": (
        "Stanford 2D-3D-S license agreement; local licensed research copy; "
        "https://github.com/alexsax/2D-3D-Semantics"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_group_distinct_samples(
    records: Iterable[dict[str, Any]],
    *,
    datasets: tuple[str, ...],
    partition: str,
    groups_per_dataset: int,
) -> list[dict[str, Any]]:
    """Select the lexicographically first source view from each distinct group."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record.get("schema") != METHOD_SCHEMA:
            raise ValueError("input record has an unexpected schema")
        if record.get("dataset_id") not in datasets:
            continue
        if record.get("partition") != partition:
            continue
        grouped[record["dataset_id"]][record["spatial_group_id"]].append(record)

    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        groups = grouped.get(dataset, {})
        if not groups:
            raise ValueError(f"no {partition} records found for {dataset}")
        for group in sorted(groups)[:groups_per_dataset]:
            candidates = sorted(
                groups[group], key=lambda item: (item["from_view_id"], item["pair_id"])
            )
            selected.append(candidates[0])
    return selected


def _sample_slug(record: dict[str, Any]) -> str:
    return record["from_view_id"].split("::", 2)[-1]


def _run_one(
    record: dict[str, Any],
    model: str,
    *,
    output_dir: Path,
    erp_height: int,
    preserve_input_resolution: bool,
    top_k: int,
    threads: int,
) -> tuple[dict[str, Any], Path]:
    source = Path(record["from_rgb_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    sample_dir = output_dir / record["dataset_id"] / _sample_slug(record)
    result_path = sample_dir / model / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (
            result.get("dataset_sample", {}).get("view_id") == record["from_view_id"]
            and result.get("input", {}).get("license")
            == DATASET_LICENSES[record["dataset_id"]]
            and result.get("input_resolution_mode")
            == ("source" if preserve_input_resolution else "resized")
            and result.get("implementation")
            == {
                "runner_sha256": _sha256(RUNNER),
                "spherical_core_sha256": _sha256(SPHERICAL_CORE),
                "fcn_adapter_sha256": _sha256(FCN_ADAPTER),
            }
        ):
            return result, result_path
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--input",
        str(source),
        "--input-license",
        DATASET_LICENSES[record["dataset_id"]],
        "--output-dir",
        str(sample_dir),
        "--erp-height",
        str(erp_height),
        "--top-k",
        str(top_k),
        "--threads",
        str(threads),
    ]
    if preserve_input_resolution:
        command.append("--preserve-input-resolution")
    completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "KMP_USE_SHM": "0", "OMP_NUM_THREADS": str(threads)},
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            break
        if "OMP: Error #179" not in completed.stderr or attempt == 2:
            break
        time.sleep(1.0)
    assert completed is not None
    if completed.returncode != 0:
        raise RuntimeError(
            f"{record['dataset_id']} {record['from_view_id']} {model} failed:\n"
            f"{completed.stderr}\n{completed.stdout[-2000:]}"
        )
    result = json.loads(result_path.read_text())
    result["dataset_sample"] = {
        "dataset_id": record["dataset_id"],
        "partition": record["partition"],
        "spatial_group_id": record["spatial_group_id"],
        "view_id": record["from_view_id"],
        "source_manifest_pair_id": record["pair_id"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result, result_path


def _contact_sheet(
    dataset: str,
    model: str,
    rows: list[tuple[dict[str, Any], Path]],
    output_path: Path,
) -> None:
    tile_size = (448, 224)
    label_height = 36
    sheet = Image.new(
        "RGB", (2 * tile_size[0], len(rows) * (tile_size[1] + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (result, result_path) in enumerate(rows):
        model_dir = result_path.parent
        source = Image.open(model_dir / "input-erp.jpg").convert("RGB")
        overlay = Image.open(model_dir / result["predictions"][0]["overlay"]).convert(
            "RGB"
        )
        source = source.resize(tile_size, Image.Resampling.LANCZOS)
        overlay = overlay.resize(tile_size, Image.Resampling.LANCZOS)
        y = row_index * (tile_size[1] + label_height)
        sheet.paste(source, (0, y + label_height))
        sheet.paste(overlay, (tile_size[0], y + label_height))
        sample = result["dataset_sample"]
        prediction = result["predictions"][0]
        draw.text((8, y + 10), f"{sample['spatial_group_id']} | input", fill="black")
        draw.text(
            (tile_size[0] + 8, y + 10),
            f"top-1 CAM: {prediction['class_name']} ({prediction['probability']:.3f})",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _aggregate(results: list[tuple[dict[str, Any], Path]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result, _ in results:
        grouped[(result["dataset_sample"]["dataset_id"], result["model"])].append(
            result
        )
    summaries: list[dict[str, Any]] = []
    for (dataset, model), values in sorted(grouped.items()):
        top = [item["predictions"][0] for item in values]
        summaries.append(
            {
                "dataset_id": dataset,
                "model": model,
                "sample_count": len(values),
                "mean_top1_probability": sum(item["probability"] for item in top)
                / len(top),
                "mean_top1_cam_spherical_fraction_ge_0_5": sum(
                    item["cam_statistics"]["spherical_fraction_ge_0_5"] for item in top
                )
                / len(top),
                "top1_classes": [item["class_name"] for item in top],
            }
        )
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_jsonl(args.inputs)
    selected = select_group_distinct_samples(
        records,
        datasets=tuple(args.datasets),
        partition=args.partition,
        groups_per_dataset=args.groups_per_dataset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[dict[str, Any], Path]] = []
    for record in selected:
        for model in args.models:
            print(f"running {record['dataset_id']} {_sample_slug(record)} {model}")
            results.append(
                _run_one(
                    record,
                    model,
                    output_dir=args.output_dir,
                    erp_height=args.erp_height,
                    preserve_input_resolution=args.preserve_input_resolution,
                    top_k=args.top_k,
                    threads=args.threads,
                )
            )

    contact_sheets: list[dict[str, str]] = []
    for dataset in args.datasets:
        for model in args.models:
            matching = [
                item
                for item in results
                if item[0]["dataset_sample"]["dataset_id"] == dataset
                and item[0]["model"] == model
            ]
            path = args.output_dir / f"contact-{dataset}-{model}.jpg"
            _contact_sheet(dataset, model, matching, path)
            contact_sheets.append(
                {"dataset_id": dataset, "model": model, "path": str(path)}
            )

    summary = {
        "schema": SCHEMA,
        "source_manifest": str(args.inputs.resolve()),
        "source_manifest_sha256": _sha256(args.inputs),
        "selection": {
            "partition": args.partition,
            "groups_per_dataset_maximum": args.groups_per_dataset,
            "rule": "first lexicographic source view per distinct spatial group",
            "selected": [
                {
                    "dataset_id": item["dataset_id"],
                    "spatial_group_id": item["spatial_group_id"],
                    "view_id": item["from_view_id"],
                    "input_sha256": _sha256(Path(item["from_rgb_path"])),
                }
                for item in selected
            ],
        },
        "models": list(args.models),
        "input_resolution_mode": (
            "source" if args.preserve_input_resolution else "resized"
        ),
        "erp_height": None if args.preserve_input_resolution else args.erp_height,
        "result_count": len(results),
        "aggregates": _aggregate(results),
        "results": [str(path) for _, path in results],
        "contact_sheets": contact_sheets,
        "license_boundary": (
            "Inputs are pre-existing locally licensed Matterport3D/Stanford2D3D "
            "data. Dataset bytes and generated outputs remain outside the repository."
        ),
        "interpretation": (
            "ImageNet top classes and CAMs are qualitative weak-localization "
            "evidence, not semantic-segmentation accuracy."
        ),
        "stanford_support_limitation": (
            "The prepared manifest provides no explicit RGB support mask. Black "
            "unobserved panorama regions remain in inputs and are not inferred as "
            "invalid from color or depth magnitude."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["aggregates"], indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS
    )
    parser.add_argument("--partition", default="development")
    parser.add_argument("--groups-per-dataset", type=int, default=3)
    parser.add_argument("--erp-height", type=int, default=224)
    parser.add_argument("--preserve-input-resolution", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.groups_per_dataset < 1:
        parser.error("--groups-per-dataset must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
