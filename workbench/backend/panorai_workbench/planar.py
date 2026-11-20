"""Workbench revisions and artifact adapters for public planar APIs only."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import uuid
import zipfile

import numpy as np
from PIL import Image

from panorai.reconstruction.planar import (
    HumanGuidedPlanarReconstruction,
    PlanarReconstructionOptions,
    SegmentPlaneRegions,
    ValidatePlanarScene,
    export_planar_result,
    planar_scene_from_document,
)


def _images(inputs):
    result = {}
    for item in inputs:
        with Image.open(item["path"]) as image:
            result[item["name"]] = np.asarray(image.convert("RGB"))
    return result


class PlanarSceneStore:
    """Content-addressed revisions. Revising never modifies an earlier run."""

    def __init__(self, project):
        self.root = Path(project) / "planar-scenes"

    def save(self, document, inputs, *, scene_id=None):
        document = deepcopy(document)
        scene_id = scene_id or f"scene_{uuid.uuid4().hex}"
        self._validate_id(scene_id)
        if len({i["name"] for i in inputs}) != len(inputs):
            raise ValueError("uploaded image names must be unique")
        document["scene_id"] = scene_id
        document.pop("revision", None)
        scene = planar_scene_from_document(document, _images(inputs))
        referenced = {v["image_key"] for v in document["views"]}
        if referenced != {i["name"] for i in inputs}:
            raise ValueError(
                "uploaded images must exactly match scene image_key values"
            )
        payload = {"document": document, "inputs": inputs}
        identity = {
            "document": document,
            "images": [{"name": i["name"], "sha256": i["sha256"]} for i in inputs],
        }
        revision = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        payload["document"]["revision"] = revision
        payload["scene_id"], payload["revision"] = scene_id, revision
        path = self.root / scene_id / f"{revision}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if existing["document"] != payload["document"]:
                raise ValueError("revision collision")
        return {
            "scene_id": scene_id,
            "revision": revision,
            "document": document,
            "view_count": len(scene.views),
            "review_required": [
                r.plane_id for r in scene.regions if r.state != "accepted"
            ],
        }

    @staticmethod
    def _validate_id(scene_id):
        if not re.fullmatch(r"scene_[0-9a-f]{32}", scene_id):
            raise ValueError("invalid scene identifier")

    def load(self, scene_id, revision):
        self._validate_id(scene_id)
        if not re.fullmatch(r"[0-9a-f]{64}", revision):
            raise ValueError("invalid revision")
        path = self.root / scene_id / f"{revision}.json"
        if not path.is_file():
            raise KeyError("scene revision not found")
        return json.loads(path.read_text(encoding="utf-8"))


def execute_planar_run(run, directory):
    from .runs import _artifact, _png_bytes
    from panorai.pipeline import Pipeline

    snapshot = run.get("planar_scene")
    if snapshot is None:
        raise ValueError(
            "create this application run through a saved planar scene revision"
        )
    scene = planar_scene_from_document(snapshot, _images(run["inputs"]))
    params = run["parameters"]
    sam = None
    segment_target = None
    if params["operation"] == "segment":
        index = int(params.get("region_index", -1))
        if index >= 0:
            if index >= len(scene.regions):
                raise ValueError("selected region index is outside the saved revision")
            target = scene.regions[index]
            segment_target = {"plane_id": target.plane_id, "view_id": target.view_id}
            scene = replace(
                scene,
                regions=(target,),
                correspondences=tuple(
                    t for t in scene.correspondences if t.plane_id == target.plane_id
                ),
            )
        if params["segmentation"] == "sam":
            from .resources import require_sam21_runtime

            require_sam21_runtime()
            from panorai.experimental.deep_learning.segmentation import (
                load_sam21_hiera_large,
            )

            sam = load_sam21_hiera_large(
                accept_upstream_terms=params["accept_sam_terms"] == "yes"
            )
        pipeline = Pipeline(
            ("validate", ValidatePlanarScene()),
            (
                "segment",
                SegmentPlaneRegions(
                    params["segmentation"], int(params["face_size"]), sam
                ),
            ),
        )
    else:
        options = PlanarReconstructionOptions(
            automatic_matching=params["matching"] == "sift",
            feature_face_size=int(params["face_size"]),
            plane_threshold_m=float(params["plane_threshold_m"]),
            maximum_range_sigma_m=float(params["maximum_range_sigma_m"]),
        )
        pipeline = HumanGuidedPlanarReconstruction(options).pipeline
    result = pipeline.run(scene)
    manifest = export_planar_result(
        result.output, directory / "planar-export", pipeline_manifest=result.manifest()
    )
    export = directory / "planar-export"
    (export / "scene.json").write_text(
        json.dumps(snapshot, allow_nan=False), encoding="utf-8"
    )
    # Include the exact scene revision and runtime in the bundle manifest.
    scene_bytes = (export / "scene.json").read_bytes()
    manifest["files"].append(
        {
            "file": "scene.json",
            "sha256": hashlib.sha256(scene_bytes).hexdigest(),
            "size_bytes": len(scene_bytes),
        }
    )
    manifest["runtime"] = run["runtime"]
    manifest["input_images"] = [
        {"name": i["name"], "sha256": i["sha256"]} for i in run["inputs"]
    ]
    manifest["sam"] = None if sam is None else sam.assets.to_dict()
    manifest["segmentation_target"] = segment_target
    (export / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    artifacts = []
    summary = json.loads((export / "result.json").read_text())
    _artifact(
        directory,
        artifacts,
        "planar-result",
        "planes",
        "Planos, correspondências e revisão",
        summary,
        extension="json",
    )
    _artifact(
        directory,
        artifacts,
        "pipeline",
        "pipeline-trace",
        "Etapas executadas",
        result.manifest(),
        extension="json",
    )
    for index, ranges in enumerate(result.output.ranges):
        view = scene.views[index]
        valid = ranges.validity
        preview = view.image.copy()
        preview[valid] = np.round(
            preview[valid] * 0.55 + np.array([255, 200, 87]) * 0.45
        ).astype(np.uint8)
        preview[ranges.conflict] = (240, 70, 85)
        _artifact(
            directory,
            artifacts,
            f"overlay-{index}",
            "preview",
            f"{view.view_id} · regiões reconstruídas",
            _png_bytes(preview),
        )
        range_preview = np.zeros_like(view.image)
        if valid.any():
            import cv2

            maximum = float(np.percentile(ranges.range[valid], 99))
            scaled = np.where(
                valid, np.clip(ranges.range / max(maximum, 1e-6), 0, 1), 0
            )
            range_preview = cv2.applyColorMap(
                (scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )[..., ::-1]
            range_preview[~valid] = 0
        _artifact(
            directory,
            artifacts,
            f"range-{index}",
            "range",
            f"{view.view_id} · range radial",
            _png_bytes(range_preview),
        )
    archive_path = directory / "planar-export.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in export.iterdir():
            archive.write(path, path.name)
    artifacts.append(
        {
            "id": "planar-export",
            "type": "archive",
            "label": "Exportar NPZ, PLY, máscaras e manifesto",
            "mime_type": "application/zip",
            "file": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
        }
    )
    metrics = {
        "plane_count": len(result.output.planes),
        "accepted_planes": sum(p.state == "accepted" for p in result.output.planes),
        "triangulated_tracks": sum(t.reason is None for t in result.output.tracks),
        "native_valid_pixels": {
            r.view_id: int(r.validity.sum()) for r in result.output.ranges
        },
        "revision": scene.revision,
    }
    return (
        artifacts,
        metrics,
        [
            "A incerteza considera poses fixas e não inclui erros de máscara ou calibração."
        ],
    )
