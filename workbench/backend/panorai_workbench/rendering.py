"""Semantic, type-first rendering for automatically executed public results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image


_DISCRETE_NAMES = {"labels", "label", "mask", "support", "validity", "inlier_mask"}
_CONTINUOUS_NAMES = {
    "range",
    "radial_range",
    "depth",
    "confidence",
    "matching_cost",
    "cost",
    "disparity",
    "residuals_rad",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "result"


def _normalized_u8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.bool_:
        return values.astype(np.uint8) * 255
    if values.dtype == np.uint8:
        return values
    floating = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(floating)
    if not finite.any():
        return np.zeros(floating.shape, dtype=np.uint8)
    low, high = np.percentile(floating[finite], [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    result = np.zeros(floating.shape, dtype=np.float64)
    result[finite] = np.clip((floating[finite] - low) / (high - low), 0.0, 1.0)
    return np.round(result * 255.0).astype(np.uint8)


def _label_rgb(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    return np.stack(
        (
            (labels * 37 + 23) % 251,
            (labels * 67 + 71) % 253,
            (labels * 97 + 131) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)


def _preview_bytes(values: np.ndarray, *, semantic: str) -> bytes | None:
    array = np.asarray(values)
    if array.size == 0 or any(dimension == 0 for dimension in array.shape):
        return None
    if array.ndim == 2:
        if semantic in _DISCRETE_NAMES and array.dtype != np.bool_:
            raster = _label_rgb(array)
        else:
            raster = _normalized_u8(array)
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        raster = _normalized_u8(array)
        if raster.shape[-1] == 1:
            raster = raster[..., 0]
    else:
        return None
    stream = BytesIO()
    Image.fromarray(raster).save(stream, format="PNG")
    return stream.getvalue()


class ResultRenderer:
    """Persist exact values plus lightweight previews and structured summaries."""

    def __init__(self, directory: Path, *, context: Mapping[str, Any] | None = None):
        self.directory = directory / "artifacts"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.artifacts: list[dict[str, Any]] = []
        self._used: set[str] = set()
        self.context = dict(context or {})

    def _id(self, path: str) -> str:
        base = _slug(path)
        candidate = base
        index = 2
        while candidate in self._used:
            candidate = f"{base}-{index}"
            index += 1
        self._used.add(candidate)
        return candidate

    def _record(
        self,
        artifact_id: str,
        kind: str,
        label: str,
        filename: str,
        mime_type: str,
        *,
        interpolation: str | None = None,
    ) -> None:
        path = self.directory / filename
        item: dict[str, Any] = {
            "id": artifact_id,
            "type": kind,
            "label": label,
            "mime_type": mime_type,
            "file": f"artifacts/{filename}",
            "size_bytes": path.stat().st_size,
        }
        if interpolation is not None:
            item["interpolation"] = interpolation
        self.artifacts.append(item)

    def _array(self, value: Any, path: str, semantic: str) -> dict[str, Any]:
        array = np.asarray(value)
        artifact_id = self._id(path)
        filename = f"{artifact_id}.npy"
        np.save(self.directory / filename, array, allow_pickle=False)
        kind = (
            "mask"
            if array.dtype == np.bool_ or semantic in {"mask", "support", "validity"}
            else "labels"
            if semantic in {"labels", "label", "inlier_mask"}
            else "image"
            if array.ndim == 3
            and array.shape[-1] in {1, 3, 4}
            and semantic not in _CONTINUOUS_NAMES
            else "raster"
            if array.ndim in {2, 3}
            else "array"
        )
        interpolation = "nearest" if kind in {"mask", "labels"} else "bilinear"
        self._record(
            artifact_id,
            kind,
            path,
            filename,
            "application/x-npy",
            interpolation=interpolation,
        )
        preview = _preview_bytes(array, semantic=semantic)
        if preview is not None:
            preview_id = self._id(f"{path}-preview")
            preview_name = f"{preview_id}.png"
            (self.directory / preview_name).write_bytes(preview)
            self._record(
                preview_id,
                "preview",
                f"{path} preview",
                preview_name,
                "image/png",
                interpolation=interpolation,
            )
        return {
            "$artifact": artifact_id,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "semantic": semantic,
            "interpolation": interpolation,
        }

    def _walk(self, value: Any, path: str, semantic: str | None = None) -> Any:
        semantic = semantic or path.rsplit(".", 1)[-1].lower()
        if isinstance(value, np.ndarray) or hasattr(value, "detach"):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return self._array(value, path, semantic)
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        # Stable typed raster semantics take precedence over field-name guesses.
        if getattr(value, "interface", None) == "panorai-spherical-raster-field/v1":
            kind = str(value.kind)
            return {
                "interface": value.interface,
                "name": value.name,
                "kind": kind,
                "units": value.units,
                "values": self._walk(value.values, f"{path}.values", kind),
                "support": self._walk(value.support, f"{path}.support", "support"),
                "validity": self._walk(value.validity, f"{path}.validity", "validity"),
                "grid": self._walk(value.grid.describe(), f"{path}.grid"),
            }

        if type(value).__name__ == "EquirectangularImage":
            rendered: dict[str, Any] = {
                "interface": getattr(value, "interface", None),
                "shape_hw": list(value.shape_hw),
            }
            for name in ("image", "depth", "labels"):
                field_value = getattr(value, name, None)
                if field_value is None:
                    continue
                semantic = "range" if name == "depth" else name
                rendered[name] = self._walk(field_value, f"{path}.{name}", semantic)
                rendered[f"{name}_support"] = self._walk(
                    value.support(name), f"{path}.{name}.support", "support"
                )
                rendered[f"{name}_validity"] = self._walk(
                    value.validity(name), f"{path}.{name}.validity", "validity"
                )
            return rendered

        if type(value).__name__ == "GnomonicFaceSet":
            description = value.describe()
            modalities = tuple(description.get("modalities", {}))
            return {
                "description": self._walk(description, f"{path}.description"),
                "faces": [
                    {
                        "id": getattr(face, "face_id", str(index)),
                        **{
                            name: self._walk(
                                face.field(name), f"{path}.{index}.{name}", name
                            )
                            for name in modalities
                        },
                        **{
                            f"{name}_validity": self._walk(
                                face.validity(name),
                                f"{path}.{index}.{name}.validity",
                                "validity",
                            )
                            for name in modalities
                        },
                        "support": self._walk(
                            face.support_mask,
                            f"{path}.{index}.support",
                            "support",
                        ),
                    }
                    for index, face in enumerate(value)
                ],
            }

        if type(value).__name__ == "SphericalFeatureSet":
            points = np.asarray(value.source_erp_xy)
            return {
                "interface": getattr(value, "interface", None),
                "feature_count": len(value),
                "erp_points": self._array(points, f"{path}.erp-points", "erp-points"),
                "descriptors": self._array(
                    value.descriptors, f"{path}.descriptors", "descriptors"
                ),
                "description": self._walk(value.describe(), f"{path}.description"),
            }
        if type(value).__name__ == "SphericalKeypointSet":
            working_points = np.asarray(value.source_erp_xy)
            reported_points = working_points
            source_shape = self.context.get("source_shape_hw")
            if (
                isinstance(source_shape, (tuple, list))
                and len(source_shape) == 2
                and len(value.source_shape_hw) == 2
            ):
                working_height, working_width = value.source_shape_hw
                source_height, source_width = (int(item) for item in source_shape)
                reported_points = working_points * np.asarray(
                    [source_width / working_width, source_height / working_height],
                    dtype=np.float64,
                )
            return {
                "interface": getattr(value, "interface", None),
                "keypoint_count": len(value),
                "erp_points": self._array(
                    reported_points, f"{path}.erp-points", "erp-points"
                ),
                "working_erp_points": self._array(
                    working_points, f"{path}.working-erp-points", "erp-points"
                ),
                "bearings": self._array(value.bearings, f"{path}.bearings", "bearings"),
                "responses": self._array(
                    value.responses, f"{path}.responses", "responses"
                ),
                "scales_deg": self._array(
                    value.scales_deg, f"{path}.scales-deg", "scales"
                ),
                "working_shape_hw": list(value.source_shape_hw),
                "reported_shape_hw": (
                    list(source_shape)
                    if isinstance(source_shape, (tuple, list))
                    else list(value.source_shape_hw)
                ),
                "diagnostics": self._walk(value.diagnostics, f"{path}.diagnostics"),
            }
        if isinstance(value, Mapping):
            return {
                str(key): self._walk(item, f"{path}.{key}", str(key).lower())
                for key, item in value.items()
            }
        if is_dataclass(value):
            return {
                item.name: self._walk(
                    getattr(value, item.name), f"{path}.{item.name}", item.name.lower()
                )
                for item in fields(value)
            }
        if isinstance(value, (tuple, list)):
            if len(value) > 5000:
                return {"count": len(value), "truncated": True}
            return [
                self._walk(item, f"{path}.{index}") for index, item in enumerate(value)
            ]
        if hasattr(value, "describe") and callable(value.describe):
            return self._walk(value.describe(), f"{path}.description")
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    def render(self, result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        summary = self._walk(result, "result")
        summary_id = self._id("result-summary")
        filename = f"{summary_id}.json"
        (self.directory / filename).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        self._record(
            summary_id,
            "json",
            "Structured result",
            filename,
            "application/json",
        )
        metrics = {
            "artifact_count": len(self.artifacts),
            "result_type": f"{type(result).__module__}.{type(result).__qualname__}",
        }
        return self.artifacts, metrics


def render_result(
    result: Any,
    directory: Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return ResultRenderer(directory, context=context).render(result)


__all__ = ["ResultRenderer", "render_result"]
