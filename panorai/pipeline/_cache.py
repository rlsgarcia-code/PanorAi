"""Opt-in content-addressed cache for safe NumPy pipeline results."""

from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from ._serialization import _canonical_json


CACHE_FORMAT = "panorai-pipeline-cache/v1"
GEOMETRY_CONTRACT = "geometry-v1"


class PipelineCacheError(ValueError):
    """A cache identity or payload is unsupported, incomplete, or corrupt."""


def _hash_array(hasher: Any, value: Any, *, path: str) -> None:
    module = type(value).__module__
    backend = "numpy"
    device = "cpu"
    if module == "torch" or module.startswith("torch."):
        if bool(value.requires_grad):
            raise PipelineCacheError(
                f"{path}: tensors requiring gradients cannot be cached because "
                "cache reuse must never disconnect autograd"
            )
        if str(value.dtype) == "torch.bfloat16":
            raise PipelineCacheError(
                f"{path}: torch.bfloat16 has no safe NumPy cache codec"
            )
        backend = "torch"
        device = str(value.device)
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise PipelineCacheError(
            f"{path}: cache supports numeric NumPy arrays and non-autograd tensors"
        )
    if value.dtype.hasobject:
        raise PipelineCacheError(f"{path}: object arrays cannot be cached safely")
    contiguous = np.ascontiguousarray(value)
    header = _canonical_json(
        {
            "backend": backend,
            "device": device,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "strides": list(value.strides),
            "c_contiguous": bool(value.flags.c_contiguous),
            "f_contiguous": bool(value.flags.f_contiguous),
        }
    )
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    payload = memoryview(contiguous).cast("B")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _workflow_identity(value: Any, hasher: Any) -> str:
    if callable(getattr(value, "_workflow_data", None)):
        metadata = value._require_workflow()
        data = value._workflow_data()
        descriptor = {
            "type": "EquirectangularImage",
            "fields": sorted(data),
            "shadow_angle": float(value.shadow_angle),
            "shadow_padded": bool(value.shadow_padded),
            "lat": float(value.lat),
            "lon": float(value.lon),
        }
        hasher.update(_canonical_json(descriptor))
        for name in sorted(data):
            item = metadata[name]
            semantic_keys = (
                "kind",
                "field_kind",
                "units",
                "interpolation",
                "invalid_policy",
                "blend",
            )
            hasher.update(
                _canonical_json(
                    {
                        "name": name,
                        "metadata": {
                            key: item[key] for key in semantic_keys if key in item
                        },
                    }
                )
            )
            _hash_array(hasher, data[name], path=f"input.{name}.values")
            _hash_array(
                hasher,
                value._workflow_support[name],
                path=f"input.{name}.support",
            )
            _hash_array(
                hasher,
                item["validity"],
                path=f"input.{name}.validity",
            )
        return "erp"
    module = type(value).__module__
    if (
        isinstance(value, np.ndarray)
        or module == "torch"
        or module.startswith("torch.")
    ):
        hasher.update(_canonical_json({"type": "array"}))
        _hash_array(hasher, value, path="input")
        return "array"
    raise PipelineCacheError(
        "initial cache inputs must be NumPy arrays or workflow "
        "EquirectangularImage objects"
    )


def cache_identity(pipeline: Any, value: Any) -> dict[str, Any]:
    from panorai import __version__
    from ._serialization import pipeline_torch_state_sha256

    config = pipeline.to_config()
    try:
        torch_state_sha256 = pipeline_torch_state_sha256(pipeline)
    except ValueError as exc:
        raise PipelineCacheError(str(exc)) from exc
    config_payload = _canonical_json(config)
    input_hasher = sha256()
    input_kind = _workflow_identity(value, input_hasher)
    input_digest = input_hasher.hexdigest()
    pipeline_digest = sha256(config_payload).hexdigest()
    key_payload = _canonical_json(
        {
            "format": CACHE_FORMAT,
            "panorai_version": __version__,
            "pipeline_interface": pipeline.interface,
            "geometry_contract": GEOMETRY_CONTRACT,
            "pipeline_sha256": pipeline_digest,
            "input_sha256": input_digest,
            "input_kind": input_kind,
            "torch_state_sha256": torch_state_sha256,
        }
    )
    return {
        "key": sha256(key_payload).hexdigest(),
        "pipeline_sha256": pipeline_digest,
        "input_sha256": input_digest,
        "input_kind": input_kind,
        "geometry_contract": GEOMETRY_CONTRACT,
        "panorai_version": __version__,
        "torch_state_sha256": torch_state_sha256,
    }


def _cache_root(memory: Any) -> Path:
    root = Path(memory)
    if root.exists() and not root.is_dir():
        raise PipelineCacheError("pipeline memory must name a directory")
    return root / CACHE_FORMAT.replace("/", "-")


def _entry_path(memory: Any, key: str) -> Path:
    return _cache_root(memory) / key[:2] / key


def _field_metadata(value: Any, name: str) -> dict[str, Any]:
    item = value._workflow_metadata[name]
    return {
        "name": name,
        "metadata": {
            key: item[key]
            for key in (
                "kind",
                "field_kind",
                "units",
                "interpolation",
                "invalid_policy",
                "blend",
            )
            if key in item
        },
    }


def _encode_result(result: Any, source: Any) -> tuple[bytes, dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    workflow_data = (
        result._workflow_data()
        if callable(getattr(result, "_workflow_data", None))
        else None
    )
    if workflow_data is not None and all(
        isinstance(item, np.ndarray) for item in workflow_data.values()
    ):
        data = workflow_data
        fields = []
        for index, name in enumerate(sorted(data)):
            for suffix, array in (
                ("values", data[name]),
                ("support", result._workflow_support[name]),
                ("validity", result._workflow_metadata[name]["validity"]),
            ):
                if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                    raise PipelineCacheError(
                        f"output.{name}.{suffix}: cache supports numeric NumPy arrays"
                    )
                arrays[f"field_{index}_{suffix}"] = array
            fields.append(_field_metadata(result, name))
        state = {
            "kind": "erp",
            "fields": fields,
            "shadow_angle": float(result.shadow_angle),
            "shadow_padded": bool(result.shadow_padded),
            "lat": float(result.lat),
            "lon": float(result.lon),
            "preserve_sampler": getattr(result, "sampler", None)
            is getattr(source, "sampler", object()),
            "preserve_projection": getattr(result, "projection", None)
            is getattr(source, "projection", object()),
        }
    else:
        from ._cache_typed import encode_typed

        try:
            arrays, root = encode_typed(result)
        except (TypeError, ValueError) as exc:
            raise PipelineCacheError(str(exc)) from exc
        state = {"kind": "typed", "root": root}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue(), state


def store_cached_result(
    memory: Any, identity: dict[str, Any], result: Any, source: Any
) -> None:
    payload, state = _encode_result(result, source)
    entry = _entry_path(memory, identity["key"])
    if entry.exists():
        return
    parent = entry.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity['key']}.", dir=parent))
    manifest = {
        "format": CACHE_FORMAT,
        **identity,
        "payload_file": "result.npz",
        "payload_sha256": sha256(payload).hexdigest(),
        "state": state,
    }
    try:
        (temporary / "result.npz").write_bytes(payload)
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
        try:
            os.replace(temporary, entry)
        except OSError:
            if not entry.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_cached_result(
    memory: Any, identity: dict[str, Any], source: Any
) -> Any | None:
    entry = _entry_path(memory, identity["key"])
    if not entry.exists():
        return None
    manifest_path = entry / "manifest.json"
    payload_path = entry / "result.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineCacheError("cache entry is incomplete or unreadable") from exc
    expected_identity = {key: identity[key] for key in identity}
    actual_identity = {key: manifest.get(key) for key in identity}
    if manifest.get("format") != CACHE_FORMAT or actual_identity != expected_identity:
        raise PipelineCacheError("cache manifest identity mismatch")
    if manifest.get("payload_file") != "result.npz":
        raise PipelineCacheError("cache payload filename is invalid")
    if sha256(payload).hexdigest() != manifest.get("payload_sha256"):
        raise PipelineCacheError("cache payload checksum mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise PipelineCacheError("cache payload is not a safe NumPy archive") from exc
    state = manifest.get("state")
    if not isinstance(state, dict):
        raise PipelineCacheError("cache result state is missing")
    if state.get("kind") == "ndarray":
        if set(arrays) != {"value"}:
            raise PipelineCacheError("cached ndarray payload inventory mismatch")
        return arrays["value"]
    if state.get("kind") == "typed":
        if not isinstance(state.get("root"), dict):
            raise PipelineCacheError("typed cache state is missing")
        from ._cache_typed import decode_typed

        try:
            return decode_typed(arrays, state["root"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineCacheError(f"typed cache payload is invalid: {exc}") from exc
    if state.get("kind") != "erp" or not isinstance(state.get("fields"), list):
        raise PipelineCacheError("unknown cached result kind")
    from panorai.data import EquirectangularImage
    from panorai.data._workflow import clone_array
    from panorai.data.fields import RasterFieldSpec

    data = {}
    metadata = {}
    supports = {}
    expected_arrays = set()
    for index, field in enumerate(state["fields"]):
        if not isinstance(field, dict) or not isinstance(field.get("name"), str):
            raise PipelineCacheError("cached ERP field metadata is invalid")
        name = field["name"]
        value_key = f"field_{index}_values"
        support_key = f"field_{index}_support"
        validity_key = f"field_{index}_validity"
        expected_arrays.update((value_key, support_key, validity_key))
        try:
            data[name] = arrays[value_key]
            supports[name] = arrays[support_key]
            validity = arrays[validity_key]
        except KeyError as exc:
            raise PipelineCacheError(
                "cached ERP array inventory is incomplete"
            ) from exc
        field_metadata = field.get("metadata")
        if not isinstance(field_metadata, dict):
            raise PipelineCacheError("cached ERP semantic metadata is invalid")
        metadata[name] = dict(field_metadata)
        metadata[name]["validity"] = validity
        semantic_kind = metadata[name].get("field_kind", metadata[name].get("kind"))
        metadata[name]["field_spec"] = (
            RasterFieldSpec.categorical()
            if semantic_kind in {"labels", "mask"}
            else RasterFieldSpec.continuous(units=metadata[name].get("units"))
        )
    if set(arrays) != expected_arrays:
        raise PipelineCacheError("cached ERP array inventory mismatch")
    result = EquirectangularImage(
        data,
        shadow_angle=state.get("shadow_angle", 0.0),
        lat=state.get("lat", 0.0),
        lon=state.get("lon", 0.0),
        shadow_padded=state.get("shadow_padded", False),
    )
    result._workflow_metadata = metadata
    result._workflow_support = supports
    result.support_mask = clone_array(next(iter(supports.values())))
    if state.get("preserve_sampler") and hasattr(source, "sampler"):
        result.sampler = source.sampler
    if state.get("preserve_projection") and hasattr(source, "projection"):
        result.projection = source.projection
    return result


__all__ = [
    "PipelineCacheError",
    "cache_identity",
    "load_cached_result",
    "store_cached_result",
]
