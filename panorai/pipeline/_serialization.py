"""Safe registered-component persistence for Experimental pipelines."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np


CONFIG_FORMAT = "panorai-pipeline-config/v1"
ARCHIVE_FORMAT = "panorai-pipeline-archive/v1"
_TAG = "__panorai_type__"
_MAX_CONFIG_BYTES = 10 * 1024 * 1024
_MAX_INLINE_ARRAY_ELEMENTS = 1_000_000


class PipelineSerializationError(ValueError):
    """A pipeline cannot be represented or verified safely."""


def _registry() -> tuple[dict[type[Any], str], dict[str, type[Any]]]:
    from panorai.features import SphericalDoGDetectorConfig
    from panorai.ops.spherical import (
        AreaAverage,
        AreaHistogram,
        BilateralFilter,
        BoxBlur,
        Canny,
        DoGDetector,
        Filter2D,
        GaussianBlur,
        GaussianPyramid,
        Gradient,
        Laplacian,
        LaplacianPyramid,
        MedianBlur,
        Resample,
        Resize,
        Rotate,
        SLIC,
        Sobel,
        SphericalNMS,
    )

    from ._fields import Output
    from ._routing import Attach, Parallel, Select
    from ._stages import (
        AggregateViews,
        LiftToBearings,
        OnSphere,
        PerView,
        PerViewParallel,
        ReconstructERP,
        SampleViews,
    )

    entries = {
        SphericalDoGDetectorConfig: "panorai.features.SphericalDoGDetectorConfig",
        AreaAverage: "panorai.ops.spherical.AreaAverage",
        AreaHistogram: "panorai.ops.spherical.AreaHistogram",
        BilateralFilter: "panorai.ops.spherical.BilateralFilter",
        BoxBlur: "panorai.ops.spherical.BoxBlur",
        Canny: "panorai.ops.spherical.Canny",
        DoGDetector: "panorai.ops.spherical.DoGDetector",
        Filter2D: "panorai.ops.spherical.Filter2D",
        GaussianBlur: "panorai.ops.spherical.GaussianBlur",
        GaussianPyramid: "panorai.ops.spherical.GaussianPyramid",
        Gradient: "panorai.ops.spherical.Gradient",
        Laplacian: "panorai.ops.spherical.Laplacian",
        LaplacianPyramid: "panorai.ops.spherical.LaplacianPyramid",
        MedianBlur: "panorai.ops.spherical.MedianBlur",
        Resample: "panorai.ops.spherical.Resample",
        Resize: "panorai.ops.spherical.Resize",
        Rotate: "panorai.ops.spherical.Rotate",
        SLIC: "panorai.ops.spherical.SLIC",
        Sobel: "panorai.ops.spherical.Sobel",
        SphericalNMS: "panorai.ops.spherical.SphericalNMS",
        Output: "panorai.pipeline.Output",
        OnSphere: "panorai.pipeline.OnSphere",
        SampleViews: "panorai.pipeline.SampleViews",
        PerView: "panorai.pipeline.PerView",
        PerViewParallel: "panorai.pipeline.PerViewParallel",
        ReconstructERP: "panorai.pipeline.ReconstructERP",
        LiftToBearings: "panorai.pipeline.LiftToBearings",
        AggregateViews: "panorai.pipeline.AggregateViews",
        Parallel: "panorai.pipeline.Parallel",
        Select: "panorai.pipeline.Select",
        Attach: "panorai.pipeline.Attach",
    }
    reverse = {identifier: component for component, identifier in entries.items()}
    return entries, reverse


def registered_pipeline_components() -> tuple[str, ...]:
    """Return the exact safe component identifiers accepted by the codec."""

    return tuple(sorted(_registry()[1]))


def _component_parameters(value: Any) -> dict[str, Any]:
    from ._routing import Attach, Parallel
    from ._stages import PerViewParallel, ReconstructERP

    if isinstance(value, Parallel):
        return {"branches": dict(value.branches)}
    if isinstance(value, Attach):
        return {
            "target": value.target,
            "replace": value.replace,
            "fields": dict(value.fields),
        }
    if isinstance(value, PerViewParallel):
        return {
            "operations": dict(value.operations),
            "input": value.input,
            "replace": value.replace,
            "batch": value.batch,
            "max_batch_elements": value.max_batch_elements,
        }
    if isinstance(value, ReconstructERP):
        return {
            "blend": value.blend,
            "modalities": value.modalities,
            "eq_shape": value.eq_shape,
            "field_blends": dict(value.field_blends),
        }
    if is_dataclass(value):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    raise PipelineSerializationError(
        f"registered component {type(value).__name__} has no parameter codec"
    )


def _encode(
    value: Any,
    *,
    path: str,
    torch_modules: list[tuple[str, Any, str]] | None = None,
) -> Any:
    from ._pipeline import Pipeline
    from ._torch_serialization import encode_torch_module, is_torch_module

    by_type, _ = _registry()
    component_id = by_type.get(type(value))
    if component_id is not None:
        parameters = _component_parameters(value)
        return {
            _TAG: "component",
            "id": component_id,
            "parameters": {
                name: _encode(
                    item,
                    path=f"{path}.{name}",
                    torch_modules=torch_modules,
                )
                for name, item in parameters.items()
            },
        }
    if is_torch_module(value):
        try:
            identifier, parameters = encode_torch_module(value)
        except (TypeError, ValueError) as exc:
            raise PipelineSerializationError(f"{path}: {exc}") from exc
        state_id = f"module_{0 if torch_modules is None else len(torch_modules):06d}"
        if torch_modules is not None:
            torch_modules.append((state_id, value, identifier))
        return {
            _TAG: "torch_module",
            "id": identifier,
            "state_id": state_id,
            "parameters": {
                name: _encode(
                    item,
                    path=f"{path}.{name}",
                    torch_modules=torch_modules,
                )
                for name, item in parameters.items()
            },
        }
    if isinstance(value, Pipeline):
        return {
            _TAG: "pipeline",
            "config": pipeline_to_config(value, _torch_modules=torch_modules),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biuf":
            raise PipelineSerializationError(
                f"{path}: only boolean, integer, and floating arrays are safe "
                "pipeline parameters"
            )
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise PipelineSerializationError(
                f"{path}: non-finite array values are forbidden"
            )
        if value.size > _MAX_INLINE_ARRAY_ELEMENTS:
            raise PipelineSerializationError(
                f"{path}: inline array exceeds {_MAX_INLINE_ARRAY_ELEMENTS} elements"
            )
        return {
            _TAG: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, tuple):
        return {
            _TAG: "tuple",
            "items": [
                _encode(
                    item,
                    path=f"{path}[{index}]",
                    torch_modules=torch_modules,
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (dict, MappingProxyType)):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineSerializationError(
                    f"{path}: mapping keys must be strings"
                )
            if key == _TAG:
                raise PipelineSerializationError(
                    f"{path}: mapping key {_TAG!r} is reserved by the safe codec"
                )
            result[key] = _encode(
                item,
                path=f"{path}.{key}",
                torch_modules=torch_modules,
            )
        return result
    if isinstance(value, list):
        return [
            _encode(
                item,
                path=f"{path}[{index}]",
                torch_modules=torch_modules,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return _encode(value.item(), path=path, torch_modules=torch_modules)
    if isinstance(value, float) and not math.isfinite(value):
        raise PipelineSerializationError(f"{path}: non-finite floats are forbidden")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PipelineSerializationError(
        f"{path}: unregistered component {type(value).__module__}."
        f"{type(value).__qualname__}; arbitrary callables and pickle are not "
        "serializable"
    )


def _exact_keys(value: dict[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PipelineSerializationError(
            f"{path}: expected keys {sorted(expected)}, got {sorted(actual)}"
        )


def _construct(
    component_type: type[Any], parameters: dict[str, Any], *, path: str
) -> Any:
    from ._routing import Attach, Parallel
    from ._stages import PerViewParallel, ReconstructERP

    try:
        if component_type is Parallel:
            _exact_keys(parameters, {"branches"}, path=f"{path}.parameters")
            return Parallel(**parameters["branches"])
        if component_type is Attach:
            _exact_keys(
                parameters,
                {"target", "replace", "fields"},
                path=f"{path}.parameters",
            )
            return Attach(
                target=parameters["target"],
                replace=parameters["replace"],
                **parameters["fields"],
            )
        if component_type is PerViewParallel:
            _exact_keys(
                parameters,
                {
                    "operations",
                    "input",
                    "replace",
                    "batch",
                    "max_batch_elements",
                },
                path=f"{path}.parameters",
            )
            return PerViewParallel(
                parameters["operations"],
                input=parameters["input"],
                replace=parameters["replace"],
                batch=parameters["batch"],
                max_batch_elements=parameters["max_batch_elements"],
            )
        if component_type is ReconstructERP:
            _exact_keys(
                parameters,
                {"blend", "modalities", "eq_shape", "field_blends"},
                path=f"{path}.parameters",
            )
            field_blends = parameters.pop("field_blends")
            if field_blends:
                return ReconstructERP(
                    eq_shape=parameters["eq_shape"],
                    **field_blends,
                )
            return ReconstructERP(**parameters)
        if not is_dataclass(component_type):
            raise PipelineSerializationError(
                f"{path}: registered component has no safe constructor"
            )
        expected = {item.name for item in fields(component_type)}
        _exact_keys(parameters, expected, path=f"{path}.parameters")
        return component_type(**parameters)
    except PipelineSerializationError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise PipelineSerializationError(
            f"{path}: invalid parameters for registered component "
            f"{component_type.__name__}: {exc}"
        ) from exc


def _decode(
    value: Any,
    *,
    path: str,
    torch_states: dict[str, dict[str, Any]] | None = None,
) -> Any:
    if isinstance(value, list):
        return [
            _decode(
                item,
                path=f"{path}[{index}]",
                torch_states=torch_states,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        if isinstance(value, float) and not math.isfinite(value):
            raise PipelineSerializationError(f"{path}: non-finite floats are forbidden")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise PipelineSerializationError(f"{path}: unsupported JSON value")
    tag = value.get(_TAG)
    if tag is None:
        if any(not isinstance(key, str) for key in value):
            raise PipelineSerializationError(f"{path}: mapping keys must be strings")
        return {
            key: _decode(
                item,
                path=f"{path}.{key}",
                torch_states=torch_states,
            )
            for key, item in value.items()
        }
    if tag == "tuple":
        _exact_keys(value, {_TAG, "items"}, path=path)
        if not isinstance(value["items"], list):
            raise PipelineSerializationError(f"{path}.items must be a list")
        return tuple(
            _decode(
                item,
                path=f"{path}.items[{index}]",
                torch_states=torch_states,
            )
            for index, item in enumerate(value["items"])
        )
    if tag == "ndarray":
        _exact_keys(value, {_TAG, "dtype", "shape", "values"}, path=path)
        try:
            dtype = np.dtype(value["dtype"])
        except TypeError as exc:
            raise PipelineSerializationError(f"{path}: invalid NumPy dtype") from exc
        if dtype.hasobject or dtype.kind not in "biuf":
            raise PipelineSerializationError(
                f"{path}: unsafe inline array dtype {dtype}"
            )
        array = np.asarray(value["values"], dtype=dtype)
        if array.size > _MAX_INLINE_ARRAY_ELEMENTS:
            raise PipelineSerializationError(f"{path}: inline array is too large")
        if list(array.shape) != value["shape"]:
            raise PipelineSerializationError(f"{path}: array shape metadata mismatch")
        if dtype.kind == "f" and not np.isfinite(array).all():
            raise PipelineSerializationError(
                f"{path}: non-finite array values are forbidden"
            )
        return array
    if tag == "pipeline":
        _exact_keys(value, {_TAG, "config"}, path=path)
        return pipeline_from_config(
            value["config"],
            path=f"{path}.config",
            _torch_states=torch_states,
        )
    if tag == "torch_module":
        _exact_keys(
            value,
            {_TAG, "id", "state_id", "parameters"},
            path=path,
        )
        if torch_states is None:
            raise PipelineSerializationError(
                f"{path}: registered Torch modules require a safetensors pipeline "
                "archive; topology JSON alone has no authenticated weights"
            )
        identifier = value["id"]
        state_id = value["state_id"]
        if (
            not isinstance(identifier, str)
            or not isinstance(state_id, str)
            or not isinstance(value["parameters"], dict)
        ):
            raise PipelineSerializationError(
                f"{path}: invalid registered Torch module record"
            )
        parameters = {
            name: _decode(
                item,
                path=f"{path}.parameters.{name}",
                torch_states=torch_states,
            )
            for name, item in value["parameters"].items()
        }
        from ._torch_serialization import decode_torch_module

        try:
            module = decode_torch_module(identifier, parameters)
            state = torch_states[state_id]
            incompatible = module.load_state_dict(state, strict=False)
        except KeyError as exc:
            raise PipelineSerializationError(
                f"{path}: safetensors state {state_id!r} is missing"
            ) from exc
        except (TypeError, ValueError, RuntimeError) as exc:
            raise PipelineSerializationError(
                f"{path}: invalid Torch module {identifier!r}: {exc}"
            ) from exc
        allowed_missing = {name for name in module.state_dict() if "." in name}
        if (
            set(incompatible.missing_keys) != allowed_missing
            or incompatible.unexpected_keys
        ):
            raise PipelineSerializationError(
                f"{path}: Torch state does not match registered architecture"
            )
        module.eval()
        return module
    if tag == "component":
        _exact_keys(value, {_TAG, "id", "parameters"}, path=path)
        _, by_id = _registry()
        identifier = value["id"]
        if not isinstance(identifier, str) or identifier not in by_id:
            raise PipelineSerializationError(
                f"{path}: unknown or unregistered component id {identifier!r}"
            )
        if not isinstance(value["parameters"], dict):
            raise PipelineSerializationError(f"{path}.parameters must be a mapping")
        parameters = {
            name: _decode(
                item,
                path=f"{path}.parameters.{name}",
                torch_states=torch_states,
            )
            for name, item in value["parameters"].items()
        }
        return _construct(by_id[identifier], parameters, path=path)
    raise PipelineSerializationError(f"{path}: unknown serialized tag {tag!r}")


def pipeline_to_config(
    pipeline: Any,
    *,
    _torch_modules: list[tuple[str, Any, str]] | None = None,
) -> dict[str, Any]:
    from ._pipeline import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline_to_config requires a Pipeline")
    torch_modules = [] if _torch_modules is None else _torch_modules
    return {
        "format": CONFIG_FORMAT,
        "interface": pipeline.interface,
        "steps": [
            {
                "name": name,
                "stage": _encode(
                    stage,
                    path=f"steps.{name}",
                    torch_modules=torch_modules,
                ),
            }
            for name, stage in pipeline.named_steps.items()
        ],
    }


def pipeline_from_config(
    config: Any,
    *,
    path: str = "config",
    _torch_states: dict[str, dict[str, Any]] | None = None,
) -> Any:
    from ._pipeline import Pipeline

    if not isinstance(config, dict):
        raise PipelineSerializationError(f"{path} must be a mapping")
    _exact_keys(config, {"format", "interface", "steps"}, path=path)
    if config["format"] != CONFIG_FORMAT:
        raise PipelineSerializationError(f"{path}: unsupported pipeline config format")
    if config["interface"] != Pipeline.interface:
        raise PipelineSerializationError(f"{path}: incompatible pipeline interface")
    if not isinstance(config["steps"], list) or not config["steps"]:
        raise PipelineSerializationError(f"{path}.steps must be a non-empty list")
    steps = []
    for index, item in enumerate(config["steps"]):
        item_path = f"{path}.steps[{index}]"
        if not isinstance(item, dict):
            raise PipelineSerializationError(f"{item_path} must be a mapping")
        _exact_keys(item, {"name", "stage"}, path=item_path)
        if not isinstance(item["name"], str):
            raise PipelineSerializationError(f"{item_path}.name must be a string")
        stage = _decode(
            item["stage"],
            path=f"{item_path}.stage",
            torch_states=_torch_states,
        )
        steps.append((item["name"], stage))
    try:
        return Pipeline(*steps)
    except (TypeError, ValueError) as exc:
        raise PipelineSerializationError(
            f"{path}: invalid pipeline topology: {exc}"
        ) from exc


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PipelineSerializationError(
            f"pipeline config is not canonical JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PipelineSerializationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise PipelineSerializationError(f"{name} contains non-finite {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except PipelineSerializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineSerializationError(f"{name} is not readable JSON") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _torch_weight_payload(
    modules: list[tuple[str, Any, str]],
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        import torch
        from safetensors.torch import save as save_safetensors
    except ImportError as exc:
        raise PipelineSerializationError(
            "registered Torch module persistence requires "
            "`pip install 'panorai[torch]'`"
        ) from exc

    tensors = {}
    inventory = []
    for state_id, module, identifier in modules:
        if bool(module.training):
            raise PipelineSerializationError(
                f"{state_id}: Torch modules must be in eval mode before saving "
                "or content-addressed caching"
            )
        state_keys = []
        for name, tensor in module.state_dict().items():
            if "." in name:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise PipelineSerializationError(
                    f"{state_id}.{name}: state_dict values must be tensors"
                )
            if tensor.layout != torch.strided:
                raise PipelineSerializationError(
                    f"{state_id}.{name}: sparse or non-strided tensors are unsupported"
                )
            key = f"{state_id}.{name}"
            tensors[key] = tensor.detach().cpu().contiguous().clone()
            state_keys.append(name)
        inventory.append(
            {
                "state_id": state_id,
                "identifier": identifier,
                "state_keys": state_keys,
            }
        )
    try:
        return save_safetensors(tensors), inventory
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PipelineSerializationError(
            f"could not encode registered Torch state safely: {exc}"
        ) from exc


def pipeline_torch_state_sha256(pipeline: Any) -> str | None:
    """Return the safe registered-weight identity used by pipeline cache keys."""

    modules: list[tuple[str, Any, str]] = []
    pipeline_to_config(pipeline, _torch_modules=modules)
    if not modules:
        return None
    payload, _ = _torch_weight_payload(modules)
    return sha256(payload).hexdigest()


def save_pipeline(
    pipeline: Any, directory: str | Path, *, overwrite: bool = False
) -> Path:
    root = Path(directory)
    if root.exists() and not root.is_dir():
        raise PipelineSerializationError("pipeline archive path must be a directory")
    if root.exists():
        contents = list(root.iterdir())
        allowed = {"pipeline.json", "manifest.json", "weights.safetensors"}
        unknown = [item.name for item in contents if item.name not in allowed]
        if unknown:
            raise PipelineSerializationError(
                f"refusing to write into a directory with unrelated files: {sorted(unknown)}"
            )
        if contents and not overwrite:
            raise FileExistsError(f"pipeline archive already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    torch_modules: list[tuple[str, Any, str]] = []
    config = pipeline_to_config(pipeline, _torch_modules=torch_modules)
    config_payload = _canonical_json(config)
    components = sorted(
        {
            item["id"]
            for step in config["steps"]
            for item in _walk_dicts(step["stage"])
            if item.get(_TAG) == "component"
        }
    )
    from panorai import __version__

    manifest = {
        "format": ARCHIVE_FORMAT,
        "config_file": "pipeline.json",
        "config_sha256": sha256(config_payload).hexdigest(),
        "panorai_version": __version__,
        "pipeline_interface": pipeline.interface,
        "registered_components": components,
    }
    if torch_modules:
        weights_payload, torch_inventory = _torch_weight_payload(torch_modules)
        manifest.update(
            {
                "weights_file": "weights.safetensors",
                "weights_sha256": sha256(weights_payload).hexdigest(),
                "registered_torch_modules": torch_inventory,
            }
        )
        _atomic_write(root / "weights.safetensors", weights_payload)
    _atomic_write(root / "pipeline.json", config_payload)
    _atomic_write(root / "manifest.json", _canonical_json(manifest))
    return root


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def _read_limited(path: Path, *, max_bytes: int = _MAX_CONFIG_BYTES) -> bytes:
    if not path.is_file():
        raise PipelineSerializationError(f"pipeline archive is missing {path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise PipelineSerializationError(f"{path.name} exceeds the safe size limit")
    return path.read_bytes()


def load_pipeline(directory: str | Path) -> Any:
    root = Path(directory)
    try:
        manifest_payload = _read_limited(root / "manifest.json")
        manifest = _load_json(manifest_payload, name="manifest.json")
    except OSError as exc:
        raise PipelineSerializationError("manifest.json is not readable JSON") from exc
    if not isinstance(manifest, dict):
        raise PipelineSerializationError("manifest.json must contain a mapping")
    ordinary_keys = {
        "format",
        "config_file",
        "config_sha256",
        "panorai_version",
        "pipeline_interface",
        "registered_components",
    }
    torch_keys = {
        "weights_file",
        "weights_sha256",
        "registered_torch_modules",
    }
    actual_keys = set(manifest)
    if actual_keys == ordinary_keys:
        has_torch_weights = False
    elif actual_keys == ordinary_keys | torch_keys:
        has_torch_weights = True
    else:
        raise PipelineSerializationError(
            f"manifest keys differ from the accepted archive schema: "
            f"{sorted(actual_keys)}"
        )
    if manifest["format"] != ARCHIVE_FORMAT:
        raise PipelineSerializationError("unsupported pipeline archive format")
    if manifest["config_file"] != "pipeline.json":
        raise PipelineSerializationError("manifest config_file must be pipeline.json")
    config_payload = _read_limited(root / "pipeline.json")
    if sha256(config_payload).hexdigest() != manifest["config_sha256"]:
        raise PipelineSerializationError("pipeline.json checksum mismatch")
    config = _load_json(config_payload, name="pipeline.json")
    torch_records = [
        item
        for step in config["steps"]
        for item in _walk_dicts(step["stage"])
        if item.get(_TAG) == "torch_module"
    ]
    torch_states = None
    if torch_records:
        if not has_torch_weights:
            raise PipelineSerializationError(
                "pipeline topology contains registered Torch modules but the "
                "safetensors manifest is missing"
            )
        if manifest["weights_file"] != "weights.safetensors":
            raise PipelineSerializationError(
                "manifest weights_file must be weights.safetensors"
            )
        weights_payload = _read_limited(
            root / "weights.safetensors",
            max_bytes=4 * 1024 * 1024 * 1024,
        )
        if sha256(weights_payload).hexdigest() != manifest["weights_sha256"]:
            raise PipelineSerializationError("weights.safetensors checksum mismatch")
        try:
            from safetensors.torch import load as load_safetensors

            flat_states = load_safetensors(weights_payload)
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            raise PipelineSerializationError(
                f"weights.safetensors is not a readable safe tensor archive: {exc}"
            ) from exc
        expected_inventory = manifest["registered_torch_modules"]
        if not isinstance(expected_inventory, list):
            raise PipelineSerializationError(
                "manifest registered_torch_modules must be a list"
            )
        for index, item in enumerate(expected_inventory):
            if not isinstance(item, dict):
                raise PipelineSerializationError(
                    f"manifest.registered_torch_modules[{index}] must be a mapping"
                )
            _exact_keys(
                item,
                {"state_id", "identifier", "state_keys"},
                path=f"manifest.registered_torch_modules[{index}]",
            )
            if (
                not isinstance(item["state_id"], str)
                or not isinstance(item["identifier"], str)
                or not isinstance(item["state_keys"], list)
                or any(not isinstance(name, str) for name in item["state_keys"])
            ):
                raise PipelineSerializationError(
                    f"manifest.registered_torch_modules[{index}] has invalid types"
                )
        state_ids = [item["state_id"] for item in expected_inventory]
        if len(set(state_ids)) != len(state_ids):
            raise PipelineSerializationError(
                "manifest registered Torch state identifiers must be unique"
            )
        actual_inventory = [
            {
                "state_id": item["state_id"],
                "identifier": item["id"],
                "state_keys": next(
                    (
                        record["state_keys"]
                        for record in expected_inventory
                        if record.get("state_id") == item["state_id"]
                    ),
                    None,
                ),
            }
            for item in torch_records
        ]
        if (
            any(item["state_keys"] is None for item in actual_inventory)
            or actual_inventory != expected_inventory
        ):
            raise PipelineSerializationError(
                "manifest registered Torch module inventory mismatch"
            )
        expected_flat_keys = {
            f"{item['state_id']}.{name}"
            for item in expected_inventory
            for name in item["state_keys"]
        }
        if set(flat_states) != expected_flat_keys:
            raise PipelineSerializationError(
                "safetensors state-key inventory does not match the manifest"
            )
        torch_states = {
            item["state_id"]: {
                name: flat_states[f"{item['state_id']}.{name}"]
                for name in item["state_keys"]
            }
            for item in expected_inventory
        }
    elif has_torch_weights:
        raise PipelineSerializationError(
            "manifest declares Torch weights but pipeline topology has no Torch modules"
        )
    pipeline = pipeline_from_config(config, _torch_states=torch_states)
    actual_components = sorted(
        {
            item["id"]
            for step in config["steps"]
            for item in _walk_dicts(step["stage"])
            if item.get(_TAG) == "component"
        }
    )
    if manifest["registered_components"] != actual_components:
        raise PipelineSerializationError("manifest component inventory mismatch")
    if manifest["pipeline_interface"] != pipeline.interface:
        raise PipelineSerializationError("manifest pipeline interface mismatch")
    return pipeline


__all__ = [
    "PipelineSerializationError",
    "load_pipeline",
    "registered_pipeline_components",
    "save_pipeline",
]
