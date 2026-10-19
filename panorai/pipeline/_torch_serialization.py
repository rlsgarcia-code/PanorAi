"""Explicit registry for safe optional-Torch pipeline persistence."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any, Callable


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{2,127}$")


@dataclass(frozen=True, slots=True)
class _Registration:
    identifier: str
    module_type: type[Any]
    encode: Callable[[Any], dict[str, Any]]
    decode: Callable[[dict[str, Any]], Any]
    spherical_semantics: str | None


_BY_ID: dict[str, _Registration] = {}
_BY_TYPE: dict[type[Any], _Registration] = {}
_BUILTINS_READY = False


def register_torch_module(
    identifier: str,
    module_type: type[Any],
    *,
    encode: Callable[[Any], dict[str, Any]],
    decode: Callable[[dict[str, Any]], Any],
    spherical_semantics: str | None = None,
) -> None:
    """Register one architecture without enabling file-directed imports.

    Registration is process-local and explicit. Loading a custom architecture
    therefore requires trusted application code to register the same identifier
    before opening its archive. ``spherical_semantics='pointwise'`` is a trusted
    assertion that the custom module preserves every spatial sample independently;
    without it, a custom module cannot execute directly through ``OnSphere``.
    """

    if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(
            "Torch module identifiers must be 3-128 characters using letters, "
            "digits, dot, underscore, or hyphen"
        )
    if not isinstance(module_type, type):
        raise TypeError("module_type must be a class")
    if not callable(encode) or not callable(decode):
        raise TypeError("encode and decode must be callable")
    if spherical_semantics not in {None, "pointwise"}:
        raise ValueError("spherical_semantics must be None or 'pointwise'")
    existing_id = _BY_ID.get(identifier)
    existing_type = _BY_TYPE.get(module_type)
    if existing_id is not None and existing_id.module_type is not module_type:
        raise ValueError(
            f"Torch module identifier {identifier!r} is already registered"
        )
    if existing_type is not None and existing_type.identifier != identifier:
        raise ValueError(
            f"Torch module type {module_type.__qualname__} is already registered "
            f"as {existing_type.identifier!r}"
        )
    if spherical_semantics is not None:
        from panorai.ops.spherical import nn as spherical_nn

        spherical_nn._register_module_semantics(module_type, spherical_semantics)
    registration = _Registration(
        identifier, module_type, encode, decode, spherical_semantics
    )
    _BY_ID[identifier] = registration
    _BY_TYPE[module_type] = registration


def _dtype_name(module: Any) -> str | None:
    parameters = list(module.parameters())
    if not parameters:
        return None
    return str(parameters[0].dtype).removeprefix("torch.")


def _dtype(value: str | None) -> Any:
    if value is None:
        return None
    import torch

    result = getattr(torch, value, None)
    if result is None or not isinstance(result, torch.dtype):
        raise ValueError(f"unsupported Torch dtype {value!r}")
    return result


def _conv_parameters(module: Any, *, transpose: bool = False) -> dict[str, Any]:
    result = {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": tuple(module.kernel_size),
        "stride": tuple(module.stride),
        "padding": tuple(module.padding),
        "groups": module.groups,
        "bias": module.bias is not None,
        "dilation": tuple(module.dilation),
        "dtype": _dtype_name(module),
        "max_sampled_elements": module.max_sampled_elements,
    }
    if transpose:
        result["output_padding"] = tuple(module.output_padding)
    return result


def _register_builtins() -> None:
    global _BUILTINS_READY
    if _BUILTINS_READY:
        return
    import torch
    from panorai.ops.spherical import nn as spherical_nn

    register_torch_module(
        "torch.nn.Sequential",
        torch.nn.Sequential,
        encode=lambda module: {"modules": tuple(module.children())},
        decode=lambda values: torch.nn.Sequential(*values["modules"]),
    )
    register_torch_module(
        "torch.nn.ReLU",
        torch.nn.ReLU,
        encode=lambda module: {"inplace": bool(module.inplace)},
        decode=lambda values: torch.nn.ReLU(**values),
    )
    register_torch_module(
        "torch.nn.Identity",
        torch.nn.Identity,
        encode=lambda module: {},
        decode=lambda values: torch.nn.Identity(),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.Conv2d",
        spherical_nn.Conv2d,
        encode=lambda module: _conv_parameters(module),
        decode=lambda values: spherical_nn.Conv2d(
            values["in_channels"],
            values["out_channels"],
            values["kernel_size"],
            stride=values["stride"],
            padding=values["padding"],
            dilation=values["dilation"],
            groups=values["groups"],
            bias=values["bias"],
            dtype=_dtype(values["dtype"]),
            max_sampled_elements=values["max_sampled_elements"],
        ),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.ConvTranspose2d",
        spherical_nn.ConvTranspose2d,
        encode=lambda module: _conv_parameters(module, transpose=True),
        decode=lambda values: spherical_nn.ConvTranspose2d(
            values["in_channels"],
            values["out_channels"],
            values["kernel_size"],
            stride=values["stride"],
            padding=values["padding"],
            output_padding=values["output_padding"],
            groups=values["groups"],
            bias=values["bias"],
            dilation=values["dilation"],
            dtype=_dtype(values["dtype"]),
            max_sampled_elements=values["max_sampled_elements"],
        ),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.MaxPool2d",
        spherical_nn.MaxPool2d,
        encode=lambda module: {
            "kernel_size": tuple(module.kernel_size),
            "stride": tuple(module.stride),
            "padding": tuple(module.padding),
            "dilation": tuple(module.dilation),
            "ceil_mode": bool(module.ceil_mode),
            "invalid_policy": module.invalid_policy,
            "max_sampled_elements": module.max_sampled_elements,
        },
        decode=lambda values: spherical_nn.MaxPool2d(**values),
    )
    for name, module_type in (
        ("Closing", spherical_nn.Closing),
        ("Dilate", spherical_nn.Dilate),
        ("Erode", spherical_nn.Erode),
        ("Opening", spherical_nn.Opening),
    ):
        register_torch_module(
            f"panorai.ops.spherical.nn.{name}",
            module_type,
            encode=lambda module: {
                "footprint": module.footprint,
                "max_sampled_elements": module.max_sampled_elements,
            },
            decode=lambda values, module_type=module_type: module_type(**values),
        )
    register_torch_module(
        "panorai.ops.spherical.nn.AvgPool2d",
        spherical_nn.AvgPool2d,
        encode=lambda module: {
            "kernel_size": tuple(module.kernel_size),
            "stride": tuple(module.stride),
            "padding": tuple(module.padding),
            "dilation": tuple(module.dilation),
            "ceil_mode": bool(module.ceil_mode),
            "invalid_policy": module.invalid_policy,
            "min_valid_weight": module.min_valid_weight,
            "max_sampled_elements": module.max_sampled_elements,
        },
        decode=lambda values: spherical_nn.AvgPool2d(**values),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.Resize",
        spherical_nn.Resize,
        encode=lambda module: {
            "output_shape_hw": module.output_shape_hw,
            "scale_factor": module.scale_factor,
            "interpolation": module.interpolation,
        },
        decode=lambda values: spherical_nn.Resize(**values),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.Rotate",
        spherical_nn.Rotate,
        encode=lambda module: {
            "rotation": tuple(
                tuple(float(item) for item in row)
                for row in module.rotation_matrix.cpu().numpy()
            ),
            "output_shape_hw": module.output_shape_hw,
            "interpolation": module.interpolation,
        },
        decode=lambda values: spherical_nn.Rotate(**values),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.Interpolate",
        spherical_nn.Interpolate,
        encode=lambda module: {
            "size": module.output_shape_hw,
            "scale_factor": module.scale_factor,
            "mode": module.interpolation,
        },
        decode=lambda values: spherical_nn.Interpolate(**values),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.AdaptiveAvgPool2d",
        spherical_nn.AdaptiveAvgPool2d,
        encode=lambda module: {"output_size": tuple(module.output_size)},
        decode=lambda values: spherical_nn.AdaptiveAvgPool2d(**values),
    )
    register_torch_module(
        "panorai.ops.spherical.nn.GlobalAveragePool",
        spherical_nn.GlobalAveragePool,
        encode=lambda module: {
            "keepdim": module.keepdim,
            "invalid_policy": module.invalid_policy,
            "min_valid_weight": module.min_valid_weight,
        },
        decode=lambda values: spherical_nn.GlobalAveragePool(**values),
    )
    _BUILTINS_READY = True


def is_torch_module(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return bool(torch is not None and isinstance(value, torch.nn.Module))


def encode_torch_module(value: Any) -> tuple[str, dict[str, Any]]:
    _register_builtins()
    registration = _BY_TYPE.get(type(value))
    if registration is None:
        raise ValueError(
            f"unregistered Torch module {type(value).__module__}."
            f"{type(value).__qualname__}; call register_torch_module() from trusted "
            "application code before saving"
        )
    parameters = registration.encode(value)
    if not isinstance(parameters, dict):
        raise TypeError("registered Torch module encoder must return a dict")
    return registration.identifier, parameters


def decode_torch_module(identifier: str, parameters: dict[str, Any]) -> Any:
    _register_builtins()
    registration = _BY_ID.get(identifier)
    if registration is None:
        raise ValueError(
            f"unknown registered Torch module identifier {identifier!r}; "
            "custom architectures must be registered by trusted code before loading"
        )
    result = registration.decode(parameters)
    if type(result) is not registration.module_type:
        raise TypeError(
            f"Torch module decoder for {identifier!r} returned "
            f"{type(result).__qualname__}, expected {registration.module_type.__qualname__}"
        )
    return result


def registered_torch_modules() -> tuple[str, ...]:
    """Return identifiers accepted without importing classes from an archive."""

    _register_builtins()
    return tuple(sorted(_BY_ID))


__all__ = [
    "register_torch_module",
    "registered_torch_modules",
]
