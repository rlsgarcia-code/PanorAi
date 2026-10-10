"""Experimental direct-spherical InternImage/DCNv3 portability.

The adapter in this module does not redistribute InternImage source or weights.
It accepts the official Hugging Face model at a pinned revision, preserves its
learned parameters, replaces DCNv3 sampling by an ERP tangent-plane analogue,
and exposes an exact dense decomposition of the model's attention classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from panorai.image_processing.torch import SphericalPortReport, port_module_with_report


SPHERICAL_INTERNIMAGE_INTERFACE = "panorai-spherical-internimage/v1"
INTERNIMAGE_G_REPOSITORY = "OpenGVLab/internimage_g_22kto1k_512"
INTERNIMAGE_G_REVISION = "f5c00e40859f34b29951722e56646a89c64d21a4"


@dataclass(frozen=True, slots=True)
class InternImageShardSpec:
    """Pinned identity of one official InternImage-G safetensors shard."""

    filename: str
    size_bytes: int
    sha256: str


INTERNIMAGE_G_SHARDS = (
    InternImageShardSpec(
        "model-00001-of-00003.safetensors",
        4_997_736_288,
        "39d79d4f3c70bc9dabb9a7d66a9c5cd1c616f984e9291444c9759d86eeb287e9",
    ),
    InternImageShardSpec(
        "model-00002-of-00003.safetensors",
        4_753_657_088,
        "a139992cce454fd4752275f18114fa4e94c80baee493c06e1e4f22bce6f1a1f2",
    ),
    InternImageShardSpec(
        "model-00003-of-00003.safetensors",
        2_554_250_296,
        "f87cceddec3d3fbafa1fd629638a122250284a75258f7accd41b043e8db934cc",
    ),
)

_INTERNIMAGE_FILES = (
    "README.md",
    "config.json",
    "configuration_internimage.py",
    "dcnv3.py",
    "dcnv3_func.py",
    "model.safetensors.index.json",
    "modeling_internimage.py",
    "preprocessor_config.json",
    *(item.filename for item in INTERNIMAGE_G_SHARDS),
)


class InternImageTermsNotAcceptedError(PermissionError):
    """Raised before acquiring externally licensed InternImage assets."""


@dataclass(frozen=True, slots=True)
class InternImageGAssets:
    """Verified external InternImage-G snapshot and audit manifest."""

    snapshot_path: str
    repository: str
    revision: str
    shard_specs: tuple[InternImageShardSpec, ...]
    manifest_path: str
    previously_cached: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance record."""

        return {
            "schema": "panorai-internimage-assets/v1",
            "snapshot_path": self.snapshot_path,
            "repository": self.repository,
            "revision": self.revision,
            "shard_specs": [asdict(spec) for spec in self.shard_specs],
            "manifest_path": self.manifest_path,
            "previously_cached": self.previously_cached,
            "upstream_license": "MIT model card/repository; dataset terms remain upstream",
            "distribution_boundary": (
                "The source snapshot and 12.3 GB checkpoint remain in the external "
                "user cache and are not part of PanorAi source, wheels, or sdists."
            ),
        }


@dataclass(frozen=True, slots=True)
class LoadedInternImageG:
    """Official classifier, processor metadata, and verified external assets."""

    model: nn.Module
    processor: Any
    assets: InternImageGAssets
    categories: tuple[str, ...]
    image_size: int = 512


@dataclass(frozen=True, slots=True)
class DenseInternImageOutput:
    """Multiscale features and exact attention-class contribution lattice."""

    features: Tensor
    logits: Tensor
    global_logits: Tensor
    attention: Tensor


@dataclass(frozen=True, slots=True)
class SphericalDCNv3Record:
    """One official DCNv3 module replaced by spherical tangent sampling."""

    path: str
    source_type: str
    groups: int
    channels: int
    kernel_size: int
    parameter_identity_preserved: bool


@dataclass(frozen=True, slots=True)
class SphericalInternImagePortReport:
    """Auditable report for both DCNv3 and ordinary spatial layer ports."""

    interface: str
    dcnv3_layers: tuple[SphericalDCNv3Record, ...]
    spatial_layers: SphericalPortReport
    remaining_dcnv3_layers: tuple[str, ...]
    parameter_identity_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "interface": self.interface,
            "stability": "experimental",
            "dcnv3_layer_count": len(self.dcnv3_layers),
            "dcnv3_layers": [asdict(item) for item in self.dcnv3_layers],
            "spatial_layers": self.spatial_layers.to_dict(),
            "remaining_dcnv3_layers": list(self.remaining_dcnv3_layers),
            "parameter_identity_preserved": self.parameter_identity_preserved,
        }


def _default_cache_dir() -> Path:
    override = os.environ.get("PANORAI_CACHE_HOME")
    if override:
        return Path(override).expanduser() / "experimental" / "internimage-g"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "panorai" / "experimental" / "internimage-g"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_shards(snapshot: Path) -> None:
    for spec in INTERNIMAGE_G_SHARDS:
        path = snapshot / spec.filename
        if not path.is_file():
            raise RuntimeError(f"InternImage-G shard is missing: {path}")
        if path.stat().st_size != spec.size_bytes:
            raise RuntimeError(
                f"InternImage-G shard size mismatch for {path}: expected "
                f"{spec.size_bytes}, received {path.stat().st_size}"
            )
        digest = _sha256(path)
        if digest != spec.sha256:
            raise RuntimeError(
                f"InternImage-G SHA-256 mismatch for {path}: expected "
                f"{spec.sha256}, received {digest}"
            )


def acquire_internimage_g(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
) -> InternImageGAssets:
    """Download the pinned official snapshot and verify all weight shards."""

    if not accept_upstream_terms:
        raise InternImageTermsNotAcceptedError(
            "InternImage-G acquisition requires accept_upstream_terms=True after "
            "reviewing its MIT model card and the ImageNet/Joint-427M dataset terms"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "InternImage-G acquisition requires huggingface-hub; install "
            'PanorAi with `pip install "panorai[deep-learning-internimage]"`.'
        ) from error

    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    )
    snapshot_target = root / INTERNIMAGE_G_REVISION
    previously_cached = all(
        (snapshot_target / name).is_file() for name in _INTERNIMAGE_FILES
    )
    downloaded = Path(
        snapshot_download(
            repo_id=INTERNIMAGE_G_REPOSITORY,
            revision=INTERNIMAGE_G_REVISION,
            allow_patterns=list(_INTERNIMAGE_FILES),
            local_dir=snapshot_target,
        )
    )
    _verify_shards(downloaded)
    manifest_path = root / "manifest.json"
    assets = InternImageGAssets(
        snapshot_path=str(downloaded),
        repository=INTERNIMAGE_G_REPOSITORY,
        revision=INTERNIMAGE_G_REVISION,
        shard_specs=INTERNIMAGE_G_SHARDS,
        manifest_path=str(manifest_path),
        previously_cached=previously_cached,
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(assets.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assets


def load_pretrained_internimage_g(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    portable_core: bool = True,
) -> LoadedInternImageG:
    """Load the verified official 90.1%-top-1 InternImage-G classifier.

    By default the official pure-PyTorch DCNv3 reference core is selected so
    that the checkpoint can be audited on hosts without OpenGVLab's optional
    compiled CUDA extension.  It has the same parameters and planar sampling
    equation as the extension and is replaced in place by :class:`SphericalDCNv3`.
    """

    assets = acquire_internimage_g(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
    )
    try:
        from transformers import AutoModelForImageClassification, CLIPImageProcessor
        from torchvision.models import ResNet18_Weights
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "InternImage-G loading requires transformers, timm, safetensors, and "
            "torchvision; install panorai[deep-learning-internimage]"
        ) from error
    snapshot = assets.snapshot_path
    model_kwargs: dict[str, Any] = {}
    if portable_core:
        model_kwargs["core_op"] = "DCNv3_pytorch"
    model = AutoModelForImageClassification.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        **model_kwargs,
    ).eval()
    processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    categories = tuple(ResNet18_Weights.DEFAULT.meta["categories"])
    if len(categories) != 1000:
        raise RuntimeError("Torchvision ImageNet-1K category metadata is incomplete")
    return LoadedInternImageG(
        model=model,
        processor=processor,
        assets=assets,
        categories=categories,
    )


def _base_offsets(
    kernel_size: int,
    dilation: int,
    *,
    remove_center: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    extent = dilation * (kernel_size - 1) / 2
    horizontal = torch.linspace(
        -extent, extent, kernel_size, device=device, dtype=dtype
    )
    vertical = torch.linspace(-extent, extent, kernel_size, device=device, dtype=dtype)
    # Match the official DCNv3 flattening order: horizontal coordinate first.
    east = horizontal[:, None].expand(kernel_size, kernel_size).reshape(-1)
    south = vertical[None, :].expand(kernel_size, kernel_size).reshape(-1)
    offsets = torch.stack((east, south), dim=-1)
    if remove_center:
        center = offsets.shape[0] // 2
        offsets = torch.cat((offsets[:center], offsets[center + 1 :]), dim=0)
    return offsets


def _dynamic_tangent_grid(
    offsets: Tensor,
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    row_start: int,
    row_stop: int,
) -> Tensor:
    """Map per-group pixel offsets through each ERP cell's exponential map."""

    batch, _, output_width, groups, points, dimensions = offsets.shape
    if dimensions != 2 or output_width != output_shape[1]:
        raise ValueError("offset shape does not match the requested output lattice")
    input_height, input_width = input_shape
    output_height, _ = output_shape
    dtype = offsets.dtype
    device = offsets.device
    y = torch.arange(row_start, row_stop, device=device, dtype=dtype)[:, None]
    x = torch.arange(output_width, device=device, dtype=dtype)[None, :]
    longitude = ((x + 0.5) / output_width) * (2 * math.pi) - math.pi
    latitude = math.pi / 2 - ((y + 0.5) / output_height) * math.pi
    longitude = longitude.expand(row_stop - row_start, output_width)
    latitude = latitude.expand(row_stop - row_start, output_width)
    sin_lon, cos_lon = longitude.sin(), longitude.cos()
    sin_lat, cos_lat = latitude.sin(), latitude.cos()
    rays = torch.stack((sin_lon * cos_lat, sin_lat, cos_lon * cos_lat), dim=-1)
    east_basis = torch.stack((cos_lon, torch.zeros_like(cos_lon), -sin_lon), dim=-1)
    north_basis = torch.stack((-sin_lon * sin_lat, cos_lat, -cos_lon * sin_lat), dim=-1)

    east = offsets[..., 0] * (2 * math.pi / input_width)
    north = -offsets[..., 1] * (math.pi / input_height)
    radius = torch.hypot(east, north)
    sinc = torch.where(
        radius == 0,
        torch.ones_like(radius),
        torch.sinc(radius / math.pi),
    )
    tangent = (
        east[..., None] * east_basis[None, :, :, None, None]
        + north[..., None] * north_basis[None, :, :, None, None]
    )
    sample_rays = (
        radius.cos()[..., None] * rays[None, :, :, None, None]
        + sinc[..., None] * tangent
    )
    sample_rays = F.normalize(sample_rays, dim=-1)
    sample_lon = torch.atan2(sample_rays[..., 0], sample_rays[..., 2])
    sample_lat = torch.asin(sample_rays[..., 1].clamp(-1, 1))
    sample_x = (sample_lon + math.pi) / (2 * math.pi) * input_width - 0.5
    sample_y = (math.pi / 2 - sample_lat) / math.pi * input_height - 0.5
    extended_x = sample_x + 1
    normalized_x = 2 * extended_x / (input_width + 1) - 1
    normalized_y = (
        torch.zeros_like(sample_y)
        if input_height == 1
        else 2 * sample_y / (input_height - 1) - 1
    )
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    return grid.permute(0, 3, 1, 2, 4, 5).reshape(
        batch * groups, row_stop - row_start, output_width * points, 2
    )


def spherical_dcnv3_core(
    values: Tensor,
    offset: Tensor,
    mask: Tensor,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    group_channels: int,
    offset_scale: float,
    remove_center: bool = False,
    max_sampled_elements: int = 16_000_000,
) -> Tensor:
    """Apply DCNv3 offsets in local spherical tangent planes.

    ``values`` and the result use NHWC layout, matching the official module.
    The released InternImage-G uses same-size, stride-one DCNv3 blocks; other
    stride/padding configurations fail explicitly rather than silently changing
    their lattice semantics.
    """

    if values.ndim != 4 or offset.ndim != 4 or mask.ndim != 4:
        raise ValueError("values, offset, and mask must use NHWC layout")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    if stride != 1 or padding != dilation * (kernel_size - 1) // 2:
        raise ValueError(
            "spherical DCNv3 currently requires same-size stride-one blocks"
        )
    if values.shape[-1] != groups * group_channels:
        raise ValueError("values channels do not equal groups * group_channels")
    points = kernel_size * kernel_size - int(remove_center)
    expected_offset = groups * points * 2
    expected_mask = groups * points
    if offset.shape[-1] != expected_offset or mask.shape[-1] != expected_mask:
        raise ValueError("offset or mask channel count is inconsistent with DCNv3")
    if offset.shape[:3] != values.shape[:3] or mask.shape[:3] != values.shape[:3]:
        raise ValueError("released stride-one DCNv3 requires matching spatial shapes")
    if max_sampled_elements < 1:
        raise ValueError("max_sampled_elements must be positive")

    batch, height, width, _ = values.shape
    base = _base_offsets(
        kernel_size,
        dilation,
        remove_center=remove_center,
        device=values.device,
        dtype=values.dtype,
    )
    learned = offset.reshape(batch, height, width, groups, points, 2)
    combined = (learned + base[None, None, None, None]) * offset_scale
    weights = mask.reshape(batch, height, width, groups, points)
    grouped = values.reshape(batch, height, width, groups, group_channels)
    grouped = grouped.permute(0, 3, 4, 1, 2).reshape(
        batch * groups, group_channels, height, width
    )
    wrapped = torch.cat((grouped[..., -1:], grouped, grouped[..., :1]), dim=-1)
    elements_per_row = batch * groups * group_channels * width * points
    rows_per_chunk = max(1, max_sampled_elements // elements_per_row)
    outputs: list[Tensor] = []
    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(height, row_start + rows_per_chunk)
        grid = _dynamic_tangent_grid(
            combined[:, row_start:row_stop],
            input_shape=(height, width),
            output_shape=(height, width),
            row_start=row_start,
            row_stop=row_stop,
        )
        sampled = F.grid_sample(
            wrapped,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(
            batch,
            groups,
            group_channels,
            row_stop - row_start,
            width,
            points,
        )
        chunk_weights = weights[:, row_start:row_stop].permute(0, 3, 1, 2, 4)
        result = (sampled * chunk_weights[:, :, None]).sum(dim=-1)
        outputs.append(
            result.permute(0, 3, 4, 1, 2).reshape(
                batch, row_stop - row_start, width, groups * group_channels
            )
        )
    return torch.cat(outputs, dim=1)


class SphericalDCNv3(nn.Module):
    """Reuse one official DCNv3 module with spherical tangent sampling."""

    interface = SPHERICAL_INTERNIMAGE_INTERFACE
    stability = "experimental"

    def __init__(self, source: nn.Module, *, max_sampled_elements: int = 16_000_000):
        super().__init__()
        required = (
            "channels",
            "kernel_size",
            "stride",
            "pad",
            "dilation",
            "group",
            "group_channels",
            "offset_scale",
            "remove_center",
            "dw_conv",
            "offset",
            "mask",
            "input_proj",
            "output_proj",
        )
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(
                "source is not an official DCNv3 module; missing " + ", ".join(missing)
            )
        self.channels = int(source.channels)
        self.kernel_size = int(source.kernel_size)
        self.stride = int(source.stride)
        self.pad = int(source.pad)
        self.dilation = int(source.dilation)
        self.group = int(source.group)
        self.group_channels = int(source.group_channels)
        self.offset_scale = float(source.offset_scale)
        self.remove_center = bool(source.remove_center)
        self.max_sampled_elements = int(max_sampled_elements)
        self.dw_conv = source.dw_conv
        self.offset = source.offset
        self.mask = source.mask
        self.input_proj = source.input_proj
        self.output_proj = source.output_proj
        self.center_feature_scale = bool(source.center_feature_scale)
        if self.center_feature_scale:
            self.center_feature_scale_proj_weight = (
                source.center_feature_scale_proj_weight
            )
            self.center_feature_scale_proj_bias = source.center_feature_scale_proj_bias
            self.center_feature_scale_module = source.center_feature_scale_module

    def forward(self, values: Tensor) -> Tensor:
        projected = self.input_proj(values)
        center = projected
        context = self.dw_conv(values.permute(0, 3, 1, 2))
        offset = self.offset(context)
        batch, height, width, _ = values.shape
        mask = self.mask(context).reshape(batch, height, width, self.group, -1)
        mask = F.softmax(mask, dim=-1).reshape(batch, height, width, -1)
        result = spherical_dcnv3_core(
            projected,
            offset,
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad,
            dilation=self.dilation,
            groups=self.group,
            group_channels=self.group_channels,
            offset_scale=self.offset_scale,
            remove_center=self.remove_center,
            max_sampled_elements=self.max_sampled_elements,
        )
        if self.center_feature_scale:
            scale = self.center_feature_scale_module(
                context,
                self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias,
            )
            scale = (
                scale[..., None]
                .repeat(1, 1, 1, 1, self.channels // self.group)
                .flatten(-2)
            )
            result = result * (1 - scale) + center * scale
        return self.output_proj(result)


def _looks_like_dcnv3(module: nn.Module) -> bool:
    return type(module).__name__ in {"DCNv3", "DCNv3_pytorch"} and all(
        hasattr(module, name)
        for name in ("offset", "mask", "input_proj", "output_proj", "group_channels")
    )


def _replace_dcnv3(
    module: nn.Module,
    *,
    prefix: str,
    max_sampled_elements: int,
    records: list[SphericalDCNv3Record],
) -> None:
    for name, child in tuple(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if _looks_like_dcnv3(child):
            before = {id(parameter) for parameter in child.parameters()}
            replacement = SphericalDCNv3(
                child, max_sampled_elements=max_sampled_elements
            )
            after = {id(parameter) for parameter in replacement.parameters()}
            records.append(
                SphericalDCNv3Record(
                    path=path,
                    source_type=type(child).__name__,
                    groups=replacement.group,
                    channels=replacement.channels,
                    kernel_size=replacement.kernel_size,
                    parameter_identity_preserved=before == after,
                )
            )
            setattr(module, name, replacement)
        else:
            _replace_dcnv3(
                child,
                prefix=path,
                max_sampled_elements=max_sampled_elements,
                records=records,
            )


def port_internimage_to_spherical(
    model: nn.Module,
    *,
    max_sampled_elements: int = 16_000_000,
) -> SphericalInternImagePortReport:
    """Replace every official DCNv3 and conventional spatial layer in place."""

    if max_sampled_elements < 1:
        raise ValueError("max_sampled_elements must be positive")
    before = {id(parameter) for parameter in model.parameters()}
    records: list[SphericalDCNv3Record] = []
    _replace_dcnv3(
        model,
        prefix="",
        max_sampled_elements=max_sampled_elements,
        records=records,
    )
    spatial = port_module_with_report(model, max_sampled_elements=max_sampled_elements)
    remaining = tuple(
        name for name, child in model.named_modules() if _looks_like_dcnv3(child)
    )
    after = {id(parameter) for parameter in model.parameters()}
    return SphericalInternImagePortReport(
        interface=SPHERICAL_INTERNIMAGE_INTERFACE,
        dcnv3_layers=tuple(records),
        spatial_layers=spatial,
        remaining_dcnv3_layers=remaining,
        parameter_identity_preserved=before == after,
    )


def _single_layer_norm(module: nn.Module) -> nn.LayerNorm:
    layers = [child for child in module.modules() if isinstance(child, nn.LayerNorm)]
    if len(layers) != 1:
        raise TypeError("expected exactly one LayerNorm in the official fc_norm")
    return layers[0]


def attention_pool_class_contributions(
    tokens: Tensor,
    projector: nn.Module,
    fc_norm: nn.Module,
    head: nn.Linear,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dense logits whose spatial mean exactly equals official logits.

    InternImage-G uses one attention-pooled query followed by LayerNorm and a
    linear ImageNet head. For a fixed input LayerNorm is affine; distributing
    its centering, bias, and the linear-head bias across tokens yields an exact
    additive decomposition rather than an approximate Grad-CAM.
    """

    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError("tokens must have shape (N, T, C) with T positive")
    if projector.training or fc_norm.training or head.training:
        raise ValueError("exact dense decomposition requires evaluation mode")
    cross = projector.cross_dcn
    query = projector.norm1_q(tokens.mean(dim=1, keepdim=True))
    keys = projector.norm1_k(tokens)
    values = projector.norm1_v(tokens)
    batch, token_count, channels = tokens.shape
    heads = int(cross.num_heads)
    head_channels = channels // heads

    query = F.linear(query, cross.q.weight, cross.q_bias)
    keys = F.linear(keys, cross.k.weight, cross.k_bias)
    values = F.linear(values, cross.v.weight, cross.v_bias)
    query = query.reshape(batch, 1, heads, head_channels).permute(0, 2, 1, 3)
    keys = keys.reshape(batch, token_count, heads, head_channels).permute(0, 2, 1, 3)
    values = values.reshape(batch, token_count, heads, head_channels).permute(
        0, 2, 1, 3
    )
    attention = ((query * cross.scale) @ keys.transpose(-2, -1)).softmax(dim=-1)
    attention = cross.attn_drop(attention)
    local_heads = attention.squeeze(2)[..., None] * values
    local_concat = local_heads.permute(0, 2, 1, 3).reshape(batch, token_count, channels)
    local_projected = F.linear(local_concat, cross.proj.weight, None)
    if cross.proj.bias is not None:
        local_projected = local_projected + cross.proj.bias / token_count
    pooled = local_projected.sum(dim=1)

    norm = _single_layer_norm(fc_norm)
    mean = pooled.mean(dim=-1, keepdim=True)
    inverse_std = torch.rsqrt(
        (pooled - mean).square().mean(dim=-1, keepdim=True) + norm.eps
    )
    local_normalized = (local_projected - mean[:, None] / token_count) * (
        inverse_std[:, None]
    )
    if norm.weight is not None:
        local_normalized = local_normalized * norm.weight
    if norm.bias is not None:
        local_normalized = local_normalized + norm.bias / token_count
    local_logits = F.linear(local_normalized, head.weight, None)
    if head.bias is not None:
        local_logits = local_logits + head.bias / token_count
    global_logits = local_logits.sum(dim=1)
    # Scaling converts additive contributions to a density whose spatial mean
    # is the exact official classifier output, matching conventional CAM usage.
    return local_logits * token_count, global_logits, attention


class InternImageGDenseClassifier(nn.Module):
    """Expose exact dense ImageNet evidence from the official G classifier."""

    interface = SPHERICAL_INTERNIMAGE_INTERFACE
    stability = "experimental"

    def __init__(self, official_model: nn.Module) -> None:
        super().__init__()
        core = getattr(official_model, "model", official_model)
        if not hasattr(core, "forward_features_seq_out"):
            raise TypeError("official_model does not expose the InternImage backbone")
        required = (
            "dcnv3_head_x4",
            "dcnv3_head_x3",
            "clip_projector",
            "fc_norm",
            "head",
        )
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise TypeError(
                "InternImage-G classifier head is incomplete: " + ", ".join(missing)
            )
        self.core = core

    def forward_dense(self, values: Tensor) -> DenseInternImageOutput:
        features = self.core.forward_features_seq_out(values)
        x3 = features[2].permute(0, 3, 1, 2)
        x4 = features[3].permute(0, 3, 1, 2)
        fused = self.core.dcnv3_head_x4(x4) + self.core.dcnv3_head_x3(x3)
        height, width = fused.shape[-2:]
        tokens = fused.flatten(-2).transpose(1, 2).contiguous()
        dense, global_logits, attention = attention_pool_class_contributions(
            tokens,
            self.core.clip_projector,
            self.core.fc_norm,
            self.core.head,
        )
        logits = dense.transpose(1, 2).reshape(
            dense.shape[0], dense.shape[-1], height, width
        )
        return DenseInternImageOutput(
            features=fused,
            logits=logits,
            global_logits=global_logits,
            attention=attention,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.forward_dense(values).global_logits


__all__ = [
    "SPHERICAL_INTERNIMAGE_INTERFACE",
    "INTERNIMAGE_G_REPOSITORY",
    "INTERNIMAGE_G_REVISION",
    "INTERNIMAGE_G_SHARDS",
    "DenseInternImageOutput",
    "InternImageGAssets",
    "InternImageGDenseClassifier",
    "InternImageShardSpec",
    "InternImageTermsNotAcceptedError",
    "LoadedInternImageG",
    "SphericalDCNv3",
    "SphericalDCNv3Record",
    "SphericalInternImagePortReport",
    "acquire_internimage_g",
    "attention_pool_class_contributions",
    "load_pretrained_internimage_g",
    "port_internimage_to_spherical",
    "spherical_dcnv3_core",
]
