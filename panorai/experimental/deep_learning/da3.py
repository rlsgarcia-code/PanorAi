"""Experimental, adapter-only Depth Anything 3 metric tangent inference.

PanorAi does not distribute Depth Anything 3 source or weights. Acquisition is
an explicit, checksum-pinned operation into a caller-controlled external cache.
The adapter keeps the official network unchanged and owns only fixed-geometry
tangent input, metric scaling, and axial-depth to radial-range conversion.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Iterator

import numpy as np

from panorai.geometry import GnomonicSpec, gnomonic_intrinsics

from .depth import ExternalArtifactRecord, ExternalArtifactSpec, _download, _sha256

SPHERICAL_DA3_METRIC_DEPTH_INTERFACE = (
    "panorai-depth-anything-3-metric-tangent/v1-experimental"
)
DA3_SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_CHECKPOINT_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
DA3_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
DA3_CANONICAL_FOCAL_PX = 300.0
DA3_PATCH_SIZE = 14
_DA3_LICENSE_SHA256 = "78446e29c48900cda82620a8df183cca61f0a595e05a49d0401a5fd604dd1870"

DA3_SOURCE = ExternalArtifactSpec(
    kind="source",
    url=(
        "https://github.com/ByteDance-Seed/Depth-Anything-3/archive/"
        f"{DA3_SOURCE_COMMIT}.tar.gz"
    ),
    filename=f"Depth-Anything-3-{DA3_SOURCE_COMMIT}.tar.gz",
    sha256="98aa2dd53ab44b96cef5190ae4841c6ae51797d099b784e08e12a1a55bd3a69b",
    size_bytes=23_650_987,
    license_statement="Depth Anything 3 source is Apache-2.0 at the pinned commit.",
    redistribution="external-only; PanorAi does not package the source archive",
)
DA3METRIC_LARGE = ExternalArtifactSpec(
    kind="checkpoint",
    url=(
        "https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/"
        f"{DA3_CHECKPOINT_REVISION}/model.safetensors"
    ),
    filename="da3metric-large-model.safetensors",
    sha256="bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
    size_bytes=1_336_734_448,
    license_statement=("The official DA3METRIC-LARGE model card declares Apache-2.0."),
    redistribution="external-only; PanorAi never redistributes this checkpoint",
)


@dataclass(frozen=True, slots=True)
class DA3Assets:
    """Verified official source tree and checkpoint used by the adapter."""

    source: ExternalArtifactRecord
    checkpoint: ExternalArtifactRecord
    source_root: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": SPHERICAL_DA3_METRIC_DEPTH_INTERFACE,
            "model_id": DA3_MODEL_ID,
            "source": self.source.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "source_root": self.source_root,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True, slots=True)
class LoadedDA3MetricModel:
    """Strictly loaded DA3Metric-Large network plus provenance."""

    model: Any
    assets: DA3Assets
    device: str
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def tangent(self, spec: GnomonicSpec) -> "DA3MetricTangent":
        """Bind the network to one fixed tangent-view intrinsic contract."""

        return DA3MetricTangent(self.model, spec, device=self.device)


class DA3TermsNotAcceptedError(PermissionError):
    """Raised before network/cache mutation when opt-in was omitted."""


def _default_cache_dir() -> Path:
    override = os.environ.get("PANORAI_CACHE_HOME")
    if override:
        return Path(override).expanduser() / "experimental" / "da3"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "panorai" / "experimental" / "da3"


def _safe_extract_da3(archive_path: Path, destination: Path) -> Path:
    source_name = f"Depth-Anything-3-{DA3_SOURCE_COMMIT}"
    source_root = destination / source_name
    marker = source_root / ".panorai-source.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source_sha256") == DA3_SOURCE.sha256:
            return source_root
        raise RuntimeError(f"source extraction marker does not match {source_root}")
    if source_root.exists():
        raise RuntimeError(f"unverified DA3 source directory exists: {source_root}")

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".da3-extract-", dir=destination))
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"unsafe DA3 archive member: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(f"unsupported DA3 archive member: {member.name}")
            archive.extractall(staging, filter="data")

        extracted = staging / source_name
        config = (
            extracted / "src" / "depth_anything_3" / "configs" / "da3metric-large.yaml"
        )
        license_path = extracted / "LICENSE"
        if not config.is_file() or not license_path.is_file():
            raise RuntimeError("DA3 archive lacks LICENSE or da3metric-large.yaml")
        if _sha256(license_path) != _DA3_LICENSE_SHA256:
            raise RuntimeError("DA3 source LICENSE checksum changed")

        os.replace(extracted, source_root)
        marker.write_text(
            json.dumps(
                {
                    "source_commit": DA3_SOURCE_COMMIT,
                    "source_sha256": DA3_SOURCE.sha256,
                    "license": "Apache-2.0",
                    "license_sha256": _DA3_LICENSE_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return source_root


def acquire_da3metric_large(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> DA3Assets:
    """Download and verify the exact official DA3Metric-Large artifacts.

    ``accept_upstream_terms=True`` is mandatory even when artifacts are already
    cached. It records caller intent; it is not a PanorAi license grant.
    """

    if not accept_upstream_terms:
        raise DA3TermsNotAcceptedError(
            "DA3 acquisition requires accept_upstream_terms=True after reviewing "
            "the Apache-2.0 source and model-card terms. PanorAi redistributes "
            "neither source nor checkpoint."
        )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")

    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    )
    downloads = root / "downloads"
    source_archive = downloads / DA3_SOURCE.filename
    checkpoint = downloads / DA3METRIC_LARGE.filename
    source_cached = _download(
        DA3_SOURCE, source_archive, timeout_seconds=timeout_seconds
    )
    checkpoint_cached = _download(
        DA3METRIC_LARGE, checkpoint, timeout_seconds=timeout_seconds
    )
    source_root = _safe_extract_da3(source_archive, root / "source")
    manifest = root / "da3metric-large.json"
    assets = DA3Assets(
        source=ExternalArtifactRecord(
            spec=DA3_SOURCE,
            path=str(source_archive),
            previously_cached=source_cached,
        ),
        checkpoint=ExternalArtifactRecord(
            spec=DA3METRIC_LARGE,
            path=str(checkpoint),
            previously_cached=checkpoint_cached,
        ),
        source_root=str(source_root),
        manifest_path=str(manifest),
    )
    manifest.write_text(
        json.dumps(assets.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assets


@contextmanager
def _upstream_import_path(source_root: Path) -> Iterator[None]:
    source_path = source_root / "src"
    if not source_path.is_dir():
        raise FileNotFoundError(f"DA3 source package not found: {source_path}")
    existing = sys.modules.get("depth_anything_3")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if source_path.resolve() not in existing_path.parents:
            raise RuntimeError(
                "depth_anything_3 is already imported from a different source tree"
            )
    value = str(source_path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive against upstream mutation
            pass


def _construct_official_da3metric_large(source_root: Path) -> Any:
    from torch import nn

    with _upstream_import_path(source_root):
        from depth_anything_3.model.da3 import DepthAnything3Net
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dpt import DPT

        backbone = DinoV2(
            name="vitl",
            out_layers=[4, 11, 17, 23],
            alt_start=-1,
            qknorm_start=-1,
            rope_start=-1,
            cat_token=False,
        )
        head = DPT(
            dim_in=1024,
            output_dim=1,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )

        class _CheckpointEnvelope(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = DepthAnything3Net(backbone, head)

        return _CheckpointEnvelope()


def load_da3metric_large(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    timeout_seconds: float = 120.0,
) -> LoadedDA3MetricModel:
    """Acquire and strictly load the official DA3Metric-Large network.

    The official high-level resizing API is intentionally bypassed. Network
    construction exactly mirrors the pinned ``da3metric-large.yaml``.
    """

    assets = acquire_da3metric_large(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    try:
        from safetensors.torch import load_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DA3 loading requires safetensors; install " "panorai[deep-learning-depth]"
        ) from exc

    envelope = _construct_official_da3metric_large(Path(assets.source_root))
    missing, unexpected = load_model(
        envelope,
        assets.checkpoint.path,
        strict=True,
        device="cpu",
    )
    missing_keys = tuple(sorted(missing))
    unexpected_keys = tuple(sorted(unexpected))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "DA3 checkpoint/model mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model = envelope.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedDA3MetricModel(
        model=model,
        assets=assets,
        device=device,
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def _axial_to_radial_range(
    axial_m: np.ndarray,
    *,
    fx: float,
    fy: float,
) -> np.ndarray:
    height, width = axial_m.shape
    x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / fx
    y = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / fy
    scale = np.sqrt(1.0 + y[:, None] ** 2 + x[None, :] ** 2)
    return np.asarray(axial_m * scale, dtype=np.float32)


class DA3MetricTangent:
    """Fixed-geometry HWC RGB adapter returning radial range in metres.

    The official metric rule scales canonical axial depth by the mean tangent
    focal length divided by 300. Pixel-centre rays then convert axial depth to
    PanorAi radial range. No resize, crop, padding, clipping, or smoothing is
    performed.
    """

    def __init__(self, model: Any, spec: GnomonicSpec, *, device: str = "cpu") -> None:
        height, width = spec.output_shape_hw
        if height % DA3_PATCH_SIZE or width % DA3_PATCH_SIZE:
            raise ValueError(
                "DA3 tangent dimensions must be divisible by patch size 14"
            )
        intrinsics = np.asarray(gnomonic_intrinsics(spec), dtype=np.float64)
        self.model = model
        self.spec = spec
        self.device = device
        self.fx = float(intrinsics[0, 0])
        self.fy = float(intrinsics[1, 1])
        self.metric_focal_px = (self.fx + self.fy) / 2.0

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        values = np.asarray(rgb)
        expected_shape = (*self.spec.output_shape_hw, 3)
        if values.shape != expected_shape:
            raise ValueError(
                f"tangent RGB must have shape {expected_shape}; got {values.shape}"
            )
        if values.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise TypeError("tangent RGB must be uint8 or float32 HWC data")
        if values.dtype == np.float32 and not np.isfinite(values).all():
            raise ValueError("tangent RGB must contain only finite values")
        if values.size and (float(values.min()) < 0.0 or float(values.max()) > 255.0):
            raise ValueError("tangent RGB values must be in the range [0, 255]")

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                values.astype(np.float32, copy=False).transpose(2, 0, 1)
            )
        )
        tensor = tensor / 255.0
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = ((tensor - mean) / std)[None, None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        if "depth" not in output:
            raise RuntimeError("DA3 output does not contain 'depth'")
        canonical = output["depth"]
        expected_output_shape = (1, 1, *self.spec.output_shape_hw)
        if tuple(canonical.shape) != expected_output_shape:
            raise RuntimeError(
                "DA3 depth output must preserve the fixed tangent lattice; "
                f"expected {expected_output_shape}, got {tuple(canonical.shape)}"
            )
        canonical_axial = canonical[0, 0].detach().float().cpu().numpy()
        axial_m = canonical_axial * (self.metric_focal_px / DA3_CANONICAL_FOCAL_PX)
        radial_m = _axial_to_radial_range(axial_m, fx=self.fx, fy=self.fy)
        valid = np.isfinite(radial_m) & (radial_m > 0.0)
        radial_m[~valid] = np.nan
        return radial_m, valid


__all__ = [
    "DA3Assets",
    "DA3MetricTangent",
    "DA3TermsNotAcceptedError",
    "DA3_CANONICAL_FOCAL_PX",
    "DA3_CHECKPOINT_REVISION",
    "DA3_MODEL_ID",
    "DA3_PATCH_SIZE",
    "DA3_SOURCE",
    "DA3_SOURCE_COMMIT",
    "DA3METRIC_LARGE",
    "LoadedDA3MetricModel",
    "SPHERICAL_DA3_METRIC_DEPTH_INTERFACE",
    "acquire_da3metric_large",
    "load_da3metric_large",
]
