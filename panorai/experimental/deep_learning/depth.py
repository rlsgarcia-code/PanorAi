"""Experimental, adapter-only spherical monocular metric depth.

PanorAi does not distribute Metric3D source or weights. Acquisition is an
explicit, checksum-pinned operation into a caller-controlled external cache.
The checkpoint repository does not publish separate weight-license terms, so
callers must review and accept the upstream terms before any network access.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import types
from typing import Any, Iterator
from urllib.request import Request, urlopen


SPHERICAL_METRIC_DEPTH_INTERFACE = "panorai-spherical-metric-depth/v1-experimental"
METRIC3D_SOURCE_COMMIT = "fd90d0dad05d913c4f45ed7a9789b8f1dfde22e0"
METRIC3D_CHECKPOINT_REVISION = "80d2d1410afb4b23cd9d18c6be9144483d4b70b6"


@dataclass(frozen=True, slots=True)
class ExternalArtifactSpec:
    """Immutable identity and legal boundary for one external artifact."""

    kind: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    license_statement: str
    redistribution: str


METRIC3D_SOURCE = ExternalArtifactSpec(
    kind="source",
    url=(
        f"https://github.com/YvanYin/Metric3D/archive/{METRIC3D_SOURCE_COMMIT}.tar.gz"
    ),
    filename=f"Metric3D-{METRIC3D_SOURCE_COMMIT}.tar.gz",
    sha256="929432fd1f7d1f2c45f51407e4ef30ca5758eb207b15e51c9b480048176ee365",
    size_bytes=49_543_110,
    license_statement="Metric3D source is BSD-2-Clause at the pinned commit.",
    redistribution="external-only; PanorAi does not package the source archive",
)
METRIC3D_CONVNEXT_TINY_V1 = ExternalArtifactSpec(
    kind="checkpoint",
    url=(
        "https://huggingface.co/JUGGHM/Metric3D/resolve/"
        f"{METRIC3D_CHECKPOINT_REVISION}/convtiny_hourglass_v1.pth"
    ),
    filename="convtiny_hourglass_v1.pth",
    sha256="bc41f5f919bb0388bbc88fe1d9e60b49b826c620b4acc1b6c10f473f6d4741a5",
    size_bytes=123_851_407,
    license_statement=(
        "No separate checkpoint/model-card license was published in the "
        "evaluated Hugging Face repository; do not infer BSD-2-Clause coverage."
    ),
    redistribution="unresolved; PanorAi never redistributes this checkpoint",
)


@dataclass(frozen=True, slots=True)
class ExternalArtifactRecord:
    """Verified local materialization of an external artifact."""

    spec: ExternalArtifactSpec
    path: str
    previously_cached: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["verified_sha256"] = self.spec.sha256
        return result


@dataclass(frozen=True, slots=True)
class Metric3DAssets:
    """Pinned source tree and checkpoint used by the adapter."""

    source: ExternalArtifactRecord
    checkpoint: ExternalArtifactRecord
    source_root: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": SPHERICAL_METRIC_DEPTH_INTERFACE,
            "source": self.source.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "source_root": self.source_root,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True, slots=True)
class LoadedSphericalDepthModel:
    """Loaded resize-free model plus acquisition and port evidence."""

    model: Any
    assets: Metric3DAssets
    port_report: Any
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


class UpstreamTermsNotAcceptedError(PermissionError):
    """Raised before network/cache mutation when opt-in was omitted."""


def _default_cache_dir() -> Path:
    override = os.environ.get("PANORAI_CACHE_HOME")
    if override:
        return Path(override).expanduser() / "experimental" / "metric3d"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "panorai" / "experimental" / "metric3d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: ExternalArtifactSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise RuntimeError(
            f"{spec.kind} size mismatch for {path}: expected {spec.size_bytes}, "
            f"received {size}"
        )
    digest = _sha256(path)
    if digest != spec.sha256:
        raise RuntimeError(
            f"{spec.kind} SHA-256 mismatch for {path}: expected {spec.sha256}, "
            f"received {digest}"
        )


def _download(
    spec: ExternalArtifactSpec,
    destination: Path,
    *,
    timeout_seconds: float,
) -> bool:
    if destination.is_file():
        _verify(destination, spec)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(spec.url, headers={"User-Agent": "PanorAi-experimental/1"})
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    try:
        with (
            urlopen(request, timeout=timeout_seconds) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        _verify(temporary, spec)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return False


def _safe_extract_metric3d(archive_path: Path, destination: Path) -> Path:
    source_root = destination / f"Metric3D-{METRIC3D_SOURCE_COMMIT}"
    marker = source_root / ".panorai-source.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source_sha256") == METRIC3D_SOURCE.sha256:
            return source_root
        raise RuntimeError(f"source extraction marker does not match {source_root}")
    if source_root.exists():
        raise RuntimeError(
            f"unverified Metric3D source directory already exists: {source_root}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".metric3d-extract-", dir=destination))
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"unsafe Metric3D archive member: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(
                        f"unsupported Metric3D archive member: {member.name}"
                    )
            archive.extractall(staging, filter="data")
        extracted = staging / source_root.name
        if (
            not (extracted / "LICENSE").is_file()
            or not (extracted / "hubconf.py").is_file()
        ):
            raise RuntimeError("Metric3D archive lacks LICENSE or hubconf.py")
        license_digest = _sha256(extracted / "LICENSE")
        expected_license = (
            "cb2dba18d598b6ad0b34f0f8b0ad01f9d495e657cae9bf0544786190cb26f9f9"
        )
        if license_digest != expected_license:
            raise RuntimeError("Metric3D source LICENSE checksum changed")
        os.replace(extracted, source_root)
        marker.write_text(
            json.dumps(
                {
                    "source_commit": METRIC3D_SOURCE_COMMIT,
                    "source_sha256": METRIC3D_SOURCE.sha256,
                    "license": "BSD-2-Clause",
                    "license_sha256": expected_license,
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


def acquire_metric3d_convnext_tiny_v1(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> Metric3DAssets:
    """Download and verify the one supported model into an external cache.

    ``accept_upstream_terms=True`` is mandatory even when artifacts are already
    cached. It records caller intent; it is not a PanorAi license grant.
    """

    if not accept_upstream_terms:
        raise UpstreamTermsNotAcceptedError(
            "Metric3D acquisition requires accept_upstream_terms=True after "
            "reviewing the BSD-2-Clause source license and the unresolved "
            "checkpoint-license statement. PanorAi does not redistribute either."
        )
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    )
    downloads = root / "downloads"
    source_archive = downloads / METRIC3D_SOURCE.filename
    checkpoint = downloads / METRIC3D_CONVNEXT_TINY_V1.filename
    source_cached = _download(
        METRIC3D_SOURCE, source_archive, timeout_seconds=timeout_seconds
    )
    checkpoint_cached = _download(
        METRIC3D_CONVNEXT_TINY_V1, checkpoint, timeout_seconds=timeout_seconds
    )
    source_root = _safe_extract_metric3d(source_archive, root / "source")
    manifest = root / "metric3d-convnext-tiny-v1.json"
    assets = Metric3DAssets(
        source=ExternalArtifactRecord(
            spec=METRIC3D_SOURCE,
            path=str(source_archive),
            previously_cached=source_cached,
        ),
        checkpoint=ExternalArtifactRecord(
            spec=METRIC3D_CONVNEXT_TINY_V1,
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
    value = str(source_root)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:  # pragma: no cover - defensive against upstream mutation
            pass


def _patch_metric3d_v1_device_assumptions(model: Any) -> None:
    """Replace two upstream CUDA literals without changing learned state."""

    import torch

    decoder = model.depth_model.decoder

    def get_bins(self: Any, bins_num: int) -> Any:
        parameter = next(self.parameters())
        return torch.exp(
            torch.linspace(
                math.log(self.min_val),
                math.log(self.max_val),
                bins_num,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    def create_mesh_grid(
        self: Any,
        height: int,
        width: int,
        batch: int,
        device: Any = None,
        set_buffer: bool = True,
    ) -> Any:
        del device, set_buffer
        parameter = next(self.parameters())
        y, x = torch.meshgrid(
            torch.arange(height, device=parameter.device, dtype=parameter.dtype),
            torch.arange(width, device=parameter.device, dtype=parameter.dtype),
            indexing="ij",
        )
        return torch.stack((x, y))[None].repeat(batch, 1, 1, 1)

    decoder.get_bins = types.MethodType(get_bins, decoder)
    decoder.create_mesh_grid = types.MethodType(create_mesh_grid, decoder)


class SphericalMetric3D:
    """Resize-free ERP wrapper returning radial range in metres.

    Input is floating NCHW RGB in the explicit range ``[0, 255]``. Shape must
    be a full 2:1 ERP and both spatial dimensions must be divisible by 32. No
    hidden resizing, padding, crop, or prefilter is performed.
    """

    interface = SPHERICAL_METRIC_DEPTH_INTERFACE
    stability = "experimental"
    depth_semantics = "radial_range_m"
    input_layout = "NCHW RGB"
    input_range = "[0, 255]"
    model_depth_range_m = (0.3, 150.0)

    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, values: Any) -> Any:
        import torch

        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim != 4 or values.shape[1] != 3:
            raise ValueError("values must use NCHW RGB layout")
        height, width = values.shape[-2:]
        if width != 2 * height:
            raise ValueError("resize-free spherical depth requires a 2:1 ERP")
        if height % 32 or width % 32:
            raise ValueError("ERP height and width must be divisible by 32")
        if values.dtype != torch.float32:
            raise TypeError("values must use torch.float32 in [0, 255]")
        parameters = getattr(self.model, "parameters", None)
        parameter = next(parameters(), None) if callable(parameters) else None
        if parameter is not None and values.device != parameter.device:
            raise ValueError(
                f"input device {values.device} does not match model device "
                f"{parameter.device}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        minimum = values.detach().amin().item()
        maximum = values.detach().amax().item()
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError("values must lie in the explicit range [0, 255]")
        mean = values.new_tensor((123.675, 116.28, 103.53))[None, :, None, None]
        std = values.new_tensor((58.395, 57.12, 57.375))[None, :, None, None]
        normalized = (values - mean) / std
        prediction = self.model.inference({"input": normalized})[0]
        effective_focal = width / (2.0 * math.pi)
        radial_m = prediction[:, :1] * (effective_focal / 1000.0)
        return radial_m.clamp(*self.model_depth_range_m)

    def eval(self) -> SphericalMetric3D:
        self.model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> SphericalMetric3D:
        self.model.to(*args, **kwargs)
        return self


def load_spherical_metric3d_convnext_tiny_v1(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    device: str | Any = "cpu",
    max_sampled_elements: int | None = 100_000_000,
    timeout_seconds: float = 120.0,
) -> LoadedSphericalDepthModel:
    """Acquire, load, and fully port Metric3D-v1 ConvNeXt-Tiny/Hourglass."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional environment
        raise ImportError(
            "Torch is required; install `panorai[deep-learning-depth]`."
        ) from error
    from panorai.image_processing.torch import port_module_with_report

    assets = acquire_metric3d_convnext_tiny_v1(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    source_root = Path(assets.source_root)
    try:
        with _upstream_import_path(source_root):
            model = torch.hub.load(
                str(source_root),
                "metric3d_convnext_tiny",
                source="local",
                pretrain=False,
            )
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "Metric3D runtime dependencies are incomplete; install "
            "`panorai[deep-learning-depth]`. The upstream source remains external."
        ) from error
    payload = torch.load(
        assets.checkpoint.path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("Metric3D checkpoint lacks model_state_dict")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    expected_unexpected = ("depth_model.decoder.depth_expectation_anchor",)
    if missing or unexpected not in ((), expected_unexpected):
        raise RuntimeError(
            "Metric3D checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    _patch_metric3d_v1_device_assumptions(model)
    port_report = port_module_with_report(
        model, max_sampled_elements=max_sampled_elements
    )
    learned = [
        layer
        for layer in port_report.layers
        if layer.source_type in {"Conv2d", "ConvTranspose2d"}
    ]
    if any(layer.parameter_identity_preserved is not True for layer in learned):
        raise RuntimeError("spherical port replaced a learned Parameter object")
    if port_report.remaining_planar_spatial_layers:
        raise RuntimeError(
            "spherical port left planar spatial layers: "
            + ", ".join(port_report.remaining_planar_spatial_layers)
        )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedSphericalDepthModel(
        model=SphericalMetric3D(model),
        assets=assets,
        port_report=port_report,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


__all__ = [
    "SPHERICAL_METRIC_DEPTH_INTERFACE",
    "METRIC3D_CHECKPOINT_REVISION",
    "METRIC3D_CONVNEXT_TINY_V1",
    "METRIC3D_SOURCE",
    "METRIC3D_SOURCE_COMMIT",
    "ExternalArtifactRecord",
    "ExternalArtifactSpec",
    "LoadedSphericalDepthModel",
    "Metric3DAssets",
    "SphericalMetric3D",
    "UpstreamTermsNotAcceptedError",
    "acquire_metric3d_convnext_tiny_v1",
    "load_spherical_metric3d_convnext_tiny_v1",
]
