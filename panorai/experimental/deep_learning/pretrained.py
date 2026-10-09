"""Lazy acquisition of supported Torchvision ImageNet checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlparse


SUPPORTED_IMAGENET_MODELS = ("alexnet", "vgg16", "resnet18")


@dataclass(frozen=True, slots=True)
class ImageNetWeightRecord:
    """Provenance for one externally cached Torchvision checkpoint."""

    model_name: str
    weights_name: str
    url: str
    cache_path: str
    sha256: str
    size_bytes: int
    previously_cached: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedImageNetModel:
    """A model, its Torchvision transform metadata, and checkpoint record."""

    model: Any
    weights: Any
    checkpoint: ImageNetWeightRecord


def _resolve_model(model_name: str) -> tuple[Any, Any]:
    name = model_name.strip().lower()
    if name not in SUPPORTED_IMAGENET_MODELS:
        raise ValueError(
            f"model_name must be one of {SUPPORTED_IMAGENET_MODELS}; received "
            f"{model_name!r}"
        )
    try:
        from torchvision import models
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Torchvision is required for pretrained ImageNet models; install "
            'PanorAi with `pip install "panorai[deep-learning]"`.'
        ) from error

    registry = {
        "alexnet": (models.alexnet, models.AlexNet_Weights.DEFAULT),
        "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    }
    return registry[name]


def _checkpoint_path(weights: Any) -> Path:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Torch is required for pretrained ImageNet models; install PanorAi "
            'with `pip install "panorai[deep-learning]"`.'
        ) from error

    filename = Path(urlparse(weights.url).path).name
    if not filename:
        raise RuntimeError(f"Torchvision weight URL has no filename: {weights.url!r}")
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash_prefix(weights: Any) -> str:
    filename = Path(urlparse(weights.url).path).name
    match = re.search(r"-([0-9a-fA-F]{8,64})\.(?:pt|pth)$", filename)
    if match is None:
        raise RuntimeError(
            "Torchvision checkpoint filename does not expose a hash prefix: "
            f"{filename!r}"
        )
    return match.group(1).lower()


def _verified_record(
    model_name: str,
    weights: Any,
    path: Path,
    *,
    previously_cached: bool,
) -> ImageNetWeightRecord:
    if not path.is_file():
        raise RuntimeError(f"Torchvision did not materialize checkpoint {path}")
    digest = _sha256(path)
    expected_prefix = _expected_hash_prefix(weights)
    if not digest.startswith(expected_prefix):
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {path}: expected prefix "
            f"{expected_prefix}, received {digest}"
        )
    return ImageNetWeightRecord(
        model_name=model_name,
        weights_name=str(weights),
        url=str(weights.url),
        cache_path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        previously_cached=previously_cached,
    )


def prefetch_imagenet_weights(
    model_names: Iterable[str] = SUPPORTED_IMAGENET_MODELS,
    *,
    progress: bool = True,
) -> tuple[ImageNetWeightRecord, ...]:
    """Download and verify supported DEFAULT weights in the external Torch cache.

    No checkpoint is copied into PanorAi's source tree or distribution. Existing
    files are verified before Torchvision deserializes them; newly downloaded
    files are verified afterwards against the hash prefix in the official URL.
    """

    names = tuple(name.strip().lower() for name in model_names)
    if not names:
        raise ValueError("model_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("model_names must not contain duplicates")

    records: list[ImageNetWeightRecord] = []
    for name in names:
        _, weights = _resolve_model(name)
        path = _checkpoint_path(weights)
        previously_cached = path.is_file()
        if previously_cached:
            _verified_record(name, weights, path, previously_cached=True)

        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        del state_dict
        records.append(
            _verified_record(
                name,
                weights,
                path,
                previously_cached=previously_cached,
            )
        )
    return tuple(records)


def load_pretrained_imagenet_model(
    model_name: str,
    *,
    progress: bool = True,
) -> LoadedImageNetModel:
    """Download if needed, verify, and load one supported DEFAULT model."""

    name = model_name.strip().lower()
    builder, weights = _resolve_model(name)
    checkpoint = prefetch_imagenet_weights((name,), progress=progress)[0]
    model = builder(weights=weights, progress=False).eval()
    return LoadedImageNetModel(
        model=model,
        weights=weights,
        checkpoint=checkpoint,
    )


__all__ = [
    "SUPPORTED_IMAGENET_MODELS",
    "ImageNetWeightRecord",
    "LoadedImageNetModel",
    "load_pretrained_imagenet_model",
    "prefetch_imagenet_weights",
]
