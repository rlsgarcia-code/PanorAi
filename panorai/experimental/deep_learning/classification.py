"""Frozen external classifiers for Experimental dense spherical inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence
from urllib.request import Request, urlopen

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from panorai.image_processing.torch import spherical_area_average

from .fcn import ImageNetFCN


SPHERICAL_CLASSIFICATION_INTERFACE = "panorai-spherical-classification/v1"
PLACES365_SOURCE_COMMIT = "8a953ed56438726dc98bdef3796d042e7f1f171e"
OPENCLIP_REPOSITORY_REVISION = "ec3d92cf63a5f9d591f0d611b736895966c73076"


@dataclass(frozen=True, slots=True)
class ClassificationArtifactSpec:
    """Pinned metadata for one externally acquired classifier artifact."""

    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    provenance: str


PLACES365_RESNET18 = ClassificationArtifactSpec(
    name="places365-resnet18",
    filename="resnet18_places365.pth.tar",
    url=("http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"),
    sha256="2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588",
    size_bytes=45_506_139,
    license="CC-BY; see the official Places365 model zoo",
    provenance=(
        "CSAILVision Places365 PyTorch ResNet18 checkpoint; official model-zoo "
        "transport is HTTP, so the complete pinned SHA-256 is mandatory"
    ),
)

PLACES365_CATEGORIES = ClassificationArtifactSpec(
    name="places365-categories",
    filename="categories_places365.txt",
    url=(
        "https://raw.githubusercontent.com/CSAILVision/places365/"
        f"{PLACES365_SOURCE_COMMIT}/categories_places365.txt"
    ),
    sha256="2affba635eb657e7ca95f4e6cc69bd9fac29ef4c32aeb83cafdfcd06ec6a1ea6",
    size_bytes=6_833,
    license="MIT repository metadata; image copyrights remain with image owners",
    provenance=f"CSAILVision/places365 commit {PLACES365_SOURCE_COMMIT}",
)

OPENCLIP_RN50_OPENAI = ClassificationArtifactSpec(
    name="openclip-rn50-openai",
    filename="open_clip_model.safetensors",
    url=(
        "https://huggingface.co/timm/resnet50_clip.openai/resolve/"
        f"{OPENCLIP_REPOSITORY_REVISION}/open_clip_model.safetensors"
    ),
    sha256="da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee",
    size_bytes=408_291_932,
    license="MIT as declared by the pinned Hugging Face model card",
    provenance=(
        "OpenAI CLIP RN50 converted for OpenCLIP and hosted by timm at pinned "
        f"revision {OPENCLIP_REPOSITORY_REVISION}"
    ),
)


class ClassificationTermsNotAcceptedError(PermissionError):
    """Raised before acquiring external classifier assets without opt-in."""


@dataclass(frozen=True, slots=True)
class ClassificationArtifactRecord:
    """Verified local identity of one external artifact."""

    spec: ClassificationArtifactSpec
    path: str
    previously_cached: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["spec"] = asdict(self.spec)
        return result


@dataclass(frozen=True, slots=True)
class Places365Assets:
    """Verified external files required by the Places365 ResNet18 loader."""

    checkpoint: ClassificationArtifactRecord
    categories: ClassificationArtifactRecord
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "panorai-classification-assets/v1",
            "model": "places365-resnet18",
            "checkpoint": self.checkpoint.to_dict(),
            "categories": self.categories.to_dict(),
            "manifest_path": self.manifest_path,
            "distribution_boundary": (
                "External assets remain in the user-controlled cache and are not "
                "part of PanorAi source, wheels, or sdists."
            ),
        }


@dataclass(frozen=True, slots=True)
class OpenCLIPAssets:
    """Verified external file required by the OpenCLIP RN50 loader."""

    checkpoint: ClassificationArtifactRecord
    repository_revision: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "panorai-classification-assets/v1",
            "model": "openclip-rn50-openai",
            "checkpoint": self.checkpoint.to_dict(),
            "repository_revision": self.repository_revision,
            "manifest_path": self.manifest_path,
            "distribution_boundary": (
                "External assets remain in the user-controlled cache and are not "
                "part of PanorAi source, wheels, or sdists."
            ),
        }


@dataclass(frozen=True, slots=True)
class LoadedPlaces365Model:
    """Frozen Places365 model, ordered categories, and verified assets."""

    model: nn.Module
    categories: tuple[str, ...]
    assets: Places365Assets
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    image_size: int


@dataclass(frozen=True, slots=True)
class LoadedOpenCLIPRN50:
    """Frozen OpenCLIP model, tokenizer, and verified external checkpoint."""

    model: nn.Module
    tokenizer: Any
    assets: OpenCLIPAssets
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    image_size: int
    open_clip_version: str


@dataclass(frozen=True, slots=True)
class DenseOpenVocabularyOutput:
    """Final visual lattice, local embeddings, and prompt-score lattice."""

    features: Tensor
    embeddings: Tensor
    logits: Tensor


def _default_cache_dir() -> Path:
    override = os.environ.get("PANORAI_CACHE_HOME")
    if override:
        return Path(override).expanduser() / "experimental" / "classification"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "panorai" / "experimental" / "classification"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(spec: ClassificationArtifactSpec, path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"external artifact is missing: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise RuntimeError(
            f"artifact size mismatch for {path}: expected {spec.size_bytes}, "
            f"received {size}"
        )
    digest = _sha256(path)
    if digest != spec.sha256:
        raise RuntimeError(
            f"artifact SHA-256 mismatch for {path}: expected {spec.sha256}, "
            f"received {digest}"
        )


def _download(
    spec: ClassificationArtifactSpec,
    destination: Path,
    *,
    timeout_seconds: float,
) -> bool:
    """Acquire one pinned file atomically; return whether it was cached."""

    if destination.exists():
        _verify_artifact(spec, destination)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(spec.url, headers={"User-Agent": "PanorAi/experimental"})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with urlopen(request, timeout=timeout_seconds) as response:
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        _verify_artifact(spec, temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return False


def _require_terms(accept_upstream_terms: bool) -> None:
    if not accept_upstream_terms:
        raise ClassificationTermsNotAcceptedError(
            "external classifier acquisition requires "
            "accept_upstream_terms=True after reviewing the checkpoint, model "
            "card, dataset and attribution terms; PanorAi does not redistribute "
            "these assets or grant their licenses"
        )


def acquire_places365_resnet18(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 180.0,
) -> Places365Assets:
    """Acquire and verify the official Places365 ResNet18 external assets."""

    _require_terms(accept_upstream_terms)
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    ) / "places365-resnet18"
    checkpoint_path = root / PLACES365_RESNET18.filename
    categories_path = root / PLACES365_CATEGORIES.filename
    checkpoint_cached = _download(
        PLACES365_RESNET18, checkpoint_path, timeout_seconds=timeout_seconds
    )
    categories_cached = _download(
        PLACES365_CATEGORIES, categories_path, timeout_seconds=timeout_seconds
    )
    manifest_path = root / "manifest.json"
    assets = Places365Assets(
        checkpoint=ClassificationArtifactRecord(
            spec=PLACES365_RESNET18,
            path=str(checkpoint_path),
            previously_cached=checkpoint_cached,
        ),
        categories=ClassificationArtifactRecord(
            spec=PLACES365_CATEGORIES,
            path=str(categories_path),
            previously_cached=categories_cached,
        ),
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(
        json.dumps(assets.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assets


def _parse_places365_categories(path: Path) -> tuple[str, ...]:
    categories: list[str] = []
    for expected_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        raw_name, raw_index = line.rsplit(maxsplit=1)
        if int(raw_index) != expected_index:
            raise RuntimeError("Places365 category indices are not contiguous")
        parts = raw_name.split("/", maxsplit=2)
        name = parts[-1].replace("_", " ")
        categories.append(name)
    if len(categories) != 365:
        raise RuntimeError(
            f"expected 365 Places365 categories, received {len(categories)}"
        )
    return tuple(categories)


def load_places365_resnet18(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 180.0,
) -> LoadedPlaces365Model:
    """Acquire, verify, safely deserialize, and strictly load Places365 RN18."""

    assets = acquire_places365_resnet18(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    try:
        from torchvision import models
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError(
            "Torchvision is required for Places365 ResNet18; install PanorAi "
            'with `pip install "panorai[deep-learning]"`.'
        ) from error
    payload = torch.load(
        assets.checkpoint.path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("arch") != "resnet18":
        raise RuntimeError("unexpected Places365 checkpoint structure")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict):
        raise RuntimeError("Places365 checkpoint has no state_dict")
    state = {
        str(key).removeprefix("module."): value for key, value in raw_state.items()
    }
    model = models.resnet18(weights=None, num_classes=365).eval()
    model.load_state_dict(state, strict=True)
    categories = _parse_places365_categories(Path(assets.categories.path))
    return LoadedPlaces365Model(
        model=model,
        categories=categories,
        assets=assets,
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        image_size=224,
    )


def acquire_openclip_rn50(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 600.0,
) -> OpenCLIPAssets:
    """Acquire and verify the pinned OpenCLIP RN50 safetensors checkpoint."""

    _require_terms(accept_upstream_terms)
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    ) / "openclip-rn50-openai"
    checkpoint_path = root / OPENCLIP_RN50_OPENAI.filename
    checkpoint_cached = _download(
        OPENCLIP_RN50_OPENAI,
        checkpoint_path,
        timeout_seconds=timeout_seconds,
    )
    manifest_path = root / "manifest.json"
    assets = OpenCLIPAssets(
        checkpoint=ClassificationArtifactRecord(
            spec=OPENCLIP_RN50_OPENAI,
            path=str(checkpoint_path),
            previously_cached=checkpoint_cached,
        ),
        repository_revision=OPENCLIP_REPOSITORY_REVISION,
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(
        json.dumps(assets.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assets


def load_openclip_rn50(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 600.0,
) -> LoadedOpenCLIPRN50:
    """Acquire and load the pinned OpenCLIP RN50 model and tokenizer."""

    assets = acquire_openclip_rn50(
        accept_upstream_terms=accept_upstream_terms,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    try:
        import open_clip
    except ImportError as error:  # pragma: no cover - optional installation
        raise ImportError(
            "OpenCLIP RN50 requires open-clip-torch; install PanorAi with "
            '`pip install "panorai[deep-learning-openclip]"`.'
        ) from error
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",
        pretrained=assets.checkpoint.path,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer("RN50")
    mean = tuple(float(value) for value in preprocess.transforms[-1].mean)
    std = tuple(float(value) for value in preprocess.transforms[-1].std)
    image_size = int(model.visual.image_size)
    return LoadedOpenCLIPRN50(
        model=model,
        tokenizer=tokenizer,
        assets=assets,
        normalization_mean=mean,
        normalization_std=std,
        image_size=image_size,
        open_clip_version=str(getattr(open_clip, "__version__", "unknown")),
    )


class Places365ResNet18FCN(ImageNetFCN):
    """Dense Places365 scene classifier using the frozen ResNet18 weights."""

    interface = SPHERICAL_CLASSIFICATION_INTERFACE
    stability = "experimental"

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model, "resnet18")


class OpenCLIPRN50Dense(nn.Module):
    """Dense open-vocabulary map from an OpenCLIP modified-ResNet visual tower.

    The fixed-resolution attention-pool classifier is intentionally removed.
    Its exact ``v_proj`` and ``c_proj`` parameters become a pointwise spatial
    projection, following the fully convolutional dense-CLIP interpretation.
    The trained query/key projections and absolute 7x7 positional embedding do
    not participate in this dense head and are reported as a limitation; this
    output is a local image-text similarity lattice, not the original global
    CLIP classification logit.
    """

    interface = SPHERICAL_CLASSIFICATION_INTERFACE
    stability = "experimental"

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        visual = getattr(clip_model, "visual", None)
        required = (
            "conv1",
            "bn1",
            "act1",
            "conv2",
            "bn2",
            "act2",
            "conv3",
            "bn3",
            "act3",
            "avgpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "attnpool",
        )
        if visual is None or any(not hasattr(visual, name) for name in required):
            raise TypeError("clip_model must expose an OpenCLIP RN50 visual tower")
        attnpool = visual.attnpool
        if not all(hasattr(attnpool, name) for name in ("v_proj", "c_proj")):
            raise TypeError("OpenCLIP RN50 attention pool has unexpected structure")
        self.visual = visual
        self.value_projection = attnpool.v_proj
        self.output_projection = attnpool.c_proj
        self.logit_scale = clip_model.logit_scale

    def forward_features(self, values: Tensor) -> Tensor:
        """Return the last NCHW modified-ResNet feature lattice."""

        visual = self.visual
        values = visual.act1(visual.bn1(visual.conv1(values)))
        values = visual.act2(visual.bn2(visual.conv2(values)))
        values = visual.act3(visual.bn3(visual.conv3(values)))
        values = visual.avgpool(values)
        values = visual.layer1(values)
        values = visual.layer2(values)
        values = visual.layer3(values)
        return visual.layer4(values)

    def project_features(self, features: Tensor) -> Tensor:
        """Apply the exact attention value/output projections at every site."""

        tokens = features.permute(0, 2, 3, 1)
        tokens = F.linear(
            tokens,
            self.value_projection.weight,
            self.value_projection.bias,
        )
        tokens = F.linear(
            tokens,
            self.output_projection.weight,
            self.output_projection.bias,
        )
        return tokens.permute(0, 3, 1, 2)

    def forward_dense(
        self,
        values: Tensor,
        normalized_text_features: Tensor,
    ) -> DenseOpenVocabularyOutput:
        """Return local embeddings and scaled prompt-similarity maps."""

        if normalized_text_features.ndim != 2:
            raise ValueError("normalized_text_features must have shape (classes, D)")
        features = self.forward_features(values)
        embeddings = self.project_features(features)
        if embeddings.shape[1] != normalized_text_features.shape[1]:
            raise ValueError("visual and text embedding dimensions differ")
        normalized_embeddings = F.normalize(embeddings, dim=1)
        text = F.normalize(
            normalized_text_features.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            ),
            dim=1,
        )
        logits = self.logit_scale.exp() * torch.einsum(
            "ndhw,cd->nchw", normalized_embeddings, text
        )
        return DenseOpenVocabularyOutput(
            features=features,
            embeddings=normalized_embeddings,
            logits=logits,
        )

    def forward(
        self,
        values: Tensor,
        normalized_text_features: Tensor,
        *,
        spherical_average: bool = True,
    ) -> Tensor:
        logits = self.forward_dense(values, normalized_text_features).logits
        if spherical_average:
            return spherical_area_average(logits)
        return logits.mean(dim=(-2, -1))


def encode_openclip_prompts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
) -> Tensor:
    """Encode and normalize a non-empty ordered prompt sequence."""

    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain non-empty strings")
    tokens = tokenizer(list(prompts))
    with torch.inference_mode():
        return model.encode_text(tokens, normalize=True)


__all__ = [
    "SPHERICAL_CLASSIFICATION_INTERFACE",
    "PLACES365_SOURCE_COMMIT",
    "OPENCLIP_REPOSITORY_REVISION",
    "PLACES365_RESNET18",
    "PLACES365_CATEGORIES",
    "OPENCLIP_RN50_OPENAI",
    "ClassificationArtifactRecord",
    "ClassificationArtifactSpec",
    "ClassificationTermsNotAcceptedError",
    "DenseOpenVocabularyOutput",
    "LoadedOpenCLIPRN50",
    "LoadedPlaces365Model",
    "OpenCLIPAssets",
    "OpenCLIPRN50Dense",
    "Places365Assets",
    "Places365ResNet18FCN",
    "acquire_openclip_rn50",
    "acquire_places365_resnet18",
    "encode_openclip_prompts",
    "load_openclip_rn50",
    "load_places365_resnet18",
]
