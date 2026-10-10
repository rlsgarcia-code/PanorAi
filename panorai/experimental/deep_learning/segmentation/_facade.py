"""One-object facade for pretrained industrial spherical segmentation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._models import (
    DenseSemanticEvidence,
    SphericalSegmentationConfig,
    SphericalSegmentationResult,
)
from ._pipeline import SphericalSemanticSegmenter
from ._semantic import (
    INDUSTRIAL_IMAGENET_PROXY_VOCABULARY,
    concept_evidence_from_imagenet,
)

SPHERICAL_INDUSTRIAL_SEGMENTATION_INTERFACE = (
    "panorai-spherical-industrial-segmentation/v1"
)

ProgressCallback = Callable[[dict[str, object]], None]


def _nearest_resize(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values)
    if source.shape[:2] == shape_hw:
        return source.copy()
    source_height, source_width = source.shape[:2]
    height, width = shape_hw
    rows = np.minimum(
        ((np.arange(height) + 0.5) / height * source_height).astype(np.int64),
        source_height - 1,
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) / width * source_width).astype(np.int64),
        source_width - 1,
    )
    return source[rows[:, None], columns[None, :]]


def _imagenet_proxy_indices() -> np.ndarray:
    return np.asarray(
        sorted(
            {
                class_index
                for concept in INDUSTRIAL_IMAGENET_PROXY_VOCABULARY
                for class_index, _, _ in concept.class_weights
            }
        ),
        dtype=np.int64,
    )


def _release_accelerator_memory(device: Any) -> None:
    """Release allocator caches without importing Torch at module import time."""

    gc.collect()
    if getattr(device, "type", None) == "mps":
        import torch

        torch.mps.empty_cache()
    elif getattr(device, "type", None) == "cuda":  # pragma: no cover - guarded API
        import torch

        torch.cuda.empty_cache()


class SphericalIndustrialSegmenter:
    """Run the complete pretrained industrial-proxy pipeline with one object.

    The facade accepts one canonical HWC uint8 equirectangular panorama.  It
    loads and sphericalizes InternImage-G, computes dense ImageNet proxy
    evidence, releases that large model, and only then loads SAM 2.1 for the
    streamed gnomonic segmentation stage.  Checkpoints are downloaded lazily
    on the first :meth:`predict` call and remain in verified external caches.

    Use :class:`SphericalSemanticSegmenter` directly when semantic evidence or
    a mask backend is already available.
    """

    interface = SPHERICAL_INDUSTRIAL_SEGMENTATION_INTERFACE
    stability = "experimental"
    supported_semantic_model = "internimage-g"
    supported_mask_model = "sam2.1-hiera-large"

    def __init__(
        self,
        *,
        accept_upstream_terms: bool,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        config: SphericalSegmentationConfig | None = None,
        semantic_input_height: int = 512,
        working_height: int = 1024,
        max_sampled_elements: int = 16_000_000,
    ) -> None:
        if not isinstance(accept_upstream_terms, bool):
            raise TypeError("accept_upstream_terms must be bool")
        if device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', or 'mps'; CUDA is not used")
        for name, value in (
            ("semantic_input_height", semantic_input_height),
            ("working_height", working_height),
            ("max_sampled_elements", max_sampled_elements),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.accept_upstream_terms = accept_upstream_terms
        self.device = device
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.config = config or SphericalSegmentationConfig()
        self.semantic_input_height = semantic_input_height
        self.working_height = working_height
        self.max_sampled_elements = max_sampled_elements
        self._mask_backend: Any | None = None
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        semantic_model: str = "internimage-g",
        mask_model: str = "sam2.1-hiera-large",
        device: str = "auto",
        accept_upstream_terms: bool,
        cache_dir: str | Path | None = None,
        config: SphericalSegmentationConfig | None = None,
        semantic_input_height: int = 512,
        working_height: int = 1024,
        max_sampled_elements: int = 16_000_000,
    ) -> SphericalIndustrialSegmenter:
        """Configure the pinned pretrained models for lazy verified loading."""

        if semantic_model != cls.supported_semantic_model:
            raise ValueError(f"semantic_model must be {cls.supported_semantic_model!r}")
        if mask_model != cls.supported_mask_model:
            raise ValueError(f"mask_model must be {cls.supported_mask_model!r}")
        return cls(
            accept_upstream_terms=accept_upstream_terms,
            device=device,
            cache_dir=cache_dir,
            config=config,
            semantic_input_height=semantic_input_height,
            working_height=working_height,
            max_sampled_elements=max_sampled_elements,
        )

    def _resolved_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return torch.device(self.device)

    def _semantic_shape(self, source_shape_hw: tuple[int, int]) -> tuple[int, int]:
        height, width = source_shape_hw
        target_height = min(height, self.semantic_input_height)
        target_width = max(1, round(width * target_height / height))
        return target_height, target_width

    def _default_output_shape(
        self, source_shape_hw: tuple[int, int]
    ) -> tuple[int, int]:
        height, width = source_shape_hw
        target_height = min(height, self.working_height)
        target_width = max(1, round(width * target_height / height))
        return target_height, target_width

    def _infer_semantic_evidence(
        self,
        rgb: np.ndarray,
        support: np.ndarray,
        *,
        progress: ProgressCallback | None,
    ) -> tuple[DenseSemanticEvidence, dict[str, object]]:
        import torch
        from PIL import Image
        from torch.nn import functional as functional

        from ..internimage import (
            INTERNIMAGE_G_REPOSITORY,
            INTERNIMAGE_G_REVISION,
            InternImageGDenseClassifier,
            load_pretrained_internimage_g,
            port_internimage_to_spherical,
        )

        device = self._resolved_device()
        if progress is not None:
            progress({"phase": "semantic_model_loading"})
        loaded: Any | None = None
        classifier: Any | None = None
        dense: Any | None = None
        normalized: Any | None = None
        feature_support: Any | None = None
        try:
            loaded = load_pretrained_internimage_g(
                accept_upstream_terms=self.accept_upstream_terms,
                cache_dir=self.cache_dir,
                portable_core=True,
            )
            classifier = InternImageGDenseClassifier(loaded.model).eval()
            classifier.to(device=device, dtype=torch.float32)
            port_report = port_internimage_to_spherical(
                classifier,
                max_sampled_elements=self.max_sampled_elements,
            )
            if port_report.remaining_dcnv3_layers:
                raise RuntimeError("DCNv3 layers remain after the spherical port")
            if port_report.spatial_layers.remaining_planar_spatial_layers:
                raise RuntimeError(
                    "planar spatial layers remain after the spherical port"
                )
            if not port_report.parameter_identity_preserved:
                raise RuntimeError(
                    "the spherical port did not preserve parameter identity"
                )

            semantic_shape = self._semantic_shape(rgb.shape[:2])
            if semantic_shape == rgb.shape[:2]:
                semantic_rgb = rgb.copy()
            else:
                semantic_rgb = np.asarray(
                    Image.fromarray(rgb, mode="RGB").resize(
                        semantic_shape[::-1], Image.Resampling.BILINEAR
                    )
                )
            semantic_support = _nearest_resize(support, semantic_shape).astype(bool)
            normalized = (
                torch.from_numpy(semantic_rgb.copy())
                .permute(2, 0, 1)
                .to(device=device, dtype=torch.float32)[None]
                / 255.0
            )
            mean = torch.tensor(
                loaded.processor.image_mean,
                device=device,
                dtype=torch.float32,
            )[None, :, None, None]
            std = torch.tensor(
                loaded.processor.image_std,
                device=device,
                dtype=torch.float32,
            )[None, :, None, None]
            normalized = (normalized - mean) / std
            if progress is not None:
                progress(
                    {
                        "phase": "semantic_inference",
                        "input_shape_hw": list(semantic_shape),
                    }
                )
            with torch.inference_mode():
                dense = classifier.forward_dense(normalized)
                support_tensor = torch.from_numpy(semantic_support).to(device=device)
                feature_support = functional.interpolate(
                    support_tensor[None, None].float(),
                    size=dense.logits.shape[-2:],
                    mode="nearest",
                ).bool()
                indices = _imagenet_proxy_indices()
                selected = (
                    dense.logits[0, indices.tolist()].detach().float().cpu().numpy()
                )
                evidence_support = (
                    feature_support[0, 0].detach().cpu().numpy().astype(bool)
                )
            provenance = {
                "interface": self.interface,
                "model": "spherical InternImage-G",
                "repository": INTERNIMAGE_G_REPOSITORY,
                "revision": INTERNIMAGE_G_REVISION,
                "semantic_input_shape_hw": list(semantic_shape),
                "evidence_shape_hw": list(selected.shape[-2:]),
                "device": str(device),
                "assets": loaded.assets.to_dict(),
                "port": port_report.to_dict(),
            }
            evidence = concept_evidence_from_imagenet(
                selected,
                indices,
                evidence_support,
                provenance=provenance,
            )
            return evidence, provenance
        finally:
            del feature_support, normalized, dense, classifier, loaded
            _release_accelerator_memory(device)
            if progress is not None:
                progress({"phase": "semantic_model_released"})

    def _load_mask_backend(self, *, progress: ProgressCallback | None) -> Any:
        if self._mask_backend is None:
            if progress is not None:
                progress({"phase": "mask_model_loading"})
            from ._sam21 import load_sam21_hiera_large

            self._mask_backend = load_sam21_hiera_large(
                accept_upstream_terms=self.accept_upstream_terms,
                cache_dir=self.cache_dir,
                device=self.device,
            )
        return self._mask_backend

    def _release_mask_backend(self) -> None:
        backend = self._mask_backend
        self._mask_backend = None
        if backend is None:
            return
        device = getattr(backend, "device", None)
        backend.release()
        del backend
        _release_accelerator_memory(device)

    def predict(
        self,
        panorama: np.ndarray,
        *,
        projection: str = "equirectangular",
        support_mask: np.ndarray | None = None,
        output_shape_hw: tuple[int, int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> SphericalSegmentationResult:
        """Segment one canonical ERP and return compact spherical instances.

        If ``support_mask`` is omitted, the whole sphere is declared supported;
        support is never inferred from pixel values.  By default, masks use an
        aspect-preserving lattice capped at ``working_height``.  Pass the source
        ``(H, W)`` explicitly when source-resolution masks are required.
        """

        if self._closed:
            raise RuntimeError("this SphericalIndustrialSegmenter is closed")
        if projection != "equirectangular":
            raise ValueError("only projection='equirectangular' is supported")
        rgb = np.asarray(panorama)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("panorama must be an HWC uint8 RGB array")
        if support_mask is None:
            support = np.ones(rgb.shape[:2], dtype=bool)
            support_source = "implicit-full-sphere"
        else:
            support = np.asarray(support_mask, dtype=bool)
            if support.shape != rgb.shape[:2] or not support.any():
                raise ValueError(
                    "support_mask must be a non-empty HW array matching panorama"
                )
            support_source = "explicit"
        target_shape = (
            self._default_output_shape(rgb.shape[:2])
            if output_shape_hw is None
            else output_shape_hw
        )
        if len(target_shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in target_shape
        ):
            raise ValueError("output_shape_hw must contain two positive integers")

        evidence, semantic_provenance = self._infer_semantic_evidence(
            rgb, support, progress=progress
        )
        backend = self._load_mask_backend(progress=progress)
        try:
            result = SphericalSemanticSegmenter(backend, self.config).segment(
                rgb,
                support,
                evidence=evidence,
                output_shape_hw=target_shape,
                progress=progress,
            )
            mask_assets = (
                backend.assets.to_dict() if hasattr(backend, "assets") else None
            )
        finally:
            del backend
            self._release_mask_backend()
        diagnostics = dict(result.diagnostics)
        diagnostics["facade"] = {
            "interface": self.interface,
            "semantic_model": self.supported_semantic_model,
            "mask_model": self.supported_mask_model,
            "source_shape_hw": list(rgb.shape[:2]),
            "output_shape_hw": list(target_shape),
            "source_resolution_output": target_shape == rgb.shape[:2],
            "support_source": support_source,
            "semantic": semantic_provenance,
            "mask_assets": mask_assets,
            "model_lifecycle": (
                "InternImage and SAM loaded sequentially and released after each stage"
            ),
        }
        return SphericalSegmentationResult(
            segments=result.segments,
            proposals=result.proposals,
            panoptic_map=result.panoptic_map,
            support=result.support,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        """Release any active model backend and make the facade unusable."""

        if self._closed:
            return
        self._release_mask_backend()
        self._closed = True

    def __enter__(self) -> SphericalIndustrialSegmenter:
        if self._closed:
            raise RuntimeError("this SphericalIndustrialSegmenter is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "SPHERICAL_INDUSTRIAL_SEGMENTATION_INTERFACE",
    "SphericalIndustrialSegmenter",
]
