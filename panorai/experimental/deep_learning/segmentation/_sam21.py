"""Pinned, lazy SAM 2.1 Hiera Large image-prompt backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ._prompts import SegmentPrompt, mask_stability_score

SAM21_HIERA_LARGE_REPOSITORY = "facebook/sam2.1-hiera-large"
SAM21_HIERA_LARGE_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM21_HIERA_LARGE_LICENSE = "Apache-2.0"
SAM21_HIERA_LARGE_FILES = {
    "config.json": (5705, None),
    "preprocessor_config.json": (683, None),
    "processor_config.json": (95, None),
    "model.safetensors": (
        897_897_416,
        "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230",
    ),
}


class Sam21TermsNotAcceptedError(PermissionError):
    """Raised before acquiring the external SAM 2.1 checkpoint."""


@dataclass(frozen=True, slots=True)
class Sam21FileRecord:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Sam21Assets:
    snapshot_path: str
    repository: str
    revision: str
    files: tuple[Sam21FileRecord, ...]
    previously_cached: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "panorai-sam21-assets/v1",
            "snapshot_path": self.snapshot_path,
            "repository": self.repository,
            "revision": self.revision,
            "files": [asdict(item) for item in self.files],
            "previously_cached": self.previously_cached,
            "upstream_license": SAM21_HIERA_LARGE_LICENSE,
            "distribution_boundary": "checkpoint remains in the external user cache",
        }


@dataclass(frozen=True, slots=True)
class SamMultimaskOutput:
    masks: np.ndarray
    logits: np.ndarray
    predicted_iou: tuple[float, float, float]
    object_presence_logit: float
    stability_scores: tuple[float, float, float]

    def __post_init__(self) -> None:
        masks = np.array(self.masks, dtype=bool, copy=True)
        logits = np.array(self.logits, dtype=np.float32, copy=True)
        if masks.ndim != 3 or masks.shape[0] != 3 or logits.shape != masks.shape:
            raise ValueError("SAM output must contain three aligned HW masks/logits")
        masks.setflags(write=False)
        logits.setflags(write=False)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "logits", logits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_sam21_hiera_large(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
) -> Sam21Assets:
    """Acquire and verify the pinned official checkpoint in an external cache."""

    if not accept_upstream_terms:
        raise Sam21TermsNotAcceptedError(
            "SAM 2.1 acquisition requires accept_upstream_terms=True after "
            "reviewing the upstream model and dataset terms"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "SAM 2.1 acquisition requires panorai[deep-learning-sam]"
        ) from error
    cache = None if cache_dir is None else str(Path(cache_dir).expanduser())
    snapshot = Path(
        snapshot_download(
            repo_id=SAM21_HIERA_LARGE_REPOSITORY,
            revision=SAM21_HIERA_LARGE_REVISION,
            cache_dir=cache,
            allow_patterns=list(SAM21_HIERA_LARGE_FILES),
        )
    )
    records: list[Sam21FileRecord] = []
    for filename, (expected_size, expected_hash) in SAM21_HIERA_LARGE_FILES.items():
        path = snapshot / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"SAM 2.1 file size mismatch or missing: {path}")
        digest = _sha256(path)
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"SAM 2.1 SHA-256 mismatch for {path}")
        records.append(Sam21FileRecord(filename, expected_size, digest))
    return Sam21Assets(
        snapshot_path=str(snapshot),
        repository=SAM21_HIERA_LARGE_REPOSITORY,
        revision=SAM21_HIERA_LARGE_REVISION,
        files=tuple(records),
        previously_cached=True,
    )


class Sam21HieraLargeSegmenter:
    """Reusable image embedding with transient prompt decoding."""

    def __init__(self, model: Any, processor: Any, *, device: Any, assets: Sam21Assets):
        self.model = model
        self.processor = processor
        self.device = device
        self.assets = assets

    def encode(self, rgb: np.ndarray) -> list[Any]:
        from PIL import Image

        import torch

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(pixels)
        self.synchronize()
        del inputs, pixels, image
        return embeddings

    def predict(
        self,
        embeddings: list[Any],
        shape_hw: tuple[int, int],
        prompt: SegmentPrompt,
    ) -> SamMultimaskOutput:
        import torch

        height, width = shape_hw
        arguments: dict[str, Any] = {
            "original_sizes": [[height, width]],
            "input_points": [[[list(point) for point in prompt.points_xy]]],
            "input_labels": [[list(prompt.labels)]],
            "return_tensors": "pt",
        }
        if prompt.box_xyxy is not None:
            arguments["input_boxes"] = [[list(prompt.box_xyxy)]]
        inputs = self.processor(**arguments)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_points", "input_labels", "input_boxes"}
        }
        if prompt.prior_mask is not None:
            model_inputs["input_masks"] = torch.from_numpy(
                np.asarray(prompt.prior_mask, dtype=np.float32)[None, None]
            ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(
                image_embeddings=embeddings,
                **model_inputs,
                multimask_output=True,
            )
        self.synchronize()
        raw_logits = outputs.pred_masks.detach().cpu()
        processed = self.processor.post_process_masks(
            raw_logits,
            [[height, width]],
            binarize=False,
        )[0]
        logits = processed.numpy()
        while logits.ndim > 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        if logits.shape[0] != 3:
            raise RuntimeError(f"SAM 2.1 returned unexpected masks {logits.shape}")
        masks = logits > 0
        scores = tuple(
            float(value)
            for value in outputs.iou_scores.detach().cpu().reshape(-1).numpy()
        )
        presence = float(outputs.object_score_logits.detach().cpu().reshape(-1)[0])
        stability = tuple(mask_stability_score(item) for item in logits)
        del outputs, raw_logits, processed, model_inputs, inputs
        return SamMultimaskOutput(masks, logits, scores, presence, stability)

    def predict_batch(
        self,
        embeddings: list[Any],
        shape_hw: tuple[int, int],
        prompts: tuple[SegmentPrompt, ...],
    ) -> tuple[SamMultimaskOutput, ...]:
        """Decode independent point prompts together for one image embedding.

        SAM calls this axis the point-batch dimension.  Discovery prompts all
        contain the same number of points and do not use boxes or mask priors;
        richer expansion prompts deliberately continue through ``predict``.
        """

        if not prompts:
            return ()
        point_counts = {len(prompt.points_xy) for prompt in prompts}
        if (
            len(point_counts) != 1
            or any(prompt.box_xyxy is not None for prompt in prompts)
            or any(prompt.prior_mask is not None for prompt in prompts)
        ):
            return tuple(
                self.predict(embeddings, shape_hw, prompt) for prompt in prompts
            )

        import torch

        height, width = shape_hw
        inputs = self.processor(
            original_sizes=[[height, width]],
            input_points=[
                [[list(point) for point in prompt.points_xy] for prompt in prompts]
            ],
            input_labels=[[list(prompt.labels) for prompt in prompts]],
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_points", "input_labels"}
        }
        with torch.inference_mode():
            outputs = self.model(
                image_embeddings=embeddings,
                **model_inputs,
                multimask_output=True,
            )
        self.synchronize()
        raw_logits = outputs.pred_masks.detach().cpu()
        processed = self.processor.post_process_masks(
            raw_logits,
            [[height, width]],
            binarize=False,
        )[0]
        logits = processed.numpy()
        while logits.ndim > 4 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        if logits.shape != (len(prompts), 3, height, width):
            raise RuntimeError(
                "SAM 2.1 returned unexpected batched masks "
                f"{logits.shape}; expected {(len(prompts), 3, height, width)}"
            )
        score_values = outputs.iou_scores.detach().cpu().numpy()
        presence_values = outputs.object_score_logits.detach().cpu().numpy()
        while score_values.ndim > 2 and score_values.shape[0] == 1:
            score_values = score_values.squeeze(0)
        while presence_values.ndim > 1 and presence_values.shape[0] == 1:
            presence_values = presence_values.squeeze(0)
        result = tuple(
            SamMultimaskOutput(
                batch_logits > 0,
                batch_logits,
                tuple(float(value) for value in score_values[index]),
                float(np.asarray(presence_values[index]).reshape(-1)[0]),
                tuple(mask_stability_score(item) for item in batch_logits),
            )
            for index, batch_logits in enumerate(logits)
        )
        del outputs, raw_logits, processed, model_inputs, inputs
        return result

    def synchronize(self) -> None:
        import torch

        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def release(self, embeddings: list[Any] | None = None) -> None:
        import torch

        if embeddings is not None:
            del embeddings
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_sam21_hiera_large(
    *,
    accept_upstream_terms: bool,
    cache_dir: str | Path | None = None,
    device: str = "auto",
) -> Sam21HieraLargeSegmenter:
    """Load the verified official model without requiring CUDA."""

    assets = acquire_sam21_hiera_large(
        accept_upstream_terms=accept_upstream_terms, cache_dir=cache_dir
    )
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "SAM 2.1 loading requires panorai[deep-learning-sam]"
        ) from error
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    target = torch.device(device)
    snapshot = assets.snapshot_path
    processor = Sam2Processor.from_pretrained(snapshot, local_files_only=True)
    model = (
        Sam2Model.from_pretrained(snapshot, local_files_only=True, use_safetensors=True)
        .eval()
        .to(target)
    )
    return Sam21HieraLargeSegmenter(model, processor, device=target, assets=assets)
