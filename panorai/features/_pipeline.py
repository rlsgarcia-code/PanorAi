"""The high-level PanorAi spherical-feature facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ._config import FaceSetSpec, SphericalFeaturePipelineConfig
from ._extractor import FeatureExtractor, _array_checksum, _as_panorama
from ._matcher import FeatureMatcher
from ._models import GnomonicRig, GnomonicRigCamera
from ._presets import MINIMUM_OPENCV_VERSION, PRESET_VERSION, preset_components
from .backends.opencv import OpenCVFeatureBackend


class SphericalFeaturePipeline:
    """Extract and match panorama features without exposing OpenCV objects."""

    def __init__(
        self,
        config: SphericalFeaturePipelineConfig,
        *,
        backend: OpenCVFeatureBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or OpenCVFeatureBackend()
        self.extractor = FeatureExtractor(
            config.extractor,
            backend=self.backend,
            minimum_opencv_version=config.minimum_opencv_version,
        )
        self.matcher = FeatureMatcher(
            config.matcher,
            backend=self.backend,
            minimum_opencv_version=config.minimum_opencv_version,
        )

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        face_sampler: str = "icosahedron",
        face_fov_deg: float | tuple[float, float] = 80.0,
        face_shape_hw: int | tuple[int, int] = (1024, 1024),
        face_overlap_deg: float = 0.0,
        count: int | None = None,
        subdivisions: int = 0,
        edge_margin_px: int | None = None,
        ratio_test: float | None = None,
        cross_check: bool | None = None,
        max_distance: float | None = None,
        max_features: int | None = None,
        angular_dedup_threshold_deg: float | None = None,
        backend: OpenCVFeatureBackend | None = None,
    ) -> "SphericalFeaturePipeline":
        extractor, matcher = preset_components(name)
        extractor_changes = {}
        matcher_changes = {}
        if edge_margin_px is not None:
            extractor_changes["edge_margin_px"] = edge_margin_px
        if max_features is not None:
            extractor_changes["max_features"] = max_features
        if angular_dedup_threshold_deg is not None:
            extractor_changes["angular_dedup_threshold_deg"] = (
                angular_dedup_threshold_deg
            )
        if ratio_test is not None:
            matcher_changes["ratio_test"] = ratio_test
        if cross_check is not None:
            matcher_changes["cross_check"] = cross_check
        if max_distance is not None:
            matcher_changes["max_distance"] = max_distance
        extractor = replace(extractor, **extractor_changes)
        matcher = replace(matcher, **matcher_changes)
        if isinstance(face_shape_hw, int):
            shape = (face_shape_hw, face_shape_hw)
        else:
            shape = tuple(face_shape_hw)
        if isinstance(face_fov_deg, (int, float)) and not isinstance(
            face_fov_deg, bool
        ):
            fov = (float(face_fov_deg), float(face_fov_deg))
        else:
            fov = tuple(face_fov_deg)
        face_set = FaceSetSpec(
            sampler=face_sampler,
            shape_hw=shape,
            fov_deg=fov,
            overlap_deg=face_overlap_deg,
            count=count,
            subdivisions=subdivisions,
        )
        config = SphericalFeaturePipelineConfig(
            extractor=extractor,
            matcher=matcher,
            face_set=face_set,
            preset_name=name.strip().lower(),
            preset_version=PRESET_VERSION,
            minimum_opencv_version=MINIMUM_OPENCV_VERSION,
        )
        return cls(config, backend=backend)

    def extract(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: Any | None = None,
    ):
        return self.extractor.extract(
            panorama,
            face_set_spec=self.config.face_set,
            panorama_id=panorama_id,
            validity_mask=validity_mask,
        )

    def match(self, features_a: Any, features_b: Any):
        return self.matcher.match(features_a, features_b)

    def extract_and_match(
        self,
        panorama_a: Any,
        panorama_b: Any,
        *,
        panorama_id_a: str | None = None,
        panorama_id_b: str | None = None,
        validity_mask_a: Any | None = None,
        validity_mask_b: Any | None = None,
    ):
        features_a = self.extract(
            panorama_a,
            panorama_id=panorama_id_a,
            validity_mask=validity_mask_a,
        )
        features_b = self.extract(
            panorama_b,
            panorama_id=panorama_id_b,
            validity_mask=validity_mask_b,
        )
        return self.match(features_a, features_b)

    def build_virtual_camera_rig(
        self, panorama: Any, *, panorama_id: str | None = None
    ) -> GnomonicRig:
        panorama = _as_panorama(panorama)
        panorama_id = panorama_id or f"panorama-{_array_checksum(panorama.image)[:16]}"
        spec = self.config.face_set
        views = panorama.views(
            spec.sampler,
            size=spec.shape_hw,
            fov=spec.effective_fov_deg,
            count=spec.count,
            subdivisions=spec.subdivisions,
        )
        cameras = []
        for index, face in enumerate(views):
            geometry = face.geometry
            cameras.append(
                GnomonicRigCamera(
                    camera_id=f"{panorama_id}:camera-{index:04d}",
                    face_id=face.face_id,
                    width=face.spec.output_shape_hw[1],
                    height=face.spec.output_shape_hw[0],
                    K=_to_numpy(geometry.K),
                    R_panorama_from_face=_to_numpy(geometry.R_panorama_from_face),
                )
            )
        return GnomonicRig(
            panorama_id=panorama_id,
            cameras=tuple(cameras),
            face_set_spec=spec,
        )

    def export_pycolmap(
        self,
        *,
        rig: GnomonicRig,
        features: Any,
        output_database: Any,
        matches: Any | None = None,
    ):
        from ._pycolmap import export_pycolmap

        return export_pycolmap(
            rig=rig,
            features=features,
            output_database=output_database,
            matches=matches,
        )

    def describe(self) -> dict[str, Any]:
        try:
            from panorai._version import __commit_id__, __version__
        except ImportError:
            __commit_id__ = None
            __version__ = "0+unknown"
        result = self.config.to_dict()
        result["panorai"] = {
            "version": __version__,
            "generating_commit": __commit_id__,
        }
        result["backend"] = {
            "name": self.backend.name,
            "version": self.backend.version,
        }
        return result

    def __repr__(self) -> str:
        return (
            "SphericalFeaturePipeline("
            f"preset={self.config.preset_name!r}, "
            f"sampler={self.config.face_set.sampler!r}, "
            f"extractor={self.config.extractor.method!r}, "
            f"matcher={self.config.matcher.method!r})"
        )


def _to_numpy(value: Any) -> np.ndarray:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        return value.detach().cpu().numpy()
    return np.asarray(value)
