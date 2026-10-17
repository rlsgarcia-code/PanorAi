"""Rectified-view planning and independent-view pairing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    gnomonic_pixels_to_rays,
    gnomonic_rotation,
    rays_to_gnomonic_pixels,
)

from ._types import (
    PanoramicStereoPose,
    RectifiedStereoViewSpec,
    StereoViewDescriptor,
    StereoViewPairScore,
    StereoViewPlan,
)


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be one finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def spec_from_axes(
    forward: np.ndarray,
    right: np.ndarray,
    *,
    hfov_deg: float,
    vfov_deg: float,
    output_shape_hw: tuple[int, int],
) -> GnomonicSpec:
    """Create a rolled gnomonic spec from desired forward and right axes."""

    forward = _unit(forward, "forward")
    right = _unit(right - np.dot(right, forward) * forward, "right")
    latitude = math.degrees(math.asin(float(np.clip(forward[1], -1.0, 1.0))))
    longitude = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    unrolled = GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        output_shape_hw=output_shape_hw,
    )
    basis = np.asarray(gnomonic_rotation(unrolled), dtype=np.float64)
    base_right = basis[:, 0]
    base_down = basis[:, 1]
    roll = math.degrees(
        math.atan2(float(np.dot(right, base_down)), float(np.dot(right, base_right)))
    )
    return GnomonicSpec(
        center_lat_deg=latitude,
        center_lon_deg=longitude,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        roll_deg=roll,
        output_shape_hw=output_shape_hw,
    )


@dataclass(frozen=True, slots=True)
class RectifiedStereoViewPlanner:
    """Plan a ring of virtual pinhole pairs rectified by construction.

    The local x-axis follows the physical A-to-B baseline.  Corresponding
    epipolar great circles therefore become horizontal raster rows in both
    faces.  The two epipolar singularities are deliberately left outside the
    finite pinhole coverage and reported by the plan.
    """

    hfov_deg: float = 90.0
    vfov_deg: float = 100.0
    overlap_deg: float = 20.0
    output_shape_hw: tuple[int, int] = (512, 512)

    def __post_init__(self) -> None:
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError("hfov_deg must be between 0 and 180")
        if not 0.0 < self.vfov_deg < 180.0:
            raise ValueError("vfov_deg must be between 0 and 180")
        if not 0.0 <= self.overlap_deg < self.hfov_deg:
            raise ValueError("overlap_deg must be in [0, hfov_deg)")
        if len(self.output_shape_hw) != 2 or min(self.output_shape_hw) < 8:
            raise ValueError("output_shape_hw must contain dimensions >= 8")
        object.__setattr__(
            self,
            "output_shape_hw",
            (int(self.output_shape_hw[0]), int(self.output_shape_hw[1])),
        )

    def plan(self, pose: PanoramicStereoPose) -> StereoViewPlan:
        rotation = pose.rotation_source_from_reference
        translation = pose.translation_source_from_reference
        # Camera B centre expressed in reference panorama A coordinates.
        baseline_axis = _unit(-rotation.T @ translation, "baseline axis")
        seed = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(seed, baseline_axis))) > 0.9:
            seed = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        first = _unit(seed - np.dot(seed, baseline_axis) * baseline_axis, "ring axis")
        second = _unit(np.cross(baseline_axis, first), "ring axis")
        step = self.hfov_deg - self.overlap_deg
        count = max(4, int(math.ceil(360.0 / step)))
        pairs: list[RectifiedStereoViewSpec] = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            forward_a = math.cos(angle) * first + math.sin(angle) * second
            right_a = baseline_axis
            forward_b = rotation @ forward_a
            right_b = rotation @ right_a
            reference_spec = spec_from_axes(
                forward_a,
                right_a,
                hfov_deg=self.hfov_deg,
                vfov_deg=self.vfov_deg,
                output_shape_hw=self.output_shape_hw,
            )
            source_spec = spec_from_axes(
                forward_b,
                right_b,
                hfov_deg=self.hfov_deg,
                vfov_deg=self.vfov_deg,
                output_shape_hw=self.output_shape_hw,
            )
            pairs.append(
                RectifiedStereoViewSpec(
                    pair_id=f"rectified-{index:03d}",
                    reference_spec=reference_spec,
                    source_spec=source_spec,
                    longitude_index=index,
                )
            )
        coverage = float(math.sin(math.radians(self.vfov_deg) / 2.0))
        return StereoViewPlan(
            tuple(pairs),
            baseline_axis,
            estimated_angular_coverage=coverage,
            omitted_epipolar_cone_deg=90.0 - self.vfov_deg / 2.0,
        )


def _sample_view_rays(spec: GnomonicSpec, samples: int = 9) -> np.ndarray:
    height, width = spec.output_shape_hw
    x = np.linspace(-0.5, width - 0.5, samples)
    y = np.linspace(-0.5, height - 0.5, samples)
    xx, yy = np.meshgrid(x, y)
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    return np.asarray(gnomonic_pixels_to_rays(pixels, spec).rays_xyz)


class SphericalStereoViewPairer:
    """Score many-to-many pairs for independently sampled spherical views."""

    def __init__(
        self,
        *,
        min_score: float = 0.20,
        min_angular_overlap: float = 0.05,
        max_pairs_per_reference: int | None = 3,
    ) -> None:
        self.min_score = float(min_score)
        self.min_angular_overlap = float(min_angular_overlap)
        self.max_pairs_per_reference = max_pairs_per_reference

    @staticmethod
    def _overlap(
        reference: StereoViewDescriptor,
        source: StereoViewDescriptor,
        rotation: np.ndarray,
    ) -> float:
        rays_a = _sample_view_rays(reference.spec)
        rays_b = rays_a @ rotation.T
        projected_b = rays_to_gnomonic_pixels(rays_b, source.spec)
        source_to_reference = _sample_view_rays(source.spec) @ rotation
        projected_a = rays_to_gnomonic_pixels(source_to_reference, reference.spec)
        return 0.5 * (
            float(np.asarray(projected_b.valid, dtype=np.float64).mean())
            + float(np.asarray(projected_a.valid, dtype=np.float64).mean())
        )

    @staticmethod
    def _support(view: StereoViewDescriptor) -> float:
        if view.support_mask is None:
            return 1.0
        return float(view.support_mask.mean())

    def score(
        self,
        reference: StereoViewDescriptor,
        source: StereoViewDescriptor,
        pose: PanoramicStereoPose,
        *,
        matches: Any | None = None,
        inlier_mask: np.ndarray | None = None,
    ) -> StereoViewPairScore:
        overlap = self._overlap(reference, source, pose.rotation_source_from_reference)
        support = math.sqrt(self._support(reference) * self._support(source))
        sparse = 0.0
        epipolar = 0.0
        if matches is not None:
            valid = np.asarray(matches.valid, dtype=bool)
            if inlier_mask is not None:
                valid &= np.asarray(inlier_mask, dtype=bool)
            selected = (
                valid
                & (np.asarray(matches.face_ids_a, dtype=object) == reference.view_id)
                & (np.asarray(matches.face_ids_b, dtype=object) == source.view_id)
            )
            total = max(1, int(valid.sum()))
            sparse = min(1.0, 8.0 * float(selected.sum()) / total)
            if np.any(selected):
                a = np.asarray(matches.bearings_a[selected], dtype=np.float64)
                b = np.asarray(matches.bearings_b[selected], dtype=np.float64)
                t = pose.translation_source_from_reference / pose.baseline
                rb = a @ pose.rotation_source_from_reference.T
                residual = np.abs(np.einsum("ij,j->i", np.cross(rb, b), t))
                epipolar = float(np.exp(-np.median(residual) / 0.01))
        score = support * (0.60 * overlap + 0.25 * sparse + 0.15 * epipolar)
        return StereoViewPairScore(
            reference.view_id,
            source.view_id,
            float(score),
            overlap,
            sparse,
            epipolar,
            support,
        )

    def pair(
        self,
        reference_views: Iterable[StereoViewDescriptor],
        source_views: Iterable[StereoViewDescriptor],
        pose: PanoramicStereoPose,
        *,
        matches: Any | None = None,
        inlier_mask: np.ndarray | None = None,
    ) -> tuple[StereoViewPairScore, ...]:
        sources = tuple(source_views)
        accepted: list[StereoViewPairScore] = []
        for reference in reference_views:
            scored = [
                self.score(
                    reference,
                    source,
                    pose,
                    matches=matches,
                    inlier_mask=inlier_mask,
                )
                for source in sources
            ]
            scored = [
                item
                for item in scored
                if item.score >= self.min_score
                and item.angular_overlap >= self.min_angular_overlap
            ]
            scored.sort(key=lambda item: (-item.score, item.source_view_id))
            if self.max_pairs_per_reference is not None:
                scored = scored[: self.max_pairs_per_reference]
            accepted.extend(scored)
        accepted.sort(
            key=lambda item: (item.reference_view_id, -item.score, item.source_view_id)
        )
        return tuple(accepted)

    def match(
        self,
        reference_views: Iterable[StereoViewDescriptor],
        source_views: Iterable[StereoViewDescriptor],
        *,
        pose: PanoramicStereoPose,
        sparse_matches: Any | None = None,
        inlier_mask: np.ndarray | None = None,
    ) -> tuple[StereoViewPairScore, ...]:
        """Match independently sampled views with the documented façade."""

        return self.pair(
            reference_views,
            source_views,
            pose,
            matches=sparse_matches,
            inlier_mask=inlier_mask,
        )


__all__ = [
    "RectifiedStereoViewPlanner",
    "SphericalStereoViewPairer",
    "spec_from_axes",
]
