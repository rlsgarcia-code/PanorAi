"""Seeded segmentation on transient gnomonic charts; authoritative ERP masks."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from panorai.geometry import (
    GnomonicSpec,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_to_equirectangular,
)
from ._models import readonly


def segment_planar_region(
    view, region, *, method="simple", sam_backend=None, face_size=512
):
    """Return a draft region and alternatives; inference never accepts a mask.

    SAM backend follows encode/predict/release used by PanorAi segmentation.
    Positive/negative strokes are hard constraints after spherical fusion.
    """
    if method not in {"manual", "simple", "sam", "watershed", "classifier"}:
        raise ValueError("method must be manual, simple, sam, watershed or classifier")
    if method == "manual":
        return region, (region.mask,)
    if method == "sam" and sam_backend is None:
        raise ValueError("SAM requires an explicitly injected mask backend")
    if not isinstance(face_size, int) or not 64 <= face_size <= 2048:
        raise ValueError("face_size must be in [64, 2048]")
    positive = region.positive if region.positive is not None else region.mask
    negative = (
        region.negative if region.negative is not None else np.zeros_like(region.mask)
    )
    roi = region.roi if region.roi is not None else view.support
    positive = positive & roi & view.support & ~negative
    if not positive.any():
        raise ValueError("segmentation requires a supported positive seed")
    if region.roi is None and method in {"simple", "watershed", "classifier"}:
        raise ValueError(
            f"{method} seeded segmentation requires an explicit ROI polygon"
        )
    if method == "classifier":
        from ._interactive import color_classifier

        mask, diagnostics = color_classifier(
            view.image, positive, negative, roi, view.support
        )
        mask = readonly(mask)
        return replace(
            region,
            mask=mask,
            state="draft",
            source=method,
            interpretation="final",
            history=region.history + ({"action": "segment", **diagnostics},),
        ), (mask,)

    from panorai.experimental.deep_learning.segmentation import (
        SegmentPrompt,
        extract_rgb_face,
        consensus_frontiers,
        advance_chart,
    )

    ys, xs = np.where(positive)
    rays = erp_pixels_to_rays(np.stack((xs, ys), axis=-1).astype(float), positive.shape)
    # Greedy angular coverage, chosen from actual seeds (seam-safe).
    uncovered = np.ones(len(rays), bool)
    centers = []
    while uncovered.any():
        center = rays[np.flatnonzero(uncovered)[len(np.flatnonzero(uncovered)) // 2]]
        centers.append(center)
        uncovered &= rays @ center < np.cos(np.deg2rad(28))
        if len(centers) >= 64:
            raise ValueError("selection requires more than 64 charts; split the region")
    candidates = [np.zeros_like(positive) for _ in range(3 if method == "sam" else 1)]
    chart_support = np.zeros_like(positive)
    seen = set()
    for center in centers:
        lon = np.degrees(np.arctan2(center[0], center[2]))
        lat = np.degrees(np.arcsin(np.clip(center[1], -1, 1)))
        spec = GnomonicSpec(
            center_lat_deg=float(lat),
            center_lon_deg=float(lon),
            hfov_deg=80,
            vfov_deg=80,
            output_shape_hw=(face_size, face_size),
        )
        key = (round(float(lon) / 12), round(float(lat) / 12))
        if key in seen:
            continue
        seen.add(key)
        image = extract_rgb_face(view.image, spec)
        prior = candidates[0] if method != "sam" else np.sum(candidates, axis=0) >= 2
        masks = [
            equirectangular_to_gnomonic(m, spec, interpolation="nearest").data.astype(
                bool
            )
            for m in (positive | prior, negative, roi, view.support)
        ]
        fg, bg, face_roi, support = masks
        if not fg.any():
            continue
        if method == "simple":
            initialization = np.where(
                face_roi & support, cv2.GC_PR_BGD, cv2.GC_BGD
            ).astype(np.uint8)
            initialization[fg] = cv2.GC_FGD
            initialization[bg | ~support] = cv2.GC_BGD
            # Unobserved/outside-ROI pixels provide definite background.
            initialization[[0, -1], :] = cv2.GC_BGD
            initialization[:, [0, -1]] = cv2.GC_BGD
            if (
                not np.any(initialization == cv2.GC_BGD)
                or np.count_nonzero(initialization == cv2.GC_FGD) < 2
            ):
                continue
            cv2.grabCut(
                image,
                initialization,
                None,
                np.zeros((1, 65)),
                np.zeros((1, 65)),
                5,
                cv2.GC_INIT_WITH_MASK,
            )
            outputs = [
                (initialization == cv2.GC_FGD) | (initialization == cv2.GC_PR_FGD)
            ]
        elif method == "watershed":
            from ._interactive import marker_watershed

            outputs = [marker_watershed(image, fg, bg, face_roi, support)]
        else:
            py, px = np.where(fg)
            ny, nx = np.where(bg)
            positions = np.linspace(0, len(px) - 1, min(12, len(px)), dtype=int)
            points = [(float(px[i]), float(py[i])) for i in positions]
            labels = [1] * len(points)
            if len(nx):
                positions = np.linspace(0, len(nx) - 1, min(12, len(nx)), dtype=int)
                points += [(float(nx[i]), float(ny[i])) for i in positions]
                labels += [0] * len(positions)
            ry, rx = np.where(face_roi & support)
            box = (float(rx.min()), float(ry.min()), float(rx.max()), float(ry.max()))
            prior_face = equirectangular_to_gnomonic(
                prior, spec, interpolation="nearest"
            ).data.astype(bool)
            prompt = SegmentPrompt(
                tuple(points),
                tuple(labels),
                box,
                prior_face if prior_face.any() else None,
            )
            embeddings = sam_backend.encode(image)
            try:
                output = sam_backend.predict(embeddings, (face_size, face_size), prompt)
                outputs = output.masks
            finally:
                sam_backend.release(embeddings)

        # ERP masks use nearest-only backprojection through the existing API.
        def backproject(mask):
            projected = gnomonic_to_equirectangular(
                mask, spec, positive.shape, interpolation="nearest", fill_value=False
            )
            return projected.data.astype(bool) & projected.support_mask & view.support

        projected_support = backproject(support)
        chart_support |= projected_support
        for index, output in enumerate(outputs):
            projected = backproject(output & face_roi & support)
            candidates[index] |= projected
        consensus = np.sum(outputs, axis=0) >= (2 if method == "sam" else 1)
        consensus &= face_roi & support & ~bg
        # Continue only through mask frontiers. New charts retain the previous
        # mask as a prompt, and all generated masks remain drafts for review.
        if len(centers) < 24:
            frontiers = consensus_frontiers(
                consensus,
                band_pixels=max(2, face_size // 16),
                minimum_pixels=max(4, face_size // 8),
            )
            for _, frontier in frontiers:
                next_spec, _ = advance_chart(spec, frontier, step_fraction=0.45)
                next_key = (
                    round(next_spec.center_lon_deg / 12),
                    round(next_spec.center_lat_deg / 12),
                )
                if next_key not in seen and len(centers) < 24:
                    longitude, latitude = np.deg2rad(
                        [next_spec.center_lon_deg, next_spec.center_lat_deg]
                    )
                    centers.append(
                        np.array(
                            [
                                np.cos(latitude) * np.sin(longitude),
                                np.sin(latitude),
                                np.cos(latitude) * np.cos(longitude),
                            ]
                        )
                    )
    if not chart_support.any():
        raise ValueError("no supported segmentation chart")
    candidates = [
        readonly((m | positive) & ~negative & roi & view.support) for m in candidates
    ]
    # Alternative indices are local decoder choices, not a cross-chart identity.
    # The UI exposes each aggregate and requires human acceptance.
    result = (
        candidates[0] if method != "sam" else readonly(np.sum(candidates, axis=0) >= 2)
    )
    history = region.history
    if method == "watershed":
        history += (
            {
                "action": "segment",
                "algorithm": "marker-lab-gradient-watershed/v1",
                "background": "negative-strokes-and-outside-roi-chart-border",
                "face_size": face_size,
                "charts": len(seen),
                "gaussian_sigma": 1.0,
            },
        )
    return replace(
        region,
        mask=result,
        state="draft",
        source=method,
        interpretation="final",
        history=history,
    ), tuple(candidates)
