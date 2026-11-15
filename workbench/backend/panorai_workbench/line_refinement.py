"""Reviewable Workbench line suggestions built on public PanorAi APIs."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def _line_frame(line: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    direction = line[1] - line[0]
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < 8.0:
        raise ValueError("manual line must span at least 8 chart pixels")
    direction /= length
    return direction, np.array([-direction[1], direction[0]]), length


def _line_band_points(
    edge_mask: np.ndarray,
    manual_xy: np.ndarray,
    band_radius_px: float,
) -> np.ndarray:
    direction, normal, length = _line_frame(manual_xy)
    y, x = np.nonzero(edge_mask)
    points = np.stack((x, y), axis=1).astype(np.float64)
    relative = points - manual_xy[0]
    along = relative @ direction
    across = np.abs(relative @ normal)
    return points[
        (along >= -band_radius_px)
        & (along <= length + band_radius_px)
        & (across <= band_radius_px)
    ]


def robust_line_suggestion(
    edge_mask: np.ndarray,
    manual_xy: np.ndarray,
    *,
    band_radius_px: float = 12.0,
    inlier_threshold_px: float = 1.5,
    iterations: int = 384,
) -> dict[str, Any] | None:
    """Fit one deterministic TLS-refined line near a reviewed manual segment.

    This is a Workbench annotation assist, not a PanorAi scientific estimator.
    """

    mask = np.asarray(edge_mask, dtype=bool)
    line = np.asarray(manual_xy, dtype=np.float64)
    if mask.ndim != 2 or line.shape != (2, 2):
        raise ValueError("edge_mask must be HW and manual_xy must be (2, 2)")
    if not np.isfinite(line).all() or not 2.0 <= float(band_radius_px) <= 64.0:
        raise ValueError("line and band radius must be finite and bounded")
    direction, _, length = _line_frame(line)
    candidates = _line_band_points(mask, line, float(band_radius_px))
    if len(candidates) < 8:
        return None
    seed = int(np.round(np.sum(line * np.array([[17.0, 31.0], [43.0, 59.0]])))) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray] | None = None
    manual_angle = np.arctan2(direction[1], direction[0])
    for _ in range(min(max(32, int(iterations)), 2048)):
        pair = candidates[rng.choice(len(candidates), 2, replace=False)]
        delta = pair[1] - pair[0]
        pair_length = float(np.linalg.norm(delta))
        if pair_length < max(6.0, length * 0.2):
            continue
        candidate_direction = delta / pair_length
        angle = abs(np.arctan2(candidate_direction[1], candidate_direction[0]) - manual_angle)
        angle = min(angle, np.pi - min(angle, np.pi))
        if angle > np.deg2rad(18.0):
            continue
        candidate_normal = np.array([-candidate_direction[1], candidate_direction[0]])
        distances = np.abs((candidates - pair[0]) @ candidate_normal)
        inliers = distances <= inlier_threshold_px
        if int(inliers.sum()) < 6:
            continue
        projected = (candidates[inliers] - pair[0]) @ candidate_direction
        score = float(inliers.sum()) + 0.02 * float(np.ptp(projected))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        return None
    inlier_points = candidates[best[1]]
    centre = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centre, full_matrices=False)
    fitted_direction = vh[0]
    if fitted_direction @ direction < 0:
        fitted_direction = -fitted_direction
    fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
    recommended = np.stack(
        [point - fitted_normal * ((point - centre) @ fitted_normal) for point in line]
    )
    displacement = np.linalg.norm(recommended - line, axis=1)
    if float(np.max(displacement)) > float(band_radius_px):
        return None
    residuals = np.abs((inlier_points - centre) @ fitted_normal)
    return {
        "recommended_chart_xy": recommended.tolist(),
        "candidate_count": int(len(candidates)),
        "inlier_count": int(len(inlier_points)),
        "inlier_fraction": float(len(inlier_points) / len(candidates)),
        "median_residual_px": float(np.median(residuals)),
        "endpoint_displacement_px": displacement.tolist(),
    }


def refine_erp_line(
    image: np.ndarray,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Create a bounded suggestion while preserving the submitted manual line."""

    from panorai.geometry import (
        GnomonicSpec,
        equirectangular_to_gnomonic,
        erp_pixels_to_rays,
        gnomonic_pixels_to_rays,
        rays_to_erp_pixels,
        rays_to_gnomonic_pixels,
    )
    import cv2

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("image must use HWC RGB layout")
    manual_erp = np.asarray(request.get("manual_erp_xy"), dtype=np.float64)
    if manual_erp.shape != (2, 2) or not np.isfinite(manual_erp).all():
        raise ValueError("manual_erp_xy must contain two finite ERP points")
    chart = request.get("chart") or {}
    shape = tuple(int(value) for value in chart.get("output_shape_hw", ()))
    if len(shape) != 2 or not (128 <= shape[0] <= 1400 and 128 <= shape[1] <= 1800):
        raise ValueError("chart output_shape_hw is outside the supported preview bounds")
    spec = GnomonicSpec(
        center_lat_deg=float(chart["center_lat_deg"]),
        center_lon_deg=float(chart["center_lon_deg"]),
        hfov_deg=float(chart["hfov_deg"]),
        vfov_deg=float(chart["vfov_deg"]),
        output_shape_hw=shape,
    )
    low = float(request.get("threshold_low", 0.035))
    high = float(request.get("threshold_high", 0.09))
    band = float(request.get("band_radius_px", 12.0))
    chart_rgb = equirectangular_to_gnomonic(
        source.astype(np.float32), spec, interpolation="bilinear"
    ).data.clip(0, 255).astype(np.uint8)
    manual_rays = erp_pixels_to_rays(manual_erp, source.shape[:2])
    manual_projection = rays_to_gnomonic_pixels(manual_rays, spec)
    if not np.asarray(manual_projection.valid).all():
        raise ValueError("manual line is not fully visible in the declared chart")
    manual_chart = np.asarray(manual_projection.pixels_xy, dtype=np.float64)
    margin = max(18, int(np.ceil(band * 2)))
    low_xy = np.floor(manual_chart.min(axis=0) - margin).astype(int)
    high_xy = np.ceil(manual_chart.max(axis=0) + margin).astype(int)
    x0, y0 = np.maximum(low_xy, 0)
    x1, y1 = np.minimum(high_xy + 1, np.array([shape[1], shape[0]]))
    crop = chart_rgb[y0:y1, x0:x1]
    manual_crop = manual_chart - [x0, y0]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 1.0)
    threshold_passes = [(low, high), (low * 0.5, high * 0.5)]
    suggestion = None
    edges_crop = np.zeros(crop_gray.shape, dtype=bool)
    selected_thresholds = threshold_passes[0]
    for pass_low, pass_high in threshold_passes:
        edges_crop = cv2.Canny(
            crop_gray,
            pass_low * 255.0,
            pass_high * 255.0,
            L2gradient=True,
        ) > 0
        selected_thresholds = (pass_low, pass_high)
        suggestion = robust_line_suggestion(
            edges_crop,
            manual_crop,
            band_radius_px=band,
            inlier_threshold_px=float(request.get("inlier_threshold_px", 1.5)),
        )
        if suggestion is not None:
            break
    candidates = _line_band_points(edges_crop, manual_crop, band)
    stream = BytesIO()
    Image.fromarray(crop).save(stream, format="PNG", optimize=True)
    payload: dict[str, Any] = {
        "schema": "panorai-workbench-line-refinement/v1",
        "status": "suggested" if suggestion else "no-supported-edge",
        "manual_erp_xy": manual_erp.tolist(),
        "manual_chart_xy": manual_crop.tolist(),
        "crop_bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "crop_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
        "candidate_count": int(len(candidates)),
        "edge_points_crop_xy": candidates[
            :: max(1, len(candidates) // 1500)
        ].tolist(),
        "execution": {
            "edge_detector": "opencv.Canny",
            "edge_domain": "gnomonic-crop",
            "projection_public_api": "panorai.geometry.equirectangular_to_gnomonic",
            "fit_owner": "panorai-workbench-annotation-assist",
            "automatic_accept": False,
            "reported_coordinate_grid": "original-erp-pixel-centres",
        },
        "parameters": {
            "threshold_low": low,
            "threshold_high": high,
            "selected_threshold_low": selected_thresholds[0],
            "selected_threshold_high": selected_thresholds[1],
            "adaptive_pass": selected_thresholds != threshold_passes[0],
            "band_radius_px": band,
            "crop_shape_hw": [int(crop.shape[0]), int(crop.shape[1])],
        },
    }
    if suggestion:
        recommended_crop = np.asarray(suggestion.pop("recommended_chart_xy"))
        recommended_chart = recommended_crop + [x0, y0]
        rays = gnomonic_pixels_to_rays(recommended_chart, spec)
        recommended_erp = rays_to_erp_pixels(rays.rays_xyz, source.shape[:2])
        payload.update(suggestion)
        payload["recommended_erp_xy"] = np.asarray(recommended_erp.pixels_xy).tolist()
        payload["recommended_chart_xy"] = recommended_crop.tolist()
    return payload
