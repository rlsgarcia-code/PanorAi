"""Registered bearing triangulation, robust plane fitting and radial ranges."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.optimize import least_squares

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from ._models import (
    PlaneEstimate,
    PlanarRangeResult,
    PlanarRegion,
    TriangulatedPlaneTrack,
    readonly,
)


def _angles(a, b):
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    a = a / np.maximum(a_norm, 1e-15)
    b = b / np.maximum(b_norm, 1e-15)
    angles = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    )
    return np.where((a_norm[..., 0] > 1e-15) & (b_norm[..., 0] > 1e-15), angles, 180.0)


def sample_mask(mask, pixels):
    h, w = mask.shape
    xy = np.asarray(pixels)
    finite = np.isfinite(xy).all(axis=-1)
    safe = np.where(finite[..., None], xy, 0)
    x = np.floor(safe[..., 0] + 0.5).astype(np.int64) % w
    y = np.floor(safe[..., 1] + 0.5).astype(np.int64)
    return finite & (y >= 0) & (y < h) & mask[np.clip(y, 0, h - 1), x]


def triangulate_workspace(workspace, options):
    views = {v.view_id: v for v in workspace.scene.views}
    masks = {(r.plane_id, r.view_id): r for r in workspace.scene.regions}
    tracks = []
    for track in workspace.correspondences:
        origins, directions, weights = [], [], []
        reason = None
        accepted_views = {
            r.view_id
            for r in workspace.scene.regions
            if r.plane_id == track.plane_id and r.state == "accepted" and r.mask.any()
        }
        if not accepted_views:
            tracks.append(
                TriangulatedPlaneTrack(
                    track, None, 0, 0, 180, "no-accepted-reference-mask"
                )
            )
            continue
        observations = (
            tuple(o for o in track.observations if o.view_id in accepted_views)
            if len(accepted_views) >= 2
            else track.observations
        )
        if len(observations) < 2:
            tracks.append(
                TriangulatedPlaneTrack(
                    track, None, 0, 0, 180, "insufficient-reviewed-views"
                )
            )
            continue
        track = replace(track, observations=observations)
        for obs in observations:
            view = views[obs.view_id]
            pixel = np.asarray(obs.pixel_xy)[None]
            region = masks.get((track.plane_id, obs.view_id))
            if not sample_mask(view.support, pixel)[0]:
                reason = "outside-support"
            if region is not None and (
                region.state == "rejected"
                or (
                    region.state == "accepted"
                    and not sample_mask(region.mask, pixel)[0]
                )
            ):
                reason = "outside-accepted-mask"
            bearing = erp_pixels_to_rays(pixel, view.support.shape)[0]
            directions.append(view.pose.R.T @ bearing)
            origins.append(view.pose.center)
            weights.append(obs.weight)
        centers, rays = np.asarray(origins), np.asarray(directions)
        weights = np.asarray(weights) / np.sum(weights)
        angles = [_angles(rays[i], rays[j]) for i in range(len(rays)) for j in range(i)]
        # Parallel and antiparallel rays both have poor intersection conditioning.
        parallax = float(max((min(a, 180 - a) for a in angles), default=0))
        projectors = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
        normal = np.einsum("n,nij->ij", weights, projectors)
        rhs = np.einsum("n,nij,nj->i", weights, projectors, centers)
        point = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        delta = point - centers
        residual = float(np.max(_angles(delta, rays)))
        if parallax < options.minimum_parallax_deg or np.linalg.cond(normal) > 1e8:
            reason = reason or "insufficient-parallax"
        elif np.any(np.sum(delta * rays, axis=1) <= 0):
            reason = reason or "negative-range"
        elif residual > options.maximum_reprojection_deg:
            reason = reason or "angular-reprojection"
        sigma = max(
            0.001,
            float(np.median(np.linalg.norm(delta, axis=1)))
            * math.radians(max(residual, 0.02))
            / max(math.sin(math.radians(parallax)), 1e-6),
        )
        tracks.append(
            TriangulatedPlaneTrack(
                track,
                readonly(point) if reason is None else None,
                sigma,
                parallax,
                residual,
                reason,
            )
        )
    return replace(workspace, tracks=tuple(tracks))


def _basis(normal):
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return np.stack((first, np.cross(normal, first)), axis=1)


def _tls(points, weights):
    center = np.average(points, axis=0, weights=weights)
    _, singular, vectors = np.linalg.svd(
        (points - center) * np.sqrt(weights[:, None]), full_matrices=False
    )
    if len(singular) < 3 or singular[1] <= max(1e-8, singular[0] * 1e-3):
        raise ValueError("collinear-or-concentrated-support")
    return vectors[-1], -float(vectors[-1] @ center)


def fit_workspace(workspace, options):
    views = {v.view_id: v for v in workspace.scene.views}
    estimates = []
    for plane_id in sorted({r.plane_id for r in workspace.scene.regions}):
        tracks = [
            t
            for t in workspace.tracks
            if t.correspondence.plane_id == plane_id and t.point_world is not None
        ]
        accepted_views = {
            r.view_id
            for r in workspace.scene.regions
            if r.plane_id == plane_id and r.state == "accepted" and r.mask.any()
        }
        if len(tracks) < options.minimum_fit_tracks:
            estimates.append(
                PlaneEstimate(
                    plane_id,
                    None,
                    None,
                    None,
                    None,
                    "needs-review",
                    {"reason": "insufficient-tracks", "track_count": len(tracks)},
                )
            )
            continue
        points = np.asarray([t.point_world for t in tracks])
        sigma = np.asarray([t.sigma_m for t in tracks])
        rng = np.random.default_rng(options.random_seed)
        order = rng.permutation(len(points))
        held_count = min(int(len(points) * options.holdout_fraction), len(points) - 3)
        held = order[:held_count]
        train = order[held_count:]
        best, score = None, (-1, -np.inf)
        for _ in range(options.ransac_trials):
            indices = rng.choice(train, 3, replace=False)
            a, b, c = points[indices]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n /= norm
            d = -n @ a
            residuals = np.abs(points[train] @ n + d)
            inliers = residuals <= options.plane_threshold_m
            candidate = (
                int(inliers.sum()),
                -float(np.minimum(residuals, options.plane_threshold_m).sum()),
            )
            if candidate > score:
                best, score = train[inliers], candidate
        if best is None or len(best) < 3 or len(best) < len(train) * 0.5:
            estimates.append(
                PlaneEstimate(
                    plane_id,
                    None,
                    None,
                    None,
                    None,
                    "needs-review",
                    {"reason": "no-planar-consensus"},
                )
            )
            continue
        try:
            weights = 1 / np.square(sigma[best])
            n, d = _tls(points[best], weights)
            for _ in range(8):
                residual = np.abs(points[best] @ n + d)
                robust = np.minimum(
                    1, options.plane_threshold_m / np.maximum(residual, 1e-10)
                )
                n, d = _tls(points[best], weights * robust)
        except ValueError as error:
            estimates.append(
                PlaneEstimate(
                    plane_id,
                    None,
                    None,
                    None,
                    None,
                    "needs-review",
                    {"reason": str(error)},
                )
            )
            continue
        # Orient toward the first supporting camera for deterministic normal signs.
        reference = views[tracks[best[0]].correspondence.observations[0].view_id]
        if n @ reference.pose.center + d < 0:
            n, d = -n, -d
        basis = _basis(n)
        covariance_origin = np.average(points[best], axis=0, weights=weights)
        local_offset = float(n @ covariance_origin + d)

        def parameters(x):
            normal = n + basis @ x[:2]
            normal /= np.linalg.norm(normal)
            return normal, local_offset + x[2] - normal @ covariance_origin

        def angular_residuals(x):
            normal, offset = parameters(x)
            result = []
            for index in best:
                observations = tracks[index].correspondence.observations
                first = observations[0]
                source = views[first.view_id]
                ray = (
                    source.pose.R.T
                    @ erp_pixels_to_rays(
                        np.array(first.pixel_xy)[None], source.support.shape
                    )[0]
                )
                denominator = normal @ ray
                distance = -(normal @ source.pose.center + offset) / (
                    denominator
                    if abs(denominator) > 1e-8
                    else math.copysign(1e-8, denominator)
                )
                point = source.pose.center + distance * ray
                for observation in observations[1:]:
                    target = views[observation.view_id]
                    observed = erp_pixels_to_rays(
                        np.array(observation.pixel_xy)[None], target.support.shape
                    )[0]
                    predicted = target.pose.R @ (point - target.pose.center)
                    predicted /= max(np.linalg.norm(predicted), 1e-12)
                    # Bearing-vector residual is chordal, locally equal to angular error.
                    result.extend(np.sqrt(observation.weight) * (predicted - observed))
            return np.asarray(result)

        optimized = least_squares(
            angular_residuals,
            np.zeros(3),
            loss="huber",
            f_scale=math.radians(0.1),
            max_nfev=80,
        )
        new_n, new_d = parameters(optimized.x)
        if np.median(np.abs(points[best] @ new_n + new_d)) <= options.plane_threshold_m:
            n, d = new_n, new_d
        basis = _basis(n)
        residuals = np.abs(points @ n + d)
        train_inliers = train[residuals[train] <= options.plane_threshold_m]
        design = np.column_stack(
            (
                (points[train_inliers] - covariance_origin) @ basis,
                np.ones(len(train_inliers)),
            )
        )
        information = design.T @ (design / np.square(sigma[train_inliers, None]))
        covariance = np.linalg.pinv(information)
        condition = float(np.linalg.cond(information))
        held_rmse = float(np.sqrt(np.mean(residuals[held] ** 2))) if len(held) else None
        held_angular = []
        # Independent points never participate in fitting or its refinement.
        for index in held:
            observations = tracks[index].correspondence.observations
            first = observations[0]
            source = views[first.view_id]
            ray = (
                source.pose.R.T
                @ erp_pixels_to_rays(
                    np.array(first.pixel_xy)[None], source.support.shape
                )[0]
            )
            denominator = n @ ray
            if abs(denominator) < 1e-10:
                held_angular.append(180.0)
                continue
            distance = -(n @ source.pose.center + d) / denominator
            point = source.pose.center + distance * ray
            for observation in observations[1:]:
                target = views[observation.view_id]
                observed = erp_pixels_to_rays(
                    np.array(observation.pixel_xy)[None], target.support.shape
                )[0]
                held_angular.append(
                    float(
                        _angles(target.pose.R @ (point - target.pose.center), observed)
                    )
                )
        state = "accepted" if len(accepted_views) >= 2 else "provisional"
        reason = None
        if condition > 1e10 or len(train_inliers) < 3:
            state, reason = "needs-review", "ill-conditioned-plane"
        if held_rmse is not None and (
            held_rmse > options.plane_threshold_m
            or max(held_angular, default=0) > options.maximum_reprojection_deg
        ):
            state, reason = "needs-review", "heldout-validation-failed"
        diagnostics = {
            "track_count": len(tracks),
            "inlier_count": len(train_inliers),
            "fit_rmse_m": float(np.sqrt(np.mean(residuals[train_inliers] ** 2))),
            "heldout_rmse_m": held_rmse,
            "heldout_max_reprojection_deg": max(held_angular, default=None),
            "condition": condition,
            "accepted_views": sorted(accepted_views),
            "reason": reason,
            "uncertainty": "linearized triangulation noise; fixed poses; excludes mask/pose/systematic errors",
        }
        estimates.append(
            PlaneEstimate(
                plane_id,
                readonly(n),
                float(d),
                readonly(covariance),
                readonly(basis),
                state,
                diagnostics,
                tuple(tracks[i].correspondence.track_id for i in train_inliers),
                tuple(tracks[i].correspondence.track_id for i in held),
                covariance_origin_world=readonly(covariance_origin),
            )
        )
    return replace(workspace, planes=tuple(estimates))


def intersect_plane_pixels(view, plane, pixels, options):
    rays = erp_pixels_to_rays(pixels, view.support.shape) @ view.pose.R
    normal, offset = plane.normal_world, plane.offset_m
    denominator = rays @ normal
    numerator = -(normal @ view.pose.center + offset)
    distances = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=np.abs(denominator) >= options.minimum_incidence,
    )
    points = view.pose.center + distances[..., None] * rays
    safe_denominator = np.where(
        np.abs(denominator) >= options.minimum_incidence, denominator, np.nan
    )
    jacobian = np.concatenate(
        (
            -np.einsum(
                "...i,ij->...j",
                points
                - (
                    plane.covariance_origin_world
                    if plane.covariance_origin_world is not None
                    else 0
                ),
                plane.tangent_basis,
            )
            / safe_denominator[..., None],
            -1 / safe_denominator[..., None],
        ),
        axis=-1,
    )
    variance = np.einsum("...i,ij,...j->...", jacobian, plane.covariance, jacobian)
    sigma = np.sqrt(np.maximum(variance, 0))
    valid = (
        np.isfinite(distances)
        & np.isfinite(sigma)
        & (distances >= options.minimum_range_m)
        & (distances <= options.maximum_range_m)
        & (sigma <= options.maximum_range_sigma_m)
    )
    return distances, sigma, valid


def propagate_workspace(workspace, options):
    views = {v.view_id: v for v in workspace.scene.views}
    existing = {(r.plane_id, r.view_id) for r in workspace.scene.regions}
    proposals = []
    for plane in workspace.planes:
        if plane.normal_world is None or plane.state == "needs-review":
            continue
        regions = [
            r
            for r in workspace.scene.regions
            if r.plane_id == plane.plane_id and r.state == "accepted"
        ]
        for target in workspace.scene.views:
            if (plane.plane_id, target.view_id) in existing:
                continue
            h, w = target.support.shape
            projected_mask = np.zeros((h, w), dtype=bool)
            # Inverse warp: test target plane intersections against each source
            # mask, avoiding holes caused by forward point splatting.
            for row in range(0, h, options.block_rows):
                yy, xx = np.mgrid[row : min(h, row + options.block_rows), :w]
                pixels = np.stack((xx, yy), axis=-1).astype(np.float64)
                distances, _, valid = intersect_plane_pixels(
                    target, plane, pixels, options
                )
                rays = erp_pixels_to_rays(pixels, (h, w)) @ target.pose.R
                points = target.pose.center + distances[..., None] * rays
                local = np.zeros(xx.shape, bool)
                for region in regions:
                    source = views[region.view_id]
                    vectors = (points - source.pose.center) @ source.pose.R.T
                    safe = np.where(np.isfinite(vectors), vectors, 0)
                    source_pixels = rays_to_erp_pixels(
                        safe, source.support.shape
                    ).pixels_xy
                    local |= sample_mask(region.mask & source.support, source_pixels)
                projected_mask[row : row + xx.shape[0]] = (
                    local & valid & target.support[row : row + xx.shape[0]]
                )
            if projected_mask.any():
                proposals.append(
                    PlanarRegion(
                        plane.plane_id,
                        target.view_id,
                        projected_mask,
                        label=regions[0].label,
                        source="plane-transfer",
                    )
                )
    return replace(workspace, proposals=tuple(proposals))


def rasterize_workspace(workspace, options):
    results = []
    plane_ids = tuple(p.plane_id for p in workspace.planes)
    planes = {p.plane_id: p for p in workspace.planes if p.state == "accepted"}
    for view in workspace.scene.views:
        h, w = view.support.shape
        ranges = np.full((h, w), np.nan, np.float32)
        sigma_map = ranges.copy()
        index = np.full((h, w), -1, np.int32)
        conflict = np.zeros((h, w), bool)
        valid = np.zeros((h, w), bool)
        for region in workspace.scene.regions:
            plane = planes.get(region.plane_id)
            if (
                region.view_id != view.view_id
                or region.state != "accepted"
                or plane is None
            ):
                continue
            for row in range(0, h, options.block_rows):
                end = min(h, row + options.block_rows)
                ys, xs = np.where(region.mask[row:end] & view.support[row:end])
                if not len(ys):
                    continue
                ys += row
                pixels = np.stack((xs, ys), axis=-1).astype(np.float64)
                distance, sigma, supported = intersect_plane_pixels(
                    view, plane, pixels, options
                )
                previous = valid[ys, xs]
                # Two accepted masks on one pixel require explicit human repair;
                # nearest-plane choice alone does not certify visibility.
                conflict[ys, xs] |= previous & supported
                write = supported & ~previous
                y, x = ys[write], xs[write]
                ranges[y, x], sigma_map[y, x] = distance[write], sigma[write]
                index[y, x] = plane_ids.index(plane.plane_id)
                valid[y, x] = True
        valid &= ~conflict
        ranges[~valid] = np.nan
        sigma_map[~valid] = np.nan
        index[~valid] = -1
        if view.baseline_range is None:
            composed = np.full((h, w), np.nan, np.float32)
            composed_valid = np.zeros((h, w), bool)
        else:
            composed = np.array(view.baseline_range, copy=True)
            composed_valid = view.baseline_validity.copy()
        source = np.where(composed_valid, 1, 0).astype(np.uint8)
        composed[valid], composed_valid[valid], source[valid] = ranges[valid], True, 2
        results.append(
            PlanarRangeResult(
                view.view_id,
                *(
                    readonly(a)
                    for a in (
                        ranges,
                        valid,
                        sigma_map,
                        index,
                        conflict,
                        composed,
                        composed_valid,
                        source,
                    )
                ),
                plane_ids,
            )
        )
    return replace(workspace, ranges=tuple(results))
