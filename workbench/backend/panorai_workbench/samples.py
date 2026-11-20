"""Redistributable Workbench samples with explicit geometric provenance."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ERP = REPOSITORY_ROOT / "docs/_static/tutorials/nature-reserve-forest-erp.jpg"
STEREO_TEXTURE = REPOSITORY_ROOT / "docs/_static/tutorials/poly-haven-studio-erp.jpg"

PAIR_ID = "poly-haven-studio-known-pose-v1"
PAIR_SHAPE_HW = (256, 512)
PAIR_X_OFFSET_METERS = 0.35
PAIR_YAW_DEGREES = 6.0
PAIR_CENTER_B_IN_A = np.asarray((PAIR_X_OFFSET_METERS, 0.03, 0.12), dtype=np.float64)
KNOWN_HEIGHT_ROOM_ID = "known-height-room-v1"
KNOWN_HEIGHT_ROOM_SHAPE_HW = (256, 512)
KNOWN_HEIGHT_CAMERA_M = 1.6


def _rotation_b_from_a() -> np.ndarray:
    angle = np.deg2rad(PAIR_YAW_DEGREES)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, 0.0, -sine), (0.0, 1.0, 0.0), (sine, 0.0, cosine)),
        dtype=np.float64,
    )


def _pair_pose() -> dict:
    rotation = _rotation_b_from_a()
    translation = -(rotation @ PAIR_CENTER_B_IN_A)
    translation_direction = translation / np.linalg.norm(translation)
    return {
        "convention": "x_b = R_b_from_a @ x_a + t_b_from_a",
        "frame": "PanorAi canonical camera frame: +X right, +Y up, +Z forward",
        "rotation_b_from_a": rotation.tolist(),
        "translation_direction_b": translation_direction.tolist(),
        "translation_b_from_a_m": translation.tolist(),
        "center_b_in_a_m": PAIR_CENTER_B_IN_A.tolist(),
        "baseline_m": float(np.linalg.norm(PAIR_CENTER_B_IN_A)),
        "yaw_b_from_a_deg": PAIR_YAW_DEGREES,
    }


def sample_catalog() -> dict:
    """Return local examples that are safe to load without remote I/O."""

    return {
        "schema": "panorai-workbench-samples/v1",
        "samples": [
            {
                "id": "nature-reserve-forest-cc0",
                "name": "Nature Reserve Forest",
                "filename": "nature-reserve-forest-erp.jpg",
                "kind": "erp-rgb",
                "shape_hw": [512, 1024],
                "default_for": ["single"],
                "url": "/api/v1/samples/nature-reserve-forest-cc0/image",
                "license": "CC0-1.0",
                "license_url": "https://polyhaven.com/license",
                "source_url": "https://polyhaven.com/a/nature_reserve_forest",
                "attribution": (
                    "Nature Reserve Forest · Dimitrios Savva and Jarod Guest · "
                    "Poly Haven"
                ),
            },
            {
                "id": "poly-haven-studio-cc0",
                "name": "Poly Haven Studio",
                "filename": "poly-haven-studio-erp.jpg",
                "kind": "erp-rgb",
                "shape_hw": [512, 1024],
                "default_for": ["known-height-planar-real"],
                "url": "/api/v1/samples/poly-haven-studio-cc0/image",
                "license": "CC0-1.0",
                "license_url": "https://polyhaven.com/license",
                "source_url": "https://polyhaven.com/a/poly_haven_studio",
                "attribution": "Poly Haven Studio · Greg Zaal · Poly Haven",
            },
            {
                "id": KNOWN_HEIGHT_ROOM_ID,
                "name": "Sala planar sintética · altura conhecida",
                "filename": "known-height-room.png",
                "kind": "erp-rgb-known-height-planar",
                "shape_hw": list(KNOWN_HEIGHT_ROOM_SHAPE_HW),
                "default_for": ["known-height-planar-mapping"],
                "url": f"/api/v1/samples/{KNOWN_HEIGHT_ROOM_ID}/image.png",
                "scene_url": f"/api/v1/samples/{KNOWN_HEIGHT_ROOM_ID}/scene",
                "license": "MIT",
                "license_url": "https://github.com/rlsgarcia-code/PanorAi/blob/main/LICENSE",
                "source_url": "generated-by-panorai-workbench",
                "attribution": "Fixture analítico gerado localmente pelo PanorAi",
            },
        ],
        "pairs": [
            {
                "id": PAIR_ID,
                "name": "Poly Haven Studio · pose conhecida",
                "kind": "erp-rgb-pair",
                "shape_hw": list(PAIR_SHAPE_HW),
                "default_for": ["two"],
                "image_a_url": f"/api/v1/samples/{PAIR_ID}/a.png",
                "image_b_url": f"/api/v1/samples/{PAIR_ID}/b.png",
                "planar_scene_url": f"/api/v1/samples/{PAIR_ID}/planar-scene",
                "filename_a": "studio-known-pose-a.png",
                "filename_b": "studio-known-pose-b.png",
                "pose": _pair_pose(),
                "scene": {
                    "model": "CC0 ERP texture mapped to an analytic sphere",
                    "sphere_radius_m": 4.0,
                    "sphere_center_in_a_m": [0.6, -0.25, 0.35],
                },
                "license": "CC0-1.0",
                "license_url": "https://polyhaven.com/license",
                "source_url": "https://polyhaven.com/a/poly_haven_studio",
                "attribution": "Poly Haven Studio · Greg Zaal · Poly Haven",
            }
        ],
    }


def known_pose_planar_scene() -> dict:
    """Return an empty, registered two-view scene ready for manual planes."""

    rotation = _rotation_b_from_a()
    return {
        "schema": "panorai-planar-scene/v1",
        "geometry_convention": "panorai-geometry-v1",
        "units": "m",
        "views": [
            {
                "view_id": "a",
                "image_key": "studio-known-pose-a.png",
                "rotation": np.eye(3, dtype=np.float64).tolist(),
                "center_world": [0.0, 0.0, 0.0],
                "pose_convention": "world_to_camera",
                "support": "full",
            },
            {
                "view_id": "b",
                "image_key": "studio-known-pose-b.png",
                "rotation": rotation.tolist(),
                "center_world": PAIR_CENTER_B_IN_A.tolist(),
                "pose_convention": "world_to_camera",
                "support": "full",
            },
        ],
        "regions": [],
        "correspondences": [],
        "provenance": {
            "sample_id": PAIR_ID,
            "pose": _pair_pose(),
            "license": "CC0-1.0",
            "source": "https://polyhaven.com/a/poly_haven_studio",
        },
    }


@lru_cache(maxsize=1)
def known_height_room_sample() -> tuple[bytes, dict]:
    """Render a deterministic axis-aligned room and its independent oracle."""

    from panorai.reconstruction.planar import encode_planar_mask

    shape = KNOWN_HEIGHT_ROOM_SHAPE_HW
    height, width = shape
    y, x = np.indices(shape, dtype=np.float64)
    rays = np.asarray(erp_pixels_to_rays(np.stack((x, y), axis=-1), shape))
    camera = np.asarray((0.0, KNOWN_HEIGHT_CAMERA_M, 0.0), dtype=np.float64)
    bounds_low = np.asarray((-3.0, 0.0, -4.0))
    bounds_high = np.asarray((3.0, 3.0, 5.0))
    candidates = []
    faces = []
    for axis, bound, face in (
        (0, bounds_low[0], "left"),
        (0, bounds_high[0], "right"),
        (1, bounds_low[1], "floor"),
        (1, bounds_high[1], "ceiling"),
        (2, bounds_low[2], "back"),
        (2, bounds_high[2], "front"),
    ):
        denominator = rays[..., axis]
        distance = np.divide(
            bound - camera[axis],
            denominator,
            out=np.full(shape, np.inf),
            where=np.abs(denominator) > 1e-9,
        )
        points = camera + distance[..., None] * rays
        inside = (distance > 0) & np.all(
            (points >= bounds_low - 1e-8) & (points <= bounds_high + 1e-8),
            axis=-1,
        )
        candidates.append(np.where(inside, distance, np.inf))
        faces.append(face)
    distances = np.stack(candidates)
    face_index = np.argmin(distances, axis=0)
    radial = np.min(distances, axis=0)
    points = camera + radial[..., None] * rays
    palette = np.asarray(
        (
            (138, 119, 96),
            (121, 107, 91),
            (126, 135, 112),
            (188, 183, 164),
            (105, 125, 139),
            (165, 146, 115),
        ),
        dtype=np.float64,
    )
    image = palette[face_index]
    checker = (
        np.floor(points[..., 0] * 2)
        + np.floor(points[..., 1] * 2)
        + np.floor(points[..., 2] * 2)
    ).astype(np.int64)
    image *= np.where(checker[..., None] % 2 == 0, 1.0, 0.82)
    # High-contrast metric guides improve visual orientation without becoming
    # support or validity evidence.
    guides = (
        (np.abs(points[..., 0] - np.round(points[..., 0])) < 0.018)
        | (np.abs(points[..., 1] - np.round(points[..., 1])) < 0.018)
        | (np.abs(points[..., 2] - np.round(points[..., 2])) < 0.018)
    )
    image[guides] *= 0.55
    output = BytesIO()
    Image.fromarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8)).save(
        output, format="PNG"
    )

    masks = {name: face_index == index for index, name in enumerate(faces)}
    floor = masks["floor"]
    front = masks["front"]
    ceiling = masks["ceiling"]
    seed = floor & (np.abs(points[..., 0]) < 0.35) & (points[..., 2] > 0.8)
    horizon_rays = np.asarray(
        ((1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    horizon_rays /= np.linalg.norm(horizon_rays, axis=1, keepdims=True)
    horizon = np.asarray(rays_to_erp_pixels(horizon_rays, shape).pixels_xy).tolist()
    floor_contact = np.asarray(
        rays_to_erp_pixels(
            np.asarray(((-2.5, 0.0, 5.0), (2.5, 0.0, 5.0))) - camera,
            shape,
        ).pixels_xy
    ).tolist()
    ceiling_contact = np.asarray(
        rays_to_erp_pixels(
            np.asarray(((-2.5, 3.0, 5.0), (2.5, 3.0, 5.0))) - camera,
            shape,
        ).pixels_xy
    ).tolist()

    def region(plane_id: str, mask: np.ndarray, state: str, interpretation: str):
        payload = {
            "plane_id": plane_id,
            "view_id": "single-view",
            "label": {
                "floor": "Piso",
                "front": "Parede frontal",
                "ceiling": "Teto",
            }.get(plane_id, plane_id),
            "state": state,
            "mask": encode_planar_mask(mask),
            "interpretation": interpretation,
            "history": [{"source": "analytic-known-height-room-v1"}],
            "source": "analytic-sample",
        }
        if interpretation == "seed":
            payload["positive"] = encode_planar_mask(mask)
        return payload

    common = {
        "schema": "panorai-known-height-planar-scene/v1",
        "scene_id": KNOWN_HEIGHT_ROOM_ID,
        "camera_height_m": KNOWN_HEIGHT_CAMERA_M,
        "camera_height_sigma_m": 0.01,
        "down_erp_xy": [width / 2 - 0.5, height - 1.0],
        "horizon_erp_xy": horizon,
        "provenance": {
            "fixture": "axis-aligned analytic room",
            "frame": "+Y up, camera centre [0,1.6,0] m",
            "bounds_m": {"low": bounds_low.tolist(), "high": bounds_high.tolist()},
        },
    }
    starter = {
        **common,
        "revision": "starter-v1",
        "regions": [region("floor", seed, "draft", "seed")],
        "constraints": [
            {
                "constraint_id": "floor-anchor",
                "plane_id": "floor",
                "kind": "floor-anchor",
                "reference_plane_id": None,
                "contact_erp_xy": [],
                "state": "proposed",
                "source": "analytic-sample",
            }
        ],
    }
    reference = {
        **common,
        "revision": "reference-v1",
        "regions": [
            region("floor", floor, "accepted", "final"),
            region("wall", front, "accepted", "final"),
            region("ceiling", ceiling, "accepted", "final"),
        ],
        "constraints": [
            {
                "constraint_id": "floor-anchor",
                "plane_id": "floor",
                "kind": "floor-anchor",
                "reference_plane_id": None,
                "contact_erp_xy": [],
                "state": "accepted",
                "source": "analytic-sample",
            },
            {
                "constraint_id": "wall-floor",
                "plane_id": "wall",
                "kind": "vertical-to-floor",
                "reference_plane_id": "floor",
                "contact_erp_xy": floor_contact,
                "state": "accepted",
                "source": "analytic-sample",
            },
            {
                "constraint_id": "ceiling-wall",
                "plane_id": "ceiling",
                "kind": "horizontal-to-plane",
                "reference_plane_id": "wall",
                "contact_erp_xy": ceiling_contact,
                "state": "accepted",
                "source": "analytic-sample",
            },
        ],
    }
    document = {
        "schema": "panorai-workbench-known-height-sample/v1",
        "image_url": f"/api/v1/samples/{KNOWN_HEIGHT_ROOM_ID}/image.png",
        "starter": starter,
        "reference": reference,
    }
    return output.getvalue(), document


def _sample_texture(texture: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    """Sample RGB texture bilinearly, wrapping only the ERP seam."""

    height, width = texture.shape[:2]
    x = np.mod(pixels_xy[..., 0], width)
    y = np.clip(pixels_xy[..., 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = texture[y0, x0] * (1.0 - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (1.0 - wx) + texture[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _render_sphere(
    center_in_a: np.ndarray,
    rays_in_a: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    sphere_center = np.asarray((0.6, -0.25, 0.35), dtype=np.float64)
    radius = 4.0
    offset = center_in_a - sphere_center
    projection = np.sum(rays_in_a * offset, axis=-1)
    discriminant = projection * projection + radius * radius - float(offset @ offset)
    radial_range = -projection + np.sqrt(np.maximum(discriminant, 0.0))
    points = center_in_a + radial_range[..., None] * rays_in_a
    texture_rays = (points - sphere_center) / radius
    texture_pixels = rays_to_erp_pixels(texture_rays, texture.shape[:2]).pixels_xy
    return _sample_texture(texture, np.asarray(texture_pixels))


@lru_cache(maxsize=1)
def known_pose_pair_pngs() -> tuple[bytes, bytes]:
    """Render a deterministic stereo ERP pair through public PanorAi geometry."""

    if not STEREO_TEXTURE.is_file():
        raise FileNotFoundError(STEREO_TEXTURE)
    with Image.open(STEREO_TEXTURE) as source:
        texture = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.0

    height, width = PAIR_SHAPE_HW
    y, x = np.indices((height, width), dtype=np.float64)
    camera_rays = np.asarray(
        erp_pixels_to_rays(np.stack((x, y), axis=-1), PAIR_SHAPE_HW)
    )
    rotation_b_from_a = _rotation_b_from_a()
    rays_b_in_a = camera_rays @ rotation_b_from_a
    image_a = _render_sphere(np.zeros(3), camera_rays, texture)
    image_b = _render_sphere(PAIR_CENTER_B_IN_A, rays_b_in_a, texture)

    payloads: list[bytes] = []
    for image in (image_a, image_b):
        encoded = Image.fromarray(
            np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        buffer = BytesIO()
        encoded.save(buffer, format="PNG", optimize=True)
        payloads.append(buffer.getvalue())
    return payloads[0], payloads[1]
