"""Dataset-neutral loader for organized RGB plus XYZ rasters stored as PLY."""

from __future__ import annotations

from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

_SHAPE_SUFFIX = re.compile(r"_(\d+)x(\d+)(?:_encrypted)?\.ply$")


def extract_hw_from_ply_filename(filename: str | Path) -> tuple[int, int]:
    """Read the organized raster shape from a ``*_HxW[_encrypted].ply`` name."""

    match = _SHAPE_SUFFIX.search(Path(filename).name)
    if match is None:
        raise ValueError(f"filename does not contain an _HxW PLY suffix: {filename}")
    return int(match.group(1)), int(match.group(2))


def _organized_arrays(
    point_cloud: Any, shape_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape_hw
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = np.asarray(point_cloud.colors, dtype=np.float64)
    expected = height * width
    if points.shape != (expected, 3) or colors.shape != (expected, 3):
        raise ValueError(
            "organized PLY must contain exactly H*W XYZ points and RGB colors"
        )
    xyz_image = points.reshape(height, width, 3)
    rgb_image = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8).reshape(
        height, width, 3
    )
    return xyz_image, rgb_image


def read_ply_xyz_image(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz_image, rgb_image)`` from one organized plaintext PLY."""

    path = Path(filename)
    point_cloud = _read_point_cloud(path)
    return _organized_arrays(point_cloud, extract_hw_from_ply_filename(path))


def read_encrypted_ply_xyz_image(
    filename: str | Path, cipher: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Decrypt one organized PLY to a temporary file and return its rasters."""

    if cipher is None:
        raise ValueError("cipher is required for an encrypted PLY")
    path = Path(filename)
    decrypted = cipher.decrypt(path.read_bytes())
    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as temporary:
        temporary.write(decrypted)
        temporary.flush()
        point_cloud = _read_point_cloud(Path(temporary.name))
    return _organized_arrays(point_cloud, extract_hw_from_ply_filename(path))


def _read_point_cloud(path: Path) -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Open3D is required to read organized PLY files") from exc
    return o3d.io.read_point_cloud(str(path))


class XYZImageDataset(Dataset):
    """Load dataset-neutral organized XYZ/RGB images.

    The consumer supplies either ``root`` or an explicit ordered ``files``
    iterable. PanorAi does not infer train/validation/test membership from file
    names. Each sample declares its XYZ frame, units, validity, radial depth,
    and optional shadow angle. Dataset licensing and split construction remain
    the consuming application's responsibility.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        files: Iterable[str | Path] | None = None,
        cipher: Any | None = None,
        transform: Any | None = None,
        transform_uses_cache_envelope: bool = False,
        n_angles: int = 1,
        coordinate_frame: str,
        units: str = "metres",
        shadow_angle: float = 0.0,
    ) -> None:
        if (root is None) == (files is None):
            raise ValueError("provide exactly one of root or files")
        if not coordinate_frame.strip():
            raise ValueError("coordinate_frame must be explicit")
        if not units.strip():
            raise ValueError("units must be explicit")
        if n_angles < 1:
            raise ValueError("n_angles must be positive")
        if files is None:
            assert root is not None
            selected = sorted(Path(root).glob("*.ply"))
        else:
            selected = [Path(item) for item in files]
        if not selected:
            raise ValueError("no organized PLY files were provided")
        self.files = tuple(selected)
        self.cipher = cipher
        self.transform = transform
        self.transform_uses_cache_envelope = bool(transform_uses_cache_envelope)
        self.n_angles = int(n_angles)
        self.coordinate_frame = coordinate_frame
        self.units = units
        self.shadow_angle = float(shadow_angle)

    def _load_sample(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        if "encrypted" in path.stem:
            xyz_image, rgb_image = read_encrypted_ply_xyz_image(path, self.cipher)
        else:
            xyz_image, rgb_image = read_ply_xyz_image(path)
        radial_depth = np.linalg.norm(xyz_image, axis=-1)
        validity = np.isfinite(xyz_image).all(axis=-1) & (radial_depth > 0.0)
        return {
            "rgb_image": rgb_image,
            "xyz_image": xyz_image,
            "radial_depth": radial_depth,
            "validity_mask": validity,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "shadow_angle": self.shadow_angle,
            "image_path": str(path),
        }

    def __getitem__(self, index: int) -> Any:
        path = self.files[index]
        angle_index = random.randrange(self.n_angles)
        flip = random.random() < 0.5
        sample = self._load_sample(index)
        if self.transform is not None and self.transform_uses_cache_envelope:
            return self.transform(
                {
                    "key": str(path),
                    "data": sample,
                    "angle_idx": angle_index,
                    "flip": flip,
                }
            )
        if self.transform is None:
            return sample
        return self.transform(sample)

    def __len__(self) -> int:
        return len(self.files)


__all__ = [
    "XYZImageDataset",
    "extract_hw_from_ply_filename",
    "read_encrypted_ply_xyz_image",
    "read_ply_xyz_image",
]
