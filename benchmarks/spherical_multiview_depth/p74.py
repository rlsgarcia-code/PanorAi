"""P74 native-polar image adapter kept outside the reusable PanorAi API."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


P74_POLAR_LIMIT_RAD = math.radians(150.0)


def load_native_angular_rgb(
    rgb_path: Path,
    output_shape_hw: tuple[int, int],
    *,
    row_chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Regrid an endpoint-inclusive P74 polar raster to a canonical ERP.

    OpenCV is imported only for a real benchmark execution. Importing the
    benchmark geometry regressions therefore does not add it to PanorAi's
    runtime dependency surface.
    """

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional benchmark runtime
        raise ImportError(
            "P74 benchmark image loading requires panorai[features]"
        ) from error
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {rgb_path}")
    source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_height, source_width = source.shape[:2]
    period = source_width - 1
    height, width = output_shape_hw
    longitude = (np.arange(width, dtype=np.float64) + 0.5) / width * (
        2.0 * math.pi
    ) - math.pi
    map_x_row = np.mod(-longitude / (2.0 * math.pi) * period, period)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    support = np.empty((height, width), dtype=bool)
    shadow = np.asarray([124, 116, 104], dtype=np.uint8)
    for start in range(0, height, row_chunk):
        stop = min(height, start + row_chunk)
        rows = np.arange(start, stop, dtype=np.float64)
        polar = (rows + 0.5) / height * math.pi
        row_support = polar <= P74_POLAR_LIMIT_RAD + 1e-12
        map_x = np.broadcast_to(map_x_row[None], (stop - start, width)).astype(
            np.float32
        )
        map_y = np.broadcast_to(
            (polar / P74_POLAR_LIMIT_RAD * (source_height - 1))[:, None],
            map_x.shape,
        ).astype(np.float32)
        chunk = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        chunk[~row_support] = shadow
        rgb[start:stop] = chunk
        support[start:stop] = row_support[:, None]
    return rgb, support


__all__ = ["P74_POLAR_LIMIT_RAD", "load_native_angular_rgb"]
