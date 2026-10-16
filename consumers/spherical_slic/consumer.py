#!/usr/bin/env python3
"""Installed-wheel consumer for the SphericalSLIC application."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np


def _assert_installed_origin(package_file: str, source_root: Path) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return origin
    raise AssertionError(f"panorai resolved inside the source checkout: {origin}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    import panorai
    from panorai.applications import Application, SphericalSLIC
    from panorai.pipeline import RasterField

    origin = _assert_installed_origin(panorai.__file__, args.source_root)
    y, x = np.indices((32, 64), dtype=np.float32)
    image = np.stack(
        (x / 63.0, y / 31.0, ((x // 8 + y // 8) % 2).astype(np.float32)),
        axis=-1,
    )
    valid = np.ones((32, 64), dtype=bool)
    valid[10:14, 20:27] = False
    application = SphericalSLIC(n_segments=16, max_num_iter=3)
    result = application(image, validity_mask=valid)

    assert application.interface == "panorai-spherical-slic/v1"
    assert application.stability == "stable"
    assert isinstance(result, RasterField)
    assert result.kind == "labels"
    assert result.grid.is_canonical_erp
    assert result.values.shape == image.shape[:2]
    assert np.array_equal(result.validity, valid)
    assert np.all(result.values[result.validity] >= 1)
    assert type(application.with_params(n_segments=12)) is SphericalSLIC

    with tempfile.TemporaryDirectory(prefix="panorai-slic-config-") as directory:
        path = application.save(Path(directory) / "slic.json")
        restored = Application.load(path)
        assert restored == application
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == "panorai-spherical-slic-config/v1"

    print(
        json.dumps(
            {
                "status": "ok",
                "consumer": "spherical-slic",
                "version": version("panorai"),
                "origin": str(origin),
                "shape_hw": list(result.values.shape),
                "label_count": int(np.unique(result.values[result.validity]).size),
                "valid_pixels": int(result.validity.sum()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
