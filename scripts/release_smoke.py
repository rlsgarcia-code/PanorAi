#!/usr/bin/env python3
"""Installed-package smoke test for wheels and TestPyPI."""

from __future__ import annotations

import argparse
from importlib import resources
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def assert_installed_origin(
    package_file: str,
    *,
    source_root: Path,
    package_version: str,
    expected_version: str | None = None,
) -> Path:
    """Prove that an import came from an installed distribution, not the checkout."""

    origin = Path(package_file).resolve()
    source_root = source_root.resolve()
    try:
        origin.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"panorai resolved inside the source checkout: {origin} (root={source_root})"
        )

    distribution_version = version("panorai")
    assert package_version == distribution_version, (
        f"package/metadata version mismatch: {package_version!r} != "
        f"{distribution_version!r}"
    )
    if expected_version is not None:
        assert distribution_version == expected_version, (
            f"unexpected installed version: {distribution_version!r} != "
            f"{expected_version!r}"
        )
    return origin


def assert_native_estimator() -> None:
    """Require the compiled essential-estimation backend and exercise it."""

    from panorai.estimators import (
        native_kernels_available,
        solve_five_point_essential,
    )

    assert native_kernels_available(), "compiled estimator kernels are unavailable"
    points = np.asarray(
        (
            (-1.0, -0.5, 4.0),
            (0.5, -0.8, 5.0),
            (1.2, 0.4, 6.0),
            (-0.3, 0.9, 7.0),
            (0.8, 1.1, 8.0),
        ),
        dtype=np.float64,
    )
    angle = np.deg2rad(4.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    first = points / np.linalg.norm(points, axis=1, keepdims=True)
    transformed = points @ rotation.T + np.asarray((0.8, 0.1, 0.05))
    second = transformed / np.linalg.norm(transformed, axis=1, keepdims=True)
    solutions = solve_five_point_essential(first, second, backend="native")
    assert solutions, "native five-point solver produced no solution"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true")
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="fail if panorai resolves from the source checkout",
    )
    parser.add_argument(
        "--source-checkout",
        action="store_true",
        help="explicitly import panorai from this checkout for source-only checks",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="checkout path that must not provide the installed import",
    )
    parser.add_argument("--expected-version")
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="fail unless the compiled estimator kernels load and execute",
    )
    args = parser.parse_args()
    if args.source_checkout and args.require_installed:
        parser.error("--source-checkout and --require-installed are mutually exclusive")
    if args.source_checkout and str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    torch_was_loaded = "torch" in sys.modules
    open3d_was_loaded = "open3d" in sys.modules

    import panorai
    import panorai.depth as depth
    import panorai.pcd as pcd
    from panorai.geometry import GnomonicSpec, equirectangular_to_gnomonic

    if args.require_installed:
        origin = assert_installed_origin(
            panorai.__file__,
            source_root=args.source_root,
            package_version=panorai.__version__,
            expected_version=args.expected_version,
        )
    else:
        assert panorai.__version__ == version("panorai")
        origin = Path(panorai.__file__).resolve()

    assert ("torch" in sys.modules) is torch_was_loaded
    assert ("open3d" in sys.modules) is open3d_was_loaded
    assert set(depth.ModelRegistry.list_models()) == {
        "dav2",
        "dust3r",
        "m3dv2",
        "zoe",
    }
    for loader_name in (
        "load_dav2_model",
        "load_dust3r_model",
        "load_m3dv2_model",
        "load_zoe_model",
    ):
        assert callable(getattr(depth, loader_name))
    assert pcd.__name__ == "panorai.pcd"
    contract = resources.files("panorai.geometry").joinpath("geometry-v1.yaml")
    assert "contract: geometry-v1" in contract.read_text(encoding="utf-8")

    spec = GnomonicSpec(hfov_deg=90, vfov_deg=60, output_shape_hw=(7, 11))
    image = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    result = equirectangular_to_gnomonic(image, spec)
    assert result.data.shape == (7, 11)
    assert result.support_mask.all()

    if args.require_native:
        assert_native_estimator()

    if args.torch:
        import torch

        tensor = torch.from_numpy(image).requires_grad_(True)
        projected = equirectangular_to_gnomonic(tensor, spec)
        projected.data.sum().backward()
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0

    print(f"release smoke: OK version={panorai.__version__} origin={origin}")


if __name__ == "__main__":
    main()
