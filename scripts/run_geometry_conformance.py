#!/usr/bin/env python3
"""Run the public geometry-v1 fixtures without pytest."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from verify_geometry_fixture_integrity import verify_fixture_integrity

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_ROOT = REPOSITORY_ROOT / "tests/fixtures/geometry/v1"


def _load(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def _nan_array(value, dtype):
    def decode(item):
        if isinstance(item, list):
            return [decode(child) for child in item]
        return np.nan if item == "NaN" else item

    return np.asarray(decode(value), dtype=dtype)


def _assert_installed_origin(
    package_file: str,
    *,
    source_root: Path,
    package_version: str,
    expected_version: str | None,
) -> Path:
    origin = Path(package_file).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"panorai resolved inside the source checkout: {origin} "
            f"(root={source_root.resolve()})"
        )
    distribution_version = version("panorai")
    assert package_version == distribution_version
    if expected_version is not None:
        assert distribution_version == expected_version
    return origin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-root", type=Path, default=DEFAULT_FIXTURES_ROOT)
    parser.add_argument(
        "--source-checkout",
        action="store_true",
        help="explicitly import panorai from this checkout for source-only checks",
    )
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="fail if panorai resolves from the source checkout",
    )
    parser.add_argument("--source-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    if args.source_checkout and args.require_installed:
        parser.error("--source-checkout and --require-installed are mutually exclusive")
    if args.source_checkout and str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    import panorai
    from panorai.geometry import (
        CUBE_FACE_ORDER,
        GnomonicSpec,
        equirectangular_to_cubemap,
        equirectangular_to_gnomonic,
        erp_pixels_to_rays,
    )

    if args.require_installed:
        origin = _assert_installed_origin(
            panorai.__file__,
            source_root=args.source_root,
            package_version=panorai.__version__,
            expected_version=args.expected_version,
        )
    else:
        origin = Path(panorai.__file__).resolve()

    root = args.fixtures_root.resolve()
    verify_fixture_integrity(root)
    manifest = _load(root, "manifest.json")

    analytic = _load(root, "analytic.json")
    rays = erp_pixels_to_rays(
        np.asarray(analytic["pixel_centers_xy"], dtype=np.float64),
        tuple(analytic["erp_shape_hw"]),
    )
    np.testing.assert_allclose(
        rays,
        np.asarray(analytic["expected_rays_xyz"]),
        atol=analytic["absolute_tolerance"],
    )

    cases = _load(root, "modalities.json")
    spec_args = dict(cases["spec"])
    spec_args["output_shape_hw"] = tuple(spec_args["output_shape_hw"])
    spec = GnomonicSpec(**spec_args)
    modalities = (
        ("labels_int16", "expected_labels_nearest", np.int16),
        ("mask_bool", "expected_mask_nearest", np.bool_),
        ("depth_float32", "expected_depth_nearest", np.float32),
        ("rgb_uint8", "expected_rgb_nearest", np.uint8),
    )
    for source_name, expected_name, dtype in modalities:
        source = _nan_array(cases[source_name], dtype)
        expected = _nan_array(cases[expected_name], dtype)
        actual = equirectangular_to_gnomonic(source, spec, interpolation="nearest").data
        np.testing.assert_allclose(
            actual, expected, atol=manifest["absolute_tolerance"], equal_nan=True
        )

    # Independent four-neighbour oracle: lon=lat=0 in this 2x4 panorama maps
    # to x=1.5,y=0.5. Two valid neighbours contribute 0.25 each, so q=0.5
    # and the normalized value is (2*0.25 + 4*0.25) / 0.5 = 3.
    normalized_source = np.array(
        ((2.0, 2.0, 4.0, 4.0), (2.0, 2.0, 4.0, 4.0)), dtype=np.float64
    )
    normalized_validity = np.array(
        ((False, True, True, False), (False, False, False, False)), dtype=bool
    )
    normalized = equirectangular_to_gnomonic(
        normalized_source,
        GnomonicSpec(output_shape_hw=(1, 1)),
        invalid_policy="renormalize",
        validity_mask=normalized_validity,
        min_valid_weight=0.5,
    )
    np.testing.assert_allclose(normalized.data, [[3.0]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        normalized.valid_weight, [[0.5]], atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(normalized.validity_mask, [[True]])
    np.testing.assert_array_equal(normalized.support_mask, [[True]])

    labels = np.asarray(cases["labels_int16"], dtype=np.int16)
    cubemap = equirectangular_to_cubemap(labels, (2, 2), interpolation="nearest")
    assert tuple(cubemap) == CUBE_FACE_ORDER
    for face in CUBE_FACE_ORDER:
        np.testing.assert_array_equal(
            cubemap[face].data,
            np.asarray(cases["expected_cubemap_labels_2x2"][face], dtype=np.int16),
        )
    print(f"geometry-v1 conformance: OK version={panorai.__version__} origin={origin}")


if __name__ == "__main__":
    main()
