"""Smoke the Workbench against an installed PanorAi outside its source tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forbid-source-root",
        type=Path,
        required=True,
        help="PanorAi source root that must not contain the imported module",
    )
    args = parser.parse_args()

    import panorai
    from panorai.geometry import GnomonicProjector, GnomonicSpec
    from panorai_workbench.catalog import experiment_catalog, runtime_fingerprint

    source_root = args.forbid_source_root.resolve()
    module_path = Path(panorai.__file__).resolve()
    if module_path == source_root or source_root in module_path.parents:
        raise SystemExit(
            f"isolation failed: panorai resolved inside source checkout: {module_path}"
        )

    image = np.linspace(0.0, 1.0, 16 * 32 * 3, dtype=np.float32).reshape(
        16, 32, 3
    )
    spec = GnomonicSpec(
        center_lon_deg=0.0,
        center_lat_deg=0.0,
        hfov_deg=90.0,
        vfov_deg=70.0,
        output_shape_hw=(12, 16),
    )
    result = GnomonicProjector(spec).project(image)
    if result.data.shape != (12, 16, 3) or not result.support_mask.all():
        raise SystemExit("installed-wheel gnomonic smoke returned an invalid result")

    catalog = experiment_catalog()
    runtime = runtime_fingerprint()
    print(f"panorai_version={panorai.__version__}")
    print(f"panorai_module={module_path}")
    print(f"installation_type={runtime['panorai']['installation_type']}")
    print(f"catalog_release={catalog['catalog']['release']}")
    print(f"catalog_surfaces={catalog['counts']['surfaces']}")
    print(f"workbench_experiments={catalog['counts']['experiments']}")
    print("gnomonic_shape=12x16x3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
