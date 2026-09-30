# Changelog

PanorAi follows semantic versioning and uses `vX.Y.Z` release tags.

## 3.1.0 — 2026-09-30

### Added

- Stable `panorai.geometry` API for ERP pixels, Cartesian rays, rectangular
  gnomonic projections, and canonical cubemaps.
- Immutable `GnomonicProjector` and `CubemapProjector` interfaces.
- NumPy `HW`/`HWC` and optional Torch `HW`/`CHW`/`NCHW` backends.
- Explicit support masks and Torch gradients with respect to input values.
- Named bilinear invalid-data policies: strict `propagate` by default and
  opt-in mask-authoritative `renormalize` with reported validity and weight.
- Python 3.10, 3.11, and 3.12 release gates.

### Changed

- `to_gnomonic(..., x_points=..., y_points=...)` now honors the requested
  output resolution.
- The legacy gnomonic center now uses its analytic latitude/longitude limit
  instead of passing a platform-dependent `NaN` coordinate to OpenCV.
- New container calls using `spec=GnomonicSpec(...)` use the canonical engine;
  legacy calls retain their 3.0 numerical convention outside that previously
  undefined center sample.
- Package metadata and dependencies now live in `pyproject.toml`.
- Torch, Open3D, point-cloud and depth dependencies are optional.
- Versions are derived from `vX.Y.Z` tags with `setuptools-scm`.
- Blender validity now comes exclusively from explicit masks; valid zero/black
  samples are preserved, unsupported non-zero samples are ignored, and valid
  non-finite samples fail explicitly.
- Supported fusion blenders preserve input shape and accept
  `return_mask=True`; Gnomonic face-set reconstruction retains its fused
  support mask.
- Gaussian blending uses the real gnomonic back-projection interface, and
  Huber blending now supports both scalar and channel inputs without forcing a
  three-channel result.
- Depth loader names and registry keys are lightweight lazy adapters to
  separately installed upstream projects. Deep vendored model namespaces and
  research training/data helpers are classified internal and are no longer
  distributed; PCD names and container `to_pcd` methods remain available.

### Packaging

- Wheel and sdist contain normal Python source, without PyArmor or generated
  stubs.
- Wheel and sdist exclude vendored Depth Anything V2, DUSt3R/CroCo, Metric3D,
  legacy ZoeDepth, datasets, and training helpers. The resulting PanorAi code
  remains MIT; upstream adapters report their separate license requirements.
- A single release workflow builds artifacts once, verifies them on TestPyPI,
  and promotes those same files to PyPI after environment approval.

## 3.0.19

- Last public release before the stable geometry API.
