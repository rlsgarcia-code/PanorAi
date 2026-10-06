# Changelog

PanorAi follows semantic versioning and uses `vX.Y.Z` release tags.

## 3.4.0 — 2026-10-04

### Added (Experimental)

- `panorai.stereo` direct spherical dense stereo. A supplied metric relative
  pose constrains every inverse-range hypothesis to its spherical epipolar
  locus; four-path edge-aware cost aggregation and bidirectional consistency
  return radial range, validity, confidence, matching cost and hypothesis
  indices without changing geometry-v1.
- Adaptive dense range pyramids keep caller-supplied high-resolution pose fixed,
  run a broad inverse-range sweep at a coarse ERP level, and refine
  uncertainty-aware local offset lattices through the original resolution.
  Local normalization, east/north gradients, and cost-volume filtering now
  reuse PanorAi's C++/NumPy spherical-convolution contract.
- Dependency-light visualization helpers for colorizing radial range and
  rendering labeled RGB diagnostic panels through the required OpenCV runtime.
- A reproducible ten-pair Matterport360/Stanford2D3D development study selected
  from successful frozen five-point/RANSAC poses with metric range references.
  Dataset bytes and generated study results are not distributed.
- `panorai.image_processing` with tangent-plane spherical convolution;
  averaging, Gaussian, median, and bilateral smoothing; canonical rotation and
  resize; Sobel/Scharr gradients and Laplacian; geodesic Canny; Gaussian and
  Laplacian pyramids; and solid-angle-aware grayscale histogram equalization.
- Compatible NumPy `float32`/`float64` ``HW``/``HWC`` convolution dispatches
  to a first-party C++17 kernel while retaining an explicit NumPy reference
  backend and a single public numerical contract.
- Experimental ``panorai-spherical-dog-sift/v1`` extraction: Gaussian/DoG
  extrema and edge rejection run directly on spherical tangent neighbourhoods,
  while one local gnomonic patch per keypoint reuses OpenCV's SIFT descriptor.
  The existing matcher and spherical feature/match result objects remain the
  integration boundary.

### Documentation

- Added an OpenCV-style spherical stereo tutorial with pose convention,
  equations, API signatures, field/parameter reference, failure behavior,
  real-study limitations and next performance gates.
- Added an original spherical epipolar diagram and a checksum-pinned diagnostic
  panel using a real CC0 panoramic photograph with analytic geometry. No
  evaluation-dataset image is redistributed.
- Replaced unsupported Mermaid flowchart fences with publishable SVG diagrams
  and expanded two-view guidance for known camera heights, floor-plane scale,
  metric triangulation and degeneracy handling.
- Added an illustrated, executable spherical image-processing tutorial derived
  from a checksum-pinned online CC0 panorama, with reproducible filtering,
  edge, equalization, and latitude-weight figures.
- Added executable and illustrated coverage for direct spherical DoG,
  tangent-patch SIFT description, cyclic-longitude matching, masks, and the
  Stable face-based alternative; plus a user-first spherical computer-vision
  guide organized around projection, image processing, features/matching,
  two-view geometry, multiview reconstruction, dense stereo, and SLAM.

### Validation

- The analytic stereo oracle verifies radial range, ERP seam wrapping,
  bidirectional consistency, output immutability and visualization behavior.
- Local 128×256 evaluation over ten selected real pairs measured 0.180
  pixel-weighted AbsRel, 0.879 delta<1.25 and 0.295 median accepted coverage
  when estimated rotation/translation direction used reference baseline
  magnitude. This is post-hoc Experimental evidence, not a Stable promotion or
  independent metric-scale claim.
- A preliminary P-74 reference-pose smoke run at the native 8192×4096 delivery
  resolution reduced one-way runtime from 273.1 s for a global 96-label sweep
  to 64.0 s for a three-level adaptive sweep, with AbsRel 0.878→0.807 and
  eligible-reference coverage 0.385→0.342 on that single pair.

### Changed (Experimental)

- The default spherical relative-pose estimator now uses
  `hypothesis_ranking="msac-first"` with
  `nonminimal_refit_max_steps=100`, promoting the strongest evidence-backed
  real-pair path without requiring application-specific opt-in. The former
  count-first/no-refit behavior remains available through explicit options.
- A metadata-separated real calibrated-pair benchmark validates this path for
  feature pairs with adequate support. Across 30 derived Hilti cam0 pairs it
  reduced descriptive median R/t errors by 49%/29% and P95 errors by 79%/56%,
  while a frozen 12-pair phase independently confirmed the translation gain
  but missed its preregistered 25% rotation-reduction target. Low-match and
  near-zero-baseline pairs still require explicit rejection.
- Relative pose now offers an opt-in decoupled Wahba/translation-consensus
  refinement for far-background-dominated spherical pairs. It separates
  rotation-only support from high-parallax translation evidence, uses a
  multiscale epipolar-cheirality score, and abstains when the translation
  direction is unobservable or insufficiently separated from alternatives.
  The existing joint estimator remains the default.
- Relative-pose experiments can ablate count-first, MSAC-first, and
  scale-marginal-first hypothesis ranking and can enable a bounded all-inlier
  linear Essential refit before nonlinear pose refinement. MSAC-first with a
  100-step cap is the default; count-first ranking with a zero-step cap
  reproduces the previous estimator behavior.
- Spherical Essential decomposition now selects translation orientation with
  bounded parallax-weighted cheirality by default, reports both weighted and
  historical raw-count evidence, and retains an explicit
  `positive-depth-count` A/B compatibility option. This improves sign
  selection when numerous distant rays have noise-dominated depth signs
  without claiming observability in pure-rotation or uniformly low-parallax
  scenes.
- The two-view tutorial and companion methods paper now specify the complete
  current spherical relative-pose pipeline: five-point roots, tangent-Sampson
  scoring, hypothesis ordering, bounded LO-RANSAC/IRLS refinement of rotation
  and unit translation, four-way decomposition, reported costs, and limits of
  the synthetic orientation benchmark.

## 3.3.1 — 2026-10-04

### Fixed

- Geometry-v1 JSON fixtures are now explicitly checked out with LF line
  endings on every platform, preserving their byte-level SHA-256 manifest on
  Windows as well as Linux and macOS.
- Pre-release CI now verifies the raw fixture manifest on a Windows checkout,
  closing the coverage gap found by the 3.3.0 publication gate. The 3.3.0
  workflow stopped before TestPyPI; no 3.3.0 package files were published to
  TestPyPI or PyPI.

## 3.3.0 — 2026-10-04

### Added (Experimental)

- `panorai.features` façade with versioned SIFT/ORB/AKAZE and BF/FLANN
  presets backed by OpenCV, while exposing only PanorAi spherical feature and
  match objects in the normal API.
- Vectorized gnomonic pixel↔panorama-ray conversion, explicit virtual-camera
  intrinsics/direction transforms, and optional face→ERP source maps.
- Mask-aware extraction, deterministic angular overlap deduplication,
  panorama-frame bearing correspondences, and serializable provenance.
- Optional PyCOLMAP export of PINHOLE cameras, fixed virtual-camera rigs,
  keypoints, descriptors, and matches. COLMAP remains responsible for SfM.
- Advanced routes for injecting OpenCV-compatible extractor and matcher
  objects without creating a second implementation of their algorithms.
- Isolated `panorai.estimators` spherical relative-pose prototype with a
  PanorAi-owned numerical five-correspondence essential kernel, locally
  optimized RANSAC, tangent-Sampson scoring, cheirality, and explicit
  low-parallax diagnostics. It returns rotation and unit translation direction
  only and remains Experimental.
- Injectable five-point samplers within RANSAC, including a default spatially
  weighted proposal with angular diversity, conditioning gates, progressive
  relaxation, uniform fallback, and explicit sampling diagnostics. Sampling
  never prefilters the correspondences used for scoring or refinement.
- PanorAi-owned polynomial five-point root enumeration, scale-marginal robust
  scoring and IRLS refinement, deterministic subset-stability diagnostics,
  Essential/rotation/spherical-homography model competition, explicit pose
  acceptance evidence, and an isotonic confidence calibrator that rejects
  calibration/evaluation sample-ID leakage.
- Experimental `panorai.reconstruction` global spherical mapper with
  quality-gated view graphs, robust rotation averaging, deterministic
  conflict-free tracks, BATA-style camera-point positioning, two-stage
  spherical bundle adjustment, filtering, retriangulation, arbitrary-scale
  gauges, and inspectable failure diagnostics. The implementation is
  PanorAi-owned NumPy/SciPy code and does not call COLMAP or PyCOLMAP geometry.
- Experimental `panorai.slam` calibrated-fisheye visual SLAM facade with an
  incremental temporal feature graph, conservative pairwise admission,
  arbitrary-scale global reconstruction, immutable trajectory results, and
  explicit failures. A metadata-blind Hilti replay adapter and separate
  post-freeze Sim(3) evaluator remain development tooling; no dataset bytes
  are distributed.
- Experimental central-ERP ``SphericalIncrementalSLAM`` with immediate
  per-frame poses/status, automatic keyframes, geometric-inlier landmark
  tracking, triangulation and culling, spherical 2D--3D refinement, bounded
  local bundle adjustment, relocalization, loop closure/global correction,
  immutable map snapshots, and a metadata-separated real-ERP replay tool.
- Optional first-party C++17 essential-estimation kernels shared by the
  five-point solver, spherical relative pose, global reconstruction and
  incremental SLAM. ``auto`` uses the compiled path when installed, while
  explicit ``numpy`` preserves the reference implementation and explicit
  ``native`` fails rather than silently falling back.
- Optional first-party C++17 spherical bundle-adjustment residuals and
  analytic Jacobian blocks for rotations, camera centers and world points.
  The global mapper retains its Python policies and SciPy optimizer, records
  the resolved backend, and preserves the NumPy finite-difference oracle.

### Changed

- The extraction and matching core of `panorai.features` is promoted to
  Stable as `panorai-spherical-features/v1` after OpenCV compatibility and
  an installed-wheel Essential consumer. Multiscale routing, virtual-camera
  rig/PyCOLMAP export, relative pose, reconstruction, and SLAM retain their
  Experimental tiers.
- The modality-aware object workflow is promoted to Stable as
  `panorai-object-workflow/v1` after two separate installed-wheel consumer
  flows covered arbitrary-N NumPy/Torch reconstruction and
  features/pose/PyCOLMAP composition. Existing 3.0 container methods remain a
  Compatibility surface; pose, PyCOLMAP export, reconstruction, and SLAM
  retain their Experimental tiers.
- `GnomonicFaceSet` reconstructs arbitrary N-view sampler outputs through
  reusable selective back-projection plans. Built-in average/closest/Gaussian
  workflow fusion no longer materializes one full ERP per view; the legacy
  default-average path retains its OpenCV numerics while sampling only pixels
  inside each face support.
- Compatible NumPy `float32`/`float64` arbitrary-N view generation and
  Gaussian reconstruction now dispatch automatically to fused first-party
  C++17 kernels. The optional implementation preserves the Stable workflow
  contract and keeps the existing Python/Torch paths for unsupported dtypes,
  validity-normalized depth, custom projectors, and non-Gaussian blends.
- README is now a concise use-case router. The supporting OpenCV-style tutorial
  path teaches sphere/ray geometry, projection surfaces, samplers/blenders,
  SIFT/ORB/AKAZE extraction, BF/FLANN matching, spherical Essential geometry,
  two-view triangulation and multiview reconstruction with executable contract
  checks, diagrams, and reproducible plots from a checksum-pinned CC0 panorama.
- Native distributions are built as an explicit 20-wheel CPython 3.11--3.14
  matrix for manylinux x86_64/aarch64, macOS x86_64/arm64, and Windows AMD64.
  Every wheel must load and execute both compiled backends before the immutable
  wheel/sdist set can advance to TestPyPI.
- OpenCV ``>=4.9,<5`` is the supported feature-backend range. The oldest
  dependency gate pins 4.9.0.80 and the newest gate resolves the latest
  compatible 4.x release. OpenCV 5 remains excluded until a dedicated backend
  migration restores and validates the AKAZE contract.
- PyCOLMAP export stores standard OpenCV SIFT descriptors as lossless 128-byte
  rows, translates PanorAi pixel-centre coordinates to COLMAP's half-pixel
  database convention, and rejects unsupported floating descriptor encodings
  explicitly.

## 3.2.0 — 2026-09-30

### Added (Experimental)

- Immutable `with_depth()` and `with_labels()` modality composition on
  `EquirectangularImage`, with direct image/depth/labels and validity access.
- Deterministic `views()` presets for cube, Fibonacci, icosahedron and spiral
  layouts, including automatic angular-density sizing and rectangular FOV.
- Modality-aware `GnomonicFaceSet.map()` and `reconstruct()` plus the exact
  `process_views()` convenience chain and structured `describe()` provenance.
- NumPy `HW`/`HWC` and optional Torch `HW`/`CHW`/`NCHW` workflow support
  without eager Torch import on the NumPy path.

The new surface is Experimental; `panorai.geometry` remains the stable
mathematical API and all 3.0 compatibility names remain available.

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
