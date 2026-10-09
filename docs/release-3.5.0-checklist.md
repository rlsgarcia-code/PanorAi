# PanorAi 3.5.0 release checklist

This minor release packages the current protected-main feature and
documentation line after the GitHub-only `v3.4.0` publication attempt. It adds
an evidence-backed relative-pose feature profile and Experimental resolution
selection, descriptor-free spherical detection, coarse-to-fine proposals,
tangent-patch descriptors and organized RGB+XYZ loading. Stable geometry-v1,
projection, object-workflow and spherical-feature-core contracts remain
compatible.

## Candidate identity

- [ ] One exact reviewed protected-main commit contains every release input and
  has no uncommitted or untracked release changes.
- [ ] `CHANGELOG.md`, the publisher workflow, Pages recovery default,
  setuptools-scm fallback, manifest, artifact policy and contract tests all
  identify exactly 3.5.0 / `v3.5.0`.
- [ ] A fresh review checks the complete `v3.4.0..candidate` diff and confirms
  that every new surface is documented at its actual stability tier.
- [ ] The immutable `v3.4.0` tag and GitHub Release remain unchanged. Read-back
  confirms that no 3.4.0 file reached TestPyPI or PyPI and that 3.3.1 remains
  the latest package-index version before this release.

## Public contract and compatibility

- [ ] Stable geometry-v1, object-workflow, spherical-feature-core, blender and
  3.x compatibility inventories pass without removed names.
- [ ] `SphericalFeaturePipeline.for_relative_pose()` preserves general preset
  defaults while freezing only the documented reference profile.
- [ ] `shadow_padded` distinguishes cropped and materialized scanner rasters;
  padding is idempotent and geometric support is not inferred from black RGB.
- [ ] Resolution selection, direct/coarse spherical detection, tangent patches,
  descriptor v2 and downstream pose/reconstruction/SLAM/stereo surfaces remain
  explicitly Experimental.
- [ ] Removal of project-specific P74/P77 source helpers changes no wheel/sdist
  member or Stable/Compatibility export promised by the published package.

## Feature and data validation

- [ ] Direct spherical detector localization covers seam, high-latitude,
  mask/support, deterministic selection, scale and empty-input cases.
- [ ] Coarse-to-fine proposals preserve source-ERP provenance and compose with
  descriptor-neutral tangent patches and SIFT/ORB adapters.
- [ ] Tangent descriptor v2 verifies photometric normalization, effective
  support, scale/orientation hypotheses, ordering and parallel determinism.
- [ ] The resolution-selection report distinguishes converged plateaus from
  highest-resolution fallback and never resizes caller data implicitly.
- [ ] `XYZImageDataset` requires explicit frame/units, validates organized PLY
  shape, keeps Open3D optional and ships no dataset bytes.

## Documentation and media

- [ ] The README, user guide, capability map, benchmark guide and API reference
  agree on sphere-native, projection-domain and hybrid routes.
- [ ] Strict Sphinx builds with zero warnings from source and extracted sdist;
  all executable examples pass from the installed wheel where applicable.
- [ ] Only checksum-pinned approved tutorial media appear in the sdist and no
  raster documentation media appear in wheels.
- [ ] Benchmarks label source, development wheel and published-release evidence
  separately and make no promotion claim for Experimental APIs.

## Source, wheel and platform validation

- [ ] Full warnings-as-errors tests pass on Python 3.11–3.14, including NumPy,
  Torch CPU/autograd, conformance/oracle, multimodal consumers and new feature
  and data suites.
- [ ] Both first-party native extensions build and execute on Linux x86_64 and
  aarch64, macOS Intel and arm64, and Windows AMD64 for Python 3.11–3.14.
- [ ] The hosted aggregate contains exactly 20 wheels and one normalized sdist;
  every file passes `twine check`, checksum, metadata, license, content and
  provenance audits.
- [ ] Fresh installed-wheel smoke outside the checkout proves version 3.5.0,
  installed origin, native availability, canonical imports and the new public
  feature/data exports.
- [ ] A distinct fresh review reproduces the exact artifact inventory and the
  release-required installed-package evidence.

## Publication sequence — separate authorization required

1. Merge the reviewed release PR only after every hosted job passes on its
   exact head and repeat the complete matrix on the protected-main merge.
2. Record `GO-CANDIDATE` only after the exact merged commit and hosted aggregate
   are independently reviewed.
3. After explicit publication authorization, change only the three protected
   environment tag selectors from `v3.4.0` to `v3.5.0`; preserve required
   reviewers, administrator enforcement, custom policies and Trusted
   Publishers.
4. Create one signed annotated `v3.5.0` tag on the exact GO-CANDIDATE commit and
   publish one GitHub Release, the sole trigger of `python-publish.yml`.
5. Verify TestPyPI serves the exact workflow-built files, then run installed
   smoke, conformance, oracle, Torch and documentation checks.
6. Allow the protected workflow to promote the same files to PyPI and deploy
   Pages; no manual upload may bypass a failed gate.
7. Install `panorai==3.5.0` from production PyPI in a fresh environment and
   verify version, native extensions, public exports and artifact identity.

The `v3.4.0` tag must never be moved, deleted or recreated. Published index
files are immutable and may not be replaced.
