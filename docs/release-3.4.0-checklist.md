# PanorAi 3.4.0 release checklist

This minor release publishes direct spherical dense stereo as an explicitly
Experimental API. Stable geometry-v1, projection, feature and object-workflow
contracts remain unchanged.

## Candidate identity

- [ ] The exact release commit contains the reviewed stereo implementation,
  visualization helpers, generated documentation assets, release metadata and
  no uncommitted files.
- [ ] The changelog uses the actual publication date and every workflow,
  recovery default, protected environment selector, fallback and artifact
  policy identifies 3.4.0 / `v3.4.0`.
- [ ] A fresh review checks the complete `v3.3.1..candidate` diff and records
  any limitations without promoting Experimental behavior to Stable.

## Experimental stereo contract

- [ ] The public pose convention is
  `X_b = R_b_from_a @ X_a + t_b_from_a`; output is radial range and inherits
  translation units.
- [ ] Every range candidate is pose-constrained, horizontal ERP sampling wraps,
  invalid pixels are explicit and A→B→A consistency is tested.
- [ ] Analytic range, seam, validation, immutability, visualization and frozen
  export-inventory tests pass with warnings as errors.
- [ ] The ten-case study records exactly five Matterport360 and five
  Stanford2D3D pose-success pairs, selection hashes, per-case panels, raw
  arrays, accuracy and coverage while stating that reference baseline magnitude
  supplies metric scale.
- [ ] Matterport360/Stanford2D3D media and local study outputs are absent from
  wheel, sdist and published documentation.

## Documentation and media

- [ ] Strict Sphinx copies and renders the method diagram and the real CC0
  photograph/analytic-geometry diagnostic panel without warnings.
- [ ] Tutorial signatures, parameters, result fields, example, pose convention,
  failure behavior, evaluation limitations and future C++ boundary match the
  published code.
- [ ] The original source image provenance and CC0 license remain bundled; the
  generated PNG has the reviewed SHA-256 enforced by the artifact auditor.
- [ ] Pages is deployed from the exact signed release tag, never from a mutable
  branch or a locally built documentation tree.

## Source, wheel and platform validation

- [ ] Full warnings-as-errors tests pass on Python 3.11–3.14, including NumPy,
  Torch CPU/autograd, geometry conformance/oracle, multimodal flows and the new
  stereo suite.
- [ ] Strict documentation and executable examples pass from source, installed
  wheel and extracted sdist where applicable.
- [ ] Both existing native extensions build and execute on the complete Linux,
  macOS and Windows wheel matrix; the Python/OpenCV stereo module is present in
  every wheel without introducing a new compiled extension.
- [ ] The hosted aggregate contains exactly 20 wheels and one normalized sdist;
  every file passes metadata, license, content, provenance and checksum audits
  plus `twine check`.
- [ ] Fresh installed-wheel smoke imports and runs `panorai.stereo` outside the
  checkout and confirms optional Torch/Open3D remain unloaded.

## Authorized publication sequence

1. Merge the reviewed PR only after all required hosted checks pass.
2. Read back the three protected environment policies and change only their
   exact tag selector from `v3.3.1` to `v3.4.0`.
3. Create a signed annotated `v3.4.0` tag on the exact GO-CANDIDATE merge.
4. Publish one GitHub Release; it is the sole trigger of
   `python-publish.yml`.
5. Independently inventory the workflow-built artifacts and attestations.
6. Verify TestPyPI serves byte-identical files and installed validation passes.
7. Allow the workflow to promote those same files to PyPI and deploy Pages.
8. Install `panorai==3.4.0` from production PyPI in a clean environment and
   repeat public smoke, native-extension and stereo checks.

No manual upload may bypass the release workflow or protected environments.
