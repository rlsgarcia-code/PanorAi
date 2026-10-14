# PanorAi 3.2.0 release checklist

This checklist is the release gate for PanorAi 3.2.0. Do not create the tag or
publish a GitHub release while any required item remains unchecked.

Every checked item must link to evidence for one exact reviewed commit. Results
from a dirty checkout, editable install, stale `dist/` archive, or local SCM
development version do not satisfy a release item.

## Candidate identity and review

- [ ] The exact release commit contains every release input and has been
  reviewed independently.
- [ ] The release worktree has no uncommitted or untracked release changes.
- [ ] Gate 0 and Gates A-I have been reproduced for that commit.
- [ ] The evidence ledger names the source commit, artifact checksums,
  environments, commands, and limitations.
- [ ] No pre-existing artifact in `dist/` is used as candidate evidence.

## Repository and publisher setup

- [ ] `rlsgarcia-code/PanorAi` is public and protected `main` contains the
  reviewed candidate.
- [ ] Required PR and push checks cover Python 3.10-3.12, Torch CPU, installed
  wheels, artifact policy, and strict documentation.
- [ ] The protected `testpypi`, `pypi`, and `github-pages` environments retain
  their reviewed publisher and approval policies.
- [ ] Trusted Publishing targets `rlsgarcia-code/PanorAi`, workflow
  `python-publish.yml`, and the matching protected environment.

## Code, semantics, and compatibility

- [ ] The complete canonical and legacy source suite passes with warnings as
  errors on Python 3.10, 3.11, and 3.12.
- [ ] Geometry-v1 conformance, the independent oracle, and Torch gradcheck
  pass against the installed wheel outside the checkout.
- [ ] All recorded 3.0.19 names and compatibility fixtures remain available.
- [ ] The Experimental 3.2 workflow passes literal
  `views().map().reconstruct()` and `process_views()` examples.
- [ ] Cube, Fibonacci, icosahedron, spiral, automatic/explicit sizes,
  rectangular FOV, rotations, and deterministic ordering are covered.
- [ ] NumPy HW/HWC and Torch HW/CHW/NCHW preserve backend, batch, dtype,
  device, support, validity, and input autograd.
- [ ] Image/depth use bilinear, labels use nearest, strict depth propagation is
  default, and renormalization requires explicit validity and threshold.
- [ ] Expanded and shortcut workflows are numerically identical and leave
  inputs unchanged.
- [ ] `describe()` reports the executed geometry-v1 plan and the new surface is
  visibly classified Experimental.
- [ ] NumPy-only imports do not load Torch or Open3D.

## Licensing and artifact identity

- [ ] The authorized adapter-only boundary remains unchanged: wheel and sdist
  contain the lazy depth loaders/registry and PCD surface, but no vendored
  model, training, dataset, checkpoint, or research tree.
- [ ] Every artifact member has known project ownership or recorded
  third-party provenance and required notices.
- [ ] Package metadata and public documentation accurately describe licensing.
- [ ] Wheel and sdist are built from the exact candidate commit in clean
  environments and report exactly version `3.2.0`.
- [ ] Filenames, sizes, SHA-256 checksums, complete inventories, build command,
  Python/dependency versions, platform, and hardware are recorded.
- [ ] `twine check`, the content/license auditor, dependency check, and package
  size budgets pass.
- [ ] The sdist rebuilds a byte-identical wheel under the recorded reproducible
  build procedure.
- [ ] Clean installed-wheel checks pass on Python 3.10, 3.11, and 3.12 without
  resolving imports to the checkout.
- [ ] `panorai.__version__` equals distribution metadata and the geometry spec
  plus `py.typed` are present.
- [ ] Strict documentation and all public examples execute against the exact
  installed wheel.

## Pre-publication boundary

- [ ] Protected PR/push CI is green for the exact candidate commit.
- [ ] The release workflow is the only package publisher, targets 3.2.0, builds
  once, verifies exact TestPyPI files before PyPI, and deploys documentation.
- [ ] The coordinator records `GO-CANDIDATE` only after independent review of
  the exact commit and artifacts.
- [ ] No tag, GitHub Release, index upload, environment approval, or publisher
  mutation occurs without separate explicit user authorization.

## Publication sequence — requires separate user authorization

1. Create signed annotated tag `v3.2.0` on the exact GO-CANDIDATE commit.
2. Publish one GitHub Release for `v3.2.0`, the sole release-workflow trigger.
3. Confirm the workflow-built wheel/sdist identities and provenance.
4. Confirm TestPyPI serves byte-identical files and installed NumPy/Torch
   smoke, conformance, oracle, gradcheck, and examples pass.
5. Approve protected PyPI promotion only after every preceding job is green.
6. Verify PyPI, TestPyPI, tag, GitHub Release, checksums, provenance, and Pages
   all identify the same version, files, and commit.
7. Install `panorai==3.2.0` from production PyPI in a new environment and
   repeat the public validation.

The workflow remains Experimental in 3.2.0. Promotion to Stable still requires
two real consumer flows and compatibility evidence; that adoption requirement
does not block publishing the explicitly Experimental surface.
