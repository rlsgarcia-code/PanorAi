# PanorAi 3.3.0 release checklist

This is the release gate for PanorAi 3.3.0. A checked item must identify one
exact reviewed commit and reproducible evidence. A dirty checkout, editable
install, stale archive, or development version is not release evidence.

## Candidate identity and review

- [ ] The exact release commit contains every source, workflow, documentation,
  version and changelog input and the tracked tree is clean.
- [ ] A fresh reviewer has inspected the complete release diff and reproduced
  Gate 0 and Gates A-I for the same commit.
- [ ] Evidence records commands, environments, raw artifact inventories,
  checksums and known limitations; no pre-existing `dist/` file is reused.
- [ ] Protected PR and post-merge CI are green for the exact trees they name.

## Stable and Experimental boundaries

- [ ] `geometry-v1`, `panorai-object-workflow/v1` and the extraction/matching
  core of `panorai-spherical-features/v1` remain the only promoted surfaces.
- [ ] Pairwise pose, multiscale routing, PyCOLMAP export, reconstruction and
  SLAM remain visibly Experimental and make no new stability claim.
- [ ] Native acceleration is an internal automatic implementation choice;
  Python/NumPy and Torch fallbacks preserve public values, ordering, validity,
  support, dtype, shape and documented failure behavior.
- [ ] All recorded 3.0 compatibility names and fixtures remain available.

## Runtime, dependency and native safety

- [ ] Metadata, README, CI and wheel selectors agree on Python 3.11-3.14.
- [ ] Core warning-strict tests pass on 3.11, 3.12, 3.13 and 3.14.
- [ ] The pinned oldest boundary (`numpy==1.26.4`, OpenCV 4.9.0.80 and the
  declared minimum scientific stack) and newest allowed boundary both pass
  `pip check` plus stable-surface tests.
- [ ] OpenCV stays constrained to `>=4.9,<5`; no OpenCV 5 compatibility is
  implied.
- [ ] Both native extensions build and execute on Linux, macOS and Windows at
  the oldest and newest supported Python versions.
- [ ] C++ entry points translate allocation/thread failures, restore the GIL,
  join partially started thread groups and allocate no worker scratch memory.
- [ ] Concurrent public arbitrary-N generation and reconstruction remain
  deterministic and leave the interpreter live.

## Semantics and regression gates

- [ ] The complete source suite passes with warnings as errors.
- [ ] Geometry conformance, independent oracle and Torch gradcheck pass against
  an installed wheel outside the checkout.
- [ ] ERP↔gnomonic, ERP↔cubemap and arbitrary-N sampler workflows cover seams,
  poles, rectangular views, rotations, masks, labels, depth and all supported
  NumPy/Torch layouts.
- [ ] Cube, Fibonacci, icosahedron and spiral ordering and tie-breaking remain
  deterministic; expanded and one-call object workflows remain equal.
- [ ] Strict Sphinx and every executable public example pass against the exact
  installed wheel.

## Artifact and publisher integrity

- [ ] A detached clean build produces exactly 20 native wheels and one sdist
  for version `3.3.0`; every file passes `twine check` and content/license
  audit.
- [ ] The sdist rebuilds the expected wheel under the recorded reproducible
  procedure; inventories, sizes and SHA-256 checksums are stored.
- [ ] Clean installed-wheel jobs pass on all 20 OS/architecture/Python
  combinations without resolving imports to the checkout.
- [ ] `panorai.__version__`, distribution metadata, tag, geometry contract and
  `py.typed` identity agree.
- [ ] The release workflow is the only package publisher, builds artifacts
  once, gates TestPyPI on installed wheels and strict docs, verifies exact
  TestPyPI files before PyPI, and deploys Pages from the same tag.
- [ ] Trusted Publishing and protected environment policies are read back and
  unchanged except for an explicitly authorized exact-version selector.

## Publication boundary

- [ ] The coordinator records `GO-CANDIDATE` only after the exact merged commit
  and artifacts receive fresh review.
- [ ] The changelog's `Unreleased` marker is replaced by the real publication
  date before the final candidate is reviewed; no post-review source delta is
  allowed.
- [ ] Creating/pushing `v3.3.0`, publishing a GitHub Release, approving a
  protected environment, changing publisher/policy configuration, uploading
  to TestPyPI/PyPI, or deploying release Pages has separate explicit user
  authorization.

## Authorized publication sequence

1. Create the signed annotated `v3.3.0` tag on the exact GO-CANDIDATE commit.
2. Publish one GitHub Release, the sole trigger of `python-publish.yml`.
3. Verify the workflow-built 20-wheel/one-sdist identities and provenance.
4. Verify TestPyPI serves byte-identical files and installed NumPy/Torch
   smoke, conformance, oracle, gradcheck and examples pass.
5. Approve production promotion only after all preceding jobs are green.
6. Verify PyPI, TestPyPI, tag, GitHub Release, checksums, provenance and Pages
   identify the same version, files and commit.
7. Install `panorai==3.3.0` from production PyPI in a new environment and
   repeat the public validation.
