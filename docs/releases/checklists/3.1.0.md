# PanorAi 3.1.0 release checklist

This checklist is the release gate for PanorAi 3.1.0. Do not create the tag or
publish a GitHub release while any required item remains unchecked.

Every checked item must link to evidence for one exact reviewed commit. Results
from a dirty checkout, editable install, stale `dist/` archive, or local SCM
development version do not satisfy a release item.

## Candidate identity and review

- [ ] The exact release commit has been reviewed and contains every release
  input.
- [ ] The release worktree has no uncommitted or untracked release changes.
- [ ] A distinct reviewer has reproduced Gate 0 and Gates A-I.
- [ ] The coordinator's evidence ledger links every green gate to its commit,
  artifact checksum, environment, and evidence record; durable release evidence
  is also attached to the reviewed pull request or GitHub Release.
- [ ] No pre-existing artifact in `dist/` is being reused without verified
  provenance.

## Repository and publisher setup

- [ ] The `rlsgarcia-code/PanorAi` repository is public and the default branch
  contains the reviewed release changes.
- [ ] The GitHub environments `testpypi` and `pypi` exist.
- [ ] The `pypi` environment requires approval from `rlsgarcia-code`.
- [ ] Trusted Publishing on TestPyPI targets owner `rlsgarcia-code`, repository
  `PanorAi`, workflow `python-publish.yml`, environment `testpypi`.
- [ ] Trusted Publishing on PyPI targets owner `rlsgarcia-code`, repository
  `PanorAi`, workflow `python-publish.yml`, environment `pypi`.
- [ ] GitHub Pages is configured to use GitHub Actions as its source.

## Code and compatibility

- [ ] The complete legacy and canonical test suite passes.
- [ ] Python 3.10, 3.11 and 3.12 core jobs pass.
- [ ] The Torch CPU parity and autograd job passes.
- [ ] The geometry-v1 conformance runner passes.
- [ ] The conformance runner passes against an installed wheel from outside the
  source repository and cannot resolve imports to the checkout.
- [ ] Imports work without Torch and Open3D installed.
- [ ] Legacy exports and all defined samples in the public 3.0.19 projection
  fixture remain unchanged; the documented `x_points`/`y_points` and
  zero-radius `NaN` defects are corrected.
- [ ] Stable blenders consume explicit masks and pass end-to-end tests with real
  projectors; all remaining legacy/experimental blenders are visibly labeled.
- [ ] Black RGB, zero labels, zero radial ranges, `NaN`, and unsupported regions
  have explicit tested semantics.

## Licensing and package identity

- [ ] Every third-party source tree in wheel and sdist has recorded origin,
  version/commit, modifications, license, notice obligations, and distribution
  decision.
- [ ] Required third-party licenses and notices are included in the artifacts.
- [ ] No non-commercial or share-alike material is represented by an MIT-only
  package claim.
- [ ] Any adapter-only or excluded legacy component has user approval and a
  name-by-name 3.0 compatibility assessment.
- [ ] The wheel and sdist contain the four lazy depth loaders/registry keys and
  PCD surface, but no Depth Anything V2, DUSt3R/CroCo, Metric3D, legacy
  ZoeDepth, training, trainer, or custom-data tree.
- [ ] Package metadata and public documentation accurately describe all shipped
  licensing terms.

## Artifact validation

- [ ] Wheel and sdist were built exactly once in a clean environment from the
  release commit.
- [ ] Both artifacts report version exactly `3.1.0`, with no `.dev` or local
  version segment.
- [ ] Filenames, sizes, SHA-256 checksums, member inventories, build command,
  and build environment are recorded.
- [ ] The candidate artifacts pass `twine check`.
- [ ] The wheel is at most 2 MB and the sdist is at most 5 MB.
- [ ] The artifact audit finds no datasets, checkpoints, caches, notebooks,
  media, PyArmor output, `panorai_models`, or `ZoeDepth_not_used`.
- [ ] The wheel installs in empty Python 3.10, 3.11 and 3.12 environments.
- [ ] `panorai.__version__` equals the installed distribution metadata.
- [ ] NumPy and Torch smoke tests pass from the installed wheel.
- [ ] Smoke tests run outside the repository and verify the imported file comes
  from the clean environment's site-packages.
- [ ] The documentation builds successfully without critical warnings.
- [ ] Functional, projector, container, RGB, depth, mask, label, cubemap, batch,
  and autograd examples execute against the installed wheel.

## Pre-merge continuous integration

- [ ] Pull requests and relevant pushes run Python 3.10-3.12 core tests.
- [ ] Torch CPU parity/autograd, installed-wheel conformance, documentation,
  package-content, and license jobs are required before merge.
- [ ] The release workflow remains the only publishing workflow and publication
  is triggered only by `release.published`.

## Publication

1. Create the signed tag `v3.1.0` on the exact reviewed commit.
2. Publish one GitHub Release for `v3.1.0`; this is the only release workflow
   trigger.
3. Confirm that the workflow builds the wheel and sdist exactly once and records
   their SHA-256 checksums.
4. Confirm publication of those exact checksums to TestPyPI.
5. Run and inspect the TestPyPI NumPy/Torch smoke job from a clean environment.
6. Confirm the TestPyPI-installed artifacts match the recorded filenames,
   versions, and checksums.
7. Approve the protected `pypi` environment only after every preceding job is
   green.
8. Confirm that PyPI, the tag, the GitHub Release artifacts, checksums, and
   provenance all identify the same commit and version.
9. Confirm that the documentation was deployed to GitHub Pages.
10. Install `panorai==3.1.0` from PyPI in a new environment and rerun the public
    NumPy and Torch smoke examples.

## Post-release work (not a 3.1.0 publication gate)

- Expand the contract fixtures and independent conformance runner as consumers
  report real-world cases.
- Publish the planned NumPy/Torch comparison with py360convert and pyequilib,
  including raw measurements, environment, hardware, median, and P95.
- Validate adoption by two or three independent consumers pinned to
  `panorai~=3.1` before planning 3.2.
