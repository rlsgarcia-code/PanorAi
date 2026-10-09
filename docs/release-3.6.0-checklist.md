# PanorAi 3.6.0 release checklist

This minor release adds opt-in Experimental spherical deep-learning model-port
surfaces and refreshes the package summary. Stable geometry-v1, object workflow,
spherical-feature core, projection, blending and every 3.x compatibility name
remain unchanged.

## Candidate identity

- [ ] One exact reviewed protected-main commit contains every release input and
  has no uncommitted or untracked release changes.
- [ ] `CHANGELOG.md`, package metadata, release workflow, Pages recovery,
  manifest, artifact policy and contract tests identify exactly 3.6.0 / `v3.6.0`.
- [ ] A fresh review covers the complete `v3.5.0..candidate` diff and verifies
  that every new deep-learning surface remains Experimental and opt-in.
- [ ] Immutable published tags and index files through 3.5.0 remain unchanged.

## Public contract and dependency isolation

- [ ] Stable/Compatibility inventories pass with no removed or re-tiered names.
- [ ] `import panorai`, `import panorai.geometry` and
  `import panorai.experimental` do not load Torch, Torchvision or external model
  code and cause no network/cache side effect.
- [ ] `panorai.experimental.deep_learning` is the only opt-in import boundary;
  Torchvision loads only when a pretrained ImageNet helper is called.
- [ ] `deep-learning` and `deep-learning-depth` remain optional extras; base
  dependencies and supported Python 3.11–3.14 boundaries are unchanged.

## Model portability and safety

- [ ] Analytic spherical convolution covers exact 1x1 equivalence, parameter
  identity, seam/longitude equivariance, finite poles, shapes and autograd.
- [ ] Planar FCN conversion reproduces the original AlexNet, VGG16 and ResNet18
  classifier logits within the documented tolerance using official external
  weights.
- [ ] The metric-depth loader requires explicit terms acceptance, pins source
  and checkpoint identities, verifies size/SHA-256, rejects traversal/links and
  uses `torch.load(..., weights_only=True)`.
- [ ] The real external model port accounts for all 45 `Conv2d`, three
  `ConvTranspose2d` and four absorbed reflection pads while preserving learned
  parameter identity and leaving no learned planar spatial layer.
- [ ] No accuracy, semantic-equivalence, spherical-equivariance or suitability
  claim is inferred from mechanical model loading and parameter preservation.

## Private-data and artifact boundary

- [ ] No private dataset name, identifier, byte, prediction, metric, result,
  generated visualization or narrative appears in tracked release content,
  wheel, sdist, documentation output or release notes.
- [ ] No checkpoint, model archive, extracted third-party source, cache,
  manifest, generated CAM or prediction appears in wheel or sdist.
- [ ] The unresolved checkpoint-license boundary is explicit: opt-in is not a
  license grant and PanorAi never redistributes the checkpoint.
- [ ] Wheel and sdist content/license audits require every new PanorAi-owned
  source module and reject prohibited model/data extensions.

## Source, documentation and artifact validation

- [ ] Full warnings-as-errors tests pass on Python 3.11–3.14; Torch CPU runs
  geometry, FCN/CAM, acquisition-contract and metric-depth adapter tests.
- [ ] Strict Sphinx builds without warnings and executable examples pass.
- [ ] Both native extensions build and execute on Linux x86_64/aarch64, macOS
  Intel/arm64 and Windows AMD64 for Python 3.11–3.14.
- [ ] The hosted aggregate contains exactly 20 wheels and one normalized sdist;
  every file passes Twine, checksum, metadata, content/license and provenance
  audits.
- [ ] A fresh compatible installed wheel outside the checkout passes version,
  origin, native, import-isolation and `--deep-learning` smoke tests without
  downloading a model.
- [ ] A distinct fresh review reproduces the artifact inventory and installed
  evidence from the exact candidate.

## Publication sequence

1. Merge only after every hosted PR job passes and repeat the complete matrix
   on the protected-main merge commit.
2. Record `GO-CANDIDATE` only after exact merged-source, aggregate-artifact and
   installed-wheel review.
3. Change only the three protected environment selectors from `v3.5.0` to
   `v3.6.0`; preserve reviewers, administrator enforcement, custom policies and
   Trusted Publishers.
4. Create one signed annotated `v3.6.0` tag on the exact candidate and publish
   one GitHub Release, the sole trigger of `python-publish.yml`.
5. Verify TestPyPI serves the exact workflow-built files and run installed
   native, geometry, Torch, deep-learning and documentation checks.
6. Promote the identical files to PyPI, attach checksums/provenance and deploy
   Pages; no manual upload may bypass a failed gate.
7. Install `panorai[deep-learning]==3.6.0` from production PyPI in a fresh
   environment and verify exact artifact identity and installed smoke without
   acquiring external model files.

Published tags and index files are immutable and may not be moved, deleted,
recreated or replaced.
