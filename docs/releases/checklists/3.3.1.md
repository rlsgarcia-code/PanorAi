# PanorAi 3.3.1 release checklist

This patch release repairs the cross-platform fixture gate that stopped the
3.3.0 workflow before TestPyPI. The signed `v3.3.0` tag and GitHub Release stay
immutable, and no 3.3.0 package files are promoted or uploaded manually.

## Patch identity and incident boundary

- [ ] The candidate is based on the exact reviewed 3.3.0 merge commit and
  contains no runtime API, numerical, native-kernel or dependency change.
- [ ] The changelog records the failed 3.3.0 publication boundary and confirms
  that neither TestPyPI nor PyPI received 3.3.0 package files.
- [ ] The release version, workflow selector, recovery workflow, documentation
  and fallback metadata all identify 3.3.1.
- [ ] The complete patch diff receives review and protected PR plus post-merge
  CI pass for the exact trees they name.

## Windows fixture correction

- [ ] `.gitattributes` forces LF for every checksum-pinned geometry-v1 JSON
  fixture on all checkout platforms.
- [ ] The standalone integrity verifier reads raw bytes and reproduces the
  manifest check used by installed-wheel conformance.
- [ ] Tests prove the canonical fixture set passes and an otherwise identical
  CRLF-mutated `analytic.json` is rejected.
- [ ] A hosted `windows-latest` job verifies fixture integrity directly after
  checkout, independently of package build and wheel smoke tests.
- [ ] Linux, macOS and Windows installed-wheel conformance all consume the same
  manifest and byte-identical canonical fixtures.

## Inherited stable-release gates

- [ ] The full source suite passes with warnings as errors on Python 3.11-3.14,
  including NumPy, Torch, conformance, oracle, gradcheck and multimodal flows.
- [ ] Strict Sphinx and executable documentation examples pass from source,
  installed wheel and extracted sdist where applicable.
- [ ] Both native extensions build and execute on the full supported Linux,
  macOS and Windows wheel matrix.
- [ ] A clean build produces exactly 20 wheels and one normalized sdist for
  version 3.3.1; all artifacts pass metadata, content, license and provenance
  audits plus `twine check`.
- [ ] The hosted aggregate is downloaded, independently inventoried and its
  stored checksums match the release workflow output.
- [ ] Stable/Experimental boundaries and all public geometry-v1 semantics are
  unchanged from the reviewed 3.3.0 candidate.

## Authorized publication sequence

1. Read back the three protected environment policies and change only their
   exact tag selector from `v3.3.0` to `v3.3.1`.
2. Create a signed annotated `v3.3.1` tag on the exact GO-CANDIDATE commit.
3. Publish one GitHub Release, the sole trigger of `python-publish.yml`.
4. Verify the workflow-built 20-wheel/one-sdist identities and provenance.
5. Verify TestPyPI serves byte-identical files and installed validation passes.
6. Allow the workflow to promote those same files to PyPI and deploy Pages.
7. Install `panorai==3.3.1` from production PyPI in a new environment and
   repeat the public smoke, native-extension and conformance validation.

The 3.3.0 tag must never be moved or deleted, and no manual upload may bypass
the release workflow or any protected environment.
