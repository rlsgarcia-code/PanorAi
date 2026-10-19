# Contributing to PanorAi

PanorAi accepts focused changes that preserve explicit coordinate, validity,
provenance, dependency, and stability boundaries.

Developers adding a user-facing application, Pipeline stage, spherical
operation, or optional Torch layer must follow the
[public API component contract](../docs/development/application_extensions.rst).
It defines the supported signatures, result/grid/mask obligations, persistence
rules, exports, tests, and the boundary between a wrapper and its domain
algorithm.

Before opening a pull request:

1. identify the affected public contract and stability tier;
2. keep algorithm development in its owning domain package;
3. avoid importing optional backends from base-package imports;
4. add conformance or regression tests proportional to the change;
5. update `contracts/public-api-surfaces-v2.json` for every public export;
6. assign every public surface a documentation role and update its catalog,
   reference, concept, guide/example and README policy;
7. document frames, units, scale, split, access role, calibration, and
   provenance for new geometric evidence;
8. run the focused tests, the full suite, and strict documentation build.

Ruff is the single formatter and linter for Python sources. Before submitting,
run `ruff check .` and `ruff format --check .`. Keeping one formatting engine
avoids version-dependent Black/Ruff rewrites in release commits.

Stable and Compatibility imports are preserved throughout 3.x. Experimental
does not mean untested: every Experimental surface needs a versioned contract,
fail-closed behavior, import-isolation coverage, and explicit promotion gates.

Benchmark claims must identify whether they are conformance, regression,
performance, or scientific evidence. Source datasets and model weights remain
external unless their redistribution and license are explicitly approved.
Never tune on a frozen test split or present post-hoc evidence as prospective.

Pull requests should be narrow enough that API review, numerical review, and
documentation review can be performed independently. Release tagging and
publication are maintainer actions performed only after the candidate commit
passes the release checklist.
