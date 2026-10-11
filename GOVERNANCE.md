# Governance

PanorAi is maintainer-led and contract-driven. Maintainers decide releases,
public API tiers, compatibility policy, and repository scope. Contributors may
propose changes through pull requests with tests and evidence.

Decisions follow these priorities:

1. mathematical and provenance correctness;
2. preservation of Stable and Compatibility promises;
3. fail-closed handling of incompatible evidence;
4. reproducibility from installed artifacts;
5. measured performance and scientific usefulness.

An API is not promoted because it is broad, heavily used internally, or part
of a release consolidation. Promotion requires the gates in the canonical API
inventory and evidence from an installed artifact. Negative findings remain
visible and may justify a do-not-promote recommendation.

The public roadmap communicates product direction. `.agents/` contains current
operational coordination only and is not a parallel product history. Git,
release notes, public contracts, and `docs/development/history.rst` are the
historical record.

Tags, package-index publication, irreversible data removal, and changes to
Stable contracts require explicit maintainer authorization.
