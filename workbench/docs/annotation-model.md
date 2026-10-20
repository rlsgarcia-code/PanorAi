# Annotation model

This document is a draft contract, not an implemented file format.

## Coordinate source of truth

An annotation is stored in spherical coordinates or on the canonical ERP grid.
A gnomonic view records enough projection metadata to reproduce the editing
view, but its pixels are not the authoritative geometry.

Every record should identify:

- panorama identifier and native dimensions;
- coordinate convention and PanorAi geometry-contract version;
- support/validity mask when the panorama is only partially observed;
- semantic label, annotation state, and annotator provenance;
- spherical region geometry and an explicit wrap/seam representation;
- links to observations of the same instance in other registered views;
- optional model proposal provenance, kept separate from human acceptance.

## Region types to evaluate

The first prototype should compare only the minimum useful set:

1. spherical centre plus horizontal and vertical field of view;
2. seam-safe spherical polygon;
3. ERP mask with nearest-neighbour reprojection semantics.

The project should select a canonical representation only after round-trip
tests at the seam, poles, and support boundaries. Derived planar boxes must not
silently replace the spherical annotation.

## Review states

Suggested states are `draft`, `accepted`, `rejected`, and `needs-review`.
Machine-generated geometry starts as `draft`. An export intended as reference
data should include only explicitly accepted records, unless the export format
states otherwise.
