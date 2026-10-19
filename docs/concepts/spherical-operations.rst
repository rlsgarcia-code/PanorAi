Sphere-aware operations
=======================

An equirectangular panorama is a rectangular storage raster for a sphere; its
ordinary planar neighbourhoods are not geometrically uniform. Longitude wraps,
horizontal metric scale changes with latitude, and raw pixel counts overweight
polar surface area.

.. panorai-api-components:: operation

.. panorai-api-components:: torch-layer

Local tangent neighbourhoods
----------------------------

Linear filters and derivatives define taps in a local east/north tangent frame
at each output bearing, transport those taps to the sphere and sample the ERP
with longitude wrapping. The operation therefore keeps an angular meaning
across latitude instead of applying the same planar pixel offsets everywhere.

Nonlinear operations follow the same domain contract. Median and bilateral
filters use geodesic neighbourhoods; Canny performs tangent-direction
suppression and seam-aware connectivity; area reductions weight latitude bands
by solid angle.

Functions, configured objects and layers
----------------------------------------

Functional calls are convenient for one expression. Immutable operation
objects add parameter inspection, ``transform_grid`` and safe Pipeline
composition. Optional Torch layers extend the same spatial contract to learned
convolution, pooling and interpolation while preserving autograd.

Grid and mask transforms are part of the operation
--------------------------------------------------

A spatial operation transforms values, geometric support, data validity and
grid identity together. Strict validity propagation is the default.
Renormalization exists only where a normalized mathematical formula is
defined, such as averaging or Gaussian weighting; arbitrary learned weights
cannot be reinterpreted as valid-weight averages.

Sphere-native does not always mean preferable
---------------------------------------------

Projected-view processing remains appropriate for existing pinhole algorithms
and models. The choice is architectural: direct operations avoid face
boundaries and fusion, while projected views reuse mature planar components.
Pipeline can combine both routes without changing geometry conventions.

See :doc:`the Spherical Ops tutorial <../guides/api-levels/spherical_ops>` and
:doc:`Operations API <../reference/spherical_ops>`.
