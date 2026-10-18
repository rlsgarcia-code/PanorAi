The geometry contract
=====================

Geometry is the common language beneath every PanorAi API level. It fixes what
a pixel means, how a ray is oriented and which samples are covered before any
application or operation is chosen.

.. panorai-api-components:: typed-data

.. panorai-api-components:: geometry

Coordinate frame and pixels
---------------------------

PanorAi uses ``+X`` right, ``+Y`` up and ``+Z`` forward. Raster coordinates
start at the top-left and address pixel centres. Longitude is periodic in
``[-pi, pi)`` and latitude runs north to south. ERP pixels convert to unit
bearings; a radial range multiplies a bearing to produce a 3-D point.

Depth is radial range
---------------------

Unqualified depth means distance from the panorama centre along the bearing,
not the ``Z`` component of a virtual camera. APIs that use axial depth must say
so and convert with an explicit frame and projection.

Support, validity and value
---------------------------

Geometric support answers whether a projection covers a sample. Data validity
answers whether a usable observation contributed. Numeric value is the
measurement. A black RGB pixel, label zero or range zero can be valid; a
non-zero fill value can be unsupported. These three facts must never be
collapsed into ``value != 0``.

Sampling follows modality
-------------------------

Continuous fields may use nearest or bilinear interpolation under a declared
invalid-data policy. Integer labels and boolean masks use nearest sampling.
Projection results carry support and validity so back-projection and blending
do not infer either from color or magnitude.

Specs make virtual cameras reproducible
---------------------------------------

Immutable gnomonic and cubemap specs define view direction, full FOV, roll,
output shape and face order. Projectors bind those specs to interpolation and
reusable plans. The normative :doc:`geometry-v1 specification <../geometry-v1>`
defines seams, poles, boundaries and tie rules precisely.

See :doc:`the Geometry tutorial <../guides/api-levels/geometry>` and
:doc:`Geometry API <../reference/geometry>`.
