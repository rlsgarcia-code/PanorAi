Spherical region analysis
=========================

Region analysis turns masks or label fields into topology and geometry.  A
planar implementation is not sufficient for an ERP: its left and right edges
are the same meridian, its pixels cover unequal solid angles, and a missing
band must not connect regions merely because its numeric values are zero.

Topology contract
-----------------

``connected_components`` uses 4- or 8-neighbour ERP pixel-centre adjacency
with periodic longitude.  Latitude does not wrap. Unsupported or invalid
pixels are absent vertices, so they cannot bridge components. Positive labels
are deterministic and ordered by the first raster index of each component;
zero is background.

``watershed`` floods the same graph from positive integer markers. It
minimizes the maximum gradient along a path, then path length; the smaller
marker label resolves a remaining tie. Marker pixels never change. This is a
declared discrete graph watershed, not a planar watershed applied to an ERP
rectangle.

Distance and boundaries
-----------------------

``geodesic_distance_transform`` builds unit bearings for ERP centres and
returns the exact great-circle angle to the nearest selected centre. Values
outside support or validity are NaN, not zero. ``region_boundaries`` uses
topological neighbours and never invents a boundary at the longitude seam.

Angular perimeter is measured along cell interfaces. North-south interfaces
have length :math:`\Delta\phi`; east-west interfaces at latitude
:math:`\phi` have length :math:`\cos(\phi)\Delta\lambda`. Solid angle comes
from the existing spherical core and sums to :math:`4\pi` over a complete
ERP. Centroids are normalized solid-angle-weighted bearing sums. Symmetric
regions whose vector sum is undefined return ``None`` rather than an arbitrary
direction.

Holes on a closed sphere
------------------------

A sphere has no privileged exterior component. ``fill_holes`` therefore
requires ``max_hole_solid_angle_sr``. It fills only background components no
larger than that physical area and never fills a component touching missing
support or invalid data. The threshold is part of the scientific method, not
an implementation detail.

API placement
-------------

Functions and configured raster operations live in
``panorai.ops.spherical``. Configured operations compose through
``EquirectangularImage.apply_spherical`` or ``OnSphere`` and safely persist in
a Pipeline. ``SphericalWatershed`` is a two-field Stage; ``AnalyzeRegions``
turns a label field into ``RegionAnalysisResult``. Immutable measurements,
adjacency graphs, conservative enclosing caps and boundary samples live in
``panorai.data``.

The boundary type intentionally stores unordered pixel-centre bearing samples,
not a polygon. Applications requiring an ordered sub-pixel contour must use a
future contour contract rather than infer ordering from these samples.

This surface is Experimental. It does not yet include spherical
skeletonization or an ordered contour/polygon tracer.
