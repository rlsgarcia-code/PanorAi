Spherical raster fields and observations
========================================

``panorai-spherical-observations/v1`` is the Stable typed-output layer of the
object workflow. Its canonical imports live in ``panorai.data``. Projection
formulas remain owned by :doc:`geometry`.

Four output topologies
----------------------

.. list-table::
   :header-rows: 1

   * - Planar output
     - Face-set operation
     - Spherical result and reduction
   * - dense raster
     - ``map(..., spec=RasterFieldSpec(...))``
     - ``reconstruct()`` with a declared blend
   * - points
     - ``map_points()`` / ``face.lift_points()``
     - ``SphericalPointSet.angular_nms()``
   * - boxes, polygons, or masks
     - ``map_regions()`` / ``face.lift_regions()``
     - solid-angle ``nms()`` or ``merge()``
   * - one score vector or embedding per view
     - ``map_views()`` / ``face.lift_view()``
     - explicit ``coverage_mean`` or score-only ``max``

Raster fields
-------------

``RasterFieldSpec`` declares ``continuous`` or ``categorical`` semantics,
units, interpolation, default blend, and optional component IDs. Continuous
fields cover CAM, logits, confidence, and dense embeddings whose components do
not change with the local basis. Categorical fields require nearest sampling
and closest-view fusion. Optical flow, normals, vectors, and covectors are
rejected by this contract because transporting their components requires an
explicit Jacobian and frame convention.

``RasterFieldResult`` lets an operator return data, data validity, and
geometric support separately. If validity is absent, ``map()`` inherits the
input validity intersected with output support. A supplied validity mask is
also intersected with support. Numeric zero remains valid evidence.

Use ``EquirectangularImage.from_field()`` for a field-only ERP,
``with_field()`` for a multimodal bundle, and ``apply_spherical()`` for a
shape-preserving operator that already works directly on the ERP. Existing
``image``, radial ``depth``, and ``labels`` calls retain their 3.x behavior.

Points
------

``FacePointBatch`` uses ``N×2`` continuous face pixels plus aligned validity,
scores, class IDs, and attributes. Lifting records the original face pixel,
source ERP pixel when known, unit bearing, face ID, source ID, projection spec,
and validity. ``angular_nms()`` is deterministic: score descends first, then
face ID and observation order resolve ties. Invalid inputs stay auditable but
do not participate in reduction.

Regions
-------

Boxes use continuous ``xyxy`` pixel-centre coordinates; the complete face is
``(-0.5, -0.5, W-0.5, H-0.5)``. Boxes and polygons retain a boundary of unit
bearings connected by great-circle arcs. Masks require an explicit boolean
dtype, are support-clipped, nearest-neighbour reprojected, and stored as
``SphericalBinaryMask``; integer or floating masks are never thresholded or
cast implicitly. Finite geometry outside a face is clipped and marked;
inverted or non-finite geometry fails.

Region IoU and containment are weighted by ERP pixel solid angle. NMS retains
one observation and a complete suppression report. Merge stores only the
compact union mask and provenance; temporary dense comparison masks are not
part of the result contract. Class-aware operations do not combine
incompatible class IDs.
When source regions have no score, merge keeps the score absent rather than
inventing a numeric confidence.

View evidence
-------------

``FaceViewOutput`` carries scores, class IDs, an embedding, or both. Every
lifted observation retains the exact ``GnomonicSpec`` and effective support.
``coverage_mean`` first divides overlapping contributions per direction, then
integrates with solid-angle weights. ``max`` is allowed for scores only.
Embedding L2 normalization is controlled explicitly.

Backend and mutability
----------------------

Dense raster processing preserves NumPy or Torch backend, dtype, device, and
batch. Non-raster observations are copied to read-only NumPy arrays. Their
ordinary constructors reject Torch tensors; ``from_torch()`` performs the
visible ``detach → CPU → NumPy`` conversion. Importing ``panorai.data`` does
not import Torch.
