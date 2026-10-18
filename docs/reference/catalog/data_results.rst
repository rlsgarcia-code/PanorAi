Typed data and results catalog
==============================

Typed values make composition safe.  They prevent an array on a reduced
lattice from being mistaken for a canonical ERP and keep numeric values,
geometric support and data validity distinct.

Registered typed-data surfaces
------------------------------

.. panorai-api-components:: typed-data

Object workflow
---------------

.. list-table:: Stable spherical image objects
   :header-rows: 1
   :widths: 26 24 50

   * - Component
     - Represents
     - Key responsibility
   * - ``panorai.EquirectangularImage``
     - one ERP with one or more fields
     - Image-style immutable I/O and transforms backed by canonical spherical geometry.
   * - ``panorai.GnomonicFace``
     - one projected view
     - Values, projection spec, support, validity and source identity.
   * - ``panorai.GnomonicFaceSet``
     - ordered projected views
     - Per-view mapping, typed fields and reconstruction.
   * - ``panorai.SphericalTranspose``
     - named spherical orientation transform
     - Explicit alternative to ambiguous planar transpose conventions.

Pipeline values
---------------

.. list-table:: Stable values from ``panorai.pipeline``
   :header-rows: 1
   :widths: 26 27 47

   * - Component
     - Represents
     - Carries
   * - ``RasterGrid``
     - spatial identity of a raster
     - domain, shape, ERP reference, anchor, stride, periodicity, view identity and transform chain.
   * - ``RasterField``
     - one named raster modality
     - values, kind, units, support, validity and ``RasterGrid``.
   * - ``ResultBundle``
     - named heterogeneous branches
     - raster and non-raster results without forcing one common topology.
   * - ``Output``
     - callable-output declaration
     - kind, units, invalid policy and minimum valid weight.
   * - ``StageResult``
     - one Stage transition
     - output value plus JSON-friendly execution details.
   * - ``StageRecord``
     - recorded Stage execution
     - name, identity, details and provenance used by ``PipelineRun``.
   * - ``PipelineRun``
     - complete recorded execution
     - final output, retained values, ordered records and manifest metadata.

Per-view and spherical observations
-----------------------------------

.. list-table:: Stable observation results
   :header-rows: 1
   :widths: 28 25 47

   * - Component
     - Topology
     - Use
   * - ``ViewResult`` / ``ViewObservations``
     - one view
     - Retain face ID, projection, support and validity with a non-raster result.
   * - ``PerViewResults``
     - ordered views
     - Generic keypoints, detections or another declared per-view kind.
   * - ``ClassificationResult``
     - score vector per view
     - Aggregate or retain classifications without inventing a raster grid.
   * - ``EmbeddingResult``
     - embedding per view
     - Preserve feature vectors, source fields and view support.
   * - ``SphericalObservations``
     - lifted observations on the sphere
     - Bearings, regions or evidence after an explicit lifting step.
   * - ``SphericalPointSet`` / ``SphericalRegionSet`` / ``SphericalViewSet``
     - points, regions or view evidence
     - Geometry-aware reductions such as angular NMS, solid-angle overlap and coverage aggregation.

Feature and pose values
-----------------------

.. list-table:: Experimental two-view exchange types
   :header-rows: 1
   :widths: 27 25 48

   * - Component
     - Produced by
     - Meaning
   * - ``SphericalImagePair``
     - application input adapter
     - Ordered reference/source ERP pair and optional IDs.
   * - ``SphericalFeaturePair``
     - ``ExtractSphericalFeaturePair``
     - Ordered feature sets before matching.
   * - ``SphericalFeatureMatches``
     - ``MatchSphericalFeatures``
     - Matched features, validity and provenance.
   * - ``SphericalBearingCorrespondences``
     - ``ToBearingCorrespondences``
     - Weighted unit bearings in both panorama frames.
   * - ``RelativePoseResult``
     - spherical relative-pose estimator
     - Source-from-reference ``R``, unit ``t``, inliers and complete quality evidence.
   * - ``RelativePoseQualityReport``
     - estimator diagnostics
     - Residual, coverage, parallax, cheirality, stability and model-competition evidence.

Dense stereo returns ``SphericalStereoResult``: radial range, confidence,
validity and diagnostics.  Semantic applications return the segmentation and
proposal types listed in the :doc:`complete public inventory
<public_surfaces>`.

See :doc:`spherical observations <../spherical_observations>` and
:doc:`EquirectangularImage <../equirectangular_image>` for detailed contracts.
