Computer-vision research capability roadmap
============================================

PanorAi aims to become a reference substrate for **spherical computer
vision**, not a replacement for OpenCV, scikit-image, PyTorch or a model
training framework.  Its distinctive responsibility is to make geometry,
grid, support, validity, units and provenance survive an entire vision
workflow.

This page records the delivered evaluation and region-analysis foundations,
the bounded Experimental registration/stitching implementation and its
remaining development gates. A component becomes public only after it enters
the audited API manifest, documentation catalog, tests and the appropriate
stability process.

The ordering is deliberate.  Evaluation comes first so later methods can be
compared without inventing metrics after seeing results.  Region analysis
comes next because it completes the current morphology and SLIC foundation.
Registration and stitching then build on features, regions, robust estimation,
projection and fusion as one end-to-end application.

Priority 1 — research evaluation
--------------------------------

The Stable ``panorai.evaluation`` namespace is the public evaluation
vocabulary for spherical results.
Evaluation must use the same coordinate, support and validity contracts as the
algorithms it measures.

Delivered metric families
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Raster and regions:** solid-angle-weighted IoU, Dice, precision, recall,
  coverage and geodesic boundary F-score.
* **Features and correspondence:** repeatability, angular localization error,
  scale error, match precision and verified-inlier measures.
* **Motion:** tangent-vector angular endpoint error, warp consistency,
  occlusion-aware coverage and forward/backward consistency.
* **Pose and registration:** SO(3) geodesic error, translation-direction error,
  residual distributions, overlap coverage and model-selection diagnostics.
* **Depth and reconstruction:** radial-range error, validity coverage,
  confidence calibration, point-to-surface error and completeness.
* **Equivariance and robustness:** non-axis-aligned rotations, longitude seam,
  poles, partial support, invalid data and resolution changes.

Every metric must declare modality, units, reduction, weighting, support and
invalid-data behavior.  ``value == 0`` is never an implicit validity rule.
Area-based metrics use spherical solid angle rather than equal ERP-pixel
weight unless a caller explicitly selects another measure.

Evaluation records and protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reproducible evaluation record should bind:

* algorithm and configuration identity;
* input, reference and split identities without bundling private datasets;
* coordinate contract, modality, units and masks;
* package version, commit or artifact hash, backend and environment;
* per-sample values, aggregate policy and confidence intervals;
* abstentions, failures and excluded samples with explicit reasons.

Protocol definitions must be frozen before a held-out result is opened.
Development, calibration and held-out splits remain distinct.  Dataset
downloaders, experiment orchestration, training and checkpoints remain outside
the core package; small analytic fixtures and dataset adapters may be used when
their provenance and licenses are explicit.

Evaluation Stable boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

Research evaluation is usable when independent code can reproduce at least
one region, feature, pose and depth metric against an installed wheel; analytic
seam/pole/rotation oracles pass; and the same inputs give documented NumPy and
optional Torch behavior where both backends are offered.

Priority 2 — region analysis
----------------------------

Region analysis extends pixel or mask processing into spherical topology.  It
must operate on the sphere rather than joining or separating regions according
to their accidental ERP layout.

Stable primitives
~~~~~~~~~~~~~~~~~

* connected components with longitude-periodic adjacency;
* geodesic distance transform;
* watershed and marker-controlled watershed;
* contour and boundary extraction;
* hole filling and intrinsic sampled-medial-axis skeletonization;
* region adjacency graphs;
* region measurements including solid angle, angular perimeter, spherical
  centroid, support fraction and enclosing spherical region;
* typed conversion between label fields, binary masks, contours and region
  observations.

These capabilities should compose as:

.. code-block:: text

   threshold or model output
       -> spherical morphology
       -> connected components or watershed
       -> typed regions and measurements
       -> evaluation or a downstream Pipeline Stage

Neighbourhoods and distances require an explicit angular or topological
definition.  Labels and masks remain nearest-sampled.  Invalid or unsupported
pixels cannot silently connect two regions, but callers may select a documented
boundary policy when partial support is intentional.

Region-analysis API placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local raster transforms belong in spherical operations.  An operation may be
used directly, through ``EquirectangularImage`` or inside ``OnSphere``.  Typed
region and contour results belong in the data/result layer.  Pipeline Stages
adapt those operations for routing; they do not duplicate the mathematics.
An Application is warranted only for a complete task with natural inputs and
an independently useful result.

Region-analysis Stable boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first region-analysis release must cover analytic caps and annuli, seam-
crossing and polar regions, disconnected invalid bands, non-axis-aligned
rotations and the planar low-distortion limit.  Connected-component topology,
distance values, watershed partitions and region measurements need independent
oracles and installed-wheel composition evidence.

Priority 3 — registration and stitching
----------------------------------------

The Experimental :class:`panorai.applications.PanoramaStitcher` accepts
:math:`N \geq 2` overlapping planar pinhole images and constructs a partial
typed ERP even when orientation, horizontal field of view or EXIF metadata was
not supplied. Available metadata is a prior and provenance, not unquestioned
ground truth.

``No metadata`` does not mean ``no assumptions``.  Some image sets do not
contain enough information to recover calibration or a common panorama.  The
system must estimate only observable quantities and fail with diagnostics
instead of inventing geometry.

Implemented bounded profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Validate numeric image arrays plus independent support and validity masks.
2. Detect masked SIFT or ORB evidence and infer candidate overlap pairs.
3. Robustly validate a planar homography and fit a central SO(3) rotation.
4. Retain supplied horizontal FOV or estimate one shared missing value only
   when the correspondence geometry makes it observable.
5. Require one connected accepted graph, initialize all orientations and
   optimize its SO(3) residuals jointly.
6. Warp values, support and validity independently to one ERP lattice.
7. Estimate optional global positive gains and apply average or feather
   blending.
8. Return the partial typed panorama, recovered intrinsics/orientations,
   contribution maps, residuals, rejected pairs and diagnostics.

The reusable :mod:`panorai.calibration` and
:mod:`panorai.model_fitting` namespaces are not private stitching helpers.
Their pinhole geometry, pair calibration, RANSAC/MSAC, rotation and homography
contracts can be invoked directly or adapted by another Pipeline Stage.

Model families and scene limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A camera rotating about one optical centre admits a central panoramic model
  and is the primary target.
* Homography currently validates pairwise planar transfer evidence; it is not
  reported as recovered 3D camera geometry or used as a separate mosaic mode.
* Translation and parallax are rejected by the current profile. A later local
  warp or seam-selection mode requires its own measured contract.
* Strong parallax, independently moving objects, rolling shutter, incompatible
  unknown lens distortion or multiple disconnected overlap groups can make a
  single central panorama invalid.  Such cases require rejection, separate
  components or a later depth-aware representation.

Minimum diagnostics include a disconnected overlap graph, insufficient
texture, repetitive matches, degenerate motion, unobservable intrinsics,
excessive reprojection error, strong parallax, photometric inconsistency and
unsupported output regions.

Stitching API placement
~~~~~~~~~~~~~~~~~~~~~~~

* **Geometry** owns camera/ray models, coordinate conversion, warps and output
  projection.  It does not choose matches or decide which model explains a
  scene.
* **Spherical operations** own local image transforms and photometric
  primitives.
* **Estimators** own pairwise and global geometric inference with uncertainty
  and diagnostics.
* **Pipeline Stages** own feature extraction, overlap discovery, evidence
  routing, estimation, warping, gain compensation and fusion as composable
  steps.
* **Application** owns the ergonomic ``N images -> panorama`` workflow and
  exposes its Pipeline for lower-level control.
* **Research evaluation** measures registration, seam, photometric and final
  panorama quality without being embedded in the estimator.

The exact Experimental imports are cataloged under Applications, Stages,
estimators, Geometry and typed results. Presence in the catalog does not imply
Stable status.

Remaining implementation work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current profile does not estimate principal point, skew or lens
distortion; choose among central-rotation, planar or local-warp models; perform
joint camera/lens bundle adjustment; find seams; perform multi-band blending;
or compensate spatially varying photometry. These are explicit capability
gaps, not hidden implementation details.

Stitching completion gate
~~~~~~~~~~~~~~~~~~~~~~~~~

The application must be tested on synthetic camera/scene oracles and
prospectively selected public image sets.  Required cases include shuffled
input order, missing metadata, partial and disconnected overlap, exposure
differences, seam crossing, rotation-only capture, planar scenes and parallax
that must be rejected.  Evaluation records must report geometric residual,
coverage, seam and photometric quality, runtime, peak memory and failure
reasons.  Exact installed-wheel consumers and independent review are required
before any Stable claim.

Later capability families
-------------------------

After these priorities, the same contracts can support spherical optical flow,
tracking and stabilization; broader 3D reconstruction and surface fusion;
photometric processing such as HDR and tone mapping; and spectral spherical
processing.  Training frameworks, datasets and bundled checkpoints remain
outside the core mission.
