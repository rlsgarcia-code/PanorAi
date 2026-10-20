Applications: solve the task first
==================================

Choose an Application when the method is already known and only its parameters
or inputs change. A wrapper validates configuration, invokes existing PanorAi
algorithms and returns a public typed result; it does not duplicate the
algorithm.

.. panorai-api-components:: application

1. Run the smallest Stable application
---------------------------------------

The following example is self-contained and executed in CI. It uses intrinsic
spherical SLIC, so no projected faces are created.

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_START = None
   :end-before: DOCS_LEVEL_APPLICATION_END = None
   :dedent: 4

2. Reconfigure immutably
------------------------

Application instances are configurations. ``with_params`` returns a new
instance and leaves the original untouched:

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_CONFIG_START = None
   :end-before: DOCS_LEVEL_APPLICATION_CONFIG_END = None
   :dedent: 4

Use ``get_params()`` for tuning tools and ``describe()`` for an auditable view
of input, output, execution domain and lower-level composition.

3. Persist only registered configurations
------------------------------------------

``SphericalSLIC.save("slic.json")`` writes canonical JSON through an explicit
registry. Loading never imports a class named by the file, and arbitrary
callables are not serialized. This makes the saved configuration portable
without using pickle.

4. Use the fixed DoG candidate without hiding its status
---------------------------------------------------------

The reviewed DoG factory fixes the complete detector, tangent-patch and
RootSIFT configuration. It is useful for reproducible research now, but it is
still Experimental while its prospective external gates remain open::

   from panorai import EquirectangularImage
   from panorai.applications import SphericalDoGFeatures

   panorama = EquirectangularImage.open("room_360.jpg")
   method = SphericalDoGFeatures.reviewed_profile()
   features = method(panorama, panorama_id="room-360")

   assert method.profile_id == "panorai-spherical-dog-features/reviewed-v1"
   print(features.bearings, features.descriptors)

Backend selection is not a scientific change::

   numpy_method = method.with_params(config__convolution_backend="numpy")
   assert numpy_method.profile_id == method.profile_id

Changing the contrast threshold is a different method. The immutable update
therefore drops the reviewed identity and returns the generic Experimental
wrapper::

   custom = method.with_params(config__contrast_threshold=0.001)
   assert custom.profile_id is None

The exact protocol and cohort-freezing commands are documented in
:doc:`Spherical application promotion protocol
<../../development/spherical_application_promotion>`.

5. Estimate Essential, R and translation direction
---------------------------------------------------

``SphericalRelativePose`` accepts an ordered pair of ERP images. Its named
reference profile includes both the frozen frontend and the selected
MSAC-first/all-inlier-refit estimator; selecting another extractor, matcher or
pose strategy is therefore explicit::

   from panorai.applications import SphericalRelativePose

   application = SphericalRelativePose.reference()
   pose = application(reference, source)
   if pose is None or not pose.quality_report.accepted:
       raise RuntimeError("relative pose was not trustworthy")

   E = pose.essential_matrix
   R_source_from_reference = pose.R
   t_direction_source_from_reference = pose.t

The translation is a unit direction, never metric scale. The equivalent
composition is visible and configurable instead of being hidden inside the
wrapper::

   pipeline = application.as_pipeline()
   print(tuple(pipeline.named_steps))
   # extract_features, match_features, bearing_correspondences, estimate_pose

Capture-time evidence remains separate. When independently registered depth
is available, prepend the scanner-assisted eligibility Stage::

   from panorai.applications import (
       RelativePoseCaptureMetadata,
       RelativePoseCapturePolicy,
   )

   application = SphericalRelativePose.reference(
       capture_policy=RelativePoseCapturePolicy.scanner_assisted_reference()
   )
   pose = application(
       reference,
       source,
       capture_metadata=RelativePoseCaptureMetadata(
           registered_cloud_overlap_min=0.72,
       ),
   )

The measured 50% overlap minimum is an eligibility condition, not a pose
quality guarantee. The post-estimation ``quality_report.accepted`` decision
remains mandatory.

These are sparse feature/geometry stages. No raster ``OnSphere`` operation or
Torch layer is required: feature extraction runs on canonical projected views,
matching and bearing conversion are non-differentiable algorithms, and the
Essential estimator runs directly on spherical bearings.

6. Descend when the method changes
----------------------------------

``application.as_pipeline()`` exposes the equivalent composition. Descend to
:doc:`Pipeline <pipeline>` when you need a different stage, branch, batch or
fusion policy. Descend to :doc:`Spherical Ops <spherical_ops>` when the
operation itself changes.

7. Construct a panorama from overlapping planar images
-------------------------------------------------------

``PanoramaStitcher`` is an Experimental, non-flagship Application for a
central camera rotating about one optical centre.  It accepts two or more
overlapping planar pinhole images and returns a partial typed ERP plus
registration diagnostics.  Its ``as_pipeline()`` exposes feature detection,
overlap discovery, calibration/registration, warping, exposure compensation
and blending as public Stages.

Use :doc:`the dedicated guide <../registration_stitching/index>` before
running it: missing metadata invokes explicit shared-FOV assumptions, and
translation, strong parallax, rolling shutter and unknown lens distortion are
outside the current model.

Current application boundary
----------------------------

``SphericalSLIC`` is Stable. ``SphericalDoGFeatures.reviewed_profile()`` is a
fixed but still Experimental promotion candidate. The generic
``SphericalDoGFeatures``,
``SphericalStereoDepth``, ``SphericalRelativePose`` and custom
``FullSphereDenseStereo`` configurations are Experimental. The exact
``FullSphereDenseStereo.reviewed_profile()`` supplied-metric-pose entry point
is Stable. ``SphericalMaskRCNNWorkflow`` and ``PanoramaStitcher`` are executable
Experimental wrappers with separate scientific and dependency gates. A common
wrapper shape does not imply common stability.

Understand the design in :doc:`Application wrappers
<../../concepts/application-wrappers>` and look up signatures in
:doc:`Applications API <../../reference/applications>`.
