Applications: solve the task first
==================================

Choose an Application when the method is already known and only its parameters
or inputs change. A wrapper validates configuration, invokes existing PanorAi
algorithms and returns a public typed result; it does not duplicate the
algorithm.

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

4. Estimate Essential, R and translation direction
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

5. Descend when the method changes
----------------------------------

``application.as_pipeline()`` exposes the equivalent composition. Descend to
:doc:`Pipeline <pipeline>` when you need a different stage, branch, batch or
fusion policy. Descend to :doc:`Spherical Ops <spherical_ops>` when the
operation itself changes.

Current application boundary
----------------------------

``SphericalSLIC`` is Stable. ``SphericalDoGFeatures``,
``SphericalStereoDepth``, ``SphericalRelativePose`` and
``SphericalMaskRCNNWorkflow`` are executable Experimental wrappers with separate
scientific and dependency gates. A common wrapper shape does not imply common
stability.

Understand the design in :doc:`Application wrappers
<../../concepts/application-wrappers>` and look up signatures in
:doc:`Applications API <../../reference/applications>`.
