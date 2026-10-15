Composed spherical API migration
================================

Status and compatibility
------------------------

The three Stable workflow containers remain the compatibility foundation:
``EquirectangularImage``, ``GnomonicFace``, and
``GnomonicFaceSet``. Existing imports in ``panorai.image_processing``,
``panorai.features`` and ``panorai.stereo`` remain available. The new
generic Pipeline and general NumPy operation contracts are Stable. Application
wrappers, SLIC, DoG, optional Torch layers, and CAM/SAM instance stages remain
Experimental until their own tracked promotion gates pass.

From manual view workflows to Pipeline
--------------------------------------

The existing Stable form remains valid:

.. code-block:: python

   result = panorama.views("cube", size=256).map(model).reconstruct(
       blend="gaussian"
   )

The equivalent composable form is:

.. code-block:: python

   from panorai.pipeline import Pipeline, SampleViews, PerView, ReconstructERP

   result = Pipeline(
       SampleViews("cube", size=256),
       PerView(model),
       ReconstructERP(blend="gaussian"),
   )(panorama)

Third-party stages now implement public ``execute(value) -> StageResult``
with declared ``input_kind`` and ``output_kind``. They do not subclass a
private base or implement ``_run``.

Typed results and grids
-----------------------

Spatial Pipeline outputs are ``RasterField`` values carrying support,
validity and ``RasterGrid``. Classification and embedding outputs are
``ClassificationResult`` and ``EmbeddingResult``. Mixed branches use
``ResultBundle``; ``ReconstructERP`` fuses only raster branches and
retains the rest.

Shape-changing operations create a derived spherical lattice. They cannot be
attached as canonical ERP fields until an explicit regridding operation
returns them to ERP pixel centres.

Applications
------------

The wrappers ``SphericalSLIC``, ``SphericalDoGFeatures``,
``SphericalStereoDepth`` and ``SphericalMaskRCNNPaper`` expose
``run()``, ``__call__()``, parameters, immutable variants, descriptions
and their lower-level Pipeline. They delegate to the existing algorithms.

Mask R-CNN users should distinguish:

* ``mask_rcnn.paper(...)`` returns the executable paper method. CAM or SAM
  supplies proposals, SAM supplies refinement or reusable masks, and an
  abstaining heuristic supplies instance decisions.
* ``SphericalMaskRCNN`` describes the future trainable
  Backbone/RPN/RoIAlign/Box/Mask topology. It remains Experimental and
  non-executable; PanorAi does not add a trainer, dataset framework or
  checkpoints.

Persistence and cache
---------------------

Only registered components can be saved. Arbitrary callables continue to work
in memory, but persistence fails with the exact parameter path. Pipeline
topology is canonical JSON. Registered Torch modules use a trusted identifier
registry and safetensors weights; an archive cannot select a class to import.

Torch cache entries require eval mode, cannot contain tensors that require
gradients, and include the registered weight hash. These rules prevent cache
reuse from silently disconnecting autograd or crossing model identities.

Promotion boundary
------------------

Installed-wheel execution is necessary but not sufficient for Stable
promotion of a scientific algorithm or application. The content-hashed
``contracts/spherical-application-promotion-v1.json`` freezes analytic and
external held-out gates for SLIC, DoG, StereoDepth and Mask R-CNN Paper.
Missing datasets, checkpoints, licenses or held-out results leave only that
scientific surface Experimental; they do not demote the generic Pipeline or
general NumPy operations. Thresholds are not relaxed after results are opened.
