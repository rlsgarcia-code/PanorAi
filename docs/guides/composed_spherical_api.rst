Composed spherical API
======================

.. note::

   The generic Pipeline, typed grid/result, routing, persistence, cache,
   general NumPy spherical-operation contracts, and generic Torch layer
   grammar are Stable. The Stable Torch device contract covers CPU
   ``float32`` and ``float64``; CUDA and MPS remain outside that guarantee.
   ``SphericalSLIC`` and the SLIC operation are separately Stable as
   ``panorai-spherical-slic/v1``. The generic application framework, DoG,
   Stereo, Mask R-CNN Workflow, and CAM/SAM instance stages remain Experimental.
   A shared namespace never promotes an Experimental name by implication. The
   original Stable object workflow remains
   ``panorai.EquirectangularImage``, ``panorai.GnomonicFace``, and
   ``panorai.GnomonicFaceSet``.

The composition API has four levels. Start at the highest level that expresses
the task; descend only when an application needs different stages.

1. Applications
---------------

SLIC
~~~~

.. code-block:: python

   from panorai.applications import SphericalSLIC

   application = SphericalSLIC(n_segments=100, compactness=10.0)
   labels = application(rgb)

``labels`` is a ``RasterField(kind="labels")`` on the canonical ERP grid.
Support and data validity remain independent, and label resampling is nearest
only. ``application.as_pipeline()`` exposes the exact lower-level composition.
The wrapper, configured operation, and historical function share the Stable
``panorai-spherical-slic/v1`` contract.

DoG features
~~~~~~~~~~~~

.. code-block:: python

   from panorai.applications import SphericalDoGFeatures

   features = SphericalDoGFeatures()(rgb, panorama_id="frame-0001")
   bearings = features.bearings
   descriptors = features.descriptors

This wrapper delegates to ``SphericalDoGSIFTExtractor``. It does not create a
second scale-space implementation. OpenCV SIFT is an external optional
dependency.

Stereo depth
~~~~~~~~~~~~

.. code-block:: python

   from panorai.applications import SphericalStereoDepth

   estimate = SphericalStereoDepth()(reference, source, R_source_from_reference, t)

The result is radial range in the unit of ``t``, plus confidence, validity and
matching diagnostics. The pose convention is
``X_source = R_source_from_reference @ X_reference + t``.

Mask R-CNN workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panorai.experimental.deep_learning import mask_rcnn

   method = mask_rcnn.workflow(
       proposal_mode="cam",
       proposal=cam_adapter,
       mask=sam_adapter,
       heuristic=instance_heuristic,
       concepts=("pipe", "equipment"),
       num_classes=3,
   )
   instances = method(rgb)

In CAM mode, CAM generates spherical search regions and SAM refines their
masks. In SAM mode, SAM generates the proposals and
``reuse_proposal_masks=True`` makes those masks the MaskHead output. The
heuristic makes abstaining per-instance decisions in the BoxHead role; it is
not described as a learned classifier. ``mask_rcnn.paper()`` remains a
deprecated compatibility alias for the original Experimental factory name.

2. Pipeline composition
-----------------------

This section describes the Stable ``panorai-spherical-pipeline/v1`` core.
The CAM/SAM proposal and instance-fusion stages shown in the Mask R-CNN method
remain Experimental.

.. code-block:: python

   from panorai.pipeline import (
       Output, PerView, Pipeline, ReconstructERP, SampleViews,
   )

   pipeline = Pipeline(
       SampleViews("fibonacci", count=100, size=256, fov=100),
       PerView.parallel(
           batch="auto",
           labels=Output(segmenter, kind="labels"),
           depth=Output(depth_model, kind="depth", units="m"),
           detections=Output(detector, kind="detections"),
       ),
       ReconstructERP(labels="closest", depth="gaussian"),
   )

``PerView.parallel`` is a logical multi-head branch, not a promise of threads.
Raster outputs carry a ``RasterGrid``; non-raster results retain face identity,
projection, support and validity. Reconstruction consumes raster branches and
keeps the other branches in the result bundle. ``Select`` and ``Attach`` route
paths such as ``views.depth`` and ``erp.labels``.

Third-party stages implement the public ``Stage`` protocol: declare
``input_kind`` and ``output_kind`` and return ``StageResult`` from
``execute(value)``. A ``Pipeline`` is itself a stage and can be nested.

Batching is deterministic. ``None`` executes each face, an integer fixes chunk
size, and ``"auto"`` is available only to adapters that declare batch support.
PanorAi does not retry silently after an out-of-memory failure.

3. Operations and Torch
-----------------------

``panorai.ops.spherical.functional`` contains direct functions. Immutable
objects with matching names are suitable for ``OnSphere`` and safe JSON
topology persistence. ``panorai.ops.spherical.algorithms`` provides SLIC, DoG,
spherical NMS, Canny, and Gaussian/Laplacian pyramids.

General NumPy operations are Stable as ``panorai-spherical-ops/v1``. SLIC is
Stable under its own ``panorai-spherical-slic/v1`` contract. DoG and the mixed
``algorithms`` module remain Experimental until their remaining gates pass.

The optional, Stable ``panorai.ops.spherical.nn`` namespace provides ``Conv2d``,
``ConvTranspose2d``, max/average pooling, global pooling, resize, interpolation,
``from_planar`` and ``validate``. Learned convolution uses strict validity
propagation. Renormalization is defined only for mathematically normalized
average/Gaussian operations; arbitrary learned weights are never
renormalized. CPU ``float32`` and ``float64`` are the Stable device contract;
CUDA and MPS remain outside that guarantee pending independent device matrices.
The validator rejects opaque functional spatial calls and custom container
semantics rather than guessing that they are sphere-safe.

Every spatial operation implements ``transform_grid(input_grid)``. Convolution
and pooling retain anchor and stride history. A reduced lattice cannot be
attached or reconstructed as canonical ERP without explicit regridding.

4. Contracts, persistence and architecture
-------------------------------------------

Pipeline topology is canonical JSON over an explicit PanorAi allowlist. Files
never name an arbitrary class to import. Registered Torch architectures store
topology by identifier and weights in a checksum-bound
``weights.safetensors`` file. Built-in spherical layers and
``torch.nn.Sequential`` are registered. Trusted application code may call
``register_torch_module()`` for another architecture; the same registration
must exist before loading. A custom direct ``OnSphere`` module must also
declare ``spherical_semantics="pointwise"`` at registration when it preserves
each spatial sample independently. Unknown custom modules and forward hooks
fail closed.

Local cache entries use canonical JSON state, safe NumPy archives, checksums
and atomic writes; pickle is forbidden.
Cache identity includes values, support, validity, grid, layout, backend,
device, PanorAi version, pipeline configuration and registered weight identity.
Torch modules must be in eval mode. A tensor requiring gradients fails before
caching so cache reuse cannot detach autograd silently.

``method.architecture`` exposes Backbone, ProposalGenerator, RoIAlign, BoxHead,
MaskHead and RoIHeads. ``SphericalMaskRCNN`` exposes the conventional future
trainable Backbone→RPN→RoIAlign→Box/Mask topology, but execution raises
``UnboundMaskRCNNError``. No trainer, dataset framework or checkpoint is part
of PanorAi's projection core.

Remaining Experimental promotion blockers
-----------------------------------------

Each remaining application or scientific algorithm stays Experimental until
its own frozen criteria pass. These failures do not revoke the Stable generic
Pipeline, NumPy-operation, or SLIC contracts:

* DoG analytic blobs and external repeatability/localization/inlier thresholds;
* stereo analytic and independent Matterport3D/Stanford2D3D thresholds;
* real external CAM and SAM checkpoints plus annotated panoramic proposal,
  mask, semantic and support measurements;
* exact clean wheel/sdist consumers, benchmarks, license audit and independent
  review.

Missing external evidence blocks only the corresponding promotion; it is not
converted into a passing source-only claim.
