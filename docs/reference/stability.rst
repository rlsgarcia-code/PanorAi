API stability in 3.x
====================

The tier describes support expectations, not how useful an object may be for a
particular research project.

.. list-table:: Stability tiers
   :header-rows: 1
   :widths: 18 30 52

   * - Tier
     - Surface
     - Contract
   * - Stable
     - ``panorai.geometry``; supported fusion blenders
     - Tested mathematical and mask contract. Compatible throughout 3.x;
       intentional corrections are documented in migration notes.
   * - Compatibility
     - 3.0 containers, factories, samplers, projection registry, depth
       loader/registry names, PCD names
     - Public names remain through 3.x. They adapt to canonical geometry where
       practical but do not define new geometry behavior.
   * - Experimental
     - 3.2 ergonomic ``with_*``/``views``/``map``/``reconstruct``/
       ``process_views`` workflow; ``panorai.features`` façade,
       ``panorai.estimators`` relative pose and PyCOLMAP export; Huber
       spatial/no-confidence and bundle-adjustment blenders
     - Shape and mask behavior is tested; numerical quality lacks an
       independent reference oracle and the surface may evolve with
       documentation. No 3.0 name is removed.
   * - Frozen/internal
     - Underscore-prefixed geometry implementation modules, deep vendored
       model namespaces, research training/data helpers, and release tooling
     - Public behavior is exposed through the stable API. Internal structure
       may change without creating a second geometry contract.

Depth adapter boundary
----------------------

PanorAi 3.1 retains ``ModelRegistry``, loader names and the ``dav2``,
``m3dv2``, ``dust3r`` and ``zoe`` keys as lightweight compatibility adapters.
Their upstream implementations, training code, datasets and checkpoints are
not included in the wheel or sdist. Deep paths under the formerly vendored
Depth Anything V2, DUSt3R/CroCo, Metric3D and legacy ZoeDepth trees are
experimental/internal and are not a stable 3.x API promise.

Invoking an unavailable adapter gives installation and upstream-license
guidance. PanorAi does not claim numerical equivalence without an independently
tested upstream version and checkpoint. DUSt3R remains subject to its separate
CC BY-NC-SA 4.0 terms and is never covered by PanorAi's MIT license.

Stable blenders
---------------

``average``, ``gaussian``, ``feathering``, ``closest``, and ``huber`` accept
explicit masks and preserve the input image shape. ``counter``, ``std``, and
``std_feathered`` are stable diagnostic maps with ``(H, W, 1)`` output.

Optional backends
-----------------

NumPy is required. Torch and Open3D are optional. Importing
``panorai.geometry`` must not require or load either optional backend. Torch
geometry is stable for input-value autograd on the documented layouts; Open3D
belongs to the compatibility PCD surface.

Deprecation policy
------------------

No stable public 3.0 name is removed before 4.0. This promise does not promote
deep vendored or research implementation files into stable API. Corrections to
broken behavior require regression tests and migration notes.

Experimental 3.2 workflow
-------------------------

``EquirectangularImage.with_depth()``, ``with_labels()``, ``views()`` and
``process_views()``, plus ``GnomonicFaceSet.map()``, ``reconstruct()`` and
``describe()``, form one experimental workflow over the existing objects.
They compose the stable ``geometry-v1`` engine and keep modality data,
geometric support and explicit validity separate. Promotion to Stable requires
two real consumer flows and compatibility evidence.

Experimental spherical features
-------------------------------

``panorai.features`` is versioned separately as
``panorai-spherical-features/v1``. OpenCV remains the implementation of SIFT,
ORB, AKAZE, BFMatcher, and FLANN. PanorAi provides geometry, masks, angular
deduplication, provenance, and public panorama-domain objects. Optional
PyCOLMAP export materializes virtual-camera rigs and visual evidence, while
COLMAP remains responsible for SfM. Promotion requires real downstream
Essential and rig-SfM consumers plus compatibility evidence.

Experimental spherical relative pose
------------------------------------

``panorai.estimators`` is versioned as
``panorai-spherical-relative-pose/v1``. Its first implementation owns a
polynomial five-correspondence essential kernel, locally optimized robust
consensus, spherical tangent-Sampson residuals, pose refinement, cheirality
selection, competing-model evidence, stability diagnostics, and explicit
acceptance policy. Its raw quality score is not a probability; the isotonic
calibrator requires disjoint calibration and evaluation sample IDs. It
estimates a panorama-frame rotation and unit translation direction only;
translation scale, tracks, triangulation, bundle adjustment and SfM are not
claimed. Promotion requires external geometric fixtures, a separately frozen
confidence-calibration corpus, real panorama-pair consumers, broader
degeneracy evaluation, and evidence-backed performance.
