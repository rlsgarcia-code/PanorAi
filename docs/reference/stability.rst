API stability in 3.1
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
     - Huber spatial/no-confidence and bundle-adjustment blenders
     - Shape and mask behavior is tested; numerical quality lacks an
       independent reference oracle and may evolve with documentation.
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
