Quick start
===========

This page starts with one complete Stable result, then reveals the three lower
levels that implement it. The examples use synthetic ``HWC`` float RGB so they
run without downloading an image, model or checkpoint.

Install
-------

Install the latest published package with:

.. code-block:: console

   python -m pip install panorai

This site follows ``main``. APIs explicitly marked as belonging to the
unreleased source line require an installed wheel built from that line until
the next release is published. Development checkouts can use
``python -m pip install -e .``. See :doc:`../reference/stability` before
depending on an Experimental name.

Load the spherical image object
-------------------------------

Real workflows normally begin with an owned container rather than an anonymous
array:

.. code-block:: python

   from panorai import EquirectangularImage

   panorama = EquirectangularImage.open("room_360.jpg")
   print(panorama.size, panorama.mode)

The object provides immutable spherical ``resize``, 3D ``rotate``, tangent
``crop``, registered ``filter`` and projection-aware ``paste`` operations.
See the :doc:`spherical image object tutorial
<../guides/tutorials/equirectangular_image>` for an executable progression and
the :doc:`design rationale <../concepts/equirectangular-image>` for the precise
differences from a planar Pillow image.

1. Run an application
---------------------

``SphericalSLIC`` is the smallest complete example: it validates configuration,
runs intrinsic spherical segmentation and returns a typed result.

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_START = None
   :end-before: DOCS_LEVEL_APPLICATION_END = None
   :dedent: 4

``labels.values`` is an integer ``HW`` raster. ``labels.grid`` says where those
values live, while ``labels.support`` and ``labels.validity`` answer different
questions: whether geometry covers a sample and whether valid data contributed.

2. Reveal the composition
-------------------------

Every Pipeline-backed application can expose its lower-level form:

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_APPLICATION_CONFIG_START = None
   :end-before: DOCS_LEVEL_APPLICATION_CONFIG_END = None
   :dedent: 4

The same block also demonstrates immutable reconfiguration. Use the application
while its behavior fits; move to Pipeline when a stage, branch or reconstruction
policy must change.

3. Choose the next level
------------------------

.. list-table:: Continue according to the decision you need to own
   :header-rows: 1
   :widths: 20 38 42

   * - Level
     - Choose it when
     - Continue
   * - Applications
     - Parameters change, but the method does not
     - :doc:`Configure and persist wrappers <../guides/api-levels/applications>`
   * - Pipeline
     - Stages, branches, batching or fusion change
     - :doc:`Compose direct and projected processing <../guides/api-levels/pipeline>`
   * - Spherical Ops
     - The operation or neural layer itself changes
     - :doc:`Apply sphere-aware operations <../guides/api-levels/spherical_ops>`
   * - Geometry
     - Rays, projections or sampling policy change
     - :doc:`Control geometry explicitly <../guides/api-levels/geometry>`

4. Keep the contract visible
-----------------------------

Before using real depth, labels or masks, read
:doc:`Data modalities <../guides/how-to/data_modalities>`. Labels and masks use
nearest sampling. Continuous depth requires explicit validity and units. The
mathematical convention is defined normatively in :doc:`../geometry-v1`.
