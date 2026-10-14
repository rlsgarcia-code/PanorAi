PanorAi
========

Build computer-vision methods for spherical images from two execution routes:
sample pinhole views and fuse them, or operate directly on the ERP lattice.

Three objects
-------------

``EquirectangularImage`` holds the complete spherical raster.
``GnomonicFace`` is one projected pinhole view.
``GnomonicFaceSet`` processes, lifts and fuses a collection of views.

1. Multi-view: sample, process, fuse
------------------------------------

.. code-block:: text

   EquirectangularImage
     └─ views(...) → GnomonicFaceSet[GnomonicFace × N]
          └─ map(pinhole_operation) → processed views
               └─ reconstruct(...) → EquirectangularImage

.. image:: _static/tutorials/pinhole-workflow-sundowner.jpg
   :alt: ERP sampled into gnomonic views, processed and fused into an ERP

Choose the view layout and overlap during sampling; choose how overlapping
rasters are combined during fusion. See
:doc:`reference/samplers_blenders` and the runnable :doc:`tutorials/00_quick_start`.

2. Native spherical operations
------------------------------

.. code-block:: text

   ERP → SphericalConv2d → SphericalResize → SphericalMaxPool2d → ERP

These operations sample tangent neighbourhoods directly on the ERP lattice.
They create no faces and require no fusion. NumPy/C++ and differentiable Torch
implementations share the same geometric specifications. See
:doc:`reference/core_operations`.

3. View output types
--------------------

.. list-table::
   :header-rows: 1

   * - Output from each view
     - Face-set operation
     - Spherical result
   * - Dense raster
     - ``map()`` then ``reconstruct()``
     - ``EquirectangularImage``
   * - Points
     - ``map_points()``
     - ``SphericalPointSet``
   * - Regions
     - ``map_regions()``
     - ``SphericalRegionSet``
   * - View score or embedding
     - ``map_views()``
     - ``SphericalViewSet``

Rasters are fused. Points, regions and view evidence are lifted into spherical
observations. See :doc:`explanation/workflow-evolution`.

.. toctree::
   :hidden:
   :maxdepth: 2

   start/index
   applications/index
   capabilities/index
   concepts/index
   papers/index
   reference/index
   evidence/index
   development/index
   releases/index
