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

Sampling chooses face centres, pinhole resolution, field of view, and therefore
overlap. Every output pixel is back-projected through its face specification;
fusion then combines only supported, valid contributors. ``average`` forms a
uniform continuous mean, ``gaussian`` favours face centres, and ``closest``
preserves one discrete value.

Keep ``views().map().reconstruct()`` explicit to inspect or batch faces. For a
single raster operation, ``EquirectangularImage.process_views()`` performs the
same complete path and returns the reconstructed ERP directly. See
:doc:`reference/samplers_blenders` and the runnable :doc:`tutorials/00_quick_start`.

2. Native spherical operations
------------------------------

.. code-block:: text

   ERP ─┬─ spherical blur σ₁ ─┐
        └─ spherical blur σ₂ ─┴─ subtract → DoG octave 0
                       └─ SphericalResize(0.5) → DoG octave 1

.. image:: _static/tutorials/spherical-dog-sift.jpg
   :alt: Sphere-native multiscale DoG detections and tangent descriptors

These operations sample tangent neighbourhoods directly on the ERP lattice.
They create no faces and require no fusion. NumPy/C++ and differentiable Torch
implementations share the same geometric specifications. See
:doc:`reference/core_operations`.

The same core ports pretrained CNNs by preserving learned parameters and
replacing ``Conv2d`` with ``SphericalConv2d``, ``ConvTranspose2d`` with
``SphericalConvTranspose2d``, and ``MaxPool2d`` with
``SphericalMaxPool2d``. The Experimental Metric3D adapter returns radial
range; the ImageNet FCN adapter returns dense class evidence and CAMs. Both
consume an ERP directly rather than evaluating projected faces.

.. image:: _static/tutorials/spherical-cnn-porting.svg
   :alt: Pretrained planar CNN layers converted to direct spherical depth and CAM inference

See :doc:`tutorials/10_spherical_monocular_depth` and
:doc:`tutorials/09_spherical_fcn_cam` for the model, artifact and evidence
contracts.

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
