Spherical Ops and layers catalog
================================

Spherical Ops are reusable computations whose spatial behavior is explicit.
Configured NumPy objects can run directly, through
``EquirectangularImage.filter()``, or inside ``OnSphere``.  Functions provide
the shortest call when configuration reuse is unnecessary.  Torch layers are
ordinary modules with spherical sampling and grid/mask transforms.

Registered operation and layer surfaces
---------------------------------------

.. panorai-api-components:: operation

.. panorai-api-components:: torch-layer

NumPy functional and configured pairs
-------------------------------------

All entries below are Stable and imported from ``panorai.ops.spherical``;
their function forms live in ``panorai.ops.spherical.functional``.

.. list-table:: Direct-sphere NumPy operations
   :header-rows: 1
   :widths: 24 24 18 34

   * - Configured object
     - Function
     - Output class
     - Spherical responsibility
   * - ``Resize``
     - ``resize``
     - resampled ERP
     - Periodic longitude, pole-aware interpolation and explicit modality rules.
   * - ``Rotate``
     - ``rotate``
     - rotated ERP
     - 3D spherical rotation rather than a planar image rotation.
   * - ``Resample``
     - ``resample``
     - regridded ERP
     - Explicit source-to-target spherical lattice conversion.
   * - ``Filter2D``
     - ``filter2d``
     - filtered ERP
     - Latitude-aware neighbourhood sampling with seam continuity.
   * - ``GaussianBlur`` / ``BoxBlur`` / ``MedianBlur`` / ``BilateralFilter``
     - ``gaussian_blur`` / ``box_blur`` / ``median_blur`` / ``bilateral_filter``
     - filtered ERP
     - Familiar filters with spherical neighbourhood semantics.
   * - ``Sobel`` / ``Gradient`` / ``Laplacian`` / ``Canny``
     - ``sobel`` / ``gradient`` / ``laplacian`` / ``canny``
     - derivative or edge field
     - Differential/edge operators that respect the spherical lattice.
   * - ``Dilate`` / ``Erode`` / ``Opening`` / ``Closing``
     - ``dilate`` / ``erode`` / ``opening`` / ``closing``
     - scalar or boolean ERP
     - ``GeodesicDisk`` neighbourhoods measured in great-circle degrees, with
       an adjoint finite-lattice contract.
   * - ``AreaAverage`` / ``AreaHistogram``
     - ``area_average`` / ``area_histogram``
     - scalar or histogram
     - Solid-angle-weighted reductions instead of equal ERP-pixel weighting.

Algorithms
----------

.. list-table:: Multi-step spherical algorithms
   :header-rows: 1
   :widths: 24 13 23 40

   * - Component
     - Stability
     - Output
     - Use
   * - ``SLIC``
     - Stable
     - integer ERP labels
     - Intrinsic, seam-periodic spherical superpixels; also powers ``SphericalSLIC``.
   * - ``GaussianPyramid``
     - Stable
     - tuple of spherical scale levels
     - Multi-scale smoothing without treating ERP pixels as equal-area samples.
   * - ``LaplacianPyramid``
     - Stable
     - reconstructible band tuple
     - Spherical multi-resolution decomposition.
   * - ``SphericalNMS``
     - Stable
     - filtered observations
     - Suppression by angular or spherical overlap rather than planar distance.
   * - ``DoGDetector``
     - Experimental
     - spherical keypoint set
     - Thin configured adapter around the direct spherical DoG detector.

Optional Torch layers
---------------------

The Stable ``panorai.ops.spherical.nn`` surface contains:

.. list-table:: Sphere-aware Torch layer families
   :header-rows: 1
   :widths: 28 25 47

   * - Layer or helper
     - Spatial effect
     - Contract
   * - ``Conv2d`` / ``SphericalConv2d``
     - neighbourhood transform
     - Spherical sampling, transformed grid, strict learned-weight validity propagation.
   * - ``ConvTranspose2d`` / ``SphericalConvTranspose2d``
     - learned upsampling
     - Explicit output lattice and support propagation.
   * - ``MaxPool2d`` / ``SphericalMaxPool2d``
     - reduced lattice
     - Declared invalid handling and updated anchors/stride.
   * - ``Dilate`` / ``Erode`` / ``Opening`` / ``Closing``
     - unchanged ERP lattice
     - Intrinsic nearest-sampled morphology with strict masks and autograd.
   * - ``AvgPool2d`` / ``AdaptiveAvgPool2d`` / ``GlobalAveragePool``
     - reduced or scalar output
     - Spherical area weighting where applicable.
   * - ``Resize`` / ``Interpolate`` / ``Rotate``
     - resampled lattice
     - Explicit spherical resampling rather than planar functional calls.
   * - ``from_planar``
     - model conversion
     - Port supported planar modules while sharing learned parameters.
   * - ``validate``
     - no data output
     - Fail-closed report of spherical layers, forbidden planar layers and grid transitions.
   * - ``execute_with_masks`` / ``transform_grid`` / ``transform_mask``
     - execution metadata
     - Keep values, support, validity and spatial identity synchronized.

Torch is optional and lazily imported.  Stable device evidence covers CPU
``float32`` and ``float64``; other devices are not implied by the API tier.

See :doc:`the Spherical Ops tutorial <../../guides/api-levels/spherical_ops>`,
:doc:`the operation concept <../../concepts/spherical-operations>`, and the
signature-level :doc:`operations API <../spherical_ops>`.

For the neighbourhood definition and finite-lattice laws, use the dedicated
:doc:`spherical morphology concept <../../concepts/spherical-morphology>` and
:doc:`tutorial <../../guides/tutorials/spherical_morphology>`.
