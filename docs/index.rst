PanorAi
========

Build computer-vision methods for spherical images by processing sampled
pinhole views or by composing operations directly on the ERP sphere.

Three objects
-------------

``EquirectangularImage`` is the complete spherical raster.
``GnomonicFace`` is one pinhole view. ``GnomonicFaceSet`` returns the views
together for processing, lifting and fusion.

1. Apply pinhole methods over the sphere
-----------------------------------------

Start with one operation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   views = panorama.views("icosahedron", size=512, fov=100)
   blurred = views.map(planar_blur)
   result = blurred.reconstruct(blend="gaussian")

.. image:: _static/tutorials/pinhole-workflow-sundowner.jpg
   :alt: ERP, sampled views, blurred views and fused ERP result

Sampling chooses centres, resolution, FOV and overlap. Every output returns
through its own ``face.spec``. Fusion combines only supported, valid samples:
``average`` and ``gaussian`` handle continuous fields; ``closest`` preserves
labels and masks. See :doc:`reference/samplers_blenders`.

Combine pinhole operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``process_views()`` when the callback produces one raster. A callback can
compose Gaussian smoothing and Canny, for example, and the method returns the
fused ERP. The same route accepts pretrained planar models: the
:doc:`Depth Anything view adapter <applications/pose_range/depth_anything_views>`
adds tangent intrinsics, axial-to-radial conversion, validity and fusion while
leaving the upstream model unchanged.

2. Operate directly on the sphere
---------------------------------

One native operation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panorai.core.numpy import SphericalGradient

   components = SphericalGradient(operator="scharr")(gray_nchw)
   magnitude = np.hypot(components[:, 0::2], components[:, 1::2])

The result is already an ERP; no views or fusion are created.

.. image:: _static/tutorials/spherical-image-processing.jpg
   :alt: Direct spherical blur, Scharr tangent gradient and geodesic Canny

In two installed-wheel macOS arm64 trials at 256×512, 15 reused native C++
Scharr calls measured 1.89–1.91 ms versus 22.61–27.71 ms for cube → OpenCV
Scharr → fusion (11.9–14.5×). Yaw-equivariance MAE was 0.00405 versus 0.00515.
This is a reproducible configuration, not a universal ranking; see
:ref:`image-processing-routes`.

Compose operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   smooth_0 = blur_sigma_1(gray_nchw)
   dog_0 = blur_sigma_2(gray_nchw) - smooth_0
   octave_1 = resize_half(smooth_0)
   dog_1 = blur_sigma_2(octave_1) - blur_sigma_1(octave_1)

.. image:: _static/tutorials/spherical-dog-pyramid.jpg
   :alt: Input ERP and three sphere-native DoG octaves

The same :doc:`core operations <reference/core_operations>` provide
``SphericalConv2d``, ``SphericalResize``, ``SphericalMaxPool2d``, gradients and
reductions in NumPy/C++ and Torch.

Convert pretrained CNNs
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/tutorials/spherical-cnn-porting.svg
   :alt: Planar CNN spatial layers converted to direct spherical ERP inference

Experimental adapters retain learned parameters and replace planar spatial
layers with spherical convolution, transposed convolution and pooling. The
converted CNN consumes the ERP directly. See
:doc:`Metric3D depth <tutorials/10_spherical_monocular_depth>` and
:doc:`ImageNet CAM <tutorials/09_spherical_fcn_cam>`.

3. Return the right spherical output
------------------------------------

.. list-table::
   :header-rows: 1

   * - Face result
     - Return path
     - Spherical result
   * - Dense image, depth, labels or C-channel logits
     - ``map()`` + ``reconstruct()``
     - ``EquirectangularImage``
   * - Keypoints or point detections
     - ``map_points()`` + angular NMS
     - ``SphericalPointSet``
   * - Boxes, polygons or masks
     - ``map_regions()`` + solid-angle NMS
     - ``SphericalRegionSet``
   * - Whole-view scores or embeddings
     - ``map_views()`` + coverage aggregation
     - ``SphericalViewSet``

Dense channels remain aligned and are fused independently. Points become unit
bearings for detection or feature matching; regions retain spherical support;
view embeddings are aggregated without counting overlap twice. See
:doc:`explanation/workflow-evolution` and the runnable
:doc:`tutorials/00_quick_start`.

.. code-block:: python

   dense = views.map(predict_logits, output="logits", spec=logit_spec)
   points = views.map_points(detect_keypoints, source_id="features")
   regions = views.map_regions(detect_boxes, source_id="detector")
   evidence = views.map_views(embed_face, source_id="encoder")

Use ``SphericalFeaturePipeline`` when extraction and descriptor matching should
be handled by a ready-made facade rather than composed from point callbacks.

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
