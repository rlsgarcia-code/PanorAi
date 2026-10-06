Spherical image processing
==========================

For a visual, end-to-end walkthrough using a reproducible online CC0
panorama, start with :doc:`../tutorials/spherical_image_processing`.

.. currentmodule:: panorai.image_processing

``panorai.image_processing`` is an Experimental, NumPy-first counterpart to
the familiar OpenCV image-processing progression.  It does not apply a planar
kernel directly to an equirectangular raster.  Instead, every kernel is placed
in the local east/down tangent plane and its taps are carried to the unit
sphere with the exponential map.  The resulting angular footprint is the same
at the equator, near the longitude seam, and at high latitude.

The module uses the canonical PanorAi frame (``+X`` right, ``+Y`` up, ``+Z``
forward), pixel-centre ERP coordinates, horizontal seam wrapping, and polar
row clamping.  It currently accepts NumPy ``HW`` and ``HWC`` arrays.  Filtering
produces ``float32`` for ``float32`` input and ``float64`` otherwise, so signed
edge responses are never silently clipped to ``uint8``.

Convolution and smoothing
-------------------------

``spherical_filter2d`` follows OpenCV's correlation convention: the supplied
kernel is not flipped.  ``angular_step_deg`` controls the angular distance
between adjacent taps and defaults to one vertical ERP pixel (``180 / H``
degrees).  Compatible ``float32``/``float64`` inputs automatically use the
first-party C++17 kernel when it is installed; ``backend="numpy"`` selects the
reference implementation and ``backend="native"`` requires the compiled
path.  Both paths implement the same public contract.

.. code-block:: python

   import numpy as np
   from panorai.image_processing import (
       spherical_bilateral_filter,
       spherical_box_blur,
       spherical_filter2d,
       spherical_gaussian_blur,
       spherical_median_blur,
   )

   erp = np.random.default_rng(7).random((256, 512), dtype=np.float32)
   sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)

   sharpened = spherical_filter2d(erp, sharpen)
   average = spherical_box_blur(erp, ksize=5)
   gaussian = spherical_gaussian_blur(erp, ksize=5, sigma=1.2)
   median = spherical_median_blur(erp, ksize=5)
   bilateral = spherical_bilateral_filter(
       erp, ksize=5, sigma_color=0.08, sigma_space=1.2
   )

The box and Gaussian operators reuse spherical convolution.  Median and
bilateral filtering use the same geodesic neighbourhood but are nonlinear;
their first implementation is the NumPy reference path.

Geometric transforms
--------------------

``spherical_rotate`` takes a proper 3×3 rotation that actively maps source
rays to output rays.  It reverse-maps every output pixel through ``R.T`` and
therefore has no uncovered border.  ``spherical_resize`` samples the exact
pixel-centre rays of the requested ERP shape.  Resize does not silently add an
antialiasing policy; Gaussian pyramids prefilter explicitly before reducing.

.. code-block:: python

   angle = np.deg2rad(30.0)
   yaw = np.array(
       [
           [np.cos(angle), 0.0, np.sin(angle)],
           [0.0, 1.0, 0.0],
           [-np.sin(angle), 0.0, np.cos(angle)],
       ]
   )
   turned = spherical_rotate(erp, yaw)
   half = spherical_resize(turned, (128, 256))

Gradients, Laplacian, and Canny
-------------------------------

``spherical_gradient`` returns signed east/north components, magnitude, and
orientation.  ``operator="sobel"`` and ``operator="scharr"`` combine local
smoothing and differentiation.  ``normalize_by_angle=True`` expresses the
derivatives per radian; the default preserves OpenCV-like one-tap response
scale.  ``spherical_laplacian`` supplies the four-neighbour second derivative.

``spherical_canny`` performs Gaussian denoising, tangent Sobel/Scharr
gradients, geodesic directional non-maximum suppression, and double-threshold
hysteresis.  Its binary result is ``HW uint8`` with values 0 and 255.  Unlike a
planar ERP implementation, both suppression and connectivity cross the
longitude seam using spherical neighbours.

.. code-block:: python

   from panorai.image_processing import (
       spherical_canny,
       spherical_gradient,
       spherical_laplacian,
   )

   gradient = spherical_gradient(erp, operator="scharr")
   laplacian = spherical_laplacian(erp)
   edges = spherical_canny(erp, threshold_low=0.05, threshold_high=0.12)

Pyramids
--------

Gaussian pyramid levels are spherical-Gaussian prefiltered and then sampled at
the next ERP raster.  Laplacian levels retain the residual against an expanded
coarser level, so ``reconstruct_laplacian_pyramid`` reconstructs the original
floating signal up to numerical precision.

.. code-block:: python

   from panorai.image_processing import (
       reconstruct_laplacian_pyramid,
       spherical_gaussian_pyramid,
       spherical_laplacian_pyramid,
   )

   gaussian_levels = spherical_gaussian_pyramid(erp, levels=4)
   laplacian_levels = spherical_laplacian_pyramid(erp, levels=4)
   reconstructed = reconstruct_laplacian_pyramid(laplacian_levels)

Histogram equalization
----------------------

An ERP contains more raster samples per unit solid angle near the poles.
``spherical_equalize_histogram`` therefore weights each row by its exact
relative spherical area.  It accepts grayscale ``HW uint8`` input, which is
the direct preprocessing form used by common keypoint detectors.  Set
``area_weighted=False`` to reproduce ordinary pixel-count histogram
equalization (including OpenCV's mapping for unmasked input).

.. code-block:: python

   from panorai.image_processing import spherical_equalize_histogram

   grayscale_u8 = np.rint(erp * 255).astype(np.uint8)
   equalized = spherical_equalize_histogram(grayscale_u8)

Current boundary
----------------

The ``panorai-spherical-image-processing/v1`` surface is Experimental.  It has
analytic seam/pole and native/reference evidence, but still needs broad real
panorama feature/matching evaluation, Torch parity, performance measurements,
and calibrated Canny/scale-space studies before promotion.  Boolean masks and
categorical labels are not valid inputs for interpolating filters.  The module
does not yet define invalid-data renormalization for convolution; non-finite
floating neighbours propagate through a nonzero kernel tap.

API summary
-----------

.. automodule:: panorai.image_processing
   :members:
   :undoc-members:
