Spherical morphology
====================

Mathematical morphology asks whether nearby values satisfy an ordering: a
dilation takes a supremum, an erosion takes an infimum, and opening/closing
compose those two operations.  On a sphere, the important question is what
``nearby`` means.

Why a Pillow-style image window is insufficient
-------------------------------------------------

A rectangular window on an equirectangular raster is not a fixed physical
neighbourhood.  Longitude pixels converge toward the poles, the left and right
columns are adjacent, and a planar disk changes angular size when the ERP is
resized.  Applying Pillow, OpenCV, or SciPy morphology directly to ERP indices
therefore makes the result depend on latitude, seam placement, and resolution.

PanorAi's ``GeodesicDisk`` instead measures its radius in degrees of
great-circle distance.  At every ERP pixel centre it samples a disk in the
local east/north tangent plane and maps that disk onto the sphere.  The
operation wraps longitude and remains meaningful at high latitudes.

The finite-lattice contract
---------------------------

For a finite ERP, a continuous geodesic disk must be sampled.  The default
sampling step follows the ERP angular lattice; callers can declare a fixed
``angular_step_deg`` when comparisons must remain independent of raster
resolution.  The sampled neighbour relation is symmetrized.  This detail gives
the discrete dilation and erosion an adjunction, so the expected morphology
laws hold exactly on the finite lattice:

* dilation is extensive and erosion is anti-extensive;
* both are monotone;
* erosion is dual to dilation under sign inversion;
* opening and closing are idempotent.

The approximation to a continuum cap is bounded by the declared sampling
lattice, rather than being silently described as exact continuum morphology.
Execution is deterministically chunked by default; ``max_sampled_elements``
can lower the temporary-memory budget or be set to ``None`` to request one
unbounded pass without changing the numerical result.

Types, masks and validity
-------------------------

The Stable NumPy v1 contract accepts boolean masks and scalar ``float32`` or
``float64`` ``HW`` fields.  It rejects multi-class labels because numeric label
order has no semantic meaning, and rejects untyped ``HWC`` channel vectors
because component-wise max/min does not define a canonical vector order.  The
Torch API accepts floating ``NCHW`` tensors and defines each channel as an
independent scalar field; it does not order vectors across channels. Sampling
is nearest-only.

Support and data validity remain separate and propagate strictly: an output is
supported/valid only when every sample required by the structuring element is
supported/valid.  The operation never treats ``False``, numeric zero, or black
as missing data.

Use :doc:`the tutorial <../guides/tutorials/spherical_morphology>` for
container, Pipeline and Torch composition, and :doc:`the API reference
<../reference/spherical_ops>` for the exact public surface.
