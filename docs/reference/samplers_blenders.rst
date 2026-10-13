Sampling and fusion methods
===========================

``EquirectangularImage.views()`` samples an ERP into canonical gnomonic views.
``GnomonicFaceSet.reconstruct()`` back-projects raster results and fuses only
explicitly valid contributions. Sampling determines where pinhole methods run;
fusion determines how overlapping results become one output.

Samplers
--------

The Stable object workflow accepts four layouts by name. It also accepts a
sampler object that supplies its own tangent points. ``size`` selects output
view shape and ``fov`` selects horizontal/vertical angular coverage; neither is
chosen implicitly by the sampler.

Cube
~~~~

``views("cube", ...)`` returns six fixed faces in ``front``, ``right``,
``back``, ``left``, ``up``, ``down`` order. It is the smallest predictable
all-sphere layout and is a useful default when six broad views provide enough
overlap for the face-local method.

Icosahedron
~~~~~~~~~~~

``views("icosahedron", subdivisions=n, ...)`` starts from 12 symmetric
icosahedron vertices. Each subdivision adds normalized edge midpoints and
increases directional density. Use it when more uniform overlap than a cube is
needed; the view count is controlled by ``subdivisions``, not ``count``.

Fibonacci
~~~~~~~~~

``views("fibonacci", count=n, ...)`` uses a golden-angle construction for a
requested number of nearly uniform directions. It is useful for controlled
coverage/compute studies and for dense local-model evaluation without the
hierarchical levels of the icosahedron sampler.

Spiral
~~~~~~

``views("spiral", count=n, ...)`` returns directions ordered along a simple
spherical spiral. Use it when deterministic traversal order is useful. It is
not an equal-area partition and should not be assumed to give the same local
overlap as Fibonacci sampling at the same count.

Custom sampler objects
~~~~~~~~~~~~~~~~~~~~~~

Pass an object exposing ``get_tangent_points() -> [(latitude_deg,
longitude_deg), ...]`` as the first argument to ``views()``. The object owns
its direction generation and reproducibility. ``count`` and ``subdivisions``
must then be configured by the object rather than passed to ``views()``.

The registry also contains ``BlueNoiseSampler`` for stochastic, well-spaced
research sampling. Supply it as an object and control the random seed outside
PanorAi when exact reproduction matters; it is not a named Stable shortcut of
the object workflow.

The optional ``rotations=((dlat, dlon), ...)`` argument appends rotated copies
of the selected base layout. This increases overlap; it does not rotate the
source ERP or replace a calibrated camera pose.

Blenders
--------

Blender validity contract
~~~~~~~~~~~~~~~~~~~~~~~~~

Every contribution carries a support/validity mask. Black RGB, label zero, and
zero depth participate when their mask is true; a non-zero value is ignored
when its mask is false. Valid floating values must be finite.

``reconstruct()`` accepts ``average``, ``gaussian``, and ``closest`` directly
by name on NumPy and Torch data. A modality bundle selects independently, for
example ``blend={"image": "gaussian", "labels": "closest"}``. Additional
blender objects below are supported for NumPy workflow data.

Average
~~~~~~~

``blend="average"`` computes a uniform mean over explicitly valid overlapping
continuous samples. It is the default for image and depth modalities. Do not
use it for categorical label IDs.

Gaussian
~~~~~~~~

``blend="gaussian"`` weights valid contributions by angular proximity to each
face centre. It reduces the influence of the more distorted face boundaries
and is usually the first fusion policy to test for RGB, logits, and continuous
range.

Closest
~~~~~~~

``blend="closest"`` selects the valid contribution farthest inside a face's
support boundary. It is the default for labels because it preserves one
discrete value instead of averaging class IDs.

Feathering
~~~~~~~~~~

``FeatheringBlender`` uses a normalized distance transform inside every
explicit mask and smoothly reduces contribution near support boundaries. Pass
the object to ``reconstruct()`` for NumPy continuous rasters.

Huber
~~~~~

``HuberBlender`` estimates a robust per-value location across valid overlaps.
It can reduce isolated numeric outliers but costs more than the first-class
average/Gaussian paths. It preserves the input raster shape and is available
for NumPy workflow data.

Diagnostic blenders
~~~~~~~~~~~~~~~~~~~

``OverlapCounterBlender``, ``OverlapStdBlender``, and
``OverlapStdFeatheredBlender`` produce overlap count or dispersion rasters.
They are audit products with shape ``(H, W, 1)``, not fused predictions.

Experimental blenders
~~~~~~~~~~~~~~~~~~~~~

``HuberNoConfidenceBlender`` and ``HuberSpatialBlender`` are research robust
estimators with tested mask/shape contracts but no reference numerical oracle.
``BundleAdjustmentBlender`` accepts scalar radius maps only and is not part of
the Stable projection path.
