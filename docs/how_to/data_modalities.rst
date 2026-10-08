Choose modality, layout, and interpolation
===========================================

PanorAi separates three concepts:

``support_mask``
   Whether projection geometry covers an output pixel.

data validity
   Whether a supported sample is meaningful, such as a finite radial range.

numeric value
   The sample itself. Zero and black are ordinary values, not missingness.

RGB
---

Use floating ``HWC`` RGB for bilinear interpolation. Convert integer RGB to a
floating range deliberately, or select ``nearest`` when class-like integer
values must remain exact.

Labels and masks
----------------

Integer labels and boolean masks require ``nearest``. Bilinear interpolation
of these modalities fails explicitly rather than creating fractional labels or
soft booleans.

Depth
-----

Depth means radial range in the canonical frame. State units alongside the
array (metres in this example). Floating ``NaN`` stays invalid data and must
not silently become a valid zero. Support remains a separate boolean mask.

The default bilinear policy, ``propagate``, keeps strict IEEE ``NaN``
propagation. For sparse measurements, opt into ``renormalize`` with an
explicit boolean validity mask and a scientifically chosen
``min_valid_weight``. The result separately reports geometric
``support_mask``, thresholded ``validity_mask``, and the contributing
``valid_weight``. PanorAi does not choose one threshold for every sensor or
modality.

Dataset-neutral XYZ images
--------------------------

``XYZImageDataset`` loads organized PLY rasters without embedding dataset
names or split rules in PanorAi. The consumer supplies an explicit directory
or ordered file list, coordinate frame, units, and optional shadow angle::

   from panorai.data import XYZImageDataset

   samples = XYZImageDataset(
       files=authorized_files,
       coordinate_frame="scanner-local-right-up-forward",
       units="metres",
       shadow_angle=0.0,
   )

Each filename declares ``_HxW`` before the optional ``_encrypted`` suffix.
Each sample returns ``rgb_image``, the original ``xyz_image``, derived
``radial_depth``, ``validity_mask``, frame, units, shadow angle, and source
path. Split construction and licensed Matterport3D or Stanford2D3D access
remain application responsibilities. Importing ``panorai.data`` does not
import Open3D; PLY decoding loads that optional dependency only when needed.

Stable object workflow
----------------------

The Stable ``panorai-object-workflow/v1`` surface records these semantics
without a user-authored channel dictionary::

   import panorai as pa

   pano = (
       pa.EquirectangularImage(rgb)
       .with_depth(radial_range_m, valid=depth_is_valid, units="m")
       .with_labels(labels)
   )
   views = pano.views("cube")

``views`` projects each modality separately: image and depth use bilinear,
while labels use nearest. ``views.describe()`` reports the resolved policy.
Legacy dictionaries remain accepted by the existing methods, but the typed
workflow rejects an untyped dictionary rather than guessing its semantics.

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :dedent: 4
   :start-after: DOCS_MODALITIES_START = None
   :end-before: DOCS_MODALITIES_END = None

Mask-aware blending
-------------------

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :dedent: 4
   :start-after: DOCS_BLENDER_START = None
   :end-before: DOCS_BLENDER_END = None
