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
