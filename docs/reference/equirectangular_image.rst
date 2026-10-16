EquirectangularImage API
========================

The Stable ``panorai-object-workflow/v1`` container is available from both
``panorai.EquirectangularImage`` and ``panorai.data.EquirectangularImage``.
Pillow-style sizes are ``(width, height)``; canonical geometry shapes and
``GnomonicSpec.output_shape_hw`` are ``(height, width)``.

Construction and I/O
--------------------

.. autoclass:: panorai.EquirectangularImage

.. automethod:: panorai.EquirectangularImage.open

.. automethod:: panorai.EquirectangularImage.sample

.. automethod:: panorai.EquirectangularImage.constant

.. automethod:: panorai.EquirectangularImage.new

.. automethod:: panorai.EquirectangularImage.fromarray

.. automethod:: panorai.EquirectangularImage.from_pil

.. automethod:: panorai.EquirectangularImage.to_pil

.. automethod:: panorai.EquirectangularImage.save

``save()`` writes one raster field, not the typed multimodal contract.  Floating
RGB/RGBA output is quantized from an explicit value range; invalid samples use
``invalid_fill`` only in the encoded copy.

Image-style spherical transforms
--------------------------------

.. automethod:: panorai.EquirectangularImage.copy

.. automethod:: panorai.EquirectangularImage.convert

.. automethod:: panorai.EquirectangularImage.resize

.. automethod:: panorai.EquirectangularImage.thumbnail

.. automethod:: panorai.EquirectangularImage.rotate

.. automethod:: panorai.EquirectangularImage.transpose

.. autoclass:: panorai.SphericalTranspose
   :members:

Projection, filtering and composition
-------------------------------------

.. automethod:: panorai.EquirectangularImage.crop

.. automethod:: panorai.EquirectangularImage.filter

.. automethod:: panorai.EquirectangularImage.composite

.. automethod:: panorai.EquirectangularImage.paste

The lower-level ``views()``, ``process_views()``, ``with_depth()``,
``with_labels()``, ``with_field()`` and ``apply_spherical()`` methods remain
part of the same Stable object workflow.  See :doc:`pipeline` and
:doc:`spherical_ops` for composition and operation contracts.

Properties
----------

``size``, ``width`` and ``height`` use image-style dimensions. ``shape_hw`` is
the geometry-style shape. ``mode`` is the logical image mode, ``dtype`` and
``backend`` describe storage, and ``format`` records the decoded file format
until a transform creates a new raster.
