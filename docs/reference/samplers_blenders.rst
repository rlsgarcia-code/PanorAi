.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/reference/samplers_blenders.rst  ── dropdown tables
.. ───────────────────────────────────────────────────────────────

.. cut:start

Samplers
========


.. dropdown:: Available samplers

   .. list-table::
      :widths: 20 80
      :header-rows: 1

      * - Name
        - Description
      * - ``cube``
        - 6 orthogonal faces.
      * - ``icosahedron``
        - Vertices of an icosahedron; hierarchical density.
      * - ``fibonacci``
        - Nearly uniform distribution via golden-angle spiral.
      * - ``spiral``
        - Simple incremental spiral over sphere.
      * - ``blue_noise``
        - Random yet well-spaced points.


Blenders
========

Blender validity contract
-------------------------

The ``masks`` argument is authoritative. Each mask is a boolean-compatible
``(H, W)`` array (``(H, W, 1)`` and channel-matched masks are also accepted).
Pixel values never imply validity: black RGB and zero depth participate when
their mask is true, while a non-zero value is ignored when its mask is false.
Valid floating samples must be finite.

Fusion blenders return the same shape as one input image. Pass
``return_mask=True`` to receive ``(data, support_mask)``; the returned mask is
the union of contributing support. Diagnostic overlap blenders return
``(H, W, 1)`` data plus the same optional support mask.

.. dropdown:: Available blenders

   .. list-table::
      :widths: 20 20 60
      :header-rows: 1

      * - Name
        - Stability
        - Description
      * - ``average``
        - Supported
        - Uniform mean of explicitly valid overlaps.
      * - ``gaussian``
        - Supported
        - Distance-based Gaussian weights using real gnomonic back-projection
          support.
      * - ``feathering``
        - Supported
        - Distance-to-boundary feathering inside each explicit mask.
      * - ``closest``
        - Supported
        - Pixel farthest inside a contributing face's support boundary.
      * - ``huber``
        - Supported
        - Per-value robust location estimate preserving the input shape.
      * - ``counter``, ``std``, ``std_feathered``
        - Diagnostic
        - Overlap count or dispersion maps; output shape is ``(H, W, 1)``.
      * - ``huber_no_confidence``, ``huber_spatial``
        - Experimental
        - Research robust estimators with a tested mask/shape contract but no
          reference numerical oracle yet.
      * - ``bundle_adjustment``
        - Experimental
        - Scalar-radius optimization only; not part of the stable projection
          path.

.. cut:end
