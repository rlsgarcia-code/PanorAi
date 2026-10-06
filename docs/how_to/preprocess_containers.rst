.. _howto-preprocess-containers:

Preprocessing Containers
========================

``EquirectangularImage`` exposes a ``preprocess()`` method so you can
materialize a scanner shadow cap, resize, or rotate a panorama before sampling
faces.

Example
-------

.. code-block:: python

    from panorai.data import EquirectangularImage

    eq = EquirectangularImage(
        observed_rgb,
        shadow_angle=20.0,
        shadow_padded=False,
    )
    eq.preprocess(delta_lat=10, resize_factor=0.5)
    faces = eq.to_gnomonic_face_set(fov=90)

The preprocessing step modifies the container in place. Subsequent projection
calls use the updated data. The Stable ``views()`` workflow and spherical
feature extraction additionally propagate the updated geometric support.

Shadow angle and raster state
-----------------------------

``shadow_angle`` is the angular size of the region that a 3D scanner cannot
capture. It is measured from the South Pole (bottom) upward.
``shadow_padded`` describes the raster state independently:

.. list-table:: Shadow-cap input states
   :header-rows: 1

   * - ``shadow_angle``
     - ``shadow_padded``
     - Meaning and action
   * - ``0``
     - ``False``
     - Complete-sphere raster; no shadow padding is needed.
   * - ``> 0``
     - ``False``
     - Only observed rows exist. ``preprocess()`` appends the missing
       south-polar rows exactly once.
   * - ``> 0``
     - ``True``
     - The south-polar rows already exist. ``preprocess()`` leaves the height
       unchanged and preserves that band as geometric non-support.

``shadow_padded=True`` with a zero angle is rejected. Once padding has been
materialized, changing the angle or requesting padding again is also rejected.
Projection and feature extraction reject a panorama with a positive angle and
``shadow_padded=False`` until ``preprocess()`` has materialized the rows.

For an observed raster of height :math:`H_o`, padding produces
:math:`H=\operatorname{round}(H_o/(1-\alpha/180))`, where :math:`\alpha` is
``shadow_angle`` in degrees. This preserves the input's existing vertical
angular sampling; it does not force a 2:1 pixel aspect ratio.

The black value is only a fill value. PanorAi creates the south-polar support
mask from the declared angle and state, never by looking for black pixels.
Consequently, genuinely black pixels in the observed region remain valid data.

Already-padded input is declared at construction time:

.. code-block:: python

    padded = EquirectangularImage(
        padded_rgb,
        shadow_angle=30.0,
        shadow_padded=True,
    )
    padded.preprocess()  # safe: the shadow rows are not added again

For cropped input, materialize the rows before projection:

.. code-block:: python

    cropped = EquirectangularImage(
        observed_rgb,
        shadow_angle=30.0,
        shadow_padded=False,
    )
    cropped.preprocess()
    assert cropped.shadow_padded

Array-level preprocessing
-------------------------

The lower-level NumPy preprocessor exposes the same switch:

.. code-block:: python

    from panorai.preprocessing.config import PreprocessorConfig
    from panorai.preprocessing.preprocessor import Preprocessor

    cfg = PreprocessorConfig(
        shadow_angle=10.0,
        shadow_padded=False,
        delta_lat=5.0,
        delta_lon=15.0,
        resize_factor=0.5,
    )

    processed = Preprocessor.preprocess_eq(
        observed_rgb,
        shadow_angle=cfg.shadow_angle,
        shadow_padded=cfg.shadow_padded,
        delta_lat=cfg.delta_lat,
        delta_lon=cfg.delta_lon,
        resize_factor=cfg.resize_factor,
        config=cfg,
    )

The array-level function is stateless, so the caller must pass the correct
``shadow_padded`` value on every call. Prefer the container when a panorama is
processed more than once: it retains the state and prevents double padding.

``panorai.preprocessing.config.PreprocessorConfig`` exposes the remaining
preprocessing options.
