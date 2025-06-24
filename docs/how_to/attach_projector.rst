.. _howto-attach-projector:

Attaching a Projector
=====================

Projectors perform the geometric mapping between an
:class:`EquirectangularImage` and a :class:`GnomonicFace`.
Each container attaches a gnomonic projector by default but you can
override it or supply parameters.

Basic usage
-----------

.. code-block:: python

    face = eq.to_gnomonic(lat=0, lon=0, fov=90)
    face.attach_projection("gnomonic", lat=30, lon=60, fov=120)
    back = face.to_equirectangular(eq_shape=(512, 1024))

You may also attach a projector to the panorama first:

.. code-block:: python

      eq.attach_projection("gnomonic", fov=75)
      face = eq.to_gnomonic(lat=10, lon=10)

Tuning parameters
-----------------

``gnomonic``
    Adjust grid resolution or interpolation. Example::

        eq.attach_projection(
            "gnomonic", lat=0, lon=0, fov=90,
            x_points=512, y_points=512,
            interpolation="INTER_LINEAR"
        )

Other projections can be registered through :class:`~panorai.factory.PanoraiFactory`.

Using a configuration object
----------------------------

You can also supply a :class:`~panorai.projections.gnomonic.config.GnomonicConfig`
for more explicit control::

    from panorai.projections.gnomonic.config import GnomonicConfig

    cfg = GnomonicConfig(fov_deg=110, x_points=256, y_points=256)
    face.attach_projection("gnomonic", config=cfg)

``config`` takes precedence over keyword arguments.  See
:mod:`panorai.projections.gnomonic.config` for the full list of
parameters.

.. seealso::

   :mod:`panorai.projections.gnomonic_projection` -- constructor parameters
   for ``GnomonicProjection``.

