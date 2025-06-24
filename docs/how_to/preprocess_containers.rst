.. _howto-preprocess-containers:

Preprocessing Containers
=======================

`EquirectangularImage` exposes a :meth:`preprocess` method so you can
resize or rotate a panorama before sampling faces.

Example
-------

.. code-block:: python

    from panorai.data import DataFactory

    eq = DataFactory.from_file("pano.jpg", data_type="equirectangular")
    eq.preprocess(delta_lat=10, shadow_angle=20, resize_factor=0.5)
    faces = eq.to_gnomonic_face_set(fov=90)

The preprocessing step modifies the container in place. After running it
any call to ``to_gnomonic`` or ``to_gnomonic_face_set`` will use the
updated data.

``shadow_angle`` describes the section of the panorama that a 3D scanner
cannot normally capture.  It is measured from the South Pole (bottom) of
the image upward.  Setting a non‐zero value pads the panorama so that the
subsequent projection can cover the missing region.
