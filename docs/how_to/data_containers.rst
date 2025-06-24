.. _howto-data-containers:

Switching Between Containers
============================

The three core containers –
:class:`panorai.data.equirectangular_image.EquirectangularImage`,
:class:`panorai.data.gnomonic_image.GnomonicFace` and
:class:`panorai.data.gnomonic_imageset.GnomonicFaceSet` –
convert to each other using built-in helpers.

Example workflow
----------------

.. code-block:: python

    from panorai.data import DataFactory

    eq = DataFactory.from_file("pano.jpg", data_type="equirectangular")
    face = eq.to_gnomonic(lat=0, lon=0, fov=90)
    faces = eq.to_gnomonic_face_set(fov=60, sampling_method="cube")

    # Process face or faces here

    back_from_face = face.to_equirectangular(eq_shape=(512, 1024))
    back_from_set = faces.to_equirectangular(eq_shape=(512, 1024))

Each container keeps track of the attached projector or blender so the
round trip can be performed seamlessly.
