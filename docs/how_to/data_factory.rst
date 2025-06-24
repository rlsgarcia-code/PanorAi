.. _howto-data-factory:

Using DataFactory
=================

:class:`panorai.data.factory.DataFactory` provides handy classmethods to build
any of the spherical data containers from different sources.

Basic usage
-----------

.. code-block:: python

    from panorai.data import DataFactory

    # Load an image file as an equirectangular panorama
    eq = DataFactory.from_file("pano.jpg", data_type="equirectangular")

    # Convert raw arrays or PIL images as well
    face = DataFactory.from_array(arr, data_type="gnomonic_face")
    face2 = DataFactory.from_pil(pil_img, data_type="gnomonic_face")

List of faces
-------------

.. code-block:: python

    from panorai.data import GnomonicFace

    faces = [GnomonicFace(arr1, 0, 0, 90), GnomonicFace(arr2, 45, 0, 90)]
    face_set = DataFactory.from_list(faces, channel_name="rgb")

`from_list` attaches the default ``average`` blender to the returned
:class:`panorai.data.gnomonic_imageset.GnomonicFaceSet`.
