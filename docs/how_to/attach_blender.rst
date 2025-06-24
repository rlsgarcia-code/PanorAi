.. _howto-attach-blender:

Attaching a Blender
===================

A :class:`panorai.data.gnomonic_imageset.GnomonicFaceSet` blends its
faces back into an equirectangular panorama using a blender object.
A new face set starts with the ``average`` blender.

Basic usage
-----------

.. code-block:: python

    faces = eq.to_gnomonic_face_set(fov=90)
    faces.attach_blender("gaussian", sig=1.2)
    pano = faces.to_equirectangular(eq_shape=(512, 1024))

Other blenders may accept different parameters such as feathering
radius or voxel size.

Tuning parameters
-----------------

``gaussian``
    Set the mean and sigma of the weighting kernel::

        faces.attach_blender("gaussian", mu=0, sig=1.5)

``feathering``
    Control the blend radius in pixels::

        faces.attach_blender("feathering", radius=30)

``closest``
    No parameters – picks the pixel closest to each centre.

See also
--------

The available blenders and their arguments are documented in
:mod:`panorai.blenders`.  You can build them directly with
:class:`~panorai.blenders.registry.BlenderRegistry` or via
:class:`~panorai.factory.PanoraiFactory`.
