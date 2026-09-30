.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/how_to/image_processing.rst  ── Samplers · Blenders · Preprocessing
.. ───────────────────────────────────────────────────────────────

Image-processing recipes
========================

High-level workflow
-------------------

.. code-block:: python

    from panorai.data import DataFactory

    # Load a panorama and sample several faces
    eq = DataFactory.from_file("pano.jpg", data_type="equirectangular")
    faces = eq.to_gnomonic_face_set(fov=90, sampling_method="cube")

    # Process faces here (e.g. neural network)

    # Blend the results back
    faces.attach_blender("gaussian")
    pano = faces.to_equirectangular(eq_shape=(512, 1024))
    pano.show()

Convert many faces
------------------

.. code-block:: python

    faces = eq.to_gnomonic_face_set(fov=75, sampling_method="fibonacci", n_points=20)

Attach a custom blender
-----------------------

.. code-block:: python

    faces.attach_blender("closest")  # pixel from nearest centre
    pano = faces.to_equirectangular(eq_shape=(512, 1024))

Multi-channel trick
-------------------

.. code-block:: python

    from panorai.data.multi_handler import MultiChannelHandler

    handler = MultiChannelHandler({"rgb": rgb, "mask": mask})
    handler.apply_projection(projector.project)

Preprocess without containers
-----------------------------

.. code-block:: python

    from panorai.preprocessing.preprocessor import Preprocessor

    arr = Preprocessor.preprocess_eq(eq.data, delta_lat=5, resize_factor=0.5)

Preprocess with containers
--------------------------

The :class:`~panorai.data.equirectangular_image.EquirectangularImage` class
provides a :meth:`preprocess` helper so you can update the data before
projecting it::

    eq = DataFactory.from_file("pano.jpg", data_type="equirectangular")
    eq.preprocess(delta_lat=2, resize_factor=0.5)
    face = eq.to_gnomonic(lat=0, lon=0, fov=90)

Samplers & blenders at a glance
-------------------------------

.. include:: ../reference/samplers_blenders.rst
   :start-after: .. cut:start
   :end-before: .. cut:end
