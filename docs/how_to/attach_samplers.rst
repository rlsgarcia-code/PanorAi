.. _howto-attach-samplers:

Attaching Samplers
==================

Samplers choose the tangent points from which multiple rectilinear
faces are extracted.  If you call
:meth:`EquirectangularImage.to_gnomonic_face_set` without specifying a
sampler, ``cube`` is attached by default.

Basic usage
-----------

.. code-block:: python

    faces = eq.to_gnomonic_face_set(fov=90)       # uses cube sampler
    eq.attach_sampler("fibonacci", n_points=20)
    faces = eq.to_gnomonic_face_set(fov=60)

You can pass extra parameters depending on the sampler.  For example
``fibonacci`` accepts ``n_points`` to control sampling density.

Tuning parameters
-----------------

Other samplers expose additional options:

``icosahedron``
    Subdivide the base icosahedron with ``subdivisions``::

        eq.attach_sampler("icosahedron", subdivisions=2)

``spiral``
    Specify how many points along the spiral with ``n_points``::

        eq.attach_sampler("spiral", n_points=50)

``blue_noise``
    Control the number of randomly distributed samples::

        eq.attach_sampler("blue_noise", n_points=100)
