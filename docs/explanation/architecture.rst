.. ───────────────────────────────────────────────────────────────
.. 🗂  docs/explanation/architecture.rst  ── “What is PanorAi?”
.. ───────────────────────────────────────────────────────────────

Architecture overview
=====================

PanorAi lets you **cut** a full-sphere panorama into rectilinear tiles, 
**process** each tile any way you like (classical CV, CNN, whatever) and 
**stitch** the results back.  
Everything revolves around three container objects and three attachable helpers.

Containers
----------

* :class:`~panorai.data.equirectangular_image.EquirectangularImage` – the
  panorama container providing
  :meth:`~panorai.data.equirectangular_image.EquirectangularImage.to_gnomonic`
  and
  :meth:`~panorai.data.equirectangular_image.EquirectangularImage.to_gnomonic_face_set`.
* :class:`~panorai.data.gnomonic_image.GnomonicFace` – a single rectilinear
  view offering
  :meth:`~panorai.data.gnomonic_image.GnomonicFace.to_equirectangular`.
* :class:`~panorai.data.gnomonic_imageset.GnomonicFaceSet` – a collection of
  faces that can be blended back with
  :meth:`~panorai.data.gnomonic_imageset.GnomonicFaceSet.to_equirectangular`.

See :doc:`../how_to/data_factory` for ways to build these containers and
:doc:`../how_to/data_containers` for conversion examples.

Attachables
-----------

* **Projector** – geometric mapping (default : Gnomonic)
* **Sampler**   – where to place faces (cube, fibonacci, …)
* **Blender**   – how to merge overlaps (average, gaussian, …)

Guides on attaching these components with parameters are available in
:doc:`../how_to/attach_projector`,
:doc:`../how_to/attach_samplers` and
:doc:`../how_to/attach_blender`.


Lists of the built-in samplers and blenders are shown in
:doc:`../reference/samplers_blenders`.  Available projectors are listed in
:doc:`../reference/projectors`.  All configuration options are detailed in
:doc:`../api_objects`.


=======


Data-flow diagram
-----------------

.. mermaid::

   graph LR
     EQ([EquirectangularImage]) -->|to_gnomonic| GF[(GnomonicFace)]
     EQ -->|to_gnomonic_face_set| GFS[(GnomonicFaceSet)]
     GF -->|to_equirectangular| EQ
     GFS -->|blend ⟶| EQ


Detailed data-flow
------------------

.. mermaid::

   graph TD
      A[EquirectangularImage]
      S[[Sampler]]
      P[[Projector]]
      F[GnomonicFaceSet]
      P2[[Projector]]
      B[[Blender]]

      A -->|attach_sampler| S
      A -->|attach_projector| P
      S -->|tangent points| P
      P -->|faces| F
      F -->|process| F
      F -->|project back| P2
      P2 -->|eq patches| B
      B -->|blend patches| A



- :doc:`../reference/samplers_blenders` – overview of built-in samplers and
  blenders.
- :doc:`../reference/projectors` – summary of available projectors.
- :doc:`../api_objects` – complete list of objects and parameters.
Why this matters
----------------

Splitting a panorama into faces lets you reuse
existing 2D algorithms with minimal changes. You can run
filters or neural networks on each face, then project the
results back to obtain an updated panorama. PanorAi provides
registries for samplers, projections and blenders so you can
tailor every step of the pipeline.

Putting it all together
-----------------------

A typical workflow is:

#. Load an :class:`EquirectangularImage` via :class:`DataFactory`.
#. Convert it to a :class:`GnomonicFaceSet` using a sampler such as
   ``cube`` or ``fibonacci``.
#. Process each :class:`GnomonicFace` individually (for instance with
   a neural network).
#. Blend the results back to an equirectangular panorama with a chosen
   blender.

These steps are demonstrated in the :doc:`../tutorials/00_quick_start` tutorial.

Further reading
---------------

- :doc:`../how_to/index` – practical how-to guides.
- :doc:`../reference/index` – complete API reference for all classes and methods.
- :doc:`../reference/samplers_blenders` – overview of built-in samplers and
  blenders.
- :doc:`../reference/projectors` – summary of available projectors.
- :doc:`../api_objects` – complete list of objects and parameters.

