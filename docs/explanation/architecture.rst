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

* **``EquirectangularImage``** – the global view (`to_gnomonic`, `to_gnomonic_face_set`)
* **``GnomonicFace``** – one rectilinear patch (`to_equirectangular`)
* **``GnomonicFaceSet``** – collection of faces that can be blended back

Attachables
-----------

* **Projector** – geometric mapping (default : Gnomonic)
* **Sampler**   – where to place faces (cube, fibonacci, …)
* **Blender**   – how to merge overlaps (average, gaussian, …)

Data-flow diagram
-----------------

.. mermaid::

   graph LR
     EQ([EquirectangularImage]) -->|to_gnomonic| GF[(GnomonicFace)]
     EQ -->|to_gnomonic_face_set| GFS[(GnomonicFaceSet)]
     GF -->|to_equirectangular| EQ
     GFS -->|blend ⟶| EQ


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
