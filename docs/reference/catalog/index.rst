Component catalog
=================

Use this catalog when you know the role a component must play, but not yet
which PanorAi namespace owns it.  It complements the API reference: the
catalog answers *what can I compose?*, while the module pages document exact
signatures and members.

PanorAi has four API levels, but more than four component roles.  An
Application is a configured end-to-end entry point.  A Stage transforms one
typed value inside a Pipeline.  An operation transforms a spherical raster or
lattice.  An estimator turns observations into a geometric inference.  Typed
values carry the grid, support, validity, units and provenance between them.
Geometry owns the coordinate transformations used by every level.

Choose by role
--------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Applications
      :link: applications
      :link-type: doc

      Complete tasks with the smallest configured surface.

   .. grid-item-card:: Pipeline Stages
      :link: stages
      :link-type: doc

      Compose domain transitions, routing and reusable task steps.

   .. grid-item-card:: Spherical Ops and layers
      :link: operations
      :link-type: doc

      Process an ERP directly with NumPy objects or Torch layers.

   .. grid-item-card:: Estimators and policies
      :link: estimators
      :link-type: doc

      Infer geometry and make quality decisions from observations.

   .. grid-item-card:: Typed data and results
      :link: data_results
      :link-type: doc

      Find the values exchanged between components.

   .. grid-item-card:: Geometry
      :link: geometry
      :link-type: doc

      Work directly with pixels, rays, projectors and projection results.

   .. grid-item-card:: Advanced subsystems
      :link: subsystems
      :link-type: doc

      Navigate features, depth, semantics, reconstruction, graph and SLAM.

Quick chooser
-------------

.. list-table:: Start from the smallest role that owns your decision
   :header-rows: 1
   :widths: 20 36 22 22

   * - You need to…
     - Start with
     - Can run alone?
     - Typical composition
   * - solve a complete supported task
     - :doc:`Application <applications>`
     - yes
     - Application exposes ``as_pipeline()``
   * - add, remove or reorder processing steps
     - :doc:`Stage <stages>`
     - yes, through ``execute()``; normally in a Pipeline
     - ``Pipeline(stage_a, stage_b)``
   * - filter, resize, detect or pool on a sphere
     - :doc:`Spherical operation <operations>`
     - yes
     - direct call, ``EquirectangularImage.filter()``, or ``OnSphere``
   * - infer pose or another model from observations
     - :doc:`Estimator <estimators>`
     - yes
     - estimator Stage at the end of a Pipeline
   * - preserve spatial meaning between steps
     - :doc:`Typed data/result <data_results>`
     - it is data, not execution
     - Stage input/output and retained Pipeline values
   * - control projections or coordinate conversions
     - :doc:`Geometry <geometry>`
     - yes
     - used directly or behind higher-level components

How to read the catalog
-----------------------

Each curated table names the canonical import, stability, accepted input,
returned output, execution domain and common composition.  ``Stable`` means a
3.x compatibility promise.  ``Experimental`` means public and tested but
eligible to evolve; placing an Experimental component in a Stable Pipeline
does not promote it.

The :doc:`complete public-surface inventory <public_surfaces>` is generated
from ``contracts/public-api-surfaces-v2.json``.  It lists every audited public
symbol, including advanced research surfaces and compatibility gateways.  If
a symbol is absent there, it is not part of the audited public API even if it
can be imported from an implementation module.

Composition model
-----------------

The usual direction is::

   Application
       └── Pipeline
             ├── Stage ── Spherical Op / estimator
             ├── Stage ── domain transition
             └── typed value or result
                    └── Geometry contract

This is not a restriction.  Operations, estimators and Geometry functions can
be used independently.  Conversely, a custom Application may wrap a Pipeline
assembled entirely from public Stages.  See :doc:`the extension contract
<../../development/application_extensions>` before publishing a new wrapper or
Stage.

.. toctree::
   :maxdepth: 1

   applications
   stages
   operations
   estimators
   data_results
   geometry
   subsystems
   public_surfaces
