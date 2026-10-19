Component catalog
=================

Use this catalog when you know the role a component must play, but not yet
which PanorAi namespace owns it.  It complements the API reference: the
catalog answers *what can I compose?*, while the module pages document exact
signatures and members.

PanorAi has four API levels, but more than four component roles.  An
Application is a configured end-to-end entry point.  A Stage transforms one
typed value inside a Pipeline.  An operation transforms a spherical raster or
lattice.  An estimator turns observations into a geometric inference.  An
evaluator compares results under a frozen research protocol.  Typed values
carry the grid, support, validity, units and provenance between them. Geometry
owns the coordinate transformations used by every level.

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

   .. grid-item-card:: Research evaluation
      :link: evaluation
      :link-type: doc

      Measure spherical results with frozen, reproducible protocols.

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
   * - compare results for research or promotion evidence
     - :doc:`Research evaluator <evaluation>`
     - yes
     - algorithm output plus independent reference evidence
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

The same manifest assigns every public surface to one or more roles and routes
each role to a catalog, signature reference, concept and progressive guide.
The registration tables at the start of the role pages are generated from
that contract. CI rejects an unclassified public module, a surface without a
role, a missing documentation route, or an incomplete Stable Application
example/consumer contract.

Installed catalog
-----------------

Consumers can inspect the same contract without cloning the repository or
accessing the network::

   from panorai.catalog import public_catalog

   catalog = public_catalog()
   slic = catalog.surface("spherical-applications-slic")
   docs = catalog.documentation_for(
       "spherical-applications-slic",
       component="SphericalSLIC",
   )

   print(slic.tier, slic.public_symbols)
   print(docs.primary.url)
   print([link.url for link in docs.learn])
   print([link.url for link in docs.help])

``public_catalog()`` reads byte-identical JSON resources from the installed
wheel and returns immutable records.  ``learn`` contains role guides and
concepts, ``help`` contains role references and catalogs, and ``context``
links the site home, global component catalog, and complete surface inventory.

The returned ``DocumentationSite`` deliberately reports
``scope="current-main"`` and ``versioned=False``.  The GitHub Pages URLs point
to the current documentation for ``main``; they are not immutable documentation
for the installed package version.  The catalog data and stability metadata,
by contrast, always come from that installed distribution.

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
   evaluation
   subsystems
   public_surfaces
