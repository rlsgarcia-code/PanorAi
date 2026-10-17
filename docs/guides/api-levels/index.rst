The four API levels
===================

Use the smallest surface that owns the decision you need to make. These
tutorials are ordered from least setup to most control and reuse one geometric
contract throughout.

.. image:: ../../_static/readme/api-levels.svg
   :alt: Applications, Pipeline, Spherical Ops and Geometry arranged from less
         setup to more control.
   :width: 100%

.. toctree::
   :maxdepth: 1

   applications
   pipeline
   spherical_ops
   geometry

.. list-table:: What changes as you move right
   :header-rows: 1
   :widths: 18 28 27 27

   * - Level
     - You provide
     - PanorAi provides
     - Typical result
   * - Applications
     - method parameters and inputs
     - the complete method composition
     - typed task result
   * - Pipeline
     - stages, branches and routing
     - execution, provenance and typed transitions
     - raster or result bundle
   * - Spherical Ops
     - an operation or network topology
     - sphere-aware sampling and grid transforms
     - transformed spherical field
   * - Geometry
     - coordinates, specs and interpolation
     - canonical ray and projection mathematics
     - projected data plus support/validity

Applications may expose Pipelines; Pipelines may contain Spherical Ops; all
three depend on Geometry. Moving between levels does not change axis, pixel,
unit or validity conventions.
