Public roadmap
==============

Current foundation — PanorAi 3.8
--------------------------------

* Preserve ``geometry-v1``, the Stable object workflow, typed Pipeline,
  spherical-operation library, CPU Torch layers, SLIC, morphology and the
  audited component catalog.
* Keep Stable and Compatibility 3.x imports working.  Experimental pose,
  stereo, reconstruction, semantics, graph and SLAM surfaces retain their
  existing evidence gates.
* Preserve exact artifact identity across source, wheel, sdist, TestPyPI and
  PyPI.  Source-checkout success alone is not release evidence.

Next — spherical computer-vision research
-----------------------------------------

Development follows three ordered capability priorities:

1. **Research evaluation**: public, geometry-aware metrics, evaluation
   records and reproducible protocols for spherical algorithms.
2. **Region analysis**: topology and measurements that continue the Stable
   morphology and SLIC foundations into connected regions, distances,
   contours and watershed-style grouping.
3. **Registration and stitching**: construct one typed panorama from
   :math:`N \geq 2` overlapping images, estimating missing image geometry
   when it is observable instead of requiring metadata for every image.

The detailed :doc:`computer-vision research capability roadmap
<computer_vision_research_roadmap>` defines scope, API placement, scientific
gates and explicit non-goals.  It is a statement of intent, not an implemented
API or a release-date promise.

Beyond those priorities
-----------------------

* Extend motion and video support with spherical vector fields, optical flow,
  tracks, motion segmentation and stabilization after the representation and
  evaluation contracts exist.
* Mature existing pose, depth, reconstruction and SLAM paths through their
  prospective scientific gates rather than promoting them by breadth alone.
* Continue graph association, localization, confidence and query research on
  spatially independent multi-view corpora.
* Add photometric, spectral, rendering or learned-model capabilities only when
  they preserve the package focus and have a clear sphere-specific contract.

Later — PanorAi 4.0
-------------------

* Apply the published import-destination map with one coherent migration.
* Replace implicit global registries with immutable configuration and explicit
  constructors where evidence supports the change.
* Consider optional point-cloud I/O adapters and additional graph storage
  backends only after their portable contracts are established.

Promotion remains evidence-driven.  This roadmap does not promote an
Experimental surface, authorize a release, or weaken an existing scientific
gate.
