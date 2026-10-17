Public roadmap
==============

Now — PanorAi 3.8
-----------------

* Publish the typed Pipeline, spherical-operation library, generic CPU Torch
  layers, and spherical image-object workflow without changing geometry-v1.
* Keep all Stable and Compatibility 3.x imports working and make every new
  public component discoverable through the role-based catalog.
* Ship SLIC as a separately evidenced Stable algorithm/application while DoG,
  stereo, relative pose, CAM/SAM, graph, reconstruction, and SLAM retain their
  explicit Experimental gates.
* Preserve exact artifact identity across source, wheel, sdist, TestPyPI, and
  PyPI; source-checkout success alone is not release evidence.

Next
----

* Validate graph association, localization, confidence, and query behavior on
  public, spatially independent multi-view corpora.
* Measure archive scale, query latency, and bounded-memory behavior.
* Add new graph posterior implementations only after the baseline contract has
  independent evidence; particle and mixture protocols are not advertised as
  implemented methods.
* Produce a 4.0 compatibility preview and migration tooling before moving any
  import.

Later — PanorAi 4.0
-------------------

* Apply the published import-destination map with one coherent migration.
* Replace implicit global registries with immutable configuration and explicit
  constructors where evidence supports the change.
* Consider optional point-cloud I/O adapters and additional graph storage
  backends only after the portable JSONL/NPZ contract is established.

Promotion is evidence-driven. This roadmap does not promote any Experimental
surface or promise a release date.
