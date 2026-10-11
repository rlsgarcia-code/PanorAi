Public roadmap
==============

Now — PanorAi 3.7
-----------------

* Publish the product-family taxonomy and canonical API inventory.
* Keep all Stable and Compatibility 3.x imports working.
* Deliver ``panorai.graph`` as an import-light Experimental contract that
  consumes prepared evidence, preserves uncertainty, and supports replay.
* Separate conformance, regression, performance, and scientific evidence.

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
