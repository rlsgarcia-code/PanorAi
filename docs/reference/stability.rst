API stability in 3.x
====================

PanorAi classifies public surfaces by product family, implementation domain,
stability tier, visibility, and canonical import. The machine-readable source
of truth is ``contracts/public-api-surfaces-v2.json``
(``panorai-public-api-surfaces/v2``), validated by the adjacent JSON Schema.
This page is rendered from that inventory during the strict Sphinx build.

The tier describes the compatibility promise, not scientific usefulness.
Participation in the 3.7 consolidation does not promote an API.

.. list-table:: Stability policy
   :header-rows: 1
   :widths: 18 82

   * - Tier
     - Promise
   * - Stable
     - Preserved throughout 3.x. Removal or incompatible relocation is reserved
       for 4.0 and requires migration guidance.
   * - Compatibility
     - Existing 3.x imports remain available, but the canonical 4.0 destination
       may differ.
   * - Experimental
     - Public and tested, but eligible to evolve between minor releases. Every
       surface has explicit promotion gates.
   * - Frozen/internal
     - No import-path compatibility promise.

Public surface matrix
---------------------

.. panorai-api-inventory::

Product-family view
-------------------

The primary navigation groups the granular contracts into product families.
The IDs remain granular so a component can be tested or promoted without
implicitly promoting its entire family.

.. panorai-api-families::

Compatibility rules for 3.7
---------------------------

* No Stable or Compatibility import receives a new runtime warning.
* ``panorai.features.__all__`` is preserved.
* ``panorai.object_localization`` remains importable and preserves its v1
  result models, IDs, scores, and states. Its pipeline delegates association
  and pair localization to ``panorai.graph``.
* ``panorai.graph`` is Experimental and import-light. Importing it does not
  load Torch, Open3D, PyCOLMAP, checkpoints, or a database backend.
* Native acceleration is not a separate public numerical contract.

The 4.0 destination map is documented in :doc:`../development/migration-4.0`.
