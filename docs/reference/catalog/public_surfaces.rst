Complete public-surface inventory
=================================

This page is exhaustive rather than introductory.  Its tables are generated
from ``contracts/public-api-surfaces-v2.json`` during every strict Sphinx
build.  They include primary products, advanced Experimental components and
3.x compatibility gateways.  Repeated symbols are intentional when one name
participates in more than one independently versioned contract.

Foundations
-----------

.. panorai-api-symbols::
   :family: foundations

Visual evidence
---------------

.. panorai-api-symbols::
   :family: visual-evidence

Pose and geometry
-----------------

.. panorai-api-symbols::
   :family: pose-geometry

Mapping and depth
-----------------

.. panorai-api-symbols::
   :family: mapping-depth

Semantic evidence
-----------------

.. panorai-api-symbols::
   :family: semantic-evidence

Spatial-semantic graph
----------------------

.. panorai-api-symbols::
   :family: spatial-semantic-graph

Internal implementation files are deliberately omitted from the reader
catalog.  They have no import-path compatibility promise.  See
:doc:`API stability <../stability>` for the complete surface matrix,
visibility and promotion policy.
