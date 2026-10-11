Build and query a spatial-semantic graph
========================================

``panorai.graph`` consumes prepared evidence. Applications call pose, depth,
semantic, oracle, or external providers explicitly and then pass typed results
to the builder. The builder never selects a model, runs matching, or downloads
a checkpoint.

The minimal lifecycle is:

1. create ``SphericalViewNode`` and ``SemanticRegionNode`` observations with
   frame, split, access role, and provenance;
2. adapt the exact matches and inliers that supported a pose into
   ``PairFeatureEvidence`` and ``RelativePoseEdge``;
3. call ``associate_regions`` and ``localize_region_pair`` independently;
4. add accepted records to ``SpatialSemanticGraphBuilder``;
5. call ``snapshot()``, archive with ``save_graph_archive``, and query through
   ``SpatialSemanticGraphQuery``.

No scale is invented. Without finite depth or metric scale, localization is
reported as unbounded angular or scale-free support. Metric sparse-voxel
support is only produced when metric evidence exists. One view never confirms
an entity under the conservative-v1 policy.

Archives contain ``graph.jsonl`` plus SHA-256-addressed ``arrays/*.npz``
sidecars. Loading verifies every sidecar before reconstructing the immutable
snapshot. Text queries require an encoder that explicitly declares compatible
text-image alignment.

See :doc:`../reference/graph` for the public classes and
:doc:`../reference/stability` for the Experimental boundary.
