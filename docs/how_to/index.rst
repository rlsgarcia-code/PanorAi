.. _howto-index:

How-to guides
=============

Short recipes that solve one specific problem at a time.
They assume
you are familiar with the basics explained in the tutorials and API
reference.

Available guides
----------------

- :doc:`image_processing` – Convert panoramas into rectilinear faces and
  blend them back using different samplers and blenders.
- :doc:`multichannel` – Stack multiple arrays (RGB + masks, depth, ...)
  so they are projected together.
- :doc:`point_cloud` – Export processed faces to an Open3D point cloud
  for 3D visualisation.
- :doc:`data_factory` – Build data objects from files, arrays or lists.
- :doc:`data_containers` – Convert between panoramas and rectilinear faces.
- :doc:`attach_samplers` – Choose how tangent points are generated.
- :doc:`attach_blender` – Select a blender when merging multiple faces.
- :doc:`attach_projector` – Override the projection parameters used.
- :doc:`preprocess_containers` – Preprocess a panorama before projection.

.. toctree::
   :maxdepth: 1

   image_processing
   multichannel
   point_cloud
   data_factory
   data_containers
   attach_samplers
   attach_blender
   attach_projector
   preprocess_containers
