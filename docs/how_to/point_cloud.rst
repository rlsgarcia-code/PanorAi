.. _howto-point-cloud:

Exporting to Point Clouds
=========================

Both :class:`panorai.data.gnomonic_image.GnomonicFace` and
:class:`panorai.data.gnomonic_imageset.GnomonicFaceSet` provide a
:meth:`to_pcd` method that converts the image data into an Open3D
point cloud.  This is handy for visualising depth maps or for further
3D processing.

Quick example::

   pcd = face.to_pcd()
   pcd_handler = face_set.to_pcd()
   pcd_handler.create_axis_arrows()

The resulting :class:`panorai.pcd.handler.PCDHandler` offers helper
functions such as :func:`create_axis_arrows` for visualisation and
gradient masking utilities.
