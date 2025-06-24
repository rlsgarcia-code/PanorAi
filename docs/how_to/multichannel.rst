.. _howto-multichannel:

Using MultiChannelHandler
=========================

When your data comes as several aligned arrays (e.g. an RGB image
and its mask), :class:`panorai.data.multi_handler.MultiChannelHandler`
lets you project all channels at once.  It stacks them, applies a
projection, then unpacks the result.

Quick pipeline
--------------

.. code-block:: python

    from panorai.data.multi_handler import MultiChannelHandler
    from panorai.projections.gnomonic_projection import GnomonicProjection

    # Stack channels together and project with a gnomonic projector
    handler = MultiChannelHandler({"rgb": rgb, "mask": mask})
    projector = GnomonicProjection(fov_deg=90)
    handler.apply_projection(projector.project)

    # Work on handler.data here

    # Unstack back to individual arrays
    rgb_arr, mask_arr = handler.unstack()

Example::

   from panorai.data.multi_handler import MultiChannelHandler
   from panorai.projections.gnomonic_projection import GnomonicProjection

   data = {
       "rgb": rgb_array,  # shape (H, W, 3)
       "mask": mask_array  # shape (H, W, 1)
   }

   handler = MultiChannelHandler(data)
   projector = GnomonicProjection(fov_deg=90)
   handler.apply_projection(projector.project)

After processing you can split the channels back with
:meth:`MultiChannelHandler.unstack`.
