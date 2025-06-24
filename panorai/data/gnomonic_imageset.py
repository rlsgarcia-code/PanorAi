"""
gnomonic_imageset.py
====================

Implements the GnomonicFaceSet class, which represents a collection of 
gnomonic faces (GnomonicFace objects) and allows easy batch operations 
and blending back into an equirectangular image.
"""

from typing import List, Callable, Iterator, Any, Tuple, Union
import numpy as np


class GnomonicFaceSet(Iterator):
    """
    Represents a collection of GnomonicFace objects.

    - Supports multi-channel data among the faces.
    - Can attach blending methods to reconstruct an equirectangular image.
    - Allows easy iteration, addition, and transformation of faces.
    """

    def __init__(
        self,
        faces: Union[List["GnomonicFace"], None] = None,
        channel_name: str = "default"
    ):
        """Initialize a :class:`GnomonicFaceSet`.

        Args:
            faces: Optional list of :class:`GnomonicFace` objects.
            channel_name: Label used to identify the data channel.

        Examples:
            >>> fs = GnomonicFaceSet([])
        """
        from .gnomonic_image import GnomonicFace
        self._faces: List[GnomonicFace] = faces if faces else []
        self.channel_name = channel_name
        self._index = 0

        # Attach a default blender
        self.blender = None
        self.attach_blender("average")

    def __iter__(self):
        """Resets iteration and returns self."""
        self._index = 0
        return self

    def __next__(self) -> "GnomonicFace":
        """Iterate over the faces in the set."""
        if self._index >= len(self._faces):
            raise StopIteration
        face = self._faces[self._index]
        self._index += 1
        return face

    def __len__(self):
        """Returns number of faces in the set."""
        return len(self._faces)

    def __getitem__(self, idx: int) -> "GnomonicFace":
        """Enable indexed access to faces."""
        return self._faces[idx]

    def __repr__(self):
        return f"GnomonicFaceSet(channel={self.channel_name}, faces={len(self._faces)})"

    def add_face(self, face: "GnomonicFace"):
        """
        Add a gnomonic face to this set.

        Args:
            face: The :class:`GnomonicFace` to add.

        Examples:
            >>> fs.add_face(face)
        """
        self._faces.append(face.clone())

    def get_faces(self) -> List["GnomonicFace"]:
        """
        Returns the list of all gnomonic faces.

        Returns:
            List[GnomonicFace]

        Examples:
            >>> faces = fs.get_faces()
        """
        return self._faces

    def apply_to_all(self, func: Callable[["GnomonicFace"], None]):
        """
        Applies a given function to each face in the set.

        Args:
            func: Function that accepts a :class:`GnomonicFace`.

        Examples:
            >>> fs.apply_to_all(lambda f: f.show())
        """
        for face in self._faces:
            func(face)

    def attach_blender(self, name: str, **kwargs):
        """
        Dynamically attach a named blender for reconstructing an equirectangular image.

        Args:
            name: Blender name such as ``"average"``.
            **kwargs: Additional blender configuration.

        Examples:
            >>> fs.attach_blender("average")
        """
        try:
            from panorai.factory.panorai_factory import PanoraiFactory
        except ModuleNotFoundError:
            # Optional dependency missing during lightweight unit tests.
            self.blender = None
            return
        self.blender = PanoraiFactory.get_blender(name, **kwargs)

    def to_equirectangular(
        self,
        eq_shape: Tuple[int, int],
        preserve_dtype: bool = True,
        blend_method: Union[str, None] = None
    ) -> "EquirectangularImage":
        """
        Convert the GnomonicFaceSet back into a single EquirectangularImage.

        - If multiple faces, uses a blender to merge them.
        - If only one face, returns the single converted face.

        Args:
            eq_shape: Target panorama shape ``(height, width)``.
            preserve_dtype: Keep the original dtype if ``True``.
            blend_method: If provided, attach or switch to this blender.

        Returns:
            EquirectangularImage

        Examples:
            >>> pano = fs.to_equirectangular((512, 1024))
        """
        
        if not self._faces:
            raise ValueError("No gnomonic faces available for back-projection.")
        
        if blend_method:
            self.attach_blender(blend_method)

        # Convert each face to equirectangular
        eq_faces = [face.to_equirectangular(eq_shape) for face in self._faces]

        if len(eq_faces) == 1:
            return eq_faces[0]

        # Blend multiple equirectangular images
        return self.blend_channels(eq_faces, preserve_dtype, self.blender)

    def blend_channels(
        self,
        projected_faces: List["EquirectangularImage"],
        preserve_dtype: bool,
        blender: Callable
    ) -> "EquirectangularImage":
        """
        Blend multiple equirectangular images channel-wise.

        Args:
            projected_faces: The images to blend.
            preserve_dtype: Whether to cast back to the original dtype.
            blender: Blender object with a ``blend`` method.

        Returns:
            EquirectangularImage: The blended panorama.

        Examples:
            >>> blended = fs.blend_channels(faces, True, blender)
        """
        from .equirectangular_image import EquirectangularImage
        first_face = projected_faces[0]
        # Single-channel vs multi-channel
        if first_face.is_multi_channel():
            # Multi-channel blending
            blended_dict = {}
            all_channels = first_face.get_channels()
            for ch in all_channels:
                # Gather arrays for the same channel from each face
                channel_arrays = [pf.data[ch] for pf in projected_faces]
                # Blend them
                blended = blender.blend(channel_arrays, channel_arrays)
                # Preserve dtype if needed
                if preserve_dtype:
                    blended = blended.astype(channel_arrays[0].dtype)
                blended_dict[ch] = blended
            img = EquirectangularImage(blended_dict)
            img.multi_channel_handler.squeeze_singleton_channels()
            return img
        else:
            # Single-channel data
            inputs = [face.data for face in projected_faces]
            blended_data = blender.blend(inputs, inputs)
            if preserve_dtype:
                blended_data = blended_data.astype(inputs[0].dtype)
            img = EquirectangularImage(blended_data)
            img.multi_channel_handler.squeeze_singleton_channels()
            return img

    def to_pcd(
        self,
        model=None,
        depth: np.ndarray = None,
        eq_shape: Tuple[int, int] = (512, 1024),
        grad_threshold: float = 0.1,
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        blender_name: str = 'simple'
    ):
        """
        Convert this entire GnomonicFaceSet into a single, merged PCD object.

        Args:
            model: Model used to infer depth.
            depth: Optional depth array used instead of ``model``.
            eq_shape: Resolution for back-projection prior to PCD.
            grad_threshold: Gradient threshold in PCD creation.
            min_radius: Minimum radius in the PCD.
            max_radius: Maximum radius in the PCD.
            blender_name: Which blender to use before creating the PCD.

        Returns:
            A PCD object from PCDHandler (adjust for real code).

        Examples:
            >>> pcd = fs.to_pcd(model=my_model)
        """
        if (not model) & (not isinstance(depth, np.ndarray)):
            raise ValueError('You need to pass either a monocular depth estimation model as "model" or a numpy array as depth.')
        else:
            if len(eq_shape) < 2:
                raise ValueError("eq_shape must have at least two dimensions")
            eq_shape = eq_shape[:2]

            from ..pcd.handler import PCDHandler
            return PCDHandler.gnomonic_faceset_to_pcd(
                model=model,
                depth=depth,
                faceset=self,
                eq_shape=eq_shape,
                grad_threshold=grad_threshold,
                min_radius=min_radius,
                max_radius=max_radius,
                blender_name=blender_name
            )

    def clone(self) -> "GnomonicFaceSet":
        """
        Create a deep copy of this GnomonicFaceSet, including attached blender.

        Returns:
            GnomonicFaceSet

        Examples:
            >>> fs_copy = fs.clone()
        """
        new_set = GnomonicFaceSet(
            faces=[face.clone() for face in self._faces],
            channel_name=self.channel_name
        )
        new_set.blender = self.blender
        return new_set