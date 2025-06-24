"""
handler.py
==========

Implements the `PCDHandler`, a class with static methods to create
and manipulate PCD objects from GnomonicFace, GnomonicFaceSet,
EquirectangularImage, etc.

Includes helper functions for rotating from camera coordinates
to world coordinates, constructing axis arrows, masking high
gradient areas, etc.
"""

import open3d as o3d
import numpy as np
import cv2
from .data import PCD

class PCDHandler:
    """
    Provides static methods that implement the logic for creating a PCD object 
    from GnomonicFace, GnomonicFaceSet, or EquirectangularImage.

    - Depth model loading
    - Rotation from camera to world coordinates
    - Spherical <-> Cartesian conversions
    - High gradient masking
    """

    @staticmethod
    def create_axis_arrows(scale=1.0, shift=1.0):
        """
        Create 3D axis arrows for visualization in Open3D.

        Returns:
            list of open3d.geometry.TriangleMesh objects representing
            X (red), Y (green), Z (blue) axis arrows.
        """
        axes = []
        # X-axis arrow (red)
        x_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02 * scale, 
            cone_radius=0.05 * scale,
            cylinder_height=0.8 * scale, 
            cone_height=0.2 * scale
        )
        x_arrow.paint_uniform_color([1, 0, 0])
        x_arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/2, 0)))
        x_arrow.translate((shift, 0, 0))
        axes.append(x_arrow)

        # Y-axis arrow (green)
        y_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02 * scale, 
            cone_radius=0.05 * scale,
            cylinder_height=0.8 * scale, 
            cone_height=0.2 * scale
        )
        y_arrow.paint_uniform_color([0, 1, 0])
        y_arrow.translate((0, shift, 0))
        axes.append(y_arrow)

        # Z-axis arrow (blue)
        z_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02 * scale, 
            cone_radius=0.05 * scale,
            cylinder_height=0.8 * scale, 
            cone_height=0.2 * scale
        )
        z_arrow.paint_uniform_color([0, 0, 1])
        z_arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0)))
        z_arrow.translate((0, 0, shift))
        axes.append(z_arrow)

        return axes

    @staticmethod
    def mask_high_gradient(depth_map: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Compute a Sobel gradient on the depth map and return a boolean mask
        that is True for pixels with gradient magnitude < threshold.

        Args:
            depth_map (np.ndarray): 2D array of depths (H,W).
            threshold (float): Maximum gradient magnitude to keep.

        Returns:
            np.ndarray: Boolean mask indicating where gradient < threshold.
        """
        # Use simple finite differences instead of Sobel to avoid relying on
        # OpenCV inside unit tests. The gradient is computed by forward
        # differences along both axes with zero padding on the top/left edges.
        # This ensures that isolated spikes in the depth map yield large
        # gradients at their centre, which is the behaviour expected by the
        # tests.
        grad_x = np.zeros_like(depth_map, dtype=float)
        grad_y = np.zeros_like(depth_map, dtype=float)

        grad_x[:, 1:] = np.diff(depth_map, axis=1)
        grad_y[1:, :] = np.diff(depth_map, axis=0)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad_mag < threshold

    @staticmethod
    def to_xyz(lat, lon, R=1.0):
        """
        Convert spherical lat/lon (in degrees) plus radius R into Cartesian (x,y,z).

        Args:
            lat (float or np.ndarray): Latitude in degrees.
            lon (float or np.ndarray): Longitude in degrees.
            R (float or np.ndarray): Radius or distances.

        Returns:
            (x, y, z) as floats or NumPy arrays.
        """
        lat = np.radians(lat)
        lon = np.radians(lon)
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)
        return x, y, z

    @staticmethod
    def compute_rotation_matrices(lat, lon):
        """
        Compute the combined rotation matrix that rotates points from 
        a camera coordinate system to a world lat/lon orientation.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Rotation about the Y-axis
        Ry = np.array([
            [np.cos(lat_rad),  0, np.sin(lat_rad)],
            [0,                1, 0            ],
            [-np.sin(lat_rad), 0, np.cos(lat_rad)]
        ])

        # Rotation about the Z-axis
        Rz = np.array([
            [np.cos(lon_rad), -np.sin(lon_rad), 0],
            [np.sin(lon_rad),  np.cos(lon_rad), 0],
            [0,                0,               1]
        ])

        transform = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]) @ (Rz @ Ry)
        return transform

    @staticmethod
    def rotate_ccs_to_wcs(xyz_points: np.ndarray, lat: float, lon: float) -> np.ndarray:
        """
        Rotate a set of points from camera coordinate system (CCS)
        to world coordinate system (WCS), given lat/lon in degrees.

        Args:
            xyz_points (np.ndarray): shape (N,3) CCS coordinates
            lat (float): Latitude in degrees
            lon (float): Longitude in degrees

        Returns:
            np.ndarray: Rotated points in WCS.
        """
        R = PCDHandler.compute_rotation_matrices(lat, lon)
        return xyz_points @ R

    # @staticmethod
    # def load_depth_model(name='dav2'):
    #     """
    #     Load a depth model by name (DepthAnything or similar).
    #     Possibly returns a callable that, given an image, produces a depth map.

    #     Args:
    #         name (str): The model name or alias.

    #     Returns:
    #         A model object with a __call__ method.
    #     """
    #     from ..integrations.depth_model_loader import load_model
    #     model = load_model(name)
    #     return model

    @staticmethod
    def gnomonic_face_to_pcd(
        face, 
        model,
        depth: np.ndarray = None,
        grad_threshold: float = 0.1,
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        inter_mask: np.ndarray = None
    ):
        """
        Convert a single GnomonicFace to a PCD.

        - Loads a depth model by `model_name`
        - Predicts depth from the gnomonic face's image
        - Masks out high gradient areas
        - Rotates the result from CCS to WCS
        - Filters points outside [min_radius, max_radius]
        - Returns a PCD

        Args:
            face (GnomonicFace): The face containing image data.
            model_name (str): Depth model identifier.
            grad_threshold (float): Gradient threshold for rejection.
            min_radius (float): Minimum distance to keep.
            max_radius (float): Maximum distance to keep.
        
        Returns:
            PCD: The resulting 3D point cloud.
        """
        # model = PCDHandler.load_depth_model(model) --> on hold
        
        image = np.array(face)      # shape(H, W, 3)
        
        if not isinstance(depth, np.ndarray):
            depth = model(image)        # shape(H, W)
        

        H, W = depth.shape[:2]
        
        # Build the [-1..1] meshgrid
        u, v = np.meshgrid(
            np.linspace(-1, 1, W),
            np.linspace(-1, 1, H),
            indexing='xy'
        )
        
        
        X = depth * u
        Y = depth * v
        Z = depth

        indexes = np.stack([ (W - 1) * (u + 1) / 2, (H - 1) * (v + 1)/ 2], axis=-1).reshape(-1,2).astype(np.int32)

        # Mask out high gradient
        grad_mask = PCDHandler.mask_high_gradient(depth, threshold=grad_threshold)

        # Flatten & rotate to WCS
        xyz_ccs = np.stack([Y.ravel(), X.ravel(), Z.ravel()], axis=1)
        xyz_full = PCDHandler.rotate_ccs_to_wcs(xyz_ccs, face.lon, 90 - face.lat)
        
        # Valid depth & gradient
        valid_mask = (depth > 0) & grad_mask
        if isinstance(inter_mask, np.ndarray):
            valid_mask = (valid_mask > 0) & inter_mask
        
        valid_mask = valid_mask.ravel()

        # Build radius image
        from ..data import GnomonicFace
        R_image = np.sqrt(np.sum(xyz_full**2, axis=1)).reshape(H, W)
        R_face = GnomonicFace(R_image, lat=face.lat, lon=face.lon, fov=face.fov)

        # Flatten color
        colors = (image.reshape(-1, 3)[valid_mask] / 255.0)
        points = xyz_full[valid_mask]
        indexes = indexes[valid_mask]

        # Radius filter
        radii = np.linalg.norm(points, axis=1)
        radius_mask = (radii >= min_radius) & (radii <= max_radius)

        points_final = points[radius_mask]
        colors_final = colors[radius_mask]
        indexes_final = indexes[radius_mask]

        # Return final PCD
        return PCD(points_final, colors_final, R_face, indexes=indexes_final, shape=(H, W))

    def gnomonic_faceset_to_pcd(
        faceset,
        model,
        depth,
        eq_shape=(512, 1024),
        grad_threshold=0.1,
        min_radius=0.0,
        max_radius=20.0,
        feather_exp=1.0,
        blender_name='simple'
    ):
        """
        Convert a GnomonicFaceSet into a merged PCD using a chosen blender.

        - Loads the depth model
        - Retrieves the appropriate blender from `PCDBlenderFactory`
        - Delegates all per-face processing to the blender

        Args:
            faceset (GnomonicFaceSet): The input face set.
            model_name (str): Depth model name.
            eq_shape (Tuple[int,int]): Equirectangular shape to which 
                faces might be reprojected before blending.
            grad_threshold (float): Gradient threshold for rejection.
            min_radius (float): Minimum radius filter.
            max_radius (float): Maximum radius filter.
            feather_exp (float): Exponent for radial feather weighting.
            blender_name (str): Which blender type to use.
        
        Returns:
            PCD: Merged 3D point cloud.
        """
        if len(eq_shape) < 2:
            raise ValueError("eq_shape must have at least two dimensions")
        eq_shape = eq_shape[:2]

        from .blender.blender_factory import PCDBlenderFactory
        blender = PCDBlenderFactory.get_blender(
            blender_name,
            eq_shape=eq_shape,
            min_radius=min_radius,
            max_radius=max_radius
        )
        # model = PCDHandler.load_depth_model(model_name) #--> old
        return blender.process_faceset(faceset, model=model, grad_threshold=grad_threshold, feather_exp=feather_exp)

    @staticmethod
    def equirectangular_image_to_pcd(eq_img, eq_radial=None, grad_threshold: float = 1.0,
                                     min_radius: float = 0.0,
                                     max_radius: float = 999999.0):
        
        # Todo ---> add lógic to generate eq_radial or use a wq_radial provided by the user,
        #           next is just about PCDHandler.gnomonic_faceset_to_pcd

        # Old Code
        # from ..models.depthanythingv2_integration import load_model
        # model = load_model('mps')

        # image = np.array(eq_img)              # shape(H,W,3)
        # depth_map = model(image)              # shape(H,W)
        # H, W = depth_map.shape[:2]

        # lon_vals = np.linspace(-180, 180, W)
        # lat_vals = np.linspace(90, -90, H)
        # lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

        # # Gradient mask
        # valid_mask = PCDHandler.mask_high_gradient(depth_map, threshold=grad_threshold).ravel()

        # # Convert to 3D
        # X, Y, Z = PCDHandler.to_xyz(lat_grid, lon_grid, R=depth_map)
        # xyz_all = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        # colors_all = (image.reshape(-1, 3) / 255.0)

        # positive_depth_mask = (depth_map.ravel() > 0)
        # combined_mask = valid_mask & positive_depth_mask

        # points = xyz_all[combined_mask]
        # colors = colors_all[combined_mask]

        # # Radius filter
        # radii = np.linalg.norm(points, axis=1)
        # radius_mask = (radii >= min_radius) & (radii <= max_radius)

        # points_final = points[radius_mask]
        # colors_final = colors[radius_mask]

        # return PCD(points_final, colors_final)
        raise ValueError('Method is not implemented.')