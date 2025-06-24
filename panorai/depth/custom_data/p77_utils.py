import numpy as np
import cv2
from scipy.spatial.distance import cdist
import open3d as o3d


def select_height(points):
    standard_sizes = np.array([2**n for n in range(20)])
    distances= np.abs(standard_sizes - np.sqrt(points.shape[0] // 2))
    return standard_sizes[np.argmin(distances)]
    
def load_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    point_cloud_in_numpy = np.asarray(pcd.points) 
    color_cloud_in_numpy = np.asarray(pcd.colors) 
    return point_cloud_in_numpy, color_cloud_in_numpy
    
def geometric_median(points, tol=1e-5):
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.
    
    Parameters:
        points (numpy.ndarray): N x 3 array of (x, y, z) points.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Estimated center of the points.
    """
    median = np.mean(points, axis=0)  # Initial guess: arithmetic mean
    while True:
        distances = np.linalg.norm(points - median, axis=1)
        nonzero_distances = np.where(distances == 0, np.finfo(float).eps, distances)  # Avoid division by zero
        weights = 1 / nonzero_distances
        new_median = np.average(points, axis=0, weights=weights)
        
        if np.linalg.norm(new_median - median) < tol:
            break
        median = new_median
    
    return median


class P77_Dataset:
    def __init__(self, points, colors, H, W):
        """
        Initialize the projection with 3D points, colors, and image dimensions.
        
        :param points: Nx3 numpy array of 3D coordinates
        :param colors: Nx3 numpy array of RGB colors
        :param H: Height of the output equirectangular image
        :param W: Width of the output equirectangular image
        """
        self.points = points
        self.colors = ( 255 * colors).astype(np.uint8)
        self.H = H
        self.W = W
        self.image = np.zeros((H, W, 3), dtype=np.uint8)

    def normalize_points(self):
        """Normalize points to lie on a unit sphere."""
        median = geometric_median(self.points.copy(), tol=1e-5)
        self._points = self.points - median
        self.points = self._points / np.linalg.norm(self._points, axis=1, keepdims=True)  # Normalize to radius 1
        # print("Estimated Sphere Center:", median)
        

    def project_to_sphere(self):
        """
        Project 3D points onto a unit sphere and convert them to spherical coordinates.
        Returns:
            - latitudes (theta) in range [-pi/2, pi/2]
            - longitudes (phi) in range [-pi, pi]
        """
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        
        # Compute latitude (theta) and longitude (phi)
        theta = np.arcsin(z)  # Latitude: arcsin(y) -> [-pi/2, pi/2]
        phi = np.arctan2(x, y)  # Longitude: atan2(x, z) -> [-pi, pi]
        
        return theta, phi

    def create_equirectangular_grid(self):
        """
        Create a grid of latitudes and longitudes for the equirectangular image.
        """
        lat_grid = np.linspace(np.pi / 2, -np.pi / 2, self.H)  # [-pi/2, pi/2]
        lon_grid = np.linspace(-np.pi, np.pi, self.W)  # [-pi, pi]
        lon_map, lat_map = np.meshgrid(lon_grid, lat_grid)  # Shape (H, W)

        return lat_map, lon_map

    def map_to_equirectangular(self):
        """
        Map the projected spherical coordinates to the equirectangular image using cv2.remap.
        """
        self.normalize_points()
        theta, phi = self.project_to_sphere()

        # Convert spherical coordinates to 2D image space
        u = ((phi + np.pi) / (2 * np.pi)) * (self.W - 1)  # Scale longitude to [0, W-1]
        v = ((np.pi / 2 - theta) / np.pi) * (self.H - 1)  # Scale latitude to [0, H-1]

        # Convert to float32 for remap
        u = np.clip(u, 0, self.W - 1).astype(np.float32)
        v = np.clip(v, 0, self.H - 1).astype(np.float32)

        # Create an empty color image and coordinate maps
        map_x = np.zeros((self.H, self.W), dtype=np.float32)
        map_y = np.zeros((self.H, self.W), dtype=np.float32)
        color_map = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Assign mapped coordinates for interpolation
        map_x[v.astype(int), u.astype(int)] = u
        map_y[v.astype(int), u.astype(int)] = v
        color_map[v.astype(int), u.astype(int)] = self.colors

        # Apply interpolation using cv2.remap
        self.image = cv2.remap(color_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        color_map = np.zeros((self.H, self.W), dtype=np.float32)

        # Assign mapped coordinates for interpolation
        map_x[v.astype(int), u.astype(int)] = u
        map_y[v.astype(int), u.astype(int)] = v
        color_map[v.astype(int), u.astype(int)] = np.sqrt(np.sum(self._points**2,axis=1))

        
        # Apply interpolation using cv2.remap
        self.depthmap= cv2.remap(color_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    

    def get_image(self):
        """Return the generated equirectangular image."""
        return self.image, self.depthmap


