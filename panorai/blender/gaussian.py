# panorai/pipeline/blender/feathering.py

import numpy as np
from .base_blenders import BaseBlender
from typing import Any

import numpy as np

def multivariate_gaussian_2d(x, mean, cov):
    """
    Computes the 2D multivariate Gaussian (normal) probability density function.
    
    Parameters
    ----------
    x : np.ndarray
        Coordinates at which to evaluate the PDF.
        - Can be shape (2,) for a single point, or shape (N, 2) for N points.
    mean : np.ndarray of shape (2,)
        Mean vector of the distribution.
    cov : np.ndarray of shape (2, 2)
        Covariance matrix (must be positive definite).
        
    Returns
    -------
    pdf : float or np.ndarray
        The PDF value(s) at x. Returns a float if x is shape (2,),
        or an array of shape (N,) if x is shape (N, 2).
    """
    # Make sure mean is 1D of shape (2,)
    mean = np.asarray(mean).reshape(-1)
    
    # Ensure x is 2D for vectorized operations: shape (N, 2)
    x = np.atleast_2d(x)
    
    # Inverse of covariance and determinant
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    
    if det_cov <= 0:
        raise ValueError("Covariance matrix must be positive definite (det > 0).")
    
    # Normalization factor for 2D
    norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    
    # Center each point by the mean
    diff = x - mean  # shape (N, 2)
    
    # Compute the exponent term for each point:
    #   exponent = -0.5 * (diff @ inv_cov @ diff.T)
    # Using einsum for vectorized quadratic form:
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff)
    
    # Final PDF values
    pdf_vals = norm_factor * np.exp(exponent)
    
    # If original x was just a single point, return a float
    if pdf_vals.shape[0] == 1:
        return pdf_vals[0]
    return pdf_vals


def get_distribution(fov_deg, H, W, mu=0, sig=1):
    v_max = u_max = np.tan(np.deg2rad(fov_deg/2))
    v_min = u_min = -u_max
    grid = np.stack(np.meshgrid(np.linspace(u_min, u_max, W),np.linspace(v_min, v_max, H)))
    probs = multivariate_gaussian_2d(grid.reshape(2,-1).T, mean=np.array([mu,mu]), cov=np.diag(np.array([sig,sig]))).reshape((512,512))
    return probs


class GaussianBlender(BaseBlender):


    def blend(self, images, masks, **kwargs):
        """
        Blends images using feathering.

        :param images: List of image arrays.
        :param masks: List of corresponding masks for weighting.
        :return: Blended image.
        """
        if not images or not masks or len(images) != len(masks):
            raise ValueError("Images and masks must have the same non-zero length.")

        img_shape = images[0].shape
        combined = np.zeros(img_shape, dtype=np.float32)
        weight_map = np.zeros(img_shape[:2], dtype=np.float32)

        required_keys = ['fov_deg', 'projector', 'tangent_points']
        missing_keys = [key for key in required_keys if key not in self.params]
        if missing_keys:
            raise ValueError(f"Error: Missing required parameters: {', '.join(missing_keys)}")
        fov_deg = self.params.get('fov_deg')
        tangent_points = self.params.get('tangent_points')
        projector = self.params.get('projector')
        mu = self.params.get('mu',0)
        sig = self.params.get('sig', 1)

        
        for img, mask, (lat_deg, lon_deg) in zip(images, masks, tangent_points):
            
            projector.config.update(
                phi1_deg=lat_deg,
                lam0_deg=lon_deg,
            )

            # Feather the mask using Euclidean distance transform
            distance = get_distribution(fov_deg, projector.config.x_points, projector.config.y_points, mu=mu, sig=sig)
            distance = np.dstack([distance, distance, distance])
            equirect_weights = projector.backward(distance, return_mask=False)[ :, :, 0]
            equirect_mask = equirect_weights / distance.max()  # Normalize to [0, 1]

            # Apply blending
            combined += img * equirect_mask[..., None]
            weight_map += equirect_mask

        # Normalize the blended image
        valid_weights = weight_map > 0
        combined[valid_weights] /= weight_map[valid_weights, None]

        # Ensure zero weights remain zero
        combined[~valid_weights] = 0
        return combined