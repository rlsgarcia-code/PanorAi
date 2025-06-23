from panorai import EquirectangularImage
import numpy as np
import random
import cv2

import numpy as np
from panorai import EquirectangularImage
import random

import numpy as np
from panorai import EquirectangularImage
import os

class DeterministicProjectionTransform:
    """
    Torch-style transform that deterministically applies:
    - longitude shift (angle_idx)
    - horizontal flip
    - gnomonic projection

    Args:
        size (int): Output face size (square).
        n_angles (int): Number of longitude buckets between 0–45°.
    """
    def __init__(self, size=518, n_angles=8):
        self.size = size
        self.n_angles = n_angles
        self.lon_angles = np.linspace(0, 45, n_angles)

    def __call__(self, sample_with_key):
        # Expect: { 'key': str, 'data': sample_dict }
        key = sample_with_key['key']
        sample = sample_with_key['data']

        idx = int(key.split('_')[0])
        angle_idx = int(key.split('_a')[1].split('_')[0])
        flip = int(key.split('_f')[1]) == 1

        delta_lon = float(self.lon_angles[angle_idx])
        H = W = 14 * (self.size // 14)

        out = {}
        for k, v in sample.items():
            if flip:
                v = np.flip(v, axis=-1)

            e = EquirectangularImage(v)
            e.preprocess(shadow_angle=30, delta_lat=0.0, delta_lon=delta_lon)
            e.attach_projection("gnomonic", x_points=W, y_points=H)
            faceset = np.array(e.to_gnomonic_face_set())

            if 'xyz' in k:
                u, v_ = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H), indexing='xy')
                faceset = np.sqrt(faceset**2 / (u[None]**2 + v_[None]**2 + 1))

            out[k] = np.ascontiguousarray(faceset.astype(np.float32))

        return out
    
class DeterministicAugmentor:
    def __init__(self, size, n_angles=8, enable_flip=True):
        self.size = size
        self.n_angles = n_angles
        self.lon_angles = np.linspace(0, 45, n_angles)
        self.enable_flip = enable_flip

    def __call__(self, sample, angle_idx=0, flip_idx=0):
        """
        Args:
            sample (dict): {'rgb_image': ..., 'xyz_image': ...}
            angle_idx (int): which angle bucket to use (0 to n_angles-1)
            flip_idx (int): 0 = no flip, 1 = flip
        """
        delta_lon = float(self.lon_angles[angle_idx])
        do_flip = (flip_idx == 1 and self.enable_flip)

        new_sample = {}
        H, W = 14 * (self.size // 14), 14 * (self.size // 14)

        for k, v in sample.items():
            if do_flip:
                v = np.flip(v, axis=-1)

            e = EquirectangularImage(v)
            e.preprocess(shadow_angle=30, delta_lat=0.0, delta_lon=delta_lon)
            e.attach_projection("gnomonic", x_points=W, y_points=H)
            faceset = np.array(e.to_gnomonic_face_set())

            if 'xyz' in k:
                u, v_ = np.meshgrid(
                    np.linspace(-1, 1, W),
                    np.linspace(-1, 1, H),
                    indexing='xy'
                )
                faceset = np.sqrt(faceset**2 / (u[None]**2 + v_[None]**2 + 1))

            new_sample[k] = np.array(faceset).astype(np.float32)
            new_sample[k] = np.ascontiguousarray(new_sample[k])

        return new_sample
    
import numpy as np
import random
from panorai import EquirectangularImage


import torch
import torch.nn.functional as F

import numpy as np
import cv2

def zero_depth_edges_batch(depth_maps, ksize=3, threshold=10):
    """
    Applies Sobel edge detection on a batch of depth maps and zeroes out edges.

    Args:
        depth_maps (np.ndarray): Batch of depth maps, shape (B, H, W).
        ksize (int): Sobel kernel size (must be 1, 3, 5, or 7).
        threshold (float): Gradient magnitude threshold for edge detection.

    Returns:
        np.ndarray: Depth maps with edges zeroed out (same shape as input).
    """
    batch_out = []

    for depth in depth_maps:
        # Convert to float32 for precision
        depth_f = depth.astype(np.float32)

        # Compute Sobel gradients
        grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=ksize)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Create a mask: 1 where no edge, 0 where edge
        edge_mask = (grad_mag < threshold).astype(np.float32)

        # Apply mask
        depth_masked = depth * edge_mask
        batch_out.append(depth_masked)

    return np.stack(batch_out)


class PrepareForNet:
    def __init__(self, size, seed=123456, n_angles=4, max_angle_deg=45):
        self.size = size
        self.rng = random.Random(seed)
        # self.__eval_mode = eval_mode
        # self.__train_mode = not eval_mode

        self.n_angles = n_angles
        self.max_angle_deg = max_angle_deg
        self.lon_angles = np.linspace(0, max_angle_deg, n_angles)

    # def eval(self):
    #     self.__eval_mode = True
    #     self.__train_mode = False

    # def train(self):
    #     self.__train_mode = True
    #     self.__eval_mode = False

    def __call__(self, sample_with_key):

        sample = sample_with_key["data"]
        angle_idx = sample_with_key["angle_idx"]
        do_flip = sample_with_key["flip"]

        delta_lon = float(self.lon_angles[angle_idx])
        delta_lat = 0.0

        origin = sample['origin']
        sample.pop('origin')


        H, W = 14 * (self.size // 14), 14 * (self.size // 14)
        new_sample = {}

        for k, v in sample.items():
            if k == ' origin':
                continue

            if do_flip:
                # print(k,v.shape)
                if 'rgb'in k:
                    v = np.flip(v,axis=-2)
                else:
                    v = np.flip(v, axis=-1)

            # if 'rgb' in k: print(f'PrepareForNet v_max: {v.max()}')
                
            e = EquirectangularImage(v)
            
            # if os.environ.get('PLOT_DEBUG', False):
            #     import matplotlib.pyplot as plt
            #     plt.title(f'PrepareForNet: {k}')
            #     plt.imshow(e)
            #     plt.show()
            e.preprocess(shadow_angle=30 if 'p74' in origin else 0, delta_lat=delta_lat, delta_lon=delta_lon)
            
            # if os.environ.get('PLOT_DEBUG', False):
            #     plt.title(f'PrepareForNet: {k} - processed')
            #     plt.imshow(e)
            #     plt.show()
            e.attach_projection("gnomonic", x_points=W, y_points=H)
            
            faceset_a = np.array(e.to_gnomonic_face_set())

            e.attach_sampler("fibonacci", x_points=W, y_points=H, n_points=16)
            
            faceset_b = np.array(e.to_gnomonic_face_set())

            faceset = np.vstack([faceset_a, faceset_b])

            # print(f'[FACESET SHAPE]: {faceset.shape}')

            # if 'rgb' in k: print(f'PrepareForNet faceset_max: {faceset.max()}')

            if 'xyz' in k:
               u, v_ = np.meshgrid(
                   np.linspace(-1, 1, W),
                   np.linspace(-1, 1, H),
                   indexing='xy'
               )
               faceset = np.sqrt(faceset**2 / (u[None]**2 + v_[None]**2 + 1))
            #    faceset = zero_depth_edges_batch(faceset, ksize=3, threshold=.1)
            
            #if 'rgb' in k:
            #    faceset = faceset / 255.
            
            # if 'rgb' in k: print(f'PrepareForNet faceset_max: {faceset.max()}')

            # if os.environ.get('PLOT_DEBUG', False):
            import matplotlib.pyplot as plt
            for face in faceset:
                plt.title(f'[zeroed edges] PrepareForNet: {k}')
                plt.imshow(face)
                plt.show()
                break

            new_sample[k] = faceset.astype(np.float32)
            new_sample[k] = np.ascontiguousarray(new_sample[k])

        # Add metadata
        new_sample["angle_idx"] = angle_idx
        new_sample["flip"] = do_flip
        

        return new_sample
    
class _PrepareForNet:
    def __init__(self, size, seed=123456, eval_mode = False):
        self.size = size
        self.rng = random.Random(seed)
        self.__eval_mode = eval_mode
        self.__train_mode= not eval_mode

    
    def eval(self):
        self.__eval_mode = True
        self.__train_mode = False

    def train(self):
        self.__train_mode = True
        self.__eval_mode = False
        
    def __call__(self, sample):
        if self.__train_mode:
            delta_lon = self.rng.uniform(0, 45)
            delta_lat = 0.0
            do_flip = self.rng.random() < 0.5  # 50% chance to flip
            do_blur = self.rng.random() < 0.5  # 50% chance to flip


        else:
            delta_lon, delta_lat = (0.0, 0.0)
            do_flip = False
            do_blur = False

        new_sample = {}
        H, W = 14 * (self.size // 14), 14 * (self.size // 14)
        for k,v in sample.items():
            if do_flip:
                v = np.flip(v, axis=-1)
            #if (do_blur) & (self.__train_mode) & ('rgb' in k):
            #    blur_val = random.randint(5,15) #blur value random
            #    v = cv2.blur(v,(blur_val, blur_val))
                
            e = EquirectangularImage(v)
            e.preprocess(shadow_angle=30, delta_lat=delta_lat, delta_lon=delta_lon)
            
            e.attach_projection("gnomonic", x_points= W, y_points = H)
            faceset = np.array(e.to_gnomonic_face_set())

            if 'xyz' in k:

                u,v = np.meshgrid(
                    np.linspace(-1, 1, W),
                    np.linspace(-1, 1, H),
                    indexing='xy'
                )
    
                faceset = np.sqrt( faceset**2 / (u[None]**2 + v[None]**2 + 1))

            
            new_sample[k] = np.array(faceset).astype(np.float32)
            new_sample[k] = np.ascontiguousarray(new_sample[k])
        return new_sample
        

class NormalizeImage(object):
    """Normlize image by given mean and std.
    """
    __keyless__ = True
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["rgb_image"] = (sample["rgb_image"] / 255. - self.__mean) / self.__std

        return sample

