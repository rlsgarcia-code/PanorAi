# custom_data/py4.py
import sys
sys.path.insert(1,'/Users/robinsongarcia/projects/FineTuneDepthAnythingv2/Depth-Anything-V2/metric_depth/')


import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


import numpy as np

 

from pathlib import Path
import matplotlib.pyplot as plt
from panorai import PanoraiFactory
import time

import random

import re

import open3d as o3d
import numpy as np
import tempfile


import re

import re
from collections import defaultdict

from .utils import sample_n_points, CachedTransform
from .disk_cached_transform import DiskCachedTransform
import time



def split_dataset(filenames, module_for_test=None, module_for_validation=None):
    """
    Splits a list of filenames into train, test, and validation sets 
    based on a given module name.

    Parameters:
    - filenames: List of file paths.
    - module_for_test: Module name (e.g., "MD-04") to be used for the test set.
    - module_for_validation: Module name (e.g., "MD-05") to be used for the validation set.

    Returns:
    - A dictionary containing lists of filenames for 'train', 'test', and 'val'.
    """

    if not module_for_test or not module_for_validation:
        raise Exception("Must set both module_for_test and module_for_validation")

    train_files, test_files, val_files = [], [], []

    # Count occurrences of each module
    module_counts = defaultdict(int)

    for filename in filenames:
        match = re.search(r'(MD-\d+)', filename)  # Extract module name (e.g., "MD-04")
        if match:
            module_name = match.group(1)
            module_counts[module_name] += 1  # Count occurrences

            if module_name == module_for_test:
                test_files.append(filename)
            elif module_name == module_for_validation:
                val_files.append(filename)
            else:
                train_files.append(filename)
        else:
            raise Exception(f"⚠️ Warning: Could not extract module name from {filename}")

    # Print module statistics
    print("\n🔹 **Number of Images Per Module:**")
    for module, count in module_counts.items():
        print(f"   📂 {module}: {count} images")

    print("\n📊 **Dataset Split Summary:**")
    print(f"   🏋️‍♂️ Train Set: {len(train_files)} files")
    print(f"   🧪 Test Set ({module_for_test}): {len(test_files)} files")
    print(f"   🎯 Validation Set ({module_for_validation}): {len(val_files)} files")

    return {'train': train_files, 'test': test_files, 'val': val_files}

def extract_hw_from_ply_filename(filename):
    """
    Extracts H and W from a filename formatted as 'originalname_HxW.ply' 
    or 'originalname_HxW_encrypted.ply'.
    
    Parameters:
    - filename: The .ply filename with embedded H and W.

    Returns:
    - H: Image height (int).
    - W: Image width (int).
    """
    match = re.search(r"_(\d+)x(\d+)(?:_encrypted)?\.ply$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # (H, W)
    else:
        raise ValueError(f"Filename does not contain valid HxW dimensions: {filename}")


def read_ply_and_rebuild_arrays(ply_filename, cipher=None):
    """
    Read a PLY file and reconstruct the original XYZ and RGB image arrays.
    
    Parameters:
    - ply_filename: The filename of the PLY file with H and W encoded.

    Returns:
    - xyz_image: (H, W, 3) numpy array with XYZ coordinates.
    - rgb_image: (H, W, 3) numpy array with RGB values (uint8).
    """
    # Extract H and W from the filename
    H, W = extract_hw_from_ply_filename(ply_filename)

    # Read the PLY file
    pcd = o3d.io.read_point_cloud(ply_filename)
    
    # Convert to NumPy arrays
    points = np.asarray(pcd.points)  # Shape: (H*W, 3)
    colors = np.asarray(pcd.colors)  # Shape: (H*W, 3), values in [0,1]

    # Convert colors back to uint8
    colors = (colors * 255).astype(np.uint8)

    # Reshape back to (H, W, 3)
    xyz_image = points.reshape(H, W, 3)
    rgb_image = colors.reshape(H, W, 3)

    return xyz_image, rgb_image



def read_encrypted_ply_and_rebuild_arrays(encrypted_ply_filename, cipher):
    """
    Read an encrypted PLY file, decrypt it in memory, and reconstruct the original XYZ and RGB image arrays.

    Parameters:
    - encrypted_ply_filename: The filename of the encrypted PLY file.
    - cipher: A cryptography.Fernet cipher object for decryption.

    Returns:
    - xyz_image: (H, W, 3) numpy array with XYZ coordinates.
    - rgb_image: (H, W, 3) numpy array with RGB values (uint8).
    """
    # Extract H and W from the filename
    H, W = extract_hw_from_ply_filename(encrypted_ply_filename.split('/')[-1])

    # Read encrypted file
    with open(encrypted_ply_filename, "rb") as file:
        encrypted_data = file.read()

    # Decrypt file
    decrypted_data = cipher.decrypt(encrypted_data)

    # Use a temporary file to hold the decrypted .ply content
    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as temp_file:
        temp_file.write(decrypted_data)
        temp_file.flush()  # Ensure data is written

        # Read the PLY file using Open3D
        pcd = o3d.io.read_point_cloud(temp_file.name)

    # Convert to NumPy arrays
    points = np.asarray(pcd.points)  # Shape: (H*W, 3)
    colors = np.asarray(pcd.colors)  # Shape: (H*W, 3), values in [0,1]

    # Convert colors back to uint8
    colors = (colors * 255).astype(np.uint8)

    # Reshape back to (H, W, 3)
    xyz_image = points.reshape(H, W, 3)
    rgb_image = colors.reshape(H, W, 3)

    return xyz_image, rgb_image

def _sample_n_points(sample, N):
    """Randomly sample N points from the dataset."""
    total_samples = len(sample['image'])
    N = min(N, total_samples)  # Ensure we don't sample more than available
    indices = random.sample(range(total_samples), N)

    sampled_data = {
        'image': [sample['image'][i] for i in indices],
        'depth': [sample['depth'][i] for i in indices],
        'valid_mask': [sample['valid_mask'][i] for i in indices],
        'image_path': [sample['image_path'][i] for i in indices]
    }

    return sampled_data


class P74(Dataset):
    def __init__(self, module_for_test, module_for_validation, mode, size=(518, 518), cypher=None, transform=None):
        self.cypher = cypher
        self.files = split_dataset(
            [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p74/p74_encrypted/").glob('*.ply')],
            module_for_test, module_for_validation
        )[mode]
        self.mode = mode
        self.transform = transform

    def _load_sample(self, idx):
        filename = self.files[idx]
        xyz_image, rgb_image = read_encrypted_ply_and_rebuild_arrays(filename, self.cypher)
        R = np.linalg.norm(xyz_image, axis=-1)  # Radius/depth from XYZ
        return {'rgb_image': rgb_image, 'xyz_image': R}

    def __getitem__(self, idx):
        angle_idx = idx % 8
        flip = (hash(str(idx)) % 2 == 0)
        key = f"{idx}_a{angle_idx}_f{int(flip)}"

        sample = self._load_sample(idx)
        if isinstance(self.transform, CachedTransform):
            return self.transform(key)
        elif isinstance(self.transform, DiskCachedTransform):
            return self.transform({'key': key, 'data': sample})
        else:
            return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.files)

class _P74(Dataset):
    def __init__(self, module_for_test,  
                 module_for_validation, 
                 mode, size=(518, 518), 
                 cypher=None,
                 transform=None):

        self.cypher = cypher

        _files = [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p74/p74_encrypted/").glob('*.ply')]
        
        self.files = split_dataset(_files, module_for_test, module_for_validation)[mode]
        
        self.mode = mode
        net_w, net_h = size

        self.transform = transform

        if isinstance(self.transform, CachedTransform):
            data_dict = {
                str(idx): self._load_sample(idx) for idx in range(len(self.files))
            }
            self.transform.attach_data(data_dict)

        if isinstance(self.transform, DiskCachedTransform):
            self.transform.attach_lookup(
                    lambda key: self._load_sample(int(key.split('_')[0]), apply_transform=False)
                )

    
    def _load_sample(self, item):
        start = time.time()
        filename = self.files[item]
        
        xyz_image, rgb_image = read_encrypted_ply_and_rebuild_arrays(filename, self.cypher)
        

        R = np.sqrt(np.sum(np.array(xyz_image)**2, axis=-1))


        
        sample = {'xyz_image': R, 'rgb_image' : rgb_image}

        end = time.time()
        
        #if self.transform:
        #    sample = self.transform(sample)
        
        return sample

    def __getitem__(self, idx):
        angle_idx = idx % 8
        flip = (hash(str(idx)) % 2 == 0)
        key = f"{idx}_a{angle_idx}_f{int(flip)}"

        sample = self._load_sample(idx)
        return {"key": key, "data": sample}

    def __len__(self):
        return len(self.files)
