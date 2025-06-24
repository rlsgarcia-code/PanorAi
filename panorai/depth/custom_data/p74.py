import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import re
import time
from .disk_cached_transform import DiskCachedTransform, LMDBCachedTransform

import open3d as o3d
import os

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

import tempfile
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

import random

from panorai.path_config import get_path

class P74(Dataset):
    def __init__(self, module_for_test, module_for_validation, mode, size=(518, 518),
                 cypher=None, transform=None, n_angles=3):
        self.n_angles = n_angles
        self.cypher = cypher
        root = get_path("datasets", "p74")
        if root is None:
            raise FileNotFoundError("Dataset path for 'p74' not configured in paths.yaml")
        _files = [str(i) for i in Path(root).glob('*.ply')]
        self.files = self.split_dataset(_files, module_for_test, module_for_validation)[mode]
        self.mode = mode
        self.transform = transform

    def split_dataset(self, filenames, module_for_test, module_for_validation):
        train_files, test_files, val_files = [], [], []
        module_counts = defaultdict(int)
        for filename in filenames:
            match = re.search(r'(MD-\d+)', filename)
            if match:
                module_name = match.group(1)
                module_counts[module_name] += 1
                if module_name == module_for_test:
                    test_files.append(filename)
                elif module_name == module_for_validation:
                    val_files.append(filename)
                else:
                    train_files.append(filename)
            else:
                raise Exception(f"Could not extract module name from {filename}")
        print("\nModule Statistics:")
        for m, cnt in module_counts.items():
            print(f"{m}: {cnt} images")
        return {'train': train_files, 'test': test_files, 'val': val_files}

    def _load_sample(self, idx):
        filename = self.files[idx]
        xyz_image, rgb_image = read_encrypted_ply_and_rebuild_arrays(filename, self.cypher)
        R = np.sqrt(np.sum(xyz_image**2,axis=-1))  #np.linalg.norm(xyz_image, axis=-1)
        return {'rgb_image': rgb_image, 'xyz_image': R}

    def __getitem__(self, idx):

        file_path = self.files[idx]

        # Generate consistent but varied keys
        angle_idx = random.randint(0, self.n_angles - 1)
        flip = random.random() < 0.5

        key = file_path #f"{file_path}_a{angle_idx}_f{int(flip)}"
        sample = self._load_sample(idx)
        sample['origin']='p74'

        # if os.environ.get('PLOT_DEBUG', False):
        #     import matplotlib.pyplot as plt
        #     print(f'P74 shapes: \nrgb_image: {sample["rgb_image"].shape}\nxyz_image: {sample["xyz_image" ].shape}' )
        #     plt.title(' p74 - dataset')
        #     plt.imshow(sample['rgb_image'])
        #     plt.show()
        #     plt.imshow(sample['xyz_image'])
        #     plt.show()

        if isinstance(self.transform, DiskCachedTransform | LMDBCachedTransform):
            return self.transform({"key": key, "data": sample, "angle_idx": angle_idx, "flip": flip})
        else:
            return self.transform(sample)

    def __len__(self):
        return len(self.files)