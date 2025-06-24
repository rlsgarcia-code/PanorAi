# custom_data/p77.py
import sys
sys.path.insert(1,'/Users/robinsongarcia/projects/FineTuneDepthAnythingv2/Depth-Anything-V2/metric_depth/')



import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


import numpy as np

from .p77_utils import P77_Dataset, load_pcd, select_height
from .encrypted import load_encrypted_pcd

from pathlib import Path
import matplotlib.pyplot as plt
import time

import random
from .utils import sample_n_points, CachedTransform
import time

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


import torch
from torch.utils.data import Dataset
from pathlib import Path
import time
import random

from .utils import sample_n_points, CachedTransform
from .disk_cached_transform import DiskCachedTransform
from .p77_utils import P77_Dataset, load_pcd, select_height
from .encrypted import load_encrypted_pcd

class P77(Dataset):
    def __init__(self, size=(518, 518), transform=None, cypher=None):
        self.cypher = cypher
        self.files = [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p77/p77_encrypted").glob('*.ply')]
        self.transform = transform

    def _load_sample(self, idx):
        filename = self.files[idx]
        points, colors = load_encrypted_pcd(filename, self.cypher)
        H = select_height(points) // 2
        W = 2 * H
        projector = P77_Dataset(points, colors, H, W)
        projector.map_to_equirectangular()
        rgb, xyz = projector.get_image()
        return {'rgb_image': rgb, 'xyz_image': np.linalg.norm(xyz, axis=-1)}

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
    

class _P77(Dataset):
    def __init__(self, size=(518, 518), transform=None, cypher=None):
        self.cypher = cypher
        self.files = [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p77/p77_encrypted").glob('*.ply')]
        self.transform = transform
        self.size = size

        if isinstance(self.transform, CachedTransform):
            # Attach untransformed data by key
            data_dict = {
                f"{idx}": self._load_sample(idx, apply_transform=False)
                for idx in range(len(self.files))
            }
            self.transform.attach_data(data_dict)

        if isinstance(self.transform, DiskCachedTransform):
            self.transform.attach_lookup(
                    lambda key: self._load_sample(int(key.split('_')[0]), apply_transform=False)
                )

    def _load_sample(self, idx, apply_transform=True):
        filename = self.files[idx]
        points, colors = load_encrypted_pcd(filename, self.cypher)

        H = select_height(points) // 2
        W = 2 * H

        projector = P77_Dataset(points, colors, H, W)
        projector.map_to_equirectangular()
        image, R = projector.get_image()

        sample = {
            'rgb_image': image,
            'xyz_image': R,
        }

        if apply_transform and self.transform:
            if isinstance(self.transform, CachedTransform):
                raise RuntimeError("Should not apply CachedTransform directly to samples — pass key instead.")
            return self.transform(sample)
        return sample

    def __getitem__(self, idx):
        # Deterministic angle & flip
        angle_idx = idx % 8
        flip = (hash(str(idx)) % 2 == 0)
        key = f"{idx}_a{angle_idx}_f{int(flip)}"

        if isinstance(self.transform, CachedTransform):
            return self.transform(key)
        else:
            sample = self._load_sample(idx, apply_transform=False)
            return self.transform({'key': key, 'data': sample}) if self.transform else sample

    def __len__(self):
        return len(self.files)
    
class _P77(Dataset):
    def __init__(self, size=(518, 518), transform=None, cypher=None):

        self.cypher = cypher

        self.files = [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p77/p77_encrypted").glob('*.ply')]
        
     
        net_w, net_h = size
        self.transform = transform

        if isinstance(self.transform, CachedTransform):
            data_dict = {
                str(idx): self._load_sample(idx) for idx in range(len(self.files))
            }
            self.transform.attach_data(data_dict)
        

    
    def _load_sample(self, item):
        start = time.time()
        filename = self.files[item]

        points, colors = load_encrypted_pcd(filename, self.cypher)

        # Image parameters
        H = select_height(points) // 2
        W = 2 * H  # Equirectangular image dimensions
        
        # Create the projection
        projector = P77_Dataset(points, colors, H, W)
        projector.map_to_equirectangular()
        
        # Retrieve and display the image
        image, R = projector.get_image()

        sample = {
                'rgb_image': image,  #PreprocessEquirectangularImage.rotate(image, delta_lat=0, delta_lon=0),
                'xyz_image': R,  #PreprocessEquirectangularImage.rotate(depthmap, delta_lat=0, delta_lon=0)
            }
        
        end = time.time()
        #print(f'[P77] Data Processing Time: {(end - start):.3f}')
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __getitem__(self, idx):
        if isinstance(self.transform, CachedTransform):
            return self.transform(str(idx))
        else:
            sample = self._load_sample(idx)
            return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.files)


#from torch.utils.data import DataLoader
#def collate_fn(batch):
#    sample = batch[0]
#    for k,v in sample.items():
#        if k == 'image_path':
#            continue
#    sample['image'] = torch.as_tensor(sample['image'])#.permute(0,3,1,2)
#    sample['depth'] = torch.as_tensor(sample['depth'])#.permute(0,3,1,2)
#    sample['valid_mask'] = torch.as_tensor(sample['valid_mask'])#.permute(0,3,1,2)
    #sample['image_path'] = [torch.tensor(i) for i in sample['image_path'] ]
#    return sample

#def load_dataloader(cypher):
#    df = P77(cypher=cypher)
#    return DataLoader(df, batch_size=1, shuffle=True, collate_fn=collate_fn)