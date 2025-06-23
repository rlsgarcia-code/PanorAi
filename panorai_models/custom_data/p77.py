import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import time
import random
from .utils import sample_n_points, CachedTransform
from .encrypted import load_encrypted_pcd
from .p77_utils import P77_Dataset, select_height
from .disk_cached_transform import DiskCachedTransform, LMDBCachedTransform
import os
from panorai.path_config import get_path

class P77(Dataset):
    def __init__(self, size=(518, 518), cypher=None, transform=None, n_angles=3):
        self.cypher = cypher
        root = get_path("datasets", "p77")
        if root is None:
            raise FileNotFoundError("Dataset path for 'p77' not configured in paths.yaml")
        self.files = [str(i) for i in Path(root).glob('*.ply')]
        self.transform = transform
        self.n_angles = n_angles

    def _load_sample(self, idx):
        filename = self.files[idx]
        points, colors = load_encrypted_pcd(filename, self.cypher)
        H = select_height(points) // 2
        W = 2 * H
        projector = P77_Dataset(points, colors, H, W)
        projector.map_to_equirectangular()
        rgb, xyz = projector.get_image()
        return {'rgb_image': rgb, 'xyz_image': xyz}

    def __getitem__(self, idx):
        
        file_path = self.files[idx]

        # Generate consistent but varied keys
        angle_idx = random.randint(0, self.n_angles - 1)
        flip = random.random() < 0.5

        key = file_path#f"{file_path}_a{angle_idx}_f{int(flip)}"
        sample = self._load_sample(idx)
        sample['origin']='p77'

        # if os.environ.get('PLOT_DEBUG', False):
        #     import matplotlib.pyplot as plt
        #     print(f'P77 shapes: \nrgb_image: {sample["rgb_image"].shape}\nxyz_image: {sample["xyz_image" ].shape}' )
        #     plt.title(' p77 - dataset')
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