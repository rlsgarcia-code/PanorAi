import sys
sys.path.insert(1,'/Users/robinsongarcia/projects/FineTuneDepthAnythingv2/Depth-Anything-V2/metric_depth/')



import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet
from panorai.pipeline.utils import PreprocessEquirectangularImage

import numpy as np

from . import P77_Dataset, load_pcd
from .utils import select_height
from .encrypted import load_encrypted_pcd

from pathlib import Path
import matplotlib.pyplot as plt
from panorai import ProjectionPipeline, PipelineData
import time

import random
def sample_n_points(sample, N):
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

class P77(Dataset):
    def __init__(self, size=(518, 518), cypher=None):

        self.cypher = cypher

        self.files = [str(i) for i in Path("/Users/robinsongarcia/projects/.Datasets/PointCloud-DepthMaps/p77/p77_encrypted").glob('*.ply')]
        self.pipeline = ProjectionPipeline(sampler_name='FibonacciSampler', blender_name='AverageBlender', projection_name='gnomonic')
     
        net_w, net_h = size
        self.transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        filename = self.files[item]
        points, colors = load_encrypted_pcd(filename, self.cypher)

        # Image parameters
        H = select_height(points) // 2
        W = 2 * H  # Equirectangular image dimensions
        
        # Create the projection
        projector = P77_Dataset(points, colors, H, W)
        projector.map_to_equirectangular()
        
        # Retrieve and display the image
        image , depthmap = projector.get_image()
        data = PipelineData.from_dict(
            {
                'rgb' : PreprocessEquirectangularImage.rotate(image, delta_lat=0, delta_lon=0),
                'depth': PreprocessEquirectangularImage.rotate(depthmap, delta_lat=0, delta_lon=0)
            }
        )
    
        faces = self.pipeline.project(data, x_points=560, y_points=560, n_points=14)
        faces.pop('stacked')

        # TODO divide image by 255
        sample={
            'image':[],
            'depth':[],
            'valid_mask':[],
            'image_path':[]
        }
        images=[]
        depths=[]
        for facename in faces.keys():
            image = faces[facename]['rgb'] / 255.
            depth = faces[facename]['depth']
            transformed = self.transform({'image': image, 'depth': depth})
            sample['image'].append(transformed['image'])
            sample['depth'].append(transformed['depth'])
            sample['valid_mask'].append( (depth > 0.01))#[:,:,None])
            sample['image_path'].append(str(filename))

        sample = sample_n_points(sample, N=6)


        for k,v in sample.items():
            if k == 'image_path':
                continue
                
            sample[k] = np.stack(v)
        
        return sample

    def __len__(self):
        return len(self.files)


from torch.utils.data import DataLoader
def collate_fn(batch):
    sample = batch[0]
    for k,v in sample.items():
        if k == 'image_path':
            continue
    sample['image'] = torch.as_tensor(sample['image'])#.permute(0,3,1,2)
    sample['depth'] = torch.as_tensor(sample['depth'])#.permute(0,3,1,2)
    sample['valid_mask'] = torch.as_tensor(sample['valid_mask'])#.permute(0,3,1,2)
    #sample['image_path'] = [torch.tensor(i) for i in sample['image_path'] ]
    return sample

def load_dataloader(cypher):
    df = P77(cypher=cypher)
    return DataLoader(df, batch_size=1, shuffle=True, collate_fn=collate_fn)