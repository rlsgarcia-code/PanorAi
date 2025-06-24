"""
dust3r.py
=========

Lazy loader for DUSt3R model.
"""

import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(__file__))
from ..registry import ModelRegistry
from pathlib import Path
#"naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", #

def build_model():
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path
    from PIL import Image
    import cv2

    from .dust3r.inference import inference as dust3r_inference
    from .dust3r.model import AsymmetricCroCo3DStereo
    from .dust3r.utils.image import load_images
    


    class Dust3RStereoWrapper(nn.Module):
        def __init__(self, checkpoint_path: str, device: torch.device, size: int = 518):
            super().__init__()
            print(f"[dust3r] Loading from {checkpoint_path}")
            self.model = AsymmetricCroCo3DStereo.from_pretrained(checkpoint_path).to(device)
            self.device = device
            self.size = size
            self.cache_dir = Path(".cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        def forward(self, image_np: np.ndarray) -> torch.Tensor:
            """
            Expects input as a NumPy array with shape (H, W, 3), dtype uint8 or float32 scaled to [0,255].
            Returns a torch.Tensor depth map resized to original shape.
            """
            assert image_np.ndim == 3 and image_np.shape[2] == 3, "Expected image shape (H, W, 3)"
            H, W = image_np.shape[:2]

            tmp_file = self.cache_dir / "tmp.png"
            Image.fromarray(image_np.astype(np.uint8)).save(tmp_file)

            filename = str(tmp_file)
            images = load_images([filename, filename], size=self.size)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

            with torch.no_grad():
                output = dust3r_inference(pairs, self.model, self.device, batch_size=1)

            depth = output['pred1']['pts3d'][0, :, :, -1].cpu().numpy()
            depth_resized = cv2.resize(depth, (W, H))

            return torch.from_numpy(depth_resized).float()
    return Dust3RStereoWrapper


@ModelRegistry.register("dust3r", default_args={
    "checkpoint_path": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    "device": "mps",
    "size": 512,
    "return_model": False,
    "max_depth": 80,
})
def load_dust3r_model(
    checkpoint_path,
    device,
    size,
    return_model,
    max_depth
):
    """
    Loads DUSt3R from checkpoint, returns an inference(filename) -> depth function.
    """
    from .dust3r.inference import inference as dust3r_inference
    from .dust3r.model import AsymmetricCroCo3DStereo
    from .dust3r.utils.image import load_images
    from .dust3r.image_pairs import make_pairs

    print(f"[dust3r] Loading from {checkpoint_path}")
    if return_model:
        model = build_model()
        return model(checkpoint_path, device, size)
    model = AsymmetricCroCo3DStereo.from_pretrained(checkpoint_path).to(device)

    def inference(arr: np.ndarray) -> np.ndarray:
        # Stereo trick: pass same image twice
        H, W = arr.shape[:2]
        cache_dir = Path(".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_dir / "tmp.png"
        Image.fromarray(arr).save(tmp_file)
        
        filename = str(tmp_file)
        images = load_images([filename, filename], size=size)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

        output = dust3r_inference(pairs, model, device, batch_size=1)

        # Depth is pred1['pts3d'][0,:,:, -1]
        pred1 = output['pred1']
        depth = pred1['pts3d'][0, :, :, -1].numpy()

        return cv2.resize(depth, (W, H))
    
    if return_model:
        inference

    return inference