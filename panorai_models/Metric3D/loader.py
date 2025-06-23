"""
m3dv2.py
========

Lazy loader for Metric3D-based monodepth with various backbones.
"""

import torch
import cv2
import numpy as np
from PIL import Image
try:
    from mmcv.utils import Config
except:
    from mmengine.config import Config
import os, sys
from pathlib import Path
from panorai.path_config import get_path

sys.path.append(os.path.dirname(__file__))
from ..registry import ModelRegistry


class Data:
    def __init__(self, pred_depth, confidence, outdict):
        self._pred_depth = pred_depth
        self._confidence = confidence
        self._outdict = outdict

    @property
    def pred_depth(self):
        return self._pred_depth

    @property
    def confidence(self):
        return self._confidence

    @property
    def outdict(self):
        return self._outdict

@ModelRegistry.register("m3dv2", default_args={
    "backbone": "ViT-Large",
    "cfg_file": None,
    "ckpt_file": None,
    "device": "mps"
})
def load_m3dv2_model(
    backbone,
    cfg_file,
    ckpt_file,
    device
):
    """
    Build a Metric3D model, load checkpoint, return inference(filename)->depth.
    """
    
    from .mono.model.monodepth_model import get_configured_monodepth_model


    if cfg_file is None:
        cfg_file = get_path("metric3d", "cfg_file")
    if ckpt_file is None:
        ckpt_file = get_path("metric3d", "ckpt_file")
    if cfg_file is None or ckpt_file is None:
        raise FileNotFoundError(
            "cfg_file or ckpt_file not provided and 'metric3d' paths not set in paths.yaml"
        )

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    print(f"[m3dv2] Loading from {ckpt_file}")

    if ckpt_file.startswith("http"):
        state = torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"]
    else:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        state = checkpoint["model_state_dict"]
    model.load_state_dict(state, strict=False)

    model.to(device).eval()

    def _preprocess(rgb: np.ndarray) -> torch.Tensor:
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)
        std  = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)

        x = torch.from_numpy(rgb.transpose(2,0,1)).float()
        x = (x - mean) / std
        return x.unsqueeze(0).to(device)

    class Model:
        def inference(self, arr: np.ndarray) -> np.ndarray:
            shape = arr.shape
            x = _preprocess(arr)
            with torch.no_grad():
                pred_depth, confidence, outdict = model.inference({"input": x})
            
            

            self.data = Data(pred_depth, confidence, outdict)
            
            depth = pred_depth[0,0].cpu().numpy()
            return cv2.resize(depth, shape[:2])
        
        def __call__(self, arr: np.ndarray) -> np.ndarray:
            return self.inference(arr)

    return Model()