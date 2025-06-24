"""train_utils.py
=================
Factory helpers to wire **datasets**, **model + optim + scheduler**, **loss_fn**
and the `TransformerUnfreezeScheduler` that cooperates with the *DepthTrainer*
(re-written 2025-05-09).

Key features
------------
* Zero duplicate class definitions.
* Cosine-Annealing-WarmRestarts scheduler contado em **micro-steps**.
* Albumentations wrapper with NaN-mask fix, *picklable* (multiprocess-safe).
* Boom-log: every unfreeze event appended to ``unfreeze_log.jsonl``.
* Fully-typed public API:
    ─ ``build_dataloaders``
    ─ ``build_model_optim_sched``
    ─ ``build_loss_fn``
    ─ ``search_best_scale``
    ─ ``TransformerUnfreezeScheduler``
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# project-internal ————————————————————————————————————————————————
from panorai.depth import ModelRegistry
from panorai.depth.trainers.losses import FixedDepthLoss
from panorai.depth.trainers.transforms import PrepareForNet
from panorai.depth.trainers.estimate_scales import estimate_best_scale
from custom_data import load_datasets as _load_datasets, get_cypher, collate_fn
from custom_data.disk_cached_transform import DiskCachedTransform
from torchvision.transforms import Compose

# ════════════════════════════════════════════════════════════════════════════
# Albumentations: global pipelines  →  picklable
# ════════════════════════════════════════════════════════════════════════════
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

AUG_TRAIN = A.Compose([
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
    # A.GaussianBlur(blur_limit=3, p=0.2),


    A.CoarseDropout(
            num_holes_range=[1, 5],
            hole_height_range=[0.1, 0.2],
            hole_width_range=[0.1, 0.2],
            fill_mask=0.0001,
            fill=0,
            p=0.5
        ),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

AUG_VAL = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# class AlbumentationsWrapper:
#     """Callable que aplica Albumentations e corrige *NaNs* na depth."""

#     def __init__(self, aug: A.Compose):
#         self.aug = aug

#     def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
#         img = sample["rgb_image"].astype(np.uint8)
#         depth = sample["xyz_image"]

#         img_out, depth_out = [], []
#         for i in range(img.shape[0]):
#             res = self.aug(image=np.ascontiguousarray(img[i]))
#             aug_img = res["image"]

#             mask = np.all(aug_img == 0, axis=-1)
#             d = depth[i].copy()
#             d[mask] = 1e-6  # evita NaNs na loss

#             img_out.append(aug_img)
#             depth_out.append(d)

#         sample["rgb_image"] = np.stack(img_out, 0)
#         sample["xyz_image"] = np.stack(depth_out, 0)
#         return sample
import cv2
import numpy as np
import matplotlib.pyplot as plt

EDGE_DEBUG_MAX_PLOTS = 5
edge_debug_counter = [0]  # mutable global counter

class AlbumentationsWrapper:
    """Callable que aplica Albumentations, corrige *NaNs* e remove bordas ruidosas na profundidade."""

    def __init__(self, aug: A.Compose, edge_ksize: int = 3, edge_thresh: float = 10.0, debug: bool = False):
        self.aug = aug
        self.edge_ksize = edge_ksize
        self.edge_thresh = edge_thresh
        self.debug = debug

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = sample["rgb_image"].astype(np.uint8)
        depth = sample["xyz_image"]

        img_out, depth_out = [], []
        for i in range(img.shape[0]):
            res = self.aug(image=np.ascontiguousarray(img[i]))
            aug_img = res["image"]

            mask = np.all(aug_img == 0, axis=-1)
            d = depth[i].copy()
            d[mask] = 1e-6  # evita NaNs na loss

            img_out.append(aug_img)
            depth_out.append(d)

        sample["rgb_image"] = np.stack(img_out, 0)
        sample["xyz_image"] = np.stack(depth_out, 0)

        # # # ───────────────────────
        # # # Mask strong Z-gradients
        # # # ───────────────────────
        # # masked = []
        # # for d in sample["xyz_image"]:
        # #     if d.ndim != 3 or d.shape[2] < 3:
                
        # #         masked.append(d)
        # #         continue

        # #     z = d[:, :, 2].astype(np.float32)
        # #     gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=self.edge_ksize)
        # #     gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=self.edge_ksize)
        # #     gmag = np.sqrt(gx**2 + gy**2)
        # #     mask = (gmag < self.edge_thresh).astype(np.float32)

        # #     d_out = d.copy()
        # #     d_out[:, :, 2] *= mask
        # #     masked.append(d_out)

        # #     # ─── Debug Plot ─────────────────────────
        # #     if self.debug and edge_debug_counter[0] < EDGE_DEBUG_MAX_PLOTS:
        # #         edge_debug_counter[0] += 1
        # #         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # #         axs[0].imshow(z, cmap='viridis')
        # #         axs[0].set_title("Original Z-Depth")
        # #         axs[1].imshow(gmag, cmap='inferno')
        # #         axs[1].set_title("Z Gradient Magnitude")
        # #         axs[2].imshow(z * mask, cmap='viridis')
        # #         axs[2].set_title("Masked Z (after filter)")
        # #         for ax in axs:
        # #             ax.axis('off')
        # #         plt.tight_layout()
        # #         plt.show()

        # sample["xyz_image"] = np.stack(masked, 0)
        return sample

def _get_aug(eval_mode: bool = False) -> AlbumentationsWrapper:
    """Devolve wrapper picklável (usado pelos *DataLoader* workers)."""
    return AlbumentationsWrapper(AUG_VAL if eval_mode else AUG_TRAIN)

# # ════════════════════════════════════════════════════════════════════════════
# # Data loaders
# # ════════════════════════════════════════════════════════════════════════════
# def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
#     ds = cfg["dataset"]
#     size = ds.get("size", 518)
#     n_tr, n_val = ds.get("n_angles_train", 10), ds.get("n_angles_val", 1)
#     bs = ds.get("batch_size", 2)

#     tr_tf = DiskCachedTransform(
#         transform_fn=Compose([
#             PrepareForNet(size=size, n_angles=n_tr, max_angle_deg=45, eval_mode=False),
#         ]),
#         post_transform_fn=_get_aug(eval_mode=False),
#         cache_dir=ds.get("train_cache_dir", ".cache/train"),
#     )
#     val_tf = DiskCachedTransform(
#         transform_fn=Compose([
#             PrepareForNet(size=size, n_angles=n_val, max_angle_deg=45, eval_mode=True),
#         ]),
#         post_transform_fn=_get_aug(eval_mode=True),
#         cache_dir=ds.get("val_cache_dir", ".cache/val"),
#     )

#     datasets = _load_datasets(
#         ds["train_name"], ds["val_name"], get_cypher(),
#         train_transform=tr_tf, valid_transform=val_tf,
#         n_angles_train=n_tr, n_angles_val=n_val,
#     )

#     tr_loader = DataLoader(
#         datasets["trainset"], batch_size=bs, shuffle=True, drop_last=True,
#         num_workers=ds.get("train_workers", 3), prefetch_factor=2,
#         collate_fn=collate_fn, persistent_workers=True,
#     )
#     val_loader = DataLoader(
#         datasets["valset"], batch_size=1, shuffle=False, drop_last=False,
#         num_workers=ds.get("val_workers", 4), prefetch_factor=2,
#         collate_fn=collate_fn, persistent_workers=True,
#     )
#     return tr_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
from custom_data.disk_cached_transform import TeacherCacher

# def build_dataloaders(cfg: Dict,
#                       teacher_model: torch.nn.Module | None = None
#                      ) -> Tuple[DataLoader, DataLoader]:
#     """
#     If `teacher_model` is passed, the train transform will call / cache
#     its coarse predictions.  Validation stays teacher-free.
#     """
#     ds   = cfg["dataset"]
#     size = ds.get("size", 518)
#     n_tr, n_val = ds.get("n_angles_train", 10), ds.get("n_angles_val", 1)
#     bs   = ds.get("batch_size", 2)

#     # ─── optional teacher cacher ---------------------------------------
#     teacher_cacher = None
#     if teacher_model is not None:
#         teacher_cacher = TeacherCacher(
#             cfg,
#             cache_root=ds.get("teacher_cache_dir", ".cache/teacher"),
#             device = cfg.get("device", "mps")
#         )

#     # # ─── train transform (with teacher) --------------------------------
#     # tr_tf = DiskCachedTransform(
#     #     transform_fn=Compose([
#     #         PrepareForNet(size=size, n_angles=n_tr, max_angle_deg=45),
#     #     ]),
#     #     post_transform_fn=_get_aug(eval_mode=False),
#     #     cache_dir=ds.get("train_cache_dir", ".cache/train"),
#     #     teacher_cacher=teacher_cacher,          # ← inject here
#     #     train_mode=True,
#     #     n_angles=n_tr,
#     #     max_angle_deg=45
#     # )

#     # # ─── validation transform (no teacher) -----------------------------
#     # val_tf = DiskCachedTransform(
#     #     transform_fn=Compose([
#     #         PrepareForNet(size=size, n_angles=n_val, max_angle_deg=45),
#     #     ]),
#     #     post_transform_fn=_get_aug(eval_mode=True),
#     #     cache_dir=ds.get("val_cache_dir", ".cache/val"),
#     #     teacher_cacher=teacher_cacher,                     # keep val clean
#     #     train_mode=False,
#     #     n_angles=n_val,
#     #     max_angle_deg=45
#     # )

#     # ─── train transform (with teacher) --------------------------------
#     tr_tf = DiskCachedTransform(
#         transform_fn=Compose([
#             PrepareForNet(size=size, n_angles=n_tr, max_angle_deg=45),
#         ]),
#         post_transform_fn=_get_aug(eval_mode=False),
#         cache_dir=ds.get("train_cache_dir", ".cache/train"),
#         teacher_cacher=teacher_cacher,          # teacher only in training
#         train_mode=True,
#         n_angles=n_tr,
#         max_angle_deg=45,
#         # only cache two of your four angles (for example)
#         #cache_angle_idxs=[0, 2],
#     )

#     # ─── validation transform (no teacher) -----------------------------
#     val_tf = DiskCachedTransform(
#         transform_fn=Compose([
#             PrepareForNet(size=size, n_angles=n_val, max_angle_deg=45),
#         ]),
#         post_transform_fn=_get_aug(eval_mode=True),
#         cache_dir=ds.get("val_cache_dir", ".cache/val"),
#         teacher_cacher=None,                    # no teacher at validation
#         train_mode=False,
#         n_angles=n_val,
#         max_angle_deg=45,
#         # for val we only ever need the zero‐angle, so cache that alone
#         cache_angle_idxs=[0],
#     )

#     # ─── dataset loading ----------------------------------------------
#     datasets = _load_datasets(
#         ds["train_name"], ds["val_name"], get_cypher(),
#         train_transform=tr_tf, valid_transform=val_tf,
#         n_angles_train=n_tr, n_angles_val=n_val,
#     )

#     # ─── data loaders --------------------------------------------------
#     tr_loader = DataLoader(
#         datasets["trainset"], batch_size=bs, shuffle=True, drop_last=True,
#         num_workers=ds.get("train_workers", 3), prefetch_factor=2,
#         collate_fn=collate_fn, persistent_workers=True,
#     )
#     val_loader = DataLoader(
#         datasets["valset"], batch_size=1, shuffle=False, drop_last=False,
#         num_workers=ds.get("val_workers", 4), prefetch_factor=2,
#         collate_fn=collate_fn, persistent_workers=True,
#     )
#     return tr_loader, val_loader

from custom_data.disk_cached_transform import TeacherCacher, DiskCachedTransform, LMDBCachedTransform

def build_dataloaders(cfg: Dict,
                      teacher_model: torch.nn.Module | None = None
                     ) -> Tuple[DataLoader, DataLoader]:
    """
    If `teacher_model` is passed, the train transform will use LMDB caching
    for expensive projections via LMDBCachedTransform. Validation remains teacher-free.
    """
    ds   = cfg["dataset"]
    size = ds.get("size", 518)
    n_tr, n_val = ds.get("n_angles_train", 10), ds.get("n_angles_val", 1)
    bs   = ds.get("batch_size", 2)

    # Optional teacher cacher
    teacher_cacher = None
    if teacher_model is not None:
        teacher_cacher = TeacherCacher(
            cfg,
            cache_root=ds.get("teacher_cache_dir", ".cache/teacher"),
            device=cfg.get("device", "mps")
        )

    # Train transform with LMDB cache
    tr_tf = LMDBCachedTransform(
        transform_fn=Compose([
            PrepareForNet(size=size, n_angles=n_tr, max_angle_deg=45),
        ]),
        cache_dir=ds.get("train_cache_dir", ".cache/train"),
        lmdb_dir=ds.get("train_lmdb_dir", None),
        post_transform_fn=_get_aug(eval_mode=False),
        teacher_cacher=teacher_cacher,
        train_mode=True,
        seed=cfg.get("seed", 42),
        n_angles=n_tr,
        max_angle_deg=45,
        cache_angle_idxs=ds.get("cache_angle_idxs_train", None),
    )

    # Validation transform without teacher, optional LMDB
    val_tf = LMDBCachedTransform(
        transform_fn=Compose([
            PrepareForNet(size=size, n_angles=n_val, max_angle_deg=45),
        ]),
        cache_dir=ds.get("val_cache_dir", ".cache/val"),
        lmdb_dir=ds.get("val_lmdb_dir", None),
        post_transform_fn=_get_aug(eval_mode=True),
        teacher_cacher=None,
        train_mode=False,
        seed=cfg.get("seed", 42),
        n_angles=n_val,
        max_angle_deg=45,
        cache_angle_idxs=ds.get("cache_angle_idxs_val", [0]),
    )

    # Dataset loading
    datasets = _load_datasets(
        ds["train_name"], ds["val_name"], get_cypher(),
        train_transform=tr_tf, valid_transform=val_tf,
        n_angles_train=n_tr, n_angles_val=n_val,
    )

    # DataLoaders
    tr_loader = DataLoader(
        datasets["trainset"], batch_size=bs, shuffle=True, drop_last=True, pin_memory=True,
        num_workers=ds.get("train_workers", 3), prefetch_factor=2,
        collate_fn=collate_fn, persistent_workers=True,
    )
    val_loader = DataLoader(
        datasets["valset"], batch_size=1, shuffle=False, drop_last=False,pin_memory=True,
        num_workers=ds.get("val_workers", 4), prefetch_factor=1,
        collate_fn=collate_fn, persistent_workers=True,
    )
    return tr_loader, val_loader


# ════════════════════════════════════════════════════════════════════════════
# Model + optimiser + scheduler
# ════════════════════════════════════════════════════════════════════════════
# In train_utils.py, replace your existing _freeze_initial with this:

# def _freeze_initial(model: torch.nn.Module):
#     # 1) Freeze everything
#     for p in model.parameters():
#         p.requires_grad = False

#     # 2) Unfreeze entire depth head only
#     for name, p in model.named_parameters():
#         if name.startswith("depth_head.scratch."):
#             p.requires_grad = True

def _freeze_initial(model: torch.nn.Module):
    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze all of depth head
    for name, p in model.named_parameters():
        if name.startswith("depth_head.scratch."):
            p.requires_grad = True

    # 3) Unfreeze global layer norm (used for scaling transformer outputs)
    for name, p in model.named_parameters():
        if name.startswith("pretrained.norm.weight"):
            p.requires_grad = True
import re
import torch
import re
import torch

def _freeze_initial(
    model: torch.nn.Module,
    *,
    min_block: int = 20
):
    """
    Freeze everything except:
      • your core depth_head outputs
      • LayerNorm weight & bias in backbone blocks >= min_block
      • the final pretrained.norm
      • (you can still tweak patch/embed etc. if you like)
    """
    named = dict(model.named_parameters())

    # 1) freeze all
    for p in named.values():
        p.requires_grad = False

    # 2) depth_head outputs
    head_allow = (
        "depth_head.projects.",
        "depth_head.scratch.output_conv1",
        "depth_head.scratch.output_conv2.0",
        "depth_head.scratch.output_conv2.2",
        "depth_head.scratch.head_logstd",
        "depth_head.scratch.head_masklog",
    )
    for name, p in named.items():
        if any(name.startswith(pref) for pref in head_allow):
            p.requires_grad = True

    # 3) LayerNorm in blocks >= min_block
    block_norm_re = re.compile(
        r"^pretrained\.blocks\.(\d+)\.norm[12]\.(weight|bias)$"
    )
    for name, p in named.items():
        m = block_norm_re.match(name)
        if m and int(m.group(1)) >= min_block:
            p.requires_grad = True

    # 4) final pretrained.norm
    for suffix in ("weight", "bias"):
        key = f"pretrained.norm.{suffix}"
        if key in named:
            named[key].requires_grad = True

    # # 6) Unfreeze patch embedding & positional embeddings
    # for emb in ("patch_embed.proj.weight",
    #             "patch_embed.proj.bias",
    #             "pos_embed",
    #             "cls_token"):
    #     key = f"pretrained.{emb}"
    #     if key in named:
    #         named[key].requires_grad = True

     # ── graph cut: detach outputs of blocks 0..min_block-1 ─────────────


# def _layerwise_lr(model: torch.nn.Module, base_lr: float, decay: float = 0.95):
#     """
#     Assign higher learning rates to head layers and norm for fast adaptation,
#     while decaying backbone block rates.
#     """
#     groups = []
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         # LayerNorm: allow rapid recalibration
#         if name.startswith("pretrained.norm."):
#             lr = base_lr * 2.0

#         # Depth head, final conv (output_conv2): most aggressive
#         elif name.startswith("depth_head.scratch.output_conv2"):
#             lr = base_lr * 5.0

#         # Depth head, first conv (output_conv1): strong but less than last layer
#         elif name.startswith("depth_head.scratch.output_conv1"):
#             lr = base_lr * 3.0

#         # Any other head parameters
#         elif name.startswith("depth_head."):
#             lr = base_lr * 2.0

#         # Backbone blocks: layer‐wise decay from start_block down
#         elif name.startswith("pretrained.blocks."):
#             blk = int(name.split(".")[2])
#             lr = base_lr * (decay ** (23 - blk))

#         # Bias terms: usually lower rate
#         elif name.endswith(".bias"):
#             lr = base_lr * 0.5

#         # Default
#         else:
#             lr = base_lr

#         groups.append({"params": [p], "lr": lr})

#     return groups

def _layerwise_lr(model: torch.nn.Module, base_lr: float,
                  decay: float = 0.95,
                  refiner_scale: float = 1.0):      # ← novo arg
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # ─── 1. refiner primeiro ─────────────────────────────
        if name.startswith("depth_head.refiner."):
            lr = base_lr * refiner_scale             # <<< aqui
        # ─── 2. regras existentes ────────────────────────────
        elif name.startswith("pretrained.norm."):
            lr = base_lr * 2.0
        elif name.startswith("depth_head.scratch.output_conv2"):
            lr = base_lr * 5.0
        elif name.startswith("depth_head.scratch.output_conv1"):
            lr = base_lr * 3.0
        elif name.startswith("depth_head."):
            lr = base_lr * 2.0
        elif name.startswith("pretrained.blocks."):
            blk = int(name.split(".")[2])
            lr = base_lr * (decay ** (23 - blk))
        elif name.endswith(".bias"):
            lr = base_lr * 0.5
        else:
            lr = base_lr

        groups.append({"params": [p], "lr": lr})

    return groups

class ResettableScheduler:
    """Cosine-annealing + warm restarts in *micro-steps*."""
    def __init__(self, optim: AdamW, cfg: Dict, steps_per_epoch: int, resetable: bool = True):
        self.resetable = resetable
        self.optim = optim
        self.cfg = cfg
        self.spe = max(1, steps_per_epoch)
        self._microstep = 0
        self._make_sched()

    def _make_sched(self):
        T_0 = self.cfg.get("sgdr_restart_period", 30) * self.spe
        self.base = CosineAnnealingWarmRestarts(
            self.optim, T_0=T_0,
            T_mult=self.cfg.get("sgdr_restart_mult", 2),
            eta_min=float(self.cfg.get("min_lr", 1e-7)),
        )

    def step(self):
        self._microstep += 1
        self.base.step(self._microstep)  # Important!

    def reset(self):
        if self.resetable:
            self._make_sched()
            self._microstep = 0


import warnings, torch

# ─── compile helper ──────────────────────────────────────────────────────────
import warnings, torch

# panorai_models/utils/compile_helper.py
import torch, warnings, os


def maybe_compile(model, *, enabled=True, mode="reduce-overhead", backend="aot_eager"):
    """
    Compila se estiver habilitado; em caso de erro (ou backend MPS instável)
    volta para eager automaticamente.
    """
    if not enabled or not hasattr(torch, "compile"):
        return model

    # Se estiver no MPS + backend padrão = inductor → não arrisque
    if any(p.device.type == "mps" for p in model.parameters()):
        warnings.warn("torch.compile desabilitado em MPS – usando eager.")
        return model

    try:
        
        compiled = torch.compile(model, mode=mode, backend=backend)
        if hasattr(compiled, "train"):
            return compiled
        warnings.warn("torch.compile devolveu callable sem .train – fallback eager.")
    except Exception as exc:
        warnings.warn(f"torch.compile falhou ({exc}); fallback eager.")

    return model

TEACHER_SIZE = 518

def add_depth_noise(depth, σ_pix=0.12, p_salt=1e-3, blur_ksize=3):
    """depth (B,1,H,W) → depth_noisy (same shape)"""
    B, _, H, W = depth.shape
    depth_np = depth.cpu().numpy()
    out = []
    for d in depth_np:
        d = cv2.GaussianBlur(d[0], (blur_ksize, blur_ksize), 0)
        out.append(d)
    depth_blur = torch.from_numpy(np.stack(out, 0)).unsqueeze(1).to(depth.device)

    g = torch.randn_like(depth_blur) * (depth_blur.abs() * σ_pix + 1e-3)
    noisy = depth_blur + g
    if p_salt:
        mask = torch.rand_like(noisy) < p_salt
        noisy = torch.where(mask, torch.rand_like(noisy) * depth_blur.max(), noisy)
    return noisy


import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────── helpers ───────────────────────────────────────
class ConvGNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

class DWConvGNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.GroupNorm(max(1, in_ch // 4), in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(max(1, out_ch // 4), out_ch),
            nn.ReLU(inplace=True),
        )

# ───────────────────────── ARCH #1 : ROBUST ──────────────────────────────
class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, p_drop):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.drop = nn.Dropout2d(p=p_drop)
        self.conv = nn.Sequential(
            ConvGNReLU(out_ch + skip_ch, out_ch),
            ConvGNReLU(out_ch, out_ch),
        )
    def forward(self, x, skip):
        x = self.drop(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))
# refiner_archs.py  (or wherever the classes live)
import math, torch
import torch.nn as nn
import torch.nn.functional as F
# … helpers stay unchanged …


# --------------------------------------------------------------------------
# shared helper – generic Kaiming initialiser
# --------------------------------------------------------------------------
def _kaiming_all(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# refiner_archs.py  ─────────────────────────────────────────────────────────
import math, torch, torch.nn as nn, torch.nn.functional as F
# helpers stay unchanged …  ConvGNReLU, DWConvGNReLU, _UpBlock, _kaiming_all


# ═══════════ 1) ROBUST  ════════════════════════════════════════════════════
class RobustDepthRefiner(nn.Module):
    """Heavy U-Net (≈1.2 M params). 3 heads: Δz, logσ², mask-logits."""
    def __init__(self, in_ch=3, base_ch=32, p_drop=0.2, **kw):
        super().__init__()
        c = base_ch
        # ─ encoder / decoder
        self.e1 = nn.Sequential(ConvGNReLU(in_ch, c),  ConvGNReLU(c, c))
        self.e2 = nn.Sequential(ConvGNReLU(c, c*2, s=2), ConvGNReLU(c*2, c*2))
        self.e3 = nn.Sequential(ConvGNReLU(c*2, c*4, s=2), ConvGNReLU(c*4, c*4))
        self.e4 = nn.Sequential(ConvGNReLU(c*4, c*8, s=2), ConvGNReLU(c*8, c*8))
        self.drop = nn.Dropout2d(p_drop)
        self.d3 = _UpBlock(c*8, c*4, c*4, p_drop)
        self.d2 = _UpBlock(c*4, c*2, c*2, p_drop)
        self.d1 = _UpBlock(c*2, c,   c,   p_drop)
        # ─ heads
        self.head_res   = nn.Conv2d(c, 1, 3, padding=1)   # residual Δz
        self.head_logv  = nn.Conv2d(c, 1, 3, padding=1)   # ½ logσ²
        self.head_mask  = nn.Conv2d(c, 1, 3, padding=1)   # mask **logits**
        self._init()

    def _init(self):
        _kaiming_all(self)
        # specialised init
        nn.init.normal_(self.head_res.weight, 0, 1e-4)
        nn.init.zeros_ (self.head_res.bias)
        nn.init.normal_(self.head_logv.weight, 0, 1e-4)
        nn.init.constant_(self.head_logv.bias, -3)         # σ≈e⁻³
        nn.init.zeros_(self.head_mask.weight)
        nn.init.constant_(self.head_mask.bias, 0.0)        #  p≈0.5 start

    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(s1); s3 = self.e3(s2)
        b  = self.drop(self.e4(s3))
        x  = self.d3(b, s3); x = self.d2(x, s2); x = self.d1(x, s1)
        res   = self.head_res(x)
        logv  = (2 * self.head_logv(x)).clamp_(-8, 4)      # log σ²
        m_logits = self.head_mask(x)                       # raw logits
        return res, logv, m_logits
# ───────────────────────────────────────────────────────────────────────────


# ═══════════ 2) LITE  ═════════════════════════════════════════════════════
class LiteDepthRefiner(nn.Module):
    """Depth-wise U-Net-lite (≈140 k)."""
    def __init__(self, in_ch=3, mid_ch=12, p_drop=0.1, **kw):
        super().__init__()
        c = mid_ch
        self.enc1 = DWConvGNReLU(in_ch, c)
        self.enc2 = DWConvGNReLU(c, c*2, s=2)
        self.bottleneck = nn.Sequential(nn.Dropout2d(p_drop),
                                        DWConvGNReLU(c*2, c*2))
        self.dec1 = DWConvGNReLU(c*2 + c, c)
        self.head = nn.Conv2d(c, 3, 3, 1, 1)       # [Δz, logσ², mask]
        self._init()

    def _init(self):
        _kaiming_all(self)
        nn.init.normal_(self.head.weight, 0, 1e-4)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([0.0, -3.0, .4])

    def forward(self, x):
        s1 = self.enc1(x)
        b  = self.bottleneck(self.enc2(s1))
        u1 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, s1], 1))
        res, raw_logv, m_logits = torch.split(self.head(u1), 1, dim=1)
        logv = F.softplus(raw_logv).clamp(-6, 5)
        return res, logv, m_logits
# ───────────────────────────────────────────────────────────────────────────


# ═══════════ 3) TINY  ═════════════════════════════════════════════════════
class TinyDepthRefiner(nn.Module):
    """Ultra-compact refiner (~10 k)."""
    def __init__(self, in_ch=3, base_ch=8, **kw):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, base_ch, 1, bias=False)
        self.sep1   = DWConvGNReLU(base_ch, base_ch)
        self.sep2   = DWConvGNReLU(base_ch, base_ch)
        self.head   = nn.Conv2d(base_ch, 3, 1)     # [Δz, logσ², mask]
        self._init()

    def _init(self):
        _kaiming_all(self)
        nn.init.normal_(self.head.weight, 0, 1e-4)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([0.0, -3.0, 0.0])

    def forward(self, x):
        x = self.sep2(self.sep1(self.reduce(x)))
        res, raw_logv, m_logits = torch.split(self.head(x), 1, dim=1)
        logv = F.softplus(raw_logv).clamp(-6, 5)
        return res, logv, m_logits
# ───────────────────────────────────────────────────────────────────────────

# ─────────────────────────── factory ─────────────────────────────────────
_ARCHS = {
    "robust": RobustDepthRefiner,
    "lite":   LiteDepthRefiner,
    "tiny":   TinyDepthRefiner,
}

def build_refiner(name: str, **kwargs) -> nn.Module:
    """
    Example YAML:
    refiner:
      arch: robust        # robust | lite | tiny
      in_ch: 3
      base_ch: 32         # only for 'robust'
      p_drop: 0.2         # optional
    """
    name = str(name).lower()
    if name not in _ARCHS:
        raise ValueError(f"Unknown refiner '{name}'. "
                         f"Choose one of {list(_ARCHS.keys())}.")
    return _ARCHS[name](**kwargs)


import types, torch.nn as nn

def attach_extra_heads(model, cfg):
    if not cfg.get('model', {}).get('attach_heads', False):
        return

    scratch = model.depth_head.scratch

    # 1) find the high-res conv in output_conv2
    highres_conv = scratch.output_conv2[0]  # the 128→32 conv
    features = {}
    handle = highres_conv.register_forward_hook(
        lambda m, inp, out: features.setdefault('feat_hr', out)
    )
    model._extra_head_handle = handle

    # 2) attach your new heads on that 32-channel high-res feature
    feat_ch = highres_conv.out_channels  # 32
    scratch.head_logstd  = nn.Conv2d(feat_ch, 1, 3, padding=1, bias=True)
    scratch.head_masklog = nn.Conv2d(feat_ch, 1, 3, padding=1, bias=True)
    nn.init.constant_(scratch.head_logstd.bias, -3.0)
    nn.init.zeros_(scratch.head_masklog.bias)

    # 3) wrap forward to always return 4-tuple
    orig_fwd = model.forward
    def forward_with_heads(self, *args, **kwargs):
        features.clear()
        out = orig_fwd(*args, **kwargs)
        depth = out[0] if isinstance(out, tuple) else out
        feat_hr = features.get('feat_hr', None)
        logstd  = scratch.head_logstd(feat_hr)  if feat_hr is not None else None
        masklog = scratch.head_masklog(feat_hr) if feat_hr is not None else None
        teacher_used = depth.clone()
        return depth, logstd, masklog, teacher_used

    model.forward = types.MethodType(forward_with_heads, model)


def attach_refiner(model: nn.Module, cfg: dict) -> None:
    """
    Step 2: if cfg['model']['attach_refiner']=True, project features + upsampled
    (depth, logstd, masklog) into a small 'refiner', and wrap forward to ALWAYS
    return a 4-tuple: (depth_refined, logv_refined, mask_refined, teacher_used=depth_u).
    """
    if not cfg.get('model', {}).get('attach_refiner', False):
        return

    # ensure extra heads exist
    if not cfg.get('model', {}).get('attach_heads', False):
        attach_extra_heads(model, cfg)

    device = next(model.parameters()).device
    mh     = cfg['model']
    scratch = model.depth_head.scratch

    # 1) connector
    conn_ch = mh['connector_ch']
    feat_ch = scratch.output_conv1.in_channels
    scratch.connector = nn.Conv2d(feat_ch, conn_ch, 1, bias=False)

    # 2) build and attach refiner
    ref_cfg = mh['refiner'].copy()
    arch    = ref_cfg.pop('arch')
    in_ch   = 3 + conn_ch
    scratch.refiner = build_refiner(arch, in_ch=in_ch, **ref_cfg).to(device)

    # 3) freeze backbone; unfreeze heads, connector, refiner
    for p in model.parameters():
        p.requires_grad = False
    for name, sub in scratch.named_children():
        if name in ('head_res', 'head_logstd', 'head_masklog', 'connector', 'refiner'):
            for p in sub.parameters():
                p.requires_grad = True

    # 4) wrap forward
    orig_fwd = model.forward
    def forward_with_refiner(self, rgb, *args, **kwargs):
        # A) coarse outputs: depth, logstd, masklog, feat
        depth, logstd, masklog, feat = orig_fwd(rgb, *args, **kwargs)
        B,_,h,w = depth.shape
        H,W    = rgb.shape[-2:]

        # B) upsample to full resolution
        depth_u  = F.interpolate(depth,  (H,W), mode='bilinear', align_corners=False)
        logstd_u = F.interpolate(logstd, (H,W), mode='bilinear', align_corners=False) if logstd is not None else torch.zeros_like(depth_u)
        mask_u   = F.interpolate(masklog,(H,W), mode='bilinear', align_corners=False) if masklog is not None else torch.zeros_like(depth_u)
        feat_u   = F.interpolate(feat,   (H,W), mode='bilinear', align_corners=False) if feat is not None else None

        # C) project features
        proj = scratch.connector(feat_u) if feat_u is not None else torch.zeros(B, conn_ch, H, W, device=device)

        # D) refine
        inp = torch.cat([depth_u, logstd_u, mask_u, proj], dim=1)
        res, logv_r, mask_r = scratch.refiner(inp)
        depth_r = depth_u + res

        # always return 4-tuple: (refined, logv, mask_r, teacher_used=depth_u)
        return depth_r, logv_r, mask_r, depth_u

    model.forward = types.MethodType(forward_with_refiner, model)
# import types
# import torch.nn.functional as F


# def attach_refiner(
#     model: torch.nn.Module,
#     cfg=None
# ):
#     device = next(model.parameters()).device

#     ref_cfg = cfg["model"]["refiner"]
#     refiner = build_refiner(ref_cfg.pop("arch"), **ref_cfg).to(device)
#     model.depth_head.add_module("refiner", refiner)
#     orig_fwd = model.forward

#     # freeze everything but the refiner
#     for p in model.parameters():
#         p.requires_grad = False
#     for p in refiner.parameters():
#         p.requires_grad = True

#     def forward_with_refine(self, rgb, depth_gt=None, teacher_pred=None):
#         """
#         • warm-up: use GT depth only (no teacher)
#         • normal: use cached teacher_pred if passed, otherwise call orig_fwd once
#         """
#         B, _, H, W = rgb.shape

#         # ─── warm-up branch ────────────────────────────────────────────────
#         if depth_gt is not None:
#             noisy = add_depth_noise(depth_gt)  # (B,1,H,W)
#             gray  = rgb.mean(1, keepdim=True)  # (B,1,H,W)
#             flag  = torch.zeros_like(gray)     # (B,1,H,W)
#             # align to GT shape just in case
#             _,_,h0,w0 = noisy.shape
#             gray = F.interpolate(gray, (h0,w0), mode="bilinear", align_corners=False)
#             flag = F.interpolate(flag, (h0,w0), mode="nearest")
#             residual, log_var = self.depth_head.refiner(torch.cat([noisy, gray, flag], dim=1))
#             if residual.shape[2:] != (H, W):
#                 residual = F.interpolate(residual, (H, W), mode="bilinear", align_corners=False)
#                 log_var   = F.interpolate(log_var,   (H, W), mode="bilinear", align_corners=False)
#             return noisy + residual, log_var, None

#         # ─── normal branch ─────────────────────────────────────────────────
#         # 1) either use cached teacher_pred or run coarse model once
#         if teacher_pred is not None:
#             # print('Skipping coarse inference, teacher_pred was given')
#             # teacher_pred: (B,1,h_t,w_t) coming in
#             z_noisy = teacher_pred
#         else:
#             print('Running inference...,, teacher_pred wasnt given')
#             rgb_coarse = F.interpolate(rgb, (TEACHER_SIZE, TEACHER_SIZE),
#                                     mode="bilinear", align_corners=False)
#             with torch.no_grad():
#                 z_out = orig_fwd(rgb_coarse)            # (B,1,ht,wt) or (B,ht,wt)
#             if z_out.ndim == 3:
#                 z_noisy = z_out.unsqueeze(1)
#             else:
#                 z_noisy = z_out
#         # 2) upsample teacher to match rgb
#         z_noisy_up = F.interpolate(z_noisy, (H, W),
#                                 mode="bilinear", align_corners=False)

#         # 3) prepare refiner inputs
#         gray = rgb.mean(1, keepdim=True)               # (B,1,H,W)
#         ids  = torch.arange(B, device=rgb.device) % 6  
#         flag = torch.where(ids==4, 1.0, torch.where(ids==5, -1.0, 0.0))
#         flag_map = flag.view(B,1,1,1).expand(-1,1,H,W)

#         # 4) align spatial dims in case right now gray/flag != z_noisy_up
#         _,_,h_up,w_up = z_noisy_up.shape
#         gray     = F.interpolate(gray,     (h_up,w_up), mode="bilinear", align_corners=False)
#         flag_map = F.interpolate(flag_map, (h_up,w_up), mode="nearest")

#         # 5) run refiner
#         inp = torch.cat([z_noisy_up, gray, flag_map], dim=1)
#         residual, log_var, mask_pred = self.depth_head.refiner(inp)
#         if residual.shape[2:] != (H, W):
#             residual = F.interpolate(residual, (H, W), mode="bilinear", align_corners=False)
#             log_var   = F.interpolate(log_var,   (H, W), mode="bilinear", align_corners=False)
#             mask_pred   = F.interpolate(mask_pred,   (H, W), mode="bilinear", align_corners=False)

#         return z_noisy_up + residual, log_var, z_noisy_up, mask_pred 

#     model.forward = types.MethodType(forward_with_refine, model)



# ──────────────────────────────────────────────────────────────────────
# Re-write of build_model_optim_sched
# ──────────────────────────────────────────────────────────────────────
# ─── helpers.py (or train_utils.py) ──────────────────────────────────────
from panorai.depth import ModelRegistry
import torch

def build_teacher_model(cfg: dict) -> torch.nn.Module:
    """
    Loads DepthAnything-v2 (or whatever `model_name` points to),
    moves to eval-only mode, freezes grads, and returns it.
    """
    teacher = ModelRegistry.load(
        cfg["model_name"],
        dataset = cfg["pretrained_on_dataset"],
        encoder = cfg["encoder"],
        return_model = True,
    ).eval()

        # checkpoint (antes da compilação!)
    if cfg.get("load_from"):
        ckpt = torch.load(cfg["load_from"], map_location='cpu', weights_only=False)
        target = teacher._orig_mod if hasattr(teacher, "_orig_mod") else teacher
        target.load_state_dict(ckpt["model"], strict=False)  # strict=False ignora chaves extras
        print(f"✅ checkpoint {cfg['load_from']} carregado")

    for p in teacher.parameters():
        p.requires_grad = False

    return teacher

# def build_model_optim_sched(cfg: Dict, trainloader: DataLoader):
#     device = cfg.get("device", "mps")
#     torch.set_float32_matmul_precision("high")

#     # ── load teacher (unchanged) ───────────────────────────────────────────
#     model = ModelRegistry.load(
#         cfg["model_name"],
#         max_depth=cfg.get("max_depth", 20.0),
#         dataset=cfg["pretrained_on_dataset"],
#         encoder=cfg["encoder"],
#         return_model=True,
#     ).to(torch.float32).to(device)

#     # ── optionally attach residual refiner ─────────────────────────────────
#     if cfg.get("refine", False):
#         attach_refiner(model, cfg=cfg)
#         # sanity: at least one param from refiner must be trainable
#         assert any(p.requires_grad for n,p in model.named_parameters()
#                 if n.startswith("depth_head.refiner")), "Refiner ended up frozen!"
#     elif cfg.get("freeze", True):
#         _freeze_initial(model)

#     # ── optional torch.compile (safe because model object is intact) ───────
#     if cfg.get("compile", True):
#         model = maybe_compile(model, mode="reduce-overhead")

#     # ── optimiser & scheduler (frozen params are ignored automatically) ────
#     base_lr = float(cfg.get("lr", 1e-4))
#     ref_scale = float(cfg.get("lr_refiner_scale", 1.0))

#     name_of = {p: n for n, p in model.named_parameters()}

#     optim = AdamW(
#         _layerwise_lr(model, base_lr, refiner_scale=ref_scale),
#         betas=(0.9, 0.999),
#         weight_decay=float(cfg.get("weight_decay", 1e-3)),
#     )

#     steps_per_epoch = len(trainloader) // cfg.get("grad_accum", 1)
#     sched = ResettableScheduler(optim, cfg, steps_per_epoch)

#     # ─── clear weight-decay for refiner only ──────────────
#     for g in optim.param_groups:
#         if any('depth_head.refiner' in name_of[p] for p in g['params']):
#             g['weight_decay'] = 0.0

#     return model, optim, sched, AdamW
class NoOpScheduler:
    """Bypass scheduler that does nothing but keeps the interface intact."""
    def __init__(self, optim: AdamW, cfg: Dict, steps_per_epoch: int):
        self.optim = optim
        self.cfg = cfg
        self.spe = max(1, steps_per_epoch)
        self._microstep = 0

    def step(self):
        """No-op step."""
        self._microstep += 1
        # No scheduler step is performed.

    def reset(self):
        """No-op reset."""
        self._microstep = 0

def build_model_optim_sched(cfg: Dict, trainloader: DataLoader):
    device = cfg.get("device", "mps")
    torch.set_float32_matmul_precision("high")

    # ── load your base DAV2 (ViT-L) model ────────────────────────────────
    model = ModelRegistry.load(
        cfg["model_name"],
        max_depth=cfg.get("max_depth", 20.0),
        dataset=cfg["pretrained_on_dataset"],
        encoder=cfg["encoder"],
        return_model=True,
    ).to(torch.float32).to(device)

    # ── Stage 1: attach coarse heads? ────────────────────────────────────
    # controlled by cfg["model"]["attach_heads"]
    if cfg.get("model", {}).get("attach_heads", False):
        attach_extra_heads(model, cfg)

    # ── Stage 2: attach high-res refiner? ───────────────────────────────
    # controlled by cfg["model"]["attach_refiner"]
    if cfg.get("model", {}).get("attach_refiner", False):
        attach_refiner(model, cfg)
        # sanity check: refiner must have trainable params
        assert any(
            p.requires_grad for n,p in model.named_parameters()
            if "refiner" in n
        ), "Refiner ended up frozen!"
    # ── otherwise: freeze everything except any heads you did attach ─────
    else:
        _freeze_initial(model)

    # ── optional torch.compile (no problem since we've only patched on top) ─
    if cfg.get("compile", True):
        model = maybe_compile(model, mode="reduce-overhead")

    # ── optimizer & scheduler setup ──────────────────────────────────────
    base_lr   = float(cfg.get("lr", 1e-4))
    ref_scale = float(cfg.get("lr_refiner_scale", 1.0))
    name_of   = {p: n for n, p in model.named_parameters()}

    optim = AdamW(
        _layerwise_lr(model, base_lr, refiner_scale=ref_scale),
        betas=(0.9, 0.999),
        weight_decay=float(cfg.get("weight_decay", 1e-3)),
    )

    steps_per_epoch = len(trainloader) // cfg.get("grad_accum", 1)
    sched = ResettableScheduler(optim, cfg, steps_per_epoch, resetable=True)
    # sched = NoOpScheduler(optim, cfg, steps_per_epoch)

    

    # ── make sure any refiner group has zero weight-decay ───────────────
    for g in optim.param_groups:
        if any("refiner" in name_of[p] for p in g["params"]):
            g["weight_decay"] = 0.0

    return model, optim, sched, AdamW

# # ════════════════════════════════════════════════════════════════════════════
# # Loss & utilities
# # ════════════════════════════════════════════════════════════════════════════
# def build_loss_fn(cfg: Dict) -> FixedDepthLoss:
#     lc = cfg.get("loss", {})

#     return FixedDepthLoss(
#         # ------------ scalar weights -----------------------------------
#         silog_weight   = lc.get("silog_weight",   1.0),
#         l1_weight      = lc.get("l1_weight",      1.0),
#         grad_weight    = lc.get("grad_weight",    1.0),   # <─ was normal_weight
#         normal_weight  = lc.get("normal_weight",  1.0),
#         smooth_weight  = lc.get("smooth_weight",  1.0),
#         thin_weight    = lc.get("thin_weight",    0.0),
#         nll_weight     = lc.get("nll_weight",    0.1),

#         # ------------ loss toggles -------------------------------------
#         use_l1_loss        = lc.get("use_l1_loss",        True),
#         use_silog_loss     = lc.get("use_silog_loss",     False),
#         use_normal_loss    = lc.get("use_normal_loss",    True),
#         use_gradient       = lc.get("use_gradient",       False),
#         use_smoothness     = lc.get("use_smoothness",     False),
#         use_thin_structure = lc.get("use_thin_structure", False),
#         use_l1_attention   = lc.get("use_l1_attention",   False),
#         use_disparity      = lc.get("use_disparity",      False),

#         # ------------ NEW features -------------------------------------
#         hetero          = lc.get("hetero",          False),  # log-var / NLL
#         silog_lambda    = lc.get("silog_lambda",    0.5),    # keep existing
#         residual_reg_weight = lc.get("residual_reg_weight",      .02),
#         residual_gate_thresh = lc.get("rresidual_gate_thresh",      .2),
#     )

# ════════════════════════════════════════════════════════════════════════════
# Loss & utilities
# ════════════════════════════════════════════════════════════════════════════
def build_loss_fn(cfg: Dict) -> FixedDepthLoss:
    lc = cfg.get("loss", {})

    # helper → zero out if the flag is False, otherwise take the weight
    def _w(key_weight, key_flag, default):
        return 0.0 if not lc.get(key_flag, True) else lc.get(key_weight, default)
    
    loss_fn = FixedDepthLoss(
        # ─── primary weights (flags control zeros) ───
        silog_weight        = _w("silog_weight",  "use_silog_loss",     1.0),
        grad_weight         = _w("grad_weight",   "use_gradient",       0.1),
        smooth_weight       = _w("smooth_weight", "use_smoothness",     0.1),
        thin_weight         = _w("thin_weight",   "use_thin_structure", 0.0),
        normal_weight       = _w("normal_weight", "use_normal_loss",    0.0),
        l1_weight           = _w("l1_weight",     "use_l1_loss",        0.0),

        # ─── L1 extras ───────────────────────────────
        l1_3d_mode          = lc.get("l1_3d_mode",      "none"),
        use_smooth_l1       = lc.get("use_smooth_l1",   True),

        # ─── SiLog & char-b parameters ───────────────
        silog_lambda        = lc.get("silog_lambda",    0.5),

        # ─── gradient / thin hyperparams ────────────
        k_edge_grad         = lc.get("k_edge_grad",     3.0),
        thin_pct            = lc.get("thin_pct",        0.70),

        # ─── hetero & NLL ────────────────────────────
        hetero               = lc.get("hetero",         False),
        nll_weight           = lc.get("nll_weight",     1.0),
        nll_ramp_epochs      = lc.get("nll_ramp_epochs",5),

        # ─── residual / refiner reg ─────────────────
        residual_reg_weight  = lc.get("residual_reg_weight",0.02),
        residual_tau_mode    = lc.get("residual_tau_mode",  "adaptive"),

        # ─── mask‐BCE head ───────────────────────────
        use_mask_bce         = lc.get("use_mask_bce",    False),
        mask_weight          = lc.get("mask_weight",     0.2),

        # ─── 3D‐L1 weighting ─────────────────────────
        sixth_weight         = lc.get("sixth_weight",    0.2),

        # ─── scheduling (warm‐up/ramp) ───────────────
        schedule_keys        = lc.get("schedule_keys",   ["thin_w", "normal_w"]),
        warmup_epochs        = lc.get("warmup_epochs",   3),
        ramp_epochs          = lc.get("ramp_epochs",     0),
    )
    
    try:
        return torch.jit.script(loss_fn)
    except Exception as e:
        print(['[jit compile failed]: {e}'])
        return loss_fn

# def build_loss_fn(cfg: Dict) -> FixedDepthLoss:
#     lc = cfg.get("loss", {})

#     # helper → weight = 0 if component switched off
#     def _w(key_weight, key_flag, default):
#         return 0.0 if not lc.get(key_flag, True) else lc.get(key_weight, default)

#     return FixedDepthLoss(
#         # ─── primary weights (flags control zeros) ───
#         silog_weight  = _w("silog_weight",  "use_silog_loss",  1.0),
#         l1_weight     = _w("l1_weight",     "use_l1_loss",     1.0),
#         grad_weight   = _w("grad_weight",   "use_gradient",    0.1),
#         smooth_weight = _w("smooth_weight", "use_smoothness",  0.1),
#         thin_weight   = _w("thin_weight",   "use_thin_structure", 0.0),
#         normal_weight = _w("normal_weight", "use_normal_loss", 1.0),

#         # ─── heteroscedastic & residual stuff ───
#         hetero                = lc.get("hetero", False),
#         nll_weight            = lc.get("nll_weight", 0.5),
#         nll_ramp_epochs       = lc.get("nll_ramp_epochs", 5),
#         residual_reg_weight   = lc.get("residual_reg_weight", 0.0),
#         residual_tau_mode     = "fixed",   # kept simple; change if you need adaptive
#         k_edge_grad           = 3.0,
#         thin_pct              = 0.70,

#         # ─── L1 extras ───
#         l1_3d_mode      = lc.get("l1_3d_mode", "log"),
#         sixth_weight    = lc.get("sixth_weight", 0.2),
#         use_smooth_l1   = lc.get("use_smooth_l1", True),

#         # ─── mask term ───
#         use_mask_bce = lc.get("use_mask_bce", False),
#         mask_weight  = lc.get("mask_weight", 0.2),
#     )
def search_best_scale(model, trainloader: DataLoader, cfg: Dict) -> float:
    return estimate_best_scale(
        model, trainloader,
        device=cfg.get("device", "mps"),
        max_depth=cfg.get("max_depth", 80.0),
        robust=True, plot=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# TransformerUnfreezeScheduler
# ════════════════════════════════════════════════════════════════════════════
# class TransformerUnfreezeScheduler:
#     """Validation/gradient-driven unfreeze with boom-log."""

#     def __init__(
#         self,
#         model: torch.nn.Module,
#         *,
#         start_block: int = 23,
#         min_block: int = 17,
#         grad_threshold: float = 1e-3,
#         patience: int = 10,
#         warmup_epochs: int = 5,
#         min_epochs_per_unfreeze: int = 5,
#         log_path: str | Path = "unfreeze_log.jsonl",
#         logger=print,
#     ) -> None:
#         self.m = model
#         self.sblk = start_block
#         self.cur_min = start_block
#         self.min_blk = min_block
#         self.grad_thr = grad_threshold
#         self.patience = patience
#         self.warmup = warmup_epochs
#         self.min_epochs = min_epochs_per_unfreeze
#         self.best_val = float("inf")
#         self.wait = 0
#         self.epoch = 0
#         self.last_unfreeze_epoch = 0
#         self.log = logger
#         self.log_path = Path(log_path)
#         self.log_path.parent.mkdir(parents=True, exist_ok=True)

#     # ——— trainer hooks ———
#     def update(self, *, epoch: int, model, val_logs: Dict, **_):
#         self.epoch = epoch
#         self._step_validation(val_logs.get("total", float("inf")))
#         self._step_gradients(model)

#     def step_gradients(self, model):
#         self._step_gradients(model)

#     # ——— internal ———
#     def _can_unfreeze(self):
#         return (self.epoch - self.last_unfreeze_epoch) >= self.min_epochs and self.cur_min > self.min_blk

#     def _step_validation(self, val_loss: float):
#         if self.epoch < self.warmup or not np.isfinite(val_loss):
#             return
#         if val_loss < self.best_val - 1e-4:
#             self.best_val, self.wait = val_loss, 0
#         else:
#             self.wait += 1
#         if self.wait >= self.patience and self._can_unfreeze():
#             self.cur_min -= 1
#             self.wait = 0
#             self._unfreeze_range(self.cur_min, self.sblk, "val_plateau")

#     def _step_gradients(self, model):
#         if not self._can_unfreeze():
#             return
#         blk = f"pretrained.blocks.{self.cur_min}"
#         grads = [p.grad.norm().item() for n, p in model.named_parameters()
#                  if n.startswith(blk) and p.grad is not None]
#         if grads and max(grads) < self.grad_thr:
#             self.cur_min -= 1
#             self._unfreeze_range(self.cur_min, self.sblk, "weak_grad")

#     def _unfreeze_range(self, mn: int, mx: int, reason: str):
#         if hasattr(self, "_rebuild"):
#             self._rebuild()  
# #        also reset the LR schedule so unfrozen blocks get a fresh cycle
#         try:
#             self._rebuild_trainer.scheduler.reset()
#         except Exception:
#             pass

#         names: List[str] = []
#         for n, p in self.m.named_parameters():
#             if any(n.startswith(f"pretrained.blocks.{i}") for i in range(mn, mx + 1)) and not p.requires_grad:
#                 p.requires_grad = True
#                 names.append(n)
#         if not names:
#             return
#         self.last_unfreeze_epoch = self.epoch
#         self._boom_log(names, reason, mn, mx)

#     def _boom_log(self, names: List[str], reason: str, mn: int, mx: int):
#         msg = (f"🔓 epoch={self.epoch} | blocks {mn}–{mx} | "
#                f"layers={len(names)} | reason={reason}")
#         self.log(msg)
#         evt = {
#             "epoch": self.epoch,
#             "reason": reason,
#             "blocks": [mn, mx],
#             "layers": names,
#             "time": time.strftime("%Y-%m-%d %H:%M:%S"),
#         }
#         with self.log_path.open("a") as fp:
#             fp.write(json.dumps(evt) + "\n")

#     def set_logger(self, logger_fn):
#         self.log = logger_fn

#     def set_rebuild_callback(self, fn):
#         self._rebuild = fn

class TransformerUnfreezeScheduler:
    """Validation/gradient-driven unfreeze with boom-log, one group per epoch."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        start_block: int = 23,
        min_block: int = 17,
        grad_threshold: float = 1e-3,
        patience: int = 10,
        warmup_epochs: int = 5,
        min_epochs_per_unfreeze: int = 5,
        log_path: str | Path = "unfreeze_log.jsonl",
        logger=print,
    ) -> None:
        self.m = model
        self.grad_thr = grad_threshold
        self.patience = patience
        self.warmup = warmup_epochs
        self.min_epochs = min_epochs_per_unfreeze
        self.best_val = float("inf")
        self.wait = 0
        self.epoch = 0
        self.last_unfreeze_epoch = -1  # allow unfreezing on epoch 0 if ready
        self.log = logger
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # ─── fresh start: delete any existing log from previous runs ───
        if self.log_path.exists():
            self.log_path.unlink()

        # Define unfreeze groups: norm, head_conv2, head_conv1, then backbone blocks
        self.unfreeze_groups: List[Dict[str, List[str]]] = []
        self.unfreeze_groups.append({"name": "norm",       "prefixes": ["pretrained.norm."]})
        self.unfreeze_groups.append({"name": "head_conv2", "prefixes": ["depth_head.scratch.output_conv2."]})
        self.unfreeze_groups.append({"name": "head_conv1", "prefixes": ["depth_head.scratch.output_conv1"]})
        for b in range(start_block, min_block - 1, -1):
            self.unfreeze_groups.append({
                "name": f"block{b}",
                "prefixes": [f"pretrained.blocks.{b}."]
            })

        self.next_group = 0
        self._rebuild = None

    def update(self, *, epoch: int, model, val_logs: Dict, **_):
        self.epoch = epoch
        self._step_validation(val_logs.get("total", float("inf")))
        self._step_gradients(model)

    def step_gradients(self, model):
        self._step_gradients(model)

    def _can_unfreeze(self) -> bool:
        return (
            self.next_group < len(self.unfreeze_groups)
            and (self.epoch - self.last_unfreeze_epoch) >= self.min_epochs
            and self.epoch >= self.warmup
        )

    def _step_validation(self, val_loss: float) -> None:
        if self.epoch < self.warmup or not np.isfinite(val_loss):
            return
        if val_loss < self.best_val - 1e-4:
            self.best_val, self.wait = val_loss, 0
        else:
            self.wait += 1
        if self.wait >= self.patience and self._can_unfreeze():
            if self._unfreeze_next_group("val_plateau"):
                self.wait = 0

    def _step_gradients(self, model) -> None:
        if not self._can_unfreeze():
            return
        grp = self.unfreeze_groups[self.next_group]
        grads = [
            p.grad.norm().item()
            for n, p in model.named_parameters()
            if any(n.startswith(pref) for pref in grp["prefixes"]) and p.grad is not None
        ]
        if grads and max(grads) < self.grad_thr:
            self._unfreeze_next_group("weak_grad")

    def _unfreeze_next_group(self, reason: str) -> bool:
        if self.next_group >= len(self.unfreeze_groups):
            return False

        grp = self.unfreeze_groups[self.next_group]

        # Rebuild optimizer/scheduler if needed
        if self._rebuild:
            self._rebuild()
            try:
                trainer = self._rebuild.__self__
                trainer.scheduler.reset()
            except Exception:
                pass

        names = []
        for n, p in self.m.named_parameters():
            if any(n.startswith(pref) for pref in grp["prefixes"]) and not p.requires_grad:
                p.requires_grad = True
                names.append(n)

        if not names:
            self.next_group += 1
            return False

        if self.last_unfreeze_epoch == self.epoch:
            self.log(f"⚠️ Already unfrozen a group in epoch={self.epoch}, skipping.")
            return False

        self.last_unfreeze_epoch = self.epoch
        self._boom_log(names, reason, grp["name"])
        self.next_group += 1
        return True

    def _boom_log(self, names: List[str], reason: str, group: str) -> None:
        msg = f"🔓 epoch={self.epoch} | group={group} | layers={len(names)} | reason={reason}"
        self.log(msg)
        evt = {
            "epoch": self.epoch,
            "group": group,
            "reason": reason,
            "layers": names,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self.log_path.open("a") as fp:
            fp.write(json.dumps(evt) + "\n")

    def set_logger(self, logger_fn) -> None:
        self.log = logger_fn

    def set_rebuild_callback(self, fn) -> None:
        self._rebuild = fn

# EOF




class EMAAdaptiveClipper:
    def __init__(self, decay=0.99, max_factor=2.0, min_scale=0.25):
        self.decay = decay
        self.max_factor = max_factor
        self.min_scale = min_scale
        self.ema_norms = {}  # name → running norm


    def step(self, model: torch.nn.Module, verbose=False):
        clipped = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            gnorm = param.grad.norm().item()
            ema = self.ema_norms.get(name, gnorm)
            ema = self.decay * ema + (1 - self.decay) * gnorm
            self.ema_norms[name] = ema

            if gnorm > self.max_factor * ema:
                ratio = gnorm / ema
                scale = 1 / (1 + (ratio - 1) ** 2)
                scale = max(scale, self.min_scale)
                param.grad.mul_(scale)
                clipped.append((name, gnorm, ema, scale))

        if verbose and clipped:
            print("[EMA Adaptive Clipping Summary]")
            for name, g, ema, s in clipped:
                print(f"  [clip] {name:40s} grad={g:.2f} ema={ema:.2f} scale=×{s:.3f}")





# EOF