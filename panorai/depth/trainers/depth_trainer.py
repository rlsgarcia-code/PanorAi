"""Panorai DepthTrainer — Modular Version (2025-05-09)
=======================================================
✓ Implements unfreezing, adaptive scaling, and detailed logging.
✓ Now modularized for easier maintenance and extensibility.
"""

from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from .metrics import MonocularDepthMetrics
from panorai.depth.training.train_utils import maybe_compile

# ──────────────── Utility Classes ────────────────

class RunningAverage:
    def __init__(self): self.reset()
    def update(self, v: float, n: int = 1): self.t += v * n; self.c += n
    def mean(self) -> float: return self.t / max(self.c, 1)
    def reset(self): self.t, self.c = 0.0, 0


class RunningLossLogger:
    def __init__(self): self.logs: Dict[str, RunningAverage] = {}
    def update(self, d: Dict[str, float], n=1):
        for k, v in d.items():
            self.logs.setdefault(k, RunningAverage()).update(v, n)
    def compute(self) -> Dict[str, float]:
        return {k: rv.mean() for k, rv in self.logs.items()}
    def reset(self): self.logs.clear()

import matplotlib.pyplot as plt
import torch

def  unnormalize_image(x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        return x * std + mean

def debug_depth_triplet(rgb, pred, gt, mask=None, idx: int = 0,
                        cmap="viridis", save_path: str | None = None,
                        title_prefix: str = ""):
    """
    Show input RGB, predicted depth, and target depth for a single sample.
    
    Args
    ----
    rgb, pred, gt : tensors shaped (B, C, H, W) – pred/gt in metres
    mask          : (B, 1, H, W) boolean / 0-1 – optional valid-pixel mask
    idx           : which sample in the batch to visualise
    cmap          : colormap for depth
    save_path     : if given, figure is saved instead of shown
    title_prefix  : optional string before each subplot title
    """
    assert rgb.ndim == 4 and pred.shape == gt.shape[:3] + pred.shape[3:], \
        "Shapes must be (B,C,H,W)"

    # pick sample & move to CPU
    img_np  = unnormalize_image(rgb[idx:idx+1]).squeeze(0).permute(1,2,0).cpu().numpy()
    pred_np = pred[idx,0].detach().cpu().numpy()
    gt_np   =  gt[idx,0].detach().cpu().numpy()
    
    if mask is not None:
        mask_np = mask[idx,0].bool().cpu().numpy()
        pred_np = np.where(mask_np, pred_np, np.nan)
        gt_np   = np.where(mask_np,   gt_np, np.nan)

    vmin = np.nanpercentile(gt_np,  1)
    vmax = np.nanpercentile(gt_np, 99)

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(img_np.clip(0,1))
    ax[0].set_title(f"{title_prefix}RGB")
    im1 = ax[1].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title(f"{title_prefix}Pred depth")
    im2 = ax[2].imshow(gt_np,   cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title(f"{title_prefix}GT depth")

    for a in ax: a.axis("off")
    fig.colorbar(im2, ax=ax.ravel().tolist(), shrink=0.6, pad=0.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def check_refiner_devices(model):
    print("\n[Device check: refiner]")
    for n, p in model.named_parameters():
        if "refiner" in n:
            print(f"{n:<60} {str(p.device)}")

# ──────────────── Logging & Formatting ────────────────

def _metrics_to_str(m: Dict[str, float]) -> str:
    keys = {
        "MAE": "MAE", "RMSE": "RMSE",
        "Threshold Accuracy (δ=1.25)": "δ=1.25",
        "Threshold Accuracy (δ=1.25^2)": "δ=1.25²",
        "Threshold Accuracy (δ=1.25^3)": "δ=1.25³",
        "Structural Similarity (SSIM)": "SSIM"
    }
    return " ".join(f"{keys[k]}={m[k]:.3f}" for k in keys if k in m)


def _top_grad_norms(m: torch.nn.Module, k=10):
    pairs = [(n, p.grad.norm().item()) for n, p in m.named_parameters() if p.grad is not None]
    pairs.sort(key=lambda x: x[1], reverse=True)
    print("\n[Top Grad Norms]")
    for n, v in pairs[:k]:
        print(f"  - {n:<50} {v:.4f}")
    return pairs[:k]


def to_float(x):  # JSON fix for numpy scalars
    return float(x) if isinstance(x, (np.floating, np.integer)) else x


def log_top_parameter_updates(model, top_k=10):
    updates = compute_update_magnitudes(model)
    if updates:
        print("\n[Top Parameter Updates]")
        for name, delta in updates[:top_k]:
            print(f"  - {name:50} Δ={delta:.6f}")


def print_trainable_params(model: torch.nn.Module):
    print("\n[Trainable Parameters]")
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            print(f"  - {name:<60} {count:>8,d}")
    print(f"Total Trainable Parameters: {total:,d}")

import time
from typing import Dict
from contextlib import contextmanager

class TimeProfiler:
    """
    Utility to profile named segments using context managers.
    Usage:
        profiler = TimeProfiler()
        with profiler.time('data'):
            ...
        print(profiler.times)
    """
    def __init__(self):
        self.times: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str):
        t0 = time.time()
        yield
        self.times[name] = time.time() - t0

# ──────────────── Gradient Handling ────────────────


def _max_norm(name: str, default: float = 5.0) -> float:
    if "bias" in name: 
        return 1.0
    if "norm" in name: 
        return 2.0
    return default


def adaptive_grad_scaling(model, max_norm=5.0, verbose=False):
    """Scales gradients softly if they exceed a layer-specific max_norm."""
    clipped = []

    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.norm()
        m = _max_norm(n, max_norm)

        if g > m:
            # Smart: only scale significantly if g >> m
            if g < 2 * m:
                # Linear soft clip: scale = 1.0 at g=m, scale→m/g at g=2m
                scale = 1.0 - ((g - m) / m) * (1.0 - (m / g))
            else:
                scale = m / (g + 1e-6)

            p.grad.mul_(scale)
            clipped.append((n, g.item(), scale))

    if verbose and clipped:
        print("[Adaptive Clipping Summary]")
        for n, g, scale in clipped:
            print(f"  [clip] {n:40s} grad={g:.2f} scale=×{scale:.3f}")


def compute_update_magnitudes(model: torch.nn.Module) -> List[Tuple[str, float]]:
    updates = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if hasattr(param, "_prev"):
                delta = (param.data - param._prev).norm().item()
                updates.append((name, delta))
            param._prev = param.data.clone().detach()  # store for next step
    return sorted(updates, key=lambda x: x[1], reverse=True)


# ─── helpers.py (or depth_trainer.py) ──────────────────────────────
# ─── WandB depth-panel helper (fixed) ───────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def _unnormalize_image(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return x * std + mean

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb


def log_depth_panel(
        rgb_batch: torch.Tensor,      # (B,3,H,W)  – torch, still normalised
        depth_gt  : torch.Tensor,      # (B,1,H,W)
        depth_pred: torch.Tensor,      # (B,1,H,W)
        step      : int,
        max_depth : float = 80.,
        vmin_err  : float = 0.,
        vmax_err  : float = 2.,
        sel_idx   : int   = 0,         # which item from the batch to draw
        tag       : str   = "vis/panel",
        cmap_depth: str   = "viridis",
        cmap_err  : str   = "plasma"
):
    """
    Logs a 2×2 panel to Weights & Biases:

        ┌───────────┬────────────┐
        │ RGB       │ GT depth   │
        ├───────────┼────────────┤
        │ Pred depth│ |Pred-GT|  │
        └───────────┴────────────┘

    * Error panel shows only **valid** pixels (0.001 < z < max_depth).
    * Error colour-bar is clamped to [0, 2] metres by default.
    """
    # ------------------------------------------------------------------ helpers
    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def _unnorm(x: torch.Tensor) -> np.ndarray:   # (3,H,W) → (H,W,3)
        x = x.cpu() * STD + MEAN
        return np.clip(x.permute(1, 2, 0).numpy(), 0., 1.)

    # pick sample -------------------------------------------------------------
    # pick sample -------------------------------------------------------------
    rgb   = rgb_batch[sel_idx].detach()                # (3,H,W) – still torch
    gt    = depth_gt [sel_idx, 0].detach().cpu().numpy()
    pred  = depth_pred[sel_idx, 0].detach().cpu().numpy()

    # valid-pixel mask --------------------------------------------------------
    mask  = (gt > 1e-3) & (gt < max_depth)
    err   = np.abs(pred - gt)
    err_masked = np.where(mask, err, np.nan)       # for transparency

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(_unnorm(rgb));           ax[0, 0].set_title("RGB")
    im1 = ax[0, 1].imshow(gt,   cmap=cmap_depth, vmin=0, vmax=max_depth)
    ax[0, 1].set_title("GT depth")
    im2 = ax[1, 0].imshow(pred, cmap=cmap_depth, vmin=0, vmax=max_depth)
    ax[1, 0].set_title("Pred depth")
    im3 = ax[1, 1].imshow(err_masked, cmap=cmap_err, vmin=vmin_err, vmax=vmax_err)
    ax[1, 1].set_title("|Pred − GT|")

    for a in ax.ravel():
        a.axis("off")

    # colour-bars (same height)
    for col, im in zip((0, 1), (im1, im3)):
        cb = fig.colorbar(im, ax=ax[:, col], fraction=0.047, pad=0.01)
        cb.ax.tick_params(labelsize=8)

    plt.tight_layout()

    # ------------------------------------------------------------------ wandb
    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)


# ----------------------------------------------------------------------
# helper: (B,3,H,W) → (B*H*W, 3) colour array in [0,1]
# ----------------------------------------------------------------------
def _tensor2rgb(t: torch.Tensor) -> np.ndarray:
    """
    (3,H,W) float tensor in range [0,1]  →  (H,W,3) uint8 RGB np-array
    Works on CUDA / MPS by detaching → cpu → numpy.
    """
    t = t.detach().cpu()          #  ←  this line fixes the crash
    arr = t.clamp_(0, 1).mul_(255).byte().permute(1, 2, 0).numpy()
    return arr


# ----------------------------------------------------------------------
# wandb 3-D point-cloud logger
# ----------------------------------------------------------------------
# utils/vis.py  (or wherever you keep helpers)
import io, json, wandb, numpy as np, torch
from panorai import GnomonicFace              # ⇦ new import

import open3d as o3d

# ------------------------------------------------------------------------- #
# depth  ➜  coloured point-cloud  ➜  W&B Object3D
# ------------------------------------------------------------------------- #
@torch.no_grad()
def log_depth_pointcloud(
    rgb_bchw   : torch.Tensor,       # (B,3,H,W) *normalised*
    depth_bch1 : torch.Tensor,       # (B,1,H,W) metres
    step       : int,
    max_depth  : float = 80.0,
    grad_thr   : float = 10.0,
    sixth_w    : float = 0.2,        # not used here – kept for interface parity
):
    """
    • takes the FIRST item in the batch
    • masks out depth ∉ (1 cm … max_depth)
    • converts to point-cloud via panorai.faces.GnomonicFace
    • logs to W&B as “pcd/step_x” so you get one tab with all clouds
    """
    # ─── move ↦ CPU, detach, squeeze ────────────────────────────────────
    rgb   = rgb_bchw[0].detach().cpu()          # (3,H,W)
    depth = depth_bch1[0,0].detach().cpu()      # (H,W)

    # ─── build validity mask (inside trainer you already have 'valid') ─
    valid = (depth > 1e-3) & (depth < max_depth)

    if valid.sum() == 0:
        print(f"[log_depth_pointcloud] step={step} – no valid pixels, skip.")
        return None

    # ─── un-normalise & to HWC uint8 ────────────────────────────────────
    rgb_np = _unnormalize_image(rgb).permute(1,2,0).numpy()   # (H,W,3) in [0,1]
    rgb_np = (rgb_np * 255).astype(np.uint8)

    # ─── panorai: rectilinear → metric point-cloud ──────────────────────
    face = GnomonicFace(rgb_np, lat=0, lon=0, fov=90)
    pcd  = face.to_pcd(depth.numpy(), grad_threshold=grad_thr,
                       max_radius=max_depth, mask=valid.numpy())

    # colour by RGB (already copied inside to_pcd, but we can be explicit)
    pcd.o3d.colors = o3d.utility.Vector3dVector(
        rgb_np.reshape(-1,3)[valid.flatten()] / 255.0
    )

    # ─── W&B Object3D serialisation -------------------------------------
    buf = io.BytesIO()
    o3d.io.write_point_cloud(buf, pcd.o3d, write_ascii=True)  # → PLY in-memory
    obj3d = wandb.Object3D(json.loads(buf.getvalue().decode()))

    wandb.log({f"pcd/step_{step:06d}": obj3d}, step=step)

    return pcd  # handy if you also want local o3d.visualisation.draw_geometries()
# ──────────────── DepthTrainer Class ────────────────
from panorai.depth.training.train_utils import EMAAdaptiveClipper, add_depth_noise
import wandb

class DepthTrainer:
    def __init__(self, model: torch.nn.Module, trainloader, valloader, *,
                 max_depth: float, loss_fn, device: str = "mps", grad_accum: int = 1,
                 debug: bool = False, best_scale: Optional[float] = None,
                 num_warmup_epochs: int = 0, adaptive_scaling: bool = False,
                 grad_clip: float = 5.0, verbose: bool = True,
                 compile_model: bool = True,
                 noise_warmup_epochs: int = 0,
                 compute_metrics: bool = False):
        self.compute_metrics = compute_metrics 
        print(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        self.acc = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="fp16" , log_with="wandb" )
        self.acc.init_trackers("PanoraiDepth", {"wandb": {"project": "Panorai-Depth-Training"}})

        self.print = self.acc.print
        self.device = self.acc.device
        self.verbose = verbose

        if compile_model:
            model = maybe_compile(model, mode="reduce-overhead")

        model = model.to(self.device)
        
        model.eval()
        
        example = torch.randn(
            1, 3,
            700,  # e.g. 518
            700,
            device=self.device
        )
        # trace and switch back to train mode
        self.model = torch.jit.trace(model, example).train()

        self._original_model = model
        self.trainloader, self.valloader = trainloader, valloader
        self.loss_fn, self.max_depth = loss_fn, max_depth

        self.grad_accum = grad_accum
        self.grad_clip = grad_clip
        self.debug = debug
        self.cur_epoch = 0
        self.best_val = math.inf
        self.best_scale = best_scale or 1.0
        self.adaptive_scaling = adaptive_scaling
        self.num_warmup_epochs = num_warmup_epochs

        self.optimizer, self.scheduler = None, None
        self._lr_getter = None
        self.unfreeze_scheduler = None

        self.lr_log: List[float] = []
        self.train_metrics, self.eval_metrics = [], []
        self.metrics_log: List[Dict[str, Any]] = []

        self.adaptive_clipper = EMAAdaptiveClipper(
            decay=0.99, max_factor=3.0, min_scale=0.5
        )
        self.noise_warmup_epochs = noise_warmup_epochs
        self.global_step=0


    def set_accelerator(self, optimizer, scheduler=None, lr_getter=None, base_optim_class=None):
        self._lr_getter = lr_getter
        self._base_optim_class = base_optim_class
        objects = [self.model, optimizer, self.trainloader, self.valloader]
        if scheduler: objects.append(scheduler)
        prepped = self.acc.prepare(*objects)
        if scheduler:
            self.model, self.optimizer, self.trainloader, self.valloader, self.scheduler = prepped
        else:
            self.model, self.optimizer, self.trainloader, self.valloader = prepped
        wandb.watch(self.model, log="all", log_freq=100)

    def set_unfreeze_scheduler(self, sched):
        self.unfreeze_scheduler = sched
        sched.set_logger(self.print)
        sched.set_rebuild_callback(self.rebuild_optimizer)

    # def _forward_step(self, img, depth, teacher_pred=None):
    #     """
    #     • warm-up  → model(img, depth)                   returns (pred, log_var, None)
    #     • normal   → model(img, None, teacher_pred)      returns (pred, log_var, z_noisy_up)
    #     • fallback → model(img) (if neither depth_gt nor teacher_pred)
    #     """
    #     # --- ensure channel dims ---
    #     if depth.ndim == 3:
    #         depth = depth.unsqueeze(1)

    #     # --- decide which forward to call ---
    #     out = self.model(img)
        
    #     # unify into exactly 4 outputs:
    #     if not isinstance(out, tuple):
    #         out = (out, None, None, None)
    #     elif len(out) == 1:
    #         out = (out[0], None, None, None)
    #     elif len(out) == 2:
    #         out = (out[0], out[1], None, None)
    #     elif len(out) == 3:
    #         out = (out[0], out[1], out[2], None)
    #     # else len(out) == 4:
    #     #     continue
    #     pred, log_var, mask_pred, teacher_used = out

    #     # --- ensure pred channel ---
    #     if pred.ndim == 3:
    #         pred = pred.unsqueeze(1)

    #     # --- valid mask ---
    #     valid = (
    #         (depth >= 1e-3) & (depth <= self.max_depth) &
    #         torch.isfinite(depth) & torch.isfinite(pred)
    #     )

    #     # --- optional adaptive scaling ---
    #     scale = self.best_scale
    #     if self.adaptive_scaling:
    #         pv, dv = pred[valid], depth[valid]
    #         if pv.numel() > 0:
    #             idx = torch.randperm(pv.numel(), device=pv.device)[:50_000]
    #             scale = (pv[idx] * dv[idx]).mean() / (pv[idx] ** 2).mean()
    #     alpha = min(self.cur_epoch / max(self.num_warmup_epochs, 1), 1.0)
    #     pred = (1 - alpha) * (scale * pred) + alpha * pred

    #     # after a forward pass inside training/validation loop
    #     # debug_depth_triplet(img, pred, depth,
    #     #                     mask=depth>.01, idx=0,
    #     #                     title_prefix=f"Epoch{self.cur_epoch}_")

    #     # --- compute loss ---
    #     loss, logs = self.loss_fn(
    #         pred, depth, valid,
    #         image=img,
    #         extra=log_var,
    #         teacher=teacher_used,
    #         pred_mask=mask_pred,
    #         epoch=self.cur_epoch  
    #     )

    #     return loss, logs, pred, valid, mask_pred
# inside your DepthTrainer class:

    def _forward_step(self, img, depth, teacher_pred=None):
        """
        Returns: loss, logs, pred, valid, mask_pred, fwd_times
        where fwd_times = {'model':…, 'post':…, 'loss':…}
        """
        prof = TimeProfiler()

        # ensure channel dims
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        # 1) model call
        with prof.time('model'):
            # you can branch here on warmup / teacher_pred etc.
            out = self.model(img)

        # unpack into exactly 4 outputs
        if not isinstance(out, tuple):
            out = (out, None, None, None)
        elif len(out) == 1:
            out = (out[0], None, None, None)
        elif len(out) == 2:
            out = (out[0], out[1], None, None)
        elif len(out) == 3:
            out = (out[0], out[1], out[2], None)
        pred, log_var, mask_pred, teacher_used = out

        # ensure pred channel
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # 2) post-processing: valid mask + scaling
        with prof.time('post'):
            valid = (
                (depth >= 1e-3) & (depth <= self.max_depth) &
                torch.isfinite(depth) & torch.isfinite(pred)
            )
            scale = self.best_scale
            if self.adaptive_scaling:
                pv, dv = pred[valid], depth[valid]
                if pv.numel() > 0:
                    idx   = torch.randperm(pv.numel(), device=pv.device)[:50_000]
                    scale = (pv[idx]*dv[idx]).mean() / (pv[idx]**2).mean()
            alpha = min(self.cur_epoch / max(self.num_warmup_epochs, 1), 1.0)
            pred  = (1 - alpha) * (scale * pred) + alpha * pred

        # 3) loss computation
        with prof.time('loss'):
            loss, logs = self.loss_fn(
                pred, depth, valid,
                image=img,
                extra=log_var,
                teacher=teacher_used,
                pred_mask=mask_pred,
                epoch=self.cur_epoch
            )

        return loss, logs, pred, valid, mask_pred, log_var, prof.times
    
    # def _train_epoch(self):
    #     import time  # ensure imported for timing
    #     self.model.train()
    #     loss_avg, logger, metrics = RunningAverage(), RunningLossLogger(), MonocularDepthMetrics()
        
    #     for step, sample in enumerate(self.trainloader):
    #         with self.acc.accumulate(self.model):
    #             # 1. data to device
    #             t0 = time.time()
    #             img   = sample["rgb_image"].to(self.device)
    #             depth = sample["xyz_image"].to(self.device)
    #             t_data = time.time() - t0

    #             # 2. unpack & flatten teacher_pred
    #             t0 = time.time()
    #             _tp = sample.get("teacher_pred", None)
    #             if _tp is None:
    #                 tp = None
    #             else:
    #                 tp = _tp if isinstance(_tp, torch.Tensor) else torch.from_numpy(_tp)
    #                 B, F, H, W = tp.shape
    #                 tp = tp.reshape(B * F, H, W).unsqueeze(1).to(self.device)
    #             t_unpack = time.time() - t0

    #             # 3. forward + loss
    #             t0 = time.time()
    #             loss, logs, pred, valid, mask_pred = self._forward_step(img, depth, teacher_pred=tp)
    #             t_forward = time.time() - t0
    #             if not torch.isfinite(loss):
    #                 continue
    #             if self.verbose:
    #                 print(f"[Loss Breakdown] " + " | ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    #             # 4. backward
    #             t0 = time.time()
    #             self.acc.backward(loss)
    #             refiner_norms = {}
    #             for n, p in self.model.named_parameters():
    #                 if "refiner" in n and p.grad is not None:
    #                     refiner_norms[n] = p.grad.norm().item()
    #             # print or wandb.log them:
    #             print("refiner grad norms:", refiner_norms)
    #             wandb.log({"grad/refiner_"+k: v for k,v in refiner_norms.items()}, step=self.global_step)

    #             # for n,p in self.model.named_parameters():
    #             #     if "refiner" in n:

    #             #         print(n, "requires_grad=", p.requires_grad)
                
    #             t_backward = time.time() - t0

    #             # 5. optimizer step & grad clipping
    #             t0 = time.time()
    #             if self.acc.sync_gradients:
    #                 self.adaptive_clipper.step(self.model, verbose=self.verbose)

    #                 if self.grad_clip > 0:
    #                     grad_norm = self.acc.clip_grad_norm_(
    #                         [p for p in self.model.parameters() if p.requires_grad],
    #                         max_norm=self.grad_clip
    #                     )
    #                 else:
    #                     grad_norm = torch.norm(torch.stack([
    #                         p.grad.norm() for p in self.model.parameters()
    #                         if p.grad is not None and p.requires_grad
    #                     ]), p=2)

    #                 if self.verbose:
    #                     post = torch.norm(torch.stack([
    #                         p.grad.norm() for p in self.model.parameters()
    #                         if p.grad is not None and p.requires_grad
    #                     ]))
    #                     print(f"[GradNorm] pre={grad_norm:.2f} → post={post:.2f}")
    #                     log_top_parameter_updates(self.model)


    #                 # -------- before optimizer.step() ---------------
    #                 snapshot = {
    #                     n: p.detach().clone()
    #                     for n,p in self.model.named_parameters()
    #                     if n.startswith("depth_head.refiner")
    #                 }

    #                 self.optimizer.step()
    #                 self.optimizer.zero_grad(set_to_none=True)

    #                 # -------- after optimizer.step() ----------------
    #                 for n, p in self.model.named_parameters():
    #                     if n in snapshot:
    #                         delta = (p - snapshot[n]).abs().max().item()
    #                         print(f"[update] {n:<60} Δmax = {delta:.8e}")


    #                 if self.scheduler:
    #                     self.scheduler.step()
    #                 self.lr_log.append(self.optimizer.param_groups[0]["lr"])
    #             else:
    #                 grad_norm = 0.0
    #             t_update = time.time() - t0


    #             # 6. metric accumulation (skipped if compute_metrics=False)
    #             t0 = time.time()
    #             loss_avg.update(loss.item(), img.size(0))
    #             logger.update(logs, img.size(0))
    #             if self.compute_metrics:
    #                 pred_np  = pred.detach().cpu().squeeze(1).numpy()
    #                 depth_np = depth.detach().cpu().squeeze(1).numpy()
    #                 mask_np  = valid.detach().cpu().squeeze(1).numpy()
    #                 for p, d, m in zip(pred_np, depth_np, mask_np):
    #                     if m.sum() > 0:
    #                         metrics.update(p, d, m)
    #             t_metrics = time.time() - t0

    #             # 7. timing report
    #             if self.verbose:
    #                 print(
    #                     f"[Timing] step {step+1}/{len(self.trainloader)} | "
    #                     f"data={t_data:.3f}s unpack={t_unpack:.3f}s "
    #                     f"forward={t_forward:.3f}s backward={t_backward:.3f}s "
    #                     f"update={t_update:.3f}s metrics={t_metrics:.3f}s"
    #                 )

    #             # 8. train summary (omit metrics display if disabled)
    #             metrics_str = _metrics_to_str(metrics.compute()) if self.compute_metrics else ""
    #             self.print(
    #                 f"[Train {step+1:>4}/{len(self.trainloader)}] "
    #                 f"Loss={logs['total']:.4f} {metrics_str} "
    #                 f"| GradNorm={grad_norm:.2f} ({'✅ update' if self.acc.sync_gradients else '🟡 accum'}) "
    #                 f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
    #             )


    #             # then in your loop, instead of wandb.log, do:
    #             self.acc.log(
    #                 {f"train/{k}": float(v) for k, v in logs.items()},
    #                 step=self.global_step
    #             )
    #             self.acc.log({"train/total_loss": loss.item()}, step=self.global_step)
    #             self.global_step += 1

    #             # log a panel at global steps 4, 10 and 35
    #             # … inside the for-loop, right after logs are computed …  #
    #             l1_val = logs.get("l1", None)          # might be absent if that loss is off

    #             magic_steps = {0, 4, 10, 35}
    #             should_log = (
    #                 (self.global_step in magic_steps) or
    #                 (l1_val is not None and l1_val > 1)
    #             )

    #             if should_log:
    #                 log_depth_panel(img, depth, pred, self.global_step, self.max_depth)
    #                 # log_depth_pointcloud(
    #                 #     img, pred, valid,
    #                 #     step=self.global_step,          # renamed param in the helper
    #                 #     max_depth=self.max_depth,
    #                 # )

    #             # --------------------------------------------------------------------------
    #             # DEBUG: sanity-check the mask branch
    #             # --------------------------------------------------------------------------
    #             # if self.cur_epoch == 0 and step == 0:                 # run once
    #             #     with torch.no_grad():
    #             #         mask = (
    #             #                 (depth >= 1e-3) & (depth <= self.max_depth) &
    #             #                 torch.isfinite(depth) & torch.isfinite(pred)
    #             #             )
    #             #         gt          = mask.float()                    # (B,1,H,W)
    #             #         logit       = mask_pred                      # raw network output
    #             #         prob        = torch.sigmoid(logit)

    #             #         # ① class balance in the ground-truth
    #             #         p_pos = gt.mean().item()                      # fraction of 1s
    #             #         self.print(f"[Mask DBG] GT positive-ratio    : {p_pos:5.3f}")

    #             #         # ② average model confidence for each class
    #             #         pos_conf = prob[gt.bool()].mean().item() if gt.bool().any() else float('nan')
    #             #         neg_conf = prob[~gt.bool()].mean().item() if (~gt.bool()).any() else float('nan')
    #             #         self.print(f"[Mask DBG] Prob(valid|gt=1)      : {pos_conf:5.3f}")
    #             #         self.print(f"[Mask DBG] Prob(valid|gt=0)      : {neg_conf:5.3f}")

    #             #         # ③ raw-logit statistics – should be ≈−3 … +3 at the start
    #             #         self.print(f"[Mask DBG] Logit μ/σ : {logit.mean():5.2f} / {logit.std():5.2f}")

    #             #         # ④ visual overlay to confirm orientation
    #             #         import matplotlib.pyplot as plt
    #             #         plt.figure(figsize=(8,3))
    #             #         plt.subplot(1,3,1); plt.title("GT mask");   plt.imshow(gt[0,0].cpu());   plt.axis("off")
    #             #         plt.subplot(1,3,2); plt.title("Prob");      plt.imshow(prob[0,0].cpu()); plt.axis("off")
    #             #         plt.subplot(1,3,3); plt.title("Logits");    plt.imshow(logit[0,0].cpu());plt.axis("off")
    #             #         plt.tight_layout(); plt.show()

    #             # ─── visualise predicted validity mask ────────────────────────────────
    #             if should_log and (mask_pred is not None):
    #                 # mask_pred is raw logits B×1×H×W
    #                 mask_prob = torch.sigmoid(mask_pred[0,0])      # → (H,W) in [0,1]
    #                 mask_np   = mask_prob.detach().cpu().numpy()

    #                 # Option A: pure grayscale
    #                 wandb.log({
    #                     "viz/pred_mask": wandb.Image(
    #                         mask_np,
    #                         mode="L",                    # single‐channel
    #                         caption=f"mask-prob step {self.global_step}"
    #                     )
    #                 }, step=self.global_step)


    #     self.last_train_logs = logger.compute()
    #     self.train_metrics.append(metrics.compute())
    #     return loss_avg.mean()

    # def _train_epoch(self):        
    #     """
    #     Single-epoch training loop with built-in timing via TimeProfiler.
    #     Stages: data, unpack, forward, backward, update, metrics.
    #     """

    #     self.model.train()
    #     loss_avg = RunningAverage()
    #     logger   = RunningLossLogger()
    #     metrics  = MonocularDepthMetrics()

    #     for step, sample in enumerate(self.trainloader):
    #         profiler = TimeProfiler() 
    #         with self.acc.accumulate(self.model):
    #             # 1. data to device
    #             with profiler.time('io'):
    #                 img   = sample["rgb_image"].to(self.device)
    #                 depth = sample["xyz_image"].to(self.device)

    #             # 2. unpack & flatten teacher_pred
    #             with profiler.time('unpack'):
    #                 _tp = sample.get("teacher_pred", None)
    #                 if _tp is None:
    #                     tp = None
    #                 else:
    #                     tp = _tp if isinstance(_tp, torch.Tensor) else torch.from_numpy(_tp)
    #                     B, F, H, W = tp.shape
    #                     tp = tp.reshape(B * F, H, W).unsqueeze(1).to(self.device)

    #             # 3. forward + loss
    #             with profiler.time('forward'):
    #                 loss, logs, pred, valid, mask_pred, log_var, fwd_times = self._forward_step(img, depth, teacher_pred=tp)
    #             if not torch.isfinite(loss):
    #                 continue
    #             if self.verbose:
    #                 print("[Loss Breakdown] " + " | ".join(f"{k}={v:.4f}" for k, v in logs.items()))
    #                 m = fwd_times
    #                 print(f"     → [FWD split] model={m['model']:.3f}s post={m['post']:.3f}s loss={m['loss']:.3f}s")

    #             # 4. backward
    #             with profiler.time('backward'):
    #                 self.acc.backward(loss)
    #                 # log any refiner gradients
    #                 refiner_norms = {n: p.grad.norm().item()
    #                                   for n, p in self.model.named_parameters()
    #                                   if 'refiner' in n and p.grad is not None}
    #                 print("refiner grad norms:", refiner_norms)
    #                 wandb.log({f"grad/refiner_{n}": v for n, v in refiner_norms.items()}, step=self.global_step)

    #             # 5. optimizer step & grad clipping
    #             with profiler.time('update'):
    #                 if self.acc.sync_gradients:
    #                     self.adaptive_clipper.step(self.model, verbose=self.verbose)
    #                     if self.grad_clip > 0:
    #                         grad_norm = self.acc.clip_grad_norm_(
    #                             [p for p in self.model.parameters() if p.requires_grad],
    #                             max_norm=self.grad_clip)
    #                     else:
    #                         grad_norm = torch.norm(torch.stack([
    #                             p.grad.norm() for p in self.model.parameters()
    #                             if p.grad is not None and p.requires_grad
    #                         ]), p=2)
    #                     self.optimizer.step()
    #                     self.optimizer.zero_grad(set_to_none=True)
    #                     if self.scheduler:
    #                         self.scheduler.step()
    #                     self.lr_log.append(self.optimizer.param_groups[0]['lr'])
    #                 else:
    #                     grad_norm = 0.0

    #             # 6. loss & log accumulation (always cheap)
    #             #with profiler.time('metrics'):
    #             import time
    #             t0 = time.time()
    #             loss_avg.update(loss.item(), img.size(0))
    #             logger.update(logs, img.size(0))
    #             profiler.times['logger'] = time.time() - t0


    #             # only run the expensive metric-updates every N steps (e.g. every 10 batches)
    #             if self.compute_metrics and (step % 10 == 0):
    #                 t0 = time.time()
    #                 pred_np  = pred.detach().cpu().squeeze(1).numpy()
    #                 depth_np = depth.detach().cpu().squeeze(1).numpy()
    #                 mask_np  = valid.detach().cpu().squeeze(1).numpy()
    #                 for p_arr, d_arr, m_arr in zip(pred_np, depth_np, mask_np):
    #                     if m_arr.sum() > 0:
    #                         metrics.update(p_arr, d_arr, m_arr)
    #                 profiler.times['metrics'] = profiler.times.get('metrics', 0.0) + (time.time() - t0)

    #             # 7. timing report
    #             if self.verbose:
    #                 t = profiler.times
    #                 print(
    #         f"[Timing] step {step+1}/{len(self.trainloader)} | "
    #         f"io={t['io']:.3f}s unpack={t['unpack']:.3f}s "
    #         f"forward={t['forward']:.3f}s backward={t['backward']:.3f}s "
    #         f"update={t['update']:.3f}s logger={t['logger']:.3f}s"
    #     )

    #             # 8. train summary & wandb logging (unchanged)
    #             metrics_str = _metrics_to_str(metrics.compute()) if self.compute_metrics else ""
    #             self.print(
    #                 f"[Train {step+1:>4}/{len(self.trainloader)}] "
    #                 f"Loss={logs['total']:.4f} {metrics_str} "
    #                 f"| GradNorm={grad_norm:.2f} ({'✅ update' if self.acc.sync_gradients else '🟡 accum'}) "
    #                 f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
    #             )

    #             self.acc.log({f"train/{k}": float(v) for k, v in logs.items()}, step=self.global_step)
    #             self.acc.log({"train/total_loss": loss.item()}, step=self.global_step)
    #             if self.compute_metrics:
    #                 train_metrics = metrics.compute()
    #                 # prefix as you like; here “train_metric/…”
    #                 self.acc.log(
    #                     {f"train_metric/{mk}": float(mv) for mk, mv in train_metrics.items()},
    #                     step=self.global_step
    #                 )
    #             self.global_step += 1



    #             # log a panel at global steps 4, 10 and 35
    #             # … inside the for-loop, right after logs are computed …  #

    #             magic_steps = {0, 35}
    #             should_log = (
    #                 (self.global_step in magic_steps) 
    #             )

    #             if should_log:
    #                 log_depth_panel(img, depth, pred, self.global_step, self.max_depth)
    #                 # log_depth_pointcloud(
    #                 #     img, pred, valid,
    #                 #     step=self.global_step,          # renamed param in the helper
    #                 #     max_depth=self.max_depth,
    #                 # )
    #             # ─── visualise predicted validity mask ────────────────────────────────
    #             if should_log and mask_pred is not None:
    #                 # Mask
    #                 mask_prob = torch.sigmoid(mask_pred[0,0])    # → (H,W)
    #                 mask_np   = mask_prob.detach().cpu().numpy()

    #                 # Log‐var (first view only)
    #                 log_var_np = log_var.detach().cpu().squeeze(1).numpy()[0]  # → (H,W)

    #                 # Make the heatmap figure
    #                 fig, ax = plt.subplots(figsize=(4,4))
    #                 im = ax.imshow(log_var_np, cmap="viridis")
    #                 ax.set_title(f"log-variance heatmap @ step {self.global_step}")
    #                 ax.axis("off")
    #                 fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #                 # Log both images to WandB
    #                 wandb.log({
    #                     "viz/pred_mask": wandb.Image(mask_np, mode="L", caption=f"mask-prob step {self.global_step}"),
    #                     "viz/log_var_heatmap": wandb.Image(fig, caption=f"log-var step {self.global_step}")
    #                 }, step=self.global_step)

    #                 plt.close(fig)

    #     self.last_train_logs = logger.compute()
    #     self.train_metrics.append(metrics.compute())
    #     return loss_avg.mean()


    def _train_epoch(self):        
        """
        Single-epoch training loop with built-in timing via TimeProfiler.
        Stages: data, unpack, forward, backward, update, metrics.
        """

        self.model.train()
        loss_avg = RunningAverage()
        logger   = RunningLossLogger()
        metrics  = MonocularDepthMetrics()

        for step, sample in enumerate(self.trainloader):
            profiler = TimeProfiler() 
            with self.acc.accumulate(self.model):
                # 1. data to device
                with profiler.time('io'):
                    img   = sample["rgb_image"].to(self.device)
                    depth = sample["xyz_image"].to(self.device)

                # 2. unpack & flatten teacher_pred
                with profiler.time('unpack'):
                    _tp = sample.get("teacher_pred", None)
                    if _tp is None:
                        tp = None
                    else:
                        tp = _tp if isinstance(_tp, torch.Tensor) else torch.from_numpy(_tp)
                        B, F, H, W = tp.shape
                        tp = tp.reshape(B * F, H, W).unsqueeze(1).to(self.device)

                # 3. forward + loss
                with profiler.time('forward'):
                    loss, logs, pred, valid, mask_pred, log_var, fwd_times = self._forward_step(img, depth, teacher_pred=tp)
                if not torch.isfinite(loss):
                    continue
                if self.verbose:
                    print("[Loss Breakdown] " + " | ".join(f"{k}={v:.4f}" for k, v in logs.items()))
                    m = fwd_times
                    print(f"     → [FWD split] model={m['model']:.3f}s post={m['post']:.3f}s loss={m['loss']:.3f}s")

                # 4. backward
                with profiler.time('backward'):
                    self.acc.backward(loss)
                    # log any refiner gradients
                    refiner_norms = {n: p.grad.norm().item()
                                      for n, p in self.model.named_parameters()
                                      if 'refiner' in n and p.grad is not None}
                    print("refiner grad norms:", refiner_norms)
                    wandb.log({f"grad/refiner_{n}": v for n, v in refiner_norms.items()}, step=self.global_step)

                # 5. optimizer step & grad clipping
                with profiler.time('update'):
                    if self.acc.sync_gradients:
                        self.adaptive_clipper.step(self.model, verbose=self.verbose)
                        if self.grad_clip > 0:
                            grad_norm = self.acc.clip_grad_norm_(
                                [p for p in self.model.parameters() if p.requires_grad],
                                max_norm=self.grad_clip)
                        else:
                            grad_norm = torch.norm(torch.stack([
                                p.grad.norm() for p in self.model.parameters()
                                if p.grad is not None and p.requires_grad
                            ]), p=2)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()
                        self.lr_log.append(self.optimizer.param_groups[0]['lr'])
                    else:
                        grad_norm = 0.0

                # 6. loss & log accumulation (always cheap)
                #with profiler.time('metrics'):
                import time
                t0 = time.time()
                loss_avg.update(loss.item(), img.size(0))
                logger.update(logs, img.size(0))
                profiler.times['logger'] = time.time() - t0


                # only run the expensive metric-updates every N steps (e.g. every 10 batches)
                if self.compute_metrics and (step % 10 == 0):
                    t0 = time.time()
                    pred_np  = pred.detach().cpu().squeeze(1).numpy()
                    depth_np = depth.detach().cpu().squeeze(1).numpy()
                    mask_np  = valid.detach().cpu().squeeze(1).numpy()
                    for p_arr, d_arr, m_arr in zip(pred_np, depth_np, mask_np):
                        if m_arr.sum() > 0:
                            metrics.update(p_arr, d_arr, m_arr)
                    profiler.times['metrics'] = profiler.times.get('metrics', 0.0) + (time.time() - t0)

                # 7. timing report
                if self.verbose:
                    t = profiler.times
                    print(
            f"[Timing] step {step+1}/{len(self.trainloader)} | "
            f"io={t['io']:.3f}s unpack={t['unpack']:.3f}s "
            f"forward={t['forward']:.3f}s backward={t['backward']:.3f}s "
            f"update={t['update']:.3f}s logger={t['logger']:.3f}s"
        )

                  # 8. TRAIN SUMMARY & W&B LOGGING (keep self.acc)
                metrics_str = _metrics_to_str(metrics.compute()) if self.compute_metrics else ""
                self.print(
                    f"[Train {step+1:>4}/{len(self.trainloader)}] "
                    f"Loss={logs['total']:.4f} {metrics_str} "
                    f"| GradNorm={grad_norm:.2f} "
                    f"({'✅ update' if self.acc.sync_gradients else '🟡 accum'}) "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # >>> NEW: batch up all scalar logs into one call (commit=False)
                scalars = {f"train/{k}": float(v) for k, v in logs.items()}
                scalars["train/total_loss"] = loss.item()
                self.acc.log(scalars, step=self.global_step)

                # decide when to log images
                magic_steps = {10, 35}
                should_log = (
                    (self.global_step in magic_steps)
                )

                if should_log and mask_pred is not None:
                    # Mask
                    mask_prob = torch.sigmoid(mask_pred[0,0])
                    mask_np   = mask_prob.detach().cpu().numpy()

                    # Log‐var (first view only)
                    log_var_np = log_var.detach().cpu().squeeze(1).numpy()[0]

                    # >>> NEW: downsample heatmap to ~100×100 px
                    fig, ax = plt.subplots(figsize=(2,2), dpi=50)
                    im = ax.imshow(log_var_np, cmap="viridis")
                    ax.set_title(f"log-var @ step {self.global_step}")
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # >>> NEW: log images via self.acc.log, commit=True
                    images = {
                        "viz/pred_mask": wandb.Image(
                            mask_np,
                            mode="L",
                            caption=f"mask @ step {self.global_step}"
                        ),
                        "viz/log_var": wandb.Image(
                            fig,
                            caption=f"log-var @ step {self.global_step}"
                        )
                    }
                    self.acc.log(images, step=self.global_step)
                    plt.close(fig)


                self.global_step += 1

        self.last_train_logs = logger.compute()
        self.train_metrics.append(metrics.compute())
        return loss_avg.mean()


    def _validate(self):
        self.model.eval()
        avg, logger, metrics = RunningAverage(), RunningLossLogger(), MonocularDepthMetrics()
        with torch.no_grad():
            for step, sample in enumerate(self.valloader):
                img, depth = sample["rgb_image"].to(self.device), sample["xyz_image"].to(self.device)
                # ─── optional debug plot in validation ────────────────────
                # if getattr(self, "debug", False) and step == 0:
                #     import matplotlib.pyplot as plt
                #     imgs  = img.cpu().permute(0,2,3,1).numpy()
                #     depths= depth.cpu().squeeze(1).numpy()
                #     n     = min(4, imgs.shape[0])
                #     fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
                #     for i in range(n):
                #         ax_img = axes[i, 0] if n>1 else axes[0]
                #         ax_dep = axes[i, 1] if n>1 else axes[1]
                #         ax_img.imshow(imgs[i]);  ax_img.set_title(f"Val RGB {i}");  ax_img.axis("off")
                #         ax_dep.imshow(depths[i], cmap="viridis")
                #         ax_dep.set_title(f"Val GT depth {i}"); ax_dep.axis("off")
                #     plt.tight_layout(); plt.show()

                # ─── unpack & flatten teacher_pred ─────────────────────────
                _tp = sample.get("teacher_pred", None)
                if _tp is None:
                    tp = None
                else:
                    tp = _tp if isinstance(_tp, torch.Tensor) else torch.from_numpy(_tp)
                    B, F, H, W = tp.shape
                    tp = tp.reshape(B * F, H, W).unsqueeze(1).to(self.device)

                # ─── forward + loss ───────────────────────────────────────
                loss, logs, pred, valid, mask_pred, log_var, _ = self._forward_step(img, depth, teacher_pred=tp)
                if not torch.isfinite(loss): continue
                if self.verbose:
                    print(f"[Val Loss Breakdown] " + " | ".join(f"{k}={v:.4f}" for k, v in logs.items()))
                avg.update(loss.item(), img.size(0))
                logger.update(logs, img.size(0))
                
                pred_np = pred.detach().cpu().squeeze(1).numpy()
                depth_np = depth.detach().cpu().squeeze(1).numpy()
                mask_np = valid.detach().cpu().squeeze(1).numpy()

                # log_var may be absent depending on the model's forward output
                if log_var is not None:
                    log_var_np = log_var.detach().cpu().squeeze(1).numpy()
                else:
                    log_var_np = None

                for p, d, m in zip(pred_np, depth_np, mask_np):
                    metrics.update(p, d, m)
                self.print(
                    f"[Val {step+1:>4}/{len(self.valloader)}] "  # ✅ use correct loader
                    f"Loss={logs['total']:.4f} "
                    f"{_metrics_to_str(metrics.compute())} "
                )
        self.last_val_logs = logger.compute()
        self.eval_metrics.append(metrics.compute())
        return avg.mean()

    def rebuild_optimizer(self):
        if not self._lr_getter or not self._base_optim_class:
            self.print("⚠️ rebuild_optimizer: missing _lr_getter or _base_optim_class.")
            return
        
        new_params = self._lr_getter(self._original_model)
        new_optim = self._base_optim_class(new_params, lr=self.optimizer.param_groups[0]["lr"])
        self.optimizer = new_optim
        self.model, self.optimizer = self.acc.prepare(self.model, self.optimizer)
        self.print("🔄 Optimizer rebuilt after unfreeze.")

        # 1) re-group params (e.g. layerwise_lr) via lr_getter
        param_groups = self._lr_getter(self._original_model)

        # 2) build new optimizer with same base class
        current_lr = self.optimizer.param_groups[0]["lr"]
        new_optim = self._base_optim_class(param_groups, lr=current_lr)

        # 3) reset & re‐prepare
        self.optimizer = new_optim
        self.model, self.optimizer = self.acc.prepare(self.model, self.optimizer)

        # 4) reset the LR scheduler so new params start at max‐LR
        if self.scheduler and hasattr(self.scheduler, "reset"):
            self.scheduler.reset()

        self.print("🔄 Optimizer & scheduler rebuilt after unfreeze.")

    def train(self, *, epochs: int, save_dir: str | Path = "checkpoints", val_interval=1, early_stop_patience=None, start_epoch=0):
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        best_val, no_improve = self.best_val, 0
        self.model.to('mps')
        check_refiner_devices(self.model)
        
        for epoch in range(start_epoch, start_epoch + epochs):

            if self.verbose:
                print_trainable_params(self.model)

            self.cur_epoch = epoch
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss = self._validate() if (epoch + 1) % val_interval == 0 else math.inf
            duration = time.time() - t0

            self.print(f"[epoch {epoch+1}] train={train_loss:.4f} val={val_loss:.4f} t={duration:.1f}s")

            log = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "dur": duration}
            log.update({f"train_{k}": float(v) for k, v in self.last_train_logs.items()})
            log.update({f"val_{k}": float(v) for k, v in self.last_val_logs.items()})
            self.metrics_log.append(log)

            (save_dir / "metrics.json").write_text(json.dumps(self.metrics_log, indent=2))
            (save_dir / "train_metrics.json").write_text(json.dumps(self.train_metrics, indent=2, default=to_float))
            (save_dir / "eval_metrics.json").write_text(json.dumps(self.eval_metrics, indent=2, default=to_float))

            ckpt = {"epoch": epoch, "model": self.acc.unwrap_model(self.model).state_dict(), "optim": self.optimizer.state_dict()}
            torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")

            if val_loss < best_val - 1e-4:
                best_val, no_improve = val_loss, 0
                torch.save(ckpt, save_dir / "best.pth")
            else:
                no_improve += 1
                if early_stop_patience and no_improve >= early_stop_patience:
                    self.print("🛑 Early stopping.")
                    break

            if self.unfreeze_scheduler:
                self.unfreeze_scheduler.update(
                    epoch=epoch + 1,
                    model=self.acc.unwrap_model(self.model),
                    train_logs=self.last_train_logs,
                    val_logs=self.last_val_logs,
                    grad_logs={"top": _top_grad_norms(self.acc.unwrap_model(self.model))},
                )

            # in train():
            # val_loss = self._validate()
            # wandb.log({"val/total_loss": val_loss, "epoch": epoch+1}, step=self.global_step)
            wandb.log(log, step=self.global_step)



        self.print("✅ training complete")
