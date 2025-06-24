# edge_aware_depth_losses.py
"""
Re‑implementation of the depth‑refinement loss suite, focused on:
────────────────────────────────────────────────────────────────────
• robust gradient loss (Charbonnier + edge weighting)
• percentile‑based thin‑structure loss with 1‑px dilation
• adaptive residual gate (per‑batch median)
• optional 3‑D linear / log‑linear L₁ in point space
• heteroscedastic NLL, surface–normal loss, smoothness, etc.

All routines are torch‑script friendly and drop‑in compatible with
existing *FixedDepthLoss* call‑sites.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
#   helpers: image un‑normalisation & cached grids
# ────────────────────────────────────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def unnormalize_rgb(img: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in ‑1…1 → 0…1 float."""
    return img * _STD.to(img.device) + _MEAN.to(img.device)

# cached normalised xy‑grids  (device,H,W) → (u,v)
_grid_cache: Dict[Tuple[torch.device, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

def get_uv_grid(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, h, w)
    if key not in _grid_cache:
        u = torch.linspace(-1, 1, w, device=device)
        v = torch.linspace(-1, 1, h, device=device)
        gv, gu = torch.meshgrid(v, u, indexing="ij")
        _grid_cache[key] = (gu.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
                             gv.unsqueeze(0).unsqueeze(0))
    return _grid_cache[key]

# ────────────────────────────────────────────────────────────────────────────────
#   robust gradient loss (Charbonnier, edge‑weighted)
# ────────────────────────────────────────────────────────────────────────────────

def _charb(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)

def robust_gradient_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    k_edge: float = 3.0
) -> torch.Tensor:
    
    # log-domain
    pred_log = torch.log(pred + 1e-6)
    tgt_log  = torch.log(tgt  + 1e-6)

    # spatial gradients
    dx_p = pred_log[..., 1:, :] - pred_log[..., :-1, :]
    dy_p = pred_log[..., :, 1:] - pred_log[..., :, :-1]
    dx_t = tgt_log[..., 1:, :] - tgt_log[..., :-1, :]
    dy_t = tgt_log[..., :, 1:] - tgt_log[..., :, :-1]

    # valid‐pair mask for each gradient map
    dx_m = mask[..., 1:, :] * mask[..., :-1, :]
    dy_m = mask[..., :, 1:] * mask[..., :, :-1]

    # edge weights separately for dx and dy
    edge_w_x = 1.0 + k_edge * dx_t.abs()
    edge_w_y = 1.0 + k_edge * dy_t.abs()

    # Charbonnier helper
    def _charb(x, eps=1e-3):
        return torch.sqrt(x * x + eps * eps)

    # horizontal + vertical losses
    loss_x = (_charb(dx_p - dx_t) * edge_w_x * dx_m).mean()
    loss_y = (_charb(dy_p - dy_t) * edge_w_y * dy_m).mean()

    return 0.5 * (loss_x + loss_y)

# ────────────────────────────────────────────────────────────────────────────────
#   thin‑structure loss (percentile threshold + dilation)
# ────────────────────────────────────────────────────────────────────────────────

def _gradient_mag(z: torch.Tensor) -> torch.Tensor:
    gx = z[..., 1:, :] - z[..., :-1, :]
    gy = z[..., :, 1:] - z[..., :, :-1]
    g  = torch.zeros_like(z)
    g[..., 1:,  :] += gx.abs()
    g[..., :-1, :] += gx.abs()
    g[..., :, 1:] += gy.abs()
    g[..., :, :-1] += gy.abs()
    return 0.5 * g  # simple average of two contributions

def thin_structure_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
                        pct: float = .70, debug: bool = False) -> torch.Tensor:
    grad_t = _gradient_mag(tgt)
    grad_p = _gradient_mag(pred)

    thr = torch.quantile(grad_t[mask.bool()], pct)
    thin = (grad_t > thr).float()
    thin = F.max_pool2d(thin, 3, 1, 1)           # 1‑px dilate

    diff  = (grad_p - grad_t).abs()
    valid = thin * mask
    if valid.sum() < 1:
        return torch.tensor(0., device=pred.device)
    loss = (diff * valid).sum() / valid.sum()

    if debug:
        with torch.no_grad():
            plt.figure(figsize=(6,3))
            plt.subplot(131); plt.imshow(grad_t[0,0].cpu(), cmap='gray'); plt.title('∇Z target'); plt.axis('off')
            plt.subplot(132); plt.imshow(thin[0,0].cpu(), cmap='gray'); plt.title('thin mask'); plt.axis('off')
            plt.subplot(133); plt.imshow(diff[0,0].cpu()*valid[0,0].cpu(), cmap='hot'); plt.title('error×mask'); plt.axis('off')
            plt.tight_layout(); plt.show()
    return loss

# ────────────────────────────────────────────────────────────────────────────────
#   curvature‑based attention (optionally inverted)
# ────────────────────────────────────────────────────────────────────────────────

def curvature_attention(depth: torch.Tensor, mask: torch.Tensor, inverse: bool = False,
                        eps: float = 1e-6, k: float = 10.) -> torch.Tensor:
    lap = F.conv2d(depth, depth.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3), padding=1).abs()
    flat = lap[mask.bool()]
    maxv = torch.quantile(flat, 0.99)
    norm = (lap / (maxv + eps)).clamp(0, 1)
    att  = (1. - norm) if inverse else norm
    return att

# ────────────────────────────────────────────────────────────────────────────────
#   point‑space utilities
# ────────────────────────────────────────────────────────────────────────────────

def depth_to_points(depth: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth.shape
    u, v = get_uv_grid(H, W, depth.device)
    return torch.cat([depth * u, depth * v, depth], dim=1)  # (B,3,H,W)

# ────────────────────────────────────────────────────────────────────────────────
#   FixedDepthLoss class
# ────────────────────────────────────────────────────────────────────────────────

# class FixedDepthLoss(nn.Module):
#     def __init__(self, *,
#                  silog_weight=1., grad_weight=1., smooth_weight=1.,
#                  thin_weight=.0, normal_weight=.0,
#                  l1_weight=.0, l1_3d_mode="none", use_smooth_l1=True,
#                  silog_lambda=.5, k_edge_grad=3.0, thin_pct=.70,
#                  residual_reg_weight=.02, residual_tau_mode="adaptive",
#                  hetero=False, nll_weight=1., nll_ramp_epochs=5,
#                  use_mask_bce=False,            # ← already there
#                  mask_weight=0.2,               # ← NEW  (default from yaml)
#                  sixth_weight=.2):
#         super().__init__()

#         # ---------------- save everything ----------------
#         self.silog      = SiLogLoss(lambd=silog_lambda)

#         # weights
#         self.silog_w, self.grad_w, self.smooth_w = silog_weight, grad_weight, smooth_weight
#         self.thin_w,  self.normal_w, self.l1_w   = thin_weight,  normal_weight, l1_weight

#         # flags & hyper-params
#         self.l1_3d_mode     = l1_3d_mode
#         self.use_smooth_l1  = use_smooth_l1
#         self.k_edge_grad    = k_edge_grad
#         self.thin_pct       = thin_pct
#         self.residual_w     = residual_reg_weight
#         self.res_tau_mode   = residual_tau_mode
#         self.hetero, self.nll_w, self.nll_epochs = hetero, nll_weight, nll_ramp_epochs
#         self.sixth_w        = sixth_weight
#         self.use_mask_bce   = use_mask_bce
#         self.mask_weight    = mask_weight          # ← store it!

#     # ---------------------------------------------------------------------
#     def forward(self, pred, tgt, mask, *, pred_mask: torch.Tensor | None = None, extra=None, teacher=None, epoch: int = 0, image=None):
#         mask_f = mask.float(); mask_sum = mask_f.sum().clamp(min=1.)
#         loss, logs = 0., {}

#         # ---------- heteroscedastic NLL ----------
#         if self.hetero and extra is not None:
#             logv = extra.clamp_(-8., 4.)
#             invv = torch.exp(-logv)
#             nll  = 0.5 * ((pred - tgt) ** 2 * invv + logv + math.log(2*math.pi))
#             nll  = (nll * mask_f).sum() / mask_sum
#             ramp = min(1., epoch / float(self.nll_epochs))
#             loss += ramp * self.nll_w * nll;   logs['nll'] = nll.item()

#         # ---------- residual regulariser ----------
#         if teacher is not None and self.residual_w > 0:
#             if teacher.ndim == 3: teacher = teacher.unsqueeze(1)
#             if teacher.shape != pred.shape:
#                 teacher = F.interpolate(teacher, pred.shape[-2:], mode='bilinear', align_corners=False)
#             res  = pred - teacher.detach()
#             errc = (teacher - tgt).abs()
#             tau  = 0.5 * torch.quantile(errc[mask.bool()], 0.5) if self.res_tau_mode == 'adaptive' else 0.20
#             gate = (errc < tau).float()
#             Lres = ((res ** 2) * gate * mask_f).sum() / (gate*mask_f).sum().clamp(min=1.)
#             loss += self.residual_w * Lres; logs['residual_reg'] = Lres.item()
#             logs['gate_ratio'] = gate.mean().item()

#         # ---------- SiLog ----------
#         if self.silog_w:
#             sil = self.silog(pred, tgt, mask); loss += self.silog_w * sil; logs['silog'] = sil.item()

#         # ---------- L1 (depth or 3‑D) ----------
#         if self.l1_w:
#             if self.l1_3d_mode != 'none':
#                 pt_p, pt_t = depth_to_points(pred), depth_to_points(tgt)
#                 if self.l1_3d_mode == 'log':
#                     pt_p, pt_t = torch.log(pt_p.clamp_min(1e-6)), torch.log(pt_t.clamp_min(1e-6))
#                 diff = (pt_p - pt_t).abs() * mask_f
#                 per_item = diff.view(diff.size(0), -1).mean(1)
#                 if self.l1_3d_mode == 'log' and self.sixth_w != 1.:
#                     w = torch.ones_like(per_item); w[5::6] = self.sixth_w; per_item *= w
#                 l1 = per_item.mean()
#             else:
#                 beta = .15
#                 abs_e = (pred - tgt).abs()
#                 if self.use_smooth_l1:
#                     l1_map = torch.where(abs_e < beta, 0.5*abs_e*abs_e/beta, abs_e-0.5*beta)
#                 else:
#                     l1_map = abs_e
#                 l1 = (l1_map * mask_f).sum() / mask_sum
#             loss += self.l1_w * l1; logs['l1'] = l1.item()

#         # ---------- gradient & thin losses ----------
#         if self.grad_w:
#             g = robust_gradient_loss(pred, tgt, mask, k_edge=self.k_edge_grad)
#             loss += self.grad_w * g; logs['grad'] = g.item()
#         if self.thin_w:
#             t = thin_structure_loss(pred, tgt, mask, pct=self.thin_pct)
#             loss += self.thin_w * t; logs['thin'] = t.item()

#         # inside FixedDepthLoss.forward, where you do:
#         if self.use_mask_bce and pred_mask is not None:
#             # pred_mask is raw logits, mask is {0,1}-valued GT where 1=valid
#             mask_f = mask.float()

#             # invert the GT so that 1→0 and 0→1
#             inv_mask = 1.0 - mask_f

#             bce = F.binary_cross_entropy_with_logits(
#                 pred_mask,         # (B,1,H,W) logits
#                 inv_mask,          # now 1 for invalid, 0 for valid
#                 reduction="mean"
#             )
#             loss += self.mask_weight * bce
#             logs["mask_bce"] = bce.item()

#         # (optional) smoothness & normals can be re‑added here …
#         logs['total'] = loss.item(); return loss, logs

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedDepthLoss(nn.Module):
    def __init__(
        self,
        *,
        silog_weight=1., grad_weight=1., smooth_weight=1.,
        thin_weight=0., normal_weight=0.,
        l1_weight=0., l1_3d_mode="none", use_smooth_l1=True,
        silog_lambda=0.5, k_edge_grad=3.0, thin_pct=0.70,
        residual_reg_weight=0.02, residual_tau_mode="adaptive",
        hetero=False, nll_weight=1., nll_ramp_epochs=5,
        use_mask_bce=False, mask_weight=0.2,
        sixth_weight=0.2,
        # scheduler arguments
        schedule_keys: list[str] = None,
        warmup_epochs: int = 0,
        ramp_epochs: int = 0,
    ):
        super().__init__()

        # ──────────────── save everything ────────────────
        self.silog = SiLogLoss(lambd=silog_lambda)

        # weights
        self.silog_w, self.grad_w, self.smooth_w = silog_weight, grad_weight, smooth_weight
        self.thin_w, self.normal_w, self.l1_w = thin_weight, normal_weight, l1_weight

        # flags & hyper-params
        self.l1_3d_mode = l1_3d_mode
        self.use_smooth_l1 = use_smooth_l1
        self.k_edge_grad = k_edge_grad
        self.thin_pct = thin_pct
        self.residual_w = residual_reg_weight
        self.res_tau_mode = residual_tau_mode
        self.hetero, self.nll_w, self.nll_epochs = hetero, nll_weight, nll_ramp_epochs
        self.sixth_w = sixth_weight
        self.use_mask_bce = use_mask_bce
        self.mask_weight = mask_weight

        # ───────────── schedule config ─────────────
        self.schedule_keys = schedule_keys or []
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        # capture original values for scheduled keys
        self._orig_weights = {k: getattr(self, k) for k in self.schedule_keys}

    def schedule(self, epoch: int):
        """
        Zero out or ramp specified weights for the first warmup_epochs,
        then optionally ramp over ramp_epochs.
        """
        if epoch < self.warmup_epochs:
            scale = 0.0
        elif self.ramp_epochs > 0 and epoch < self.warmup_epochs + self.ramp_epochs:
            scale = (epoch - self.warmup_epochs) / float(self.ramp_epochs)
        else:
            scale = 1.0
        for key, original in self._orig_weights.items():
            setattr(self, key, scale * original)

    def forward(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        *,
        pred_mask: torch.Tensor | None = None,
        extra=None,
        teacher=None,
        epoch: int = 0,
        image=None,
    ):
        # apply warmup/ramp schedule at the start
        self.schedule(epoch)
        #  # ─── DEBUG PRINTS ────────────────────────────────────────────────
        # print(f"[Loss Debug] pred={tuple(pred.shape)}, tgt={tuple(tgt.shape)}, mask={tuple(mask.shape)}")
        # if isinstance(extra, torch.Tensor):
        #     print(f"[Loss Debug] extra(log_var)={tuple(extra.shape)}")
        # if isinstance(pred_mask, torch.Tensor):
        #     print(f"[Loss Debug] pred_mask={tuple(pred_mask.shape)}")
        

        mask_f = mask.float(); mask_sum = mask_f.sum().clamp(min=1.)
        loss, logs = 0.0, {}

        # ---------- heteroscedastic NLL ----------
        if self.hetero and extra is not None:
            logv = extra.clamp_(-8., 4.)
            invv = torch.exp(-logv)
            nll = 0.5 * ((pred - tgt) ** 2 * invv + logv + math.log(2 * math.pi))
            nll = (nll * mask_f).sum() / mask_sum
            ramp = min(1.0, epoch / float(self.nll_epochs))
            loss += ramp * self.nll_w * nll; logs['nll'] = nll.item()

        # ---------- residual regulariser ----------
        if teacher is not None and self.residual_w > 0:
            if teacher.ndim == 3:
                teacher = teacher.unsqueeze(1)
            if teacher.shape != pred.shape:
                teacher = F.interpolate(teacher, pred.shape[-2:], mode='bilinear', align_corners=False)
            res = pred - teacher.detach()
            errc = (teacher - tgt).abs()
            if self.res_tau_mode == 'adaptive':
                tau = 0.5 * torch.quantile(errc[mask.bool()], 0.5)
            else:
                tau = 0.20
            gate = (errc < tau).float()
            Lres = ((res ** 2) * gate * mask_f).sum() / (gate * mask_f).sum().clamp(min=1.)
            loss += self.residual_w * Lres; logs['residual_reg'] = Lres.item()
            logs['gate_ratio'] = gate.mean().item()

        # ---------- SiLog ----------
        if self.silog_w:
            sil = self.silog(pred, tgt, mask); loss += self.silog_w * sil; logs['silog'] = sil.item()

        # ---------- L1 (depth or 3D) ----------
        if self.l1_w:
            if self.l1_3d_mode != 'none':
                pt_p, pt_t = depth_to_points(pred), depth_to_points(tgt)
                if self.l1_3d_mode == 'log':
                    pt_p = torch.log(pt_p.clamp_min(1e-6)); pt_t = torch.log(pt_t.clamp_min(1e-6))
                diff = (pt_p - pt_t).abs() * mask_f
                per_item = diff.view(diff.size(0), -1).mean(1)
                if self.l1_3d_mode == 'log' and self.sixth_w != 1.0:
                    w = torch.ones_like(per_item); w[5::6] = self.sixth_w; per_item *= w
                l1 = per_item.mean()
            else:
                beta = 0.15
                abs_e = (pred - tgt).abs()
                if self.use_smooth_l1:
                    l1_map = torch.where(abs_e < beta,
                                         0.5 * abs_e * abs_e / beta,
                                         abs_e - 0.5 * beta)
                else:
                    l1_map = abs_e
                l1 = (l1_map * mask_f).sum() / mask_sum
            loss += self.l1_w * l1; logs['l1'] = l1.item()

        # ---------- gradient & thin losses ----------
        if self.grad_w:
            g = robust_gradient_loss(pred, tgt, mask, k_edge=self.k_edge_grad)
            loss += self.grad_w * g; logs['grad'] = g.item()
        if self.thin_w:
            t = thin_structure_loss(pred, tgt, mask, pct=self.thin_pct)
            loss += self.thin_w * t; logs['thin'] = t.item()

        # ---------- mask BCE ----------
        if self.use_mask_bce and pred_mask is not None:
            inv_mask = 1.0 - mask_f
            bce = F.binary_cross_entropy_with_logits(
                pred_mask,
                inv_mask,
                reduction="mean"
            )
            loss += self.mask_weight * bce; logs['mask_bce'] = bce.item()

        # any optional smoothness or normal terms can be added here
        logs['total'] = loss.item()
        return loss, logs
    
# ──────────────────────────────────────────────────────────────────
class SiLogLoss(nn.Module):
    def __init__(self, lambd: float = .5):
        super().__init__(); self.lambd = lambd
    def forward(self, p: torch.Tensor, t: torch.Tensor, m: torch.Tensor):
        pl, tl = torch.log(p+1e-6), torch.log(t+1e-6)
        d = (pl - tl) * m
        mu = d.mean(); var = (d**2).mean() - self.lambd * mu * mu
        return torch.sqrt(torch.clamp(var, min=0.))
