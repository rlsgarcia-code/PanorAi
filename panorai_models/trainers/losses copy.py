import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# ===============================================================
# Logger Setup
# ===============================================================
logger = logging.getLogger("LossDebug")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ===============================================================
# Utility Functions
# ===============================================================

def validate_depth_tensors(pred, target, mask, name="DepthLoss"):
    try:
        assert pred.shape == target.shape == mask.shape, f"{name}: Incompatible shapes!"
        if (mask.sum() == 0):
            raise ValueError(f"{name}: No valid pixels (mask.sum() == 0)")
        if ((pred * mask).numel() != (target * mask).numel()):
            raise ValueError(f"{name}: Element count mismatch after masking")
    except Exception as e:
        dump_dir = os.path.join("debug_skips", str(uuid.uuid4())[:8])
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(pred.detach().cpu(), os.path.join(dump_dir, "pred.pt"))
        torch.save(target.detach().cpu(), os.path.join(dump_dir, "target.pt"))
        torch.save(mask.detach().cpu(), os.path.join(dump_dir, "mask.pt"))
        print(f"{name}: Skipped sample due to error. Dumped to {dump_dir}")
        raise RuntimeError(f"{name}: SkipSample")

def unnormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return image * std + mean

# ===============================================================
# SiLog Loss (Safe)
# ===============================================================
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, mask):
        pred = pred * mask
        target = target * mask

        diff_log = torch.log(target + 1e-8) - torch.log(pred + 1e-8)
        loss = torch.sqrt((diff_log**2).mean() - self.lambd * (diff_log.mean())**2)
        return loss

# ===============================================================
# Gradient Loss (Safe)
# ===============================================================
def gradient_loss(pred, target, mask):
    dx_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dx_target = target[..., 1:, :] - target[..., :-1, :]
    dx_mask = mask[..., 1:, :] * mask[..., :-1, :]

    dy_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dy_target = target[..., :, 1:] - target[..., :, :-1]
    dy_mask = mask[..., :, 1:] * mask[..., :, :-1]

    dx_loss = ((dx_pred - dx_target) ** 2) * dx_mask
    dy_loss = ((dy_pred - dy_target) ** 2) * dy_mask

    return (dx_loss.mean() + dy_loss.mean()) * 0.5

def log_gradient_loss(pred, target, mask, eps=1e-6):
    """
    Computes squared difference of log-depth gradients along x and y directions.
    """
    pred_log = torch.log(pred + eps)
    target_log = torch.log(target + eps)

    dx_pred = pred_log[..., 1:, :] - pred_log[..., :-1, :]
    dx_target = target_log[..., 1:, :] - target_log[..., :-1, :]
    dx_mask = mask[..., 1:, :] * mask[..., :-1, :]

    dy_pred = pred_log[..., :, 1:] - pred_log[..., :, :-1]
    dy_target = target_log[..., :, 1:] - target_log[..., :, :-1]
    dy_mask = mask[..., :, 1:] * mask[..., :, :-1]

    dx_loss = ((dx_pred - dx_target) ** 2) * dx_mask
    dy_loss = ((dy_pred - dy_target) ** 2) * dy_mask

    return 0.5 * (dx_loss.mean() + dy_loss.mean())

def log_gradient_loss_abs(pred, target, mask, eps=1e-6):
    pred_log = torch.log(pred + eps)
    target_log = torch.log(target + eps)

    dx_pred = pred_log[..., 1:, :] - pred_log[..., :-1, :]
    dx_target = target_log[..., 1:, :] - target_log[..., :-1, :]
    dx_mask = mask[..., 1:, :] * mask[..., :-1, :]

    dy_pred = pred_log[..., :, 1:] - pred_log[..., :, :-1]
    dy_target = target_log[..., :, 1:] - target_log[..., :, :-1]
    dy_mask = mask[..., :, 1:] * mask[..., :, :-1]

    dx_loss = torch.abs(dx_pred - dx_target) * dx_mask
    dy_loss = torch.abs(dy_pred - dy_target) * dy_mask

    return 0.5 * (dx_loss.sum() + dy_loss.sum()) / (dx_mask.sum() + dy_mask.sum() + 1e-6)

# ===============================================================
# Smoothness Loss (Safe)
# ===============================================================
def edge_aware_smoothness_loss(pred, image):
    image = unnormalize_image(image)

    dx_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dy_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

    dx_img = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)
    dy_img = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)

    weight_x = torch.exp(-dx_img)
    weight_y = torch.exp(-dy_img)

    return (dx_pred * weight_x).mean() + (dy_pred * weight_y).mean()

import matplotlib.pyplot as plt
import os

# Global control (only plots on the first N calls)
THIN_DEBUG_MAX_PLOTS = 5
thin_debug_counter = [0]  # use list for mutability inside function
def thin_structure_loss(pred, target, mask, threshold=0.005, debug=True):
    """
    Penalizes errors in thin structures based on gradient magnitude.
    """

    # --- Compute gradients ---
    dx_target = target[..., 1:, :-1] - target[..., :-1, :-1]
    dy_target = target[..., :-1, 1:] - target[..., :-1, :-1]
    grad_mag_target = torch.sqrt(dx_target**2 + dy_target**2)

    dx_pred = pred[..., 1:, :-1] - pred[..., :-1, :-1]
    dy_pred = pred[..., :-1, 1:] - pred[..., :-1, :-1]
    grad_mag_pred = torch.sqrt(dx_pred**2 + dy_pred**2)

    # --- Thin structure mask ---
    thin_mask = torch.sigmoid(20 * (grad_mag_target - threshold))  # Soft mask
    # thin_mask = grad_mag_target > threshold

    # --- Compute L1 difference ---
    diff = torch.abs(grad_mag_pred - grad_mag_target)

    # --- Valid mask alignment ---
    mask_cropped = mask[..., :-1, :-1]
    total_mask = thin_mask * mask_cropped

    if total_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Reweight a little towards higher gradients
    weight = 1.0 / (target[..., :-1, :-1] + 1e-3)  # Same

    # NEW: Small bonus for strong thin areas
    final_weight = (1.0 + 5.0 * thin_mask) * weight  

    loss = (diff * final_weight * mask_cropped).sum() / (mask_cropped.sum() + 1e-8)

    # --- Debug plots ---
    if debug and thin_debug_counter[0] < THIN_DEBUG_MAX_PLOTS:
        thin_debug_counter[0] += 1

        grad_target_np = grad_mag_target[0,0].detach().cpu().numpy()
        grad_pred_np = grad_mag_pred[0,0].detach().cpu().numpy()
        thin_mask_np = thin_mask[0,0].detach().cpu().numpy()
        diff_np = diff[0,0].detach().cpu().numpy()
        weight_np = weight[0,0].detach().cpu().numpy()
        total_mask_np = total_mask[0,0].detach().cpu().numpy()

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs[0,0].imshow(grad_target_np, cmap='plasma')
        axs[0,0].set_title('GT Gradient Magnitude')
        axs[0,1].imshow(grad_pred_np, cmap='plasma')
        axs[0,1].set_title('Pred Gradient Magnitude')
        axs[0,2].imshow(thin_mask_np, cmap='gray')
        axs[0,2].set_title('Thin Structure Mask')
        axs[1,0].imshow(diff_np, cmap='hot')
        axs[1,0].set_title('Error in Thin Regions')
        axs[1,1].imshow(total_mask_np, cmap='gray')
        axs[1,1].set_title('Final Mask (thin × valid)')
        axs[1,2].imshow(diff_np * weight_np * total_mask_np, cmap='hot')
        axs[1,2].set_title('Masked Error (weighted)')

        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return loss

# ===============================================================
# Global Normal Debug Counter
# ===============================================================
NORMAL_DEBUG_MAX_PLOTS = 5
normal_debug_counter = [0]

# # ===============================================================
# # Surface Normals Computation
# # ===============================================================
# import torch
# import torch.nn.functional as F

# def compute_surface_normals(depth):
#     """Compute normals from depth maps via 3D gnomonic backprojection and cross product (FOV=90° assumed)."""

#     B, C, H, W = depth.shape
#     assert C == 1, "Depth must have 1 channel."

#     device = depth.device

#     # Create a fixed [-1, 1] grid
#     u = torch.linspace(-1, 1, W, device=device)
#     v = torch.linspace(-1, 1, H, device=device)
#     grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)

#     grid_u = grid_u.expand(B, 1, H, W)  # (B,1,H,W)
#     grid_v = grid_v.expand(B, 1, H, W)

#     # Correct backprojection
#     X = depth * (grid_u )
#     Y = depth * (grid_v)
#     Z = depth 

#     points = torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)

#     # Compute spatial gradients
#     dx = points[:, :, :, 1:] - points[:, :, :, :-1]  # (B, 3, H, W-1)
#     dy = points[:, :, 1:, :] - points[:, :, :-1, :]  # (B, 3, H-1, W)

#     # Crop to align shapes
#     dx = dx[:, :, :-1, :]    # (B, 3, H-1, W-1)
#     dy = dy[:, :, :, :-1]    # (B, 3, H-1, W-1)

#     # Cross product to get normals
#     normals = torch.cross(dy, dx, dim=1)  # (B, 3, H-1, W-1)

#     # Normalize normals
#     normals = F.normalize(normals, dim=1)

#     return normals



# # ===============================================================
# # Surface Normal Loss (with Debug Visualization)
# # ===============================================================
# def surface_normal_loss(pred, target, mask, debug=True):
#     """L2 loss between predicted and target normals, masked."""
#     pred_normals = compute_surface_normals(pred)
#     target_normals = compute_surface_normals(target)
#     mask_cropped = mask[..., :-1, :-1]

#     # 🛠 FIX: Squeeze if needed
#     if pred_normals.ndim == 5:
#         pred_normals = pred_normals.squeeze(2)
#     if target_normals.ndim == 5:
#         target_normals = target_normals.squeeze(2)

#     # 🛠 Compute dot-product safely
#     dot_product = (pred_normals * target_normals).sum(dim=1, keepdim=True)  # (B,1,H-1,W-1)
#     loss = 1.0 - dot_product  # (B,1,H-1,W-1)

#     masked_loss = loss * mask_cropped
#     final_loss = masked_loss.mean()

#     # --- Debug plots ---
#     if debug and normal_debug_counter[0] < NORMAL_DEBUG_MAX_PLOTS:
#         normal_debug_counter[0] += 1

#         pred_np = pred_normals[0].permute(1,2,0).detach().cpu().numpy()  # (H,W,3)
#         target_np = target_normals[0].permute(1,2,0).detach().cpu().numpy()  # (H,W,3)
#         error_np = masked_loss[0,0].detach().cpu().numpy()  # (H,W)
#         mask_np = mask_cropped[0,0].detach().cpu().numpy()  # (H,W)

#         # Normalize for visualization: map from [-1, 1] to [0, 1]
#         pred_rgb = (pred_np + 1.0) * 0.5
#         target_rgb = (target_np + 1.0) * 0.5

#         import matplotlib.pyplot as plt
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#         axs[0,0].imshow(target_rgb)
#         axs[0,0].set_title("Target Normals (RGB)")
#         axs[0,1].imshow(pred_rgb)
#         axs[0,1].set_title("Predicted Normals (RGB)")
#         axs[1,0].imshow(error_np, cmap='hot', vmin=0, vmax=np.pi)
#         axs[1,0].set_title("Angular Error (radians)")
#         axs[1,1].imshow(mask_np, cmap='gray')
#         axs[1,1].set_title("Valid Mask")

#         for ax in axs.flat:
#             ax.axis('off')
#         plt.tight_layout()
#         plt.show()

#     return final_loss

# ===============================================================
# Depth Median Smoothing Utility
# ===============================================================
def median_smooth_depth(depth, kernel_size=9):
    """Apply median filter to depth to clean noise before computing normals."""
    B, C, H, W = depth.shape
    assert C == 1, "Depth must have 1 channel."

    pad = kernel_size // 2
    depth_padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')

    # Extract sliding local blocks
    patches = depth_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.contiguous().view(B, C, H, W, -1)

    median = patches.median(dim=-1)[0]
    return median

# ===============================================================
# Surface Normals Computation (with smoothing!)
# ===============================================================
def compute_surface_normals(depth, apply_median=True):
    """Compute normals from depth maps via 3D gnomonic backprojection and cross product (FOV=90° assumed).
    Optionally smooth depth first.
    """
    if apply_median:
        depth = median_smooth_depth(depth, kernel_size=3)

    B, C, H, W = depth.shape
    assert C == 1, "Depth must have 1 channel."

    device = depth.device

    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)

    grid_u = grid_u.expand(B, 1, H, W)
    grid_v = grid_v.expand(B, 1, H, W)

    # 3D coordinates
    X = depth * grid_u
    Y = depth * grid_v
    Z = depth

    points = torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)

    dx = points[:, :, :, 1:] - points[:, :, :, :-1]
    dy = points[:, :, 1:, :] - points[:, :, :-1, :]

    dx = dx[:, :, :-1, :]
    dy = dy[:, :, :, :-1]

    normals = torch.cross(dy, dx, dim=1)
    normals = F.normalize(normals, dim=1)

    return normals

def smooth_normals(normals, kernel_size=5):
    B, C, H, W = normals.shape
    smoothed = F.avg_pool2d(normals, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return F.normalize(smoothed, dim=1)

# ===============================================================
# Surface Normal Loss (with Smoothing)
# ===============================================================
# def surface_normal_loss(pred, target, mask, debug=False):
#     """L2 loss between predicted and target normals, masked."""
#     pred_normals = compute_surface_normals(pred, apply_median=True)
#     target_normals = compute_surface_normals(target, apply_median=True)

#     mask_cropped = mask[..., :-1, :-1]

#     if pred_normals.ndim == 5:
#         pred_normals = pred_normals.squeeze(2)
#     if target_normals.ndim == 5:
#         target_normals = target_normals.squeeze(2)

#     dot_product = (pred_normals * target_normals).sum(dim=1, keepdim=True)
#     loss = 1.0 - dot_product

#     masked_loss = loss * mask_cropped
#     final_loss = masked_loss.mean()

#     # --- Debug Plots ---
#     if debug and normal_debug_counter[0] < NORMAL_DEBUG_MAX_PLOTS:
#         normal_debug_counter[0] += 1

#         pred_np = pred_normals[0].permute(1, 2, 0).detach().cpu().numpy()
#         target_np = target_normals[0].permute(1, 2, 0).detach().cpu().numpy()
#         error_np = masked_loss[0, 0].detach().cpu().numpy()
#         mask_np = mask_cropped[0, 0].detach().cpu().numpy()

#         pred_rgb = (pred_np + 1.0) * 0.5
#         target_rgb = (target_np + 1.0) * 0.5

#         print(pred_rgb.shape)
#         print(target_rgb.shape)

#         import matplotlib.pyplot as plt
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#         axs[0,0].imshow(target_rgb)
#         axs[0,0].set_title("Target Normals (RGB)")
#         axs[0,1].imshow(pred_rgb)
#         axs[0,1].set_title("Predicted Normals (RGB)")
#         axs[1,0].imshow(error_np, cmap='hot', vmin=0, vmax=np.pi)
#         axs[1,0].set_title("Angular Error (radians)")
#         axs[1,1].imshow(mask_np, cmap='gray')
#         axs[1,1].set_title("Valid Mask")
#         for ax in axs.flat:
#             ax.axis('off')
#         plt.tight_layout()
#         plt.show()

#     return final_loss

def surface_normal_loss(pred, target, mask, sixth_weight=0.2, debug=False):
    """
    L2 loss between predicted and target normals, masked.
    Downweights every 6th item in the batch by `sixth_weight`.
    """
    B = pred.shape[0]
    pred_normals = compute_surface_normals(pred, apply_median=True)
    target_normals = compute_surface_normals(target, apply_median=True)
    mask_cropped = mask[..., :-1, :-1]

    if pred_normals.ndim == 5:
        pred_normals = pred_normals.squeeze(2)
    if target_normals.ndim == 5:
        target_normals = target_normals.squeeze(2)

    dot_product = (pred_normals * target_normals).sum(dim=1, keepdim=True)
    loss = 1.0 - dot_product
    masked_loss = loss * mask_cropped  # (B, 1, H-1, W-1)

    # Mean loss per sample (B,)
    loss_per_item = masked_loss.view(B, -1).mean(dim=1)

    # Create weighting: downweight every 6th item
    weights = torch.ones(B, device=pred.device)
    weights[5::6] = sixth_weight

    weighted_loss = (loss_per_item * weights).sum() / weights.sum()

    # --- Debug Plots ---
    if debug and normal_debug_counter[0] < NORMAL_DEBUG_MAX_PLOTS:
        normal_debug_counter[0] += 1

        pred_np = pred_normals[0].permute(1, 2, 0).detach().cpu().numpy()
        target_np = target_normals[0].permute(1, 2, 0).detach().cpu().numpy()
        error_np = masked_loss[0, 0].detach().cpu().numpy()
        mask_np = mask_cropped[0, 0].detach().cpu().numpy()

        pred_rgb = (pred_np + 1.0) * 0.5
        target_rgb = (target_np + 1.0) * 0.5

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(target_rgb)
        axs[0, 0].set_title("Target Normals (RGB)")
        axs[0, 1].imshow(pred_rgb)
        axs[0, 1].set_title("Predicted Normals (RGB)")
        axs[1, 0].imshow(error_np, cmap='hot', vmin=0, vmax=np.pi)
        axs[1, 0].set_title("Angular Error (radians)")
        axs[1, 1].imshow(mask_np, cmap='gray')
        axs[1, 1].set_title("Valid Mask")
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return weighted_loss
# ===============================================================
# L1 Depth Loss (new)
# ===============================================================
def l1_depth_loss(pred, target, mask):
    """Simple L1 loss between predicted and target depths, masked."""
    pred = pred * mask
    target = target * mask
    return (pred - target).abs().mean()

import torch.nn.functional as F

def laplacian(depth):
    laplace_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    return F.conv2d(depth, laplace_kernel, padding=1)


CURVE_PLOTS = 5
curve_debug_counter = [0]
def curvature_attention(depth, mask, sharpness=10.0, inverse=True, debug=True):
    # 1. Compute curvature
    curvature = laplacian(depth).abs() * mask

    # 2. Use robust max to avoid outliers dominating
    flat = curvature.view(curvature.shape[0], -1)
    max_val = torch.quantile(flat, 0.99, dim=1, keepdim=True).view(-1, 1, 1, 1)
    normed = curvature / (max_val + 1e-6)
    normed = normed.clamp(0, 1)

    # 3. Compute attention
    if inverse:
        attention = 1.0 - normed
    else:
        attention = torch.sigmoid(sharpness * (normed - 0.15))  # optional soft

    # 4. Debug visualization
    if debug and curve_debug_counter[0] < CURVE_PLOTS:
        curve_debug_counter[0] += 1
        att_vis = attention[0, 0].detach().cpu().numpy()
        plt.imshow(att_vis, cmap='hot', vmin=0, vmax=1)
        plt.title("Curvature-Based Inverse Attention")
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return attention

def l1_depth_loss_with_attention(pred, target, mask, attention=None):
    pred = pred * mask
    target = target * mask
    loss = (pred - target).abs()
    if attention is not None:
        loss = loss * attention
    return loss.mean()

# ===============================================================
# Updated FixedDepthLoss (SiLog + L1 + Normal)
# ===============================================================
# Updated version of FixedDepthLoss with GT edge removal using compute_surface_normals
# Updated version of FixedDepthLoss with GT edge removal using angular normal deviation

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

REMOVE_EDGE_PLOTS = 5
remove_debug_counter = [0]

def compute_surface_normals(depth, apply_median=True):
    if apply_median:
        depth = median_smooth_depth(depth)

    B, C, H, W = depth.shape
    assert C == 1
    device = depth.device

    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

    grid_u = grid_u.expand(B, 1, H, W)
    grid_v = grid_v.expand(B, 1, H, W)

    X = depth * grid_u
    Y = depth * grid_v
    Z = depth

    points = torch.cat([X, Y, Z], dim=1)

    dx = points[:, :, :, 1:] - points[:, :, :, :-1]
    dy = points[:, :, 1:, :] - points[:, :, :-1, :]

    dx = dx[:, :, :-1, :]
    dy = dy[:, :, :, :-1]

    normals = torch.cross(dy, dx, dim=1)
    normals = F.normalize(normals, dim=1)
    return normals

def remove_edges_from_gt(depth_tensor: torch.Tensor, mask: torch.Tensor, threshold=0.2):
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    normals = compute_surface_normals(depth_tensor)
    normals = smooth_normals(normals)

    nx1 = normals[:, :, :, :-1]
    nx2 = normals[:, :, :, 1:]
    ny1 = normals[:, :, :-1, :]
    ny2 = normals[:, :, 1:, :]

    dot_x = (nx1 * nx2).sum(dim=1, keepdim=True).clamp(-1, 1)
    dot_y = (ny1 * ny2).sum(dim=1, keepdim=True).clamp(-1, 1)

    angle_x = torch.acos(dot_x)
    angle_y = torch.acos(dot_y)
    angle_mag = torch.sqrt(angle_x[:, :, :-1, :]**2 + angle_y[:, :, :, :-1]**2)

    angle_mag = F.interpolate(angle_mag, size=(H, W), mode='bilinear', align_corners=False)
    edge_mask = (angle_mag < threshold).float()

    full_mask = mask * edge_mask
    masked_depth = depth_tensor * full_mask

    # if remove_debug_counter[0] < REMOVE_EDGE_PLOTS:
    #     remove_debug_counter[0] += 1
    #     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #     axs[0].imshow(depth_tensor[0, 0].detach().cpu().numpy(), cmap='viridis')
    #     axs[0].set_title("Original GT Depth")
    #     axs[1].imshow(angle_mag[0, 0].detach().cpu().numpy(), cmap='hot')
    #     axs[1].set_title("Normal Angle Magnitude")
    #     axs[2].imshow(masked_depth[0, 0].detach().cpu().numpy(), cmap='viridis')
    #     axs[2].set_title("Masked GT Depth")
    #     for ax in axs.flat:
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()

    return masked_depth

def l1_loss_3d_points(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss in 3D point space from depth maps.
    Projects depth into 3D and compares per-pixel 3D coordinates.
    
    Args:
        pred:    (B, 1, H, W) predicted depth
        target:  (B, 1, H, W) ground truth depth
        mask:    (B, 1, H, W) binary valid depth mask

    Returns:
        Scalar L1 loss in 3D space
    """
    assert pred.shape == target.shape == mask.shape
    B, C, H, W = pred.shape
    device = pred.device

    # Generate 2D normalized coordinates
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)

    grid_u = grid_u.expand(B, 1, H, W)
    grid_v = grid_v.expand(B, 1, H, W)

    # Backproject depth into 3D
    def to_points(depth):
        X = depth * grid_u
        Y = depth * grid_v
        Z = depth
        return torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)

    pred_xyz = to_points(pred)
    target_xyz = to_points(target)

    # Compute absolute difference in 3D space
    diff = torch.abs(pred_xyz - target_xyz) * mask  # broadcasted (B, 3, H, W)

    # Reduce to scalar loss
    return diff.mean()

# def log_l1_loss_3d_points(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """
#     Computes log-L1 loss in 3D point space from depth maps.
#     Projects depth into 3D and compares log-difference per coordinate.

#     Args:
#         pred:   (B, 1, H, W) predicted depth
#         target: (B, 1, H, W) ground truth depth
#         mask:   (B, 1, H, W) binary valid mask

#     Returns:
#         Scalar log-difference loss in 3D point space
#     """
#     assert pred.shape == target.shape == mask.shape
#     B, C, H, W = pred.shape
#     device = pred.device

#     # Generate normalized image plane coordinates
#     u = torch.linspace(-1, 1, W, device=device)
#     v = torch.linspace(-1, 1, H, device=device)
#     grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

#     grid_u = grid_u.expand(B, 1, H, W)
#     grid_v = grid_v.expand(B, 1, H, W)

#     def to_points(depth):
#         X = depth * grid_u
#         Y = depth * grid_v
#         Z = depth
#         return torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)

#     pred_xyz = to_points(pred)
#     target_xyz = to_points(target)

#     # Avoid log(0) or negative values by clamping
#     pred_xyz = torch.clamp(pred_xyz, min=eps)
#     target_xyz = torch.clamp(target_xyz, min=eps)

#     # Compute log-diff (absolute)
#     log_diff = torch.abs(torch.log(pred_xyz) - torch.log(target_xyz)) * mask

#     return log_diff.mean()

def log_l1_loss_3d_points(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sixth_weight: float = 0.2,
) -> torch.Tensor:
    """
    Computes L1 loss in 3D space from depth maps with downweighting every 6th element in the batch.

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        mask: (B, 1, H, W)
        sixth_weight: float weight multiplier for every 6th item (default = 0.2)

    Returns:
        Scalar weighted L1 loss
    """
    B, C, H, W = pred.shape
    device = pred.device

    # Generate 2D grid
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    grid_u = grid_u.expand(B, 1, H, W)
    grid_v = grid_v.expand(B, 1, H, W)

    # Convert depth to 3D
    def to_points(depth):
        X = depth * grid_u
        Y = depth * grid_v
        Z = depth
        return torch.cat([X, Y, Z], dim=1)

    pred_xyz = to_points(pred)
    target_xyz = to_points(target)

    # Compute 3D difference and apply mask
    diff = torch.abs(pred_xyz - target_xyz) * mask
    loss_per_item = diff.view(B, -1).mean(dim=1)  # (B,)

    # Build weights: every 6th item is downweighted
    weights = torch.ones(B, device=device)
    weights[5::6] = sixth_weight  # 0-based index → every 6th element

    # Apply weights and normalize
    weighted_loss = (loss_per_item * weights).sum() / weights.sum()
    return weighted_loss

# class FixedDepthLoss(nn.Module):
#     def __init__(self,
#                  silog_weight=1.0,
#                  grad_weight=1.0,
#                  smooth_weight=1.0,
#                  thin_weight=0.0,
#                  normal_weight=0.0,
#                  l1_weight=0.0,               # 🆕 NEW
#                  silog_lambda=0.5,
#                  use_l1_attention=False,
#                  use_gradient=False,
#                  use_smoothness=False,
#                  use_disparity=False,
#                  use_thin_structure=False,
#                  use_normal_loss=False,
#                  use_l1_loss=False,
#                  use_silog_loss=False,
#                  hetero=False,
#                  nll_weight=1.0,
#                  nll_ramp_epochs=5,
#                  residual_reg_weight=0.02,        # λ_res
#                  residual_gate_thresh=0.20,       # τ  (metres)):
#     ):
#         super().__init__()

#         self.hetero = hetero
#         self.nll_ramp_epochs = nll_ramp_epochs
#         self.residual_reg_weight  = float(residual_reg_weight)
#         self.residual_gate_thresh = float(residual_gate_thresh)

#         self.nll_weight = nll_weight
#         self.use_gradient = use_gradient
#         self.use_smoothness = use_smoothness
#         self.use_disparity = use_disparity
#         self.use_thin_structure = use_thin_structure
#         self.use_normal_loss = use_normal_loss
#         self.use_l1_loss = use_l1_loss
#         self.use_l1_attention = use_l1_attention
#         self.use_silog_loss = use_silog_loss

#         self.silog = SiLogLoss(lambd=silog_lambda)
#         self.silog_weight = silog_weight
#         self.grad_weight = grad_weight
#         self.smooth_weight = smooth_weight
#         self.thin_weight = thin_weight
#         self.normal_weight = normal_weight
#         self.l1_weight = l1_weight

#     # def forward(self, pred, target, mask, image=None,
#     #             extra=None, teacher=None):
#     #     validate_depth_tensors(pred, target, mask)

#     #     # --- Remove GT edges before any loss is computed ---
#     #     # target = remove_edges_from_gt(target, mask)

#     #     if self.use_disparity:
#     #         eps = 1e-8
#     #         pred = 1.0 / (pred + eps)
#     #         target = 1.0 / (target + eps)

#     #     loss = 0.0
#     #     logs = {}


    
#     #     # ----- heteroscedastic term (only if extra is provided) ---------------
#     #     if getattr(self, "hetero", False) and extra is not None:
#     #         inv_var = torch.exp(-extra)
#     #         error = (pred - target)**2
#     #         assert error.shape == inv_var.shape == extra.shape == mask.shape, "Hetero: shape mismatch"
#     #         nll = (error * inv_var + extra)[mask.bool()].mean()
#     #         loss += self.l1_weight * nll
#     #         logs["nll"] = nll.item()

#     #     # ----- self-distillation (only if teacher is passed) ------------------
#     #     if getattr(self, "distill_weight", 0.0) > 0 and teacher is not None:
#     #         if teacher.shape != pred.shape:
#     #             teacher = teacher.unsqueeze(1) if teacher.ndim == 3 else teacher
#     #             teacher = F.interpolate(teacher, size=pred.shape[-2:], mode="bilinear", align_corners=False)

#     #         if teacher.shape != pred.shape:
#     #             print(f"[DistillSkip] Final shape mismatch: pred={pred.shape}, teacher={teacher.shape}")
#     #         else:
#     #             valid_mask = mask.bool()
#     #             if valid_mask.sum() == 0:
#     #                 print("[Distill] No valid pixels.")
#     #             else:
#     #                 dist = ((pred - teacher)**2)[valid_mask].mean()
#     #                 loss += self.distill_weight * dist
#     #                 logs["distill"] = dist.item()

#     #         valid_mask = mask.bool()
#     #         if valid_mask.sum() == 0:
#     #             print("[Distill] No valid pixels.")
#     #         else:
#     #             dist = ((pred - teacher)**2)[valid_mask].mean()
#     #             # print(f"[Distill Debug] distill = {dist.item():.4f}")
#     #             loss += self.distill_weight * dist
#     #             logs["distill"] = dist.item()

#     #     # --- SiLog Loss
#     #     if self.use_silog_loss:
#     #         silog = self.silog(pred, target, mask)
#     #         loss += self.silog_weight * silog
#     #         logs["silog"] = silog.item()

#     #     # --- L1 Depth Loss (new)
#     #     if self.use_l1_loss:
#     #         if self.use_l1_attention:
#     #             attention = curvature_attention(target, mask)
#     #             l1 = l1_depth_loss_with_attention(pred, target, mask, attention=attention)
#     #         else:
#     #             # l1 = l1_depth_loss(pred, target, mask)
#     #             l1 = log_l1_loss_3d_points(pred, target, mask)
#     #         loss += self.l1_weight * l1
#     #         logs["l1"] = l1.item()

#     #     # --- Gradient Loss
#     #     if self.use_gradient:
#     #         # grad = gradient_loss(pred, target, mask)
#     #         grad = log_gradient_loss(pred, target, mask)
            
#     #         loss += self.grad_weight * grad
#     #         logs["grad"] = grad.item()

#     #     # --- Smoothness Loss
#     #     if self.use_smoothness and image is not None:
#     #         smooth = edge_aware_smoothness_loss(pred, image)
#     #         loss += self.smooth_weight * smooth
#     #         logs["smooth"] = smooth.item()

#     #     # --- Thin Structure Loss (optional)
#     #     if self.use_thin_structure:
#     #         thin = thin_structure_loss(pred, target, mask)
#     #         loss += self.thin_weight * thin
#     #         logs["thin"] = thin.item()

#     #     # --- Normal Loss (optional)
#     #     if self.use_normal_loss:
#     #         normal = surface_normal_loss(pred, target, mask)
#     #         loss += self.normal_weight * normal
#     #         logs["normal"] = normal.item()

#     #     logs["total"] = loss.item()
#     #     if teacher is not None and teacher.shape == pred.shape:
#     #         diff = (pred - teacher)
#     #         print(f"[Distill Δ] mean={diff.abs().mean().item():.6f} | max={diff.abs().max().item():.6f}")
#     #     return loss, logs
    
#     def forward(self, pred, target, mask, image=None,
#                 extra=None, teacher=None, current_epoch:int=0):
#         validate_depth_tensors(pred, target, mask)
#         # print(f"[Shapes] pred: {pred.shape}, target: {target.shape}, mask: {mask.shape}, teacher: {None if teacher is None else teacher.shape}")

#         # convert to float mask once
#         mask_f = mask.float()
#         mask_sum = mask_f.sum().clamp(min=1.0)  # avoid div by zero

#         if self.use_disparity:
#             eps = 1e-8
#             pred   = 1.0 / (pred   + eps)
#             target = 1.0 / (target + eps)

#         loss = 0.0
#         logs = {}

#         # ────────────────────────────────────────────────────────────────
#         # Heteroscedastic negative-log-likelihood  (per-pixel σ²)
#         # ────────────────────────────────────────────────────────────────
#         if getattr(self, "hetero", False) and extra is not None:
#             # extra  : raw network output  (B × 1 × H × W)  =  log σ²
#             # mask_f : float mask  (same shape)   |   mask_sum : mask_f.sum()
#             error   = (pred - target).pow(2)

#             # 1) Clamp log-variance to a sane band  → prevents NaNs / spikes
#             log_var = extra.clamp_(min=-8.0, max=4.0)      # σ² ∈ [≈3e-4, ≈55]
#             inv_var = torch.exp(-log_var)                  # = 1 / σ²

#             # 2) Per-pixel NLL map
#             const    = math.log(2.0 * math.pi)
#             nll_map  = 0.5 * (error * inv_var + log_var + const)

#             # 3) Masked reduction (same denominator every batch → stable)
#             nll = (nll_map * mask_f).sum() / mask_sum      # scalar
            
#             ramp = min(1.0, current_epoch / float(self.nll_ramp_epochs))
#             # print("DEBUG nll_weight type:", type(self.nll_weight), "value:", self.nll_weight)

#             loss = loss + ramp * self.nll_weight * nll
#             logs["nll"] = nll.item()

#         # ── NEW: confidence-gated residual regulariser ───────────────────
#         if (teacher is not None) and (self.residual_reg_weight > 0):
#             if teacher.ndim == 3:
#                 teacher = teacher.unsqueeze(1)              # ensure B×1×H×W
#             if teacher.shape != pred.shape:
#                 teacher = F.interpolate(
#                     teacher, size=pred.shape[-2:],
#                     mode="bilinear", align_corners=False
#                 )

#             # residual the refiner is applying
#             residual = (pred - teacher.detach())

#             # coarse error magnitude
#             err_coarse = (teacher - target).abs()

#             # gate: 1 where coarse error < τ (no big changes needed)
#             gate = (err_coarse < self.residual_gate_thresh).float()

#             gated_mask = gate * mask_f
#             num_pix    = gated_mask.sum().clamp(min=1.0)

#             L_res = ((residual ** 2) * gated_mask).sum() / num_pix

#             loss += self.residual_reg_weight * L_res
#             logs["residual_reg"] = L_res.item()

#         # --- SiLog Loss
#         if self.use_silog_loss:
#             silog = self.silog(pred, target, mask)
#             loss += self.silog_weight * silog
#             logs["silog"] = silog.item()

#         # --- L1 Depth Loss (new)
#         if self.use_l1_loss:
#             if self.use_l1_attention:
#                 attention = curvature_attention(target, mask)
#                 l1 = l1_depth_loss_with_attention(pred, target, mask, attention=attention)
#             else:
#                 l1 = log_l1_loss_3d_points(pred, target, mask)
#             loss += self.l1_weight * l1
#             logs["l1"] = l1.item()

#         # --- Gradient Loss
#         if self.use_gradient:
#             grad = log_gradient_loss(pred, target, mask)
#             loss += self.grad_weight * grad
#             logs["grad"] = grad.item()

#         # --- Smoothness Loss
#         if self.use_smoothness and image is not None:
#             smooth = edge_aware_smoothness_loss(pred, image)
#             loss  += self.smooth_weight * smooth
#             logs["smooth"] = smooth.item()

#         # --- Thin Structure Loss (optional)
#         if self.use_thin_structure:
#             thin = thin_structure_loss(pred, target, mask)
#             loss += self.thin_weight * thin
#             logs["thin"] = thin.item()

#         # --- Normal Loss (optional)
#         if self.use_normal_loss:
#             normal = surface_normal_loss(pred, target, mask)
#             loss   += self.normal_weight * normal
#             logs["normal"] = normal.item()

#         logs["total"] = loss.item()

#         # debug distill delta
#         # if teacher is not None and teacher.shape == pred.shape:
#         #     diff = (pred - teacher)
#             # print(f"[Distill Δ] mean={diff.abs().mean().item():.6f} | max={diff.abs().max().item():.6f}")

#         return loss, logs


# ---------------------------------------------------------------------------
# helpers – cached 3-D grid maker  (works for any H×W seen during training)
# ---------------------------------------------------------------------------
_grid_cache = {}          # key = (device, H, W)  →  (grid_u, grid_v)

def _get_normalised_grids(H: int, W: int, device: torch.device):
    key = (device, H, W)
    if key not in _grid_cache:
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
        # (1,1,H,W) so it broadcasts over batch
        _grid_cache[key] = (grid_u.unsqueeze(0).unsqueeze(0),
                            grid_v.unsqueeze(0).unsqueeze(0))
    return _grid_cache[key]          # (1,1,H,W), (1,1,H,W)

def _depth_to_points(depth, grid_u, grid_v):
    X = depth * grid_u
    Y = depth * grid_v
    Z = depth
    return torch.cat([X, Y, Z], dim=1)   # (B,3,H,W)
# ---------------------------------------------------------------------------


class FixedDepthLoss(nn.Module):
    def __init__(self, *,
                 # task weights
                 silog_weight=1.0, grad_weight=1.0, smooth_weight=1.0,
                 thin_weight=0.0,  normal_weight=0.0,  l1_weight=0.0,
                 # flags
                 silog_lambda=0.5, use_l1_attention=False,
                 use_smooth_l1: bool = True,
                 use_gradient=False, use_smoothness=False,
                 use_disparity=False, use_thin_structure=False,
                 use_normal_loss=False, use_l1_loss=False, use_silog_loss=False,
                 hetero=False, nll_weight=1.0, nll_ramp_epochs=5,
                 residual_reg_weight=0.02, residual_gate_thresh=0.20,
                 l1_beta: float = 0.15,
                 # NEW ► choose which 3-D loss to use
                 l1_3d_mode: str = "none",
                 sixth_weight: float = 0.2,    # only for log mode
    ):
        super().__init__()

        # store hyper-params
        self.hetero, self.nll_weight, self.nll_ramp_epochs = hetero, float(nll_weight), int(nll_ramp_epochs)
        self.residual_reg_weight  = float(residual_reg_weight)
        self.residual_gate_thresh = float(residual_gate_thresh)
        self.use_gradient, self.use_smoothness = use_gradient, use_smoothness
        self.use_disparity, self.use_thin_structure = use_disparity, use_thin_structure
        self.use_normal_loss, self.use_l1_loss = use_normal_loss, use_l1_loss
        self.use_l1_attention, self.use_silog_loss = use_l1_attention, use_silog_loss
        self.use_smooth_l1 = use_smooth_l1
        self.l1_beta       = float(l1_beta)      # ← store

        # weights
        self.silog_weight, self.grad_weight = silog_weight, grad_weight
        self.smooth_weight, self.thin_weight = smooth_weight, thin_weight
        self.normal_weight, self.l1_weight   = normal_weight, l1_weight

        # 3-D loss selector
        assert l1_3d_mode in ("linear", "log", "none"), "l1_3d_mode must be 'linear' or 'log'"
        self.l1_3d_mode  = l1_3d_mode
        self.sixth_weight = float(sixth_weight)

        # sub-loss objects
        self.silog = SiLogLoss(lambd=silog_lambda)

    # --------------------------------------------------------------------- #
    def forward(self, pred, target, mask, *, image=None,
                extra=None, teacher=None, current_epoch: int = 0):

        validate_depth_tensors(pred, target, mask)

        mask_f   = mask.float()
        mask_sum = mask_f.sum().clamp(min=1.0)

        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(mask_f[0,0].detach().cpu().numpy())
        # plt.subplot(122)
        # plt.imshow(mask_f[1,0].detach().cpu().numpy())
        # plt.show()

        # disparity option
        if self.use_disparity:
            eps = 1e-8
            pred, target = 1.0 / (pred + eps), 1.0 / (target + eps)

        # ➊ shared 2-D error tensors
        diff    = pred - target
        abs_err = diff.abs()
        sq_err  = diff.pow(2)

        loss, logs = 0.0, {}

        # ---------- Heteroscedastic NLL ----------
        if self.hetero and extra is not None:
            log_var = extra.clamp_(min=-8.0, max=4.0)
            inv_var = torch.exp(-log_var)
            nll_map = 0.5 * (sq_err * inv_var + log_var + math.log(2.0 * math.pi))
            nll     = (nll_map * mask_f).sum() / mask_sum
            ramp    = min(1.0, current_epoch / float(self.nll_ramp_epochs))
            loss   += ramp * self.nll_weight * nll
            logs["nll"] = nll.item()

        # ---------- Residual regulariser ----------
        if teacher is not None and self.residual_reg_weight > 0:
            if teacher.ndim == 3: teacher = teacher.unsqueeze(1)
            if teacher.shape != pred.shape:
                teacher = F.interpolate(teacher, pred.shape[-2:], mode="bilinear", align_corners=False)
            residual   = pred - teacher.detach()
            err_coarse = (teacher - target).abs()
            gate       = (err_coarse < self.residual_gate_thresh).float()
            gated_mask = gate * mask_f
            num_pix    = gated_mask.sum().clamp(min=1.0)
            L_res = ((residual ** 2) * gated_mask).sum() / num_pix
            loss += self.residual_reg_weight * L_res
            logs["residual_reg"] = L_res.item()

            with torch.no_grad():

                logs["gate_ratio"]        = gate.mean().item()           # % pixels protected
                logs["residual_std_good"] = residual[gate.bool()].std().item()
                logs["residual_std_bad"]  = residual[(1-gate).bool()].std().item()

        # ---------- SiLog ----------
        if self.use_silog_loss:
            silog = self.silog(pred, target, mask)
            loss += self.silog_weight * silog
            logs["silog"] = silog.item()

        # ---------- 3-D L1 / log-L1 ----------
        if self.use_l1_loss:
            if self.l1_3d_mode in ("linear", "log"):
                B, _, H, W = pred.shape
                grid_u, grid_v = _get_normalised_grids(H, W, pred.device)

                # project once
                pred_xyz   = _depth_to_points(pred,   grid_u, grid_v)   # (B,3,H,W)
                target_xyz = _depth_to_points(target, grid_u, grid_v)

                if self.l1_3d_mode == "linear":
                    diff_3d = (pred_xyz - target_xyz).abs() * mask_f
                    l1_per_item = diff_3d.view(B, -1).mean(dim=1)

                else:   # "log"
                    eps = 1e-6
                    pred_xyz   = pred_xyz.clamp(min=eps)
                    target_xyz = target_xyz.clamp(min=eps)
                    diff_3d = (torch.log(pred_xyz) - torch.log(target_xyz)).abs() * mask_f
                    l1_per_item = diff_3d.view(B, -1).mean(dim=1)

                    # down-weight every 6-th item (if desired)
                    if self.sixth_weight != 1.0:
                        weights = torch.ones(B, device=pred.device)
                        weights[5::6] = self.sixth_weight
                        l1_per_item = l1_per_item * weights

                l1 = l1_per_item.mean()
                loss += self.l1_weight * l1
                logs["l1"] = l1.item()
            else:
                # >>> 2-D depth-space branch <<<
                if self.use_smooth_l1:
                    beta = self.l1_beta
                    # Smooth-L1 element-wise
                    l1_map = torch.where(
                        abs_err < beta,
                        0.5 * sq_err / beta,      # quadratic
                        abs_err - 0.5 * beta      # linear
                    )
                else:          # plain depth L1
                    l1_map = abs_err

                if self.use_l1_attention:
                    attention = curvature_attention(target, mask)
                    l1_map *= attention

                l1 = (l1_map * mask_f).sum() / mask_sum
                loss += self.l1_weight * l1
                logs["l1"] = l1.item()
        # ---------- Other auxiliary losses ----------
        if self.use_gradient:
            grad = log_gradient_loss(pred, target, mask)
            loss += self.grad_weight * grad
            logs["grad"] = grad.item()

        if self.use_smoothness and image is not None:
            smooth = edge_aware_smoothness_loss(pred, image)
            loss  += self.smooth_weight * smooth
            logs["smooth"] = smooth.item()

        if self.use_thin_structure:
            thin = thin_structure_loss(pred, target, mask)
            loss += self.thin_weight * thin
            logs["thin"] = thin.item()

        if self.use_normal_loss:
            normal = surface_normal_loss(pred, target, mask)
            loss  += self.normal_weight * normal
            logs["normal"] = normal.item()

        logs["total"] = loss.item()
        return loss, logs