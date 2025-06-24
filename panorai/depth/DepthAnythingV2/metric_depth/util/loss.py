import torch
from torch import nn



# https://hiddenlayers.tech/blog/the-mystery-of-silog-loss/
class _SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.pow(diff_log, 2).mean() - \
            self.lambd * torch.pow(diff_log.mean(), 2)

        return loss

import logging

logger = logging.getLogger("LossDebug")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import torch
import torch.nn as nn

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        """
        Computes the SiLog loss without slicing tensors — safe for gradient accumulation.
        
        Args:
            pred (Tensor): (B, 1, H, W) predicted depth
            target (Tensor): (B, 1, H, W) ground truth depth
            valid_mask (Tensor): (B, 1, H, W) boolean or 0/1 mask

        Returns:
            loss (Tensor): scalar
        """

        valid_mask = valid_mask.detach().float()  # Ensure it's float (0s and 1s)

        # Full diff log
        diff_log = torch.log(target + 1e-8) - torch.log(pred + 1e-8)

        # Masked computations
        diff_log = diff_log * valid_mask

        # Normalize by valid pixel count
        n_valid = valid_mask.sum() + 1e-8  # Avoid division by zero

        mean_squared = (diff_log ** 2).sum() / n_valid
        squared_mean = (diff_log.sum() / n_valid) ** 2

        loss = torch.sqrt(mean_squared - self.lambd * squared_mean)

        return loss
    
class _SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        with torch.no_grad():
            num_nan_pred = torch.isnan(pred_valid).sum().item()
            num_nan_target = torch.isnan(target_valid).sum().item()
            num_neg_pred = (pred_valid < 0).sum().item()
            num_neg_target = (target_valid < 0).sum().item()
            num_zero_pred = (pred_valid == 0).sum().item()
            num_zero_target = (target_valid == 0).sum().item()

            logger.debug(f"[SiLogLoss] valid pixels: {pred_valid.numel()}")
            logger.debug(f"[SiLogLoss] NaNs → pred: {num_nan_pred}, target: {num_nan_target}")
            logger.debug(f"[SiLogLoss] Negatives → pred: {num_neg_pred}, target: {num_neg_target}")
            logger.debug(f"[SiLogLoss] Zeros → pred: {num_zero_pred}, target: {num_zero_target}")

        if pred_valid.shape != target_valid.shape:
            print(f"\n[🔥 SiLogLoss CRITICAL] Masked shape mismatch!")
            print(f"  → pred[mask]:   {pred_valid.shape}")
            print(f"  → target[mask]: {target_valid.shape}")
            print(f"  → original pred shape:   {pred.shape}")
            print(f"  → original target shape: {target.shape}")
            print(f"  → mask sum: {valid_mask.sum().item()} (mask shape = {valid_mask.shape})")
            print(f"  → unique mask values: {torch.unique(valid_mask)}")
            raise ValueError("Masked pred and target shapes differ")
        
        # Use pred_valid and target_valid that you already verified!
        try:
            assert pred_valid.shape == target_valid.shape, (
                f"[SiLogLoss] ❌ Shape mismatch after masking: "
                f"pred[mask]={pred_valid.shape}, target[mask]={target_valid.shape}"
            )
            diff_log = torch.log(target_valid) - torch.log(pred_valid)

        except Exception as e:
            import os
            import uuid
            from torchvision.utils import save_image

            sample_id = str(uuid.uuid4())[:8]
            dump_dir = os.path.join("debug_skips", sample_id)
            os.makedirs(dump_dir, exist_ok=True)

            # Save the raw tensors
            torch.save(pred.detach().cpu(), os.path.join(dump_dir, "pred.pt"))
            torch.save(target.detach().cpu(), os.path.join(dump_dir, "target.pt"))
            torch.save(valid_mask.detach().cpu(), os.path.join(dump_dir, "mask.pt"))

            # Optionally save the mask as an image (rescaled to [0, 1])
            try:
                mask_img = valid_mask.float().cpu().squeeze()
                if mask_img.ndim == 2:  # [H, W]
                    save_image(mask_img.unsqueeze(0), os.path.join(dump_dir, "mask.png"))
            except Exception as im_err:
                print(f"[WARN] Failed to save mask image: {im_err}")

            print(f"[SKIP] Sample failed. Dumped to {dump_dir}. Reason: {e}")
            return torch.tensor(0.0, device=pred.device)

        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                        self.lambd * torch.pow(diff_log.mean(), 2))
        return loss
