import torch
from .p74 import P74
from .p77 import P77
from .encrypted import get_cypher

def load_datasets(module_for_validation, module_for_test, cypher, train_transform=None, valid_transform=None,
                  n_angles_train=10, n_angles_val=1):
    trainset = P74(n_angles=n_angles_train, module_for_test=module_for_test, module_for_validation=module_for_validation, mode='train', cypher=cypher, transform=train_transform)
    valset = P74(n_angles=n_angles_val, module_for_test=module_for_test, module_for_validation=module_for_validation, mode='val', cypher=cypher, transform=valid_transform)
    testset = P74(n_angles=n_angles_val, module_for_test=module_for_test, module_for_validation=module_for_validation, mode='test', cypher=cypher, transform=valid_transform)
    trainset_append = P77(n_angles=n_angles_train, cypher=cypher, transform=train_transform)
    return {
        'trainset': torch.utils.data.ConcatDataset([trainset, trainset_append]),
        'valset': valset,
        'testset': testset
    }

from torch.utils.data import DataLoader
# def collate_fn(batch):
#     sample = batch[0]
#     for k, v in sample.items():
#         if k == 'image_path':
#             continue
#     sample['rgb_image'] = torch.as_tensor(sample['rgb_image'], dtype=torch.float32).permute(0,3,1,2).contiguous()
#     sample['xyz_image'] = torch.as_tensor(sample['xyz_image'][:,:,:,None], dtype=torch.float32).permute(0,3,1,2).contiguous()
#     return sample

import torch

# def collate_fn(batch):
#     """
#     Collate list[dict] → single dict with stacked / concatenated tensors,
#     while printing debug information for each key.
#     """
#     rgb_list, xyz_list, path_list = [], [], []

#     # print(f"[collate_fn] ── incoming batch of {len(batch)} samples ───────────")
#     for idx, sample in enumerate(batch):
#         # print(f"  sample {idx}: rgbImage {sample['rgb_image'].shape} "
#         #       f"| xyzImage {sample['xyz_image'].shape}")

#         # ── RGB processing ───────────────────────────────────────────────
#         rgb = torch.as_tensor(sample["rgb_image"], dtype=torch.float32)
#         if rgb.ndim == 4 and rgb.shape[-1] == 3:           # (N, H, W, 3)
#             # print("    ↳ RGB in NHWC, permuting to NCHW")
#             rgb = rgb.permute(0, 3, 1, 2).contiguous()     # → (N, 3, H, W)
#         rgb_list.append(rgb)

#         # ── Depth / XYZ processing ──────────────────────────────────────
#         xyz = torch.as_tensor(sample["xyz_image"], dtype=torch.float32)
#         if xyz.ndim == 3:                                  # (N, H, W)
#             # print("    ↳ xyz has no channel dim, unsqueezing")
#             xyz = xyz.unsqueeze(-1)                        # (N, H, W, 1)
#         if xyz.ndim == 4 and xyz.shape[-1] == 1:           # (N, H, W, 1)
#             # print("    ↳ xyz in NHWC, permuting to NCHW")
#             xyz = xyz.permute(0, 3, 1, 2).contiguous()     # → (N, 1, H, W)
#         xyz_list.append(xyz)

#         # ── optional metadata ───────────────────────────────────────────
#         if "image_path" in sample:
#             ip = sample["image_path"]
#             path_list.extend(ip if isinstance(ip, (list, tuple)) else [ip])

#     # ── concatenate along leading N dimension ‑‑ final shapes ───────────
#     rgb_batch = torch.cat(rgb_list, dim=0)
#     xyz_batch = torch.cat(xyz_list, dim=0)

#     # print(f"[collate_fn] ⇒ rgb_batch {tuple(rgb_batch.shape)}  "
#     #       f"| xyz_batch {tuple(xyz_batch.shape)}")
#     # if path_list:
#     #     print(f"[collate_fn] kept {len(path_list)} image_path entries")

#     batch_out = {
#         "rgb_image": rgb_batch,
#         "xyz_image": xyz_batch,
#     }
#     if path_list:
#         batch_out["image_path"] = path_list

#     return batch_out

# def collate_fn(batch):
#     """
#     Collate list[dict]  →  dict with
#       rgb_image    : (B,3,H,W)
#       xyz_image    : (B,1,H,W)
#       teacher_pred : (B,1,518,518)   # present only if samples contain it
#       image_path   : list[str]       # optional
#     """
#     rgb_list, xyz_list, teacher_list, path_list = [], [], [], []

#     for sample in batch:
#         # -------------- RGB --------------------------------------------
#         rgb = torch.as_tensor(sample["rgb_image"], dtype=torch.float32)
#         if rgb.ndim == 4 and rgb.shape[-1] == 3:      # NHWC → NCHW
#             rgb = rgb.permute(0, 3, 1, 2).contiguous()
#         rgb_list.append(rgb)

#         # -------------- Depth / XYZ ------------------------------------
#         xyz = torch.as_tensor(sample["xyz_image"], dtype=torch.float32)
#         if xyz.ndim == 3:                              # (N,H,W) → (N,1,H,W)
#             xyz = xyz.unsqueeze(-1)
#         if xyz.ndim == 4 and xyz.shape[-1] == 1:      # NHWC → NCHW
#             xyz = xyz.permute(0, 3, 1, 2).contiguous()
#         xyz_list.append(xyz)

#         # -------------- Teacher pred  (optional) -----------------------
#         if "teacher_pred" in sample:
#             tp = torch.as_tensor(sample["teacher_pred"], dtype=torch.float32)
#             if tp.ndim == 3:                           # (H,W) or (1,H,W)?
#                 tp = tp.unsqueeze(0)                  # ensure (1,H,W)
#             if tp.ndim == 4 and tp.shape[-1] == 1:    # NHWC → NCHW
#                 tp = tp.permute(0, 3, 1, 2).contiguous()
#             teacher_list.append(tp)

#         # -------------- metadata path (optional) -----------------------
#         if "image_path" in sample:
#             ip = sample["image_path"]
#             path_list.extend(ip if isinstance(ip, (list, tuple)) else [ip])

#     # ---------- stack / concat -----------------------------------------
#     rgb_batch = torch.cat(rgb_list, dim=0)
#     xyz_batch = torch.cat(xyz_list, dim=0)

#     batch_out = {
#         "rgb_image": rgb_batch,
#         "xyz_image": xyz_batch,
#     }

#     if teacher_list:                                   # add only if present
#         batch_out["teacher_pred"] = torch.cat(teacher_list, dim=0)

#     if path_list:
#         batch_out["image_path"] = path_list

#     return batch_out

import torch

def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    3-D (H,W,C)   -> (C,H,W)  
    4-D (N,H,W,C) -> (N,C,H,W)
    Otherwise returns x unchanged.
    """
    if x.ndim == 3 and x.shape[-1] in {1, 3}:
        return x.permute(2, 0, 1).contiguous()
    if x.ndim == 4 and x.shape[-1] in {1, 3}:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def collate_fn(batch):
    """
    • RGB / XYZ are guaranteed to stay aligned with teacher_pred and path  
    • Outputs always have the same first-dimension length
        rgb_image    : (B, 3,  H,  W)
        xyz_image    : (B, 1,  H,  W)
        teacher_pred : (B, 1, 518, 518)  (zeros if a sample had none)
        image_path   : list[str] length B
    """
    rgb_all, xyz_all, teacher_all, path_all = [], [], [], []

    for sample in batch:
        # ---------- RGB ----------------------------------------------------
        rgb = torch.as_tensor(sample["rgb_image"], dtype=torch.float32)
        rgb = _to_nchw(rgb)                       # (F,3,H,W) or (3,H,W)

        # ---------- depth --------------------------------------------------
        xyz = torch.as_tensor(sample["xyz_image"], dtype=torch.float32)
        if xyz.ndim == 3:                         # (F,H,W) -> (F,1,H,W)
            xyz = xyz.unsqueeze(-1)
        xyz = _to_nchw(xyz)                       # (F,1,H,W)

        # ---------- teacher pred (optional) --------------------------------
        if "teacher_pred" in sample:
            tp = torch.as_tensor(sample["teacher_pred"], dtype=torch.float32)
            if tp.ndim == 3:                      # (H,W) or (1,H,W)
                tp = tp.unsqueeze(0)
            tp = _to_nchw(tp)                     # (F,1,518,518)
        else:
            tp = torch.empty(0, 1, 518, 518)      # zero-length placeholder

        # ---------- image path / uid ---------------------------------------
        ip = sample.get("image_path", [None] * rgb.shape[0])
        if not isinstance(ip, (list, tuple)):
            ip = [ip] * rgb.shape[0]

        # ---------- append, keeping order intact ---------------------------
        rgb_all.append(rgb)
        xyz_all.append(xyz)
        teacher_all.append(tp)
        path_all.extend(ip)

    # ---------- concatenate along batch dimension --------------------------
    rgb_batch  = torch.cat(rgb_all,     dim=0)         # (B,3,H,W)
    xyz_batch  = torch.cat(xyz_all,     dim=0)         # (B,1,H,W)
    tp_batch   = torch.cat(teacher_all, dim=0)         # (B,1,518,518) or (0,1,518,518)

    out = {
        "rgb_image": rgb_batch,
        "xyz_image": xyz_batch,
        "image_path": path_all,        # always length B
    }

    if tp_batch.numel():               # add only if at least one sample has it
        out["teacher_pred"] = tp_batch

    return out

__all__ = ["load_datasets", "get_cypher", "collate_fn"]

