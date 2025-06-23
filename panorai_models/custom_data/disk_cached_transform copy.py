# import os
# import time
# import hashlib
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# import cv2
# import matplotlib.pyplot as plt

# TEACHER_SIZE = 518


# class TeacherCacher:
#     """
#     Caches teacher outputs under .cache/teacher/<tag>/<md5(full_key)>.pt.

#     Supports:
#      - __call__(key, rgb_t): single image
#      - batch_predict(full_key, rgb_batch): entire batch under one file
#     """
#     def __init__(self, cfg, cache_root: str = ".cache/teacher", device: str = "cuda"):
#         self.cfg = cfg
#         self.device = device
#         self.teacher = None

#         tag = hashlib.sha1(
#             f"{cfg['model_name']}_{cfg['encoder']}_{cfg['pretrained_on_dataset']}".encode()
#         ).hexdigest()[:7]
#         self.cache_dir = Path(cache_root) / tag
#         self.cache_dir.mkdir(parents=True, exist_ok=True)

#     def _ensure_teacher(self):
#         if self.teacher is None:
#             from panorai_models import ModelRegistry
#             self.teacher = ModelRegistry.load(
#                 self.cfg["model_name"],
#                 dataset=self.cfg["pretrained_on_dataset"],
#                 encoder=self.cfg["encoder"],
#                 return_model=True
#             ).eval().to(self.device)

#             if self.cfg.get("load_from"):
#                 ckpt = torch.load(self.cfg["load_from"], map_location="cpu", weights_only=False)
#                 target = getattr(self.teacher, "_orig_mod", self.teacher)
#                 target.load_state_dict(ckpt["model"], strict=False)
#                 print(f"✅ checkpoint {self.cfg['load_from']} carregado")

#             for p in self.teacher.parameters():
#                 p.requires_grad = False

#     def __call__(self, key: str, rgb_t: torch.Tensor):
#         self._ensure_teacher()
#         path = self.cache_dir / f"{key}.pt"

#         if path.exists():
#             print(f"[TeacherCache] LOADED  {key}")
#             data = torch.load(path, map_location="cpu")
#             data["__cached__"] = True
#             return data

#         x = rgb_t.unsqueeze(0) if rgb_t.ndim == 3 else rgb_t[None]
#         inp = F.interpolate(x.to(self.device), size=(TEACHER_SIZE, TEACHER_SIZE),
#                              mode="bilinear", align_corners=False)
#         with torch.no_grad():
#             raw = self.teacher(inp)
#             z = raw[0] if isinstance(raw, (tuple, list)) else raw

#         z = z.cpu()
#         if z.ndim == 2:
#             z = z.unsqueeze(0)
#         torch.save({"z": z.half()}, path)
#         print(f"[TeacherCache] CREATED {key}")
#         return {"z": z, "__cached__": False}

#     def batch_predict(self, full_key: str, rgb_batch: torch.Tensor):
#         self._ensure_teacher()
#         md5k = hashlib.md5(full_key.encode()).hexdigest()
#         cache_path = self.cache_dir / f"{md5k}.pt"

#         if cache_path.exists():
#             data = torch.load(cache_path, map_location="cpu")
#             return data["z"].float(), False

#         inp = F.interpolate(
#             rgb_batch.to(self.device),
#             size=(TEACHER_SIZE, TEACHER_SIZE),
#             mode="bilinear", align_corners=False,
#         )
#         with torch.no_grad():
#             raw = self.teacher(inp)
#             depth = raw[0] if isinstance(raw, (tuple, list)) else raw

#         if depth.ndim == 3:           # (N, H, W)
#             depth = depth.unsqueeze(1)  # → (N, 1, H, W)
#         elif depth.ndim == 4 and depth.shape[1] > 1:
#             depth = depth[:, :1]

#         z_batch = depth.cpu()
#         torch.save({"z": z_batch.half()}, cache_path)
#         return z_batch.float(), True


# class DiskCachedTransform:
#     """
#     1) choose angle_idx / flip
#     2) cache gnomonic faces under MD5(raw_key_aXX_fX).npz
#     3) post‐transform
#     4) call teacher_cacher.batch_predict once per sample
#     5) debug‐plot on batch‐miss
#     """
#     def __init__(self,
#                  transform_fn,
#                  cache_dir: str,
#                  post_transform_fn=None,
#                  teacher_cacher: TeacherCacher = None,
#                  train_mode: bool = True,
#                  seed: int = 42,
#                  n_angles: int = 4,
#                  max_angle_deg: float = 45.0):
#         self.transform_fn = transform_fn
#         self.post_transform_fn = post_transform_fn
#         self.teacher_cacher = teacher_cacher
#         self.train_mode = train_mode

#         self.n_angles = n_angles
#         self.lon_angles = np.linspace(0, max_angle_deg, n_angles)
#         self.rng = np.random.default_rng(seed)

#         os.makedirs(cache_dir, exist_ok=True)
#         self.cache_dir = cache_dir

#     def _key_to_path(self, full_key: str) -> str:
#         h = hashlib.md5(full_key.encode()).hexdigest()
#         return os.path.join(self.cache_dir, f"{h}.npz")

#     def __call__(self, sample_with_key: dict) -> dict:
#         raw_key = sample_with_key["key"]

#         # ── 1) pick angle & flip ─────────────────────────────────────
#         if self.train_mode:
#             angle_idx = int(self.rng.integers(0, self.n_angles))
#             flip = bool(self.rng.random() < 0.5)
#         else:
#             angle_idx = 0
#             flip = False

#         sample_with_key["angle_idx"] = angle_idx
#         sample_with_key["flip"] = flip

#         full_key = f"{raw_key}_a{angle_idx:02d}_f{int(flip)}"
#         cache_path = self._key_to_path(full_key)

#         # ── 2) RGB/XYZ cache ─────────────────────────────────────────
#         if os.path.exists(cache_path):
#             print(f"[DiskCache] LOADED  {full_key}")
#             with np.load(cache_path, allow_pickle=True) as npz:
#                 transformed = {k: npz[k] for k in npz.files}
#         else:
#             print(f"[DiskCache] CREATED {full_key}")
#             transformed = self.transform_fn({
#                 "key": raw_key,
#                 "data": sample_with_key["data"],
#                 "angle_idx": angle_idx,
#                 "flip": flip
#             })
#             np.savez_compressed(cache_path, **transformed)

#         # ── 3) post‐transform ─────────────────────────────────────────
#         if self.post_transform_fn:
#             transformed = self.post_transform_fn(transformed)

#         # ── 4) teacher batch cache + debug ───────────────────────────
#         if self.teacher_cacher is not None:
#             faces = transformed["rgb_image"]      # (N, H, W, 3)
#             N, H, W, _ = faces.shape

#             face_batch = (
#                 torch.from_numpy(faces.astype(np.float32))
#                      .permute(0, 3, 1, 2)
#                      .to(self.teacher_cacher.device)
#             )

#             z_batch, was_miss = self.teacher_cacher.batch_predict(full_key, face_batch)
#             pred_np = z_batch.squeeze(1).cpu().numpy()
#             transformed["teacher_pred"] = pred_np

#             print(f"[DEBUG] full_key = {full_key}")
#             print(f"  faces      : {faces.shape}      dtype={faces.dtype}   min={faces.min():.3f} max={faces.max():.3f}")
#             print(f"  teacher_np : {pred_np.shape}   dtype={pred_np.dtype}   min={pred_np.min():.3f} max={pred_np.max():.3f}")

#             if was_miss and "xyz_image" in transformed:
#                 for i in range(N):
#                     gt = transformed["xyz_image"][i]
#                     print(f"  gt[{i}]     : {gt.shape}   dtype={gt.dtype}   min={gt.min():.3f} max={gt.max():.3f}")

#                     p = cv2.resize(pred_np[i],
#                                    (gt.shape[1], gt.shape[0]),
#                                    interpolation=cv2.INTER_LINEAR)
#                     print(f"  pred_rs[{i}]: {p.shape}   dtype={p.dtype}   min={p.min():.3f} max={p.max():.3f}")

#                     err = np.abs(gt - p)
#                     l1 = float(err.mean())
#                     print(f"  err[{i}]    : {err.shape}   mean={l1:.3f}")

#                     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#                     axs[0].imshow(gt, cmap="viridis");    axs[0].set_title("GT Depth");      axs[0].axis("off")
#                     axs[1].imshow(p,  cmap="viridis");    axs[1].set_title("Teacher Pred");   axs[1].axis("off")
#                     axs[2].imshow(err,cmap="hot");        axs[2].set_title("Absolute Error"); axs[2].axis("off")
#                     plt.suptitle(f"{full_key}_face{i}   L1={l1:.3f} m")
#                     plt.tight_layout()
#                     plt.show()

#         return transformed

import os
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

TEACHER_SIZE = 518
import os
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


TEACHER_SIZE = 518

class TeacherCacher:
    def __init__(self, cfg, cache_root: str = ".cache/teacher", device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.teacher = None

        tag = hashlib.sha1(
            f"{cfg['model_name']}_{cfg['encoder']}_{cfg['pretrained_on_dataset']}".encode()
        ).hexdigest()[:7]

        # ensure cache_root is inside your project .cache folder
        self.cache_dir = Path(cache_root).resolve() / tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_teacher(self):
        if self.teacher is None:
            from panorai_models import ModelRegistry
            self.teacher = ModelRegistry.load(
                self.cfg["model_name"],
                dataset=self.cfg["pretrained_on_dataset"],
                encoder=self.cfg["encoder"],
                return_model=True,
            ).eval().to(self.device)

            if self.cfg.get("load_from"):
                ckpt = torch.load(self.cfg["load_from"], map_location="cpu", weights_only=False)
                target = getattr(self.teacher, "_orig_mod", self.teacher)
                target.load_state_dict(ckpt["model"], strict=False)

            for p in self.teacher.parameters():
                p.requires_grad = False

    def batch_predict(self, full_key: str, rgb_batch: torch.Tensor):
        """
        Process an entire batch of faces at once, cache under one .pt file.
        Returns:
          z_batch  : (N,1,h,w) float32 on CPU
          was_miss : True if we ran the teacher, False if loaded cache
        """
        self._ensure_teacher()

        # hash full_key to get deterministic filename:
        md5k = hashlib.md5(full_key.encode()).hexdigest()
        cache_path = self.cache_dir / f"{md5k}.pt"

        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu")
            z_batch = data["z"].float()
            # print(f"[TeacherCache] LOADED batch → {cache_path}  shape={tuple(z_batch.shape)}")
            return z_batch, False

        # resize and run teacher on whole batch
        inp = F.interpolate(
            rgb_batch.to(self.device),
            size=(TEACHER_SIZE, TEACHER_SIZE),
            mode="bilinear",
            align_corners=False
        )
        with torch.no_grad():
            z_out = self.teacher(inp)  # returns (N, C, h, w)

        # take channel 0 as depth
        z_batch = z_out.cpu()

        # save half-precision on disk
        torch.save({"z": z_batch.half()}, cache_path)
        # print(f"[TeacherCache] CREATED batch → {cache_path}  shape={tuple(z_batch.shape)}")
        return z_batch.float(), True

import os
import hashlib
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class DiskCachedTransform:
    def __init__(
        self,
        transform_fn,
        cache_dir: str,
        post_transform_fn=None,
        teacher_cacher=None,
        train_mode: bool = True,
        seed: int = 42,
        n_angles: int = 4,
        max_angle_deg: float = 45.0,
    ):
        self.transform_fn      = transform_fn
        self.post_transform_fn = post_transform_fn
        self.teacher_cacher    = teacher_cacher
        self.train_mode        = train_mode

        self.n_angles   = n_angles
        self.lon_angles = np.linspace(0, max_angle_deg, n_angles)
        self.rng        = np.random.default_rng(seed)

        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key_to_path(self, full_key: str) -> str:
        h = hashlib.md5(full_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npz")

    def __call__(self, sample_with_key: dict) -> dict:
        raw_key = sample_with_key["key"]
        # print(f"[DEBUG] incoming raw_key: {raw_key}")
        # print(f"[DEBUG] sample_with_key before pick: angle_idx={sample_with_key.get('angle_idx')} flip={sample_with_key.get('flip')}")


        # plt.imshow(sample_with_key['data']['rgb_image'])
        # plt.show()

        # 1) pick angle & flip
        if self.train_mode:
            angle_idx = int(self.rng.integers(0, self.n_angles))
            flip      = bool(self.rng.random() < 0.5)
        else:
            angle_idx, flip = 0, False

        # print(f"[DEBUG] chosen angle_idx={angle_idx}, flip={flip}")
        # print(f"[DEBUG] lon_angles: {self.lon_angles}")

        sample_with_key["angle_idx"] = angle_idx
        sample_with_key["flip"]      = flip

        full_key   = f"{raw_key}_a{angle_idx:02d}_f{int(flip)}"
        # print(f"[DEBUG] full_key string: {full_key}")
        cache_path = self._key_to_path(full_key)
        # print(f"[DEBUG] cache_path: {cache_path}")

        # 2) LOAD or CREATE cache
        if os.path.exists(cache_path):
            # print(f"[DiskCache] LOADED  {cache_path}")
            with np.load(cache_path, allow_pickle=True) as npz:
                transformed = {k: npz[k] for k in npz.files}
        else:
            # print(f"[DiskCache] CREATED {cache_path}")
            transformed = self.transform_fn({
                "key":       raw_key,
                "data":      sample_with_key["data"],
                "angle_idx": angle_idx,
                "flip":      flip,
            })
            np.savez_compressed(cache_path, **transformed)

        # 3) post‐transform
        if self.post_transform_fn:
            transformed = self.post_transform_fn(transformed)

        # for ii in range(transformed["rgb_image"].shape[0]):
        #     import matplotlib.pyplot as plt
        #     plt.subplot(121)
        #     plt.imshow(transformed["rgb_image"][ii])
        #     plt.subplot(122)
        #     plt.imshow(transformed["xyz_image"][ii])
        #     plt.show()

        # 4) teacher batch + debug
        if self.teacher_cacher is not None:
            faces = transformed["rgb_image"]         # (N, H, W, 3)
            N, H, W, _ = faces.shape

            batch = (
                torch.from_numpy(faces.astype(np.float32))
                     .permute(0,3,1,2)
                     .to(self.teacher_cacher.device)
            )
            z_batch, was_miss = self.teacher_cacher.batch_predict(full_key, batch)
            # print(f'z_batch_shape: {z_batch.shape}')
            pred_np = z_batch.squeeze(1).cpu().numpy()
            transformed["teacher_pred"] = pred_np

            if was_miss and "xyz_image" in transformed:
                for i in range(N):
                    gt = transformed["xyz_image"][i]
                    p  = cv2.resize(pred_np[i], (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

                    valid = (gt >= 0.001) & (gt <= 80.0)
                    if valid.any():
                        ev = np.abs(gt[valid] - p[valid])
                        l1 = float(ev.mean())
                        eps = 1e-6
                        r = np.maximum(gt[valid]/(p[valid]+eps), p[valid]/(gt[valid]+eps))
                        δ125 = float((r < 1.25).mean() * 100)
                    else:
                        l1, δ125 = 0.0, 0.0

                    # print(f"{full_key}_face{i}   L1={l1:.3f}   δ<1.25={δ125:.1f}%")

                    # fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                    # RGB

                    # rgb_face = faces[i]
                    # # undo mean/std normalization:
                    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # your normalization mean
                    # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # your normalization std

                    # # if rgb_face is normalized, first un-normalize:
                    # rgb_face = (rgb_face * std[None,None,:] + mean[None,None,:])

                    # # now bring back to [0..1] range and scale to 0-255 uint8
                    # rgb_face = np.clip(rgb_face, 0.0, 1.0)
                    # rgb_face = (rgb_face * 255).astype(np.uint8)

                    # axs[0].imshow(rgb_face)
                    # axs[0].set_title("RGB Face");    axs[0].axis("off")
                    # # GT
                    # axs[1].imshow(gt, cmap="viridis")
                    # axs[1].set_title("GT Depth");    axs[1].axis("off")
                    # # Pred
                    # axs[2].imshow(p, cmap="viridis")
                    # axs[2].set_title("Teacher Pred");axs[2].axis("off")
                    # # Error
                    # axs[3].imshow(np.abs(gt - p), cmap="hot")
                    # axs[3].set_title("Abs Error");   axs[3].axis("off")

                    # plt.suptitle(f"{full_key}_face{i}   L1={l1:.3f}   δ<1.25={δ125:.1f}%")
                    # plt.show()
                    

        return transformed