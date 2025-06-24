import os
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

TEACHER_SIZE = 518

class TeacherCacher:
    def __init__(self, cfg, cache_root: str = ".cache/teacher", device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.teacher = None

        tag = hashlib.sha1(
            f"{cfg['model_name']}_{cfg['encoder']}_{cfg['pretrained_on_dataset']}".encode()
        ).hexdigest()[:7]

        self.cache_dir = Path(cache_root).resolve() / tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_teacher(self):
        if self.teacher is None:
            from panorai.depth import ModelRegistry
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
        
        md5k = hashlib.md5(full_key.encode()).hexdigest()
        cache_path = self.cache_dir / f"{md5k}.pt"

        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu")
            z_batch = data["z"].float()
            print(f"[TeacherCache] LOADED batch → {cache_path}  shape={tuple(z_batch.shape)}")
            return z_batch, False

        self._ensure_teacher()
        inp = F.interpolate(
            rgb_batch.to(self.device),
            size=(TEACHER_SIZE, TEACHER_SIZE),
            mode="bilinear",
            align_corners=False
        )
        with torch.no_grad():
            z_out = self.teacher(inp)
        z_batch = z_out.cpu()
        torch.save({"z": z_batch.half()}, cache_path)
        print(f"[TeacherCache] CREATED batch → {cache_path}  shape={tuple(z_batch.shape)}")
        return z_batch.float(), True

class DiskCachedTransform:
    """
    1) choose angle_idx (from whitelist) + flip (per-face)
    2) cache expensive cube-projection under MD5(raw_key_aXX).npz
    3) apply flip in-memory
    4) post-transform
    5) call teacher_cacher.batch_predict
    6) debug-plot on teacher miss (with resizing & threshold accuracy)
    """
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
        cache_angle_idxs: list[int] | None = None,
    ):
        self.transform_fn = transform_fn
        self.post_transform_fn = post_transform_fn
        self.teacher_cacher = teacher_cacher
        self.train_mode = train_mode

        self.n_angles = n_angles
        self.lon_angles = np.linspace(0, max_angle_deg, n_angles)
        self.rng = np.random.default_rng(seed)

        self.cache_angle_idxs = cache_angle_idxs or list(range(n_angles))
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key_to_path(self, base_key: str) -> str:
        h = hashlib.md5(base_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npz")

    def __call__(self, sample_with_key: dict) -> dict:
        raw_key = sample_with_key["key"]
        data = sample_with_key.get("data")

        # 1) pick angle_idx and flip
        if self.train_mode:
            angle_idx = int(self.rng.choice(self.cache_angle_idxs))
            flip = bool(self.rng.random() < 0.5)
        else:
            angle_idx, flip = 0, False

        sample_with_key["angle_idx"] = angle_idx
        sample_with_key["flip"] = flip

        # 2) build cache key only on angle
        base_key = f"{raw_key}_a{angle_idx:02d}"
        cache_path = self._key_to_path(base_key)

        # 3) load or create expensive projection (no flip)
        # if os.path.exists(cache_path):
        #     print(f"[DiskCache] LOADED {base_key}")
        #     with np.load(cache_path, allow_pickle=True) as npz:
        #         transformed = {k: npz[k] for k in npz.files}
        # else:
        #     print(f"[DiskCache] CREATED {base_key}")
        #     transformed = self.transform_fn({
        #         "key": raw_key,
        #         "data": data,
        #         "angle_idx": angle_idx,
        #         "flip": False,
        #     })
        #     np.savez_compressed(cache_path, **transformed)

        if os.path.exists(cache_path):
            try:
                print(f"[DiskCache] LOADED {base_key}")
                with np.load(cache_path, allow_pickle=False) as npz:
                    transformed = {k: npz[k] for k in npz.files}
            except (zipfile.BadZipFile, EOFError) as e:
                print(f"[DiskCache] CORRUPTED {base_key} ({e}); regenerating")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                transformed = self.transform_fn({
                    "key": raw_key,
                    "data": data,
                    "angle_idx": angle_idx,
                    "flip": False,
                })
                np.savez_compressed(cache_path, **transformed)
        else:
            print(f"[DiskCache] CREATED {base_key}")
            transformed = self.transform_fn({
                "key": raw_key,
                "data": data,
                "angle_idx": angle_idx,
                "flip": False,
            })
            np.savez_compressed(cache_path, **transformed)

        # # 4) apply flip in-memory per-face
        # if flip:
        #     transformed["rgb_image"] = transformed["rgb_image"][:, :, ::-1, :]
        #     if "xyz_image" in transformed:
        #         transformed["xyz_image"] = transformed["xyz_image"][:, :, ::-1]

        transformed["angle_idx"] = angle_idx
        transformed["flip"] = flip

        # 5) post-transform
        if self.post_transform_fn:
            transformed = self.post_transform_fn(transformed)

        # 6) teacher caching + debug plot
        if self.teacher_cacher:
            faces = transformed["rgb_image"]  # (N, H, W, 3)
            batch = (
                torch.from_numpy(faces.astype(np.float32))
                     .permute(0, 3, 1, 2)
                     .to(self.teacher_cacher.device)
            )
            z_batch, was_miss = self.teacher_cacher.batch_predict(base_key, batch)
            preds = z_batch.squeeze(1).cpu().numpy()
            transformed["teacher_pred"] = preds

            if was_miss and "xyz_image" in transformed:
                gt_images = transformed["xyz_image"]
                for i in range(gt_images.shape[0]):
                    gt = gt_images[i]
                    # resize teacher prediction to match GT resolution
                    pred_resized = cv2.resize(
                        preds[i],
                        (gt.shape[1], gt.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # compute error and threshold accuracy
                    err = np.abs(gt - pred_resized)
                    mean_err = err.mean()
                    eps = 1e-6
                    ratio = np.maximum(gt / (pred_resized + eps), pred_resized / (gt + eps))
                    delta1 = float((ratio < 1.25).mean() * 100)
                    # print metrics
                    # print(f"[DEBUG] {base_key}_face{i}: mean_err={mean_err:.3f}, δ<1.25={delta1:.1f}%")

                    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    # axs[0].imshow(gt, cmap="viridis"); axs[0].set_title("GT Depth"); axs[0].axis("off")
                    # axs[1].imshow(pred_resized, cmap="viridis"); axs[1].set_title("Teacher Pred"); axs[1].axis("off")
                    # axs[2].imshow(err, cmap="hot"); axs[2].set_title("Abs Error"); axs[2].axis("off")
                    # plt.suptitle(f"{base_key}_face{i} | mean err={mean_err:.3f} | δ<1.25={delta1:.1f}%")
                    # plt.tight_layout()
                    # plt.show()

        return transformed


import os
import hashlib
import lmdb
import pickle
import numpy as np
import torch

# class LMDBCachedTransform:
#     """
#     LMDB-only cache (no NPZ writes). Flow:
#       1) If key in LMDB: use it (no writes).
#       2) Else if NPZ exists: load NPZ, write to LMDB, then use LMDB thereafter.
#       3) Else: compute via transform_fn, write to LMDB, never write NPZ.

#     Once populated in LMDB, NPZ is never read again.
#     """
#     def __init__(
#         self,
#         transform_fn,
#         cache_dir: str,
#         lmdb_dir: str = None,
#         post_transform_fn=None,
#         teacher_cacher=None,
#         train_mode: bool = True,
#         seed: int = 42,
#         n_angles: int = 4,
#         max_angle_deg: float = 45.0,
#         cache_angle_idxs: list[int] | None = None,
#         map_size: int = 1 << 40,
#     ):
#         self.transform_fn      = transform_fn
#         self.post_transform_fn = post_transform_fn
#         self.teacher_cacher    = teacher_cacher
#         self.train_mode        = train_mode
#         self.rng               = np.random.default_rng(seed)
#         self.n_angles          = n_angles
#         self.lon_angles        = np.linspace(0, max_angle_deg, n_angles)
#         self.cache_angle_idxs  = cache_angle_idxs or list(range(n_angles))

#         # cache paths
#         self.cache_dir = os.path.abspath(cache_dir)
#         os.makedirs(self.cache_dir, exist_ok=True)
#         self.lmdb_dir  = os.path.abspath(lmdb_dir or os.path.join(self.cache_dir, "lmdb"))
#         os.makedirs(self.lmdb_dir, exist_ok=True)

#         # lazy-init LMDB env
#         self._env      = None
#         self._map_size = map_size

#     def _get_env(self):
#         if self._env is None:
#             self._env = lmdb.open(
#                 self.lmdb_dir,
#                 map_size   = self._map_size,
#                 readonly   = False,
#                 lock       = True,
#                 readahead  = False,
#                 meminit    = False,
#             )
#         return self._env

#     def _build_keys(self, raw_key: str, angle_idx: int):
#         base     = f"{raw_key}_a{angle_idx:02d}"
#         digest   = hashlib.md5(base.encode()).hexdigest()
#         npz_path = os.path.join(self.cache_dir, f"{digest}.npz")
#         return base, digest.encode(), npz_path
    
#     def __del__(self):
#         if self._env is not None:
#             self._env.close()

#     def __call__(self, sample: dict) -> dict:
#         raw_key = sample.get("key")
#         if raw_key is None:
#             # no key: direct transform
#             out = self.transform_fn(sample)
#             if self.post_transform_fn:
#                 out = self.post_transform_fn(out)
#             if self.teacher_cacher:
#                 faces = out.get("rgb_image")
#                 if faces is not None:
#                     batch = torch.from_numpy(faces.astype(np.float32)).permute(0,3,1,2).to(self.teacher_cacher.device)
#                     z, _ = self.teacher_cacher.batch_predict("", batch)
#                     out["teacher_pred"] = z.squeeze(1).cpu().numpy()
#             return out

#         # choose angle & flip
#         if self.train_mode:
#             angle_idx = int(self.rng.choice(self.cache_angle_idxs))
#             flip      = bool(self.rng.random() < 0.5)
#         else:
#             angle_idx, flip = 0, False
#         sample["angle_idx"], sample["flip"] = angle_idx, flip

#         base, lmdb_key, npz_path = self._build_keys(raw_key, angle_idx)
#         env = self._get_env()

#         # 1) Try LMDB
#         with env.begin(write=False) as txn:
#             blob = txn.get(lmdb_key)
#         if blob is not None:
#             print(f"[LMDB] HIT {base}")
#             data = pickle.loads(blob)
#         else:
#             print(f"[LMDB] MISS {base}")
#             # 2) NPZ fallback
#             if os.path.exists(npz_path):
#                 print(f"[LMDB] LOADING NPZ {npz_path}")
#                 with np.load(npz_path, allow_pickle=False) as npz:
#                     data = {k: npz[k] for k in npz.files}
#             else:
#                 print(f"[LMDB] NPZ MISSING, running transform_fn")
#                 data = self.transform_fn(sample)

#             # write only to LMDB, never NPZ
#             blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
#             with env.begin(write=True) as txn:
#                 txn.put(lmdb_key, blob)
#             print(f"[LMDB] WRITE {base}")

#         # in-memory flip
#         if flip and data.get("rgb_image") is not None:
#             data["rgb_image"] = data["rgb_image"][..., ::-1, :]
#             if "xyz_image" in data:
#                 data["xyz_image"] = data["xyz_image"][..., ::-1]

#         # post-transform
#         if self.post_transform_fn:
#             data = self.post_transform_fn(data)

#         # teacher caching
#         if self.teacher_cacher and data.get("rgb_image") is not None:
#             faces = data["rgb_image"]
#             batch = torch.from_numpy(faces.astype(np.float32)).permute(0,3,1,2).to(self.teacher_cacher.device)
#             z, _ = self.teacher_cacher.batch_predict(base, batch)
#             data["teacher_pred"] = z.squeeze(1).cpu().numpy()

#         return data


import os
import lmdb
import pickle
import zipfile
import numpy as np
import torch
import hashlib

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# train_utils.py (or wherever you keep helpers)

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def debug_plot_images(
    data: dict,
    n_cols: int = 4,
    max_images: int = None,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  tuple[float, float, float] = (0.229, 0.224, 0.225),
    undo_norm: bool = True
):
    """
    Collects every image-like array/tensor in `data` (B×H×W×C, B×C×H×W, H×W×C or C×H×W),
    unwraps all batch entries, undoes normalization if requested, then plots up to
    `max_images` of them in an n_cols-wide grid, labeling each with its index.
    """
    imgs = []

    # prepare mean/std arrays for undo
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr  = np.array(std).reshape(1, 1, 3)

    for v in data.values():
        arr = v
        # to numpy
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if not isinstance(arr, np.ndarray):
            continue

        # Batched 4D
        if arr.ndim == 4:
            B = arr.shape[0]
            # torch-style B×C×H×W
            if arr.shape[1] in (1, 3):
                for i in range(B):
                    img = np.transpose(arr[i], (1, 2, 0))
                    imgs.append(img)
            # NHWC
            elif arr.shape[3] in (1, 3):
                for i in range(B):
                    imgs.append(arr[i])

        # Single 3D image H×W×C
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            imgs.append(arr)

    if max_images is not None and len(imgs) > max_images:
        # pick a random subset instead of the first N
        idxs = np.random.permutation(len(imgs))[:max_images]
        imgs = [imgs[i] for i in idxs]

    n = len(imgs)
    if n == 0:
        print("⚠️ debug_plot_images: no images found")
        return

    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for idx, img in enumerate(imgs):
        ax = axes[idx]

        # undo normalization for RGB
        if undo_norm and img.ndim == 3 and img.shape[2] == 3:
            img = img * std_arr + mean_arr
            img = np.clip(img, 0.0, 1.0)

        # grayscale if single‐channel
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        ax.set_title(f"{idx}")
        ax.axis("off")

    # hide any extra axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

import numpy as np
import torch
def truncate_batches(
    data: dict,
    n: int,
    p_cube: float = 0.5,
    p_bottom: float = 0.1,
):
    """
    In-place: for every numpy array or torch.Tensor in `data` whose first axis is a batch,
    keep only `n` entries along axis 0 (if there are >n), sampling without replacement
    by allocating total mass p_cube to indices 0–5 (cube faces),
    of which index 5 (bottom) gets p_bottom and the other cube faces share (p_cube - p_bottom)
    equally; the remaining mass (1-p_cube) is shared equally among indices >=6.
    """
    # 1) find a batch‐style entry to get B
    B = None
    for v in data.values():
        if isinstance(v, np.ndarray) and ((v.ndim == 4) or (v.ndim == 3 and v.shape[-1] not in (1,3,4))):
            B = v.shape[0]; break
        if isinstance(v, torch.Tensor) and ((v.ndim == 4) or (v.ndim == 3 and v.shape[-1] not in (1,3,4))):
            B = v.shape[0]; break

    if B is None or B <= n:
        return data

    # 2) build the same probability vector once
    p_noncube = 1.0 - p_cube
    p = np.zeros(B, dtype=float)
    cube_cnt = min(6, B)
    non_cnt  = max(0, B - 6)

    if cube_cnt > 0:
        bottom_mass = min(p_bottom, p_cube)
        side_mass   = p_cube - bottom_mass
        per_side    = side_mass / max(cube_cnt - 1, 1)
        p[:cube_cnt] = per_side
        if cube_cnt == 6:
            p[5] = bottom_mass

    if non_cnt > 0:
        p[6:] = p_noncube / non_cnt

    p /= p.sum()

    # 3) sample idx once
    rng    = np.random.default_rng()
    idx_np = rng.choice(B, size=n, replace=False, p=p)
    idx_t  = torch.from_numpy(idx_np).long()

    # 4) apply to all batch‐style arrays/tensors
    for k, v in list(data.items()):
        if isinstance(v, np.ndarray):
            is_batch = (v.ndim == 4) or (v.ndim == 3 and v.shape[-1] not in (1,3,4))
            if is_batch:
                data[k] = v[idx_np].copy()
        elif isinstance(v, torch.Tensor):
            is_batch = (v.ndim == 4) or (v.ndim == 3 and v.shape[-1] not in (1,3,4))
            if is_batch:
                data[k] = v[idx_t.to(v.device)]

    return data

class LMDBCachedTransform:
    """
    LMDB-only cache (no NPZ writes). Flow:
      1) If key in LMDB: use it (no writes).
      2) Else if NPZ exists: load NPZ, write to LMDB, then use LMDB thereafter.
      3) Else: compute via transform_fn, write to LMDB, never write NPZ.
    """
    def __init__(
        self,
        transform_fn,
        cache_dir: str,
        lmdb_dir: str = None,
        post_transform_fn=None,
        teacher_cacher=None,
        train_mode: bool = True,
        seed: int = 42,
        n_angles: int = 4,
        max_angle_deg: float = 45.0,
        cache_angle_idxs: list[int] | None = None,
        map_size: int = 1 << 40,
    ):
        self.transform_fn      = transform_fn
        self.post_transform_fn = post_transform_fn
        self.teacher_cacher    = teacher_cacher
        self.train_mode        = train_mode
        self.rng               = np.random.default_rng(seed)
        self.n_angles          = n_angles
        self.lon_angles        = np.linspace(0, max_angle_deg, n_angles)
        self.cache_angle_idxs  = cache_angle_idxs or list(range(n_angles))

        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lmdb_dir  = os.path.abspath(lmdb_dir or os.path.join(self.cache_dir, "lmdb"))
        os.makedirs(self.lmdb_dir, exist_ok=True)

        self._env      = None
        self._map_size = map_size

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_dir,
                map_size   = self._map_size,
                readonly   = False,
                lock       = True,
                readahead  = False,
                meminit    = False,
            )
        return self._env

    def _build_keys(self, raw_key: str, angle_idx: int):
        base     = f"{raw_key}_a{angle_idx:02d}"
        digest   = hashlib.md5(base.encode()).hexdigest()
        npz_path = os.path.join(self.cache_dir, f"{digest}.npz")
        return base, digest.encode(), npz_path

    def __del__(self):
        if self._env is not None:
            self._env.close()

    def __call__(self, sample: dict) -> dict:
        raw_key = sample.get("key")
        if raw_key is None:
            out = self.transform_fn(sample)
            if self.post_transform_fn:
                out = self.post_transform_fn(out)
            if self.teacher_cacher and out.get("rgb_image") is not None:
                faces = out["rgb_image"]
                batch = torch.from_numpy(faces.astype(np.float32)).permute(0,3,1,2).to(self.teacher_cacher.device)
                z, _ = self.teacher_cacher.batch_predict("", batch)
                out["teacher_pred"] = z.squeeze(1).cpu().numpy()
            return out

        # pick angle & flip
        if self.train_mode:
            angle_idx = int(self.rng.choice(self.cache_angle_idxs))
            flip      = bool(self.rng.random() < 0.5)
        else:
            angle_idx, flip = 0, False
        sample["angle_idx"], sample["flip"] = angle_idx, flip

        base, lmdb_key, npz_path = self._build_keys(raw_key, angle_idx)
        env = self._get_env()

        # 1) Try LMDB
        with env.begin(write=False) as txn:
            blob = txn.get(lmdb_key)
        if blob is not None:
            print(f"[LMDB] HIT {base}")
            data = pickle.loads(blob)
        else:
            print(f"[LMDB] MISS {base}")
            # 2) NPZ fallback with corruption handling
            if os.path.exists(npz_path):
                try:
                    print(f"[LMDB] LOADING NPZ {npz_path}")
                    with np.load(npz_path, allow_pickle=False) as npz:
                        data = {k: npz[k] for k in npz.files}
                except (zipfile.BadZipFile, EOFError) as e:
                    print(f"[LMDB] CORRUPTED NPZ {npz_path} ({e}); regenerating")
                    try:
                        os.remove(npz_path)
                    except OSError:
                        pass
                    data = self.transform_fn(sample)
            else:
                print(f"[LMDB] NPZ MISSING, running transform_fn")
                data = self.transform_fn(sample)

            # write into LMDB
            blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            with env.begin(write=True) as txn:
                txn.put(lmdb_key, blob)
            print(f"[LMDB] WRITE {base}")

        # in-memory flip
        # if flip:
        #     for k, v in list(data.items()):
        #         # numpy arrays
        #         if isinstance(v, np.ndarray):
        #             # if it's at least 2D, flip the width axis (second-to-last)
        #             if v.ndim >= 2:
        #                 data[k] = np.flip(v, axis=-2).copy()

        #         # torch tensors
        #         elif isinstance(v, torch.Tensor):
        #             # if it's at least 2D, flip the width axis
        #             if v.ndim >= 2:
        #                 data[k] = v.flip(-1)

        # post-transform
        if self.post_transform_fn:
            data = self.post_transform_fn(data)

        # teacher caching
        if self.teacher_cacher and data.get("rgb_image") is not None:
            faces = data["rgb_image"]
            batch = torch.from_numpy(faces.astype(np.float32)).permute(0,3,1,2).to(self.teacher_cacher.device)
            z, _ = self.teacher_cacher.batch_predict(base, batch)
            data["teacher_pred"] = z.squeeze(1).cpu().numpy()

        truncate_batches(data, 6, p_cube=0.6, p_bottom=0.05)
        # debug_plot_images(data, n_cols=5, max_images=None,
        #           mean=(0.485, 0.456, 0.406),
        #           std=(0.229, 0.224, 0.225),
        #           undo_norm=True)    
        return data