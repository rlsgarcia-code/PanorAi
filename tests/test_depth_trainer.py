import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from contextlib import contextmanager
import math
import pytest

try:
    import torch
except Exception as e:  # pragma: no cover - only executed when torch import fails
    pytest.skip(
        f"Skipping depth trainer tests because torch failed to import: {e}",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Stub external dependencies (accelerate, wandb)
# ---------------------------------------------------------------------------
class DummyAccelerator:
    def __init__(self, *a, gradient_accumulation_steps=1, mixed_precision=None, log_with=None, **kw):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cpu")
        self.print = print
        self.sync_gradients = True

    def init_trackers(self, *a, **k):
        pass

    def prepare(self, *objs):
        return objs

    @contextmanager
    def accumulate(self, *a, **k):
        yield

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def unwrap_model(self, model):
        return model

    def log(self, *a, **k):
        pass

accelerate_stub = ModuleType("accelerate")
accelerate_stub.Accelerator = DummyAccelerator
sys.modules.setdefault("accelerate", accelerate_stub)

# -- wandb --------------------------------------------------------------
class DummyImage:
    def __init__(self, data, *a, **k):
        self.data = data

def _noop(*a, **k):
    pass

wandb_stub = ModuleType("wandb")
wandb_stub.init = _noop
wandb_stub.log = _noop
wandb_stub.Image = DummyImage
wandb_stub.Object3D = DummyImage
wandb_stub.watch = _noop
sys.modules.setdefault("wandb", wandb_stub)

# ---------------------------------------------------------------------------
# Stub project modules with heavy dependencies
# ---------------------------------------------------------------------------
train_utils_stub = ModuleType("train_utils")

def maybe_compile(model, *a, **k):
    return model

class EMAAdaptiveClipper:
    def __init__(self, *a, **k):
        pass

    def step(self, *a, **k):
        pass

def add_depth_noise(x, *a, **k):
    return x

train_utils_stub.maybe_compile = maybe_compile
train_utils_stub.EMAAdaptiveClipper = EMAAdaptiveClipper
train_utils_stub.add_depth_noise = add_depth_noise
sys.modules.setdefault("train_utils", train_utils_stub)

metrics_stub = ModuleType("panorai_models.trainers.metrics")

class MonocularDepthMetrics:
    def __init__(self):
        pass

    def reset(self):
        pass

    def update(self, *a, **k):
        pass

    def compute(self):
        return {}

metrics_stub.MonocularDepthMetrics = MonocularDepthMetrics
sys.modules.setdefault("panorai_models.trainers.metrics", metrics_stub)

# ---------------------------------------------------------------------------
# Stub panorai and open3d if unavailable
# ---------------------------------------------------------------------------
panorai_stub = ModuleType("panorai")

class EquirectangularImage:
    def __init__(self, *a, **k):
        pass

class DummyPCD:
    def __init__(self):
        self.o3d = SimpleNamespace(colors=None)

class GnomonicFace:
    def __init__(self, *a, **k):
        pass

    def to_pcd(self, *a, **k):
        return DummyPCD()

panorai_stub.EquirectangularImage = EquirectangularImage
panorai_stub.GnomonicFace = GnomonicFace
sys.modules.setdefault("panorai", panorai_stub)

o3d_stub = ModuleType("open3d")
o3d_stub.io = ModuleType("open3d.io")
o3d_stub.utility = ModuleType("open3d.utility")

def _noop(*a, **k):
    pass

o3d_stub.io.write_point_cloud = _noop
o3d_stub.utility.Vector3dVector = _noop
sys.modules.setdefault("open3d", o3d_stub)

# ---------------------------------------------------------------------------
# Import DepthTrainer after stubbing
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DepthTrainer = importlib.import_module("panorai_models.trainers.depth_trainer").DepthTrainer

# ---------------------------------------------------------------------------
# Minimal stub model and dataset
# ---------------------------------------------------------------------------
class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class RandomDepthDataset(torch.utils.data.Dataset):
    def __init__(self, n=8, size=16):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rgb = torch.randn(3, self.size, self.size)
        depth = torch.rand(1, self.size, self.size)
        return {"rgb_image": rgb, "xyz_image": depth}

# ---------------------------------------------------------------------------
# Dummy loss function matching DepthTrainer expectations
# ---------------------------------------------------------------------------
def dummy_loss(pred, tgt, mask, **_):
    loss = torch.mean(torch.abs(pred - tgt))
    return loss, {"total": loss.item()}

# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------
def test_depth_trainer_single_epoch(tmp_path):
    train_ds = RandomDepthDataset()
    val_ds = RandomDepthDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2)

    model = TinyCNN()
    trainer = DepthTrainer(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
        max_depth=10.0,
        loss_fn=dummy_loss,
        device="cpu",
        grad_accum=1,
        compile_model=False,
        verbose=False,
    )

    optim = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.set_accelerator(optim)

    # Patch methods that require unavailable backends
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
        mp.setattr(torch.jit, "trace", lambda model, example: model)
        trainer.train(epochs=1, save_dir=tmp_path)

    ckpt = tmp_path / "epoch_000.pth"
    assert ckpt.exists()
    assert math.isfinite(trainer.metrics_log[-1]["train_loss"])
