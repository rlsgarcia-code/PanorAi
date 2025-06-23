import importlib.util
import sys
from types import ModuleType
from pathlib import Path
from contextlib import contextmanager
import math
import pytest

torch = pytest.importorskip("torch")

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
