import importlib
import sys
from types import ModuleType
from pathlib import Path
from contextlib import contextmanager
import math
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("open3d")

try:
    import torch
except Exception as e:  # pragma: no cover - only executed when torch import fails
    pytest.skip(
        f"Skipping full pipeline test because torch failed to import: {e}",
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
# Patch heavy train_utils dependency
# ---------------------------------------------------------------------------
train_utils_stub = ModuleType("panorai_models.training.train_utils")

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
sys.modules.setdefault("panorai_models.training.train_utils", train_utils_stub)

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

# Import DepthTrainer after stubbing
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DepthTrainer = importlib.import_module("panorai_models.trainers.depth_trainer").DepthTrainer

# Import packages to ensure registries are populated
import panorai.samplers
import panorai.blenders
import panorai.projections

from panorai.factory.panorai_factory import PanoraiFactory
from panorai.preprocessing.preprocessor import Preprocessor


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


def dummy_loss(pred, tgt, mask, **_):
    loss = torch.mean(torch.abs(pred - tgt))
    return loss, {"total": loss.item()}


class SingleSampleDataset(torch.utils.data.Dataset):
    def __init__(self, rgb, depth, n=4):
        self.rgb = torch.tensor(rgb.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        self.depth = torch.tensor(depth[None], dtype=torch.float32)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"rgb_image": self.rgb, "xyz_image": self.depth}


def test_full_pipeline(tmp_path):
    # Create simple equirectangular image
    data = np.zeros((8, 16, 3), dtype=np.uint8)
    data[..., 0] = 255

    eq = PanoraiFactory.create_data_from_array(data, "equirectangular")

    processed = Preprocessor.preprocess_eq(
        eq.data,
        shadow_angle=5.0,
        delta_lat=2.0,
        delta_lon=3.0,
        resize_factor=1.0,
    )
    eq.data = processed

    faceset = eq.to_gnomonic_face_set(fov=90)
    assert len(faceset) == 6

    eq_back = faceset.to_equirectangular(eq.shape)
    assert eq_back.shape == eq.shape

    def depth_model(img):
        return np.ones(img.shape[:2], dtype=np.float32)

    pcd = faceset.to_pcd(model=depth_model, eq_shape=eq.shape)
    assert pcd.points.shape[1] == 3
    assert pcd.colors.shape[1] == 3

    dataset = SingleSampleDataset(np.array(eq_back), pcd.radius_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = TinyCNN()
    trainer = DepthTrainer(
        model=model,
        trainloader=loader,
        valloader=loader,
        max_depth=10.0,
        loss_fn=dummy_loss,
        device="cpu",
        grad_accum=1,
        compile_model=False,
        verbose=False,
    )

    optim = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.set_accelerator(optim)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
        mp.setattr(torch.jit, "trace", lambda model, example: model)
        trainer.train(epochs=1, save_dir=tmp_path)

    ckpt = tmp_path / "epoch_000.pth"
    assert ckpt.exists()
    assert math.isfinite(trainer.metrics_log[-1]["train_loss"])
