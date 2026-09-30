import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Provide a lightweight NumPy stub for the validation module
# ---------------------------------------------------------------------------
class Array:
    def __init__(self, shape):
        self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)


numpy_stub = ModuleType("numpy")
numpy_stub.ndarray = Array
numpy_stub.zeros = lambda shape: Array(shape)
numpy_stub.ones = lambda shape: Array(shape)
sys.modules.setdefault("numpy", numpy_stub)
np = numpy_stub

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

# Minimal package structure to avoid executing panorai.__init__
panorai_pkg = ModuleType("panorai")
panorai_pkg.__path__ = [str(ROOT / "panorai")]
sys.modules.setdefault("panorai", panorai_pkg)

utils_pkg = ModuleType("panorai.utils")
utils_pkg.__path__ = [str(ROOT / "panorai" / "utils")]
sys.modules.setdefault("panorai.utils", utils_pkg)

exceptions = _load_module("panorai.utils.exceptions", "panorai/utils/exceptions.py")
validation = _load_module("panorai.utils.validation", "panorai/utils/validation.py")

validate_image_data = validation.validate_image_data
validate_gnomonic_data = validation.validate_gnomonic_data
ChannelMismatchError = exceptions.ChannelMismatchError
InvalidDataError = exceptions.InvalidDataError


def test_validate_image_data_valid():
    validate_image_data(np.zeros((2, 2)))
    validate_image_data(np.zeros((2, 2, 3)))
    validate_image_data({"r": np.zeros((4, 4)), "g": np.ones((4, 4))})


def test_validate_image_data_channel_mismatch():
    data = {"a": np.zeros((2, 2)), "b": np.zeros((3, 3))}
    with pytest.raises(ChannelMismatchError):
        validate_image_data(data)


def test_validate_image_data_invalid_type():
    with pytest.raises(InvalidDataError):
        validate_image_data([1, 2, 3])


def test_validate_gnomonic_data_valid():
    validate_gnomonic_data(np.zeros((3, 3)))
    validate_gnomonic_data(np.zeros((3, 3, 1)))
    validate_gnomonic_data({"x": np.zeros((5, 5)), "y": np.zeros((5, 5))})


def test_validate_gnomonic_data_channel_mismatch():
    data = {"left": np.zeros((2, 3)), "right": np.zeros((3, 3))}
    with pytest.raises(ChannelMismatchError):
        validate_gnomonic_data(data)


def test_validate_gnomonic_data_invalid_type():
    with pytest.raises(InvalidDataError):
        validate_gnomonic_data("invalid")
