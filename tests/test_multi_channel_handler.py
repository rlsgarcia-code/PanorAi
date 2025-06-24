import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import pytest
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture()
def MultiChannelHandler(monkeypatch):
    panorai_pkg = ModuleType("panorai")
    panorai_pkg.__path__ = [str(ROOT / "panorai")]
    monkeypatch.setitem(sys.modules, "panorai", panorai_pkg)

    data_pkg = ModuleType("panorai.data")
    data_pkg.__path__ = [str(ROOT / "panorai" / "data")]
    monkeypatch.setitem(sys.modules, "panorai.data", data_pkg)

    utils_pkg = ModuleType("panorai.utils")
    utils_pkg.__path__ = [str(ROOT / "panorai" / "utils")]
    monkeypatch.setitem(sys.modules, "panorai.utils", utils_pkg)

    data_utils_pkg = ModuleType("panorai.data.utils")
    data_utils_pkg.__path__ = [str(ROOT / "panorai" / "data" / "utils")]
    monkeypatch.setitem(sys.modules, "panorai.data.utils", data_utils_pkg)

    _load_module("panorai.utils.exceptions", "panorai/utils/exceptions.py")
    _load_module("panorai.data.utils.shape_manager", "panorai/data/utils/shape_manager.py")
    module = _load_module("panorai.data.multi_handler", "panorai/data/multi_handler.py")
    yield module.MultiChannelHandler

    for mod in [
        "panorai.data.multi_handler",
        "panorai.utils.exceptions",
        "panorai.data.utils.shape_manager",
        "panorai.data.utils",
        "panorai.utils",
        "panorai.data",
        "panorai",
    ]:
        sys.modules.pop(mod, None)

def test_stack_and_unstack_updates(MultiChannelHandler):
    Handler = MultiChannelHandler
    data = {"a": np.zeros((2, 2)), "b": np.ones((2, 2))}
    handler = Handler(data)

    stacked, keys, counts = handler.stack()
    assert stacked.shape == (2, 2, 2)
    assert keys == ["a", "b"]
    assert counts == [1, 1]

    stacked[..., 0] = 3
    stacked[..., 1] = 4
    handler.unstack(stacked, keys, counts)

    assert handler.data["a"].shape == (2, 2, 1)
    assert handler.data["b"].shape == (2, 2, 1)
    assert np.all(handler.data["a"] == 3)
    assert np.all(handler.data["b"] == 4)

def test_apply_projection_single_and_multi(MultiChannelHandler):
    Handler = MultiChannelHandler
    proj = lambda arr: arr + 2

    single = Handler(np.zeros((2, 2)))
    out_single = single.apply_projection(proj)
    assert np.array_equal(out_single, np.full((2, 2), 2))
    assert np.array_equal(single.data, np.zeros((2, 2)))

    multi = Handler({"x": np.zeros((2, 2)), "y": np.zeros((2, 2))})
    out_multi = multi.apply_projection(proj)
    assert isinstance(out_multi, dict)
    for key in ["x", "y"]:
        assert np.array_equal(out_multi[key], np.full((2, 2, 1), 2))
        assert np.array_equal(multi.data[key], np.full((2, 2, 1), 2))
