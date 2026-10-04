from __future__ import annotations

import copy
from collections.abc import Iterator
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import panorai as pa

# Resolve the lazy root exports during collection. A legacy trainer test later
# replaces ``sys.modules['panorai']`` with a local stub at collection time; the
# explicit resolution keeps these public-root API tests isolated from that
# unrelated compatibility fixture.
_ROOT_EXPORTS = (
    pa.EquirectangularImage,
    pa.GnomonicFace,
    pa.GnomonicFaceSet,
)


def _rgb(height=8, width=16):
    y, x = np.indices((height, width), dtype=np.float32)
    return np.stack((x / width, y / height, (x + y) / (height + width)), axis=-1)


def _multimodal():
    rgb = _rgb()
    depth = np.linspace(1.0, 4.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    valid = np.ones((8, 16), dtype=bool)
    valid[2:4, 5:7] = False
    labels = (np.arange(8 * 16).reshape(8, 16) % 5).astype(np.int16)
    return (
        pa.EquirectangularImage(rgb)
        .with_depth(depth, valid=valid, units="m")
        .with_labels(labels)
    )


def _snapshot_panorama(pano):
    return {
        name: value.copy()
        for name, value in pano._workflow_data().items()
    }, {
        name: pano.validity(name)
        for name in pano._workflow_metadata
    }


def test_construction_modalities_are_typed_direct_and_immutable():
    rgb = (_rgb() * 255).astype(np.uint8)
    depth = np.ones((8, 16), dtype=np.float32)
    valid = np.ones((8, 16), dtype=bool)
    valid[0, 0] = False
    labels = np.zeros((8, 16), dtype=np.int64)

    base = pa.EquirectangularImage(rgb)
    with_depth = base.with_depth(depth, valid=valid, units=None)
    result = with_depth.with_labels(labels)

    assert base.depth is None and base.labels is None
    assert with_depth.labels is None
    assert result.image is not rgb
    assert result.image.dtype == np.uint8
    assert result.depth.dtype == np.float32
    assert result.labels.dtype == np.int64
    assert np.array_equal(result.validity("depth"), valid)
    assert result._workflow_metadata["depth"]["units"] is None

    result.image[0, 0] = 255
    result.depth[0, 0] = 9
    assert not np.array_equal(base.image[0, 0], result.image[0, 0])
    assert depth[0, 0] == 1


def test_modality_validation_rejects_unsafe_inputs():
    pano = pa.EquirectangularImage(_rgb())
    with pytest.raises(TypeError, match="floating-point"):
        pano.with_depth(np.ones((8, 16), dtype=np.uint16))
    with pytest.raises(TypeError, match="integer or boolean"):
        pano.with_labels(np.ones((8, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="spatial shape"):
        pano.with_depth(np.ones((7, 16), dtype=np.float32))
    bad = np.ones((8, 16), dtype=np.float32)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="marked as valid"):
        pano.with_depth(bad)
    valid = np.ones((8, 16), dtype=bool)
    valid[1, 1] = False
    accepted = pano.with_depth(bad, valid=valid)
    assert np.isnan(accepted.depth[1, 1])
    bad_image = _rgb()
    bad_image[0, 0] = np.nan
    legacy_compatible = pa.EquirectangularImage(bad_image)
    with pytest.raises(ValueError, match="marked as valid"):
        legacy_compatible.views()


def test_legacy_dictionary_is_preserved_but_rejected_by_new_workflow():
    legacy = pa.EquirectangularImage({"rgb": _rgb(), "other": _rgb()})
    assert legacy.is_multi_channel()
    with pytest.raises(TypeError, match="semantically typed"):
        legacy.views()


@pytest.mark.parametrize(
    ("layout", "kwargs", "count"),
    [
        ("cube", {}, 6),
        ("fibonacci", {}, 20),
        ("fibonacci", {"count": 7}, 7),
        ("spiral", {}, 20),
        ("spiral", {"count": 9}, 9),
        ("icosahedron", {}, 12),
        ("icosahedron", {"subdivisions": 1}, 42),
    ],
)
def test_view_presets_are_deterministic(layout, kwargs, count):
    pano = pa.EquirectangularImage(_rgb(10, 22))
    first = pano.views(layout, **kwargs)
    second = pano.views(layout, **kwargs)
    assert len(first) == count
    assert first.describe()["specs"] == second.describe()["specs"]
    assert first.describe()["view_shape_hw"] == (5, 6)
    assert first.describe()["view_order"] == second.describe()["view_order"]


def test_explicit_size_rectangular_fov_rotations_and_sampler_object():
    pano = pa.EquirectangularImage(_rgb())
    views = pano.views("cube", size=(3, 5), fov=(100, 70), rotations=((0, 5),))
    assert len(views) == 12
    assert views[0].spec.output_shape_hw == (3, 5)
    assert views[0].hfov_deg == 100
    assert views[0].vfov_deg == 70
    twice = pano.views("cube", size=3, rotations=((0, 5), (0, 10)))
    assert len(twice) == 18

    class Sampler:
        def get_tangent_points(self):
            return [(0, 0), (10, 20)]

    custom = pano.views(Sampler(), size=4)
    assert len(custom) == 2
    assert custom.describe()["layout"] == "Sampler"

    from panorai.geometry import GnomonicProjector, GnomonicSpec

    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(1, 1)))
    injected = pano.views("cube", size=4, projector=projector)
    assert injected.describe()["projector"] == "GnomonicProjector"
    assert injected.reconstruct().image.shape == pano.image.shape
    default = pano.views("cube", size=4)
    assert np.allclose(injected[0].image, default[0].image)
    with pytest.raises(TypeError, match="GnomonicProjector"):
        pano.views("cube", size=4, projector=object())


def test_projection_is_modality_specific_and_inputs_do_not_change():
    pano = _multimodal()
    data_before, validity_before = _snapshot_panorama(pano)
    views = pano.views(size=5)

    assert views[0].image.dtype == np.float32
    assert views[0].depth.dtype == np.float32
    assert views[0].labels.dtype == np.int16
    assert set(np.unique(views[0].labels)).issubset(set(np.unique(pano.labels)))
    assert np.any(np.modf(views[0].depth[np.isfinite(views[0].depth)])[0] != 0)
    for name, original in data_before.items():
        assert np.array_equal(pano._workflow_data()[name], original, equal_nan=True)
        assert np.array_equal(pano.validity(name), validity_before[name])


def test_depth_policy_contract_requires_explicit_threshold():
    pano = _multimodal()
    with pytest.raises(ValueError, match="min_valid_weight is required"):
        pano.views(depth_policy="renormalize")
    with pytest.raises(ValueError, match="only valid"):
        pano.views(min_valid_weight=0.5)
    for invalid in (0.0, 1.1, float("nan")):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            pano.views(
                depth_policy="renormalize",
                min_valid_weight=invalid,
            )
    with pytest.raises(TypeError, match="finite real"):
        pano.views(depth_policy="renormalize", min_valid_weight=True)
    views = pano.views(
        size=4,
        depth_policy="renormalize",
        min_valid_weight=0.75,
    )
    description = views.describe()
    assert description["depth_policy"] == "renormalize"
    assert description["min_valid_weight"] == 0.75


def test_map_array_and_validity_output_collision_and_immutability():
    views = pa.EquirectangularImage(_rgb()).views(size=4)
    original = views[0].image.copy()

    mapped = views.map(lambda image: image * 2)
    assert np.array_equal(views[0].image, original)
    assert np.allclose(mapped[0].image, original * 2)

    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    depth = views.map(
        lambda image: (image[..., 0], valid),
        output="depth",
        units="m",
    )
    assert depth[0].depth.shape == (4, 4)
    assert not depth[0].validity("depth")[0, 0]
    assert depth[0].image is not None
    with pytest.raises(ValueError, match="already exists"):
        depth.map(lambda image: image[..., 0], input="image", output="depth")
    replaced = depth.map(
        lambda image: image[..., 0] + 1,
        input="image",
        output="depth",
        replace=True,
    )
    assert replaced[0].depth[1, 1] > depth[0].depth[1, 1]


def test_map_array_only_validity_is_scoped_to_geometric_support():
    support = np.ones((8, 16), dtype=bool)
    support[2:4, 5:7] = False
    views = pa.EquirectangularImage(_rgb(), support_mask=support).views(size=6)

    face_supports = [
        face._workflow_support["image"] for face in views
    ]
    call_index = 0

    def model(image):
        nonlocal call_index
        output = np.nan_to_num(image)
        output[~face_supports[call_index]] = np.nan
        call_index += 1
        return output

    mapped = views.map(model)

    assert np.array_equal(
        mapped[0].validity("image"),
        mapped[0]._workflow_support["image"],
    )
    assert not mapped[0].validity("image").all()
    assert np.isfinite(
        mapped[0].image[mapped[0].validity("image")]
    ).all()


def test_map_requires_input_for_bundle_and_validates_model_contract():
    views = _multimodal().views(size=4)
    with pytest.raises(ValueError, match="input is required"):
        views.map(lambda value: value)
    with pytest.raises(ValueError, match="spatial shape"):
        views.map(lambda value: value[:-1], input="depth")
    with pytest.raises(ValueError, match="non-finite"):
        views.map(
            lambda value: np.full_like(value, np.nan),
            input="depth",
        )
    with pytest.raises(ValueError, match="tuple results"):
        views.map(lambda value: (value, value, value), input="depth")


def test_reconstruct_defaults_preserve_modalities_and_process_views_is_exact():
    pano = _multimodal()
    expanded_views = pano.views(size=6)
    processed = expanded_views.map(lambda image: image + 0.25, input="image")
    expanded = processed.reconstruct()
    shortcut = pano.process_views(
        lambda image: image + 0.25,
        size=6,
        input="image",
    )

    assert expanded.image.shape == pano.image.shape
    assert expanded.depth.shape == pano.depth.shape
    assert expanded.labels.shape == pano.labels.shape
    assert expanded.labels.dtype == pano.labels.dtype
    assert np.array_equal(expanded.validity("depth"), shortcut.validity("depth"))
    for name in ("image", "depth"):
        assert np.allclose(
            expanded._workflow_data()[name],
            shortcut._workflow_data()[name],
            equal_nan=True,
        )
    assert np.array_equal(expanded.labels, shortcut.labels)


def test_reconstruct_selection_blends_and_manual_shape_requirement():
    views = _multimodal().views(size=5)
    labels = views.reconstruct(modalities="labels", blend="closest")
    assert labels.image is None
    assert labels.labels.shape == (8, 16)
    image = views.reconstruct(modalities="image", blend="gaussian")
    assert image.image.shape == (8, 16, 3)
    with pytest.raises(ValueError, match="mapping by modality"):
        views.reconstruct(blend="average")

    manual = pa.GnomonicFaceSet([pa.GnomonicFace(np.ones((2, 2)), 0, 0, 90)])
    with pytest.raises(ValueError, match="eq_shape is required"):
        manual.reconstruct()


def test_describe_and_repr_report_executed_choices_exactly():
    views = _multimodal().views("fibonacci", count=3, size=(4, 6), fov=(80, 70))
    description = views.describe()
    assert description["contract"] == "geometry-v1"
    assert description["interface"] == "panorai-object-workflow/v1"
    assert description["stability"] == "stable"
    assert description["layout"] == "fibonacci"
    assert description["view_count"] == 3
    assert description["view_shape_hw"] == (4, 6)
    assert description["specs"][0]["hfov_deg"] == 80
    assert description["specs"][0]["vfov_deg"] == 70
    assert description["modalities"]["labels"]["interpolation"] == "nearest"
    assert description["modalities"]["labels"]["blend"] == "closest"
    assert description["modalities"]["depth"]["units"] == "m"
    assert "layout='fibonacci'" in repr(views)


def test_face_set_iteration_is_reentrant():
    views = pa.EquirectangularImage(_rgb()).views(size=3)
    pairs = [(left.lat, right.lon) for left in views for right in views]
    assert len(pairs) == len(views) ** 2
    assert len(list(views)) == len(views)
    assert len(list(views)) == len(views)


def test_face_set_preserves_legacy_iterator_protocol():
    views = pa.EquirectangularImage(_rgb()).views(size=3)

    assert isinstance(views, Iterator)
    assert next(views) is views[0]
    assert next(views) is views[1]
    assert len(list(views)) == len(views)


def test_importing_numpy_workflow_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
import numpy as np
import panorai as pa
p = pa.EquirectangularImage(np.zeros((4, 8, 3), dtype=np.float32))
v = p.views(size=2)
assert 'torch' not in sys.modules
print(json.dumps(v.describe()['modalities']['image']))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["backend"] == "numpy"


@pytest.mark.parametrize("shape", [(8, 16), (3, 8, 16), (2, 3, 8, 16)])
def test_torch_hw_chw_nchw_preserve_backend_batch_dtype_device(shape):
    torch = pytest.importorskip("torch")
    image = torch.rand(shape, dtype=torch.float32)
    pano = pa.EquirectangularImage(image)
    views = pano.views(size=(4, 5))
    mapped = views.map(lambda value: value + 1)
    result = mapped.reconstruct()
    assert isinstance(result.image, torch.Tensor)
    assert tuple(result.image.shape) == shape
    assert result.image.dtype == image.dtype
    assert result.image.device == image.device
    assert views.describe()["backend"] == "torch"


def test_torch_multimodal_batch_and_mixed_backend_rejection():
    torch = pytest.importorskip("torch")
    image = torch.rand((2, 3, 8, 16))
    depth = torch.ones((2, 1, 8, 16))
    valid = torch.ones((2, 8, 16), dtype=torch.bool)
    valid[:, 0, 0] = False
    labels = torch.zeros((2, 1, 8, 16), dtype=torch.int64)
    pano = pa.EquirectangularImage(image).with_depth(depth, valid=valid).with_labels(labels)
    model_validity = torch.zeros((2, 4, 4), dtype=torch.bool)
    result = pano.views(size=4).map(
        lambda value: (value + 2, model_validity),
        input="depth",
    ).reconstruct()
    assert tuple(result.depth.shape) == tuple(depth.shape)
    assert tuple(result.validity("depth").shape) == tuple(valid.shape)
    assert result.labels.dtype == labels.dtype
    assert result._workflow_metadata["depth"]["units"] == "m"
    assert result.support_mask.all()
    assert not result.validity("depth").any()
    with pytest.raises(ValueError, match="share backend"):
        pa.EquirectangularImage(_rgb()).with_depth(depth[0, 0])


def test_torch_image_validity_support_and_clone_are_backend_safe():
    torch = pytest.importorskip("torch")
    source = torch.rand((2, 3, 8, 16), requires_grad=True)
    valid = torch.ones((2, 8, 16), dtype=torch.bool)
    valid[:, 1, 2] = False
    support = torch.ones((2, 8, 16), dtype=torch.bool)
    support[:, 0, 0] = False

    pano = pa.EquirectangularImage(source, support_mask=support, valid=valid)
    assert torch.equal(pano.validity("image"), valid)
    assert torch.equal(pano.support_mask, support)
    assert not torch.equal(pano.validity("image"), pano.support_mask)

    cloned = pano.clone()
    assert cloned is not pano
    assert cloned.image.data_ptr() != pano.image.data_ptr()
    cloned.image.sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()

    views = pano.views(size=4)
    cloned_face = views[0].clone()
    assert cloned_face.image.data_ptr() != views[0].image.data_ptr()
    copied_set = pa.GnomonicFaceSet([])
    copied_set.add_face(views[0])
    assert copied_set[0].image.data_ptr() != views[0].image.data_ptr()


@pytest.mark.parametrize("shape", [(8, 16), (3, 8, 16), (2, 3, 8, 16)])
def test_torch_explicit_image_validity_layouts(shape):
    torch = pytest.importorskip("torch")
    image = torch.rand(shape)
    valid_shape = shape[-2:] if len(shape) < 4 else (shape[0], *shape[-2:])
    valid = torch.ones(valid_shape, dtype=torch.bool)
    pano = pa.EquirectangularImage(image, valid=valid)
    assert torch.equal(pano.validity("image"), valid)
    assert tuple(pano.views(size=3)[0].validity("image").shape) == tuple(valid_shape[:-2] + (3, 3))
