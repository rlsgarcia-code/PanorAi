from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import numpy as np
import pytest

from panorai.geometry import (
    CUBE_FACE_ORDER,
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    cubemap_to_equirectangular,
    equirectangular_to_gnomonic,
)
from panorai.geometry import _engine


def _faces(shape=(13, 17), *, dtype=np.float32, channels=3):
    result = {}
    for index, face in enumerate(CUBE_FACE_ORDER):
        rng = np.random.default_rng(100 + index)
        array = rng.random((*shape, channels), dtype=np.float64).astype(dtype)
        result[face] = array if channels > 1 else array[..., 0]
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_selective_numpy_cubemap_matches_full_oracle_exactly(
    dtype, channels, interpolation
) -> None:
    faces = _faces(dtype=dtype, channels=channels)
    output_shape = (31, 64)
    first = faces[CUBE_FACE_ORDER[0]]
    selective = cubemap_to_equirectangular(
        faces, output_shape, interpolation=interpolation
    )
    full_plan = _engine._cubemap_full_back_plan(output_shape, first.shape[:2], first)
    full = _engine._cubemap_to_equirectangular_with_plan(
        faces,
        output_shape,
        interpolation=interpolation,
        invalid_policy="propagate",
        validity_masks=None,
        min_valid_weight=None,
        plan=full_plan,
    )

    np.testing.assert_array_equal(selective.data, full.data)
    np.testing.assert_array_equal(selective.support_mask, full.support_mask)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.bool_])
def test_selective_numpy_cubemap_preserves_nearest_modalities(dtype) -> None:
    faces = {
        face: np.full((9, 11), index % 2 if dtype == np.bool_ else index, dtype=dtype)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }

    result = cubemap_to_equirectangular(faces, (24, 48), interpolation="nearest")

    assert result.data.dtype == dtype
    assert set(np.unique(result.data)).issubset(
        {bool(index % 2) if dtype == np.bool_ else index for index in range(6)}
    )


def test_selective_numpy_matches_full_validity_and_weight() -> None:
    faces = _faces(dtype=np.float64, channels=2)
    masks = {}
    for index, face in enumerate(CUBE_FACE_ORDER):
        mask = np.ones((13, 17), dtype=bool)
        mask[(index + 1) :: 4, index % 3 :: 5] = False
        masks[face] = mask
        faces[face][~mask] = np.nan
    output_shape = (27, 56)
    first = faces[CUBE_FACE_ORDER[0]]
    selective = cubemap_to_equirectangular(
        faces,
        output_shape,
        invalid_policy="renormalize",
        validity_masks=masks,
        min_valid_weight=0.25,
    )
    full = _engine._cubemap_to_equirectangular_with_plan(
        faces,
        output_shape,
        interpolation="bilinear",
        invalid_policy="renormalize",
        validity_masks=masks,
        min_valid_weight=0.25,
        plan=_engine._cubemap_full_back_plan(output_shape, first.shape[:2], first),
    )

    np.testing.assert_array_equal(selective.data, full.data)
    np.testing.assert_array_equal(selective.validity_mask, full.validity_mask)
    np.testing.assert_array_equal(selective.valid_weight, full.valid_weight)


def test_projector_cache_is_private_bounded_and_lru() -> None:
    projector = GnomonicProjector(
        GnomonicSpec(output_shape_hw=(7, 9)), interpolation="nearest"
    )
    images = [np.zeros((8 + index, 16 + 2 * index), np.float32) for index in range(5)]
    for image in images[:4]:
        projector.project(image)
    assert len(projector._plans) == 4
    oldest_key = projector._plans.keys()[0]

    projector.project(images[0])
    assert projector._plans.keys()[-1] == oldest_key
    evicted_key = projector._plans.keys()[0]
    projector.project(images[4])

    assert len(projector._plans) == 4
    assert evicted_key not in projector._plans.keys()
    assert oldest_key in projector._plans.keys()


def test_cache_key_includes_direction_shape_dtype_backend_and_device() -> None:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    image32 = np.ones((12, 24), np.float32)
    image64 = image32.astype(np.float64)

    face = projector.project(image32).data
    projector.project(image64)
    projector.back_project(face, (12, 24))

    keys = projector._plans.keys()
    assert len(keys) == 3
    assert {key[0] for key in keys} == {"erp_to_gnomonic", "gnomonic_to_erp"}
    assert {key[-2] for key in keys} == {"<f4", "<f8"}
    assert {key[-1] for key in keys} == {"cpu"}


def test_cache_does_not_affect_repr_equality_or_hash() -> None:
    left = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    right = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    before = repr(left)
    left.project(np.ones((12, 24), np.float32))

    assert repr(left) == before == repr(right)
    assert left == right
    assert hash(left) == hash(right)


def test_returned_source_pixels_cannot_poison_cached_plan() -> None:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(7, 11)))
    image = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    first = projector.project(image, return_source_pixels=True)
    expected = first.source_pixels_xy.copy()
    first.source_pixels_xy[...] = -999.0

    second = projector.project(image, return_source_pixels=True)

    np.testing.assert_array_equal(second.source_pixels_xy, expected)
    assert second.source_pixels_xy is not first.source_pixels_xy


def test_returned_torch_source_pixels_are_cloned_from_cached_plan() -> None:
    torch = pytest.importorskip("torch")
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(7, 11)))
    image = torch.arange(16 * 32, dtype=torch.float32).reshape(1, 1, 16, 32)
    first = projector.project(image, return_source_pixels=True)
    expected = first.source_pixels_xy.clone()
    first.source_pixels_xy.fill_(-999.0)

    second = projector.project(image, return_source_pixels=True)

    assert torch.equal(second.source_pixels_xy, expected)
    assert second.source_pixels_xy.data_ptr() != first.source_pixels_xy.data_ptr()


def test_functional_api_retains_no_plan_between_calls(monkeypatch) -> None:
    calls = 0
    original = _engine._gnomonic_forward_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_engine, "_gnomonic_forward_plan", counted)
    image = np.ones((12, 24), np.float32)
    spec = GnomonicSpec(output_shape_hw=(5, 7))
    equirectangular_to_gnomonic(image, spec)
    equirectangular_to_gnomonic(image, spec)

    assert calls == 2


def test_plan_cache_is_thread_safe_for_one_projector() -> None:
    projector = CubemapProjector(CubemapSpec((9, 11)), interpolation="nearest")
    image = np.arange(24 * 48, dtype=np.float32).reshape(24, 48)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: projector.project(image), range(12)))

    assert len(projector._plans) == 1
    expected = results[0]["front"].data
    for result in results[1:]:
        np.testing.assert_array_equal(result["front"].data, expected)


def test_cached_plan_is_released_with_projector() -> None:
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(64, 96)))
    projector.project(np.ones((128, 256), np.float32))
    plan = next(iter(projector._plans._entries.values()))
    pixels = plan.pixels_xy
    reference = weakref.ref(pixels)

    del pixels, plan, projector
    gc.collect()

    assert reference() is None


def test_torch_grids_are_built_on_device_without_numpy_grid_roundtrip(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy grid construction was used by a Torch plan")

    monkeypatch.setattr(_engine, "_erp_rays_numpy", forbidden)
    monkeypatch.setattr(_engine, "_numpy_grid", forbidden)
    image = torch.ones((1, 2, 12, 24), dtype=torch.float32)
    gnomonic = GnomonicProjector(GnomonicSpec(output_shape_hw=(7, 9)))
    face = gnomonic.project(image)
    restored = gnomonic.back_project(face, image.shape[-2:])
    cubemap = CubemapProjector(CubemapSpec((7, 9)))
    faces = cubemap.project(image)
    cube_restored = cubemap.back_project(faces, image.shape[-2:])

    assert restored.data.device == image.device
    assert cube_restored.data.device == image.device


@pytest.mark.parametrize("layout", ["HW", "CHW", "NCHW"])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_torch_selective_cubemap_matches_cached_full_grid(
    layout, interpolation
) -> None:
    torch = pytest.importorskip("torch")
    shape = {"HW": (9, 11), "CHW": (2, 9, 11), "NCHW": (2, 2, 9, 11)}[layout]
    faces = {face: torch.rand(shape, dtype=torch.float64) for face in CUBE_FACE_ORDER}
    output_shape = (24, 48)
    first = faces[CUBE_FACE_ORDER[0]]
    selective = cubemap_to_equirectangular(
        faces, output_shape, interpolation=interpolation
    )
    full = _engine._cubemap_to_equirectangular_with_plan(
        faces,
        output_shape,
        interpolation=interpolation,
        invalid_policy="propagate",
        validity_masks=None,
        min_valid_weight=None,
        plan=_engine._cubemap_full_back_plan(output_shape, (9, 11), first),
    )

    assert torch.equal(selective.data, full.data)
    assert torch.equal(selective.support_mask, full.support_mask)


def test_torch_selective_cubemap_preserves_input_gradients() -> None:
    torch = pytest.importorskip("torch")
    selective_faces = {
        face: torch.rand((1, 2, 7, 9), dtype=torch.float64, requires_grad=True)
        for face in CUBE_FACE_ORDER
    }
    full_faces = {
        face: value.detach().clone().requires_grad_(True)
        for face, value in selective_faces.items()
    }
    output_shape = (18, 36)
    selective = cubemap_to_equirectangular(selective_faces, output_shape)
    full = _engine._cubemap_to_equirectangular_with_plan(
        full_faces,
        output_shape,
        interpolation="bilinear",
        invalid_policy="propagate",
        validity_masks=None,
        min_valid_weight=None,
        plan=_engine._cubemap_full_back_plan(
            output_shape, (7, 9), full_faces[CUBE_FACE_ORDER[0]]
        ),
    )
    selective.data.square().sum().backward()
    full.data.square().sum().backward()

    assert torch.equal(selective.data, full.data)
    for face in CUBE_FACE_ORDER:
        assert torch.allclose(
            selective_faces[face].grad,
            full_faces[face].grad,
            atol=1e-12,
            rtol=1e-12,
        )


def test_torch_cpu_projector_uses_selective_plan_after_measured_margin() -> None:
    torch = pytest.importorskip("torch")
    projector = CubemapProjector(CubemapSpec((7, 9)))
    faces = {
        face: torch.ones((1, 2, 7, 9), dtype=torch.float32) for face in CUBE_FACE_ORDER
    }

    projector.back_project(faces, (18, 36))

    plan = next(iter(projector._plans._entries.values()))
    assert isinstance(plan, _engine._CubemapSelectiveBackPlan)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_torch_device_has_distinct_cache_entry_when_available(device) -> None:
    torch = pytest.importorskip("torch")
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} is not available")
    projector = GnomonicProjector(GnomonicSpec(output_shape_hw=(5, 7)))
    projector.project(torch.ones((1, 1, 12, 24), device="cpu"))
    result = projector.project(torch.ones((1, 1, 12, 24), device=device))

    assert len(projector._plans) == 2
    assert result.data.device.type == device
