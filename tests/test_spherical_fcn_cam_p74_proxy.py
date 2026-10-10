import numpy as np
import pytest
import torch

from benchmarks.spherical_fcn_cam.run_p74_imagenet_proxy_study import (
    PROXY_CANDIDATES,
    VISUALIZED_INDICES,
    masked_class_activation_map,
    masked_cam_statistics,
    spherical_masked_average,
)
from benchmarks.spherical_fcn_cam.run_p74_internimage_proxy_study import (
    compare_to_resnet18,
    write_dense_proxy_evidence,
)


def test_spherical_masked_average_ignores_invalid_extreme_logits() -> None:
    logits = torch.tensor(
        [[[[2.0, 2.0], [999.0, 999.0]], [[-3.0, -3.0], [-999.0, -999.0]]]]
    )
    support = torch.tensor([[[[True, True], [False, False]]]])

    actual = spherical_masked_average(logits, support)

    torch.testing.assert_close(actual, torch.tensor([[2.0, -3.0]]))


def test_spherical_masked_average_rejects_empty_support() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        spherical_masked_average(
            torch.ones((1, 2, 2, 2)), torch.zeros((1, 1, 2, 2), dtype=torch.bool)
        )


def test_masked_cam_normalizes_only_valid_support_and_zeros_invalid_pixels() -> None:
    logits = torch.tensor([[[[1.0, 3.0], [100.0, 100.0]]]])
    support = torch.tensor([[[[True, True], [False, False]]]])

    cam = masked_class_activation_map(logits, 0, support, output_shape=(2, 2))

    torch.testing.assert_close(cam, torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]))


def test_masked_helpers_validate_layouts_and_class_index() -> None:
    logits = torch.ones((1, 2, 2, 2))
    support = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="N1HW"):
        spherical_masked_average(logits, support[:, 0])
    with pytest.raises(ValueError, match="outside"):
        masked_class_activation_map(logits, 2, support, output_shape=(2, 2))


def test_masked_cam_statistics_exclude_unsupported_pixels() -> None:
    cam = np.asarray([[0.0, 1.0], [1.0, 1.0]])
    support = np.asarray([[True, True], [False, False]])

    statistics = masked_cam_statistics(cam, support)

    assert statistics["spherical_mean"] == pytest.approx(0.5)
    assert statistics["spherical_fraction_ge_0_5"] == pytest.approx(0.5)
    assert statistics["peak_latitude_degrees"] == pytest.approx(45.0)
    assert statistics["peak_longitude_degrees"] == pytest.approx(90.0)


def test_internimage_comparison_reports_rank_and_direction_agreement() -> None:
    def record(rank_offset: int, seconds: float) -> dict:
        predictions = []
        for position, (index, name) in enumerate(PROXY_CANDIDATES.items(), start=1):
            item = {
                "class_index": index,
                "class_name": name,
                "rank": position + rank_offset,
            }
            if index in VISUALIZED_INDICES:
                item["cam_statistics"] = {
                    "peak_longitude_degrees": 20.0,
                    "peak_latitude_degrees": -10.0,
                }
            predictions.append(item)
        return {
            "sample_id": "W050",
            "inference_seconds": seconds,
            "top_20": [{"class_index": index} for index in list(PROXY_CANDIDATES)[:20]],
            "candidate_predictions": predictions,
        }

    comparison = compare_to_resnet18(
        [record(0, 4.0)],
        {"schema": "baseline/v1", "records": [record(1, 2.0)]},
    )

    assert comparison["samples"][0]["top_20_jaccard"] == 1.0
    assert comparison["samples"][0]["inference_time_ratio_vs_resnet18"] == 2.0
    assert (
        comparison["candidates"][0]["median_rank_delta_internimage_minus_resnet18"]
        == -1.0
    )
    visualized = next(
        item
        for item in comparison["candidates"]
        if item["class_index"] in VISUALIZED_INDICES
    )
    assert visualized["mean_peak_direction_distance_degrees"] < 1e-5


def test_dense_proxy_export_preserves_declared_order_and_native_lattice(
    tmp_path,
) -> None:
    logits = torch.arange(1000 * 2 * 3, dtype=torch.float32).reshape(1, 1000, 2, 3)
    support = torch.tensor([[[[True, True, False], [True, False, False]]]])

    report = write_dense_proxy_evidence(tmp_path / "evidence.npz", logits, support)
    with np.load(tmp_path / "evidence.npz") as stored:
        indices = stored["class_indices"]
        assert indices.tolist() == list(PROXY_CANDIDATES)
        assert stored["class_names"].tolist() == list(PROXY_CANDIDATES.values())
        np.testing.assert_array_equal(stored["logits"], logits[0, indices].numpy())
        np.testing.assert_array_equal(stored["support"], support[0, 0].numpy())
        assert stored["schema"].item() == "panorai-internimage-proxy-evidence/v1"
    assert report["shape"] == [len(PROXY_CANDIDATES), 2, 3]
    assert report["dtype"] == "float32"
