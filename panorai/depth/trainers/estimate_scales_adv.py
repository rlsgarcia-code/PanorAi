import torch
import numpy as np
import matplotlib.pyplot as plt
from panorai.depth.trainers.metrics import MonocularDepthMetrics

def optimize_scale_for_metrics(
    model,
    dataloader,
    device,
    max_depth,
    robust=True,
    plot=True,
    mae_weight=1.0,
    rmse_weight=1.0,
    threshold_weight=2.0,
):
    """
    Optimize global scale by minimizing a normalized combined metric.
    Prints detailed step-by-step metrics during the sweep.
    """
    model.eval()

    pred_all = []
    target_all = []

    with torch.no_grad():
        for sample in dataloader:
            rgb = sample["rgb_image"].to(device, non_blocking=True)
            xyz = sample["xyz_image"].to(device, non_blocking=True)

            pred = model(rgb).unsqueeze(1)

            valid = (
                (xyz > 1e-3)
                & (xyz < max_depth)
                & torch.isfinite(xyz)
                & torch.isfinite(pred)
            )

            if valid.sum() == 0:
                continue

            pred_valid = pred[valid]
            xyz_valid = xyz[valid]

            if robust:
                errors = (pred_valid - xyz_valid).abs()
                lower = torch.quantile(errors, 0.05)
                upper = torch.quantile(errors, 0.95)
                mask = (errors >= lower) & (errors <= upper)
                pred_valid = pred_valid[mask]
                xyz_valid = xyz_valid[mask]

            pred_all.append(pred_valid.cpu().numpy())
            target_all.append(xyz_valid.cpu().numpy())

    pred_all_flat = np.concatenate(pred_all)
    target_all_flat = np.concatenate(target_all)

    # 🔥 Sweep scales
    scales = np.linspace(0.5, 1.5, 200)
    all_mae, all_rmse, all_thresh = [], [], []

    print("\n🔎 Sweeping Scales:")
    print(f"{'Scale':>8} | {'MAE':>8} | {'RMSE':>8} | {'δ<1.25':>8}")

    for s in scales:
        temp_metrics = MonocularDepthMetrics()
        temp_metrics.update(pred_all_flat * s, target_all_flat, np.ones_like(pred_all_flat))
        m = temp_metrics.compute()
        all_mae.append(m["MAE"])
        all_rmse.append(m["RMSE"])
        all_thresh.append(m["Threshold Accuracy (δ=1.25)"])

        print(f"{s:8.4f} | {m['MAE']:8.4f} | {m['RMSE']:8.4f} | {m['Threshold Accuracy (δ=1.25)']:8.4f}")

    all_mae = np.array(all_mae)
    all_rmse = np.array(all_rmse)
    all_thresh = np.array(all_thresh)

    # --- Normalize each term ---
    mae_norm = (all_mae - all_mae.min()) / (all_mae.max() - all_mae.min() + 1e-8)
    rmse_norm = (all_rmse - all_rmse.min()) / (all_rmse.max() - all_rmse.min() + 1e-8)
    thresh_norm = (all_thresh - all_thresh.min()) / (all_thresh.max() - all_thresh.min() + 1e-8)

    total_score = (
        mae_weight * mae_norm +
        rmse_weight * rmse_norm +
        threshold_weight * (1.0 - thresh_norm)
    )

    best_idx = np.argmin(total_score)
    best_scale = scales[best_idx]

    print(f"\n✅ Optimized global scale (normalized metrics): {best_scale:.6f}")
    print(f"Lowest combined score: {total_score[best_idx]:.6f}")

    # 🖼️ Plot (optional)
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(scales, total_score, label="Total Score (normalized)")
        ax.axvline(best_scale, color="red", linestyle="--", label=f"Best Scale: {best_scale:.4f}")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Combined Normalized Score")
        ax.set_title("Scale Optimization (Normalized Metrics)")
        ax.legend()
        ax.grid(True)
        plt.show()

    return best_scale