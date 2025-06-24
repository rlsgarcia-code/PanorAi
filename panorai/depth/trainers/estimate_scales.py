import torch
import numpy as np
import matplotlib.pyplot as plt
from panorai.depth.trainers.metrics import MonocularDepthMetrics

def estimate_best_scale(model, dataloader, device, max_depth, robust=True, plot=True):
    """
    Estimate optimal global scale for depth predictions and display metrics before/after scaling.
    """
    print('starting...')
    model.eval()
    running_metrics_before = MonocularDepthMetrics()
    running_metrics_after = MonocularDepthMetrics()

    numerator_sum = 0.0
    denominator_sum = 0.0
    skipped_batches = 0

    pred_all = []
    target_all = []

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
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
                skipped_batches += 1
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

            numerator_sum += (pred_valid * xyz_valid).sum().item()
            denominator_sum += (pred_valid ** 2).sum().item()

            pred_np = pred_valid.cpu().numpy()
            xyz_np = xyz_valid.cpu().numpy()
            mask_np = np.ones_like(pred_np)

            pred_all.append(pred_np)
            target_all.append(xyz_np)

            # 🔥 Update running metric BEFORE scaling
            running_metrics_before.update(pred_np, xyz_np, mask_np)

            # 🔥 Print selected metrics after each batch
            batch_metrics = running_metrics_before._compute_batch(pred_np, xyz_np, mask_np)
            print(f"[Batch {idx:03d}] ➔ MAE: {batch_metrics['MAE']:.4f}, RMSE: {batch_metrics['RMSE']:.4f}, "
                  f"Threshold@1.25: {batch_metrics['Threshold Accuracy (δ=1.25)']:.4f}")

    if denominator_sum == 0:
        raise RuntimeError("No valid samples found to estimate scale.")

    best_scale = numerator_sum / denominator_sum
    print(f"\n✅ Estimated best global scale: {best_scale:.6f}")

    pred_all_flat = np.concatenate(pred_all)
    target_all_flat = np.concatenate(target_all)

    # Update metrics after scaling
    running_metrics_after.update(pred_all_flat * best_scale, target_all_flat, np.ones_like(pred_all_flat))

    metrics_before = running_metrics_before.compute()
    metrics_after = running_metrics_after.compute()

    # --- Pretty print as table ---
    print("\n📋 Monocular Depth Metrics (Before vs After Scaling)")
    print(f"{'Metric':<40} | {'Before Scaling':>15} | {'After Scaling':>15}")
    print("-" * 75)
    for k in metrics_before.keys():
        print(f"{k:<40} | {metrics_before[k]:>15.6f} | {metrics_after[k]:>15.6f}")

    # --- Optional Plot: sweep scales ---
    if plot:
        scales = np.linspace(0.5 * best_scale, 1.5 * best_scale, 100)
        mae_vals, rmse_vals = [], []

        for s in scales:
            temp_metrics = MonocularDepthMetrics()
            temp_metrics.update(pred_all_flat * s, target_all_flat, np.ones_like(pred_all_flat))
            m = temp_metrics.compute()
            mae_vals.append(m["MAE"])
            rmse_vals.append(m["RMSE"])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(scales, mae_vals, label="MAE")
        ax[0].set_xlabel("Scale")
        ax[0].set_ylabel("MAE")
        ax[0].set_title("MAE vs Scale")
        ax[0].grid(True)

        ax[1].plot(scales, rmse_vals, label="RMSE", color="orange")
        ax[1].set_xlabel("Scale")
        ax[1].set_ylabel("RMSE")
        ax[1].set_title("RMSE vs Scale")
        ax[1].grid(True)

        plt.suptitle("Error vs Scale Sweep")
        plt.show()

    return best_scale


# vits - vkitti 
# [Batch 000] ➔ MAE: 8.9976, RMSE: 10.6388, Threshold@1.25: 0.0034
# [Batch 001] ➔ MAE: 7.3705, RMSE: 8.4862, Threshold@1.25: 0.0539
# [Batch 002] ➔ MAE: 8.5107, RMSE: 9.3536, Threshold@1.25: 0.0000
# [Batch 003] ➔ MAE: 7.7446, RMSE: 8.5850, Threshold@1.25: 0.0004
# [Batch 004] ➔ MAE: 8.3726, RMSE: 9.8455, Threshold@1.25: 0.0007
# [Batch 005] ➔ MAE: 12.7063, RMSE: 15.2494, Threshold@1.25: 0.0000
# [Batch 006] ➔ MAE: 8.7642, RMSE: 9.9018, Threshold@1.25: 0.0001
# [Batch 007] ➔ MAE: 8.5178, RMSE: 9.4808, Threshold@1.25: 0.0001
# [Batch 008] ➔ MAE: 7.7298, RMSE: 8.4373, Threshold@1.25: 0.0000
# [Batch 009] ➔ MAE: 6.6480, RMSE: 7.4339, Threshold@1.25: 0.0012
# [Batch 010] ➔ MAE: 7.6968, RMSE: 8.3327, Threshold@1.25: 0.0000
# [Batch 011] ➔ MAE: 7.6813, RMSE: 8.2763, Threshold@1.25: 0.0000
# [Batch 012] ➔ MAE: 8.1639, RMSE: 8.8679, Threshold@1.25: 0.0000
# [Batch 013] ➔ MAE: 10.2002, RMSE: 11.9927, Threshold@1.25: 0.0000
# [Batch 014] ➔ MAE: 7.6242, RMSE: 8.5716, Threshold@1.25: 0.0006
# [Batch 015] ➔ MAE: 8.0665, RMSE: 8.7659, Threshold@1.25: 0.0004
# [Batch 016] ➔ MAE: 8.6451, RMSE: 9.2322, Threshold@1.25: 0.0000
# [Batch 017] ➔ MAE: 8.8801, RMSE: 9.6231, Threshold@1.25: 0.0001
# [Batch 018] ➔ MAE: 7.9777, RMSE: 8.6382, Threshold@1.25: 0.0011
# [Batch 019] ➔ MAE: 8.7988, RMSE: 9.5209, Threshold@1.25: 0.0000
# [Batch 020] ➔ MAE: 9.4369, RMSE: 10.0778, Threshold@1.25: 0.0000
# [Batch 021] ➔ MAE: 8.1795, RMSE: 8.9530, Threshold@1.25: 0.0000
# [Batch 022] ➔ MAE: 8.7086, RMSE: 10.0106, Threshold@1.25: 0.0002
# [Batch 023] ➔ MAE: 8.4706, RMSE: 9.1163, Threshold@1.25: 0.0000
# [Batch 024] ➔ MAE: 10.7608, RMSE: 11.7140, Threshold@1.25: 0.0000
# [Batch 025] ➔ MAE: 8.2676, RMSE: 9.1005, Threshold@1.25: 0.0000
# [Batch 026] ➔ MAE: 8.4975, RMSE: 9.6011, Threshold@1.25: 0.0010
# [Batch 027] ➔ MAE: 8.3504, RMSE: 8.9348, Threshold@1.25: 0.0000
# [Batch 028] ➔ MAE: 7.2693, RMSE: 7.6696, Threshold@1.25: 0.0000
# [Batch 029] ➔ MAE: 8.6185, RMSE: 9.1238, Threshold@1.25: 0.0000
# [Batch 030] ➔ MAE: 8.8810, RMSE: 9.3533, Threshold@1.25: 0.0000
# [Batch 031] ➔ MAE: 8.7946, RMSE: 10.4737, Threshold@1.25: 0.0000
# [Batch 032] ➔ MAE: 8.0891, RMSE: 8.9569, Threshold@1.25: 0.0000
# [Batch 033] ➔ MAE: 10.3998, RMSE: 12.1816, Threshold@1.25: 0.0000
# [Batch 034] ➔ MAE: 7.5560, RMSE: 8.3268, Threshold@1.25: 0.0015
# [Batch 035] ➔ MAE: 9.1049, RMSE: 10.1069, Threshold@1.25: 0.0000
# [Batch 036] ➔ MAE: 8.2404, RMSE: 9.4570, Threshold@1.25: 0.0014
# [Batch 037] ➔ MAE: 8.0747, RMSE: 8.6145, Threshold@1.25: 0.0000
# [Batch 038] ➔ MAE: 8.3378, RMSE: 9.0366, Threshold@1.25: 0.0000
# [Batch 039] ➔ MAE: 11.2442, RMSE: 14.1607, Threshold@1.25: 0.0000
# [Batch 040] ➔ MAE: 11.3234, RMSE: 12.8905, Threshold@1.25: 0.0000
# [Batch 041] ➔ MAE: 9.3365, RMSE: 10.6806, Threshold@1.25: 0.0007
# [Batch 042] ➔ MAE: 8.7397, RMSE: 9.8808, Threshold@1.25: 0.0001
# [Batch 043] ➔ MAE: 9.2294, RMSE: 10.1011, Threshold@1.25: 0.0000
# [Batch 044] ➔ MAE: 8.8955, RMSE: 10.0270, Threshold@1.25: 0.0001
# [Batch 045] ➔ MAE: 9.1667, RMSE: 10.1461, Threshold@1.25: 0.0001
# [Batch 046] ➔ MAE: 7.7857, RMSE: 9.1326, Threshold@1.25: 0.0208
# [Batch 047] ➔ MAE: 7.9152, RMSE: 8.6695, Threshold@1.25: 0.0000
# [Batch 048] ➔ MAE: 9.3871, RMSE: 10.9178, Threshold@1.25: 0.0003
# [Batch 049] ➔ MAE: 7.6246, RMSE: 8.6969, Threshold@1.25: 0.0001
# [Batch 050] ➔ MAE: 8.5806, RMSE: 9.7975, Threshold@1.25: 0.0001
# [Batch 051] ➔ MAE: 8.5873, RMSE: 9.4907, Threshold@1.25: 0.0000
# [Batch 052] ➔ MAE: 9.7305, RMSE: 10.8320, Threshold@1.25: 0.0000
# [Batch 053] ➔ MAE: 8.9008, RMSE: 9.6403, Threshold@1.25: 0.0000
# [Batch 054] ➔ MAE: 8.5400, RMSE: 9.0841, Threshold@1.25: 0.0000
# [Batch 055] ➔ MAE: 8.0953, RMSE: 8.7258, Threshold@1.25: 0.0000
# [Batch 056] ➔ MAE: 7.8157, RMSE: 8.4089, Threshold@1.25: 0.0000
# [Batch 057] ➔ MAE: 7.9931, RMSE: 8.4378, Threshold@1.25: 0.0000
# [Batch 058] ➔ MAE: 7.9680, RMSE: 8.4935, Threshold@1.25: 0.0000
# [Batch 059] ➔ MAE: 8.8904, RMSE: 9.6132, Threshold@1.25: 0.0000
# [Batch 060] ➔ MAE: 8.5015, RMSE: 9.2776, Threshold@1.25: 0.0000
# [Batch 061] ➔ MAE: 8.6684, RMSE: 9.2868, Threshold@1.25: 0.0000
# [Batch 062] ➔ MAE: 8.8191, RMSE: 9.5058, Threshold@1.25: 0.0000
# [Batch 063] ➔ MAE: 8.1804, RMSE: 8.9781, Threshold@1.25: 0.0000
# [Batch 064] ➔ MAE: 8.6181, RMSE: 9.4277, Threshold@1.25: 0.0000
# [Batch 065] ➔ MAE: 8.4455, RMSE: 9.6421, Threshold@1.25: 0.0063
# [Batch 066] ➔ MAE: 8.8802, RMSE: 9.5693, Threshold@1.25: 0.0000
# [Batch 067] ➔ MAE: 9.0495, RMSE: 10.2644, Threshold@1.25: 0.0000
# [Batch 068] ➔ MAE: 7.9147, RMSE: 8.4758, Threshold@1.25: 0.0004
# [Batch 069] ➔ MAE: 11.7118, RMSE: 14.3321, Threshold@1.25: 0.0000
# [Batch 070] ➔ MAE: 8.2185, RMSE: 9.1688, Threshold@1.25: 0.0006
# [Batch 071] ➔ MAE: 9.0712, RMSE: 9.9217, Threshold@1.25: 0.0001
# [Batch 072] ➔ MAE: 8.4113, RMSE: 9.1239, Threshold@1.25: 0.0001
# [Batch 073] ➔ MAE: 7.3551, RMSE: 7.8526, Threshold@1.25: 0.0000
# [Batch 074] ➔ MAE: 8.4109, RMSE: 9.1608, Threshold@1.25: 0.0000
# [Batch 075] ➔ MAE: 8.3361, RMSE: 8.9632, Threshold@1.25: 0.0000
# [Batch 076] ➔ MAE: 8.0179, RMSE: 8.5771, Threshold@1.25: 0.0000
# [Batch 077] ➔ MAE: 7.9070, RMSE: 8.8147, Threshold@1.25: 0.0000
# [Batch 078] ➔ MAE: 8.5240, RMSE: 9.1722, Threshold@1.25: 0.0001
# [Batch 079] ➔ MAE: 8.2968, RMSE: 8.9702, Threshold@1.25: 0.0000
# [Batch 080] ➔ MAE: 8.9315, RMSE: 9.7594, Threshold@1.25: 0.0000
# [Batch 081] ➔ MAE: 8.0588, RMSE: 8.7384, Threshold@1.25: 0.0000
# [Batch 082] ➔ MAE: 9.0641, RMSE: 9.9567, Threshold@1.25: 0.0000
# [Batch 083] ➔ MAE: 7.7583, RMSE: 9.2246, Threshold@1.25: 0.0028
# [Batch 084] ➔ MAE: 7.5647, RMSE: 8.6720, Threshold@1.25: 0.0009
# [Batch 085] ➔ MAE: 8.5926, RMSE: 9.7555, Threshold@1.25: 0.0006
# [Batch 086] ➔ MAE: 8.4024, RMSE: 9.2570, Threshold@1.25: 0.0001
# [Batch 087] ➔ MAE: 10.3909, RMSE: 12.8374, Threshold@1.25: 0.0000
# [Batch 088] ➔ MAE: 8.7626, RMSE: 9.5883, Threshold@1.25: 0.0000
# [Batch 089] ➔ MAE: 7.6730, RMSE: 8.5160, Threshold@1.25: 0.0006
# [Batch 090] ➔ MAE: 8.4335, RMSE: 9.4133, Threshold@1.25: 0.0002
# [Batch 091] ➔ MAE: 9.0101, RMSE: 10.5239, Threshold@1.25: 0.0006
# [Batch 092] ➔ MAE: 7.9560, RMSE: 9.2438, Threshold@1.25: 0.0000
# [Batch 093] ➔ MAE: 8.5249, RMSE: 9.6098, Threshold@1.25: 0.0004
# [Batch 094] ➔ MAE: 7.1548, RMSE: 7.7943, Threshold@1.25: 0.0001
# [Batch 095] ➔ MAE: 8.5253, RMSE: 9.7067, Threshold@1.25: 0.0000
# [Batch 096] ➔ MAE: 8.4484, RMSE: 9.1563, Threshold@1.25: 0.0000
# [Batch 097] ➔ MAE: 7.6130, RMSE: 8.2127, Threshold@1.25: 0.0000
# [Batch 098] ➔ MAE: 8.4649, RMSE: 9.0593, Threshold@1.25: 0.0001
# [Batch 099] ➔ MAE: 7.8173, RMSE: 8.4939, Threshold@1.25: 0.0000
# [Batch 100] ➔ MAE: 9.1987, RMSE: 9.9100, Threshold@1.25: 0.0000
# [Batch 101] ➔ MAE: 9.0493, RMSE: 9.7371, Threshold@1.25: 0.0000
# [Batch 102] ➔ MAE: 8.6206, RMSE: 9.3466, Threshold@1.25: 0.0002
# [Batch 103] ➔ MAE: 8.0421, RMSE: 8.7366, Threshold@1.25: 0.0000
# [Batch 104] ➔ MAE: 7.8419, RMSE: 8.6743, Threshold@1.25: 0.0000
# [Batch 105] ➔ MAE: 7.8453, RMSE: 8.4272, Threshold@1.25: 0.0000
# [Batch 106] ➔ MAE: 10.2318, RMSE: 12.5084, Threshold@1.25: 0.0032
# [Batch 107] ➔ MAE: 7.4846, RMSE: 8.5205, Threshold@1.25: 0.0010
# [Batch 108] ➔ MAE: 8.3147, RMSE: 9.2227, Threshold@1.25: 0.0010
# [Batch 109] ➔ MAE: 8.6882, RMSE: 9.6644, Threshold@1.25: 0.0001
# [Batch 110] ➔ MAE: 8.7526, RMSE: 9.5828, Threshold@1.25: 0.0000
# [Batch 111] ➔ MAE: 9.6610, RMSE: 10.4125, Threshold@1.25: 0.0000
# [Batch 112] ➔ MAE: 8.9039, RMSE: 9.5515, Threshold@1.25: 0.0000
# [Batch 113] ➔ MAE: 8.7365, RMSE: 9.2927, Threshold@1.25: 0.0001
# [Batch 114] ➔ MAE: 7.7440, RMSE: 8.2372, Threshold@1.25: 0.0001
# [Batch 115] ➔ MAE: 7.9458, RMSE: 8.7027, Threshold@1.25: 0.0002
# [Batch 116] ➔ MAE: 8.7514, RMSE: 9.5148, Threshold@1.25: 0.0000
# [Batch 117] ➔ MAE: 7.9374, RMSE: 9.6150, Threshold@1.25: 0.0047
# [Batch 118] ➔ MAE: 8.5436, RMSE: 9.0443, Threshold@1.25: 0.0000
# [Batch 119] ➔ MAE: 7.5679, RMSE: 8.1773, Threshold@1.25: 0.0001
# [Batch 120] ➔ MAE: 8.1188, RMSE: 8.7026, Threshold@1.25: 0.0000
# [Batch 121] ➔ MAE: 8.2699, RMSE: 8.8879, Threshold@1.25: 0.0000
# [Batch 122] ➔ MAE: 7.8744, RMSE: 8.7793, Threshold@1.25: 0.0035
# [Batch 123] ➔ MAE: 6.7696, RMSE: 7.2044, Threshold@1.25: 0.0000
# [Batch 124] ➔ MAE: 8.2719, RMSE: 8.9640, Threshold@1.25: 0.0000
# [Batch 125] ➔ MAE: 8.1661, RMSE: 9.0843, Threshold@1.25: 0.0000
# [Batch 126] ➔ MAE: 8.0799, RMSE: 8.7036, Threshold@1.25: 0.0000
# [Batch 127] ➔ MAE: 8.5529, RMSE: 9.3773, Threshold@1.25: 0.0000
# [Batch 128] ➔ MAE: 8.9365, RMSE: 9.6058, Threshold@1.25: 0.0000
# [Batch 129] ➔ MAE: 8.8363, RMSE: 9.5822, Threshold@1.25: 0.0000
# [Batch 130] ➔ MAE: 9.5423, RMSE: 11.1762, Threshold@1.25: 0.0000
# [Batch 131] ➔ MAE: 7.0026, RMSE: 7.4989, Threshold@1.25: 0.0000
# [Batch 132] ➔ MAE: 8.0963, RMSE: 8.8111, Threshold@1.25: 0.0000
# [Batch 133] ➔ MAE: 7.2161, RMSE: 7.8808, Threshold@1.25: 0.0000
# [Batch 134] ➔ MAE: 9.4294, RMSE: 9.9142, Threshold@1.25: 0.0000
# [Batch 135] ➔ MAE: 9.9369, RMSE: 11.7462, Threshold@1.25: 0.0000
# [Batch 136] ➔ MAE: 9.2570, RMSE: 11.1173, Threshold@1.25: 0.0000
# [Batch 137] ➔ MAE: 7.6856, RMSE: 8.1406, Threshold@1.25: 0.0008

# ✅ Estimated best global scale: 0.211495

# 📋 Monocular Depth Metrics (Before vs After Scaling)
# Metric                                   |  Before Scaling |   After Scaling
# ---------------------------------------------------------------------------
# MAE                                      |        8.530967 |        0.802510
# RMSE                                     |        9.463261 |        1.392293
# Scale-Invariant RMSE                     |        1.681826 |        0.480206
# Threshold Accuracy (δ=1.25)              |        0.000857 |        0.473812
# Threshold Accuracy (δ=1.25^2)            |        0.006464 |        0.721845
# Threshold Accuracy (δ=1.25^3)            |        0.021475 |        0.866570
# Log RMSE                                 |        1.681826 |        0.480206
# Mean Relative Error                      |        4.843648 |        0.443563
# Mean Squared Log Error                   |        1.737379 |        0.100089
# Structural Similarity (SSIM)             |        0.376024 |        0.857760
# Edge-Aware Loss                          |        2.528269 |        0.059895


# vitl - vkitti (frozen encoder)
# [Batch 000] ➔ MAE: 5.8295, RMSE: 7.1271, Threshold@1.25: 0.0385
# [Batch 001] ➔ MAE: 4.9571, RMSE: 5.7419, Threshold@1.25: 0.0046
# [Batch 002] ➔ MAE: 4.7056, RMSE: 5.6883, Threshold@1.25: 0.0090
# [Batch 003] ➔ MAE: 4.9512, RMSE: 5.6685, Threshold@1.25: 0.0044
# [Batch 004] ➔ MAE: 6.4717, RMSE: 6.9787, Threshold@1.25: 0.0000
# [Batch 005] ➔ MAE: 4.2073, RMSE: 4.7342, Threshold@1.25: 0.0135
# [Batch 006] ➔ MAE: 6.1043, RMSE: 6.9237, Threshold@1.25: 0.0002
# [Batch 007] ➔ MAE: 6.0658, RMSE: 7.1875, Threshold@1.25: 0.0006
# [Batch 008] ➔ MAE: 5.2430, RMSE: 5.5496, Threshold@1.25: 0.0029
# [Batch 009] ➔ MAE: 4.0170, RMSE: 4.7731, Threshold@1.25: 0.0038
# [Batch 010] ➔ MAE: 5.8750, RMSE: 6.8754, Threshold@1.25: 0.0132
# [Batch 011] ➔ MAE: 4.6210, RMSE: 5.2420, Threshold@1.25: 0.0499
# [Batch 012] ➔ MAE: 5.4828, RMSE: 6.1950, Threshold@1.25: 0.0200
# [Batch 013] ➔ MAE: 5.1298, RMSE: 6.2140, Threshold@1.25: 0.0050
# [Batch 014] ➔ MAE: 5.2422, RMSE: 6.3204, Threshold@1.25: 0.0050
# [Batch 015] ➔ MAE: 5.2789, RMSE: 6.1299, Threshold@1.25: 0.0088
# [Batch 016] ➔ MAE: 4.9666, RMSE: 5.7565, Threshold@1.25: 0.0091
# [Batch 017] ➔ MAE: 6.0730, RMSE: 7.5087, Threshold@1.25: 0.0931
# [Batch 018] ➔ MAE: 4.3125, RMSE: 5.5710, Threshold@1.25: 0.0923
# [Batch 019] ➔ MAE: 4.9908, RMSE: 5.8840, Threshold@1.25: 0.0147
# [Batch 020] ➔ MAE: 4.8233, RMSE: 5.8859, Threshold@1.25: 0.1041
# [Batch 021] ➔ MAE: 5.6232, RMSE: 6.1104, Threshold@1.25: 0.0005
# [Batch 022] ➔ MAE: 6.8334, RMSE: 7.9469, Threshold@1.25: 0.0194
# [Batch 023] ➔ MAE: 5.3344, RMSE: 5.7834, Threshold@1.25: 0.0004
# [Batch 024] ➔ MAE: 5.1341, RMSE: 5.8761, Threshold@1.25: 0.0002
# [Batch 025] ➔ MAE: 5.8416, RMSE: 6.8449, Threshold@1.25: 0.0022
# [Batch 026] ➔ MAE: 5.2456, RMSE: 5.7579, Threshold@1.25: 0.0033
# [Batch 027] ➔ MAE: 5.6167, RMSE: 6.2109, Threshold@1.25: 0.0019
# [Batch 028] ➔ MAE: 8.3525, RMSE: 9.7122, Threshold@1.25: 0.0003
# [Batch 029] ➔ MAE: 5.3057, RMSE: 6.2457, Threshold@1.25: 0.0005
# [Batch 030] ➔ MAE: 5.5411, RMSE: 6.1769, Threshold@1.25: 0.0004
# [Batch 031] ➔ MAE: 5.4289, RMSE: 6.6419, Threshold@1.25: 0.0005
# [Batch 032] ➔ MAE: 3.5350, RMSE: 3.8949, Threshold@1.25: 0.0009
# [Batch 033] ➔ MAE: 5.0462, RMSE: 5.5831, Threshold@1.25: 0.0359
# [Batch 034] ➔ MAE: 4.6518, RMSE: 5.4194, Threshold@1.25: 0.0449
# [Batch 035] ➔ MAE: 4.2571, RMSE: 5.1869, Threshold@1.25: 0.0171
# [Batch 036] ➔ MAE: 5.3837, RMSE: 6.6095, Threshold@1.25: 0.0006
# [Batch 037] ➔ MAE: 5.4049, RMSE: 6.0431, Threshold@1.25: 0.0003
# [Batch 038] ➔ MAE: 4.1175, RMSE: 4.7395, Threshold@1.25: 0.0173
# [Batch 039] ➔ MAE: 5.2549, RMSE: 6.3276, Threshold@1.25: 0.0519
# [Batch 040] ➔ MAE: 4.8166, RMSE: 5.3243, Threshold@1.25: 0.0034
# [Batch 041] ➔ MAE: 3.9109, RMSE: 4.4189, Threshold@1.25: 0.0073
# [Batch 042] ➔ MAE: 5.4649, RMSE: 6.5275, Threshold@1.25: 0.0084
# [Batch 043] ➔ MAE: 5.0706, RMSE: 6.0490, Threshold@1.25: 0.0068
# [Batch 044] ➔ MAE: 6.5505, RMSE: 7.4162, Threshold@1.25: 0.0003
# [Batch 045] ➔ MAE: 3.6408, RMSE: 4.1679, Threshold@1.25: 0.0694
# [Batch 046] ➔ MAE: 5.0548, RMSE: 5.8297, Threshold@1.25: 0.0005
# [Batch 047] ➔ MAE: 3.8433, RMSE: 4.7030, Threshold@1.25: 0.0602
# [Batch 048] ➔ MAE: 5.4620, RMSE: 5.9501, Threshold@1.25: 0.0009
# [Batch 049] ➔ MAE: 6.0678, RMSE: 6.8795, Threshold@1.25: 0.0015
# [Batch 050] ➔ MAE: 5.4420, RMSE: 6.0556, Threshold@1.25: 0.0073
# [Batch 051] ➔ MAE: 3.3034, RMSE: 3.8083, Threshold@1.25: 0.0282
# [Batch 052] ➔ MAE: 5.5133, RMSE: 5.9836, Threshold@1.25: 0.0001
# [Batch 053] ➔ MAE: 5.6988, RMSE: 6.3610, Threshold@1.25: 0.0009
# [Batch 054] ➔ MAE: 6.0704, RMSE: 6.6646, Threshold@1.25: 0.0001
# [Batch 055] ➔ MAE: 4.4873, RMSE: 5.1802, Threshold@1.25: 0.0225
# [Batch 056] ➔ MAE: 5.8818, RMSE: 6.8970, Threshold@1.25: 0.0044
# [Batch 057] ➔ MAE: 5.7097, RMSE: 6.8664, Threshold@1.25: 0.0037
# [Batch 058] ➔ MAE: 7.5310, RMSE: 8.9312, Threshold@1.25: 0.0003
# [Batch 059] ➔ MAE: 6.3671, RMSE: 7.1463, Threshold@1.25: 0.0016
# [Batch 060] ➔ MAE: 4.2889, RMSE: 5.4626, Threshold@1.25: 0.0688
# [Batch 061] ➔ MAE: 4.7550, RMSE: 5.4041, Threshold@1.25: 0.0039
# [Batch 062] ➔ MAE: 5.6146, RMSE: 6.4625, Threshold@1.25: 0.0135
# [Batch 063] ➔ MAE: 4.8419, RMSE: 5.3410, Threshold@1.25: 0.0083
# [Batch 064] ➔ MAE: 4.2109, RMSE: 4.8316, Threshold@1.25: 0.0138
# [Batch 065] ➔ MAE: 6.9926, RMSE: 7.8656, Threshold@1.25: 0.0021
# [Batch 066] ➔ MAE: 4.1447, RMSE: 4.7587, Threshold@1.25: 0.0705
# [Batch 067] ➔ MAE: 4.2728, RMSE: 4.9101, Threshold@1.25: 0.0011
# [Batch 068] ➔ MAE: 5.5408, RMSE: 6.1769, Threshold@1.25: 0.0027
# [Batch 069] ➔ MAE: 6.3373, RMSE: 7.2314, Threshold@1.25: 0.0211
# [Batch 070] ➔ MAE: 6.2252, RMSE: 7.1039, Threshold@1.25: 0.0019
# [Batch 071] ➔ MAE: 5.7270, RMSE: 6.5465, Threshold@1.25: 0.0021
# [Batch 072] ➔ MAE: 5.4642, RMSE: 6.6211, Threshold@1.25: 0.0257
# [Batch 073] ➔ MAE: 5.2484, RMSE: 6.1500, Threshold@1.25: 0.0070
# [Batch 074] ➔ MAE: 4.3410, RMSE: 5.4290, Threshold@1.25: 0.0209
# [Batch 075] ➔ MAE: 5.7913, RMSE: 6.5561, Threshold@1.25: 0.0076
# [Batch 076] ➔ MAE: 4.8479, RMSE: 5.5890, Threshold@1.25: 0.0042
# [Batch 077] ➔ MAE: 4.6647, RMSE: 5.2901, Threshold@1.25: 0.0015
# [Batch 078] ➔ MAE: 4.8820, RMSE: 5.4238, Threshold@1.25: 0.0004
# [Batch 079] ➔ MAE: 6.3464, RMSE: 7.0443, Threshold@1.25: 0.0046
# [Batch 080] ➔ MAE: 4.3969, RMSE: 4.7267, Threshold@1.25: 0.0015
# [Batch 081] ➔ MAE: 6.1853, RMSE: 6.8681, Threshold@1.25: 0.0011
# [Batch 082] ➔ MAE: 5.3767, RMSE: 6.1882, Threshold@1.25: 0.0055
# [Batch 083] ➔ MAE: 5.9104, RMSE: 7.5689, Threshold@1.25: 0.0491
# [Batch 084] ➔ MAE: 4.9551, RMSE: 6.2956, Threshold@1.25: 0.0021
# [Batch 085] ➔ MAE: 5.3458, RMSE: 6.0483, Threshold@1.25: 0.0071
# [Batch 086] ➔ MAE: 3.5154, RMSE: 4.0587, Threshold@1.25: 0.0273
# [Batch 087] ➔ MAE: 3.9336, RMSE: 4.4348, Threshold@1.25: 0.0076
# [Batch 088] ➔ MAE: 4.0487, RMSE: 5.0593, Threshold@1.25: 0.0366
# [Batch 089] ➔ MAE: 5.1674, RMSE: 6.2908, Threshold@1.25: 0.0088
# [Batch 090] ➔ MAE: 5.0491, RMSE: 5.8622, Threshold@1.25: 0.0051
# [Batch 091] ➔ MAE: 4.9399, RMSE: 5.5658, Threshold@1.25: 0.0056
# [Batch 092] ➔ MAE: 5.5226, RMSE: 6.4333, Threshold@1.25: 0.0036
# [Batch 093] ➔ MAE: 5.3183, RMSE: 6.3474, Threshold@1.25: 0.0584
# [Batch 094] ➔ MAE: 6.2193, RMSE: 6.8584, Threshold@1.25: 0.0009
# [Batch 095] ➔ MAE: 4.3049, RMSE: 5.1010, Threshold@1.25: 0.0688
# [Batch 096] ➔ MAE: 6.9374, RMSE: 8.2332, Threshold@1.25: 0.0011
# [Batch 097] ➔ MAE: 4.6026, RMSE: 5.1105, Threshold@1.25: 0.0037
# [Batch 098] ➔ MAE: 6.4164, RMSE: 7.2975, Threshold@1.25: 0.0007
# [Batch 099] ➔ MAE: 4.7068, RMSE: 5.2243, Threshold@1.25: 0.0155
# [Batch 100] ➔ MAE: 3.8904, RMSE: 4.6941, Threshold@1.25: 0.0218
# [Batch 101] ➔ MAE: 4.7029, RMSE: 5.1696, Threshold@1.25: 0.0002
# [Batch 102] ➔ MAE: 5.2945, RMSE: 5.9782, Threshold@1.25: 0.0008
# [Batch 103] ➔ MAE: 5.5215, RMSE: 6.1026, Threshold@1.25: 0.0007
# [Batch 104] ➔ MAE: 5.1469, RMSE: 5.8078, Threshold@1.25: 0.0144
# [Batch 105] ➔ MAE: 4.2155, RMSE: 4.8825, Threshold@1.25: 0.0088
# [Batch 106] ➔ MAE: 4.6465, RMSE: 5.9003, Threshold@1.25: 0.0176
# [Batch 107] ➔ MAE: 4.7744, RMSE: 5.3219, Threshold@1.25: 0.0051
# [Batch 108] ➔ MAE: 3.7127, RMSE: 4.1790, Threshold@1.25: 0.0073
# [Batch 109] ➔ MAE: 4.2721, RMSE: 4.9565, Threshold@1.25: 0.0281
# [Batch 110] ➔ MAE: 4.7954, RMSE: 5.3627, Threshold@1.25: 0.0001
# [Batch 111] ➔ MAE: 4.9725, RMSE: 5.5546, Threshold@1.25: 0.0019
# [Batch 112] ➔ MAE: 5.4155, RMSE: 6.7831, Threshold@1.25: 0.0256
# [Batch 113] ➔ MAE: 4.7706, RMSE: 5.3305, Threshold@1.25: 0.0032
# [Batch 114] ➔ MAE: 4.9864, RMSE: 5.4783, Threshold@1.25: 0.0004
# [Batch 115] ➔ MAE: 5.7239, RMSE: 6.2768, Threshold@1.25: 0.0005
# [Batch 116] ➔ MAE: 6.3986, RMSE: 7.4234, Threshold@1.25: 0.0035
# [Batch 117] ➔ MAE: 5.9201, RMSE: 6.5532, Threshold@1.25: 0.0071
# [Batch 118] ➔ MAE: 3.9106, RMSE: 4.6791, Threshold@1.25: 0.0431
# [Batch 119] ➔ MAE: 2.7830, RMSE: 3.3938, Threshold@1.25: 0.1044
# [Batch 120] ➔ MAE: 5.5669, RMSE: 6.1373, Threshold@1.25: 0.0013
# [Batch 121] ➔ MAE: 5.4090, RMSE: 6.1861, Threshold@1.25: 0.0005
# [Batch 122] ➔ MAE: 5.7799, RMSE: 6.3839, Threshold@1.25: 0.0002
# [Batch 123] ➔ MAE: 6.5744, RMSE: 7.2065, Threshold@1.25: 0.0049
# [Batch 124] ➔ MAE: 4.4811, RMSE: 5.0630, Threshold@1.25: 0.0024
# [Batch 125] ➔ MAE: 6.3409, RMSE: 7.0271, Threshold@1.25: 0.0001
# [Batch 126] ➔ MAE: 5.9426, RMSE: 7.1378, Threshold@1.25: 0.0078
# [Batch 127] ➔ MAE: 6.1383, RMSE: 6.6729, Threshold@1.25: 0.0002
# [Batch 128] ➔ MAE: 7.3019, RMSE: 8.1069, Threshold@1.25: 0.0001
# [Batch 129] ➔ MAE: 5.3274, RMSE: 6.4743, Threshold@1.25: 0.0567
# [Batch 130] ➔ MAE: 3.6825, RMSE: 4.2738, Threshold@1.25: 0.0638
# [Batch 131] ➔ MAE: 4.9403, RMSE: 5.7665, Threshold@1.25: 0.0272
# [Batch 132] ➔ MAE: 6.2529, RMSE: 7.1806, Threshold@1.25: 0.0015
# [Batch 133] ➔ MAE: 6.2935, RMSE: 7.0687, Threshold@1.25: 0.0006
# [Batch 134] ➔ MAE: 6.0733, RMSE: 7.2286, Threshold@1.25: 0.0009
# [Batch 135] ➔ MAE: 5.3028, RMSE: 5.6127, Threshold@1.25: 0.0000
# [Batch 136] ➔ MAE: 6.2159, RMSE: 7.1040, Threshold@1.25: 0.0034
# [Batch 137] ➔ MAE: 6.5591, RMSE: 7.1679, Threshold@1.25: 0.0000

# ✅ Estimated best global scale: 0.293254

# 📋 Monocular Depth Metrics (Before vs After Scaling)
# Metric                                   |  Before Scaling |   After Scaling
# ---------------------------------------------------------------------------
# MAE                                      |        5.250513 |        1.017236
# RMSE                                     |        6.040760 |        1.734655
# Scale-Invariant RMSE                     |        1.362282 |        0.608966
# Threshold Accuracy (δ=1.25)              |        0.014327 |        0.359861
# Threshold Accuracy (δ=1.25^2)            |        0.064504 |        0.629548
# Threshold Accuracy (δ=1.25^3)            |        0.136337 |        0.787079
# Log RMSE                                 |        1.362282 |        0.608966
# Mean Relative Error                      |        3.205184 |        0.523242
# Mean Squared Log Error                   |        1.053446 |        0.160998
# Structural Similarity (SSIM)             |        0.506085 |        0.830840
# Edge-Aware Loss                          |        3.822493 |        2.265678



# vitl - hypersim (frozen encoder)
# [Batch 000] ➔ MAE: 3.6188, RMSE: 5.1308, Threshold@1.25: 0.0360
# [Batch 001] ➔ MAE: 2.4517, RMSE: 3.1896, Threshold@1.25: 0.1569
# [Batch 002] ➔ MAE: 2.6946, RMSE: 3.6693, Threshold@1.25: 0.1478
# [Batch 003] ➔ MAE: 3.2061, RMSE: 4.1759, Threshold@1.25: 0.1184
# [Batch 004] ➔ MAE: 2.3979, RMSE: 2.7537, Threshold@1.25: 0.0092
# [Batch 005] ➔ MAE: 3.8699, RMSE: 4.8881, Threshold@1.25: 0.0242
# [Batch 006] ➔ MAE: 2.3344, RMSE: 2.6337, Threshold@1.25: 0.0304
# [Batch 007] ➔ MAE: 2.6216, RMSE: 3.1735, Threshold@1.25: 0.0333
# [Batch 008] ➔ MAE: 2.4385, RMSE: 2.6710, Threshold@1.25: 0.0063
# [Batch 009] ➔ MAE: 4.2890, RMSE: 5.6715, Threshold@1.25: 0.0147
# [Batch 010] ➔ MAE: 2.4511, RMSE: 2.8664, Threshold@1.25: 0.0074
# [Batch 011] ➔ MAE: 2.6999, RMSE: 3.0074, Threshold@1.25: 0.0120
# [Batch 012] ➔ MAE: 2.8456, RMSE: 3.5164, Threshold@1.25: 0.0534
# [Batch 013] ➔ MAE: 2.2954, RMSE: 3.2395, Threshold@1.25: 0.0195
# [Batch 014] ➔ MAE: 2.5812, RMSE: 3.4045, Threshold@1.25: 0.0308
# [Batch 015] ➔ MAE: 3.0670, RMSE: 3.7416, Threshold@1.25: 0.0519
# [Batch 016] ➔ MAE: 2.6130, RMSE: 3.0764, Threshold@1.25: 0.1140
# [Batch 017] ➔ MAE: 1.9961, RMSE: 2.4449, Threshold@1.25: 0.1312
# [Batch 018] ➔ MAE: 2.4798, RMSE: 3.0617, Threshold@1.25: 0.0657
# [Batch 019] ➔ MAE: 2.8484, RMSE: 3.5198, Threshold@1.25: 0.0474
# [Batch 020] ➔ MAE: 2.1456, RMSE: 2.8224, Threshold@1.25: 0.0786
# [Batch 021] ➔ MAE: 1.9787, RMSE: 2.4084, Threshold@1.25: 0.0250
# [Batch 022] ➔ MAE: 2.2221, RMSE: 2.6851, Threshold@1.25: 0.1203
# [Batch 023] ➔ MAE: 2.5778, RMSE: 3.1172, Threshold@1.25: 0.0561
# [Batch 024] ➔ MAE: 2.9637, RMSE: 3.7548, Threshold@1.25: 0.0249
# [Batch 025] ➔ MAE: 3.1778, RMSE: 3.7524, Threshold@1.25: 0.1002
# [Batch 026] ➔ MAE: 2.9032, RMSE: 3.6390, Threshold@1.25: 0.0470
# [Batch 027] ➔ MAE: 3.2236, RMSE: 3.8734, Threshold@1.25: 0.0395
# [Batch 028] ➔ MAE: 2.2539, RMSE: 2.6426, Threshold@1.25: 0.0662
# [Batch 029] ➔ MAE: 2.2013, RMSE: 2.6350, Threshold@1.25: 0.0846
# [Batch 030] ➔ MAE: 2.5014, RMSE: 3.1419, Threshold@1.25: 0.0869
# [Batch 031] ➔ MAE: 2.2116, RMSE: 2.6833, Threshold@1.25: 0.1157
# [Batch 032] ➔ MAE: 2.9626, RMSE: 3.8099, Threshold@1.25: 0.0702
# [Batch 033] ➔ MAE: 2.0317, RMSE: 2.3958, Threshold@1.25: 0.0238
# [Batch 034] ➔ MAE: 2.3404, RMSE: 2.8248, Threshold@1.25: 0.1074
# [Batch 035] ➔ MAE: 2.7308, RMSE: 3.1862, Threshold@1.25: 0.0426
# [Batch 036] ➔ MAE: 2.5253, RMSE: 3.0902, Threshold@1.25: 0.1260
# [Batch 037] ➔ MAE: 3.3618, RMSE: 3.8373, Threshold@1.25: 0.0064
# [Batch 038] ➔ MAE: 3.5245, RMSE: 4.2918, Threshold@1.25: 0.0159
# [Batch 039] ➔ MAE: 2.8943, RMSE: 3.4007, Threshold@1.25: 0.0422
# [Batch 040] ➔ MAE: 2.9490, RMSE: 3.5784, Threshold@1.25: 0.0617
# [Batch 041] ➔ MAE: 3.0413, RMSE: 3.8218, Threshold@1.25: 0.0203
# [Batch 042] ➔ MAE: 2.9190, RMSE: 3.5910, Threshold@1.25: 0.0181
# [Batch 043] ➔ MAE: 2.8177, RMSE: 3.3000, Threshold@1.25: 0.0225
# [Batch 044] ➔ MAE: 3.1321, RMSE: 3.7245, Threshold@1.25: 0.0285
# [Batch 045] ➔ MAE: 2.8069, RMSE: 3.5214, Threshold@1.25: 0.0340
# [Batch 046] ➔ MAE: 3.1557, RMSE: 3.7799, Threshold@1.25: 0.1176
# [Batch 047] ➔ MAE: 2.3996, RMSE: 2.9260, Threshold@1.25: 0.0938
# [Batch 048] ➔ MAE: 2.4153, RMSE: 3.0019, Threshold@1.25: 0.1645
# [Batch 049] ➔ MAE: 2.4498, RMSE: 2.8026, Threshold@1.25: 0.0071
# [Batch 050] ➔ MAE: 2.8023, RMSE: 3.6489, Threshold@1.25: 0.2549
# [Batch 051] ➔ MAE: 2.6976, RMSE: 3.1262, Threshold@1.25: 0.0997
# [Batch 052] ➔ MAE: 2.7440, RMSE: 3.5085, Threshold@1.25: 0.0637
# [Batch 053] ➔ MAE: 2.9687, RMSE: 3.8002, Threshold@1.25: 0.0492
# [Batch 054] ➔ MAE: 2.7017, RMSE: 3.3819, Threshold@1.25: 0.0990
# [Batch 055] ➔ MAE: 2.3399, RMSE: 2.7430, Threshold@1.25: 0.0566
# [Batch 056] ➔ MAE: 2.1556, RMSE: 2.4347, Threshold@1.25: 0.0093
# [Batch 057] ➔ MAE: 3.1730, RMSE: 4.0280, Threshold@1.25: 0.0487
# [Batch 058] ➔ MAE: 2.3892, RMSE: 2.7659, Threshold@1.25: 0.0707
# [Batch 059] ➔ MAE: 2.3589, RMSE: 2.6698, Threshold@1.25: 0.0165
# [Batch 060] ➔ MAE: 2.6583, RMSE: 3.3914, Threshold@1.25: 0.0835
# [Batch 061] ➔ MAE: 2.5994, RMSE: 2.9259, Threshold@1.25: 0.0150
# [Batch 062] ➔ MAE: 2.4759, RMSE: 2.8773, Threshold@1.25: 0.0233
# [Batch 063] ➔ MAE: 2.7244, RMSE: 3.1139, Threshold@1.25: 0.0108
# [Batch 064] ➔ MAE: 2.3988, RMSE: 3.2091, Threshold@1.25: 0.0150
# [Batch 065] ➔ MAE: 3.3033, RMSE: 4.4407, Threshold@1.25: 0.0074
# [Batch 066] ➔ MAE: 2.1970, RMSE: 2.4706, Threshold@1.25: 0.0189
# [Batch 067] ➔ MAE: 3.4051, RMSE: 4.4805, Threshold@1.25: 0.0170
# [Batch 068] ➔ MAE: 2.8378, RMSE: 3.3572, Threshold@1.25: 0.0189
# [Batch 069] ➔ MAE: 3.0946, RMSE: 4.3444, Threshold@1.25: 0.1962
# [Batch 070] ➔ MAE: 2.4886, RMSE: 3.0062, Threshold@1.25: 0.0935
# [Batch 071] ➔ MAE: 2.6325, RMSE: 3.4623, Threshold@1.25: 0.1759
# [Batch 072] ➔ MAE: 2.3185, RMSE: 2.8556, Threshold@1.25: 0.0545
# [Batch 073] ➔ MAE: 2.1948, RMSE: 2.6871, Threshold@1.25: 0.0107
# [Batch 074] ➔ MAE: 2.1911, RMSE: 2.5103, Threshold@1.25: 0.0231
# [Batch 075] ➔ MAE: 2.7524, RMSE: 3.4019, Threshold@1.25: 0.0255
# [Batch 076] ➔ MAE: 2.4986, RMSE: 2.9035, Threshold@1.25: 0.0218
# [Batch 077] ➔ MAE: 2.5593, RMSE: 2.9548, Threshold@1.25: 0.0851
# [Batch 078] ➔ MAE: 3.3537, RMSE: 4.3046, Threshold@1.25: 0.0478
# [Batch 079] ➔ MAE: 3.2078, RMSE: 4.3329, Threshold@1.25: 0.0123
# [Batch 080] ➔ MAE: 2.5214, RMSE: 3.1463, Threshold@1.25: 0.0512
# [Batch 081] ➔ MAE: 3.0055, RMSE: 3.7575, Threshold@1.25: 0.0269
# [Batch 082] ➔ MAE: 2.1065, RMSE: 2.4630, Threshold@1.25: 0.0397
# [Batch 083] ➔ MAE: 2.7535, RMSE: 3.2139, Threshold@1.25: 0.0245
# [Batch 084] ➔ MAE: 2.9197, RMSE: 3.2783, Threshold@1.25: 0.0190
# [Batch 085] ➔ MAE: 2.5972, RMSE: 3.2478, Threshold@1.25: 0.0854
# [Batch 086] ➔ MAE: 2.5997, RMSE: 3.3002, Threshold@1.25: 0.1444
# [Batch 087] ➔ MAE: 2.3241, RMSE: 2.7913, Threshold@1.25: 0.0709
# [Batch 088] ➔ MAE: 2.5995, RMSE: 3.0970, Threshold@1.25: 0.0188
# [Batch 089] ➔ MAE: 3.7298, RMSE: 5.1719, Threshold@1.25: 0.1596
# [Batch 090] ➔ MAE: 2.5482, RMSE: 2.8865, Threshold@1.25: 0.0093
# [Batch 091] ➔ MAE: 3.5107, RMSE: 4.6347, Threshold@1.25: 0.0161
# [Batch 092] ➔ MAE: 2.6464, RMSE: 3.2897, Threshold@1.25: 0.0598
# [Batch 093] ➔ MAE: 3.0874, RMSE: 3.8193, Threshold@1.25: 0.1999
# [Batch 094] ➔ MAE: 2.8068, RMSE: 3.7207, Threshold@1.25: 0.1250
# [Batch 095] ➔ MAE: 2.5397, RMSE: 3.1919, Threshold@1.25: 0.0108
# [Batch 096] ➔ MAE: 2.1683, RMSE: 2.7039, Threshold@1.25: 0.1003
# [Batch 097] ➔ MAE: 2.8139, RMSE: 3.2118, Threshold@1.25: 0.0155
# [Batch 098] ➔ MAE: 2.0324, RMSE: 2.4609, Threshold@1.25: 0.0983
# [Batch 099] ➔ MAE: 3.2870, RMSE: 4.2011, Threshold@1.25: 0.1280
# [Batch 100] ➔ MAE: 2.3884, RMSE: 3.0257, Threshold@1.25: 0.0192
# [Batch 101] ➔ MAE: 2.1315, RMSE: 2.6918, Threshold@1.25: 0.0875
# [Batch 102] ➔ MAE: 2.9405, RMSE: 3.6100, Threshold@1.25: 0.0568
# [Batch 103] ➔ MAE: 2.9575, RMSE: 3.4930, Threshold@1.25: 0.0360
# [Batch 104] ➔ MAE: 2.9940, RMSE: 3.5032, Threshold@1.25: 0.0276
# [Batch 105] ➔ MAE: 2.7133, RMSE: 2.9962, Threshold@1.25: 0.0183
# [Batch 106] ➔ MAE: 2.8518, RMSE: 3.3873, Threshold@1.25: 0.0132
# [Batch 107] ➔ MAE: 2.5011, RMSE: 3.3229, Threshold@1.25: 0.0403
# [Batch 108] ➔ MAE: 1.6393, RMSE: 2.1385, Threshold@1.25: 0.0581
# [Batch 109] ➔ MAE: 2.5602, RMSE: 2.9799, Threshold@1.25: 0.0738
# [Batch 110] ➔ MAE: 2.5624, RMSE: 3.1201, Threshold@1.25: 0.0245
# [Batch 111] ➔ MAE: 2.6335, RMSE: 3.1365, Threshold@1.25: 0.0233
# [Batch 112] ➔ MAE: 1.8567, RMSE: 2.0992, Threshold@1.25: 0.0101
# [Batch 113] ➔ MAE: 1.6883, RMSE: 2.1175, Threshold@1.25: 0.0731
# [Batch 114] ➔ MAE: 2.2696, RMSE: 2.7397, Threshold@1.25: 0.0733
# [Batch 115] ➔ MAE: 1.8378, RMSE: 2.4487, Threshold@1.25: 0.1578
# [Batch 116] ➔ MAE: 2.2222, RMSE: 2.6308, Threshold@1.25: 0.0947
# [Batch 117] ➔ MAE: 2.1347, RMSE: 2.5904, Threshold@1.25: 0.0991
# [Batch 118] ➔ MAE: 2.0503, RMSE: 2.4340, Threshold@1.25: 0.0748
# [Batch 119] ➔ MAE: 2.5728, RMSE: 3.2395, Threshold@1.25: 0.0549
# [Batch 120] ➔ MAE: 2.1760, RMSE: 2.5385, Threshold@1.25: 0.0196
# [Batch 121] ➔ MAE: 2.7088, RMSE: 3.2733, Threshold@1.25: 0.0630
# [Batch 122] ➔ MAE: 1.8489, RMSE: 2.0915, Threshold@1.25: 0.0097
# [Batch 123] ➔ MAE: 3.0262, RMSE: 3.5566, Threshold@1.25: 0.0526
# [Batch 124] ➔ MAE: 2.1813, RMSE: 2.5779, Threshold@1.25: 0.0310
# [Batch 125] ➔ MAE: 2.3777, RMSE: 2.8206, Threshold@1.25: 0.0077
# [Batch 126] ➔ MAE: 3.2166, RMSE: 3.9604, Threshold@1.25: 0.0392
# [Batch 127] ➔ MAE: 1.8445, RMSE: 2.3483, Threshold@1.25: 0.1104
# [Batch 128] ➔ MAE: 2.7071, RMSE: 3.2766, Threshold@1.25: 0.0921
# [Batch 129] ➔ MAE: 1.4827, RMSE: 1.8887, Threshold@1.25: 0.1105
# [Batch 130] ➔ MAE: 2.5206, RMSE: 2.9786, Threshold@1.25: 0.0681
# [Batch 131] ➔ MAE: 2.7356, RMSE: 3.3357, Threshold@1.25: 0.1085
# [Batch 132] ➔ MAE: 2.4936, RMSE: 2.8936, Threshold@1.25: 0.0863
# [Batch 133] ➔ MAE: 2.6990, RMSE: 3.2850, Threshold@1.25: 0.1312
# [Batch 134] ➔ MAE: 2.5049, RMSE: 2.9785, Threshold@1.25: 0.0209
# [Batch 135] ➔ MAE: 2.9844, RMSE: 3.4697, Threshold@1.25: 0.0319
# [Batch 136] ➔ MAE: 2.4738, RMSE: 3.2695, Threshold@1.25: 0.0341
# [Batch 137] ➔ MAE: 2.3457, RMSE: 3.0184, Threshold@1.25: 0.1177

# ✅ Estimated best global scale: 0.477741

# 📋 Monocular Depth Metrics (Before vs After Scaling)
# Metric                                   |  Before Scaling |   After Scaling
# ---------------------------------------------------------------------------
# MAE                                      |        2.632409 |        0.817887
# RMSE                                     |        3.230919 |        1.390249
# Scale-Invariant RMSE                     |        0.868915 |        0.516239
# Threshold Accuracy (δ=1.25)              |        0.060416 |        0.490084
# Threshold Accuracy (δ=1.25^2)            |        0.198762 |        0.748863
# Threshold Accuracy (δ=1.25^3)            |        0.408214 |        0.866125
# Log RMSE                                 |        0.868915 |        0.516239
# Mean Relative Error                      |        1.421924 |        0.404618
# Mean Squared Log Error                   |        0.400360 |        0.110118
# Structural Similarity (SSIM)             |        0.685935 |        0.817795
# Edge-Aware Loss                          |        0.845369 |        0.709265
