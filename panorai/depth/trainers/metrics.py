import numpy as np
from skimage.metrics import structural_similarity as ssim

class MonocularDepthMetrics:
    def __init__(self):
        self.sums = {}
        self.count = 0

    def reset(self):
        self.sums = {}
        self.count = 0

    def _compute_batch(self, predictions, ground_truth, mask):
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        mask = np.array(mask)

        # 🚨 Strict valid mask: positive depth, positive pred, valid mask
        valid_mask = (mask > 0) & (predictions > 0) & (ground_truth > 0)

        if np.sum(valid_mask) == 0:
            # No valid pixels → return NaNs
            return {k: float('nan') for k in [
                "MAE", "RMSE", "Scale-Invariant RMSE",
                "Threshold Accuracy (δ=1.25)", "Threshold Accuracy (δ=1.25^2)",
                "Threshold Accuracy (δ=1.25^3)", "Log RMSE",
                "Mean Relative Error", "Mean Squared Log Error",
                "Structural Similarity (SSIM)", "Edge-Aware Loss"
            ]}

        pred = predictions[valid_mask]
        gt = ground_truth[valid_mask]

        # 📏 Basic errors
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt) ** 2))

        # 📏 Log errors (no +1e-8 needed because all preds and gt > 0)
        log_diff = np.log(pred) - np.log(gt)
        scale_inv_rmse = np.sqrt(np.mean(log_diff ** 2))
        log_rmse = np.sqrt(np.mean(log_diff ** 2))

        # 📏 Threshold accuracy
        ratio = np.maximum(pred / gt, gt / pred)
        thresh1 = np.mean(ratio < 1.25)
        thresh2 = np.mean(ratio < 1.25 ** 2)
        thresh3 = np.mean(ratio < 1.25 ** 3)

        # 📏 Mean Relative Error
        mean_rel = np.mean(np.abs(pred - gt) / gt)

        # 📏 MSLE (using log1p: safer for low depths if needed, though here all > 0)
        msle = np.mean((np.log1p(pred) - np.log1p(gt)) ** 2)

        # 🧠 SSIM
        try:
            ssim_value = ssim(
                (ground_truth * mask).astype(np.float32),
                (predictions * mask).astype(np.float32),
                data_range=(predictions * mask).max() - (predictions * mask).min()
            )
        except Exception:
            ssim_value = float('nan')

        # 🧠 Edge-aware loss
        grad_pred = np.gradient(predictions * mask)
        grad_truth = np.gradient(ground_truth * mask)
        edge_loss = np.mean(
            np.abs(grad_pred[0] - grad_truth[0]) +
            np.abs(grad_pred[1] - grad_truth[1])
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "Scale-Invariant RMSE": scale_inv_rmse,
            "Threshold Accuracy (δ=1.25)": thresh1,
            "Threshold Accuracy (δ=1.25^2)": thresh2,
            "Threshold Accuracy (δ=1.25^3)": thresh3,
            "Log RMSE": log_rmse,
            "Mean Relative Error": mean_rel,
            "Mean Squared Log Error": msle,
            "Structural Similarity (SSIM)": ssim_value,
            "Edge-Aware Loss": edge_loss,
        }

    def update(self, predictions, ground_truth, mask):
        """Update metric sums from a batch."""
        batch_metrics = self._compute_batch(predictions, ground_truth, mask)
        if not self.sums:
            self.sums = {k: 0.0 for k in batch_metrics.keys()}
        for k, v in batch_metrics.items():
            self.sums[k] += v
        self.count += 1

    def compute(self):
        """Return averaged metrics over all updates."""
        if self.count == 0:
            return {k: float('nan') for k in self.sums.keys()}
        return {k: v / self.count for k, v in self.sums.items()}