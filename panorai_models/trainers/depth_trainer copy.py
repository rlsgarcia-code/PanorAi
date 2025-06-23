from accelerate import Accelerator
import os
import time
import heapq
import torch
import torch.nn.functional as F
import numpy as np
from panorai_models.DepthAnythingV2.metric_depth.util.loss import SiLogLoss
from .metrics import MonocularDepthMetrics  # adjust this import if needed
import json

class DepthTrainer:
    def __init__(self, model, trainloader, valloader, max_depth, loss_fn, device='mps'):
        self.accelerator = Accelerator(gradient_accumulation_steps=2)
        self.device = self.accelerator.device
        self.print = self.accelerator.print
        self.max_depth = max_depth

        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.criterion = loss_fn
        self.best_models = []
        self.best_val_loss = float('inf')
        self.apply_quadratic = False

        self.train_losses: List[float] = [] 

    def set_accelerator(self, optimizer, scheduler):
        self.scheduler = scheduler
        self.optimizer = optimizer
        if scheduler:
            self.model, self.optimizer, self.trainloader, self.valloader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.trainloader, self.valloader, self.scheduler
            )
        else:
            print('no scheduler set')
            self.model, self.optimizer, self.trainloader, self.valloader= self.accelerator.prepare(
                self.model, self.optimizer, self.trainloader, self.valloader
            )
        
    def train(self, epochs=10, 
            save_path="checkpoints", val_interval=1, start_epoch=0):

        if save_path: os.makedirs(save_path, exist_ok=True)

        metrics_log = []  # list of dicts, one per epoch

        for epoch in range(start_epoch, epochs):
            self.print(f"\n[Train] Epoch {epoch+1}/{epochs} START")
            self.criterion.set_epoch(epoch)

            # Train
            train_loss, train_metrics = self.run_train_epoch()

            # Optionally Validate
            if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
                val_loss, val_metrics = self.run_val_epoch()
            else:
                val_loss = float('inf')
                val_metrics = {k: float('inf') for k in train_metrics}

            self.print(f"[Epoch {epoch+1}]")
            self.print(f"  Train Loss: {train_loss:.4f}")
            
            self._print_epoch_metric_table(epoch + 1, train_metrics, val_metrics, train_loss, val_loss)

            # Save model
            if save_path:
                ckpt_path = os.path.join(save_path, f"epoch_{epoch:03d}.pth")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, ckpt_path)

            # Save metrics for this epoch
            metrics_log.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            })

            # Write JSON log every time (overwrite)
            if save_path:
                with open(os.path.join(save_path, "metrics.json"), "w") as f:
                    json.dump(metrics_log, f, indent=2)

            self._print_metric_progress_log(metrics_log)

        self.print("[Train] Done.")

    def run_train_epoch(self):
        self.model.train()
        running_loss = RunningLoss()
        all_metrics_list = []
        batch_losses: List[float] = [] 

        for i, sample in enumerate(self.trainloader):
            try:
                with self.accelerator.accumulate(self.model):
                    start_t = time.time()
                    self.optimizer.zero_grad()

                    img = sample['rgb_image'].to(self.device)
                    depth = sample['xyz_image'].to(self.device)

                    pred = self.model(img).unsqueeze(1)
                    # pred = self.apply_quadratic_correction(pred)
                    #if pred.dim() == 4 and pred.shape[1] == 1:
                    
                    valid = (depth >= 0.001) & (depth <= self.max_depth)
                    if valid.sum() < 1:
                        continue


                    # loss = self.criterion(pred, depth, valid)
                    loss, log_dict = self.criterion(pred, depth, valid, img)
                    # Safety check
                    if not torch.isfinite(loss):
                        print("[FATAL] Non-finite loss encountered:", loss.item())
                        raise ValueError("Loss is NaN or Inf")

                    self.accelerator.backward(loss)
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # for name, param in self.model.named_parameters():
                    #     if not param.is_floating_point():
                    #         print(f"[ERROR] Param {name} is not a floating point tensor: {param.dtype}")
                    #     if param.dtype != torch.float32:
                    #         print(f"[ERROR] Param {name} is {param.dtype}, expected float32")
                    #     if not param.device.type == 'mps':
                    #         print(f"[WARNING] Param {name} is on {param.device}, not MPS")
                    #     if torch.isnan(param).any() or torch.isinf(param).any():
                    #         print(f"[ERROR] Param {name} contains NaNs or Infs!")

                    self.optimizer.step()


                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None and not torch.isfinite(param.grad).all():
                    #         self.print(f"⚠️ Non-finite gradient detected in {name}")

                    running_loss.update(loss.item())
                    batch_losses.append(loss.item())

                    lr_curr = self.optimizer.param_groups[0]['lr']

                    with torch.no_grad():
                        pred_np = pred[valid].detach().cpu().numpy()
                        depth_np = depth[valid].detach().cpu().numpy()
                        metrics = MonocularDepthMetrics(pred_np, depth_np, (depth_np >= .001) & (depth_np <= self.max_depth)).compute_all_metrics()
                        all_metrics_list.append(metrics)

                    end_t = time.time()
                    
                    # for name, param in self.criterion.named_parameters():
                    #     self.print(f"[DEBUG] {name} grad: {param.grad}")
                    
                    # self.print(
                    #     f"[Train Iter {i}/{len(self.trainloader)}] "
                    #     f"SiLog = {log_dict['silog']:.4f} | "
                    #     f"Grad = {log_dict.get('grad', 0):.4f} | "
                    #     f"Smooth = {log_dict.get('smooth', 0):.4f} | "
                    #     f"Total = {log_dict['total']:.4f} | "
                    #     f"MAE = {metrics['MAE']:.2f} | "
                    #     f"#ValidPx = {valid.sum()} | "
                    #     f"Time = {end_t - start_t:.2f}s | "
                    #     f"Log σ²: silog = {log_dict.get('log_var_silog', 0):.2f}, "
                    #     f"grad = {log_dict.get('log_var_grad', 0):.2f}, "
                    #     f"smooth = {log_dict.get('log_var_smooth', 0):.2f}"
                    # )
                    self.print(
                        f"[Train Iter {i}/{len(self.trainloader)}] "
                        f"Loss = {log_dict['total']:.4f} | "
                        f"GradNorm = {grad_norm:.2f} | "
                        f"LR = {lr_curr:.2e} | "
                        f"Time = {end_t - start_t:.2f}s"
                    )

            except Exception as e:
                print(ValueError(f'Training Loop error: {e}'))
                continue

        self.train_losses = batch_losses  
        
        avg_loss = running_loss.average()
        return avg_loss, self._average_metrics(all_metrics_list)

    def run_val_epoch(self):
        self.model.eval()
        running_loss = RunningLoss()
        all_metrics_list = []

        self.print("[Val] Starting validation epoch...")

        with torch.no_grad():
            for i, sample in enumerate(self.valloader):
                try:

                    img = sample['rgb_image'].to(self.device)
                    depth = sample['xyz_image'].to(self.device)

                    pred = self.model(img).unsqueeze(1)
                    pred = self.apply_quadratic_correction(pred)
                    
                    #pred = F.interpolate(pred, depth.shape[-2:], mode='nearest')#, align_corners=True)

                    valid = (depth >= 0.001) & (depth <= self.max_depth)
                    if valid.sum() < 1:
                        self.print(f"[Val] Sample {i+1} has 0 valid pixels. Skipping.")
                        continue

                    loss, log_dict = self.criterion(pred, depth, valid, img)
                    running_loss.update(loss.item())

                    pred_np = pred[valid].detach().cpu().numpy()
                    depth_np = depth[valid].detach().cpu().numpy()
                    mask_np = (depth_np > .001) & (depth_np  < self.max_depth)
                    metrics = MonocularDepthMetrics(pred_np, depth_np, mask_np).compute_all_metrics()
                    all_metrics_list.append(metrics)

                    self.print(
                            f"[Val] Sample {i+1}: "
                            f"Loss = {loss.item():.4f} | "
                            f"SiLog = {log_dict['silog']:.4f} | "
                            f"Grad = {log_dict.get('grad', 0):.4f} | "
                            f"Smooth = {log_dict.get('smooth', 0):.4f} | "
                            f"Total = {log_dict['total']:.4f} | "
                            f"MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}"
                        )

                except Exception as e:
                    print(f'Validation loop error: {e}')
                    continue

        avg_loss = running_loss.average()
        self.print(f"[Val] Epoch Average Loss: {avg_loss:.4f}")
        self.print("[Val] Average Metrics:")
        for k, v in self._average_metrics(all_metrics_list).items():
            self.print(f"    {k}: {v:.4f}")
        return avg_loss, self._average_metrics(all_metrics_list)

    def _average_metrics(self, metrics_list):
        if not metrics_list:
            return {k: float('inf') for k in [
                "MAE", "RMSE", "Scale-Invariant RMSE",
                "Threshold Accuracy (δ=1.25)", "Threshold Accuracy (δ=1.25^2)", "Threshold Accuracy (δ=1.25^3)",
                "Log RMSE", "Mean Relative Error", "Mean Squared Log Error",
                "Structural Similarity (SSIM)", "Edge-Aware Loss"
            ]}
        avg = {}
        keys = metrics_list[0].keys()
        for k in keys:
            avg[k] = float(np.mean([m[k] for m in metrics_list]))
        return avg

    def _print_epoch_metric_table(self, epoch, train_metrics, val_metrics, train_loss=None, val_loss=None):
        def format_metric_row(name, train_val, val_val):
            train_str = f"{train_val:.4f}" if np.isfinite(train_val) else "   ---   "
            val_str   = f"{val_val:.4f}" if np.isfinite(val_val) else "   ---   "
            return f"{name:<32} | {train_str:^10} | {val_str:^10}"

        self.print(f"\n📊 Metric Summary (Epoch {epoch})")
        header = f"{'Metric':<32} | {'Train':^10} | {'Val':^10}"
        self.print(header)
        self.print("-" * len(header))

        if train_loss is not None and val_loss is not None:
            self.print(format_metric_row("Loss", train_loss, val_loss))

        for k in sorted(train_metrics.keys()):
            self.print(format_metric_row(k, train_metrics[k], val_metrics[k]))

    def _print_metric_progress_log(self, epoch_log):
        self.print("\n📈 Training Progress (per Epoch)")
        self.print("-" * 60)
        for entry in epoch_log:
            epoch = entry["epoch"]
            self.print(f"📅 Epoch {epoch}")
            self.print(f"   Train Loss: {entry['train_loss']:.4f}")
            self.print(f"   Val   Loss: {entry['val_loss']:.4f}")

            self.print("   🔹 Metrics:")
            for k in sorted(entry["train_metrics"].keys()):
                train_val = entry["train_metrics"].get(k, float("inf"))
                val_val = entry["val_metrics"].get(k, float("inf"))
                t_str = f"{train_val:.4f}" if np.isfinite(train_val) else "---"
                v_str = f"{val_val:.4f}" if np.isfinite(val_val) else "---"
                self.print(f"      {k:<40}: Train = {t_str}, Val = {v_str}")
            self.print("-" * 60)

    def compute_mae_vs_prediction_curve(self, num_bins=40):
        self.model.eval()
        self.print(f"\n📉 Computing MAE vs Prediction Curve ({num_bins} bins)...")

        # Initialize bins
        bin_edges = np.linspace(0.01, self.max_depth, num_bins + 1)
        bin_mae = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        with torch.no_grad():
            for i, sample in enumerate(self.trainloader):
                if i > 20:
                    break
                img = sample['rgb_image'].to(self.device)
                gt_depth = sample['xyz_image'].to(self.device)

                pred = self.model(img).unsqueeze(1)
                if pred.dim() == 4 and pred.shape[1] == 1:
                    pred = F.interpolate(pred, gt_depth.shape[-2:], mode='nearest', align_corners=True)

                valid = (gt_depth >= 0.001) & (gt_depth <= self.max_depth)
                if valid.sum() < 1:
                    continue

                pred_np = pred[valid].cpu().numpy()
                gt_np = gt_depth[valid].cpu().numpy()
                #error = np.abs(pred_np - gt_np)
                error = pred_np - gt_np

                # Bin by predicted depth
                bin_indices = np.digitize(pred_np, bin_edges) - 1
                for b in range(num_bins):
                    bin_mask = bin_indices == b
                    if bin_mask.any():
                        bin_mae[b] += error[bin_mask].sum()
                        bin_counts[b] += bin_mask.sum()

        # Compute MAE per bin
        bin_mae = bin_mae / np.maximum(bin_counts, 1)

        # Return bin centers and MAE
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, bin_mae

    def plot_mae_vs_prediction(self, bin_centers, bin_mae, epoch, save_dir="checkpoints/mae_curves"):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Fit a robust quadratic curve
        from sklearn.linear_model import RANSACRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        # Prepare X and y
        X = bin_centers.reshape(-1, 1)
        y = bin_mae

        # Create a pipeline: Polynomial (degree 2) + RANSAC regression
        model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor())
        model.fit(X, y)

        # Predict over smooth x
        x_smooth = np.linspace(min(bin_centers), max(bin_centers), 200).reshape(-1, 1)
        y_smooth = model.predict(x_smooth)

        # Extract coefficients from fitted model
        coefs = model.named_steps['ransacregressor'].estimator_.coef_
        intercept = model.named_steps['ransacregressor'].estimator_.intercept_
        a = coefs[2]  # x^2
        b = coefs[1]  # x
        c = intercept  # constant

        # Create plot
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, bin_mae, 'o', label="Binned MAE")
        plt.plot(x_smooth, y_smooth, 'r--', label="Quadratic Fit")

        # Add equation to plot
        equation = f"MAE = {a:.4f}x² + {b:.4f}x + {c:.4f}"
        plt.title(f"MAE vs Predicted Depth (Epoch {epoch})\n{equation}")
        plt.xlabel("Predicted Depth (m)")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(save_dir, f"mae_vs_pred_epoch_{epoch:03d}.png")
        plt.savefig(save_path)
        plt.close()

        self.print(f"📉 Saved MAE vs Predicted Depth plot with quadratic fit to: {save_path}")
        self.print(f"📐 Quadratic Coefficients (Epoch {epoch}): a = {a:.6f}, b = {b:.6f}, c = {c:.6f}")

    def set_quadratic_correction(self, a, b, c):
        """Enable and store quadratic coefficients for correction."""
        self.quadratic_coeffs = (a, b, c)
        self.apply_quadratic = True
        self.print(f"[Correction] Quadratic correction enabled: MAE = {a:.4f}x² + {b:.4f}x + {c:.4f}")

    def apply_quadratic_correction(self, pred):
        """Safely applies quadratic correction to predicted depth."""
        if not self.apply_quadratic or self.quadratic_coeffs is None:
            return pred

        a, b, c = self.quadratic_coeffs

        #with torch.no_grad():
        #correction = a * pred.detach()**2 + b * pred.detach() + c
        correction = b * pred + c
        correction = torch.clamp(correction, -20.0, 20.0)  # 🔒 tighter control

        corrected = pred - correction
        corrected = torch.clamp(corrected, min=1e-3, max=100.0)  # 🧼 safe depth range

        return corrected


class RunningLoss:
    def __init__(self):
        self.reset()
    def update(self, loss, n=1):
        self.total_loss += loss * n
        self.count += n
    def average(self):
        return self.total_loss / self.count if self.count > 0 else 0.0
    def reset(self):
        self.total_loss = 0.0
        self.count = 0