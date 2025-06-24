import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from accelerate import Accelerator
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random


class DepthModelTester:
    def __init__(self, model, trainloader, device='mps', lr=1e-4, steps=100, max_depth=10.0, loss_fn=None):
        self.model = model
        self.device = torch.device(device)
        self.lr = lr
        self.steps = steps
        self.max_depth = max_depth
        self.loss_fn = loss_fn if loss_fn is not None else self.default_loss
        self.losses = []
        self.trainloader = trainloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def show_samples(self, loader, n_samples=5):
        import os
        os.environ['PLOT_DEBUG'] = 'True'
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])

        for t, sample in enumerate(loader):
            if 'data' in sample:
                sample = sample['data']

            if t > n_samples:
                break

            print(f'image shape: {sample["rgb_image"].shape}')
            print(f'image shape: {sample["xyz_image"].shape}')

            img = sample['rgb_image'][0].permute(1, 2, 0).detach().cpu().numpy()
            depth = sample['xyz_image'][0, 0].detach().cpu().numpy()

            denorm_img = img * std + mean
            denorm_img = np.clip(denorm_img, 0, 255).astype(np.uint8)

            plt.figure(figsize=(12, 4))
            plt.suptitle(f'Tester output: norm img range [{img.min():.2f}, {img.max():.2f}]')

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Normalized Image')

            plt.subplot(1, 3, 2)
            plt.imshow(denorm_img)
            plt.title('Denormalized Image')

            plt.subplot(1, 3, 3)
            plt.imshow(depth, cmap='viridis')
            plt.title('Depth Map')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        os.environ['PLOT_DEBUG'] = 'false'

    def default_loss(self, pred, target, valid):
        return F.l1_loss(pred[valid], target[valid]), {}

    def _valid_mask(self, depth):
        return (depth >= 0.001) & (depth <= self.max_depth)

    def _show_stats(self, step, pred, ref_weight):
        print(f"[Step {step}] pred: mean={pred.mean():.4f}, min={pred.min():.4f}, max={pred.max():.4f}")
        delta = (self.model.depth_head.scratch.output_conv1.weight.data - ref_weight).abs().mean().item()
        print(f"[Step {step}] ΔWeight: {delta:.6f}")

    def _plot_results(self, img, depth, pred):
        img = img[0].detach().cpu()
        pred = pred[0, 0].detach().cpu()
        depth = depth[0, 0].detach().cpu()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img.permute(1, 2, 0))
        axs[0].set_title("RGB Image")
        axs[1].imshow(depth, cmap='viridis')
        axs[1].set_title("Ground Truth Depth")
        axs[2].imshow(pred, cmap='viridis')
        axs[2].set_title("Final Prediction")
        plt.tight_layout()
        plt.show()

        plt.plot(self.losses)
        plt.title("Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def run_trainer_one_epoch(self, epoch_val_interval: int = 1) -> float:
        from panorai.depth.trainers.depth_trainer import DepthTrainer

        # Get a random subset of the real training loader
        dataset = self.trainloader.dataset
        indices = random.sample(range(len(dataset)), min(32, len(dataset)))
        subset_loader = DataLoader(Subset(dataset, indices), batch_size=4, shuffle=True, collate_fn=self.trainloader.collate_fn)

        trainer = DepthTrainer(
            model=self.model,
            trainloader=subset_loader,
            valloader=subset_loader,
            loss_fn=self.loss_fn,
            max_depth=self.max_depth,
            device=self.device,
            debug=True,
        )

        trainer.set_accelerator(self.optimizer, scheduler=None)
        trainer.train(epochs=1, val_interval=epoch_val_interval, save_path=None)

        epoch_loss = float(np.mean(trainer.train_losses)) if trainer.train_losses else float("nan")
        self.losses.append(epoch_loss)
        return epoch_loss

    def run_trainer_until_convergence(
        self,
        *,
        target_loss: float = 0.01,
        patience: int = 5,
        min_delta: float = 1e-4,
        max_epochs: int = 50,
        epoch_val_interval: int = 1,
    ):
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            epoch_loss = self.run_trainer_one_epoch(epoch_val_interval=epoch_val_interval)
            print(f"[Epoch {epoch}] loss = {epoch_loss:.6f}")

            if epoch_loss <= target_loss:
                print("✅ Target loss reached – stopping early.")
                break

            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping – no improvement for {patience} epochs.")
                    break
        else:
            print("⚠️  Reached max_epochs without convergence.")
