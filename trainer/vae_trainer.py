import torch
import torch.nn.functional as F
from pathlib import Path

from VAE.VAE import VAE
from Dataset.XRayChest.xraychest import XRayChestDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm


class VAETrainer:
    def __init__(self, vae: VAE, dataset: Dataset, val_dataset: Dataset = None):
        # Auto-select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.vae = vae.to(self.device)
        self.val_dataset = val_dataset

        # Hyperparameters
        self.batch_size = 8
        self.lr = 1e-4
        self.num_epochs = 100

        # KL warmup: linearly increase kl_weight from 0 to kl_weight_max
        # over the first kl_warmup_epochs epochs
        self.kl_weight_max = 0.001
        self.kl_warmup_epochs = 20
        self.kl_weight = 0.0

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        self.val_dataloader = None
        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

        # Output directories
        self.ckpt_dir = Path("checkpoints")
        self.sample_dir = Path("samples")
        self.ckpt_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)

    @staticmethod
    def vae_loss(x, x_hat, mu, logvar, kl_weight=1.0):
        # Sum over pixels, mean over batch — keeps rec and kl on the same scale
        rec_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2, 3)).mean()
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
        return rec_loss + kl_weight * kl_loss, rec_loss, kl_loss

    def train_one_epoch(self, epoch):
        self.vae.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        pbar = tqdm(self.dataloader, desc=f"epoch {epoch}", leave=False)

        for x in pbar:
            x = x.to(self.device)
            x_hat, mu, logvar = self.vae(x)

            loss, rec, kl = self.vae_loss(x, x_hat, mu, logvar, self.kl_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()
            pbar.set_postfix(
                loss=loss.item(), rec=rec.item(),
                kl=kl.item(), kl_w=self.kl_weight
            )

        n = len(self.dataloader)
        return total_loss / n, total_rec / n, total_kl / n

    @torch.no_grad()
    def validate(self):
        if self.val_dataloader is None:
            return 0.0, 0.0, 0.0
        self.vae.eval()
        val_loss = 0.0
        val_rec = 0.0
        val_kl = 0.0
        for x in self.val_dataloader:
            x = x.to(self.device)
            x_hat, mu, logvar = self.vae(x)
            loss, rec, kl = self.vae_loss(x, x_hat, mu, logvar, self.kl_weight)
            val_loss += loss.item()
            val_rec += rec.item()
            val_kl += kl.item()
        n = len(self.val_dataloader)
        return val_loss / n, val_rec / n, val_kl / n

    @torch.no_grad()
    def save_samples(self, epoch, n=8):
        """Sample from random latent vectors and save generated images"""
        self.vae.eval()
        z = torch.randn(n, 64, 8, 8, device=self.device)
        samples = self.vae.decoder(z)
        save_image(samples, self.sample_dir / f"epoch_{epoch:03d}.png", nrow=4)

    @torch.no_grad()
    def reconstruct_samples(self, epoch, n=8):
        """Reconstruct real images and save side-by-side comparison"""
        self.vae.eval()
        x = next(iter(self.val_dataloader or self.dataloader))[:n].to(self.device)
        x_hat, _, _ = self.vae(x)
        comparison = torch.cat([x, x_hat], dim=0)
        save_image(comparison, self.sample_dir / f"recon_{epoch:03d}.png", nrow=n)

    def save_checkpoint(self, epoch, loss):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.vae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
        }, self.ckpt_dir / f"vae_epoch_{epoch:03d}.pt")

    def train(self):
        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            # KL warmup: linearly ramp kl_weight from 0 to kl_weight_max
            if epoch < self.kl_warmup_epochs:
                self.kl_weight = self.kl_weight_max * (epoch / self.kl_warmup_epochs)
            else:
                self.kl_weight = self.kl_weight_max

            train_loss, rec_loss, kl_loss = self.train_one_epoch(epoch)
            self.scheduler.step()
            val_loss, val_rec, val_kl = self.validate()

            tqdm.write(
                f"epoch {epoch:3d} | kl_w {self.kl_weight:.5f} | "
                f"train {train_loss:.2f} (rec {rec_loss:.2f}, kl {kl_loss:.2f}) | "
                f"val {val_loss:.2f} (rec {val_rec:.2f}, kl {val_kl:.2f})"
            )

            # Save samples and checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_samples(epoch)
                self.reconstruct_samples(epoch)
                self.save_checkpoint(epoch, train_loss)

            # Save best model based on validation rec loss
            if val_rec < best_loss:
                best_loss = val_rec
                torch.save(self.vae.state_dict(), self.ckpt_dir / "vae_best.pt")


if __name__ == "__main__":
    train_dataset = XRayChestDataset(split="train", img_size=256)
    val_dataset = XRayChestDataset(split="val", img_size=256)

    vae = VAE(latent_channels=64)

    trainer = VAETrainer(vae=vae, dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()