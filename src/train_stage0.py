"""
Stage 0: Decoder Domain Adaptation.

Train g_phi as a conditional autoencoder so it learns to reconstruct
FrozenLake images conditioned on E_v features. This adapts the decoder
to the visual domain before receiving latent-block conditioning.

Loss:  L_diff(φ) = MSE( g_φ(pool(E_v(o_t))), o_{t+1} )
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from model import WorldModel, K, D_LAT
from losses import stage0_loss, compute_psnr
from dataset import get_loaders

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR = "/data/koe/ECE285-Final/results"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
K_LAT      = K
D_LATENT   = D_LAT


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[Stage 0] Device: {DEVICE}")
    model = WorldModel(K=K_LAT, d_lat=D_LATENT).to(DEVICE)

    # Stage 0: train E_v + g_phi together
    # f_theta (latent predictor) is NOT used in Stage 0 forward pass
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE)

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for obs, action, next_obs in tqdm(train_loader, desc=f"E{epoch:02d} Train", leave=False):
            obs      = obs.to(DEVICE)
            next_obs = next_obs.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                # Stage 0 forward: use E_v to get proxy latent, then decode
                vis_tokens = model.encoder(obs)                    # (B, M, d_v)
                z_proxy    = model.encoder.pool_to_k(vis_tokens)  # (B, K, d_v=64)
                obs_hat    = model.decoder(obs, z_proxy)           # (B, 3, 64, 64)
                loss       = stage0_loss(obs_hat, next_obs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        psnr_vals = []

        with torch.no_grad():
            for obs, action, next_obs in val_loader:
                obs      = obs.to(DEVICE)
                next_obs = next_obs.to(DEVICE)
                vis_tokens = model.encoder(obs)
                z_proxy    = model.encoder.pool_to_k(vis_tokens)
                obs_hat    = model.decoder(obs, z_proxy)
                loss       = stage0_loss(obs_hat, next_obs)
                val_loss  += loss.item()
                psnr_vals.append(compute_psnr(obs_hat, next_obs))

        val_loss /= len(val_loader)
        val_psnr  = sum(psnr_vals) / len(psnr_vals)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_PSNR={val_psnr:.2f}dB")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            }, f"{CKPT_DIR}/stage0.pth")
            print(f"  ✓ Saved best checkpoint (val_PSNR={val_psnr:.2f}dB)")

    # Save history
    with open(f"{RESULT_DIR}/stage0_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n[Stage 0] Done. Best val_PSNR:", max(history["val_psnr"]), "dB")


if __name__ == "__main__":
    train()
