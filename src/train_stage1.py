"""
Stage 1: Latent Format Stabilization (SFT).

Train f_theta to emit well-formed latent block tokens z_hat that align
with E_v(o_{t+1}) embeddings. Both E_v and g_phi are frozen.

Loss:
  z_target = pool_to_k( E_v(o_{t+1}) )       # (B, K, d_v)
  L_format(θ) = 0.9 * MSE(z_hat, z_target) + 0.1 * (1 - cos_sim)

Monitoring: 'format success rate' = fraction where token norms ∈ [0.1, 10]
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from model import WorldModel, K, D_LAT
from losses import stage1_loss
from dataset import get_loaders

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_DIR    = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR  = "/data/koe/ECE285-Final/results"
STAGE0_CKPT = f"{CKPT_DIR}/stage0.pth"
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
K_LAT       = K
D_LATENT    = D_LAT


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[Stage 1] Device: {DEVICE}")

    model = WorldModel(K=K_LAT, d_lat=D_LATENT).to(DEVICE)

    # Load Stage 0 weights
    ckpt = torch.load(STAGE0_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded Stage 0 checkpoint (val_PSNR={ckpt['val_psnr']:.2f}dB)")

    # Freeze E_v and g_phi; train only f_theta and action encoder
    freeze(model.encoder)
    freeze(model.decoder)
    params = list(model.latent_predictor.parameters()) + \
             list(model.action_encoder.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE)

    history = {"train_loss": [], "val_loss": [], "val_cos_sim": [], "format_rate": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        model.encoder.eval()   # keep BN in eval mode (frozen)
        model.decoder.eval()

        train_loss = 0.0
        train_cos  = 0.0

        for obs, action, next_obs in tqdm(train_loader, desc=f"E{epoch:02d} Train", leave=False):
            obs      = obs.to(DEVICE)
            action   = action.to(DEVICE)
            next_obs = next_obs.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                # z_target: pool E_v(o_{t+1}) to K tokens
                with torch.no_grad():
                    vis_next   = model.encoder(next_obs)          # (B, M, d_v)
                    z_target   = model.encoder.pool_to_k(vis_next) # (B, K, d_v)

                # f_theta forward
                vis_tokens = model.encoder(obs)
                action_tok = model.action_encoder(action)
                z_hat      = model.latent_predictor(vis_tokens, action_tok)

                loss, sub = stage1_loss(z_hat, z_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_cos  += sub["cos_sim"]

        train_loss /= len(train_loader)
        train_cos  /= len(train_loader)
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_cos     = 0.0
        format_ok   = 0
        total_tok   = 0

        with torch.no_grad():
            for obs, action, next_obs in val_loader:
                obs      = obs.to(DEVICE)
                action   = action.to(DEVICE)
                next_obs = next_obs.to(DEVICE)

                vis_next   = model.encoder(next_obs)
                z_target   = model.encoder.pool_to_k(vis_next)
                vis_tokens = model.encoder(obs)
                action_tok = model.action_encoder(action)
                z_hat      = model.latent_predictor(vis_tokens, action_tok)

                loss, sub  = stage1_loss(z_hat, z_target)
                val_loss  += loss.item()
                val_cos   += sub["cos_sim"]

                # Format success rate: token norms in [0.1, 10]
                norms = z_hat.reshape(-1, z_hat.shape[-1]).norm(dim=-1)
                format_ok += ((norms >= 0.1) & (norms <= 10.0)).sum().item()
                total_tok += norms.numel()

        val_loss  /= len(val_loader)
        val_cos   /= len(val_loader)
        fmt_rate   = format_ok / total_tok

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cos_sim"].append(val_cos)
        history["format_rate"].append(fmt_rate)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"cos_sim={val_cos:.4f}  "
              f"format_rate={fmt_rate:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cos_sim": val_cos,
            }, f"{CKPT_DIR}/stage1.pth")
            print(f"  ✓ Saved best checkpoint (cos_sim={val_cos:.4f})")

    with open(f"{RESULT_DIR}/stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n[Stage 1] Done. Best val cos_sim:", max(history["val_cos_sim"]))


if __name__ == "__main__":
    train()
