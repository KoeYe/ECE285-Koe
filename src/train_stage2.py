"""
Stage 2: Transition Grounding.

Full pipeline: (o_t, a_t) → z_hat_{t+1} → o_hat_{t+1}
Gradients flow through g_phi (frozen but differentiable) into f_theta.

Loss:
  L_ground(θ) = L_pixel + λ * L_embed
  L_pixel = ||o_hat - o_{t+1}||_2^2
  L_embed = ||E_v(o_hat) - E_v(o_{t+1})||_2^2

Usage:
  python train_stage2.py                   # default λ=0.1
  python train_stage2.py --lambda_embed 0.0
  python train_stage2.py --K 4
  python train_stage2.py --no_stage0       # start from scratch (no stage0 ckpt)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from model import WorldModel, K as DEFAULT_K, D_LAT
from losses import stage2_loss, compute_psnr, compute_ssim, compute_mse
from dataset import get_loaders

# ── Config ─────────────────────────────────────────────────────────────────
CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR = "/data/koe/ECE285-Final/results"
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(lambda_embed=0.1, K_lat=DEFAULT_K, d_lat=D_LAT,
          no_stage0=False, freeze_decoder=True, tag="default"):
    """
    Main training function for Stage 2.

    Args:
        lambda_embed:   weight for embedding loss
        K_lat:          number of latent tokens
        d_lat:          latent token dimension
        no_stage0:      if True, don't load any pretrained checkpoint
        freeze_decoder: if True, keep g_phi frozen; if False, unfreeze last block
        tag:            name suffix for checkpoint/results
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[Stage 2] tag={tag}  λ={lambda_embed}  K={K_lat}  freeze_dec={freeze_decoder}")
    print(f"          Device: {DEVICE}")

    model = WorldModel(K=K_lat, d_lat=d_lat).to(DEVICE)

    # Load Stage 1 checkpoint (or Stage 0 if no stage1 exists)
    # Filter out shape-mismatched keys (e.g. when K differs from default)
    if not no_stage0:
        s1_path = f"{CKPT_DIR}/stage1.pth"
        s0_path = f"{CKPT_DIR}/stage0.pth"
        load_path = s1_path if os.path.exists(s1_path) else (s0_path if os.path.exists(s0_path) else None)
        if load_path:
            ckpt_sd = torch.load(load_path, map_location=DEVICE)["model_state"]
            cur_sd  = model.state_dict()
            filtered = {k: v for k, v in ckpt_sd.items()
                        if k in cur_sd and v.shape == cur_sd[k].shape}
            skipped = len(ckpt_sd) - len(filtered)
            model.load_state_dict(filtered, strict=False)
            tag_name = "Stage 1" if load_path == s1_path else "Stage 0"
            if skipped:
                print(f"  Loaded {tag_name} (skipped {skipped} shape-mismatched keys for K={K_lat})")
            else:
                print(f"  Loaded {tag_name} checkpoint")

    # Freeze E_v; conditionally freeze/unfreeze g_phi
    freeze(model.encoder)

    if freeze_decoder:
        freeze(model.decoder)
        # Gradients still flow through g_phi because it's differentiable
        # We just don't update its parameters
        params = list(model.latent_predictor.parameters()) + \
                 list(model.action_encoder.parameters())
    else:
        # Unfreeze last decoder block (dec1 + out_conv) for partial adaptation
        freeze(model.decoder)
        unfreeze(model.decoder.dec1)
        unfreeze(model.decoder.out_conv)
        params = list(model.latent_predictor.parameters()) + \
                 list(model.action_encoder.parameters()) + \
                 list(model.decoder.dec1.parameters()) + \
                 list(model.decoder.out_conv.parameters())

    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_warmup_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    scaler    = GradScaler()

    train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE)

    history = {
        "train_loss": [], "train_l_pixel": [], "train_l_embed": [],
        "val_loss":   [], "val_psnr": [], "val_ssim": [], "val_mse": [],
    }
    best_val_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        model.encoder.eval()

        if freeze_decoder:
            model.decoder.eval()

        t_loss = t_pixel = t_embed = 0.0

        for obs, action, next_obs in tqdm(train_loader, desc=f"E{epoch:02d}", leave=False):
            obs      = obs.to(DEVICE)
            action   = action.to(DEVICE)
            next_obs = next_obs.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                # Forward: (o_t, a_t) → z_hat → o_hat
                vis_tokens = model.encoder(obs)
                action_tok = model.action_encoder(action)
                z_hat      = model.latent_predictor(vis_tokens, action_tok)
                obs_hat    = model.decoder(obs, z_hat)

                # Compute embedding loss with frozen E_v
                with torch.no_grad():
                    vis_true = model.encoder(next_obs)
                vis_hat = model.encoder(obs_hat)   # gradients flow through obs_hat → f_theta

                loss, sub = stage2_loss(
                    obs_hat, next_obs, vis_hat, vis_true,
                    lambda_embed=lambda_embed
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            scaler.step(optimizer)
            scaler.update()

            t_loss  += loss.item()
            t_pixel += sub["l_pixel"]
            t_embed += sub["l_embed"]

        t_loss  /= len(train_loader)
        t_pixel /= len(train_loader)
        t_embed /= len(train_loader)
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss = v_psnr = v_ssim = v_mse = 0.0
        n_batch = 0

        with torch.no_grad():
            for obs, action, next_obs in val_loader:
                obs      = obs.to(DEVICE)
                action   = action.to(DEVICE)
                next_obs = next_obs.to(DEVICE)

                vis_tokens = model.encoder(obs)
                action_tok = model.action_encoder(action)
                z_hat      = model.latent_predictor(vis_tokens, action_tok)
                obs_hat    = model.decoder(obs, z_hat)

                vis_true = model.encoder(next_obs)
                vis_hat  = model.encoder(obs_hat)

                loss, _   = stage2_loss(obs_hat, next_obs, vis_hat, vis_true, lambda_embed)
                v_loss   += loss.item()
                v_psnr   += compute_psnr(obs_hat, next_obs)
                v_ssim   += compute_ssim(obs_hat, next_obs)
                v_mse    += compute_mse(obs_hat, next_obs)
                n_batch  += 1

        v_loss /= n_batch; v_psnr /= n_batch; v_ssim /= n_batch; v_mse /= n_batch

        history["train_loss"].append(t_loss)
        history["train_l_pixel"].append(t_pixel)
        history["train_l_embed"].append(t_embed)
        history["val_loss"].append(v_loss)
        history["val_psnr"].append(v_psnr)
        history["val_ssim"].append(v_ssim)
        history["val_mse"].append(v_mse)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={t_loss:.4f} (px={t_pixel:.4f} em={t_embed:.4f})  "
              f"val: PSNR={v_psnr:.2f}dB  SSIM={v_ssim:.4f}  MSE={v_mse:.4f}")

        if v_psnr > best_val_psnr:
            best_val_psnr = v_psnr
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": v_psnr,
                "val_ssim": v_ssim,
                "val_mse":  v_mse,
                "config":   {"lambda_embed": lambda_embed, "K": K_lat, "tag": tag},
            }, f"{CKPT_DIR}/stage2_{tag}.pth")
            print(f"  ✓ Saved best (PSNR={v_psnr:.2f}dB, SSIM={v_ssim:.4f})")

    with open(f"{RESULT_DIR}/stage2_{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Stage 2/{tag}] Done. Best PSNR: {best_val_psnr:.2f}dB")
    return best_val_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_embed", type=float, default=0.1)
    parser.add_argument("--K",            type=int,   default=DEFAULT_K)
    parser.add_argument("--no_stage0",    action="store_true")
    parser.add_argument("--unfreeze_dec", action="store_true")
    parser.add_argument("--tag",          type=str,   default="default")
    args = parser.parse_args()

    train(
        lambda_embed   = args.lambda_embed,
        K_lat          = args.K,
        no_stage0      = args.no_stage0,
        freeze_decoder = not args.unfreeze_dec,
        tag            = args.tag,
    )
