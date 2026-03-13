"""
All loss functions for the three-stage training pipeline.

Matches paper notation exactly:
  Stage 0: L_diff(φ)    = MSE reconstruction
  Stage 1: L_format(θ)  = 0.9 * MSE_tokens + 0.1 * cosine_loss
  Stage 2: L_ground(θ)  = L_pixel + λ * L_embed

  Eval:    Drift_H       = (1/H) Σ pixel_err + α * embed_err
"""

import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import numpy as np


# ── Stage 0 ────────────────────────────────────────────────────────────────

def stage0_loss(obs_hat: torch.Tensor, obs_true: torch.Tensor) -> torch.Tensor:
    """
    L_diff(φ) = ||o_hat - o_{t+1}||_2^2
    Images in [-1, 1].
    """
    return F.mse_loss(obs_hat, obs_true)


# ── Stage 1 ────────────────────────────────────────────────────────────────

def stage1_loss(
    z_hat: torch.Tensor,
    z_target: torch.Tensor,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    L_format(θ) = 0.9 * L_MSE + 0.1 * L_cosine

    Args:
        z_hat:    (B, K, d) – predicted latent block
        z_target: (B, K, d) – target from encoder pool
        alpha:    weight for cosine similarity loss
    Returns:
        total loss tensor, dict of sub-losses
    """
    B, K, d = z_hat.shape

    # Token-level MSE
    l_mse = F.mse_loss(z_hat, z_target)

    # Cosine similarity loss (averaged over B*K pairs)
    z_hat_flat   = z_hat.reshape(-1, d)       # (B*K, d)
    z_tgt_flat   = z_target.reshape(-1, d)
    cos_sim = F.cosine_similarity(z_hat_flat, z_tgt_flat, dim=-1).mean()
    l_cos   = 1.0 - cos_sim

    total = (1.0 - alpha) * l_mse + alpha * l_cos
    return total, {"l_mse": l_mse.item(), "l_cos": l_cos.item(), "cos_sim": cos_sim.item()}


# ── Stage 2 ────────────────────────────────────────────────────────────────

def stage2_loss(
    obs_hat: torch.Tensor,
    obs_true: torch.Tensor,
    vis_hat: torch.Tensor,
    vis_true: torch.Tensor,
    lambda_embed: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    L_ground(θ) = L_pixel + λ * L_embed

    Args:
        obs_hat:     (B, 3, 64, 64) – predicted next observation
        obs_true:    (B, 3, 64, 64) – ground truth next observation
        vis_hat:     (B, M, d_v)    – E_v(o_hat)
        vis_true:    (B, M, d_v)    – E_v(o_{t+1})
        lambda_embed: weight for embedding loss
    Returns:
        total loss tensor, dict of sub-losses
    """
    l_pixel = F.mse_loss(obs_hat, obs_true)
    l_embed = F.mse_loss(vis_hat, vis_true)
    total   = l_pixel + lambda_embed * l_embed
    return total, {"l_pixel": l_pixel.item(), "l_embed": l_embed.item()}


# ── Evaluation Metrics ─────────────────────────────────────────────────────

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (B, 3, H, W) tensor in [-1,1] to (B, H, W, 3) uint8 numpy."""
    arr = (t.clamp(-1, 1).permute(0, 2, 3, 1).cpu().float().numpy() + 1) * 127.5
    return arr.astype(np.uint8)


def compute_psnr(obs_hat: torch.Tensor, obs_true: torch.Tensor) -> float:
    """Mean PSNR over batch. Inputs in [-1, 1]."""
    pred = tensor_to_uint8(obs_hat)
    true = tensor_to_uint8(obs_true)
    psnr_vals = [
        skimage_psnr(true[i], pred[i], data_range=255)
        for i in range(len(pred))
    ]
    return float(np.mean(psnr_vals))


def compute_ssim(obs_hat: torch.Tensor, obs_true: torch.Tensor) -> float:
    """Mean SSIM over batch. Inputs in [-1, 1]."""
    pred = tensor_to_uint8(obs_hat)
    true = tensor_to_uint8(obs_true)
    ssim_vals = [
        skimage_ssim(true[i], pred[i], data_range=255, channel_axis=2)
        for i in range(len(pred))
    ]
    return float(np.mean(ssim_vals))


def compute_mse(obs_hat: torch.Tensor, obs_true: torch.Tensor) -> float:
    """Pixel-space MSE in [0, 1] normalized."""
    return F.mse_loss(
        obs_hat.clamp(-1, 1) * 0.5 + 0.5,
        obs_true.clamp(-1, 1) * 0.5 + 0.5,
    ).item()


def compute_embed_mse(vis_hat: torch.Tensor, vis_true: torch.Tensor) -> float:
    """MSE in embedding space."""
    return F.mse_loss(vis_hat, vis_true).item()


def rollout_drift(
    model,
    obs0: torch.Tensor,
    actions: torch.Tensor,
    gt_obs: torch.Tensor,
    alpha: float = 1.0,
    device: str = "cuda",
) -> dict:
    """
    Compute Drift_H for H ∈ {5, 10, 20} steps.

    Args:
        model:   WorldModel (eval mode)
        obs0:    (B, 3, 64, 64) – initial observation
        actions: (B, H_max) – action sequence
        gt_obs:  (B, H_max, 3, 64, 64) – ground truth future observations
        alpha:   weight for embedding error in drift
    Returns:
        dict with keys "drift_5", "drift_10", "drift_20" and per-step psnr list
    """
    model.eval()
    B, H_max, C, H, W = gt_obs.shape
    H_eval = min(H_max, 20)

    current_obs = obs0.clone().to(device)
    pixel_errs  = []
    embed_errs  = []
    psnr_list   = []

    with torch.no_grad():
        for step in range(H_eval):
            act_step = actions[:, step].to(device)
            out = model(current_obs, act_step)
            obs_hat = out["obs_hat"]

            gt_step = gt_obs[:, step].to(device)
            pe = F.mse_loss(obs_hat.clamp(-1,1), gt_step.clamp(-1,1)).item()

            # Embedding error
            vis_hat  = model.encoder(obs_hat)
            vis_true = model.encoder(gt_step)
            ee = F.mse_loss(vis_hat, vis_true).item()

            pixel_errs.append(pe)
            embed_errs.append(ee)
            psnr_list.append(compute_psnr(obs_hat, gt_step))

            # Next input = predicted observation (compounding error)
            current_obs = obs_hat.detach()

    def drift(H):
        H = min(H, H_eval)
        return (1 / H) * sum(
            pixel_errs[k] + alpha * embed_errs[k] for k in range(H)
        )

    return {
        "drift_5":  drift(5),
        "drift_10": drift(10),
        "drift_20": drift(20),
        "psnr_per_step": psnr_list,
    }
