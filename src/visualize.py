"""
Visualization pipeline:
  1. Rollout strips (ground truth | predicted | pixel diff)
  2. PSNR vs rollout step (drift curves, multi-config)
  3. Training loss curves (3-stage)
  4. Latent space PCA (z_hat colored by action / cell type)
  5. Sample predictions grid

All figures saved to results/figures/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from tqdm import tqdm
import json

from model import WorldModel, K, D_LAT
from dataset import get_loaders, DATA_PATH, TransitionDataset
from eval import build_rollout_sequences

CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR = "/data/koe/ECE285-Final/results"
FIG_DIR    = "/data/koe/ECE285-Final/results/figures"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [-1,1] → (H, W, 3) uint8."""
    arr = (t.clamp(-1, 1).permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5
    return arr.astype(np.uint8)


def load_model(tag="default"):
    ckpt = torch.load(f"{CKPT_DIR}/stage2_{tag}.pth", map_location=DEVICE)
    cfg  = ckpt.get("config", {})
    m    = WorldModel(K=cfg.get("K", K), d_lat=D_LAT).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


# ── 1. Rollout Strips ──────────────────────────────────────────────────────

def plot_rollout_strips(model, sequences, n_seqs=4, H=10, save_path=None):
    """
    For each sequence: 3 rows (Ground Truth | Predicted | |Diff|×10)
    across H steps.
    """
    fig, axes = plt.subplots(
        n_seqs * 3, H,
        figsize=(H * 1.2, n_seqs * 3.6),
        dpi=100
    )
    if n_seqs == 1:
        axes = axes.reshape(3, H)

    row_labels = ["Ground Truth", "Predicted", "|Diff|×10"]

    for si, seq in enumerate(sequences[:n_seqs]):
        # seq is a tuple (obs0_tensor, actions_tensor, gt_frames_tensor)
        obs0    = seq[0].unsqueeze(0).to(DEVICE)
        actions = seq[1]
        gt_obs  = seq[2]
        H_seq   = min(len(actions), H)

        current = obs0.clone()
        preds   = []

        with torch.no_grad():
            for step in range(H_seq):
                act  = actions[step:step+1].to(DEVICE)
                vis  = model.encoder(current)
                atok = model.action_encoder(act)
                z    = model.latent_predictor(vis, atok)
                pred = model.decoder(current, z)
                preds.append(pred.squeeze(0))
                current = pred.detach()

        for step in range(H_seq):
            gt_img   = tensor_to_img(gt_obs[step])
            pred_img = tensor_to_img(preds[step])
            diff_img = np.clip(
                np.abs(gt_img.astype(float) - pred_img.astype(float)) * 10, 0, 255
            ).astype(np.uint8)

            row0 = si * 3
            for ri, img in enumerate([gt_img, pred_img, diff_img]):
                ax = axes[row0 + ri, step]
                ax.imshow(img)
                ax.axis("off")
                if step == 0:
                    ax.set_ylabel(row_labels[ri], fontsize=7, rotation=90, labelpad=3)
                if si == 0 and ri == 0:
                    ax.set_title(f"t+{step+1}", fontsize=7)

    plt.suptitle("Multi-Step Rollout: Ground Truth vs. Predicted", fontsize=11, y=1.01)
    plt.tight_layout()
    path = save_path or f"{FIG_DIR}/rollout_strips.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved rollout strips → {path}")


# ── 2. Drift Curves ────────────────────────────────────────────────────────

def plot_drift_curves(configs: dict, save_path=None):
    """
    configs: {label: psnr_per_step_list}
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for i, (label, psnr_steps) in enumerate(configs.items()):
        steps = list(range(1, len(psnr_steps) + 1))
        ax.plot(steps, psnr_steps, label=label, color=colors[i], linewidth=2, marker="o",
                markersize=3)

    ax.set_xlabel("Rollout Step", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Multi-Step Prediction Quality (Rollout Drift)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(len(v) for v in configs.values()))
    plt.tight_layout()

    path = save_path or f"{FIG_DIR}/drift_curves.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved drift curves → {path}")


# ── 3. Training Loss Curves ────────────────────────────────────────────────

def plot_training_curves(save_path=None):
    """Load all stage history JSONs and plot 3-panel training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    stage_files = [
        ("Stage 0\n(Decoder Adapt.)", f"{RESULT_DIR}/stage0_history.json",
         "val_loss", "Val MSE Loss"),
        ("Stage 1\n(Format SFT)", f"{RESULT_DIR}/stage1_history.json",
         "val_cos_sim", "Val Cosine Sim."),
        ("Stage 2\n(Grounding)", f"{RESULT_DIR}/stage2_default_history.json",
         "val_psnr", "Val PSNR (dB)"),
    ]

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, (title, fpath, key, ylabel), color in zip(axes, stage_files, colors):
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=11)
            continue
        with open(fpath) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist["train_loss"]) + 1))
        ax.plot(epochs, hist["train_loss"], label="Train loss", color=color,
                linewidth=1.5, linestyle="--", alpha=0.7)
        if key in hist:
            ax2 = ax.twinx()
            ax2.plot(epochs, hist[key], label=ylabel, color=color, linewidth=2)
            ax2.set_ylabel(ylabel, fontsize=10, color=color)
            ax2.tick_params(axis="y", colors=color)

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Train Loss", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Three-Stage Training Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{FIG_DIR}/training_curves.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


# ── 4. Latent Space PCA ────────────────────────────────────────────────────

def plot_latent_pca(model, n_samples=2000, save_path=None):
    """PCA of z_hat tokens colored by action taken."""
    ds = TransitionDataset("test", DATA_PATH)
    idxs = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)

    z_hats  = []
    actions = []

    model.eval()
    with torch.no_grad():
        batch_size = 64
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            obs_list   = []
            act_list   = []
            for i in batch_idx:
                o, a, _ = ds[int(i)]
                obs_list.append(o)
                act_list.append(a)
            obs_t = torch.stack(obs_list).to(DEVICE)
            act_t = torch.stack(act_list).to(DEVICE)

            vis  = model.encoder(obs_t)
            atok = model.action_encoder(act_t)
            z    = model.latent_predictor(vis, atok)   # (B, K, d)
            # Pool K tokens to one vector per sample
            z_pool = z.mean(dim=1)                     # (B, d)
            z_hats.append(z_pool.cpu().float().numpy())
            actions.extend(act_t.cpu().numpy())

    z_all = np.concatenate(z_hats, axis=0)
    a_all = np.array(actions)

    pca  = PCA(n_components=2)
    z_2d = pca.fit_transform(z_all)

    action_names = ["Left", "Down", "Right", "Up"]
    colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for ai, (name, color) in enumerate(zip(action_names, colors)):
        mask = a_all == ai
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=name,
                   s=8, alpha=0.6, linewidths=0)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("Latent Space PCA of $\\hat{z}_{t+1}$ (colored by action)", fontsize=11)
    ax.legend(title="Action", fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = save_path or f"{FIG_DIR}/latent_pca.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved latent PCA → {path}")


# ── 5. Sample Predictions Grid ─────────────────────────────────────────────

def plot_sample_predictions(model, n=8, save_path=None):
    """Side-by-side: input | predicted next | ground truth next."""
    ds = TransitionDataset("test", DATA_PATH)
    idxs = np.random.choice(len(ds), n, replace=False)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_labels = ["Input $o_t$", "Predicted $\\hat{o}_{t+1}$", "Ground Truth $o_{t+1}$"]

    model.eval()
    with torch.no_grad():
        for col, idx in enumerate(idxs):
            obs, act, next_obs = ds[int(idx)]
            obs_t = obs.unsqueeze(0).to(DEVICE)
            act_t = act.unsqueeze(0).to(DEVICE)

            vis  = model.encoder(obs_t)
            atok = model.action_encoder(act_t)
            z    = model.latent_predictor(vis, atok)
            pred = model.decoder(obs_t, z)

            imgs = [
                tensor_to_img(obs),
                tensor_to_img(pred.squeeze(0)),
                tensor_to_img(next_obs),
            ]
            for row, img in enumerate(imgs):
                ax = axes[row, col]
                ax.imshow(img)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8)

    plt.suptitle("Sample One-Step Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{FIG_DIR}/sample_predictions.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved sample predictions → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model = load_model("default")

    # 1. Rollout strips — filter out stationary (no-op) sequences
    seqs_raw = build_rollout_sequences(max_seqs=200)
    seqs = [s for s in seqs_raw
            if not all(torch.allclose(s[2][i], s[2][0], atol=0.05)
                       for i in range(1, len(s[2])))]
    # report version: 2 seqs (matches caption), compact
    plot_rollout_strips(model, seqs, n_seqs=2, H=10,
                        save_path=f"{FIG_DIR}/rollout_strips.pdf")
    # slides landscape version: 2 seqs, wide layout
    plot_rollout_strips(model, seqs, n_seqs=2, H=10,
                        save_path=f"{FIG_DIR}/rollout_strips_landscape.pdf")

    # 2. Drift curves (load from metrics JSONs)
    drift_configs = {}
    for tag in ["default", "no_stage0", "lam0p0", "lam1p0", "K4"]:
        mpath = f"{RESULT_DIR}/eval_{tag}.json"
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            label_map = {
                "default":  "Full (λ=0.1, K=8)",
                "no_stage0":"No Stage 0",
                "lam0p0":   "λ=0.0 (pixel only)",
                "lam1p0":   "λ=1.0",
                "K4":       "K=4",
            }
            drift_configs[label_map.get(tag, tag)] = m["rollout"]["psnr_per_step"]
    if drift_configs:
        plot_drift_curves(drift_configs)

    # 3. Training curves
    plot_training_curves()

    # 4. Latent PCA
    plot_latent_pca(model)

    # 5. Sample predictions
    plot_sample_predictions(model)

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
