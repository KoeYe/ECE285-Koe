"""
Evaluation script: compute one-step and multi-step rollout metrics on test set.

Metrics:
  One-step:  MSE, PSNR, SSIM, Embed-MSE
  Multi-step: Drift_H for H in {5, 10, 20}, PSNR-per-step

Usage:
  python eval.py                              # evaluate default stage2 checkpoint
  python eval.py --ckpt checkpoints/stage2_lambda0.0.pth --tag lambda0.0
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import WorldModel, K, D_LAT
from losses import (
    compute_psnr, compute_ssim, compute_mse,
    compute_embed_mse, rollout_drift,
)
from dataset import TransitionDataset, DATA_PATH

CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR = "/data/koe/ECE285-Final/results"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
ROLLOUT_H  = 20


def build_rollout_sequences(data_path=DATA_PATH, max_seqs=200, H=ROLLOUT_H):
    """
    Build rollout sequences from the test split.
    Each sequence: (obs0, actions[H], gt_obs[H])
    """
    data = np.load(data_path)
    mask = data["test_mask"]

    obs_all     = data["obs"][mask]
    next_all    = data["next_obs"][mask]
    acts_all    = data["actions"][mask].astype(np.int64)
    states_all  = data["states"][mask]
    next_st_all = data["next_states"][mask]
    map_ids     = data["map_ids"][mask]

    def to_tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

    sequences = []
    unique_maps = np.unique(map_ids)

    for mid in unique_maps[:max_seqs]:
        idxs = np.where(map_ids == mid)[0]
        if len(idxs) < H:
            continue
        chain = [idxs[0]]
        state_to_idx = {}
        for i in idxs:
            s = int(states_all[i])
            if s not in state_to_idx:
                state_to_idx[s] = i
        for _ in range(H - 1):
            cur_next = int(next_st_all[chain[-1]])
            if cur_next in state_to_idx:
                chain.append(state_to_idx[cur_next])
            else:
                break
        if len(chain) < H:
            continue

        obs0    = to_tensor(obs_all[chain[0]])
        actions = torch.tensor([acts_all[i] for i in chain], dtype=torch.long)
        gt_frames = torch.stack([to_tensor(next_all[i]) for i in chain])
        sequences.append((obs0, actions, gt_frames))
        if len(sequences) >= max_seqs:
            break

    return sequences


def evaluate_one_step(model, loader, device=DEVICE):
    model.eval()
    all_psnr = []
    all_ssim = []
    all_mse  = []
    all_emse = []

    with torch.no_grad():
        for obs, action, next_obs in tqdm(loader, desc="One-step eval"):
            obs      = obs.to(device)
            action   = action.to(device)
            next_obs = next_obs.to(device)

            vis_tokens = model.encoder(obs)
            action_tok = model.action_encoder(action)
            z_hat      = model.latent_predictor(vis_tokens, action_tok)
            obs_hat    = model.decoder(obs, z_hat)
            vis_true   = model.encoder(next_obs)
            vis_hat    = model.encoder(obs_hat)

            all_psnr.append(compute_psnr(obs_hat, next_obs))
            all_ssim.append(compute_ssim(obs_hat, next_obs))
            all_mse.append(compute_mse(obs_hat, next_obs))
            all_emse.append(compute_embed_mse(vis_hat, vis_true))

    return {
        "psnr":      float(np.mean(all_psnr)),
        "ssim":      float(np.mean(all_ssim)),
        "mse":       float(np.mean(all_mse)),
        "embed_mse": float(np.mean(all_emse)),
    }


def evaluate_rollout(model, sequences, device=DEVICE):
    model.eval()
    all_drift5  = []
    all_drift10 = []
    all_drift20 = []
    psnr_steps  = [[] for _ in range(ROLLOUT_H)]

    for obs0, actions, gt_frames in tqdm(sequences, desc="Rollout eval"):
        obs0_b    = obs0.unsqueeze(0).to(device)
        actions_b = actions.unsqueeze(0).to(device)
        gt_b      = gt_frames.unsqueeze(0).to(device)

        result = rollout_drift(model, obs0_b, actions_b, gt_b, device=device)
        all_drift5.append(result["drift_5"])
        all_drift10.append(result["drift_10"])
        all_drift20.append(result["drift_20"])
        for step, psnr_val in enumerate(result["psnr_per_step"]):
            psnr_steps[step].append(psnr_val)

    psnr_per_step = [float(np.mean(s)) if s else 0.0 for s in psnr_steps]
    return {
        "drift_5":  float(np.mean(all_drift5)),
        "drift_10": float(np.mean(all_drift10)),
        "drift_20": float(np.mean(all_drift20)),
        "psnr_per_step": psnr_per_step,
    }


def main(ckpt_path, tag):
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"[Eval] Checkpoint: {ckpt_path}  Tag: {tag}  Device: {DEVICE}")

    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    cfg   = ckpt.get("config", {})
    K_use = cfg.get("K", K)          # honour K stored in checkpoint
    model = WorldModel(K=K_use, d_lat=D_LAT).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Config: {cfg}")

    test_ds     = TransitionDataset("test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    one_step = evaluate_one_step(model, test_loader)
    print(f"\nOne-step:  PSNR={one_step['psnr']:.2f}dB  SSIM={one_step['ssim']:.4f}  "
          f"MSE={one_step['mse']:.4f}  EmbedMSE={one_step['embed_mse']:.4f}")

    print(f"\nBuilding rollout sequences...")
    sequences = build_rollout_sequences(max_seqs=200, H=ROLLOUT_H)
    print(f"  Built {len(sequences)} sequences of H={ROLLOUT_H} steps")
    rollout = evaluate_rollout(model, sequences)
    print(f"Rollout:   Drift@5={rollout['drift_5']:.4f}  "
          f"Drift@10={rollout['drift_10']:.4f}  Drift@20={rollout['drift_20']:.4f}")

    results = {"tag": tag, "checkpoint": ckpt_path, "config": cfg,
               "one_step": one_step, "rollout": rollout}
    out_path = f"{RESULT_DIR}/eval_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=f"{CKPT_DIR}/stage2_default.pth")
    parser.add_argument("--tag",  type=str, default="default")
    args = parser.parse_args()
    main(args.ckpt, args.tag)
