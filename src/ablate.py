"""
Ablation runner: sweeps over 4 ablation configurations.

Ablation 1: No Stage 0 (skip decoder pretraining)
Ablation 2: Latent block length K ∈ {2, 4, 8, 16}
Ablation 3: Lambda (embedding loss weight) ∈ {0.0, 0.01, 0.1, 1.0}
Ablation 4: Decoder freeze strategy (frozen / partial / full)

Each config runs Stage 2 training and evaluation.
Results saved to results/ablation_results.json.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch

from train_stage2 import train as train_stage2
from eval import main as eval_main
from model import K as DEFAULT_K, D_LAT

RESULT_DIR = "/data/koe/ECE285-Final/results"
CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"

# Reduce epochs for ablation runs (faster)
import train_stage2 as ts2
ABLATION_EPOCHS = 5    # vs 30 for main run (quick ablations)


def run_ablation(tag, lambda_embed=0.1, K_lat=DEFAULT_K, no_stage0=False,
                 freeze_decoder=True):
    """Run one ablation config."""
    # Temporarily reduce epochs
    orig_epochs = ts2.EPOCHS
    ts2.EPOCHS  = ABLATION_EPOCHS

    try:
        best_psnr = train_stage2(
            lambda_embed   = lambda_embed,
            K_lat          = K_lat,
            d_lat          = D_LAT,
            no_stage0      = no_stage0,
            freeze_decoder = freeze_decoder,
            tag            = tag,
        )
        ckpt_path = f"{CKPT_DIR}/stage2_{tag}.pth"
        results = eval_main(ckpt_path, tag)
        return {"tag": tag, "best_psnr": best_psnr, "metrics": results}
    finally:
        ts2.EPOCHS = orig_epochs


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/ablation_results.json"

    # Load partial results if they exist (resume after crash)
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"Resuming: loaded {len(all_results)} existing results")
    else:
        all_results = {}

    def run_and_save(tag, **kwargs):
        if tag in all_results:
            print(f"  [SKIP] {tag} already done")
            return all_results[tag]
        r = run_ablation(tag, **kwargs)
        all_results[tag] = r
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        return r

    print("=" * 60)
    print("ABLATION 1: Effect of Stage 0 (decoder pretraining)")
    print("=" * 60)
    run_and_save("no_stage0", lambda_embed=0.1, K_lat=DEFAULT_K, no_stage0=True)

    print("\n" + "=" * 60)
    print("ABLATION 2: Latent block length K")
    print("=" * 60)
    for K_val in [2, 4, 16]:
        run_and_save(f"K{K_val}", lambda_embed=0.1, K_lat=K_val)

    print("\n" + "=" * 60)
    print("ABLATION 3: Lambda (embedding loss weight)")
    print("=" * 60)
    for lam in [0.0, 0.01, 1.0]:
        tag = f"lam{lam}".replace(".", "p")
        run_and_save(tag, lambda_embed=lam, K_lat=DEFAULT_K)

    print("\n" + "=" * 60)
    print("ABLATION 4: Decoder freeze strategy")
    print("=" * 60)
    run_and_save("dec_partial", lambda_embed=0.1, K_lat=DEFAULT_K, freeze_decoder=False)

    print(f"\nAll ablation results saved to {out_path}")

    # Print summary table
    print("\n── Ablation Summary ──────────────────────────────────────────")
    print(f"{'Config':<18} {'PSNR':>7} {'SSIM':>7} {'Drift@5':>9} {'Drift@10':>10}")
    print("-" * 55)

    # Load default result
    default_path = f"{RESULT_DIR}/eval_default.json"
    if os.path.exists(default_path):
        with open(default_path) as f:
            def_r = json.load(f)
        print(f"{'default (K=8,λ=0.1)':<22} "
              f"{def_r['one_step']['psnr']:>7.2f} "
              f"{def_r['one_step']['ssim']:>7.4f} "
              f"{def_r['rollout']['drift_5']:>9.5f} "
              f"{def_r['rollout']['drift_10']:>10.5f}")

    for tag, res in all_results.items():
        if res is None:
            continue
        m = res.get("metrics") or res   # eval_main returns results dict directly
        d10 = m["rollout"]["drift_10"] or 0.0
        print(f"{tag:<22} "
              f"{m['one_step']['psnr']:>7.2f} "
              f"{m['one_step']['ssim']:>7.4f} "
              f"{m['rollout']['drift_5']:>9.5f} "
              f"{d10:>10.5f}")


if __name__ == "__main__":
    main()
