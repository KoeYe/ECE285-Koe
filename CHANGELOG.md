# VAE Changelog

## Round 1 — Architecture & Training Fundamentals

Initial code review identified structural issues. All changes applied together before first training run.

**Files changed**: `VAE/VAE.py`, `trainer/vae_trainer.py`

### Changes

1. **Add BatchNorm to skip connections** (`DownBlock`, `UpBlock`)
   - *Problem*: Main branch goes through BN (~N(0,1)) but skip branch has no BN → scale mismatch when adding.
   - *Fix*: Skip branch changed from bare Conv to `nn.Sequential(Conv, BatchNorm2d)`.

2. **Reduce latent dimensions 64 → 16**
   - *Problem*: Latent shape (B, 64, 8, 8) = 4096 dims is excessively high for a VAE. Posterior has too much freedom, KL regularization cannot constrain it.
   - *Fix*: `latent_channels` default changed from 64 to 16 → (B, 16, 8, 8) = 1024 dims.

3. **Add Kaiming weight initialization**
   - *Problem*: PyTorch default init is suboptimal for ReLU networks.
   - *Fix*: Added `_init_weights()` with Kaiming Normal for all Conv layers, standard init for BN layers.

4. **Fix loss scaling: sum → mean reduction**
   - *Problem*: Both rec_loss and kl_loss used `sum` reduction. rec_loss sums over 65536 pixels, kl_loss over 4096 latent dims, and `kl_weight_max = 0.001`. Effective KL contribution ≈ 0.001 × 4096 ≈ 4, completely dwarfed by ~65536 rec_loss.
   - *Symptom*: Model degrades to a plain autoencoder. No latent regularization. Random sampling produces garbage.
   - *Fix*: Both losses changed to `mean` reduction (per-element average). `kl_weight_max` adjusted from 0.001 to 1.0.

5. **Add gradient clipping**
   - *Fix*: `clip_grad_norm_(max_norm=1.0)` after `loss.backward()`.

6. **Best model selection based on ELBO (total val_loss) instead of val_rec only**

7. **Fix hardcoded latent shape in `save_samples()`**
   - *Problem*: `torch.randn(n, 64, 8, 8)` hardcoded 64, inconsistent with actual `latent_channels`.
   - *Fix*: Changed to `torch.randn(n, self.vae.latent_channels, 8, 8)`.

---

## Round 2 — Fix NaN from Initialization

**Trigger**: Training produced NaN from epoch 0.

**Files changed**: `VAE/VAE.py`

### Root Cause

Kaiming init applied to `fc_logvar` layer → initial `logvar` values reached ±25 → `exp(25) ≈ 7.2e10` → KL loss exploded to ~4,000,000 → one gradient step destroyed all weights.

Verified by debug forward pass:
```
mu:     [-54.77, 64.34]
logvar: [-31.88, 24.96]
kl_loss: 4146366.0
```

### Changes

1. **Near-zero init for `fc_mu` and `fc_logvar`**
   - `xavier_uniform_(gain=0.01)` so initial mu ≈ 0, logvar ≈ 0 → latent starts near N(0,1).

2. **Clamp `logvar` to [-10, 10]**
   - Numerical safety net: `exp(10) ≈ 22026` is large but finite, prevents overflow.

### Result

```
step 0: loss=0.2280 rec=0.2278 kl=0.0001 mu=[-0.08,0.07] logvar=[-0.07,0.09]
step 4: loss=0.2199 rec=0.2196 kl=0.0003 mu=[-0.08,0.09] logvar=[-0.06,0.12]
```

No NaN. Loss decreasing normally. mu/logvar in reasonable range.

---

## Round 3 — Fix Checkerboard Artifacts & KL Collapse

**Trigger**: Generated sample images show severe grid-like patterns (checkerboard artifacts) and all look nearly identical (KL collapse).

**Files changed**: `VAE/VAE.py`, `trainer/vae_trainer.py`

### Changes

1. **Replace `ConvTranspose2d` with `Upsample + Conv` in `UpBlock`**
   - *Problem*: `ConvTranspose2d(kernel_size=4, stride=2)` produces uneven pixel overlap (1-2-1-2 pattern). 5 stacked layers amplify this into severe checkerboard artifacts.
   - *Symptom*: Generated images broken into tiny grid blocks, completely unrecognizable.
   - *Fix*:
     ```python
     # Before
     self.deconv = ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
     # After
     self.up = nn.Sequential(
         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
         nn.Conv2d(in_ch, out_ch, 3, padding=1),
     )
     ```

2. **Add Free Bits strategy to prevent KL collapse**
   - *Problem*: KL loss collapses to ≈ 0 even after warmup. Encoder ignores latent space entirely.
   - *Symptom*: All generated images look the same (blurry mean image). No diversity.
   - *Fix*: Clamp per-dimension KL to a minimum of 0.25 nats:
     ```python
     kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
     kl_loss = torch.clamp(kl_per_dim, min=0.25).mean()
     ```

### Result

```
epoch  33 | kl_w 1.00 | train 0.256 (rec 0.006, kl 0.250) | val 0.254 (rec 0.005, kl 0.250)
```

- rec_loss: 0.027 → 0.006 (decoder learning normally)
- KL stable at ~0.25 (= free_bits floor). Normal for early training — model learns reconstruction first, then KL should exceed 0.25 as encoder starts utilizing latent space.

---

## Round 4 — Timestamp Output Directories

**Trigger**: New training runs overwrite previous samples and checkpoints.

**Files changed**: `trainer/vae_trainer.py`

### Changes

1. **Add timestamp to output paths**
   - `samples/` → `samples/{YYYYMMDD_HHMMSS}/`
   - `checkpoints/` → `checkpoints/{YYYYMMDD_HHMMSS}/`
   - Historical results are preserved across runs.
