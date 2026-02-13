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

---

## Round 5 — Fix Generation Ghosting (Reconstruction-Generation Gap)

**Trigger**: Reconstruction quality is good, but random generation shows ghosting / double-image artifacts.

**Root Cause**: Latent space is not well-regularized to N(0,1). Sampling from N(0,1) lands in regions the decoder never saw during training, producing blended/ghosted outputs.

**Contributing factors identified**:

1. **`kl_weight_max = 0.1` too low** — KL barely constrains the posterior. Encoder maps different images to distant clusters with large gaps. Random z falls between clusters → decoder outputs superposition of multiple modes → ghosting.

2. **Latent dimensionality too high** (16 × 8 × 8 = 1024 dims) — High-dimensional N(0,1) concentrates on a thin shell; encoder posteriors form irregular clusters that don't cover this shell. More empty space = worse generation.

3. **Free bits (`min=0.25`) counterproductive at low KL weight** — Forces every dimension to carry ≥0.25 nats of information, preventing the posterior from collapsing _toward_ the prior. Combined with low `kl_weight`, the posterior stays far from N(0,1) but the gradient signal to fix it is weak.

4. **`reparameterize()` uses mu (no noise) at eval** — Makes reconstruction deterministic (looks great) but means validation loss doesn't reflect true generative quality, masking the gap.

### Recommended Changes (not yet applied)

| Priority | Change | Rationale |
|----------|--------|-----------|
| **P0** | `kl_weight_max`: 0.1 → **0.5** | Main fix. Stronger KL pressure forces posterior toward N(0,1) |
| **P1** | Remove free bits (or reduce to 0.05) | Conflicting with low KL weight; let posterior collapse toward prior |
| **P1** | `kl_warmup_epochs`: 20 → **80** | Prevent KL shock with higher weight; give encoder time to adapt |
| **P2** | `latent_channels`: 16 → **8** | Reduce latent dims (512 vs 1024), fewer gaps in latent space |
| **P3** | Always sample in `reparameterize()` | More honest val loss, better tracks generation quality |

### Expected Tradeoff

Reconstruction quality may slightly decrease (normal VAE rec-gen tradeoff), but generation should improve significantly with fewer ghosting artifacts.

---

## Round 6 — Combat Generation Blurriness

**Trigger**: After Round 5 hyperparameter tuning (`kl_weight=0.5`, `latent_channels=8`, `free_bits=0.05`, `warmup=40`), ghosting is resolved but reconstruction degraded and generation remains blurry.

**Root Cause**: Pixel-level losses (L1/SSIM) reward the decoder for producing the *average* of plausible outputs → inherently blurry. Single-conv UpBlocks lack capacity to synthesize sharp detail. Pure convolution has no global receptive field.

**Files changed**: `VAE/VAE.py`, `trainer/vae_trainer.py`

### Architecture Changes (`VAE.py`)

1. **Deeper `UpBlock`: 1 conv → 2 convs per block**
   - *Problem*: Each UpBlock had only one 3×3 conv after upsampling — limited capacity to synthesize detail.
   - *Fix*: Added second `Conv2d(out_ch, out_ch, 3, padding=1)` + `BatchNorm2d` in residual path. Doubles decoder depth without changing tensor shapes.

2. **`LeakyReLU(0.2)` in `UpBlock`**
   - *Problem*: ReLU kills negative gradients — hurts generative path where all activations are "invented" (not reflected from data).
   - *Fix*: Replaced `ReLU` with `LeakyReLU(0.2)` in UpBlock.

3. **Self-Attention at 16×16 resolution**
   - *Problem*: Convolutions only see local windows (3×3 or 4×4). Global structure (symmetry, organ placement) requires global receptive field.
   - *Fix*: Added `SelfAttention(256)` module after `down4` in Encoder and after `up1` in Decoder. Uses scaled dot-product attention with GroupNorm. Output projection initialized to zero (starts as identity, no disruption to training).
   - 16×16 chosen because attention is O(n²) — affordable at 256 tokens but prohibitive at 256×256 (65536 tokens).

### Loss Changes (`vae_trainer.py`)

4. **Multi-scale reconstruction loss (3 scales)**
   - *Problem*: Loss at native resolution only — if coarse structure is wrong, gradient signal is diluted across 65536 pixels.
   - *Fix*: Compute L1+SSIM at scales 1× (256), 2× (128), 4× (64) via `avg_pool2d`, average equally. Ensures both coarse structure and fine detail are supervised.

5. **Spectral (FFT) loss**
   - *Problem*: L1/SSIM don't directly penalize loss of high-frequency content (edges, textures). Model can minimize loss by blurring.
   - *Fix*: Added `_spectral_loss()` — computes L1 of complex FFT difference (`rfft2` with ortho normalization). Weight = 0.1. Directly penalizes missing high-frequency detail.

### Weight Initialization

6. **GroupNorm init + SelfAttention proj zero init**
   - Added `nn.GroupNorm` to the initialization loop (weight=1, bias=0).
   - SelfAttention `proj` layer explicitly zeroed after Kaiming init loop, so attention starts as identity mapping.
