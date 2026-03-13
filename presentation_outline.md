# ECE 285 Final Presentation Outline
## "Shaping Latent World Models via Latent Reasoning for Embodied Planning"
### 3 minutes | 9 slides

---

## Slide 1 — Title (20s)
**Title:** Shaping Latent World Models via Latent Reasoning for Embodied Planning and Generalization

**Subtitle:** ECE 285 Final Project

**Visual:** FrozenLake grid → latent block → predicted frame (pipeline schematic)

**Say:**
> "We built a world model that predicts future observations from actions. The key idea is to route all dynamics through a structured latent token block — bridging semantic reasoning and pixel generation."

---

## Slide 2 — Why Latent State Carriers? (20s)
**Visual:** Two paths: (A) pixel-to-pixel prediction (opaque, hard to iterate) vs (B) pixel → latent → pixel (interpretable, iterable)

**Key points:**
- Pure pixel predictors lack interpretable intermediate state
- Pure latent models can't produce visual outputs
- Our latent block z̃ ∈ R^{K×d} gives both: structure + decodability

**Say:**
> "Prior work either predicts directly in pixels — losing semantic structure — or purely in latent space — losing visual interpretability. Our latent block gives us both."

---

## Slide 3 — Architecture (30s)
**Visual:** Block diagram with 4 components:
```
o_t ──► E_v (CNN) ──────────────────────┐
                                          ├──► f_θ (Transformer) ──► ẑ_{t+1} ──► g_φ (U-Net) ──► ô_{t+1}
a_t ──► ActionEmb ──────────────────────┘          cross-attn bottleneck
```

**Key specs table:**
| Component | Type | Params |
|---|---|---|
| E_v | 4-layer CNN | ~120K |
| ActionEnc | Embedding(4,64) | tiny |
| f_θ | 4-layer Transformer | ~1.5M |
| g_φ | U-Net + cross-attention | ~4M |

**Say:**
> "The vision encoder produces 64 patch tokens. The Transformer latent predictor outputs K=8 latent tokens conditioned on visual features and the action. The U-Net decoder conditions on these tokens via cross-attention to produce the predicted frame."

---

## Slide 4 — Three-Stage Training Pipeline (30s)
**Visual:** Three boxes in sequence with lock/unlock icons:

**Stage 0 (Decoder Adaptation)**
- Train E_v + g_φ together as autoencoder
- L_diff = MSE(g_φ(pool(E_v(o_t))), o_{t+1})
- Result: g_φ learns FrozenLake visual domain

**Stage 1 (Latent Format SFT)**
- Freeze g_φ, E_v; train f_θ
- L_format = 0.9·MSE(ẑ, z*) + 0.1·(1-cos(ẑ, z*))
- Result: f_θ emits encoder-compatible tokens

**Stage 2 (Transition Grounding)**
- Freeze E_v; train f_θ end-to-end
- L_ground = L_pixel + λ·L_embed
- Result: full pipeline predicts next observations

**Say:**
> "We train in three stages. First we adapt the decoder to our domain. Then we stabilize the latent channel. Finally we ground everything with pixel plus embedding losses. Each stage has a clear, isolated objective."

---

## Slide 5 — FrozenLake Setup (15s)
**Visual:** Grid of 4 sample frames showing different map seeds + agent positions

**Key facts:**
- 8×8 FrozenLake-v1, is_slippery=False (deterministic)
- Custom 64×64 RGB renderer (ice=blue, hole=black, goal=green, agent=orange)
- 200 random map seeds → ~148K transitions (o_t, a_t, o_{t+1})
- 70/15/15 train/val/test split by map seed

**Say:**
> "We use FrozenLake as a controlled testbed. Deterministic transitions mean prediction errors reflect model capability, not environment stochasticity. We collected 148K transitions from 200 unique maps."

---

## Slide 6 — Main Results Table (30s)
**Visual:** Results table (large, center-aligned)

| Model | PSNR↑ | SSIM↑ | Drift₅↓ | Drift₁₀↓ |
|---|---|---|---|---|
| Copy (o_t) | 28.14 | 0.890 | 0.00821 | 0.01243 |
| Stage 0 proxy | 31.30 | 0.932 | 0.00612 | 0.01041 |
| **Full model** | **35.84** | **0.961** | **0.00312** | **0.00581** |

**Sub-bullet:** +4.54 dB over Stage 0 proxy, +7.7 dB over copy baseline

**Say:**
> "Our full three-stage model achieves 35.84 dB PSNR — over 4.5 dB better than using just the Stage 0 proxy decoder and 7.7 dB better than simply copying the previous frame. Rollout drift is 0.006 at 10 steps."

---

## Slide 7 — Rollout Visualization (25s)
**Visual:** Figure from results/figures/rollout_strips.pdf (4 sequences × 10 steps)
- Row 1: Ground Truth
- Row 2: Predicted
- Row 3: |Diff| × 10

**Key observation:** Model tracks agent position correctly for 5-8 steps; errors concentrate at holes and boundaries

**Say:**
> "Here are 4 test rollouts over 10 steps. The model accurately predicts agent movement and surrounding cell appearance for the first several steps. Errors appear mainly near holes, where a small positional error produces large pixel differences."

---

## Slide 8 — Drift Curves + Ablations (20s)
**Visual:** Two panels side by side
- LEFT: Drift curve figure (PSNR vs. step, multi-line)
- RIGHT: Mini ablation table highlighting the two most important factors

**Key ablation findings:**
1. **No Stage 0** → -2.2 dB PSNR (decoder adaptation is critical)
2. **λ=0** (pixel-only) → higher Drift₁₀ (embedding loss reduces long-horizon drift)
3. **K=2** → -1.3 dB (insufficient capacity)
4. **K=8 optimal** (K=16 marginal improvement)

**Say:**
> "The drift curves confirm that embedding loss significantly slows quality degradation. Ablations show Stage 0 and the embedding loss are the two most important components — removing either causes measurable regression."

---

## Slide 9 — Conclusion (10s)
**Visual:** One clean summary box + future work bullet

**Summary:**
- Three-stage pipeline learns reliable latent world models from scratch
- 5.8M params, single GPU, ~2.5h total training
- 35.84 dB PSNR, Drift₁₀ = 0.006

**Future work:**
- Scheduled sampling for multi-step training
- Stochastic variants (is_slippery=True)
- Continuous action spaces and complex visual domains

**Say:**
> "In summary, our three-stage approach cleanly separates decoder adaptation, latent stabilization, and transition grounding. The resulting model achieves strong one-step and rollout performance from scratch. Thank you."

---

## Timing Guide
| Slide | Time |
|---|---|
| 1. Title | 20s |
| 2. Why latent? | 20s |
| 3. Architecture | 30s |
| 4. Training pipeline | 30s |
| 5. Setup | 15s |
| 6. Main results | 30s |
| 7. Rollout viz | 25s |
| 8. Drift + ablations | 20s |
| 9. Conclusion | 10s |
| **Total** | **3:00** |
