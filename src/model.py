"""
World Model for FrozenLake: three components matching the paper's notation.

  E_v     : VisionEncoder   – CNN, maps (B,3,64,64) → (B, M, d_v)
  Enc(a)  : ActionEncoder   – Embedding, maps (B,) → (B, 1, d)
  f_theta : LatentPredictor – Transformer, maps (obs_tokens, action) → (B, K, d)
  g_phi   : PixelDecoder    – U-Net w/ cross-attention, maps (B, K, d) → (B, 3, 64, 64)

Paper notation preserved throughout: z_hat, K, d, M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Global hyperparameters ─────────────────────────────────────────────────
D_VIS   = 64    # vision token dim (d_v in paper)
D_ACT   = 64    # action embedding dim
D_LAT   = 64    # latent token dim (d in paper)
M_PATCH = 64    # number of visual patch tokens (M = 8×8 spatial grid)
K       = 8     # latent block length
D_MODEL = 128   # Transformer hidden dim
N_HEAD  = 4
N_LAYER = 4
IMG_C   = 3
IMG_H   = 64


# ── Vision Encoder E_v ─────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    CNN that maps a 64x64 RGB image to M=64 patch tokens of dimension d_v=64.
    Mirrors a ViT's patch embedding but implemented as a strided CNN.

    Input:  (B, 3, 64, 64)
    Output: (B, M, d_v)  where M=64, d_v=64
    """

    def __init__(self, d_out: int = D_VIS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),  # 32×32
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),  # 16×16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),  # 8×8
        )
        self.proj = nn.Linear(64, d_out)  # 64 channels → d_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 64, 64)
        returns: (B, M=64, d_v=64)  – M spatial tokens
        """
        feat = self.backbone(x)          # (B, 64, 8, 8)
        B, C, H, W = feat.shape
        tokens = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, 64, 64)
        return self.proj(tokens)                            # (B, 64, 64)

    def pool_to_k(self, tokens: torch.Tensor, K: int = K) -> torch.Tensor:
        """
        Average-pool M tokens into K groups → (B, K, d_v).
        Used to create z_target in Stage 1.
        """
        B, M, d = tokens.shape
        assert M % K == 0, f"M={M} must be divisible by K={K}"
        group_size = M // K
        return tokens.view(B, K, group_size, d).mean(dim=2)  # (B, K, d_v)


# ── Action Encoder ─────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """
    Embedding lookup for discrete actions.
    4 actions (0=L, 1=D, 2=R, 3=U) → d_act-dimensional token.

    Input:  (B,) long
    Output: (B, 1, d_act)
    """

    def __init__(self, num_actions: int = 4, d_act: int = D_ACT):
        super().__init__()
        self.emb = nn.Embedding(num_actions, d_act)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.emb(a).unsqueeze(1)   # (B, 1, d_act)


# ── Latent Predictor f_θ ───────────────────────────────────────────────────

class LatentPredictor(nn.Module):
    """
    Transformer f_θ: given (visual tokens h_t, action token u_t),
    autoregressively emits K latent tokens z_hat_{t+1}.

    Sequence fed to Transformer:
        [CLS | h_t (M tokens) | u_t (1 token)] → predict K output tokens

    Input:
        vis_tokens: (B, M, d_v)
        action_tok: (B, 1, d_act)
    Output:
        z_hat: (B, K, d)  – the latent state carrier
    """

    def __init__(
        self,
        d_vis: int = D_VIS,
        d_act: int = D_ACT,
        d_model: int = D_MODEL,
        d_lat: int = D_LAT,
        K: int = K,
        n_head: int = N_HEAD,
        n_layer: int = N_LAYER,
        M: int = M_PATCH,
    ):
        super().__init__()
        self.K = K
        self.M = M

        # Input projections
        self.vis_proj = nn.Linear(d_vis, d_model)
        self.act_proj = nn.Linear(d_act, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Latent start/end delimiter tokens (learned embeddings)
        self.latent_start = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.latent_end   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding for context (1 CLS + M vis + 1 act = M+2)
        self.pos_emb = nn.Parameter(torch.randn(1, M + 2, d_model) * 0.02)

        # Transformer encoder (not causal; we predict all K tokens at once
        # by appending K query slots after the context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 2, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # K learnable query tokens (one per latent position)
        self.latent_queries = nn.Parameter(torch.randn(1, K, d_model) * 0.02)

        # Output projection to latent dim d
        self.out_proj = nn.Linear(d_model, d_lat)

    def forward(self, vis_tokens: torch.Tensor, action_tok: torch.Tensor) -> torch.Tensor:
        """
        vis_tokens: (B, M, d_v)
        action_tok: (B, 1, d_act)
        returns:    (B, K, d)
        """
        B = vis_tokens.shape[0]

        # Project to d_model
        v = self.vis_proj(vis_tokens)                          # (B, M, d_model)
        a = self.act_proj(action_tok)                          # (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, d_model)

        # Context sequence: [CLS | vis_tokens | action_token]
        context = torch.cat([cls, v, a], dim=1)                # (B, M+2, d_model)
        context = context + self.pos_emb                       # add positional encoding

        # Latent query tokens
        queries = self.latent_queries.expand(B, -1, -1)        # (B, K, d_model)

        # Full sequence: context + queries
        seq = torch.cat([context, queries], dim=1)             # (B, M+2+K, d_model)
        out = self.transformer(seq)                            # (B, M+2+K, d_model)

        # Extract the K output positions (last K tokens)
        z_latent = out[:, -self.K:, :]                         # (B, K, d_model)
        z_hat = self.out_proj(z_latent)                        # (B, K, d)
        return z_hat


# ── Pixel Decoder g_φ ──────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Simple residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CrossAttention(nn.Module):
    """
    Cross-attention: queries from spatial features, keys/values from z_hat.
    Implements the conditioning mechanism: 'U-Net blocks cross-attend to z_hat.'
    """

    def __init__(self, d_query: int, d_kv: int, n_head: int = 4):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_query // n_head
        self.scale  = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_query, d_query)
        self.k_proj = nn.Linear(d_kv,    d_query)
        self.v_proj = nn.Linear(d_kv,    d_query)
        self.out    = nn.Linear(d_query, d_query)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  (B, HW, d_query)  – spatial features
        kv: (B, K,  d_kv)     – latent tokens z_hat
        """
        B, N, D = q.shape
        H = self.n_head
        dh = self.d_head

        Q = self.q_proj(q).view(B, N, H, dh).transpose(1, 2)   # (B, H, N, dh)
        K = self.k_proj(kv).view(B, -1, H, dh).transpose(1, 2) # (B, H, K, dh)
        V = self.v_proj(kv).view(B, -1, H, dh).transpose(1, 2) # (B, H, K, dh)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class PixelDecoder(nn.Module):
    """
    U-Net decoder g_φ conditioned on the latent block z_hat via cross-attention.

    Architecture:
        Encoder: 64×64 → 32×32 → 16×16 → 8×8 (bottleneck)
        Cross-attention at bottleneck: spatial tokens attend to z_hat tokens
        Decoder: 8×8 → 16×16 → 32×32 → 64×64

    Input:  x (B, 3, 64, 64), z_hat (B, K, d_lat)
    Output: (B, 3, 64, 64) in [-1, 1]
    """

    def __init__(self, d_lat: int = D_LAT, K: int = K):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,  64, 3, stride=2, padding=1),  # 64 → 32
            ResBlock(64),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32 → 16
            ResBlock(128),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 16 → 8
            ResBlock(256),
        )

        # Bottleneck cross-attention: spatial tokens (256-d) attend to z_hat
        self.z_proj      = nn.Linear(d_lat, 256)
        self.cross_attn  = CrossAttention(d_query=256, d_kv=256, n_head=4)
        self.attn_norm   = nn.LayerNorm(256)
        self.bot_res     = ResBlock(256)

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),  # 8 → 16
            ResBlock(128),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),   # 16 → 32
            ResBlock(64),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),     # 32 → 64
            ResBlock(32),
        )
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, 3, 64, 64) – conditioning image (current observation o_t)
        z_hat: (B, K, d_lat)  – predicted latent block
        returns: (B, 3, 64, 64) – predicted next observation
        """
        # Encode
        e1 = self.enc1(x)    # (B, 64,  32, 32)
        e2 = self.enc2(e1)   # (B, 128, 16, 16)
        e3 = self.enc3(e2)   # (B, 256,  8,  8)

        # Bottleneck cross-attention
        B, C, H, W = e3.shape
        spatial = e3.view(B, C, H * W).permute(0, 2, 1)   # (B, HW=64, 256)
        z_kv    = self.z_proj(z_hat)                       # (B, K, 256)
        spatial = spatial + self.cross_attn(self.attn_norm(spatial), z_kv)
        bot     = self.bot_res(spatial.permute(0, 2, 1).view(B, C, H, W))

        # Decode with skips
        d3 = self.dec3(torch.cat([bot, e3], dim=1))  # (B, 128, 16, 16)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B,  64, 32, 32)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B,  32, 64, 64)
        return self.out_conv(d1)                     # (B,   3, 64, 64)


# ── Full World Model ───────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Full world model: (o_t, a_t) → z_hat_{t+1} → o_hat_{t+1}

    This wraps E_v, Enc(a), f_theta, g_phi into a single forward pass.
    The three-stage training scripts selectively freeze/unfreeze components.
    """

    def __init__(self, K: int = K, d_lat: int = D_LAT):
        super().__init__()
        self.encoder         = VisionEncoder(d_out=D_VIS)
        self.action_encoder  = ActionEncoder(num_actions=4, d_act=D_ACT)
        self.latent_predictor = LatentPredictor(
            d_vis=D_VIS, d_act=D_ACT, d_model=D_MODEL, d_lat=d_lat, K=K, M=M_PATCH
        )
        self.decoder = PixelDecoder(d_lat=d_lat, K=K)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → visual tokens (B, M, d_v)"""
        return self.encoder(obs)

    def predict_latent(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(obs, action) → z_hat (B, K, d)"""
        vis_tokens  = self.encoder(obs)
        action_tok  = self.action_encoder(action)
        return self.latent_predictor(vis_tokens, action_tok)

    def decode(self, obs: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """(obs, z_hat) → o_hat (B, 3, 64, 64)"""
        return self.decoder(obs, z_hat)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """
        Full forward pass.
        Returns dict with keys: z_hat, obs_hat, vis_tokens
        """
        vis_tokens  = self.encoder(obs)
        action_tok  = self.action_encoder(action)
        z_hat       = self.latent_predictor(vis_tokens, action_tok)
        obs_hat     = self.decoder(obs, z_hat)
        return {"z_hat": z_hat, "obs_hat": obs_hat, "vis_tokens": vis_tokens}


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  Trainable: {train:,}"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WorldModel(K=K, d_lat=D_LAT).to(device)
    print("WorldModel params:", count_params(model))

    enc = model.encoder
    lp  = model.latent_predictor
    dec = model.decoder
    print(f"  VisionEncoder:    {count_params(enc)}")
    print(f"  LatentPredictor:  {count_params(lp)}")
    print(f"  PixelDecoder:     {count_params(dec)}")

    B = 4
    obs    = torch.randn(B, 3, 64, 64).to(device)
    action = torch.randint(0, 4, (B,)).to(device)

    out = model(obs, action)
    print(f"\nForward pass (B={B}):")
    print(f"  z_hat shape:   {out['z_hat'].shape}")    # (4, 8, 64)
    print(f"  obs_hat shape: {out['obs_hat'].shape}")  # (4, 3, 64, 64)
    print(f"  z_hat range:   [{out['z_hat'].min():.3f}, {out['z_hat'].max():.3f}]")
    print(f"  obs_hat range: [{out['obs_hat'].min():.3f}, {out['obs_hat'].max():.3f}]")
