import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn(self.conv(x)) + identity)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample + Conv avoids checkerboard artifacts from ConvTranspose2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second conv for more decoder capacity
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.up(x)))
        out = self.bn2(self.conv2(out)) + identity
        return self.act(out)


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention for spatial feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W)
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


# VAE
# Encoder(x) -> mu, logvar
# z = mu + eps * exp(0.5 * logvar),  eps ~ N(0, 1)
# Decoder(z) -> x_hat

class Encoder(nn.Module):
    """Input (B, 1, 256, 256) -> Output mu, logvar each (B, latent_ch, 8, 8)"""
    def __init__(self, latent_channels=16):
        super().__init__()
        self.down1 = DownBlock(1, 32)       # 256 -> 128
        self.down2 = DownBlock(32, 64)      # 128 -> 64
        self.down3 = DownBlock(64, 128)     # 64  -> 32
        self.down4 = DownBlock(128, 256)    # 32  -> 16
        self.attn = SelfAttention(256)      # self-attention at 16x16
        self.down5 = DownBlock(256, 512)    # 16  -> 8

        # Separate 1x1 convolutions to produce mu and logvar independently
        self.fc_mu = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(512, latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.attn(x)
        x = self.down5(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-10, 10)
        return mu, logvar


class Decoder(nn.Module):
    """Input z (B, latent_ch, 8, 8) -> Output (B, 1, 256, 256)"""
    def __init__(self, latent_channels=16):
        super().__init__()
        self.proj = nn.Conv2d(latent_channels, 512, kernel_size=1)
        self.up1 = UpBlock(512, 256)    # 8  -> 16
        self.attn = SelfAttention(256)  # self-attention at 16x16
        self.up2 = UpBlock(256, 128)    # 16 -> 32
        self.up3 = UpBlock(128, 64)     # 32 -> 64
        self.up4 = UpBlock(64, 32)      # 64 -> 128
        self.up5 = UpBlock(32, 32)      # 128 -> 256
        self.head = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, z):
        z = self.proj(z)
        z = self.up1(z)
        z = self.attn(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        z = self.up5(z)
        return torch.sigmoid(self.head(z))


class VAE(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Near-zero init for mu/logvar heads so latent starts near N(0,1)
        nn.init.xavier_uniform_(self.encoder.fc_mu.weight, gain=0.01)
        nn.init.zeros_(self.encoder.fc_mu.bias)
        nn.init.xavier_uniform_(self.encoder.fc_logvar.weight, gain=0.01)
        nn.init.zeros_(self.encoder.fc_logvar.bias)
        # Zero init for self-attention output projections (start as identity)
        for m in self.modules():
            if isinstance(m, SelfAttention):
                nn.init.zeros_(m.proj.weight)
                nn.init.zeros_(m.proj.bias)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample during training, use mu at inference"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
