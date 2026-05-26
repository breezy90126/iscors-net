import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BilinearUp(nn.Module):
    """
    Bilinear upsampling + 1x1 conv. No checkerboard artifacts.
    PixelShuffle without ICNR initialisation produces 2x2 grid artifacts.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(self.up(x))


class PISSLTauEncoder(nn.Module):
    """
    U-Net for Physics-Informed Spatial Sampling.

    Input channels (v3.6+): K G_norm channels + K-1 log-log slope channels = 2K-1.
    Output: (γ, α) — 2 channels (amplitude removed).

    Direction D (v3.6): α uses ELU+1 activation instead of Sigmoid×2.
      ELU(x) + 1:  x=0 → α=1.0 (healthy default)
                   x>0 → α = x+1 (linear, no saturation — good for super-diffusion)
                   x<0 → α = e^x (approaches 0, non-zero gradient everywhere)
      Compare Sigmoid×2: saturates at both ends; gradient → 0 near α≈0 or α≈2.
    """
    def __init__(self, num_tau_channels=8, predict_amplitude=True):
        super(PISSLTauEncoder, self).__init__()
        self.predict_amplitude = predict_amplitude
        out_channels = 3 if predict_amplitude else 2

        # Encoder
        self.inc   = DoubleConv(num_tau_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder
        self.up1      = BilinearUp(512, 256)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2      = BilinearUp(256, 128)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3      = BilinearUp(128, 64)
        self.conv_up3 = DoubleConv(128, 64)

        # Physics projection: ch0=γ, ch1=α, ch2=A (optional)
        self.physics_projection = nn.Conv2d(64, out_channels, kernel_size=1)

        if predict_amplitude:
            with torch.no_grad():
                self.physics_projection.bias[2].fill_(-6.9)  # Softplus(-6.9) ≈ 1e-3

        self.gamma_activation = nn.Sigmoid()   # γ ∈ (0, 1)
        self.amp_activation   = nn.Softplus()  # A > 0

    def forward(self, x):
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decode with skip connections
        u1 = self.up1(x4)
        u1 = self.conv_up1(torch.cat([x3, u1], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([x2, u2], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([x1, u3], dim=1))

        p = self.physics_projection(u3)

        gamma_map = self.gamma_activation(p[:, 0:1])           # (0, 1)

        # Direction D: ELU+1 for α — linear gradient for α>1, no saturation
        # x=0 → 1.0; x<0 → e^x (sub-diffusion); x>0 → x+1 (super-diffusion)
        alpha_map = (F.elu(p[:, 1:2]) + 1.001).clamp(max=2.0) # (0.001, 2]

        if self.predict_amplitude:
            amp_map = self.amp_activation(p[:, 2:3])
            return torch.cat([gamma_map, alpha_map, amp_map], dim=1)  # (B, 3, H, W)
        return torch.cat([gamma_map, alpha_map], dim=1)               # (B, 2, H, W)


if __name__ == "__main__":
    # v3.6: 2K-1 = 19 input channels (K=10)
    dummy_input = torch.randn(4, 19, 64, 64)
    model = PISSLTauEncoder(num_tau_channels=19, predict_amplitude=False)
    output = model(dummy_input)

    print(f"Input shape  (B, 2K-1, H, W) : {dummy_input.shape}")
    print(f"Output shape (B, [γ,α], H, W): {output.shape}")
    print(f"Gamma range: [{output[:,0].min().item():.4f}, {output[:,0].max().item():.4f}]")
    print(f"Alpha range: [{output[:,1].min().item():.4f}, {output[:,1].max().item():.4f}]")
