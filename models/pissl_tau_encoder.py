import torch
import torch.nn as nn

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
    A 2D U-Net architecture specifically designed for Physics-Informed Temporal Sampling.
    The input channels represent fixed exponential time delays (tau).

    v3.2: optional 3rd output channel for amplitude A = Var(I)/<I>².
    Shape-only normalised loss (v3.1) discards A and leaves cell-body γ stuck
    near 0; predicting A explicitly lets G_theory = A/(1+γτ^α) be compared to
    G_empirical without normalisation.
    """
    def __init__(self, num_tau_channels=8, predict_amplitude=True):
        super(PISSLTauEncoder, self).__init__()
        self.predict_amplitude = predict_amplitude
        out_channels = 3 if predict_amplitude else 2

        # 1. Encoder (Downsampling)
        # Input channels = number of tau slices (e.g., 8)
        self.inc = DoubleConv(num_tau_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # 2. Decoder (Upsampling) with PixelShuffle
        self.up1 = BilinearUp(512, 256)
        self.conv_up1 = DoubleConv(512, 256) # 256 + skip 256

        self.up2 = BilinearUp(256, 128)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = BilinearUp(128, 64)
        self.conv_up3 = DoubleConv(128, 64)

        # 3. Physics-Guided Latent Projection Layer
        # Channel 0: γ ∈ (0, 1)   diffusion coefficient
        # Channel 1: α ∈ (0, 2)   anomalous exponent
        # Channel 2: A > 0        amplitude Var(I)/<I>² (v3.2+, optional)
        self.physics_projection = nn.Conv2d(64, out_channels, kernel_size=1)

        # G_empirical is ~1e-3 for typical iSCAT data; biasing the amplitude
        # output so Softplus starts near that value avoids the first epochs
        # being spent rescaling A from ~0.7 (Softplus(0)) down to 1e-3.
        if predict_amplitude:
            with torch.no_grad():
                self.physics_projection.bias[2].fill_(-6.9)  # Softplus(-6.9) ≈ 1e-3

        self.gamma_activation = nn.Sigmoid()           # γ ∈ (0, 1)
        self.alpha_activation = nn.Sigmoid()           # α ∈ (0, 2) after ×2
        self.amp_activation   = nn.Softplus()          # A > 0, unbounded above

    def forward(self, x):
        # x shape: (B, num_tau_channels, H, W)

        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decode with skip connections
        u1 = self.up1(x4)
        u1 = torch.cat([x3, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x2, u2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x1, u3], dim=1)
        u3 = self.conv_up3(u3)

        physics_maps = self.physics_projection(u3)

        gamma_map = self.gamma_activation(physics_maps[:, 0:1, :, :])         # (0, 1)
        alpha_map = self.alpha_activation(physics_maps[:, 1:2, :, :]) * 2.0   # (0, 2)
        if self.predict_amplitude:
            amp_map = self.amp_activation(physics_maps[:, 2:3, :, :])         # > 0
            return torch.cat([gamma_map, alpha_map, amp_map], dim=1)          # (B, 3, H, W)
        return torch.cat([gamma_map, alpha_map], dim=1)                       # (B, 2, H, W)

if __name__ == "__main__":
    dummy_input = torch.randn(4, 8, 64, 64)
    model = PISSLTauEncoder(num_tau_channels=8, predict_amplitude=True)
    output = model(dummy_input)

    print(f"Input shape (B, Tau, H, W): {dummy_input.shape}")
    print(f"Output shape (B, [γ,α,A], H, W): {output.shape}")
    print(f"Gamma range: [{output[:,0].min().item():.4f}, {output[:,0].max().item():.4f}]")
    print(f"Alpha range: [{output[:,1].min().item():.4f}, {output[:,1].max().item():.4f}]")
    print(f"Amp   range: [{output[:,2].min().item():.4e}, {output[:,2].max().item():.4e}]")
