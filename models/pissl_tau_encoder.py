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
    """
    def __init__(self, num_tau_channels=8):
        super(PISSLTauEncoder, self).__init__()
        
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
        # Maps the high-dimensional features (64 channels) to exactly 2 parameter maps:
        # Channel 0: Gamma (Diffusion coefficient)
        # Channel 1: Alpha (Anomalous exponent)
        self.physics_projection = nn.Conv2d(64, 2, kernel_size=1)
        
        # Gamma ∈ (0, 1.0)  — GT max is 0.5, Sigmoid keeps prediction in range
        # Softplus was unbounded and caused gamma_pred to drift to ~7.7 (no convergence)
        self.gamma_activation = nn.Sigmoid()
        # Alpha ∈ (0, 2.0)
        self.alpha_activation = nn.Sigmoid()

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
        
        # Physical Projection
        # physics_maps shape: (B, 2, H, W)
        physics_maps = self.physics_projection(u3)
        
        # Enforce physical constraints on the two channels
        gamma_map = self.gamma_activation(physics_maps[:, 0:1, :, :])          # (0, 1)
        alpha_map = self.alpha_activation(physics_maps[:, 1:2, :, :]) * 2.0   # (0, 2)
        
        # Return as (B, 2, H, W)
        return torch.cat([gamma_map, alpha_map], dim=1)

if __name__ == "__main__":
    # Test the model structure
    # Batch size 4, 8 Tau channels, 64x64 patch
    dummy_input = torch.randn(4, 8, 64, 64)
    model = PISSLTauEncoder(num_tau_channels=8)
    output = model(dummy_input)
    
    print(f"Input shape (B, Tau, H, W): {dummy_input.shape}")
    print(f"Output shape (B, Gamma/Alpha, H, W): {output.shape}")
    print(f"Gamma Map range: [{output[:,0].min().item():.3f}, {output[:,0].max().item():.3f}]")
    print(f"Alpha Map range: [{output[:,1].min().item():.3f}, {output[:,1].max().item():.3f}]")
