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

    v3.9+: input is (B, 2K, P, P) = K G_norm channels || K τ-PE channels.

    τ positional encoding (v3.9):
        tau_pe[k] = log(τ_k) / log(τ_max)  ∈ [0, 1], constant spatially.
        Concatenated as K extra channels before the first conv.

    Why τ-PE matters:
        Without PE, the model sees K G_norm values with implicit ordering. It can
        identify (γ,α) from the SET of values (bag-of-values) without needing τ order.
        Consequence: shuffle test shows small diff in homogeneous regions (spatial
        propagation dominates), large diff only at boundaries.

        With PE, each G_norm channel carries an explicit τ label. Shuffling G_norm
        while PE stays fixed creates G_norm(τ_k) ↔ τ_{perm(k)} mismatch everywhere.
        The model MUST learn the τ→G_norm functional relationship, not just value patterns.
        Shuffle test diff becomes large at every pixel, not just boundaries.

    Spatial context is still used (U-Net), exploiting the microscopy redundancy prior
    (nearby pixels share similar physics). Per-pixel MLP is not used.

    Output activations:
        γ: Sigmoid → (0, 1)
        α: ELU+1 (Direction D) — (F.elu(x) + 1.001).clamp(max=2.0) → (0.001, 2]
            x=0 → α=1.0 (healthy default)
            x>0 → linear, no saturation for super-diffusion
            x<0 → exponential approach to 0, non-zero gradient everywhere
    """
    def __init__(self, recon_taus, predict_amplitude=False):
        super().__init__()
        self.predict_amplitude = predict_amplitude
        out_channels = 3 if predict_amplitude else 2

        K = len(recon_taus)
        self.K = K

        # τ positional encoding: log(τ_k) / log(τ_max) ∈ [0, 1]
        tau_t   = torch.tensor(list(recon_taus), dtype=torch.float32)
        log_tau = torch.log(tau_t.clamp(min=1.0))
        tau_pe  = log_tau / (log_tau.max() + 1e-8)            # (K,)
        self.register_buffer("tau_pe", tau_pe)

        # Encoder — input is 2K channels (K G_norm + K τ-PE)
        self.inc   = DoubleConv(K * 2, 64)
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
        """
        Args:
            x: (B, K, H, W) — K masked G_norm channels (τ-shuffled or normal).
        """
        B, K, H, W = x.shape
        # τ-PE broadcast: (K,) → (1, K, 1, 1) → (B, K, H, W)
        tau_pe_sp = self.tau_pe.view(1, K, 1, 1).expand(B, K, H, W)
        # Concatenate G_norm and τ-PE: (B, 2K, H, W)
        # When G_norm channels are τ-shuffled (perm), τ-PE stays in correct order
        # → creates G_norm(τ_k) ↔ τ_{perm⁻¹(k)} mismatch → physics inconsistency
        x_in = torch.cat([x, tau_pe_sp], dim=1)               # (B, 2K, H, W)

        # Encode
        x1 = self.inc(x_in)
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

        # Direction D: ELU+1 — no saturation at either extreme
        alpha_map = (F.elu(p[:, 1:2]) + 1.001).clamp(max=2.0) # (0.001, 2]

        if self.predict_amplitude:
            amp_map = self.amp_activation(p[:, 2:3])
            return torch.cat([gamma_map, alpha_map, amp_map], dim=1)
        return torch.cat([gamma_map, alpha_map], dim=1)         # (B, 2, H, W)


if __name__ == "__main__":
    taus  = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)   # K=10
    model = PISSLTauEncoder(recon_taus=taus, predict_amplitude=False)
    x     = torch.randn(4, 10, 64, 64)               # (B, K, H, W)
    out   = model(x)

    print(f"Input  (B, K, H, W)    : {x.shape}")
    print(f"Output (B, [γ,α], H, W): {out.shape}")
    print(f"Gamma range: [{out[:,0].min():.4f}, {out[:,0].max():.4f}]")
    print(f"Alpha range: [{out[:,1].min():.4f}, {out[:,1].max():.4f}]")
    print(f"τ-PE values: {model.tau_pe.tolist()}")

    # Verify shuffle test detects physics mismatch
    perm     = torch.randperm(10)
    out_shuf = model(x[:, perm, :, :])               # shuffled G_norm, fixed τ-PE
    dg = (out[:,0] - out_shuf[:,0]).abs().mean().item()
    da = (out[:,1] - out_shuf[:,1]).abs().mean().item()
    print(f"\nShuffle diff (random init, no training): |Δγ|={dg:.4f}  |Δα|={da:.4f}")
