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
    U-Net for Physics-Informed Spatial Sampling.

    v3.9+: input is (B, 2K, P, P) = K G_norm channels || K τ-PE channels.
    v4.1+: optionally (B, 3K, P, P) = K G_norm || K τ-PE || K σ_G_norm channels
           when use_sigma=True. σ_G_norm lets the model learn to down-weight
           unreliable τ channels (high noise) automatically.

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

    Output activations (v4.6, Direction E — scaled-sigmoid):
        γ: gamma_scale · Sigmoid(x) → (0, gamma_scale), default scale 2.0
            Empirical g0_norm fits reach ≈1.7–2.0 — plain Sigmoid (0,1)
            (v4.5) was clipping real signal at the upper end.
        α: 2 · Sigmoid(x) → (0, 2)
            x=0 → α=1.0 (healthy default, same as v4.5 ELU+1)
            Smooth saturation at both ends — gradient shrinks but never
            hits exactly zero, so there is no boundary "pile-up" the way
            the v4.5 hard .clamp(max=2.0) produced at α=2.0.
    """
    def __init__(self, recon_taus, predict_amplitude=False, use_sigma=False, gamma_scale=2.0,
                 n_components=1):
        super().__init__()
        assert n_components in (1, 2), "n_components must be 1 or 2"
        self.predict_amplitude = predict_amplitude
        self.use_sigma = use_sigma
        self.gamma_scale = gamma_scale
        self.n_components = n_components
        if n_components == 2:
            assert not predict_amplitude, \
                "predict_amplitude is only supported for n_components=1"
            out_channels = 4               # f, γ_slow, Δγ(→softplus), α
        else:
            out_channels = 3 if predict_amplitude else 2

        K = len(recon_taus)
        self.K = K

        # τ positional encoding: log(τ_k) / log(τ_max) ∈ [0, 1]
        tau_t   = torch.tensor(list(recon_taus), dtype=torch.float32)
        log_tau = torch.log(tau_t.clamp(min=1.0))
        tau_pe  = log_tau / (log_tau.max() + 1e-8)            # (K,)
        self.register_buffer("tau_pe", tau_pe)

        # Encoder — 2K channels (G_norm + τ-PE) or 3K (+ σ_G_norm) if use_sigma
        in_ch = K * 3 if use_sigma else K * 2
        self.inc   = DoubleConv(in_ch, 64)
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

        # v4.6: smooth scaled-sigmoid activations (Direction E).
        # Both replace hard clamps with saturating-but-never-flat curves —
        # gradient is small but never exactly zero anywhere in range, so
        # there is no boundary "pile-up" (cf. v4.5 ELU+1 clamp at α=2.0).
        # γ ∈ (0, gamma_scale): empirical g0_norm fits (checkerboard gridA,
        #   1/D_map GT) reach ≈1.7–2.0 — γ ∈ (0,1) was clipping real signal.
        #   gamma_scale=2.0 gives headroom while keeping training stable
        #   (a fully unbounded Softplus risks early-training blow-up).
        # α ∈ (0, 2): 2 is the genuine physical ceiling (ballistic motion,
        #   MSD ~ t²). 2·sigmoid(x) is symmetric: x=0 → α=1.0 (healthy
        #   default, same as v4.5), and approaches 0 / 2 smoothly from
        #   both sides — no more asymmetric ELU compression of α<1.
        self.gamma_activation = nn.Sigmoid()   # scaled to (0, gamma_scale) in forward()
        self.alpha_activation = nn.Sigmoid()   # scaled to (0, 2) in forward()
        self.amp_activation   = nn.Softplus()  # A > 0
        self.delta_activation = nn.Softplus()  # Δγ ≥ 0 → γ_fast = γ_slow + Δγ (2-comp)

    def forward(self, x, sigma_g_norm=None):
        """
        Args:
            x:            (B, K, H, W) — K masked G_norm channels.
            sigma_g_norm: (B, K, H, W) optional — normalised σ_G channels.
                          Required when use_sigma=True; ignored otherwise.
        """
        B, K, H, W = x.shape
        # τ-PE broadcast: (K,) → (1, K, 1, 1) → (B, K, H, W)
        tau_pe_sp = self.tau_pe.view(1, K, 1, 1).expand(B, K, H, W)
        if self.use_sigma and sigma_g_norm is not None:
            # 3K channels: G_norm || τ-PE || σ_G_norm
            x_in = torch.cat([x, tau_pe_sp, sigma_g_norm], dim=1)  # (B, 3K, H, W)
        else:
            # 2K channels: G_norm || τ-PE  (backward-compatible default)
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

        if self.n_components == 2:
            # Two-component (shared-α) mixture. Output channels:
            #   f       ∈ (0,1)             fast-component fractional amplitude
            #   γ_slow  ∈ (0, gamma_scale)
            #   γ_fast  = γ_slow + Δγ       (Δγ = softplus ≥ 0 → enforces γ_fast ≥ γ_slow,
            #                                breaking the label-swap symmetry that would
            #                                otherwise give the loss two equivalent minima)
            #   α       ∈ (0, 2)            shared anomalous exponent
            # The loss builds G_norm(τ)=f/(1+γ_fast τ^α)+(1-f)/(1+γ_slow τ^α). Giving
            # heterogeneity its own d.o.f. (f, γ_fast-γ_slow) frees α to represent genuine
            # anomaly instead of absorbing distribution width (the single-component
            # mean-regression). Returns (B, 4, H, W) = [f, γ_slow, γ_fast, α].
            f_map  = self.gamma_activation(p[:, 0:1])                      # sigmoid → (0,1)
            g_slow = self.gamma_scale * self.gamma_activation(p[:, 1:2])   # (0, gamma_scale)
            g_fast = g_slow + self.delta_activation(p[:, 2:3])             # ≥ γ_slow
            alpha_map = 2.0 * self.alpha_activation(p[:, 3:4])             # (0, 2)
            return torch.cat([f_map, g_slow, g_fast, alpha_map], dim=1)    # (B, 4, H, W)

        # Single component (default). Direction E scaled-sigmoid — smooth saturation,
        # no zero-gradient pile-up at either boundary (cf. v4.5 hard clamp at α=2.0).
        gamma_map = self.gamma_scale * self.gamma_activation(p[:, 0:1])  # (0, gamma_scale)
        alpha_map = 2.0 * self.alpha_activation(p[:, 1:2])               # (0, 2)

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

    # ── Two-component model smoke test ───────────────────────────────────────
    m2  = PISSLTauEncoder(recon_taus=taus, n_components=2)
    o2  = m2(x)
    print(f"\n[n_components=2] output shape: {o2.shape}  (expect (B,4,H,W))")
    f_, gs_, gf_, a_ = o2[:, 0], o2[:, 1], o2[:, 2], o2[:, 3]
    print(f"  f      range: [{f_.min():.3f}, {f_.max():.3f}]   (expect ⊂ (0,1))")
    print(f"  γ_slow range: [{gs_.min():.3f}, {gs_.max():.3f}]")
    print(f"  γ_fast range: [{gf_.min():.3f}, {gf_.max():.3f}]")
    print(f"  α      range: [{a_.min():.3f}, {a_.max():.3f}]")
    assert bool((gf_ >= gs_ - 1e-5).all()), "ordering γ_fast ≥ γ_slow violated"
    print("  ordering γ_fast ≥ γ_slow: OK ✓")
