"""
SN2N-style spatial diagonal resampling and σ_G estimation for iSCAT video.

Reference: SN2N (Self-supervised Noise-to-Noise) — spatial XY checkerboard
subsampling. For xyt data: per-frame spatial resampling, all T frames preserved
→ G(τ) computation on either grid is completely unaffected by the split.

Two complementary grids from one video:
  Grid A: average of (top-left, bottom-right) of each 2×2 block
  Grid B: average of (top-right, bottom-left) of each 2×2 block

Both grids have size H/2 × W/2, same T dimension, and statistically independent
measurement noise. G(τ) on Grid A and Grid B are independent estimates of the
same underlying physics — enabling cross-validation without GT labels.

Usage for checkerboard validation:
    gridA, gridB = diagonal_resample(video)
    # Traditional iSCORS on gridA → quasi-GT at H/2 × W/2
    # Model inference on gridB   → prediction at H/2 × W/2
    # Compare: model vs quasi-GT on held-out spatial grid

σ_G estimation:
    σ²_G(τ; y,x) ≈ (2/T) * [G(0; y,x)² + G(τ; y,x)²]
    Derived from Wiener-Khinchin noise propagation for finite-T autocorrelation.
    G(0; y,x) = Var(δI(y,x)) / ⟨I(y,x)⟩² — zero-lag autocorrelation.
"""

import numpy as np


def diagonal_resample(video):
    """
    Split (T, H, W) video into two statistically independent spatial grids.

    Each 2×2 pixel block is split into its two diagonals:
      Grid A: (top-left  + bottom-right) / 2   (even-diagonal)
      Grid B: (top-right + bottom-left)  / 2   (odd-diagonal)

    The two grids share no pixels and have independent photon noise, so
    autocorrelations G(τ) computed on each grid are independent estimates
    of the same diffusion physics.

    Args:
        video: (T, H, W) float32 array.

    Returns:
        gridA: (T, H//2, W//2) float32
        gridB: (T, H//2, W//2) float32
    """
    T, H, W = video.shape
    H2, W2 = H // 2, W // 2
    # Reshape into 2×2 blocks: (T, H2, 2, W2, 2)
    block = video[:, :H2 * 2, :W2 * 2].reshape(T, H2, 2, W2, 2)
    gridA = (block[:, :, 0, :, 0] + block[:, :, 1, :, 1]) / 2.0  # even diagonal
    gridB = (block[:, :, 0, :, 1] + block[:, :, 1, :, 0]) / 2.0  # odd diagonal
    return gridA.astype(np.float32), gridB.astype(np.float32)


def compute_sigma_g(video, taus, g_map=None):
    """
    Per-pixel noise estimate σ_G(τ) from Wiener-Khinchin noise propagation.

    σ²_G(τ; y,x) ≈ (2/T) * [G(0; y,x)² + G(τ; y,x)²]

    Properties:
      - Large at small τ when G(τ) is large (shot-noise-limited regime).
      - Large when G(τ_ref) ≈ 0 (normalization instability for fast dynamics).
      - Combined with Fisher weights → only τ channels that are BOTH
        physically informative AND reliably measured contribute to the loss.

    Args:
        video: (T, H, W) float32, mean ≈ 1.0 (after background correction).
        taus:  tuple/list of integer τ lag values matching g_map channel order.
        g_map: (H, W, K) float32 — pre-computed G_empirical at the given taus.
               Pass this to avoid redundant computation. If None, G(τ) is
               estimated directly from video.

    Returns:
        sigma_g: (H, W, K) float32 — noise std for each pixel and τ channel.
    """
    video = video.astype(np.float32)
    T, H, W = video.shape
    mean_I  = video.mean(axis=0)          # (H, W)
    delta_I = video - mean_I
    denom   = mean_I ** 2 + 1e-10

    # G(0) = Var(δI) / ⟨I⟩²  — zero-lag autocorrelation
    g_zero = (delta_I ** 2).mean(axis=0) / denom   # (H, W)

    if g_map is None:
        K = len(taus)
        g_vals = np.zeros((H, W, K), dtype=np.float32)
        for i, tau in enumerate(taus):
            n = T - int(tau)
            g_vals[:, :, i] = (delta_I[:n] * delta_I[int(tau):]).mean(axis=0) / denom
    else:
        g_vals = g_map  # (H, W, K)

    # σ²_G(τ) = (2/T) * (G(0)² + G(τ)²)
    sigma_g = np.sqrt((2.0 / T) * (g_zero[:, :, None] ** 2 + g_vals ** 2))
    return sigma_g.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    T, H, W = 500, 64, 64
    rng = np.random.default_rng(0)
    video = rng.standard_normal((T, H, W)).astype(np.float32) + 100.0

    # Test diagonal_resample
    gA, gB = diagonal_resample(video)
    print(f"Input : {video.shape}")
    print(f"Grid A: {gA.shape}")
    print(f"Grid B: {gB.shape}")

    # Pixel-level independence check: correlation of A and B fluctuations should be low
    dA = gA - gA.mean(axis=0)
    dB = gB - gB.mean(axis=0)
    corr = float((dA * dB).mean() / (dA.std() * dB.std() + 1e-10))
    print(f"A-B temporal correlation (expect ≈0 for independent noise): {corr:.4f}")

    # Test compute_sigma_g
    taus = (1, 2, 4, 8, 16, 32)
    sigma = compute_sigma_g(video, taus)
    print(f"\nσ_G shape: {sigma.shape}")
    print(f"σ_G range: [{sigma.min():.4e}, {sigma.max():.4e}]")
    print(f"σ_G at τ=1 (mean): {sigma[:,:,0].mean():.4e}")
    print(f"σ_G at τ=32 (mean): {sigma[:,:,-1].mean():.4e}")
    print("Expected: σ ∝ 1/sqrt(T) ≈ {:.4e}".format(1.0 / T**0.5))
