import math
import os

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def generate_signal_with_autocorr(T, gamma, alpha, mean_val=100.0, variance=10.0):
    """
    Generates a 1D time series of length T with autocorrelation decay
    G(tau) = 1 / (1 + gamma * tau^alpha) via Wiener-Khinchin theorem.
    """
    if gamma == 0:
        return np.random.normal(mean_val, np.sqrt(variance), T)

    tau     = np.arange(T)
    tau_sym = np.minimum(tau, T - tau)
    g_tau   = 1.0 / (1.0 + gamma * (tau_sym ** alpha))

    psd     = np.fft.fft(g_tau).real
    psd     = np.maximum(psd, 0)

    white_f = np.fft.fft(np.random.randn(T))
    colored = np.fft.ifft(white_f * np.sqrt(psd)).real
    colored -= colored.mean()
    colored  = colored / (colored.std() + 1e-12) * np.sqrt(variance)
    return colored + mean_val


def _identifiability_summary(regions, tau=128):
    """Print G_norm and dG/dα at tau for each region."""
    print(f"  Region identifiability at tau_max={tau}:")
    for name, g, a in regions:
        gn   = (1 + g) / (1 + g * tau ** a)
        dGda = -(1+g) * g * (tau**a) * math.log(tau) / (1 + g * tau**a)**2
        print(f"    {name:<28} γ={g:.2f} α={a:.1f}:  "
              f"G_norm({tau})={gn:.4f}  dG/dα({tau})={dGda:.4f}")


def create_test_video(output_path="./data/test_synthetic_cell.tif",
                      T=2000, H=128, W=128):
    """
    Synthetic iSCAT test video — three nested circles + near-static background.

    Training / held-out evaluation video (v1 geometry).
    Hard-edge boundaries; used as the primary benchmark since v3.0.

    T=2000 rationale:
        G_empirical error ≈ 2.3% at tau=128  (vs 12% for T=200).

    Region design:
        Large circle  (cell body / euchromatin):   gamma=0.10, alpha=1.0
        Small circle 1 (condensed, fast):          gamma=0.50, alpha=1.5
        Small circle 2 (confined sub-diffusion):   gamma=0.05, alpha=0.5
    Background: near-static (CV=0.05%).
    """
    print(f"Generating synthetic video v1 (T={T}, H={H}, W={W})...")

    video        = np.zeros((T, H, W), dtype=np.float32)
    gt_gamma_map = np.zeros((H, W), dtype=np.float32)
    gt_alpha_map = np.zeros((H, W), dtype=np.float32)

    MEAN     = 100.0
    CELL_VAR = 10.0
    BG_STD   = 0.05

    cy_l,  cx_l,  r_l  = 64, 64, 50
    cy_s1, cx_s1, r_s1 = 45, 45, 15
    cy_s2, cx_s2, r_s2 = 85, 85, 15

    gamma_l,  alpha_l  = 0.10, 1.0
    gamma_s1, alpha_s1 = 0.50, 1.5
    gamma_s2, alpha_s2 = 0.05, 0.5

    for y in range(H):
        for x in range(W):
            d_l  = np.sqrt((x - cx_l)  ** 2 + (y - cy_l)  ** 2)
            d_s1 = np.sqrt((x - cx_s1) ** 2 + (y - cy_s1) ** 2)
            d_s2 = np.sqrt((x - cx_s2) ** 2 + (y - cy_s2) ** 2)

            if d_s1 <= r_s1:
                g, a = gamma_s1, alpha_s1
            elif d_s2 <= r_s2:
                g, a = gamma_s2, alpha_s2
            elif d_l <= r_l:
                g, a = gamma_l, alpha_l
            else:
                video[:, y, x] = np.random.normal(MEAN, BG_STD, T)
                continue

            video[:, y, x]     = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)
            gt_gamma_map[y, x] = g
            gt_alpha_map[y, x] = a

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, video)
    print(f"Saved video -> {output_path}")

    gt_path = output_path.replace('.tif', '_gt.npz')
    np.savez(gt_path, gamma=gt_gamma_map, alpha=gt_alpha_map)
    print(f"Saved GT maps -> {gt_path}")

    print(f"\nT={T}:")
    _identifiability_summary([
        ("Cell body (normal)",         0.10, 1.0),
        ("Fast spot (super-diffusion)", 0.50, 1.5),
        ("Slow spot (sub-diffusion)",   0.05, 0.5),
    ])
    print()


def create_test_video_v2(output_path="./data/test_synthetic_cell_v2.tif",
                         T=2000, H=128, W=128, sigma_edge=3.0):
    """
    Synthetic iSCAT test video — concentric rings + soft boundaries.

    Purpose: held-out overfitting test.
      Train model on v1 (nested circles, hard edges).
      Evaluate on v2 (concentric rings, different parameters, soft edges).
      Consistent quality across both → method generalises beyond specific geometry.

    Two differences from v1 that probe different failure modes:

    (1) Different geometry — concentric rings instead of nested circles.
        v1: three separate circles (overlapping allowed, inner takes priority).
        v2: radially ordered zones (inner disc → middle ring → outer ring).
        The model cannot exploit the "circle at fixed offset" prior it may have
        seen in v1; it must use temporal physics, not spatial familiarity.

    (2) Soft (physically realistic) boundaries — Gaussian blur of parameter maps.
        sigma_edge = 3.0 px ≈ 1 PSF width (at 65nm/px, 532nm, NA=1.4).
        Why soft edges matter:
          - v1 hard edges test if the model can reconstruct step functions.
          - Real diffusion domains (chromatin condensates, lipid rafts) have
            gradual parameter transitions on the scale of the PSF.
          - A model that only recovers hard-edge boundaries but fails on gradual
            ones is learning boundary detection, not physics inversion.
          - The GT for v2 is the BLURRED parameter map, which is the true
            generating function. MAE is computed against blurred GT.
        Trade-off: soft GT has smaller dynamic range near boundaries; the model
        should correctly predict intermediate values there, not hard jumps.

    Region design (different values from v1 to prevent value memorization):
        Inner disc   (r < 20):        γ=0.08, α=0.60  (slow sub-diffusion)
        Middle ring  (20 ≤ r < 40):   γ=0.35, α=1.30  (moderate super-diffusion)
        Outer ring   (40 ≤ r < 54):   γ=0.12, α=0.90  (near-normal diffusion)
        Background   (r ≥ 54):        γ=0.0,  α=0.0   (near-static)

    Note: sigma_edge=0 gives hard-edge v2 (same as v1 approach, different layout).
    """
    print(f"Generating synthetic video v2 — concentric rings, sigma_edge={sigma_edge}px")
    print(f"  T={T}, H={H}, W={W}")

    MEAN     = 100.0
    CELL_VAR = 10.0
    BG_STD   = 0.05

    cx, cy = 64, 64
    R_INNER = 20    # inner disc boundary
    R_MID   = 40    # middle ring boundary
    R_OUTER = 54    # cell boundary

    G_INNER, A_INNER = 0.08, 0.60
    G_MID,   A_MID   = 0.35, 1.30
    G_OUTER, A_OUTER = 0.12, 0.90

    # ---- Build hard-edge parameter maps -------------------------------------
    gt_gamma_hard = np.zeros((H, W), dtype=np.float32)
    gt_alpha_hard = np.zeros((H, W), dtype=np.float32)
    cell_mask     = np.zeros((H, W), dtype=bool)

    yy, xx = np.mgrid[0:H, 0:W]
    dist    = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    gt_gamma_hard[dist < R_INNER] = G_INNER
    gt_alpha_hard[dist < R_INNER] = A_INNER

    ring_mid = (dist >= R_INNER) & (dist < R_MID)
    gt_gamma_hard[ring_mid] = G_MID
    gt_alpha_hard[ring_mid] = A_MID

    ring_out = (dist >= R_MID) & (dist < R_OUTER)
    gt_gamma_hard[ring_out] = G_OUTER
    gt_alpha_hard[ring_out] = A_OUTER

    cell_mask[dist < R_OUTER] = True

    # ---- Apply Gaussian blur for soft boundaries ----------------------------
    # Blur γ and α independently; boundary pixels get interpolated values.
    # Background (γ=0, α=0) blends into cell → transition zone has small γ,α.
    # The CV filter in PhysReconDataset will handle the background classification.
    if sigma_edge > 0:
        gt_gamma_map = gaussian_filter(gt_gamma_hard.astype(np.float64),
                                       sigma=sigma_edge).astype(np.float32)
        gt_alpha_map = gaussian_filter(gt_alpha_hard.astype(np.float64),
                                       sigma=sigma_edge).astype(np.float32)
        print(f"  Applied Gaussian blur sigma={sigma_edge}px to parameter maps.")
        print(f"  γ range after blur: [{gt_gamma_map.min():.4f}, {gt_gamma_map.max():.4f}]")
        print(f"  α range after blur: [{gt_alpha_map.min():.4f}, {gt_alpha_map.max():.4f}]")
    else:
        gt_gamma_map = gt_gamma_hard.copy()
        gt_alpha_map = gt_alpha_hard.copy()

    # ---- Generate pixel-wise signals using (possibly blurred) parameters ----
    video = np.zeros((T, H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            g = float(gt_gamma_map[y, x])
            a = float(gt_alpha_map[y, x])

            if g < 1e-4 and not cell_mask[y, x]:
                # True background: near-static
                video[:, y, x] = np.random.normal(MEAN, BG_STD, T)
            else:
                # Cell pixel (or blurred boundary): generate autocorrelated signal
                # For very small gamma (blur transition), signal approaches white noise
                video[:, y, x] = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)

    # ---- Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, video)
    print(f"Saved video v2 -> {output_path}")

    gt_path = output_path.replace('.tif', '_gt.npz')
    np.savez(gt_path,
             gamma=gt_gamma_map, alpha=gt_alpha_map,
             gamma_hard=gt_gamma_hard, alpha_hard=gt_alpha_hard,
             sigma_edge=np.float32(sigma_edge))
    print(f"Saved GT maps -> {gt_path}  (includes blurred + hard-edge versions)")

    print(f"\nT={T}, sigma_edge={sigma_edge}:")
    _identifiability_summary([
        ("Inner disc (slow sub-diff)",    G_INNER, A_INNER),
        ("Middle ring (moderate super)",  G_MID,   A_MID),
        ("Outer ring (near-normal)",      G_OUTER, A_OUTER),
    ])
    print()


if __name__ == "__main__":
    create_test_video("./data/test_synthetic_cell.tif")
    create_test_video_v2("./data/test_synthetic_cell_v2.tif")
