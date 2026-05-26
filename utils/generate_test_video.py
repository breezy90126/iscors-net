import numpy as np
import tifffile
import os


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


def create_test_video(output_path="./data/test_synthetic_cell.tif",
                      T=2000, H=128, W=128):
    """
    Synthetic iSCAT test video — three diffusion regions + near-static background.

    T=2000 rationale:
        At tau=128, N_eff = T-128.
        T=200  -> N_eff=72  -> G_empirical error ~12%
        T=2000 -> N_eff=1872 -> G_empirical error ~2.3%  (26x SNR gain)
        Separates "physics unidentifiable" from "statistics insufficient".

    Region design (motivated by chromatin dynamics in iSCORS):
        Large circle  (cell body / euchromatin):   gamma=0.10, alpha=1.0
            Normal diffusion, moderate speed.
            Gradient concern: dG/dalpha ~= gamma*log(tau) — weak at small tau.

        Small circle 1 (condensed domain, fast):   gamma=0.50, alpha=1.5
            Super-diffusion (active transport / chromatin loop extrusion).
            High gamma -> gradient is healthy.

        Small circle 2 (confined sub-diffusion):   gamma=0.05, alpha=0.5
            Slow, anomalous. Realistic for heterochromatin or membrane-
            tethered proteins at high frame rate (1000 fps: tau_D ~ 20 frames).
            Hardest case: small gamma AND alpha < 1.

    Background:
        Near-static (std=0.05, CV=0.05%) -> CV filter safely skips these pixels.
    """
    print(f"Generating synthetic video (T={T}, H={H}, W={W})...")

    video        = np.zeros((T, H, W), dtype=np.float32)
    gt_gamma_map = np.zeros((H, W), dtype=np.float32)
    gt_alpha_map = np.zeros((H, W), dtype=np.float32)

    MEAN     = 100.0
    CELL_VAR = 10.0    # CV ~ 3.2% — typical iSCAT cell pixel
    BG_STD   = 0.05    # CV = 0.05% — near-static background

    # Large circle — cell body (normal diffusion)
    cy_l,  cx_l,  r_l  = 64, 64, 50
    gamma_l,  alpha_l  = 0.10, 1.0

    # Small circle 1 — fast anomalous region (upper-left)
    cy_s1, cx_s1, r_s1 = 45, 45, 15
    gamma_s1, alpha_s1 = 0.50, 1.5

    # Small circle 2 — slow sub-diffusion region (lower-right)
    cy_s2, cx_s2, r_s2 = 85, 85, 15
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
                g, a = 0.0, 0.0
                video[:, y, x] = np.random.normal(MEAN, BG_STD, T)
                gt_gamma_map[y, x] = g
                gt_alpha_map[y, x] = a
                continue

            video[:, y, x]   = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)
            gt_gamma_map[y, x] = g
            gt_alpha_map[y, x] = a

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, video)
    print(f"Saved video -> {output_path}")

    gt_path = output_path.replace('.tif', '_gt.npz')
    np.savez(gt_path, gamma=gt_gamma_map, alpha=gt_alpha_map)
    print(f"Saved GT maps -> {gt_path}")

    # Print identifiability summary
    print(f"\nRegion identifiability at tau_max=128 (T={T}):")
    import math
    for name, g, a in [("Cell body  ", 0.10, 1.0),
                       ("Fast spot  ", 0.50, 1.5),
                       ("Slow spot  ", 0.05, 0.5)]:
        tau = 128
        gn  = (1 + g) / (1 + g * tau ** a)         # G_norm at tau=128
        dGda = -(1+g)*g*(tau**a)*math.log(tau) / (1+g*tau**a)**2
        print(f"  {name} gamma={g:.2f} alpha={a:.1f}: "
              f"G_norm(128)={gn:.3f}  dG/dalpha(128)={dGda:.4f}")
    print()


if __name__ == "__main__":
    create_test_video("./data/test_synthetic_cell.tif")
