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

    tau = np.arange(T)
    tau_sym = np.minimum(tau, T - tau)
    g_tau = 1.0 / (1.0 + gamma * (tau_sym ** alpha))

    psd = np.fft.fft(g_tau).real
    psd = np.maximum(psd, 0)

    white_f = np.fft.fft(np.random.randn(T))
    colored = np.fft.ifft(white_f * np.sqrt(psd)).real

    colored -= colored.mean()
    colored = colored / (colored.std() + 1e-12) * np.sqrt(variance)
    return colored + mean_val


def create_test_video(output_path="./data/test_synthetic_cell.tif", T=500, H=128, W=128):
    """
    Synthetic iSCAT test video with three diffusion regions and a near-static background.

    Background design:
        gamma=0, alpha=0 — but implemented as near-static (std=0.05) so that
        CV-based filtering in fit_physical_parameters correctly skips these pixels.
        The old approach (white noise) gave CV~3%, same as cell pixels, causing
        curve fitting to return garbage gamma values for background.

    Cell regions:
        T=500 frames so that autocorrelation at tau up to 64 has good statistics
        (500-64=436 valid starting points vs. 136 with T=200).
    """
    print(f"Generating synthetic video (T={T}, H={H}, W={W})...")

    video = np.zeros((T, H, W), dtype=np.float32)
    gt_gamma_map = np.zeros((H, W), dtype=np.float32)
    gt_alpha_map = np.zeros((H, W), dtype=np.float32)

    MEAN = 100.0
    CELL_VAR = 10.0          # cell pixel temporal variance (std ≈ 3.16, CV ≈ 3.2%)
    BG_STD = 0.05            # background std → CV = 0.05/100 = 0.05% → safely filtered

    # Large circle — cell body (normal diffusion)
    cy_l, cx_l, r_l = 64, 64, 50
    gamma_l, alpha_l = 0.1, 1.0

    # Small circle 1 — organelle (fast anomalous diffusion)
    cy_s1, cx_s1, r_s1 = 45, 45, 15
    gamma_s1, alpha_s1 = 0.5, 1.5

    # Small circle 2 — organelle (slow subdiffusion)
    cy_s2, cx_s2, r_s2 = 85, 85, 15
    gamma_s2, alpha_s2 = 0.05, 0.5

    for y in range(H):
        for x in range(W):
            d_l  = np.sqrt((x - cx_l) ** 2  + (y - cy_l) ** 2)
            d_s1 = np.sqrt((x - cx_s1) ** 2 + (y - cy_s1) ** 2)
            d_s2 = np.sqrt((x - cx_s2) ** 2 + (y - cy_s2) ** 2)

            if d_s1 <= r_s1:
                g, a = gamma_s1, alpha_s1
                trace = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)
            elif d_s2 <= r_s2:
                g, a = gamma_s2, alpha_s2
                trace = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)
            elif d_l <= r_l:
                g, a = gamma_l, alpha_l
                trace = generate_signal_with_autocorr(T, g, a, MEAN, CELL_VAR)
            else:
                # Near-static background — mimics glass surface in real iSCAT.
                # CV = BG_STD/MEAN = 0.05% << cell CV (3.2%), so the CV filter
                # in fit_physical_parameters will correctly return (0.0, 0.0).
                g, a = 0.0, 0.0
                trace = np.random.normal(MEAN, BG_STD, T)

            gt_gamma_map[y, x] = g
            gt_alpha_map[y, x] = a
            video[:, y, x] = trace

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, video)
    print(f"Saved video → {output_path}")

    gt_path = output_path.replace('.tif', '_gt.npz')
    np.savez(gt_path, gamma=gt_gamma_map, alpha=gt_alpha_map)
    print(f"Saved GT maps → {gt_path}")
    print("Done.")


if __name__ == "__main__":
    create_test_video("./data/test_synthetic_cell.tif")
