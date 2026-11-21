"""Validate the iPSF kernel K[tau, D] (utils/ipsf_kernel.py) and show what it buys.

PART 1 — does the analytic kernel match a brute-force simulation?
    Particles diffuse (2D, periodic box) under a fine-sampled iPSF; one binned camera
    pixel is recorded for T frames; the pipeline's 'nor1' estimator is averaged over
    many repeats. That empirical G(tau) must match ipsf_kernel(..., T=T) — including
    the finite-T bias — and differ from the ideal (T=None) column at long tau.

PART 2 — single population: P(D) from the iPSF kernel vs the DLS exponential kernel.
    The exponential kernel (what utils/laplace_spectrum.contin uses) turns a SINGLE D
    into a broad rate spectrum (the q-integration); the iPSF kernel returns one peak
    at the right D.

PART 3 — two populations at ratio R, at an ROI-level noise: which R is resolved.

Run:  python3 scripts/validate_ipsf_kernel.py          (numpy only, ~1-2 min)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ipsf_kernel import (scalar_ipsf, ipsf_spectrum, effective_q2, ipsf_kernel,  # noqa: E402
                               invert_kernel, spectrum_summary)

TAUS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)            # = RECON_TAUS
CAM_PIX, BIN = 72e-9, 2                                 # camera pixel, BIN_FACTOR
UP = 4                                                  # fine sampling for the MC
DT = 1.0                                                # work in frames: D in m^2/frame


def make_ipsf(n_cam=64, z=70e-9):
    fine = scalar_ipsf(n_pix=n_cam * UP, pixel_size=CAM_PIX / UP, z=z)
    return fine, CAM_PIX / UP, BIN * UP


def binned_pixel_image(fine, b):
    """Circular b x b box average of the fine iPSF = signal of the binned camera pixel
    as a function of particle offset."""
    n = fine.shape[0]
    box = np.zeros((n, n))
    box[:b, :b] = 1.0 / b ** 2
    box = np.roll(box, (-(b // 2), -(b // 2)), axis=(0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(fine) * np.fft.fft2(box)))


def bilinear_periodic(img, x, y):
    n = img.shape[0]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x0 %= n
    y0 %= n
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1]
            + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])


def simulate_g(fine, px, b, D, T=2000, n_part=8, reps=300, seed=0):
    """'nor1' G_hat(tau) of the binned pixel over `reps` independent pixels.

    Returns (ratio_of_means, mean_of_ratio, se):
      ratio_of_means = mean_px C_hat(tau) / mean_px C_hat(0)   <- what the kernel models
      mean_of_ratio  = mean_px [C_hat(tau) / C_hat(0)]         <- per-pixel g_norm, then ROI mean
    """
    rng = np.random.default_rng(seed)
    pix_img = binned_pixel_image(fine, b)
    n = fine.shape[0]
    step = np.sqrt(2 * D * DT) / px                      # in fine pixels
    pos = rng.uniform(0, n, size=(2, reps, n_part))
    trace = np.empty((T, reps))
    for t in range(T):
        trace[t] = bilinear_periodic(pix_img, pos[0], pos[1]).sum(axis=1)
        pos += step * rng.standard_normal(pos.shape)
    d = trace - trace.mean(axis=0)
    c0 = (d ** 2).mean(axis=0)
    C = np.array([(d[:T - tau] * d[tau:]).mean(axis=0) for tau in TAUS])  # (k, reps)
    rom = C.mean(axis=1) / c0.mean()
    # delta-method standard error of the ratio of means
    r = C - rom[:, None] * c0[None, :]
    se = r.std(axis=1) / np.sqrt(reps) / c0.mean()
    return rom, (C / c0).mean(axis=1), se


def part1():
    print("=== PART 1 - analytic kernel vs brute-force simulation ===")
    fine, px, b = make_ipsf()
    q2, S = ipsf_spectrum(fine, px, bin_factor=b)
    q2e = effective_q2(q2, S)
    T = 2000
    for label, target in (("fast", 0.3), ("slow", 0.03)):
        D = target / (q2e * DT)                          # D q_eff^2 dt = target
        g_sim, g_mor, se = simulate_g(fine, px, b, D, T=T)
        k_T, _ = ipsf_kernel(TAUS, DT, q2, S, D=[D], T=T)
        k_id, _ = ipsf_kernel(TAUS, DT, q2, S, D=[D], T=None)
        z_T = (g_sim - k_T[:, 0]) / se
        print(f"\n  {label}: D*q_eff^2*dt = {target}")
        print("   tau   sim(ratio of means) +-se   kernel(T)  kernel(ideal)  (sim-kT)/se"
              "   sim(mean of ratios)")
        for i, tau in enumerate(TAUS):
            print(f"  {tau:4d}      {g_sim[i]:8.4f}  {se[i]:7.4f}   {k_T[i, 0]:8.4f}   "
                  f"{k_id[i, 0]:8.4f}      {z_T[i]:+6.2f}        {g_mor[i]:8.4f}")
        chi2 = float((z_T ** 2).mean())
        chi2_id = float((((g_sim - k_id[:, 0]) / se) ** 2).mean())
        print(f"  mean z^2: finite-T kernel {chi2:.2f}   ideal kernel {chi2_id:.2f}"
              f"   (~1 = consistent with the simulation)")
    print("\n  The kernel models E[C_hat(tau)]/E[C_hat(0)]. Averaging per-pixel RATIOS adds a\n"
          "  further (Jensen) bias when C_hat(0) fluctuates, i.e. sparse/intermittent\n"
          "  signals. For the ROI curve, average C_hat(tau) and C_hat(0) separately and\n"
          "  divide: g_roi = sum(g_norm * c0) / sum(c0) over the ROI.")


def exp_kernel_inversion(g, taus, n=80, lam=1e-3):
    taus = np.asarray(taus, float)
    rates = np.logspace(np.log10(1 / (3 * taus.max())), np.log10(3 / taus.min()), n)
    K = np.exp(-np.outer(taus, rates))
    return invert_kernel(g, K, lam=lam), rates


def part2_3():
    fine, px, b = make_ipsf()
    q2, S = ipsf_spectrum(fine, px, bin_factor=b, n_q2_bins=512)
    q2e = effective_q2(q2, S)
    T = 5000
    K, Dg = ipsf_kernel(TAUS, DT, q2, S, T=T, n_D=80)
    D0 = 0.05 / (q2e * DT)
    rng = np.random.default_rng(1)

    print("\n=== PART 2 - single D: iPSF kernel vs exponential (DLS) kernel ===")
    k1, _ = ipsf_kernel(TAUS, DT, q2, S, D=[D0], T=T)
    g = k1[:, 0]
    p = invert_kernel(g, K)
    s = spectrum_summary(p, Dg)
    pe, rates = exp_kernel_inversion(g, TAUS, lam=1e-4)
    se_ = spectrum_summary(pe, rates)
    print(f"  true D = {D0:.3e}")
    print(f"  iPSF kernel : peaks at D = {np.array2string(s['peak_D'], precision=3)}, "
          f"width {s['width_dec']:.2f} dec")
    print(f"  exp kernel  : {len(se_['peak_D'])} peak(s), width {se_['width_dec']:.2f} dec "
          f"<- the q-integration showing up as fake polydispersity")

    print("\n=== PART 3 - two populations (50/50), ROI noise, 20 noise draws each ===")
    print("  noise is per-tau RMS on G (1 px ~ 4.1%, ~1000 px ~ 0.13%)")
    print("   R     noise    resolved(2 peaks within 0.15 dec of truth)/20")
    for noise in (0.0013, 0.0004):
        for R in (2, 3, 5, 10, 30):
            D1, D2 = D0 / np.sqrt(R), D0 * np.sqrt(R)
            kk, _ = ipsf_kernel(TAUS, DT, q2, S, D=[D1, D2], T=T)
            g_true = 0.5 * kk[:, 0] + 0.5 * kk[:, 1]
            ok = 0
            for _ in range(20):
                gn = g_true + noise * rng.standard_normal(len(TAUS))
                p = invert_kernel(gn, K)
                pk = spectrum_summary(p, Dg)["peak_D"]
                if len(pk) == 2 and all(np.min(np.abs(np.log10(pk) - np.log10(d))) < 0.15
                                        for d in (D1, D2)):
                    ok += 1
            print(f"  {R:4d}  {noise:7.4f}   {ok:2d}/20")


if __name__ == "__main__":
    part1()
    part2_3()
