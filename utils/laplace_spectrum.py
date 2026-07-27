"""CONTIN-style Laplace (decay-rate spectrum) inversion for iSCORS G(tau).

Companion to `scripts/laplace_identifiability.py`, which establishes WHEN this is
worth running. The short version of that study:

  * On RECON_TAUS = (1..128) at the measured per-pixel noise (~4.1% RMS on G),
    the Laplace kernel supports ~4 spectral components — and "4 components" means
    4 smooth singular directions (amplitude, centroid, width, a little skew), NOT
    4 separable populations.
  * alpha in G(tau) = A/(1+gamma tau^alpha) IS the spectrum's width: alpha=1 maps
    exactly onto p(Gamma) = (1/gamma) exp(-Gamma/gamma), width 0.557 decades;
    lower alpha = broader spectrum. So inverting cannot "split alpha out" — it
    re-parameterises the same one degree of freedom.
  * A genuine two-population mixture is therefore invisible per pixel (it just
    lowers the fitted alpha) and only clears the noise on ROI-AVERAGED curves:
    ~100 px for a 10x rate ratio, ~900 px for 3x.

Hence this module's entry point is ROI-level, not per-pixel: `roi_spectrum`,
validated by the same half-split the rest of the pipeline uses. Nothing here
produces a per-pixel map, by design.

Numpy only (no torch) — these are K-vectors, not videos.
"""

import numpy as np


# ───────────────────────────── rate grid / kernel ───────────────────────────
def rate_grid(taus, n=80, pad=3.0):
    """Log-spaced decay rates Gamma spanning (a bit beyond) what the taus can see."""
    taus = np.asarray(taus, float)
    return np.logspace(np.log10(1.0 / (pad * taus.max())),
                       np.log10(pad / taus.min()), n)


def laplace_kernel(taus, rates):
    """K[k, j] = exp(-Gamma_j tau_k), shape (len(taus), len(rates))."""
    return np.exp(-np.outer(np.asarray(taus, float), np.asarray(rates, float)))


def laplace_dof(taus, noise, n_grid=120, pad=3.0):
    """How many spectral components this tau grid supports at `noise` (relative).

    Returns (singular_values_normalised, n_resolvable). Components whose singular
    value falls below the noise are supplied by the regulariser, not the data.
    """
    rates = rate_grid(taus, n_grid, pad)
    w = np.sqrt(np.log(rates[-1] / rates[0]) / (n_grid - 1))     # d(log Gamma) measure
    sv = np.linalg.svd(laplace_kernel(taus, rates) * w, compute_uv=False)
    sv = sv / sv[0]
    return sv, int((sv > noise).sum())


# ───────────────────────── non-negative inversion ───────────────────────────
def contin(g, taus, rates=None, n_iter=4000, smooth=0.0):
    """Non-negative p(Gamma) with K p ~= g, by Richardson-Lucy.

    RL keeps p >= 0 without any active-set bookkeeping, which is what makes a
    Laplace inversion behave at all. `smooth` in (0,1) applies a 3-tap blur of p
    each iteration — the regularisation that plays the role of CONTIN's penalty.

    Args:
        g      : (K,) the measured G(tau), strictly positive (clip the tail first).
        taus   : (K,) integer lags matching g.
        rates  : (M,) Gamma grid, or None for `rate_grid(taus)`.
        smooth : 0 = none (sharpest, noisiest); 0.2-0.5 for noisy data.
    Returns (p, rates) with p summing to ~g extrapolated to tau=0.
    """
    g = np.asarray(g, float)
    if rates is None:
        rates = rate_grid(taus)
    K = laplace_kernel(taus, rates)
    if np.any(g <= 0):
        g = np.clip(g, 1e-6, None)
    colsum = K.sum(axis=0)
    colsum[colsum < 1e-12] = 1e-12
    p = np.full(len(rates), 1.0 / len(rates))
    for _ in range(n_iter):
        pred = K @ p
        p = p * ((K.T @ (g / np.maximum(pred, 1e-12))) / colsum)
        if smooth > 0:
            p = ((1 - smooth) * p
                 + 0.5 * smooth * (np.r_[p[0], p[:-1]] + np.r_[p[1:], p[-1]]))
    return p, rates


def spectrum_moments(p, rates):
    """(centroid, width) of p(Gamma) in decades of log10(Gamma)."""
    p = np.asarray(p, float)
    tot = p.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    w = p / tot
    lg = np.log10(rates)
    mean = float((w * lg).sum())
    return mean, float(np.sqrt((w * (lg - mean) ** 2).sum()))


def count_peaks(p, rel=0.05):
    """Local maxima above `rel` of the peak — the 'how many populations' readout."""
    p = np.asarray(p, float)
    mx = p.max() if p.size else 0.0
    if mx <= 0:
        return 0
    inner = p[1:-1]
    return int(np.sum((inner > p[:-2]) & (inner >= p[2:]) & (inner > rel * mx)))


def alpha_spectrum_width(gamma, alpha, taus, **kw):
    """Width (decades) of the rate spectrum equivalent to a given (gamma, alpha).

    Demonstrates the degeneracy: alpha and spectral width are the same quantity.
    """
    taus = np.asarray(taus, float)
    p, rates = contin(1.0 / (1.0 + gamma * taus ** alpha), taus, **kw)
    return spectrum_moments(p, rates)[1]


# ───────────────────────── one-component reference fit ──────────────────────
def fit_gamma_alpha(g, taus, gamma_scale=2.0):
    """Least-squares (gamma, alpha) for G=1/(1+gamma tau^alpha): grid then refine.

    The reference the spectrum must beat — if a 1-component fit already explains
    the curve to within the noise, the spectrum's extra structure is regularisation.
    Returns (rms_residual, gamma, alpha).
    """
    g = np.asarray(g, float)
    taus = np.asarray(taus, float)
    gams = np.logspace(-4, np.log10(gamma_scale), 120)
    alps = np.linspace(0.2, 2.0, 91)
    pred = 1.0 / (1.0 + gams[:, None, None] * taus[None, None, :] ** alps[None, :, None])
    sse = ((pred - g[None, None, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    gam, alp, best = gams[i], alps[j], sse[i, j]
    for scale in (0.3, 0.1, 0.03, 0.01):
        for _ in range(200):
            improved = False
            for dg, da in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                gg = gam * (1 + scale * dg)
                aa = min(2.0, max(0.05, alp + 0.5 * scale * da))
                r = float(((1.0 / (1.0 + gg * taus ** aa) - g) ** 2).sum())
                if r < best:
                    best, gam, alp, improved = r, gg, aa, True
            if not improved:
                break
    return float(np.sqrt(best / len(taus))), float(gam), float(alp)


# ───────────────────────── the ROI test worth running ───────────────────────
def roi_spectrum(G, mask, taus, G1=None, G2=None, smooth=0.3, n_iter=4000):
    """ROI-averaged Laplace spectrum + the half-split check that it is real.

    Feed this the notebook's CACHE: G/G1/G2 are the (H,W,K) g_norm stacks for the
    full record and the two halves, `mask` is `nuc` (or a sub-region of it).

    The verdict rests on three things, in order:
      1. `n_px` — below ~100 px nothing multi-modal is trustworthy (see the study);
      2. `resid_1comp` vs `noise_rms` — if a single (gamma, alpha) already fits to
         within the half-split noise, the extra peaks are the regulariser talking;
      3. `n_peaks_half1 == n_peaks_half2` AND matching centroids — a peak that does
         not reproduce across halves is noise, the same standard Section 2 applies.

    Returns a dict with p, rates, centroid, width, n_peaks, the 1-component
    reference fit, the measured noise, and `verdict`.
    """
    mask = np.asarray(mask, bool)
    taus = np.asarray(taus, float)

    def _roi(stack):                    # ROI mean over finite pixels only
        rows = np.asarray(stack, float)[mask]
        return np.nanmean(rows[np.isfinite(rows).all(axis=1)], axis=0)

    g = _roi(G)
    n_px = int(mask.sum())

    out = {"n_px": n_px, "g_roi": g}

    noise_rms = np.nan
    if G1 is not None and G2 is not None:
        g1, g2 = _roi(G1), _roi(G2)
        # half-split difference is a direct, assumption-free noise estimate:
        # each half has ~2x the variance of the full record, hence the /2.
        noise_rms = float(np.sqrt(((g1 - g2) ** 2).mean()) / 2.0)
        for tag, gh in (("half1", g1), ("half2", g2)):
            ph, rates_h = contin(gh, taus, smooth=smooth, n_iter=n_iter)
            c, w = spectrum_moments(ph, rates_h)
            out[f"p_{tag}"], out[f"centroid_{tag}"] = ph, c
            out[f"width_{tag}"], out[f"n_peaks_{tag}"] = w, count_peaks(ph)
        out["rates_halves"] = rates_h
    out["noise_rms"] = noise_rms

    p, rates = contin(g, taus, smooth=smooth, n_iter=n_iter)
    centroid, width = spectrum_moments(p, rates)
    resid_1comp, gam, alp = fit_gamma_alpha(g, taus)
    out.update(p=p, rates=rates, centroid=centroid, width=width,
               n_peaks=count_peaks(p), resid_1comp=resid_1comp,
               gamma_1comp=gam, alpha_1comp=alp)

    reasons = []
    if n_px < 100:
        reasons.append(f"only {n_px} px — below the ~100 px floor for any multi-modal claim")
    if np.isfinite(noise_rms):
        if resid_1comp < noise_rms:
            reasons.append(f"a single (gamma={gam:.3f}, alpha={alp:.2f}) already fits to "
                           f"{resid_1comp:.4f} < noise {noise_rms:.4f}")
        if out.get("n_peaks_half1") != out.get("n_peaks_half2"):
            reasons.append(f"peak count does not reproduce across halves "
                           f"({out.get('n_peaks_half1')} vs {out.get('n_peaks_half2')})")
    out["verdict"] = ("single population / no evidence for split: " + "; ".join(reasons)
                      if reasons else
                      f"{out['n_peaks']} reproducible peak(s) beyond the 1-component fit")
    return out


if __name__ == "__main__":
    # Self-test: the analytic anchor. alpha=1 in G=1/(1+gamma tau) is EXACTLY the
    # Laplace transform of p(Gamma) = (1/gamma) exp(-Gamma/gamma), whose log10
    # width is pi/(sqrt(6) ln 10) = 0.5573 decades. If the inversion is right it
    # recovers that number, and the width then falls monotonically as alpha rises.
    taus = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
    print(f"analytic width for alpha=1: {np.pi / (np.sqrt(6) * np.log(10)):.3f} decades")
    print("\n alpha   recovered width (decades)   peaks")
    for alpha in (0.6, 0.8, 1.0, 1.2, 1.5):
        p, rates = contin(1.0 / (1.0 + 0.15 * np.asarray(taus, float) ** alpha), taus)
        _, w = spectrum_moments(p, rates)
        print(f" {alpha:5.2f}   {w:23.2f}   {count_peaks(p):5d}")

    sv, dof = laplace_dof(taus, noise=0.041)
    print(f"\nlaplace_dof at the measured 4.1% per-pixel noise: {dof} components")
    print("sigma_i/sigma_0:", " ".join(f"{s:.1e}" for s in sv[:8]))

    # A mixture is NOT visible per pixel — it re-appears as a lower alpha.
    t = np.asarray(taus, float)
    for R in (10, 100):
        gs, gf = 0.15 / np.sqrt(R), 0.15 * np.sqrt(R)
        g = 0.5 / (1 + gs * t) + 0.5 / (1 + gf * t)
        r, gam, alp = fit_gamma_alpha(g, taus)
        print(f"\ntwo populations at rate ratio {R}: best 1-comp fit "
              f"gamma={gam:.3f} alpha={alp:.2f}, residual {r:.5f} "
              f"(per-pixel noise ~0.041)")
