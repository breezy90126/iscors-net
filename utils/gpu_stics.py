"""
GPU iMSD (image Mean Square Displacement) — independent α verifier for the ACF line.

The per-pixel ACF (TICS, what iSCORS-net does) cannot separate genuine anomalous
diffusion (α<1) from a distribution of normal rates (σ_D): both stretch the temporal
decay the same way. iMSD breaks that degeneracy because it reads α from a DIFFERENT
projection — how fast the signal spreads in SPACE as a function of lag τ:

    STICS(Δ, τ) = <δI(r,t)·δI(r+Δ, t+τ)>_{r,t}          (spatially-averaged)
    σ²_space(τ) = 2nd moment of the central peak of STICS(·, τ)
    σ²_space(τ) = σ²_PSF + Γ·τ^α                          (iMSD fit)

α here is the genuine MSD exponent, independent of the ACF α/σ_D degeneracy:
  α_iMSD ≈ α_ACF        → the anomaly is real (two projections agree)
  α_iMSD ≈ 1, α_ACF<1   → the ACF α was a heterogeneity (σ_D) leak, not real anomaly

Honest degradation: if the diffusion spread does not exceed the PSF over the accessible
τ range, σ²(τ) is floor-dominated and α is unresolvable — the fit flags `resolved=False`
rather than returning a spurious number.

Pre-processing matches the rest of the pipeline: δI = video − mean_t (this also removes
static iSCAT speckle, which is constant in time and lands entirely in the temporal mean).

Computation is GPU-batched via spatial FFT (Wiener–Khinchin in 2D), all frames at once —
seconds-to-minutes, the same scale as the ACF GPU fitter. Only a GLOBAL (cell-averaged)
α is needed to verify the model's global α, so no per-pixel STICS is required.
"""

import numpy as np
import torch


def _to_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def compute_stics_imsd(video, taus, max_shift=12, mask=None, device=None, verbose=True):
    """Spatially-averaged STICS width σ²(τ) for each lag τ (the iMSD curve).

    Args:
        video:     (T, H, W) array (numpy or torch).
        taus:      iterable of integer lags τ.
        max_shift: half-size (px) of the central window the 2nd moment is taken over.
        mask:      optional (H, W) bool — zero δI outside it (restrict to the cell).

    Returns:
        taus  : (K,) float
        sig2  : (K,) float — σ²_space(τ) in px² (2nd moment of the central STICS peak).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, device)
    T, H, W = v.shape
    dI = v - v.mean(dim=0, keepdim=True)                     # remove static structure/speckle
    if mask is not None:
        m = _to_t(mask, device)
        dI = dI * m.unsqueeze(0)

    if verbose:
        print(f"[iMSD] FFT of {T} frames ({H}×{W}) ...")
    F = torch.fft.fft2(dI)                                   # (T,H,W) complex

    # coordinate grid for the 2nd moment over the central window
    ax = torch.arange(-max_shift, max_shift + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    rr = yy ** 2 + xx ** 2                                   # |Δ|²  (w,w)
    cy, cx = H // 2, W // 2
    eps = 1e-10

    sig2 = []
    for tau in [int(t) for t in taus]:
        n = T - tau
        cross = (F[:n] * torch.conj(F[tau:])).mean(dim=0)   # (H,W) complex
        r = torch.fft.fftshift(torch.fft.ifft2(cross).real) # centred real correlation
        win = r[cy - max_shift:cy + max_shift + 1,
                cx - max_shift:cx + max_shift + 1]           # (w,w)
        border = torch.cat([win[0], win[-1], win[:, 0], win[:, -1]])
        rpos = (win - border.median()).clamp(min=0)          # baseline-subtracted peak
        s2 = (rr * rpos).sum() / (rpos.sum() + eps)
        sig2.append(float(s2.item()))

    return np.asarray([int(t) for t in taus], dtype=float), np.asarray(sig2, dtype=float)


def fit_imsd_alpha(taus, sig2):
    """Fit σ²(τ) = σ0² + Γ·τ^α and decide if α is resolvable above the PSF floor.

    Returns dict(alpha, Gamma, sigma0_2, r2, resolved, growth_ratio).
      resolved=False ⇒ the spread never exceeds ~30% of the floor over the τ range,
      so α is PSF-dominated and should NOT be trusted (honest degradation).
    """
    from scipy.optimize import curve_fit
    taus = np.asarray(taus, float); sig2 = np.asarray(sig2, float)

    def _m(t, s0, g, a):
        return s0 + g * t ** a

    p0 = [max(sig2.min(), 1e-6), max(sig2.max() - sig2.min(), 1e-3), 1.0]
    try:
        popt, _ = curve_fit(_m, taus, sig2, p0=p0,
                            bounds=([0.0, 0.0, 0.1], [np.inf, np.inf, 2.0]), maxfev=10000)
        s0, g, a = [float(x) for x in popt]
        pred = _m(taus, *popt)
        ss_res = float(((sig2 - pred) ** 2).sum())
        ss_tot = float(((sig2 - sig2.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    except Exception as e:
        return dict(alpha=float('nan'), Gamma=float('nan'), sigma0_2=float('nan'),
                    r2=float('nan'), resolved=False, growth_ratio=0.0, error=str(e))

    growth = g * taus.max() ** a                             # spread gained over the τ range
    growth_ratio = growth / (s0 + 1e-12)
    return dict(alpha=a, Gamma=g, sigma0_2=s0, r2=r2,
                resolved=bool(growth_ratio > 0.3), growth_ratio=float(growth_ratio))


def stics_alpha(video, taus, max_shift=12, mask=None, device=None, verbose=True):
    """video (T,H,W) → independent global α via iMSD. Returns the fit dict + the curve."""
    t, s2 = compute_stics_imsd(video, taus, max_shift=max_shift, mask=mask,
                               device=device, verbose=verbose)
    out = fit_imsd_alpha(t, s2)
    out.update(taus=t, sigma2=s2)
    if verbose:
        tag = 'RESOLVED' if out['resolved'] else 'PSF-DOMINATED (α unreliable)'
        print(f"[iMSD] α = {out['alpha']:.3f}  (R²={out['r2']:.3f}, "
              f"σ0²={out['sigma0_2']:.2f}px², growth×{out['growth_ratio']:.2f}) → {tag}")
    return out


if __name__ == "__main__":
    # ── Self-test: a field that spreads as σ²∝τ (normal diffusion) → expect α≈1 ──
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, H, W = 400, 96, 96
    taus = (1, 2, 4, 8, 16, 32, 48, 64)
    # build frames by progressively Gaussian-blurring a static random field over time,
    # so the spatial correlation width grows ~linearly with frame index (α≈1 diffusion).
    from scipy.ndimage import gaussian_filter
    base = rng.standard_normal((H, W)).astype(np.float32)
    vid = np.empty((T, H, W), np.float32)
    for t in range(T):
        vid[t] = gaussian_filter(base, sigma=0.3 + 0.05 * t) + 0.05 * rng.standard_normal((H, W))
        vid[t] += 100.0
    out = stics_alpha(vid, taus, max_shift=15, verbose=True)
    print(f"\nσ²(τ): {np.round(out['sigma2'], 2)}")
    print(f"recovered α = {out['alpha']:.3f}  (synthetic spreading ⇒ should be > 0, monotone σ²)")
    assert np.all(np.diff(out['sigma2']) >= -1e-6), "σ²(τ) should be non-decreasing"
    print("σ²(τ) monotone non-decreasing: OK ✓")
