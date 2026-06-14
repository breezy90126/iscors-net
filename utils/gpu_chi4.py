"""
GPU four-point dynamic susceptibility χ4(τ) — feasibility probe for iSCAT.

χ4 measures DYNAMIC HETEROGENEITY: not "how fast does it relax" (the ACF) but
"is the relaxation spatially cooperative" — do clusters of pixels relax together,
how big, and on what timescale. It is the only projection that separates

  static (quenched) heterogeneity  — different regions permanently differ; no τ*
  dynamic heterogeneity            — the SAME region switches fast/slow over time;
                                     χ4(τ) PEAKS at a characteristic τ* (glass / LLPS)

Definition (field / imaging form):
  δÎ(r,t)  = z-scored intensity fluctuation per pixel, AFTER removing the per-frame
             spatial mean (common-mode global illumination flicker — otherwise it
             dominates χ4 with a spurious co-fluctuation).
  q(r,t,τ) = δÎ(r,t)·δÎ(r,t+τ)                     # per-pixel instantaneous overlap
  Q(t,τ)   = <q(r,t,τ)>_r over cell pixels          # global overlap order parameter
  χ4(τ)    = N · Var_t[Q(t,τ)]                      # four-point susceptibility

For independent pixels χ4 ~ O(1); cooperative relaxation makes Var(Q) ≫ 1/N so
χ4(τ) ≫ 1 with a peak whose height ≈ the number of dynamically-correlated pixels.

χ4 is 4th-order → very noise-sensitive. Two guards:
  - common-mode removal (per-frame spatial-mean subtraction) kills global flicker;
  - a TIME-SHUFFLE control (destroys temporal order) gives the noise/artifact floor:
    real dynamic heterogeneity ⇒ χ4_real(τ*) ≫ χ4_shuffled.

GPU: all elementwise products + reductions over cell pixels; seconds.
"""

import numpy as np
import torch


def _to_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _zscore_fluct(video, mask, remove_common_mode, device, shuffle=False, seed=0):
    """video (T,H,W) → z-scored cell-pixel fluctuations (T, Ncell)."""
    v = _to_t(video, device)
    T = v.shape[0]
    dI = v - v.mean(dim=0, keepdim=True)                 # remove static structure/speckle
    m = _to_t(mask, device).bool()
    dIc = dI[:, m]                                        # (T, Ncell)
    if remove_common_mode:
        dIc = dIc - dIc.mean(dim=1, keepdim=True)         # remove per-frame global flicker
    if shuffle:                                           # destroy temporal order (control)
        g = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(T, generator=g).to(device)
        dIc = dIc[perm]
    dIc = dIc / (dIc.std(dim=0, keepdim=True) + 1e-8)     # per-pixel z-score
    return dIc                                            # (T, Ncell)


def compute_chi4(video, taus, mask=None, min_cv=0.005, remove_common_mode=True,
                 device=None, shuffle=False, seed=0):
    """χ4(τ) = N·Var_t[Q(t,τ)] over cell pixels. Returns (taus, chi4, relax, N).

    relax(τ) = <Q(t,τ)>_t — the spatially-averaged overlap (an ACF-like relaxation
    curve), for sanity (should decay with τ).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, device)
    T, H, W = v.shape
    if mask is None:
        mean_I = v.mean(0); cv = v.std(0) / (mean_I.abs() + 1e-8)
        mask = (cv >= min_cv)
    dh = _zscore_fluct(v, mask, remove_common_mode, device, shuffle=shuffle, seed=seed)
    N = dh.shape[1]
    chi4, relax = [], []
    for tau in [int(t) for t in taus]:
        q = dh[:T - tau] * dh[tau:]                       # (T-τ, Ncell)
        Q = q.mean(dim=1)                                 # (T-τ,)
        chi4.append(float(N * Q.var(unbiased=True).item()))
        relax.append(float(Q.mean().item()))
    return np.asarray([int(t) for t in taus], float), np.asarray(chi4), np.asarray(relax), int(N)


def find_tau_star(taus, chi4):
    """Detect an interior peak (rise-then-fall). Returns (tau_star, peak, has_peak)."""
    i = int(np.argmax(chi4))
    has_peak = (0 < i < len(chi4) - 1) and (chi4[i] > chi4[0]) and (chi4[i] > chi4[-1])
    return (float(taus[i]) if has_peak else float('nan')), float(chi4[i]), bool(has_peak)


def chi4_probe(video, taus, mask=None, min_cv=0.005, device=None, verbose=True):
    """Run χ4 + the time-shuffle control and return a verdict dict."""
    t, c4, relax, N = compute_chi4(video, taus, mask=mask, min_cv=min_cv, device=device)
    _, c4s, _, _   = compute_chi4(video, taus, mask=mask, min_cv=min_cv, device=device,
                                  shuffle=True, seed=0)
    tau_star, peak, has_peak = find_tau_star(t, c4)
    # signal above the shuffle (noise/artifact) floor at the peak
    floor = float(np.median(c4s))
    snr = peak / (floor + 1e-12)
    resolved = bool(has_peak and peak > 1.5 * floor and snr > 2.0)
    out = dict(taus=t, chi4=c4, chi4_shuffled=c4s, relax=relax, N=N,
               tau_star=tau_star, peak=peak, shuffle_floor=floor, snr=snr,
               has_peak=has_peak, resolved=resolved)
    if verbose:
        tag = (f"PEAK at τ*={tau_star:.0f} (χ4={peak:.1f}, {snr:.1f}× shuffle floor) → "
               "dynamic heterogeneity RESOLVED" if resolved else
               "no significant peak above the shuffle floor → "
               "homogeneous / static / SNR-starved (χ4 not resolvable)")
        print(f"[χ4] N={N} cell px | peak χ4={peak:.1f} | shuffle floor={floor:.1f} | {tag}")
    return out


if __name__ == "__main__":
    # ── Self-test: synthesise dynamic heterogeneity → expect a χ4 peak ───────
    rng = np.random.default_rng(0)
    T, H, W = 600, 64, 64
    taus = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
    # cooperative switching: a blob's pixels share a slowly-flipping state → χ4 peak.
    state = np.sign(np.sin(np.arange(T) / 20.0))                      # global slow flip ~τ*≈30
    blob = np.zeros((H, W), np.float32); blob[20:44, 20:44] = 1.0
    vid = rng.standard_normal((T, H, W)).astype(np.float32)
    vid += (state[:, None, None] * blob[None]) * 2.0                  # blob co-fluctuates
    vid += 100.0
    out = chi4_probe(vid, taus, verbose=True)
    print("χ4(τ)        :", np.round(out['chi4'], 1))
    print("χ4 shuffled  :", np.round(out['chi4_shuffled'], 1))
    assert out['chi4'].max() > out['chi4_shuffled'].max(), "real χ4 should exceed shuffled"
    print("real χ4 > shuffled: OK ✓")
