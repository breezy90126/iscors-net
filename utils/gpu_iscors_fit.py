"""
GPU-batched classical iSCORS fitter (no ML).

Purpose — this is the *control* the ML pipeline was missing:
  (A) Ablation baseline. Fits (γ, α) per pixel with the SAME objective the
      network minimises (g0_norm target 1/(1+γτ^α), Fisher × reliability τ-weights),
      but each pixel INDEPENDENTLY — no U-Net, no spatial pooling. Comparing its
      γ/α/R² maps against the model isolates exactly what the spatial network adds.
  (B) Fast dense quasi-GT. Replaces the per-pixel scipy.optimize.curve_fit loop
      (minutes, with convergence failures) used for the checkerboard gridA reference
      with a single batched GPU optimisation over all cell pixels (seconds, no failures).

Relation to MATLAB correlation.m
  correlation.m computes the raw per-lag correlation with a naive O(N²) double loop
  and offers two normalisations:
      nor_1: CorrF / CorrF(0)            (zero-lag)
      nor_2: CorrF / (mean(x)·mean(y))   (standard FCS 1 + C/⟨I⟩²)
  Fed fluctuations δI = I − ⟨I⟩, `correlation(δI, δI, 'nor_1')` = C(τ)/C(0) — exactly
  this module's `norm='nor1'` g0_norm target (1 at τ→0, →0 at large τ), which is what
  the network is trained against. We compute the same quantity vectorised (the O(N²)
  loop is unnecessary). `norm='nor2'` reproduces the raw 1+C/⟨I⟩² convention for the
  rare case the external GT used it; it does not affect the ablation / quasi-GT uses,
  which are internally self-consistent.

The fit uses the same activations as models/pissl_tau_encoder.py so the parameter
ranges match the network exactly:
    γ = gamma_scale · sigmoid(θ_γ) ∈ (0, gamma_scale)
    α = 2 · sigmoid(θ_α)           ∈ (0, 2)
and for n_components=2 the shared-α two-rate mixture with γ_fast = γ_slow + softplus(Δ).
"""

import numpy as np
import torch


def _to_t(x, dtype, device):
    """To a tensor on `device`, whether x is numpy or a (possibly CUDA) torch tensor.

    np.asarray() on a CUDA tensor raises ('can't convert cuda tensor to numpy'),
    so torch tensors must be moved with .to() rather than re-wrapped via numpy.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)



# ───────────────────────── condensation (V_DLS / D) ────────────────────────
def condensation_projection(v_dls, inv_Dstar, cell_mask, slope=3.0, b=None):
    """iSCORS condensation via the slope-3 log-log perpendicular projection.

    Per pixel:  X = log10(1/D*) (slowness),  Y = log10(V_DLS) (CV² fluctuation energy).
    Baseline (pure-density scaling):  Y = slope·X + b   with slope FIXED = 3.
      b = total mass density — fixed-slope least-squares intercept over the cell:
          b = mean(Y - slope·X).
    Condensation level = position along the baseline = X_p of the perpendicular foot:
          X_p = (X + slope·Y - slope·b) / (1 + slope²)        # = (X+3Y-3b)/10 for slope=3
          Y_p = slope·X_p + b
    Projecting removes the off-line (density/noise) scatter, leaving condensation.

    Args:
        v_dls     : (H,W) CV² = G(0) map (= compute_density).
        inv_Dstar : (H,W) 1/D* map (∝ 1/γ from the ACF fit).
        cell_mask : (H,W) bool.
    Returns (condensation X_p map with NaN outside cell, info dict with b, Xp, Yp, X, Y).
    """
    eps = 1e-12
    X = np.log10(np.clip(np.asarray(inv_Dstar, np.float64), eps, None))
    Y = np.log10(np.clip(np.asarray(v_dls,     np.float64), eps, None))
    m = np.asarray(cell_mask, bool) & np.isfinite(X) & np.isfinite(Y)
    if b is None:
        b = float(np.mean(Y[m] - slope * X[m]))             # fixed-slope intercept = density
    denom = 1.0 + slope ** 2                                 # = 10 for slope = 3
    Xp = (X + slope * Y - slope * b) / denom
    Yp = slope * Xp + b
    cond = Xp.astype(np.float32).copy(); cond[~np.asarray(cell_mask, bool)] = np.nan
    return cond, dict(b=float(b), slope=float(slope), Xp=Xp, Yp=Yp, X=X, Y=Y)


def condensation(density, gamma, alpha=None, blur='gamma'):
    """iSCORS condensation map = CV² / Φ(D*)  (the 'V_DLS / D' quantity).

    density : CV² = G(0) map (from compute_density) — illumination-corrected total
              fluctuation energy, but suppressed by motion.
    gamma   : per-pixel ACF decay rate (∝ D*); the motion-blur factor.
    Φ(D*)   : blur='gamma' → Φ = γ            (γ ∝ D; the standard V_DLS/D form)
              blur='tauD'  → Φ = γ^(1/α) = 1/τ_D  (anomalous-corrected characteristic rate)
    Dividing un-blurs the variance → dense+slow ⇒ high, dilute+fast ⇒ low.
    """
    g = np.asarray(gamma, dtype=np.float64)
    if blur == 'tauD' and alpha is not None:
        a = np.clip(np.asarray(alpha, np.float64), 0.1, 2.0)
        phi = np.power(np.clip(g, 1e-6, None), 1.0 / a)              # γ^(1/α) = 1/τ_D
    else:
        phi = np.clip(g, 1e-6, None)                                # Φ = γ ∝ D
    return (np.asarray(density, np.float64) / (phi + 1e-10)).astype(np.float32)


# ───────────────────────── density channel (amplitude) ─────────────────────
def compute_density(video, min_cv=0.005, device=None):
    """Robust 'how much / how many' map: G(0) = CV² = Var_t(I)/⟨I⟩² per pixel.

    This is the amplitude/density channel iSCORS *normalises away*. It needs no
    curve fit, is high-SNR (a variance), and is the clean condensation-like map
    (≈ the iSCORS MATLAB Cond_map). Complementary to the dynamics (γ,α): density
    is "how much", γ is "how fast". Returns (density CV² map, cell_mask).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, torch.float32, device)
    mean_I = v.mean(dim=0)
    var_I  = ((v - mean_I) ** 2).mean(dim=0)
    cv2 = (var_I / (mean_I ** 2 + 1e-10))                # CV² = G(0)
    cell = (torch.sqrt(var_I.clamp(min=0)) / (mean_I.abs() + 1e-10)) >= min_cv
    dens = cv2.cpu().numpy(); dens[~cell.cpu().numpy()] = np.nan
    return dens, cell.cpu().numpy()


# ───────────────────────── G_norm computation ──────────────────────────────
def compute_g_norm_torch(video, recon_taus, norm='nor1', min_cv=0.005, device=None):
    """Vectorised C(τ)/C(0) (≡ correlation(δI,δI,'nor1')) for every pixel on GPU.

    Args:
        video:      (T, H, W) array (numpy or torch).
        recon_taus: iterable of integer τ lags.
        norm:       'nor1' → C(τ)/C(0) (g0_norm, matches the network target);
                    'nor2' → 1 + C(τ)/⟨I⟩² (raw FCS convention).
        min_cv:     CV threshold for the cell mask.

    Returns:
        g_norm    : (H, W, K) float32 torch tensor (0 outside the cell mask).
        cell_mask : (H, W) bool torch tensor.
        g_zero    : (H, W) float32 — C(0)=Var (diagnostics / σ normalisation).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, torch.float32, device)
    T, H, W = v.shape
    taus = [int(t) for t in recon_taus]
    eps = 1e-10

    mean_I = v.mean(dim=0)                                   # (H, W)
    delta  = v - mean_I                                      # (T, H, W)
    c0     = (delta ** 2).mean(dim=0)                        # (H, W) = Var = C(0)
    cv_map = torch.sqrt(c0.clamp(min=0)) / (mean_I.abs() + eps)
    cell_mask = cv_map >= min_cv

    g = torch.zeros((H, W, len(taus)), dtype=torch.float32, device=device)
    for i, tau in enumerate(taus):
        n = T - tau
        c_tau = (delta[:n] * delta[tau:]).mean(dim=0)        # (H, W) = C(τ)
        if norm == 'nor2':
            g[:, :, i] = 1.0 + c_tau / (mean_I ** 2 + eps)
        else:                                                # 'nor1' = g0_norm
            g[:, :, i] = c_tau / (c0 + eps)

    g[~cell_mask] = 0.0
    return g, cell_mask, c0


# ───────────────────────── τ weighting (matches the loss) ───────────────────
def _fisher_weights_g0(taus_t, gamma_0=0.15, alpha_0=0.8):
    """Fisher information of G_norm(τ)=1/(1+γτ^α) at a prior (γ₀,α₀).

    Identical formula to PhysicsReconLoss._fisher_weights_g0 (g0_norm mode):
        ∂G/∂γ = -τ^α/(1+γτ^α)² ,  ∂G/∂α = -γτ^α·log τ/(1+γτ^α)²
    Returns weights summing to 1 over τ.
    """
    t_a = taus_t.clamp(min=1e-8) ** alpha_0
    d2  = (1.0 + gamma_0 * t_a) ** 2
    log_t = torch.log(taus_t.clamp(min=1.0))
    fisher = (t_a / d2) ** 2 + (gamma_0 * t_a * log_t / d2) ** 2
    return fisher / (fisher.sum() + 1e-10)


# ───────────────────────── batched non-linear fit ──────────────────────────
def fit_gamma_alpha_batched(g_norm, cell_mask, recon_taus,
                            n_components=1, gamma_scale=2.0, fix_alpha=False,
                            global_alpha=False,
                            weight='fisher', sigma_g_norm=None,
                            gamma0=0.15, alpha0=0.8,
                            n_steps=500, lr=0.05, lambda_occam=0.02,
                            device=None, verbose=True):
    """Fit (γ, α) — or the 2-component mixture — to every cell pixel at once (Adam).

    Same objective as the network: weighted MSE of g_theory vs the g0_norm target,
    with the same scaled-sigmoid activations. Each pixel is independent (no spatial
    coupling) — that is the whole point of the baseline.

    Args:
        g_norm:       (H, W, K) g0_norm curves (from compute_g_norm_torch, norm='nor1').
        cell_mask:    (H, W) bool.
        n_components: 1 → [γ,α];  2 → shared-α two-rate mixture (effective γ returned).
        weight:       'fisher' | 'uniform'. With sigma_g_norm given, multiplied by 1/σ.
        sigma_g_norm: (H, W, K) optional per-τ reliability (down-weights noisy lags).
        lambda_occam: parsimony penalty (2-comp only) min(f,1-f)·(γ_fast-γ_slow).

    Returns dict of (H, W) float32 numpy maps: gamma, alpha, r2 (+ f, g_slow, g_fast
    when n_components=2); non-cell pixels are NaN.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_norm = _to_t(g_norm, torch.float32, device)
    cm     = _to_t(cell_mask, torch.bool, device)
    taus_t = torch.tensor([int(t) for t in recon_taus], dtype=torch.float32, device=device)
    K, eps = taus_t.numel(), 1e-10

    gpix = g_norm[cm]                                        # (N, K)
    N = gpix.shape[0]
    if N == 0:
        raise ValueError("no cell pixels to fit")

    # τ weights (1,K) or per-pixel (N,K) when reliability is supplied
    if weight == 'fisher':
        w = _fisher_weights_g0(taus_t, gamma0, alpha0).view(1, K)
    else:
        w = torch.full((1, K), 1.0 / K, device=device)
    if sigma_g_norm is not None:
        s = _to_t(sigma_g_norm, torch.float32, device)[cm]  # (N,K)
        w = w * (1.0 / (s + eps))
    w = w / (w.sum(dim=-1, keepdim=True) + eps)             # normalise over τ

    def _logit(p):                                          # inverse of sigmoid
        p = min(max(p, 1e-4), 1 - 1e-4)
        return float(np.log(p / (1 - p)))

    # ---- parameters (initialised at the prior) ----
    if n_components == 1:
        tg = torch.full((N,), _logit(gamma0 / gamma_scale), device=device, requires_grad=True)
        # α: per-pixel (free) | one shared scalar (global_alpha) | none (fix_alpha → α≡1)
        if fix_alpha:
            ta = None
        elif global_alpha:
            ta = torch.full((1,), _logit(alpha0 / 2.0), device=device, requires_grad=True)
        else:
            ta = torch.full((N,), _logit(alpha0 / 2.0), device=device, requires_grad=True)
        params = [tg] if ta is None else [tg, ta]
    else:
        tf  = torch.zeros(N, device=device, requires_grad=True)                       # f≈0.5
        tgs = torch.full((N,), _logit((gamma0 * 0.5) / gamma_scale), device=device, requires_grad=True)
        tgd = torch.zeros(N, device=device, requires_grad=True)                        # Δ via softplus
        # α: per-pixel (free), or ONE shared scalar (global_alpha), or unused (fix_alpha)
        ta_n = 1 if global_alpha else N
        ta  = torch.full((ta_n,), _logit(alpha0 / 2.0), device=device, requires_grad=True)
        params = [tf, tgs, tgd, ta]

    opt = torch.optim.Adam(params, lr=lr)
    sp  = torch.nn.functional.softplus

    def _forward():
        if n_components == 1:
            gamma = gamma_scale * torch.sigmoid(tg)         # (N,)
            if fix_alpha:
                alpha = torch.ones_like(gamma)               # α≡1 (1-param)
            elif global_alpha:
                alpha = (2.0 * torch.sigmoid(ta)).expand_as(gamma)  # one shared scalar
            else:
                alpha = 2.0 * torch.sigmoid(ta)              # per-pixel
            g_th  = 1.0 / (1.0 + gamma[:, None] * taus_t[None, :] ** alpha[:, None])
            extras = {}
        else:
            f      = torch.sigmoid(tf)
            g_slow = gamma_scale * torch.sigmoid(tgs)
            g_fast = g_slow + sp(tgd)
            if fix_alpha:
                alpha = torch.ones_like(g_slow)
            elif global_alpha:
                alpha = (2.0 * torch.sigmoid(ta)).expand_as(g_slow)     # shared scalar
            else:
                alpha = 2.0 * torch.sigmoid(ta)                          # per-pixel
            ta_    = taus_t[None, :] ** alpha[:, None]
            g_th   = (f[:, None] / (1.0 + g_fast[:, None] * ta_)
                      + (1 - f)[:, None] / (1.0 + g_slow[:, None] * ta_))
            gamma  = f * g_fast + (1 - f) * g_slow           # effective γ
            extras = {'f': f, 'g_slow': g_slow, 'g_fast': g_fast}
        return g_th, gamma, alpha, extras

    for step in range(n_steps):
        opt.zero_grad()
        g_th, gamma, alpha, extras = _forward()
        loss = ((g_th - gpix) ** 2 * w).sum(dim=1).mean()
        if n_components == 2 and lambda_occam > 0:
            loss = loss + lambda_occam * (torch.minimum(extras['f'], 1 - extras['f'])
                                          * (extras['g_fast'] - extras['g_slow'])).mean()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % max(1, n_steps // 5) == 0:
            print(f"  [gpu-fit] step {step+1}/{n_steps}  loss={loss.item():.4e}")

    # ---- final maps + per-pixel R² ----
    with torch.no_grad():
        g_th, gamma, alpha, extras = _forward()
        ss_res = ((g_th - gpix) ** 2).sum(dim=1)
        ss_tot = ((gpix - gpix.mean(dim=1, keepdim=True)) ** 2).sum(dim=1) + eps
        r2 = 1.0 - ss_res / ss_tot

    H, W = cm.shape
    def _scatter(vec):
        m = torch.full((H, W), float('nan'), device=device)
        m[cm] = vec
        return m.cpu().numpy()

    out = {'gamma': _scatter(gamma), 'alpha': _scatter(alpha),
           'r2': _scatter(r2), 'cell_mask': cm.cpu().numpy()}
    if n_components == 2:
        out.update(f=_scatter(extras['f']), g_slow=_scatter(extras['g_slow']),
                   g_fast=_scatter(extras['g_fast']))
    return out


# ───────────────────────── one-call entry point ────────────────────────────
def gpu_fit_maps(video, recon_taus, n_components=1, gamma_scale=2.0, fix_alpha=False,
                 global_alpha=False, min_cv=0.005, weight='fisher', gamma0=0.15, alpha0=0.8,
                 n_steps=500, lr=0.05, device=None, verbose=True):
    """video (T,H,W) → dense classical (γ, α, R²) maps via GPU-batched fitting.

    Convenience wrapper: compute_g_norm_torch (nor1/g0_norm) → fit_gamma_alpha_batched.
    Returns the dict from fit_gamma_alpha_batched.
    """
    g_norm, cell_mask, _ = compute_g_norm_torch(
        video, recon_taus, norm='nor1', min_cv=min_cv, device=device)
    if verbose:
        print(f"[gpu-fit] cell pixels: {int(cell_mask.sum())}/{cell_mask.numel()}  "
              f"n_components={n_components}  steps={n_steps}")
    return fit_gamma_alpha_batched(
        g_norm, cell_mask, recon_taus, n_components=n_components,
        gamma_scale=gamma_scale, fix_alpha=fix_alpha, global_alpha=global_alpha,
        weight=weight, gamma0=gamma0, alpha0=alpha0,
        n_steps=n_steps, lr=lr, device=device, verbose=verbose)


if __name__ == "__main__":
    # ── Self-test: recover known (γ, α) from synthetic g0_norm curves ────────
    torch.manual_seed(0)
    taus = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
    taus_t = torch.tensor(taus, dtype=torch.float32)

    # three "regions" with known parameters → build C(τ)/C(0)=1/(1+γτ^α) + noise
    true = torch.tensor([[0.10, 1.0], [0.50, 1.5], [0.05, 0.5]])
    reps = 400
    gamma_true = true[:, 0].repeat_interleave(reps)
    alpha_true = true[:, 1].repeat_interleave(reps)
    g_clean = 1.0 / (1.0 + gamma_true[:, None] * taus_t[None, :] ** alpha_true[:, None])
    g_noisy = (g_clean + 0.02 * torch.randn_like(g_clean)).clamp(min=1e-4)

    # pack into a fake (H,W,K) with all pixels = cell
    NP = g_noisy.shape[0]
    g_map = g_noisy.view(NP, 1, len(taus)).permute(1, 0, 2).contiguous()   # (1, NP, K)
    cell  = torch.ones((1, NP), dtype=torch.bool)

    out = fit_gamma_alpha_batched(g_map, cell, taus, n_components=1,
                                  n_steps=600, lr=0.05, verbose=True)
    gfit = torch.tensor(out['gamma'][0]); afit = torch.tensor(out['alpha'][0])
    print("\n=== recovery (n_components=1) ===")
    for r, (gt, at) in enumerate(true.tolist()):
        sl = slice(r * reps, (r + 1) * reps)
        print(f"  region γ={gt:.2f} α={at:.2f}  →  "
              f"γ̂={gfit[sl].mean():.3f}±{gfit[sl].std():.3f}  "
              f"α̂={afit[sl].mean():.3f}±{afit[sl].std():.3f}")
    print(f"  γ MAE={ (gfit-gamma_true).abs().mean():.4f}   "
          f"α MAE={ (afit-alpha_true).abs().mean():.4f}   "
          f"R² mean={np.nanmean(out['r2']):.3f}")
