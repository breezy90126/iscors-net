"""
Phase-correlation transport probe — cheap directional-transport gate for OT/optical-flow.

iMSD (STICS) reads σ²(τ), an ISOTROPIC spread — it averages over direction, so a
COHERENT drift (all nearby pixels moving the same way, e.g. Zidovska-style chromatin
motion) gets partially cancelled in the average and, if the net displacement stays
sub-PSF over the accessible window, disappears into the noise floor entirely
(see gpu_stics.py: resolved=False for this dataset). The ACF is direction-blind too.

A transport map — real Sinkhorn OT or the cheap analogue used here — reads a VECTOR
field per patch instead of a scalar spread, so a coherent sub-PSF drift can still show
up as a reproducible, spatially-coherent displacement even when iMSD sees only floor.

This module is the CHEAP GATE: FFT phase correlation per patch (no Sinkhorn, no ground
cost / regularization tuning) standing in for a rigid-translation optimal-transport map.
It is orders of magnitude cheaper and answers exactly the question that decides whether
real OT is worth building:

  1. REPRODUCIBLE — half-split (first half vs second half in time) correlation of the
     per-patch transport magnitude. r≈0 ⇒ noise, not signal.
  2. INDEPENDENT  — transport magnitude vs CV² (density) and vs γ (ACF decay rate),
     the two established axes. High correlation ⇒ OT is not adding a new axis.
  3. COHERENT     — neighbouring patches' transport VECTORS point the same way more
     than a spatial shuffle of the same vectors would. This is what iMSD cannot see
     (it never looks at direction) and is OT's unique selling point.

Pass all three ⇒ a real Sinkhorn OT map is worth the engineering cost. Fail any one ⇒
honest degradation, same posture as gpu_stics/gpu_chi4: report the number, don't force
a positive result.

Per-patch computation mirrors the Wiener–Khinchin trick in gpu_stics.py (average the
cross-power spectrum over t, then invert once) but keeps the PHASE only (divide by
|·|) so the inverse FFT peak is a displacement estimate, not a spread.
"""

import numpy as np
import torch


def _to_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _parabolic_peak(win):
    """(2w+1,2w+1) real surface -> subpixel (dy,dx) offset from centre + peak/background."""
    h, w = win.shape
    idx = int(torch.argmax(win).item())
    py, px = divmod(idx, w)
    dy, dx = float(py - h // 2), float(px - w // 2)
    if 0 < py < h - 1:
        num = float(win[py - 1, px] - win[py + 1, px])
        den = float(win[py - 1, px] - 2 * win[py, px] + win[py + 1, px])
        if abs(den) > 1e-8:
            dy += 0.5 * num / den
    if 0 < px < w - 1:
        num = float(win[py, px - 1] - win[py, px + 1])
        den = float(win[py, px - 1] - 2 * win[py, px] + win[py, px + 1])
        if abs(den) > 1e-8:
            dx += 0.5 * num / den
    border = torch.cat([win[0], win[-1], win[:, 0], win[:, -1]])
    conf = float(win[py, px].item() - border.median().item())
    return dy, dx, conf


def compute_patch_transport(video, tau, patch=24, stride=None, mask=None,
                             max_shift=None, min_valid_frac=0.6, device=None):
    """Per-patch displacement vector at lag tau via FFT phase correlation.

    video : (T,H,W) array/tensor.
    Returns dict(dy, dx, mag, conf, valid) — each an (R,C) numpy grid (NaN where
    the patch was skipped: not enough cell-mask coverage, or fewer than 8 usable t).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, device)
    T, H, W = v.shape
    dI = v - v.mean(dim=0, keepdim=True)
    stride = stride or patch
    if max_shift is None:
        max_shift = max(2, patch // 2 - 2)
    m = _to_t(mask, device).bool() if mask is not None else None

    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    R, C = len(ys), len(xs)
    dy = np.full((R, C), np.nan, np.float32)
    dx = np.full((R, C), np.nan, np.float32)
    mag = np.full((R, C), np.nan, np.float32)
    conf = np.full((R, C), np.nan, np.float32)
    valid = np.zeros((R, C), dtype=bool)

    n = T - tau
    if n < 8:
        return dict(dy=dy, dx=dx, mag=mag, conf=conf, valid=valid,
                    ys=np.array(ys), xs=np.array(xs))
    cy = cx = patch // 2

    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            if m is not None:
                frac = m[y0:y0 + patch, x0:x0 + patch].float().mean().item()
                if frac < min_valid_frac:
                    continue
            stack = dI[:, y0:y0 + patch, x0:x0 + patch]
            F = torch.fft.fft2(stack)
            cross = (F[:n] * torch.conj(F[tau:])).mean(dim=0)
            phase = cross / cross.abs().clamp(min=1e-8)
            r = torch.fft.fftshift(torch.fft.ifft2(phase).real)
            win = r[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
            dyv, dxv, cval = _parabolic_peak(win)
            dy[i, j], dx[i, j] = dyv, dxv
            mag[i, j] = (dyv ** 2 + dxv ** 2) ** 0.5
            conf[i, j] = cval
            valid[i, j] = True

    return dict(dy=dy, dx=dx, mag=mag, conf=conf, valid=valid,
                ys=np.array(ys), xs=np.array(xs))


def _combine_taus(fields):
    """List of per-tau transport dicts -> confidence-weighted mean vector field."""
    valid = np.all([f['valid'] for f in fields], axis=0)
    w = np.clip(np.stack([f['conf'] for f in fields]), 0, None) + 1e-6
    dy = np.average(np.stack([f['dy'] for f in fields]), axis=0, weights=w) if len(fields) > 1 \
        else fields[0]['dy']
    dx = np.average(np.stack([f['dx'] for f in fields]), axis=0, weights=w) if len(fields) > 1 \
        else fields[0]['dx']
    dy = np.where(valid, dy, np.nan)
    dx = np.where(valid, dx, np.nan)
    mag = np.sqrt(dy ** 2 + dx ** 2)
    return dict(dy=dy, dx=dx, mag=mag, valid=valid,
                ys=fields[0]['ys'], xs=fields[0]['xs'])


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float('nan')
    a, b = a[ok], b[ok]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _downsample_to_patchgrid(map2d, ys, xs, patch):
    R, C = len(ys), len(xs)
    out = np.full((R, C), np.nan, np.float32)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            blk = map2d[y0:y0 + patch, x0:x0 + patch]
            blk = blk[np.isfinite(blk)]
            if blk.size > 0:
                out[i, j] = blk.mean()
    return out


def _local_coherence(dy, dx, valid, rng, n_shuffle=200):
    """Mean cosine similarity of each valid patch's vector with its 4-neighbours,
    vs. the same statistic under random spatial permutation of the vectors (null)."""
    R, C = valid.shape
    n = np.sqrt(dy ** 2 + dx ** 2)
    n = np.where(n > 1e-6, n, np.nan)
    uy, ux = dy / n, dx / n

    def _mean_cos(uy_, ux_, valid_):
        sims = []
        for i in range(R):
            for j in range(C):
                if not valid_[i, j] or not np.isfinite(uy_[i, j]):
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    i2, j2 = i + di, j + dj
                    if i2 < R and j2 < C and valid_[i2, j2] and np.isfinite(uy_[i2, j2]):
                        sims.append(uy_[i, j] * uy_[i2, j2] + ux_[i, j] * ux_[i2, j2])
        return float(np.mean(sims)) if sims else float('nan')

    obs = _mean_cos(uy, ux, valid)

    idx = np.argwhere(valid & np.isfinite(uy))
    if len(idx) < 8 or not np.isfinite(obs):
        return dict(observed=obs, shuffled_mean=float('nan'), shuffled_std=float('nan'), z=float('nan'))

    null = []
    for _ in range(n_shuffle):
        perm = rng.permutation(len(idx))
        uy_s = np.full_like(uy, np.nan)
        ux_s = np.full_like(ux, np.nan)
        for k, (i, j) in enumerate(idx):
            i2, j2 = idx[perm[k]]
            uy_s[i, j], ux_s[i, j] = uy[i2, j2], ux[i2, j2]
        null.append(_mean_cos(uy_s, ux_s, valid))
    null = np.asarray(null, float)
    mu, sd = float(np.nanmean(null)), float(np.nanstd(null)) + 1e-12
    return dict(observed=obs, shuffled_mean=mu, shuffled_std=sd, z=(obs - mu) / sd)


def ot_probe(video, taus, patch=24, stride=None, mask=None,
             density_map=None, gamma_map=None, n_shuffle=200,
             r_reproducible=0.4, r_independent=0.3, z_coherent=3.0,
             device=None, seed=0, verbose=True):
    """Cheap directional-transport feasibility gate (see module docstring).

    video       : (T,H,W).
    taus        : lags (px, in frame units) to probe; results are confidence-weighted
                  averaged across them into one transport-magnitude field.
    mask        : optional (H,W) bool cell mask (restrict patches to the cell).
    density_map, gamma_map : optional (H,W) CV² / gamma maps (e.g. from
                  utils.gpu_iscors_fit) for the independence check. Skipped if None.

    Returns a dict with the full/half-split fields plus:
      r_half        : half-split reproducibility of transport magnitude.
      r_density,r_gamma : correlation of magnitude vs the two established axes.
      coherence     : dict(observed, shuffled_mean, shuffled_std, z).
      reproducible, independent, coherent, resolved (all three) : bool.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = _to_t(video, device)
    T = v.shape[0]
    taus = [int(t) for t in taus]
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[OT probe] {T} frames, patch={patch}, taus={taus} ...")
    full_fields = [compute_patch_transport(v, t, patch, stride, mask, device=device) for t in taus]
    full = _combine_taus(full_fields)

    h = T // 2
    h1_fields = [compute_patch_transport(v[:h], t, patch, stride, mask, device=device) for t in taus]
    h2_fields = [compute_patch_transport(v[h:2 * h], t, patch, stride, mask, device=device) for t in taus]
    half1 = _combine_taus(h1_fields)
    half2 = _combine_taus(h2_fields)

    n_valid = int(full['valid'].sum())
    r_half = _pearson(half1['mag'], half2['mag'])

    r_density = float('nan')
    if density_map is not None:
        dens_grid = _downsample_to_patchgrid(np.asarray(density_map), full['ys'], full['xs'], patch)
        r_density = _pearson(full['mag'], dens_grid)
    r_gamma = float('nan')
    if gamma_map is not None:
        gamma_grid = _downsample_to_patchgrid(np.asarray(gamma_map), full['ys'], full['xs'], patch)
        r_gamma = _pearson(full['mag'], gamma_grid)

    coh = _local_coherence(full['dy'], full['dx'], full['valid'], rng, n_shuffle=n_shuffle)

    reproducible = bool(n_valid >= 16 and np.isfinite(r_half) and r_half > r_reproducible)
    ind_vals = [r for r in (r_density, r_gamma) if np.isfinite(r)]
    independent = bool(all(abs(r) < r_independent for r in ind_vals)) if ind_vals else True
    coherent = bool(np.isfinite(coh['z']) and coh['z'] > z_coherent)
    resolved = reproducible and independent and coherent

    if verbose:
        print(f"[OT probe] valid patches   : {n_valid}")
        print(f"[OT probe] reproducible    : r(half1,half2)={r_half:.3f}  "
              f"({'PASS' if reproducible else 'FAIL'}, thr={r_reproducible})")
        if ind_vals:
            print(f"[OT probe] independent     : r(vs CV²)={r_density:.3f}  r(vs γ)={r_gamma:.3f}  "
                  f"({'PASS' if independent else 'FAIL'}, thr=|r|<{r_independent})")
        print(f"[OT probe] coherent        : cos_sim={coh['observed']:.3f}  "
              f"shuffled={coh['shuffled_mean']:.3f}±{coh['shuffled_std']:.3f}  z={coh['z']:.2f}  "
              f"({'PASS' if coherent else 'FAIL'}, thr=z>{z_coherent})")
        tag = ('GO — real Sinkhorn OT worth building' if resolved else
               'NO-GO — no directional transport signal beyond CV²/γ/noise')
        print(f"[OT probe] VERDICT: {tag}")

    return dict(taus=np.array(taus), patch=patch, ys=full['ys'], xs=full['xs'],
                dy=full['dy'], dx=full['dx'], mag=full['mag'], valid=full['valid'],
                mag_half1=half1['mag'], mag_half2=half2['mag'],
                n_valid=n_valid, r_half=r_half, r_density=r_density, r_gamma=r_gamma,
                coherence=coh, reproducible=reproducible, independent=independent,
                coherent=coherent, resolved=resolved)


if __name__ == "__main__":
    # ── Self-test 1: coherent sub-patch drift (all patches share one global drift
    # direction) + noise -> expect reproducible ∧ coherent. Independence is trivially
    # true here (no CV²/γ maps passed).
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, H, W = 600, 96, 96
    base = rng.standard_normal((H, W)).astype(np.float32)
    vx, vy = 0.15, 0.08                          # px/frame coherent drift, sub-pixel-per-frame
    vid = np.empty((T, H, W), np.float32)
    from scipy.ndimage import shift as nd_shift
    for t in range(T):
        vid[t] = nd_shift(base, (vy * t, vx * t), mode='wrap') + 0.3 * rng.standard_normal((H, W))
        vid[t] += 50.0
    out = ot_probe(vid, taus=(4, 8, 16), patch=24, verbose=True)
    assert out['reproducible'] and out['coherent'], "coherent-drift synthetic should pass both gates"
    print("coherent-drift synthetic: reproducible & coherent -> OK\n")

    # ── Self-test 2: pure temporal noise (independent frames) -> expect NOT
    # reproducible and NOT coherent (honest degradation, like gpu_stics resolved=False).
    vid2 = 50.0 + 3.0 * rng.standard_normal((T, H, W)).astype(np.float32)
    out2 = ot_probe(vid2, taus=(4, 8, 16), patch=24, verbose=True)
    assert not out2['resolved'], "pure-noise synthetic must NOT pass the gate"
    print("pure-noise synthetic: correctly NOT resolved -> OK")
