"""iPSF-based kernel K[tau, D] for a CONTIN-style inversion of iSCORS G(tau) into P(D).

Why a new kernel
----------------
`utils/laplace_spectrum.contin` inverts with the DLS kernel exp(-Gamma tau). As
`docs/laplace_projection_alpha.md` section 1 explains, that is a kernel mismatch for
iSCAT: one camera pixel does not see a single q, it sees every spatial frequency the
iPSF contains. For a particle (or many independent particles) diffusing laterally with
coefficient D, Wiener-Khinchin gives the pixel's intensity autocovariance as

    C(tau; D) = sum_q S(q) exp(-D |q|^2 tau dt),     S(q) = |FT[iPSF (x) pixel box](q)|^2

with the q = 0 term removed (the pipeline subtracts the temporal mean). The pipeline's
'nor1' estimator is C_hat(tau)/C_hat(0), so one column of the kernel is

    K[k, j] = C(tau_k; D_j) / C(0; D_j).

Every column is a positive mixture of exponentials whose weights are fixed by the
optics, so an inversion on this kernel returns P(D) directly, with the q-weighting of
the iPSF already accounted for instead of being smeared into a fake spectral width.

The same machinery also forward-models the finite-T bias (docs section 6): with
`T` given, each column is the EXPECTED value of the sample-mean estimator rather than
the ideal ACF, so the bias lives in the model instead of in the data.

Scope / assumptions (be explicit about what this is NOT)
-------------------------------------------------------
* Lateral (x, y) diffusion only. The iPSF depends on z; `ipsf_stack` + `z_weights`
  averages over a STATIC distribution of heights (each particle keeps its z). Axial
  motion during the lag (which modulates the interference phase) is not in the kernel;
  check it against a full 3D simulation from the generator before trusting it for fast
  axial dynamics.
* Particles are independent and dilute (no interactions; number fluctuations at q=0
  removed by the mean subtraction).
* The iPSF must be sampled finely enough (camera pixel 72 nm vs lambda/2NA ~190 nm is
  fine) and the field of view large enough that the iPSF has decayed at the edges.

Numpy only. The iPSF itself comes either from the generator
(`scripts/export_ipsf_stack.py`, needs torch) or from `scalar_ipsf` below, a
self-contained scalar pupil model used for tests and when torch is not available.
"""
import numpy as np


# ───────────────────────────── iPSF sources ─────────────────────────────────
def scalar_ipsf(n_pix=128, pixel_size=72e-9, wavelength=532e-9, NA=1.4, n_medium=1.33,
                z=70e-9, ref_phase=0.0, scat_amp=0.05, geometry="reflection"):
    """Scalar-pupil iPSF contrast image (fallback when the torch generator is absent).

    E_s(x, y) = IFT[ pupil(k) * exp(i k_z z) ], pupil = disk |k| <= k0*min(NA, n_medium);
    iPSF = 2 s Re(E_s e^{-i ref_phase}) + s^2 |E_s|^2, normalised so max|E_s| = 1.
    Captures the ring structure and the z-dependent contrast flip; ignores
    polarisation, supercritical angles and aberrations (the generator has those).

    geometry: "reflection" (iSCAT; the scattered wave picks up exp(2 i k_z z)) or
              "transmission" (COBRI, the setup of the iSCORS paper; exp(i k_z z)).
    """
    k0 = 2 * np.pi / wavelength
    k_max = k0 * min(NA, n_medium)
    k = 2 * np.pi * np.fft.fftfreq(n_pix, d=pixel_size)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX ** 2 + KY ** 2
    pupil = (K2 <= k_max ** 2).astype(complex)
    kz = np.sqrt(np.clip((k0 * n_medium) ** 2 - K2, 0, None))
    n_pass = 2 if geometry == "reflection" else 1
    E = np.fft.fftshift(np.fft.ifft2(pupil * np.exp(1j * n_pass * kz * z)))
    E = E / np.abs(E).max()
    return (2 * scat_amp * np.real(E * np.exp(-1j * ref_phase))
            + scat_amp ** 2 * np.abs(E) ** 2)


def scalar_ipsf_stack(z_list, **kw):
    """(nz, H, W) stack of scalar_ipsf over heights z_list (m)."""
    return np.stack([scalar_ipsf(z=z, **kw) for z in z_list])


def roi_curve(G, c0, mask):
    """ROI-averaged normalised ACF that matches the kernel: sum(G*c0)/sum(c0).

    G : (H, W, K) per-pixel nor1 curves; c0 : (H, W) per-pixel C_hat(0).
    Averaging per-pixel ratios instead adds a Jensen bias whenever C_hat(0)
    fluctuates (see scripts/validate_ipsf_kernel.py PART 1).
    """
    G = np.asarray(G, float)
    c0 = np.asarray(c0, float)
    m = np.asarray(mask, bool) & np.isfinite(c0) & np.isfinite(G).all(axis=-1) & (c0 > 0)
    w = c0[m]
    return (G[m] * w[:, None]).sum(axis=0) / w.sum(), int(m.sum())


def best_single_d(g, K, D):
    """Single-population reference: the kernel column closest to g (least squares,
    free amplitude). Returns (D_best, rms_residual)."""
    g = np.asarray(g, float)
    amp = (K * g[:, None]).sum(axis=0) / np.maximum((K * K).sum(axis=0), 1e-30)
    res = np.sqrt(((K * amp[None, :] - g[:, None]) ** 2).mean(axis=0))
    j = int(np.argmin(res))
    return float(D[j]), float(res[j])


def load_ipsf_npz(path):
    """Load the stack written by scripts/export_ipsf_stack.py -> (ipsf[nz,H,W], z, pixel_size)."""
    d = np.load(path)
    return d["ipsf"], d["z"], float(d["pixel_size"])


# ───────────────────────────── q-space weights ──────────────────────────────
def box_transfer(n_pix, bin_factor):
    """|FT| of a bin_factor x bin_factor pixel box on the n_pix camera grid (separable)."""
    if bin_factor <= 1:
        return np.ones((n_pix, n_pix))
    u = np.pi * np.fft.fftfreq(n_pix)                    # (q * pixel) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.sin(bin_factor * u) / (bin_factor * np.sin(u))
    h[0] = 1.0
    return np.outer(h, h)


def ipsf_spectrum(ipsf, pixel_size, bin_factor=1, z_weights=None, n_q2_bins=None):
    """Power spectrum weights S over |q|^2 for the (binned) pixel signal.

    Args:
        ipsf       : (H, W) image or (nz, H, W) stack of iPSF contrast.
        pixel_size : camera pixel (m).
        bin_factor : the pipeline's BIN_FACTOR (pixel signal = mean over b x b pixels).
        z_weights  : (nz,) weights for a stack (static height distribution). None = uniform.
        n_q2_bins  : compress to this many |q|^2 bins (exact if None; 512 is plenty).

    Returns (q2, S): 1-D arrays, q = 0 removed, S >= 0. Only |q|^2 matters for
    isotropic diffusion, so an anisotropic iPSF (polarisation) is handled exactly.
    """
    stack = np.asarray(ipsf, float)
    if stack.ndim == 2:
        stack = stack[None]
    nz, H, W = stack.shape
    if H != W:
        raise ValueError("square iPSF images only")
    w = np.ones(nz) if z_weights is None else np.asarray(z_weights, float)
    w = w / w.sum()
    box2 = box_transfer(H, bin_factor) ** 2
    S = np.zeros((H, W))
    for wi, img in zip(w, stack):
        S += wi * np.abs(np.fft.fft2(img - img.mean())) ** 2 * box2
    k = 2 * np.pi * np.fft.fftfreq(H, d=pixel_size)
    q2 = (k[:, None] ** 2 + k[None, :] ** 2).ravel()
    S = S.ravel()
    keep = (q2 > 0) & (S > 0)
    q2, S = q2[keep], S[keep]
    if n_q2_bins:
        edges = np.linspace(0, q2.max() * (1 + 1e-9), n_q2_bins + 1)
        idx = np.digitize(q2, edges) - 1
        Sb = np.bincount(idx, weights=S, minlength=n_q2_bins)
        Qb = np.bincount(idx, weights=S * q2, minlength=n_q2_bins)
        m = Sb > 0
        q2, S = Qb[m] / Sb[m], Sb[m]                     # S-weighted bin centres
    return q2, S / S.sum()


def effective_q2(q2, S):
    """S-weighted mean |q|^2 -> the D*q2_eff*dt scale that sets where the decay sits."""
    return float((q2 * S).sum() / S.sum())


# ───────────────────────────── ACF columns ──────────────────────────────────
def acf_ideal(lags, D, q2, S, dt):
    """Ideal normalised ACF C(tau)/C(0) at integer `lags` for one D (C(0) = sum S = 1)."""
    lags = np.asarray(lags, float)
    return np.exp(-np.outer(lags * dt * D, q2)) @ S


def finite_t_expected(c, taus, T):
    """Expected 'nor1' estimator E[C_hat(tau)] / E[C_hat(0)] for a stationary series.

    c    : ideal autocovariance C(0..T-1) (length T).
    taus : integer lags requested.
    Matches gpu_iscors_fit.compute_g_norm_torch: sample mean over all T frames,
    C_hat(tau) = mean_{t<T-tau} dx_t dx_{t+tau}, C_hat(0) = mean_t dx_t^2.

    With a_t = Cov(x_t, xbar) = (1/T) sum_s C(|t-s|) and V = Var(xbar):
        E[C_hat(tau)] = C(tau) - (1/n) sum_{t<n} (a_t + a_{t+tau}) + V,  n = T - tau.
    """
    c = np.asarray(c, float)
    if len(c) < T:
        raise ValueError("need C at every lag 0..T-1")
    P = np.cumsum(c[:T])                                # P[m] = sum_{k<=m} C(k)
    t = np.arange(T)
    a = (P[t] + P[T - 1 - t] - c[0]) / T
    V = a.mean()
    A = np.r_[0.0, np.cumsum(a)]                        # A[m] = sum_{t<m} a_t

    def e_hat(tau):
        n = T - tau
        return c[tau] - (A[n] + (A[T] - A[tau])) / n + V

    e0 = e_hat(0)
    return np.array([e_hat(int(tau)) for tau in taus]) / e0


# ───────────────────────────── the kernel ───────────────────────────────────
def d_grid(taus, dt, q2_eff, n=80, pad=3.0):
    """Log-spaced D (m^2/s) whose decay times span (a bit beyond) the tau window."""
    taus = np.asarray(taus, float)
    lo = 1.0 / (pad * taus.max() * dt * q2_eff)
    hi = pad / (taus.min() * dt * q2_eff)
    return np.logspace(np.log10(lo), np.log10(hi), n)


def ipsf_kernel(taus, dt, q2, S, D=None, T=None, n_D=80, pad=3.0):
    """K[k, j] = expected normalised G(tau_k) for a population with diffusion D_j.

    Args:
        taus : integer lags (frames), e.g. RECON_TAUS.
        dt   : frame time (s).
        q2,S : from ipsf_spectrum.
        D    : (M,) grid in m^2/s, or None for d_grid(...).
        T    : record length in frames. None = ideal ACF; an int = the finite-T
               expectation of the pipeline's estimator (recommended; docs section 6).
    Returns (K, D) with K shape (len(taus), M).
    """
    taus = np.asarray(taus, int)
    if D is None:
        D = d_grid(taus, dt, effective_q2(q2, S), n_D, pad)
    D = np.asarray(D, float)
    K = np.empty((len(taus), len(D)))
    for j, Dj in enumerate(D):
        if T is None:
            K[:, j] = acf_ideal(taus, Dj, q2, S, dt)
        else:
            K[:, j] = finite_t_expected(acf_ideal(np.arange(T), Dj, q2, S, dt), taus, T)
    return K, D


# ───────────────────────────── inversion on a general kernel ────────────────
def invert_kernel(g, K, lam=1e-4, weights=None, g0_anchor=1.0, nugget=True,
                  n_iter=20000, tol=1e-10, return_nugget=False):
    """CONTIN-style  min ||W(K p - g)||^2 + lam ||L p||^2,  p >= 0  (FISTA).

    L is the second-difference operator on the (log-spaced) grid — CONTIN's default
    smoothness penalty. Unlike Richardson-Lucy this needs no positivity of g or K,
    which matters because the finite-T kernel goes slightly negative at long tau.

    Args:
        g       : (k,) measured normalised G at the kernel's taus.
        K       : (k, M) kernel.
        lam     : regularisation strength (relative to ||K||^2). Scan it: a peak that
                  moves or vanishes as lam changes by 10x is the regulariser, not data.
        weights : (k,) 1/sigma per tau, or None.
        g0_anchor : the 'nor1' estimator is exactly 1 at tau=0, so a tau=0 row is free
                  information (sum of all weights = 1). None = off.
        nugget  : with the anchor, add a separate non-negative weight that is 1 at
                  tau=0 and 0 at every tau>=1: the white-noise share of the variance
                  (shot/read noise, anything decorrelating within one frame). Without
                  it the 1-G(1) drop is forced into the fastest D column and shows up
                  as a fake peak at the top edge of the grid.
        return_nugget : also return the nugget weight.
    Returns p (M,) (and the nugget if asked); with the anchor sum(p)+nugget ~ 1.
    """
    g = np.asarray(g, float)
    K = np.asarray(K, float)
    M = K.shape[1]
    W = np.ones(len(g)) if weights is None else np.asarray(weights, float)
    use_nug = nugget and g0_anchor is not None
    if g0_anchor is not None:
        g = np.r_[g0_anchor, g]
        K = np.vstack([np.ones(M), K])
        W = np.r_[W.max() * 10.0, W]
    if use_nug:
        col = np.zeros((K.shape[0], 1)); col[0, 0] = 1.0
        K = np.hstack([K, col])
    A, b = K * W[:, None], g * W
    N = K.shape[1]
    L = np.zeros((M - 2, N)); L[:, :M] = np.diff(np.eye(M), 2, axis=0)   # nugget not smoothed
    data_rows = A[1:] if g0_anchor is not None else A
    scale = np.linalg.norm(data_rows, 2) ** 2           # lam relative to the data, not the anchor
    H = A.T @ A + lam * scale * (L.T @ L)
    f = A.T @ b
    step = 1.0 / np.linalg.norm(H, 2)
    p = y = np.full(N, max(g[0], 1e-3) / N)
    tk = 1.0
    for _ in range(n_iter):
        p_new = np.maximum(y - step * (H @ y - f), 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        y = p_new + ((tk - 1) / t_new) * (p_new - p)
        if np.linalg.norm(p_new - p) <= tol * (np.linalg.norm(p) + 1e-30):
            p = p_new
            break
        p, tk = p_new, t_new
    nug = float(p[M]) if use_nug else 0.0
    p = p[:M]
    return (p, nug) if return_nugget else p


def spectrum_summary(p, D, rel=0.05, edge=4):
    """Centroid / width (decades of log10 D) and peaks of P(D), with the grid edges
    reported separately.

    Mass in the `edge` bins at either end is NOT a population at that D: at the slow
    end it means "slower than the tau window can resolve" (still correlated at
    tau_max — a lower bound bucket), at the fast end "decays within ~one frame".
    Peaks are only counted in the interior; centroid/width are of the interior mass.
    """
    p = np.asarray(p, float)
    tot = p.sum()
    lg = np.log10(D)
    inner_m = np.zeros(len(p), bool); inner_m[edge:len(p) - edge] = True
    pin = np.where(inner_m, p, 0.0)
    w = pin / pin.sum() if pin.sum() > 0 else pin
    mean = float((w * lg).sum())
    width = float(np.sqrt((w * (lg - mean) ** 2).sum()))
    pp = np.r_[0.0, pin, 0.0]
    mid = pp[1:-1]
    peaks = np.where((mid > pp[:-2]) & (mid >= pp[2:]) & (mid > rel * max(pin.max(), 1e-30))
                     & inner_m)[0]
    frac = (lambda x: float(x / tot)) if tot > 0 else (lambda x: float("nan"))
    return {"centroid_log10D": mean, "centroid_D": float(10 ** mean), "width_dec": width,
            "peak_D": D[peaks],
            "peak_mass": [frac(pin[max(edge, i - 3):i + 4].sum()) for i in peaks],
            "slow_edge_frac": frac(p[:edge].sum()), "fast_edge_frac": frac(p[-edge:].sum()),
            "interior_frac": frac(pin.sum())}
