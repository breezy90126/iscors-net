"""Finite-T bias correction for per-pixel g_norm curves (docs/laplace_projection_alpha.md 6.4, item 3).

The pipeline estimator ('nor1', compute_g_norm_torch) subtracts the SAMPLE mean and divides
by C_hat(0). Over a record of T frames this depresses the slow tail by roughly
2 tau_int C(0) / T -- more for slow pixels, so the bias is gamma-dependent (D4).

Model (same family as the deliverable, plus an explicit white-noise share):
    C(0) = 1,  C(k >= 1) = A / (1 + gamma k^alpha)        (1 - A = white-noise share)
The exact expected estimator is linear in C before the ratio, so
    E[C_hat(tau)] = (1 - A) (delta_tau - 1/T) + A e_f(tau; gamma, alpha, T)
where e_f is utils.ipsf_kernel.finite_t_expected's numerator for the unit-amplitude
algebraic part. We tabulate e_f on a (alpha, gamma) grid once, then:

    fit (A, gamma, alpha) -> bias = E[g_hat] - g_model -> g_corr = g - bias -> refit

and iterate. Because the bias is computed for the actual T, the full record (T) and the
halves (T/2) should agree after correction -- that is the self-test D4 runs.

Numpy only. Fitting is a vectorised grid search: A is solved in closed form per (pixel,
gamma); gamma on a log grid with parabolic refinement; alpha shared (global) or per pixel.
"""
import numpy as np


# ───────────────────────── expected estimator, unnormalised ─────────────────────────
def _expected_unnorm(c, taus, T):
    """E[C_hat(tau)] (NOT divided by E[C_hat(0)]) for tau in [0] + taus."""
    P = np.cumsum(c[:T])
    t = np.arange(T)
    a = (P[t] + P[T - 1 - t] - c[0]) / T
    V = a.mean()
    A = np.r_[0.0, np.cumsum(a)]
    out = []
    for tau in (0,) + tuple(int(x) for x in taus):
        n = T - tau
        out.append(c[tau] - (A[n] + (A[T] - A[tau])) / n + V)
    return np.array(out)


def _algebraic(k, gamma, alpha):
    k = np.asarray(k, float)
    return 1.0 / (1.0 + gamma * k ** alpha)


class BiasTable:
    """e_f(tau) on a (alpha, gamma) grid for one record length T."""

    def __init__(self, taus, T, alphas, gammas):
        self.taus = np.asarray(taus, int)
        self.T = int(T)
        self.alphas = np.atleast_1d(np.asarray(alphas, float))
        self.gammas = np.asarray(gammas, float)
        k = np.arange(self.T, dtype=float)
        K = len(self.taus)
        self.f = np.empty((len(self.alphas), len(self.gammas), K))     # ideal shape at taus
        self.e = np.empty((len(self.alphas), len(self.gammas), K + 1))  # E[C_hat], tau=0 first
        for i, a in enumerate(self.alphas):
            ka = k ** a
            for j, g in enumerate(self.gammas):
                c = 1.0 / (1.0 + g * ka)
                c[0] = 1.0
                self.e[i, j] = _expected_unnorm(c, self.taus, self.T)
                self.f[i, j] = c[self.taus]

    def expected_ratio(self, ia, ig, A):
        """E[g_hat(taus)] for pixels with grid indices (ia, ig) and amplitude A."""
        T = self.T
        e = self.e[ia, ig]                                  # (N, K+1)
        A = np.asarray(A, float)[:, None]
        num = -(1.0 - A) / T + A * e[:, 1:]
        den = (1.0 - A[:, 0]) * (1.0 - 1.0 / T) + A[:, 0] * e[:, 0]
        return num / den[:, None]


# ───────────────────────────── vectorised fit ─────────────────────────────
def fit_grid(g, f_table, w=None):
    """Least-squares A/(1+gamma tau^alpha) over a table of shapes.

    g       : (N, K) curves.
    f_table : (M, K) candidate shapes (one alpha, M gammas).
    Returns best index (N,), A (N,), rms (N,), and the full rms (N, M) for refinement.
    """
    g = np.asarray(g, float)
    F = np.asarray(f_table, float)
    if w is None:
        w = np.ones(g.shape[1])
    gw = g * w; Fw = F * w
    num = gw @ Fw.T                                         # (N, M)
    den = (Fw * Fw).sum(1)                                  # (M,)
    A = np.clip(num / den, 0.0, 1.0)
    # rms = ||g - A f||^2 / K, expanded
    gg = (gw * gw).sum(1)[:, None]
    sse = gg - 2 * A * num + A ** 2 * den
    rms = np.sqrt(np.maximum(sse, 0) / g.shape[1])
    j = np.argmin(rms, axis=1)
    n = np.arange(len(g))
    return j, A[n, j], rms[n, j], rms


def _refine_log(grid, j, rms_row_vals):
    """Parabolic refinement of the argmin on a log grid; returns refined value."""
    lg = np.log(grid)
    j = np.clip(j, 1, len(grid) - 2)
    y0, y1, y2 = rms_row_vals
    d = (y0 - 2 * y1 + y2)
    off = np.where(np.abs(d) > 1e-30, 0.5 * (y0 - y2) / np.where(d == 0, 1, d), 0.0)
    off = np.clip(off, -1, 1)
    step = lg[1] - lg[0]
    return np.exp(lg[j] + off * step)


def fit_curves(g, table, global_alpha=True, w=None):
    """Fit every curve in g (N, K). Returns dict(gamma, A, alpha, ia, ig, rms)."""
    g = np.asarray(g, float)
    N = len(g)
    best = None
    per = []
    for ia in range(len(table.alphas)):
        j, A, r, rr = fit_grid(g, table.f[ia], w)
        per.append((j, A, r, rr))
    if global_alpha:
        tot = [np.nanmean(p[2] ** 2) for p in per]
        ia = int(np.argmin(tot))
        j, A, r, rr = per[ia]
        ia_arr = np.full(N, ia)
    else:
        R = np.stack([p[2] for p in per])                  # (n_alpha, N)
        ia_arr = np.argmin(R, axis=0)
        n = np.arange(N)
        j = np.stack([p[0] for p in per])[ia_arr, n]
        A = np.stack([p[1] for p in per])[ia_arr, n]
        r = R[ia_arr, n]
        rr = None
    n = np.arange(N)
    if rr is not None:
        jc = np.clip(j, 1, len(table.gammas) - 2)
        gam = _refine_log(table.gammas, j, (rr[n, jc - 1], rr[n, jc], rr[n, jc + 1]))
        gam = np.where((j == 0) | (j == len(table.gammas) - 1), table.gammas[j], gam)
    else:
        gam = table.gammas[j]
    return dict(gamma=gam, A=A, alpha=table.alphas[ia_arr], ia=ia_arr, ig=j, rms=r)


def model_table(table, A_grid):
    """Expected-estimator curves r[alpha, gamma, A, tau] (forward model incl. finite T)."""
    T = table.T
    b = 1.0 / T
    e = table.e                                              # (na, ng, K+1)
    A = np.asarray(A_grid, float)[None, None, :, None]
    num = A * (e[:, :, None, 1:] + b) - b
    den = (1.0 - b) + A[..., 0] * (e[:, :, None, 0] - 1.0 + b)
    return num / den[..., None]


def _fit_one_alpha(g, R, chunk=4096):
    """g (N,K); R (ng, nA, K). Returns best (ig, iA), sse (N,), sse along gamma at best A."""
    ng, nA, K = R.shape
    Rf = R.reshape(ng * nA, K)
    RR = (Rf * Rf).sum(1)
    N = len(g)
    ig = np.empty(N, int); iA = np.empty(N, int); sse = np.empty(N)
    prof = np.empty((N, ng))
    for s in range(0, N, chunk):
        gg = g[s:s + chunk]
        E = (gg * gg).sum(1)[:, None] - 2 * gg @ Rf.T + RR[None]   # (n, ng*nA)
        E = E.reshape(len(gg), ng, nA)
        best_A = E.argmin(2)                                       # (n, ng)
        Eg = np.take_along_axis(E, best_A[..., None], 2)[..., 0]  # (n, ng)
        j = Eg.argmin(1)
        n = np.arange(len(gg))
        ig[s:s + chunk] = j; iA[s:s + chunk] = best_A[n, j]
        sse[s:s + chunk] = Eg[n, j]; prof[s:s + chunk] = Eg
    return ig, iA, np.maximum(sse, 0), prof


def fit_expected(g, table, A_grid=None, global_alpha=True, alpha_subset=3000, seed=0):
    """Forward-model fit: g ~ E[g_hat | A, gamma, alpha, T]. No iteration, so no overshoot.

    Global alpha is chosen on a random subset of curves (alpha_subset), then every curve is
    fitted at that alpha. Returns dict(gamma, A, alpha, ia, ig, rms, model).
    """
    g = np.asarray(g, float)
    N, K = g.shape
    if A_grid is None:
        A_grid = np.linspace(0.02, 1.0, 50)
    A_grid = np.asarray(A_grid, float)
    R = model_table(table, A_grid)                           # (na, ng, nA, K)
    if global_alpha:
        rs = np.random.RandomState(seed)
        sub = g[rs.choice(N, size=min(N, alpha_subset), replace=False)]
        tot = [np.mean(_fit_one_alpha(sub, R[a])[2]) for a in range(len(table.alphas))]
        ia_list = [int(np.argmin(tot))]
    else:
        ia_list = list(range(len(table.alphas)))
    best = None
    for a in ia_list:
        ig, iA, sse, prof = _fit_one_alpha(g, R[a])
        if best is None:
            best = [np.full(N, a), ig, iA, sse, prof]
        else:
            m = sse < best[3]
            for arr, new in zip(best, (np.full(N, a), ig, iA, sse, prof)):
                arr[m] = new[m]
    ia, ig, iA, sse, prof = best
    n = np.arange(N)
    jc = np.clip(ig, 1, len(table.gammas) - 2)
    gam = _refine_log(table.gammas, ig, (prof[n, jc - 1], prof[n, jc], prof[n, jc + 1]))
    gam = np.where((ig == 0) | (ig == len(table.gammas) - 1), table.gammas[ig], gam)
    return dict(gamma=gam, A=A_grid[iA], alpha=table.alphas[ia], ia=ia, ig=ig,
                rms=np.sqrt(sse / K), model=R[ia, ig, iA])


def debias(g, taus, T, alphas=None, gammas=None, global_alpha=True, table=None,
           A_grid=None, **_):
    """Finite-T corrected fit of curves g (N, K) measured on a T-frame record.

    Fits the EXPECTED estimator (finite-T bias inside the model, same idea as the D6
    kernel), then returns the bias-corrected curves g_corr = g - (E[g_hat] - g_ideal)
    for downstream use (D5), plus a like-for-like naive fit (T -> infinity) for comparison.
    Returns dict(g_corr, gamma, A, alpha, rms, gamma_raw, alpha_raw, A_raw, table).
    """
    g = np.asarray(g, float)
    if table is None:
        if alphas is None:
            alphas = np.round(np.arange(0.20, 1.21, 0.02), 3)
        if gammas is None:
            gammas = np.logspace(-4, 1.5, 100)
        table = BiasTable(taus, T, alphas, gammas)
    fit = fit_expected(g, table, A_grid, global_alpha)
    naive = fit_curves(g, table, global_alpha)              # ideal-ACF fit, same grids
    ideal = fit['A'][:, None] * table.f[fit['ia'], fit['ig']]
    gc = g - (fit['model'] - ideal)
    return dict(g_corr=gc, gamma=fit['gamma'], A=fit['A'], alpha=fit['alpha'], rms=fit['rms'],
                gamma_raw=naive['gamma'], alpha_raw=naive['alpha'], A_raw=naive['A'],
                table=table)
