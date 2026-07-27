"""Can a Laplace (rate-spectrum) inversion of this project's G(tau) separate
distinct alpha populations, the way CONTIN separates size populations in DLS?

This script answers that quantitatively, on the pipeline's ACTUAL settings:
    RECON_TAUS = (1,2,4,8,16,32,48,64,96,128)   -> 10 lags, 2.11 decades
    model       G(tau) = A / (1 + gamma*tau^alpha),  gamma0 ~ 0.15, alpha0 ~ 0.8
    N_FRAMES = 5000 (half-split -> 2500),  BIN_FACTOR = 2  (4 px averaged)

Five experiments:
  1. conditioning of the Laplace kernel on this tau grid -> how many spectral
     components are resolvable at a given noise level;
  2. the actual per-pixel noise of G_hat(tau) at T=2500 (simulated, not assumed);
  3. the Laplace spectrum of a SINGLE (gamma, alpha) curve -> alpha is a spectrum
     WIDTH, not a separable label;
  4. two populations at rate ratio R vs one (gamma, alpha) -> when does the
     mixture become visible above the noise;
  5. the reverse degeneracy -> an anomalous alpha=0.6 curve fitted by a
     two-exponential mixture.

Deliberately stdlib-only (no numpy): the matrices are 10 x M, and this way it
runs anywhere, including a bare Colab/CI shell.
"""
import math
import random

TAUS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]        # = RECON_TAUS in the notebook
K = len(TAUS)
GAMMA0, ALPHA0 = 0.15, 0.8                           # the fitter's priors


def model(tau, gamma, alpha):
    """The pipeline's per-pixel ACF model, G(tau) = 1/(1 + gamma tau^alpha)."""
    return 1.0 / (1.0 + gamma * tau ** alpha)


# ───────────────────────────── small linear algebra ─────────────────────────
def jacobi_eigvals(A, iters=100):
    """Eigenvalues of a small symmetric matrix (cyclic Jacobi)."""
    n = len(A)
    M = [row[:] for row in A]
    for _ in range(iters):
        off = math.sqrt(sum(M[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < 1e-14:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(M[p][q]) < 1e-18:
                    continue
                theta = (M[q][q] - M[p][p]) / (2 * M[p][q])
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):                                  # right rotation
                    Mkp, Mkq = M[k][p], M[k][q]
                    M[k][p] = c * Mkp - s * Mkq
                    M[k][q] = s * Mkp + c * Mkq
                for k in range(n):                                  # left rotation
                    Mpk, Mqk = M[p][k], M[q][k]
                    M[p][k] = c * Mpk - s * Mqk
                    M[q][k] = s * Mpk + c * Mqk
    return sorted((M[i][i] for i in range(n)), reverse=True)


def rate_grid(n=120, taus=None, lo=None, hi=None):
    """Log-spaced decay rates spanning (a little beyond) what the tau window sees."""
    taus = taus or TAUS
    lo = lo or 1.0 / (3 * max(taus))
    hi = hi or 3.0 / min(taus)
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# ───────────── EXP 1 — how many spectral components does tau allow? ─────────
def laplace_dof(taus=None, noise_levels=(0.01, 0.03, 0.041, 0.10), verbose=True):
    """Singular values of K[k,j] = exp(-Gamma_j tau_k); count those above noise.

    A Laplace inversion can only recover the singular directions whose singular
    value stands above the data noise — everything below is pure regularisation
    (i.e. the prior, not the data). This is the hard ceiling on the answer.
    """
    taus = taus or TAUS
    G = rate_grid(120, taus)
    w = math.sqrt(math.log(G[-1] / G[0]) / (len(G) - 1))     # d(log Gamma) measure
    Kmat = [[w * math.exp(-g * t) for g in G] for t in taus]
    KKt = [[sum(Kmat[i][j] * Kmat[k][j] for j in range(len(G))) for k in range(len(taus))]
           for i in range(len(taus))]
    sv = [math.sqrt(max(e, 0.0)) for e in jacobi_eigvals(KKt)]
    s0 = sv[0]
    if verbose:
        print(f"\n=== EXP 1 — Laplace kernel conditioning, {len(taus)} lags, "
              f"{math.log10(max(taus) / min(taus)):.2f} decades ===")
        print("    sigma_i/sigma_0: " + "  ".join(f"{s / s0:.2e}" for s in sv[:12]))
        for nl in noise_levels:
            print(f"    noise/signal = {nl * 100:5.2f}%  ->  "
                  f"{sum(1 for s in sv if s / s0 > nl)} resolvable components")
    return sv


# ───────────── Richardson-Lucy positive (CONTIN-like) inversion ─────────────
def invert_spectrum(g, G, iters=4000, taus=None):
    """Non-negative p(Gamma) with sum_j p_j exp(-Gamma_j tau_k) ~= g_k."""
    taus = taus or TAUS
    Kmat = [[math.exp(-gm * t) for gm in G] for t in taus]
    colsum = [sum(Kmat[k][j] for k in range(len(taus))) for j in range(len(G))]
    p = [1.0 / len(G)] * len(G)
    for _ in range(iters):
        pred = [sum(Kmat[k][j] * p[j] for j in range(len(G))) for k in range(len(taus))]
        ratio = [g[k] / max(pred[k], 1e-12) for k in range(len(taus))]
        for j in range(len(G)):
            p[j] *= sum(Kmat[k][j] * ratio[k] for k in range(len(taus))) / max(colsum[j], 1e-12)
    return p


def spectrum_stats(p, G):
    """Centroid and width of p(Gamma), both in decades of log10(Gamma)."""
    tot = sum(p)
    if tot <= 0:
        return 0.0, 0.0
    w = [x / tot for x in p]
    lg = [math.log10(x) for x in G]
    mean = sum(w[i] * lg[i] for i in range(len(G)))
    return mean, math.sqrt(sum(w[i] * (lg[i] - mean) ** 2 for i in range(len(G))))


def count_peaks(p, rel=0.05):
    mx = max(p) or 1.0
    return sum(1 for i in range(1, len(p) - 1)
               if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > rel * mx)


def rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


# ───────────── EXP 2 — what IS the per-pixel noise on G_hat(tau)? ───────────
def simulate_acf_noise(gamma=GAMMA0, alpha=ALPHA0, T=2500, reps=120,
                       n_modes=24, bin_px=4, seed=0):
    """Std of the per-pixel G_hat(tau) from a finite record, measured not assumed.

    The synthetic process is a sum of independent AR(1) modes whose weights are
    solved (by the same RL inversion) so its true ACF IS 1/(1+gamma tau^alpha) —
    the very curve the pipeline fits. bin_px = the 4 pixels BIN_FACTOR=2 averages.
    """
    rnd = random.Random(seed)
    target = [model(t, gamma, alpha) for t in TAUS]
    G = rate_grid(n_modes)
    p = invert_spectrum(target, G)
    tot = sum(p) or 1.0
    amp = [math.sqrt(x / tot) for x in p]
    phi = [math.exp(-g) for g in G]                      # AR(1) coefficient per mode

    sums, sums2 = [0.0] * K, [0.0] * K
    for _ in range(reps):
        acc, acc0 = [0.0] * K, 0.0
        for _px in range(bin_px):
            state = [rnd.gauss(0, 1) for _ in range(n_modes)]
            x = [0.0] * T
            for t in range(T):
                s = 0.0
                for m in range(n_modes):
                    state[m] = phi[m] * state[m] + math.sqrt(1 - phi[m] ** 2) * rnd.gauss(0, 1)
                    s += amp[m] * state[m]
                x[t] = s
            mu = sum(x) / T
            d = [v - mu for v in x]
            acc0 += sum(v * v for v in d) / T
            for i, tau in enumerate(TAUS):
                n = T - tau
                acc[i] += sum(d[j] * d[j + tau] for j in range(n)) / n
        for i in range(K):
            g = acc[i] / acc0
            sums[i] += g
            sums2[i] += g * g

    print(f"\n=== EXP 2 — per-pixel G(tau) noise, T={T} frames, {bin_px}px bin ===")
    print("     tau    G_true    G_mean        sd   sd/G_true    bias")
    sds = []
    for i, tau in enumerate(TAUS):
        mean = sums[i] / reps
        sd = math.sqrt(max(sums2[i] / reps - mean * mean, 0.0))
        sds.append(sd)
        print(f"    {tau:4d}  {target[i]:8.4f}  {mean:8.4f}  {sd:8.4f}  {sd / max(target[i], 1e-9) * 100:8.1f}%"
              f"  {(mean - target[i]) / max(target[i], 1e-9) * 100:+7.1f}%")
    r = rms(sds)
    print(f"    RMS noise on G(tau) = {r:.4f}  ({r * 100:.1f}% of G(0)=1)")
    print("    NOTE the systematic negative bias at large tau: finite-T mean subtraction.")
    print("    It sits exactly on the slow lags a Laplace inversion leans on.")
    return sds, r


# ───────────── EXP 3 — alpha IS the spectrum width ──────────────────────────
def exp3_alpha_is_width(noise_rms):
    print("\n=== EXP 3 — Laplace spectrum of ONE (gamma, alpha) curve ===")
    print("    (were alpha a separable 'population' label, these would be multi-peaked)")
    print("   alpha   residual   width(decades)   peaks   representable?")
    G = rate_grid(120)
    for alpha in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
        g = [model(t, GAMMA0, alpha) for t in TAUS]
        p = invert_spectrum(g, G)
        pred = [sum(math.exp(-G[j] * t) * p[j] for j in range(len(G))) for t in TAUS]
        r = rms([g[k] - pred[k] for k in range(K)])
        _, width = spectrum_stats(p, G)
        print(f"   {alpha:5.2f}  {r:9.5f}   {width:14.2f}   {count_peaks(p):5d}   "
              f"{'yes' if r < noise_rms else 'NO positive spectrum fits'}")
    print("    Analytic check: alpha=1 gives p(Gamma) = (1/gamma) exp(-Gamma/gamma) exactly,")
    print("    whose log10 width is pi/(sqrt(6) ln10) = 0.557 decades.")


# ───────────── EXP 4 — two populations vs one (gamma, alpha) ────────────────
def fit_gamma_alpha(g):
    """Least-squares fit of the pipeline's 1-component model (grid then refine)."""
    best = (1e9, GAMMA0, ALPHA0)
    for gi in range(-40, 21):
        gam = 10 ** (gi / 10.0)
        for ai in range(4, 41):
            al = ai / 20.0
            r = sum((g[k] - model(TAUS[k], gam, al)) ** 2 for k in range(K))
            if r < best[0]:
                best = (r, gam, al)
    _, gam, al = best
    for scale in (0.3, 0.1, 0.03, 0.01):
        for _ in range(60):
            improved = False
            for dg, da in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                gg = gam * (1 + scale * dg)
                aa = max(0.05, min(2.0, al + scale * da * 0.5))
                r = sum((g[k] - model(TAUS[k], gg, aa)) ** 2 for k in range(K))
                if r < best[0]:
                    best, gam, al, improved = (r, gg, aa), gg, aa, True
            if not improved:
                break
    return math.sqrt(best[0] / K), best[1], best[2]


def exp4_two_populations(noise_rms):
    print("\n=== EXP 4 — two populations (rate ratio R) vs a single (gamma, alpha) ===")
    print(f"    per-pixel noise floor = {noise_rms:.4f}")
    print("      R   1-comp residual   per-pixel?   px to average   best 1-comp fit")
    resid = {}
    for R in (2, 3, 5, 10, 30, 100):
        gs, gf = GAMMA0 / math.sqrt(R), GAMMA0 * math.sqrt(R)
        g = [0.5 * model(t, gs, 1.0) + 0.5 * model(t, gf, 1.0) for t in TAUS]
        r, gam, al = fit_gamma_alpha(g)
        resid[R] = r
        thr = 3 * noise_rms / math.sqrt(K)
        npx = (thr / r) ** 2 if r > 0 else float("inf")
        print(f"    {R:3d}   {r:15.5f}   {'YES' if r > thr else 'no':>10s}   {npx:13.0f}"
              f"   gamma={gam:.3f}, alpha={al:.2f}")
    print("    Read the last column: a mixture does not vanish — it re-appears AS a")
    print("    lower alpha. alpha already carries the polydispersity.")
    return resid


def exp4b_pixel_averaging(resid, noise_rms=0.0413):
    print("\n=== EXP 4b — the only lever that works: average N pixels ===")
    print("   N px    noise      R=2    R=3    R=5   R=10   R=30")
    for N in (1, 4, 16, 100, 400, 900, 3600):
        nz = noise_rms / math.sqrt(N)
        thr = 3 * nz / math.sqrt(K)
        row = "  ".join(f"{'YES' if resid[R] > thr else 'no':>5s}" for R in (2, 3, 5, 10, 30))
        print(f"  {N:5d}   {nz * 100:5.2f}%   {row}")


# ───────────── EXP 5 — the reverse degeneracy ───────────────────────────────
def exp5_alpha_vs_mixture(noise_rms):
    print("\n=== EXP 5 — is 'anomalous alpha' distinguishable from 'a mixture'? ===")
    g = [model(t, GAMMA0, 0.6) for t in TAUS]
    best = (1e9, 0.0, 0.0, 0.0)
    for i in range(-35, 16):
        gs = 10 ** (i / 10.0)
        for j in range(i + 1, 26):
            gf = 10 ** (j / 10.0)
            for fi in range(1, 20):
                f = fi / 20.0
                r = sum((g[k] - (f * model(TAUS[k], gf, 1.0)
                                 + (1 - f) * model(TAUS[k], gs, 1.0))) ** 2 for k in range(K))
                if r < best[0]:
                    best = (r, gs, gf, f)
    r = math.sqrt(best[0] / K)
    print(f"    alpha=0.6 curve, best 2-exponential (alpha=1) mixture:")
    print(f"      gamma_slow={best[1]:.4f}  gamma_fast={best[2]:.4f}  f={best[3]:.2f}")
    print(f"      residual {r:.5f}  vs per-pixel noise {noise_rms:.4f}  ->  "
          f"{'INDISTINGUISHABLE' if r < noise_rms else 'distinguishable'}")


# ───────────── lever A — would a longer tau window help? ────────────────────
def exp6_tau_window(noise_rms):
    print("\n=== EXP 6 — lever A: does a longer tau window buy components? ===")
    grids = {
        "current  1..128  (10 lags, 2.11 dec)": TAUS,
        "extended 1..512  (12 lags, 2.71 dec)": [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512],
        "extended 1..2048 (13 lags, 3.31 dec)": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048],
    }
    for name, tg in grids.items():
        sv = laplace_dof(tg, verbose=False)
        counts = [sum(1 for s in sv if s / sv[0] > nl) for nl in (noise_rms, 0.0041, 0.0014)]
        print(f"    {name}:  at 4.1% noise -> {counts[0]} comps | "
              f"at 0.41% -> {counts[1]} | at 0.14% -> {counts[2]}")
    print("    Conditioning is set by decades of tau, and it saturates fast: at the real")
    print("    per-pixel noise, 16x more lags still buys nothing. Noise is the binding")
    print("    constraint, not the tau window.")


if __name__ == "__main__":
    laplace_dof()
    _, noise = simulate_acf_noise()
    exp3_alpha_is_width(noise)
    resid = exp4_two_populations(noise)
    exp4b_pixel_averaging(resid, noise)
    exp5_alpha_vs_mixture(noise)
    exp6_tau_window(noise)
    print("\nConclusion: on this tau grid and at this per-pixel noise, a Laplace")
    print("inversion cannot separate alpha populations per pixel — alpha IS the")
    print("spectrum width. It only becomes a real measurement on ROI-averaged")
    print("curves (>= ~100 px), and only for rate ratios >= ~10.")
