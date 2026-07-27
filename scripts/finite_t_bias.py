"""The finite-T bias on G_hat(tau): is it uniform, and does it matter?

Two questions, because the answers point in opposite directions:

  PART 1 — is the bias uniform across pixels?
      If uniform it shifts every map by a constant and every downstream step
      (the fixed-slope intercept b, z-scoring, GEVD) absorbs it -> harmless.
      If it depends on the pixel's own gamma it is a signal-dependent distortion
      of the very contrast being mapped, AND it is identical in both halves,
      so half-split reproducibility cannot detect it.

  PART 2 — how big is it next to the things we want to measure?
      Per pixel it sits under the noise. On an ROI average the noise falls as
      1/sqrt(N) but the bias does not fall at all, so bulk analyses are
      systematics-limited, not noise-limited.

Run: python3 scripts/finite_t_bias.py     (stdlib only; PART 1 takes a few minutes)
"""
import math
import random

from laplace_identifiability import (TAUS, K, GAMMA0, ALPHA0, model, fit_gamma_alpha,
                                     invert_spectrum, rate_grid, rms)

PIXEL_NOISE = 0.0413            # measured in laplace_identifiability EXP 2


# ─────────── PART 1 — does the bias depend on the pixel's own gamma? ────────
def simulate(gamma, alpha=ALPHA0, T=2500, reps=60, n_modes=24, seed=1):
    """Mean G_hat(tau) (normalised by C_hat(0)) and C_hat(0)/C_true(0) at this gamma.

    True C(0) = 1 by construction, so the second return value IS the CV^2 bias.
    """
    rnd = random.Random(seed)
    G = rate_grid(n_modes)
    p = invert_spectrum([model(t, gamma, alpha) for t in TAUS], G)
    tot = sum(p) or 1.0
    amp = [math.sqrt(x / tot) for x in p]
    phi = [math.exp(-g) for g in G]
    sig = [math.sqrt(1 - f * f) for f in phi]
    gauss = rnd.gauss

    acc, acc0 = [0.0] * K, 0.0
    for _ in range(reps):
        state = [gauss(0, 1) for _ in range(n_modes)]
        x = [0.0] * T
        for t in range(T):
            s = 0.0
            for m in range(n_modes):
                state[m] = phi[m] * state[m] + sig[m] * gauss(0, 1)
                s += amp[m] * state[m]
            x[t] = s
        mu = sum(x) / T
        d = [v - mu for v in x]
        c0 = sum(v * v for v in d) / T
        acc0 += c0
        for i, tau in enumerate(TAUS):
            n = T - tau
            acc[i] += (sum(d[j] * d[j + tau] for j in range(n)) / n) / c0
    return [a / reps for a in acc], acc0 / reps


def fit_fixed_alpha(g, alpha=ALPHA0):
    """gamma-only fit — what global_alpha=True effectively does per pixel."""
    best, bg = 1e9, None
    for gi in range(-400, 201):
        gam = 10 ** (gi / 100.0)
        r = sum((g[k] - model(TAUS[k], gam, alpha)) ** 2 for k in range(K))
        if r < best:
            best, bg = r, gam
    return bg


def part1():
    print("=== PART 1 — finite-T bias vs the pixel's own dynamics "
          f"(T=2500, alpha_true={ALPHA0}) ===")
    print("    X = log10(1/gamma), Y = log10(CV^2); the bias moves BOTH.\n")
    print("  gamma_true   gamma_hat(fixed a)   gamma_hat(free)   alpha_hat   "
          "CV2_hat/CV2       dX         dY")
    rows = []
    for gamma in (0.03, 0.06, 0.12, 0.25, 0.5, 1.0):
        curve, c0ratio = simulate(gamma)
        g_fix = fit_fixed_alpha(curve)
        _, g_free, a_free = fit_gamma_alpha(curve)
        dX = -(math.log10(g_fix) - math.log10(gamma))          # X = log10(1/gamma)
        dY = math.log10(c0ratio)
        rows.append((gamma, g_fix, dX, dY))
        print(f"  {gamma:9.3f}   {g_fix:18.4f}   {g_free:15.4f}   {a_free:9.3f}   "
              f"{c0ratio:11.4f}   {dX:+8.4f}   {dY:+8.4f}")

    print("\n  -- the artifact's trajectory in the (X, Y) plane --")
    for i in range(1, len(rows)):
        ddX, ddY = rows[i][2] - rows[i - 1][2], rows[i][3] - rows[i - 1][3]
        if abs(ddX) > 1e-6:
            print(f"    gamma {rows[i-1][0]:.3f} -> {rows[i][0]:.3f}:  "
                  f"artifact slope dY/dX = {ddY/ddX:+.2f}")

    span_true = math.log10(rows[-1][0] / rows[0][0])
    span_meas = math.log10(rows[-1][1] / rows[0][1])
    print(f"\n  true log10 gamma span     : {span_true:.3f} decades")
    print(f"  measured log10 gamma span : {span_meas:.3f} decades")
    print(f"  -> gamma contrast compressed to {span_meas / span_true * 100:.1f}% of truth")

    # effect on the measured within-cell Y-vs-X slope
    dX_range = rows[-1][2] - rows[0][2]
    dY_range = rows[-1][3] - rows[0][3]
    X_range = math.log10(1 / rows[-1][0]) - math.log10(1 / rows[0][0])
    dstretch, dtilt = dX_range / X_range, dY_range / X_range
    print(f"\n  effect on a measured slope s: s_meas ~ (s_true {dtilt:+.4f}) / "
          f"{1 + dstretch:.4f}")
    for s_true in (0.4, 0.5, 3.0):
        print(f"    s_true = {s_true:.1f}  ->  s_measured = "
              f"{(s_true + dtilt) / (1 + dstretch):.3f}")


# ─────────── PART 2 — bias vs the signal a bulk analysis looks for ──────────
G_TRUE = [model(t, GAMMA0, ALPHA0) for t in TAUS]
G_MEAS = [0.8725, 0.7910, 0.6799, 0.5437, 0.3988,           # measured, EXP 2
          0.2653, 0.2000, 0.1620, 0.1092, 0.0811]


def part2():
    bias = [G_MEAS[i] - G_TRUE[i] for i in range(K)]
    rb = rms(bias)
    print("\n\n=== PART 2 — the bias in the same units as everything else ===")
    print("     tau:  " + " ".join(f"{t:7d}" for t in TAUS))
    print("    bias:  " + " ".join(f"{b:+7.4f}" for b in bias))
    print(f"    RMS systematic bias = {rb:.4f}   (it does NOT fall with pixel averaging)")
    for n in (1, 100, 1000, 3000):
        print(f"    vs statistical noise at {n:5d} px = {PIXEL_NOISE/math.sqrt(n):.4f}"
              f"  ->  bias/noise = {rb/(PIXEL_NOISE/math.sqrt(n)):5.1f}x")

    print("\n  -- mixture signatures a bulk spectrum test hunts for (f=0.5) --")
    for R in (3, 10, 30, 100):
        gs, gf = GAMMA0 / math.sqrt(R), GAMMA0 * math.sqrt(R)
        g = [0.5 * model(t, gf, 1.0) + 0.5 * model(t, gs, 1.0) for t in TAUS]
        r, _, _ = fit_gamma_alpha(g)
        print(f"    R={R:4d}: 1-comp residual {r:.5f}  "
              f"({'SMALLER' if r < rb else 'larger'} than the {rb:.4f} bias)")

    print("\n  -- does the bias ALONE fake a second component? --")
    r_true, g_t, a_t = fit_gamma_alpha(G_TRUE)
    r_meas, g_m, a_m = fit_gamma_alpha(G_MEAS)
    thr = 3 * (PIXEL_NOISE / math.sqrt(1000)) / math.sqrt(K)
    print(f"    fit to TRUE curve  : residual {r_true:.5f}  gamma={g_t:.4f} alpha={a_t:.3f}")
    print(f"    fit to BIASED curve: residual {r_meas:.5f}  gamma={g_m:.4f} alpha={a_m:.3f}")
    print(f"    3-sigma bulk threshold at 1000 px = {thr:.5f}")
    print(f"    -> the biased curve exceeds it by {r_meas/thr:.1f}x on bias alone.")
    print(f"    alpha {a_t:.3f} -> {a_m:.3f} (inflated): a depressed slow tail reads as")
    print("    LESS slow power, so it MASKS a real slow population (false negative),")
    print("    while an over-correction manufactures one (false positive).")

    print("\n  -- bulk sensitivity vs minority fraction f (1000 px, statistical only) --")
    print("    minority f |     R=3        R=10       R=30      R=100")
    for f in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        cells = []
        for R in (3, 10, 30, 100):
            gs, gf = GAMMA0 / math.sqrt(R), GAMMA0 * math.sqrt(R)
            g = [f * model(t, gf, 1.0) + (1 - f) * model(t, gs, 1.0) for t in TAUS]
            r, _, _ = fit_gamma_alpha(g)
            cells.append(f"{r:.5f}{'*' if r > thr else ' '}")
        print(f"       {f:5.2f}   | " + "  ".join(cells))
    print("    (* = above the 3-sigma STATISTICAL threshold only; none of it")
    print("     survives the systematic until the bias is corrected)")


if __name__ == "__main__":
    part1()
    part2()
