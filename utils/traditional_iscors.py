import numpy as np
from scipy.optimize import curve_fit

def compute_autocorrelation(trace):
    """
    FFT-based autocorrelation with Hann windowing (Wiener-Khinchin theorem).
    Hann window reduces spectral leakage; FFT gives O(N log N) vs O(N^2).
    Window bias is corrected by dividing by the window self-correlation per lag.
    """
    N = len(trace)
    mean_I = np.mean(trace)
    if mean_I == 0:
        return np.zeros(N // 2)

    fluct = trace - mean_I

    # Hann window to reduce spectral leakage
    window = np.hanning(N)
    fluct_w = fluct * window

    # Zero-pad to next power of 2 for FFT efficiency
    fft_len = int(2 ** np.ceil(np.log2(2 * N)))

    # Autocorrelation via Wiener-Khinchin: IFFT(|FFT(x)|^2)
    F = np.fft.rfft(fluct_w, n=fft_len)
    corr = np.fft.irfft(F * np.conj(F), n=fft_len)[:N].real

    # Window bias correction: divide by window self-correlation at each lag
    W = np.fft.rfft(window, n=fft_len)
    w_corr = np.fft.irfft(W * np.conj(W), n=fft_len)[:N].real

    # G(tau) = <dI(t)dI(t+tau)> / <I>^2
    g_tau = corr / (w_corr * mean_I ** 2 + 1e-10)
    return g_tau

def theoretical_g_tau(tau, gamma, alpha, amplitude):
    """
    Theoretical anomalous diffusion decay model.
    Note: Real iSCORS often uses G(tau) = G(0) / (1 + (tau/tau_D)^alpha)
    Here we implement a general power-law/exponential-like decay based on 
    the concept: G(tau) proportional to Gamma * tau^(-alpha).
    You can adjust this exact mathematical equation based on your specific lab's model.
    """
    # Avoid divide by zero for tau=0 if formula demands it, 
    # but normally tau >= 1 for this fitting
    return amplitude / (1.0 + gamma * (tau ** alpha))

def fit_physical_parameters(trace, max_tau=64):
    """
    Compute autocorrelation of a pixel trace and fit gamma / alpha.

    Background detection uses coefficient of variation (CV = std/mean):
        - Near-static background: CV < 0.5%  → return (0.0, 0.0)
        - Cell pixels:            CV ≈ 3%    → proceed with fitting
    This replaces the old `np.var < 1e-8` check, which missed background
    pixels generated with white noise (variance=10, so var >> 1e-8).
    """
    mean_I = np.mean(trace)
    if mean_I == 0:
        return 0.0, 0.0

    cv = np.std(trace) / (mean_I + 1e-10)
    if cv < 0.005:   # CV < 0.5% → near-static, no dynamics to fit
        return 0.0, 0.0
        
    # 1. Compute empirical G_rough(tau)
    g_tau = compute_autocorrelation(trace)
    
    # We only fit up to max_tau (or the length of the trace, whichever is smaller)
    max_tau = min(max_tau, len(g_tau) - 1)
    
    # Tau values for fitting (usually start from tau=1 to avoid shot noise at tau=0)
    taus_to_fit = np.arange(1, max_tau + 1)
    g_to_fit = g_tau[1:max_tau + 1]
    
    # 2. Curve Fitting
    # Initial guesses: Gamma=0.1, Alpha=1.0, Amplitude=g_tau[1]
    p0 = [0.1, 1.0, g_to_fit[0] if len(g_to_fit)>0 else 1.0]
    # Bounds: Gamma > 0, 0 < Alpha < 2, Amplitude > 0
    bounds = ([0.0, 0.0, 0.0], [np.inf, 2.0, np.inf])
    
    try:
        popt, pcov = curve_fit(theoretical_g_tau, taus_to_fit, g_to_fit, p0=p0, bounds=bounds, maxfev=1000)
        gamma_fit, alpha_fit, amp_fit = popt
        y_pred  = theoretical_g_tau(taus_to_fit, *popt)
        ss_res  = np.sum((g_to_fit - y_pred) ** 2)
        ss_tot  = np.sum((g_to_fit - g_to_fit.mean()) ** 2)
        r2      = float(1.0 - ss_res / (ss_tot + 1e-10))
        return gamma_fit, alpha_fit, r2
    except Exception:
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    # Test the traditional algorithm
    t = np.arange(500)
    # Generate a dummy trace with some decay-like correlation 
    # (just random walk to simulate Brownian-ish motion)
    trace = np.cumsum(np.random.randn(500)) + 1000 
    
    gamma, alpha, r2 = fit_physical_parameters(trace)
    print(f"Fitted Gamma: {gamma:.4f}, Fitted Alpha: {alpha:.4f}, R²: {r2:.4f}")
