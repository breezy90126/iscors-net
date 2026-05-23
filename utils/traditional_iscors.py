import numpy as np
from scipy.optimize import curve_fit

def compute_autocorrelation(trace):
    """
    Computes the temporal autocorrelation of a 1D intensity trace.
    Uses numpy.correlate for efficiency.
    """
    # Normalize trace by subtracting mean
    mean_I = np.mean(trace)
    if mean_I == 0:
        return np.zeros(len(trace) // 2)
        
    fluct = trace - mean_I
    # Full cross-correlation of fluct with itself
    corr = np.correlate(fluct, fluct, mode='full')
    # Take the second half (tau >= 0)
    corr = corr[len(corr)//2:]
    
    # Normalize by the number of overlapping points and mean intensity squared
    N = len(trace)
    lags = np.arange(len(corr))
    # G(tau) = <dI(t)dI(t+tau)> / <I>^2
    g_tau = corr / ((N - lags) * (mean_I ** 2) + 1e-10)
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
    Given a single pixel's time trace (e.g., 500 frames), compute the 
    autocorrelation and fit it to extract Gamma and Alpha.
    
    Returns:
        gamma, alpha
    """
    if np.var(trace) < 1e-8:
        # Static background, no dynamics
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
        return gamma_fit, alpha_fit
    except Exception as e:
        # If curve fitting fails (e.g., too noisy or flat), return safe defaults
        return 0.0, 0.0

if __name__ == "__main__":
    # Test the traditional algorithm
    t = np.arange(500)
    # Generate a dummy trace with some decay-like correlation 
    # (just random walk to simulate Brownian-ish motion)
    trace = np.cumsum(np.random.randn(500)) + 1000 
    
    gamma, alpha = fit_physical_parameters(trace)
    print(f"Fitted Gamma: {gamma:.4f}, Fitted Alpha: {alpha:.4f}")
