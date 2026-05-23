import numpy as np
import tifffile
import os

def generate_signal_with_autocorr(T, gamma, alpha, mean_val=100.0, variance=10.0):
    """
    Generates a 1D time series of length T with a specific 
    autocorrelation decay: G(tau) = 1 / (1 + gamma * tau^alpha).
    Uses the Wiener-Khinchin theorem (Frequency domain coloring).
    """
    if gamma == 0:
        # Pure white noise
        return np.random.normal(mean_val, np.sqrt(variance), T)
        
    # 1. Define theoretical G(tau)
    tau = np.arange(T)
    # Mirror tau to make it symmetric for FFT: [0, 1, 2... T/2, -T/2+1 ... -1]
    tau_symmetric = np.minimum(tau, T - tau)
    
    g_tau = 1.0 / (1.0 + gamma * (tau_symmetric ** alpha))
    
    # 2. Power Spectral Density (PSD) is the FFT of the autocorrelation
    psd = np.fft.fft(g_tau).real
    # Ensure non-negative (can sometimes have small negative numerical artifacts)
    psd = np.maximum(psd, 0)
    
    # 3. Generate White Noise in frequency domain
    white_noise = np.random.randn(T)
    white_noise_f = np.fft.fft(white_noise)
    
    # 4. Color the noise: multiply by sqrt(PSD)
    colored_noise_f = white_noise_f * np.sqrt(psd)
    
    # 5. Inverse FFT to get time domain signal
    colored_noise = np.fft.ifft(colored_noise_f).real
    
    # 6. Adjust mean and variance
    colored_noise = colored_noise - np.mean(colored_noise)
    colored_noise = (colored_noise / np.std(colored_noise)) * np.sqrt(variance)
    
    return colored_noise + mean_val

def create_test_video(output_path="./data/test_synthetic_cell.tif", T=200, H=128, W=128):
    print(f"Generating synthetic video (T={T}, H={H}, W={W})...")
    
    video = np.zeros((T, H, W), dtype=np.float32)
    
    # Ground truth parameter maps (one value per pixel)
    gt_gamma_map = np.zeros((H, W), dtype=np.float32)
    gt_alpha_map = np.zeros((H, W), dtype=np.float32)
    
    # Define Regions
    # Large Circle (Cell body)
    cy_large, cx_large, r_large = 64, 64, 50
    gamma_large, alpha_large = 0.1, 1.0
    
    # Small Circle 1 (Organelle 1 - Fast Anomalous)
    cy_s1, cx_s1, r_s1 = 45, 45, 15
    gamma_s1, alpha_s1 = 0.5, 1.5
    
    # Small Circle 2 (Organelle 2 - Slow Subdiffusion)
    cy_s2, cx_s2, r_s2 = 85, 85, 15
    gamma_s2, alpha_s2 = 0.05, 0.5
    
    # Generate pixel by pixel
    for y in range(H):
        for x in range(W):
            # Check which region the pixel belongs to
            d_large = np.sqrt((x - cx_large)**2 + (y - cy_large)**2)
            d_s1 = np.sqrt((x - cx_s1)**2 + (y - cy_s1)**2)
            d_s2 = np.sqrt((x - cx_s2)**2 + (y - cy_s2)**2)
            
            if d_s1 <= r_s1:
                # Small circle 1
                g, a = gamma_s1, alpha_s1
            elif d_s2 <= r_s2:
                # Small circle 2
                g, a = gamma_s2, alpha_s2
            elif d_large <= r_large:
                # Large circle
                g, a = gamma_large, alpha_large
            else:
                # Background
                g, a = 0.0, 0.0 # White noise
                
            gt_gamma_map[y, x] = g
            gt_alpha_map[y, x] = a
            trace = generate_signal_with_autocorr(T, gamma=g, alpha=a)
            video[:, y, x] = trace
            
    print(f"Saving video to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, video)
    
    # Save the perfect GT maps alongside the video
    gt_path = output_path.replace('.tif', '_gt.npz')
    np.savez(gt_path, gamma=gt_gamma_map, alpha=gt_alpha_map)
    print(f"Saved GT maps to {gt_path}")
    print("Done!")

if __name__ == "__main__":
    create_test_video("./data/test_synthetic_cell.tif")
