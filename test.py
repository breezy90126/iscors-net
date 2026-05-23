import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from models.iscat_encoder import ISCATEncoder
from datasets.sequence_dataset import ISCATSequenceDataset
from torch.utils.data import DataLoader
from models.loss.msd_loss import compute_empirical_msd

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data (Use train data for MVP test or separate test folder)
    # Ideally test on 'data/test' (unseen).
    # Assuming we generated test data in main.py? 
    # main.py generated 'data/train'. Let's use 'data/train' for verification now.
    dataset = ISCATSequenceDataset(
        dataPath=config['train_folder'],
        mode='test',
        stack_size=5,
        sequence_length=50  # Reduced from 100 to fit in 100-frame movies
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = ISCATEncoder(in_channels=5, out_channels=5, final_sigmoid=True, latent_dim=1).to(device)
    
    # Load Checkpoint
    # Find latest checkpoint
    checkpoint_dir = config['checkpoint_path']
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found. Running with random weights.")
    else:
        latest = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest)))
        print(f"Loaded checkpoint {latest}")
    
    model.eval()
    
    # Inference
    all_latents = []
    all_gts = []
    
    with torch.no_grad():
        for i, (seq_data, gt_z) in enumerate(loader):
            # seq_data: [B, T_seq, Stack, H, W]
            # gt_z: [B, T_seq, N_particles]
            seq_data = seq_data.to(device)
            B, T_seq, Stack, H, W = seq_data.shape
            
            # Match training normalization
            median_bg = torch.median(seq_data, dim=1, keepdim=True).values 
            seq_data_norm = seq_data / (median_bg + 1e-6) - 1.0
            
            flat_input = seq_data_norm.view(-1, Stack, H, W)
            
            recons_full, latent_full, pred_msd_full = model(flat_input)
            
            denoised_frames = recons_full[:, Stack // 2, :, :]
            denoised_seq = denoised_frames.view(B, T_seq, H, W)
            noisy_seq = seq_data[:, :, Stack // 2, :, :]
            
            latent_seq = latent_full.view(B, T_seq)
            pred_msd_seq = pred_msd_full.view(B, T_seq, -1)
            
            all_latents.append(latent_seq.cpu().numpy())
            all_gts.append(gt_z.cpu().numpy())
            avg_pred_msd = torch.mean(pred_msd_seq, dim=1).cpu().numpy()
            all_pred_msds = avg_pred_msd
            
            test_noisy_video = noisy_seq[0].cpu().numpy()
            test_denoised_video = denoised_seq[0].cpu().numpy()
            
            if i >= 0: break
    
    # Analyze
    latents = all_latents[0][0] # [T_seq]
    gts = all_gts[0][0] # [T_seq, N_particles]
    pred_msd = all_pred_msds[0]
    
    # 1. Trajectory Comparison Logic for Multi-Particle
    # Pick the mean GT Z for bulk comparison
    gt_mean = np.mean(gts, axis=1)
    
    # Find the particle that most correlates with the model's output
    def norm_01(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    latents_norm = norm_01(latents)
    
    correlations = []
    for p in range(gts.shape[1]):
        corr = np.corrcoef(latents_norm, norm_01(gts[:, p]))[0, 1]
        correlations.append(corr)
    best_p = np.argmax(correlations)
    
    plt.figure(figsize=(14, 7))
    # Plot all GT particles with high transparency
    for p in range(gts.shape[1]):
        alpha = 0.5 if p == best_p else 0.15
        ls = '-' if p == best_p else '--'
        color = 'red' if p == best_p else 'gray'
        plt.plot(norm_01(gts[:, p]), ls=ls, color=color, alpha=alpha, 
                 label=f'Particle {p+1}' if p == best_p or p < 3 else "")
    
    # Plot Model Prediction
    plt.plot(latents_norm, 'b-', linewidth=2.5, label='Model Estimated z(t)')
    
    plt.title(f"Identity Verification: Model vs All 10 Particles (Best Match: P{best_p+1}, corr={correlations[best_p]:.2f})")
    plt.xlabel("Time (Frame)")
    plt.ylabel("Normalized Z-Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(config['results_dir'], 'trajectory_comparison.png'))
    plt.close()
    
    # 2. Plot MSD Comparison
    plt.figure(figsize=(10, 5))
    tau_range = list(range(1, 1 + len(pred_msd)))
    latent_tensor = torch.tensor(latents).unsqueeze(0)
    empirical_msd_values = compute_empirical_msd(latent_tensor, tau_range)[0].numpy()
    
    plt.plot(tau_range, empirical_msd_values, 'o-', linewidth=2, label='Empirical MSD (from z)')
    plt.plot(tau_range, pred_msd, 'x--', linewidth=2, label='Predicted MSD (from features)')
    plt.title("MSD Consistency Check")
    plt.xlabel("Lag Time (tau)")
    plt.ylabel("MSD Value")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(config['results_dir'], 'msd_comparison.png'))
    plt.close()
    
    # 3. Save Denoised Video Comparison (GIF) with Robust Contrast & Colorbar
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    print("Generating improved denoised video comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Robust scaling: 1st and 99th percentile
    def get_vrange(data):
        return np.percentile(data, [1, 99])
    
    v1_min, v1_max = get_vrange(test_noisy_video)
    v2_min, v2_max = get_vrange(test_denoised_video)
    
    im1 = ax1.imshow(test_noisy_video[0], cmap='gray', vmin=v1_min, vmax=v1_max)
    ax1.set_title("Original (Noisy)")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    
    im2 = ax2.imshow(test_denoised_video[0], cmap='magma', vmin=v2_min, vmax=v2_max)
    ax2.set_title("Denoised (FAST Output)")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    
    # Add pixel scale label (proxy)
    ax1.set_xlabel("Pixels")
    ax2.set_xlabel("Pixels")

    def update(frame_idx):
        im1.set_data(test_noisy_video[frame_idx])
        im2.set_data(test_denoised_video[frame_idx])
        return [im1, im2]
    
    ani = FuncAnimation(fig, update, frames=len(test_noisy_video), interval=100, blit=True)
    gif_path = os.path.join(config['results_dir'], 'denoised_comparison.gif')
    ani.save(gif_path, writer='pillow')
    plt.close()
    
    print(f"Test complete. Results saved to {config['results_dir']}")

if __name__ == "__main__":
    if os.path.exists('params.json'):
        config = load_config('params.json')
        test(config)
