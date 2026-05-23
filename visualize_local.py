import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datasets.xyt_dataset_generator import EnhancedSimulationPipeline
import json
import os

def visualize_demo():
    print("Initializing Generator on CPU...")
    # Load basic config/params
    if os.path.exists('params.json'):
        with open('params.json', 'r') as f:
            config = json.load(f)
        optical_params = config['optical_params']
    else:
        # Fallback defaults
        optical_params = {
            'cam_npixels': 128,
            'cam_pixelsize': 7.2e-8,
            'wavelength': 5.32e-7,
            'NA': 1.4,
            'n_medium': 1.33
        }

    # Ensure required params exist
    optical_params.setdefault('polarization', [1.0, 0.0]) # Linear polarization
    optical_params.setdefault('z_camera', 0.0)
    
    # Force CPU
    device = torch.device('cpu')
    pipeline = EnhancedSimulationPipeline(optical_params, device=device)
    
    print("Generating a short sequence (20 frames)...")
    
    # 1. Generate one random particle config
    fov = (optical_params.get('cam_npixels', 128) * optical_params.get('cam_pixelsize', 7.2e-8)) / 2.0
    gen_params = {
        'n_particles': 2,
        'type': 'random',
        'z_range': [50e-9, 100e-9],
        'field_of_view': fov,
        'min_dist': 5e-7
    }
    coords_list = pipeline.generate_particle_coordinates(n_configs=1, generation_params=gen_params)
    
    # 2. Generate a background
    surf_params = {
        'N': optical_params.get('cam_npixels', 128),
        'pixel_size': optical_params.get('cam_pixelsize', 7.2e-8),
        'sigma': 2e-9,
        'xi': 2e-7
    }
    surfaces, _ = pipeline.generate_rough_surfaces(n_surfaces=1, surface_params=surf_params, save_files=False)
    
    # 3. Create a mini dataset from this (cheat input format)
    # We just need to feed it into generate_xyt_dataset
    # But generate_xyt_dataset expects a dictionary. Let's construct a minimal one.
    
    # Or better, just call the logic directly to save overhead
    # ... Actually let's use the method we just implemented to verify it works too!
    
    # Construct "dataset_2d" mock
    mock_dataset_2d = {
        'coords_physical': np.array(coords_list), # [1, N, 3]
        'clean_images': np.zeros((1, 128, 128)), # Dummy
        'background_images': np.zeros((1, 128, 128)), # Dummy background, we will pass roughness manually if needed
        # Wait, generate_xyt_dataset uses 'sample_metadata' to get photon_scale
        'sample_metadata': [{'photon_scale': 40000}]
    }
    
    # We need to hack the pipeline to use the roughness we generated
    # The current implementation of generate_xyt_dataset tries to reconstruct roughness from background_images.
    # Let's bypass that and use a lower level loop for this visualization script to be robust.
    
    n_frames = 20
    D = 5e-14
    init_coords = coords_list[0].numpy() # [N, 3]
    roughness = torch.tensor(surfaces[0], device=device).float()
    
    trajectory = pipeline._simulate_brownian_motion(init_coords, n_frames, D, dt=0.01)
    
    print("Simulating frames...")
    frames = []
    
    for t in range(n_frames):
        coords_step = torch.tensor(trajectory[t], device=device).unsqueeze(0) # [1, N, 3]
        with torch.no_grad():
            res = pipeline.optical_model.simulate_with_background(
                particle_coords=coords_step,
                roughness_field=roughness.unsqueeze(0),
                add_noise=True,
                photon_scale=40000
            )
        # Normalize for visualization: (Raw / Median - 1)
        # But for raw video play, let's just show Raw
        frames.append(res['noisy_image'].squeeze().numpy())

    frames = np.array(frames)
    print(f"Generated shape: {frames.shape}")
    
    # Animation
    fig, ax = plt.subplots()
    ax.set_title("iSCAT Simulation (Local Preview)")
    ax.axis('off')
    
    # Use global normalization for consistent plotting
    vmin, vmax = frames.min(), frames.max()
    im = ax.imshow(frames[0], cmap='gray', vmin=vmin, vmax=vmax)
    
    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    
    # Save GIF
    gif_path = "iscat_demo.gif"
    ani.save(gif_path, writer='pillow')
    print(f"Saved animation to {os.path.abspath(gif_path)}")
    
    plt.show()

if __name__ == "__main__":
    visualize_demo()
