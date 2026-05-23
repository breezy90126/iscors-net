import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.xyt_dataset_generator import EnhancedSimulationPipeline
# We will import other modules as we implement them (e.g., train_model, ISCATEncoder)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_data(config):
    """
    Generates synthetic iSCAT data using EnhancedSimulationPipeline.
    """
    print("Initializing Data Generator...")
    optical_params = config.get('optical_params')
    noise_params = config.get('noise_params')
    
    if not optical_params:
        raise ValueError("optical_params missing in config")

    # Handle Special Parameters (Tensors, Complex numbers) that JSON can't represent
    import math
    if 'polarization' not in optical_params:
        optical_params['polarization'] = torch.tensor([1.0, 1j], dtype=torch.complex128) / torch.sqrt(torch.tensor(2.0))
    
    if 'qwp_angle' not in optical_params:
         optical_params['qwp_angle'] = math.pi/4*4

    pipeline = EnhancedSimulationPipeline(optical_params, noise_params)
    
    # Generate rough surfaces for background
    print("Generating rough surfaces...")
    # Prepare surface params with defaults if not present
    default_surface_params = {
        'N': optical_params.get('cam_npixels', 128),
        'pixel_size': optical_params.get('cam_pixelsize', 7.2e-8),
        'sigma': 2e-9,  # Typical roughness RMS (2 nm)
        'xi': 2e-7      # Typical correlation length (200 nm)
    }
    surface_params = config.get('surface_params', default_surface_params)
    n_surfaces = config.get('n_surfaces', 5)
    rough_surfaces, _ = pipeline.generate_rough_surfaces(n_surfaces=n_surfaces, surface_params=surface_params)
    
    # Generate particle coordinates
    print("Generating particle coordinates...")
    n_particles = config.get('n_particles', 1)
    n_configs = config.get('n_configs', 5)
    # Calculate field of view for random placement
    fov = (optical_params.get('cam_npixels', 128) * optical_params.get('cam_pixelsize', 7.2e-8)) / 2.0
    
    generation_params = {
        'n_particles': n_particles,
        'z_range': config.get('z_range', [60e-9, 80e-9]),
        'min_dist': 5e-7,
        'type': 'random',
        'field_of_view': fov
    }
    particle_coords_list = pipeline.generate_particle_coordinates(n_configs=n_configs, generation_params=generation_params)

    # Generate integrated dataset (2D)
    print("Generating integrated 2D dataset...")
    dataset_params = {
        'add_noise': True,
        'photon_scale_range': config.get('photon_scale_range', [30000, 50000]),
        'n_samples_per_combo': 1, # 10x10 = 100 unique movies
        'seed': config.get('seed', 42)
    }
    # Dynamic ranges could be added here if needed
    
    # We want to save to specific directories
    train_dir = config.get('train_folder', './data/train')
    os.makedirs(train_dir, exist_ok=True)
    
    # This generates the base 2D dataset
    dataset_2d = pipeline.generate_integrated_dataset_with_variations(
        particle_coords_list, 
        rough_surfaces, 
        dataset_params,
        output_dir=train_dir,
        save_files=False 
    )
    
    # Generate XYT dataset (Time-series)
    print("Generating XYT dataset...")
    n_frames = config.get('n_frames', 100)
    
    # pipeline.generate_xyt_dataset saves files to output_dir
    xyt_dataset = pipeline.generate_xyt_dataset(
        dataset_2d, 
        n_frames=n_frames, 
        output_dir=train_dir, 
        save_files=True
    )
    
    print(f"Data generation complete. Saved to {train_dir}")
    print("Next step: Run `python verify_data.py` to check the quality.")
    return xyt_dataset

def main():
    config_path = 'params.json'
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        return

    config = load_config(config_path)
    
    # ensure directories exist
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_path'], exist_ok=True)
    os.makedirs(config['train_folder'], exist_ok=True)

    # 1. Generate Data
    generate_data(config)

if __name__ == "__main__":
    main()
